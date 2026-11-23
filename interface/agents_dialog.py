"""Fenêtre du bouton 3 : système multi-agents Gemini et validation humaine de la configuration."""
from __future__ import annotations
import logging
import queue
import threading
import tkinter as tk
import webbrowser
from tkinter import ttk, messagebox

from agents.agent_manager import run_agents, AgentsSession
from agents.gemini_client import GeminiConfigError
from agents.schemas import AgentResult, PARAM_KEYS, SchemaError, validate_config
from config.settings import (COLORS as C, POPULATION_CHOICES, CROSSOVER_CHOICES, GENERATION_CHOICES,
                             SELECTION_CHOICES)

log = logging.getLogger(__name__)
LABELS = {"population_size": "Population", "selection_method": "Sélection", "mutation_rate": "Mutation",
          "crossover_rate": "Croisement", "generations": "Générations", "elitism": "Élitisme"}
SEL_FR = {"tournament": "Tournoi", "roulette": "Roue biaisée"}


def fmt(key, v):
    if v is None:
        return "—"
    if key == "selection_method":
        return SEL_FR.get(v, v)
    if key == "elitism":
        return "oui" if v else "non"
    if key == "mutation_rate":
        return f"{v:.4g}"
    return str(v)


class AgentsDialog(tk.Toplevel):
    def __init__(self, master, problem, on_apply):
        super().__init__(master)
        self.title("Réglage par agents IA — Gemini (Web Research · Scientifique · Arbitre)")
        self.configure(bg=C["bg"])
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(int(sw * 0.9), 1900), min(int(sh * 0.88), 1250)
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{max(0, (sh - h) // 4)}")
        self.minsize(1000, 650)
        self.problem, self.on_apply = problem, on_apply
        self.q: queue.Queue = queue.Queue()
        self.session: AgentsSession | None = None
        self.running = False
        self.editing = False
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(100, self._pump)
        self.start()

    # ------------------------------------------------------------------ UI
    def _build(self):
        top = ttk.Frame(self, padding=(10, 8))
        top.pack(fill="x")
        self.status = ttk.Label(top, text="", style="H.TLabel")
        self.status.pack(side="left")
        self.btn_rerun = ttk.Button(top, text="↻ Relancer les agents", command=self.start, state="disabled")
        self.btn_rerun.pack(side="right")
        self.progress = ttk.Progressbar(self, maximum=100)
        self.progress.pack(fill="x", padx=10)
        self.model_lbl = ttk.Label(self, text="", style="Muted.TLabel")
        self.model_lbl.pack(fill="x", padx=10, pady=(2, 4))

        # --- bas : validation humaine (réservé en premier pour ne jamais être rogné) ---
        bot = ttk.Frame(self, padding=10)
        bot.pack(side="bottom", fill="x")
        ttk.Label(bot, text="Configuration finale (validation humaine)", style="H.TLabel").grid(
            row=0, column=0, columnspan=12, sticky="w")
        self.v = {"population_size": tk.StringVar(), "selection_method": tk.StringVar(),
                  "mutation_rate": tk.StringVar(), "crossover_rate": tk.StringVar(),
                  "generations": tk.StringVar(), "elitism": tk.BooleanVar(value=True)}
        widgets = [
            ("population_size", ttk.Combobox(bot, textvariable=self.v["population_size"], width=7,
                                             values=[str(x) for x in POPULATION_CHOICES])),
            ("selection_method", ttk.Combobox(bot, textvariable=self.v["selection_method"], width=12,
                                              values=list(SELECTION_CHOICES))),
            ("mutation_rate", ttk.Entry(bot, textvariable=self.v["mutation_rate"], width=8)),
            ("crossover_rate", ttk.Combobox(bot, textvariable=self.v["crossover_rate"], width=6,
                                            values=[str(x) for x in CROSSOVER_CHOICES])),
            ("generations", ttk.Combobox(bot, textvariable=self.v["generations"], width=7,
                                         values=[str(x) for x in GENERATION_CHOICES])),
            ("elitism", ttk.Checkbutton(bot, variable=self.v["elitism"], text="")),
        ]
        self.fields = []
        for i, (key, w) in enumerate(widgets):
            ttk.Label(bot, text=LABELS[key]).grid(row=1, column=2 * i, sticky="e", padx=(8 if i else 0, 4), pady=6)
            w.grid(row=1, column=2 * i + 1, sticky="w")
            w.configure(state="disabled")
            self.fields.append(w)
        btns = ttk.Frame(bot)
        btns.grid(row=2, column=0, columnspan=12, sticky="ew", pady=(4, 0))
        self.btn_accept = ttk.Button(btns, text="✔ Accepter et lancer la simulation", style="Accent.TButton",
                                     command=self._accept, state="disabled")
        self.btn_accept.pack(side="left")
        self.btn_modify = ttk.Button(btns, text="✎ Modifier", command=self._modify, state="disabled")
        self.btn_modify.pack(side="left", padx=6)
        self.btn_refuse = ttk.Button(btns, text="✖ Refuser", style="Danger.TButton", command=self._refuse,
                                     state="disabled")
        self.btn_refuse.pack(side="left")
        self.save_lbl = ttk.Label(btns, text="", style="Muted.TLabel")
        self.save_lbl.pack(side="right")

        # --- milieu : 3 agents + divergences ---
        mid = ttk.PanedWindow(self, orient="vertical")
        mid.pack(fill="both", expand=True, padx=10)
        cards = ttk.PanedWindow(mid, orient="horizontal")
        mid.add(cards, weight=5)
        self.cards = {}
        for key, title in (("web", "① Agent Web Research"), ("sci", "② Agent Scientifique"),
                           ("arb", "③ Agent Arbitre — décision")):
            f = ttk.Frame(cards, padding=8)
            cards.add(f, weight=1)
            self.cards[key] = self._card(f, title, sources=(key == "web"))

        div = ttk.Frame(mid, padding=(0, 6))
        mid.add(div, weight=3)
        ttk.Label(div, text="Divergences et décisions de l'arbitre", style="H.TLabel").pack(anchor="w")
        cols = [("p", "Paramètre", 110), ("w", "Web", 90), ("s", "Scientifique", 100), ("f", "Final", 90),
                ("st", "Statut", 115), ("r", "Décision de l'arbitre", 600)]
        self.tree = ttk.Treeview(div, columns=[c[0] for c in cols], show="headings", height=6)
        for k, label, w in cols:
            self.tree.heading(k, text=label)
            self.tree.column(k, width=w, minwidth=w, anchor="w" if k == "r" else "center", stretch=k == "r")
        self.tree.tag_configure("div", foreground=C["warn"])
        self.tree.tag_configure("ok", foreground=C["accent2"])
        # La décision complète est réservée en bas (jamais rognée) ; le tableau prend le reste.
        self.reason_lbl = tk.Label(div, text="", bg=C["panel"], fg=C["fg"], anchor="w", justify="left",
                                   wraplength=1400, font=("Segoe UI", 9))
        self.reason_lbl.pack(side="bottom", fill="x")
        self.tree.pack(fill="both", expand=True, pady=(4, 2))
        div.bind("<Configure>", lambda e: self.reason_lbl.configure(wraplength=max(300, e.width - 20)))
        self.tree.bind("<<TreeviewSelect>>", self._show_reason)
        st = ttk.Style(self)
        st.configure("Treeview", background=C["panel2"], fieldbackground=C["panel2"], foreground=C["fg"],
                     rowheight=24, borderwidth=0)
        st.configure("Treeview.Heading", background=C["panel"], foreground=C["fg"], relief="flat")

    def _card(self, parent, title, sources=False) -> dict:
        ttk.Label(parent, text=title, style="H.TLabel").pack(anchor="w")
        badge = tk.Label(parent, text="en attente…", bg=C["panel2"], fg=C["muted"], anchor="w",
                         justify="left", wraplength=520, padx=6, pady=3, font=("Segoe UI", 9))
        badge.pack(fill="x", pady=(2, 4))
        grid = ttk.Frame(parent)
        grid.pack(fill="x")
        vals = {}
        for i, k in enumerate(PARAM_KEYS):
            ttk.Label(grid, text=LABELS[k], style="Muted.TLabel").grid(row=i // 3 * 2, column=i % 3, sticky="w", padx=(0, 14))
            v = ttk.Label(grid, text="—", font=("Segoe UI Semibold", 11))
            v.grid(row=i // 3 * 2 + 1, column=i % 3, sticky="w", padx=(0, 14))
            vals[k] = v
        out = {"badge": badge, "vals": vals}
        text_parent = parent
        if sources:
            # Deux onglets : chacun dispose de toute la hauteur de la carte (lisible à toute résolution).
            st = ttk.Style(self)
            st.configure("Dark.TNotebook", background=C["panel"], borderwidth=0)
            st.configure("Dark.TNotebook.Tab", background=C["panel2"], foreground=C["muted"], padding=(10, 3))
            st.map("Dark.TNotebook.Tab", background=[("selected", C["accent"])],
                   foreground=[("selected", "#ffffff")])
            nb = ttk.Notebook(parent, style="Dark.TNotebook")
            nb.pack(fill="both", expand=True, pady=(6, 0))
            tab_j = ttk.Frame(nb)
            tab_s = ttk.Frame(nb, padding=(0, 4))
            nb.add(tab_j, text="Justification")
            nb.add(tab_s, text="Sources")
            ttk.Label(tab_s, text="★ citée par Gemini · survol ou clic : URL et extrait · double-clic : ouvrir",
                      style="Muted.TLabel").pack(anchor="w")
            # La liste prend toute la hauteur ; le détail s'affiche en info-bulle (survol ou sélection).
            lb = tk.Listbox(tab_s, height=4, bg=C["panel2"], fg=C["fg"], relief="flat",
                            highlightthickness=0, selectbackground=C["accent"], font=("Segoe UI", 9))
            lb.pack(fill="both", expand=True)
            lb.bind("<Double-Button-1>", self._open_source)
            lb.bind("<<ListboxSelect>>", self._show_source)
            lb.bind("<Motion>", self._hover_source)
            lb.bind("<Leave>", lambda e: self._hide_tip())
            out.update(sources=lb, notebook=nb, tab_sources=tab_s)
            text_parent = tab_j
        txt = tk.Text(text_parent, wrap="word", bg=C["panel2"], fg=C["fg"], relief="flat", height=6,
                      font=("Segoe UI", 9), padx=6, pady=4)
        txt.pack(fill="both", expand=True, pady=(0 if sources else 6, 0))
        txt.configure(state="disabled")
        out["text"] = txt
        parent.bind("<Configure>", lambda e, b=badge: b.configure(wraplength=max(200, e.width - 30)), add="+")
        return out

    # ------------------------------------------------------------ exécution
    def start(self):
        if self.running:
            return
        self.running = True
        self.session = None
        self.progress["value"] = 0
        self.status.configure(text="Agents en cours…")
        for card in self.cards.values():
            self._fill_card(card, None, "en attente…", C["muted"])
        self.tree.delete(*self.tree.get_children())
        self._set_buttons()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            s = run_agents(self.problem, progress=lambda m, p: self.q.put(("progress", m, p)))
            self.q.put(("done", s))
        except GeminiConfigError as e:
            self.q.put(("config_error", str(e)))
        except Exception as e:  # noqa: BLE001
            log.exception("Agents")
            self.q.put(("error", str(e)))

    def _pump(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    self.status.configure(text=msg[1])
                    self.progress["value"] = msg[2]
                elif kind == "done":
                    self.running = False
                    self._show(msg[1])
                elif kind in ("config_error", "error"):
                    self.running = False
                    self.status.configure(text="Échec des agents", foreground=C["danger"])
                    self._set_buttons()
                    messagebox.showerror("Gemini" if kind == "config_error" else "Erreur des agents",
                                         msg[1], parent=self)
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(150, self._pump)

    # ------------------------------------------------------------ affichage
    def _fill_card(self, card, res: AgentResult | None, badge: str, color: str, body: str = ""):
        card["badge"].configure(text=badge, fg=color)
        for k, lbl in card["vals"].items():
            lbl.configure(text=fmt(k, getattr(res.config, k)) if res and res.config else "—")
        t = card["text"]
        t.configure(state="normal")
        t.delete("1.0", "end")
        t.insert("1.0", body)
        t.configure(state="disabled")
        if "sources" in card:
            card["sources"].delete(0, "end")
            card["sources"].uris = []
            card["sources"].items = []

    def _show(self, s: AgentsSession):
        self.session = s
        self.progress["value"] = 100
        self.model_lbl.configure(text=f"Modèle Gemini utilisé : {s.model}   ·   Disponibilité testée : "
                                      + " | ".join(s.probe) + f"   ·   Session : {s.saved_to}")
        web, sci, arb = s.web, s.scientific, s.arbiter
        # Agent web
        if web.error:
            self._fill_card(self.cards["web"], web, f"ERREUR : {web.error}", C["danger"])
        else:
            color = C["accent2"] if web.web_search_used else C["warn"]
            n_cited = sum(x.cited for x in web.sources)
            queries = web.extra.get("queries", [])
            badge = (f"✔ Recherche DuckDuckGo réelle — {len(queries)} requête(s), {len(web.sources)} source(s), "
                     f"{n_cited} citée(s) (citations vérifiées par Python)" if web.web_search_used
                     else "⚠ " + web.note.split("\nDétail")[0])
            body = web.config.justification
            if queries:
                body += "\n\nREQUÊTES EFFECTUÉES :\n" + "\n".join(f"• {q}" for q in queries)
            self._fill_card(self.cards["web"], web, badge, color, body)
            lb = self.cards["web"]["sources"]
            ordered = sorted(web.sources, key=lambda x: (not x.cited, int(x.id[1:]) if x.id[1:].isdigit() else 0))
            lb.items = ordered
            lb.uris = [x.uri for x in ordered]
            for i, src in enumerate(ordered):
                lb.insert("end", f"{'★' if src.cited else '  '} [{src.id}] {src.title}")
                lb.itemconfigure(i, foreground=C["accent2"] if src.cited else C["muted"])
            if not web.sources:
                lb.insert("end", "Aucune source vérifiée (pas de recherche web).")
            self.cards["web"]["notebook"].tab(self.cards["web"]["tab_sources"],
                                              text=f"Sources ({len(web.sources)} · {n_cited} citée(s))")
        # Agent scientifique
        if sci.error:
            self._fill_card(self.cards["sci"], sci, f"ERREUR : {sci.error}", C["danger"])
        else:
            self._fill_card(self.cards["sci"], sci, "Analyse théorique (sans web)", C["muted"], sci.config.justification)
        # Arbitre
        if arb is None or arb.error:
            msg = arb.error if arb else "non exécuté (aucune proposition valide)"
            self._fill_card(self.cards["arb"], arb, f"ERREUR : {msg}", C["danger"])
            self.status.configure(text="Pas de configuration finale : relancez les agents.")
        else:
            e = arb.extra
            body = (f"CONSENSUS : {e.get('consensus', '')}\n\nJUSTIFICATION :\n{arb.config.justification}\n\n"
                    "CONVERGENCES :\n" + "\n".join(f"• {c}" for c in e.get("convergences", [])) +
                    "\n\nÉCARTÉ :\n" + ("\n".join(f"• {r}" for r in e.get("rejected", [])) or "• (rien)"))
            self._fill_card(self.cards["arb"], arb, "Configuration finale proposée — à valider", C["accent2"], body)
            self._fill_divergences(web, sci, arb)
            self._set_form(arb.config.params())
            self.status.configure(text="Propositions reçues : acceptez, modifiez ou refusez la configuration.")
        self._set_buttons()

    def _fill_divergences(self, web, sci, arb):
        self.tree.delete(*self.tree.get_children())
        reasons = {d["parameter"]: f"{d['decision']} — {d['reason']}" for d in arb.extra.get("divergences", [])}
        for k in PARAM_KEYS:
            wv = getattr(web.config, k) if web.config else None
            sv = getattr(sci.config, k) if sci.config else None
            fv = getattr(arb.config, k)
            agree = wv is not None and sv is not None and wv == sv
            reason = reasons.get(k, "Accord des deux agents." if agree else "")
            self.tree.insert("", "end", values=(LABELS[k], fmt(k, wv), fmt(k, sv), fmt(k, fv),
                                                "accord" if agree else "DIVERGENCE", reason),
                             tags=("ok" if agree else "div",))
        first_div = next((i for i in self.tree.get_children()
                          if self.tree.item(i, "values")[4] == "DIVERGENCE"), None)
        if first_div:
            self.tree.selection_set(first_div)
            self._show_reason()
        else:
            self.reason_lbl.configure(text="Aucune divergence : les deux agents proposent la même configuration.")

    def _show_reason(self, _e=None):
        sel = self.tree.selection()
        if sel:
            v = self.tree.item(sel[0], "values")
            self.reason_lbl.configure(text=f"{v[0]} : {v[5]}")

    def _set_form(self, p: dict):
        inv = {v: k for k, v in SELECTION_CHOICES.items()}
        self.v["population_size"].set(str(p["population_size"]))
        self.v["selection_method"].set(inv[p["selection_method"]])
        self.v["mutation_rate"].set(f"{p['mutation_rate']:.4g}")
        self.v["crossover_rate"].set(str(p["crossover_rate"]))
        self.v["generations"].set(str(p["generations"]))
        self.v["elitism"].set(bool(p["elitism"]))

    def _set_buttons(self):
        ready = (not self.running and self.session is not None and self.session.arbiter is not None
                 and self.session.arbiter.config is not None)
        st = "normal" if ready else "disabled"
        for b in (self.btn_accept, self.btn_modify, self.btn_refuse):
            b.configure(state=st)
        self.btn_rerun.configure(state="disabled" if self.running else "normal")

    # ------------------------------------------------ détail d'une source
    def source_detail(self, index: int) -> str | None:
        items = getattr(self.cards["web"]["sources"], "items", [])
        if 0 <= index < len(items):
            s = items[index]
            return (f"[{s.id}] {s.title}\n{s.uri}\nRequête : {s.query}\n"
                    f"« {self._short(s.snippet, 300) or 'aucun extrait'} »"
                    + ("\n★ citée par Gemini" if s.cited else ""))
        return None

    def _tip_at(self, index: int, x_root: int, y_root: int):
        text = self.source_detail(index)
        if text is None:
            self._hide_tip()
            return
        if getattr(self, "_tip", None) is None or not self._tip.winfo_exists():
            self._tip = tk.Toplevel(self)
            self._tip.wm_overrideredirect(True)
            self._tip_lbl = tk.Label(self._tip, bg=C["panel"], fg=C["fg"], justify="left", anchor="w",
                                     wraplength=560, font=("Segoe UI", 9), padx=8, pady=6,
                                     highlightthickness=1, highlightbackground=C["border"])
            self._tip_lbl.pack()
        self._tip_lbl.configure(text=text)
        self._tip.wm_geometry(f"+{x_root + 16}+{y_root + 12}")
        self._tip.deiconify()
        self._tip.lift()

    def _hide_tip(self):
        if getattr(self, "_tip", None) is not None and self._tip.winfo_exists():
            self._tip.withdraw()

    def _hover_source(self, e):
        lb = e.widget
        i = lb.nearest(e.y)
        bbox = lb.bbox(i)
        if bbox and bbox[1] <= e.y <= bbox[1] + bbox[3]:
            self._tip_at(i, e.x_root, e.y_root)
        else:
            self._hide_tip()

    def _show_source(self, _e):
        lb = self.cards["web"]["sources"]
        sel = lb.curselection()
        if sel:
            bbox = lb.bbox(sel[0]) or (0, 0, 0, 0)
            self._tip_at(sel[0], lb.winfo_rootx() + bbox[0] + 40, lb.winfo_rooty() + bbox[1] + bbox[3])

    @staticmethod
    def _short(text: str, n: int = 170) -> str:
        text = " ".join((text or "").split())
        return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"

    def _open_source(self, _e):
        lb = self.cards["web"]["sources"]
        sel = lb.curselection()
        if sel and getattr(lb, "uris", None) and sel[0] < len(lb.uris):
            webbrowser.open(lb.uris[sel[0]])

    # ------------------------------------------------------------ décisions
    def _modify(self):
        self.editing = True
        for w in self.fields:
            w.configure(state="readonly" if isinstance(w, ttk.Combobox) else "normal")
        self.btn_accept.configure(text="✔ Accepter la version modifiée et lancer")
        self.btn_modify.configure(state="disabled")

    def _read_form(self) -> dict:
        try:
            mut = float(self.v["mutation_rate"].get().replace(",", "."))
            data = {"population_size": int(self.v["population_size"].get()),
                    "selection_method": SELECTION_CHOICES[self.v["selection_method"].get()],
                    "mutation_rate": mut, "crossover_rate": float(self.v["crossover_rate"].get()),
                    "generations": int(self.v["generations"].get()), "elitism": self.v["elitism"].get(),
                    "justification": self.session.arbiter.config.justification}
        except (ValueError, KeyError) as e:
            raise SchemaError(f"Valeur illisible : {e}") from e
        return validate_config(data).params()

    def _accept(self):
        try:
            params = self._read_form()
        except SchemaError as e:
            messagebox.showwarning("Configuration invalide", str(e), parent=self)
            return
        modified = params != self.session.arbiter.config.params()
        self.session.decision = "modifiée puis acceptée" if modified else "acceptée"
        self.session.applied = params
        self.session.save()
        self.on_apply(params, self.session, modified)
        self.destroy()

    def _refuse(self):
        if not messagebox.askyesno("Refuser", "Refuser la recommandation des agents ?\n"
                                              "Aucun paramètre ne sera modifié.", parent=self):
            return
        self.session.decision = "refusée"
        self.session.save()
        self.destroy()

    def _close(self):
        if self.session and self.session.decision == "en attente" and self.session.arbiter:
            self.session.decision = "fermée sans décision"
            self.session.save()
        self.destroy()
