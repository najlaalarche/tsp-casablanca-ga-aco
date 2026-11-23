"""Fenêtre « Verdicts des agents » : les 3 agents du SMA côte à côte, texte complet."""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk

from config.settings import COLORS as C

TITLES = ("Agent web", "Agent sémantique / scientifique", "Agent arbitre")
ROLES = ("recherche DuckDuckGo réelle, sources citées vérifiées par Python",
         "analyse théorique de l'Ant System (sans recherche web)",
         "compare les deux propositions et décide")


class VerdictsDialog(tk.Toplevel):
    def __init__(self, master, on_rerun=None):
        super().__init__(master)
        self.title("Verdicts des agents (SMA)")
        self.configure(bg=C["bg"])
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(int(sw * 0.85), 1800), min(int(sh * 0.8), 1000)
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{max(0, (sh - h) // 4)}")
        self.minsize(900, 500)
        bot = ttk.Frame(self, padding=(10, 0, 10, 10))
        bot.pack(fill="x", side="bottom")
        self.decision = ttk.Label(bot, text="", style="H.TLabel")
        self.decision.pack(side="left")
        ttk.Button(bot, text="Fermer", command=self.destroy).pack(side="right")
        self.btn_rerun = ttk.Button(bot, text="↻ Relancer les agents en échec", state="disabled",
                                    command=on_rerun or (lambda: None))
        self.btn_rerun.pack(side="right", padx=(0, 8))
        self.file_lbl = ttk.Label(bot, text="", style="Muted.TLabel")
        self.file_lbl.pack(side="right", padx=12)
        body = ttk.Frame(self, padding=10)
        body.pack(fill="both", expand=True)
        self.cols = []
        for i in range(3):
            body.columnconfigure(i, weight=1, uniform="c")
            f = ttk.Frame(body, padding=(8, 6))
            f.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 8, 0))
            ttk.Label(f, text=TITLES[i], style="H.TLabel").pack(anchor="w")
            ttk.Label(f, text=ROLES[i], style="Muted.TLabel").pack(anchor="w")
            status = ttk.Label(f, text="", style="Muted.TLabel", wraplength=480, justify="left")
            status.pack(anchor="w", pady=(4, 0))
            params = tk.Label(f, text="", bg=C["panel"], fg=C["accent2"], font=("Segoe UI Semibold", 11),
                              justify="left", anchor="w")
            params.pack(fill="x", pady=(2, 6))
            tf = ttk.Frame(f)
            tf.pack(fill="both", expand=True)
            txt = tk.Text(tf, height=8, wrap="word", bg=C["panel2"], fg=C["fg"], relief="flat", font=("Segoe UI", 10),
                          padx=10, pady=8, highlightthickness=0)
            sb = ttk.Scrollbar(tf, orient="vertical", command=txt.yview)
            txt.configure(yscrollcommand=sb.set)
            txt.pack(side="left", fill="both", expand=True)
            sb.pack(side="right", fill="y")
            txt.tag_configure("h", font=("Segoe UI Semibold", 10), foreground=C["accent"], spacing1=8)
            txt.tag_configure("err", foreground=C["danger"])
            txt.tag_configure("muted", foreground=C["muted"])
            self.cols.append((status, params, txt))
        body.rowconfigure(0, weight=1)

    def refresh(self, results, session=None, running: bool = False, can_rerun: bool = False):
        """results : [AgentResult | None] × 3 ; session : AgentsSession finale (décision humaine)."""
        for i, (status, params, txt) in enumerate(self.cols):
            r = results[i] if i < len(results) else None
            txt.configure(state="normal")
            txt.delete("1.0", "end")
            if r is None:
                status.configure(text="en cours…" if running else "pas encore consulté", foreground=C["muted"])
                params.configure(text="—")
                txt.insert("end", "Cliquez sur BC4 Réglage SMA pour consulter les agents."
                           if not running else "En attente de la réponse de l'agent…", "muted")
            elif r.config is None:
                status.configure(text=f"ÉCHEC · {r.model or '—'}", foreground=C["danger"])
                params.configure(text="Aucune proposition valide")
                txt.insert("end", "Erreur\n", "h")
                txt.insert("end", (r.error or "erreur inconnue") + "\n", "err")
                calls = r.extra.get("tool_calls") or []
                if i == 0 and calls:
                    txt.insert("end", "Appels d'outil effectués (recherche_web)\n", "h")
                    txt.insert("end", "".join(
                        f"• recherche_web(\"{c['requete']}\") → {c['resultats']} source(s)"
                        + (f" — {c['erreur']}" if c.get("erreur") else "") + "\n" for c in calls), "muted")
            elif r.extra.get("fallback"):
                status.configure(text="DÉCISION DE SECOURS (arbitre Gemini indisponible) · calculée par Python",
                                 foreground=C["warn"])
                p = r.config.params()
                params.configure(text=f"{p['n_ants']} fourmis · ρ = {p['rho']} · α = {p['alpha']} · "
                                      f"β = {p['beta']} · {p['cycles']} cycles")
                txt.insert("end", "Règle appliquée par Python\n", "h")
                txt.insert("end", r.extra["rule"] + ".\n")
                txt.insert("end", "Aucune analyse de Gemini : cette décision n'est pas une justification d'agent.\n", "muted")
                if r.extra.get("basis"):
                    txt.insert("end", "Éléments utilisés\n", "h")
                    txt.insert("end", "".join(f"• {b}\n" for b in r.extra["basis"]))
                if r.extra.get("divergences"):
                    txt.insert("end", "Paramètres arbitrés par la règle\n", "h")
                    txt.insert("end", "".join(f"• {d['parameter']} → {d['decision']} : {d['reason']}\n"
                                              for d in r.extra["divergences"]))
                txt.insert("end", "Erreur de l'arbitre Gemini\n", "h")
                txt.insert("end", (r.extra.get("gemini_error") or "—") + "\n", "err")
            else:
                status.configure(text=f"OK · {r.model}" if r.extra.get("sdk") else f"OK · modèle {r.model}",
                                 foreground=C["accent2"])
                p = r.config.params()
                params.configure(text=f"{p['n_ants']} fourmis · ρ = {p['rho']} · α = {p['alpha']} · "
                                      f"β = {p['beta']} · {p['cycles']} cycles")
                if i == 2:
                    ex = r.extra
                    if ex.get("basis"):
                        txt.insert("end", "Sur quoi l'arbitre fonde sa décision\n", "h")
                        txt.insert("end", "".join(f"• {b}\n" for b in ex["basis"]))
                txt.insert("end", "Justification\n", "h")
                txt.insert("end", r.config.justification + "\n")
                if r.note and i != 2:
                    txt.insert("end", "Note\n", "h")
                    txt.insert("end", r.note + "\n", "muted")
                if i == 2:
                    ex = r.extra
                    for key, title in (("convergences", "Convergences"), ("rejected", "Arguments écartés")):
                        if ex.get(key):
                            txt.insert("end", title + "\n", "h")
                            txt.insert("end", "".join(f"• {x}\n" for x in ex[key]))
                    if ex.get("divergences"):
                        txt.insert("end", "Divergences tranchées\n", "h")
                        txt.insert("end", "".join(f"• {d['parameter']} → {d['decision']} : {d['reason']}\n"
                                                  for d in ex["divergences"]))
                    if ex.get("python_diff"):
                        txt.insert("end", "Écarts calculés par Python (web / scientifique)\n", "h")
                        txt.insert("end", "".join(
                            f"• {d['parameter']} : {d['web']} / {d['scientific']} "
                            f"({'accord' if d['agree'] else 'divergence'})\n" for d in ex["python_diff"]), "muted")
                    if ex.get("consensus"):
                        txt.insert("end", "Consensus\n", "h")
                        txt.insert("end", ex["consensus"] + "\n")
                calls = r.extra.get("tool_calls") or []
                if i == 0 and calls:
                    txt.insert("end", "Appels d'outil effectués (recherche_web)\n", "h")
                    txt.insert("end", "".join(
                        f"• recherche_web(\"{c['requete']}\") → {c['resultats']} source(s)"
                        + (f" — {c['erreur']}" if c.get("erreur") else "") + "\n" for c in calls), "muted")
                cited = [s for s in r.sources if s.cited]
                if cited:
                    txt.insert("end", "Sources citées (vérifiées)\n", "h")
                    txt.insert("end", "".join(f"[{s.id}] {s.title}\n    {s.uri}\n" for s in cited), "muted")
            txt.configure(state="disabled")
        self.btn_rerun.configure(state="normal" if can_rerun else "disabled")
        if session is not None:
            self.decision.configure(text=f"Décision humaine : {session.decision}")
            self.file_lbl.configure(text=session.saved_to or "")
        else:
            self.decision.configure(text="Décision humaine : en attente" if running else "")
            self.file_lbl.configure(text="")
