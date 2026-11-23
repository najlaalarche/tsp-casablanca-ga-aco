"""Cadre défilable verticalement (Canvas + barre de défilement), au thème sombre."""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk

from config.settings import COLORS as C


class VScrollFrame(ttk.Frame):
    """Le contenu se place dans self.inner. La barre n'apparaît que si le contenu dépasse la hauteur."""

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.canvas = tk.Canvas(self, bg=C["panel"], highlightthickness=0, bd=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self._update())
        self.canvas.bind("<Configure>", lambda e: self._update())
        # molette active seulement quand la souris est au-dessus du quadrant
        self.bind_all_wheel = False
        self.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._wheel))
        self.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

    def _update(self):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        req = self.inner.winfo_reqheight()
        # largeur du contenu = largeur visible ; hauteur = max(visible, demandée) pour garder l'étirement
        self.canvas.itemconfigure(self._win, width=cw, height=max(ch, req))
        self.canvas.configure(scrollregion=(0, 0, cw, max(ch, req)))
        if req > ch + 1:
            if not self.vsb.winfo_ismapped():
                self.vsb.pack(side="right", fill="y", before=self.canvas)
        elif self.vsb.winfo_ismapped():
            self.vsb.pack_forget()
            self.canvas.yview_moveto(0)

    def _wheel(self, event):
        if self.vsb.winfo_ismapped():
            # pas de défilement si la molette agit sur une liste déroulante ouverte
            if not str(event.widget).endswith("popdown.f.l"):
                self.canvas.yview_scroll(int(-event.delta / 120), "units")
