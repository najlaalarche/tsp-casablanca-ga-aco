"""Thème sombre ttk + Matplotlib."""
import tkinter as tk
from tkinter import ttk
import matplotlib

from config.settings import COLORS as C

FONT = ("Segoe UI", 10)
FONT_B = ("Segoe UI Semibold", 10)
FONT_H = ("Segoe UI Semibold", 12)
FONT_KPI = ("Segoe UI Semibold", 15)


def apply(root: tk.Tk) -> None:
    root.configure(bg=C["bg"])
    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure(".", background=C["panel"], foreground=C["fg"], fieldbackground=C["panel2"],
                bordercolor=C["border"], lightcolor=C["panel"], darkcolor=C["panel"],
                troughcolor=C["panel2"], font=FONT, insertcolor=C["fg"])
    s.configure("TFrame", background=C["panel"])
    s.configure("Bg.TFrame", background=C["bg"])
    s.configure("TLabel", background=C["panel"], foreground=C["fg"])
    s.configure("Muted.TLabel", foreground=C["muted"])
    s.configure("H.TLabel", font=FONT_H)
    s.configure("Header.TLabel", background=C["bg"], font=("Segoe UI Semibold", 15))
    s.configure("HeaderMuted.TLabel", background=C["bg"], foreground=C["muted"])
    s.configure("KPI.TLabel", font=FONT_KPI, foreground=C["accent2"])
    s.configure("TCheckbutton", background=C["panel"], foreground=C["fg"])
    s.map("TCheckbutton", background=[("active", C["panel"])])
    s.configure("TCombobox", fieldbackground=C["panel2"], background=C["panel2"],
                foreground=C["fg"], arrowcolor=C["fg"])
    s.map("TCombobox", fieldbackground=[("readonly", C["panel2"])],
          foreground=[("readonly", C["fg"])], selectbackground=[("readonly", C["panel2"])])
    root.option_add("*TCombobox*Listbox.background", C["panel2"])
    root.option_add("*TCombobox*Listbox.foreground", C["fg"])
    root.option_add("*TCombobox*Listbox.selectBackground", C["accent"])
    s.configure("TEntry", fieldbackground=C["panel2"], foreground=C["fg"])
    s.configure("TButton", background=C["panel2"], foreground=C["fg"], padding=(8, 5),
                borderwidth=1, focuscolor=C["panel2"])
    s.map("TButton", background=[("disabled", C["panel"]), ("active", C["border"])],
          foreground=[("disabled", "#5b6070")])
    s.configure("Accent.TButton", background=C["accent"], foreground="#fff", font=FONT_B)
    s.map("Accent.TButton", background=[("disabled", C["panel"]), ("active", "#3b74e0")])
    s.configure("Mode.TButton", font=FONT_B)
    s.configure("ModeOn.TButton", background=C["accent2"], foreground="#0b1512", font=FONT_B)
    s.map("ModeOn.TButton", background=[("active", "#27a98a")])
    s.configure("Danger.TButton", background="#4a2328")
    s.map("Danger.TButton", background=[("active", C["danger"]), ("disabled", C["panel"])])
    s.configure("Horizontal.TProgressbar", background=C["accent2"], troughcolor=C["panel2"],
                bordercolor=C["panel2"], lightcolor=C["accent2"], darkcolor=C["accent2"])
    s.configure("TPanedwindow", background=C["bg"])
    s.configure("Sash", sashthickness=6, background=C["bg"])

    matplotlib.rcParams.update({
        "figure.facecolor": C["panel"], "axes.facecolor": C["panel2"],
        "axes.edgecolor": C["border"], "axes.labelcolor": C["fg"], "text.color": C["fg"],
        "xtick.color": C["muted"], "ytick.color": C["muted"], "grid.color": C["border"],
        "axes.grid": True, "grid.alpha": 0.6, "legend.facecolor": C["panel"],
        "legend.edgecolor": C["border"], "font.size": 8.5, "axes.titlesize": 10,
    })
