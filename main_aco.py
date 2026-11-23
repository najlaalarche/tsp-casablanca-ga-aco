"""Point d'entrée de l'application ACO : python main_aco.py  (option : --autorun)"""
import ctypes
import logging
import sys
import tkinter as tk

from config.settings import DATA_DIR


def main():
    try:  # console Windows (cp1252) : ρ, α, β, × ne doivent pas provoquer d'erreur de log
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(DATA_DIR / "app_aco.log", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])
    if sys.platform == "win32":
        try:  # pixels physiques, cohérents avec la fenêtre WebView2 incrustée
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    from interface_aco.main_window_aco import MainWindowACO
    root = tk.Tk()
    MainWindowACO(root, autorun="--autorun" in sys.argv)
    root.mainloop()


if __name__ == "__main__":
    main()
