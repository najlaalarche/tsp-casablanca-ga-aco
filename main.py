"""Point d'entrée : python main.py"""
import ctypes
import logging
import sys
import tkinter as tk

from config.settings import DATA_DIR


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(DATA_DIR / "app.log", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])
    if sys.platform == "win32":
        try:  # coordonnées en pixels physiques, cohérentes avec la fenêtre WebView2 incrustée
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    from interface.main_window import MainWindow
    root = tk.Tk()
    MainWindow(root, autorun="--autorun" in sys.argv)
    root.mainloop()


if __name__ == "__main__":
    main()
