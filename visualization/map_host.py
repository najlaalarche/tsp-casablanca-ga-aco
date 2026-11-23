"""Processus enfant : navigateur Edge WebView2 (pywebview) qui affiche la carte Folium.

Pourquoi un processus séparé ?
- Tkinter et pywebview veulent tous deux posséder la boucle d'événements du
  thread principal ; les faire cohabiter dans le même processus est instable.
- CEF Python (cefpython3) n'existe pas pour Python 3.13 et tkinterweb
  n'exécute pas le JavaScript de Leaflet.
La fenêtre WebView2 (Chromium, JavaScript complet) est créée sans bordure et hors
écran, puis le processus parent l'incruste dans le quadrant 4 via Win32 SetParent.

Les cartes sont servies par un petit serveur HTTP local (127.0.0.1 uniquement) :
les serveurs de tuiles OpenStreetMap refusent les pages file:// (pas de Referer).

Protocole (stdin/stdout, une commande par ligne) :
  enfant -> parent : "PORT <n>" puis "HWND <entier>"
  parent -> enfant : "LOAD <nom de fichier dans le dossier servi>" | "QUIT"
"""
from __future__ import annotations
import ctypes
import functools
import os
import sys
import threading
import time
from ctypes import wintypes
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import webview

TITLE = f"TSP40-map-{os.getpid()}"


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def _serve(directory: Path) -> int:
    handler = functools.partial(_QuietHandler, directory=str(directory))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd.server_address[1]


def _find_hwnd(timeout: float = 20.0) -> int | None:
    user32 = ctypes.windll.user32
    found = []
    proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _):
        n = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        if buf.value == TITLE:
            found.append(hwnd)
            return False
        return True

    end = time.time() + timeout
    while time.time() < end:
        user32.EnumWindows(proto(cb), 0)
        if found:
            return int(found[0])
        time.sleep(0.1)
    return None


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    first = sys.argv[2] if len(sys.argv) > 2 else None
    port = _serve(root)
    base = f"http://127.0.0.1:{port}/"
    print(f"PORT {port}", flush=True)
    window = webview.create_window(TITLE, url=base + first if first else "about:blank",
                                   frameless=True, width=800, height=500,
                                   x=-10000, y=-10000, easy_drag=False)

    def worker():
        hwnd = _find_hwnd()
        if hwnd is None:
            print("ERROR no window", flush=True)
            window.destroy()
            return
        print(f"HWND {hwnd}", flush=True)
        for line in sys.stdin:  # se termine aussi si le parent disparaît
            cmd, _, arg = line.strip().partition(" ")
            if cmd == "LOAD" and arg:
                window.load_url(base + arg + f"?t={time.time():.0f}")
            elif cmd == "FIT":
                try:
                    window.evaluate_js("window.__tspRefit && window.__tspRefit()")
                except Exception:  # noqa: BLE001  (page en cours de chargement)
                    pass
            elif cmd == "EVAL" and arg:  # diagnostic
                try:
                    print(f"EVAL {window.evaluate_js(arg)}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"EVAL error {e}", flush=True)
            elif cmd == "QUIT":
                break
        window.destroy()

    webview.start(lambda: threading.Thread(target=worker, daemon=True).start(),
                  gui="edgechromium", private_mode=True)


if __name__ == "__main__":
    main()
