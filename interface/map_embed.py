"""Incrustation d'une fenêtre WebView2 (processus map_host) dans un Frame Tkinter (Windows)."""
from __future__ import annotations
import ctypes
import logging
import queue
import subprocess
import sys
import threading
import tkinter as tk
from ctypes import wintypes
from pathlib import Path

from config.settings import BASE_DIR, CACHE_DIR

log = logging.getLogger(__name__)

GWL_STYLE = -16
WS_CHILD = 0x40000000
WS_POPUP = 0x80000000
WS_CAPTION = 0x00C00000
WS_THICKFRAME = 0x00040000
WS_VISIBLE = 0x10000000
SW_SHOW = 5

if sys.platform == "win32":
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SetParent.argtypes = [wintypes.HWND, wintypes.HWND]
    user32.SetParent.restype = wintypes.HWND
    user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
    user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
    user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
    user32.MoveWindow.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_int, wintypes.BOOL]
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.IsWindow.argtypes = [wintypes.HWND]


class EmbeddedBrowser(tk.Frame):
    """Frame Tkinter qui héberge un navigateur Chromium (Edge WebView2) avec JavaScript."""

    def __init__(self, master, initial_html: Path | None = None,
                 on_status=None, **kw):
        super().__init__(master, **kw)
        self.on_status = on_status or (lambda msg, ok: None)
        self.hwnd: int | None = None
        self.port: int | None = None
        self.proc: subprocess.Popen | None = None
        self._events: queue.Queue = queue.Queue()
        self._pending: Path | None = None
        self._fit_job = None
        self.bind("<Configure>", self._resize)
        if sys.platform != "win32":
            self.on_status("Carte intégrée disponible uniquement sous Windows (WebView2).", False)
            return
        self._start(initial_html)
        self.after(100, self._poll)

    # ---------- processus enfant ----------
    def _start(self, html: Path | None):
        # Les cartes sont servies depuis CACHE_DIR par le serveur HTTP local du processus enfant.
        args = [sys.executable, "-m", "visualization.map_host", str(CACHE_DIR)]
        if html:
            args.append(Path(html).name)
        try:
            self.proc = subprocess.Popen(
                args, cwd=str(BASE_DIR), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as e:
            self.on_status(f"Impossible de démarrer le navigateur embarqué : {e}", False)
            return
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stdout(self):
        for line in self.proc.stdout:
            self._events.put(line.strip())
        self._events.put("EXIT")

    def _read_stderr(self):
        for line in self.proc.stderr:
            log.info("map_host: %s", line.rstrip())

    def _poll(self):
        try:
            while True:
                ev = self._events.get_nowait()
                if ev.startswith("PORT "):
                    self.port = int(ev.split()[1])
                elif ev.startswith("HWND "):
                    self.hwnd = int(ev.split()[1])
                    self._embed()
                elif ev == "EXIT" and self.hwnd is not None:
                    self.on_status("Le navigateur embarqué s'est arrêté.", False)
                    self.hwnd = None
                elif ev == "EXIT":
                    self.on_status("Échec du démarrage de WebView2 (voir logs).", False)
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(100, self._poll)

    def _embed(self):
        parent = self.winfo_id()
        user32.SetParent(self.hwnd, parent)
        style = user32.GetWindowLongPtrW(self.hwnd, GWL_STYLE)
        style = (style & ~(WS_POPUP | WS_CAPTION | WS_THICKFRAME)) | WS_CHILD | WS_VISIBLE
        user32.SetWindowLongPtrW(self.hwnd, GWL_STYLE, style)
        user32.ShowWindow(self.hwnd, SW_SHOW)
        self._resize()
        self.on_status("Carte Leaflet intégrée (Edge WebView2).", True)
        if self._pending:
            self.load(self._pending)

    def _resize(self, _event=None):
        if self.hwnd and user32.IsWindow(self.hwnd):
            user32.MoveWindow(self.hwnd, 0, 0, self.winfo_width(), self.winfo_height(), True)
            self._schedule_fit(400)

    def _schedule_fit(self, delay_ms: int):
        """Demande à Leaflet de recadrer les points (regroupé si plusieurs resize rapprochés)."""
        if self._fit_job is not None:
            self.after_cancel(self._fit_job)
        self._fit_job = self.after(delay_ms, self._fit)

    def _fit(self):
        self._fit_job = None
        self._send("FIT")

    def _send(self, cmd: str):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(cmd + "\n")
                self.proc.stdin.flush()
            except OSError:
                pass

    # ---------- API publique ----------
    def load(self, html: Path):
        if self.hwnd is None:
            self._pending = html
            return
        self._pending = None
        self._send(f"LOAD {Path(html).name}")
        self.after(1500, lambda: self._send("FIT"))

    def url_for(self, html: Path) -> str | None:
        return f"http://127.0.0.1:{self.port}/{Path(html).name}" if self.port else None

    def close(self):
        self._send("QUIT")
        if self.proc:
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
