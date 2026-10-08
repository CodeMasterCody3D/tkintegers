"""Embed the actual Atlas HTML application, not a native reconstruction."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk


class ModelAtlasPanel:
    def __init__(self, app, parent, colors):
        self.app, self.parent = app, parent
        self.process = None
        self.started = False
        self.closed = False
        self.resize_id = None
        self.status = tk.StringVar(parent, "Opening the original Atlas HTML interface…")
        bar = ttk.Frame(parent)
        bar.pack(fill="x", pady=(0, 6))
        ttk.Label(bar, textvariable=self.status, style="Muted.TLabel").pack(side="left")
        ttk.Button(bar, text="Reload Atlas", command=self.reload).pack(side="right")
        self.host = tk.Frame(parent, background=colors["panel"], highlightthickness=0, container=True)
        self.host.pack(fill="both", expand=True)
        self.host.bind("<Map>", self._mapped)
        self.host.bind("<Configure>", self._resize)
        self.host.bind("<Destroy>", lambda event: self.close() if event.widget is self.host else None)

    def _mapped(self, _=None):
        if not self.started:
            self.started = True
            self.parent.after_idle(self.start)
        else:
            self._resize()

    def start(self):
        if self.closed:
            return
        try:
            if self.host.tk.call("tk", "windowingsystem") != "x11":
                raise RuntimeError("Embedded Atlas currently needs Tk on X11 (including XWayland).")
            if importlib.util.find_spec("PyQt5.QtWebEngineWidgets") is None:
                raise RuntimeError("Install python3-pyqt5.qtwebengine to render the original Atlas HTML in this tab.")
            env = {**os.environ, "QT_QPA_PLATFORM":"xcb"}
            command = [sys.executable, "-u", "-X", "faulthandler", str(Path(__file__).with_name("model_atlas_webview.py")),
                       "--parent", str(self.host.winfo_id()), "--width", str(max(1,self.host.winfo_width())),
                       "--height", str(max(1,self.host.winfo_height()))]
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, text=True, env=env)
            process = self.process
            def read_events():
                for line in process.stdout:
                    try:
                        self.app.events.put(("atlas_web", json.loads(line)))
                    except json.JSONDecodeError:
                        self.app.events.put(("atlas_web", {"type":"diagnostic", "message":line.strip()}))
                code = process.wait()
                if not self.closed and self.process is process:
                    self.app.events.put(("atlas_web", {"type":"exit", "message":f"Atlas renderer closed (exit {code}). Use Reload Atlas to reopen it."}))
            def read_errors():
                for line in process.stderr:
                    # Keep renderer diagnostics in the app log, not in the UI status loop.
                    print("[Atlas renderer] " + line.rstrip(), file=sys.stderr, flush=True)
            threading.Thread(target=read_events, daemon=True).start()
            threading.Thread(target=read_errors, daemon=True).start()
        except (ImportError, OSError, RuntimeError) as exc:
            self.status.set(str(exc))

    def send(self, message):
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.stdin.write(json.dumps(message)+"\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError):
                self.status.set("Atlas renderer disconnected. Use Reload Atlas.")

    def _resize(self, _=None):
        if self.closed or self.resize_id is not None:
            return
        self.resize_id = self.parent.after_idle(self._send_size)

    def _send_size(self):
        self.resize_id = None
        if not self.closed:
            self.send({"type":"resize","width":max(1,self.host.winfo_width()),"height":max(1,self.host.winfo_height())})

    def reload(self):
        if self.app.busy:
            return
        if self.process is not None and self.process.poll() is None:
            self.send({"type":"reload"})
        else:
            self.start()

    def set_enabled(self, enabled):
        self.send({"type":"enabled", "value":enabled})

    def handle(self, event):
        kind = event.get("type")
        if kind == "loaded":
            if event.get("ok"):
                self.status.set("Original Atlas · full HTML interface · Import saved plan or Review in Tk-Integers")
            else:
                self.status.set("Atlas could not load: " + event.get("message","check the app log"))
        elif kind == "review":
            self.app.review_html_atlas(event["plan"])
        elif kind == "exported":
            self.app.last_atlas_export = event["path"]
            self.status.set("Atlas exported: " + event["path"])
        elif kind == "exit":
            self.status.set(event["message"])
        elif kind == "console" and event.get("level") == 2:
            self.status.set("Atlas script error: " + event["message"])
        # Evaluation receipts are used by UI-only integration checks.
        self.last_event = event

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.resize_id is not None:
            self.parent.after_cancel(self.resize_id)
        if self.process is not None:
            self.send({"type":"quit"})
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
