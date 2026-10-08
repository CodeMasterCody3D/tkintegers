"""Qt WebEngine child embedded in an X11 Tk frame; serves the original Atlas."""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import mimetypes
import os
from pathlib import Path
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit


ASSETS = Path(__file__).parent / "vendor" / "model_atlas"
REVIEW_BRIDGE = """
// Tk-Integers integration. Atlas itself remains the original app above.
const reviewInTk = document.createElement('button');
reviewInTk.id = 'reviewInTkIntegers';
reviewInTk.className = 'primary';
reviewInTk.textContent = 'Review in Tk-Integers →';
reviewInTk.title = 'Export this exact quant mix and open it for review in the Tk-Integers tab. Does not start conversion.';
document.querySelector('#importPlanButton').insertAdjacentElement('afterend', reviewInTk);
reviewInTk.onclick = async () => {
  try {
    const bridge = await window.tkIntegersBridge;
    bridge.reviewRecipe(JSON.stringify(buildPlan('combined')));
  } catch(error) { showNotice('Could not send recipe: ' + error.message); }
};
"""
HEAD_BRIDGE = """
<style>
/* Accommodate the one additional button without narrowing the original controls. */
.model-control .import-row {display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.35fr);gap:8px;align-items:stretch}
.model-control .import-row button {font-size:12px;padding:9px 7px;min-width:0}
#importPlanButton,#reviewInTkIntegers {white-space:nowrap}
</style>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<script>
// Qt5's browser supports ES modules; these standard helpers arrived later.
if (!Object.hasOwn) Object.hasOwn = (object, key) => Object.prototype.hasOwnProperty.call(object,key);
if (!crypto.randomUUID) crypto.randomUUID = () => {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 15) | 64; bytes[8] = (bytes[8] & 63) | 128;
  const h = Array.from(bytes, v => v.toString(16).padStart(2,'0')).join('');
  return `${h.slice(0,8)}-${h.slice(8,12)}-${h.slice(12,16)}-${h.slice(16,20)}-${h.slice(20)}`;
};
window.tkIntegersBridge = new Promise(resolve => {
  new QWebChannel(qt.webChannelTransport, channel => resolve(channel.objects.tkIntegers));
});
</script>
"""


def emit(**event):
    print(json.dumps(event),flush=True)


class AtlasAssets(BaseHTTPRequestHandler):
    def log_message(self,*args):
        pass

    def do_GET(self):
        path = unquote(urlsplit(self.path).path).lstrip("/") or "index.html"
        file = (ASSETS / path).resolve()
        if (not file.is_relative_to(ASSETS.resolve()) or not file.is_file()
                or file.suffix not in {".html", ".js", ".css", ".json"}
                or path.startswith(("quant/", "tests/"))):
            self.send_error(404)
            return
        data = file.read_bytes()
        if path == "index.html":
            data = data.replace(b"</head>",HEAD_BRIDGE.encode()+b"</head>")
        elif path == "app.js":
            data += REVIEW_BRIDGE.encode()
        self.send_response(200)
        self.send_header("Content-Type",mimetypes.guess_type(file.name)[0] or "application/octet-stream")
        self.send_header("Content-Length",str(len(data)))
        self.send_header("Cache-Control","no-store")
        self.end_headers()
        self.wfile.write(data)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--parent",type=int,required=True)
    parser.add_argument("--width",type=int,default=1100)
    parser.add_argument("--height",type=int,default=650)
    args=parser.parse_args()
    from PyQt5.QtCore import Qt, QObject, pyqtSlot, QUrl, QSocketNotifier
    from PyQt5.QtGui import QColor, QDesktopServices
    from PyQt5.QtWidgets import QApplication, QFileDialog
    from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
    from PyQt5.QtWebChannel import QWebChannel

    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app=QApplication([sys.argv[0]])
    server=ThreadingHTTPServer(("127.0.0.1",0),AtlasAssets)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    url=f"http://127.0.0.1:{server.server_port}/"

    class Bridge(QObject):
        @pyqtSlot(str)
        def reviewRecipe(self, raw):
            if len(raw.encode())>20_000_000:
                emit(type="console",level=2,message="Recipe exceeds 20 MB.")
                return
            try:
                plan=json.loads(raw)
                if not isinstance(plan,dict):
                    raise ValueError("Recipe must be a JSON object")
                emit(type="review",plan=plan)
            except (ValueError,TypeError) as exc:
                emit(type="console",level=2,message=str(exc))

    class Page(QWebEnginePage):
        def javaScriptConsoleMessage(self,level,message,line,source):
            emit(type="console",level=int(level),message=message,line=line,source=source)

        def acceptNavigationRequest(self,target,kind,main_frame):
            if main_frame and kind==QWebEnginePage.NavigationTypeLinkClicked and target.scheme() in ("http","https") and not target.toString().startswith(url):
                QDesktopServices.openUrl(target)
                return False
            return super().acceptNavigationRequest(target,kind,main_frame)

        def createWindow(self, kind):
            # Preserve Atlas's source links and Open Actions buttons. They open
            # only when the user clicks; no second browser holds the Atlas UI.
            popup=QWebEnginePage(self)
            def opened(target):
                if target.scheme() in ("http","https"):
                    QDesktopServices.openUrl(target)
                    popup.deleteLater()
            popup.urlChanged.connect(opened)
            return popup

    view=QWebEngineView()
    view.setWindowFlags(Qt.FramelessWindowHint | Qt.X11BypassWindowManagerHint)
    page=Page(view)
    page.setBackgroundColor(QColor("#f5f7f5"))
    view.setPage(page)
    channel=QWebChannel(page)
    bridge=Bridge(channel)
    channel.registerObject("tkIntegers",bridge)
    page.setWebChannel(channel)
    view.resize(args.width,args.height)
    # Qt owns the browser widget; Tk owns its native parent and tab visibility.
    view.show()
    # Reparent the native surface after Qt initializes it. QWindow.setParent on
    # a foreign Tk container crashes Qt5 WebEngine during QWidget.show().
    x11=ctypes.CDLL(ctypes.util.find_library("X11"))
    x11.XOpenDisplay.argtypes=[ctypes.c_char_p]
    x11.XOpenDisplay.restype=ctypes.c_void_p
    x11.XReparentWindow.argtypes=[ctypes.c_void_p,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_int,ctypes.c_int]
    x11.XFlush.argtypes=[ctypes.c_void_p]
    x11.XCloseDisplay.argtypes=[ctypes.c_void_p]
    display=x11.XOpenDisplay(None)
    if not display:
        raise RuntimeError("Cannot connect to the Tk X11 display")
    x11.XReparentWindow(display,int(view.winId()),args.parent,0,0)
    x11.XFlush(display)

    def download(item):
        path,_=QFileDialog.getSaveFileName(view,"Export Atlas file",item.suggestedFileName(),"JSON recipe (*.json);;All files (*)")
        if not path:
            item.cancel()
            return
        item.setDownloadDirectory(str(Path(path).parent))
        item.setDownloadFileName(Path(path).name)
        item.finished.connect(lambda: emit(type="exported",path=path) if item.state()==item.DownloadCompleted else None)
        item.accept()
    page.profile().downloadRequested.connect(download)

    def loaded(ok):
        if not ok:
            emit(type="loaded",ok=False,message="HTML load failed")
            return
        page.runJavaScript("JSON.stringify({nodes:document.querySelectorAll('#flowSvg [data-node]').length,review:!!document.querySelector('#reviewInTkIntegers'),models:document.querySelector('#modelSelect').options.length})",
            lambda value: emit(type="loaded",ok=bool(value and json.loads(value).get("review")),dom=json.loads(value) if value else {},message="" if value and json.loads(value).get("review") else "Atlas scripts did not initialize"))
    view.loadFinished.connect(loaded)
    view.load(QUrl(url))

    # Private stdin protocol: only this renderer's owning Tk application sends it.
    buffer=bytearray()
    notifier=QSocketNotifier(sys.stdin.fileno(),QSocketNotifier.Read)
    def incoming():
        chunk=os.read(sys.stdin.fileno(),65536)
        if not chunk:
            app.quit()
            return
        buffer.extend(chunk)
        while b"\n" in buffer:
            line,_,rest=buffer.partition(b"\n")
            buffer[:]=rest
            try:
                command=json.loads(line)
                kind=command.get("type")
                if kind=="resize":
                    view.resize(max(1,int(command["width"])),max(1,int(command["height"])))
                    view.move(0,0)
                elif kind=="quit": app.quit()
                elif kind=="reload": view.reload()
                elif kind=="enabled": view.setEnabled(bool(command["value"]))
                elif kind=="evaluate":
                    identifier=command.get("id")
                    page.runJavaScript(command["code"],lambda result,i=identifier:emit(type="evaluation",id=i,result=result))
            except (ValueError,KeyError) as exc:
                emit(type="diagnostic",message=str(exc))
    notifier.activated.connect(incoming)
    emit(type="embedded",window=int(view.winId()),parent=args.parent)
    try:
        app.exec_()
    finally:
        server.shutdown()
        server.server_close()
        x11.XCloseDisplay(display)


if __name__=="__main__":
    main()
