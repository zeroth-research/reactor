"""Live module visualizer with auto-updating via WebSocket.

Usage:
    from reactor.visual import show
    show(module)           # opens browser, auto-refreshes on value changes
    show(module, poll=0.5) # custom poll interval in seconds
"""
import json
import os
import socket
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import asyncio
import websockets

from ..reactor import Var, X
from ..blocks import views


_TEMPLATE_PATH = Path(__file__).parent / "template.html"


# The ids of the views of the variables start here, above the ids of any wire.
_VIEW_IDS = 1 << 40


def _view_id(view) -> int:
    """The id of the view of a variable: its latched value, or its next value. The
    blocks of the atoms give the views wires of their own, which the visualizer
    shows under the ids of the views, to link the atoms and the interfaces."""
    if isinstance(view, Var):
        return _VIEW_IDS + 2 * view.name
    return _VIEW_IDS + 2 * view.var.name + 1


def _serialize_terms(atom, kind, atom_idx, counter):
    """Serialize a block of terms to JSON-compatible dicts: the wires of the boundary
    under the ids of their views, the others under their own."""
    block = getattr(atom, kind)
    inputs, outputs = views(kind, atom.ctrl, atom.wait, atom.read)
    ids = {w: _view_id(v) for w, v in zip(block.read, inputs)}
    ids.update({w: _view_id(v) for w, v in zip(block.write, outputs)})

    result = []
    for term in block:
        tid = f"a{atom_idx}_{kind}_{counter[0]}"
        counter[0] += 1
        writes = list(term.write)
        result.append({
            "id": tid,
            "label": str(term.itype),
            "reads": [ids.get(w, w.id) for w in term.read],
            "writes": [ids.get(w, w.id) for w in writes],
            "dtype": str(writes[0].dtype) if writes else None,
        })
    return result


def _serialize_module(module):
    """Serialize a Module to a JSON-compatible dict."""
    counter = [0]
    atoms = []

    for atom_idx, atom in enumerate(module.atoms):
        atoms.append({
            "id": atom_idx,
            # ctrl and wait circles carry the ids of the next values, the ones the
            # init/update terms actually write and await; reads are latched
            "ctrl": [{"id": _view_id(X(v)), "dtype": str(v.dtype)} for v in atom.ctrl],
            "wait": [{"id": _view_id(X(v)), "dtype": str(v.dtype)} for v in atom.wait],
            "read": [{"id": _view_id(v), "dtype": str(v.dtype)} for v in atom.read],
            "init": _serialize_terms(atom, "init", atom_idx, counter),
            "update": _serialize_terms(atom, "update", atom_idx, counter),
        })

    def wire_pairs(iface):
        return [{"ltc": _view_id(v), "nxt": _view_id(X(v)), "dtype": str(v.dtype)} for v in iface]

    return {
        "atoms": atoms,
        "extl": wire_pairs(module.extl),
        "intf": wire_pairs(module.intf),
        "prvt": wire_pairs(module.prvt),
    }


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def show(module, names=None, poll=0.5, open_browser=True):
    """Start a live visualizer for the given module.

    Args:
        module: A reactor Module (or Env/torch.Module instance).
        names: Optional dict mapping Vars, Wires, or wire ids to display
               names (str); a Var names its latched value and, primed, its
               next value. If None, circles show no labels (topology only).
        poll: Interval in seconds between state pushes.
        open_browser: Whether to open the browser automatically. Overridden
               by the REACTOR_NO_BROWSER environment variable (for headless
               runs, e.g. executing the notebooks as tests).

    Returns:
        A stop function that shuts down the server.
    """
    wire_names = {}
    if names:
        for k, v in names.items():
            if isinstance(k, Var):
                wire_names[_view_id(k)] = str(v)
                wire_names[_view_id(X(k))] = str(v) + "'"
            else:
                wire_names[k.id if hasattr(k, 'id') else int(k)] = str(v)
    http_port = _find_free_port()
    ws_port = _find_free_port()

    html = _TEMPLATE_PATH.read_text().replace("{{WS_PORT}}", str(ws_port))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode())

        def log_message(self, format, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", http_port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    stop_event = threading.Event()

    async def ws_handler(websocket):
        while not stop_event.is_set():
            data = _serialize_module(module)
            data['wire_names'] = wire_names
            try:
                await websocket.send(json.dumps({"type": "full", "data": data}))
            except websockets.exceptions.ConnectionClosed:
                break
            await asyncio.sleep(poll)

    async def ws_main():
        async with websockets.serve(ws_handler, "127.0.0.1", ws_port):
            while not stop_event.is_set():
                await asyncio.sleep(0.1)

    def run_ws():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(ws_main())
        loop.close()

    threading.Thread(target=run_ws, daemon=True).start()

    url = f"http://127.0.0.1:{http_port}"
    if open_browser and not os.environ.get("REACTOR_NO_BROWSER"):
        webbrowser.open(url)
    print(f"Module viewer running at {url} (ws://127.0.0.1:{ws_port})")

    def stop():
        stop_event.set()
        httpd.shutdown()
        print("Module viewer stopped.")

    return stop
