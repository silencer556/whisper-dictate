"""
Tiny HTTP server that lets the browser extension poll for transcribed text.

GET /pending?crd=1  → {"text": "..."} or {"text": ""}
  ?crd=1  means the extension currently sees a remotedesktop.google.com tab.
  ?crd=0  means no CRD tab is open right now.

The extension sends this flag on every 250 ms poll so Python always knows
whether CRD is active without any extra round-trip.
"""

import json
import logging
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

log = logging.getLogger(__name__)

_q: queue.Queue[str] = queue.Queue()
_server: HTTPServer | None = None

# Updated by the extension on every poll; used by is_crd_active().
_crd_last_seen: float = 0.0
_CRD_TIMEOUT_SEC = 1.5   # extension polls every 250 ms, so 1.5 s is generous


def enqueue(text: str) -> None:
    """Called from the transcription thread to push text to the extension."""
    _q.put(text)


def is_crd_active() -> bool:
    """True if the extension reported a CRD tab open within the last 1.5 s."""
    return (time.monotonic() - _crd_last_seen) < _CRD_TIMEOUT_SEC


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        global _crd_last_seen
        parsed = urlparse(self.path)
        if parsed.path != "/pending":
            self.send_response(404)
            self.end_headers()
            return

        # Extension reports whether a CRD tab is currently open.
        params = parse_qs(parsed.query)
        if params.get("crd", ["0"])[0] == "1":
            _crd_last_seen = time.monotonic()

        try:
            text = _q.get_nowait()
        except queue.Empty:
            text = ""

        body = json.dumps({"text": text}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass  # silence per-request noise in stdout


def start(port: int = 9754) -> None:
    global _server
    if _server is not None:
        return
    _server = HTTPServer(("127.0.0.1", port), _Handler)
    t = threading.Thread(target=_server.serve_forever, daemon=True, name="ext-server")
    t.start()
    log.info("Extension server listening on http://127.0.0.1:%d/pending", port)


def stop() -> None:
    global _server
    if _server:
        _server.shutdown()
        _server = None
