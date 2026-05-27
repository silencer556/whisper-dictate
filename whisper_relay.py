#!/usr/bin/env python3
"""
Whisper Dictate relay — run this on the Mac.

It listens for text over a TCP socket and types it locally using the
clipboard + Cmd+V, which works in any focused app.

Usage:
    python3 whisper_relay.py            # listens on default port 9753
    python3 whisper_relay.py 9999       # custom port

Add to Mac login items or run in a terminal you leave open.
Requires: Python 3 (built into macOS), no extra packages.
"""

import socket
import subprocess
import sys
import time

PORT = 9753


def type_text(text: str) -> None:
    # Use clipboard + Cmd+V — handles all characters without escaping issues.
    # pbcopy sets the Mac clipboard; osascript fires Cmd+V into the focused app.
    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=False)
    time.sleep(0.15)  # give clipboard a moment to settle
    subprocess.run(
        ["osascript", "-e",
         'tell application "System Events" to keystroke "v" using command down'],
        check=False,
    )


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", port))
        srv.listen()
        print(f"whisper-relay listening on port {port}  (Ctrl+C to stop)")
        while True:
            conn, addr = srv.accept()
            with conn:
                chunks = []
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
                text = b"".join(chunks).decode("utf-8")
                if text:
                    print(f"[{addr[0]}] typing: {text!r}")
                    type_text(text)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
