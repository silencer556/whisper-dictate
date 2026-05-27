import time
import logging

log = logging.getLogger(__name__)


def type_text(text: str, method: str = "keystroke", trailing_space: bool = True,
              keystroke_delay_ms: int = 0,
              remote_host: str = "", remote_port: int = 9753) -> None:
    """Send *text* to the currently focused window.

    method:
        "keystroke"  — keyboard.write(), works in most apps
        "clipboard"  — copy -> Ctrl+V -> restore clipboard
        "extension"  — push to local HTTP server; browser extension injects via CDP
        "remote"     — TCP relay to whisper_relay.py on a Mac on the same LAN
    """
    if not text:
        return

    if trailing_space:
        text += " "

    # Auto-detect: CDP extension when a CRD tab is open, keystroke otherwise.
    if method == "auto":
        from . import ext_server
        method = "extension" if ext_server.is_crd_active() else "keystroke"
        log.debug("auto mode -> %s", method)

    if method == "extension":
        _type_extension(text)
    elif method == "remote":
        _type_remote(text, remote_host, remote_port)
    elif method == "clipboard":
        _type_clipboard(text)
    else:
        try:
            _type_keystroke(text, delay_sec=keystroke_delay_ms / 1000.0)
        except Exception as exc:
            log.warning("keystroke output failed (%s) — falling back to clipboard", exc)
            _type_clipboard(text)


def _type_extension(text: str) -> None:
    from . import ext_server
    ext_server.enqueue(text)
    log.debug("Queued %d chars for extension", len(text))


def _type_keystroke(text: str, delay_sec: float = 0.0) -> None:
    import keyboard, re
    # Split into printable runs and backspace runs so keyboard.write() never
    # has to deal with \b, which it may not handle on all platforms.
    for part in re.split(r'(\x08+)', text):
        if not part:
            continue
        if part[0] == '\x08':
            for _ in part:
                keyboard.send('backspace')
                if delay_sec:
                    time.sleep(delay_sec)
        else:
            keyboard.write(part, delay=delay_sec)
    log.debug("Typed %d chars via keystroke (delay=%.0fms)", len(text), delay_sec * 1000)


def _type_remote(text: str, host: str, port: int) -> None:
    import socket
    if not host:
        log.error("output.method = 'remote' but remote.host is not set in config")
        return
    try:
        with socket.create_connection((host, port), timeout=5.0) as s:
            s.sendall(text.encode("utf-8"))
        log.debug("Sent %d chars to relay %s:%d", len(text), host, port)
    except Exception as exc:
        log.error("Remote relay send failed (%s:%d): %s", host, port, exc)


def _type_clipboard(text: str) -> None:
    import pyperclip
    import keyboard

    try:
        previous = pyperclip.paste()
    except Exception:
        previous = ""

    pyperclip.copy(text)
    keyboard.send("ctrl+v")
    log.debug("Typed %d chars via clipboard paste", len(text))

    # Restore original clipboard after a short delay so Ctrl+V has time to fire
    time.sleep(0.15)
    try:
        pyperclip.copy(previous)
    except Exception:
        pass
