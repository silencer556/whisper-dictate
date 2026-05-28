"""
Voice command processing — runs after vocabulary substitution.

A command phrase may appear:
  • as the entire transcribed segment  ("whisper enter")
  • at the END of a segment            ("see you later whisper enter")

In both cases the phrase is stripped and not typed; everything before it is
typed normally first.
"""

import logging
import re

log = logging.getLogger(__name__)

_PUNCT_RE = re.compile(r'[^\w\s]')

# phrase (lowercase) -> action name
_BUILT_IN: dict[str, str] = {
    "whisper enter":        "enter",
    "whisper new line":     "enter",
    "whisper tab":          "tab",
    "whisper delete that":  "delete_last",
    "whisper scratch that": "delete_last",
    "whisper undo":         "undo",
    "whisper select all":   "select_all",
    "whisper copy":         "copy",
    "whisper paste":        "paste",
    "whisper cancel":       "cancel",
    "whisper stop":         "cancel",
    "whisper resend":       "resend",
}


def _clean(text: str) -> str:
    """Lowercase and strip punctuation for command matching.

    Whisper often inserts commas/periods around words, e.g. "Whisper, enter"
    instead of "Whisper enter".  Removing punctuation lets the matcher ignore
    those artefacts.
    """
    return re.sub(r'\s+', ' ', _PUNCT_RE.sub(' ', text.lower())).strip()


def find_command(text: str) -> tuple[str, str | None]:
    """Return (text_before_command, action_or_None).

    If the whole segment is a command, text_before_command is ''.
    If no command is found, action_or_None is None and text is unchanged.
    Punctuation inserted by Whisper (e.g. "Whisper, enter") is ignored.
    """
    stripped = text.strip()
    clean = _clean(stripped)

    # Longest match wins so "delete that" can't shadow "scratch that" etc.
    for phrase in sorted(_BUILT_IN, key=len, reverse=True):
        if clean == phrase:
            log.debug("Command: whole segment %r -> %s", phrase, _BUILT_IN[phrase])
            return ("", _BUILT_IN[phrase])
        if clean.endswith(" " + phrase):
            # Remove the command words from the end of the ORIGINAL text.
            # Count words rather than characters so punctuation differences
            # between clean and stripped don't throw off the slice.
            n_cmd_words = len(phrase.split())
            orig_words = stripped.split()
            prefix = " ".join(orig_words[:-n_cmd_words]).rstrip(" ,;.")

            # Rescue any sentence-ending punctuation Whisper attached to the
            # last command word (e.g. "enter?" or "enter.").  Without this,
            # "how are you whisper enter?" loses the "?" when "enter?" is stripped.
            last_cmd_word = orig_words[-1] if orig_words else ""
            m = re.search(r'([.!?]+)$', last_cmd_word)
            if m:
                rescued = m.group(1)
                # Only rescue if the prefix doesn't already end with sentence-ending
                # punctuation.  Whisper routinely adds "." to the last word of any
                # utterance; we don't want that period appended when the prefix
                # already ends with "!" or "?" from a vocabulary substitution.
                if not prefix or prefix[-1] not in '.!?':
                    prefix += rescued

            log.debug("Command at end: %r -> %s  (prefix=%r)",
                      phrase, _BUILT_IN[phrase], prefix)
            return (prefix, _BUILT_IN[phrase])

    return (text, None)


_PUNCT = frozenset('.!?,;:')


def _chars_to_delete(text: str) -> int:
    """How many characters to erase on "delete that".

    Walks backwards through *text*, skipping any trailing whitespace/punctuation,
    then keeps going until it hits the next punctuation mark — that's the natural
    sentence/clause break to stop at.  If none is found, deletes the whole thing.

    Examples
    --------
    "Hello world, how are you today "  ->  19  (" how are you today ")
    "I went to the store. Then home. " ->  15  (" Then home. ")
    "No punctuation here "             ->  20  (everything)
    """
    content = text.rstrip()          # strip trailing whitespace
    i = len(content) - 1

    # Skip any punctuation sitting right at the end of the content
    # (e.g. the closing '.' of a sentence — that's not a useful break point)
    while i >= 0 and content[i] in _PUNCT:
        i -= 1

    # Now walk backwards to find the PREVIOUS punctuation mark
    while i >= 0:
        if content[i] in _PUNCT:
            # Delete everything after this punctuation up to end of original text
            return len(text) - (i + 1)
        i -= 1

    # No inner punctuation found — delete the whole segment
    return len(text)


_HOTKEY_LOCALS = {
    "copy":  "ctrl+c",
    "paste": "ctrl+v",
}


def _send_hotkey(hotkey: str) -> None:
    """Send copy/paste via the extension (CRD → Mac Cmd) or local keyboard (Ctrl).

    When Chrome Remote Desktop is active the extension handles the keypress and
    forwards it as Meta+key so the Mac receives Cmd+C / Cmd+V.  Otherwise we
    fall back to a local keyboard.send() with the Windows Ctrl equivalent.
    """
    from . import ext_server
    if ext_server.is_crd_active():
        ext_server.enqueue_hotkey(hotkey)
        log.debug("Hotkey %r → extension (CRD active)", hotkey)
    else:
        try:
            import keyboard
            combo = _HOTKEY_LOCALS[hotkey]
            keyboard.send(combo)
            log.debug("Hotkey %r → keyboard.send(%r)", hotkey, combo)
        except Exception as exc:
            log.warning("Hotkey %r via keyboard failed: %s", hotkey, exc)


def run_action(action: str, last_typed: str,
               output_method: str, keystroke_delay_ms: int) -> None:
    """Execute a voice command action.  Called from the GUI transcription thread."""
    from .output import type_text

    if action == "enter":
        type_text("\n", method=output_method, trailing_space=False,
                  keystroke_delay_ms=keystroke_delay_ms)
        log.info("Command: Enter")

    elif action == "tab":
        type_text("\t", method=output_method, trailing_space=False,
                  keystroke_delay_ms=keystroke_delay_ms)
        log.info("Command: Tab")

    elif action == "delete_last":
        if last_typed:
            n = _chars_to_delete(last_typed)
            type_text("\b" * n, method=output_method,
                      trailing_space=False, keystroke_delay_ms=keystroke_delay_ms)
            log.info("Command: deleted %d of %d chars (back to last punctuation)",
                     n, len(last_typed))
        else:
            log.debug("Command: delete_last — nothing to delete")

    elif action == "undo":
        try:
            import keyboard
            keyboard.send("ctrl+z")
            log.info("Command: Ctrl+Z")
        except Exception as exc:
            log.warning("Command: undo failed: %s", exc)

    elif action == "select_all":
        try:
            import keyboard
            keyboard.send("ctrl+a")
            log.info("Command: Ctrl+A")
        except Exception as exc:
            log.warning("Command: select_all failed: %s", exc)

    elif action == "copy":
        _send_hotkey("copy")
        log.info("Command: Copy")

    elif action == "paste":
        _send_hotkey("paste")
        log.info("Command: Paste")

    else:
        log.warning("Unknown command action: %r", action)
