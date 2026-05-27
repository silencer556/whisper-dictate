import threading
import time
import logging
import tkinter as tk

from .vocabulary import apply_substitutions
from .output import type_text

log = logging.getLogger(__name__)

_COLORS = {
    "idle":         "#888888",
    "recording":    "#e74c3c",
    "transcribing": "#f5a623",
}
_LABELS = {
    "idle":         "Idle — press hotkey to start",
    "recording":    "Recording...",
    "transcribing": "Transcribing...",
}


def _play_tone(freq: int, duration_ms: int, volume: float = 0.5, sample_rate: int = 44100):
    import numpy as np
    import sounddevice as sd
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), endpoint=False)
    wave = (volume * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sd.play(wave, samplerate=sample_rate)
    sd.wait()


def _beep_start():
    try:
        _play_tone(880, 120)
        time.sleep(0.05)
        _play_tone(1320, 180)
    except Exception:
        pass


def _beep_stop():
    try:
        _play_tone(1320, 120)
        time.sleep(0.05)
        _play_tone(660, 200)
    except Exception:
        pass


class DictateGUI:
    """Push-to-talk / continuous dictation GUI."""

    def __init__(self, recorder, transcriber, vocab: dict, sample_rate: int,
                 hotkey: str = "win+`", output_method: str = "keystroke",
                 trailing_space: bool = True, keystroke_delay_ms: int = 10,
                 auto_stop_silence_sec: float = 1.5, vad_threshold_db: float = -40.0,
                 config_path=None):
        self._recorder = recorder
        self._transcriber = transcriber
        self._vocab = vocab
        self._sample_rate = sample_rate
        self._hotkey = hotkey
        self._output_method = output_method
        self._trailing_space = trailing_space
        self._keystroke_delay_ms = keystroke_delay_ms
        self._auto_stop_silence_sec = auto_stop_silence_sec
        self._vad_threshold_db = vad_threshold_db

        self._config_path = config_path

        self._state = "idle"
        self._state_lock = threading.Lock()
        self._continuous = False       # True while in continuous dictation mode
        self._trigger_was_hotkey = False
        self._last_typed = ""           # text typed in last segment (for "delete that")
        self._pending_cmd_id: str | None = None   # after() timer id for deferred command

        self.root = tk.Tk()
        self.root.title("Whisper Dictate")
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)
        self._build_ui()
        self._register_hotkey()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = tk.Frame(self.root, padx=20, pady=16)
        outer.pack()

        self._canvas = tk.Canvas(outer, width=52, height=52, highlightthickness=0, bg=self.root["bg"])
        self._canvas.pack()
        self._dot = self._canvas.create_oval(6, 6, 46, 46, fill=_COLORS["idle"], outline="")

        self._status_var = tk.StringVar(value=_LABELS["idle"])
        tk.Label(outer, textvariable=self._status_var, font=("Segoe UI", 11)).pack(pady=(2, 10))

        hotkey_label = self._hotkey.replace("+", " + ").upper()
        self._btn = tk.Button(
            outer,
            text=f"Start   [ {hotkey_label} ]",
            command=self._toggle,
            width=26,
            height=2,
            font=("Segoe UI", 10),
        )
        self._btn.pack(pady=(0, 14))

        tk.Label(outer, text="Last transcription:", anchor="w", font=("Segoe UI", 9)).pack(fill="x")
        self._text = tk.Text(
            outer,
            height=5,
            width=46,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("Segoe UI", 10),
            relief=tk.SUNKEN,
            bd=1,
        )
        self._text.pack(pady=(2, 0))

        # Pending-command countdown (shows "↵ Enter in 1.5 s — keep talking to cancel")
        self._pending_var = tk.StringVar()
        tk.Label(outer, textvariable=self._pending_var,
                 font=("Segoe UI", 8, "italic"), fg="#3498db").pack(fill="x", pady=(2, 0))

        # Clickable uncertain-words bar — each word is a link that opens the vocab dialog.
        self._conf_text = tk.Text(
            outer, height=2, font=("Segoe UI", 8),
            wrap=tk.WORD, state=tk.DISABLED,
            relief=tk.FLAT, bd=0, bg=self.root["bg"], cursor="arrow",
        )
        self._conf_text.tag_configure("label", foreground="#888888")
        self._conf_text.pack(fill="x", pady=(2, 0))

        btn_row = tk.Frame(outer)
        btn_row.pack(pady=(6, 0))

        self._copy_btn = tk.Button(
            btn_row, text="Copy text", command=self._copy_transcription,
            font=("Segoe UI", 9), width=12,
        )
        self._copy_btn.pack(side="left", padx=(0, 8))

        tk.Button(
            btn_row, text="Settings", command=self._show_settings,
            font=("Segoe UI", 9), relief=tk.FLAT, fg="#555555",
        ).pack(side="left")

    # ------------------------------------------------------------------
    # Hotkey
    # ------------------------------------------------------------------

    # Virtual-key codes for keys that aren't plain ASCII letters/digits.
    _VK_EXTRA: dict[str, int] = {
        '`': 0xC0, '~': 0xC0,
        '-': 0xBD, '=': 0xBB,
        '[': 0xDB, ']': 0xDD, '\\': 0xDC,
        ';': 0xBA, "'": 0xDE,
        ',': 0xBC, '.': 0xBE, '/': 0xBF,
        'space': 0x20, 'enter': 0x0D, 'tab': 0x09,
        'esc': 0x1B, 'escape': 0x1B,
        'f1':  0x70, 'f2':  0x71, 'f3':  0x72, 'f4':  0x73,
        'f5':  0x74, 'f6':  0x75, 'f7':  0x76, 'f8':  0x77,
        'f9':  0x78, 'f10': 0x79, 'f11': 0x7A, 'f12': 0x7B,
    }
    _MOD_FLAGS: dict[str, int] = {
        'win': 0x0008, 'ctrl': 0x0002, 'control': 0x0002,
        'alt': 0x0001, 'shift': 0x0004,
    }

    def _register_hotkey(self):
        """Register the push-to-talk hotkey.

        When the hotkey uses the Win modifier we prefer the Windows
        RegisterHotKey API.  The keyboard-library approach (suppress=True)
        installs a low-level hook that intercepts every Win+key event to
        watch for the right combo, which inadvertently blocks Win+D,
        Win+L, and other system shortcuts.  RegisterHotKey only ever
        fires for the exact registered combination.
        """
        parts = [p.strip().lower() for p in self._hotkey.split('+')]
        if 'win' in parts[:-1]:
            try:
                if self._register_via_win_api(parts):
                    return
            except Exception as exc:
                log.warning("RegisterHotKey failed (%s) — falling back to keyboard library", exc)

        # Fallback: keyboard library (fine for non-Win hotkeys).
        try:
            import keyboard
            keyboard.add_hotkey(self._hotkey, self._hotkey_triggered, suppress=True)
            log.info("Hotkey registered via keyboard library: %s", self._hotkey)
        except Exception as exc:
            log.warning("Could not register hotkey %s: %s  (try running as admin)",
                        self._hotkey, exc)

    def _register_via_win_api(self, parts: list[str]) -> bool:
        """Register using Windows RegisterHotKey.  Returns True on success.

        Runs a dedicated daemon thread with a GetMessage loop so Tkinter's
        event loop is never blocked.  The thread self-terminates if the
        application exits (daemon=True).
        """
        import ctypes
        import ctypes.wintypes as wt
        import queue as _queue
        import threading as _threading

        user32 = ctypes.windll.user32

        key_name = parts[-1]
        mods = 0x4000  # MOD_NOREPEAT
        for m in parts[:-1]:
            flag = self._MOD_FLAGS.get(m)
            if flag is None:
                log.warning("Unknown hotkey modifier %r", m)
            else:
                mods |= flag

        vk = self._VK_EXTRA.get(key_name)
        if vk is None:
            if len(key_name) == 1:
                vk = user32.VkKeyScanW(ord(key_name)) & 0xFF
            else:
                log.warning("Unknown key name %r — cannot use RegisterHotKey", key_name)
                return False

        WM_HOTKEY = 0x0312
        HOTKEY_ID = 1
        result_q: _queue.Queue[bool] = _queue.Queue()

        def _message_loop():
            ok = bool(user32.RegisterHotKey(None, HOTKEY_ID, mods, vk))
            result_q.put(ok)
            if not ok:
                log.error("RegisterHotKey failed (GetLastError=%d)", ctypes.GetLastError())
                return
            log.info("Hotkey registered via RegisterHotKey API: %s (mods=0x%x vk=0x%x)",
                     self._hotkey, mods, vk)
            msg = wt.MSG()
            try:
                while True:
                    ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                    if ret == 0 or ret == -1:
                        break
                    if msg.message == WM_HOTKEY and msg.wParam == HOTKEY_ID:
                        self._hotkey_triggered()
                    user32.TranslateMessage(ctypes.byref(msg))
                    user32.DispatchMessageW(ctypes.byref(msg))
            finally:
                user32.UnregisterHotKey(None, HOTKEY_ID)

        t = _threading.Thread(target=_message_loop, daemon=True, name="winapi-hotkey")
        t.start()
        try:
            return result_q.get(timeout=2.0)
        except _queue.Empty:
            log.error("RegisterHotKey: timed out waiting for registration result")
            return False

    def _hotkey_triggered(self):
        self._trigger_was_hotkey = True
        self.root.after(0, self._toggle)

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _set_state(self, state: str, transcription: str | None = None):
        """Must be called from the tkinter main thread."""
        self._state = state
        self._canvas.itemconfig(self._dot, fill=_COLORS.get(state, "#888888"))

        if state == "recording" and self._continuous:
            label = "Listening...  (press hotkey to stop)"
        elif state == "idle" and not self._continuous:
            label = _LABELS["idle"]
        else:
            label = _LABELS.get(state, state)
        self._status_var.set(label)

        hotkey_label = self._hotkey.replace("+", " + ").upper()
        if self._continuous:
            self._btn.config(text=f"Stop   [ {hotkey_label} ]")
        else:
            self._btn.config(text=f"Start   [ {hotkey_label} ]")

        if transcription is not None:
            self._text.config(state=tk.NORMAL)
            self._text.delete("1.0", tk.END)
            self._text.insert("1.0", transcription)
            self._text.config(state=tk.DISABLED)

    def _toggle(self):
        with self._state_lock:
            state = self._state

        if state == "idle":
            self._continuous = True
            self._trigger_was_hotkey = True  # hotkey never moves focus
            self._begin_recording(beep=True)
        elif self._continuous:
            # User pressed hotkey again — exit continuous mode
            self._continuous = False
            if state == "recording":
                self._end_recording(restart=False, beep=True)
            # If transcribing, _transcribe_worker will see _continuous=False and go idle

    def _begin_recording(self, beep: bool = False):
        self._set_state("recording")
        self._update_conf_display({})  # clear stale confidence info
        # If the user keeps talking, cancel any command that was waiting to fire.
        if self._pending_cmd_id is not None:
            self.root.after_cancel(self._pending_cmd_id)
            self._pending_cmd_id = None
            self._pending_var.set("")
            log.debug("Pending command cancelled — new speech detected")
        if beep:
            threading.Thread(target=_beep_start, daemon=True).start()
        self._recorder.start(
            on_auto_stop=self._on_vad_silence if self._auto_stop_silence_sec > 0 else None,
            silence_timeout_sec=self._auto_stop_silence_sec,
            vad_threshold_db=self._vad_threshold_db,
        )

    def _on_vad_silence(self):
        """Called from recorder thread when silence threshold is reached."""
        self.root.after(0, lambda: self._end_recording(restart=True, beep=False))

    def _end_recording(self, restart: bool = False, beep: bool = False):
        with self._state_lock:
            if self._state != "recording":
                return  # Guard against double-fire
        audio = self._recorder.stop()
        if beep:
            threading.Thread(target=_beep_stop, daemon=True).start()
        self._set_state("transcribing")
        threading.Thread(
            target=self._transcribe_worker, args=(audio, restart), daemon=True
        ).start()

    def _transcribe_worker(self, audio, restart_after: bool):
        displayed: list[str] = []
        from .commands import find_command, run_action

        # Uncertain words accumulate across segments for the whole recording session.
        _CONF_THRESHOLD = 0.75
        uncertain_seen: dict[str, float] = {}   # word -> lowest probability seen

        def on_segment(raw: str, words=None):
            text = apply_substitutions(raw, self._vocab)
            if not text:
                return

            # Track low-confidence words and update the label.
            if words:
                for word, prob in words:
                    if prob < _CONF_THRESHOLD and word:
                        key = word.lower()
                        if key not in uncertain_seen or prob < uncertain_seen[key]:
                            uncertain_seen[key] = prob
                if uncertain_seen:
                    self.root.after(0, lambda uw=dict(uncertain_seen):
                                    self._update_conf_display(uw))

            text_to_type, action = find_command(text)

            if text_to_type:
                displayed.append(text_to_type)
                self.root.after(0, lambda t=" ".join(displayed): self._update_display(t))
                type_text(text_to_type, self._output_method, self._trailing_space,
                          self._keystroke_delay_ms)
                self._last_typed = text_to_type + (" " if self._trailing_space else "")

            if action:
                _CMD_LABELS = {
                    "enter": "Enter ↵", "tab": "Tab ↹",
                    "delete_last": "Delete that", "undo": "Undo",
                    "select_all": "Select all",
                }
                label = _CMD_LABELS.get(action, action)

                def _fire_command(a=action, lt=self._last_typed, disp=displayed):
                    self._pending_cmd_id = None
                    self._pending_var.set("")
                    run_action(a, lt, self._output_method, self._keystroke_delay_ms)
                    if a == "delete_last":
                        self._last_typed = ""
                        if disp:
                            disp.pop()
                        self.root.after(0, lambda: self._update_display(
                            " ".join(disp) if disp else ""))

                # In continuous mode (VAD auto-stop) delay the command so the user
                # can keep talking to cancel it.  On an explicit stop, fire at once.
                if restart_after:
                    _DELAY_MS = 1500
                    self.root.after(0, lambda l=label: self._pending_var.set(
                        f"⏸  {l} — keep talking to cancel"))
                    self._pending_cmd_id = self.root.after(_DELAY_MS, _fire_command)
                else:
                    _fire_command()

        try:
            self._transcriber.transcribe(audio, self._sample_rate, on_segment=on_segment)
        except Exception as exc:
            log.error("Transcription failed: %s", exc)
            self.root.after(0, lambda: self._set_state("idle", transcription=f"[Error: {exc}]"))
            self._continuous = False
            return

        final = " ".join(displayed) if displayed else "(nothing detected)"

        if restart_after and self._continuous:
            # Continuous mode — show result briefly then start listening again
            self.root.after(0, lambda: self._set_state("transcribing", transcription=final))
            self.root.after(0, lambda: self._begin_recording(beep=False))
        else:
            self._continuous = False
            self.root.after(0, lambda: self._set_state("idle", transcription=final))

    def _update_display(self, text: str):
        self._text.config(state=tk.NORMAL)
        self._text.delete("1.0", tk.END)
        self._text.insert("1.0", text)
        self._text.config(state=tk.DISABLED)

    def _copy_transcription(self):
        text = self._text.get("1.0", tk.END).strip()
        if not text or text == "(nothing detected)":
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        # Brief visual feedback — button label changes for 1 second
        self._copy_btn.config(text="Copied!")
        self.root.after(1000, lambda: self._copy_btn.config(text="Copy text"))

    # ------------------------------------------------------------------
    # Confidence / vocab helpers
    # ------------------------------------------------------------------

    def _update_conf_display(self, uncertain: dict[str, float]):
        """Rebuild the clickable uncertain-words bar.  Main thread only."""
        self._conf_text.config(state=tk.NORMAL)
        self._conf_text.delete("1.0", tk.END)
        # Drop old per-word tags
        for tag in list(self._conf_text.tag_names()):
            if tag.startswith("w_"):
                self._conf_text.tag_delete(tag)

        if uncertain:
            self._conf_text.insert(tk.END, "Uncertain (click to fix):  ", "label")
            for i, (word, prob) in enumerate(sorted(uncertain.items(), key=lambda x: x[1])):
                if i:
                    self._conf_text.insert(tk.END, "   ", "label")
                tag = f"w_{i}"
                self._conf_text.tag_configure(tag, foreground="#e67e22", underline=True)
                self._conf_text.tag_bind(tag, "<Button-1>",
                    lambda e, w=word, p=prob: self._show_vocab_add(w, p))
                self._conf_text.tag_bind(tag, "<Enter>",
                    lambda e: self._conf_text.config(cursor="hand2"))
                self._conf_text.tag_bind(tag, "<Leave>",
                    lambda e: self._conf_text.config(cursor="arrow"))
                self._conf_text.insert(tk.END, f'"{word}" ({prob:.0%})', tag)

        self._conf_text.config(state=tk.DISABLED)

    def _show_vocab_add(self, heard: str, prob: float):
        """Pop up a dialog to add *heard* → replacement to the vocabulary config."""
        from .config import load_config, save_config, get_config_path

        win = tk.Toplevel(self.root)
        win.title("Add to vocabulary")
        win.resizable(False, False)
        win.attributes("-topmost", True)
        win.grab_set()

        f = tk.Frame(win, padx=18, pady=14)
        f.pack()

        tk.Label(f, text="Whisper heard:", font=("Segoe UI", 9), anchor="w").grid(
            row=0, column=0, sticky="w", pady=4)
        tk.Label(f, text=f'"{heard}"  ({prob:.0%} confidence)',
                 font=("Segoe UI", 9, "italic"), fg="#e67e22").grid(
            row=0, column=1, sticky="w", padx=(10, 0))

        tk.Label(f, text="Replace with:", font=("Segoe UI", 9), anchor="w").grid(
            row=1, column=0, sticky="w", pady=4)
        replace_var = tk.StringVar(value=heard.title())
        entry = tk.Entry(f, textvariable=replace_var, width=28, font=("Segoe UI", 10))
        entry.grid(row=1, column=1, sticky="w", padx=(10, 0))
        entry.selection_range(0, tk.END)
        entry.focus_set()

        tk.Label(f, text="Section:", font=("Segoe UI", 9), anchor="w").grid(
            row=2, column=0, sticky="w", pady=4)
        section_var = tk.StringVar(value="terminology")
        tk.OptionMenu(f, section_var, "terminology", "names", "unique", "punctuation").grid(
            row=2, column=1, sticky="w", padx=(10, 0))
        tk.Label(f, text="terminology = jargon/acronyms  |  names = people/places",
                 font=("Segoe UI", 7), fg="#888").grid(
            row=3, column=0, columnspan=2, sticky="w")

        def on_save():
            replacement = replace_var.get().strip()
            if not replacement:
                return
            section = section_var.get()
            cfg_path = self._config_path or get_config_path()
            cfg = load_config(cfg_path)
            cfg["vocabulary"].setdefault(section, {})[heard.lower()] = replacement
            save_config(cfg, cfg_path)
            # Apply immediately to the running instance
            self._vocab.setdefault(section, {})[heard.lower()] = replacement
            log.info("Vocabulary: '%s' -> '%s' added to [%s]", heard, replacement, section)
            win.destroy()

        btn_f = tk.Frame(win, pady=8)
        btn_f.pack()
        tk.Button(btn_f, text="Cancel", command=win.destroy, width=10).pack(side="left", padx=4)
        tk.Button(btn_f, text="Add", command=on_save, width=10,
                  default="active").pack(side="left", padx=4)
        win.bind("<Return>", lambda e: on_save())
        win.bind("<Escape>", lambda e: win.destroy())

    # ------------------------------------------------------------------
    # Settings dialog
    # ------------------------------------------------------------------

    def _show_settings(self):
        from .config import load_config, save_config, get_config_path
        from . import ext_server as _ext

        cfg_path = self._config_path or get_config_path()
        cfg = load_config(cfg_path)
        o = cfg["output"]

        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.resizable(False, False)
        win.attributes("-topmost", True)
        win.grab_set()

        f = tk.Frame(win, padx=18, pady=14)
        f.pack()

        def row(label, r):
            tk.Label(f, text=label, anchor="w", font=("Segoe UI", 10)).grid(
                row=r, column=0, sticky="w", pady=5, padx=(0, 12))

        row("Output method:", 0)
        method_var = tk.StringVar(value=o["method"])
        tk.OptionMenu(f, method_var, "auto", "keystroke", "clipboard", "extension").grid(
            row=0, column=1, sticky="w")
        tk.Label(f, text='  "auto" uses extension when CRD is open, keystroke otherwise',
                 font=("Segoe UI", 8), fg="#666").grid(
            row=0, column=2, sticky="w", padx=(4, 0))

        row("Extension port:", 1)
        port_var = tk.StringVar(value=str(o.get("extension_port", 9754)))
        tk.Entry(f, textvariable=port_var, width=8, font=("Segoe UI", 10)).grid(
            row=1, column=1, sticky="w")

        row("Trailing space:", 2)
        space_var = tk.BooleanVar(value=o.get("trailing_space", True))
        tk.Checkbutton(f, variable=space_var).grid(row=2, column=1, sticky="w")

        row("Keystroke delay (ms):", 3)
        delay_var = tk.StringVar(value=str(o.get("keystroke_delay_ms", 10)))
        tk.Entry(f, textvariable=delay_var, width=8, font=("Segoe UI", 10)).grid(
            row=3, column=1, sticky="w")

        def on_save():
            method = method_var.get()
            try:
                port = int(port_var.get())
            except ValueError:
                port = 9754
            trailing = space_var.get()
            try:
                delay = int(delay_var.get())
            except ValueError:
                delay = 10

            cfg["output"]["method"] = method
            cfg["output"]["extension_port"] = port
            cfg["output"]["trailing_space"] = trailing
            cfg["output"]["keystroke_delay_ms"] = delay
            save_config(cfg, cfg_path)

            self._output_method = method
            self._trailing_space = trailing
            self._keystroke_delay_ms = delay

            if method in ("extension", "auto"):
                _ext.start(port)

            log.info("Settings saved: method=%s port=%d trailing=%s delay=%d",
                     method, port, trailing, delay)
            win.destroy()

        btn_f = tk.Frame(win, pady=8)
        btn_f.pack()
        tk.Button(btn_f, text="Cancel", command=win.destroy, width=10).pack(side="left", padx=4)
        tk.Button(btn_f, text="Save", command=on_save, width=10, default="active").pack(side="left", padx=4)
        win.bind("<Return>", lambda _: on_save())
        win.bind("<Escape>", lambda _: win.destroy())

    # ------------------------------------------------------------------

    def run(self):
        self.root.mainloop()
