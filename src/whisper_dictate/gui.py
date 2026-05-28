import threading
import time
import logging
import tkinter as tk
import tkinter.ttk as ttk

from .vocabulary import apply_substitutions
from .output import type_text


log = logging.getLogger(__name__)

_COLORS = {
    "idle":         "#888888",
    "recording":    "#e74c3c",
    "transcribing": "#f5a623",
}
_LABELS = {
    "idle":         "Idle",
    "recording":    "Recording...",
    "transcribing": "Transcribing...",
}

# Dark-mode colour palette
_D = {
    "bg":         "#2b2b2b",   # window / frame backgrounds
    "bg2":        "#333333",   # dropdown menus
    "bg_input":   "#3a3a3a",   # text boxes, entries
    "bg_btn":     "#3f3f3f",   # buttons (normal)
    "bg_btn_act": "#555555",   # buttons (hover/active)
    "title":      "#1e1e1e",   # compact title bar
    "fg":         "#e0e0e0",   # primary text
    "fg_dim":     "#999999",   # secondary / muted text
    "fg_hint":    "#606060",   # very muted hints
    "accent":     "#5294e2",   # blue (pending-command label)
    "orange":     "#e8943a",   # uncertain-word highlight
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
                 idle_stop_sec: float = 30.0, streaming_interval_sec: float = 0.0,
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
        self._idle_stop_sec = idle_stop_sec
        self._stream_interval_sec = streaming_interval_sec

        self._config_path = config_path

        self._state = "idle"
        self._state_lock = threading.Lock()
        self._continuous = False       # True while in continuous dictation mode
        self._trigger_was_hotkey = False
        self._last_typed = ""           # text typed in last segment (for "delete that")
        self._pending_cmd_id: str | None = None   # after() timer id for deferred command
        self._last_speech_time: float = 0.0       # monotonic time of last transcribed speech
        self._cancel_session = False              # set True by "whisper cancel" command
        self._last_transcription = ""             # full text of last completed session (for resend)
        self._drag_ox = 0                         # drag origin for borderless compact window

        # Streaming transcription state  (_stream_interval_sec already set above from param)
        self._stream_typed: str = ""              # concatenation of all streamed text this session
        self._stream_last_sample: int = 0         # recorder sample index up to which we've streamed
        self._stream_timer: threading.Timer | None = None
        self._stream_lock = threading.Lock()
        self._stream_busy = False
        self._drag_oy = 0

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
        self._compact = False
        self.root.configure(bg=_D["bg"])

        # StringVar shared between expanded status label and compact strip
        self._status_var = tk.StringVar(value=_LABELS["idle"])

        # ── Compact horizontal strip (borderless, shown instead of _outer) ───
        # Layout: [✕] [●] [status text ............] [⊞]
        self._compact_frame = tk.Frame(self.root, bg=_D["title"])
        # (not packed yet — shown by _toggle_compact)

        tk.Button(self._compact_frame, text="✕", command=self._on_close,
                  bg=_D["title"], fg=_D["fg_dim"], font=("Segoe UI", 11),
                  relief=tk.FLAT, bd=0, activebackground="#e74c3c",
                  activeforeground="white", padx=8,
                  cursor="arrow").pack(side="left")

        self._compact_canvas = tk.Canvas(self._compact_frame, width=15, height=15,
                                         highlightthickness=0, bg=_D["title"])
        self._compact_canvas.pack(side="left", padx=(5, 0), pady=10)
        self._compact_dot = self._compact_canvas.create_oval(
            1, 1, 14, 14, fill=_COLORS["idle"], outline="")

        self._compact_lbl = tk.Label(
            self._compact_frame, textvariable=self._status_var,
            font=("Segoe UI", 10), bg=_D["title"], fg=_D["fg"],
            width=15, anchor="w",
        )
        self._compact_lbl.pack(side="left", padx=(5, 0))

        tk.Button(self._compact_frame, text="⊞", command=self._toggle_compact,
                  bg=_D["title"], fg=_D["fg_dim"], font=("Segoe UI", 11),
                  relief=tk.FLAT, bd=0, activebackground=_D["bg_btn"],
                  activeforeground=_D["fg"], padx=8).pack(side="right")

        # The whole strip is a drag handle except the two buttons
        for w in (self._compact_frame, self._compact_lbl):
            w.bind("<ButtonPress-1>", self._drag_start)
            w.bind("<B1-Motion>",     self._drag_motion)

        # ── Expanded layout ───────────────────────────────────────────
        self._outer = tk.Frame(self.root, padx=20, pady=16, bg=_D["bg"])
        self._outer.pack()

        # ── Always-visible: dot + status ──────────────────────────────
        self._canvas = tk.Canvas(self._outer, width=52, height=52,
                                 highlightthickness=0, bg=_D["bg"])
        self._canvas.pack()
        self._dot = self._canvas.create_oval(6, 6, 46, 46, fill=_COLORS["idle"], outline="")

        tk.Label(self._outer, textvariable=self._status_var, font=("Segoe UI", 11),
                 bg=_D["bg"], fg=_D["fg"]).pack(pady=(2, 4))

        # Download progress bar — hidden until a model download starts
        _pb_style = ttk.Style()
        _pb_style.theme_use("default")
        _pb_style.configure("WD.Horizontal.TProgressbar",
                            troughcolor=_D["bg_input"],
                            background=_D["accent"],
                            bordercolor=_D["bg"],
                            lightcolor=_D["accent"],
                            darkcolor=_D["accent"])
        self._progress_frame = tk.Frame(self._outer, bg=_D["bg"])
        # (not packed yet)
        self._progress_bar = ttk.Progressbar(
            self._progress_frame, length=220, mode="determinate",
            style="WD.Horizontal.TProgressbar")
        self._progress_bar.pack(side="left", padx=(0, 8))
        self._progress_pct_var = tk.StringVar(value="")
        tk.Label(self._progress_frame, textvariable=self._progress_pct_var,
                 font=("Segoe UI", 8), bg=_D["bg"], fg=_D["fg_dim"],
                 width=5, anchor="w").pack(side="left")

        # ── Collapsible section ────────────────────────────────────────
        self._detail_frame = tk.Frame(self._outer, bg=_D["bg"])
        self._detail_frame.pack()

        hotkey_label = self._hotkey.replace("+", " + ").upper()
        self._btn = tk.Button(
            self._detail_frame,
            text=f"Start   [ {hotkey_label} ]",
            command=self._toggle,
            width=26, height=2,
            font=("Segoe UI", 10),
            bg=_D["bg_btn"], fg=_D["fg"],
            activebackground=_D["bg_btn_act"], activeforeground=_D["fg"],
            relief=tk.FLAT, bd=0,
        )
        self._btn.pack(pady=(4, 14))

        tk.Label(self._detail_frame, text="Last transcription:", anchor="w",
                 font=("Segoe UI", 9), bg=_D["bg"], fg=_D["fg_dim"]).pack(fill="x")
        self._text = tk.Text(
            self._detail_frame,
            height=5, width=46,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("Segoe UI", 10),
            relief=tk.FLAT, bd=0,
            bg=_D["bg_input"], fg=_D["fg"],
            insertbackground=_D["fg"],
        )
        self._text.pack(pady=(2, 0))

        # Pending-command countdown
        self._pending_var = tk.StringVar()
        tk.Label(self._detail_frame, textvariable=self._pending_var,
                 font=("Segoe UI", 8, "italic"),
                 fg=_D["accent"], bg=_D["bg"]).pack(fill="x", pady=(2, 0))

        # Clickable uncertain-words bar
        self._conf_text = tk.Text(
            self._detail_frame, height=2, font=("Segoe UI", 8),
            wrap=tk.WORD, state=tk.DISABLED,
            relief=tk.FLAT, bd=0,
            bg=_D["bg"], fg=_D["fg"],
            cursor="arrow",
        )
        self._conf_text.tag_configure("label", foreground=_D["fg_dim"])
        self._conf_text.pack(fill="x", pady=(2, 0))

        btn_row = tk.Frame(self._detail_frame, bg=_D["bg"])
        btn_row.pack(pady=(6, 0))

        self._copy_btn = tk.Button(
            btn_row, text="Copy text", command=self._copy_transcription,
            font=("Segoe UI", 9), width=12,
            bg=_D["bg_btn"], fg=_D["fg"],
            activebackground=_D["bg_btn_act"], activeforeground=_D["fg"],
            relief=tk.FLAT, bd=0,
        )
        self._copy_btn.pack(side="left", padx=(0, 8))

        tk.Button(
            btn_row, text="Settings", command=self._show_settings,
            font=("Segoe UI", 9), relief=tk.FLAT, bd=0,
            bg=_D["bg"], fg=_D["fg_dim"],
            activebackground=_D["bg_btn"], activeforeground=_D["fg"],
        ).pack(side="left")

        tk.Button(
            btn_row, text="⊟", command=self._toggle_compact,
            font=("Segoe UI", 9), relief=tk.FLAT, bd=0,
            bg=_D["bg"], fg=_D["fg_dim"],
            activebackground=_D["bg_btn"], activeforeground=_D["fg"],
        ).pack(side="left", padx=(8, 0))

    def _toggle_compact(self):
        self._compact = not self._compact
        if self._compact:
            # Hide the full expanded panel and go borderless
            self._outer.pack_forget()
            self.root.withdraw()
            self.root.overrideredirect(True)
            self._compact_frame.pack(fill="x")
            self.root.deiconify()
            self.root.attributes("-topmost", True)
            # Freeze to a slim horizontal strip; enforce a minimum drag width
            self.root.geometry("")
            self.root.update_idletasks()
            w = max(self.root.winfo_width(), 240)
            h = self.root.winfo_height()
            self.root.geometry(f"{w}x{h}")
        else:
            # Restore the full panel
            self._compact_frame.pack_forget()
            self.root.withdraw()
            self.root.overrideredirect(False)
            self._outer.pack()
            self.root.deiconify()
            self.root.attributes("-topmost", True)
            self.root.geometry("")
            self.root.update_idletasks()

    def _drag_start(self, event):
        self._drag_ox = event.x_root - self.root.winfo_x()
        self._drag_oy = event.y_root - self.root.winfo_y()

    def _drag_motion(self, event):
        self.root.geometry(f"+{event.x_root - self._drag_ox}+{event.y_root - self._drag_oy}")

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

    def _on_model_status(self, status: str) -> None:
        """Called from the background model-loading thread."""
        if status == "downloading":
            self.root.after(0, self._show_download_start)
        elif status == "loading":
            self.root.after(0, self._show_loading)
        else:  # ready
            self.root.after(0, self._hide_progress)
            self.root.after(0, lambda: self._set_state("idle"))

    def _show_download_start(self) -> None:
        self._canvas.itemconfig(self._dot, fill=_D["accent"])
        self._compact_canvas.itemconfig(self._compact_dot, fill=_D["accent"])
        self._status_var.set("Downloading model…")
        self._progress_bar["value"] = 0
        self._progress_pct_var.set("0%")
        self._progress_frame.pack(pady=(0, 6))

    def _show_loading(self) -> None:
        self._canvas.itemconfig(self._dot, fill=_D["accent"])
        self._compact_canvas.itemconfig(self._compact_dot, fill=_D["accent"])
        self._status_var.set("Loading model…")
        self._hide_progress()   # no bar needed for cache load

    def _hide_progress(self) -> None:
        self._progress_frame.pack_forget()
        self._progress_pct_var.set("")

    def _on_progress(self, fraction: float) -> None:
        """Called from download thread with 0.0–1.0. Schedules a UI update."""
        self.root.after(0, lambda f=fraction: self._update_progress(f))

    def _update_progress(self, fraction: float) -> None:
        pct = int(fraction * 100)
        self._progress_bar["value"] = pct
        self._progress_pct_var.set(f"{pct}%")
        # Keep compact strip text in sync too
        self._status_var.set(f"Downloading…  {pct}%")

    def _set_state(self, state: str, transcription: str | None = None):
        """Must be called from the tkinter main thread."""
        self._state = state
        color = _COLORS.get(state, "#888888")
        self._canvas.itemconfig(self._dot, fill=color)
        self._compact_canvas.itemconfig(self._compact_dot, fill=color)

        if state == "recording" and self._continuous:
            label = "Listening..."
        elif state == "idle" and not self._continuous:
            label = _LABELS["idle"]
        else:
            label = _LABELS.get(state, state)
        self._status_var.set(label)

        hotkey_label = self._hotkey.replace("+", " + ").upper()
        if self._continuous:
            self._btn.config(text=f"Stop   [ {hotkey_label} ]",
                             bg="#4a2020", activebackground="#5e2828")
        else:
            self._btn.config(text=f"Start   [ {hotkey_label} ]",
                             bg=_D["bg_btn"], activebackground=_D["bg_btn_act"])

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
            self._begin_recording(beep=True, reset_idle=True)
        elif state == "transcribing" and not self._continuous:
            # Previous clip is still transcribing — start the next recording immediately.
            # The in-flight transcription will finish and type its result in the background
            # without touching the state (guarded in _transcribe_worker).
            self._continuous = True
            self._trigger_was_hotkey = True
            self._begin_recording(beep=True, reset_idle=True)
        elif self._continuous:
            # User pressed hotkey again — exit continuous mode
            self._continuous = False
            if state == "recording":
                self._end_recording(restart=False, beep=True)
            # If transcribing, _transcribe_worker will see _continuous=False and go idle

    def _begin_recording(self, beep: bool = False, reset_idle: bool = False):
        self._set_state("recording")
        self._update_conf_display({})  # clear stale confidence info
        if reset_idle:
            self._last_speech_time = time.monotonic()
        # If the user keeps talking, cancel any command that was waiting to fire.
        if self._pending_cmd_id is not None:
            self.root.after_cancel(self._pending_cmd_id)
            self._pending_cmd_id = None
            self._pending_var.set("")
            log.debug("Pending command cancelled — new speech detected")
        # Reset streaming state for this new recording session
        self._stream_typed = ""
        self._stream_last_sample = 0
        if beep:
            threading.Thread(target=_beep_start, daemon=True).start()
        self._recorder.start(
            on_auto_stop=self._on_vad_silence if self._auto_stop_silence_sec > 0 else None,
            silence_timeout_sec=self._auto_stop_silence_sec,
            vad_threshold_db=self._vad_threshold_db,
        )
        # Start streaming ticker if enabled
        log.info("Streaming: interval=%.1fs (0=disabled)", self._stream_interval_sec)
        if self._stream_interval_sec > 0:
            self._schedule_stream_tick()

    def _schedule_stream_tick(self):
        if self._stream_interval_sec > 0:
            self._stream_timer = threading.Timer(
                self._stream_interval_sec, self._stream_tick)
            self._stream_timer.daemon = True
            self._stream_timer.start()

    def _stream_tick(self):
        """Periodic background callback: transcribe the new audio chunk since the last tick."""
        with self._state_lock:
            if self._state != "recording":
                return   # recording ended before timer fired

        with self._stream_lock:
            if self._stream_busy:
                # Previous tick still transcribing — keep cadence by rescheduling now
                self._schedule_stream_tick()
                return
            self._stream_busy = True

        # Reschedule the NEXT tick immediately (before doing the slow transcription work)
        # so the interval is wall-clock-based rather than completion-based.  This keeps
        # the cadence steady regardless of how long each transcription takes.
        self._schedule_stream_tick()

        def _work():
            try:
                with self._state_lock:
                    if self._state != "recording":
                        return

                # Take a snapshot of ALL audio captured so far, then slice off the
                # portion we haven't transcribed yet.  This gives us a fixed-size chunk
                # (roughly one interval worth of audio) regardless of how long we've been
                # recording, so transcription time stays constant tick-to-tick.
                full_audio = self._recorder.snapshot()
                offset = self._stream_last_sample
                chunk = full_audio[offset:]

                chunk_dur = len(chunk) / self._sample_rate if self._sample_rate else 0
                # Require at least 1.5 s of new audio.  Very short chunks are more prone
                # to Whisper hallucinations (e.g. "Thanks for watching") triggered by
                # quiet background audio barely above the silence gate.
                min_dur = max(1.5, self._stream_interval_sec * 0.5)
                log.info("Stream tick: chunk=%.1fs (offset=%d, min=%.1fs)",
                         chunk_dur, offset, min_dur)
                if chunk_dur < min_dur:
                    return   # not enough new audio in this chunk yet

                parts: list[str] = []

                def _on_seg(raw, words=None):
                    text = apply_substitutions(raw, self._vocab)
                    if text:
                        parts.append(text)

                # Pass the previously streamed text as initial_prompt so Whisper has
                # sentence context and doesn't repeat the boundary word from the
                # previous chunk's end when starting a new chunk.
                context_prompt = (
                    (self._transcriber._initial_prompt or "") + " " + self._stream_typed
                ).strip() or None

                self._transcriber.transcribe(
                    chunk, self._sample_rate, on_segment=_on_seg,
                    initial_prompt_override=context_prompt,
                )

                if parts:
                    chunk_text = " ".join(parts)

                    # ── Partial-word / boundary-dash cleanup ──────────────────
                    # Whisper signals a mid-utterance audio cut by appending a
                    # trailing dash to the last word (e.g. "to-", "CH-",
                    # "during-").  Strip that partial word and back the sample
                    # pointer up so the next chunk re-transcribes it in full.
                    # Also strip a bare trailing " -" that sometimes appears
                    # when the chunk ends at a clause boundary.
                    _backed_up = False
                    chunk_words = chunk_text.split()
                    if chunk_words and chunk_words[-1].rstrip().endswith('-'):
                        # Remove the partial token from what we type
                        chunk_words.pop()
                        chunk_text = " ".join(chunk_words)
                        # Back up ~600 ms so the next chunk captures the full word
                        backup = int(0.6 * self._sample_rate)
                        _backed_up = True
                        log.info("Stripped trailing partial-word, will back up 0.6s")

                    # Strip a lone trailing dash left after joining segments
                    if chunk_text.endswith(' -') or chunk_text == '-':
                        chunk_text = chunk_text.rstrip(' -').strip()
                        _backed_up = True

                    if not chunk_text:
                        # Entire chunk was a partial word — don't advance the
                        # pointer so the next tick retries with more audio
                        log.info("Stream tick: chunk was entirely partial, skipping")
                        return

                    # ── Sample-pointer advance ────────────────────────────────
                    # Use the end-time of the last COMPLETE word Whisper reported
                    # rather than the raw snapshot length, so we never start the
                    # next chunk mid-syllable.
                    last_word_end = self._transcriber._last_word_end_sec
                    if last_word_end > 0:
                        pad = int(0.08 * self._sample_rate)   # 80 ms safety pad
                        advance = min(
                            offset + int(last_word_end * self._sample_rate) + pad,
                            len(full_audio),
                        )
                    else:
                        advance = len(full_audio)

                    if _backed_up:
                        backup = int(0.6 * self._sample_rate)
                        advance = max(offset, advance - backup)

                    type_text(chunk_text, self._output_method,
                              trailing_space=self._trailing_space,
                              keystroke_delay_ms=self._keystroke_delay_ms)
                    self._stream_typed = (
                        (self._stream_typed + " " + chunk_text).strip()
                        if self._stream_typed else chunk_text
                    )
                    self._stream_last_sample = advance
                    self._last_typed = chunk_text + (" " if self._trailing_space else "")
                    log.info("Streaming: typed %r  last_word_end=%.2fs  advance=%d",
                             chunk_text, last_word_end, advance)
                else:
                    log.info("Stream tick: chunk produced no segments (silence?)")

            except Exception as exc:
                log.error("Stream tick error: %s", exc)
            finally:
                with self._stream_lock:
                    self._stream_busy = False

        threading.Thread(target=_work, daemon=True).start()

    def _on_vad_silence(self):
        """Called from recorder thread when silence threshold is reached."""
        self.root.after(0, lambda: self._end_recording(restart=True, beep=False))

    def _end_recording(self, restart: bool = False, beep: bool = False):
        # Cancel the streaming timer immediately so no new tick starts
        if self._stream_timer is not None:
            self._stream_timer.cancel()
            self._stream_timer = None

        with self._state_lock:
            if self._state != "recording":
                return  # Guard against double-fire

        # Stop the recorder right away so no extra silence is captured while
        # we wait for any in-flight streaming tick to finish.
        audio = self._recorder.stop()

        if beep:
            threading.Thread(target=_beep_stop, daemon=True).start()
        self._set_state("transcribing")
        threading.Thread(
            target=self._transcribe_worker,
            args=(audio, restart), daemon=True
        ).start()

    def _transcribe_worker(self, audio, restart_after: bool):
        # If a streaming tick is still running (race against end-of-recording), wait
        # for it to finish before we start the final transcription.  This is safe here
        # because we are already in a background thread — the main thread is never blocked.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            with self._stream_lock:
                if not self._stream_busy:
                    break
            time.sleep(0.05)

        # Read streaming state NOW (after waiting) so we see the completed tick's results.
        stream_typed = self._stream_typed            # text typed by streaming ticks
        stream_last_sample = self._stream_last_sample  # sample index up to which streaming covered

        # With chunk-based streaming, the final pass only needs to handle the TAIL —
        # the audio recorded after the last streaming tick completed.  There is nothing
        # to deduplicate; the tail is genuinely new audio.
        if stream_last_sample > 0 and stream_last_sample < len(audio):
            tail_audio = audio[stream_last_sample:]
            tail_dur = len(tail_audio) / self._sample_rate if self._sample_rate else 0
            log.info("Final pass: transcribing %.1fs tail (offset=%d)", tail_dur, stream_last_sample)
        elif stream_last_sample >= len(audio):
            # Streaming covered everything — nothing left for the final pass
            tail_audio = None
            log.info("Final pass: streaming covered all audio, tail is empty")
        else:
            # No streaming was active — transcribe the full clip
            tail_audio = audio
            log.info("Final pass: no streaming, transcribing full %.1fs clip",
                     len(audio) / self._sample_rate if self._sample_rate else 0)

        # Seed the display with whatever streaming already typed so the final result
        # shows the complete session (streamed prefix + tail), not just the tail.
        displayed: list[str] = [stream_typed] if stream_typed else []

        from .commands import find_command, run_action

        # Uncertain words accumulate across segments for the whole recording session.
        _CONF_THRESHOLD = 0.75
        uncertain_seen: dict[str, float] = {}   # word -> lowest probability seen

        def on_segment(raw: str, words=None):
            if self._cancel_session:
                return   # already cancelled — ignore remaining segments
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

            if action == "cancel":
                log.info("Command: cancel — discarding segment and stopping")
                self._cancel_session = True
                return   # don't type text_to_type; _transcribe_worker checks flag after

            if action == "resend":
                if self._last_transcription:
                    log.info("Command: resend — retyping %r", self._last_transcription)
                    type_text(self._last_transcription, self._output_method,
                              self._trailing_space, self._keystroke_delay_ms)
                else:
                    log.debug("Command: resend — nothing to resend")
                return   # discard text_to_type (don't also type any prefix)

            if text_to_type:
                self._last_speech_time = time.monotonic()  # reset idle timer on real speech

                # Every segment from the final pass is genuinely new audio — type it all.
                # (Chunk-based streaming means there is never any overlap to skip.)
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

        if tail_audio is not None and len(tail_audio) > 0:
            try:
                self._transcriber.transcribe(tail_audio, self._sample_rate, on_segment=on_segment)
            except Exception as exc:
                log.error("Transcription failed: %s", exc)
                self.root.after(0, lambda: self._set_state("idle", transcription=f"[Error: {exc}]"))
                self._continuous = False
                return

        # "whisper cancel" command was detected — discard everything and go idle
        if self._cancel_session:
            self._cancel_session = False
            self._continuous = False
            threading.Thread(target=_beep_stop, daemon=True).start()
            self.root.after(0, lambda: self._set_state("idle", transcription="(cancelled)"))
            return

        final = " ".join(displayed) if displayed else "(nothing detected)"
        if displayed:
            self._last_transcription = " ".join(displayed)

        if restart_after and self._continuous:
            # Check idle-stop: if no real speech for idle_stop_sec, quit completely.
            if (self._idle_stop_sec > 0 and self._last_speech_time > 0 and
                    time.monotonic() - self._last_speech_time >= self._idle_stop_sec):
                log.info("Idle timeout (%.0fs without speech) — stopping", self._idle_stop_sec)
                self._continuous = False
                threading.Thread(target=_beep_stop, daemon=True).start()
                self.root.after(0, lambda: self._set_state("idle",
                                transcription="(stopped — no speech detected)"))
            else:
                # Continuous mode — show result briefly then start listening again
                self.root.after(0, lambda: self._set_state("transcribing", transcription=final))
                self.root.after(0, lambda: self._begin_recording(beep=False))
        else:
            # Guard: if the user started a new recording while we were transcribing,
            # don't clobber the "recording" state — just update the text display.
            with self._state_lock:
                already_recording = (self._state == "recording")
            if already_recording:
                log.debug("Transcription done but new recording active — skipping idle transition")
                if final and final != "(nothing detected)":
                    self.root.after(0, lambda: self._update_display(final))
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
                self._conf_text.tag_configure(tag, foreground=_D["orange"], underline=True)
                self._conf_text.tag_bind(tag, "<Button-1>",
                    lambda e, w=word, p=prob: self._show_vocab_add(w, p))
                self._conf_text.tag_bind(tag, "<Enter>",
                    lambda e: self._conf_text.config(cursor="hand2"))
                self._conf_text.tag_bind(tag, "<Leave>",
                    lambda e: self._conf_text.config(cursor="arrow"))
                self._conf_text.insert(tk.END, f'"{word}" ({prob:.0%})', tag)

        self._conf_text.config(state=tk.DISABLED)

    def _show_vocab_add(self, heard: str, prob: float):
        """Pop up the add/edit dialog pre-filled with a low-confidence word."""
        self._show_vocab_entry(
            parent=self.root,
            heard_init=heard,
            replace_init=heard.title(),
            section_init="names",
            subtitle=f'Whisper heard "{heard}"  ({prob:.0%} confidence)',
        )

    # ------------------------------------------------------------------
    # Vocabulary manager + add/edit dialog
    # ------------------------------------------------------------------

    def _show_vocab_entry(self, parent, heard_init: str = "", replace_init: str = "",
                          section_init: str = "names", subtitle: str = "",
                          delete_heard: str = "", delete_section: str = "",
                          on_done=None):
        """Add or edit a single vocabulary entry.

        heard_init / replace_init — pre-filled values.
        delete_heard / delete_section — if set, that key is removed before saving
        (used for in-place edits where the heard phrase may change).
        on_done — callback fired after a successful save (e.g. to refresh a list).
        """
        from .config import load_config, save_config, get_config_path
        cfg_path = self._config_path or get_config_path()

        is_edit = bool(delete_heard)
        win = tk.Toplevel(parent)
        win.title("Edit vocabulary entry" if is_edit else "Add vocabulary entry")
        win.resizable(False, False)
        win.attributes("-topmost", True)
        win.configure(bg=_D["bg"])
        win.grab_set()

        f = tk.Frame(win, padx=18, pady=14, bg=_D["bg"])
        f.pack()

        def _lbl(text, r, col=0, **kw):
            tk.Label(f, text=text, bg=_D["bg"], fg=_D["fg"],
                     font=("Segoe UI", 9), anchor="w", **kw).grid(
                row=r, column=col, sticky="w", pady=4, padx=(0, 12 if col == 0 else 0))

        def _entry_widget(var, r):
            e = tk.Entry(f, textvariable=var, width=34, font=("Segoe UI", 10),
                         bg=_D["bg_input"], fg=_D["fg"], insertbackground=_D["fg"],
                         relief=tk.FLAT, highlightthickness=1,
                         highlightbackground=_D["bg_btn"])
            e.grid(row=r, column=1, sticky="w")
            return e

        row = 0
        if subtitle:
            tk.Label(f, text=subtitle, font=("Segoe UI", 9, "italic"),
                     fg=_D["orange"], bg=_D["bg"]).grid(
                row=row, column=0, columnspan=2, sticky="w", pady=(0, 8))
            row += 1

        _lbl("When Whisper says:", row)
        heard_var = tk.StringVar(value=heard_init)
        heard_entry = _entry_widget(heard_var, row)
        row += 1

        _lbl("Type instead:", row)
        replace_var = tk.StringVar(value=replace_init)
        replace_entry = _entry_widget(replace_var, row)
        row += 1

        _lbl("Section:", row)
        section_var = tk.StringVar(value=section_init)
        om = tk.OptionMenu(f, section_var, "names", "terminology", "unique", "punctuation")
        om.config(bg=_D["bg_btn"], fg=_D["fg"], relief=tk.FLAT, highlightthickness=0,
                  activebackground=_D["bg_btn_act"], activeforeground=_D["fg"])
        om["menu"].config(bg=_D["bg2"], fg=_D["fg"],
                          activebackground=_D["accent"], activeforeground="white")
        om.grid(row=row, column=1, sticky="w")
        row += 1

        tk.Label(f,
                 text="names = people/places  ·  terminology = jargon  ·  unique = multi-word / @ symbols",
                 font=("Segoe UI", 7), fg=_D["fg_hint"], bg=_D["bg"]).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 4))
        row += 1

        # Focus the first empty field
        if not heard_init:
            heard_entry.focus_set()
        else:
            replace_entry.focus_set()
            replace_entry.selection_range(0, tk.END)

        def on_save():
            heard = heard_var.get().strip().lower()
            replacement = replace_var.get().strip()
            section = section_var.get()
            if not heard or not replacement:
                return

            cfg = load_config(cfg_path)
            vocab = cfg.setdefault("vocabulary", {})

            # Remove old key if we're editing (the heard phrase may have changed)
            if delete_heard and delete_section:
                vocab.get(delete_section, {}).pop(delete_heard.lower(), None)
                self._vocab.get(delete_section, {}).pop(delete_heard.lower(), None)

            vocab.setdefault(section, {})[heard] = replacement
            save_config(cfg, cfg_path)
            self._vocab.setdefault(section, {})[heard] = replacement
            log.info("Vocabulary: '%s' -> '%s' [%s]", heard, replacement, section)
            win.destroy()
            if on_done:
                on_done()

        btn_f = tk.Frame(win, pady=8, bg=_D["bg"])
        btn_f.pack()

        def _btn(text, cmd):
            return tk.Button(btn_f, text=text, command=cmd, width=10,
                             bg=_D["bg_btn"], fg=_D["fg"],
                             activebackground=_D["bg_btn_act"], activeforeground=_D["fg"],
                             relief=tk.FLAT, bd=0)

        _btn("Cancel", win.destroy).pack(side="left", padx=4)
        _btn("Save", on_save).pack(side="left", padx=4)
        win.bind("<Return>", lambda e: on_save())
        win.bind("<Escape>", lambda e: win.destroy())

    def _show_vocab_manager(self):
        """Full vocabulary manager: list all entries, add / edit / delete."""
        from .config import load_config, save_config, get_config_path
        cfg_path = self._config_path or get_config_path()

        win = tk.Toplevel(self.root)
        win.title("Vocabulary")
        win.resizable(True, True)
        win.attributes("-topmost", True)
        win.configure(bg=_D["bg"])
        win.grab_set()
        win.geometry("560x360")

        # ── Treeview ──────────────────────────────────────────────────
        tree_frame = tk.Frame(win, bg=_D["bg"])
        tree_frame.pack(fill="both", expand=True, padx=12, pady=(10, 0))

        cols = ("section", "heard", "replacement")
        tree = ttk.Treeview(tree_frame, columns=cols, show="headings",
                            selectmode="browse", height=13)
        tree.heading("section",     text="Section",            anchor="w")
        tree.heading("heard",       text="When Whisper says",  anchor="w")
        tree.heading("replacement", text="Type instead",       anchor="w")
        tree.column("section",     width=90,  minwidth=70,  anchor="w")
        tree.column("heard",       width=200, minwidth=100, anchor="w")
        tree.column("replacement", width=200, minwidth=100, anchor="w")

        _ts = ttk.Style()
        _ts.configure("Vocab.Treeview",
                       background=_D["bg2"], fieldbackground=_D["bg2"],
                       foreground=_D["fg"], rowheight=22, borderwidth=0)
        _ts.configure("Vocab.Treeview.Heading",
                       background=_D["bg_btn"], foreground=_D["fg_dim"],
                       relief="flat")
        _ts.map("Vocab.Treeview",
                background=[("selected", _D["accent"])],
                foreground=[("selected", "white")])
        tree.configure(style="Vocab.Treeview")

        sb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # ── Populate ──────────────────────────────────────────────────
        _SECTION_ORDER = ["unique", "names", "terminology", "punctuation"]

        def _reload():
            tree.delete(*tree.get_children())
            cfg = load_config(cfg_path)
            vocab = cfg.get("vocabulary", {})
            for sec in _SECTION_ORDER:
                entries = vocab.get(sec, {})
                if isinstance(entries, dict):
                    for heard_key, repl in sorted(entries.items()):
                        tree.insert("", "end", values=(sec, heard_key, repl))

        _reload()

        # ── Buttons ───────────────────────────────────────────────────
        btn_frame = tk.Frame(win, bg=_D["bg"], pady=8, padx=12)
        btn_frame.pack(fill="x")

        def _btn(parent, text, cmd, **kw):
            return tk.Button(parent, text=text, command=cmd, width=10,
                             bg=_D["bg_btn"], fg=_D["fg"],
                             activebackground=_D["bg_btn_act"], activeforeground=_D["fg"],
                             relief=tk.FLAT, bd=0, **kw)

        def on_add():
            self._show_vocab_entry(win, on_done=_reload)

        def on_edit():
            sel = tree.selection()
            if not sel:
                return
            sec, heard_key, repl = tree.item(sel[0], "values")
            self._show_vocab_entry(win,
                                   heard_init=heard_key,
                                   replace_init=repl,
                                   section_init=sec,
                                   delete_heard=heard_key,
                                   delete_section=sec,
                                   on_done=_reload)

        def on_delete():
            sel = tree.selection()
            if not sel:
                return
            sec, heard_key, _ = tree.item(sel[0], "values")
            cfg = load_config(cfg_path)
            cfg.get("vocabulary", {}).get(sec, {}).pop(heard_key, None)
            save_config(cfg, cfg_path)
            self._vocab.get(sec, {}).pop(heard_key, None)
            log.info("Vocabulary: deleted '%s' from [%s]", heard_key, sec)
            _reload()

        tree.bind("<Double-1>", lambda _e: on_edit())
        tree.bind("<Delete>",   lambda _e: on_delete())

        _btn(btn_frame, "+ Add",   on_add).pack(side="left", padx=(0, 4))
        _btn(btn_frame, "Edit",    on_edit).pack(side="left", padx=4)
        _btn(btn_frame, "Delete",  on_delete).pack(side="left", padx=4)
        _btn(btn_frame, "Close",   win.destroy).pack(side="right", padx=(4, 0))

        tk.Label(btn_frame,
                 text="Double-click a row to edit  ·  Delete key removes selected",
                 font=("Segoe UI", 7), fg=_D["fg_hint"], bg=_D["bg"]).pack(
            side="left", padx=8)

    # ------------------------------------------------------------------
    # Settings dialog
    # ------------------------------------------------------------------

    def _show_settings(self):
        import threading as _threading
        from .config import load_config, save_config, get_config_path
        from .transcriber import MODEL_OPTIONS, MODEL_SIZES, _model_is_cached, cuda_available
        from . import ext_server as _ext

        cfg_path = self._config_path or get_config_path()
        cfg = load_config(cfg_path)
        o = cfg["output"]
        w = cfg.get("whisper", {})

        _cuda_ok = cuda_available()

        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.resizable(False, False)
        win.attributes("-topmost", True)
        win.configure(bg=_D["bg"])
        win.grab_set()

        f = tk.Frame(win, padx=18, pady=14, bg=_D["bg"])
        f.pack()

        def row(label, r):
            tk.Label(f, text=label, anchor="w", font=("Segoe UI", 10),
                     bg=_D["bg"], fg=_D["fg"]).grid(
                row=r, column=0, sticky="w", pady=5, padx=(0, 12))

        def hint_lbl(r):
            """Return a Label in column 2 that callers can configure later."""
            lbl = tk.Label(f, text="", font=("Segoe UI", 8),
                           fg=_D["fg_hint"], bg=_D["bg"])
            lbl.grid(row=r, column=2, sticky="w", padx=(4, 0))
            return lbl

        def hint(text, r):
            tk.Label(f, text=text, font=("Segoe UI", 8),
                     fg=_D["fg_hint"], bg=_D["bg"]).grid(
                row=r, column=2, sticky="w", padx=(4, 0))

        def dark_om(var, *options):
            om = tk.OptionMenu(f, var, *options)
            om.config(bg=_D["bg_btn"], fg=_D["fg"], relief=tk.FLAT, highlightthickness=0,
                      activebackground=_D["bg_btn_act"], activeforeground=_D["fg"])
            om["menu"].config(bg=_D["bg2"], fg=_D["fg"],
                              activebackground=_D["accent"], activeforeground="white")
            return om

        def dark_entry(var, **kw):
            return tk.Entry(f, textvariable=var, font=("Segoe UI", 10),
                            bg=_D["bg_input"], fg=_D["fg"], insertbackground=_D["fg"],
                            relief=tk.FLAT, highlightthickness=1,
                            highlightbackground=_D["bg_btn"], **kw)

        # ── Output settings ───────────────────────────────────────────
        row("Output method:", 0)
        method_var = tk.StringVar(value=o["method"])
        dark_om(method_var, "auto", "keystroke", "clipboard", "extension").grid(
            row=0, column=1, sticky="w")
        hint('  "auto" uses extension when CRD is open, keystroke otherwise', 0)

        row("Extension port:", 1)
        port_var = tk.StringVar(value=str(o.get("extension_port", 9754)))
        dark_entry(port_var, width=8).grid(row=1, column=1, sticky="w")

        row("Trailing space:", 2)
        space_var = tk.BooleanVar(value=o.get("trailing_space", True))
        tk.Checkbutton(f, variable=space_var,
                       bg=_D["bg"], fg=_D["fg"],
                       selectcolor=_D["bg_input"],
                       activebackground=_D["bg"], activeforeground=_D["fg"]).grid(
            row=2, column=1, sticky="w")

        row("Keystroke delay (ms):", 3)
        delay_var = tk.StringVar(value=str(o.get("keystroke_delay_ms", 10)))
        dark_entry(delay_var, width=8).grid(row=3, column=1, sticky="w")

        row("Idle stop (sec):", 4)
        idle_stop_var = tk.StringVar(
            value=str(cfg.get("audio", {}).get("idle_stop_sec", 30.0)))
        dark_entry(idle_stop_var, width=8).grid(row=4, column=1, sticky="w")
        hint("  stop mic after this many seconds without speech  (0 = never)", 4)

        row("Streaming interval (sec):", 5)
        stream_var = tk.StringVar(
            value=str(cfg.get("audio", {}).get("streaming_interval_sec", 0.0)))
        dark_entry(stream_var, width=8).grid(row=5, column=1, sticky="w")
        hint("  type progressively while speaking  (0 = off;  3–5 = recommended)", 5)

        # ── Vocabulary section ────────────────────────────────────────
        tk.Label(f, text="  Vocabulary", font=("Segoe UI", 8, "bold"),
                 bg=_D["bg"], fg=_D["fg_dim"]).grid(
            row=6, column=0, columnspan=3, sticky="w", pady=(12, 2))
        tk.Frame(f, bg=_D["bg_btn"], height=1).grid(
            row=7, column=0, columnspan=3, sticky="ew", pady=(0, 6))

        def open_vocab_manager():
            win.grab_release()
            self._show_vocab_manager()

        tk.Button(f, text="Manage Vocabulary…",
                  command=open_vocab_manager,
                  bg=_D["bg_btn"], fg=_D["fg"],
                  activebackground=_D["bg_btn_act"], activeforeground=_D["fg"],
                  relief=tk.FLAT, bd=0, padx=10).grid(
            row=8, column=0, columnspan=2, sticky="w", pady=(0, 4))
        hint("  add names, phrases, @ symbols, or any custom substitution", 8)

        # ── Section divider ───────────────────────────────────────────
        tk.Label(f, text="  Whisper model", font=("Segoe UI", 8, "bold"),
                 bg=_D["bg"], fg=_D["fg_dim"]).grid(
            row=9, column=0, columnspan=3, sticky="w", pady=(12, 2))
        tk.Frame(f, bg=_D["bg_btn"], height=1).grid(
            row=10, column=0, columnspan=3, sticky="ew", pady=(0, 6))

        # ── Model selector ────────────────────────────────────────────
        row("Model:", 11)
        current_model = w.get("model", "medium.en")
        model_var = tk.StringVar(value=current_model)
        dark_om(model_var, *MODEL_OPTIONS).grid(row=11, column=1, sticky="w")
        model_hint = hint_lbl(11)

        def _update_model_hint(*_):
            m = model_var.get()
            cached = _model_is_cached(m)
            size = MODEL_SIZES.get(m, "")
            if cached:
                model_hint.config(text="  ✓ downloaded", fg="#5a9e5a")
            else:
                model_hint.config(text=f"  {size} — will download on save", fg=_D["orange"])

        model_var.trace_add("write", _update_model_hint)
        _update_model_hint()   # set initial state

        # ── Device selector ───────────────────────────────────────────
        row("Device:", 12)
        _DEVICE_LABELS = {"cuda": "GPU (CUDA)", "cpu": "CPU"}
        _DEVICE_VALUES = {"GPU (CUDA)": "cuda", "CPU": "cpu"}
        current_device = w.get("device", "cuda")
        device_label_var = tk.StringVar(
            value=_DEVICE_LABELS.get(current_device, "GPU (CUDA)"))

        device_om = dark_om(device_label_var, "GPU (CUDA)", "CPU")
        device_om.grid(row=12, column=1, sticky="w")

        if not _cuda_ok:
            # Show GPU option but grayed out so users know it exists
            device_om["menu"].entryconfig(0, state="disabled",
                                          foreground=_D["fg_hint"])
            device_label_var.set("CPU")
            hint("  no CUDA GPU detected — install nvidia-cublas-cu12 to enable", 12)
        else:
            hint("  GPU is ~10–40× faster than CPU for transcription", 12)

        # ── Save / Cancel ─────────────────────────────────────────────
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
            try:
                idle_stop = float(idle_stop_var.get())
                if idle_stop < 0:
                    idle_stop = 0.0
            except ValueError:
                idle_stop = 30.0
            try:
                stream_interval = float(stream_var.get())
                if stream_interval < 0:
                    stream_interval = 0.0
            except ValueError:
                stream_interval = 0.0

            new_model  = model_var.get()
            new_device = _DEVICE_VALUES.get(device_label_var.get(), "cuda")
            new_compute = "float16" if new_device == "cuda" else "int8"

            model_changed = (new_model  != w.get("model",        "medium.en") or
                             new_device != w.get("device",        "cuda"))

            cfg["output"]["method"]           = method
            cfg["output"]["extension_port"]   = port
            cfg["output"]["trailing_space"]   = trailing
            cfg["output"]["keystroke_delay_ms"] = delay
            cfg.setdefault("audio", {})["idle_stop_sec"] = idle_stop
            cfg["audio"]["streaming_interval_sec"] = stream_interval
            cfg.setdefault("whisper", {})["model"]        = new_model
            cfg["whisper"]["device"]                      = new_device
            cfg["whisper"]["compute_type"]                = new_compute
            save_config(cfg, cfg_path)

            self._output_method      = method
            self._trailing_space     = trailing
            self._keystroke_delay_ms = delay
            self._idle_stop_sec      = idle_stop
            self._stream_interval_sec = stream_interval

            if method in ("extension", "auto"):
                _ext.start(port)

            if model_changed:
                log.info("Model changed to %s on %s — reloading", new_model, new_device)
                _threading.Thread(
                    target=lambda: self._transcriber.reload(
                        model_name=new_model,
                        device=new_device,
                        compute_type=new_compute,
                        on_status=self._on_model_status,
                        on_progress=self._on_progress,
                    ),
                    daemon=True,
                ).start()

            log.info("Settings saved: method=%s model=%s device=%s", method, new_model, new_device)
            win.destroy()

        btn_f = tk.Frame(win, pady=8, bg=_D["bg"])
        btn_f.pack()

        def _dbtn(text, cmd, **kw):
            return tk.Button(btn_f, text=text, command=cmd, width=10,
                             bg=_D["bg_btn"], fg=_D["fg"],
                             activebackground=_D["bg_btn_act"], activeforeground=_D["fg"],
                             relief=tk.FLAT, bd=0, **kw)

        _dbtn("Cancel", win.destroy).pack(side="left", padx=4)
        _dbtn("Save", on_save).pack(side="left", padx=4)
        win.bind("<Return>", lambda _: on_save())
        win.bind("<Escape>", lambda _: win.destroy())

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # UI state persistence (position + compact mode)
    # ------------------------------------------------------------------

    def _ui_state_path(self):
        from .config import get_config_path
        base = self._config_path or get_config_path()
        return base.parent / "ui_state.json"

    def _load_ui_state(self) -> dict:
        import json
        try:
            p = self._ui_state_path()
            if p.exists():
                with open(p) as f:
                    return json.load(f)
        except Exception as exc:
            log.debug("Could not load UI state: %s", exc)
        return {}

    def _save_ui_state(self):
        import json
        try:
            p = self._ui_state_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "x": self.root.winfo_x(),
                "y": self.root.winfo_y(),
                "compact": self._compact,
            }
            with open(p, "w") as f:
                json.dump(state, f)
            log.debug("UI state saved: %s", state)
        except Exception as exc:
            log.warning("Could not save UI state: %s", exc)

    def _on_close(self):
        self._save_ui_state()
        self.root.destroy()

    # ------------------------------------------------------------------

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        ui = self._load_ui_state()
        if "x" in ui and "y" in ui:
            self.root.geometry(f"+{ui['x']}+{ui['y']}")
        if ui.get("compact"):
            self.root.after(100, self._toggle_compact)   # after first draw

        self.root.mainloop()
