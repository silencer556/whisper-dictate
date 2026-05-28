import logging
import math
import re
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Returned when audio is gated out (silence / too short) so the caller
# can distinguish "nothing to type" from an actual empty transcription.
GATED = ""

# Public sizes table (approximate download size, shown in the settings UI)
MODEL_SIZES: dict[str, str] = {
    "tiny.en":   "75 MB",
    "base.en":   "145 MB",
    "small.en":  "465 MB",
    "medium.en": "1.5 GB",
    "large-v2":  "3 GB",
    "large-v3":  "3 GB",
}

# Ordered list for the settings dropdown (best first)
MODEL_OPTIONS = ["large-v3", "large-v2", "medium.en", "small.en", "base.en", "tiny.en"]


def _model_is_cached(model_name: str) -> bool:
    """Return True if the model is already in the HuggingFace disk cache."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    return (cache_dir / f"models--Systran--faster-whisper-{model_name}").exists()


def cuda_available() -> bool:
    """Return True if ctranslate2 can use CUDA on this machine."""
    try:
        import ctranslate2
        return bool(ctranslate2.get_supported_compute_types("cuda"))
    except Exception:
        return False


def _rms_db(audio: np.ndarray) -> float:
    rms = math.sqrt(float(np.mean(audio ** 2)))
    if rms == 0:
        return -math.inf
    return 20 * math.log10(rms)


class Transcriber:
    def __init__(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        initial_prompt: str,
        min_duration_sec: float,
        silence_threshold_db: float,
    ):
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._initial_prompt = initial_prompt or None
        self._min_duration_sec = min_duration_sec
        self._silence_threshold_db = silence_threshold_db

        self._model = None
        self._model_lock = threading.Lock()
        self._load_failed = False
        self._last_word_end_sec: float = 0.0   # end-time of last word in most recent transcription

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _is_cached(self) -> bool:
        """Return True if the model files are already in the HuggingFace disk cache."""
        from pathlib import Path
        return _model_is_cached(self._model_name)

    def reload(self, model_name: str = None, device: str = None,
               compute_type: str = None, on_status: callable = None,
               on_progress: callable = None) -> None:
        """Swap to a different model/device without restarting. Blocking — call from a thread."""
        with self._model_lock:
            self._model = None
            self._load_failed = False
            if model_name is not None:
                self._model_name = model_name
            if device is not None:
                self._device = device
            if compute_type is not None:
                self._compute_type = compute_type
        self.load(on_status=on_status, on_progress=on_progress)

    def load(self, on_status: callable = None,
             on_progress: callable = None) -> None:
        """Load the model synchronously. Safe to call from a background thread.

        on_status(status) — called with "downloading", "loading", or "ready"
        on_progress(fraction) — called with 0.0–1.0 during download (optional)
        """
        if self._model is not None or self._load_failed:
            if on_status:
                on_status("ready")
            return

        cached = self._is_cached()
        if on_status:
            on_status("downloading" if not cached else "loading")

        with self._model_lock:
            if self._model is not None or self._load_failed:
                if on_status:
                    on_status("ready")
                return
            self._load_model(on_progress=on_progress if not cached else None)

        if on_status:
            on_status("ready")

    def _download_with_progress(self, on_progress: callable) -> None:
        """Pre-download model files via huggingface_hub with byte-level progress."""
        try:
            from huggingface_hub import snapshot_download
            from tqdm import tqdm as _BaseTqdm

            class _ProgressTqdm(_BaseTqdm):
                """tqdm subclass that forwards n/total to on_progress."""
                _cb: callable = None   # set before use, cleared after

                def update(self, n=1):
                    super().update(n)
                    if _ProgressTqdm._cb and self.total and self.total > 0:
                        _ProgressTqdm._cb(min(self.n / self.total, 1.0))

            _ProgressTqdm._cb = on_progress
            try:
                snapshot_download(
                    f"Systran/faster-whisper-{self._model_name}",
                    tqdm_class=_ProgressTqdm,
                )
            finally:
                _ProgressTqdm._cb = None

        except Exception as exc:
            # Non-fatal: WhisperModel will attempt its own download as fallback
            log.warning("Progress download failed (%s) — WhisperModel will retry", exc)

    def _load_model(self, on_progress: callable = None) -> None:
        from faster_whisper import WhisperModel

        device = self._device
        compute_type = self._compute_type

        # If not yet cached and we have a progress callback, download first
        if on_progress and not self._is_cached():
            self._download_with_progress(on_progress)
            on_progress(1.0)   # ensure bar reaches 100% before "loading" begins

        log.info("Loading Whisper model '%s' on %s (%s)…", self._model_name, device, compute_type)
        t0 = time.monotonic()
        try:
            self._model = WhisperModel(self._model_name, device=device, compute_type=compute_type)
        except Exception as exc:
            if device == "cuda":
                log.warning("CUDA load failed (%s) — falling back to CPU/int8", exc)
                try:
                    self._model = WhisperModel(self._model_name, device="cpu", compute_type="int8")
                    self._device = "cpu"
                    self._compute_type = "int8"
                except Exception as exc2:
                    log.error("CPU fallback also failed: %s", exc2)
                    self._load_failed = True
                    return
            else:
                log.error("Failed to load model: %s", exc)
                self._load_failed = True
                return

        elapsed = time.monotonic() - t0
        log.info("Model loaded in %.1fs (device=%s, compute=%s)", elapsed, self._device, self._compute_type)

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(self, audio: np.ndarray, sample_rate: int,
                   on_segment: callable = None,
                   initial_prompt_override: str | None = None,
                   min_duration_override: float | None = None,
                   silence_threshold_override: float | None = None) -> str:
        """
        Transcribe float32 mono audio. Returns the text string, or GATED ("") if
        the clip was too short or too quiet. Raises on model load failure.

        initial_prompt_override — if provided, used instead of the configured
            initial_prompt.  Pass the previous streaming text to give Whisper
            context so it doesn't repeat words at chunk boundaries.
        min_duration_override / silence_threshold_override — per-call gate
            overrides (e.g. stricter thresholds for short streaming chunks).
        """
        min_dur = min_duration_override if min_duration_override is not None else self._min_duration_sec
        sil_db  = silence_threshold_override if silence_threshold_override is not None else self._silence_threshold_db

        # Gate: duration
        duration = len(audio) / sample_rate if sample_rate > 0 else 0
        if duration < min_dur:
            log.debug("Clip too short (%.3fs < %.3fs) — gated", duration, min_dur)
            return GATED

        # Gate: silence
        db = _rms_db(audio)
        if db < sil_db:
            log.debug("Clip too quiet (%.1f dB < %.1f dB) — gated", db, sil_db)
            return GATED

        # Lazy load
        with self._model_lock:
            if self._model is None and not self._load_failed:
                self._load_model()

        if self._load_failed or self._model is None:
            raise RuntimeError("Whisper model failed to load; cannot transcribe")

        log.debug("Transcribing %.2fs clip (%.1f dB)…", duration, db)

        prompt = initial_prompt_override if initial_prompt_override is not None else self._initial_prompt

        try:
            return self._run_transcribe(audio, duration, on_segment, prompt)
        except RuntimeError as exc:
            if self._device == "cuda" and any(k in str(exc).lower() for k in ("cublas", "cuda", "cublaslt")):
                log.warning("CUDA inference failed (%s) — falling back to CPU/int8", exc)
                with self._model_lock:
                    self._model = None
                    self._device = "cpu"
                    self._compute_type = "int8"
                    self._load_model()
                if self._load_failed or self._model is None:
                    raise RuntimeError("CPU fallback model failed to load") from exc
                return self._run_transcribe(audio, duration, on_segment, prompt)
            raise

    def _run_transcribe(self, audio: np.ndarray, duration: float,
                        on_segment: callable = None,
                        initial_prompt: str | None = None) -> str:
        t0 = time.monotonic()

        # vad_filter is intentionally OFF.
        #
        # Audio sent here has already passed our own silence gate (RMS dB check
        # + min_duration_sec), so Silero VAD inside Whisper is a redundant second
        # filter that kept silently discarding legitimate speech — especially
        # quieter mid-sentence sections.  Turning it off gives Whisper the full
        # recording and lets it decide what is and isn't speech via its language
        # model, which is far more reliable for continuous dictation.
        #
        # no_speech_threshold is raised from the default 0.6 → 0.9 so that
        # Whisper only suppresses a segment when it is extremely confident there
        # is no speech — further guarding against mid-sentence dropouts.
        segments, info = self._model.transcribe(
            audio,
            language="en",
            initial_prompt=initial_prompt,
            vad_filter=False,
            no_speech_threshold=0.9,
            word_timestamps=True,
        )

        log.info("Transcribing %.2fs clip (vad_filter=off, no_speech_threshold=0.9)", duration)

        from .vocabulary import fix_missing_periods

        parts = []
        last_word_end_sec: float = 0.0   # end-time of the last word in the chunk (seconds)

        for seg in segments:
            part = seg.text.strip()
            if not part:
                continue
            # Strip Whisper's ellipsis hallucination tokens (generated at pauses/silence).
            # re.sub first handles "..." variants; replace handles the Unicode ellipsis "…".
            part = re.sub(r'\.{3,}', '', part).replace('…', '').strip()
            if not part:
                continue
            part = fix_missing_periods(part)
            parts.append(part)

            # Track end-time of the last word so callers can advance sample pointers
            # to the actual end of speech rather than the end of the audio buffer.
            if seg.words:
                last_word_end_sec = max(last_word_end_sec, seg.words[-1].end)

            if on_segment:
                # words: list of (word_text, probability) — stripped of surrounding spaces
                words = [(w.word.strip(), w.probability) for w in seg.words or []]
                on_segment(part, words)

        text = " ".join(parts)
        self._last_word_end_sec = last_word_end_sec   # stored for callers that need it

        elapsed = time.monotonic() - t0
        log.info(
            "Transcribed %.2fs -> %.2fs  [%.1fx realtime]  %r",
            duration, elapsed, duration / elapsed if elapsed > 0 else 0, text,
        )
        return text
