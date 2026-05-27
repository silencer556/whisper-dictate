import logging
import math
import threading
import time

import numpy as np

log = logging.getLogger(__name__)

# Returned when audio is gated out (silence / too short) so the caller
# can distinguish "nothing to type" from an actual empty transcription.
GATED = ""


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

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the model synchronously. Safe to call from a background thread."""
        with self._model_lock:
            if self._model is not None or self._load_failed:
                return
            self._load_model()

    def _load_model(self) -> None:
        from faster_whisper import WhisperModel

        device = self._device
        compute_type = self._compute_type

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
                   on_segment: callable = None) -> str:
        """
        Transcribe float32 mono audio. Returns the text string, or GATED ("") if
        the clip was too short or too quiet. Raises on model load failure.
        """
        # Gate: duration
        duration = len(audio) / sample_rate if sample_rate > 0 else 0
        if duration < self._min_duration_sec:
            log.debug("Clip too short (%.3fs < %.3fs) — gated", duration, self._min_duration_sec)
            return GATED

        # Gate: silence
        db = _rms_db(audio)
        if db < self._silence_threshold_db:
            log.debug("Clip too quiet (%.1f dB < %.1f dB) — gated", db, self._silence_threshold_db)
            return GATED

        # Lazy load
        with self._model_lock:
            if self._model is None and not self._load_failed:
                self._load_model()

        if self._load_failed or self._model is None:
            raise RuntimeError("Whisper model failed to load; cannot transcribe")

        log.debug("Transcribing %.2fs clip (%.1f dB)…", duration, db)

        try:
            return self._run_transcribe(audio, duration, on_segment)
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
                return self._run_transcribe(audio, duration, on_segment)
            raise

    def _run_transcribe(self, audio: np.ndarray, duration: float,
                        on_segment: callable = None) -> str:
        t0 = time.monotonic()

        segments, info = self._model.transcribe(
            audio,
            language="en",
            initial_prompt=self._initial_prompt,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=True,
        )

        parts = []
        for seg in segments:
            part = seg.text.strip()
            if not part:
                continue
            parts.append(part)
            if on_segment:
                # words: list of (word_text, probability) — stripped of surrounding spaces
                words = [(w.word.strip(), w.probability) for w in (seg.words or [])]
                on_segment(part, words)

        text = " ".join(parts)

        elapsed = time.monotonic() - t0
        log.info(
            "Transcribed %.2fs -> %.2fs  [%.1fx realtime]  %r",
            duration, elapsed, duration / elapsed if elapsed > 0 else 0, text,
        )
        return text
