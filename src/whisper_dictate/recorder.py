import math
import threading
import time
import logging
import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)


def list_input_devices() -> list[dict]:
    """Return all input-capable devices with index, name, channels, and default sample rate."""
    devices = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            devices.append({
                "index": i,
                "name": dev["name"],
                "max_input_channels": int(dev["max_input_channels"]),
                "default_samplerate": int(dev["default_samplerate"]),
            })
    return devices


def get_default_input_device() -> dict | None:
    try:
        idx = sd.default.device[0]  # (input, output) tuple
        if idx is None or idx < 0:
            return None
        dev = sd.query_devices(idx)
        return {"index": idx, "name": dev["name"]}
    except Exception:
        return None


class Recorder:
    # Minimum recording time before auto-stop can fire
    _MIN_RECORD_SEC = 0.5

    def __init__(self, sample_rate: int, channels: int, device_index: int):
        self._sample_rate = sample_rate
        self._channels = channels
        self._device: int | None = None if device_index == -1 else device_index
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None

        # VAD / auto-stop state — reset on each start()
        self._auto_stop_cb = None
        self._silence_timeout_sec: float = 0.0
        self._vad_threshold_db: float = -40.0
        self._start_time: float = 0.0
        self._last_sound_time: float = 0.0
        self._auto_stop_fired = False

    def start(self, on_auto_stop=None, silence_timeout_sec: float = 0.0,
              vad_threshold_db: float = -40.0) -> None:
        """Begin capturing audio.

        If *on_auto_stop* is provided and *silence_timeout_sec* > 0, the callback
        is fired (in a daemon thread) after that many seconds of continuous silence,
        but no sooner than _MIN_RECORD_SEC into the recording.
        """
        with self._lock:
            self._chunks.clear()

        self._auto_stop_cb = on_auto_stop
        self._silence_timeout_sec = silence_timeout_sec
        self._vad_threshold_db = vad_threshold_db
        self._start_time = time.monotonic()
        self._last_sound_time = self._start_time
        self._auto_stop_fired = False

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            device=self._device,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        log.debug(
            "Recording started — device=%s, rate=%d Hz, channels=%d, vad_timeout=%.1fs",
            self._device, self._sample_rate, self._channels, silence_timeout_sec,
        )

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            log.warning("sounddevice status: %s", status)

        with self._lock:
            self._chunks.append(indata.copy())

        # VAD silence monitoring
        if self._silence_timeout_sec > 0 and self._auto_stop_cb and not self._auto_stop_fired:
            rms = float(np.sqrt(np.mean(indata ** 2)))
            db = 20.0 * math.log10(rms) if rms > 0 else -100.0
            now = time.monotonic()

            if db >= self._vad_threshold_db:
                self._last_sound_time = now
            else:
                elapsed_total = now - self._start_time
                silence_dur = now - self._last_sound_time
                if (elapsed_total >= self._MIN_RECORD_SEC and
                        silence_dur >= self._silence_timeout_sec):
                    self._auto_stop_fired = True
                    cb = self._auto_stop_cb
                    threading.Thread(target=cb, daemon=True).start()

    def stop(self) -> np.ndarray:
        """Stop recording and return float32 mono audio in [-1, 1]."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            chunks = list(self._chunks)
            self._chunks.clear()

        if not chunks:
            log.warning("stop() called but no audio was captured")
            return np.zeros(0, dtype=np.float32)

        audio = np.concatenate(chunks, axis=0)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        duration = len(audio) / self._sample_rate
        peak = float(np.abs(audio).max()) if len(audio) else 0.0
        log.debug("Captured %.2fs of audio, peak=%.4f", duration, peak)
        return audio.astype(np.float32)
