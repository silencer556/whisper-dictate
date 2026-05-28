import argparse
import logging
import sys
import threading
import time
import wave
from pathlib import Path

from .config import load_config, get_config_path


def _setup_logging(log_path: Path, verbose: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        handlers=handlers,
    )


def _cmd_list_devices() -> None:
    from .recorder import list_input_devices, get_default_input_device
    import sounddevice as sd

    default = get_default_input_device()
    devices = list_input_devices()
    if not devices:
        print("No input devices found.")
        return

    print(f"\n{'IDX':>4}  {'NAME':<50}  {'CH':>3}  {'DEFAULT RATE':>12}")
    print("-" * 74)
    for d in devices:
        marker = " *" if default and d["index"] == default["index"] else "  "
        print(f"{d['index']:>4}{marker} {d['name']:<50}  {d['max_input_channels']:>3}  {d['default_samplerate']:>12}")

    if default:
        print(f"\n* = current system default ({default['name']})")
    print("\nSet [audio] device_index in config.toml to the IDX you want.")
    print("Use -1 to keep the system default.\n")


def _save_wav(path: Path, audio, sample_rate: int) -> None:
    import numpy as np
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _cmd_gui(cfg: dict, config_path: Path) -> None:
    from .recorder import Recorder
    from .transcriber import Transcriber
    from .gui import DictateGUI

    a_cfg = cfg["audio"]
    w_cfg = cfg["whisper"]
    o_cfg = cfg["output"]

    if o_cfg["method"] in ("extension", "auto"):
        from . import ext_server
        ext_server.start(o_cfg.get("extension_port", 9754))

    recorder = Recorder(
        sample_rate=a_cfg["sample_rate"],
        channels=a_cfg["channels"],
        device_index=a_cfg["device_index"],
    )
    transcriber = Transcriber(
        model_name=w_cfg["model"],
        device=w_cfg["device"],
        compute_type=w_cfg["compute_type"],
        initial_prompt=w_cfg["initial_prompt"],
        min_duration_sec=a_cfg["min_duration_sec"],
        silence_threshold_db=a_cfg["silence_threshold_db"],
    )

    # Build the GUI first so the status callback has a root window to post to
    gui = DictateGUI(
        recorder=recorder,
        transcriber=transcriber,
        vocab=cfg["vocabulary"],
        sample_rate=a_cfg["sample_rate"],
        hotkey=cfg["hotkey"]["key"],
        output_method=o_cfg["method"],
        trailing_space=o_cfg["trailing_space"],
        keystroke_delay_ms=o_cfg["keystroke_delay_ms"],
        auto_stop_silence_sec=a_cfg["auto_stop_silence_sec"],
        vad_threshold_db=a_cfg["vad_threshold_db"],
        idle_stop_sec=a_cfg.get("idle_stop_sec", 30.0),
        streaming_interval_sec=a_cfg.get("streaming_interval_sec", 0.0),
        config_path=config_path,
    )

    # Load (or download) the model in the background; GUI shows progress
    threading.Thread(
        target=lambda: transcriber.load(
            on_status=gui._on_model_status,
            on_progress=gui._on_progress,
        ),
        daemon=True,
    ).start()

    gui.run()


def _cmd_test_transcribe(cfg: dict, wav_path: Path) -> None:
    import wave
    import numpy as np
    from .transcriber import Transcriber

    if not wav_path.exists():
        print(f"ERROR: file not found: {wav_path}")
        sys.exit(1)

    # Read WAV into float32 numpy array
    with wave.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(n_frames)

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sampwidth, np.int16)
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    audio /= float(np.iinfo(dtype).max)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    print(f"Loaded {wav_path.name}: {len(audio)/sample_rate:.2f}s, {sample_rate} Hz, {n_channels}ch")

    w_cfg = cfg["whisper"]
    a_cfg = cfg["audio"]
    t = Transcriber(
        model_name=w_cfg["model"],
        device=w_cfg["device"],
        compute_type=w_cfg["compute_type"],
        initial_prompt=w_cfg["initial_prompt"],
        min_duration_sec=a_cfg["min_duration_sec"],
        silence_threshold_db=a_cfg["silence_threshold_db"],
    )

    print(f"Loading model '{w_cfg['model']}' on {w_cfg['device']}…")
    t.load()  # force load now so CUDA errors surface before we gate on audio
    text = t.transcribe(audio, sample_rate)

    if text == "":
        print("(gated — clip was too short or too quiet)")
    else:
        print(f"\nTranscription: {text!r}")


def _beep_start() -> None:
    """Two rising tones: ready to speak."""
    import winsound
    winsound.Beep(880, 120)
    time.sleep(0.05)
    winsound.Beep(1320, 180)


def _beep_stop() -> None:
    """Two falling tones: done recording."""
    import winsound
    winsound.Beep(1320, 120)
    time.sleep(0.05)
    winsound.Beep(660, 200)


def _cmd_test_record(cfg: dict, duration: float, out_path: Path) -> None:
    from .recorder import Recorder
    import numpy as np

    audio_cfg = cfg["audio"]
    rec = Recorder(
        sample_rate=audio_cfg["sample_rate"],
        channels=audio_cfg["channels"],
        device_index=audio_cfg["device_index"],
    )

    print(f"Recording for {duration:.1f}s — listen for the start beep, speak, then listen for the stop beep.")
    _beep_start()
    rec.start()
    time.sleep(duration)
    audio = rec.stop()
    _beep_stop()

    if len(audio) == 0:
        print("ERROR: no audio captured. Check your microphone and device_index.")
        sys.exit(1)

    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.abs(audio).max())
    duration_actual = len(audio) / audio_cfg["sample_rate"]
    print(f"Captured {duration_actual:.2f}s — peak={peak:.4f}  RMS={rms:.4f}")

    if peak < 0.001:
        print("WARNING: audio is near-silent. Check that the correct mic is selected.")

    _save_wav(out_path, audio, audio_cfg["sample_rate"])
    print(f"Saved to {out_path}")
    print("Play it back in any audio player to verify it sounds right.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="whisper-dictate", description="Local push-to-talk dictation")
    parser.add_argument("--config", metavar="PATH", type=Path, default=None,
                        help="Path to config.toml (default: %%APPDATA%%\\whisper-dictate\\config.toml)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio input devices and exit")
    parser.add_argument("--test-record", metavar="SECONDS", type=float, nargs="?", const=3.0,
                        help="Record audio for N seconds (default 3), save to test.wav, and exit")
    parser.add_argument("--out", metavar="PATH", type=Path, default=Path("test.wav"),
                        help="Output path for --test-record (default: test.wav)")
    parser.add_argument("--test-transcribe", metavar="WAV", type=Path, default=None,
                        help="Transcribe a WAV file and print the result, then exit")
    parser.add_argument("--gui", action="store_true",
                        help="Open the test GUI (record → transcribe → display)")
    args = parser.parse_args()

    config_path = args.config or get_config_path()
    log_path = config_path.parent / "log.txt"
    _setup_logging(log_path, args.verbose)

    log = logging.getLogger(__name__)

    if args.list_devices:
        _cmd_list_devices()
        return

    log.info("whisper-dictate starting up")
    log.info("Config path: %s", config_path)
    cfg = load_config(config_path)

    if args.test_record is not None:
        _cmd_test_record(cfg, args.test_record, args.out)
        return

    if args.test_transcribe is not None:
        _cmd_test_transcribe(cfg, args.test_transcribe)
        return

    if args.gui:
        _cmd_gui(cfg, config_path)
        return

    # Placeholder — full app.py wired up in a later step
    import json
    print("\nLoaded config:")
    print(json.dumps(cfg, indent=2))
    print(f"\nConfig file: {config_path}")


if __name__ == "__main__":
    main()
