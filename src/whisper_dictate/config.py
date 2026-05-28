import os
import sys
import tomllib
import tomli_w
import logging
from pathlib import Path

log = logging.getLogger(__name__)

VALID_MODELS = {"tiny.en", "base.en", "small.en", "medium.en", "large-v2", "large-v3"}
VALID_DEVICES = {"cuda", "cpu"}
VALID_COMPUTE = {"float16", "int8", "float32"}
VALID_MODES = {"push_to_talk", "toggle"}
VALID_OUTPUT_METHODS = {"auto", "keystroke", "clipboard", "extension", "remote"}

DEFAULT_CONFIG: dict = {
    "hotkey": {
        "key": "win+`",
        "mode": "push_to_talk",
    },
    "whisper": {
        "model": "medium.en",
        "device": "cuda",
        "compute_type": "float16",
        "initial_prompt": "",
    },
    "audio": {
        "sample_rate": 16000,
        "channels": 1,
        "device_index": -1,
        "min_duration_sec": 0.3,
        "silence_threshold_db": -50.0,
        "auto_stop_silence_sec": 1.5,
        "vad_threshold_db": -40.0,
        "idle_stop_sec": 30.0,
        "streaming_interval_sec": 0.0,  # 0 = disabled; >0 = type progressively while speaking
    },
    "output": {
        "method": "keystroke",
        "keystroke_delay_ms": 10,
        "trailing_space": True,
        "sound_on_start": True,
        "sound_on_stop": True,
        "extension_port": 9754,
    },
    "remote": {
        "host": "",
        "port": 9753,
    },
    "vocabulary": {
        "unique": {},
        "punctuation": {
            # Unambiguous phrases — safe to leave on by default.
            "question mark":        "?",
            "exclamation point":    "!",
            "exclamation mark":     "!",
            "full stop":            ".",
            "open paren":           "(",
            "open parenthesis":     "(",
            "close paren":          ")",
            "close parenthesis":    ")",
            "dot dot dot":          "...",
            "ellipsis":             "...",
            "new paragraph":        "\n\n",
        },
        "terminology": {},
        "names": {},
    },
}


def get_config_path() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "whisper-dictate" / "config.toml"
    # Fallback for non-Windows or missing APPDATA
    return Path.home() / ".whisper-dictate" / "config.toml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base, returning a new dict. Nested dicts are merged recursively."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _validate(cfg: dict) -> None:
    """Warn on invalid values but don't crash, except for clearly fatal issues."""
    whisper = cfg.get("whisper", {})
    model = whisper.get("model", "")
    if model not in VALID_MODELS:
        log.warning("Unknown whisper model %r — valid options: %s", model, ", ".join(sorted(VALID_MODELS)))

    device = whisper.get("device", "")
    if device not in VALID_DEVICES:
        log.warning("Unknown whisper device %r — must be 'cuda' or 'cpu'", device)

    compute = whisper.get("compute_type", "")
    if compute not in VALID_COMPUTE:
        log.warning("Unknown compute_type %r — valid: %s", compute, ", ".join(sorted(VALID_COMPUTE)))

    hotkey = cfg.get("hotkey", {})
    mode = hotkey.get("mode", "")
    if mode not in VALID_MODES:
        log.warning("Unknown hotkey mode %r — must be 'push_to_talk' or 'toggle'", mode)

    key = hotkey.get("key", "")
    if not key:
        log.error("hotkey.key is empty — the app will not respond to any key")

    output = cfg.get("output", {})
    method = output.get("method", "")
    if method not in VALID_OUTPUT_METHODS:
        log.warning("Unknown output method %r — must be 'keystroke' or 'clipboard'", method)


def load_config(path: Path | None = None) -> dict:
    """Load config from disk, creating it from defaults if missing."""
    config_path = path or get_config_path()

    if not config_path.exists():
        log.info("Config not found at %s — creating from defaults", config_path)
        cfg = DEFAULT_CONFIG
        save_config(cfg, config_path)
        return dict(cfg)

    try:
        with open(config_path, "rb") as f:
            on_disk = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        log.error("Failed to parse config at %s: %s", config_path, exc)
        log.error("Using built-in defaults — fix the file and reload")
        return dict(DEFAULT_CONFIG)

    # Merge so missing keys get their defaults
    cfg = _deep_merge(DEFAULT_CONFIG, on_disk)
    _validate(cfg)
    return cfg


def save_config(cfg: dict, path: Path | None = None) -> None:
    """Write config to disk."""
    config_path = path or get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "wb") as f:
        tomli_w.dump(cfg, f)
    log.debug("Config saved to %s", config_path)
