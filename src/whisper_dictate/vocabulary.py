import re
import logging

log = logging.getLogger(__name__)

# Section processing order — unique first so "@claire lee" wins before "claire lee"
_SECTION_ORDER = ["unique", "punctuation", "terminology", "names"]


def _flatten(vocab: dict) -> dict[str, str]:
    """Flatten a nested {section: {key: value}} vocab dict into a single {key: value} dict.

    Sections are merged in priority order (unique > terminology > names > other).
    If vocab is already flat (all string values), returns it unchanged.
    """
    if not any(isinstance(v, dict) for v in vocab.values()):
        return dict(vocab)

    flat: dict[str, str] = {}

    # Priority sections first (later sections don't overwrite earlier ones)
    seen = set()
    for section in _SECTION_ORDER:
        if section in vocab and isinstance(vocab[section], dict):
            for k, v in vocab[section].items():
                if k not in seen:
                    flat[k] = v
                    seen.add(k)

    # Any extra sections the user added
    for section, val in vocab.items():
        if section not in _SECTION_ORDER and isinstance(val, dict):
            for k, v in val.items():
                if k not in seen:
                    flat[k] = v
                    seen.add(k)

    return flat


def apply_substitutions(text: str, vocab: dict) -> str:
    """Apply vocabulary substitutions to *text* in a single regex pass.

    A single pass means a replacement can never be matched again by another rule,
    so "at claire lee" -> "@claire lee" won't then re-trigger "claire lee" -> "Claire Lee".

    Longer keys are tried before shorter ones (regex alternation, left-to-right).
    Matching is case-insensitive with word boundaries.
    """
    if not text or not vocab:
        return text

    flat = _flatten(vocab)
    if not flat:
        return text

    # Longest keys first — regex alternation picks the first match left-to-right,
    # so putting longer patterns earlier gives us longest-match behaviour.
    sorted_keys = sorted(flat.keys(), key=len, reverse=True)
    lookup = {k.lower(): v for k, v in flat.items()}

    patterns = [r"\b" + re.escape(k) + r"\b" for k in sorted_keys]
    combined = re.compile("|".join(patterns), re.IGNORECASE)

    applied: list[tuple[str, str]] = []

    def _replace(m: re.Match) -> str:
        original = m.group()
        replacement = lookup.get(original.lower(), original)
        if replacement != original:
            applied.append((original, replacement))
        return replacement

    result = combined.sub(_replace, text)

    if applied:
        log.debug("Vocabulary substitutions: %s", applied)
        # Strip any space that was left immediately before a punctuation mark
        # (e.g. "how are you ?" → "how are you?").  Only runs when something
        # was actually substituted, so normal text is never touched.
        result = re.sub(r' ([.!?,;:])', r'\1', result)
        # Remove any comma or semicolon immediately followed by a stronger
        # punctuation mark (e.g. ",!" → "!" or ",?" → "?").  Whisper sometimes
        # inserts a comma before a spoken punctuation word like "exclamation mark"
        # and the two end up adjacent after substitution.
        result = re.sub(r'[,;]([.!?])', r'\1', result)

    return result
