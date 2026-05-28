import re
import logging

log = logging.getLogger(__name__)

# Section processing order — unique first so "@claire lee" wins before "claire lee"
_SECTION_ORDER = ["unique", "punctuation", "terminology", "names"]

# ---------------------------------------------------------------------------
# Missing-period repair
# ---------------------------------------------------------------------------
# Whisper sometimes capitalises a word that starts a new sentence but omits
# the period before it.  E.g.:
#   "I went to the store And then I came home"  (missing ".")
# We repair this by looking for a lowercase letter immediately followed by a
# space and one of a fixed set of sentence-starting connectives.
# The fix is conservative — only well-known conjunctions/adverbs are listed,
# so proper nouns and ordinary mid-sentence capitals are left untouched.
_SENTENCE_CONNECTIVES = (
    "And|But|Or|So|Yet|Because|However|Therefore|Although|While|Since|"
    "Though|Unless|Whether|Whereas|Meanwhile|Furthermore|Moreover|"
    "Nevertheless|Nonetheless|Additionally|Consequently|Thus"
)
_MISSING_PERIOD_RE = re.compile(
    r'([a-z])( (?:' + _SENTENCE_CONNECTIVES + r')\b)'
)


def fix_missing_periods(text: str) -> str:
    """Insert a period where Whisper capitalised a new sentence but forgot the '.'.

    Only fires for a conservative set of sentence-starting conjunctions /
    adverbs, so ordinary proper nouns and mid-sentence capitals are left alone.
    """
    fixed, n = _MISSING_PERIOD_RE.subn(r'\1.\2', text)
    if n:
        log.debug("fix_missing_periods: %d insertion(s) in %r → %r", n, text, fixed)
    return fixed


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


# ---------------------------------------------------------------------------
# Automatic @ mention rule
# ---------------------------------------------------------------------------
# "at Name" → "@Name" whenever the word after "at" starts with an uppercase
# letter (i.e. Whisper recognised it as a proper noun).  This fires AFTER
# explicit vocabulary substitutions so a user-defined entry always wins.
#
# Guards:
#  • Requires a word boundary before "at" so "chat John" never matches.
#  • Negative lookbehind on @ and / so already-converted "@foo" and URLs
#    like "//at" are never touched again.
#  • Only matches when the name starts with [A-Z] — lowercase words like
#    "at the store", "at work", "at 3pm" are left alone.
#  • Allows hyphenated names (Jean-Pierre) but stops at apostrophes so
#    "at John's desk" becomes "@John's desk" cleanly.
_AT_MENTION_RE = re.compile(
    r'(?<![/@])\b[Aa]t\s+([A-Z][a-zA-Z-]*(?:\s+[A-Z][a-zA-Z-]*)*)'
)


def _apply_at_mentions(text: str) -> str:
    result, n = _AT_MENTION_RE.subn(lambda m: '@' + m.group(1), text)
    if n:
        log.debug("at-mention rule: %d replacement(s)  %r → %r", n, text, result)
    return result


def apply_substitutions(text: str, vocab: dict) -> str:
    """Apply vocabulary substitutions to *text* in a single regex pass, then
    apply the automatic @ mention rule.

    A single pass means a replacement can never be matched again by another rule,
    so "at claire lee" -> "@claire lee" won't then re-trigger "claire lee" -> "Claire Lee".

    Longer keys are tried before shorter ones (regex alternation, left-to-right).
    Matching is case-insensitive with word boundaries.
    """
    if not text:
        return text

    result = text

    if vocab:
        flat = _flatten(vocab)
        if flat:
            # Longest keys first — regex alternation picks the first match
            # left-to-right, so putting longer patterns earlier gives us
            # longest-match behaviour.
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

            result = combined.sub(_replace, result)

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

    # Automatic @ mention rule runs last so explicit vocab entries take priority.
    result = _apply_at_mentions(result)

    return result
