import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from whisper_dictate.vocabulary import apply_substitutions


# ---------------------------------------------------------------------------
# Flat vocab (legacy / backwards-compat)
# ---------------------------------------------------------------------------

def test_basic_flat():
    vocab = {"vdci": "VDCI"}
    assert apply_substitutions("I work at vdci", vocab) == "I work at VDCI"


def test_case_insensitive():
    vocab = {"vdci": "VDCI"}
    assert apply_substitutions("VDCI is great", vocab) == "VDCI is great"
    assert apply_substitutions("Vdci is great", vocab) == "VDCI is great"


def test_no_partial_word_match():
    vocab = {"vdci": "VDCI"}
    # "vdci" inside a longer token should not match
    assert apply_substitutions("thevdciproject", vocab) == "thevdciproject"


def test_longest_key_wins_flat():
    vocab = {
        "bim one oh one": "BIM101",
        "one": "1",
    }
    result = apply_substitutions("bim one oh one", vocab)
    assert result == "BIM101", f"Expected 'BIM101', got {result!r}"


def test_empty_text():
    assert apply_substitutions("", {"vdci": "VDCI"}) == ""


def test_empty_vocab():
    assert apply_substitutions("hello world", {}) == "hello world"


# ---------------------------------------------------------------------------
# Nested vocab sections
# ---------------------------------------------------------------------------

NESTED = {
    "terminology": {
        "bim one oh one": "BIM101",
        "bim 101": "BIM101",
        "vdci": "VDCI",
    },
    "names": {
        "claire lee": "Claire Lee",
        "old town transit": "Old Town Transit Center",
    },
    "unique": {
        "at claire lee": "@claire lee",
    },
}


def test_nested_terminology():
    assert apply_substitutions("we use bim 101 here", NESTED) == "we use BIM101 here"


def test_nested_names():
    assert apply_substitutions("contact claire lee", NESTED) == "contact Claire Lee"


def test_unique_beats_name_single_pass():
    # "at claire lee" should become "@claire lee", NOT "@Claire Lee"
    # because the single-pass means "claire lee" won't re-fire on the result
    result = apply_substitutions("message at claire lee", NESTED)
    assert result == "message @claire lee", f"Got {result!r}"


def test_unique_and_name_in_same_sentence():
    # The standalone "claire lee" (without "at") should still capitalise
    result = apply_substitutions("at claire lee and claire lee", NESTED)
    assert result == "@claire lee and Claire Lee", f"Got {result!r}"


def test_multiple_substitutions():
    result = apply_substitutions("vdci uses bim 101", NESTED)
    assert result == "VDCI uses BIM101", f"Got {result!r}"


def test_longest_wins_nested():
    result = apply_substitutions("bim one oh one not just one", NESTED)
    # "bim one oh one" should match before "one"
    # (there's no "one" rule in NESTED, but let's confirm "bim one oh one" -> "BIM101")
    assert "BIM101" in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_no_double_substitution():
    """A value produced by one rule must not be matched by another rule."""
    vocab = {
        "terminology": {"whisper": "Whisper"},
        "unique": {"at whisper": "@whisper"},
    }
    # "at whisper" fires first (longer key); "whisper" inside "@whisper" must not re-fire
    result = apply_substitutions("ping at whisper", vocab)
    assert result == "ping @whisper", f"Got {result!r}"


def test_extra_section():
    """Unknown section names should still be included."""
    vocab = {
        "jargon": {"synergy": "alignment"},
    }
    assert apply_substitutions("lots of synergy here", vocab) == "lots of alignment here"


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
