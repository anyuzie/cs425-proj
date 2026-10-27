"""Reading the user's answer to a bot question, without letting the question's words leak in.

The extractor was trained with the bot's question attached only to short CLARIFY replies (<= 6 words,
always labelled with the asked type), never to full sentences, CHALLENGE or PROBE questions. So:

1. The answer speaks for itself first: read it alone; attach the question only if that finds nothing
   ("the first one", "either is fine" need the question; "I enjoy research" does not).
2. Short answers to a CHALLENGE ("the first one" / "the second one") and to a PROBE ("I enjoy it" /
   "just what I know") are resolved by rule, from what the bot already holds.
"""
import re

from .interfaces import Reading, SlotReading

SHORT = 8   # words; longer answers are read normally

_FIRST = re.compile(r"\b(?:first|earlier|former|before|originally|previous|old one|what i said before)\b", re.I)
_SECOND = re.compile(r"\b(?:second|latter|later|latest|newer|recent|just now|now|what i just said|new one)\b", re.I)
# no bare "no": "No, I genuinely enjoy it" answers the either-or, it doesn't negate the enjoyment
_NEGATED = re.compile(r"\b(?:not|never|don'?t|doesn'?t|didn'?t|wouldn'?t|isn'?t)\b|n't\b", re.I)
_ENJOY = re.compile(r"\b(?:enjoy|enjoyed|like|love|interested|interesting|passionate|fun)\b", re.I)
_FAMILIAR = re.compile(r"\b(?:what i know|just know|experience|used to it|familiar|grew up|exposure|"
                       r"only because|just because|mainly because|habit)\b", re.I)


def asked_kind(last) -> str | None:
    """CLARIFY / CHALLENGE / PROBE for the question being answered (an ABSTAIN repeats its retry)."""
    if last is None:
        return None
    return last.payload.get("retry", "CLARIFY") if last.kind == "ABSTAIN" else last.kind


def short_answer(kind: str | None, text: str, state, slot: str | None) -> Reading | None:
    """A Reading for a short CHALLENGE or PROBE answer, or None to read the answer normally."""
    if not slot or len(text.split()) > SHORT:
        return None
    if kind == "CHALLENGE":
        flag = next((f for f in state.flags if f.slot == slot), None)
        first, second = bool(_FIRST.search(text)), bool(_SECOND.search(text))
        if not flag or first == second:          # nothing held, or ambiguous ("not the first, the second")
            return None
        chosen = flag.old if first else flag.new
        return Reading(text, [SlotReading(slot, 1 if chosen.value >= 0 else -1, min(1.0, abs(chosen.value)),
                                          chosen.confidence, chosen.reason, chosen.source, chosen.reason_confidence)])
    if kind == "PROBE":
        current = state.slots.get(slot)
        if not current:
            return None
        enjoy = bool(_ENJOY.search(text)) and not _NEGATED.search(text)
        familiar = bool(_FAMILIAR.search(text)) or (bool(_ENJOY.search(text)) and bool(_NEGATED.search(text)))
        if enjoy == familiar:                     # neither, or both ("I like it but mostly it's habit")
            return None
        sign = 1 if current.value >= 0 else -1
        # enjoyment: keep the stance, now as enjoyment. familiarity only: no real interest (value 0).
        strength = min(1.0, abs(current.value)) if enjoy else 0.0
        return Reading(text, [SlotReading(slot, sign, strength, max(current.confidence, 0.8),
                                          "enjoyment" if enjoy else "familiarity", text.strip(), 1.0)])
    return None


def read_answer(extractor, state, last, text: str) -> Reading:
    """The answer speaks for itself first; the bot's question is attached only when the answer alone says nothing."""
    kind = asked_kind(last)
    if kind not in ("CLARIFY", "CHALLENGE", "PROBE"):
        return extractor.read(text, None, None)
    rule = short_answer(kind, text, state, last.slot)
    if rule is not None:
        return rule
    alone = extractor.read(text, None, None)
    if alone.slots:
        return alone
    return extractor.read(text, state.last_reply, last.slot)
