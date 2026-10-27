"""Reading the user's answer to a bot question, without letting the question's words leak in.

The extractor was trained with the bot's question attached only to short CLARIFY replies (<= 6 words,
always labelled with the asked type), never to full sentences, CHALLENGE or PROBE questions. Attaching
the question to anything else lets its topic leak into the reading. So:

1. The answer speaks for itself first: read it alone. If that already has a stance on the asked type, use it.
2. CHALLENGE / PROBE (never trained with the question attached): the question is never attached. Short
   answers ("the first one", "I enjoy it", "just what I know") are resolved by rule from what the bot holds.
3. CLARIFY (trained with the question): attach the question when the answer alone says nothing, or when it
   is a reply to the either-or ("yes, but I hate selling"); keep only the asked type from that reading.
"""
import re

from .interfaces import Reading, SlotReading

MILD = 0.55  # labels.json strength for a mild stance
SHORT = 8   # words. Rules only judge short replies; the training replies with a question were <= 6 words.

# Phrase forms only: bare "now", "before", "later" appear in replies that state a stance of their own
# ("I now enjoy hands-on work"), so they must not pick a statement.
_FIRST = re.compile(r"\b(?:the\s+)?(?:first|earlier|former|previous|old(?:er)?)(?:\s+one)\b|\bthe\s+first\b|"
                    r"\bwhat\s+i\s+said\s+(?:first|before|earlier)\b|^\s*(?:first|the\s+first)\W*$", re.I)
_SECOND = re.compile(r"\b(?:the\s+)?(?:second|latter|latest|later|new(?:er)?|last)(?:\s+one)\b|\bthe\s+(?:second|latest)\b|"
                     r"\bwhat\s+i\s+(?:just\s+)?said\s+(?:now|last|just\s+now)\b|\bwhat\s+i\s+just\s+said\b|"
                     r"^\s*(?:second|the\s+second|latest)\W*$", re.I)
# A negation or "that was wrong" makes a first/second reply ambiguous ("not the first one"). A leading
# "no" does not: "No, the second one" still picks the second.
_REJECT = re.compile(r"\b(?:not|never|neither)\b|n't\b|\b(?:mistake|wrong|take\s+back|ignore|scratch)\b", re.I)
_ENJOY = re.compile(r"\b(?:enjoy(?:ed|s)?|lik(?:e|ed|es)|lov(?:e|ed|es)|interested|interesting|passionate|fun)\b", re.I)
# Negation only right before the enjoy word: "I don't (really) enjoy it", not "I do enjoy it, not just habit".
_NOT_ENJOY = re.compile(r"(?:\b(?:not|never)|n't)\s+(?:really\s+|actually\s+|that\s+)?"
                        r"(?:enjoy(?:ed|s)?|lik(?:e|ed|es)|lov(?:e|ed|es)|interested|into)\b", re.I)
_FAMILIAR = re.compile(r"\b(?:what\s+i\s+know|just\s+know|experience|used\s+to\s+it|familiar|grew\s+up|exposure|"
                       r"only\s+because|just\s+because|mainly\s+because|habit)\b", re.I)
_NOT_JUST = re.compile(r"\bnot\s+(?:just|only|merely)\b[^,.;!?]*", re.I)   # "not just habit" denies familiarity
# Replies to an either-or question; only these may borrow the question when they also mention another type.
_ANSWERISH = re.compile(r"\b(?:yes|yeah|yep|sure|no|nope|nah|definitely|either|both|neither|first|second|"
                        r"former|latter|rather|prefer|probably|lean|ok|okay|fine|don'?t\s+mind)\b|"
                        r"\b(?:the|that|this)\s+(?:[\w-]+\s+){0,2}one\b", re.I)


def asked_kind(last) -> str | None:
    """CLARIFY / CHALLENGE / PROBE for the question being answered (an ABSTAIN repeats its retry)."""
    if last is None:
        return None
    return last.payload.get("retry", "CLARIFY") if last.kind == "ABSTAIN" else last.kind


def short_answer(kind: str | None, text: str, state, slot: str | None) -> Reading | None:
    """A Reading for a short CHALLENGE or PROBE answer, or None when the rule cannot tell."""
    if not slot or len(text.split()) > SHORT:
        return None
    if kind == "CHALLENGE":
        flag = next((f for f in state.flags if f.slot == slot), None)
        first, second = bool(_FIRST.search(text)), bool(_SECOND.search(text))
        if not flag or first == second or _REJECT.search(text):
            return None
        chosen = flag.old if first else flag.new
        return Reading(text, [SlotReading(slot, 1 if chosen.value >= 0 else -1, min(1.0, abs(chosen.value)),
                                          chosen.confidence, chosen.reason, chosen.source, chosen.reason_confidence)])
    if kind == "PROBE":
        current = state.slots.get(slot)
        if not current or current.value <= 0:   # probes question a liking; a familiar dislike is left alone
            return None
        plain = _NOT_JUST.sub(" ", text)
        not_enjoy = bool(_NOT_ENJOY.search(text))
        enjoy = bool(_ENJOY.search(text)) and not not_enjoy
        familiar = not_enjoy or bool(_FAMILIAR.search(plain))
        if enjoy == familiar:                     # neither, or both ("I like it but mostly it's habit")
            return None
        # "I enjoy it": keep the stance, now as enjoyment. "I don't like it": a mild dislike.
        # "just what I know": no real interest either way (value 0).
        if enjoy:
            sign, strength, reason = 1, min(1.0, current.value), "enjoyment"
        elif not_enjoy:
            sign, strength, reason = -1, MILD, "enjoyment"
        else:
            sign, strength, reason = 1, 0.0, "familiarity"
        return Reading(text, [SlotReading(slot, sign, strength, current.confidence, reason, text.strip(), 1.0)])
    return None


def read_answer(extractor, state, last, text: str) -> Reading:
    """Read the answer to the last bot question without letting the question's words leak in."""
    kind = asked_kind(last)
    alone = extractor.read(text, None, None)
    if kind not in ("CLARIFY", "CHALLENGE", "PROBE") or not last.slot:
        return alone
    if any(r.slot == last.slot for r in alone.slots):
        return alone                              # the answer speaks for itself
    if kind in ("CHALLENGE", "PROBE"):
        rule = short_answer(kind, text, state, last.slot)
        return Reading(text, alone.slots + rule.slots) if rule else alone
    if alone.slots and not _ANSWERISH.search(text):
        return alone                              # a statement about something else, not an answer to the either-or
    paired = extractor.read(text, state.last_reply, last.slot)
    return Reading(text, alone.slots + [r for r in paired.slots if r.slot == last.slot])
