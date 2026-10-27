"""Mock model parts for running the whole app WITHOUT the trained models (no torch, no checkpoints).

Only the three model-backed parts are mocked; the tracker, rule, policy, ranker, question bank, server and
page are the real code. Importing this module registers the mocks in pipeline.registry; the app itself
never imports it. Results from mock runs are flow checks only, never evaluation results: the config is
labelled mock_tracker, and System refuses to label mock components actual_tracker.
"""
import json
import re
from pathlib import Path

from pipeline import registry
from pipeline.interfaces import TYPES, Reading, SlotReading

ROOT = Path(__file__).resolve().parent.parent
_WORD = re.compile(r"[a-z']+")
_SENT = re.compile(r"[^.!?\n]+[.!?]*")
POS = ("love", "enjoy", "like", "fun", "exciting", "interesting", "satisfying", "rewarding", "happy", "great",
       "passionate", "keen", "prefer", "rather", "fulfilling", "into")
NEG = ("hate", "dislike", "boring", "dull", "tedious", "awful", "nightmare", "drain", "stressful", "miserable",
       "can't stand", "not for me", "rather not")
# Replies to an either-or CLARIFY question, which name no type of their own ("the first one").
YES = ("definitely", "for sure", "absolutely", "yes", "yeah", "first", "former", "sure", "the one")
NO = ("no way", "nope", "nah", "second", "latter", "rather the other", "not at all")
DONT_MIND = ("either", "don't mind", "both ok", "both are fine", "no preference")


def _has(text, phrases):
    padded = f" {text} "
    return any(f" {p} " in padded or (" " in p and p in text) for p in phrases)


class MockKeywordExtractor:
    """Keyword reader over O*NET's own RIASEC vocabulary (data/riasec_lexicon.json)."""
    name = "mock-keyword"

    def __init__(self, lexicon_path: Path) -> None:
        lex = json.loads(Path(lexicon_path).read_text(encoding="utf-8"))
        stem = self._stem
        self.terms = {t: {stem(x["term"]) for x in lex["types"][t][:60]} for t in TYPES}
        inten = lex.get("intensity", {})
        self.negators = set(inten.get("negators", ["not", "don't", "never", "no"]))
        self.strong = [p.lower() for p in inten.get("strong_positive", []) + inten.get("strong_negative", [])]
        self.familiar = [p.lower() for p in inten.get("familiarity", [])] + [
            "that's what i know", "my dad", "my mum", "my mom", "my parents", "always been around", "grew up",
            "internship"]

    @staticmethod
    def _stem(word):
        return word[:-1] if word.endswith("s") else word

    def _direction(self, words, low):
        pos = sum(low.count(c) for c in POS)
        neg = sum(low.count(c) for c in NEG)
        for i, w in enumerate(words):   # "don't enjoy" flips a positive cue
            if w in self.negators and any(" ".join(words[i + 1:i + 3]).startswith(c) for c in POS):
                pos, neg = pos - 1, neg + 1
        return 0 if pos == neg else (1 if pos > neg else -1)

    def _sentence(self, sentence):
        words = _WORD.findall(sentence.lower())
        low = " ".join(words)
        sign = self._direction(words, low)
        familiar = _has(low, self.familiar)
        if not sign and familiar:
            sign = 1   # "my dad runs a sales company so that's what I know": a lean, justified by familiarity
        if not sign:
            return []
        stems = {self._stem(w) for w in words}
        strength = 1.0 if _has(low, self.strong) else 0.55
        reason = "familiarity" if familiar else "enjoyment"
        return [SlotReading(t, sign, strength, 0.8, reason, sentence.strip(), 0.9)
                for t in TYPES if stems & self.terms[t]]

    def read(self, text, context=None, expected_slot=None):
        best = {}
        for m in _SENT.finditer(text):
            for r in self._sentence(m.group()):
                best.setdefault(r.slot, r)
        if not best and context and expected_slot in TYPES:   # a short reply to an either-or question
            low = " ".join(_WORD.findall(text.lower()))
            if _has(low, DONT_MIND):
                best[expected_slot] = SlotReading(expected_slot, 1, 0.2, 0.8, "enjoyment", text.strip(), 0.9)
            elif _has(low, NO):
                best[expected_slot] = SlotReading(expected_slot, -1, 0.55, 0.8, "enjoyment", text.strip(), 0.9)
            elif _has(low, YES):
                best[expected_slot] = SlotReading(expected_slot, 1, 0.55, 0.8, "enjoyment", text.strip(), 0.9)
        return Reading(text, list(best.values()))


class MockNli:
    """Treats every candidate the contradiction rule finds as a real contradiction."""
    name = "mock-nli"

    def score(self, old_src, new_src):
        return 1.0


registry.REGISTRY["extractor"]["mock-keyword"] = lambda cfg, d: MockKeywordExtractor(d / "riasec_lexicon.json")
registry.REGISTRY["confirmer"]["mock-nli"] = lambda cfg, d: MockNli()
