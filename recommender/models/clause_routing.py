"""Shared clause routing for the signed BM25 and signed semantic components.

Routes every user clause, split by the explicit stance extractor (``explicit_scoped_v1``), into
``positive`` / ``neutral`` (used as query text), ``negative`` (never used as query text) or
``skipped`` (non-preference wording). The rules and patterns come from the caller's variant
parameters, so lexical (``bm25_signed``) and semantic (``occupation_dual_encoder_signed``) variants
that record the same patterns route identically.
"""

from dataclasses import asdict
import re

from .task_preferences import REVISION as EXTRACTOR_REVISION, TaskPreferenceExtractor

ROUTES = ("positive", "neutral", "negative", "skipped")
QUERY_ROUTES = ("positive", "neutral")


def normalise(text: str) -> str:
    """The extractor's own normalisation; clause spans are substrings of this text."""
    return text.replace("’", "'")


class ClauseRouter:
    def __init__(self, parameters: dict):
        if parameters["extractor_revision"] != EXTRACTOR_REVISION:
            raise ValueError("clause routing requires the explicit_scoped_v1 stance extractor")
        self.cue = re.compile(parameters["negation_cue_pattern"], re.I)
        self.non_preference = re.compile(parameters["non_preference_pattern"], re.I)
        # Routing v3 rules; absent from v1/v2 parameters.
        self.scope_split = re.compile(parameters["scope_split_pattern"], re.I) if "scope_split_pattern" in parameters else None
        self.anything_but = re.compile(parameters["anything_but_pattern"], re.I) if "anything_but_pattern" in parameters else None
        self.scope_tail_filler = frozenset(parameters.get("scope_tail_filler", ()))
        self.extractor = TaskPreferenceExtractor()

    def route(self, preference) -> str:
        cue = bool(self.cue.search(preference.source_span))
        if preference.status == "usable":
            return preference.polarity  # explicit positive / negative
        if self.non_preference.search(preference.source_span):
            return "skipped"  # "I don't mind", "not sure": no stance, and not a dislike
        if preference.rule == "uncertain_or_negated_dislike":
            return "skipped" if cue else "neutral"
        return "negative" if cue else "neutral"

    def clauses(self, user_sentences) -> list[dict]:
        """Routed clauses over the whole history, each tagged with the index of its user turn."""
        texts = list(user_sentences)
        routed = [dict(asdict(p), route=self.route(p)) for p in self.extractor.extract(texts)]
        # The extractor emits clauses turn by turn, in order; recover each clause's turn.
        turn_of = [i for i, text in enumerate(texts)
                   for piece in self.extractor._split.split(normalise(text)) if piece.strip(" ,")]
        if len(turn_of) != len(routed):
            raise RuntimeError("clause/turn alignment failed")
        for clause, turn in zip(routed, turn_of):
            clause["turn"] = turn
        if self.scope_split:
            routed = [piece for clause in routed for piece in self._split_scope(clause)]
        if self.anything_but:
            self._mark_anything_but(routed)
        return routed

    def _split_scope(self, clause: dict) -> list[dict]:
        """v3: "I love biology, just not lab work" -> "I love biology" + "just not lab work".

        The extractor does not split at these commas, so the dislike tail would take the liked part
        with it. Each piece is a substring of the original clause (so the signed semantic component can
        still delete pieces in place) and is routed on its own.
        """
        match = self.scope_split.search(clause["source_span"])
        if not match or not match.start():
            return [clause]
        span = clause["source_span"]
        pieces = [span[:match.start()].strip(" ,"), span[match.end():].strip(" ,")]
        from .bm25_ranker import tokenize
        if not [w for w in tokenize(pieces[1]) if w not in self.scope_tail_filler]:
            return [clause]  # "teaching, not as much": the tail only carries the cue, so the dislike is the whole clause
        out = []
        for piece in pieces:
            if not piece:
                continue
            for p in self.extractor.extract([piece]):
                out.append(dict(asdict(p), route=self.route(p), turn=clause["turn"], scope_split=True))
        return out or [clause]

    def _mark_anything_but(self, routed: list[dict]) -> None:
        """v3: the extractor splits on "but", so "Anything but sales" arrives as "Anything" + "sales".
        The clause after a clause ending in "anything"/"everything" (same turn) is the dislike."""
        for current, following in zip(routed, routed[1:]):
            if current["turn"] == following["turn"] and self.anything_but.search(current["source_span"]):
                current["route"], following["route"] = "skipped", "negative"
                following["rule"] = current["rule"] = "anything_but"
