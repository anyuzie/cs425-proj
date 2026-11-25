"""Semantic retrieval without the user's negative clauses.

``occupation_dual_encoder_signed@finetuned_v1`` uses the same fine-tuned checkpoint and index as
``occupation_dual_encoder@finetuned_v1``, and routes clauses exactly like
``bm25_signed@ignore_negative_v2``. Each user turn has its negative and skipped clauses deleted;
everything else is kept as typed, so a history with no such clause gives a byte-identical query
and ranking. "i like helping others. but teaching, not as much" is encoded as
"i like helping others. but".

Early stop: with no positive or neutral clause the component is ``unusable``. It wraps
``OccupationSemantic`` rather than changing it, because ``occupation_semantic.py`` is hashed into
the index identity.
"""

from copy import deepcopy
from dataclasses import replace

from ..components import ComponentRanking
from ..configuration import load_parameters
from ..occupation_semantic import OccupationSemantic
from .clause_routing import QUERY_ROUTES, ClauseRouter, normalise

PARAMETERS = load_parameters("occupation_dual_encoder_signed_finetuned_v1_parameters")
# finetuned_v2: same, with routing v3 (same rules as bm25_signed@ignore_negative_v3).
PARAMETERS_V2 = load_parameters("occupation_dual_encoder_signed_finetuned_v2_parameters")


class SignedOccupationSemantic:
    def __init__(self, occupations, parameters: dict):
        if parameters not in (PARAMETERS, PARAMETERS_V2):
            raise ValueError("unsupported signed occupation semantic parameters")
        self.parameters = deepcopy(parameters)
        self._router = ClauseRouter(parameters)
        self._base = OccupationSemantic(occupations, {"index": parameters["index"]})
        self.last_trace = {}

    @property
    def metadata(self):
        return {**self._base.metadata, "parameters": deepcopy(self.parameters),
                "query_policy": "newline_join_chronological_user_only_minus_negative_and_skipped_clauses"}

    def filtered(self, user_sentences) -> tuple[tuple[str, ...], list[dict]]:
        """User turns with negative/skipped clauses deleted, plus the routed clauses."""
        texts = list(user_sentences)
        clauses = self._router.clauses(texts)
        out = []
        for i, text in enumerate(texts):
            own = [c for c in clauses if c["turn"] == i]
            if all(c["route"] in QUERY_ROUTES for c in own):
                out.append(text)  # untouched, so a history without dislikes equals the base query
                continue
            source, cursor, pieces = normalise(text), 0, []
            for clause in own:  # clauses are in text order: locate each after the previous one
                start = source.find(clause["source_span"], cursor)
                if start < 0:
                    raise RuntimeError("clause span not found in its turn")
                end = start + len(clause["source_span"])
                pieces.append(source[cursor:start])
                if clause["route"] in QUERY_ROUTES:
                    pieces.append(source[start:end])
                cursor = end
            pieces.append(source[cursor:])
            out.append(" ".join("".join(pieces).split()))
        return tuple(out), clauses

    def _inputs(self, inputs):
        sentences, clauses = self.filtered(inputs.user_sentences)
        return replace(inputs, user_sentences=sentences), clauses

    def score(self, inputs) -> ComponentRanking:
        filtered, clauses = self._inputs(inputs)
        self.last_trace = dict(clauses=clauses, query=filtered.user_sentences,
                               removed=[c["source_span"] for c in clauses if c["route"] not in QUERY_ROUTES])
        if clauses and not any(c["route"] in QUERY_ROUTES for c in clauses):
            return ComponentRanking("unusable", reason="Signed semantic: no positive or neutral clause (early stop).")
        return self._base.score(filtered)

    def query_diagnostics(self, inputs):
        filtered, clauses = self._inputs(inputs)
        return {**self._base.query_diagnostics(filtered),
                "removed_clauses": [dict(text=c["source_span"], route=c["route"], turn=c["turn"])
                                    for c in clauses if c["route"] not in QUERY_ROUTES],
                "history_policy": "chronological newline join of user turns with negative and skipped clauses deleted; "
                                  "first 256 wordpieces, assistant text excluded"}
