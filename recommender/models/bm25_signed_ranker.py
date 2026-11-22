"""Signed lexical matching: clause polarity decides which clauses reach BM25.

Reuses the ``bm25@baseline_v1`` index and scoring; only the query text differs.
Each user clause is routed by the explicit stance extractor (``explicit_scoped_v1``):

- ``positive``: explicit positive preference -> query.
- ``neutral``: no explicit stance and no negation cue (background, topic statements) -> query.
- ``negative``: explicit negative preference, or no explicit stance but a negation cue -> never in the query.
- ``skipped``: non-preference wording ("I don't mind", "not sure", "doesn't matter"), or uncertain
  wording with a negation cue -> ignored.

Early stop: with no positive or neutral clause, lexical matching does not run (``unusable``), so
fusion uses the other components. Negative clauses never add to any occupation's score.

Variants differ only in what negative clauses do:
- ``ignore_negative_v1``: nothing.
- ``filter_negative_v1``: an occupation is removed from the lexical ranking when its BM25 score
  against the negative clauses is higher than against the positive query. Removal means "no lexical
  match", exactly like an occupation that matched nothing; it is not pushed below those.
"""

from copy import deepcopy

from ..components import ComponentInput, ComponentRanking, ComponentScore
from ..configuration import load_parameters
from .bm25_ranker import BASELINE_PARAMETERS as BM25_PARAMETERS, BM25Ranker
from .clause_routing import QUERY_ROUTES, ROUTES, ClauseRouter
from .task_preferences import REVISION as EXTRACTOR_REVISION

IGNORE_PARAMETERS = load_parameters("bm25_signed_ignore_negative_v1_parameters")
FILTER_PARAMETERS = load_parameters("bm25_signed_filter_negative_v1_parameters")
# v2: same rules, plus apostrophe-less contractions ("dont", "cant", "isnt", ...) as negation cues and
# in non-preference wording ("dont mind", "doesnt matter").
IGNORE_V2_PARAMETERS = load_parameters("bm25_signed_ignore_negative_v2_parameters")
FILTER_V2_PARAMETERS = load_parameters("bm25_signed_filter_negative_v2_parameters")
# v3: routing v3 (scope split for "X, just not Y", "anything but X", first-person-only "avoid").
IGNORE_V3_PARAMETERS = load_parameters("bm25_signed_ignore_negative_v3_parameters")
SUPPORTED = (IGNORE_PARAMETERS, FILTER_PARAMETERS, IGNORE_V2_PARAMETERS, FILTER_V2_PARAMETERS, IGNORE_V3_PARAMETERS)


class SignedBM25Ranker:
    def __init__(self, occupations, parameters: dict):
        if parameters not in SUPPORTED:
            raise ValueError("unsupported signed BM25 parameters")
        self.parameters = deepcopy(parameters)
        self._filter = parameters["negative_handling"] != "ignore"
        self._router = ClauseRouter(parameters)
        self._bm25 = BM25Ranker(occupations, deepcopy(BM25_PARAMETERS))
        self.last_trace = {}

    @property
    def metadata(self):
        return {**self._bm25.metadata, "signed": deepcopy(self.parameters)}

    def route(self, preference) -> str:
        return self._router.route(preference)

    def clauses(self, user_sentences) -> list[dict]:
        return self._router.clauses(user_sentences)

    def score(self, inputs: ComponentInput) -> ComponentRanking:
        clauses = self.clauses(inputs.user_sentences)
        query = tuple(c["source_span"] for c in clauses if c["route"] in QUERY_ROUTES)
        negative = tuple(c["source_span"] for c in clauses if c["route"] == "negative")
        self.last_trace = dict(extractor_revision=EXTRACTOR_REVISION, clauses=clauses,
                               counts={r: sum(c["route"] == r for c in clauses) for r in ROUTES}, removed=[])
        if not query:
            return ComponentRanking("unusable", reason="Signed BM25: no positive or neutral clause (early stop).")
        ranking = self._bm25.score(ComponentInput(user_sentences=query))
        if ranking.status != "usable" or not (self._filter and negative):
            return ranking
        against = self._bm25.score(ComponentInput(user_sentences=negative))
        negative_scores = {o.code: o.score for o in against.occupations}
        kept = [o for o in ranking.occupations if negative_scores.get(o.code, 0.0) <= o.score]
        self.last_trace["removed"] = [o.code for o in ranking.occupations if negative_scores.get(o.code, 0.0) > o.score]
        if not kept:
            return ComponentRanking("unusable", reason="Signed BM25: every lexical match matched the negative clauses more.")
        return ComponentRanking("usable", tuple(ComponentScore(o.code, o.score, rank) for rank, o in enumerate(kept, 1)))
