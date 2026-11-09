"""D2 (docs/LEXICAL_MATCHING_PLAN.md): post-fusion dislike stage.

Runs after fusion, on the final ranking, so a dislike is enforced whatever the components scored.
Every clause routed ``negative`` (routing v3, shared with ``bm25_signed@ignore_negative_v3`` and
``occupation_dual_encoder_signed@finetuned_v2``) gives a *target*: the clause minus cue and filler
words ("teaching, not as much" -> "teaching"; "not a people person" -> no target). An occupation
matches a target by

- ``title``: a target word and a title word share a stem (teaching / Teachers, nursing / Nurses);
- ``semantic``: cosine between the encoded target and the occupation's fine-tuned document vector
  is at least ``semantic_threshold`` (calibrated on the calibration probes only).

Matched occupations move below every non-matched occupation, keeping their relative order
(``placement: after_all_non_matched``); nothing is removed. With no target the ranking is unchanged.

v2 variants (owner review, 9 November 2026: "mainly reduce the scoring; remove only if the phrasing
is very strongly negative") grade each dislike by its wording:
- ``strong`` (hate, never, can't stand, not for me, don't want, anything but, just not, except ...):
  matched jobs move below every other job, as in v1;
- ``mild`` (not as much, not really, not a fan, rather not, don't care much ...; also the default
  when neither pattern matches): the job's fused score drops by ``mild_penalty_sd`` standard
  deviations of this query's fused scores, then the list is re-sorted.
A job matched by both keeps the strongest treatment.
v3 variants: as v2 with ``mild_penalty_sd`` 1.5 instead of 1.0 (owner decision, 9 November 2026).
"""

from copy import deepcopy
import re

from ..components import ComponentRanking, ComponentScore
from ..configuration import load_parameters
from .bm25_ranker import tokenize
from .clause_routing import ClauseRouter

_SUFFIXES = ("ations", "ation", "ions", "ion", "ists", "ist", "ings", "ing", "ers", "er", "ors", "or",
             "ies", "s", "y")  # no final "e"/"es": "care" must not become "car"


def stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:  # "labor" must not become "lab"
            return word[: -len(suffix)]
    return word


def stems_match(a: str, b: str) -> bool:
    """Equal stems; a stem of at least 5 letters that starts the other (account / accountant); or a
    stem of at least 4 letters that starts the other and differs by one letter (nurs / nurse)."""
    short, long_ = sorted((a, b), key=len)
    return a == b or (long_.startswith(short) and (len(short) >= 5 or (len(short) >= 4 and len(long_) - len(short) == 1)))


VARIANTS = {name: load_parameters(f"dislike_demotion_{name}_parameters")
            for name in ("title_v1", "semantic_v1", "title_semantic_v1", "title_v2", "semantic_v2", "title_semantic_v2",
                         "title_v3", "semantic_v3", "title_semantic_v3")}


def validate_post_fusion(selection) -> str:
    if (not isinstance(selection, dict) or set(selection) != {"algorithm", "variant"}
            or selection["algorithm"] != "dislike_demotion" or selection["variant"] not in VARIANTS):
        raise ValueError(f"post_fusion must be {{algorithm: dislike_demotion, variant: one of {sorted(VARIANTS)}}}")
    return f"dislike_demotion@{selection['variant']}"


class DislikeDemotion:
    def __init__(self, occupations, parameters: dict):
        if parameters not in VARIANTS.values() and not parameters.get("calibration_only"):
            raise ValueError("unsupported dislike demotion parameters")
        self.parameters = deepcopy(parameters)
        self._router = ClauseRouter(parameters)
        self._filler = frozenset(parameters["target_filler_words"])
        self._matchers = set(parameters["matcher"].split("_or_"))
        self._titles = {o.code: {stem(w) for w in tokenize(o.title)} for o in occupations}
        self._threshold = parameters.get("semantic_threshold")
        if "semantic" in self._matchers and not isinstance(self._threshold, (int, float)):
            raise ValueError("semantic matching needs a calibrated semantic_threshold")
        tiers = parameters.get("strength_tiers")
        self._mild = re.compile(tiers["mild_pattern"], re.I) if tiers else None
        self._strong = re.compile(tiers["strong_pattern"], re.I) if tiers else None
        self._bundle = None
        if "semantic" in self._matchers:
            from ..occupation_semantic import OccupationSemantic
            base = OccupationSemantic(occupations, {"index": parameters["semantic_index"]})
            if base.bundle is None:
                raise ValueError(f"dislike demotion needs the semantic index: {base.error}")
            self._bundle = base.bundle

    @property
    def metadata(self):
        return {"parameters": deepcopy(self.parameters)}

    def target(self, clause: dict) -> str:
        text = clause["activity_text"] if clause["polarity"] == "negative" and clause["activity_text"] else clause["source_span"]
        return " ".join(w for w in tokenize(text) if w not in self._filler)

    def strength(self, clause: dict) -> str:
        """v1: every dislike is strong. v2: mild wording first, then strong, else mild."""
        if self._mild is None:
            return "strong"
        span = clause["source_span"]
        if self._mild.search(span):
            return "mild"
        if clause.get("rule") == "anything_but" or self._strong.search(span):
            return "strong"
        return "mild"

    def targets(self, user_sentences) -> list[dict]:
        out = []
        for clause in self._router.clauses(user_sentences):
            if clause["route"] == "negative":
                target = self.target(clause)
                if target:
                    out.append(dict(clause=clause["source_span"], turn=clause["turn"], target=target,
                                    strength=self.strength(clause)))
        return out

    def matches(self, targets: list[dict]) -> dict[str, dict]:
        """Occupation code -> the first target that matched it, with the matcher and similarity."""
        found = {}
        similarities = None
        if self._bundle is not None and targets:
            import numpy as np
            model, vectors, codes, _ = self._bundle
            encoded = model.encode([t["target"] for t in targets], normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False)
            similarities = (np.asarray(encoded, dtype="float32") @ vectors.T, codes)
        for i, t in enumerate(targets):
            words = {stem(w) for w in t["target"].split()}
            if "title" in self._matchers:
                for code, title in self._titles.items():
                    if self._replaces(found.get(code), t) and any(stems_match(w, x) for w in words for x in title):
                        found[code] = dict(target=t["target"], clause=t["clause"], matcher="title", strength=t["strength"])
            if similarities is not None:
                row, codes = similarities[0][i], similarities[1]
                for code, s in zip(codes, row):
                    if self._replaces(found.get(code), t) and float(s) >= self._threshold:
                        found[code] = dict(target=t["target"], clause=t["clause"], matcher="semantic",
                                           similarity=float(s), strength=t["strength"])
        return found

    @staticmethod
    def _replaces(current, target) -> bool:
        """A first match, or a strong dislike replacing a mild one."""
        return current is None or (current["strength"] == "mild" and target["strength"] == "strong")

    def apply(self, ranking: ComponentRanking, user_sentences) -> tuple[ComponentRanking, dict]:
        targets = self.targets(user_sentences)
        matched = self.matches(targets) if targets else {}
        diagnostics = dict(targets=targets, demoted=sum(o.code in matched for o in ranking.occupations),
                           top_demoted=[dict(code=o.code, original_rank=o.rank, **matched[o.code])
                                        for o in ranking.occupations if o.code in matched][:10])
        if ranking.status != "usable" or not any(o.code in matched for o in ranking.occupations):
            return ranking, diagnostics
        scores = [o.score for o in ranking.occupations]
        mean = sum(scores) / len(scores)
        sd = (sum((x - mean) ** 2 for x in scores) / len(scores)) ** 0.5
        penalty = self.parameters.get("mild_penalty_sd", 0.0) * sd
        adjusted = {o.code: o.score - (penalty if matched.get(o.code, {}).get("strength") == "mild" else 0.0)
                    for o in ranking.occupations}
        strong = {c for c, m in matched.items() if m["strength"] == "strong"} & set(adjusted)
        offset = (max(adjusted.values()) - min(adjusted.values())) + 1.0  # strong: below every other job
        for code in strong:
            adjusted[code] -= offset
        order = sorted(adjusted, key=lambda c: (-adjusted[c], c))
        diagnostics["mild_penalty"] = penalty
        return ComponentRanking("usable", tuple(ComponentScore(c, adjusted[c], r) for r, c in enumerate(order, 1))), diagnostics
