"""Independent M1 cosine similarity over observed RIASEC dimensions."""

from ..configuration import load_parameters

from math import fsum, hypot

from ..components import ComponentInput, ComponentRanking, ComponentScore
from ..schemas import TYPES, Occupation

BASELINE_PARAMETERS = load_parameters("cosine_baseline_parameters")


class CosineSimilarity:
    def __init__(self, occupations: list[Occupation], parameters: dict):
        if parameters != BASELINE_PARAMETERS:
            raise ValueError("cosine_similarity baseline requires its documented parameters")
        self._occupations = [(o.code, dict(o.z_scores)) for o in occupations]

    def score(self, inputs: ComponentInput) -> ComponentRanking:
        if inputs.interests is None:
            return ComponentRanking("unavailable", reason="No structured profile supplied.")
        dimensions = [t for t in TYPES if t in inputs.observed_types]
        user_norm = hypot(*(inputs.interests[t] for t in dimensions))
        if user_norm == 0:
            return ComponentRanking("unusable", reason="No observed nonzero RIASEC interests; cosine is undefined.")
        user = [inputs.interests[t] / user_norm for t in dimensions]
        scored = []
        for code, z in self._occupations:
            norm = hypot(*(z[t] for t in dimensions))
            if norm == 0:
                continue  # Undefined cosine; do not invent a zero-score signal.
            score = fsum(u * (z[t] / norm) for u, t in zip(user, dimensions))
            scored.append((max(-1.0, min(1.0, score)), code))
        if not scored:
            return ComponentRanking("unusable", reason="No occupations with nonzero observed RIASEC norm.")
        scored.sort(key=lambda item: (-item[0], item[1]))
        return ComponentRanking("usable", tuple(
            ComponentScore(code, score, rank)
            for rank, (score, code) in enumerate(scored, 1)))
