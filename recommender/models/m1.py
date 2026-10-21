"""Independent weighted_dot_product implementation; fixed baseline mathematics."""

from ..configuration import load_parameters

from ..components import ComponentInput, ComponentRanking, ComponentScore
from ..schemas import TYPES, Occupation

BASELINE_PARAMETERS = load_parameters("m1_baseline_parameters")


class WeightedDotProduct:
    def __init__(self, occupations: list[Occupation], parameters: dict):
        if parameters != BASELINE_PARAMETERS:
            raise ValueError("weighted_dot_product baseline requires its documented parameters")
        # Own a snapshot: returned output and caller mutations cannot alter the pool.
        self._occupations = [(o.code, dict(o.z_scores)) for o in occupations]

    def score(self, inputs: ComponentInput) -> ComponentRanking:
        if inputs.interests is None:
            return ComponentRanking("unavailable", reason="No structured profile supplied.")
        if not any(inputs.interests[t] != 0 for t in inputs.observed_types):
            return ComponentRanking("unusable", reason="No observed nonzero RIASEC interests; M1 uses structured scoring only.")
        scored = [(sum(inputs.interests[t] * z[t] for t in TYPES if t in inputs.observed_types), code)
                  for code, z in self._occupations]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return ComponentRanking("usable", tuple(ComponentScore(code, score, rank)
                                                 for rank, (score, code) in enumerate(scored, 1)))
