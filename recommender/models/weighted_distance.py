"""Independent M1 confidence-weighted distance in occupation z-space.

score(occ) = -sum_{t in observed} w_t * (kappa * u_t - z_t)^2

u_t is the runtime tracker value in [-1, 1]; kappa maps it onto the occupation
z-scale (the mock tracker divides hidden z-slots by 3, so kappa=3 inverts it);
w_t is the tracker's cumulative slot weight (accepted evidence mass).

Unlike cosine, magnitude matters: an occupation that is "balanced but weak"
on the observed types sits far from a strong user. Unlike the dot product,
a huge z on one type cannot buy a high score regardless of fit. Unknown types
contribute nothing; an observed zero is a real target at z=0.

Oracle inputs are already z-scale and carry no weights, so this algorithm is
``unavailable`` without tracker weights instead of inventing them.
"""

from math import fsum, isfinite

from ..components import ComponentInput, ComponentRanking, ComponentScore
from ..configuration import load_parameters
from ..schemas import TYPES, Occupation

BASELINE_PARAMETERS = load_parameters("weighted_distance_kappa3_parameters")


class WeightedDistance:
    def __init__(self, occupations: list[Occupation], parameters: dict):
        if parameters != BASELINE_PARAMETERS:
            raise ValueError("weighted_distance kappa3_v1 requires its documented parameters")
        self._kappa = float(parameters["user_scale_kappa"])
        self._occupations = [(o.code, dict(o.z_scores)) for o in occupations]

    def score(self, inputs: ComponentInput) -> ComponentRanking:
        if inputs.interests is None:
            return ComponentRanking("unavailable", reason="No structured profile supplied.")
        if inputs.weights is None:
            return ComponentRanking("unavailable", reason="weighted_distance requires tracker slot weights.")
        dimensions = [t for t in TYPES if t in inputs.observed_types]
        weights = {t: float(inputs.weights.get(t, 0.0)) for t in dimensions}
        if any(not isfinite(w) or w < 0 for w in weights.values()):
            raise ValueError("slot weights must be finite and nonnegative")
        if fsum(weights.values()) == 0:
            return ComponentRanking("unusable", reason="No observed RIASEC type carries tracker weight.")
        target = {t: self._kappa * inputs.interests[t] for t in dimensions}
        scored = [(-fsum(weights[t] * (target[t] - z[t]) ** 2 for t in dimensions), code)
                  for code, z in self._occupations]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return ComponentRanking("usable", tuple(
            ComponentScore(code, score, rank) for rank, (score, code) in enumerate(scored, 1)))
