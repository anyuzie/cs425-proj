"""M1 deterministic signed RIASEC scoring; no text retrieval or fusion."""

from pathlib import Path

from .data_loader import load_eligible_occupations
from .schemas import TYPES, OracleInput, RankedOccupation, RankingRequest, RankingResponse, SignalScore


class StructuredRanker:
    def __init__(self, occupations_path: Path = Path("data/occupations.csv")) -> None:
        self._occupations = load_eligible_occupations(occupations_path)

    @property
    def eligible_count(self) -> int:
        return len(self._occupations)

    @property
    def eligible_codes(self) -> set[str]:
        return {occupation.code for occupation in self._occupations}

    def rank_oracle(self, inputs: OracleInput) -> RankingResponse:
        """Offline full-pool ranking; preserve z-scale values without runtime coercion."""
        if not isinstance(inputs, OracleInput):
            raise ValueError("rank_oracle requires OracleInput")
        inputs.__post_init__()
        return self._rank(inputs.interests, set(TYPES), self.eligible_count)

    def rank(self, interests: dict[str, float], observed_types: set[str],
             user_sentences: list[str], top_k: int = 100) -> RankingResponse:
        request = RankingRequest(interests, observed_types, user_sentences, top_k)
        return self._rank(request.interests, request.observed_types, request.top_k)

    def _rank(self, interests: dict[str, float], observed_types: set[str], top_k: int) -> RankingResponse:
        if not any(interests[t] != 0 for t in observed_types):
            return RankingResponse("no_signal", reason="No observed nonzero RIASEC interests; M1 uses structured scoring only.")
        # Fixed dimension order avoids set-iteration-dependent floating-point sums.
        scored = [
            (sum(interests[t] * occupation.z_scores[t]
                 for t in TYPES if t in observed_types), occupation)
            for occupation in self._occupations
        ]
        scored.sort(key=lambda item: (-item[0], item[1].code))
        results = [
            RankedOccupation(
                code=occupation.code, title=occupation.title, rank=rank,
                occupation_z=dict(occupation.z_scores),
                signals={"structured": SignalScore(score=score, rank=rank)},
            )
            for rank, (score, occupation) in enumerate(scored[:top_k], 1)
        ]
        return RankingResponse("ranked", occupations=results)
