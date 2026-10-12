"""M1 deterministic signed RIASEC scoring; no text retrieval or fusion."""

from pathlib import Path

from .data_loader import load_eligible_occupations
from .schemas import TYPES, RankedOccupation, RankingRequest, RankingResponse, SignalScore


class StructuredRanker:
    def __init__(self, occupations_path: Path = Path("data/occupations.csv")) -> None:
        self._occupations = load_eligible_occupations(occupations_path)

    @property
    def eligible_count(self) -> int:
        return len(self._occupations)

    def rank(self, interests: dict[str, float], observed_types: set[str],
             user_sentences: list[str], top_k: int = 100) -> RankingResponse:
        request = RankingRequest(interests, observed_types, user_sentences, top_k)
        if not request.has_structured_signal:
            return RankingResponse("no_signal", reason="No observed nonzero RIASEC interests; M1 uses structured scoring only.")
        # Fixed dimension order avoids set-iteration-dependent floating-point sums.
        scored = [
            (sum(request.interests[t] * occupation.z_scores[t]
                 for t in TYPES if t in request.observed_types), occupation)
            for occupation in self._occupations
        ]
        scored.sort(key=lambda item: (-item[0], item[1].code))
        results = [
            RankedOccupation(
                code=occupation.code, title=occupation.title, rank=rank,
                occupation_z=dict(occupation.z_scores),
                signals={"structured": SignalScore(score=score, rank=rank)},
            )
            for rank, (score, occupation) in enumerate(scored[:request.top_k], 1)
        ]
        return RankingResponse("ranked", occupations=results)
