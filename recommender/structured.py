"""Backward-compatible M1-only runtime and oracle entry point."""

from pathlib import Path

from .ranker import BASELINE_CONFIG, RankingPipeline


class StructuredRanker(RankingPipeline):
    def __init__(self, occupations_path: Path = Path("data/occupations.csv")) -> None:
        super().__init__(occupations_path, config=BASELINE_CONFIG)
