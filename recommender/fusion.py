"""STEP 5 · Final ranker: structured + BM25 + dense, fused with RRF (k=60).  STATUS: not built yet.

Replaces: StructuredKeywordRanker in pipeline/stubs.py
Switch on: configs/system.json -> "ranker": "rrf"
Needs: recommender/bm25.py, recommender/dense.py and the indexes from scripts/build_indexes.py.
The structured signal is already done (recommender/structured.py).
"""

from pathlib import Path

from .bm25 import BM25Signal
from .dense import DenseSignal
from .schemas import RankingResponse
from .structured import StructuredRanker

RRF_K = 60


class RRFRanker:
    name = "rrf"

    def __init__(self, occupations_path: Path, bm25_path: Path, faiss_path: Path, owners_path: Path,
                 embedder: str) -> None:
        raise NotImplementedError("recommender/fusion.py is not filled in yet (docs/INTEGRATION.md). "
                                  "Delete this line once the code below is written.")
        self.structured = StructuredRanker(occupations_path)          # done already
        self.bm25 = BM25Signal(bm25_path)                              # fill in bm25.py
        self.dense = DenseSignal(faiss_path, owners_path, embedder)   # fill in dense.py

    @property
    def eligible_count(self) -> int:
        return self.structured.eligible_count

    def rank(self, interests: dict, observed_types: set, user_sentences: list[str], top_k: int = 100) -> RankingResponse:
        """Same inputs and output as StructuredRanker.rank (see recommender/schemas.py).
        Each RankedOccupation must carry `evidence` (the dense task matches): the cards quote the first two."""
        # ================= PUT YOUR CODE HERE =================
        # 1. s = self.structured.rank(interests, observed_types, user_sentences, 378)
        #    b = self.bm25.rank(user_sentences);  d = self.dense.rank(user_sentences)
        # 2. RRF: score(code) = sum over signals of 1 / (RRF_K + rank_in_that_signal)
        # 3. sort desc (ties by code), keep top_k, build RankedOccupation with rrf_score, signals, evidence
        # 4. nothing from any signal -> RankingResponse("no_signal", reason="...")
        # ======================================================
        raise NotImplementedError
