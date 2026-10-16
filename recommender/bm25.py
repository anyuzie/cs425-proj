"""STEP 5a · BM25 keyword signal (milestone M3).  STATUS: not built yet.

Used by RRFRanker in recommender/fusion.py. Index built by scripts/build_indexes.py -> indexes/bm25.pkl
No model: rank_bm25 over each pool occupation's bm25_doc column (manual part A2).
"""

from pathlib import Path


class BM25Signal:
    def __init__(self, index_path: Path) -> None:
        raise NotImplementedError("recommender/bm25.py is not filled in yet (docs/INTEGRATION.md). "
                                  "Delete this line once the code below is written.")
        # ================= PUT YOUR CODE HERE: load once =================
        #   data = pickle.load(open(index_path, "rb"))
        #   self.index, self.codes = data["index"], data["codes"]   # codes[i] = occupation of doc i
        # ==================================================================

    def rank(self, user_sentences: list[str]) -> list[tuple[str, float]]:
        """user_sentences  the user's own free-text messages (not short replies like "the first one")
        Return [(occupation_code, score), ...] best first. Empty list if no usable words."""
        # ================= PUT YOUR CODE HERE =================
        # tokens = " ".join(user_sentences).lower().split()
        # scores = self.index.get_scores(tokens) -> pair with self.codes -> sort desc, drop score 0
        # ======================================================
        raise NotImplementedError
