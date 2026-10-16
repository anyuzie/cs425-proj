"""STEP 5b · MiniLM + FAISS task matching (milestone M4).  STATUS: not built yet.

Used by RRFRanker in recommender/fusion.py. Index built by scripts/build_indexes.py
-> indexes/faiss.idx (one vector per task statement) + indexes/faiss_owners.json
Model: all-MiniLM-L6-v2 (no training, used as is). Manual part A3.
The matched task statements are what the recommendation cards quote.
"""

from pathlib import Path

from .schemas import TaskEvidence  # noqa: F401


class DenseSignal:
    def __init__(self, index_path: Path, owners_path: Path, embedder: str) -> None:
        raise NotImplementedError("recommender/dense.py is not filled in yet (docs/INTEGRATION.md). "
                                  "Delete this line once the code below is written.")
        # ================= PUT YOUR CODE HERE: load once =================
        #   self.model  = SentenceTransformer(embedder)
        #   self.index  = faiss.read_index(str(index_path))
        #   self.owners = json.load(open(owners_path))   # owners[i] = {"code", "task_id", "text"} of vector i
        # ==================================================================

    def rank(self, user_sentences: list[str]) -> list[tuple[str, float, list[TaskEvidence]]]:
        """Return [(occupation_code, score, [TaskEvidence, ...]), ...] best first.
        TaskEvidence(task_id, occupation_code, text, match_score) for the tasks that matched."""
        # ================= PUT YOUR CODE HERE =================
        # 1. embed the user's sentences (normalize_embeddings=True)
        # 2. nearest 50 task vectors -> group by occupation
        # 3. occupation score = mean of its top 3 task scores; keep those tasks as evidence
        # ======================================================
        raise NotImplementedError
