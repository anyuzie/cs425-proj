"""Build the BM25 and FAISS indexes once, offline. STATUS: not built yet.

Run:   .venv/bin/python scripts/build_indexes.py
Needs: pip install -r requirements-models.txt
Writes indexes/bm25.pkl, indexes/faiss.idx, indexes/faiss_owners.json (paths in configs/system.json).
Manual parts A2 and A3.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "indexes"


def main() -> None:
    # ================= PUT YOUR CODE HERE =================
    # 1. pool = occupations.csv rows with job_zone >= 4 and RIASEC scores (378); reuse
    #    recommender.data_loader.load_eligible_occupations for the codes
    # 2. BM25: BM25Okapi([doc.lower().split() for doc in pool.bm25_doc]);
    #    pickle {"index": ..., "codes": [...]} -> OUT / "bm25.pkl"
    # 3. FAISS: tasks.csv rows whose code is in the pool; SentenceTransformer("all-MiniLM-L6-v2")
    #    .encode(tasks, normalize_embeddings=True); faiss.IndexFlatIP(384).add(V)
    #    -> OUT / "faiss.idx"; owners list [{"code", "task_id", "text"}, ...] -> OUT / "faiss_owners.json"
    # ======================================================
    raise NotImplementedError("scripts/build_indexes.py is not filled in yet")


if __name__ == "__main__":
    main()
