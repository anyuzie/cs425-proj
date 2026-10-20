"""Frozen M4 configuration; safe to import without model dependencies."""

BASELINE_PARAMETERS = {
    "model_id": "sentence-transformers/all-MiniLM-L6-v2",
    "model_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "model_dir": "artifacts/ranking/models/all-MiniLM-L6-v2_1110a243",
    "tasks_path": "data/tasks.csv",
    "cache_dir": "artifacts/ranking/minilm_mean_tasks_v1",
    "dimension": 384, "max_seq_length": 256, "batch_size": 32,
    "device": "cpu", "threads": 1, "normalize_embeddings": True,
    "index": "IndexFlatIP", "search_depth": "all_tasks",
    "sentence_splitter": "punctuation_newline_v1",
    "aggregation": "max_task_per_occupation_then_mean_sentences",
    "evidence": "top_3_tasks_by_max_sentence_similarity_provisional",
    "tie_break": "code_ascending", "seed": 425,
}


def create_dense(occupations, parameters):
    try:
        from .dense_ranker import DenseRanker
    except ImportError as error:
        raise ValueError("M4 dependencies missing; install requirements-ranking.txt") from error
    return DenseRanker.from_cache(occupations, parameters)
