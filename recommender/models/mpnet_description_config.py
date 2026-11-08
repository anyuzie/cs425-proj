"""Pinned MPNet replacement for the description encoder; same ranking workflow."""

MPNET_PARAMETERS = {
    "algorithm": "mpnet_descriptions", "variant": "mpnet_blocks10_v1", "candidate_count": 100,
    "block_size": 10,
    "model_id": "sentence-transformers/all-mpnet-base-v2",
    "model_revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
    "dimension": 768, "max_seq_length": 256,
    "query": "newline_join_user_history", "ordering": "cosine_desc_code_asc",
}

MPNET20_PARAMETERS = {**MPNET_PARAMETERS, "variant": "mpnet_blocks20_v1", "block_size": 20}
