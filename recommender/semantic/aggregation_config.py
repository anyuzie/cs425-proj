"""Frozen score aggregation experiment; existing sequential variants are separate."""
from copy import deepcopy
from ..models.description_reranker import VARIANTS

AGGREGATED_PARAMETERS = {
    "algorithm": "aggregated_descriptions",
    "variant": "equal_weights_blocks10_v1",
    "candidate_count": 100,
    "block_size": 10,
    "aggregation": "weighted_mean_cosine",
    "score_normalization": "none",
    "weights": {"minilm": 0.5, "mpnet": 0.5},
    "stages": [deepcopy(VARIANTS["blocks10_v1"]), deepcopy(VARIANTS["mpnet_blocks10_v1"])],
}
