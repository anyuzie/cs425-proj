"""Frozen ordered stage selection; each block is formed from the prior output."""
from copy import deepcopy

from .description_reranker import VARIANTS

SEQUENTIAL_PARAMETERS = {
    "algorithm": "sequential_descriptions",
    "variant": "mpnet20_minilm10_v1",
    "stages": [deepcopy(VARIANTS["mpnet_blocks20_v1"]), deepcopy(VARIANTS["blocks10_v1"])],
}

REVERSE_PARAMETERS = {
    "algorithm": "sequential_descriptions",
    "variant": "minilm20_mpnet10_v1",
    "stages": [deepcopy(VARIANTS["blocks20_v1"]), deepcopy(VARIANTS["mpnet_blocks10_v1"])],
}
SEQUENTIAL_VARIANTS = (SEQUENTIAL_PARAMETERS, REVERSE_PARAMETERS)
