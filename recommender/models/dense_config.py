"""Dense task retrieval configuration; safe to import without model dependencies."""

from ..configuration import load_parameters

BASELINE_PARAMETERS = load_parameters("dense_config_baseline_parameters")


def create_dense(occupations, parameters):
    try:
        from .dense_ranker import DenseRanker
    except ImportError as error:
        raise ValueError("M4 dependencies missing; install requirements-ranking.txt") from error
    return DenseRanker.from_cache(occupations, parameters)
