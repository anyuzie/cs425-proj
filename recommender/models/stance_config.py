"""Versioned stance experiment settings; safe without model dependencies."""

from ..configuration import load_parameters

PARAMETERS = load_parameters("stance_config_parameters")
POSITIVE_PARAMETERS = load_parameters("stance_config_positive_parameters")


def create_stance(occupations, parameters):
    from .stance_ranker import StanceTaskRanker
    return StanceTaskRanker.from_cache(occupations, parameters)
