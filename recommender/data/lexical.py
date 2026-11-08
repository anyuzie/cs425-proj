"""Deterministic ranking data preprocessing; no model dependencies."""
import re
from recommender.configuration import load_parameters
_PARAMETERS = load_parameters("bm25_ranker_baseline_parameters")
_TOKEN = re.compile(_PARAMETERS["token_pattern"])
_STOP = frozenset(_PARAMETERS["stopwords"])


def tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN.findall(text.lower()) if token not in _STOP]

