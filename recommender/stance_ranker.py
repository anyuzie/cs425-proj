"""Compatibility entry point; implementation lives in recommender.models.stance_ranker."""
import importlib
import sys

_implementation = importlib.import_module("recommender.models.stance_ranker")
if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
