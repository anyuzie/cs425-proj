"""Compatibility entry point; implementation lives in recommender.models.dense_config."""
import importlib
import sys

_implementation = importlib.import_module("recommender.models.dense_config")
if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
