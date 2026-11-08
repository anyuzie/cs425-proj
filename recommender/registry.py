"""Compatibility entry point; implementation lives in recommender.models.registry."""
import importlib
import sys

_implementation = importlib.import_module("recommender.models.registry")
if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
