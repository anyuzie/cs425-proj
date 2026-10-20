"""Compatibility entry point; implementation lives in recommender.preparation.build_dense."""
import importlib
import sys

_implementation = importlib.import_module("recommender.preparation.build_dense")
if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
