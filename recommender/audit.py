"""Compatibility entry point; implementation lives in recommender.experiments.audit."""
import importlib
import sys

_implementation = importlib.import_module("recommender.experiments.audit")
if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
