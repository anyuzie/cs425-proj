"""Compatibility entry point; implementation lives in recommender.data.data_loader."""
import importlib
import sys

_implementation = importlib.import_module("recommender.data.data_loader")
if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
