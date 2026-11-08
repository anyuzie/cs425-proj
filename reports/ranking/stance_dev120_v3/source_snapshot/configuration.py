"""Safe JSON/YAML mapping loading with explicit paths and duplicate-key errors."""

import json
from pathlib import Path

DEFAULT_EVALUATION_CONFIG = Path("configs/m2.json")
DEFAULT_RANKING_CONFIG = Path("configs/ranking.yaml")


def unique_mapping(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate configuration key: {key}")
        result[key] = value
    return result


def load_configuration(path: Path) -> tuple[dict, bytes]:
    path = Path(path)
    try:
        source = path.read_bytes()
    except FileNotFoundError as error:
        raise ValueError(f"configuration file not found: {path}") from error
    try:
        if path.suffix.lower() == ".json":
            config = json.loads(source.decode("utf-8-sig"), object_pairs_hook=unique_mapping)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as error:
                raise ValueError("YAML support requires PyYAML; run python -m pip install -r requirements.txt") from error

            class UniqueSafeLoader(yaml.SafeLoader):
                pass

            def mapping(loader, node):
                return unique_mapping((loader.construct_object(key), loader.construct_object(value))
                                      for key, value in node.value)

            UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
            try:
                config = yaml.load(source.decode("utf-8-sig"), Loader=UniqueSafeLoader)
            except yaml.YAMLError as error:
                raise ValueError(str(error)) from error
        else:
            raise ValueError("configuration must be a .json, .yaml, or .yml file")
        if not isinstance(config, dict):
            raise ValueError("configuration must be a mapping/object")
        return config, source
    except (ValueError, UnicodeError, TypeError) as error:
        raise ValueError(f"invalid configuration {path}: {error}") from error
