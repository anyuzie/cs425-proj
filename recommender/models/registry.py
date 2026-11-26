"""Small explicit algorithm/variant registry. No dynamic imports or plugins."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

from ..components import ScoringComponent
from .m1 import BASELINE_PARAMETERS, WeightedDotProduct
from ..schemas import Occupation

Factory = Callable[[list[Occupation], dict], ScoringComponent]


@dataclass(frozen=True)
class ResolvedComponent:
    component: str
    algorithm: str
    variant: str
    parameters: dict

    @property
    def identifier(self):
        return f"{self.algorithm}@{self.variant}"


class AlgorithmRegistry:
    def __init__(self):
        self._algorithms: dict[tuple[str, str], Factory] = {}
        self._variants: dict[tuple[str, str, str], dict] = {}

    def register_algorithm(self, component: str, algorithm: str, factory: Factory):
        self._validate_ids(component, algorithm)
        key = (component, algorithm)
        if key in self._algorithms:
            raise ValueError(f"algorithm already registered: {component}/{algorithm}")
        self._algorithms[key] = factory

    def register_variant(self, component: str, algorithm: str, variant: str, parameters: dict):
        self._validate_ids(component, algorithm, variant)
        if (component, algorithm) not in self._algorithms:
            raise ValueError(f"unknown algorithm: {component}/{algorithm}")
        key = (component, algorithm, variant)
        if key in self._variants:
            raise ValueError(f"variant already registered: {component}/{algorithm}@{variant}")
        # JSON-safe configuration is required for manifests and fingerprints.
        import json
        json.dumps(parameters, allow_nan=False)
        if not isinstance(parameters, dict):
            raise ValueError("variant parameters must be an object")
        self._variants[key] = deepcopy(parameters)

    @staticmethod
    def _validate_ids(component, *identifiers):
        if component not in {"m1", "m3", "m4"}:
            raise ValueError(f"unknown component: {component}")
        if any(not isinstance(value, str) or not value or "@" in value or value.strip() != value for value in identifiers):
            raise ValueError("algorithm and variant identifiers must be nonempty strings without @")

    def resolve(self, component: str, selection: dict) -> ResolvedComponent:
        if not isinstance(selection, dict) or set(selection) != {"algorithm", "variant"}:
            raise ValueError(f"{component}: selection requires algorithm and variant")
        algorithm, variant = selection["algorithm"], selection["variant"]
        self._validate_ids(component, algorithm, variant)
        if (component, algorithm) not in self._algorithms:
            raise ValueError(f"unknown algorithm: {component}/{algorithm}")
        key = (component, algorithm, variant)
        if key not in self._variants:
            raise ValueError(f"unknown variant: {component}/{algorithm}@{variant}")
        return ResolvedComponent(component, algorithm, variant, deepcopy(self._variants[key]))

    def build(self, resolved: ResolvedComponent, occupations: list[Occupation]) -> ScoringComponent:
        # Re-resolve registered identity instead of trusting mutable external parameters.
        registered = self.resolve(resolved.component, {"algorithm": resolved.algorithm, "variant": resolved.variant})
        if registered != resolved:
            raise ValueError("resolved variant parameters were modified")
        return self._algorithms[(resolved.component, resolved.algorithm)](occupations, deepcopy(resolved.parameters))


def default_registry() -> AlgorithmRegistry:
    registry = AlgorithmRegistry()
    from ..occupation_semantic import OccupationSemantic
    registry.register_algorithm('m4', 'occupation_dual_encoder', OccupationSemantic)
    for name in ('pretrained', 'finetuned'):
        registry.register_variant('m4', 'occupation_dual_encoder', name + '_v1', {
            'index': 'models/occupation_semantics_v1/' + name + '_index'})
    from .occupation_semantic_signed import (SignedOccupationSemantic, PARAMETERS as SIGNED_SEMANTIC_PARAMETERS,
                                             PARAMETERS_V2 as SIGNED_SEMANTIC_PARAMETERS_V2)
    registry.register_algorithm('m4', 'occupation_dual_encoder_signed', SignedOccupationSemantic)
    registry.register_variant('m4', 'occupation_dual_encoder_signed', 'finetuned_v1', SIGNED_SEMANTIC_PARAMETERS)
    registry.register_variant('m4', 'occupation_dual_encoder_signed', 'finetuned_v2', SIGNED_SEMANTIC_PARAMETERS_V2)  # routing v3
    registry.register_algorithm("m1", "weighted_dot_product", WeightedDotProduct)
    registry.register_variant("m1", "weighted_dot_product", "baseline_v1", BASELINE_PARAMETERS)
    from .cosine import CosineSimilarity, BASELINE_PARAMETERS as COSINE_PARAMETERS
    registry.register_algorithm("m1", "cosine_similarity", CosineSimilarity)
    registry.register_variant("m1", "cosine_similarity", "baseline_v1", COSINE_PARAMETERS)
    from .weighted_distance import WeightedDistance, BASELINE_PARAMETERS as DISTANCE_PARAMETERS
    registry.register_algorithm("m1", "weighted_distance", WeightedDistance)
    registry.register_variant("m1", "weighted_distance", "kappa3_v1", DISTANCE_PARAMETERS)
    from .bm25_ranker import BM25Ranker, BASELINE_PARAMETERS as BM25_PARAMETERS
    registry.register_algorithm("m3", "bm25", BM25Ranker)
    registry.register_variant("m3", "bm25", "baseline_v1", BM25_PARAMETERS)
    from .bm25_signed_ranker import (SignedBM25Ranker, IGNORE_PARAMETERS, FILTER_PARAMETERS,
                                     IGNORE_V2_PARAMETERS, FILTER_V2_PARAMETERS, IGNORE_V3_PARAMETERS)
    registry.register_algorithm("m3", "bm25_signed", SignedBM25Ranker)
    registry.register_variant("m3", "bm25_signed", "ignore_negative_v1", IGNORE_PARAMETERS)
    registry.register_variant("m3", "bm25_signed", "filter_negative_v1", FILTER_PARAMETERS)
    registry.register_variant("m3", "bm25_signed", "ignore_negative_v2", IGNORE_V2_PARAMETERS)  # + apostrophe-less cues
    registry.register_variant("m3", "bm25_signed", "filter_negative_v2", FILTER_V2_PARAMETERS)
    registry.register_variant("m3", "bm25_signed", "ignore_negative_v3", IGNORE_V3_PARAMETERS)  # routing v3
    from .dense_config import BASELINE_PARAMETERS as DENSE_PARAMETERS, create_dense
    registry.register_algorithm("m4", "minilm", create_dense)
    registry.register_variant("m4", "minilm", "mean_tasks_v1", DENSE_PARAMETERS)
    from .stance_config import PARAMETERS, POSITIVE_PARAMETERS, create_stance
    registry.register_algorithm("m4", "minilm_stance", create_stance)
    registry.register_variant("m4", "minilm_stance", "stance_internal_v1", PARAMETERS)
    registry.register_variant("m4", "minilm_stance", "positive_internal_v1", POSITIVE_PARAMETERS)
    from .. import configuration
    catalog, _ = configuration.load_configuration(configuration.CONFIG_ROOT / "ranking/variants.json")
    if set(catalog) != {"variants"} or not isinstance(catalog["variants"], list):
        raise ValueError("variant catalog requires a variants list")
    for entry in catalog["variants"]:
        if not isinstance(entry, dict) or set(entry) != {"component", "algorithm", "variant", "parameters"}:
            raise ValueError("invalid variant catalog entry")
        registry.register_variant(entry["component"], entry["algorithm"], entry["variant"], entry["parameters"])
    return registry
