"""Configuration-driven pipeline; owns validation, eligibility and response assembly."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from ..components import ComponentInput, ComponentRanking
from ..configuration import load_configuration
from ..data_loader import load_eligible_occupations
from .fusion import select_ranking, validate_fusion
from ..evidence import select_evidence
from .registry import AlgorithmRegistry, default_registry
from ..schemas import TYPES, OracleInput, RankedOccupation, RankingRequest, RankingResponse, SignalScore

BASELINE_CONFIG = {"components": {"m1": {"algorithm": "weighted_dot_product", "variant": "baseline_v1"},
                                  "m3": None, "m4": None}, "fusion": {"algorithm": "rrf", "k": 60}}
SIGNAL_NAMES = {"m1": "structured", "m3": "lexical", "m4": "semantic"}


class RankingPipeline:
    def __init__(self, occupations_path: Path = Path("data/occupations.csv"), *,
                 config: Path | dict | None = None, registry: AlgorithmRegistry | None = None):
        registry = registry if registry is not None else default_registry()
        if isinstance(config, (str, Path)):
            config, _ = load_configuration(Path(config))
        config = deepcopy(BASELINE_CONFIG if config is None else config)
        if not isinstance(config, dict) or not {"components", "fusion"} <= set(config) or set(config) - {"components", "fusion", "evidence_filter"}:
            raise ValueError("ranking config requires components and fusion; optional evidence_filter")
        self.evidence_filter = config.get("evidence_filter", "raw")
        if self.evidence_filter not in ("raw", "active_only"):
            raise ValueError("evidence_filter must be raw or active_only")
        selections = config["components"]
        if not isinstance(selections, dict) or set(selections) != set(SIGNAL_NAMES):
            raise ValueError("components must contain exactly m1, m3, m4; use null to disable")
        if self.evidence_filter == "active_only" and selections["m3"] is None and selections["m4"] is None:
            raise ValueError("active_only requires an enabled text ranker (m3 or m4)")
        validate_fusion(config["fusion"])
        # Resolve all selections before loading datasets or constructing components.
        self._resolved = {key: registry.resolve(key, selection) for key, selection in selections.items()
                          if selection is not None}
        self._config = config
        occupations = load_eligible_occupations(occupations_path)
        self._occupations = {o.code: o for o in occupations}
        self._components = {key: registry.build(resolved, deepcopy(occupations)) for key, resolved in self._resolved.items()}

    @property
    def eligible_count(self):
        return len(self._occupations)

    @property
    def eligible_codes(self):
        return set(self._occupations)

    @property
    def selection_config(self):
        return deepcopy(self._config)

    @property
    def resolved_config(self):
        return {"components": {key: asdict(self._resolved[key]) if key in self._resolved else None
                                for key in SIGNAL_NAMES}, "fusion": deepcopy(self._config["fusion"]),
                **({"evidence_filter": self.evidence_filter} if "evidence_filter" in self._config else {})}

    @property
    def component_identifiers(self):
        return {key: self._resolved[key].identifier if key in self._resolved else "" for key in SIGNAL_NAMES}

    @property
    def fusion_identifier(self):
        # Experiment configuration, with actual application counts recorded by evaluation.
        return "rrf_k60" if len(self._components) > 1 and self._config["fusion"] is not None else ""

    @property
    def component_metadata(self):
        return {key: deepcopy(getattr(component, "metadata", {})) for key, component in self._components.items()}

    def rank(self, interests: dict[str, float], observed_types: set[str],
             user_sentences: list[str], top_k: int = 100) -> RankingResponse:
        if self.evidence_filter != "raw":
            raise ValueError("active_only requires source_evidence; call rank_with_evidence")
        request = RankingRequest(interests, observed_types, user_sentences, top_k)
        inputs = ComponentInput(dict(request.interests), frozenset(request.observed_types), tuple(request.user_sentences))
        return self._rank(inputs, request.top_k)

    def rank_with_evidence(self, interests, observed_types, user_sentences, source_evidence, top_k=100):
        request = RankingRequest(interests, observed_types, user_sentences, top_k)
        sentences, diagnostics = select_evidence(request.user_sentences, source_evidence, self.evidence_filter)
        # Only text changes. M1 receives the exact same current vector/observed set.
        response = self._rank(ComponentInput(dict(request.interests), frozenset(request.observed_types), tuple(sentences)), top_k)
        response.query_diagnostics = diagnostics
        return response

    def rank_oracle(self, inputs: OracleInput) -> RankingResponse:
        if self.evidence_filter != "raw":
            raise ValueError("active_only requires tracker evidence snapshots; oracle inputs have no lifecycle metadata")
        if not isinstance(inputs, OracleInput):
            raise ValueError("rank_oracle requires OracleInput")
        inputs.__post_init__()
        return self._rank(ComponentInput(dict(inputs.interests), frozenset(TYPES), tuple(inputs.user_sentences)), self.eligible_count)

    def _rank(self, inputs: ComponentInput, top_k: int) -> RankingResponse:
        rankings = {}
        for key, component in self._components.items():
            # Components receive independent snapshots and cannot mutate each other.
            result = component.score(deepcopy(inputs))
            if not isinstance(result, ComponentRanking):
                raise ValueError(f"{key}: component must return ComponentRanking")
            result.__post_init__()
            result.validate_pool(self.eligible_codes)
            rankings[key] = result
        selected, fused = select_ranking(rankings, self._config["fusion"])
        selected.validate_pool(self.eligible_codes)
        if selected.status != "usable":
            return RankingResponse("no_signal", reason=selected.reason)
        scores = {key: {item.code: item for item in ranking.occupations} for key, ranking in rankings.items()}
        evidence = {}
        for ranking in rankings.values():
            for item in ranking.evidence:
                evidence.setdefault(item.occupation_code, []).append(item)
        results = []
        for rank, item in enumerate(selected.occupations[:top_k], 1):
            occupation = self._occupations[item.code]
            signals = {SIGNAL_NAMES[key]: SignalScore(values[item.code].score, values[item.code].rank)
                       for key, values in scores.items() if item.code in values}
            results.append(RankedOccupation(occupation.code, occupation.title, rank, dict(occupation.z_scores),
                                            rrf_score=item.score if fused else None, signals=signals,
                                            evidence=deepcopy(evidence.get(item.code, []))))
        return RankingResponse("ranked", occupations=results)
