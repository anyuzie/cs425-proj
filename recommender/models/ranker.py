"""Configuration-driven pipeline; owns validation, eligibility and response assembly."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from ..components import ComponentInput, ComponentRanking
from ..configuration import load_configuration
from ..data_loader import load_eligible_occupations
from .fusion import build_fusion, fusion_identifier, select_ranking, validate_fusion
from ..evidence import select_evidence
from ..profile_policy import apply_policy, validate_policy
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
        if not isinstance(config, dict) or not {"components", "fusion"} <= set(config) or set(config) - {"components", "fusion", "evidence_filter", "reranking", "profile_policy"}:
            raise ValueError("ranking config requires components and fusion; optional evidence_filter, reranking, profile_policy")
        self.evidence_filter = config.get("evidence_filter", "raw")
        if self.evidence_filter not in ("raw", "active_only"):
            raise ValueError("evidence_filter must be raw or active_only")
        self.profile_policy = validate_policy(config.get("profile_policy", "raw"))
        selections = config["components"]
        self._reranker = None
        self._rerankers = []
        self._semantic_pipeline = None
        if "reranking" in config:
            from ..semantic import SemanticMatchingPipeline
            SemanticMatchingPipeline.validate(config["reranking"])
            if selections != BASELINE_CONFIG["components"] or config["fusion"] is not None:
                raise ValueError("description reranking requires weighted dot product only and fusion=null")
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
        if "reranking" in config:
            self._semantic_pipeline = SemanticMatchingPipeline(occupations_path, set(self._occupations), config["reranking"])
            self._rerankers = self._semantic_pipeline.stages
            self._reranker = self._rerankers[0]  # Preserve the single-stage compatibility handle.
        self._components = {key: registry.build(resolved, deepcopy(occupations)) for key, resolved in self._resolved.items()}
        self._fusion = build_fusion(config["fusion"]) if config["fusion"] is not None else None

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
                **({"evidence_filter": self.evidence_filter} if "evidence_filter" in self._config else {}),
                **({"profile_policy": self.profile_policy} if "profile_policy" in self._config else {}),
                **({"reranking": deepcopy(self._config["reranking"])} if self._reranker else {})}

    @property
    def component_identifiers(self):
        return {key: self._resolved[key].identifier if key in self._resolved else "" for key in SIGNAL_NAMES}

    @property
    def fusion_identifier(self):
        # Experiment configuration, with actual application counts recorded by evaluation.
        return fusion_identifier(self._config["fusion"]) if len(self._components) > 1 else ""

    @property
    def component_metadata(self):
        return {**{key: deepcopy(getattr(component, "metadata", {})) for key, component in self._components.items()},
                **({"reranking": self._semantic_pipeline.metadata} if self._semantic_pipeline else {})}

    def _profile(self, request):
        """Stage A: the only place profile_policy alters the structured input."""
        interests, observed, diagnostics = apply_policy(
            self.profile_policy, request.interests, request.observed_types, request.slot_reasons)
        # An excluded type leaves the weights with the observed set; weights are never rescaled.
        weights = (None if request.slot_weights is None
                   else {t: float(w) for t, w in request.slot_weights.items() if t in observed})
        return interests, observed, weights, (diagnostics if self.profile_policy != "raw" else None)

    def rank(self, interests: dict[str, float], observed_types: set[str],
             user_sentences: list[str], top_k: int = 100, *, slot_reasons=None, slot_weights=None) -> RankingResponse:
        return self.rank_with_snapshot(interests, observed_types, user_sentences, top_k,
                                       slot_reasons=slot_reasons, slot_weights=slot_weights)[0]

    def rank_with_snapshot(self, interests: dict[str, float], observed_types: set[str],
                           user_sentences: list[str], top_k: int = 100, *, slot_reasons=None, slot_weights=None):
        """Same ranking as rank(), plus every eligible occupation's fused and component ranks.

        The snapshot is read-only explanation data (SEMANTIC_LEDGER_PLAN section 10.3): it is
        {"fused": {code: rank}, "<signal>": {code: rank}, ...}, or None when no snapshot applies.
        It is returned separately so RankingResponse and its exports keep their schema.
        """
        if self.evidence_filter != "raw":
            raise ValueError("active_only requires source_evidence; call rank_with_evidence")
        request = RankingRequest(interests, observed_types, user_sentences, top_k, slot_reasons=slot_reasons,
                                 slot_weights=slot_weights)
        interests, observed, weights, profile = self._profile(request)
        inputs = ComponentInput(interests, frozenset(observed), tuple(request.user_sentences), weights)
        response, snapshot = self._rank_full(inputs, request.top_k)
        if profile:
            response.query_diagnostics = {**response.query_diagnostics, "profile": profile}
        return response, snapshot

    def rank_with_evidence(self, interests, observed_types, user_sentences, source_evidence, top_k=100, *,
                           slot_reasons=None, slot_weights=None):
        request = RankingRequest(interests, observed_types, user_sentences, top_k, slot_reasons=slot_reasons,
                                 slot_weights=slot_weights)
        sentences, diagnostics = select_evidence(request.user_sentences, source_evidence, self.evidence_filter)
        interests, observed, weights, profile = self._profile(request)
        # Only text changes here; the vector changes only through profile_policy.
        response = self._rank(ComponentInput(interests, frozenset(observed), tuple(sentences), weights), top_k)
        response.query_diagnostics = {**response.query_diagnostics, **diagnostics, **({"profile": profile} if profile else {})}
        return response

    def rank_oracle(self, inputs: OracleInput) -> RankingResponse:
        if self.evidence_filter != "raw":
            raise ValueError("active_only requires tracker evidence snapshots; oracle inputs have no lifecycle metadata")
        if self.profile_policy != "raw":
            raise ValueError("profile_policy requires tracker slot reasons; oracle inputs have none")
        if not isinstance(inputs, OracleInput):
            raise ValueError("rank_oracle requires OracleInput")
        inputs.__post_init__()
        return self._rank(ComponentInput(dict(inputs.interests), frozenset(TYPES), tuple(inputs.user_sentences)), self.eligible_count)

    def _rank(self, inputs: ComponentInput, top_k: int) -> RankingResponse:
        return self._rank_full(inputs, top_k)[0]

    def _rank_full(self, inputs: ComponentInput, top_k: int):
        rankings = {}
        for key, component in self._components.items():
            # Components receive independent snapshots and cannot mutate each other.
            result = component.score(deepcopy(inputs))
            if not isinstance(result, ComponentRanking):
                raise ValueError(f"{key}: component must return ComponentRanking")
            result.__post_init__()
            result.validate_pool(self.eligible_codes)
            rankings[key] = result
        selected, fused = select_ranking(rankings, self._config["fusion"], self._fusion)
        selected.validate_pool(self.eligible_codes)
        semantic_diagnostics = {}
        if any(r.algorithm == 'occupation_dual_encoder' for r in self._resolved.values()):
            semantic_diagnostics = {'occupation_semantic': {
                'requested': self.selection_config, 'resolved': self.component_identifiers,
                'components': {key: {'status': r.status, 'reason': r.reason} for key,r in rankings.items()},
                'model_index': self._components['m4'].metadata,
                'query': self._components['m4'].query_diagnostics(inputs),
                'composition': fusion_identifier(self._config['fusion']) if fused else 'single_usable_component',
                'observed_types': sorted(inputs.observed_types),
                'xgboost_invoked': False, 'xgboost_reason': 'no compatible career-relevance labels/schema for semantic features',
                'fallback_reason': '; '.join(f'{k}: {r.reason}' for k,r in rankings.items() if r.status != 'usable') or None}}
        if selected.status != "usable":
            return RankingResponse("no_signal", reason=selected.reason, query_diagnostics=semantic_diagnostics), None
        scores = {key: {item.code: item for item in ranking.occupations} for key, ranking in rankings.items()}
        evidence = {}
        for ranking in rankings.values():
            for item in ranking.evidence:
                evidence.setdefault(item.occupation_code, []).append(item)
        results = []
        depth = max(top_k, 100) if self._reranker else top_k
        for rank, item in enumerate(selected.occupations[:depth], 1):
            occupation = self._occupations[item.code]
            signals = {SIGNAL_NAMES[key]: SignalScore(values[item.code].score, values[item.code].rank)
                       for key, values in scores.items() if item.code in values}
            results.append(RankedOccupation(occupation.code, occupation.title, rank, dict(occupation.z_scores),
                                            rrf_score=item.score if fused else None, signals=signals,
                                            evidence=deepcopy(evidence.get(item.code, []))))
        response = RankingResponse("ranked", occupations=results, query_diagnostics=semantic_diagnostics)
        snapshot = None
        if self._reranker:
            # Reranking reorders only the top of the list, so a full-pool snapshot would be wrong.
            response = self._semantic_pipeline.rerank(response, inputs.user_sentences)
        else:
            snapshot = {"fused": {item.code: item.rank for item in selected.occupations},
                        **{SIGNAL_NAMES[key]: {code: item.rank for code, item in values.items()}
                           for key, values in scores.items()}}
        response.occupations = response.occupations[:top_k]
        return response, snapshot
