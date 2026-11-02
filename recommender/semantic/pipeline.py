"""Semantic-only stage orchestration. Does not read or compute RIASEC values."""
from copy import deepcopy

from ..models.description_reranker import DescriptionReranker, VARIANTS
from ..models.sequential_description_config import SEQUENTIAL_VARIANTS
from ..schemas import SignalScore
from .aggregation_config import AGGREGATED_PARAMETERS
from .context_tasks import CONFIGURATIONS, ContextTaskReranker


class SemanticMatchingPipeline:
    @staticmethod
    def validate(parameters):
        if parameters not in CONFIGURATIONS.values() and parameters != AGGREGATED_PARAMETERS and parameters not in SEQUENTIAL_VARIANTS and parameters not in VARIANTS.values():
            raise ValueError("reranking requires a frozen description variant")

    def __init__(self, occupations_path, eligible_codes, parameters):
        self.validate(parameters)
        self.parameters = deepcopy(parameters)
        context_tasks = parameters in CONFIGURATIONS.values()
        self.sequential = parameters["algorithm"] == "sequential_descriptions"
        self.aggregated = parameters["algorithm"] == "aggregated_descriptions"
        selections = parameters["stages"] if self.sequential or self.aggregated else [parameters]
        scorer = ContextTaskReranker if context_tasks else DescriptionReranker
        if context_tasks and not (self.sequential or self.aggregated):
            selections = [next(value for value in VARIANTS.values()
                               if value["variant"] + "_context_tasks_v1" == parameters["variant"])]
        self.stages = [scorer(occupations_path, eligible_codes, parameters=stage)
                       for stage in selections]

    @property
    def metadata(self):
        if not self.sequential and not self.aggregated:
            return {**self.stages[0].metadata, "parameters": deepcopy(self.parameters)}
        return {"parameters": deepcopy(self.parameters), "stages": [stage.metadata for stage in self.stages]}

    def rerank(self, response, utterances):
        """Consume an ordered shortlist and text; preserve structured data as opaque fields."""
        if self.aggregated:
            return self._aggregate(response, utterances)
        if not self.sequential:
            return self.stages[0].rerank(response, utterances)
        for stage in self.stages:
            name = stage.parameters["algorithm"].removesuffix("_descriptions")
            response = stage.rerank(response, utterances, signal_name=f"semantic_{name}")
            for occupation in response.occupations[:100]:
                signal = occupation.signals.get(f"semantic_{name}")
                if signal is not None:
                    occupation.signals[f"stage_{name}"] = SignalScore(signal.score, occupation.rank)
        for occupation in response.occupations[:100]:
            final_name = self.stages[-1].parameters["algorithm"].removesuffix("_descriptions")
            if f"semantic_{final_name}" in occupation.signals:
                occupation.signals["semantic"] = deepcopy(occupation.signals[f"semantic_{final_name}"])
        return response

    def _aggregate(self, response, utterances):
        if response.status != "ranked" or not any(text.strip() for text in utterances):
            return response
        candidates = response.occupations[:self.parameters["candidate_count"]]
        codes = [o.code for o in candidates]
        # Both encoders receive the identical original shortlist. Neither sorts it.
        scores = {stage.parameters["algorithm"].removesuffix("_descriptions"):
                  stage.score(codes, utterances) for stage in self.stages}
        weights = self.parameters["weights"]
        combined = {code: sum(weights[name] * scores[name][code] for name in weights) / sum(weights.values())
                    for code in codes}
        rankings = {name: {code: rank for rank, code in enumerate(
                    sorted(values, key=lambda c: (-values[c], c)), 1)}
                    for name, values in {**scores, "aggregate": combined}.items()}
        size = self.parameters["block_size"]
        ordered = [o for start in range(0, len(candidates), size)
                   for o in sorted(candidates[start:start+size], key=lambda o: (-combined[o.code], o.code))]
        for rank, occupation in enumerate(ordered, 1):
            code = occupation.code
            for name, values in scores.items():
                occupation.signals[f"semantic_{name}"] = SignalScore(values[code], rankings[name][code])
            occupation.signals["semantic_aggregate"] = SignalScore(combined[code], rankings["aggregate"][code])
            occupation.signals["semantic"] = deepcopy(occupation.signals["semantic_aggregate"])
            occupation.signals["stage_aggregate"] = SignalScore(combined[code], rank)
        response.occupations = ordered + response.occupations[len(ordered):]
        for rank, occupation in enumerate(response.occupations, 1):
            occupation.rank = rank
        return response
