"""weighted_distance@kappa3_v1 scorer and slot-weight threading through the ranking pipeline (ranking only).

Tracker reason, coverage-gate and terminal tests live with those (separately committed) changes."""
from pathlib import Path
import unittest

from recommender.components import ComponentInput
from recommender.experiments.runtime_inputs import runtime_requests
from recommender.experiments.weighted_distance import subgroup_effects, subgroups
from recommender.models.ranker import RankingPipeline
from recommender.models.registry import default_registry
from recommender.models.weighted_distance import BASELINE_PARAMETERS, WeightedDistance
from recommender.schemas import TYPES, Occupation, RankingRequest

DISTANCE = {"components": {"m1": {"algorithm": "weighted_distance", "variant": "kappa3_v1"}, "m3": None, "m4": None},
            "fusion": None}


def occupation(code, **z):
    return Occupation(code, code, dict.fromkeys(TYPES, 0.0) | z)


def scorer(*occupations):
    return WeightedDistance(list(occupations), BASELINE_PARAMETERS)


class WeightedDistanceScorer(unittest.TestCase):
    def test_formula_penalises_balanced_but_weak_and_one_sided(self):
        # User at about (1.65, 1.65) on A, S: a weak balanced row and a one-sided spike both lose to a fit.
        ranking = scorer(occupation("weak", A=.39, S=.31), occupation("spike", A=4.2, S=.06),
                         occupation("fit", A=1.6, S=1.7)).score(
            ComponentInput(dict.fromkeys(TYPES, 0.0) | {"A": .55, "S": .55}, frozenset("AS"), (), {"A": 2., "S": .5}))
        self.assertEqual([o.code for o in ranking.occupations], ["fit", "weak", "spike"])
        weak = next(o for o in ranking.occupations if o.code == "weak")
        self.assertAlmostEqual(weak.score, -(2. * (1.65 - .39) ** 2 + .5 * (1.65 - .31) ** 2))

    def test_weights_change_the_order(self):
        rows = occupation("a_fits", A=1.5, S=-1.), occupation("s_fits", A=-1., S=1.5)
        interests = dict.fromkeys(TYPES, 0.0) | {"A": .5, "S": .5}
        heavy_a = scorer(*rows).score(ComponentInput(interests, frozenset("AS"), (), {"A": 3., "S": .1}))
        heavy_s = scorer(*rows).score(ComponentInput(interests, frozenset("AS"), (), {"A": .1, "S": 3.}))
        self.assertEqual(heavy_a.occupations[0].code, "a_fits")
        self.assertEqual(heavy_s.occupations[0].code, "s_fits")

    def test_observed_zero_is_a_target_and_unknown_is_ignored(self):
        rows = occupation("flat", R=0.), occupation("high_r", R=2.)
        interests = dict.fromkeys(TYPES, 0.0)
        observed_zero = scorer(*rows).score(ComponentInput(interests, frozenset("R"), (), {"R": 1.}))
        self.assertEqual([o.code for o in observed_zero.occupations], ["flat", "high_r"])
        unknown = scorer(*rows).score(ComponentInput(interests | {"A": .5}, frozenset("A"), (), {"A": 1.}))
        self.assertEqual({o.score for o in unknown.occupations}, {-2.25})   # R ignored; ties broken by code
        self.assertEqual([o.code for o in unknown.occupations], ["flat", "high_r"])

    def test_signal_status_without_weights_or_weight(self):
        rows = (occupation("x", A=1.),)
        interests = dict.fromkeys(TYPES, 0.0) | {"A": .5}
        self.assertEqual(scorer(*rows).score(ComponentInput(interests, frozenset("A"))).status, "unavailable")
        self.assertEqual(scorer(*rows).score(ComponentInput(interests, frozenset("A"), (), {"A": 0.})).status, "unusable")
        self.assertEqual(scorer(*rows).score(ComponentInput(None)).status, "unavailable")

    def test_registered_and_parameters_frozen(self):
        registry = default_registry()
        resolved = registry.resolve("m1", DISTANCE["components"]["m1"])
        self.assertEqual(resolved.parameters["user_scale_kappa"], 3.0)
        with self.assertRaisesRegex(ValueError, "documented parameters"):
            WeightedDistance([], {**BASELINE_PARAMETERS, "user_scale_kappa": 2.0})
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register_variant("m1", "weighted_distance", "kappa3_v1", BASELINE_PARAMETERS)


class SlotWeightThreading(unittest.TestCase):
    interests = dict.fromkeys(TYPES, 0.0) | {"A": .55, "S": .55}

    def test_pipeline_passes_weights_and_dot_cosine_ignore_them(self):
        ranker = RankingPipeline(config=DISTANCE)
        self.assertEqual(ranker.rank(self.interests, {"A", "S"}, [], 5).status, "no_signal")   # no weights
        ranked = ranker.rank(self.interests, {"A", "S"}, [], 5, slot_weights={"A": 1.89, "S": .53})
        self.assertEqual(ranked.status, "ranked")
        for path in ("configs/ranking/baseline_v1.json", "configs/ranking/m1_cosine.json"):
            frozen = RankingPipeline(config=Path(path))
            self.assertEqual(frozen.rank(self.interests, {"A", "S"}, [], 400),
                             frozen.rank(self.interests, {"A", "S"}, [], 400, slot_weights={"A": 9., "S": .1}))

    def test_request_validation(self):
        for bad in ({"A": 1.}, {"A": 1., "S": -1.}, {"A": 1., "S": float("nan")}, {"A": True, "S": 1.}):
            with self.assertRaisesRegex(ValueError, "slot_weights"):
                RankingRequest(self.interests, {"A", "S"}, [], slot_weights=bad)

    def test_familiarity_exclusion_drops_the_excluded_weight(self):
        config = {**DISTANCE, "profile_policy": "familiarity_excluded_v1"}
        response = RankingPipeline(config=config).rank(
            self.interests, {"A", "S"}, [], 5, slot_reasons={"A": "enjoyment", "S": "familiarity"},
            slot_weights={"A": 1., "S": 1.})
        alone = RankingPipeline(config=DISTANCE).rank(dict.fromkeys(TYPES, 0.0) | {"A": .55}, {"A"}, [], 5,
                                                      slot_weights={"A": 1.})
        self.assertEqual([o.code for o in response.occupations], [o.code for o in alone.occupations])

    def test_replay_sessions_carry_weights(self):
        profile = {"input_mode": "actual_tracker", "upstream": {"extractor": "d", "tracker": "t", "revision": "r"},
                   "turns": [{"text": "x", "interests": self.interests, "observed_types": ["A", "S"],
                              "slot_weights": {"A": 1., "S": .5}}]}
        (request,) = runtime_requests(profile, "actual_tracker")
        self.assertEqual(request.slot_weights, {"A": 1., "S": .5})

    def test_subgroups_use_observed_count_only(self):
        sessions = [{"id": i, "turns": [{"observed_types": list("RIASEC"[:n])}]} for i, n in (("a", 3), ("b", 4))]
        groups = subgroups(sessions)
        self.assertEqual(groups, {"a": "sparse_le3", "b": "dense_ge4"})
        diffs = [{"profile_id": "a", "mrr": .5, "ndcg_at_10": 0, "hit_at_5": 1, "recall_at_10": 0},
                 {"profile_id": "b", "mrr": -.1, "ndcg_at_10": 0, "hit_at_5": 0, "recall_at_10": 0}]
        effects = subgroup_effects(diffs, groups, samples=50)
        self.assertEqual(effects["sparse_le3"]["mrr"]["mean_difference"], .5)
        self.assertEqual(effects["dense_ge4"]["mrr"]["regressed"], 1)


if __name__ == "__main__":
    unittest.main()
