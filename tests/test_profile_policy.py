"""Profile policies: frozen transforms, pipeline wiring and evaluator guards."""
from dataclasses import asdict
from unittest import TestCase

from recommender.experiments.familiarity_policy import subgroup_effects, subgroups
from recommender.experiments.runtime_inputs import runtime_requests
from recommender.models.ranker import RankingPipeline
from recommender.profile_policy import POLICIES, apply_policy
from recommender.schemas import OracleInput, RankingRequest

INTERESTS = dict(R=.55, I=1., A=0., S=-.55, E=0., C=0.)
OBSERVED = {"R", "I", "S"}
REASONS = {"R": "familiarity", "I": "enjoyment", "S": "neither"}


def pipeline(scorer="cosine_similarity", policy=None):
    config = {"components": {"m1": {"algorithm": scorer, "variant": "baseline_v1"}, "m3": None, "m4": None},
              "fusion": None}
    return RankingPipeline(config={**config, "profile_policy": policy} if policy else config)


class ProfilePolicyTests(TestCase):
    def test_frozen_transforms_do_not_mutate_inputs(self):
        interests, observed = dict(INTERESTS), set(OBSERVED)
        same, kept, info = apply_policy("raw", interests, observed, None)
        self.assertEqual((same, kept, info["changed_types"]), (INTERESTS, OBSERVED, []))
        excluded, kept, info = apply_policy("familiarity_excluded_v1", interests, observed, REASONS)
        self.assertEqual(kept, {"I", "S"})
        self.assertEqual(excluded["R"], 0.)
        self.assertEqual(info["changed_types"], ["R"])
        half, kept, _ = apply_policy("familiarity_half_v1", interests, observed, REASONS)
        self.assertEqual(kept, OBSERVED)
        self.assertAlmostEqual(half["R"], .275)
        self.assertEqual((interests, observed), (INTERESTS, OBSERVED))
        with self.assertRaises(ValueError):
            apply_policy("familiarity_half_v1", interests, observed, None)
        with self.assertRaises(ValueError):
            apply_policy("other", interests, observed, REASONS)
        self.assertEqual(set(POLICIES), {"raw", "familiarity_excluded_v1", "familiarity_half_v1"})

    def test_request_validates_reasons_against_observed(self):
        RankingRequest(INTERESTS, OBSERVED, ["x"], slot_reasons=REASONS)
        for bad in ({"R": "familiarity"}, {**REASONS, "R": "boredom"}, {**REASONS, "E": "enjoyment"}):
            with self.assertRaises(ValueError):
                RankingRequest(INTERESTS, OBSERVED, ["x"], slot_reasons=bad)

    def test_pipeline_applies_policy_only_to_structured_input(self):
        raw = pipeline().rank(INTERESTS, OBSERVED, ["text"], 378)
        with_reasons = pipeline().rank(INTERESTS, OBSERVED, ["text"], 378, slot_reasons=REASONS)
        self.assertEqual([asdict(o) for o in raw.occupations], [asdict(o) for o in with_reasons.occupations])
        self.assertEqual(with_reasons.query_diagnostics, {})  # raw output is unchanged by reasons
        excluded = pipeline(policy="familiarity_excluded_v1").rank(INTERESTS, OBSERVED, ["text"], 378, slot_reasons=REASONS)
        manual = pipeline().rank({**INTERESTS, "R": 0.}, {"I", "S"}, ["text"], 378)
        self.assertEqual([o.code for o in excluded.occupations], [o.code for o in manual.occupations])
        self.assertEqual(excluded.query_diagnostics["profile"]["changed_types"], ["R"])
        half = pipeline("weighted_dot_product", "familiarity_half_v1").rank_with_evidence(
            INTERESTS, OBSERVED, ["text"], [{"id": "e1", "turn": 1, "text": "text", "status": "active"}], 378,
            slot_reasons=REASONS)
        manual = pipeline("weighted_dot_product").rank({**INTERESTS, "R": .275}, OBSERVED, ["text"], 378)
        self.assertEqual([o.code for o in half.occupations], [o.code for o in manual.occupations])
        self.assertIn("profile", half.query_diagnostics)

    def test_policy_requires_reasons_and_rejects_oracle(self):
        ranker = pipeline(policy="familiarity_half_v1")
        with self.assertRaises(ValueError):
            ranker.rank(INTERESTS, OBSERVED, ["text"], 10)
        with self.assertRaises(ValueError):
            ranker.rank_oracle(OracleInput(dict(R=1., I=0., A=0., S=0., E=0., C=0.), ["text"]))
        self.assertEqual(ranker.resolved_config["profile_policy"], "familiarity_half_v1")
        with self.assertRaises(ValueError):
            pipeline(policy="unknown")

    def test_runtime_sessions_carry_reasons(self):
        session = {"id": "s", "input_mode": "actual_tracker",
                   "upstream": {"extractor": "deberta", "tracker": "decay", "revision": "r"},
                   "turns": [{"text": "x", "interests": INTERESTS, "observed_types": sorted(OBSERVED),
                              "slot_reasons": REASONS}]}
        self.assertEqual(runtime_requests(session, "actual_tracker")[0].slot_reasons, REASONS)
        del session["turns"][0]["slot_reasons"]
        self.assertIsNone(runtime_requests(session, "actual_tracker")[0].slot_reasons)

    def test_subgroups_split_misleading_from_genuine(self):
        def session(i, reasons, interests):
            return {"id": i, "turns": [{"slot_reasons": reasons, "interests": interests}]}
        hidden = dict(R=-1., I=1., A=0., S=0., E=0., C=0.)
        sessions = [session("a", {"I": "enjoyment"}, {**INTERESTS}),
                    session("b", {"I": "familiarity"}, {**INTERESTS}),
                    session("c", {"R": "familiarity", "I": "familiarity"}, {**INTERESTS})]
        groups = subgroups(sessions, [{"id": i, "hidden_slots": hidden} for i in "abc"])
        self.assertEqual(groups, {"a": "no_familiarity", "b": "genuine_only", "c": "misleading_present"})
        diffs = [{"profile_id": i, "mrr": v, "ndcg_at_10": 0., "hit_at_5": 0} for i, v in (("a", 0.), ("b", -.2), ("c", .4))]
        effects = subgroup_effects(diffs, groups)
        self.assertEqual(effects["misleading_present"]["mrr"], .4)
        self.assertEqual(effects["genuine_only"]["profiles"], 1)
