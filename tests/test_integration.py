"""Tracker boundary, policy integration, provenance and configuration regressions."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from m5_fixture import capture, make_system, SCRIPTS
from pipeline.interfaces import State
from recommender.compare import paired_comparison
from recommender.evaluate import run
from recommender.runtime_inputs import runtime_requests
from recommender.schemas import RankingResponse


class IntegrationTests(unittest.TestCase):
    def test_actual_tracker_adapter_requires_explicit_provenance(self):
        profiles, _ = capture(make_system())
        profile = deepcopy(profiles[0])
        profile["input_mode"] = "actual_tracker"
        with self.assertRaisesRegex(ValueError, "test components"):
            runtime_requests(profile, "actual_tracker")
        # Interface acceptance only: this is not exported as an actual evaluation run.
        profile["upstream"] = {"extractor": "deberta", "tracker": "decay", "revision": "fixture-contract-only"}
        requests = runtime_requests(profile, "actual_tracker")
        self.assertEqual(requests[-1].user_sentences, SCRIPTS[0])

    def test_multiturn_conflict_and_raw_history(self):
        system = make_system()
        profiles, traces = capture(system)
        rows = traces[0]["ranking_trace"]
        self.assertNotEqual(rows[0]["occupations"][0]["code"], rows[1]["occupations"][0]["code"])
        self.assertEqual(traces[0]["actions"][1]["action"], "CHALLENGE")
        self.assertEqual(traces[1]["actions"][1]["action"], "PROBE")
        self.assertNotEqual(traces[1]["actions"][2]["action"], "CHALLENGE")
        self.assertEqual(rows[-1]["user_sentences"], SCRIPTS[0])
        self.assertEqual(rows[-1]["interests"]["R"], -1)
        self.assertEqual(len(rows[-1]["occupations"]), 100)
        self.assertEqual(system.ranker.component_identifiers["m3"], "")
        self.assertEqual(system.ranker.component_identifiers["m4"], "")
        for profile in profiles:
            self.assertEqual(len(runtime_requests(profile, "mock_tracker")), 4)

    def test_invalid_runtime_provenance_and_values(self):
        profiles, _ = capture(make_system())
        for mutation in (lambda p: p.update(input_mode="actual_tracker"),
                         lambda p: p.pop("upstream"),
                         lambda p: p["turns"][0]["interests"].update(R=2),
                         lambda p: p["turns"][0].update(observed_types=["X"]),
                         lambda p: p.update(turns=[])):
            profile = deepcopy(profiles[0])
            mutation(profile)
            with self.assertRaises(ValueError):
                runtime_requests(profile, "mock_tracker")

    def test_policy_accepts_short_and_empty_lists(self):
        system = make_system()
        rank = system.ranker.rank_with_snapshot   # System's entry point; rank() delegates to it
        for empty in (False, True):
            def shortened(*args, **kwargs):
                response = rank(*args, **kwargs)[0]
                return (RankingResponse("no_signal", reason="fixture") if empty
                        else RankingResponse("ranked", response.occupations[:2]), None)
            system.ranker.rank_with_snapshot = shortened
            system.policy.max_turns = 1
            state = State()
            result = system.step(state, SCRIPTS[0][0])
            self.assertTrue(result.done)
            self.assertEqual(len(state.ranking), 0 if empty else 2)
            self.assertEqual(result.action, "ABSTAIN" if empty else "RECOMMEND")


    @unittest.skipUnless(Path("artifacts/ranking/minilm_mean_tasks_v1/manifest.json").exists(), "prepare pinned M4 cache for real three-way integration")
    def test_three_way_cached_pipeline_tasks_and_repeated_turns(self):
        system = make_system(Path("configs/ranking/m4_three_way.json"))
        components = dict(system.ranker._components)
        _, traces = capture(system)
        tasks = {}
        with Path("data/tasks.csv").open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                tasks[(row["code"], row["Task ID"])] = row["Task"]
        fused = False
        for trace in traces:
            for turn in trace["ranking_trace"]:
                occupations = turn["occupations"]
                codes = [o["code"] for o in occupations]
                self.assertEqual(len(codes), len(set(codes)))
                self.assertTrue(set(codes) <= system.ranker.eligible_codes)
                for occupation in occupations:
                    fused |= occupation["rrf_score"] is not None
                    self.assertEqual(len(occupation["occupation_z"]), 6)
                    self.assertLessEqual(len(occupation["evidence"]), 3)
                    for task in occupation["evidence"]:
                        self.assertEqual(task["status"], "provisional")
                        self.assertEqual(task["text"], tasks[(occupation["code"], task["task_id"])])
        self.assertTrue(fused)
        self.assertTrue(all(system.ranker._components[k] is v for k, v in components.items()))
