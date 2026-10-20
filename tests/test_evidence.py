"""M6 lifecycle-aware query selection and controlled ranking tests."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from recommender.bm25_ranker import tokenize
from recommender.compare import paired_comparison
from recommender.data_loader import load_eligible_occupations
from recommender.evidence import select_evidence, validate_evidence
from recommender.evaluate import run
from recommender.ranker import BASELINE_CONFIG, RankingPipeline
from recommender.schemas import TYPES
from recommender.runtime_inputs import runtime_requests


def evidence(history, statuses):
    return [
        {"id": f"e{turn}", "turn": turn, "text": text, "status": status}
        for turn, (text, status) in enumerate(zip(history, statuses), 1)
    ]


class EvidenceSelectionTests(unittest.TestCase):
    def test_raw_and_active_only_selection_preserve_source_provenance(self):
        history = ["I enjoy repairing machinery.", "I take that back; I dislike it."]
        records = evidence(history, ["superseded", "active"])
        original = deepcopy(records)

        raw, raw_diagnostics = select_evidence(history, records, "raw")
        active, active_diagnostics = select_evidence(history, records, "active_only")

        self.assertEqual(raw, history)
        self.assertEqual(active, [history[1]])
        self.assertEqual(raw_diagnostics["selected_source_ids"], ["e1", "e2"])
        self.assertEqual(active_diagnostics["selected_source_ids"], ["e2"])
        self.assertEqual(active_diagnostics["excluded_source_ids"], ["e1"])
        self.assertEqual(active_diagnostics["source_evidence"], records)
        self.assertEqual(records, original)

    def test_empty_active_evidence_is_safe_and_unknown_filter_is_rejected(self):
        history = ["I am unsure about this."]
        records = evidence(history, ["disputed"])
        self.assertEqual(select_evidence(history, records, "active_only")[0], [])
        with self.assertRaisesRegex(ValueError, "evidence filter"):
            select_evidence(history, records, "latest_only")

    def test_evidence_snapshots_are_complete_stable_and_superseded_is_terminal(self):
        history = ["First statement.", "Correction."]
        first = evidence(history[:1], ["disputed"])
        resolved = evidence(history, ["superseded", "active"])
        self.assertEqual(validate_evidence(resolved, history, first), resolved)

        invalid_snapshots = [
            evidence(history, ["active", "active"]),  # changes prior identity
            [{"id": "e1", "turn": 1, "text": "Changed.", "status": "superseded"},
             resolved[1]],
            evidence(history, ["superseded", "active"]),
        ]
        invalid_snapshots[-1][0]["id"] = "e2"
        for invalid in invalid_snapshots:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_evidence(invalid, history, resolved)

        resurrected = evidence(history, ["active", "active"])
        with self.assertRaisesRegex(ValueError, "cannot be reactivated"):
            validate_evidence(resurrected, history, resolved)

    def test_runtime_replay_requires_cumulative_evidence_snapshots(self):
        history = ["I enjoy one activity.", "I now prefer another."]
        profile = {
            "input_mode": "mock_tracker",
            "upstream": {"extractor": "fixture", "tracker": "fixture", "revision": "test-rev"},
            "turns": [
                {"text": history[0], "interests": dict.fromkeys(TYPES, 0.0),
                 "observed_types": [], "source_evidence": evidence(history[:1], ["active"])},
                {"text": history[1], "interests": dict.fromkeys(TYPES, 0.0),
                 "observed_types": [], "source_evidence": evidence(history, ["superseded", "active"])},
            ],
        }
        requests = runtime_requests(profile, "mock_tracker")
        self.assertEqual(requests[0].source_evidence, profile["turns"][0]["source_evidence"])
        self.assertEqual(requests[1].source_evidence, profile["turns"][1]["source_evidence"])

        disappeared = deepcopy(profile)
        disappeared["turns"][1].pop("source_evidence")
        with self.assertRaisesRegex(ValueError, "cannot disappear"):
            runtime_requests(disappeared, "mock_tracker")

        changed = deepcopy(profile)
        changed["turns"][1]["source_evidence"][0]["text"] = "Rewritten original."
        with self.assertRaises(ValueError):
            runtime_requests(changed, "mock_tracker")

    def test_active_only_requires_enabled_text_retrieval(self):
        config = deepcopy(BASELINE_CONFIG)
        config["evidence_filter"] = "active_only"
        with self.assertRaisesRegex(ValueError, "requires an enabled text ranker"):
            RankingPipeline(config=config)

    def test_filter_changes_text_retrieval_not_structured_scores(self):
        ranker = RankingPipeline(config={
            "components": {
                "m1": BASELINE_CONFIG["components"]["m1"],
                "m3": {"algorithm": "bm25", "variant": "baseline_v1"},
                "m4": None,
            },
            "fusion": {"algorithm": "rrf", "k": 60},
            "evidence_filter": "raw",
        })
        occupations = list(ranker._occupations.values())
        token_sets = {occupation.code: set(tokenize(occupation.bm25_doc or ""))
                      for occupation in occupations}
        pair = next(
            ((left, right, sorted(token_sets[left.code] - token_sets[right.code])[0],
              sorted(token_sets[right.code] - token_sets[left.code])[0])
             for left in occupations for right in occupations
             if token_sets[left.code] - token_sets[right.code]
             and token_sets[right.code] - token_sets[left.code]),
            None,
        )
        self.assertIsNotNone(pair, "eligible documents must contain distinct lexical terms")
        superseded, active, stale_term, active_term = pair
        history = [stale_term, active_term]
        records = evidence(history, ["superseded", "active"])
        interests = dict.fromkeys(TYPES, 0.0) | {"I": 0.7}
        observed = {"I"}

        raw = ranker.rank_with_evidence(interests, observed, history, records, ranker.eligible_count)
        active_config = deepcopy(ranker.selection_config)
        active_config["evidence_filter"] = "active_only"
        active_ranker = RankingPipeline(config=active_config)
        filtered = active_ranker.rank_with_evidence(
            interests, observed, history, records, active_ranker.eligible_count)

        raw_by_code = {item.code: item for item in raw.occupations}
        filtered_by_code = {item.code: item for item in filtered.occupations}
        self.assertIn("lexical", raw_by_code[superseded.code].signals)
        self.assertNotIn("lexical", filtered_by_code[superseded.code].signals)
        self.assertEqual(
            {code: item.signals["structured"] for code, item in raw_by_code.items()},
            {code: item.signals["structured"] for code, item in filtered_by_code.items()},
        )
        self.assertEqual(filtered.query_diagnostics["selected_source_ids"], ["e2"])
        self.assertEqual(filtered.query_diagnostics["excluded_source_ids"], ["e1"])
        self.assertEqual(filtered.query_diagnostics["source_evidence"], records)

        no_active = evidence(history, ["disputed", "superseded"])
        empty_query = active_ranker.rank_with_evidence(
            interests, observed, history, no_active, active_ranker.eligible_count)
        self.assertEqual(empty_query.status, "ranked")
        self.assertEqual(empty_query.query_diagnostics["query_sentences"], [])

    def test_evaluator_preserves_inputs_and_records_both_filter_conditions(self):
        occupations = load_eligible_occupations(Path("data/occupations.csv"))
        first = next(o for o in occupations if tokenize(o.bm25_doc or ""))
        second = next(
            o for o in occupations
            if o.code != first.code
            and set(tokenize(o.bm25_doc or "")) - set(tokenize(first.bm25_doc or ""))
            and set(tokenize(first.bm25_doc or "")) - set(tokenize(o.bm25_doc or ""))
        )
        first_terms, second_terms = set(tokenize(first.bm25_doc or "")), set(tokenize(second.bm25_doc or ""))
        history = [sorted(first_terms - second_terms)[0], sorted(second_terms - first_terms)[0]]
        records = evidence(history, ["superseded", "active"])
        interests = dict.fromkeys(TYPES, 0.0) | {"I": 0.7}
        profile = {
            "id": "m6_eval_fixture",
            "input_mode": "mock_tracker",
            "upstream": {"extractor": "fixture", "tracker": "fixture", "revision": "m6-test"},
            "turns": [
                {"text": history[0], "interests": interests, "observed_types": ["I"],
                 "source_evidence": evidence(history[:1], ["active"])},
                {"text": history[1], "interests": interests, "observed_types": ["I"],
                 "source_evidence": records},
            ],
            "target": first.code,
            "related": {"short": [], "long": [], "supplemental": []},
        }
        evaluation = json.loads(Path("configs/m2.json").read_text(encoding="utf-8"))
        evaluation.update(run_id="m6_fixture", input_mode="mock_tracker")
        raw_config = {
            "components": {
                "m1": BASELINE_CONFIG["components"]["m1"],
                "m3": {"algorithm": "bm25", "variant": "baseline_v1"},
                "m4": None,
            },
            "fusion": {"algorithm": "rrf", "k": 60},
            "evidence_filter": "raw",
        }
        active_config = {**raw_config, "evidence_filter": "active_only"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = root / "profiles.jsonl"
            profiles.write_text(json.dumps(profile) + "\n", encoding="utf-8")
            evaluation["profiles"] = str(profiles)
            evaluation_path = root / "evaluation.json"
            evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
            raw_report, active_report = root / "raw", root / "active"
            run(evaluation_path, raw_report, history=root / "raw_history.csv", ranking_config=raw_config)
            run(evaluation_path, active_report, history=root / "active_history.csv",
                ranking_config=active_config)

            self.assertEqual((raw_report / "inputs.json").read_bytes(),
                             (active_report / "inputs.json").read_bytes())
            raw_manifest = json.loads((raw_report / "manifest.json").read_text())
            active_manifest = json.loads((active_report / "manifest.json").read_text())
            self.assertEqual(raw_manifest["evidence_policy_version"], "m6-v1")
            self.assertEqual(raw_manifest["evidence_filter"], "raw")
            self.assertEqual(active_manifest["evidence_filter"], "active_only")
            comparison, differences = paired_comparison(raw_report, active_report, samples=50)
            self.assertEqual(comparison["profiles"], 1)
            self.assertEqual(len(differences), 1)
            turn_rows = json.loads((active_report / "turn_predictions.json").read_text())
            self.assertEqual(turn_rows[-1]["query_diagnostics"]["selected_source_ids"], ["e2"])
            self.assertEqual(turn_rows[-1]["query_diagnostics"]["excluded_source_ids"], ["e1"])

            incomplete = deepcopy(profile)
            incomplete["turns"][0].pop("source_evidence")
            incomplete_profiles = root / "incomplete_profiles.jsonl"
            incomplete_profiles.write_text(json.dumps(incomplete) + "\n", encoding="utf-8")
            evaluation["profiles"] = str(incomplete_profiles)
            incomplete_evaluation = root / "incomplete_evaluation.json"
            incomplete_evaluation.write_text(json.dumps(evaluation), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires source_evidence"):
                run(incomplete_evaluation, root / "incomplete_report", ranking_config=active_config)
            self.assertFalse((root / "incomplete_report").exists())


if __name__ == "__main__":
    unittest.main()
