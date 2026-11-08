"""Controlled logits and state: no checkpoint downloads or real model inference."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

try:
    import torch
except ImportError:   # torch is optional: the rest of the suite must run without it
    torch = None

from eval.nli import evaluate, metrics, select_threshold
from pipeline.interfaces import HistoryEntry, SlotState, State
from pipeline.nli import NliConfirmer
from pipeline.rule import ContradictionRule
from pipeline.system import System


def state_for(old_value=1.0, new_value=-1.0, old_conf=0.8, new_conf=0.9):
    old = HistoryEntry(old_value, old_conf, "enjoyment", "I enjoy repairing machines.", 1)
    new = HistoryEntry(new_value, new_conf, "enjoyment", "I hate repairing machines.", 2)
    return State(turn=2, slots={"R": SlotState(new_value, new_conf, new.reason, new.source, 2, [old, new])})


@unittest.skipIf(torch is None, "needs torch (pip install -r requirements-models.txt)")
class NliTests(unittest.TestCase):
    def test_load_once_softmax_and_reordered_labels(self):
        model = Mock()
        model.to.return_value = model
        model.config.id2label = {0: "neutral", 1: "contradiction", 2: "entailment"}
        model.return_value.logits = torch.tensor([[0.0, 2.0, 1.0]])
        tokenizer = Mock()
        tokenizer.return_value.to.return_value = {"input_ids": torch.tensor([[1]])}
        with patch("transformers.AutoModelForSequenceClassification.from_pretrained", return_value=model) as load, \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
            confirmer = NliConfirmer()
            for _ in range(2):
                score = confirmer.score("earlier", "now")
                self.assertIs(type(score), float)
                self.assertAlmostEqual(score, torch.softmax(torch.tensor([0., 2., 1.]), 0)[1].item())
            load.assert_called_once()
            model.eval.assert_called_once()
            self.assertEqual(tokenizer.call_args.args, ("earlier", "now"))
            self.assertEqual(model.call_count, 2)

    def test_reject_unverified_labels(self):
        for labels in ({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},
                       {0: "contradiction", 1: "contradiction", 2: "neutral"}):
            model = Mock()
            model.config.id2label = labels
            with patch("transformers.AutoModelForSequenceClassification.from_pretrained", return_value=model):
                with self.assertRaisesRegex(ValueError, "Unverified"):
                    NliConfirmer()

    def test_invalid_model_outputs(self):
        confirmer = NliConfirmer.__new__(NliConfirmer)
        confirmer.device, confirmer.contradiction_index = "cpu", 0
        confirmer.tokenizer = Mock()
        confirmer.tokenizer.return_value.to.return_value = {}
        confirmer.model = Mock()
        for logits in (torch.zeros(1, 2), torch.tensor([[float("nan"), 0., 0.]])):
            confirmer.model.return_value.logits = logits
            with self.assertRaises(ValueError):
                confirmer.score("old", "new")


class CandidateTests(unittest.TestCase):
    def test_candidate_filter_boundaries(self):
        rule = ContradictionRule()
        for kwargs, expected in (({}, 1), ({"old_value": 0.2}, 1),
                                 ({"new_value": 1}, 0), ({"old_value": 0}, 0),
                                 ({"old_conf": 0.4}, 0), ({"new_conf": 0.4}, 0),
                                 ({"old_conf": 0.400001, "new_conf": 0.400001}, 1)):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(len(rule.candidates(state_for(**kwargs))), expected)

    def test_exclusions_and_most_recent_opposite(self):
        rule = ContradictionRule()
        for attr in ("resolved", "challenged"):
            state = state_for()
            getattr(state, attr).add("R")
            self.assertEqual(rule.candidates(state), [])
        state = state_for()
        state.turn = 3
        self.assertEqual(rule.candidates(state), [])
        state = state_for()
        state.slots["R"].history = state.slots["R"].history[-1:]
        self.assertEqual(rule.candidates(state), [])
        state = state_for()
        recent = HistoryEntry(0.55, 0.7, "enjoyment", "Another positive sentence", 1)
        state.slots["R"].history.insert(1, recent)
        self.assertIs(rule.candidates(state)[0].old, recent)
        state = state_for()
        state.slots["I"] = state.slots.pop("R")
        state.slots["I"].history = state.slots["I"].history[-1:]
        self.assertEqual(rule.candidates(state), [])  # no cross-type comparison

    def test_confirmation_boundary_preserves_evidence_and_confidence(self):
        state = state_for()
        system = System(None, None, ContradictionRule(), Mock(), None, None, None, {}, confirm_threshold=0.90)
        for score, count in ((0.899999, 0), (0.90, 0), (0.900001, 1)):
            system.confirmer.score.return_value = score
            flags = system._confirmed(state)
            self.assertEqual(len(flags), count)
            if flags:
                flag = flags[0]
                self.assertIs(flag.old, state.slots["R"].history[0])
                self.assertIs(flag.new, state.slots["R"].history[1])
                self.assertEqual((flag.old.source, flag.new.source),
                                 ("I enjoy repairing machines.", "I hate repairing machines."))
                self.assertEqual((flag.old.confidence, flag.new.confidence, flag.score), (0.8, 0.9, score))
        system.confirmer.reset_mock()
        system._confirmed(state_for(new_value=1))
        system.confirmer.score.assert_not_called()



class EvaluationTests(unittest.TestCase):
    def test_metrics_strict_boundary(self):
        result = metrics([1, 1, 0, 0], [0.9, 0.6, 0.7, 0.2], 0.6)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual((result["precision"], result["recall"], result["f1"]), (0.5, 0.5, 0.5))

    def test_selection_and_freeze_before_test(self):
        rows = [{"a": "dev positive", "b": "b", "split": "dev", "label": "contradiction"},
                {"a": "dev negative", "b": "b", "split": "dev", "label": "elaboration"},
                {"a": "test", "b": "b", "split": "test", "label": "unrelated"}]
        events = []
        def score(a, b):
            events.append(a)
            return {"dev positive": 0.8, "dev negative": 0.5, "test": 0.99}[a]
        report = evaluate(rows, SimpleNamespace(score=score), lambda r: events.append("frozen"))
        self.assertEqual(events, ["dev positive", "dev negative", "frozen", "test"])
        self.assertEqual(report["threshold"], 0.79)
        self.assertEqual(report["test"]["confusion_matrix"], [[0, 1], [0, 0]])
        self.assertEqual(len(report["dev_sweep"]), 51)
        rows[-1]["label"] = "contradiction"
        other = evaluate(rows, SimpleNamespace(score=score), lambda r: None)
        self.assertEqual(report["threshold"], other["threshold"])


if __name__ == "__main__":
    unittest.main()
