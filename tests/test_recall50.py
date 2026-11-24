"""Recall@50 cutoff, frozen extension provenance and nine-to-ten history migration."""
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from recommender.experiments.evaluation import evaluate_ranking, METRICS
from recommender.experiments.evaluate import run
from recommender.experiments.history import FIELDS
from recommender.experiments.metric_extensions import add_recall50, read_extension
from recommender.experiments.compare import paired_comparison


class Recall50Tests(unittest.TestCase):
    def test_cutoff_and_relevant_label_denominator(self):
        codes = [str(i) for i in range(60)]
        related = {"primary_short":["4","9","19","49","50","49","outside"]}
        metrics=evaluate_ranking(codes,"0",related,set(codes))
        self.assertAlmostEqual(metrics["recall_at_50"],5/6)
        self.assertEqual(evaluate_ranking(codes,None,{},set(codes))["recall_at_50"],0)
        self.assertEqual(len(METRICS),10)

