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

    def test_frozen_extension_and_history_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);report=root/"first";history=root/"history.csv"
            run(output=report,history=history)
            original={name:(report/name).read_bytes() for name in ("metrics.json","manifest.json","per_profile.json")}
            extra=add_recall50(report)
            self.assertEqual(extra["aggregate"]["recall_at_50"],json.loads(original["metrics.json"])["recall_at_50"])
            self.assertEqual(read_extension(report),extra)
            comparison,_=paired_comparison(report,report,samples=20)
            self.assertEqual(comparison["metrics"]["recall_at_50"]["ci95"],[0,0])
            self.assertTrue(all((report/name).read_bytes()==data for name,data in original.items()))
            rows=list(csv.DictReader(io.StringIO(history.read_text())))
            fields=[f for f in FIELDS if f!="recall_at_50"]
            with history.open('w',newline='',encoding='utf-8') as handle:
                writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore')
                writer.writeheader();writer.writerows(rows)
            before=history.read_bytes()
            run(output=root/"second",history=history)
            self.assertEqual(history.with_suffix('.csv.pre_recall50.bak').read_bytes(),before)
            after=list(csv.DictReader(io.StringIO(history.read_text())))
            self.assertEqual(after[0]['recall_at_50'],'')
            self.assertNotEqual(after[1]['recall_at_50'],'')
            self.assertEqual({f:after[0][f] for f in fields},{f:rows[0][f] for f in fields})
            with (report/'predictions.jsonl').open('a') as handle:handle.write('\n')
            with self.assertRaisesRegex(ValueError,'no longer matches'):
                read_extension(report)
