"""Extraction-quality diagnostics for extractor-derived ranking inputs."""
from unittest import TestCase

from recommender.experiments.extractor_inputs import extraction_diagnostics, pearson


class ExtractionDiagnosticsTests(TestCase):
    def test_observed_sign_agreement_and_top_type(self):
        hidden = dict(R=-1., I=2., A=.2, S=1., E=0., C=-.8)
        readings = [
            {"id": "a", "observed_types": ["I", "S", "C"], "interests": dict(R=0, I=1., A=0, S=.55, E=0, C=-.55)},
            {"id": "b", "observed_types": ["R", "A"], "interests": dict(R=.55, I=0, A=.2, S=0, E=0, C=0)},
            {"id": "c", "observed_types": [], "interests": dict.fromkeys("RIASEC", 0.)},
        ]
        profiles = [{"id": i, "hidden_slots": hidden} for i in "abc"]
        d = extraction_diagnostics(readings, profiles)
        self.assertEqual(d["profiles_with_no_observed_type"], 1)
        self.assertAlmostEqual(d["mean_observed_types"], 5 / 3)
        self.assertAlmostEqual(d["top_type_agreement"], 1 / 3)
        self.assertEqual(d["per_type"]["R"]["sign_agreement_when_hidden_abs_ge_0_5"], 0.)
        self.assertEqual(d["per_type"]["I"]["sign_agreement_when_hidden_abs_ge_0_5"], 1.)
        self.assertIsNone(d["per_type"]["A"]["sign_agreement_when_hidden_abs_ge_0_5"])  # |.2| < .5
        self.assertAlmostEqual(d["per_type"]["I"]["observed_rate"], 1 / 3)
        self.assertAlmostEqual(pearson([1, 2, 3], [2, 4, 6]), 1.)
        self.assertIsNone(pearson([1, 2], [1, 2]))
