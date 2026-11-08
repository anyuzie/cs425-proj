import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clause_rewrite_lab.matching import maxsim, windows


class MatchingTests(unittest.TestCase):
    def test_masked_maxsim_matches_upstream_reduction(self):
        torch.manual_seed(425)
        query = torch.randn(32, 128)
        documents = torch.randn(3, 20, 128)
        mask = torch.ones(3, 20, dtype=torch.bool)
        mask[0, 10:] = False
        mask[1, 15:] = False
        # Upstream: document padding -9999, max over documents, sum over query.
        reference = documents @ query.permute(1, 0)
        reference[~mask] = -9999
        reference = reference.max(1).values.sum(-1)
        self.assertTrue(torch.equal(maxsim(query, documents, mask), reference))

    def test_padding_cannot_win_when_valid_scores_are_negative(self):
        query = torch.tensor([[1., 0.]])
        docs = torch.tensor([[[-1., 0.], [0., 0.]]])
        mask = torch.tensor([[True, False]])
        self.assertEqual(maxsim(query, docs, mask).item(), -1.)

    def test_windows_retain_every_source_token_and_stay_within_budgets(self):
        for limit, overlap in [(29, 8), (177, 32)]:
            tokens = list(range(1000))
            chunks = windows(tokens, limit, overlap)
            self.assertEqual(set(token for chunk in chunks for token in chunk), set(tokens))
            self.assertTrue(all(len(chunk) <= limit for chunk in chunks))
        self.assertEqual(windows([], 29, 8), [[]])


if __name__ == "__main__":
    unittest.main()
