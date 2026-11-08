import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clause_rewrite_lab.pipeline import ClausePipeline, invariant_reasons


class Generator:
    def generate(self, original):
        return [("I enjoy identifying software errors.", 425)]


class Reconstructor:
    def __init__(self):
        self.inputs = []

    def reconstruct(self, rewrite):
        self.inputs.append(rewrite)
        return "I enjoy finding bugs in computer programs."


class Evaluator:
    def __init__(self, score=.9):
        self.score = score

    def evaluate(self, original, rewrite, reconstruction):
        return self.score, self.score, {key: .9 for key in ["a", "b", "c", "d"]}


class Matcher:
    def __init__(self):
        self.inputs = []

    def match(self, query, top_k):
        self.inputs.append(query)
        return [{"task_text": "Identify software defects", "query": query}]


class FlowTests(unittest.TestCase):
    original = "I enjoy debugging software."

    def test_accepted_diffuseq_output_is_the_task_query(self):
        backward, matcher = Reconstructor(), Matcher()
        result = ClausePipeline(Generator(), backward, Evaluator(), matcher).process(self.original)
        self.assertEqual(backward.inputs, [result["diffuseq_rewrite"]])
        self.assertEqual(matcher.inputs, [result["diffuseq_rewrite"]])
        self.assertNotIn(result["reconstruction"], matcher.inputs)
        self.assertTrue(result["rewrite_accepted"])
        self.assertEqual(result["matching_source"], "diffuseq")

    def test_rejected_rewrite_uses_original_and_labels_preview(self):
        matcher = Matcher()
        result = ClausePipeline(Generator(), Reconstructor(), Evaluator(.2), matcher).process(self.original)
        self.assertFalse(result["rewrite_accepted"])
        self.assertEqual(matcher.inputs, [self.original, result["diffuseq_rewrite"]])
        self.assertEqual(result["matching_source"], "original_fallback")
        self.assertNotIn(result["reconstruction"], matcher.inputs)
        self.assertTrue(result["rejected_rewrite_task_preview"])

    def test_generation_budget_failure_has_explicit_fallback(self):
        class TooLong:
            def generate(self, text):
                raise ValueError("source budget exceeded")
        matcher = Matcher()
        result = ClausePipeline(TooLong(), Reconstructor(), Evaluator(), matcher).process(self.original)
        self.assertIsNone(result["diffuseq_rewrite"])
        self.assertEqual(matcher.inputs, [self.original])
        self.assertEqual(result["generation_errors"], ["source budget exceeded"])

    def test_preference_and_qualifiers_cannot_be_silently_removed(self):
        cases = [("I hate repairing machines.", "I enjoy repairing machines.", "changed_polarity"),
                 ("I might enjoy teaching.", "I enjoy teaching.", "changed_polarity"),
                 ("I sometimes enjoy teaching.", "I enjoy teaching.", "changed_frequency"),
                 ("I love debugging.", "I enjoy debugging.", "changed_strong_preference"),
                 ("I repair 2 engines.", "I repair engines.", "changed_quantities"),
                 ("I enjoy teaching if groups are small.", "I enjoy teaching.", "changed_conditional"),
                 ("I prefer outdoors over desk work.", "I prefer outdoors.", "changed_comparison"),
                 ("I enjoy teaching.", "Do I enjoy teaching?", "question_instead_of_statement")]
        for original, rewrite, reason in cases:
            with self.subTest(original=original):
                self.assertIn(reason, invariant_reasons(original, rewrite))

    def test_empty_or_invalid_requests_are_rejected_before_models(self):
        pipeline = ClausePipeline(None, None, None, None)
        for text in ["", " ", None]:
            with self.assertRaises(ValueError):
                pipeline.process(text)
        with self.assertRaises(ValueError):
            pipeline.process("valid", 21)

    def test_negative_meaning_is_not_accepted_despite_high_cosine(self):
        result = ClausePipeline(Generator(), Reconstructor(), Evaluator(.99), Matcher()).process("I hate debugging software.")
        self.assertFalse(result["rewrite_accepted"])
        self.assertIn("changed_polarity", result["fidelity"]["rejection_reasons"])

    def test_no_existing_pipeline_imports(self):
        import ast
        package = Path(__file__).resolve().parents[1] / "src/clause_rewrite_lab"
        for path in package.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or ""]
                else:
                    continue
                self.assertFalse(any(name.split(".")[0] in {"app", "pipeline", "recommender"} for name in names), path.name)

    def test_real_generator_reserves_target_budget_without_truncating(self):
        from clause_rewrite_lab.generation import DiffuSeqGenerator
        class Tokenizer:
            def encode(self, text, **kwargs):
                return list(range(112))
        generator = DiffuSeqGenerator.__new__(DiffuSeqGenerator)
        generator.tokenizer, generator.length = Tokenizer(), 128
        with self.assertRaisesRegex(ValueError, "source budget"):
            generator.generate("long clause")


if __name__ == "__main__":
    unittest.main()
