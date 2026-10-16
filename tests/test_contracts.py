"""Runs every contract mixin against the current stand-ins."""

import unittest

from contracts import (DATA, ConfirmerContract, ExtractorContract, FillerContract, PolicyContract,
                       RankerContract, RuleContract, TrackerContract)
from pipeline import registry


def factory(kind, name):
    return lambda self: registry.REGISTRY[kind][name]({"settings": {}}, DATA)


class TestLexiconExtractor(ExtractorContract, unittest.TestCase):
    make = factory("extractor", "lexicon-stub")


class TestSimpleTracker(TrackerContract, unittest.TestCase):
    make = factory("tracker", "simple-stub")


class TestContradictionRule(RuleContract, unittest.TestCase):
    make = factory("rule", "rule")


class TestPassthroughConfirmer(ConfirmerContract, unittest.TestCase):
    make = factory("confirmer", "passthrough-stub")


class TestStructuredKeywordRanker(RankerContract, unittest.TestCase):
    make = factory("ranker", "structured+keyword-stub")


class TestRulePolicy(PolicyContract, unittest.TestCase):
    make = factory("policy", "rule-stub")


class TestQuestionBankFiller(FillerContract, unittest.TestCase):
    make = factory("filler", "question-bank")


if __name__ == "__main__":
    unittest.main()
