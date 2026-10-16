"""Runs every contract mixin: real components where they exist, test fakes (tests/fakes.py) otherwise."""

import unittest

from contracts import (DATA, ConfirmerContract, ExtractorContract, FillerContract, PolicyContract,
                       RankerContract, RuleContract, TrackerContract)
import fakes  # noqa: F401  (registers the test-* fakes)
from pipeline import registry


def factory(kind, name):
    return lambda self: registry.REGISTRY[kind][name]({"settings": {}}, DATA)


class TestScriptedExtractor(ExtractorContract, unittest.TestCase):
    make = factory("extractor", "test-scripted")


class TestSimpleTracker(TrackerContract, unittest.TestCase):
    make = factory("tracker", "test-simple-tracker")


class TestContradictionRule(RuleContract, unittest.TestCase):
    make = factory("rule", "rule")


class TestPassthroughConfirmer(ConfirmerContract, unittest.TestCase):
    make = factory("confirmer", "test-passthrough")


class TestStructuredRanker(RankerContract, unittest.TestCase):
    make = factory("ranker", "structured")


class TestStructuredKeywordRanker(RankerContract, unittest.TestCase):
    make = factory("ranker", "test-keyword-ranker")


class TestSimplePolicy(PolicyContract, unittest.TestCase):
    make = factory("policy", "test-simple-policy")


class TestQuestionBankFiller(FillerContract, unittest.TestCase):
    make = factory("filler", "question-bank")


if __name__ == "__main__":
    unittest.main()
