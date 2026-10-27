"""The four agreed rules: control arm listens to everything, unsure answers get one retry,
PROBE needs reason confidence above a setting, and recency decay as a setting."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from contracts import DATA
import fakes  # noqa: F401; register test-only perceptions
from pipeline.interfaces import Action, State
from pipeline.policy import ManualPolicy
from pipeline.system import System
from pipeline.tracker import DecayTracker
from test_dialogue_integration import reading


class RulesBase(unittest.TestCase):
    settings = {}

    def setUp(self):
        self.config = json.loads((Path(__file__).parent / "test_config.json").read_text())
        self.config["components"].update(tracker="decay", policy="manual", ranker="structured")
        self.config["settings"].update(input_mode="mock_tracker", **self.settings)
        self.system = System.from_config(DATA, self.config)
        self.state = State()
        self.system.start(self.state)

    def step(self, item):
        with patch.object(self.system.extractor, "read", return_value=item):
            return self.system.step(self.state, item.text)


class ControlArmListens(RulesBase):
    """Rule 1: with no_probing, opposite statements are merged like any other evidence."""

    def test_full_arm_holds_the_contradiction_for_a_challenge(self):
        self.step(reading("I love machines", R=(1, .9)))
        result = self.step(reading("I hate machines", R=(-1, .9)))
        self.assertEqual(result.action, "CHALLENGE")
        self.assertEqual(self.state.slots["R"].value, 1)          # held, not merged, until answered

    def test_control_arm_merges_both_statements_and_never_challenges(self):
        self.system.policy = ManualPolicy(self.system.max_turns, no_probing=True)
        self.step(reading("I love machines", R=(1, .9)))
        result = self.step(reading("I hate machines", R=(-1, .9)))
        self.assertNotEqual(result.action, "CHALLENGE")
        self.assertAlmostEqual(self.state.slots["R"].value, 0)    # both count equally: no strong view
        self.assertFalse(self.state.conflict_candidates)
        self.assertFalse(self.state.flags)


class UnsureGetsOneRetry(RulesBase):
    """Rule 2: one retry with a concrete example, then move on."""

    def test_clarify_retry_once_then_mark_no_clear_preference(self):
        first = self.step(reading("I enjoy research", I=(1, .9)))
        asked = first.action == "CLARIFY" and self.state.last_action.slot
        retry = self.step(reading("I am not sure"))
        self.assertEqual((retry.action, self.state.last_action.slot), ("ABSTAIN", asked))
        self.assertTrue(self.state.last_action.payload["repair"])            # the example wording
        moved = self.step(reading("I still cannot decide"))
        self.assertIn(asked, self.state.settled)
        self.assertNotEqual((moved.action, self.state.last_action.slot), ("ABSTAIN", asked))

    def test_usable_answer_resets_the_retry(self):
        self.step(reading("I enjoy research", I=(1, .9)))
        asked = self.state.last_action.slot
        self.step(reading("I am not sure"))
        self.step(reading("Yes, definitely", **{asked: (1, .9)}))
        self.assertNotIn(asked, self.state.strikes)
        self.assertNotIn(asked, self.state.settled)

    def test_stuck_challenge_keeps_the_newer_statement_at_half_confidence(self):
        self.step(reading("I like teaching", S=(1, .9)))
        self.assertEqual(self.step(reading("I dislike teaching", S=(-1, .8))).action, "CHALLENGE")
        self.assertEqual(self.step(reading("hmm")).action, "ABSTAIN")          # one retry of the challenge
        result = self.step(reading("hmm"))
        self.assertNotIn(result.action, ("CHALLENGE", "ABSTAIN"))
        self.assertEqual(self.state.slots["S"].value, -1)                      # newer statement kept
        self.assertAlmostEqual(self.state.slots["S"].confidence, .4)          # at half confidence
        self.assertEqual(self.state.unresolved_conflicts, ["S"])
        self.assertFalse(self.state.flags)

    def test_no_clear_preference_counts_as_covered_and_is_revisited_once(self):
        everything = reading(**{t: (.5, .9) for t in "RIASE"})
        self.step(everything)
        self.assertEqual(self.state.last_action.slot, "C")                     # the only one left
        self.step(reading("not sure"))
        self.step(reading("still not sure"))
        self.assertIn("C", self.state.settled)
        actions = []
        for _ in range(6):
            result = self.step(everything)
            actions.append((result.action, self.state.last_action.slot))
            if result.done:
                break
        self.assertEqual(sum(1 for a in actions if a == ("CLARIFY", "C")), 1)  # asked once more, not again
        self.assertEqual(result.action, "RECOMMEND")                           # C does not block recommending
        self.assertEqual(len(result.recommendations), 5)


class ProbeNeedsReasonConfidence(RulesBase):
    """Rule 3: PROBE only when reason confidence > probe_reason_conf (default 0.60)."""

    def test_default_threshold_is_point_six(self):
        self.assertEqual(self.system.policy.probe_reason_conf, .60)
        self.assertNotEqual(self.step(reading("I studied sales", "familiarity", .6, E=(.7, .9))).action, "PROBE")
        self.state = State()
        self.system.start(self.state)
        self.assertEqual(self.step(reading("I studied sales", "familiarity", .61, E=(.7, .9))).action, "PROBE")

    def test_invalid_threshold_is_rejected(self):
        for bad in (-0.1, 1.5, True, "0.6"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ManualPolicy(probe_reason_conf=bad)


class ProbeThresholdSetting(RulesBase):
    settings = {"probe_reason_conf": .8}

    def test_setting_is_read_from_the_config(self):
        self.assertEqual(self.system.policy.probe_reason_conf, .8)
        self.assertNotEqual(self.step(reading("I studied sales", "familiarity", .7, E=(.7, .9))).action, "PROBE")


class RecencyDecay(RulesBase):
    """Rule 4: decay = 1.0 keeps all turns equal (default); 0.9 makes older evidence count for less."""

    def merged(self, decay):
        tracker, state = DecayTracker(decay), State()
        for turn, value in ((1, 1.0), (6, .2)):
            state.turn = turn
            tracker.update(state, reading(I=(value, .9)))
        return state.slots["I"].value

    def test_default_is_off_and_unchanged(self):
        self.assertEqual(self.system.tracker.decay, 1.0)
        self.assertAlmostEqual(self.merged(1.0), (1 * .9 + .2 * .9) / 1.8)    # 0.60, as before

    def test_decay_makes_the_recent_statement_count_more(self):
        w_old = .9 * .9 ** 5                                                  # five turns old
        self.assertAlmostEqual(self.merged(.9), (1 * w_old + .2 * .9) / (w_old + .9))
        self.assertLess(self.merged(.9), self.merged(1.0))

    def test_invalid_decay_is_rejected(self):
        for bad in (0, 1.1, -1, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                DecayTracker(bad)


class DecaySetting(RulesBase):
    settings = {"decay": .9}

    def test_setting_is_read_from_the_config(self):
        self.assertEqual(self.system.tracker.decay, .9)


class ReviewRegressionsPolicy(RulesBase):
    def stuck_challenge(self):
        everything = reading(**{t: (.9, .9) for t in "RIASEC"})
        for _ in range(3):
            self.step(everything)
        self.assertEqual(self.step(reading("I hate machines", R=(-.9, .9))).action, "CHALLENGE")
        self.assertEqual(self.step(reading("hmm")).action, "ABSTAIN")
        return self.step(reading("hmm"))

    def test_give_up_reranks_and_does_not_recommend_on_a_stale_profile(self):
        result = self.stuck_challenge()
        self.assertNotEqual(result.action, "RECOMMEND")                      # the profile just changed
        self.assertEqual(self.state.profile_history[-1]["R"], self.state.slots["R"].value)
        fresh = self.system._rank(self.state, 100)
        self.assertEqual([o.code for o in self.state.ranking], [o.code for o in fresh])

    def test_give_up_keeps_one_ranking_trace_row_per_turn(self):
        self.stuck_challenge()
        turns = [row["turn"] for row in self.state.ranking_trace]
        self.assertEqual(turns, sorted(set(turns)))

    def test_contradiction_held_at_the_last_turn_is_settled_before_recommending(self):
        self.system.policy, self.system.max_turns = ManualPolicy(2), 2
        self.step(reading("I like teaching", S=(1, .9)))
        result = self.step(reading("I dislike teaching", S=(-1, .8)))   # confirmed on the last turn
        self.assertEqual(result.action, "RECOMMEND")
        self.assertEqual(self.state.slots["S"].value, -1)                # the newer statement counts
        self.assertEqual(self.state.unresolved_conflicts, ["S"])
        self.assertEqual([o.code for o in self.state.ranking], [o.code for o in self.system._rank(self.state, 100)])

    def test_repeated_opposite_while_held_updates_the_quote(self):
        self.step(reading("I like teaching and fixing cars", S=(1, .9), R=(1, .9)))
        result = self.step(reading("I dislike teaching and fixing cars", S=(-1, .8), R=(-1, .8)))
        self.assertEqual(result.action, "CHALLENGE")
        asked = self.state.last_action.slot
        other = ({"S", "R"} - {asked}).pop()
        self.assertIn(other, {f.slot for f in self.state.flags})            # held, not asked yet
        self.step(reading("I really can't stand it", **{other: (-1, .9)}))  # not an answer to the asked one
        flag = next(f for f in self.state.flags if f.slot == other)
        self.assertEqual(flag.new.source, "I really can't stand it")        # quotes the latest statement
        self.assertEqual(self.state.slots[other].value, 1)                  # still held, not merged

    def test_give_up_does_not_duplicate_history_and_keeps_the_statement_turn(self):
        self.stuck_challenge()
        history = self.state.slots["R"].history
        self.assertEqual([round(h.value, 2) for h in history].count(-.9), 1)
        self.assertEqual(self.state.slots["R"].turn, 4)                      # when "I hate machines" was said

    def test_everything_unsure_with_a_long_limit_does_not_crash(self):
        self.system.policy, self.system.max_turns = ManualPolicy(20), 20
        for _ in range(20):
            result = self.step(reading("hmm"))
        self.assertTrue(result.done)
        self.assertFalse(self.state.slots)

    def test_plain_retry_lets_t5_word_it(self):
        self.step(reading("I enjoy research", I=(1, .9)))
        self.step(reading("asdfgh"))
        self.assertEqual(self.state.last_action.kind, "ABSTAIN")
        self.assertNotIn("repair", self.state.last_action.payload)          # only "what do you mean" gets the example


class FullArmMergesUnconfirmedOpposites(RulesBase):
    """Review finding 7: the two arms must differ only in asking."""

    def test_rejected_by_nli_is_merged_not_dropped(self):
        self.step(reading("I like working with people", S=(1, .9)))
        with patch.object(self.system.confirmer, "score", return_value=.1):   # NLI: an elaboration, not a conflict
            result = self.step(reading("I need quiet to focus", S=(-1, .9)))
        self.assertNotEqual(result.action, "CHALLENGE")
        self.assertAlmostEqual(self.state.slots["S"].value, 0)               # same as the control arm would hold

    def test_low_confidence_opposite_is_merged(self):
        self.step(reading("I like teaching", S=(1, .9)))
        self.step(reading("teaching is meh", S=(-1, .3)))                    # rule needs > 0.4 on both
        self.assertAlmostEqual(self.state.slots["S"].value, (.9 - .3) / 1.2)


class DecayCompounds(unittest.TestCase):
    def test_three_updates(self):
        tracker, state, d = DecayTracker(.9), State(), .9
        for turn, value in ((1, 1.0), (3, .2), (6, .6)):
            state.turn = turn
            tracker.update(state, reading(I=(value, .9)))
        w1 = .9 * d ** 2 + .9                         # weight after turn 3
        v1 = (1.0 * .9 * d ** 2 + .2 * .9) / w1
        expected = (v1 * w1 * d ** 3 + .6 * .9) / (w1 * d ** 3 + .9)
        self.assertAlmostEqual(state.slots["I"].value, expected)


class EvaluationKeepsSettings(unittest.TestCase):
    def test_compare_builds_both_arms_with_the_configured_probe_threshold(self):
        from eval import dialogue
        system = type("S", (), {})()
        system.max_turns, system.policy = 10, ManualPolicy(probe_reason_conf=.75)
        seen = []
        with patch.object(dialogue, "simulate", side_effect=lambda s, p, b: seen.append(s.policy) or
                          {"metrics": {"hit_at_5": 0, "mrr": 0, "ndcg_at_10": 0}}), \
                patch.object(dialogue, "aggregate", return_value={"hit_at_5": 0, "mrr": 0, "ndcg_at_10": 0}):
            dialogue.compare(system, [{"id": "p1"}], {})
        self.assertEqual([(p.no_probing, p.probe_reason_conf) for p in seen], [(False, .75), (True, .75)])


if __name__ == "__main__":
    unittest.main()
