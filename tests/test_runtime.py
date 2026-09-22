"""The runtime's pure functions — the parts that must match Laya's reference exactly.

These need no weights and no onnxruntime session. They guard the port: a change
here that looks harmless can shift every probability the model reports.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from runtime import (  # noqa: E402
    QTYPES,
    confidence_from_probs,
    render_criterion,
    render_options,
    serialize_state,
    temp_bucket,
    to_internal,
)


class StateSerialization(unittest.TestCase):
    def test_strings_pass_through_untouched(self):
        self.assertEqual(serialize_state("already text"), "already text")

    def test_dicts_become_compact_json(self):
        self.assertEqual(serialize_state({"a": 1}), '{"a": 1}')

    def test_non_ascii_survives(self):
        """ensure_ascii=False matters: escaped text tokenizes differently."""
        self.assertIn("é", serialize_state({"x": "café"}))


class OptionRendering(unittest.TestCase):
    def test_choice_renders_label_and_description(self):
        q = to_internal({"type": "choice", "instructions": "i",
                         "criteria": {"a": "first", "b": "second"}})
        self.assertEqual(render_options(q), ["a: first", "b: second"])

    def test_choice_without_description_renders_bare_label(self):
        q = to_internal({"type": "choice", "instructions": "i", "criteria": {"a": None, "b": ""}})
        self.assertEqual(render_options(q), ["a", "b"])

    def test_choice_keeps_falsy_but_real_criteria(self):
        """0 and False are legitimate descriptions; only None and '' mean 'no description'."""
        q = to_internal({"type": "choice", "instructions": "i", "criteria": {"a": 0, "b": False}})
        self.assertEqual(render_options(q), ["a: 0", "b: false"])

    def test_choice_given_a_list_becomes_labels_without_descriptions(self):
        q = to_internal({"type": "choice", "instructions": "i", "criteria": ["x", "y"]})
        self.assertEqual(render_options(q), ["x", "y"])

    def test_option_order_follows_criteria_order(self):
        """Answers are decoded by index, so a reordering would mislabel every result."""
        crit = {"first": "1", "second": "2", "third": "3"}
        q = to_internal({"type": "choice", "instructions": "i", "criteria": crit})
        self.assertEqual([o.split(":")[0] for o in render_options(q)], list(crit))

    def test_score_renders_ordinal_levels(self):
        q = to_internal({"type": "score", "instructions": "i", "criteria": ["low", "high"]})
        self.assertEqual(render_options(q), ["level 0: low", "level 1: high"])

    def test_noul_is_always_false_then_true(self):
        q = to_internal({"type": "noul", "instructions": "i"})
        opts = render_options(q)
        self.assertEqual(len(opts), 2)
        self.assertTrue(opts[0].startswith("false:"))
        self.assertTrue(opts[1].startswith("true:"))

    def test_noul_accepts_custom_wording(self):
        q = to_internal({"type": "noul", "instructions": "i",
                         "criteria": {"false": "nope", "true": "yep"}})
        self.assertEqual(render_options(q), ["false: nope", "true: yep"])

    def test_structured_criteria_render_as_json_not_python_repr(self):
        self.assertEqual(render_criterion({"desc": "x"}), '{"desc": "x"}')
        self.assertNotIn("'", render_criterion({"desc": "x"}))


class Normalisation(unittest.TestCase):
    def test_question_keys_are_renamed_for_the_sequence_builder(self):
        q = to_internal({"type": "noul", "instructions": "text"})
        self.assertEqual(q["t"], "noul")
        self.assertEqual(q["ins"], "text")

    def test_non_string_instructions_become_json(self):
        q = to_internal({"type": "noul", "instructions": {"a": 1}})
        self.assertIsInstance(q["ins"], str)


class Confidence(unittest.TestCase):
    def test_certainty_scores_one(self):
        self.assertAlmostEqual(confidence_from_probs(np.array([1.0, 0.0]), 2), 1.0, places=6)

    def test_uniform_scores_zero(self):
        self.assertAlmostEqual(confidence_from_probs(np.array([0.5, 0.5]), 2), 0.0, places=6)

    def test_single_option_is_trivially_certain(self):
        self.assertEqual(confidence_from_probs(np.array([1.0]), 1), 1.0)

    def test_matches_the_entropy_definition(self):
        p = np.array([0.7, 0.2, 0.1])
        expected = 1.0 - (-(p * np.log(p)).sum()) / math.log(3)
        self.assertAlmostEqual(confidence_from_probs(p, 3), expected, places=9)

    def test_result_stays_within_zero_and_one(self):
        for p in (np.array([0.5, 0.5]), np.array([0.999, 0.001]), np.array([0.34, 0.33, 0.33])):
            c = confidence_from_probs(p, len(p))
            self.assertGreaterEqual(c, 0.0)
            self.assertLessEqual(c, 1.0)


class TemperatureBuckets(unittest.TestCase):
    def test_buckets_partition_by_option_count(self):
        self.assertEqual(temp_bucket(QTYPES["noul"], 2), "noul:2")
        self.assertEqual(temp_bucket(QTYPES["choice"], 4), "choice:3-5")
        self.assertEqual(temp_bucket(QTYPES["choice"], 8), "choice:6-10")
        self.assertEqual(temp_bucket(QTYPES["choice"], 28), "choice:11+")

    def test_boundaries_land_in_the_lower_bucket(self):
        self.assertEqual(temp_bucket(QTYPES["choice"], 5), "choice:3-5")
        self.assertEqual(temp_bucket(QTYPES["choice"], 10), "choice:6-10")

    def test_every_primitive_has_a_distinct_prefix(self):
        names = {temp_bucket(v, 2).split(":")[0] for v in QTYPES.values()}
        self.assertEqual(names, set(QTYPES))


if __name__ == "__main__":
    unittest.main()
