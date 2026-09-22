"""Task construction and scoring helpers. No model weights, no network."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task import NO_ACTION, Task, load_cases, majority_baseline  # noqa: E402


class TaskDefinition(unittest.TestCase):
    def setUp(self):
        self.task = Task()

    def test_actions_flatten_without_collision(self):
        counted = sum(len(g) for g in self.task.action_groups.values())
        self.assertEqual(len(self.task.actions), counted,
                         "an action name appears in two groups, so one silently shadows the other")

    def test_every_action_maps_back_to_its_group(self):
        for action in self.task.actions:
            self.assertIn(self.task.group_of[action], self.task.action_groups)

    def test_flat_question_offers_every_action_plus_none(self):
        criteria = self.task.flat_questions()["action"]["criteria"]
        self.assertEqual(set(criteria), set(self.task.actions) | {"none"})
        self.assertEqual(criteria["none"], NO_ACTION)

    def test_hierarchy_covers_the_same_actions_as_the_flat_form(self):
        """Both shapes must be able to reach every action, or they are not comparable."""
        reachable = set()
        for category in self.task.action_groups:
            reachable |= set(self.task.second_questions(category)["action"]["criteria"])
        self.assertEqual(reachable, set(self.task.actions))

    def test_every_group_has_a_category_description(self):
        self.assertTrue(set(self.task.action_groups) <= set(self.task.categories))

    def test_state_carries_the_utterance_without_losing_base_fields(self):
        state = self.task.state_for("come here")
        self.assertEqual(state["utterance"], "come here")
        for k, v in self.task.base_state.items():
            self.assertEqual(state[k], v)

    def test_state_for_does_not_mutate_the_base(self):
        before = json.dumps(self.task.base_state, sort_keys=True)
        self.task.state_for("anything")
        self.assertEqual(before, json.dumps(self.task.base_state, sort_keys=True))

    def test_side_questions_declare_a_known_primitive(self):
        for name, q in self.task.side_questions.items():
            self.assertIn(q["type"], ("choice", "score", "noul"), name)
            self.assertTrue(q["instructions"].strip(), name)


class Cases(unittest.TestCase):
    def test_labels_reference_real_actions(self):
        task = Task()
        valid = set(task.actions) | {"none"}
        for utterance, action, _, _, _ in load_cases(include_ambiguous=True):
            self.assertIn(action, valid, f"{utterance!r} is labelled with an unknown action")

    def test_vision_labels_match_the_vision_question(self):
        options = set(Task().side_questions["vision"]["criteria"])
        for utterance, _, vision, _, _ in load_cases(include_ambiguous=True):
            self.assertIn(vision, options, f"{utterance!r} has an unknown vision label")

    def test_about_self_is_boolean(self):
        for utterance, _, _, about_self, _ in load_cases(include_ambiguous=True):
            self.assertIn(about_self, (0.0, 1.0), utterance)

    def test_utterances_are_unique(self):
        seen = [c[0] for c in load_cases(include_ambiguous=True)]
        self.assertEqual(len(seen), len(set(seen)), "duplicate utterances skew the scores")

    def test_ambiguous_cases_are_excluded_by_default(self):
        self.assertLess(len(load_cases()), len(load_cases(include_ambiguous=True)))

    def test_every_action_appears_in_the_cases(self):
        """An action nobody ever asks for is untested, and its accuracy is unmeasured."""
        task = Task()
        labelled = {c[1] for c in load_cases(include_ambiguous=True)}
        self.assertEqual(set(task.actions) - labelled, set())


class MajorityBaseline(unittest.TestCase):
    def test_baseline_is_the_most_common_label(self):
        cases = [("a", "none", "none", 0.0, False)] * 7 + [("b", "sit", "none", 0.0, False)] * 3
        label, score = majority_baseline(cases, 1)
        self.assertEqual(label, "none")
        self.assertAlmostEqual(score, 0.7)

    def test_baseline_never_below_chance(self):
        cases = load_cases()
        for index in (1, 2, 3):
            _, score = majority_baseline(cases, index)
            self.assertGreaterEqual(score, 1.0 / len({str(c[index]) for c in cases}))


class CustomTaskFile(unittest.TestCase):
    def test_a_different_domain_loads_without_code_changes(self):
        """The point of the task file: retarget by editing JSON, not Python."""
        spec = {
            "name": "toy",
            "base_state": {"machine": "a kettle"},
            "action_groups": {"heat": {"boil": "bring to the boil", "warm": "warm gently"}},
            "categories": {"heat": "applying heat", "none": "nothing to do"},
            "side_questions": {"urgent": {"type": "noul", "instructions": "It is urgent."}},
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(spec, f)
            path = f.name
        try:
            task = Task(path)
            self.assertEqual(set(task.actions), {"boil", "warm"})
            self.assertEqual(set(task.flat_questions()["action"]["criteria"]),
                             {"boil", "warm", "none"})
            self.assertEqual(task.state_for("put it on")["machine"], "a kettle")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
