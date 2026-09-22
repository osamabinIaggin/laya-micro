"""The decision task: what the model is asked, and the cases it is scored on.

Pure Python — no torch, no onnxruntime — so the workstation harness and the
on-device runtime build identical questions from one definition. If these
diverged, two benchmarks of the same model would stop being comparable.

A task file supplies `action_groups`, `categories`, `base_state` and
`side_questions`; see `cases/zeus_task.json`.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TASK = os.path.join(HERE, "cases", "zeus_task.json")
DEFAULT_CASES = os.path.join(HERE, "cases", "zeus_cases.json")

NO_ACTION = "no physical action is called for"


class Task:
    """A domain: the actions to choose between, and the questions asked alongside."""

    def __init__(self, path=None):
        self.path = path or os.environ.get("LAYA_MICRO_TASK", DEFAULT_TASK)
        with open(self.path) as f:
            spec = json.load(f)
        self.name = spec.get("name", os.path.basename(self.path))
        self.action_groups = spec["action_groups"]
        self.categories = spec["categories"]
        self.base_state = spec["base_state"]
        self.side_questions = spec["side_questions"]
        self.actions = dict(a for g in self.action_groups.values() for a in g.items())
        self.group_of = {a: g for g, acts in self.action_groups.items() for a in acts}

    def state_for(self, utterance):
        return dict(self.base_state, utterance=utterance)

    def flat_questions(self):
        """Every action as one choice — more options than the model card advises."""
        return dict(self.side_questions, action={
            "type": "choice",
            "instructions": "Which single physical action should the robot perform in response?",
            "criteria": dict(self.actions, none=NO_ACTION),
        })

    def first_questions(self):
        """Stage one of the hierarchy: which kind of action."""
        return dict(self.side_questions, category={
            "type": "choice",
            "instructions": "Which kind of physical action should the robot perform in response?",
            "criteria": self.categories,
        })

    def second_questions(self, category):
        """Stage two: which action within the chosen category."""
        return {"action": {
            "type": "choice",
            "instructions": "Which single action should the robot perform in response?",
            "criteria": self.action_groups[category],
        }}


def load_cases(path=None, include_ambiguous=False):
    """Labelled utterances as (utterance, action, vision, about_self, ambiguous).

    Cases marked ambiguous are excluded by default: their intent is genuinely
    debatable, so scoring them measures the labeller rather than the model.
    """
    with open(path or DEFAULT_CASES) as f:
        data = json.load(f)
    return [(c["utterance"], c["action"], c["vision"], float(c["about_self"]),
             bool(c.get("ambiguous", False)))
            for c in data if include_ambiguous or not c.get("ambiguous")]


def majority_baseline(cases, index):
    """The score you get by always answering with the most common label.

    Reported next to every accuracy, because these classes are imbalanced enough
    that a useless model can look good without it.
    """
    from collections import Counter

    counts = Counter(str(c[index]) for c in cases)
    label, hits = counts.most_common(1)[0]
    return label, hits / len(cases)
