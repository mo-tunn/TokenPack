from __future__ import annotations

import unittest

from submission.longbench_eval.stratified_pilot import sample
from submission.longbench_eval.windowed_reference import adapt
from tokenpack.tokenization import TokenCounter


class DummyCounter:
    encoding_name = "dummy-token"

    def count(self, text):
        return len(text.split())


class StratifiedPilotTests(unittest.TestCase):
    def test_windowed_reference_is_labeled_honestly_without_changing_other_methods(self):
        methods = (
            "full-context", "tokenpack-50", "production-rag-50",
            "only-longllmlingua-rate050", "tokenpack-50+longllmlingua-rate050",
        )
        rows = []
        for n in range(60):
            for method in methods:
                rows.append({
                    "case_id": f"case-{n}",
                    "task_id": f"case-{n}::{method}", "method":method,
                    "length": ("short","medium","long")[n//20],
                    "context": "a long detailed context " * 80 if method=="full-context" else "short",
                    "source_tokens": 500, "context_tokens": 500,
                })
        ready, report = adapt(rows, token_limit=120, counter=TokenCounter())
        self.assertEqual(len(ready), 300)
        self.assertEqual(report["total_cases"], 60)
        self.assertEqual(set(x["method"] for x in ready if x["method"].startswith("windowed")), {"windowed-reference-210k"})
        self.assertTrue(all(x["reference_window_mode"]=="head_tail" for x in ready if x["method"]=="windowed-reference-210k"))
        self.assertTrue(all(x["context"]=="short" for x in ready if x["method"]!="windowed-reference-210k"))

    def setUp(self):
        self.rows = []
        for s in ("short", "medium", "long"):
            for n in range(25):
                self.rows.append({
                    "_id": f"{s}{n:02d}",
                    "length": s,
                    "context": "hello world " * 30,
                    "question": f"Question {n}?",
                    "choice_A": "A one", "choice_B": "B two",
                    "choice_C": "C three", "choice_D": "D four",
                    "answer": "A", "domain": "qa", "difficulty": "hard",
                })

    def test_determinism_and_no_gold_based_selection(self):
        a, stat = sample(self.rows, seed=107, per_stratum=20,
                         max_source_tokens=200, token_counter=DummyCounter())
        b, _ = sample(list(reversed(self.rows)), seed=107, per_stratum=20,
                      max_source_tokens=200, token_counter=DummyCounter())
        self.assertEqual([x["_id"] for x in a], [x["_id"] for x in b])
        self.assertEqual(stat["sample_counts"], {"short": 20, "medium": 20, "long": 20})
        self.assertEqual(stat["expected_tasks_for_5_methods"], 300)
        new_rows = [{**x, "answer": "D"} for x in self.rows]
        c, _ = sample(new_rows, seed=107, per_stratum=20,
                      max_source_tokens=200, token_counter=DummyCounter())
        self.assertEqual([x["_id"] for x in a], [x["_id"] for x in c])

    def test_missing_stratum_raises_instead_of_truncating(self):
        for x in self.rows:
            if x["length"] == "long" and x["_id"] != "long00":
                x["context"] = "too many tokens " * 100
        with self.assertRaisesRegex(ValueError, "Not enough"):
            sample(self.rows, seed=107, per_stratum=20,
                   max_source_tokens=200, token_counter=DummyCounter())


if __name__ == "__main__":
    unittest.main()
