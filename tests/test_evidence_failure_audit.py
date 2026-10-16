"""Deterministic, offline tests for phase-1 evidence diagnostic."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / "submission" / "experiments" / "evidence_failure_audit.py"
SPEC = importlib.util.spec_from_file_location("evidence_failure_audit", MODULE)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def qrow(case: str, method: str, context: str, evidence: list[str]) -> dict:
    return {
        "case_id":case,"method":method,"context":context,
        "evidence_texts":evidence,"context_tokens":len(context.split()),
    }


def jrow(case: str, method: str, correct: bool, answer: str="") -> dict:
    return {
        "case_id":case,"method":method,"status":"completed",
        "judge_parse_failed":False,"correctness_0_2":2 if correct else 0,
        "answer":answer,
    }


class EvidenceAuditTests(unittest.TestCase):
    def test_coverage_uses_token_multiset_and_contiguous_fivegrams(self):
        tok,shingle = audit.evidence_coverage(
            "The cat sat on the mat and the dog barked",
            ["the cat sat on the mat"],
        )
        self.assertEqual(tok,1.0)
        self.assertEqual(shingle,1.0)
        tok,shingle = audit.evidence_coverage(
            "the cat sat on the x the mat",
            ["the cat sat on the mat"],
        )
        self.assertEqual(tok,1.0)
        self.assertLess(shingle,1.0)

    def test_empty_evidence_is_unknown_not_zero(self):
        self.assertEqual(audit.evidence_coverage("text",[]),(None,None))

    def test_chunk_set_differs_from_order(self):
        a="[Chunk 1: id=chunk-aa, source=x, tokens=3]\na\n[Chunk 2: id=chunk-bb, source=x, tokens=3]\nb"
        b="[Chunk 1: id=chunk-bb, source=x, tokens=3]\nb\n[Chunk 2: id=chunk-aa, source=x, tokens=3]\na"
        result=audit.chunk_comparison(a,b)
        self.assertTrue(result["same_chunk_set"])
        self.assertFalse(result["same_chunk_order"])
        self.assertFalse(result["identical_context"])

    def test_identical_context_disagreement_is_not_evidence_loss(self):
        prompt="[Chunk 1: id=chunk-aa, tokens=8]\nthe cat sat on the mat"
        methods=["tokenpack-50","budget-top-k-50"]
        tasks=[qrow("caseA",m,prompt,["the cat sat on the mat"]) for m in methods]
        judged=[jrow("caseA",methods[0],False),jrow("caseA",methods[1],True)]
        cases,summary=audit.analyse_qasper(tasks,judged)
        self.assertEqual(cases[0]["candidate_failure"],"same_context_conflicting_outcome_check_judge")
        self.assertEqual(summary["identical_contexts"],1)
        self.assertEqual(summary["identical_context_same_prompt_outcome_disagreements"],1)
        self.assertEqual(summary["outcomes"]["tokenpack_loss"],1)

    def test_loss_with_missing_gold_evidence(self):
        tasks=[
            qrow("caseA","tokenpack-50","hello",[]),
            qrow("caseA","budget-top-k-50","world",[]),
        ]
        judged=[
            jrow("caseA","tokenpack-50",False),
            jrow("caseA","budget-top-k-50",True),
        ]
        cases,_=audit.analyse_qasper(tasks,judged)
        self.assertEqual(cases[0]["candidate_failure"],"no_gold_evidence_annotations")

    def test_same_evidence_different_answer_prompt_is_distinct(self):
        prompt="context remains identical"
        task_a=qrow("caseA","tokenpack-50",prompt,["context remains identical"])
        task_b=qrow("caseA","budget-top-k-50",prompt,["context remains identical"])
        task_a["answer_prompt_variant"]="strict"
        task_b["answer_prompt_variant"]="default"
        cases,summary=audit.analyse_qasper(
            [task_a,task_b],
            [jrow("caseA","tokenpack-50",False),jrow("caseA","budget-top-k-50",True)],
        )
        self.assertEqual(cases[0]["candidate_failure"],"same_context_different_answer_prompt_variant")
        self.assertEqual(summary["identical_context_different_prompt_variant"],1)
        self.assertEqual(summary["identical_context_same_prompt_outcome_disagreements"],0)

    def test_refusal_takes_priority_after_identical_context(self):
        cls=audit.triage_loss(
            "Insufficient information to decide",
            (1.0,1.0),(1.0,1.0),{"identical_context":False,"same_chunk_set":False,"same_chunk_order":False},
        )
        self.assertEqual(cls,"refusal_review_evidence")

    def test_longbench_paired_comparison_does_not_invent_evidence(self):
        common={"case_id":"caseA","model_id":"Qwen/Qwen2.5-14B-Instruct",
                "answer":"A","source_tokens":1000,"length":"short","domain":"QA","difficulty":"easy"}
        rows=[
            {**common,"method":"tokenpack-50","prediction":"B","correct":False,"context_tokens":500},
            {**common,"method":"production-rag-50","prediction":"A","correct":True,"context_tokens":500},
        ]
        cases,summary=audit.analyse_longbench(rows)
        self.assertEqual(summary["outcomes"]["tokenpack_loss"],1)
        self.assertEqual(cases[0]["candidate_failure"],"missing_evidence_spans_manual_review")

    def test_reject_duplicate_cases(self):
        row={"case_id":"abc","method":"tokenpack-50"}
        with self.assertRaises(ValueError):
            audit.index_by_case([row,row])

    def test_csv_writer_and_readout(self):
        with tempfile.TemporaryDirectory() as path:
            output=Path(path)/"cases.csv"
            audit.save_csv(output,[{"case_id":"A","outcome":"tied"}])
            self.assertIn("case_id",output.read_text(encoding="utf8"))


if __name__=="__main__":
    unittest.main()
