from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from submission.longbench_eval.client import build_tasks


class TinyEmbedder:
    model_name = "mock-embedder"

    def embed(self, texts):
        return [[0.8,0.6,0.0] for _ in texts]


class ForbiddenEmbedder:
    model_name = "mock-embedder"

    def embed(self, texts):
        raise AssertionError("Already-cached question must not be embedded again")


def _config(source: Path, checkpoints: Path):
    return argparse.Namespace(
        data_file=str(source), limit=1, max_scanned=1,
        source_min_tokens=0, source_max_tokens=240000,
        embedding_model="mock-embedder", embedding_allow_download=False,
        chunker="structure-aware", scoring="evidence-hybrid",
        selection_strategy="budget-top-k", budget_ratio=0.5,
        context_order="score-then-source", candidate_pool=300,
        target_tokens=30, min_tokens=10, max_tokens=60,
        skip_compression=True, fast_pilot=True,
        checkpoint_dir=str(checkpoints),
    )


class FastPilotTest(unittest.TestCase):
    def test_exact_rag25_uses_direct_quarter_budget_on_full_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/"cases.jsonl"
            case={
                "_id":"rag25-example", "length":"medium",
                "question":"Which alpha evidence is available?",
                "choice_A":"alpha", "choice_B":"beta", "choice_C":"gamma", "choice_D":"delta",
                "answer":"A",
                "context":("Alpha beta gamma evidence. " * 100)
                           +"\n\n"+("Another source document. "*100),
            }
            source.write_text(json.dumps(case)+"\n",encoding="utf-8")
            args=_config(source,root/"rag25-cache")
            args.fast_pilot=False
            args.rag25_only=True
            with patch("submission.longbench_eval.client.make_embedder",return_value=TinyEmbedder()):
                tasks,report=build_tasks(args)
            self.assertEqual(len(tasks),1)
            self.assertEqual(tasks[0]["method"],"production-rag-25")
            self.assertEqual(tasks[0]["budget_tokens"],int(tasks[0]["source_tokens"]*.25))
            self.assertEqual(tasks[0]["timing"]["stages"]["compression"]["status"],"skipped")
            # Strict source content budget, excluding the small rendered citation headers.
            self.assertLessEqual(tasks[0]["selected_content_tokens"],int(tasks[0]["source_tokens"]*.25))
            with patch("submission.longbench_eval.client.make_embedder",return_value=ForbiddenEmbedder()):
                again,_=build_tasks(args)
            self.assertEqual(again[0]["context"],tasks[0]["context"])

    def test_five_methods_resume_without_reembedding_and_fail_closed_on_source_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/"cases.jsonl"
            case={
                "_id":"mock-case", "length":"short", "domain":"Single-Document QA",
                "question":"Which target appeared?", "choice_A":"alpha",
                "choice_B":"beta", "choice_C":"gamma", "choice_D":"delta",
                "answer":"A",
                "context":("Alpha beta gamma evidence. " * 70)+
                          "\n\n"+("The reference document contains alpha. "*70),
            }
            source.write_text(json.dumps(case)+"\n",encoding="utf-8")
            args=_config(source,root/"cache")
            with patch("submission.longbench_eval.client.make_embedder",return_value=TinyEmbedder()):
                tasks,report=build_tasks(args)
            self.assertEqual(len(tasks),5)
            self.assertEqual({x["method"] for x in tasks},{
                "full-context","production-rag-50","tokenpack-50","tokenpack-25","head-tail-50"
            })
            self.assertEqual(report["cases"],1)
            self.assertEqual(len(list((root/"cache").glob("*.json"))),1)
            self.assertTrue(all(x["answer"]=="A" for x in tasks))
            self.assertLessEqual(next(x["context_tokens"] for x in tasks if x["method"]=="tokenpack-25"),
                                 next(x["context_tokens"] for x in tasks if x["method"]=="full-context"))
            # Build twice with no new model inference or chunk embeddings.
            with patch("submission.longbench_eval.client.make_embedder",return_value=ForbiddenEmbedder()):
                cached,_=build_tasks(args)
            self.assertEqual([(x["task_id"],x["context"]) for x in tasks],
                             [(x["task_id"],x["context"]) for x in cached])
            # Change only the hidden answer. Cache is rejected (not silently reused).
            case["answer"]="B"
            source.write_text(json.dumps(case)+"\n",encoding="utf-8")
            with patch("submission.longbench_eval.client.make_embedder",return_value=ForbiddenEmbedder()):
                with self.assertRaisesRegex(ValueError,"Checkpoint differs"):
                    build_tasks(args)


if __name__=="__main__":
    unittest.main()
