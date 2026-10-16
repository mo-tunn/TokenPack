"""Offline pinned cross-encoder and gold-blind scoring plumbing unit tests."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import torch

from submission.experiments import phase5b_pretrained_rerank_eval as mod
from submission.experiments.phase5b_msmarco_300_eval import METHODS as METHODS_300


def test_offline_checksum_verifier_checks_exact_weights(monkeypatch,tmp_path):
    weights=tmp_path/"model.safetensors"
    weights.write_bytes(b"tiny fixture is NOT a downloaded model")
    monkeypatch.setattr(mod,"FOLDER",tmp_path)
    monkeypatch.setattr(mod,"WEIGHTS_SHA256",hashlib.sha256(
        weights.read_bytes()).hexdigest())
    assert mod.verify_local_model()==weights.stat().st_size
    weights.write_bytes(b"tampered")
    import pytest
    with pytest.raises(AssertionError,match="SHA256"):
        mod.verify_local_model()


def test_crossencoder_candidate_score_order_preserved_with_batches():
    class Tokenizer:
        def __call__(self,questions,texts,**kwargs):
            return {"input_ids":torch.tensor(
                [[int(s.split("-")[-1])] for s in texts],dtype=torch.long)}

    class Model:
        def eval(self):return self
        def __call__(self,**inputs):
            return SimpleNamespace(logits=inputs["input_ids"].float())

    c=[SimpleNamespace(chunk=SimpleNamespace(text=f"candidate-{i}"))
       for i in range(7)]
    got=mod.score(Model(),Tokenizer(),"Which is correct?",c,
                  "cpu",512,3)
    assert got==[float(i) for i in range(7)]


def test_comparison_is_budget_and_paper_paired_by_definition():
    assert mod.RATIOS==(.25,.5)
    assert mod.METHODS==(
        "manual-render","lambdamart-render","msmarco-zero-shot")
    assert METHODS_300==(
        "manual-300","msmarco-650","msmarco-300","lambdamart-650")
