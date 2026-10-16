"""Freeze-smoke on previously seen VALIDATION only; never opens test data."""
from __future__ import annotations

import csv
from pathlib import Path

import lightgbm as lgb
from submission.experiments.phase3b_full_qasper_train import (
    DEFAULT_CACHE,DEFAULT_DATA,prepare_split_batched,
)
from submission.experiments.phase3b_native_qasper import load_local_qasper_rows
from submission.experiments.phase3d_frozen_qasper_test import (
    MODEL,select_before_gold,verify_frozen_sources,
)
from tokenpack.embeddings import make_embedder
from tokenpack.tokenization import TokenCounter


def main():
    verify_frozen_sources()
    rows=load_local_qasper_rows(
        DEFAULT_DATA/"validation-00000-of-00001.parquet")[:5]
    counter=TokenCounter()
    embedder=make_embedder(
        "sentence-transformers/all-MiniLM-L6-v2",local_files_only=True)
    questions,_=prepare_split_batched(
        "validation",rows,embedder,counter,DEFAULT_CACHE)
    model=lgb.Booster(model_file=str(MODEL))
    output,_,_=select_before_gold(questions,model,counter)
    previous={}
    old=Path("submission/results/phase3c_qasper_full889_optimized_20261011/paired_contexts.csv")
    with old.open(encoding="utf8") as f:
        for item in csv.DictReader(f):
            key=(item["case_id"],float(item["budget_ratio"]),item["method"])
            previous[key]=item
    for row in output:
        key=(row["case_id"],row["budget_ratio"],row["method"])
        orig=previous[key]
        assert row["actual_tokens"]==int(orig["rendered_tokens"]),key
        assert abs(row["gold_5gram_recall"]-float(orig["gold_5gram_recall"]))<2e-7,key
    assert len(output)==len(questions)*8
    print(f"PRETEST SMOKE VERIFIED {len(questions)} validation cases and {len(output)} matching frozen contexts.",flush=True)


if __name__=="__main__":
    main()
