# Modal QASPER Generation Quality Eval

This harness runs the short-term generation-quality experiment for TokenPack on Modal GPUs.
It keeps task construction local and sends only ready-to-answer prompts to Modal.

## What It Compares

Default setup builds 200 QASPER validation questions and four context methods:

- `full-document`
- `budget-top-k-50`
- `tokenpack-50`
- `only-llmlingua2-rate050`

Each task is answered by one open model and judged blindly by the same model with a JSON rubric.
The default remote model is `Qwen/Qwen2.5-14B-Instruct`.

## Setup

Install local tooling:

```powershell
pip install -e .[modal,compression]
modal setup
```

Optional, if you use gated Hugging Face models:

```powershell
modal secret create hf-token HF_TOKEN=your_hf_token
$env:TOKENPACK_MODAL_SECRET_NAME="hf-token"
```

Optional GPU override:

```powershell
$env:TOKENPACK_MODAL_GPU="A100-40GB"
```

Default is `L40S`.

## Dry Run

Build two-question tasks and estimate prompt sizes without launching Modal:

```powershell
python submission/modal_generation_eval/client.py --dry-run --limit 2 --skip-llmlingua2
```

For the full four-method task set, omit `--skip-llmlingua2`; this requires LLMLingua-2 dependencies and model access:

```powershell
python submission/modal_generation_eval/client.py --dry-run --limit 5 --compression-allow-download
```

## Run On Modal

Build tasks first:

```powershell
python submission/modal_generation_eval/client.py --limit 200 --compression-allow-download
```

Then run the printed Modal command, or launch directly:

```powershell
python submission/modal_generation_eval/client.py --limit 200 --compression-allow-download --run-modal
```

Equivalent explicit command:

```powershell
$env:PYTHONIOENCODING="utf-8"
python -m modal run submission/modal_generation_eval/app.py --tasks-jsonl submission/results/modal_generation_eval/qasper_generation_tasks.jsonl --output-jsonl submission/results/modal_generation_eval/qasper_generation_judged.jsonl --shard-size 100 --batch-size 2
```

The Modal app also writes per-shard JSONL files to the `tokenpack-modal-generation-eval` Volume under `/data/tokenpack_modal_eval/shards`.
The remote runner is capped at one GPU container to avoid accidental parallel cost spikes; shards run sequentially and reuse the loaded model where Modal keeps the container warm.

## Summarize Results

After a run:

```powershell
python submission/modal_generation_eval/client.py --results-jsonl submission/results/modal_generation_eval/qasper_generation_judged.jsonl
```

This writes:

- `qasper_generation_quality_summary.csv`
- `qasper_generation_quality_pairwise.csv`
- `qasper_generation_quality_table.tex`

Do not wire the generated table into `submission/paper/main.tex` until the smoke and full runs are stable.

## Recommended Progression

1. Smoke: `--limit 5 --shard-size 20`
2. Pilot: `--limit 50 --shard-size 100`
3. Full: `--limit 200 --shard-size 100`

## Targeted Score-Sorted Prompt Ablation

To test only the score-sorted TokenPack prompt variants on the same QASPER split:

```powershell
python submission/modal_generation_eval/client.py --data-file .tokenpack/data/qasper-validation.parquet --output-dir submission/results/modal_generation_eval_qasper_grounded_ablation50 --limit 50 --tokenpack-only --skip-llmlingua2 --tokenpack-variants score-sorted-grounded,score-sorted-extractive --dry-run
```

Then run the printed Modal command, or add `--run-modal`.

The paper text should describe the judge as local/open-source LLM-as-a-judge and not as a human evaluation.
