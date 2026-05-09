from __future__ import annotations

import math
import time
from typing import Protocol

from tokenpack.models import ScoredChunk


class AmiScorer(Protocol):
    model_name: str

    def ami(self, query: str, context: str) -> float:
        ...


def apply_instruction_ami(
    scored: list[ScoredChunk],
    query_text: str,
    scorer: AmiScorer,
    candidate_pool: int = 50,
    time_budget_seconds: float = 35.0,
    blend_weight: float = 0.35,
) -> dict[str, object]:
    """Rerank only the strongest candidates with model-based AMI under a wall-clock budget."""

    started = time.perf_counter()
    pool = sorted(scored, key=lambda item: item.value, reverse=True)[: max(0, candidate_pool)]
    measured: list[tuple[ScoredChunk, float]] = []
    timed_out = False

    for item in pool:
        if time.perf_counter() - started >= time_budget_seconds:
            timed_out = True
            break
        ami_value = scorer.ami(query_text, item.chunk.text)
        measured.append((item, max(0.0, ami_value)))

    normalized = _minmax([value for _, value in measured])
    measured_ids = set()
    blend = min(1.0, max(0.0, blend_weight))
    for (item, raw_value), normalized_value in zip(measured, normalized, strict=True):
        measured_ids.add(id(item))
        base_value = item.value
        item.value = (1.0 - blend) * base_value + blend * normalized_value
        item.score_components["ami"] = normalized_value
        item.score_components["ami_raw"] = raw_value
        item.score_components["ami_base_value"] = base_value
        item.score_components["ami_blend_weight"] = blend
        item.score_components["ami_model"] = scorer.model_name
        item.score_components["ami_measured"] = 1.0

    for item in pool:
        if id(item) not in measured_ids:
            item.score_components["ami_measured"] = 0.0

    return {
        "ami_model": scorer.model_name,
        "ami_candidates_requested": candidate_pool,
        "ami_candidates_scored": len(measured),
        "ami_time_budget_seconds": time_budget_seconds,
        "ami_blend_weight": blend,
        "ami_elapsed_seconds": time.perf_counter() - started,
        "ami_timed_out": timed_out,
    }


class TransformersAmiScorer:
    """Optional local causal-LM scorer. Import cost and dependencies are paid only when used."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        max_input_tokens: int = 2048,
    ) -> None:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name)
        self._model.eval()
        if device:
            self._device = device
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"
        self._model.to(self._device)
        self._base_loss_cache: dict[str, float] = {}

    def ami(self, query: str, context: str) -> float:
        base_loss = self._base_loss_cache.get(query)
        if base_loss is None:
            base_loss = self._loss_for_query(query=query, context="")
            self._base_loss_cache[query] = base_loss
        conditional_loss = self._loss_for_query(query=query, context=context)
        # Loss reduction is a stable proxy for PPL(q)-PPL(q|c) and avoids overflow on long queries.
        return base_loss - conditional_loss

    def _loss_for_query(self, query: str, context: str) -> float:
        prompt_prefix = f"{context.strip()}\n\nInstruction:\n" if context.strip() else ""
        text = prompt_prefix + query
        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        input_ids = encoded["input_ids"].to(self._device)
        labels = input_ids.clone()
        if prompt_prefix:
            prefix_ids = self._tokenizer(
                prompt_prefix,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_tokens,
            )["input_ids"]
            prefix_length = min(prefix_ids.shape[-1], labels.shape[-1])
            labels[:, :prefix_length] = -100
        with self._torch.no_grad():
            output = self._model(input_ids=input_ids, labels=labels)
        loss = float(output.loss.detach().cpu())
        return loss if math.isfinite(loss) else 0.0


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return [1.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]
