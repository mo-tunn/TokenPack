from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from tokenpack.models import Chunk
from tokenpack.tokenization import TokenCounter


class PromptCompressorBackend(Protocol):
    """Minimal interface shared by LLMLingua and test doubles."""

    def compress_prompt(self, context: list[str] | str, **kwargs: Any) -> dict[str, Any]:
        ...


@dataclass(slots=True)
class CompressionResult:
    compressed_prompt: str
    origin_tokens: int
    compressed_tokens: int
    ratio: float
    metadata: dict[str, Any]

    @property
    def saving_rate(self) -> float:
        if self.origin_tokens <= 0:
            return 0.0
        return 1.0 - self.compressed_tokens / self.origin_tokens


@dataclass(slots=True)
class CompressionConfig:
    compressor: str = "none"
    model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
    rate: float = 0.5
    target_tokens: int = -1
    question: str = ""
    instruction: str = ""
    longllmlingua: bool = False
    llmlingua2: bool = False
    use_context_level_filter: bool = False
    use_sentence_level_filter: bool = False
    use_token_level_filter: bool = True
    device_map: str = "cpu"
    local_files_only: bool = True


def compress_chunks(
    chunks: list[Chunk],
    config: CompressionConfig,
    backend: PromptCompressorBackend | None = None,
    token_counter: TokenCounter | None = None,
) -> CompressionResult:
    """Compress selected chunks while keeping TokenPack selection separate.

    The intended composition is:
    TokenPack chunk selection -> LLMLingua intra-context compression.
    """

    counter = token_counter or TokenCounter()
    contexts = [chunk.text for chunk in chunks if chunk.text.strip()]
    origin_tokens = sum(counter.count(text) for text in contexts)
    if config.compressor == "none":
        prompt = "\n\n".join(contexts)
        return CompressionResult(
            compressed_prompt=prompt,
            origin_tokens=origin_tokens,
            compressed_tokens=origin_tokens,
            ratio=1.0,
            metadata={"compressor": "none"},
        )
    if config.compressor != "llmlingua":
        raise ValueError(f"Unknown compressor: {config.compressor}")

    kwargs = _llmlingua_kwargs(config)
    llmlingua = backend or _make_llmlingua_backend(config)
    payload = llmlingua.compress_prompt(contexts, **kwargs)
    compressed_prompt = str(payload.get("compressed_prompt", ""))
    compressed_tokens = _int_or_default(payload.get("compressed_tokens"), counter.count(compressed_prompt))
    payload_origin = _int_or_default(payload.get("origin_tokens"), origin_tokens)
    ratio = _ratio_from_payload(payload.get("ratio"), payload_origin, compressed_tokens)
    return CompressionResult(
        compressed_prompt=compressed_prompt,
        origin_tokens=payload_origin,
        compressed_tokens=compressed_tokens,
        ratio=ratio,
        metadata={
            "compressor": "llmlingua",
            "model_name": config.model_name,
            "longllmlingua": config.longllmlingua,
            "llmlingua2": config.llmlingua2,
            "rate": config.rate,
            "target_tokens": config.target_tokens,
            "raw": payload,
        },
    )


def _make_llmlingua_backend(config: CompressionConfig) -> PromptCompressorBackend:
    try:
        from llmlingua import PromptCompressor
    except ImportError as exc:
        raise RuntimeError(
            "LLMLingua is not installed. Install optional compression dependencies with "
            "`pip install tokenpack-rag[compression]` or `pip install -e .[compression]`."
        ) from exc
    model_name = _resolve_local_model_path(config.model_name) if config.local_files_only else config.model_name
    return PromptCompressor(
        model_name=model_name,
        device_map=config.device_map,
        model_config={"local_files_only": config.local_files_only},
        use_llmlingua2=config.llmlingua2,
    )


def _resolve_local_model_path(model_name: str) -> str:
    if Path(model_name).exists():
        return model_name
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to resolve cached LLMLingua models."
        ) from exc
    try:
        return snapshot_download(repo_id=model_name, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"LLMLingua model is not cached locally: {model_name}. "
            "Run once with `--allow-download` for `pack`, use `--compression-allow-download` for "
            "`export-context`, or download the model before offline use."
        ) from exc


def _llmlingua_kwargs(config: CompressionConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "instruction": config.instruction,
        "question": config.question,
        "rate": config.rate,
        "target_token": config.target_tokens,
        "use_context_level_filter": config.use_context_level_filter,
        "use_sentence_level_filter": config.use_sentence_level_filter,
        "use_token_level_filter": config.use_token_level_filter,
    }
    if config.longllmlingua:
        if not config.question.strip():
            raise ValueError("LongLLMLingua compression requires a non-empty question.")
        kwargs.update(
            {
                "rank_method": "longllmlingua",
                "condition_compare": True,
                "condition_in_question": "after_condition",
                "reorder_context": "sort",
            }
        )
    return kwargs


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ratio_from_payload(value: Any, origin_tokens: int, compressed_tokens: int) -> float:
    if isinstance(value, str) and value.endswith("x"):
        try:
            return float(value[:-1])
        except ValueError:
            pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return origin_tokens / max(1, compressed_tokens)
