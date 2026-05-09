from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChunkSizeConfig:
    target_tokens: int
    min_tokens: int
    max_tokens: int


CHUNK_SIZE_PRESETS: dict[str, ChunkSizeConfig] = {
    "default": ChunkSizeConfig(target_tokens=650, min_tokens=120, max_tokens=900),
    "low-budget": ChunkSizeConfig(target_tokens=250, min_tokens=40, max_tokens=320),
}


def resolve_chunk_size_config(
    preset: str,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
) -> ChunkSizeConfig:
    if preset == "manual":
        return ChunkSizeConfig(
            target_tokens=target_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )
    try:
        return CHUNK_SIZE_PRESETS[preset]
    except KeyError as exc:
        choices = ", ".join(["manual", *CHUNK_SIZE_PRESETS])
        raise ValueError(f"Unknown chunk size preset: {preset}. Choose one of: {choices}.") from exc
