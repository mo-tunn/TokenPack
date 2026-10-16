"""Research-only, gold-blind multiresolution evidence candidate handling.

There are two intentional controls:
 - naive union: combine unique 150, 300, 650 chunks, jointly rescore and
   use the frozen Phase 3C render-aware selector without overlap handling;
 - source overlap: apply a fixed source-span redundancy filter BEFORE the
   identical renderer/selector.

These are not production paths, and do not access labels or gold evidence.
"""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Iterable

from tokenpack.models import Chunk,ScoredChunk

SCALES=(150,300,650)
OVERLAP_THRESHOLD=.80
PRIORITY_COST_EXPONENT=.25


def construct_multiresolution(chunks_by_scale:dict[int,list[Chunk]],
                               vectors_by_scale:dict[int,list[list[float]]]):
    """Deduplicate identical source+text, guarantee globally distinct IDs.

    Keep the shortest-scale representative if the very same source interval
    and text appears in multiple chunk scales. Source spans remain unmodified.
    """
    unique={}
    ordered_chunks=[]
    ordered_vectors=[]
    source_scale={}
    for scale in SCALES:
        chunks=chunks_by_scale[scale]
        vectors=vectors_by_scale[scale]
        if len(chunks)!=len(vectors):
            raise ValueError("Missing embedding for a chunk")
        for c,vector in zip(chunks,vectors,strict=True):
            key=(c.source_path,c.document_index,c.char_start,c.char_end,c.text)
            if key in unique:
                continue
            new_id=f"mres-{scale}-{c.id}"
            if any(x.id==new_id for x in ordered_chunks):
                raise ValueError("Unexpected duplicated multiresolution chunk ID")
            metadata={**c.metadata,"research_resolution":scale}
            new=replace(c,id=new_id,metadata=metadata)
            unique[key]=len(ordered_chunks)
            ordered_chunks.append(new)
            ordered_vectors.append(vector)
            source_scale[new_id]=scale
    return ordered_chunks,ordered_vectors,source_scale


def covered_fraction(short:Chunk,long:Chunk)->float:
    """Fraction of shorter source block set covered by another candidate.

    Native QASPER block char offsets reset at every block, thus they MUST
    NOT be used as global document offsets. For split chunks sharing only
    one original source block, compare local char offsets. Otherwise prefer
    shared stable source block IDs, or paragraph indices as a fallback.
    """
    if short.document_index!=long.document_index or short.source_path!=long.source_path:
        return 0.
    a=set(short.block_ids)
    b=set(long.block_ids)
    if a and b:
        same_one_block=len(a)==len(b)==1 and a==b
        if same_one_block:
            lo=max(short.char_start,long.char_start)
            hi=min(short.char_end,long.char_end)
            length_a=max(1,short.char_end-short.char_start)
            length_b=max(1,long.char_end-long.char_start)
            return min(1.,max(0,hi-lo)/min(length_a,length_b))
        return len(a&b)/min(len(a),len(b))
    # Fallback for manually created chunks without block ID provenance.
    if short.start_paragraph==short.end_paragraph==long.start_paragraph==long.end_paragraph:
        lo=max(short.char_start,long.char_start)
        hi=min(short.char_end,long.char_end)
        length_a=max(1,short.char_end-short.char_start)
        length_b=max(1,long.char_end-long.char_start)
        return min(1.,max(0,hi-lo)/min(length_a,length_b))
    aa=set(range(short.start_paragraph,short.end_paragraph+1))
    bb=set(range(long.start_paragraph,long.end_paragraph+1))
    return len(aa&bb)/max(1,min(len(aa),len(bb)))


def filter_heavy_overlap(scored:list[ScoredChunk],threshold:float=OVERLAP_THRESHOLD
                         )->tuple[list[ScoredChunk],dict]:
    """Fixed greedy near-containment filter, no labels or evidence knowledge.

    Chunks are prioritized by relevance / estimated source token cost^.25;
    ties break in source order. Items with >=80% overlap of the SHORTER
    source span with a kept chunk are rejected. This ensures at most one
    near-contained chunk is sent to the actual render-aware selector.
    """
    if not 0<threshold<=1:
        raise ValueError("Bad overlap threshold")
    if not scored:return [],{"duplicates_removed":0}
    vals=[float(x.value) for x in scored]
    if not all(math.isfinite(x) for x in vals):
        raise ValueError("Non-finite relevance")
    minimum=min(vals)
    maximum=max(vals)
    def priority(x):
        rel=.1+.9*(x.value-minimum)/max(1e-12,maximum-minimum)
        return (-rel/max(1,x.weight)**PRIORITY_COST_EXPONENT,
                x.chunk.order_key,x.chunk.id)
    prioritized=sorted(scored,key=priority)
    selected=[]
    rejected=0
    for item in prioritized:
        if any(covered_fraction(item.chunk,c.chunk)>=threshold for c in selected):
            rejected+=1
            continue
        selected.append(item)
    return selected,{"duplicates_removed":rejected,
                     "input_candidates":len(scored),
                     "remaining_candidates":len(selected)}


def count_selected_scale(chunks:list[ScoredChunk])->dict[str,int]:
    result={str(s):0 for s in SCALES}
    for scored in chunks:
        value=scored.chunk.metadata.get("research_resolution")
        if value in SCALES:result[str(value)]+=1
    return result
