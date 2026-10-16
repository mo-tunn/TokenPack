"""Faz24 proposed BASELINE-PRESERVING answer-span contraction.

Not yet run on any new Phase24 data. This module and model instructions are
frozen alongside PRE_GOLD_PROTOCOL. No gold labels in functions or prompts.
A rewrite may be lexically valid but semantically wrong; require independent
blind fidelity review, don't claim substring implies equivalent meaning.
"""
from __future__ import annotations
import math,re,unicodedata
SYSTEM=(
  "You are an answer-SPAN contractor for evidence-grounded multi-hop QA. "
  "You receive a question, a fixed original-source context, and an ALREADY "
  "GENERATED short answer. You must preserve the SAME referent and every "
  "fact asserted by the original answer. If any proposed shortening might "
  "alter its referent, geographical level, specificity required by the "
  "question, uncertainty, or truth conditions, KEEP THE ORIGINAL unchanged. "
  "When possible remove redundant appositive titles, honorifics, or location "
  "qualifiers without changing the requested answer entity. Output a SINGLE "
  "contiguous exact substring copied verbatim from the ORIGINAL ANSWER, "
  "with no commentary, JSON, or explanation. NEVER invent a different "
  "answer, use outside knowledge, or use any expected/gold answer. "
  "If no safe contraction exists, copy the original answer verbatim."
)
USER=(
  "Question:\n{question}\n\nProvided original source evidence:\n{context}"
  "\n\nORIGINAL ANSWER (copy substring only):\n{answer}\n\n"
  "Return the shortest safe answer span preserving the original referent."
)
MAX_OUTPUT_TOKENS=60
MODEL_SEED=20261012
# Only answers with overtly compound surface forms receive an extra remote
# call; all other baseline outputs are reused without additional cost.
def needs_review(original:str)->bool:
    text=str(original or "").strip()
    if not text:return False
    return len(text.split())>=3 or "," in text or " (" in text

def _normalized(x:str)->str:
    return unicodedata.normalize("NFKC",x).strip()

def validated_substring(original:str,model_output:str)->tuple[str,str]:
    """Return (protected prediction, status) without any gold lookup.

    Strict byte-identical contiguous extraction ensures no NEW content.
    Unicode letter/number boundaries prevent fragments of existing words,
    but cannot guarantee semantic equivalent identity.
    """
    answer=str(original or "").strip()
    candidate=str(model_output or "").strip()
    if not candidate:return answer,"empty_rejected"
    if not needs_review(answer):return answer,"not_eligible"
    start=answer.find(candidate)
    if start<0:return answer,"not_exact_original_substring"
    end=start+len(candidate)
    if start>0 and (answer[start-1].isalnum() or answer[start-1]=="_"):
        return answer,"left_word_boundary_fail"
    if end<len(answer) and (answer[end].isalnum() or answer[end]=="_"):
        return answer,"right_word_boundary_fail"
    if len(candidate)<2:return answer,"too_short"
    if candidate==answer:return answer,"unchanged"
    return candidate,"accepted_shorter_original_substring"

def make_contract_payload(question:str,context:str,answer:str,model:str):
    assert needs_review(answer)
    # The endpoint is the PRE-EXISTING Phase13 full-response Qwen27B relay.
    return {
       "model":model,
       "messages":[{"role":"system","content":SYSTEM},
                   {"role":"user","content":USER.format(
                       question=question,context=context,answer=answer)}],
       "temperature":0,"top_k":1,"top_p":1,"min_p":0,
       "seed":MODEL_SEED,"max_tokens":MAX_OUTPUT_TOKENS,
       "reasoning_effort":"none","stream":False}
