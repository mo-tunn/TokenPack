# LongBench v2 Error Analysis

## Profile Summary

| Profile | Method | Runs | Accuracy | Avg ctx toks | Saving |
|---|---|---:|---:|---:|---:|
| evidence-hybrid | full-context | 30 | 0.400 | 17623 | 0.000 |
| evidence-hybrid | tokenpack-50 | 30 | 0.400 | 8750 | 0.504 |
| evidence-hybrid | only-longllmlingua-rate050 | 30 | 0.333 | 8564 | 0.512 |
| evidence-hybrid | tokenpack-50+longllmlingua-rate050 | 30 | 0.300 | 4454 | 0.746 |
| query-support | full-context | 30 | 0.400 | 17623 | 0.000 |
| query-support | tokenpack-50 | 30 | 0.367 | 8778 | 0.502 |
| query-support | only-longllmlingua-rate050 | 30 | 0.333 | 8564 | 0.512 |
| query-support | tokenpack-50+longllmlingua-rate050 | 30 | 0.333 | 4483 | 0.745 |
| decision-aware | full-context | 30 | 0.400 | 17623 | 0.000 |
| decision-aware | tokenpack-50 | 30 | 0.367 | 8774 | 0.502 |
| decision-aware | only-longllmlingua-rate050 | 30 | 0.333 | 8564 | 0.512 |
| decision-aware | tokenpack-50+longllmlingua-rate050 | 30 | 0.333 | 4483 | 0.745 |
| coverage | full-context | 30 | 0.400 | 17623 | 0.000 |
| coverage | tokenpack-50 | 30 | 0.367 | 8631 | 0.512 |
| coverage | only-longllmlingua-rate050 | 30 | 0.333 | 8564 | 0.512 |
| coverage | tokenpack-50+longllmlingua-rate050 | 30 | 0.267 | 4414 | 0.749 |

## Evidence-Hybrid Outcome Counts

| Outcome | Count |
|---|---:|
| all_wrong | 15 |
| all_correct | 6 |
| full+tokenpack | 3 |
| full+llmlingua | 2 |
| tokenpack+llmlingua+cascade | 1 |
| full+llmlingua+cascade | 1 |
| tokenpack+cascade | 1 |
| tokenpack | 1 |

## TokenPack Diagnosis Counts

| Diagnosis | Count |
|---|---:|
| all_wrong_or_gold_not_surface_form | 15 |
| tokenpack_correct | 6 |
| tokenpack_loss_vs_full_generation_or_paraphrase | 3 |
| tokenpack_correct_llmlingua_wrong | 3 |
| tokenpack_correct_full_wrong_possible_distractor_removal | 3 |

## Ablation Swings vs Evidence-Hybrid TokenPack

| Profile | Swing | Count |
|---|---|---:|
| coverage | broken_by_ablation | 2 |
| coverage | fixed_by_ablation | 1 |
| coverage | unchanged | 27 |
| decision-aware | broken_by_ablation | 2 |
| decision-aware | fixed_by_ablation | 1 |
| decision-aware | unchanged | 27 |
| query-support | broken_by_ablation | 2 |
| query-support | fixed_by_ablation | 1 |
| query-support | unchanged | 27 |

## Important Case Lists

### Evidence-Hybrid TokenPack Wins over LongLLMLingua

| Case | Gold | TokenPack | LongLLMLingua | Question |
|---|---|---|---|---|
| 66ec1e3f821e116aacb1ae7d | D | D | B | Which of the following statements is incorrect? |
| 66ed1556821e116aacb1ea14 | A | A | C | In terms of data classification, which types of data are introduced in the article and the method FEDHSSL mentioned in the text uses which parts of the data during the pre-train... |
| 66ed910a821e116aacb2033b | A | A | C | Which following option is wrong, according to the topic "disaster " in the text? |
| 66f9625fbb02136c067c5456 | B | B | A | If the original audio has low clarity, after completing the audio cutting,what should I do? |
| 6719bc01bb02136c067d43fa | D | D | C | Which player wins the most golds in the game? |

### LongLLMLingua Wins over Evidence-Hybrid TokenPack

| Case | Gold | TokenPack | LongLLMLingua | Diagnosis | Question |
|---|---|---|---|---|---|
| 66eae4de5a08c7b9b35dd12d | C | B | C | tokenpack_loss_vs_full_generation_or_paraphrase | Which kind of ability is not mentioned in the essay? |
| 66ec3d1d821e116aacb1c622 | B | D | B | tokenpack_loss_vs_full_generation_or_paraphrase | Which of the following is correct？ |
| 66f3918f821e116aacb2d8b7 | D | A | D | tokenpack_loss_vs_full_generation_or_paraphrase | In Calyx & Corolla's customer base, which demographic is underestimated and could lead to greater success for the company in the future? |

### Ablations That Broke Evidence-Hybrid TokenPack

| Profile | Case | Gold | Evidence-Hybrid | Ablation | Question |
|---|---|---|---|---|---|
| query-support | 66ed910a821e116aacb2033b | A | A | C | Which following option is wrong, according to the topic "disaster " in the text? |
| query-support | 66f9625fbb02136c067c5456 | B | B | A | If the original audio has low clarity, after completing the audio cutting,what should I do? |
| decision-aware | 66ed910a821e116aacb2033b | A | A | C | Which following option is wrong, according to the topic "disaster " in the text? |
| decision-aware | 66f9625fbb02136c067c5456 | B | B | A | If the original audio has low clarity, after completing the audio cutting,what should I do? |
| coverage | 66ec356a821e116aacb1c22b | C | C | A | What do these two cases have in common? |
| coverage | 66f9625fbb02136c067c5456 | B | B | A | If the original audio has low clarity, after completing the audio cutting,what should I do? |

## Interpretation

- The main evidence-hybrid TokenPack setting remains the strongest TokenPack-50 profile in this 30-case window.
- Query-support and decision-aware fix a small number of cases, but they also break cases that evidence-hybrid gets right.
- Coverage slightly increases saving but hurts both TokenPack-50 and the cascade, so it should stay experimental.
- The next useful improvement is likely not another lexical heuristic; it is either a stronger learned reranker/value model or a second-stage compressor/reranker after TokenPack selection.

## Actionable Takeaways

- The ablation space is small: 15/30 cases are missed by every method, so the current 30-case pilot has limited room for simple selector tweaks to show large gains.
- Evidence-hybrid is more stable than the newer heuristic profiles: each of query-support, decision-aware, and coverage fixes 1 TokenPack case but breaks 2.
- TokenPack has real wins over LongLLMLingua: 5 cases where TokenPack-50 is correct and LongLLMLingua is wrong.
- LongLLMLingua has fewer wins over TokenPack: 3 cases, all currently tagged as generation/paraphrase or evidence-use losses rather than obvious token-budget failures.
- The next useful experiment should inspect selected context snippets for the 3 LongLLMLingua wins and 2 broken-by-ablation cases, then decide whether the fix should be a learned reranker, context ordering change, or cascade-first framing.
