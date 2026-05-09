# Resources QA Review Packet

Accept only records where the query is meaningful and the answer is supported by both evidence passages.
Edit awkward queries/answers before accepting. Reject records that are just bibliography noise, broken PDF text, or unsupported.

## Record 1/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about university, stanford, long, and how is it connected to code, llms, while?

Answer: LongCodeZip: Compress Long Context for Code Language Models Yuling Shi1, Yichun Qian 2, Hongyu Zhang 3, Beijun Shen 1, Xiaodong Gu 1∗ 1Shanghai Jiao Tong University, Shanghai, China 2Stanford University, Stanford, CA, USA 3Chongqing University, Chongqing, China {yuling.shi, bjshen, xiaodong.gu}@sjtu.edu.cn, ycqian@stanford.edu, hyzhang@cqu.edu.cn Abstract—Code generation under long contexts is becoming increasingly critical as Large Language Models (LLMs) are required to reason over extensive in While recent advances enable code LLMs to process long inputs, high API costs and generation latency remain substantial bottlenecks.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\2510.00446v1.pdf`

Evidence 1 page 1, paragraph 0:

```text
LongCodeZip: Compress Long Context for Code Language Models Yuling Shi1, Yichun Qian 2, Hongyu Zhang
3, Beijun Shen 1, Xiaodong Gu 1∗ 1Shanghai Jiao Tong University, Shanghai, China 2Stanford
University, Stanford, CA, USA 3Chongqing University, Chongqing, China {yuling.shi, bjshen,
xiaodong.gu}@sjtu.edu.cn, ycqian@stanford.edu, hyzhang@cqu.edu.cn Abstract—Code generation under
long contexts is becoming increasingly critical as Large Language Models (LLMs) are required to
reason over extensive information in the code- base.
```

Evidence 2 page 1, paragraph 1:

```text
While recent advances enable code LLMs to process long inputs, high API costs and generation latency
remain substantial bottlenecks. Existing context pruning techniques, such as LLMLingua, achieve
promising results for general text but overlook code-specific structures and dependencies, leading
to suboptimal performance in programming tasks. In this paper, we propose LongCodeZip, a novel plug-
and-play code compression framework designed specifically for code LLMs.
```

## Record 2/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about lines, line, perplexity, and how is it connected to budget, code, adaptive?

Answer: We treat each line of code as the smallest atomic unit and group consecutive lines based on their perplexity scores, calculated as in (3). This perplexity- guided aggregation allows blocks to capture meaningful code segments while preserving the code structure.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\2510.00446v1.pdf`

Evidence 1 page 4, paragraph 51:

```text
We treat each line of code as the smallest atomic unit and group consecutive lines based on their
perplexity scores, calculated as in (3). When a line’s perplexity exhibits a sharp local increase,
exceeding that of its neighbors by at leastαtimes of the standard deviation over all lines, we mark
it as a block boundary. Such high-perplexity lines typically mark the beginning of a new block,
reflecting underlying semantic or structural changes.
```

Evidence 2 page 4, paragraph 52:

```text
This perplexity- guided aggregation allows blocks to capture meaningful code segments while
preserving the code structure. Adaptive Budget Allocation.Functions selected in the coarse-grained
stage vary in importance. Hence, applying a uniform compression ratio across all of them is
suboptimal. To address this, we introduce an adaptive budget allocation mechanism that distributes
the fine-grained token budget pro- portionally to function importance.
```

## Record 3/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about ratio, compression, longcodezip, and how is it connected to ablation, study, understand?

Answer: TABLE V: Results with Closed-source Models Long Code Completion Long Module Summarization RepoQA Method CLAUDE-3.7-SONNET GPT-4O CLAUDE-3.7-SONNET GPT-4O CLAUDE-3.7-SONNET GPT-4O ES EM Ratio ES EM Ratio CompScore Ratio CompScore RatioAvg Acc Ratio Avg Acc Ratio No Compression 66.24 41.20 1.0x 65.13 40.80 1.0x 60.72 1.0x 58.42 1.0x 89.7 1.0x 87.8 1.0x No Context 43.97 14.20 - 42.92 14.00 - 6.58 - 6.41 - 0.0 - 0.0 - Random Token 47.61 14.00 4.4x 46.51 13.80 4.4x 37.45 1.8x 35.83 1.8x 3.8 3.6x 3.8  B.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\2510.00446v1.pdf`

Evidence 1 page 8, paragraph 98:

```text
TABLE V: Results with Closed-source Models Long Code Completion Long Module Summarization RepoQA
Method CLAUDE-3.7-SONNET GPT-4O CLAUDE-3.7-SONNET GPT-4O CLAUDE-3.7-SONNET GPT-4O ES EM Ratio ES EM
Ratio CompScore Ratio CompScore RatioAvg Acc Ratio Avg Acc Ratio No Compression 66.24 41.20 1.0x
65.13 40.80 1.0x 60.72 1.0x 58.42 1.0x 89.7 1.0x 87.8 1.0x No Context 43.97 14.20 - 42.92 14.00 -
6.58 - 6.41 - 0.0 - 0.0 - Random Token 47.61 14.00 4.4x 46.51 13.80 4.4x 37.45 1.8x 35.83 1.8x 3.8
3.6x 3.8 3.6x Random Line 52.61 22.20 4.5x 51.42 21.80 4.5x 50.12 1.8x 48.24 1.8x 12.2 3.5x 12.1
3.5x RAG (Sliding Window) 61.44 34.00 2.8x 60.03 33.20 2.8x 58.03 1.7x 55.85 1.7x 73.8 3.7x 73.0
3.7x RAG (Function Chunking)63.55 36.80 3.1x 62.01 36.00 3.1x 44.56 2.1x 42.76 2.1x 55.0 4.3x 52.5
4.3x LLMLingua 46.58 15.20 3.4x 45.53 14.80 3.4x 43.21 1.7x 41.57 1.7x 2.8 4.1x 2.7 4.1x LLMLingua-2
49.02 16.20 4.4x 47.90 15.80 4.4x 57.85 2.1x 55.48 2.1x 3.0 4.6x 2.8 4.6x LongLLMLingua 57.58 27.80
3.2x 56.24 27.20 3.2x 50.86 1.5x 48.89 1.5x 74.5 4.8x 73.2 4.8x DietCode 54.00 19.80 3.4x 52.76
19.40 3.4x 38.82 2.1x 37.21 2.1x 26.7 3.7x 25.5 3.7x SlimCode 53.03 20.80 4.5x 51.78 20.40 4.5x
48.11 2.2x 46.13 2.2x 38.3 4.3x 37.0 4.3x LongCodeZip 66.27 40.20 4.3x 64.72 38.80 4.3x 61.47 1.7x
59.04 1.7x 88.9 5.1x 88.9 5.1x TABLE VI: Comparison with Advanced RAG Methods on Long Code
Completion Model Method ES EM Ratio SEED-CODER-8B No Compression64.04 40.20 1.0x A3-CodGen 58.70
33.10 3.8x cAST 57.35 30.90 4.1x RepoGenix 60.28 34.70 3.5x RLCoder 58.14 32.30 4.0x LongCodeZip
63.11 37.40 5.6x CLAUDE-3.7-SONNET No Compression66.24 41.20 1.0x A3-CodGen 60.15 35.80 3.8x cAST
58.92 33.60 4.1x RepoGenix 62.48 37.40 3.5x RLCoder 62.76 37.90 4.0x LongCodeZip 66.27 40.20 4.3x
TABLE VII: Ablation Study Results Configuration ES EM Ratio LongCodeZip 57.55 32.40 4.3x Coarse-
grained Ablations: w/ Similarity-based Ranking 49.66(-7.89)25.20(-7.20)4.3x w/ Random Ranking
39.76(-17.79)11.50(-20.90)4.4x Fine-grained Ablations: w/o Fine-grained Compression
56.10(-1.45)31.20(-1.20)4.2x w/o Adaptive Budget Allocation 55.21(-2.34)29.40(-3.00)4.3x w/ Line
Chunking 55.98(-1.57)31.20(-1.20)4.3x w/ Random Line Selection 55.07(-2.48)29.00(-3.40)4.3x
♂lightbulbFinding 1 LongCodeZip is effective across various downstream tasks, with up to 5.6x
compression ratio without sacri- ficing downstream performance.
```

Evidence 2 page 8, paragraph 99:

```text
B. RQ2: Ablation Study To understand the contribution of each component in Long- CodeZip, we conduct
an ablation study on the Long Code Completion task using Qwen2.5-Coder-7B. For all ablations, the
total token budget and other hyper-parameters are set the same as the full method. We systematically
remove or modify key components to analyze their individual impact.
```

## Record 4/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about source, code, compression, and how is it connected to attention, across, token?

Answer: However, these methods are primarily designed for nat- ural language, and often fail to capture the structural and semantic regularities of source code, leading to suboptimal performance in code-related tasks. DietCode [26] combines static frequency-based filtering with CodeBERT attention heuristics to discard low-impact tokens, but its reliance on model-specific attention reduces adaptability across different architectures.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\2510.00446v1.pdf`

Evidence 1 page 11, paragraph 142:

```text
However, these methods are primarily designed for nat- ural language, and often fail to capture the
structural and semantic regularities of source code, leading to suboptimal performance in code-
related tasks. This has led to a growing body of research focused on code-specific compression.
Short- enDoc [72] targets docstring compression specifically, whereas our method targets source
code, which typically dominates the input in long-context scenarios.
```

Evidence 2 page 11, paragraph 143:

```text
DietCode [26] combines static frequency-based filtering with CodeBERT attention heuristics to
discard low-impact tokens, but its reliance on model-specific attention reduces adaptability across
different architectures. SlimCode[27] applies rule-based token pruning using token types and program
dependency graphs, which may not gener- alize well across languages or tasks. However, these
existing methods mainly focus on compressing single functions for short context tasks.
```

## Record 5/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about wang, arxiv, chen, and how is it connected to alistarh, gptq, accurate?

Answer: Wang, Y . Alistarh, “Gptq: Accurate post-training quantization for generative pre-trained transformers,” in The Eleventh International Conference on Learning Representations.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\2510.00446v1.pdf`

Evidence 1 page 13, paragraph 192:

```text
Wang, Y . Wang, D. Guo, J. Chen, R. Zhang, Y . Ma, and Z. Zheng, “Rlcoder: Reinforcement learning
for repository-level code completion,” arXiv preprint arXiv:2407.19487, 2024. [43] E. Frantar, S.
Ashkboos, T. Hoefler, and D.
```

Evidence 2 page 13, paragraph 193:

```text
Alistarh, “Gptq: Accurate post-training quantization for generative pre-trained transformers,” in
The Eleventh International Conference on Learning Representations. OpenReview, 2023. [44] J. Achiam,
S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J.
```

## Record 6/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about across, comprehensive, evaluation, and how is it connected to gpt-, related, work?

Answer: • Comprehensive evaluation: Results across diverse bench- marks show FABLE consistently achieves a superior balance between completeness and faithfulness, outperforming struc- tured RAG and full-context LLMs across multiple real-world QA and agent tasks. 2 Related Work 2.1 Long-Context Large Language Models Recent years have seen rapid expansion of LLM context windows, from 4K tokens in GPT-3 to 128K in GPT-4 Turbo and beyond.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\2601.18116v1.pdf`

Evidence 1 page 2, paragraph 27:

```text
• Comprehensive evaluation: Results across diverse bench- marks show FABLE consistently achieves a
superior balance between completeness and faithfulness, outperforming struc- tured RAG and full-
context LLMs across multiple real-world QA and agent tasks.
```

Evidence 2 page 2, paragraph 28:

```text
2 Related Work 2.1 Long-Context Large Language Models Recent years have seen rapid expansion of LLM
context windows, from 4K tokens in GPT-3 to 128K in GPT-4 Turbo and beyond. These advances are
enabled by architectural techniques such as efficient 2
```

## Record 7/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about evaluation, datasets, synthetic, and how is it connected to reasoning, documents, structured?

Answer: 4 Experimental Setup 4.1 Evaluation Datasets We evaluate FABLE on diverse datasets covering synthetic reason- ing, multi-hop QA, and agent-based retrieval. Real-World Knowledge Multi-hop QA: We use two subsets from LongBench [3]: HotpotQA [37] requiring reasoning across Wikipedia documents and 2Wiki [14] emphasizing multi-step entity reasoning focusing on logical reasoning over structured and un- structured sources.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\2601.18116v1.pdf`

Evidence 1 page 6, paragraph 71:

```text
4 Experimental Setup 4.1 Evaluation Datasets We evaluate FABLE on diverse datasets covering
synthetic reason- ing, multi-hop QA, and agent-based retrieval. Synthetic Knowledge QA: DragonBall [
40] contains LLM- generated documents and queries following predefined schemas, enabling controlled
evaluation without real-world confounds.
```

Evidence 2 page 6, paragraph 72:

```text
Real-World Knowledge Multi-hop QA: We use two subsets from LongBench [3]: HotpotQA [37] requiring
reasoning across Wikipedia documents and 2Wiki [14] emphasizing multi-step entity reasoning focusing
on logical reasoning over structured and un- structured sources. We adapt these two datasets from a
long-context inference benchmark to a RAG setting by treating the candidate documents as a retrieval
corpus.
```

## Record 8/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about hallucination, irrelevance, control, and how is it connected to retrieval, irrelevance, fable?

Answer: Hallucination and irrelevance control: FABLE(nodes) consistently shows the lowest hallucination rates at 4k+ tokens (6.0% at 4k, 7.0% at 8k), substantially better than fixlength-chunks and llm- chunks. For irrelevance, FABLE(nodes) achieves 4.6% at 4k and 3.9% at 8k, compared to 10.6% and 8.1% for the flat retrieval baselines, validating that fine-grained hierarchical retrieval enables more precise evidence localization.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\2601.18116v1.pdf`

Evidence 1 page 9, paragraph 117:

```text
Hallucination and irrelevance control: FABLE(nodes) consistently shows the lowest hallucination
rates at 4k+ tokens (6.0% at 4k, 7.0% at 8k), substantially better than fixlength-chunks and llm-
chunks.
```

Evidence 2 page 9, paragraph 118:

```text
For irrelevance, FABLE(nodes) achieves 4.6% at 4k and 3.9% at 8k, compared to 10.6% and 8.1% for the
flat retrieval baselines, validating that fine-grained hierarchical retrieval enables more precise
evidence localization.
```

## Record 9/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about arxiv, replug, retrieval-augmented, and how is it connected to arxiv, https, wenyu?

Answer: REPLUG: Retrieval-Augmented Black-Box Language Models.arXiv preprint arXiv:2301.12652(2023). arXiv:2401.15391 [cs.CL] https: //arxiv.org/abs/2401.15391 [33] Wenyu Tao, Xiaofen Xing, Yirong Chen, Linyi Huang, and Xiangmin Xu.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\2601.18116v1.pdf`

Evidence 1 page 10, paragraph 160:

```text
REPLUG: Retrieval-Augmented Black-Box Language Models.arXiv preprint arXiv:2301.12652(2023). [32]
Yixuan Tang and Yi Yang. 2024. MultiHop-RAG: Benchmarking Retrieval- Augmented Generation for Multi-
Hop Queries.
```

Evidence 2 page 10, paragraph 161:

```text
arXiv:2401.15391 [cs.CL] https: //arxiv.org/abs/2401.15391 [33] Wenyu Tao, Xiaofen Xing, Yirong
Chen, Linyi Huang, and Xiangmin Xu. 2025. TreeRAG: Unleashing the Power of Hierarchical Storage for
Enhanced Knowledge Retrieval in Long Documents.
```

## Record 10/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about first, sort, elements, and how is it connected to value, then, initialize?

Answer: We first sort all elements in descending order based on their semantic111 relevance scores and assign each element a rank(e). The value of each element is then computed as112 followed:113 value(e) = max_score − rank(e) (6) This design ensures that elements with higher semantic relevance within the local subgraph re-114 ceive higher value scores, and are therefore prioritized for inclusion in the final subgraph.115 Algorithm 1 Dynamic Programming for 0-1 Knapsack Prob- lem Input: Values v[1..n], Weights w[1..n], Capacity C Output: Selected items maximizing total value within C Initialize A ← array of (n + 1) × (C + 1) w

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\27887_Query_Aware_Subgraph_Pac.pdf`

Evidence 1 page 4, paragraph 40:

```text
We first sort all elements in descending order based on their semantic111 relevance scores and
assign each element a rank(e).
```

Evidence 2 page 4, paragraph 41:

```text
The value of each element is then computed as112 followed:113 value(e) = max_score − rank(e) (6)
This design ensures that elements with higher semantic relevance within the local subgraph re-114
ceive higher value scores, and are therefore prioritized for inclusion in the final subgraph.115
Algorithm 1 Dynamic Programming for 0-1 Knapsack Prob- lem Input: Values v[1..n], Weights w[1..n],
Capacity C Output: Selected items maximizing total value within C Initialize A ← array of (n + 1) ×
(C + 1) with 0 Initialize keep ← boolean array of (n + 1) × (C + 1) with False for i = 1 to n do for
c = 0 to C do if w[i] ≤ c and v[i] +A[i − 1][c − w[i]] > A[i − 1][c] then A[i][c] ← v[i] + A[i −
1][c − w[i]] keep[i][c] ← True else A[i][c] ← A[i − 1][c] Initialize S ← [], c ← C for i = n downto
1 do if keep[i][c] then Append i to S c ← c − w[i] return S 116 Structure-Aware Weight Assign-117
ment In terms of measuring struc-118 tural cost, we adopt a structure-aware119 weighting mechanism
to suppress re-120 dundancy.
```

## Record 11/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about performance, tasks, retrieval, and how is it connected to graphpack, entities, noteworthy?

Answer: Further performance reports on more graph benchmark tasks and knowledge-intensive tasks258 are presented in Appendix D.1.259 4.3 Subgraph Retrieval Strategy (RQ2)260 To verify the effectiveness of GraphPack’s graph-enhanced retrieval strategy, we evaluate its impact on261 LLMs without fine-tuning. It is noteworthy that GraphPack263 achieves a 18.61% increase in F1 Score compared to the baseline model.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\27887_Query_Aware_Subgraph_Pac.pdf`

Evidence 1 page 8, paragraph 88:

```text
Further performance reports on more graph benchmark tasks and knowledge-intensive tasks258 are
presented in Appendix D.1.259 4.3 Subgraph Retrieval Strategy (RQ2)260 To verify the effectiveness
of GraphPack’s graph-enhanced retrieval strategy, we evaluate its impact on261 LLMs without fine-
tuning. Table 4 demonstrates the performance improvements achieved by different262 strategies during
the inference of LLMs without any fine-tuning.
```

Evidence 2 page 8, paragraph 89:

```text
It is noteworthy that GraphPack263 achieves a 18.61% increase in F1 Score compared to the baseline
model. This is particularly important264 in real-world question answering scenarios, as it can
provide users with more correct candidate entities265 to choose from. Furthermore, As shown in Table
3, we analyze the performance of ChatGPT and266 GraphPack when addressing questions involving
multiple entities within labels.
```

## Record 12/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about https, arxiv, ethan, and how is it connected to https, arxiv, alec?

Answer: URL https://arxiv.org/abs/398 2408.08921.399 Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron Courville. URL https://arxiv.org/abs/1709.07871.401 Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,402 Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever.403 Learning transferable visual models from natural language supervision, 2021.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\27887_Query_Aware_Subgraph_Pac.pdf`

Evidence 1 page 11, paragraph 135:

```text
URL https://arxiv.org/abs/398 2408.08921.399 Ethan Perez, Florian Strub, Harm de Vries, Vincent
Dumoulin, and Aaron Courville. Film: Visual rea-400 soning with a general conditioning layer, 2017.
```

Evidence 2 page 11, paragraph 136:

```text
URL https://arxiv.org/abs/1709.07871.401 Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh,
Gabriel Goh, Sandhini Agarwal,402 Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen
Krueger, and Ilya Sutskever.403 Learning transferable visual models from natural language
supervision, 2021.
```

## Record 13/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about impacts, societal, does, and how is it connected to societal, impact, considerations?

Answer: Broader impacts660 Question: Does the paper discuss both potential positive societal impacts and negative661 societal impacts of the work performed?662 Answer: [NA]663 Justification: The paper does no discuss both potential positive and negative societal impacts664 of the proposed method.665 16 Guidelines:666 • The answer NA means that there is no societal impact of the work performed.667 • If the authors answer NA or No, they should explain why their work has no societal668 impact or why the paper does not address societal impact.669 • Examples of negative societal impacts include potential malicious or unintended uses670 (e.g., disinformation, generating fake profiles, surveillance), fairness considerations671 (e.g., deployment of technologies that could make decisions that unfairly 

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\27887_Query_Aware_Subgraph_Pac.pdf`

Evidence 1 page 16, paragraph 185:

```text
Broader impacts660 Question: Does the paper discuss both potential positive societal impacts and
negative661 societal impacts of the work performed?662 Answer: [NA]663 Justification: The paper does
no discuss both potential positive and negative societal impacts664 of the proposed method.665 16
```

Evidence 2 page 17, paragraph 186:

```text
Guidelines:666 • The answer NA means that there is no societal impact of the work performed.667 • If
the authors answer NA or No, they should explain why their work has no societal668 impact or why the
paper does not address societal impact.669 • Examples of negative societal impacts include potential
malicious or unintended uses670 (e.g., disinformation, generating fake profiles, surveillance),
fairness considerations671 (e.g., deployment of technologies that could make decisions that unfairly
impact specific672 groups), privacy considerations, and security considerations.673 • The conference
expects that many papers will be foundational research and not tied674 to particular applications,
let alone deployments.
```

## Record 14/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about section, nsga-ii, analysis, and how is it connected to section, discusses, bibliometric?

Answer: Section 4 addresses the literature survey, including NSGA-II implementation to MOCOPs, performance assessment, performed case stud- ies, statistical analysis, and post-Pareto optimality anal- ysis. Section 6 discusses the bibliometric analysis, and lastly, Section 7 provides the conclusion and future direc- tions drawn from this study.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\a-comprehensive-review-on-nsga-ii-for-multi-objective-combinatorial-optimization-problems.pdf`

Evidence 1 page 2, paragraph 30:

```text
Section 4 addresses the literature survey, including NSGA-II implementation to MOCOPs, performance
assessment, performed case stud- ies, statistical analysis, and post-Pareto optimality anal- ysis.
Section 5 is about the analysis of modiﬁcations in NSGA-II.
```

Evidence 2 page 2, paragraph 31:

```text
Section 6 discusses the bibliometric analysis, and lastly, Section 7 provides the conclusion and
future direc- tions drawn from this study. II. BACKGROUND In this section, we describe the concepts
of MOOP , MOCOP , and Pareto dominance. The basic structure and procedure of NSGA-II are also
discussed. A.
```

## Record 15/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about nsga-ii, problem, conventional, and how is it connected to method, study, operators?

Answer: 1) CONVENTIONAL NSGA-II FOR MOCOPS This section presents a detailed study of the conventional NSGA-II implementation to MOCOPS. In this study, the operators’ personality and decision-making styles, exper- tise in dealing with machines, and job security are also incorporated, which demonstrates the novelty of the pro- posed model.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\a-comprehensive-review-on-nsga-ii-for-multi-objective-combinatorial-optimization-problems.pdf`

Evidence 1 page 7, paragraph 80:

```text
1) CONVENTIONAL NSGA-II FOR MOCOPS This section presents a detailed study of the conventional NSGA-
II implementation to MOCOPS. The summary of the related literature is shown in Table 6. a:
ASSIGNMENT PROBLEM In [49], Azadeh et al. utilized NSGA-II to solve a large-sized cell formation
problem, a traditional problem of assignment of parts, operators, and machines to the cells.
```

Evidence 2 page 7, paragraph 81:

```text
In this study, the operators’ personality and decision-making styles, exper- tise in dealing with
machines, and job security are also incorporated, which demonstrates the novelty of the pro- posed
model. The results were validated using NSGA-II, multi-objective PSO (MOPSO), weighted sum method
(WSM), and epsilon constraint method (ECM).
```

## Record 16/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about nsga-ii, multiple, used, and how is it connected to operator, mutation, zheng?

Answer: The proposed algorithm outperformed the benchmark NSGA-II, MOPSO, SPEA-II and PAES by at least 18 % in optimizing the objective functions of various synthetic and real-world scenarios. Zheng et al.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\a-comprehensive-review-on-nsga-ii-for-multi-objective-combinatorial-optimization-problems.pdf`

Evidence 1 page 11, paragraph 123:

```text
The proposed algorithm outperformed the benchmark NSGA-II, MOPSO, SPEA-II and PAES by at least 18 %
in optimizing the objective functions of various synthetic and real-world scenarios. In [69], NSGA-
II was used to solve the multi-objective RAP in multiple input multiple output orthogonal frequency
division multiple access sys- tems. The SPX and bitwise mutation operators were used as genetic
operators for NSGA-II.
```

Evidence 2 page 11, paragraph 124:

```text
Zheng et al. [71] imple- mented NSGA-II to solve energy-aware RAPs in a cloud manufacturing
environment. The best optimal solution was then achieved using TOPSIS. The TPX operator and muta-
tion operator based on the concepts of simple mutation and uniform mutation were used as genetic
operators.
```

## Record 17/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about nsga-ii, controlled, elitist, and how is it connected to nsga-ii, improved, solution?

Answer: In [188], the controlled elitist NSGA-II improved the multi-objective process planning and scheduling in manu- facturing systems to consider the problem’s computational intractability. Ruiming [192] suggested an improved NSGA-II that improves the solution diversity and convergence for multi-objective dynamic scheduling problem in an integrated energy system.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\a-comprehensive-review-on-nsga-ii-for-multi-objective-combinatorial-optimization-problems.pdf`

Evidence 1 page 14, paragraph 166:

```text
In [188], the controlled elitist NSGA-II improved the multi-objective process planning and
scheduling in manu- facturing systems to consider the problem’s computational intractability. The
proposed algorithm was compared to the controlled elitist NSGA-II and NSGA-II for test cases, and
the results indicated that the proposed algorithm provided more optimal and robust solutions.
```

Evidence 2 page 14, paragraph 167:

```text
Ruiming [192] suggested an improved NSGA-II that improves the solution diversity and convergence for
multi-objective dynamic scheduling problem in an integrated energy system. An interactive strategy
using an external archive to update the solution helped prevent local optimiza- tion. The authors
used the traditional NSGA-II to compare the non-dominated solutions with the proposed algorithm and
found that the improved NSGA-II has a better exploration ability and uniform spread of solutions.
```

## Record 18/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about proposed, ga-mip, nise, and how is it connected to they, presented, modi-?

Answer: The proposed GA-MIP again compares with the non- inferior set estimation (NISE) method on publicly available instances. They presented a modi- ﬁed version of NSGA-II based on mutation operator only and used semi-supervised learning to generate initial populations.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\a-comprehensive-review-on-nsga-ii-for-multi-objective-combinatorial-optimization-problems.pdf`

Evidence 1 page 16, paragraph 209:

```text
The proposed GA-MIP again compares with the non- inferior set estimation (NISE) method on publicly
available instances. The GA-MIP scaled better than the NISE method on large instances and can also
ﬁnd non-supported solutions along with supported solutions. Cococcioni et al. [50] proposed a multi-
objective model for worker’s risk perception and caution to improve workers’ occupational safety at
the workplace.
```

Evidence 2 page 16, paragraph 210:

```text
They presented a modi- ﬁed version of NSGA-II based on mutation operator only and used semi-
supervised learning to generate initial populations. After that, the best Pareto optimal solution
was obtained using the TOPSIS method. Finally, the validation of the proposed methodology was
carried out using data collected from small manufacturing enterprises. Segredo et al.
```

## Record 19/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about instances, proposed, approach, and how is it connected to time, verma, comprehensive?

Answer: The proposed approach was compared with ﬁve state- of-the-art algorithms on JSSP benchmark instances using a set coverage metric. S.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\a-comprehensive-review-on-nsga-ii-for-multi-objective-combinatorial-optimization-problems.pdf`

Evidence 1 page 20, paragraph 254:

```text
The proposed approach was compared with ﬁve state- of-the-art algorithms on JSSP benchmark instances
using a set coverage metric. The authors conclude that the pro- posed method was better than other
methods in most of the test instances. Also, the estimated time obtained through 57776 VOLUME 9,
2021
```

Evidence 2 page 21, paragraph 255:

```text
S. Verma et al.: Comprehensive Review on NSGA-II for Multi-Objective COPS multi-layer perceptron has
better accuracy than the other regression techniques used for time estimations. Liu et al. [157]
studied machine scheduling under dis- ruption to minimize weighted discounted total completion time
and deviation from the initial schedule.
```

## Record 20/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about crossover, coding, representation, and how is it connected to mutation, coding, operators?

Answer: The chromosome representation is selected according to the nature of the problem. S.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\a-comprehensive-review-on-nsga-ii-for-multi-objective-combinatorial-optimization-problems.pdf`

Evidence 1 page 24, paragraph 298:

```text
The chromosome representation is selected according to the nature of the problem. The conventional
NSGA-II crossover and mutation cannot be applied to all types of COPs. Also, the inappropriate
representation may result in poor performance of the algorithm. Therefore, other crossover
operators, such as TPX, UX (binary coding), arith- metic crossover (real coding), PMX, OX (integer
coding), 57780 VOLUME 9, 2021
```

Evidence 2 page 25, paragraph 299:

```text
S. Verma et al.: Comprehensive Review on NSGA-II for Multi-Objective COPS FIGURE 4. Crossover and
mutation operators. TABLE 12. Post-Pareto optimality techniques. and mutation operators such as bit-
wise mutation, bit-ﬂip mutation (binary coding), Gaussian mutation (real coding) and inversion
mutation (integer coding) are used according to the chromosome representation.
```

## Record 21/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about ehrgott, years, multiobjective, and how is it connected to multi-objective, metaheuris-, tics?

Answer: Ehrgott, ‘‘1984-2004–20 years of multiobjective metaheuristics. 33–46, doi: 10.1007/978-3-540-31880-4_3.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\a-comprehensive-review-on-nsga-ii-for-multi-objective-combinatorial-optimization-problems.pdf`

Evidence 1 page 30, paragraph 348:

```text
Ehrgott, ‘‘1984-2004–20 years of multiobjective metaheuristics. But what about the solution of
combinatorial problems with multiple objectives?’’ in Evolutionary Multi-Criterion Optimiza- tion
(Lecture Notes in Computer Science), vol. 3410. Berlin, Germany: Springer-V erlag, 2005, pp.
```

Evidence 2 page 30, paragraph 349:

```text
33–46, doi: 10.1007/978-3-540-31880-4_3. [9] Q. Liu, X. Li, H. Liu, and Z. Guo, ‘‘Multi-objective
metaheuris- tics for discrete optimization problems: A review of the state-of- the-art,’’ Appl. Soft
Comput., vol. 93, Aug.
```

## Record 22/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about ciently, solving, multi-objective, and how is it connected to bandyopadhyay, solving, icting?

Answer: Jie, and X. Bandyopadhyay, ‘‘Solving conﬂicting bi- objective facility location problem by NSGA II evolutionary algorithm,’’ Int.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\a-comprehensive-review-on-nsga-ii-for-multi-objective-combinatorial-optimization-problems.pdf`

Evidence 1 page 31, paragraph 401:

```text
Jie, and X. Bin, ‘‘Efﬁciently solving multi-objective dynamic weapon-target assignment problems by
NSGA-II,’’ in Proc. 34th Chin. Control Conf. (CCC), Jul. 2015, pp. 2556–2561, doi:
10.1109/ChiCC.2015.7260033. [57] J. Zeng, L. Dou, and B.
```

Evidence 2 page 31, paragraph 403:

```text
Bandyopadhyay, ‘‘Solving conﬂicting bi- objective facility location problem by NSGA II evolutionary
algorithm,’’ Int. J. Adv. Manuf. Technol., vol. 51, nos. 1–4, pp. 397–414, Apr. 2010, doi:
10.1007/s00170-010-2622-6. [59] A. L.
```

## Record 23/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about liao, ting, solving, and how is it connected to cipls, open, journal?

Answer: 2020, doi: 10.1007/s12351-018- 0392-3. 107–114, doi: 10.1109/CIPLS.2013.6595207.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\a-comprehensive-review-on-nsga-ii-for-multi-objective-combinatorial-optimization-problems.pdf`

Evidence 1 page 32, paragraph 453:

```text
2020, doi: 10.1007/s12351-018- 0392-3. [105] X.-L. Liao and C.-K. Ting, ‘‘Solving the biobjective
selective pickup and delivery problem with memetic algorithm,’’ in Proc. IEEE Symp. Comput. Intell.
Prod. Logistics Syst. (CIPLS), Apr. 2013, pp.
```

Evidence 2 page 32, paragraph 454:

```text
107–114, doi: 10.1109/CIPLS.2013.6595207. [106] A. Open, A. Journal, Y . Shuai, S. Y unfeng, and Z.
Kai, ‘‘An effec- tive method for solving multiple travelling salesman problem based on NSGA-
II,’’Syst. Sci. Control Eng., vol. 7, no. 2, pp.
```

## Record 24/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about syst, manag, subashini, and how is it connected to bandyopadhyay, bhattacharya, solving?

Answer: Syst. 37, no.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\a-comprehensive-review-on-nsga-ii-for-multi-objective-combinatorial-optimization-problems.pdf`

Evidence 1 page 33, paragraph 508:

```text
Syst. Manag., vol. 26, no. 2, pp. 463–485, Sep. 2018, doi: 10.1007/s10922-017-9425-0. [154] G.
Subashini and M. C. Bhuvaneswari, ‘‘Comparison of multi-objective evolutionary approaches for task
scheduling in distributed comput- ing systems,’’ Sadhana, vol.
```

Evidence 2 page 33, paragraph 509:

```text
37, no. 6, pp. 675–694, Jan. 2012, doi: 10.1007/s12046-012-0102-4. [155] S. Bandyopadhyay and R.
Bhattacharya, ‘‘Solving multi-objective par- allel machine scheduling problem by a modiﬁed NSGA-
II,’’ Appl. Math. Model., vol.
```

## Record 25/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about blank, mostaghim, solving, and how is it connected to wang, coordinated, optimized?

Answer: 2018, doi: 10.1016/j.cie.2018.05.001. Wang, ‘‘Coordinated optimized scheduling of locks and transshipment in inland waterway transporta- tion using binary NSGA-II,’’ Int.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\a-comprehensive-review-on-nsga-ii-for-multi-objective-combinatorial-optimization-problems.pdf`

Evidence 1 page 35, paragraph 565:

```text
2018, doi: 10.1016/j.cie.2018.05.001. [204] J. Blank, K. Deb, and S. Mostaghim, ‘‘Solving the bi-
objective trav- eling thief problem with multi-objective evolutionary algorithms,’’ in Proc. Int.
Conf. Evol. Multi-Criterion Optim. (EMO), vol.
```

Evidence 2 page 35, paragraph 567:

```text
Wang, ‘‘Coordinated optimized scheduling of locks and transshipment in inland waterway transporta-
tion using binary NSGA-II,’’ Int. Trans. Oper . Res., vol. 27, no. 3, pp. 1501–1525, May 2020, doi:
10.1111/itor.12720. [206] R. Khanduzi, M. R.
```

## Record 26/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about optimization, previous, work, and how is it connected to problems, nonetheless, include?

Answer: Previous work has investigated the ability of LLMs to solve NP$hard optimization problems (e.g. Nonetheless, we include OPRO (Yang et al., 2024), one of the leading LLM$based optimizers, as a prompting strategy in the evaluation.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\a-knapsack-by-any-other-name-presentation-impacts-llm-performance-on-np-hard-problems.pdf`

Evidence 1 page 2, paragraph 22:

```text
Previous work has investigated the ability of LLMs to solve NP$hard optimization problems (e.g. Yang
et al. , 2024; Guo et al. , 2024; Wu et al. , 2025). Here we do not aim to further improve LLM$based
optimization as such; our focus is on the impact of problem presentation on LLM performance.
```

Evidence 2 page 2, paragraph 23:

```text
Nonetheless, we include OPRO (Yang et al., 2024), one of the leading LLM$based optimizers, as a
prompting strategy in the evaluation. Finally, there are a number of existing datasets for
evaluating models on NP$hard problems. NPHardEval (Fan et al., 2024 ) looks only at textbook
problems, including the three base prob$ lems we consider here.
```

## Record 27/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about always, greedy, include, and how is it connected to reasoning, strategies, evaluation?

Answer: We include results for specifying the ILPs in the domain$specific LP file format in Appendix E. 4.3 Evaluation We run all non$reasoning models with all prompt$ ing strategies and all reasoning models with zero$ shot and ILP Python strategies for all instances in EHOP.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\a-knapsack-by-any-other-name-presentation-impacts-llm-performance-on-np-hard-problems.pdf`

Evidence 1 page 5, paragraph 70:

```text
We include results for specifying the ILPs in the domain$specific LP file format in Appendix E. if
it fits in the remaining capacity. For Travel ing Salesman , we use the strategy of always moving
to the closest unvisited city. We apply the greedy baselines directly to the original problem
instances. These greedy strategies are linear$time algorithms which always produce valid solutions
but give no guarantee of optimality.
```

Evidence 2 page 5, paragraph 71:

```text
4.3 Evaluation We run all non$reasoning models with all prompt$ ing strategies and all reasoning
models with zero$ shot and ILP Python strategies for all instances in EHOP. We classify the
correctness of the outputs using the following scheme. An incompatible response is syntactically
flawed; it can’t be parsed as a solu$ tion to the problem.
```

## Record 28/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about often, real, problem, and how is it connected to would, interesting, explore?

Answer: At least standard LLMs, such as GPT$4o, seem to often recite when they appear to be reasoning. It would be interesting to explore dialogue systems performing actual collaborative problem$solving with the user.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\a-knapsack-by-any-other-name-presentation-impacts-llm-performance-on-np-hard-problems.pdf`

Evidence 1 page 9, paragraph 117:

```text
At least standard LLMs, such as GPT$4o, seem to often recite when they appear to be reasoning. One
limitation of EHOP as a dataset of real problem$solving tasks is that real users will often not be
able to spell out an instance of an everyday problem in detail, e.g. by assigning a numeric satis$
faction value to every museum in Paris.
```

Evidence 2 page 9, paragraph 118:

```text
It would be interesting to explore dialogue systems performing actual collaborative problem$solving
with the user. The costumes of EHOP could be a good starting point for such work. Acknowledgments.
We gratefully acknowl$ edge fruitful conversations with Peter Clark and the members of the
Computational Linguistics group at Saarland University.
```

## Record 29/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about range, city, quicksum, and how is it connected to objective, function, total?

Answer: Result LLM Response Optimal Solution Optimal 1,2,3,4,1 1, 2, 3, 4 Suboptimal 1,4,3,2,1 1, 4, 2, 3 Erroneous 1, 1, 1, 1, 1 1, 4, 3, 5, 2 Incompatible 1,4,1,2,3,5,1 1, 4, 2, 3, 5 from gurobipy import GRB, Model, quicksum def f(): # Create the model model = Model("Traveling Salesman Problem") # Create helper variables n = 4 # number of cities dist = [[0, 5, 11, 4], [5, 0, 1, 1], [11, 1, 0, 3], [4, 1, 3, 0]] # distance matrix # Add variables x = model.addVars(n, n, vtype=GRB.BINARY, name="x") # x[i, .

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\a-knapsack-by-any-other-name-presentation-impacts-llm-performance-on-np-hard-problems.pdf`

Evidence 1 page 13, paragraph 164:

```text
Result LLM Response Optimal Solution Optimal 1,2,3,4,1 1, 2, 3, 4 Suboptimal 1,4,3,2,1 1, 4, 2, 3
Erroneous 1, 1, 1, 1, 1 1, 4, 3, 5, 2 Incompatible 1,4,1,2,3,5,1 1, 4, 2, 3, 5 from gurobipy import
GRB, Model, quicksum def f(): # Create the model model = Model("Traveling Salesman Problem") #
Create helper variables n = 4 # number of cities dist = [[0, 5, 11, 4], [5, 0, 1, 1], [11, 1, 0, 3],
[4, 1, 3, 0]] # distance matrix # Add variables x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
# x[i, j] = 1 if we travel from city i to city j u = model.addVars(n, vtype=GRB.INTEGER, name="u") #
u[i] = order in which we visit city i # Add constraints model.addConstrs(quicksum(x[i, j] for j in
range(n)) == 1 for i in range(n)) # each city is visited exactly once model.addConstrs(quicksum(x[j,
i] for j in range(n)) == 1 for i in range(n)) # each city is left exactly once model.addConstrs(u[i]
- u[j] + n * x[i, j] <= n - 1 for i in range(n) for j in range(n) if i != j) # subtour elimination
model.addConstrs(x[i, i] == 0 for i in range(n)) # we cannot visit the same city twice
model.addConstr(u[0] == 1) # we start at city 1 # Set objective
model.setObjective(quicksum(dist[i][j] * x[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)
# Optimize/solve the model model.optimize() # Return the optimized model return model This ILP
formulation uses the following variables: .
```

Evidence 2 page 13, paragraph 165:

```text
. . The objective function is the total distance traveled, which is minimized. ILP Failure 1, 2, 3,
4 AttributeError at line 117: Unable to retrieve attribute 'X' Table 5: The following examples are
all generated by Llama for textbook Traveling Salesman with the ILP Python prompting strategy.
```

## Record 30/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about decoration, decorations, point, and how is it connected to decorations, decoration, cost?

Answer: Decoration 4 has a cost of $30 and a point value of 4. I don’t want the decorations to be the focus of the party, so I wan’t to pick the worst ones, but I still need to spend the decorations budget.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\a-knapsack-by-any-other-name-presentation-impacts-llm-performance-on-np-hard-problems.pdf`

Evidence 1 page 15, paragraph 212:

```text
Decoration 4 has a cost of $30 and a point value of 4. I can buy at most one of each decoration.
Which decorations should I purchase to make the total point value as high as possible without going
over my budget of $10? Generate a comma$separated list of the decorations I should buy, where each
decoration is represented by its number. I am planning a party, and I need to buy some decorations.
```

Evidence 2 page 15, paragraph 213:

```text
I don’t want the decorations to be the focus of the party, so I wan’t to pick the worst ones, but I
still need to spend the decorations budget. Each decoration has a cost and a point value I’ve
assigned in terms of its worth as a decoration. Here are the decorations I can buy: Decoration 1 has
a cost of $10 and a point value of 2.
```

## Record 31/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about textbook, inverted, costumed, and how is it connected to table, results, full?

Answer: Problem Variant Small Large GCP Textbook 96 56 GCP Inverted −48.0 −56.0 GCP Costumed −1.3 −16.0 KSP Textbook 96 24 KSP Inverted +0.0 +24.0 KSP Costumed −8.0 +6.7 TSP Textbook 100 36 TSP Inverted +0.0 −36.0 TSP Costumed +0.0 −30.7 Table 13: Optimization accuracies for GPT$4o on EHOP$RANDOM using OPRO. Formatting as in Table 1.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\a-knapsack-by-any-other-name-presentation-impacts-llm-performance-on-np-hard-problems.pdf`

Evidence 1 page 22, paragraph 263:

```text
Problem Variant Small Large GCP Textbook 96 56 GCP Inverted −48.0 −56.0 GCP Costumed −1.3 −16.0 KSP
Textbook 96 24 KSP Inverted +0.0 +24.0 KSP Costumed −8.0 +6.7 TSP Textbook 100 36 TSP Inverted +0.0
−36.0 TSP Costumed +0.0 −30.7 Table 13: Optimization accuracies for GPT$4o on EHOP$RANDOM using
OPRO.
```

Evidence 2 page 22, paragraph 264:

```text
Formatting as in Table 1. E Full Results Table 10, Table 11, and Table 12 present full de$
aggregated results from the experiments on GPT, Llama, and non$thinking Qwen, respectively; see
Appendix G for reasoning model results. The ta$ bles break down results using the result categories
discussed in Section 4.3.
```

## Record 32/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about study, chunking, dense, and how is it connected to retrieval, full, systems?

Answer: Overall, this study reframes chunking as a first-class design dimension in dense retrieval and RAG systems, providing empirical guid- ance for principled segmentation choices in real-world deploy- ments. The full RAG systems combine retrieval with generation, this study isolates the retrieval stage to measure chunking impact without con- founding effects from prompt design, decoding, or answer synthesis.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\a-systematic-investigation-of-document-chunking-strategies-and-embedding-sensitivity.pdf`

Evidence 1 page 2, paragraph 21:

```text
Overall, this study reframes chunking as a first-class design dimension in dense retrieval and RAG
systems, providing empirical guid- ance for principled segmentation choices in real-world deploy-
ments. Material and Methods This study benchmarks a wide range of document chunking strategies to
evaluate their impact on dense retrieval perfor- mance across multiple knowledge domains.
```

Evidence 2 page 2, paragraph 22:

```text
The full RAG systems combine retrieval with generation, this study isolates the retrieval stage to
measure chunking impact without con- founding effects from prompt design, decoding, or answer
synthesis. Retrieval quality remains the critical bottleneck for RAG success, making this controlled
analysis directly relevant. An overview of the complete pipeline is provided in Figure 1.
```

## Record 33/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about relevant, answer, retrieved, and how is it connected to gain, discounted, cumulative?

Answer: Reference Answer: {answer} Retrieved Chunk: {chunk_text} Assign a relevance score: 0 = Not relevant 1 = Partially relevant 2 = Fully relevant Respond with JSON only: { "score": 0 | 1 | 2, "reason": "short explanation" } Processing For each query q under configuration (m,d,s) comprising embedding model m, domain d, and chunking strategy s we evaluated the top K=5 retrieved chunks ordered by rank. We report Normalised Discounted Cumulative Gain at rank 5 (nDCG@5) as the primary effectiveness metric, as it jointly captures ranking order and graded relevance.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\a-systematic-investigation-of-document-chunking-strategies-and-embedding-sensitivity.pdf`

Evidence 1 page 6, paragraph 66:

```text
Reference Answer: {answer} Retrieved Chunk: {chunk_text} Assign a relevance score: 0 = Not relevant
1 = Partially relevant 2 = Fully relevant Respond with JSON only: { "score": 0 | 1 | 2, "reason":
"short explanation" } Processing For each query q under configuration (m,d,s) comprising embedding
model m, domain d, and chunking strategy s we evaluated the top K=5 retrieved chunks ordered by
rank.
```

Evidence 2 page 6, paragraph 67:

```text
We report Normalised Discounted Cumulative Gain at rank 5 (nDCG@5) as the primary effectiveness
metric, as it jointly captures ranking order and graded relevance. Letgi ∈ {0,1,2} denote the gain
at rank i. The Discounted Cumulative Gain is: 6/15
```

## Record 34/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about chunking, retrieval, while, and how is it connected to chunks, indexing, different?

Answer: While most prior evaluations of chunking strategies focus on aggregate retrieval metrics, this analysis highlights mean- ingful differences in robustness across queries, suggesting that segmentation choices also influence the reliability of retrieval outcomes for diverse information needs. Different meth- ods produce widely varying numbers of chunks per document, directly affecting indexing overhead and retrieval efficiency.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\a-systematic-investigation-of-document-chunking-strategies-and-embedding-sensitivity.pdf`

Evidence 1 page 9, paragraph 109:

```text
While most prior evaluations of chunking strategies focus on aggregate retrieval metrics, this
analysis highlights mean- ingful differences in robustness across queries, suggesting that
segmentation choices also influence the reliability of retrieval outcomes for diverse information
needs. Effectiveness–Efficiency Trade-offs The choice of chunking strategy has significant
implications for index size, storage cost, and query latency.
```

Evidence 2 page 9, paragraph 110:

```text
Different meth- ods produce widely varying numbers of chunks per document, directly affecting
indexing overhead and retrieval efficiency. While methods that generate many small chunks can
achieve high recall, they often incur substantial indexing and latency costs. Conversely, grouping
content into very large chunks reduces index size but risks diluting relevance signals. 9/15
```

## Record 35/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about arxiv, rudat, spiekermann, and how is it connected to retrieval, arxiv, european?

Answer: R., Rudat, M., Spiekermann, J. In European Conference on Information Retrieval, 345–352 (Springer).

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\a-systematic-investigation-of-document-chunking-strategies-and-embedding-sensitivity.pdf`

Evidence 1 page 14, paragraph 153:

```text
R., Rudat, M., Spiekermann, J. & Flores-Herr, N. Rethinking chunk size for long-document retrieval:
A multi-dataset analysis.arXiv preprint arXiv:2505.21700 (2025). 11. Liu, Z., Simon, C.-E. &
Caspani, F. Passage segmenta- tion of documents for extractive question answering.
```

Evidence 2 page 14, paragraph 154:

```text
In European Conference on Information Retrieval, 345–352 (Springer). 12. Lee, K., Chang, M.-W. &
Toutanova, K. Latent retrieval for weakly supervised open domain question answering. arXiv preprint
arXiv:1906.00300(2019). 13. Guu, K., Lee, K., Tung, Z., Pasupat, P.
```

## Record 36/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about prompts, soft, information, and how is it connected to prompts, section, endeavor?

Answer: Our study builds upon these foundational principles, proposing a unified framework that capitalizes on the strengths of soft prompts and advanced summarization methodologies to alle- viate the constraints of existing LLMs in efficiently handling extensive textual information. In this section, our endeavor lies in the construction of a holistic mathematical framework, embodying the amalgamation of summary vectors with prompts formatted in natural language, the convergence of utility retention with information conden- sation, and the notion of soft prompts.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\adapting-llms-for-efficient-context-processing-through-soft-prompt-compression.pdf`

Evidence 1 page 2, paragraph 22:

```text
Our study builds upon these foundational principles, proposing a unified framework that capitalizes
on the strengths of soft prompts and advanced summarization methodologies to alle- viate the
constraints of existing LLMs in efficiently handling extensive textual information. III. M
ETHODOLOGY Integrating summary vectors with natural language format- ted prompts, the fusion of
utility preservation with information compression, and the conceptualization of soft prompts are
explored within a comprehensive mathematical model.
```

Evidence 2 page 2, paragraph 23:

```text
In this section, our endeavor lies in the construction of a holistic mathematical framework,
embodying the amalgamation of summary vectors with prompts formatted in natural language, the
convergence of utility retention with information conden- sation, and the notion of soft prompts.
```

## Record 37/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about varying, resource, current, and how is it connected to your, simple, queries?

Answer: While larger models offer superior perfor- mance, their high costs makes their universal de- ployment impractical. For simple queries like “What are your business hours?", a smaller, cost- effective model might suffice.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\adaptive-llm-routing-under-budget-constraints.pdf`

Evidence 1 page 1, paragraph 4:

```text
While larger models offer superior perfor- mance, their high costs makes their universal de-
ployment impractical. This challenge is particu- larly acute given the varying pricing structures of
proprietary models and the resource requirements of deploying the open-source alternatives. *Equal
contribution †Current affiliation: Microsoft Research ‡Current affiliation: Microsoft To understand
the need for varying resource re- quirements, consider a customer service chatbot handling diverse
queries.
```

Evidence 2 page 1, paragraph 5:

```text
For simple queries like “What are your business hours?", a smaller, cost- effective model might
suffice. However, for com- plex inquiries, such as “ I’m torn between two of your smartphone models:
the X200 and the Z300. I need a phone with excellent battery life, a high- quality camera, and
robust performance for mul- titasking.
```

## Record 38/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about pretraining, balance, exploration, and how is it connected to time, formulation, posterior?

Answer: To balance exploration, particularly if online queries differ from pretraining data, we set λa as the inverse of arm a’s accuracy during the pretraining phase. With this formulation, the posterior distribution of arm (LLM) embeddings at time t becomes: p(ˆθt a|Dt) = N (˜θt a, (At a)−1) , where Dt repre- sents the observed online data up to time t.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\adaptive-llm-routing-under-budget-constraints.pdf`

Evidence 1 page 4, paragraph 53:

```text
To balance exploration, particularly if online queries differ from pretraining data, we set λa as
the inverse of arm a’s accuracy during the pretraining phase.
```

Evidence 2 page 4, paragraph 54:

```text
With this formulation, the posterior distribution of arm (LLM) embeddings at time t becomes: p(ˆθt
a|Dt) = N (˜θt a, (At a)−1) , where Dt repre- sents the observed online data up to time t.
```

## Record 39/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about pilot, analyses, exhibits, and how is it connected to pilot, qualitative, routing?

Answer: PILOT also exhibits highest performance on deployment set across various learning bucket sizes, showing its efficacy with limited data. 5.1 Qualitative Analysis of PILOT’s Routing Qualitative examination of PILOT’s routing reveals intelligent decision-making.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\adaptive-llm-routing-under-budget-constraints.pdf`

Evidence 1 page 7, paragraph 100:

```text
PILOT also exhibits highest performance on deployment set across various learning bucket sizes,
showing its efficacy with limited data. 5 Discussion and Ablations This section provides qualitative
and quantitative analyses of PILOT’s routing behavior, computa- tional efficiency, cost policy, and
sensitivity, offer- ing a holistic view. These analyses, summarized under ‘Analysis’ in Table 1,
delve deeper into PI- LOT’s operational characteristics.
```

Evidence 2 page 7, paragraph 101:

```text
5.1 Qualitative Analysis of PILOT’s Routing Qualitative examination of PILOT’s routing reveals
intelligent decision-making. For demanding tasks like MMLU and ARC Challenge, PILOT routes 90% and
89.4% of queries to GPT-4, respectively, leveraging GPT-4’s strength in complex reasoning.
```

## Record 40/40

Decision: [ ] Accept  [ ] Edit  [ ] Reject

Query: What does the document say about include, multi-llm, synthesis, and how is it connected to reward, meta-routing, estimation?

Answer: These include: (i) Multi-LLM Synthesis: LLM-Blender (Lu et al., 2024) and related methods (Jiang et al., 2023) invoke several models and fuse their responses. (ii) Meta-routing via reward estimation: Tensor- Opera Router (Stripelis et al., 2024) builds a sepa- rate reward model to guide routing decisions over multiple LLMs.

Source: `C:\Users\metehan\Desktop\llm-context-paper\resources\adaptive-llm-routing-under-budget-constraints.pdf`

Evidence 1 page 11, paragraph 148:

```text
These include: (i) Multi-LLM Synthesis: LLM-Blender (Lu et al., 2024) and related methods (Jiang et
al., 2023) invoke several models and fuse their responses. While improving output quality, these
approaches are cost-intensive and unsuitable for latency- sensitive applications.
```

Evidence 2 page 11, paragraph 149:

```text
(ii) Meta-routing via reward estimation: Tensor- Opera Router (Stripelis et al., 2024) builds a
sepa- rate reward model to guide routing decisions over multiple LLMs. However, it relies on offline
data and full supervision to train the reward predictor. Unlike these works, we focus on selecting a
single model per query and improve performance through online learning from partial feedback.
```
