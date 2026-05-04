# Gold Evidence Review Packet

Use this file to check whether each candidate query and answer is supported by the listed RAG evidence chunk.

Review decision guide:

- Accept if the answer is directly supported by the evidence text.
- Edit if the evidence is useful but the query or answer is awkward.
- Reject if the query is meaningless or the evidence does not support the answer.

```text
========================================================================================
Record 1/8
----------------------------------------------------------------------------------------
QUERY: What is TokenPack's main goal?

ANSWER: TokenPack's goal is to select the most useful semantic chunks without exceeding a limited language-model context budget.

RAG EVIDENCE DATA USED BY THIS RECORD:
- chunk_id: chunk-bc5c1d1ea66d
  source_path: submission\gold\simple_corpus\tokenpack_demo_context.pdf
  pages: 1 - 1
  token_count: 180
  text:
  TokenPack Demo Knowledge Base Prepared for Retrieval and Knapsack Testing 1 Project Overview
  TokenPack is a prototype retrieval system that treats a language model context window as a
  limited token budget. A long document is divided into semantic chunks. Each chunk receives a
  relevance value according to how similar it is to the user query. Each chunk also has a token
  weight. The goal is to select the most useful chunks without exceeding the context budget. The
  main idea is based on the 0/1 Knapsack Problem. In this mapping, every chunk is an item, the
  token count is the item weight, the semantic relevance score is the item value, and the
  available context window is the knapsack capacity. This formulation is useful because language
  models cannot always read every available document at once. TokenPack is designed as a Python
  package and command-line tool. It can ingest text or PDF les, create

```

Decision: [ ] Accept  [ ] Edit  [ ] Reject

```text
========================================================================================
Record 2/8
----------------------------------------------------------------------------------------
QUERY: How does TokenPack map context selection to the 0/1 Knapsack Problem?

ANSWER: Each chunk is an item, token count is the weight, semantic relevance is the value, and the available context window is the capacity.

RAG EVIDENCE DATA USED BY THIS RECORD:
- chunk_id: chunk-bc5c1d1ea66d
  source_path: submission\gold\simple_corpus\tokenpack_demo_context.pdf
  pages: 1 - 1
  token_count: 180
  text:
  TokenPack Demo Knowledge Base Prepared for Retrieval and Knapsack Testing 1 Project Overview
  TokenPack is a prototype retrieval system that treats a language model context window as a
  limited token budget. A long document is divided into semantic chunks. Each chunk receives a
  relevance value according to how similar it is to the user query. Each chunk also has a token
  weight. The goal is to select the most useful chunks without exceeding the context budget. The
  main idea is based on the 0/1 Knapsack Problem. In this mapping, every chunk is an item, the
  token count is the item weight, the semantic relevance score is the item value, and the
  available context window is the knapsack capacity. This formulation is useful because language
  models cannot always read every available document at once. TokenPack is designed as a Python
  package and command-line tool. It can ingest text or PDF les, create

```

Decision: [ ] Accept  [ ] Edit  [ ] Reject

```text
========================================================================================
Record 3/8
----------------------------------------------------------------------------------------
QUERY: Why can top-k retrieval be unsafe under a token budget?

ANSWER: Top-k can select the highest ranked chunks without checking their total token cost, so it may exceed the effective context budget.

RAG EVIDENCE DATA USED BY THIS RECORD:
- chunk_id: chunk-aafe696ecb9e
  source_path: submission\gold\simple_corpus\tokenpack_demo_context.pdf
  pages: 1 - 1
  token_count: 180
  text:
  chunks, compute embeddings, compare retrieval strategies, export selected context, and
  optionally send the selected context to a local language model. 2 Context Window Budget A
  context window is the maximum amount of text that a language model can receive in a single
  request. TokenPack separates the total context window into two parts. One part is reserved for
  retrieved chunks, and the other part is reserved for the model's answer. For example, if the
  total budget is 50,000 tokens and 4,000 tokens are reserved for the answer, then 46,000 tokens
  remain for selected chunks. Budget awareness is important because a retrieval method may nd
  relevant chunks but still fail if the selected text is too long. A top-k retriever can select
  the highest ranked chunks without checking their total token cost. In contrast, a budget-aware
  method must ensure that the nal selected context never exceeds the eective budget.

```

Decision: [ ] Accept  [ ] Edit  [ ] Reject

```text
========================================================================================
Record 4/8
----------------------------------------------------------------------------------------
QUERY: What does paragraph-group chunking do?

ANSWER: Paragraph-group chunking groups neighboring paragraphs until a target token range is reached, preserving local context instead of cutting at a fixed word count.

RAG EVIDENCE DATA USED BY THIS RECORD:
- chunk_id: chunk-8cfe29557457
  source_path: submission\gold\simple_corpus\tokenpack_demo_context.pdf
  pages: 1 - 1
  token_count: 158
  text:
  In TokenPack, over-budget selection is treated as invalid for production use. A method can be
  fast and relevant, but if it violates the token limit, it cannot be safely used as the nal
  context packer. 3 Semantic Chunking Semantic chunking is the process of dividing a document into
  meaningful pieces. TokenPack supports paragraph-group chunking and semantic-threshold chunking.
  Paragraph-group chunking groups neighboring paragraphs until a target token range is reached.
  This approach keeps local context together and avoids cutting every xed number of words.
  Semantic-threshold chunking compares neighboring text blocks using embedding similarity. If the
  similarity between two neighboring blocks drops below a threshold, TokenPack treats that
  location as a topic change and starts a new chunk. This helps keep unrelated topics in separate
  chunks. 1

```

Decision: [ ] Accept  [ ] Edit  [ ] Reject

```text
========================================================================================
Record 5/8
----------------------------------------------------------------------------------------
QUERY: What does semantic-threshold chunking do?

ANSWER: Semantic-threshold chunking compares neighboring text blocks with embedding similarity and starts a new chunk when similarity drops below a threshold.

RAG EVIDENCE DATA USED BY THIS RECORD:
- chunk_id: chunk-8cfe29557457
  source_path: submission\gold\simple_corpus\tokenpack_demo_context.pdf
  pages: 1 - 1
  token_count: 158
  text:
  In TokenPack, over-budget selection is treated as invalid for production use. A method can be
  fast and relevant, but if it violates the token limit, it cannot be safely used as the nal
  context packer. 3 Semantic Chunking Semantic chunking is the process of dividing a document into
  meaningful pieces. TokenPack supports paragraph-group chunking and semantic-threshold chunking.
  Paragraph-group chunking groups neighboring paragraphs until a target token range is reached.
  This approach keeps local context together and avoids cutting every xed number of words.
  Semantic-threshold chunking compares neighboring text blocks using embedding similarity. If the
  similarity between two neighboring blocks drops below a threshold, TokenPack treats that
  location as a topic change and starts a new chunk. This helps keep unrelated topics in separate
  chunks. 1

```

Decision: [ ] Accept  [ ] Edit  [ ] Reject

```text
========================================================================================
Record 6/8
----------------------------------------------------------------------------------------
QUERY: Which retrieval strategies does TokenPack compare?

ANSWER: TokenPack compares top-k, budget-top-k, MMR, knapsack, greedy strategies, and simulated annealing.

RAG EVIDENCE DATA USED BY THIS RECORD:
- chunk_id: chunk-15ce9fca2da4
  source_path: submission\gold\simple_corpus\tokenpack_demo_context.pdf
  pages: 2 - 2
  token_count: 180
  text:
  Chunking quality aects retrieval quality. If a chunk is too small, it may lose the surrounding
  explanation. If a chunk is too large, it may waste the context budget. TokenPack therefore
  stores chunk metadata such as source le, page number, paragraph order, and token count so that
  selected chunks can later be restored to their original order. 4 Retrieval Strategies TokenPack
  compares several retrieval and selection strategies. Top-k retrieval selects a xed num- ber of
  chunks by similarity score. Budget-top-k follows the similarity ranking but skips chunks that
  would exceed the remaining budget. MMR, or Maximal Marginal Relevance, also considers redundancy
  and tries to avoid selecting chunks that are too similar to each other. The knapsack strategy
  solves a 0/1 optimization problem. It tries to maximize total relevance value while keeping the
  selected token count under the budget. Dynamic programming can

- chunk_id: chunk-133877051187
  source_path: submission\gold\simple_corpus\tokenpack_demo_context.pdf
  pages: 2 - 2
  token_count: 180
  text:
  compute the exact optimal solution when the number of candidate chunks and the budget are
  manageable. Greedy strategies are faster but not always optimal. Greedy by value selects high-
  value chunks rst. Greedy by value density selects chunks with high value per token. Simulated
  annealing is a metaheuristic that explores neighboring selections and can sometimes improve over
  simple greedy choices, although it usually takes more time. 5 Evaluation Procedure TokenPack
  evaluation compares strategies under the same budget. Important metrics include evi- dence
  recall, evidence precision, selected token count, budget utilization, value density, redundancy
  score, and latency. Evidence recall measures whether the selected context includes the known
  evi- dence chunks needed to answer a query. For algorithm analysis, TokenPack also runs repeated
  synthetic knapsack experiments. The same randomly generated instances are solved by exact
  dynamic programming, greedy methods, simulated annealing, and random

```

Decision: [ ] Accept  [ ] Edit  [ ] Reject

```text
========================================================================================
Record 7/8
----------------------------------------------------------------------------------------
QUERY: Which metrics are used in TokenPack evaluation?

ANSWER: TokenPack uses evidence recall, evidence precision, selected token count, budget utilization, value density, redundancy score, and latency.

RAG EVIDENCE DATA USED BY THIS RECORD:
- chunk_id: chunk-133877051187
  source_path: submission\gold\simple_corpus\tokenpack_demo_context.pdf
  pages: 2 - 2
  token_count: 180
  text:
  compute the exact optimal solution when the number of candidate chunks and the budget are
  manageable. Greedy strategies are faster but not always optimal. Greedy by value selects high-
  value chunks rst. Greedy by value density selects chunks with high value per token. Simulated
  annealing is a metaheuristic that explores neighboring selections and can sometimes improve over
  simple greedy choices, although it usually takes more time. 5 Evaluation Procedure TokenPack
  evaluation compares strategies under the same budget. Important metrics include evi- dence
  recall, evidence precision, selected token count, budget utilization, value density, redundancy
  score, and latency. Evidence recall measures whether the selected context includes the known
  evi- dence chunks needed to answer a query. For algorithm analysis, TokenPack also runs repeated
  synthetic knapsack experiments. The same randomly generated instances are solved by exact
  dynamic programming, greedy methods, simulated annealing, and random

```

Decision: [ ] Accept  [ ] Edit  [ ] Reject

```text
========================================================================================
Record 8/8
----------------------------------------------------------------------------------------
QUERY: Why can TokenPack run without an OpenAI API key?

ANSWER: The retrieval benchmark does not require a language model, and optional local generation can use an Ollama model installed on the same computer.

RAG EVIDENCE DATA USED BY THIS RECORD:
- chunk_id: chunk-836fe30d9184
  source_path: submission\gold\simple_corpus\tokenpack_demo_context.pdf
  pages: 3 - 3
  token_count: 180
  text:
  API key. These facts are intentionally direct. A reviewer should be able to open the review
  packet, compare each answer with the quoted evidence text, and decide whether the gold record is
  valid in less than a minute per question. 8 Local Language Model Usage TokenPack can run without
  an external API key. For local generation, it can use an Ollama model installed on the same
  computer. The retrieval benchmark itself does not require a language model, because it evaluates
  whether the correct evidence chunks are selected under the token budget. Local generation is
  useful as a demo layer. After TokenPack selects chunks, the selected context can be exported and
  given to a local model with a question. The model should answer only from the provided context.
  This separation is intentional: retrieval quality can be measured o ine, while answer generation
  can be tested later with dierent local models. For the rst project version,

```

Decision: [ ] Accept  [ ] Edit  [ ] Reject
