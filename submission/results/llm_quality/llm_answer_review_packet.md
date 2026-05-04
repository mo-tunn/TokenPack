# LLM Answer Quality Review Packet

Score each completed answer: 0 = wrong, 1 = partially correct, 2 = correct and grounded.

## Record 1

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `document-prefix`
- Status: `completed`
- Context tokens: `698`
- Evidence recall: `1.00`
- Query: What is TokenPack's main goal?
- Gold answer: TokenPack's goal is to select the most useful semantic chunks without exceeding a limited language-model context budget.

Answer:

```text
The main goal of TokenPack is to select the most useful chunks from a document without exceeding the context budget, based on semantic relevance and token weight, using a 0/1 Knapsack Problem approach.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 2

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `full-document`
- Status: `invalid-over-budget`
- Context tokens: `1539`
- Evidence recall: `1.00`
- Query: What is TokenPack's main goal?
- Gold answer: TokenPack's goal is to select the most useful semantic chunks without exceeding a limited language-model context budget.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 3

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `top-k`
- Status: `completed`
- Context tokens: `660`
- Evidence recall: `1.00`
- Query: What is TokenPack's main goal?
- Gold answer: TokenPack's goal is to select the most useful semantic chunks without exceeding a limited language-model context budget.

Answer:

```text
TokenPack's main goal is to select the most useful chunks from a document, respecting the context budget by optimizing both semantic relevance and token cost.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 4

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `knapsack`
- Status: `completed`
- Context tokens: `660`
- Evidence recall: `1.00`
- Query: What is TokenPack's main goal?
- Gold answer: TokenPack's goal is to select the most useful semantic chunks without exceeding a limited language-model context budget.

Answer:

```text
TokenPack's main goal is to select the most useful chunks from a document, considering both semantic relevance and token budget, using a simplified approach similar to the 0/1 Knapsack Problem.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 5

- Model: `Medium local` / `qwen3:4b`
- Strategy: `document-prefix`
- Status: `completed`
- Context tokens: `698`
- Evidence recall: `1.00`
- Query: What is TokenPack's main goal?
- Gold answer: TokenPack's goal is to select the most useful semantic chunks without exceeding a limited language-model context budget.

Answer:

```text
Hmm, the user wants me to answer a question about TokenPack based solely on the provided context. They've given me four chunks of text from a PDF about TokenPack, and I need to find what TokenPack's main goal is.

Let me carefully read through each chunk to find the answer. 

In Chunk 1, it says: "The goal is to select the most useful chunks without exceeding the context budget." That seems like a direct statement of the main goal. 

I should check if other chunks provide additional context. Chunk 2 talks about context window budget but doesn't restate the main goal. Chunk 3 discusses semantic chunking and its importance but again doesn't state the primary goal. Chunk 4 covers retrieval strategies and chunking quality but doesn't restate the main goal.

The question is very specific: "What is TokenPack's main goal?" And Chunk
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 6

- Model: `Medium local` / `qwen3:4b`
- Strategy: `full-document`
- Status: `invalid-over-budget`
- Context tokens: `1539`
- Evidence recall: `1.00`
- Query: What is TokenPack's main goal?
- Gold answer: TokenPack's goal is to select the most useful semantic chunks without exceeding a limited language-model context budget.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 7

- Model: `Medium local` / `qwen3:4b`
- Strategy: `top-k`
- Status: `completed`
- Context tokens: `660`
- Evidence recall: `1.00`
- Query: What is TokenPack's main goal?
- Gold answer: TokenPack's goal is to select the most useful semantic chunks without exceeding a limited language-model context budget.

Answer:

```text
Hmm, the user wants me to answer a question about TokenPack based solely on the provided context. They've given me five chunks of text from a PDF document about TokenPack. The question is: "What is TokenPack's main goal?"

Let me carefully review each chunk to find the answer. 

Starting with Chunk 1: It says "TokenPack is a prototype retrieval system that treats a language model context window as a limited token budget." Then it explains the goal: "The goal is to select the most useful chunks without exceeding the context budget." That seems like a direct statement of the main goal.

Chunk 2 talks about context window budget but doesn't restate the main goal. Chunk 3 gives an example but doesn't define the goal. Chunk 4 and 5 discuss implementation details and success criteria but don't explicitly state the main goal.

The most precise answer comes from
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 8

- Model: `Medium local` / `qwen3:4b`
- Strategy: `knapsack`
- Status: `completed`
- Context tokens: `660`
- Evidence recall: `1.00`
- Query: What is TokenPack's main goal?
- Gold answer: TokenPack's goal is to select the most useful semantic chunks without exceeding a limited language-model context budget.

Answer:

```text
Hmm, the user wants me to answer a question about TokenPack based solely on the provided context. They've given me five chunks of text from a PDF document about TokenPack. The question is: "What is TokenPack's main goal?"

Let me carefully review each chunk to find the answer. 

Starting with Chunk 1: It says "TokenPack is a prototype retrieval system that treats a language model context window as a limited token budget." Then it explains the goal: "The goal is to select the most useful chunks without exceeding the context budget." That seems like a direct statement of the main goal.

Chunk 2 talks about context window budget but doesn't restate the main goal. Chunk 3 gives an example but doesn't define the goal. Chunk 4 and 5 discuss implementation details and success criteria but don't explicitly state the main goal.

The most precise answer comes from
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 9

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `document-prefix`
- Status: `completed`
- Context tokens: `698`
- Evidence recall: `1.00`
- Query: How does TokenPack map context selection to the 0/1 Knapsack Problem?
- Gold answer: Each chunk is an item, token count is the weight, semantic relevance is the value, and the available context window is the capacity.

Answer:

```text
TokenPack maps context selection to the 0/1 Knapsack Problem by selecting the most relevant chunks without exceeding the context budget, where each chunk is an item with a token weight and semantic relevance score.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 10

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `full-document`
- Status: `invalid-over-budget`
- Context tokens: `1539`
- Evidence recall: `1.00`
- Query: How does TokenPack map context selection to the 0/1 Knapsack Problem?
- Gold answer: Each chunk is an item, token count is the weight, semantic relevance is the value, and the available context window is the capacity.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 11

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `top-k`
- Status: `invalid-over-budget`
- Context tokens: `757`
- Evidence recall: `1.00`
- Query: How does TokenPack map context selection to the 0/1 Knapsack Problem?
- Gold answer: Each chunk is an item, token count is the weight, semantic relevance is the value, and the available context window is the capacity.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 12

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `knapsack`
- Status: `completed`
- Context tokens: `660`
- Evidence recall: `1.00`
- Query: How does TokenPack map context selection to the 0/1 Knapsack Problem?
- Gold answer: Each chunk is an item, token count is the weight, semantic relevance is the value, and the available context window is the capacity.

Answer:

```text
TokenPack maps context selection to the 0/1 Knapsack Problem by selecting the most useful chunks with a token weight, prioritizing relevance over the total token budget.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 13

- Model: `Medium local` / `qwen3:4b`
- Strategy: `document-prefix`
- Status: `completed`
- Context tokens: `698`
- Evidence recall: `1.00`
- Query: How does TokenPack map context selection to the 0/1 Knapsack Problem?
- Gold answer: Each chunk is an item, token count is the weight, semantic relevance is the value, and the available context window is the capacity.

Answer:

```text
Hmm, the user wants me to answer a specific question about TokenPack based solely on the provided context. They've given me four chunks of text from a PDF about TokenPack's knowledge base. The question is: "How does TokenPack map context selection to the 0/1 Knapsack Problem?"

Let me carefully review the context chunks to find the relevant information. 

Looking at Chunk 1, I see this key passage: "The main idea is based on the 0/1 Knapsack Problem. In this mapping, every chunk is an item, the token count is the item weight, the semantic relevance score is the item value, and the available context window is the knapsack capacity."

That's exactly what I need! This chunk clearly explains the mapping: 
- Chunks become items
- Token count = item weight
- Semantic relevance score = item value
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 14

- Model: `Medium local` / `qwen3:4b`
- Strategy: `full-document`
- Status: `invalid-over-budget`
- Context tokens: `1539`
- Evidence recall: `1.00`
- Query: How does TokenPack map context selection to the 0/1 Knapsack Problem?
- Gold answer: Each chunk is an item, token count is the weight, semantic relevance is the value, and the available context window is the capacity.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 15

- Model: `Medium local` / `qwen3:4b`
- Strategy: `top-k`
- Status: `invalid-over-budget`
- Context tokens: `757`
- Evidence recall: `1.00`
- Query: How does TokenPack map context selection to the 0/1 Knapsack Problem?
- Gold answer: Each chunk is an item, token count is the weight, semantic relevance is the value, and the available context window is the capacity.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 16

- Model: `Medium local` / `qwen3:4b`
- Strategy: `knapsack`
- Status: `completed`
- Context tokens: `660`
- Evidence recall: `1.00`
- Query: How does TokenPack map context selection to the 0/1 Knapsack Problem?
- Gold answer: Each chunk is an item, token count is the weight, semantic relevance is the value, and the available context window is the capacity.

Answer:

```text
Hmm, the user wants me to answer a specific question about TokenPack's mapping to the 0/1 Knapsack Problem using only the provided context. They've given me five chunks of text from a PDF about TokenPack.

Let me carefully examine the context to find the relevant information. Looking at Chunk 1, it states: "The main idea is based on the 0/1 Knapsack Problem. In this mapping, every chunk is an item, the token count is the item weight, the semantic relevance score is the item value, and the available context window is the knapsack capacity."

That's exactly what I need! This chunk clearly explains the mapping: 
- Chunks become items
- Token count becomes item weight
- Semantic relevance score becomes item value
- Available context window becomes knapsack capacity

The question is specifically about how TokenPack maps context
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 17

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `document-prefix`
- Status: `completed`
- Context tokens: `698`
- Evidence recall: `1.00`
- Query: Why can top-k retrieval be unsafe under a token budget?
- Gold answer: Top-k can select the highest ranked chunks without checking their total token cost, so it may exceed the effective context budget.

Answer:

```text
Top-k retrieval can be unsafe under a token budget because it selects a fixed number of chunks based on similarity, which may exceed the available context budget. If the selected chunks are too long, they could exceed the budget, making the approach invalid for production use.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 18

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `full-document`
- Status: `invalid-over-budget`
- Context tokens: `1539`
- Evidence recall: `1.00`
- Query: Why can top-k retrieval be unsafe under a token budget?
- Gold answer: Top-k can select the highest ranked chunks without checking their total token cost, so it may exceed the effective context budget.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 19

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `top-k`
- Status: `invalid-over-budget`
- Context tokens: `901`
- Evidence recall: `1.00`
- Query: Why can top-k retrieval be unsafe under a token budget?
- Gold answer: Top-k can select the highest ranked chunks without checking their total token cost, so it may exceed the effective context budget.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 20

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `knapsack`
- Status: `completed`
- Context tokens: `698`
- Evidence recall: `1.00`
- Query: Why can top-k retrieval be unsafe under a token budget?
- Gold answer: Top-k can select the highest ranked chunks without checking their total token cost, so it may exceed the effective context budget.

Answer:

```text
Top-k retrieval can be unsafe under a token budget because it selects a fixed number of chunks based on similarity scores, which may exceed the available budget if the selected chunks are too large. This approach can lead to over-billing, making it unsuitable for production use.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 21

- Model: `Medium local` / `qwen3:4b`
- Strategy: `document-prefix`
- Status: `completed`
- Context tokens: `698`
- Evidence recall: `1.00`
- Query: Why can top-k retrieval be unsafe under a token budget?
- Gold answer: Top-k can select the highest ranked chunks without checking their total token cost, so it may exceed the effective context budget.

Answer:

```text
Hmm, the user wants me to answer a question about TokenPack based solely on the provided context. They've given me four chunks of text from a PDF about TokenPack's knowledge base. The question is: "Why can top-k retrieval be unsafe under a token budget?"

Let me carefully review the context chunks to find the answer. 

Looking at Chunk 2, I see this relevant information: "A top-k retriever can select the highest ranked chunks without checking their total token cost. In contrast, a budget-aware method must ensure that the final selected context never exceeds the effective budget." 

This directly explains why top-k retrieval is unsafe - it doesn't check the total token cost of selected chunks. It just picks the top k chunks by relevance score without considering how much space they'll take in the token budget. 

The context also mentions in Chunk 3 that "over-budget selection is
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 22

- Model: `Medium local` / `qwen3:4b`
- Strategy: `full-document`
- Status: `invalid-over-budget`
- Context tokens: `1539`
- Evidence recall: `1.00`
- Query: Why can top-k retrieval be unsafe under a token budget?
- Gold answer: Top-k can select the highest ranked chunks without checking their total token cost, so it may exceed the effective context budget.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 23

- Model: `Medium local` / `qwen3:4b`
- Strategy: `top-k`
- Status: `invalid-over-budget`
- Context tokens: `901`
- Evidence recall: `1.00`
- Query: Why can top-k retrieval be unsafe under a token budget?
- Gold answer: Top-k can select the highest ranked chunks without checking their total token cost, so it may exceed the effective context budget.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 24

- Model: `Medium local` / `qwen3:4b`
- Strategy: `knapsack`
- Status: `completed`
- Context tokens: `698`
- Evidence recall: `1.00`
- Query: Why can top-k retrieval be unsafe under a token budget?
- Gold answer: Top-k can select the highest ranked chunks without checking their total token cost, so it may exceed the effective context budget.

Answer:

```text
Hmm, the user wants me to answer a specific question about why top-k retrieval can be unsafe under a token budget, using only the provided context. They've given me four chunks of text from what appears to be a technical document about TokenPack.

Let me carefully examine the context to find the answer. Looking at Chunk 1, it mentions that "a top-k retriever can select the highest ranked chunks without checking their total token cost." This is important because it directly addresses the issue. The context also states that "Budget awareness is important because a retrieval method may find relevant chunks but still fail if the selected text is too long."

Chunk 3 provides additional context about how chunking quality affects retrieval, but the core answer seems to be in Chunk 1. The key point is that top-k retrieval doesn't consider token costs when selecting chunks - it just picks the highest ranked ones regardless of
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 25

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `document-prefix`
- Status: `completed`
- Context tokens: `698`
- Evidence recall: `1.00`
- Query: What does paragraph-group chunking do?
- Gold answer: Paragraph-group chunking groups neighboring paragraphs until a target token range is reached, preserving local context instead of cutting at a fixed word count.

Answer:

```text
Paragraph-group chunking groups neighboring paragraphs until a target token range is reached, keeping local context together and avoiding cutting every fixed number of words.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 26

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `full-document`
- Status: `invalid-over-budget`
- Context tokens: `1539`
- Evidence recall: `1.00`
- Query: What does paragraph-group chunking do?
- Gold answer: Paragraph-group chunking groups neighboring paragraphs until a target token range is reached, preserving local context instead of cutting at a fixed word count.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 27

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `top-k`
- Status: `invalid-over-budget`
- Context tokens: `879`
- Evidence recall: `1.00`
- Query: What does paragraph-group chunking do?
- Gold answer: Paragraph-group chunking groups neighboring paragraphs until a target token range is reached, preserving local context instead of cutting at a fixed word count.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 28

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `knapsack`
- Status: `completed`
- Context tokens: `639`
- Evidence recall: `1.00`
- Query: What does paragraph-group chunking do?
- Gold answer: Paragraph-group chunking groups neighboring paragraphs until a target token range is reached, preserving local context instead of cutting at a fixed word count.

Answer:

```text
Paragraph-group chunking groups neighboring paragraphs until a target token range is reached, keeping local context together and avoiding cutting every fixed number of words.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 29

- Model: `Medium local` / `qwen3:4b`
- Strategy: `document-prefix`
- Status: `completed`
- Context tokens: `698`
- Evidence recall: `1.00`
- Query: What does paragraph-group chunking do?
- Gold answer: Paragraph-group chunking groups neighboring paragraphs until a target token range is reached, preserving local context instead of cutting at a fixed word count.

Answer:

```text
I need to find information about paragraph-group chunking in the provided context. Let me search through the chunks.

In Chunk 3, I find this relevant information:
"Semantic chunking is the process of dividing a document into meaningful pieces. TokenPack supports paragraph-group chunking and semantic-threshold chunking. Paragraph-group chunking groups neighboring paragraphs until a target token range is reached. This approach keeps local context together and avoids cutting every xed number of words."

This clearly explains what paragraph-group chunking does. According to the context, paragraph-group chunking:
1. Groups neighboring paragraphs until a target token range is reached
2. Keeps local context together
3. Avoids cutting every fixed number of words

So the answer to "What does paragraph-group chunking do?" is that it groups neighboring paragraphs until a target token range is reached, keeping local context together and avoiding cutting every fixed
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 30

- Model: `Medium local` / `qwen3:4b`
- Strategy: `full-document`
- Status: `invalid-over-budget`
- Context tokens: `1539`
- Evidence recall: `1.00`
- Query: What does paragraph-group chunking do?
- Gold answer: Paragraph-group chunking groups neighboring paragraphs until a target token range is reached, preserving local context instead of cutting at a fixed word count.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 31

- Model: `Medium local` / `qwen3:4b`
- Strategy: `top-k`
- Status: `invalid-over-budget`
- Context tokens: `879`
- Evidence recall: `1.00`
- Query: What does paragraph-group chunking do?
- Gold answer: Paragraph-group chunking groups neighboring paragraphs until a target token range is reached, preserving local context instead of cutting at a fixed word count.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 32

- Model: `Medium local` / `qwen3:4b`
- Strategy: `knapsack`
- Status: `completed`
- Context tokens: `639`
- Evidence recall: `1.00`
- Query: What does paragraph-group chunking do?
- Gold answer: Paragraph-group chunking groups neighboring paragraphs until a target token range is reached, preserving local context instead of cutting at a fixed word count.

Answer:

```text
Hmm, the user wants me to answer a specific question about TokenPack based solely on the provided context. They've given me five chunks of text from a PDF document about TokenPack, and I need to find what paragraph-group chunking does.

Let me carefully examine each chunk to find the relevant information. Looking at Chunk 1, I see this section: "TokenPack supports paragraph-group chunking and semantic-threshold chunking. Paragraph-group chunking groups neighboring paragraphs until a target token range is reached. This approach keeps local context together and avoids cutting every xed number of words."

Ah, here's the direct answer! Paragraph-group chunking groups neighboring paragraphs until a target token range is reached, and it keeps local context together while avoiding fixed word counts. That's exactly what the question is asking about.

I should check if any other chunks provide additional context, but Chunk 1 seems to
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 33

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `document-prefix`
- Status: `completed`
- Context tokens: `698`
- Evidence recall: `1.00`
- Query: What does semantic-threshold chunking do?
- Gold answer: Semantic-threshold chunking compares neighboring text blocks with embedding similarity and starts a new chunk when similarity drops below a threshold.

Answer:

```text
Semantic-threshold chunking compares neighboring text blocks using embedding similarity. If the similarity between two neighboring blocks drops below a threshold, TokenPack treats that location as a topic change and starts a new chunk. This helps keep unrelated topics in separate chunks.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 34

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `full-document`
- Status: `invalid-over-budget`
- Context tokens: `1539`
- Evidence recall: `1.00`
- Query: What does semantic-threshold chunking do?
- Gold answer: Semantic-threshold chunking compares neighboring text blocks with embedding similarity and starts a new chunk when similarity drops below a threshold.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 35

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `top-k`
- Status: `invalid-over-budget`
- Context tokens: `879`
- Evidence recall: `1.00`
- Query: What does semantic-threshold chunking do?
- Gold answer: Semantic-threshold chunking compares neighboring text blocks with embedding similarity and starts a new chunk when similarity drops below a threshold.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 36

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `knapsack`
- Status: `completed`
- Context tokens: `639`
- Evidence recall: `1.00`
- Query: What does semantic-threshold chunking do?
- Gold answer: Semantic-threshold chunking compares neighboring text blocks with embedding similarity and starts a new chunk when similarity drops below a threshold.

Answer:

```text
Semantic-threshold chunking compares neighboring text blocks using embedding similarity. If the similarity between two neighboring blocks drops below a threshold, TokenPack treats that location as a topic change and starts a new chunk.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 37

- Model: `Medium local` / `qwen3:4b`
- Strategy: `document-prefix`
- Status: `completed`
- Context tokens: `698`
- Evidence recall: `1.00`
- Query: What does semantic-threshold chunking do?
- Gold answer: Semantic-threshold chunking compares neighboring text blocks with embedding similarity and starts a new chunk when similarity drops below a threshold.

Answer:

```text
Hmm, the user wants me to answer a specific question about semantic-threshold chunking using only the provided context. They've given me four chunks of text from a TokenPack demo knowledge base.

Let me carefully examine what the context says about semantic-threshold chunking. Looking at Chunk 3, I find this relevant passage: "Semantic-threshold chunking compares neighboring text blocks using embedding similarity. If the similarity between two neighboring blocks drops below a threshold, TokenPack treats that location as a topic change and starts a new chunk. This helps keep unrelated topics in separate chunks."

This clearly explains what semantic-threshold chunking does. It compares text blocks with embedding similarity, creates new chunks when similarity drops below a threshold, and this helps separate unrelated topics.

I need to make sure I'm answering precisely what the question asks. The question is "What does semantic-threshold chunking do?" and
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 38

- Model: `Medium local` / `qwen3:4b`
- Strategy: `full-document`
- Status: `invalid-over-budget`
- Context tokens: `1539`
- Evidence recall: `1.00`
- Query: What does semantic-threshold chunking do?
- Gold answer: Semantic-threshold chunking compares neighboring text blocks with embedding similarity and starts a new chunk when similarity drops below a threshold.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 39

- Model: `Medium local` / `qwen3:4b`
- Strategy: `top-k`
- Status: `invalid-over-budget`
- Context tokens: `879`
- Evidence recall: `1.00`
- Query: What does semantic-threshold chunking do?
- Gold answer: Semantic-threshold chunking compares neighboring text blocks with embedding similarity and starts a new chunk when similarity drops below a threshold.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 40

- Model: `Medium local` / `qwen3:4b`
- Strategy: `knapsack`
- Status: `completed`
- Context tokens: `639`
- Evidence recall: `1.00`
- Query: What does semantic-threshold chunking do?
- Gold answer: Semantic-threshold chunking compares neighboring text blocks with embedding similarity and starts a new chunk when similarity drops below a threshold.

Answer:

```text
Hmm, the user wants me to answer a specific question about semantic-threshold chunking using only the provided context. They've given me five chunks of text from a PDF about TokenPack, and I need to be precise - just one or two sentences max.

Let me scan the context carefully. In Chunk 1, I find the key information: "Semantic-threshold chunking compares neighboring text blocks using embedding similarity. If the similarity between two neighboring blocks drops below a threshold, TokenPack treats that location as a topic change and starts a new chunk." That's exactly what the question is asking about.

I should check if other chunks provide additional context. Chunk 4 mentions "semantic-threshold chunking starts a new chunk when neighboring block similarity drops below a threshold" - this confirms the same point but in a different wording. The other chunks discuss chunking quality, retrieval strategies, and gold review
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 41

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `document-prefix`
- Status: `completed`
- Context tokens: `698`
- Evidence recall: `0.50`
- Query: Which retrieval strategies does TokenPack compare?
- Gold answer: TokenPack compares top-k, budget-top-k, MMR, knapsack, greedy strategies, and simulated annealing.

Answer:

```text
The context provided does not contain the answer to the question. The context is insufficient to answer the question.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 42

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `full-document`
- Status: `invalid-over-budget`
- Context tokens: `1539`
- Evidence recall: `1.00`
- Query: Which retrieval strategies does TokenPack compare?
- Gold answer: TokenPack compares top-k, budget-top-k, MMR, knapsack, greedy strategies, and simulated annealing.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 43

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `top-k`
- Status: `invalid-over-budget`
- Context tokens: `901`
- Evidence recall: `0.50`
- Query: Which retrieval strategies does TokenPack compare?
- Gold answer: TokenPack compares top-k, budget-top-k, MMR, knapsack, greedy strategies, and simulated annealing.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 44

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `knapsack`
- Status: `completed`
- Context tokens: `661`
- Evidence recall: `0.50`
- Query: Which retrieval strategies does TokenPack compare?
- Gold answer: TokenPack compares top-k, budget-top-k, MMR, knapsack, greedy strategies, and simulated annealing.

Answer:

```text
The context does not contain the answer.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 45

- Model: `Medium local` / `qwen3:4b`
- Strategy: `document-prefix`
- Status: `completed`
- Context tokens: `698`
- Evidence recall: `0.50`
- Query: Which retrieval strategies does TokenPack compare?
- Gold answer: TokenPack compares top-k, budget-top-k, MMR, knapsack, greedy strategies, and simulated annealing.

Answer:

```text
Hmm, the user wants me to answer a question about TokenPack retrieval strategies using only the provided context. They've given me four chunks of text from a PDF about TokenPack.

Let me carefully check what the context says about retrieval strategies. Looking at Chunk 4, it mentions: "TokenPack compares several retrieval and selection strategies." Then it lists specific ones: Top-k retrieval, Budget-top-k, MMR (Maximal Marginal Relevance), and the knapsack strategy.

I see that Chunk 2 also mentions "a top-k retriever" and "a budget-aware method" but doesn't name the full set. Chunk 3 talks about chunking but not retrieval strategies. Chunk 1 is more about the overall concept.

The question is specifically about which retrieval strategies TokenPack compares. From Chunk 4, it clearly states four strategies: Top-k retrieval, Budget-top-k
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 46

- Model: `Medium local` / `qwen3:4b`
- Strategy: `full-document`
- Status: `invalid-over-budget`
- Context tokens: `1539`
- Evidence recall: `1.00`
- Query: Which retrieval strategies does TokenPack compare?
- Gold answer: TokenPack compares top-k, budget-top-k, MMR, knapsack, greedy strategies, and simulated annealing.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 47

- Model: `Medium local` / `qwen3:4b`
- Strategy: `top-k`
- Status: `invalid-over-budget`
- Context tokens: `901`
- Evidence recall: `0.50`
- Query: Which retrieval strategies does TokenPack compare?
- Gold answer: TokenPack compares top-k, budget-top-k, MMR, knapsack, greedy strategies, and simulated annealing.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 48

- Model: `Medium local` / `qwen3:4b`
- Strategy: `knapsack`
- Status: `completed`
- Context tokens: `661`
- Evidence recall: `0.50`
- Query: Which retrieval strategies does TokenPack compare?
- Gold answer: TokenPack compares top-k, budget-top-k, MMR, knapsack, greedy strategies, and simulated annealing.

Answer:

```text
First, I need to answer the question: "Which retrieval strategies does TokenPack compare?" using only the provided context. If the context doesn't have the answer, I should say the context is insufficient. Keep it to one or two sentences.

Let me look at the context chunks to find information about TokenPack's retrieval strategies.

Starting with Chunk 1: It says "TokenPack compares several retrieval and selection strategies." Then it lists: "Top-k retrieval selects a fixed number of chunks by similarity score. Budget-top-k follows the similarity ranking but skips chunks that would exceed the remaining budget. MMR, or Maximal Marginal Relevance, also considers redundancy and tries to avoid selecting chunks that are too similar to each other. The knapsack strategy solves a 0/1 optimization problem. It tries to maximize total relevance value while keeping the selected token count under the budget."

So,
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 49

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `document-prefix`
- Status: `completed`
- Context tokens: `698`
- Evidence recall: `0.00`
- Query: Which metrics are used in TokenPack evaluation?
- Gold answer: TokenPack uses evidence recall, evidence precision, selected token count, budget utilization, value density, redundancy score, and latency.

Answer:

```text
The context is insufficient.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 50

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `full-document`
- Status: `invalid-over-budget`
- Context tokens: `1539`
- Evidence recall: `1.00`
- Query: Which metrics are used in TokenPack evaluation?
- Gold answer: TokenPack uses evidence recall, evidence precision, selected token count, budget utilization, value density, redundancy score, and latency.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 51

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `top-k`
- Status: `invalid-over-budget`
- Context tokens: `735`
- Evidence recall: `1.00`
- Query: Which metrics are used in TokenPack evaluation?
- Gold answer: TokenPack uses evidence recall, evidence precision, selected token count, budget utilization, value density, redundancy score, and latency.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 52

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `knapsack`
- Status: `completed`
- Context tokens: `638`
- Evidence recall: `1.00`
- Query: Which metrics are used in TokenPack evaluation?
- Gold answer: TokenPack uses evidence recall, evidence precision, selected token count, budget utilization, value density, redundancy score, and latency.

Answer:

```text
The context is insufficient.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 53

- Model: `Medium local` / `qwen3:4b`
- Strategy: `document-prefix`
- Status: `completed`
- Context tokens: `698`
- Evidence recall: `0.00`
- Query: Which metrics are used in TokenPack evaluation?
- Gold answer: TokenPack uses evidence recall, evidence precision, selected token count, budget utilization, value density, redundancy score, and latency.

Answer:

```text
I need to analyze the provided context to determine which metrics are used in TokenPack evaluation.

Let me go through each chunk to find information about evaluation metrics:

Chunk 1: This chunk describes what TokenPack is and its basic concept as a prototype retrieval system based on the 0/1 Knapsack Problem. It mentions that TokenPack is designed as a Python package and command-line tool, but doesn't specifically mention evaluation metrics.

Chunk 2: This chunk discusses the context window budget and how TokenPack separates the total context window into parts for retrieved chunks and the model's answer. It mentions that budget awareness is important and that a top-k retriever can select highest ranked chunks without checking their total token cost, while a budget-aware method must ensure the selected context doesn't exceed the effective budget. But it doesn't specifically mention evaluation metrics.

Chunk 3: This chunk discusses semantic chunking
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 54

- Model: `Medium local` / `qwen3:4b`
- Strategy: `full-document`
- Status: `invalid-over-budget`
- Context tokens: `1539`
- Evidence recall: `1.00`
- Query: Which metrics are used in TokenPack evaluation?
- Gold answer: TokenPack uses evidence recall, evidence precision, selected token count, budget utilization, value density, redundancy score, and latency.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 55

- Model: `Medium local` / `qwen3:4b`
- Strategy: `top-k`
- Status: `invalid-over-budget`
- Context tokens: `735`
- Evidence recall: `1.00`
- Query: Which metrics are used in TokenPack evaluation?
- Gold answer: TokenPack uses evidence recall, evidence precision, selected token count, budget utilization, value density, redundancy score, and latency.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 56

- Model: `Medium local` / `qwen3:4b`
- Strategy: `knapsack`
- Status: `completed`
- Context tokens: `638`
- Evidence recall: `1.00`
- Query: Which metrics are used in TokenPack evaluation?
- Gold answer: TokenPack uses evidence recall, evidence precision, selected token count, budget utilization, value density, redundancy score, and latency.

Answer:

```text
Hmm, the user wants me to answer a question about TokenPack evaluation metrics using only the provided context. They've given me five chunks of text from a PDF document about TokenPack.

Let me carefully examine each chunk to find the answer. The question is specifically about which metrics are used in TokenPack evaluation.

Looking at Chunk 2, I see this relevant passage: "TokenPack evaluation compares strategies under the same budget. Important metrics include evidence recall, evidence precision, selected token count, budget utilization, value density, redundancy score, and latency."

This directly lists all the evaluation metrics mentioned in the context. The other chunks don't seem to contain information about evaluation metrics - Chunk 1 talks about semantic chunking, Chunk 3 discusses budget constraints, Chunk 4 covers local language model usage, and Chunk 5 mentions success criteria.

The context is sufficient here because Chunk 2 explicitly states the
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 57

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `document-prefix`
- Status: `completed`
- Context tokens: `698`
- Evidence recall: `0.00`
- Query: Why can TokenPack run without an OpenAI API key?
- Gold answer: The retrieval benchmark does not require a language model, and optional local generation can use an Ollama model installed on the same computer.

Answer:

```text
The context is insufficient to answer the question.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 58

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `full-document`
- Status: `invalid-over-budget`
- Context tokens: `1539`
- Evidence recall: `1.00`
- Query: Why can TokenPack run without an OpenAI API key?
- Gold answer: The retrieval benchmark does not require a language model, and optional local generation can use an Ollama model installed on the same computer.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 59

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `top-k`
- Status: `invalid-over-budget`
- Context tokens: `758`
- Evidence recall: `1.00`
- Query: Why can TokenPack run without an OpenAI API key?
- Gold answer: The retrieval benchmark does not require a language model, and optional local generation can use an Ollama model installed on the same computer.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 60

- Model: `Small local` / `qwen3:0.6b`
- Strategy: `knapsack`
- Status: `completed`
- Context tokens: `661`
- Evidence recall: `1.00`
- Query: Why can TokenPack run without an OpenAI API key?
- Gold answer: The retrieval benchmark does not require a language model, and optional local generation can use an Ollama model installed on the same computer.

Answer:

```text
The context is insufficient.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 61

- Model: `Medium local` / `qwen3:4b`
- Strategy: `document-prefix`
- Status: `completed`
- Context tokens: `698`
- Evidence recall: `0.00`
- Query: Why can TokenPack run without an OpenAI API key?
- Gold answer: The retrieval benchmark does not require a language model, and optional local generation can use an Ollama model installed on the same computer.

Answer:

```text
Hmm, the user wants me to answer a specific question about TokenPack using only the provided context. The question is: "Why can TokenPack run without an OpenAI API key?"

Let me carefully review the context chunks to see if they contain any information about this. 

Looking at Chunk 1: It describes TokenPack as a prototype retrieval system that treats language model context windows as token budgets. It mentions TokenPack is designed as a Python package and command-line tool that can "ingest text or PDF files, create chunks, compute embeddings..." but doesn't mention anything about API keys.

Chunk 2: Talks about context window budget and how TokenPack separates the budget into retrieved chunks and model answer space. Still no mention of API keys.

Chunk 3: Describes semantic chunking methods but again, no API key information.

Chunk 4: Discusses chunking quality and retrieval strategies
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 62

- Model: `Medium local` / `qwen3:4b`
- Strategy: `full-document`
- Status: `invalid-over-budget`
- Context tokens: `1539`
- Evidence recall: `1.00`
- Query: Why can TokenPack run without an OpenAI API key?
- Gold answer: The retrieval benchmark does not require a language model, and optional local generation can use an Ollama model installed on the same computer.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 63

- Model: `Medium local` / `qwen3:4b`
- Strategy: `top-k`
- Status: `invalid-over-budget`
- Context tokens: `758`
- Evidence recall: `1.00`
- Query: Why can TokenPack run without an OpenAI API key?
- Gold answer: The retrieval benchmark does not require a language model, and optional local generation can use an Ollama model installed on the same computer.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 64

- Model: `Medium local` / `qwen3:4b`
- Strategy: `knapsack`
- Status: `completed`
- Context tokens: `661`
- Evidence recall: `1.00`
- Query: Why can TokenPack run without an OpenAI API key?
- Gold answer: The retrieval benchmark does not require a language model, and optional local generation can use an Ollama model installed on the same computer.

Answer:

```text
Hmm, the user wants me to answer a specific question about TokenPack using only the provided context. They've given me five chunks of text from a PDF document about TokenPack, and I need to find why TokenPack can run without an OpenAI API key.

Let me carefully examine each chunk to find the relevant information. 

Looking at Chunk 4, I see this relevant passage: "TokenPack can run without an external API key. For local generation, it can use an Ollama model installed on the same computer." This directly answers the question - TokenPack doesn't require an OpenAI API key because it can use the Ollama model locally on the same computer.

I should also check if other chunks provide additional context. Chunk 5 mentions that the main success criterion is selecting valuable evidence chunks while respecting the context budget, but that doesn't directly address the API question. Chunk
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 65

- Model: `Large cloud` / `gpt-4o`
- Strategy: `document-prefix`
- Status: `completed`
- Context tokens: `698`
- Evidence recall: `1.00`
- Query: What is TokenPack's main goal?
- Gold answer: TokenPack's goal is to select the most useful semantic chunks without exceeding a limited language-model context budget.

Answer:

```text
The context is insufficient.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 66

- Model: `Large cloud` / `gpt-4o`
- Strategy: `full-document`
- Status: `invalid-over-budget`
- Context tokens: `1539`
- Evidence recall: `1.00`
- Query: What is TokenPack's main goal?
- Gold answer: TokenPack's goal is to select the most useful semantic chunks without exceeding a limited language-model context budget.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 67

- Model: `Large cloud` / `gpt-4o`
- Strategy: `top-k`
- Status: `completed`
- Context tokens: `660`
- Evidence recall: `1.00`
- Query: What is TokenPack's main goal?
- Gold answer: TokenPack's goal is to select the most useful semantic chunks without exceeding a limited language-model context budget.

Answer:

```text
The context is insufficient.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 68

- Model: `Large cloud` / `gpt-4o`
- Strategy: `knapsack`
- Status: `completed`
- Context tokens: `660`
- Evidence recall: `1.00`
- Query: What is TokenPack's main goal?
- Gold answer: TokenPack's goal is to select the most useful semantic chunks without exceeding a limited language-model context budget.

Answer:

```text
The context is insufficient.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 69

- Model: `Large cloud` / `gpt-4o`
- Strategy: `document-prefix`
- Status: `completed`
- Context tokens: `698`
- Evidence recall: `1.00`
- Query: How does TokenPack map context selection to the 0/1 Knapsack Problem?
- Gold answer: Each chunk is an item, token count is the weight, semantic relevance is the value, and the available context window is the capacity.

Answer:

```text
TokenPack maps context selection to the 0/1 Knapsack Problem by treating each semantic chunk as an item, the token count as the item weight, the semantic relevance score as the item value, and the available context window as the knapsack capacity.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 70

- Model: `Large cloud` / `gpt-4o`
- Strategy: `full-document`
- Status: `invalid-over-budget`
- Context tokens: `1539`
- Evidence recall: `1.00`
- Query: How does TokenPack map context selection to the 0/1 Knapsack Problem?
- Gold answer: Each chunk is an item, token count is the weight, semantic relevance is the value, and the available context window is the capacity.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 71

- Model: `Large cloud` / `gpt-4o`
- Strategy: `top-k`
- Status: `invalid-over-budget`
- Context tokens: `757`
- Evidence recall: `1.00`
- Query: How does TokenPack map context selection to the 0/1 Knapsack Problem?
- Gold answer: Each chunk is an item, token count is the weight, semantic relevance is the value, and the available context window is the capacity.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 72

- Model: `Large cloud` / `gpt-4o`
- Strategy: `knapsack`
- Status: `completed`
- Context tokens: `660`
- Evidence recall: `1.00`
- Query: How does TokenPack map context selection to the 0/1 Knapsack Problem?
- Gold answer: Each chunk is an item, token count is the weight, semantic relevance is the value, and the available context window is the capacity.

Answer:

```text
TokenPack maps context selection to the 0/1 Knapsack Problem by treating each chunk as an item, the token count as the item weight, the semantic relevance score as the item value, and the available context window as the knapsack capacity.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 73

- Model: `Large cloud` / `gpt-4o`
- Strategy: `document-prefix`
- Status: `completed`
- Context tokens: `698`
- Evidence recall: `1.00`
- Query: Why can top-k retrieval be unsafe under a token budget?
- Gold answer: Top-k can select the highest ranked chunks without checking their total token cost, so it may exceed the effective context budget.

Answer:

```text
The context is insufficient.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 74

- Model: `Large cloud` / `gpt-4o`
- Strategy: `full-document`
- Status: `invalid-over-budget`
- Context tokens: `1539`
- Evidence recall: `1.00`
- Query: Why can top-k retrieval be unsafe under a token budget?
- Gold answer: Top-k can select the highest ranked chunks without checking their total token cost, so it may exceed the effective context budget.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 75

- Model: `Large cloud` / `gpt-4o`
- Strategy: `top-k`
- Status: `invalid-over-budget`
- Context tokens: `901`
- Evidence recall: `1.00`
- Query: Why can top-k retrieval be unsafe under a token budget?
- Gold answer: Top-k can select the highest ranked chunks without checking their total token cost, so it may exceed the effective context budget.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 76

- Model: `Large cloud` / `gpt-4o`
- Strategy: `knapsack`
- Status: `completed`
- Context tokens: `698`
- Evidence recall: `1.00`
- Query: Why can top-k retrieval be unsafe under a token budget?
- Gold answer: Top-k can select the highest ranked chunks without checking their total token cost, so it may exceed the effective context budget.

Answer:

```text
The context is insufficient.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 77

- Model: `Large cloud` / `gpt-4o`
- Strategy: `document-prefix`
- Status: `completed`
- Context tokens: `698`
- Evidence recall: `1.00`
- Query: What does paragraph-group chunking do?
- Gold answer: Paragraph-group chunking groups neighboring paragraphs until a target token range is reached, preserving local context instead of cutting at a fixed word count.

Answer:

```text
Paragraph-group chunking groups neighboring paragraphs until a target token range is reached, keeping local context together and avoiding cutting every fixed number of words.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 78

- Model: `Large cloud` / `gpt-4o`
- Strategy: `full-document`
- Status: `invalid-over-budget`
- Context tokens: `1539`
- Evidence recall: `1.00`
- Query: What does paragraph-group chunking do?
- Gold answer: Paragraph-group chunking groups neighboring paragraphs until a target token range is reached, preserving local context instead of cutting at a fixed word count.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 79

- Model: `Large cloud` / `gpt-4o`
- Strategy: `top-k`
- Status: `invalid-over-budget`
- Context tokens: `879`
- Evidence recall: `1.00`
- Query: What does paragraph-group chunking do?
- Gold answer: Paragraph-group chunking groups neighboring paragraphs until a target token range is reached, preserving local context instead of cutting at a fixed word count.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 80

- Model: `Large cloud` / `gpt-4o`
- Strategy: `knapsack`
- Status: `completed`
- Context tokens: `639`
- Evidence recall: `1.00`
- Query: What does paragraph-group chunking do?
- Gold answer: Paragraph-group chunking groups neighboring paragraphs until a target token range is reached, preserving local context instead of cutting at a fixed word count.

Answer:

```text
Paragraph-group chunking groups neighboring paragraphs until a target token range is reached, keeping local context together and avoiding cutting every fixed number of words.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 81

- Model: `Large cloud` / `gpt-4o`
- Strategy: `document-prefix`
- Status: `completed`
- Context tokens: `698`
- Evidence recall: `1.00`
- Query: What does semantic-threshold chunking do?
- Gold answer: Semantic-threshold chunking compares neighboring text blocks with embedding similarity and starts a new chunk when similarity drops below a threshold.

Answer:

```text
Semantic-threshold chunking compares neighboring text blocks using embedding similarity, and if the similarity between two neighboring blocks drops below a threshold, TokenPack treats that location as a topic change and starts a new chunk.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 82

- Model: `Large cloud` / `gpt-4o`
- Strategy: `full-document`
- Status: `invalid-over-budget`
- Context tokens: `1539`
- Evidence recall: `1.00`
- Query: What does semantic-threshold chunking do?
- Gold answer: Semantic-threshold chunking compares neighboring text blocks with embedding similarity and starts a new chunk when similarity drops below a threshold.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 83

- Model: `Large cloud` / `gpt-4o`
- Strategy: `top-k`
- Status: `invalid-over-budget`
- Context tokens: `879`
- Evidence recall: `1.00`
- Query: What does semantic-threshold chunking do?
- Gold answer: Semantic-threshold chunking compares neighboring text blocks with embedding similarity and starts a new chunk when similarity drops below a threshold.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 84

- Model: `Large cloud` / `gpt-4o`
- Strategy: `knapsack`
- Status: `completed`
- Context tokens: `639`
- Evidence recall: `1.00`
- Query: What does semantic-threshold chunking do?
- Gold answer: Semantic-threshold chunking compares neighboring text blocks with embedding similarity and starts a new chunk when similarity drops below a threshold.

Answer:

```text
Semantic-threshold chunking compares neighboring text blocks using embedding similarity, and if the similarity between two neighboring blocks drops below a threshold, TokenPack treats that location as a topic change and starts a new chunk.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 85

- Model: `Large cloud` / `gpt-4o`
- Strategy: `document-prefix`
- Status: `completed`
- Context tokens: `698`
- Evidence recall: `0.50`
- Query: Which retrieval strategies does TokenPack compare?
- Gold answer: TokenPack compares top-k, budget-top-k, MMR, knapsack, greedy strategies, and simulated annealing.

Answer:

```text
The context is insufficient.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 86

- Model: `Large cloud` / `gpt-4o`
- Strategy: `full-document`
- Status: `invalid-over-budget`
- Context tokens: `1539`
- Evidence recall: `1.00`
- Query: Which retrieval strategies does TokenPack compare?
- Gold answer: TokenPack compares top-k, budget-top-k, MMR, knapsack, greedy strategies, and simulated annealing.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 87

- Model: `Large cloud` / `gpt-4o`
- Strategy: `top-k`
- Status: `invalid-over-budget`
- Context tokens: `901`
- Evidence recall: `0.50`
- Query: Which retrieval strategies does TokenPack compare?
- Gold answer: TokenPack compares top-k, budget-top-k, MMR, knapsack, greedy strategies, and simulated annealing.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 88

- Model: `Large cloud` / `gpt-4o`
- Strategy: `knapsack`
- Status: `completed`
- Context tokens: `661`
- Evidence recall: `0.50`
- Query: Which retrieval strategies does TokenPack compare?
- Gold answer: TokenPack compares top-k, budget-top-k, MMR, knapsack, greedy strategies, and simulated annealing.

Answer:

```text
The context is insufficient.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 89

- Model: `Large cloud` / `gpt-4o`
- Strategy: `document-prefix`
- Status: `completed`
- Context tokens: `698`
- Evidence recall: `0.00`
- Query: Which metrics are used in TokenPack evaluation?
- Gold answer: TokenPack uses evidence recall, evidence precision, selected token count, budget utilization, value density, redundancy score, and latency.

Answer:

```text
The context is insufficient.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 90

- Model: `Large cloud` / `gpt-4o`
- Strategy: `full-document`
- Status: `invalid-over-budget`
- Context tokens: `1539`
- Evidence recall: `1.00`
- Query: Which metrics are used in TokenPack evaluation?
- Gold answer: TokenPack uses evidence recall, evidence precision, selected token count, budget utilization, value density, redundancy score, and latency.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 91

- Model: `Large cloud` / `gpt-4o`
- Strategy: `top-k`
- Status: `invalid-over-budget`
- Context tokens: `735`
- Evidence recall: `1.00`
- Query: Which metrics are used in TokenPack evaluation?
- Gold answer: TokenPack uses evidence recall, evidence precision, selected token count, budget utilization, value density, redundancy score, and latency.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 92

- Model: `Large cloud` / `gpt-4o`
- Strategy: `knapsack`
- Status: `completed`
- Context tokens: `638`
- Evidence recall: `1.00`
- Query: Which metrics are used in TokenPack evaluation?
- Gold answer: TokenPack uses evidence recall, evidence precision, selected token count, budget utilization, value density, redundancy score, and latency.

Answer:

```text
The metrics used in TokenPack evaluation include evidence recall, evidence precision, selected token count, budget utilization, value density, redundancy score, and latency.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 93

- Model: `Large cloud` / `gpt-4o`
- Strategy: `document-prefix`
- Status: `completed`
- Context tokens: `698`
- Evidence recall: `0.00`
- Query: Why can TokenPack run without an OpenAI API key?
- Gold answer: The retrieval benchmark does not require a language model, and optional local generation can use an Ollama model installed on the same computer.

Answer:

```text
The context is insufficient.
```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 94

- Model: `Large cloud` / `gpt-4o`
- Strategy: `full-document`
- Status: `invalid-over-budget`
- Context tokens: `1539`
- Evidence recall: `1.00`
- Query: Why can TokenPack run without an OpenAI API key?
- Gold answer: The retrieval benchmark does not require a language model, and optional local generation can use an Ollama model installed on the same computer.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 95

- Model: `Large cloud` / `gpt-4o`
- Strategy: `top-k`
- Status: `invalid-over-budget`
- Context tokens: `758`
- Evidence recall: `1.00`
- Query: Why can TokenPack run without an OpenAI API key?
- Gold answer: The retrieval benchmark does not require a language model, and optional local generation can use an Ollama model installed on the same computer.

Answer:

```text

```

Human score: [ ] 0  [ ] 1  [ ] 2

## Record 96

- Model: `Large cloud` / `gpt-4o`
- Strategy: `knapsack`
- Status: `completed`
- Context tokens: `661`
- Evidence recall: `1.00`
- Query: Why can TokenPack run without an OpenAI API key?
- Gold answer: The retrieval benchmark does not require a language model, and optional local generation can use an Ollama model installed on the same computer.

Answer:

```text
TokenPack can run without an OpenAI API key because it can use an Ollama model installed on the same computer for local generation.
```

Human score: [ ] 0  [ ] 1  [ ] 2
