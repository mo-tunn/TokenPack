# Gold Evidence Review Packet

Use this file to check whether each candidate query and answer is supported by the listed RAG evidence chunk.

Review decision guide:

- Accept if the answer is directly supported by the evidence text.
- Edit if the evidence is useful but the query or answer is awkward.
- Reject if the query is meaningless or the evidence does not support the answer.

```text
========================================================================================
Record 1/5
----------------------------------------------------------------------------------------
QUERY: text embedding late chunking retrieval embeddings

ANSWER: arXiv:2409.04701v3 [cs.CL] 7 Jul 2025 LATE CHUNKING : C ONTEXTUAL CHUNK EMBED - DINGS USING LONG -C ONTEXT EMBEDDING MODELS Michael G ¨unther1, Isabelle Mohr 1, Daniel James Williams 2, Bo Wang1, Han Xiao 1 1Jina AI GmbH, Prinzessinnenstr.

RAG EVIDENCE DATA USED BY THIS RECORD:
- chunk_id: chunk-5020842e6335
  source_path: resources\late-chunking-contextual-chunk-embeddings-using-long-context-embedding-models.pdf
  pages: 1 - 1
  token_count: 500
  text:
  arXiv:2409.04701v3 [cs.CL] 7 Jul 2025 LATE CHUNKING : C ONTEXTUAL CHUNK EMBED - DINGS USING LONG
  -C ONTEXT EMBEDDING MODELS Michael G ¨unther1, Isabelle Mohr 1, Daniel James Williams 2, Bo
  Wang1, Han Xiao 1 1Jina AI GmbH, Prinzessinnenstr. 19-20, 10969 Berlin, Germany research@jina.ai
  2Weaviate B.V ., Prinsengracht 769a, 1017JZ Amsterdam danny@weaviate.io ABSTRACT Many use cases
  require retrieving smaller portions of text, and dense vector-based retrieval systems often
  perform better with shorter text segments, as the semantics are less likely to be “over-
  compressed” in the embeddings. Consequently, practi- tioners often split text documents into
  smaller chunks and encode them separately. However, chunk embeddings created in this way can
  lose contextual information from surrounding chunks, resulting in sub-optimal representations.
  In this paper, we introduce a novel method called “late chunking”, which leverages long con-
  text embedding models to first embed all tokens of the long text, with chunking applied after
  the transformer model and just before mean pooling - hence the term “late” in its naming. The
  resulting chunk embeddings capture the full contextual information, leading to superior results
  across various retrieval tasks. The method is generic enough to be applied to a wide range of
  long-context embedding models and works without additional training. To further increase the
  effectiveness of late chunking, we propose a dedicated fine-tuning approach for embedding
  models. 1 I NTRODUCTION Neural information retrieval (IR) relies on text embedding models
  (Reimers & Gurevych, 2019) that are primarily based on the transformer architecture (Devlin et
  al., 2019) and have been pre-trained using very large text corpora. These models capture
  important elements of texts’ ...

```

Decision: [ ] Accept  [ ] Edit  [ ] Reject

```text
========================================================================================
Record 2/5
----------------------------------------------------------------------------------------
QUERY: chunking training text long embedding late

ANSWER: • Extended Algorithm for Long Documents: For encoding long documents with more tokens than long-context embedding models can handle, we propose a long late chunking approach (see Section 3.1) and prove its effectiveness in Section 4.3.

RAG EVIDENCE DATA USED BY THIS RECORD:
- chunk_id: chunk-2a5f2c6fab16
  source_path: resources\late-chunking-contextual-chunk-embeddings-using-long-context-embedding-models.pdf
  pages: 3 - 3
  token_count: 502
  text:
  • Extended Algorithm for Long Documents: For encoding long documents with more tokens than long-
  context embedding models can handle, we propose a long late chunking approach (see Section 3.1)
  and prove its effectiveness in Section 4.3. • Training for Late Chunking: While late chunking
  does not require additional training, we propose a novel training method to further enhance
  retrieval accuracy when using it (see Section 3.2). We conduct an evaluation to show its
  advantage over comparable contrastive training in Section 4.4. • Comprehensive Evaluation: We
  conduct a comprehensive empirical evaluation to identify scenarios where late chunking performs
  superior to naive chunking and scenarios where the standard method yields comparable or superior
  results (see Sections 4.1 and 4.2). 2 R ELATED WORK Most modern text embedding models are
  trained on transformer-based architectures (Devlin et al., 2019) using the training method
  proposed by Reimers & Gurevych (2019). In general, the model is equipped with a pooling operator
  which converts the token embeddings produced by the transformer into a single vector
  representation. Mean pooling is especially popular, as Reimers & Gurevych (2019) conduct
  experiments in which mean pooling shows the best performance among other meth- ods. While the
  original transformer uses absolute positional encodings, methods that encode relative positions
  like AliBi (Press et al., 2022) and RoPE (Su et al., 2024) allow effective training of em-
  bedding models with larger context lengths (G¨unther et al., 2023; Nussbaum et al., 2024). To
  address the limited context length and overcome practical issues of handling embeddings of long
  texts, chunking text before embedding it has become common practice. While simple chunking
  methods use a fixed token length ...

```

Decision: [ ] Accept  [ ] Edit  [ ] Reject

```text
========================================================================================
Record 3/5
----------------------------------------------------------------------------------------
QUERY: training text embeddings span pooling token

ANSWER: 3.2 T RAINING METHOD While late chunking works without further training, models that are trained with mean pooling to create a single embedding representation of a longer text might not be well-suited to encode chunks of token embeddings containing additional information from surrounding tokens.

RAG EVIDENCE DATA USED BY THIS RECORD:
- chunk_id: chunk-0440ee5220ec
  source_path: resources\late-chunking-contextual-chunk-embeddings-using-long-context-embedding-models.pdf
  pages: 6 - 6
  token_count: 501
  text:
  3.2 T RAINING METHOD While late chunking works without further training, models that are trained
  with mean pooling to create a single embedding representation of a longer text might not be
  well-suited to encode chunks of token embeddings containing additional information from
  surrounding tokens. Therefore, we propose a modified text embedding training method, which uses
  a technique that we call “span pooling” to train the model to encode specifically the relevant
  information contained in an annotated text span into its token embeddings. Training Data: To
  conduct the training, we prepare training data which consist of tuples (q, d, ⟨start, end ⟩) of
  two text values: a query q and a relevant document d, with additional an- notation of the
  relevant span in the document ⟨start, end ⟩ that contains the answer. Training Process: The
  fine-tuning procedure itself follows the pair training stage described in G¨unther et al.
  (2023), where the model is trained on text pairs using the InfoNCE loss function (van den Oord
  et al., 2018) which is defined on a batch B = ((x1, y1), . . . ,(xk, yk)) of k pairs and the
  cosine similarity function s: LNCE(B) := − X (xi,yi)∈B ln es(xi,yi)/τ kP i′=1 es(xi,yi′ )/τ (1)
  Here, the query vectors xi are obtained by applying the embedding model to the query text qi in
  the usual way. For the document embeddings yi, the set of token embeddings ϑi,1, . . . , ϑi,n is
  obtained by applying the model on the documents di, and executing the mean pooling operation
  only to the token embeddings within the span ⟨start, end ⟩, hence the term “span pooling”. As
  proposed by G ¨unther et al. (2023), we use a bi-directional version of the loss Lpairs, where
  B† = ((y1, x1), . . . ,(yk, xk)) is obtained from B by swapping the order of pairs: Lpairs(B) :=
  LNCE(B) + LNCE( ...

```

Decision: [ ] Accept  [ ] Edit  [ ] Reject

```text
========================================================================================
Record 4/5
----------------------------------------------------------------------------------------
QUERY: chunking triviaqa training late long pooling

ANSWER: Table 3: Evaluation results (nDCG@10 [%]) on chunked evaluation tasks when training with span pooling and mean pooling, with a fixed chunk size of 64 tokens and late chunking during inference.

RAG EVIDENCE DATA USED BY THIS RECORD:
- chunk_id: chunk-7a4d6c89fde9
  source_path: resources\late-chunking-contextual-chunk-embeddings-using-long-context-embedding-models.pdf
  pages: 9 - 9
  token_count: 501
  text:
  Table 3: Evaluation results (nDCG@10 [%]) on chunked evaluation tasks when training with span
  pooling and mean pooling, with a fixed chunk size of 64 tokens and late chunking during
  inference. Model Pooling (During Training Data Sci- Narrative- NF- TREC FiQA Training) Fact QA
  Corpus -COV J3 Span-Based TriviaQA&FEVER 72.61 44.01 36.80 77.59 48.22 TriviaQA 72.28 44.94
  36.69 77.39 47.99 J3 Mean TriviaQA&FEVER 72.59 43.83 36.77 77.21 47.40 TriviaQA 72.56 44.86
  36.78 77.36 47.35 J2s Span-Based TriviaQA&FEVER 65.20 47.29 29.96 65.18 34.52 TriviaQA 65.43
  47.76 30.04 64.95 34.29 J2s Mean TriviaQA&FEVER 64.77 47.31 29.70 64.73 33.87 TriviaQA 65.18
  47.45 29.76 64.86 33.82 4.3 E VALUATION OF LONG LATE CHUNKING To evaluate long late chunking, we
  select three of the non-synthetic reading comprehension datasets, as none of the BeIR datasets
  contain a significant amount of text values with more than 8192 tokens. We use the same
  evaluation method as described in Section 4.2 but do not truncate this time. Figure 4 shows that
  late chunking with the long late chunking method achieves superior results in comparison to
  naive chunking. Compared to the experiment of Section 4.2, the nDCG scores are higher, as
  truncation in the last experiment could lead to information loss. Long late chunking solves this
  problem. 4.4 E VALUATION OF TRAINING METHOD Table 3 captures the results from our training
  experiments. The experiments include running both span-based and regular mean pooling training
  methods on the jina-embeddings-v3 and jina-embeddings-v2-small-en long context embedding models
  in order to see whether the proposed training method achieves performance gains in combination
  with late chunking.

```

Decision: [ ] Accept  [ ] Edit  [ ] Reject

```text
========================================================================================
Record 5/5
----------------------------------------------------------------------------------------
QUERY: arxiv association computational linguistics http xiao

ANSWER: 2023.

RAG EVIDENCE DATA USED BY THIS RECORD:
- chunk_id: chunk-cb8b25d113f2
  source_path: resources\late-chunking-contextual-chunk-embeddings-using-long-context-embedding-models.pdf
  pages: 11 - 11
  token_count: 503
  text:
  2023. URL http://arxiv.org/abs/2310.19923. Rohan Jha, Bo Wang, Michael G ¨unther, Saba Sturua,
  Mohammad Kalim Akram, and Han Xiao. Jina-ColBERT-v2: A General-Purpose Multilingual Late
  Interaction Retriever. arXiv preprint arXiv:2408.16672, 2024. URL
  http://arxiv.org/abs/2408.16672. Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer.
  TriviaQA: A Large Scale Dis- tantly Supervised Challenge Dataset for Reading Comprehension. In
  Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume
  1: Long Papers) , pp. 1601–1611, 2017. Greg Kamradt. 5 Levels of Text Splitting.
  https://github.com/ FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/
  LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb, 2024. Ac- cessed: 2024-09-06. Omar
  Khattab and Matei Zaharia. ColBERT: Efficient and Effective Passage Search via Contextual- ized
  Late Interaction over BERT. InProceedings of the 43rd International ACM SIGIR conference on
  research and development in Information Retrieval, pp. 39–48, 2020. Patrick Lewis, Ethan Perez,
  Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich K¨uttler, Mike
  Lewis, Wen-tau Yih, Tim Rockt¨aschel, et al. Retrieval-Augmented Gener- ation for Knowledge-
  Intensive NLP Tasks. Advances in Neural Information Processing Systems, 33:9459–9474, 2020. Kun
  Luo, Zheng Liu, Shitao Xiao, Tong Zhou, Yubo Chen, Jun Zhao, and Kang Liu. Landmark embedding: A
  chunking-free embedding method for retrieval augmented long-context large lan- guage models. In
  Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume
  1: Long Papers), pp. 3268–3281. Association for Computational Linguistics, 2024. Zach Nussbaum,

```

Decision: [ ] Accept  [ ] Edit  [ ] Reject
