# QASPER Compression Error Analysis

## Pairwise Win Rates

- Same-saving comparison (`Only TokenPack` vs `Only LLMLingua-2`, 200 questions): TokenPack higher evidence recall on 84.0% of questions; LLMLingua-2 higher on 14.5%.
- Higher-saving comparison (`TokenPack + LLMLingua-2` vs `Only LLMLingua-2`, 200 questions): selection-first pipeline higher evidence recall on 0.0% of questions; LLMLingua-2 higher on 88.5%.

## Largest Same-Saving Gaps

### Case 1

- Paper: `2002.08795`
- Question ID: `88ab7811662157680144ed3fdd00939e36552672`
- Title: How To Avoid Being Eaten By a Grue: Exploration Strategies for Text-Adventure Agents
- Question: What are the two new strategies?
- Gold answer: a method that detects bottlenecks in text-games using the overall reward gained and the knowledge graph state to leverage knowledge graphs to improve existing exploration algorithms for dealing with combinatorial action-space
- Evidence recall: TokenPack `1.000`, Only LLMLingua-2 `0.560`, TokenPack + LLMLingua-2 `0.493`
- Token saving: TokenPack `51.1%`, Only LLMLingua-2 `51.9%`, TokenPack + LLMLingua-2 `76.6%`
- Evidence snippets:
  - More efficient exploration strategies are required to pass bottlenecks. Our contributions are two-fold. We first introduce a method that detects bottlenecks in text-games using the overall reward gained and the knowledge graph state. This method freezes the policy used to reach the bottleneck and restarts the training from there on out, additionally conducting a backtracking search to ensure that a sub-optimal policy has not been frozen. The second contribution explore how to leverage knowledge graphs to improve existing exploration algorithms for dealing with combinatorial action-spaces such as Go-Explore BIBREF9. We additionally present a comparative ablation study analyzing the performance of these methods on the popular text-game Zork1.

### Case 2

- Paper: `1908.06941`
- Question ID: `6844683935d0d8f588fa06530f5068bf3e1ed0c0`
- Title: Why So Down? The Role of Negative (and Positive) Pointwise Mutual Information in Distributional Semantics
- Question: Why are statistics from finite corpora unreliable?
- Gold answer: $\mathit {PMI}(w,c)$ goes to negative infinity when the word-context pair $(w,c)$ does not appear in the training corpus
- Evidence recall: TokenPack `1.000`, Only LLMLingua-2 `0.561`, TokenPack + LLMLingua-2 `0.526`
- Token saving: TokenPack `50.2%`, Only LLMLingua-2 `46.5%`, TokenPack + LLMLingua-2 `73.4%`
- Evidence snippets:
  - Unfortunately, $\mathit {PMI}(w,c)$ goes to negative infinity when the word-context pair $(w,c)$ does not appear in the training corpus. Due to unreliable statistics, this happens very frequently in finite corpora. Many models work around this issue by clipping negative $\mathit {PMI}$ values at 0, a measure known as Positive $\mathit {PMI}$ ($\mathit {PPMI}$), which works very well in practice. An unanswered question is: “What is lost/gained by collapsing the negative $\mathit {PMI}$ spectrum to 0?”. Understanding which type of information is captured by $\mathit {\texttt {-}PMI}$ can help in tailoring models for optimal performance.

### Case 3

- Paper: `1911.03059`
- Question ID: `f428618ca9c017e0c9c2a23515dab30a7660f65f`
- Title: A Comprehensive Comparison of Machine Learning Based Methods Used in Bengali Question Classification
- Question: what ml based approaches were compared?
- Gold answer: Multi-Layer Perceptron (MLP), Naive Bayes Classifier (NBC), Support Vector Machine (SVM), Gradient Boosting Classifier (GBC), Stochastic Gradient Descent (SGD), K Nearest Neighbour (K-NN) and Random Forest (RF)
- Evidence recall: TokenPack `1.000`, Only LLMLingua-2 `0.568`, TokenPack + LLMLingua-2 `0.527`
- Token saving: TokenPack `51.3%`, Only LLMLingua-2 `51.5%`, TokenPack + LLMLingua-2 `76.5%`
- Evidence snippets:
  - In this research, we briefly discuss the steps of QA system and compare the performance of seven machine learning based classifiers (Multi-Layer Perceptron (MLP), Naive Bayes Classifier (NBC), Support Vector Machine (SVM), Gradient Boosting Classifier (GBC), Stochastic Gradient Descent (SGD), K Nearest Neighbour (K-NN) and Random Forest (RF)) in classifying Bengali questions to classes based on their anticipated answers. Bengali questions have flexible inquiring ways, so there are many difficulties associated with Bengali QC BIBREF0. As there is no rich corpus of questions in Bengali Language available, collecting questions is an additional challenge. Different difficulties in building a QA System are mentioned in the literature BIBREF2 BIBREF3. The first work on a machine learning based approach towards Bengali question classification is presented in BIBREF0 that employ the Stochastic Gradient Descent (SGD).

### Case 4

- Paper: `1909.07575`
- Question ID: `022c365a14fdec406c7a945a1a18e7e79df37f08`
- Title: Bridging the Gap between Pre-Training and Fine-Tuning for End-to-End Speech Translation
- Question: What is the attention module pretrained on?
- Gold answer: the model is pre-trained on CTC-based ASR task and MT task in the pre-training stage.
- Evidence recall: TokenPack `1.000`, Only LLMLingua-2 `0.571`, TokenPack + LLMLingua-2 `0.571`
- Token saving: TokenPack `50.1%`, Only LLMLingua-2 `46.2%`, TokenPack + LLMLingua-2 `73.8%`
- Evidence snippets:
  - To sufficiently utilize the large dataset $\mathcal {A}$ and $\mathcal {M}$, the model is pre-trained on CTC-based ASR task and MT task in the pre-training stage.

### Case 5

- Paper: `1805.08241`
- Question ID: `14b74ad5a6f5b0506511c9b454e9c464371ef8c4`
- Title: Sparse and Constrained Attention for Neural Machine Translation
- Question: What are the language pairs explored in this paper?
- Gold answer: De-En Ja-En Ro-En
- Evidence recall: TokenPack `1.000`, Only LLMLingua-2 `0.580`, TokenPack + LLMLingua-2 `0.580`
- Token saving: TokenPack `56.9%`, Only LLMLingua-2 `51.4%`, TokenPack + LLMLingua-2 `79.2%`
- Evidence snippets:
  - We evaluated our attention transformations on three language pairs. We focused on small datasets, as they are the most affected by coverage mistakes. We use the IWSLT 2014 corpus for De-En, the KFTT corpus for Ja-En BIBREF19 , and the WMT 2016 dataset for Ro-En. The training sets have 153,326, 329,882, and 560,767 parallel sentences, respectively. Our reason to prefer smaller datasets is that this regime is what brings more adequacy issues and demands more structural biases, hence it is a good test bed for our methods. We tokenized the data using the Moses scripts and preprocessed it with subword units BIBREF20 with a joint vocabulary and 32k merge operations. Our implementation was done on a fork of the OpenNMT-py toolkit BIBREF21 with the default parameters . We used a validation set to tune hyperparameters introduced by our model. Even though our attention implementations are CPU-based using NumPy (unlike the rest of the computation which is done on the GPU), we did not observe any noticeable slowdown using multiple devices.
