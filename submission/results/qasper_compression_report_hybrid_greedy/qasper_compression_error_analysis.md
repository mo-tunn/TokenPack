# QASPER Compression Error Analysis

## Pairwise Win Rates

- Same-saving comparison (`Only TokenPack` vs `Only LLMLingua-2`, 200 questions): TokenPack higher evidence recall on 90.0% of questions; LLMLingua-2 higher on 8.5%.
- Higher-saving comparison (`TokenPack + LLMLingua-2` vs `Only LLMLingua-2`, 200 questions): selection-first pipeline higher evidence recall on 0.0% of questions; LLMLingua-2 higher on 86.5%.

## Largest Same-Saving Gaps

### Case 1

- Paper: `1811.00383`
- Question ID: `7e62a53823aba08bc26b2812db016f5ce6159565`
- Title: Addressing word-order Divergence in Multilingual Neural Machine Translation for extremely Low Resource Languages
- Question: Which dataset(s) do they experiment with?
- Gold answer: IITB English-Hindi parallel corpus BIBREF22 ILCI English-Hindi parallel corpus
- Evidence recall: TokenPack `1.000`, Only LLMLingua-2 `0.400`, TokenPack + LLMLingua-2 `0.343`
- Token saving: TokenPack `53.8%`, Only LLMLingua-2 `52.5%`, TokenPack + LLMLingua-2 `78.1%`
- Evidence snippets:
  - Datasets
  - For training English-Hindi NMT systems, we use the IITB English-Hindi parallel corpus BIBREF22 ( INLINEFORM0 sentences from the training set) and the ILCI English-Hindi parallel corpus ( INLINEFORM1 sentences). The ILCI (Indian Language Corpora Initiative) multilingual parallel corpus BIBREF23 spans multiple Indian languages from the health and tourism domains. We use the 520-sentence dev-set of the IITB parallel corpus for validation. For each child task, we use INLINEFORM2 sentences from ILCI corpus as the test set.

### Case 2

- Paper: `1811.00383`
- Question ID: `a313e98994fc039a82aa2447c411dda92c65a470`
- Title: Addressing word-order Divergence in Multilingual Neural Machine Translation for extremely Low Resource Languages
- Question: How do they match words before reordering them?
- Gold answer: CFILT-preorder system
- Evidence recall: TokenPack `1.000`, Only LLMLingua-2 `0.550`, TokenPack + LLMLingua-2 `0.550`
- Token saving: TokenPack `50.0%`, Only LLMLingua-2 `52.5%`, TokenPack + LLMLingua-2 `76.1%`
- Evidence snippets:
  - We use the CFILT-preorder system for reordering English sentences to match the Indian language word order. It contains two re-ordering systems: (1) generic rules that apply to all Indian languages BIBREF17 , and (2) hindi-tuned rules which improve the generic rules by incorporating improvements found through an error analysis of English-Hindi reordering BIBREF28 . These Hindi-tuned rules have been found to improve reordering for many English to Indian language pairs BIBREF29 .

### Case 3

- Paper: `2002.08795`
- Question ID: `88ab7811662157680144ed3fdd00939e36552672`
- Title: How To Avoid Being Eaten By a Grue: Exploration Strategies for Text-Adventure Agents
- Question: What are the two new strategies?
- Gold answer: a method that detects bottlenecks in text-games using the overall reward gained and the knowledge graph state to leverage knowledge graphs to improve existing exploration algorithms for dealing with combinatorial action-space
- Evidence recall: TokenPack `1.000`, Only LLMLingua-2 `0.560`, TokenPack + LLMLingua-2 `0.507`
- Token saving: TokenPack `50.5%`, Only LLMLingua-2 `51.9%`, TokenPack + LLMLingua-2 `76.6%`
- Evidence snippets:
  - More efficient exploration strategies are required to pass bottlenecks. Our contributions are two-fold. We first introduce a method that detects bottlenecks in text-games using the overall reward gained and the knowledge graph state. This method freezes the policy used to reach the bottleneck and restarts the training from there on out, additionally conducting a backtracking search to ensure that a sub-optimal policy has not been frozen. The second contribution explore how to leverage knowledge graphs to improve existing exploration algorithms for dealing with combinatorial action-spaces such as Go-Explore BIBREF9. We additionally present a comparative ablation study analyzing the performance of these methods on the popular text-game Zork1.

### Case 4

- Paper: `1908.06941`
- Question ID: `6844683935d0d8f588fa06530f5068bf3e1ed0c0`
- Title: Why So Down? The Role of Negative (and Positive) Pointwise Mutual Information in Distributional Semantics
- Question: Why are statistics from finite corpora unreliable?
- Gold answer: $\mathit {PMI}(w,c)$ goes to negative infinity when the word-context pair $(w,c)$ does not appear in the training corpus
- Evidence recall: TokenPack `1.000`, Only LLMLingua-2 `0.561`, TokenPack + LLMLingua-2 `0.561`
- Token saving: TokenPack `51.1%`, Only LLMLingua-2 `46.5%`, TokenPack + LLMLingua-2 `74.4%`
- Evidence snippets:
  - Unfortunately, $\mathit {PMI}(w,c)$ goes to negative infinity when the word-context pair $(w,c)$ does not appear in the training corpus. Due to unreliable statistics, this happens very frequently in finite corpora. Many models work around this issue by clipping negative $\mathit {PMI}$ values at 0, a measure known as Positive $\mathit {PMI}$ ($\mathit {PPMI}$), which works very well in practice. An unanswered question is: “What is lost/gained by collapsing the negative $\mathit {PMI}$ spectrum to 0?”. Understanding which type of information is captured by $\mathit {\texttt {-}PMI}$ can help in tailoring models for optimal performance.

### Case 5

- Paper: `1911.03059`
- Question ID: `f428618ca9c017e0c9c2a23515dab30a7660f65f`
- Title: A Comprehensive Comparison of Machine Learning Based Methods Used in Bengali Question Classification
- Question: what ml based approaches were compared?
- Gold answer: Multi-Layer Perceptron (MLP), Naive Bayes Classifier (NBC), Support Vector Machine (SVM), Gradient Boosting Classifier (GBC), Stochastic Gradient Descent (SGD), K Nearest Neighbour (K-NN) and Random Forest (RF)
- Evidence recall: TokenPack `1.000`, Only LLMLingua-2 `0.568`, TokenPack + LLMLingua-2 `0.514`
- Token saving: TokenPack `50.1%`, Only LLMLingua-2 `51.3%`, TokenPack + LLMLingua-2 `75.9%`
- Evidence snippets:
  - In this research, we briefly discuss the steps of QA system and compare the performance of seven machine learning based classifiers (Multi-Layer Perceptron (MLP), Naive Bayes Classifier (NBC), Support Vector Machine (SVM), Gradient Boosting Classifier (GBC), Stochastic Gradient Descent (SGD), K Nearest Neighbour (K-NN) and Random Forest (RF)) in classifying Bengali questions to classes based on their anticipated answers. Bengali questions have flexible inquiring ways, so there are many difficulties associated with Bengali QC BIBREF0. As there is no rich corpus of questions in Bengali Language available, collecting questions is an additional challenge. Different difficulties in building a QA System are mentioned in the literature BIBREF2 BIBREF3. The first work on a machine learning based approach towards Bengali question classification is presented in BIBREF0 that employ the Stochastic Gradient Descent (SGD).
