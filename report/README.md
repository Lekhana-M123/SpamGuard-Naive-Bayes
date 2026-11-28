
# B4 - Mathematical and Probabilistic Analysis of an AI System

## Project Context

**University:** Kyushu University - Laboratory Selection Process  
**Requirement:** [B4] Choose an artificial intelligence system to discuss/analyze from a mathematical/probabilistic point of view.  
**Selected system:** Multinomial Naive Bayes (MNB) for SMS spam detection  
**Dataset:** UCI SMS Spam Collection

This project analyzes the model's probabilistic structure and behavior. The sections below distinguish the implemented pipeline and measured results from experiments that remain planned.

## Quick Start

The project environment was validated on Windows with Python 3.14 and the versions
in `requirements.txt`.

1. Open `E:\B4_Project` in VS Code.
2. In the integrated PowerShell terminal, create the environment, install dependencies
   and launch the app:

   ```powershell
   cd E:\B4_Project
   py -3.14 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

3. If prompted, select `.venv\Scripts\python.exe` using **Python: Select Interpreter**
   in VS Code. Streamlit prints the local URL after startup.

The application loads the saved model, vectorizer and configuration from `models/`,
and its canonical evaluation files from `results/`. Launching the app does not train
the model or regenerate evaluation results.

## Repository Layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `models/` | Saved Multinomial Naive Bayes model, vectorizer and configuration |
| `dataset/raw/` | Original UCI SMS Spam Collection |
| `dataset/processed/` | Group-aware training, validation and independent-test splits |
| `src/` | Data preparation, validation analysis and canonical evaluation scripts |
| `experiments/` | Hand calculations and illustrative mathematical examples |
| `results/` | Validation and final independent-test CSV outputs |
| `graphs/` | Experiment figures and the final-test confusion matrix |
| `report/README.md` | This research report |
| `requirements.txt` | Pinned direct Python dependencies |

## Contents

- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Implemented pipeline and final results](#implemented-pipeline-and-evidence-status)
- [Completed and planned work](#completed-and-planned-work)
- [Research problem and objectives](#1-research-problem-and-objectives)
- [Mathematical foundations](#2-mathematical-foundations-bayes-theorem-and-mnb)
- [Proposed experimental design](#8-proposed-experimental-design-and-variables)
- [Completed experiment record](#completed-experiment-record-and-initial-dataset-inspection)
- [Reproducibility limitations](#reproducibility-record)

## Implemented Pipeline and Evidence Status

- **Primary classifier:** `MultinomialNB(alpha=0.1)`.
- **Final application settings:** alpha = `0.1`; classify as spam when the saved model's `predict_proba()` spam probability is at least `0.3`, otherwise classify as ham. This is the retained precision/recall operating point, not the validation maximum-F1 threshold.
- **Partitions:** 5,574 original records are split with `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, grouping identical message text. One fold is held out as the independent test set (1,115 records). The remaining 4,459 records are split again by grouped stratified 5-fold into 3,567 model-training records and 892 validation records.
- **Duplicate handling:** repeated records are retained. The grouped splits keep identical message text from crossing train/validation/test boundaries; the split checks found no identical-message overlap between the partitions.
- **Final fit:** after validation-based selection, the final vectorizer and model are fitted on all 4,459 training-plus-validation records. The independent test set is transformed and evaluated only after settings are fixed.
- **Vectorization:** code uses `CountVectorizer()` defaults: lowercase conversion enabled, token pattern `(?u)\b\w\w+\b` (so one-character tokens are omitted), no accent stripping, and no stop-word filtering. No explicit Unicode normalization is applied.
- **Saved final model:** vocabulary size 7,673; the saved model, vectorizer and configuration agree on alpha, threshold and feature count.
- **Canonical independent-test workflow and outputs:** `src/final_test_evaluation.py` fits the final model on the combined training and validation partitions, then evaluates the untouched test partition at threshold `0.3`. Its canonical outputs are `results/final_test_metrics.csv`, `results/final_test_predictions.csv` and `graphs/final_test_confusion_matrix.png`. The Streamlit application checks the prediction labels against the saved threshold and recomputes/checks the metrics before displaying them.
- **Separate validation thresholds:** `src/alpha_comparison.py` and `src/prior_comparison.py` use `model.predict()` (the default 0.5 decision boundary) for their classification metrics. `src/statistical_analysis.py` and `src/error_analysis.py` explicitly use threshold 0.5. These are validation analyses, not final-model test results. `src/threshold_analysis.py` is a validation threshold sweep; `src/calibration_analysis.py` evaluates probabilities without applying a classification threshold.
- **Historical alpha-1 baseline:** `results/model_metrics.csv` and `results/predictions.csv` were generated by the older `src/train_evaluate.py` workflow with `alpha=1.0` and `model.predict()`'s default decision rule. They are not final-model results and must not be used as independent-test tuning evidence.
- **Full-dataset prior illustration:** `experiments/prior_analysis.py` fits `alpha=1.0` models on the whole collection, including the reserved test records. Its printed example probabilities and the visualization from `experiments/prior_visualization.py` are illustrative only; they are excluded from predictive-performance claims and model selection. The illustration has no CSV output.

### Final Independent-Test Results

| Metric | Result |
|---|---:|
| Test records | 1,115 |
| Accuracy | 98.39% |
| Spam precision | 92.31% |
| Spam recall | 96.00% |
| Spam F1-score | 94.12% |
| ROC-AUC | 98.54% |
| PR-AUC (average precision) | 97.58% |

Confusion matrix (rows are actual; columns are predicted):

|  | Predicted ham | Predicted spam |
|---|---:|---:|
| Actual ham | 953 | 12 |
| Actual spam | 6 | 144 |

### Completed and Planned Work

| Status | Work |
|---|---|
| Completed | Dataset and duplicate inspection; grouped train/validation/test splitting; hand-worked classification and smoothing arithmetic; the eight-message joint-probability illustration; validation alpha, prior and threshold comparisons; validation calibration, error and message-group bootstrap analyses; training-only feature analysis; canonical independent-test evaluation. Evidence and limits are identified in the experiment record below. |
| Planned or incomplete | Learning curves; repeated stratified cross-validation; a controlled independent-versus-dependent synthetic experiment measuring classifier performance/calibration; and an automated comparison of the hand calculation with scikit-learn's implementation. No results are claimed for these items. |

The eight-message independence example compares an observed joint probability with
the product of marginals only. It does not fit or evaluate a classifier.

## 1. Research Problem and Objectives

SMS spam detection is a binary classification problem: given a message, infer whether its label is `spam` or `ham` (non-spam). MNB represents a message as word counts and applies Bayes' theorem with class-conditional multinomial likelihoods. Its compact mathematical form makes it possible to examine how class prevalence, token frequencies, the conditional-independence model, and smoothing jointly affect predictions.

The objectives below describe the research scope; they are not a claim that every
proposed experiment has been completed. See [Completed and Planned Work](#completed-and-planned-work)
and the experiment record for the evidence available in this project.

**Research problem:** Under what data and modeling conditions does MNB assign a message to spam or ham, and how do its probability assumptions and estimation choices affect those assignments and the quality of its probability estimates?

**Objectives:**

1. Derive the MNB posterior decision rule from Bayes' theorem and the multinomial likelihood.
2. Show how empirical class priors and class-conditional token probabilities are estimated from training data.
3. Explain the bag-of-words conditional-independence assumption and identify its limits for natural language.
4. Derive additive (Laplace/Lidstone) smoothing and demonstrate the zero-frequency problem it addresses.
5. Test, with controlled experiments, how priors, smoothing, vocabulary, training-set size, and dependence between tokens change scores, decisions, and evaluation metrics.
6. Evaluate both classification decisions and probability estimates without using test data to choose model settings.

## 2. Mathematical Foundations: Bayes' Theorem and MNB

Let $C \in \{\text{ham},\text{spam}\}$ be the class and let $x=(x_1,\ldots,x_V)$ be the message's bag-of-words count vector over a vocabulary of size $V$. Here, $x_j$ is the number of occurrences of vocabulary token $j$, and $L=\sum_{j=1}^{V}x_j$ is the message length in tokens.

Bayes' theorem gives

$$
P(C=c\mid x)=\frac{P(x\mid C=c)P(C=c)}{P(x)},
\qquad
P(x)=\sum_{c'\in\{\text{ham},\text{spam}\}}P(x\mid C=c')P(C=c').
$$

The posterior is the probability of a class after observing the message. The prior $P(C=c)$ represents class prevalence before observing its words; the likelihood $P(x\mid C=c)$ represents how probable the message is under that class. For classification, the denominator is identical for both classes, so the maximum-posterior rule is

$$
\hat c(x)=\arg\max_c P(C=c)P(x\mid C=c).
$$

MNB models the token counts as a multinomial distribution conditional on the class and message length. Let $n_{cj}$ be the total number of occurrences of token $j$ in training messages of class $c$, and $N_c=\sum_{j=1}^{V}n_{cj}$ the total number of in-vocabulary token occurrences in that class. With additive smoothing parameter $\alpha\geq 0$,

$$
\hat\theta_{cj}=P(\text{token }j\mid c)
=\frac{n_{cj}+\alpha}{N_c+\alpha V}.
$$

The class-conditional multinomial likelihood is

$$
P(x\mid c)=\frac{L!}{\prod_{j=1}^{V}x_j!}\prod_{j=1}^{V}\hat\theta_{cj}^{x_j}.
$$

The multinomial coefficient is the same for every candidate class for a given message, so it cancels in the classification comparison. Computing in log space avoids numerical underflow:

$$
\log P(c\mid x)=\text{constant}(x)+\log P(c)
+\sum_{j=1}^{V}x_j\log\hat\theta_{cj}.
$$

The constant includes the log evidence and, for the multinomial likelihood, the message's multinomial coefficient. For a two-class decision, the spam-versus-ham log-score difference can be written as

$$
\log\frac{P(\text{spam}\mid x)}{P(\text{ham}\mid x)}
=\log\frac{P(\text{spam})}{P(\text{ham})}
+\sum_{j=1}^{V}x_j\log\frac{\hat\theta_{\text{spam},j}}{\hat\theta_{\text{ham},j}}.
$$

This decomposition separates the prior log-odds from the sum of token evidence and provides a direct way to inspect why a score changes.

## 3. Prior Probability Calculations

For a training set of $M$ messages, let $M_c$ be the number labeled class $c$. The maximum-likelihood empirical prior is

$$
\hat P(C=c)=\frac{M_c}{M},
\qquad M=M_{\text{ham}}+M_{\text{spam}}.
$$

Calculate these counts from the training partition only. Report the counts and proportions, not just the resulting predictions. Do not assume the training prevalence equals the prevalence in a future deployment population.

Use empirical priors as the primary model setting. As a controlled comparison, also evaluate equal priors, $P(\text{ham})=P(\text{spam})=0.5$, while holding token likelihoods fixed. This isolates the prior contribution: changing priors adds a constant to the class log-score difference, while leaving each class's token estimates unchanged. Any prior-adjusted analysis must state that equal priors are an experimental condition, not an estimate of the dataset's prevalence.

## 4. Likelihood Estimation

For each training partition, count each vocabulary token's occurrences separately within ham and spam messages. Estimate each class's token probability using the smoothed relative-frequency formula in Section 2. The denominator is the total in-vocabulary token count for that class plus $\alpha V$; it is not the number of messages in that class.

The standard MNB likelihood uses token multiplicities: a token appearing three times contributes $3\log\hat\theta_{cj}$ to the class log-score. A diagnostic comparison can replace counts by binary token presence, but that changes the feature representation and is not the primary multinomial model. State the representation whenever comparing results.

Fit the vocabulary and all preprocessing from the training partition of each split only. At evaluation time, tokens absent from that fitted vocabulary are out of vocabulary (OOV) and are ignored by the primary bag-of-words pipeline. This is distinct from a vocabulary token that is known to the model but has zero count in one class; the latter is the zero-frequency case addressed by smoothing. For an explicit OOV experiment, use a predefined vocabulary with an `<UNK>` bucket and report that as a separate modeling choice.

## 5. Conditional Independence Assumption

The generative interpretation is that, conditional on class $c$, the $L$ token positions are draws from a categorical distribution with probabilities $\theta_{c1},\ldots,\theta_{cV}$. The token draws are modeled as conditionally independent and identically distributed given the class. Aggregating the draws into counts produces the multinomial likelihood.

This is not a claim that words are independent in actual language. Word order, phrases, negation, and co-occurring terms create dependencies that the bag-of-words model does not represent directly. Repeated occurrences affect the count likelihood, but sequence order does not. Consequently, the model can double-count correlated evidence or assign extreme scores when several dependent tokens appear together. The experiments should treat this as a model assumption to investigate, not as a fact established by good classification metrics.

## 6. Laplace Smoothing and the Zero-Frequency Problem

Without smoothing ($\alpha=0$), a token with $n_{cj}=0$ has estimated probability zero in class $c$. If that token occurs in a test message, the product likelihood for that class becomes zero, regardless of the other tokens. In log space, the corresponding term is $\log 0$, which is undefined (or represented as negative infinity). A single absent token can therefore dominate the comparison.

Additive smoothing assigns $\alpha$ pseudo-counts to every vocabulary token:

$$
\hat\theta_{cj}=\frac{n_{cj}+\alpha}{N_c+\alpha V}.
$$

For $\alpha>0$, every in-vocabulary token has positive estimated probability. $\alpha=1$ is Laplace smoothing; other positive values are often called Lidstone smoothing. As $\alpha$ grows, token probabilities move toward the uniform distribution over the fitted vocabulary. Smoothing changes both numerator and denominator, so its effect is not simply to add the same amount of evidence to every class score.

Demonstrate the zero-frequency failure using a small, fully specified toy training corpus and a test message containing a token observed in one class but not the other. Compare $\alpha=0$ with one or more positive values, show the class token counts and estimated probabilities, and calculate the resulting log-scores by hand. Keep this controlled demonstration separate from the SMS test results. A token never present in the fitted vocabulary is an OOV token, not a zero-count in-vocabulary feature; do not conflate these cases.

## 7. Experimental Questions and Hypotheses

The following are preregisterable questions. Hypotheses describe expected directions or behaviors to test; they are not reported findings.

| ID | Experimental question | Hypothesis to test |
|---|---|---|
| Q1 | How does the class prior affect classification when token likelihoods are fixed? | Equal and empirical priors can produce different decisions for messages near the decision boundary; the score difference shifts by the prior log-odds. |
| Q2 | How does $\alpha$ affect token probabilities, class scores, and predictive performance? | Positive $\alpha$ prevents zero in-vocabulary likelihoods; predictive metrics may vary with $\alpha$, and no value is assumed to be best in advance. |
| Q3 | How does training-set size affect estimates and evaluation? | Larger training samples are expected to make frequency estimates less variable, though measured performance need not improve monotonically in every split. |
| Q4 | What is lost by the conditional-independence and bag-of-words assumptions? | In a controlled dependent-token corpus, MNB may become overconfident or make errors because it treats correlated token evidence as independent; the size and direction of the effect are empirical. |
| Q5 | How do class imbalance and the decision threshold affect spam detection? | Empirical priors and threshold choice can change the precision-recall trade-off; their effects should be measured rather than inferred from accuracy alone. |
| Q6 | Are posterior scores reliable as probabilities, not just useful for ranking? | Discrimination and calibration can differ; calibration metrics and reliability plots may reveal overconfident probabilities even when classification metrics are acceptable. |

## 8. Proposed Experimental Design and Variables

This section describes proposed methodological extensions, not a second description
of the implemented application pipeline. The actual app uses the grouped partitions
and `CountVectorizer()` defaults documented above.

### Proposed data and preprocessing checks

For future experiments, record the exact dataset source, retrieval date, file checksum,
row and label counts, and any malformed or duplicate records. The existing app pipeline
uses `CountVectorizer()` defaults: lowercase conversion, the default token pattern,
no accent stripping and no stop-word filtering; it does not explicitly normalize
Unicode. Any alternative normalization, tokenization or deduplication policy should
be labeled as a separate experiment. Do not use test-set vocabulary, labels or
statistics during fitting.

### Split and fitting protocol

The implemented evaluation uses the group-aware nested split described above. The
following are recommendations for additional studies, not completed analyses:

1. Retain group separation if adding another fixed holdout; record the split rule and seed.
2. For estimates less dependent on one split, use repeated stratified 5-fold
   cross-validation. Fit tokenization/vocabulary and all model estimates inside each
   training fold; transform its validation fold using that fitted pipeline. Repeated
   cross-validation has not been run in this project.
3. If selecting $\alpha$ or another setting, select it using training data only (nested cross-validation for an unbiased cross-validated estimate, or a separate validation split). Do not choose settings by inspecting the final test labels or scores.
4. Keep the same folds and preprocessing when comparing priors, smoothing values, or feature representations so comparisons are paired.
5. Record random seeds, software/library versions, split indices, and all parameter settings. Report the fixed holdout as an illustrative result and cross-validation as the stability analysis; do not treat overlapping folds as independent datasets.

### Independent, dependent, and controlled variables

| Role | Variables |
|---|---|
| Independent (varied deliberately) | Prior rule (empirical/equal); smoothing $\alpha$ (including $\alpha=0$ only for a controlled zero-frequency demonstration); training fraction; count versus binary representation in a labeled diagnostic; decision threshold; token-dependence strength in a synthetic experiment. |
| Dependent (measured) | Predicted class, spam posterior/log-score, zero-likelihood or non-finite-score cases, confusion-matrix counts, precision, recall, F1, PR-AUC, log loss, Brier score, and calibration summaries. |
| Controlled | Dataset and label mapping; split/fold indices; tokenizer and normalization; fitted vocabulary policy; model implementation; class-positive convention; evaluation procedure; random seeds. |

### Proposed reproducible experiments

**A. Hand-worked Bayesian example.** Specify a tiny training corpus with explicit labels and token counts. Calculate class priors, smoothed token probabilities, class log-scores, posterior normalization, and the final decision by hand. Use the same counts to verify that the implementation follows the derivation. This is a mathematical check, not an estimate of SMS performance.

**B. Prior isolation.** Fit one vocabulary and one set of class likelihoods, then score the same held-out messages using empirical and equal priors. Show the prior log-odds and per-token log-likelihood-ratio contributions so any changed decision can be traced to its source.

**C. Zero-frequency and smoothing sweep.** First use the controlled toy corpus to demonstrate the $\alpha=0$ failure and finite scores for $\alpha>0$. Then run a prespecified positive-$\alpha$ sweep on the SMS folds (for example, $0.01, 0.1, 0.5, 1, 2, 10$). Plot token probabilities and evaluation metrics against $\alpha$. Do not interpret an $\alpha=0$ run with undefined scores as an ordinary performance comparison.

**D. Controlled independence experiment.** Generate a labeled synthetic bag-of-words dataset with known class priors and known token probabilities. Compare an independent-token generator with a generator in which a token pair is correlated conditional on class while keeping the marginal token frequencies as similar as possible. Fit the same MNB model in both settings; compare predicted scores, error rates, and calibration. Publish the generator specification and seed. This isolates assumption mismatch more clearly than attributing any real-SMS error to dependence alone.

**E. Learning curve.** Train on increasing, stratified fractions of the training data, using identical held-out evaluation messages or matched cross-validation folds. Plot training size against the selected metrics and, where practical, the variation in token-probability estimates across resamples. This tests whether estimates stabilize with more data.

**F. Imbalance and threshold analysis.** Compare empirical and equal priors while holding likelihoods fixed, then vary the spam decision threshold on validation predictions. Report the resulting spam precision and recall and the threshold. Do not tune a threshold on the final test set.

**G. Representation and vocabulary sensitivity (optional).** Compare count features with binary presence features, or compare a small number of documented tokenization policies, one factor at a time. Treat these as model/representation changes rather than evidence about smoothing alone.

## 9. Statistical Evaluation Methods

The guidance below describes sound reporting practice; it does not imply that every
listed metric or inferential procedure was run. Completed metrics are reported with
their partitions in the experiment record. Treat `spam` as the positive class and
report the confusion matrix with its label order. Report:

- **Accuracy:** overall proportion classified correctly; interpret alongside class balance.
- **Spam precision:** among messages predicted spam, the proportion that are spam.
- **Spam recall:** among actual spam messages, the proportion detected.
- **Spam F1 and macro-F1:** balance precision/recall for spam, and summarize performance across both classes, respectively.
- **Precision-recall AUC:** useful for examining ranking of the positive class when class prevalence is unequal; state the implementation and averaging convention.
- **Log loss and Brier score:** evaluate predicted probabilities, not just thresholded labels. State any clipping convention required for finite log loss.
- **Calibration plot or calibration error summary:** compare predicted spam-probability bins with observed spam frequency; include binning details and avoid overinterpreting sparse bins.

For cross-validation, report the fold/repetition distribution (mean and standard deviation, and preferably an interval) and retain paired predictions for model comparisons. Because repeated folds overlap in their training data, do not use a naive independent-samples test on fold scores. For a single held-out set, use paired message-level procedures: McNemar's test for two classifiers' paired correctness, or a paired bootstrap over test messages for metric differences and confidence intervals. If many settings are compared inferentially, identify primary comparisons in advance and control multiplicity (for example, Holm adjustment). Report effect sizes and intervals alongside p-values; statistical significance does not establish practical importance.

The final test set is for final evaluation only. Any selection among smoothing values, priors beyond the prespecified primary setting, preprocessing policies, or thresholds belongs in training/validation data. Report uncertainty and class counts so that small numbers of spam examples are visible.

## 10. Planned Additional Visualizations and Analysis

The figures below are proposed additions or extensions; no values or trends are
asserted in advance. Existing plots are identified with their completed experiments
later in this document.

1. **Class counts and priors:** bar chart of training ham/spam counts and empirical proportions, optionally marking the equal-prior experimental condition.
2. **Token probability comparison:** selected tokens' estimated $\hat\theta_{cj}$ for ham and spam, with the smoothing value and training fold stated. Prefer a principled token-selection rule rather than choosing examples after seeing the test outcomes.
3. **Score decomposition:** for a few correctly and incorrectly classified held-out messages, show prior log-odds and token-level log-likelihood-ratio contributions. Redact or paraphrase messages if needed for privacy in any later presentation.
4. **Smoothing curves:** plot $\alpha$ against spam recall, precision, F1, log loss, and Brier score, with cross-validation variability or confidence intervals. Discuss classification and calibration separately.
5. **Confusion matrices and precision-recall curves:** compare only prespecified settings on the same evaluation folds; label axes and the positive class clearly.
6. **Calibration reliability diagram:** plot predicted spam-probability bins against observed frequency, with bin counts, plus the Brier score.
7. **Learning curve:** show training-set size against evaluation performance and uncertainty across matched splits.
8. **Dependence demonstration:** in the synthetic experiment, contrast the designed joint token probability with the product of its marginals, and show how MNB scores change as conditional dependence is introduced.

For every figure, state the data partition, preprocessing, model settings, number of messages, and uncertainty method. The analysis should connect observed changes back to the prior term, token likelihoods, smoothing formula, or independence assumption, and distinguish those mechanistic explanations from empirical associations.

## Mathematical Derivations Versus Empirical Evidence

**Derived from the model:** Bayes' rule; the empirical prior estimator; the multinomial likelihood and its log-score; cancellation of class-independent terms; the smoothed token estimator; and the zero-probability consequence when an in-vocabulary token has zero class count and $\alpha=0$.

**Requires experiments:** predictive performance, the comparison of priors and smoothing, threshold trade-offs, calibration, and uncertainty. Results currently available are explicitly partitioned into validation and final-test findings above and below. Learning-curve behavior, repeated cross-validation stability, and the effect of controlled token dependence on classifier performance have not been measured in this project.

## Reproducibility Record

For reproducibility, preserve the dataset source and checksum, preprocessing specification, label mapping, split/fold indices and seeds, vocabulary policy, MNB parameters, software versions, metric definitions, and analysis/plotting settings. `requirements.txt` pins the direct dependencies to the existing working environment. The current scripts record the seed and split strategy but do not persist split indices or the downloaded dataset checksum; the downloader prints a checksum only on a fresh download. These provenance items remain limitations.

**Project status:** The methodology and several experiments are implemented. The completed and incomplete work is distinguished above; the proposed experiments below are not evidence of completed results unless corresponding outputs are identified.


## Completed Experiment Record and Initial Dataset Inspection

### Dataset Source
UCI SMS Spam Collection

### Dataset Statistics

- Total messages: 5,574
- Total columns: 2
- Ham messages: 4,827
- Spam messages: 747
- Ham proportion: 86.60%
- Spam proportion: 13.40%
- Missing values: 0

### Message Length Statistics

- Mean length: 80.48 characters
- Median length: 62 characters
- Minimum length: 2 characters
- Maximum length: 910 characters

### Duplicate Analysis

- Rows belonging to duplicate groups: 684
- Unique repeated message texts: 281
- Additional duplicate occurrences: 403
- Duplicate groups with conflicting labels: 0

### Initial Observations

1. The dataset has a class imbalance, with ham
   messages representing approximately 86.60%.
2. Repeated messages are present in the dataset.
3. No conflicting labels were found among identical
   message texts.
4. Missing values were not detected.

### Data Handling Decision

The original dataset is preserved without
deleting duplicate records.

Group-aware, stratified splitting was used to ensure
that identical message texts do not appear across
training, validation and independent-test partitions.

### Status

Dataset downloaded, inspected and duplicate
analysis completed.


## Experiment 1: Hand-Calculated Bayesian Classification

### Objective
Verify the mathematical working of the Multinomial
Naive Bayes classifier using a small artificial dataset.

### Training Dataset
- Ham messages: 3
- Spam messages: 3
- Vocabulary: free, meeting, now
- Vocabulary size: 3
- Smoothing parameter: alpha = 1

**Script:** `experiments/hand_calculation.py` (prints the manual calculation; it
does not write a result file).

### Word Counts

| Word | Ham | Spam |
|---|---:|---:|
| free | 0 | 4 |
| meeting | 2 | 0 |
| now | 2 | 2 |
| Total | 4 | 6 |

### Experimental Input
free now

### Calculated Results
- Ham probability: 24.85%
- Spam probability: 75.15%
- Predicted class: Spam

### Observation
The classifier assigned a higher posterior probability
to the spam class because the token "free" had a
higher estimated likelihood under the spam class.

`experiments/hand_calculation.py` calculates these probabilities directly from the
toy counts. An automated comparison with scikit-learn's classifier remains planned.

### Conclusion
The experiment verified the use of class priors,
smoothed word likelihoods and posterior probability
normalization in Multinomial Naive Bayes.


## Experiment 2: Laplace Smoothing Analysis

### Objective
Investigate how different smoothing parameters
affect word likelihood estimates in Multinomial
Naive Bayes.

### Experimental Conditions
- Alpha = 0 (No smoothing)
- Alpha = 1 (Laplace smoothing)
- Alpha = 5 (Stronger smoothing)

`experiments/smoothing_analysis.py` prints the probabilities below.
`experiments/smoothing_visualization.py` generates the separate alpha sweep plot
at `graphs/smoothing_probability_analysis.png`.

### Results

| Probability | Alpha 0 | Alpha 1 | Alpha 5 |
|---|---:|---:|---:|
| P(free|ham) | 0.0000 | 0.1429 | 0.2632 |
| P(meeting|ham) | 0.5000 | 0.4286 | 0.3684 |
| P(now|ham) | 0.5000 | 0.4286 | 0.3684 |
| P(free|spam) | 0.6667 | 0.5556 | 0.4286 |
| P(meeting|spam) | 0.0000 | 0.1111 | 0.2381 |
| P(now|spam) | 0.3333 | 0.3333 | 0.3333 |

### Observations

1. Without smoothing, unseen in-class vocabulary
   terms receive zero probability.
2. Laplace smoothing assigns positive probabilities
   to all vocabulary terms.
3. Stronger smoothing moves estimated probabilities
   towards a uniform distribution.
4. Smoothing changes the relative likelihoods of
   words across classes.

### Conclusion

The experiment demonstrates how additive smoothing
prevents zero-frequency likelihoods and changes
the probability estimates used by the classifier.

The best smoothing parameter cannot be determined
from this experiment alone. It requires further
evaluation on training and validation data.


### Visualization Analysis

A line graph was generated to study the effect
of different smoothing parameters on word
probability estimates.

The tested alpha values were:
0, 0.5, 1, 2, 5, 10 and 20.

### Observations

1. Zero-frequency probabilities become positive
   when smoothing is introduced.

2. Increasing alpha gradually moves word
   probabilities towards a uniform distribution.

3. As alpha approaches infinity, the estimated
   probabilities approach 1/3 for each vocabulary
   word.

4. P(now|spam) remains constant at approximately
   0.3333 because its frequency is proportional
   to the total number of spam word occurrences.

### Conclusion

The visualization demonstrates that the smoothing
parameter controls the influence of observed word
frequencies on probability estimates.

The graph is saved at:

graphs/smoothing_probability_analysis.png


## Experiment 3: Full-Dataset Prior Illustration

### Objective
Investigate the influence of class prior
probabilities on Naive Bayes predictions.

### Dataset
UCI SMS Spam Collection

Total messages: 5,574
Smoothing parameter: alpha = 1

### Actual Class Priors

- Ham: 0.8660
- Spam: 0.1340

The script `experiments/prior_analysis.py` fits both models on all 5,574 messages,
including the reserved test partition. The printed probabilities below are
illustrative only and are excluded from model selection and predictive-performance
claims. This script prints its examples; it does not save a CSV.

### Experimental Conditions

1. Actual class priors.
2. Equal class priors (0.5, 0.5).

Word likelihoods were verified to remain
identical between the two models.

### Results

| Message | Actual Priors | Equal Priors |
|---|---:|---:|
| Congratulations you won a free prize | 100.00% | 100.00% |
| Are we meeting tomorrow | 0.07% | 0.47% |
| Free entry claim your prize now | 100.00% | 100.00% |
| Please call me when you reach home | 0.02% | 0.13% |

### Observations

1. The original dataset has a substantial
   class imbalance.
2. Changing class priors affects posterior
   probabilities.
3. Strong word evidence can dominate the
   effect of prior probabilities.
4. Equal priors increased spam probabilities
   for the two ham-like example messages.

### Conclusion

The experiment demonstrates that class priors
influence posterior probability estimates,
even when word likelihoods remain unchanged.

### Prior Probability Visualization

A grouped bar chart was generated to compare
actual dataset priors with equal class priors.

### Observations

1. The actual ham prior is approximately 0.866.
2. The actual spam prior is approximately 0.134.
3. Equal priors assign 0.5 probability to both classes.
4. The original dataset has a substantial class imbalance.

### Conclusion

The visualization demonstrates how the choice
of class priors changes the initial probability
assigned to each class.

`experiments/prior_visualization.py` writes
`graphs/prior_probability_comparison.png`. This visualization uses the full-dataset
prior illustration above; it is not the validation comparison in Experiment 6.


## Experiment 4: Conditional Independence Analysis

### Objective
Investigate the conditional independence assumption
used by the Naive Bayes classifier.

### Experimental Dataset
Eight synthetic spam messages.

`experiments/independence_analysis.py` computes the joint-probability comparison.
It prints the results and does not write a CSV; `experiments/independence_visualization.py`
generates `graphs/independence_comparison.png`.

### Results

- P(offer|spam) = 0.5000
- P(prize|spam) = 0.5000
- Actual joint probability = 0.5000
- Independent estimate = 0.2500
- Difference = 0.2500

### Observations

1. Both individual word probabilities are 0.5.
2. The actual joint probability is 0.5.
3. The independence-based estimate is 0.25.
4. The estimated joint probability differs from
   the actual joint probability by 0.25.

### Conclusion

The experiment demonstrates that the conditional
independence assumption may not hold when words
are correlated.

This can affect the probability estimates produced
by Naive Bayes.

This is a fixed eight-message synthetic joint-probability demonstration. It does not
fit/evaluate a classifier, compare independent and dependent generators, or measure
word dependence in the UCI dataset. A controlled classifier-performance experiment
remains planned.


### Conditional Independence Visualization

A bar chart was generated to compare the actual
joint probability with the independence-based estimate.

- Actual joint probability: 0.50
- Independence estimate: 0.25
- Absolute difference: 0.25

The visualization demonstrates the limitations of
the conditional independence assumption when
features are correlated.

Graph:
graphs/independence_comparison.png


## Experiment 5: Alpha Sensitivity Analysis

### Objective
Investigate the effect of different Laplace smoothing
parameters on Multinomial Naive Bayes performance.

### Experimental Setup

- Training records: 3,567
- Validation records: 892
- Vocabulary size: 6,768
- Vectorization: CountVectorizer
- Script: `src/alpha_comparison.py`

Tested alpha values:
0.01, 0.1, 0.5, 1, 2, 5, 10.

### Results

| Alpha | Accuracy | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|---:|
| 0.01 | 0.9765 | 0.9298 | 0.8908 | 0.9099 | 0.9386 |
| 0.1 | 0.9765 | 0.9455 | 0.8739 | 0.9083 | 0.9461 |
| 0.5 | 0.9753 | 0.9533 | 0.8571 | 0.9027 | 0.9432 |
| 1 | 0.9742 | 0.9615 | 0.8403 | 0.8969 | 0.9287 |
| 2 | 0.9686 | 0.9596 | 0.7983 | 0.8716 | 0.9037 |
| 5 | 0.9630 | 1.0000 | 0.7227 | 0.8390 | 0.8563 |
| 10 | 0.9473 | 1.0000 | 0.6050 | 0.7539 | 0.8013 |

### Observations

1. Alpha = 0.1 achieved the highest validation PR-AUC.
2. Increasing alpha generally reduced spam recall.
3. Higher alpha values produced higher precision
   but lower recall.
4. Excessive smoothing reduced overall F1-score.

### Conclusion

The smoothing parameter has a substantial influence
on classification performance.

Alpha = 0.1 was selected based on validation PR-AUC among the tested values; it did
not have the highest validation F1. The held-out test results are reported separately
in the final independent-test section and were not used for this selection.

### Generated Files

- results/alpha_comparison.csv
- graphs/alpha_comparison.png


## Experiment 6: Validation-Only Prior Comparison

### Objective
Investigate how actual and equal class priors
affect Naive Bayes classification performance.

### Experimental Setup

- Training records: 3,567
- Validation records: 892
- Smoothing parameter: alpha = 0.1
- Model: Multinomial Naive Bayes
- Actual priors: Training class distribution
- Equal priors: 0.5 for each class
- Classification threshold: 0.5 (`model.predict()` default)
- Script: `src/prior_comparison.py`

Word likelihoods were verified to be identical.

### Results

| Metric | Actual Priors | Equal Priors |
|---|---:|---:|
| Accuracy | 0.9765 | 0.9720 |
| Precision | 0.9455 | 0.8917 |
| Recall | 0.8739 | 0.8992 |
| F1-score | 0.9083 | 0.8954 |
| PR-AUC | 0.9461 | 0.9461 |

### Confusion Matrices

Rows are actual labels and columns are predicted labels; both use the order
`[ham, spam]`.

Actual Priors:

[[767, 6],
 [15, 104]]

Equal Priors:

[[760, 13],
 [12, 107]]

### Observations

1. Equal priors improved spam recall.
2. Actual priors produced higher precision.
3. Actual priors achieved higher accuracy and F1-score.
4. PR-AUC remained unchanged.
5. Changing priors altered classification decisions
   without changing the estimated word likelihoods.

### Conclusion

Class priors influence the posterior probability
and classification decisions of Naive Bayes.

Actual priors provided a better precision and
accuracy trade-off in this validation experiment,
while equal priors detected more spam messages.

This is a validation-only comparison. The final independent-test evaluation was
performed separately using alpha = 0.1 and threshold = 0.3; its results are reported
in the canonical final-test section above.

### Output

results/prior_comparison.csv


## Experiment 7: Word Likelihood and Feature Analysis

### Objective
Identify words that contribute strongly to spam
and ham classification using conditional probabilities.

### Methodology

1. Train Multinomial Naive Bayes using training data.
2. Extract class-conditional word probabilities.
3. Calculate log-likelihood ratios.
4. Rank words based on their association with each class.
5. Generate visualizations.

The script `src/feature_analysis.py` fits on `dataset/processed/train_fit.csv`
with alpha = 0.1. The ranked terms describe training-set likelihood ratios; they
are not test-set feature importances or predictive-performance results.

### Mathematical Formula

LR(w) = log(P(w|spam) / P(w|ham))

### Strongest Spam-associated Words

| Word | Log-likelihood ratio |
|---|---:|
| claim | 7.9833 |
| prize | 7.6800 |
| uk | 7.6134 |
| 150p | 7.3821 |
| tone | 7.2913 |
| 18 | 7.0806 |
| guaranteed | 7.0806 |
| awarded | 6.8508 |

### Strongest Ham-associated Words

| Word | Log-likelihood ratio |
|---|---:|
| gt | -6.3527 |
| lt | -6.3427 |
| he | -6.1297 |
| she | -5.7370 |
| da | -5.6896 |
| lor | -5.5983 |
| later | -5.3860 |
| way | -5.3860 |
| say | -5.3038 |
| ask | -5.2144 |
| doing | -5.2144 |

### Findings

The experiment demonstrated that different words
have different conditional probabilities across
the two message classes.

Positive log-likelihood ratios indicate stronger
association with spam, while negative values
indicate stronger association with ham.

### Generated Files

- results/feature_analysis.csv
- graphs/spam_word_analysis.png
- graphs/ham_word_analysis.png


## Experiment 8: Decision Threshold Analysis

### Objective
Investigate how changing the decision threshold
affects spam classification performance.

### Experimental Setup

- Model: Multinomial Naive Bayes
- Alpha: 0.1
- Training records: 3,567
- Validation records: 892
- Thresholds: 0.1 to 0.9
- Script: `src/threshold_analysis.py`

### Results

| Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|
| 0.1 | 0.8699 | 0.8992 | 0.8843 |
| 0.2 | 0.9068 | 0.8992 | 0.9030 |
| 0.3 | 0.9298 | 0.8908 | 0.9099 |
| 0.4 | 0.9455 | 0.8739 | 0.9083 |
| 0.5 | 0.9455 | 0.8739 | 0.9083 |
| 0.6 | 0.9541 | 0.8739 | 0.9123 |
| 0.7 | 0.9541 | 0.8739 | 0.9123 |
| 0.8 | 0.9541 | 0.8739 | 0.9123 |
| 0.9 | 0.9626 | 0.8655 | 0.9115 |

### Observations

1. Lower thresholds increase spam detection sensitivity.
2. Higher thresholds improve precision.
3. Thresholds 0.6, 0.7 and 0.8
   achieved the highest F1-score.
4. The same predictions were obtained at
   thresholds 0.6, 0.7 and 0.8.

### Conclusion

Decision thresholds significantly affect
the classification trade-off between
false positives and false negatives.

A threshold of 0.6 is a candidate for further
consideration if validation F1 alone is the objective. The final configuration
retains threshold 0.3 as a precision/recall operating-point choice: on validation it
has higher recall (0.8908) than the higher-F1 thresholds 0.6–0.8 (0.8739), with
somewhat lower precision and F1. Threshold 0.3 is not claimed to maximize F1. The
final independent-test evaluation uses the fixed 0.3 threshold and is reported
separately; the test set is not used for threshold selection.

### Generated Files

- results/threshold_analysis.csv
- graphs/threshold_comparison.png


## Experiment 9: Probability Calibration Analysis

### Objective
Evaluate the reliability of probability estimates
generated by the Multinomial Naive Bayes classifier.

### Experimental Setup
- Model: Multinomial Naive Bayes
- Alpha: 0.1
- Training records: 3,567
- Validation records: 892
- Probability bins: 10
- Calibration strategy: Uniform binning
- Script: `src/calibration_analysis.py`

The Brier score, log loss, ECE and reliability bins use predicted probabilities
directly; no classification threshold is applied in this calibration analysis.

### Calibration Metrics

| Metric | Result |
|---|---:|
| Brier Score | 0.02056 |
| Log Loss | 0.17591 |
| Expected Calibration Error | 0.01870 |

### Observations

1. The Brier score was 0.02056.
2. Log loss was 0.17591.
3. ECE was 0.01870.
4. Most validation messages received low
   spam probabilities.
5. 107 messages received probabilities
   between 0.9 and 1.0.
6. Middle probability bins contained very
   few observations.

### Conclusion

The model achieved low overall probabilistic
prediction errors on the validation dataset.

However, the reliability analysis revealed
sparse observations in intermediate probability
bins. Therefore, individual bin estimates
should be interpreted cautiously.

### Generated Files

- results/calibration_results.csv
- results/calibration_metrics.csv
- graphs/calibration_curve.png


## Experiment 10: Statistical Uncertainty Analysis

### Objective
Estimate the uncertainty of classification
performance using Bootstrap Resampling.

### Experimental Setup

- Model: Multinomial Naive Bayes
- Alpha: 0.1
- Training records: 3,567
- Validation records: 892
- Bootstrap iterations: 1,000
- Confidence level: 95%
- Random seed: 42
- Script: `src/statistical_analysis.py`

### Original Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.9765 |
| Precision | 0.9455 |
| Recall | 0.8739 |
| F1-score | 0.9083 |

### Bootstrap Confidence Intervals

These are 1,000 **validation message-group bootstrap** resamples (seed 42), not
independent-test intervals. Each resample draws the 821 unique validation message
texts with replacement and keeps all repeated rows for a selected text together;
the number of rows can vary between resamples. The fixed classifier uses alpha 0.1
and the 0.5 baseline threshold for these validation statistics.

| Metric | Lower 95% CI | Upper 95% CI |
|---|---:|---:|
| Accuracy | 0.9639 | 0.9870 |
| Precision | 0.8819 | 0.9917 |
| Recall | 0.8020 | 0.9375 |
| F1-score | 0.8594 | 0.9487 |

### Observations

1. The bootstrap F1-score distribution is
   concentrated around the original score.
2. Accuracy shows relatively low variation.
3. Precision and recall show greater variation.
4. The estimated F1-score confidence interval
   ranges from 0.8594 to 0.9487, matching the values in
   `results/confidence_intervals.csv`.

### Conclusion

Bootstrap resampling provided an estimate
of uncertainty in validation performance.

The results indicate reasonably consistent
classification performance across resampled
validation datasets.

This analysis measures validation sampling uncertainty
for the fixed trained model and predictions, with repeated
message texts treated as clusters. It is not a test-set interval.

### Generated Files

- results/bootstrap_results.csv
- results/confidence_intervals.csv
- graphs/bootstrap_f1_distribution.png


## Experiment 11: Prediction Error Analysis

### Objective

Investigate incorrect predictions made by the
Multinomial Naive Bayes classifier and identify
patterns in classification errors.

### Experimental Setup

- Model: Multinomial Naive Bayes
- Alpha: 0.1
- Decision threshold: 0.5
- Validation records: 892
- Script: `src/error_analysis.py`

This is a validation error analysis at the baseline threshold, not the final
independent-test evaluation. `src/refined_error_analysis.py` uses the same training
and validation partitions and threshold to summarize terms in error messages.
### Prediction Summary

| Category | Count |
|---|---:|
| Correct predictions | 871 |
| Incorrect predictions | 21 |
| False positives | 6 |
| False negatives | 15 |

### Observations

1. The classifier correctly classified 871 messages.
2. Six legitimate messages were incorrectly
   classified as spam.
3. Fifteen spam messages were incorrectly
   classified as legitimate.
4. Average confidence of incorrect predictions
   was 90.8%.
5. Some incorrect predictions had confidence
   values reaching 100%.

### Error Analysis Findings

False positives included conversational
messages containing words associated with
promotional content.

False negatives included promotional messages,
prize announcements and other spam messages
that resembled legitimate communication.

### Conclusion

The error analysis identified two important
classification challenges:

- Legitimate messages containing promotional
  vocabulary.
- Spam messages resembling ordinary conversations.

The results demonstrate that high classification
accuracy does not eliminate individual prediction
errors or overconfident incorrect predictions.

### Generated Files

- results/error_analysis.csv
- results/error_summary.csv
- results/error_terms_by_type.csv
- graphs/error_distribution.png
- graphs/error_confidence.png
- graphs/refined_error_terms.png
