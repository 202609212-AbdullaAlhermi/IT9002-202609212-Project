# Interpretation paragraphs for notebooks/STUDENTID_Task1.ipynb (design v2), written after looking at the real outputs.
# Every number quoted here was read from the executed notebook / results files (run of 2026-10-21).

INTERP["leakage"] = r"""
**Interpretation (`results/leakage_check.csv`).** With a random clause-level split, **48 %** of held-out clauses have
their most similar training clause in the *same contract*, and 6.4 % have a near-duplicate (cosine ≥ 0.9) in
training, against 0 % and 2.7 % with a contract-level split. (The remaining 2.7 % are boilerplate that genuinely
recurs across different companies' contracts — legitimate, not leakage.) The same default model scores **0.791**
macro-F1 under the clause-level split but **0.765** under the contract-level split: about 0.03 of the apparent
performance comes from recognising the contract, not the clause type. A contract-level split is therefore required
for an honest estimate on *new* contracts — the real use case — and the same contract split is used for Task 2.
Section 4.7 repeats the check with the tuned best classical model.
"""

INTERP["hypotheses"] = r"""
**Interpretation (`results/confusion_hypotheses.csv`, Figure 2).** Every hypothesised pair is more similar than the new
class's median similarity to the other classes, so all nine are plausible confusions for a bag-of-words model. The
strongest are **Non-Transferable License vs License Grant** (cosine 0.774 against a median of 0.428),
**Competitive Restriction Exception vs Exclusivity / Non-Compete** (0.701 / 0.699) and **Joint Ip Ownership vs Ip
Ownership Assignment** (0.669). The data also points to neighbours we had not written down: the closest class to
**Liquidated Damages** is Post-Termination Services (0.610, closer than Cap On Liability at 0.492), Non-Disparagement
is closest to Non-Compete (0.590), and Source Code Escrow to Post-Termination Services (0.439). These are similarities
of class *averages*; whether a trained model actually confuses the pairs is tested on validation in section 4.4.
"""

INTERP["curation"] = r"""
**Interpretation.** Of the 13,823 expert-labelled spans in CUAD, 9,825 belong to the 37 genuine clause categories.
The iterative ≥ 60 rule removed 10 small categories (listed above), leaving 27 classes whose 8,993 annotations form
Task 2. Of those, 7,830 clauses carry exactly one kept label; removing the 45 validation/test clauses whose exact text
also appears in training leaves the **7,785 Task 1 clauses**. The rebuild from the official CUAD file matches all four
frozen files exactly, so the curation is fully reproducible, and no span is misaligned with the contract text.

*Honesty note (also in the report's Limitations):* the ≥ 60 rule counted clauses across **all** contracts, including the
test contracts. No model output or test score was involved — only class sizes — but it is a data-level choice made with
knowledge of the whole corpus.
"""

INTERP["ambiguous"] = r"""
**Why these 557 clauses are held out of Task 1, and how they are used.** Each of them was highlighted by the lawyers
under **two or more** of the 27 kept categories (510 under two, 45 under three, 2 under four). A single-label classifier
cannot be scored fairly on a clause that has several correct answers, so they are excluded from training and from the
main test score (no multi-label variant of Task 1 is built). They are **not** thrown away: in the error analysis (Part C)
we check how often the best model's prediction is **one of the clause's valid labels** on the validation and test rows
(190 clauses; the 367 training-split rows are not used because their contracts were seen in training). They are also
used by the Part F anomaly study, and Task 2 keeps them because extraction is multi-label by design.

The overlaps are informative in themselves: the most frequent combination, **License Grant + Non-Transferable License**
(163 clauses), is larger than the 96 *single-label* Non-Transferable License clauses — in CUAD a non-transferability
statement is usually part of a licence grant, and the single-label examples are the less typical remainder. Exclusivity +
License Grant (69), Expiration Date + Renewal Term (40), Anti-Assignment + Change Of Control (22) and Competitive
Restriction Exception + Exclusivity (18) mirror the confusion hypotheses of section 2.2.
"""

INTERP["labels"] = r"""
**Interpretation.** The classes are strongly imbalanced: **10.7 : 1** between Cap On Liability (650 clauses) and
Non-Disparagement (61). Three classes have fewer than 15 test clauses — **Source Code Escrow (4), Joint Ip Ownership (10)
and Non-Disparagement (12)** — so a single clause changes their test F1 by a large step; their confidence intervals will
be wide and the robustness macro-F1 over the other 24 classes is reported next to the primary metric. The validation
split mirrors the training distribution closely (it is stratified); the test split is fixed by CUAD's official contract
list and differs more, a realistic distribution shift that macro-F1 is not sensitive to (each class counts once, however
many test clauses it has).
"""

INTERP["length"] = r"""
**Interpretation.** Clauses are short: the median is 41 words, 95 % have ≤ 129 words and 94.9 % have ≤ 128, against a
median of about 5,000 words for a whole contract (section 2.4). A whole clause therefore fits in a transformer's input
window (section 3.5 measures the tokens), and the texts are short enough that TF-IDF vectors are sparse. Length differs by
class — Termination For Convenience, Governing Law and Expiration Date have medians of 27–30 words, Post-Termination
Services and No-Solicit Of Employees about 55 — so length alone is weak evidence of the type, and long clauses are a
natural subgroup for the error analysis.
"""

INTERP["study"] = r"""
**Interpretation (all numbers from `results/preprocessing_study.csv`).**

* **No preprocessing step reliably helps on the 27-class data.** The raw-text baseline scores **0.757** validation
  macro-F1. No option improved it by the pre-declared +0.003; the largest gain was Unicode normalisation (+0.0018,
  4 extra features), and every option's 95 % CI includes zero.
* **Lower-casing (−0.003), numbers → `num` (−0.004), Porter stemming (−0.004) and lemmatisation (−0.009) were rejected.**
  Merging word forms or numbers throws away distinctions the linear model can use (e.g. "Term" as a defined term versus
  "term", "30 days" versus "num days").
* **Stop words:** the full NLTK list scored −0.003 (CI −0.023 to +0.010) and the legal-aware list +0.001; both are inside
  the noise band, so the pre-declared tie-break kept the **smallest vocabulary** — full stop-word removal. This choice was
  made by the rule, not by evidence that stop words hurt or help.
* **Word pairs:** after stop-word removal, 1–2-grams (−0.015) and 1–3-grams (−0.025) were clearly worse than single words
  on this untuned bench.
* **Overall:** the selected pipeline (`secref` for cross-references, full stop-word removal, single words) scores 0.754 —
  **−0.003 against raw text, CI [−0.024, +0.011]**. The honest conclusion is that, for a linear TF-IDF model on these
  clauses, the text-level choices tested here make no measurable difference; the gains must come from the
  representation and the classifier settings (section 4).
* **Several findings of the first-version (20-class) dry run did not replicate** (that run favoured lower-casing and word
  pairs and found stop-word removal clearly harmful). The class set changed, and with it the evidence — which is why every
  choice was re-tested on the v2 validation set instead of being carried over.
* **Path dependence and the follow-up.** A greedy study decides each step given the earlier ones, with an untuned
  classifier, so a near-tie early on (stop words) can change later results (n-grams). The tuned grid in section 4 therefore
  re-tests the stop-word decision as a grid dimension. **This re-test was decided after the study and before any tuning,
  and it uses the validation set only** (no test data was involved in this or any other choice). There, keeping stop words is better for every
  word-level model (best LinearSVC 0.783 vs 0.775, Logistic Regression 0.779 vs 0.771, Complement NB 0.712 vs 0.683), and
  character n-grams — which do not depend on the stop-word step — give the best linear models.
"""

INTERP["transformer"] = r"""
**Interpretation (`results/token_lengths.json`).** On the training clauses the 99th percentile is 312 Legal-BERT tokens and
314 BERT-base tokens; the larger, rounded up to a multiple of 32, gives `max_length` = **320** for both models; fewer than 1 % of clauses in any split are truncated (0.94 % train, 0.49 % validation, 0.55 %
test), and for those the first 320 tokens normally contain the operative wording. Padding every clause to the model
maximum of 512 would make batches much longer for almost no extra text.

The two vocabularies give nearly the same *total* token count (BERT-base needs 1.9 % more tokens for the same clauses),
but the example shows *where* they differ: Legal-BERT keeps core legal terms whole ("licensee", "indemnify", "licensor"),
while BERT-base breaks them into fragments ("license ##e", "ind ##em ##ni ##fy", "li ##cens ##or"). This matters for the
controlled ablation (Legal-BERT vs BERT-base): the two models differ in their pre-training **corpus and** their
**vocabulary**, and the ablation cannot separate the two effects.
"""

INTERP["task2prep"] = r"""
**Interpretation (`results/task2_input_stats.json`).** Contracts have a median of 33,143 characters (95th percentile
162,491, maximum 338,211), i.e. a median of about 6,600 Legal-BERT tokens and up to 78,958 — up to about 150 times the
512-token limit, so sliding windows are unavoidable. With 256-token windows and a stride of 128 the 510 contracts give
**42,168 windows**. Normalising the text would break the labels: Unicode (NFKC) normalisation changes the length of 39
contracts and collapsing whitespace changes 505 of the 510, which would shift every later character offset. Task 2
therefore keeps the raw text and maps word-pieces to characters through the tokenizer's offsets (details and the oracle
check in `STUDENTID_Task2.ipynb`).
"""

INTERP["search"] = r"""
**What the search showed (5,776 configurations, `results/classical_search.csv`).**

* **The two linear models are clearly ahead and close to each other.** Best validation macro-F1: Logistic Regression
  **0.793**, LinearSVC **0.788**, Complement NB 0.712.
* **Character n-grams won for both linear models** (`char_wb`, 3–6 characters: 0.793 / 0.788 against 0.779 / 0.783 for the
  best word-level settings). Character n-grams match word stems and fragments ("assign", "liab", "escrow") without any
  stemming rule, and they are robust to the hyphenation and typing variants of the filings.
* **Balanced class weights mattered more than any text preprocessing:** all ten best configurations of both linear models
  use `class_weight="balanced"`, and the best score without weights is lower (LinearSVC 0.780, Logistic Regression 0.778).
  With 27 imbalanced classes, unweighted models under-predict the rare ones.
* **Keeping stop words helped the word-level models** (see 3.3); for Complement NB, word pairs helped (best 0.712 with
  1–2-grams vs 0.659 with single words), while the linear models did slightly better with single words.
* **Settings that hardly mattered:** `min_df`, `max_df`, `max_features` and `sublinear_tf` move the best score by about
  0.01 or less (Figure 9).
* **No selected value sits on the edge of its grid** (cell above), so the grid was wide enough.
* **The top of the search is flat:** the ten best LinearSVC configurations lie within 0.005 of each other and the ten best
  Logistic Regression configurations within 0.012. Picking the maximum out of thousands of tries on one validation set is
  optimistic (the *winner's curse*), so the test score may be lower.
"""

INTERP["valconf"] = r"""
**Interpretation (`results/top_confusions_validation.csv`, `results/confusion_hypotheses_check.csv`).** On validation,
the best classical model (Logistic Regression) makes these confusions most often: Exclusivity → License Grant (6) and
Post-Termination Services → Revenue/Profit Sharing (6), then Non-Transferable License → Anti-Assignment (5),
Renewal Term → Expiration Date, Rofr/Rofo/Rofn → License Grant and Minimum Commitment → Volume Restriction (4 each).

* **Supported hypotheses:** *Non-Transferable License ↔ Anti-Assignment* (5 clauses one way, 3 the other; only 4 of 13
  validation clauses of the new class are correct) and *Non-Transferable License ↔ License Grant* (1 + 1);
  *Competitive Restriction Exception → Exclusivity* (3 of 11) and *→ Non-Compete* (1). These are the two weakest new
  classes on validation (recall 0.31 and 0.27).
* **Not observed on validation:** No-Solicit Of Employees vs Non-Compete, Liquidated Damages vs Cap On Liability, Joint Ip
  Ownership vs Ip Ownership Assignment, Source Code Escrow vs License Grant, Non-Disparagement vs Covenant Not To Sue (0
  confusions each; these new classes reach validation recall 0.73–0.94). With only 7–16 validation clauses per new
  class, "not observed" is weak evidence; the test confusions below are a second, independent look.
* **Unexpected confusions** between *old* classes (Post-Termination Services → Revenue/Profit Sharing, Renewal Term →
  Expiration Date) show that the harder 27-class task is not only about the new classes.
"""

INTERP["test"] = r"""
**Interpretation of the test results (each model scored once; `results/test_eval_log.csv`).**

* **Objective O2 is met:** the best TF-IDF model beats the majority baseline (test macro-F1 0.006) by **0.788**, far above
  the 0.50 target.
* **Logistic Regression (0.794, 95 % CI 0.752–0.824) and LinearSVC (0.786, CI 0.750–0.817) are statistically tied:** the
  paired contract-level bootstrap gives a difference of +0.008 with CI [−0.011, +0.024]. Logistic Regression is the "best
  classical model" because it was best on *validation* (0.793 vs 0.788), the pre-declared rule. It also gives class
  probabilities, useful for ranking clauses for a reviewer.
* **Complement NB is clearly weaker** (0.713, CI 0.669–0.738).
* **The robustness number agrees:** over the 24 classes with ≥ 15 test clauses, Logistic Regression scores 0.796 and
  LinearSVC 0.790 — the conclusions do not depend on the three rare classes.
* **Validation and test agree** for the best model (0.793 vs 0.794), so the winner's-curse effect was small here.
* Accuracy (0.829) is higher than macro-F1 (0.794) because the large classes are easier than several small ones — the
  reason macro-F1 is the primary metric.
"""

INTERP["best"] = r"""
**Interpretation of the best classical model (Logistic Regression on character n-grams).**

* **Easy classes have distinctive vocabulary:** Governing Law (F1 0.994), No-Solicit Of Employees (0.968), Insurance,
  Expiration Date, Renewal Term, Audit Rights and Cap On Liability (all ≥ 0.91).
* **Six classes fall below 0.70 recall** — Exclusivity (0.57), Liquidated Damages (0.57), Non-Disparagement (0.58),
  Competitive Restriction Exception (0.65), Change Of Control (0.67) and Non-Transferable License (0.68) — so this model
  does **not** satisfy objective O5 (O5 is judged on the overall best model in Part C).
* **The test confusions repeat the validation picture:** Exclusivity → License Grant (7) and → Non-Compete (6), Change Of
  Control → Anti-Assignment (6), License Grant → Exclusivity (6), Competitive Restriction Exception → Non-Compete (4, 24 %
  of its test clauses). **Liquidated Damages → Post-Termination Services (5, 24 %)** confirms the data-driven neighbour
  found in section 2.2 rather than the hypothesised Cap On Liability. Minimum Commitment → Volume Restriction (7) is the
  most frequent confusion between two old classes.
* **Rare classes are essentially unmeasured:** all 4 Source Code Escrow test clauses come from **one** contract, so a
  contract-level CI cannot be estimated at all (F1 0.857 is a single-contract result); Joint Ip Ownership (10 clauses, 5
  contracts) has F1 0.769 with a CI of 0.33–1.00 (Figure 10). Their point values should not be over-read.
* **The strongest features are fragments of legal cue words** (`assign`, `audit`, `liab`, `insur`, `escrow`, `dispar`,
  `empl`, `renew`, `warrant`), so the model is interpretable for a legal reviewer; a few are generic (`pay`, `use`, `20`).
* **The readable word-level version** (best word-level Logistic Regression on validation: single words, stop words kept,
  balanced, validation macro-F1 0.779) shows the same picture in plain words — *assign, consent, void* (Anti-Assignment),
  *audit, records, inspect* (Audit Rights), *liquidated, damages, penalty* (Liquidated Damages), *disparage, reputation,
  goodwill* (Non-Disparagement), *sublicense, transferable* (Non-Transferable License). It also exposes a weakness: some
  strong features are company or product names from a few training contracts (*Sparkling*, *MBE*, *MLX*, *XLI*), i.e. the
  word model partly memorises parties. The contract-level split prevents this from inflating the test score, but it is
  noted in Limitations.
"""

INTERP["leakbest"] = r"""
**Interpretation (`results/leakage_check_best_classical.csv`).** For the tuned best classical pipeline, a random
clause-level split gives 0.799 ± 0.016 macro-F1 against 0.786 ± 0.012 with a contract-level split: the naive split would
overstate its performance by **+0.013**. The effect is about half of the default model's +0.026 (section 2.1) —
character n-grams with balanced weights rely less on contract-specific wording — but it is still positive in the mean,
which confirms that the contract-level split is the right protocol.
"""

INTERP["sofar"] = r"""
* **The 27-class task is learnable with simple models but clearly harder.** Tuned TF-IDF + linear classifiers reach about
  **0.79 test macro-F1** (Logistic Regression 0.794 [0.752, 0.824], LinearSVC 0.786) on contracts never seen in training, against 0.006
  for the majority baseline.
* **What mattered:** character n-grams and balanced class weights (sections 3.7 and 4.3). Text preprocessing (lower-casing,
  stemming, placeholders, stop words, n-gram range) made no measurable difference on the untuned bench (section 3.3), and
  keeping stop words was better once the classifiers were tuned.
* **Where the errors are:** between classes that overlap in legal meaning — Exclusivity vs License Grant / Non-Compete,
  Change Of Control vs Anti-Assignment, and the new classes Competitive Restriction Exception, Non-Transferable License and
  Liquidated Damages. Several of these clauses genuinely carry two labels (section 2.6), which limits any single-label
  model. The rare classes cannot be measured precisely with 4–12 test clauses.
* **Honesty notes** (also in the report's limitations): the ≥ 60 class-size rule used whole-corpus
  counts (class sizes only); a first-version 20-class pipeline was run before the proposal was approved — its outputs are
  archived and never reported, and every v2 choice was re-made on v2 validation (deviation D7 in the report); the multi-label clauses are
  held out of the main score; the contracts are US, SEC-filed and English only.
* **Next:** fine-tuned Legal-BERT and the BERT-base ablation on the same split (section 5), compared with these baselines
  with the paired contract-level bootstrap.
"""


INTERP["featuredec"] = r"""
**Interpretation (`results/feature_decisions.csv`, Figure 7).** On the same untuned bench, the representation matters far
more than any cleaning step:
* **Character n-grams vs word tokens: +0.020** validation macro-F1 without class weights (0.774 vs 0.754, CI −0.003 to
  +0.046) and the best cell overall with balanced weights (**0.787, +0.033 over the word baseline, CI +0.001 to +0.066** —
  the only change in the whole preprocessing section whose CI excludes zero). The price is a vocabulary 12 times larger
  (110,858 character features vs 9,149 words), which the linear model handles easily.
* **Balanced class weights interact with the features:** they *lower* the word model slightly (−0.007, CI includes zero) but
  *raise* the character model (0.774 → 0.787). With the much richer character features, re-weighting lets the rare classes
  use them instead of being out-voted by the large classes.
* **The sample clause shows why:** its text is partly redacted (`[ * ]`), so the word model keeps only 8 tokens, while the
  character model still sees 237 overlapping pieces (`ight`, `ghts`, `cons`…) that carry the meaning of *rights*, *consent*
  and *assign*.
* Section 4 tunes both decisions jointly with the classifiers; the tuned grid confirms them (all ten best linear
  configurations use character n-grams or balanced weights, and the selected linear models use both).
"""

INTERP["curves"] = r"""
**Interpretation of the curves (Figure 13).**
* **Warm-up.** During roughly the first half epoch the training loss stays near 3.3 — ln 27, the loss of guessing uniformly
  among 27 classes — while the learning rate ramps up from 0; once the rate is high enough the loss drops steeply. The
  warm-up let the randomly initialised classification layer settle before large updates reached the pre-trained layers.
* **Fit and generalisation.** Legal-BERT's mean training loss falls from 1.78 (epoch 1) to 0.09 (epoch 5), while its
  validation loss levels off after epoch 2 (0.50 → 0.46–0.54): the model fits the training clauses almost perfectly and its
  *confidence* stops improving on validation. Its decisions still improve until epoch 4 (validation macro-F1 0.730 → 0.795
  → 0.817 → **0.851**) and then drop slightly at epoch 5 (0.841) — the first sign of over-fitting, so epoch 4 is selected.
* **BERT-base learns more slowly:** after one epoch it reaches 0.564 validation macro-F1 against Legal-BERT's 0.730, its
  validation loss is higher at every epoch (0.55 at the end vs 0.46–0.48), and it is still improving at epoch 5 (0.814 →
  0.836), so it might gain slightly from more epochs. Starting from a legal-domain model means less has to be learnt during
  fine-tuning.
"""

INTERP["lrseeds"] = r"""
**Interpretation (Figure 14, seed table, `results/transformer_choice.json`).**
* **Learning rate.** Both models were best at **5e-5**, the top of the range recommended by Devlin et al. (2019): Legal-BERT
  0.839 / 0.834 / **0.851** and BERT-base 0.760 / 0.797 / **0.836** at 2e-5 / 3e-5 / 5e-5. BERT-base is much more sensitive to
  the rate. Because the best value is at the edge of the searched range, a slightly higher rate might be better still; we
  kept the recommended range and state this as a limitation.
* **Seeds.** At 5e-5, Legal-BERT scores **0.848 ± 0.004** and BERT-base **0.826 ± 0.009** on validation (mean ± sd over 3
  seeds). Every Legal-BERT seed (lowest 0.843) beats every BERT-base seed (highest 0.836). The pre-declared test run (seed 42)
  is the best of its three Legal-BERT seeds on validation (0.851), so its single validation number is slightly optimistic; the
  3-seed mean is the fairer validation estimate.
* **Class-weighted loss — not adopted.** Over the same 3 seeds it scored 0.849 ± 0.012 against 0.848 ± 0.004 unweighted: a
  mean gain of **+0.0004**, far below the seed spread of 0.008 (and the 0.003 minimum). It also made results *less* stable
  (sd 0.012 vs 0.004). Unlike the linear models, the fine-tuned transformer does not need re-weighting for these classes.
"""

INTERP["results5"] = r"""
**Interpretation of the results (each model scored on test once; `results/test_eval_log.csv`).**
* **Legal-BERT is the best Task 1 model so far:** test macro-F1 **0.871** (95 % CI 0.836–0.896), robustness macro-F1 over 24
  classes 0.866, accuracy 0.891.
* **Objective O3, hypothesis H1 — supported.** Legal-BERT beats the best classical model (TF-IDF character n-grams + Logistic
  Regression, 0.794) by **+0.077**, paired contract-level bootstrap 95 % CI **[+0.047, +0.110]** (robustness set: +0.070
  [+0.044, +0.102]); no resample favoured the classical model. The cost and latency part of O3 is completed by the LLM comparison.
* **Controlled ablation — legal-domain pre-training helps.** With everything else identical (architecture, size, recipe,
  learning rate, seed, split, metric), Legal-BERT beats BERT-base by **+0.045** test macro-F1, CI **[+0.018, +0.064]**
  (robustness set +0.036 [+0.015, +0.057]); on validation +0.015 for seed 42 and +0.022 on the 3-seed means. The shared
  learning rate (5e-5) is also BERT-base's own best rate, so the comparison is not tilted by tuning for Legal-BERT.
* **General pre-training alone also helps:** even BERT-base beats the best classical model (+0.032, CI [+0.007, +0.064]).
* **What the ablation cannot say.** Legal-BERT differs from BERT-base in its pre-training *corpus* **and** its *vocabulary*, so
  +0.045 is their combined effect. Clause length is not the explanation: the two tokenizers give almost the same number of
  tokens per clause (69.3 vs 70.6) and truncate under 1 % of clauses. A possible overlap between Legal-BERT's EDGAR
  pre-training text and CUAD contracts could contribute and cannot be checked.
* **Computational budget** (RTX 5080; table above, `results/compute_budget.csv`): one 5-epoch run takes about 95 s (17 s per
  epoch, peak 3.7 GB GPU memory); the whole 13-run plan took 21.0 min. The model has 109.5 M parameters and needs 439 MB on
  disk. Inference: about 1,030 clauses/s on the GPU (1.0 ms per clause) and 19 clauses/s on the CPU (52 ms per clause).
  The best classical model refits in seconds on the CPU, so the transformer costs far more compute, though it is still small
  in absolute terms for this dataset.
"""

INTERP["perclass"] = r"""
**Interpretation of the per-class analysis (`results/per_class_f1_models.csv`, `results/per_class_focus_new_classes.csv`).**
* **Legal-BERT is better than the best classical model on 23 of the 27 classes** and better than BERT-base on 22. The largest
  gains over the classical model are on classes defined by legal concepts rather than fixed phrases: Warranty Duration
  (+0.22), Volume Restriction (+0.22), **Exclusivity (+0.19)**, Liquidated Damages (+0.18), Non-Disparagement (+0.16) and
  Competitive Restriction Exception (+0.15).
* **Exclusivity**, the weakest class of the classical model (F1 0.549, recall 0.57), rises to **F1 0.738, recall 0.70**. Its
  remaining errors are the expected ones: 6 clauses predicted as License Grant ("exclusive licence") and 3 as Non-Compete.
* **The 7 new classes:** Legal-BERT is best for Competitive Restriction Exception (0.757 vs 0.611 classical), Liquidated
  Damages (0.810 vs 0.632), Non-Disparagement (0.857 vs 0.700) and Source Code Escrow (1.000 — but all 4 test clauses are in
  one contract, so this is not a reliable estimate). BERT-base is best for Joint Ip Ownership (0.900 vs 0.870; 10 clauses), and
  the classical model for **No-Solicit Of Employees** (0.968 vs 0.882 — a class with distinctive words such as "solicit" and
  "employee") and **Non-Transferable License** (0.634 vs 0.611).
* **Non-Transferable License remains the hardest class for every model** (Legal-BERT F1 0.611, recall 0.58, CI 0.32–0.80): 6 of
  its 19 test clauses are predicted as Anti-Assignment. As section 2.6 showed, most non-transferability statements in CUAD are
  annotated *together* with License Grant and are held out as ambiguous; the 96 single-label examples are the less typical
  remainder, and they read much like assignment restrictions.
* **Objective O5 (recall ≥ 0.70 for every class) — verdict: NOT MET.** O5 is judged on Legal-BERT, the pre-declared best
  Task 1 model: 2 of 27 classes fall below 0.70 recall — Non-Transferable License (0.58) and Change Of Control (0.63, often
  predicted as Anti-Assignment). For context: BERT-base misses it on 5 classes and the best classical model on 6
  (`results/metrics.json`).
* **Class size matters less for Legal-BERT** (Figure 16): the Spearman correlation between training-set size and test F1 is
  0.44 for the classical model but 0.25 for Legal-BERT. Pre-training lets small classes reach high F1 (Non-Disparagement,
  42 training clauses, F1 0.857). The rare-class values come with very wide CIs and should be read with that in mind.
"""


INTERP["sensitivity"] = r"""
**Interpretation (`results/transformer_lr_sensitivity.csv`, `results/transformer_epoch_sensitivity.json`).**
* **The optimum is not above the pre-declared grid.** For Legal-BERT, 7e-5 gives 0.850 and 1e-4 gives 0.840 against 0.851 at
  5e-5; for BERT-base, 0.830 and 0.820 against 0.836. Both models peak at 5e-5 and decline above it, so extending the grid would
  not have changed the choice, and the "best value at the edge" limitation of section 5.5 is resolved.
* **More epochs barely help BERT-base.** Trained for 8 epochs it reaches 0.838 (best at epoch 8) against 0.836 after 5, and its
  validation loss rises after epoch 3 (over-confidence). The validation gap to Legal-BERT shrinks only from +0.015 to +0.014, so
  the 5-epoch budget of the controlled ablation does not explain Legal-BERT's advantage.
"""

INTERP["ledgar_compare"] = r"""
**Interpretation (`results/ledgar_comparison.csv`, `results/ledgar_results.json`, Figure 30).**
* **The published Legal-BERT result is reproduced within its uncertainty.** Our best-validation seed scores **88.65 µ-F1 /
  82.68 m-F1** on the LEDGAR test set, against **88.2 / 83.0** in LexGLUE Table 3 (p. 4317): +0.45 and −0.32 points. Both
  published values lie inside our 95 % bootstrap intervals ([88.0, 89.3] and [81.4, 83.8]). On validation our three seeds give
  88.5 ± 0.1 µ-F1 and 81.9 ± 0.2 m-F1, against the paper's 88.5 ± 0.0 and 82.2 ± 0.3 (Tables 5–6, p. 4327). This is the main
  purpose of the comparison: our fine-tuning code, run with the published set-up on the same data, reaches the published number.
* **TF-IDF + linear SVM:** 88.09 µ-F1 / 82.84 m-F1 vs the paper's 87.2 / 82.4. Ours is slightly higher (µ-F1 by 0.9 points, just
  outside our interval). The paper does not give its grid values, so our grid (word 1–2-grams, sublinear tf, all features,
  C = 1, squared hinge) may simply be a better configuration.
* **Fine-tuning hardly helps on LEDGAR:** Legal-BERT − TF-IDF = −0.16 m-F1 points, 95 % CI [−1.15, +1.26] (+0.56 µ-F1). The paper
  shows the same picture (88.2 vs 87.2 µ-F1, 83.0 vs 82.4 m-F1).
* **Differences from the paper's run, and their likely effect:**
  1. **Epochs:** at most 10 instead of 20 (our budget cut). Seeds 43 and 44 were still at their best in epoch 10 and validation
     m-F1 was still creeping up (Figure 30), so the cut probably costs a few tenths of a point, mostly in m-F1. It is the most
     plausible reason why our m-F1 is 0.3 below the paper while our µ-F1 is above it.
  2. **Seeds:** 3 instead of 5, so the best-validation selection has fewer chances (and with 5 seeds a slightly luckier seed is
     more likely).
  3. **Optimiser details not stated in the paper:** we used AdamW without weight decay and a linear decay over 10 epochs (Hugging
     Face defaults), early-stopping patience 3 on micro-F1, gradient clipping 1.0.
  4. **Precision:** bf16 instead of fp16 (our GPU).
  5. **Hardware:** RTX 5080, about 5.8 min per epoch (3 seeds: 3,668 + 3,611 + 3,639 s ≈ 3.0 h, peak 3.3 GB). The paper reports
     21 min per epoch on its own hardware (Table 7, p. 4328).
  6. **Data version:** the current Hugging Face copy of `lex_glue/ledgar`, with the same split sizes as the paper's Table 1
     (60,000 / 10,000 / 10,000, 100 labels).
  7. **Maximum length:** 512, which the paper implies but does not state; only 1.4 % of test provisions are longer.
  Metrics, label set, split and the reporting rule (test score of the best-validation seed, evaluated once) are identical.
"""

INTERP["ledgar_relate"] = r"""
**Interpretation — relating LEDGAR to our CUAD result (`results/ledgar_vs_cuad.csv`).** The two numbers cannot be compared
directly, only explained:
* **Different tasks and labels.** CUAD Task 1 has 27 expert-defined clause *types* (legal risk categories). LEDGAR has 100
  provision *topics*, several of which overlap almost by name ("Assigns" / "Assignments", "Waivers" / "No Waivers",
  "Indemnity" / "Indemnifications", "Governing Laws" / "Applicable Laws"). Such pairs cap LEDGAR's macro-F1 for any model.
* **Different sizes.** 5,315 vs 60,000 training clauses (median 154 vs 426 per class). Macro-F1 is averaged over 27 vs 100
  classes, and LEDGAR has a long tail (the rarest label has 23 training provisions).
* **Different splits.** CUAD: by contract (test clauses come from unseen contracts). LEDGAR: chronological (test = 2019), so
  there is also drift over time.
* **Micro vs macro.** Micro-F1 (= accuracy here) is similar on both: 89.1 (CUAD) vs 88.6 (LEDGAR). Macro-F1 differs more
  (87.1 vs 82.7) because of the label-set differences above. Our CUAD macro-F1 is therefore **not** "better than LexGLUE"; the
  datasets differ.
* **The most informative contrast is the gap to TF-IDF.** On CUAD, fine-tuned Legal-BERT beats the best TF-IDF model by
  **+7.7 macro-F1 points** (0.871 vs 0.794; paired contract-bootstrap CI [+0.047, +0.110], section 5). On LEDGAR the gap is
  essentially zero. Explanation: pre-trained knowledge pays off most when labelled data are few, and CUAD's categories depend
  on what a clause *does* (an exception to a restriction, a cap that limits the time to claim) rather than on topic words.
* **The learning-curve check supports the data-size part of this explanation** (`results/ledgar_learning_curve_gap.csv`,
  validation only, one seed). With only 5,000 LEDGAR training provisions, about our CUAD size, Legal-BERT leads TF-IDF by
  **+2.25 µ-F1 / +3.61 m-F1 points** (82.3 / 72.5 vs 80.1 / 68.9). With all 60,000 the lead shrinks to +0.16 / +0.28. So on the
  *same* task the advantage of fine-tuning grows when data are scarce. The remaining difference to CUAD's +7.7 is consistent
  with the second part (functional rather than lexical categories) but is not tested by this check.
* **What is identical:** the checkpoint (`nlpaueb/legal-bert-base-uncased`), the [CLS] + softmax head, the use of validation
  for every choice and a single test evaluation. **What differs:** the hyper-parameters. LEDGAR uses the paper's set-up (lr 3e-5,
  batch 8, length 512, early stopping); CUAD uses our own recipe from section 5 (lr 5e-5, batch 16, warm-up, weight decay,
  length 320, 5 epochs).
"""

INTERP["eval_table"] = r"""
**Interpretation (`results/task1_comparison_table.csv`, `results/task1_paired_bootstraps.csv`, Figure 31).**
* **Ranking on macro-F1 (test, 27 classes):**
  - fine-tuned **Legal-BERT 0.871** [0.836, 0.896];
  - BERT-base 0.826;
  - TF-IDF + LogReg 0.794 and LinearSVC 0.786 (their difference is not significant: +0.008 [−0.011, +0.024], section 4);
  - ComplementNB 0.713;
  - the prompted Qwen2.5-3B 0.556 and Qwen2.5-1.5B 0.451;
  - majority 0.005;
  - (Part F3) Legal-BERT with LoRA 0.829 [0.796, 0.862], between BERT-base and Legal-BERT (section 9).
  The robustness number over the 24 classes with ≥ 15 test clauses gives the same order. Every key paired comparison
  excludes 0: Legal-BERT − LogReg +0.077 [+0.047, +0.110]; LogReg − majority +0.788 [+0.746, +0.819]; Legal-BERT − BERT-base
  +0.045 [+0.018, +0.064]; Legal-BERT − each LLM +0.420 and +0.314; 3B − 1.5B +0.106.
* **Cost and speed (one basis):** the classical models cost about $0.00001 per 1,000 clauses and answer in under 1 ms; the
  fine-tuned transformers about $0.0002 and 6 ms; the prompted LLMs $0.008–0.015 and 126–190 ms. Training is cheap everywhere
  except in the searches: the final fits take 0.3–6 s (classical) and about 95 s (transformers), while the model selection took
  3,433 s (classical grid) and 1,262 s (transformer plan). The LLMs need no training but are 20–35× slower than Legal-BERT at
  inference.
* **Risk-bearing classes (`results/eval_risk_class_recall.csv`):** Legal-BERT has the highest recall for Cap On Liability
  (0.948) and Liquidated Damages (0.810), and ties with Qwen2.5-3B on Termination For Convenience (0.897). It is not the best
  everywhere: on **Change Of Control** its recall is only 0.633 (LogReg 0.673, Qwen2.5-1.5B 0.714). On Non-Compete it is 0.875
  vs LogReg 0.917. Change Of Control is the weak spot of the best model (section 7.5).
* **Confusion matrices (Figure 31):** the fine-tuned and classical models show clean diagonals with a few off-diagonal cells
  along the known neighbours. The LLM matrices show *column smears*, i.e. labels predicted for many classes (Non-Transferable
  License for the 3B model; Anti-Assignment, Exclusivity and Liquidated Damages for the 1.5B model). The majority baseline
  predicts Cap On Liability for every clause. Legal-BERT's full-size matrix is Figure 17.
"""

INTERP["eval_ablation"] = r"""
**Interpretation.** With everything else held constant, legal-domain pre-training improves test macro-F1 by **+0.045**
[+0.018, +0.064] (robust-24: +0.036). On validation (3 seeds each) the means are 0.848 vs 0.826. The sensitivity runs rule out
two easy explanations. The gain is not an artefact of a learning rate that suits only one model (both peak at 5e-5, and higher
rates are worse for both). Nor is it BERT-base needing more training: with 8 epochs the validation gap shrinks only from 0.015
to 0.014. The two tokenisers produce almost the same number of tokens per clause (69.3 vs 70.6), so sequence length is not
the reason either. Because Legal-BERT changes corpus and vocabulary together, the ablation measures the effect of *legal
pre-training as a package*, not of the vocabulary alone. Per-class winners are in section 5.8.
"""

INTERP["eval_errors"] = r"""
**Interpretation (`results/error_examples_task1.csv`, `results/error_patterns_task1.json`, `results/ambiguous_check.csv`).**
* **How many, how sure.** Legal-BERT misclassifies 158 of 1,444 test clauses (10.9 %). It is usually less sure when it is
  wrong: mean confidence 0.80 on errors vs 0.97 on correct predictions, and 51 % of the errors but only 7 % of the correct
  predictions have confidence below 0.9. A review tool could route low-confidence clauses to a human and catch half of the
  errors at a small cost (future work: calibration and abstention). Many errors are still confidently wrong: the seven examples
  all have confidence ≥ 0.99.
* **Causes of the seven examples** (verified by reading each clause):
  - overlapping legal functions (an assignment restriction triggered by a merger, labelled Change Of Control);
  - multi-label reality (an exclusive licence: Exclusivity and License Grant at once);
  - annotation boundaries (a 9-word fragment that starts mid-sentence; a span that joins a warranty and a liability limitation);
  - misleading surface words in very short clauses ("insurance carriers" in a commission clause; "extend" in an initial-term
    clause);
  - jargon (a right of first negotiation phrased as a renewal negotiation).
* **Common patterns.**
  1. *Functional neighbours.* The top confusions are Change Of Control → Anti-Assignment (10 clauses, 20 % of Change Of
     Control), Non-Transferable License → Anti-Assignment (6, 32 %), Exclusivity → License Grant (6) and IP Ownership Assignment
     → Post-Termination Services (5). They mirror the label combinations of the held-out multi-label clauses (License Grant +
     Non-Transferable License 163, Exclusivity + License Grant 69, Anti-Assignment + Change Of Control 22). These categories
     genuinely co-occur in the same sentences.
  2. *Hypotheses written before training (section 2.2).* Only one is confirmed on Legal-BERT's test errors:
     Non-Transferable License → Anti-Assignment (6 errors). Non-Transferable License → License Grant, No-Solicit → Non-Compete,
     Liquidated Damages → Cap On Liability and Joint → Assignment IP ownership produce 0 errors, and Competitive Restriction
     Exception produces 1 each. The fine-tuned model separates most of the pairs that confused the classical models.
  3. *Clause length matters only a little:* error rate 12.1 % for short clauses (≤ 31 words), 9.5 % medium, 11.2 % long.
     Short clauses lack context, and long ones more often mix functions.
  4. *Contracts:* the errors fall in 53 of the 93 test contracts, but the 5 worst contracts hold 42 % of them. Drafting style
     and annotation habits of single contracts matter, which is why we resample contracts in the bootstrap.
* **The 557 held-out multi-label clauses:** Legal-BERT predicts **one of the clause's valid labels for 93.0 %** (test-split
  clauses 91.8 %). On these clauses the model is mostly "right in one of the valid ways". Single-label scoring cannot credit
  this, and it slightly overstates real-world accuracy (honesty-log item 4).
"""

INTERP["eval_published"] = r"""
**Interpretation (`results/published_comparisons.csv`).**
* **Task 1 — LEDGAR (identical dataset, metrics and reporting rule).** Our Legal-BERT reaches 88.65 µ-F1 / 82.68 m-F1 vs the
  paper's 88.2 / 83.0, and our TF-IDF + SVM 88.09 / 82.84 vs 87.2 / 82.4. The published Legal-BERT values lie inside our 95 % CIs,
  so the result is reproduced within uncertainty. What is *not* identical: at most 10 vs 20 epochs, 3 vs 5 seeds, bf16 vs fp16,
  optimiser details the paper leaves unstated, and the hardware. The CUAD score cannot be compared with LEDGAR's (section 6).
* **Task 2 — CUAD paper (equivalent, not identical).** Same official test contracts and the same `evaluate.py` matching rule,
  but 27 vs 41 categories and a windowed tagger vs extractive QA. Our CUAD-style AUPR (52.4 %; 58.7 % for the low-threshold
  candidate list) is numerically above the paper's best (48.2 % RoBERTa-large, 47.8 % DeBERTa-xlarge). However, our curves stop
  at 58 % / 69 % recall, and our precision at 80 % recall is 0 vs 44.0 % for DeBERTa-xlarge. **We therefore do not claim to beat
  the paper.** The category difficulty ranking agrees moderately (Spearman 0.66).
"""

INTERP["eval_limitations"] = r"""
**Data and coverage.** CUAD contains English, US, SEC-filed commercial contracts, mostly from large public companies. The
models have not been tested on other jurisdictions, languages, private or small-business contracts, or recent drafting styles.
They should not be assumed to transfer, and they would be unfair to users whose contracts look different. The subgroup analysis
shows this already within CUAD (section 7.8): the franchise and licence families are clearly weaker, and the franchise family
comes from only 2 test contracts.

**Privacy.** The contracts are public SEC filings, but they name companies and individual signatories (and sometimes
addresses). We used the published dataset as distributed (CC BY 4.0) and identified no one. We created no new personal data,
and the raw data are not redistributed in the submission (notebooks download them from the official source).

**Over-reliance.** This is decision support for a human reviewer, not legal advice. The best classifier is wrong on 1 clause
in 9 and is sometimes wrong with confidence ≥ 0.99. The extractor misses about 4 in 10 annotated clauses (maximum recall 58 % at
its operating point). It must never be the only check, and its output should be shown with confidence scores and in context.

**Annotation boundaries and label noise.** CUAD spans range from 9-word fragments to whole sections. 557 clauses carry several
categories and are held out of Task 1. Several of the verified errors are annotation artefacts rather than model failures.
Task 2's exact-boundary F1 (0.30) is limited by the same granularity differences.

**Rare classes.** Source Code Escrow (4 test clauses), Joint Ip Ownership (10) and Non-Disparagement (12) have very wide
per-class intervals. Macro-F1 is noisy for them, which is why we report the robustness number over 24 classes and contract-level
CIs. We claim no differences the intervals cannot support.

**Honesty-log items:**
1. The label set (≥ 60 unambiguous clauses) was chosen with whole-corpus counts, including the test contracts' counts but no
   model performance.
2. **D7:** a complete 20-class dry run, including test evaluations, existed before the proposal was approved. It is archived
   and never reported, and the test log was reset for design v2. The v2 recipe was designed with knowledge of it (grids,
   epochs, learning-rate range), but every v2 choice was made on v2 validation data and every v2 model was scored on test once.
3. Legal-BERT's pre-training corpus includes EDGAR contracts, so memorised boilerplate may help it on CUAD. There is no label
   leakage, but this cannot be checked; the Qwen models' pre-training data are not documented in enough detail to rule out
   exposure either.
4. Task 1 is single-label although clauses can be multi-label.
5. The prompted LLMs are small (1.5 B and 3 B), so their results are a lower bound for prompting.
6. Single-label classes hide cross-category similarity (the top confusions above).
7. Task 2 is an equivalent, not a replication, of CUAD's QA set-up.
8. Anomaly flagging (Part F) has no ground truth.

**Licences.**
- CUAD: CC BY 4.0. LEDGAR / LexGLUE: CC BY 4.0 (dataset card).
- Legal-BERT: CC BY-SA 4.0. bert-base-uncased: Apache-2.0.
- Qwen2.5-1.5B-Instruct: Apache-2.0. **Qwen2.5-3B-Instruct: Qwen Research Licence** — research or evaluation use only, with an
  attribution notice. It could not be used in a commercial review product without a separate licence.
- Llama-3.2-3B was not used (gated).

**Compute.** About 5.0 GPU hours (RTX 5080) and 1.3 CPU hours of measured jobs are logged in `results/compute_budget.csv`,
excluding notebook re-runs of light cells. LEDGAR is the largest single item (3.0 h). We did not estimate energy or emissions
(no verified conversion source was used).

**Specific future work.**
1. Calibrated confidence with an *abstain → human review* option (half of Legal-BERT's errors have confidence < 0.9).
2. Multi-label Task 1, to credit clauses that are both, e.g., Exclusivity and License Grant.
3. Retrieval-augmented prompting tested as a pre-declared model (v4 reached 0.694 on validation), and larger LLMs.
4. Domain-adaptive pre-training on unlabelled contracts.
5. More jurisdictions and contract families.
6. A span-ranking extractor (QA-style n-best) to raise Task 2 recall beyond 69 %.
7. Active learning with reviewers on the most confused category pairs (Change Of Control / Anti-Assignment).
"""

INTERP["eval_subgroups"] = r"""
**Interpretation (`results/subgroups_task1.csv`).**
* **Clause length:** Legal-BERT is fairly stable across the terciles (accuracy 0.879 short / 0.905 medium / 0.888 long) and is
  better than TF-IDF + LogReg in every tercile. Short clauses are the hardest for both models (macro-F1 0.822 vs 0.743), because
  a few words give little context.
* **Contract family (≥ 50 test clauses each; "other", 27 clauses, is not reported):** Legal-BERT is strongest on services
  (accuracy 0.947), development / collaboration and distribution / supply (0.928 each), and weakest on **licence / IP (0.795)**
  and **franchise (0.789)**. Licence / IP contracts are full of the confusable licence categories (License Grant,
  Non-Transferable License, Exclusivity, IP Ownership Assignment). The franchise family comes from only **2 test contracts**, so
  it says as much about two documents as about franchises. Legal-BERT beats TF-IDF in every family, but the gap is smallest
  exactly in these two weak families (+0.016 and +0.023 accuracy).
* **Implication:** a reviewer working mainly on licence or franchise agreements should expect noticeably more errors than the
  overall 89 % accuracy suggests. Per-class results (section 5.8) and the confidence-based review routing (section 7.5) matter
  most there.
"""

INTERP["anomaly_results"] = r"""
**Interpretation — PART F1 (`results/anomaly_validation.csv`, `results/anomaly_test.csv`, Figure 32). Proxy-evaluated only,
not human-validated.**
* **Choice on validation:** within the "unusual wording" family, the **fine-tuned Legal-BERT embedding kNN** is best (mean
  AUROC 0.976 over random and hard injected pairs). It beats Mahalanobis (0.958), TF-IDF kNN (0.913) and the non-fine-tuned
  embedding kNN (0.848). The doubt score 1 − p(type) is higher still (0.987), but it measures a different thing and was not
  eligible as the headline.
* **Test (headline, evaluated once):**
  - **P1 injected wrong-type pairs:** AUROC **0.988** [0.982, 0.992] with random wrong types and **0.964** [0.955, 0.973] with
    hard ones. At the 5 % operating point it flags 92 % (random) and 79 % (hard) of the injected pairs. On genuine test clauses
    the flag rate is **7.6 %** instead of the 5 % set on validation, so the thresholds transfer only approximately.
  - **P2 the classifier's own errors** (scored against Legal-BERT's predicted type): AUROC 0.867 [0.840, 0.892]. Only 25 % of the
    158 errors are flagged at the operating point, while just 1.8 % of the correct predictions are.
  - **P3 the 557 multi-type clauses:** AUROC 0.714 [0.684, 0.745]. Clauses that genuinely belong to two types look more unusual
    for each of them than clean clauses do, as expected. Only 18 % are flagged.
* **The two families, side by side.** The doubt score is at least as good on every proxy (P1 0.993 / 0.983, P2 0.875, P3 0.849).
  That is no surprise: every proxy is about the *type*, and the classifier was trained for exactly that. The contrast inside
  the wording family is the informative part. With the **non-fine-tuned** Legal-BERT space the same kNN reaches only 0.882 /
  0.823 on P1, and P3 falls *below* 0.5 (0.430). Its scores also correlate negatively with clause length (Spearman −0.35):
  general embeddings mostly react to form. The fine-tuned space has learnt what distinguishes the 27 types, so "far from the
  type's training clauses" there means "far in a type-aware sense". The headline wording score is therefore **partly type
  information inherited from fine-tuning**, not pure surface oddity. The honest conclusion is that a clause-embedding kNN in the
  fine-tuned space detects mislabelled or atypical clauses well on proxies. Whether its flags match what a lawyer finds
  "unusual" is **not assessed**, because the blind review was not filled.
"""

INTERP["anomaly_inspection"] = r"""
**Length check (descriptive, not validation).** The headline score is almost independent of clause length (Spearman 0.10), and
34 of the 110 flagged test clauses have a normal length (27–63 words). The top flags are therefore not simply very short or very
long clauses.

**Blind review.** `results/anomaly_review_sheet.csv` holds 30 flagged and 30 random unflagged test clauses, shuffled, without
scores, with empty label columns (the key is in `results/anomaly_review_key.csv` — do not open it before reviewing). It has
**not** been filled, so the flags are **proxy-evaluated only, not human-validated**, and the human part of objective O6 is
**not assessed**.

**Responsible use.** A flag directs a reviewer's attention; it is not a legal conclusion. "Unusual" does not mean "wrong" or
"risky", and false flags cost reviewer time (7.6 % of genuine test clauses are flagged).
"""

INTERP["lora"] = r"""
**Interpretation — PART F3 (`results/lora_vs_full.csv`, `results/lora_results.json`, Figure 33).**
* **What LoRA saves:**
  - trains **0.61 M parameters (0.56 %)** instead of 109.5 M;
  - **11.4 s vs 17.4 s per epoch (−34 %)**;
  - **2.3 vs 3.7 GB peak GPU memory (−37 %)**;
  - **3.2 MB vs 439 MB** of saved weights (an adapter per task instead of a full model copy).
* **What it costs:** test macro-F1 **0.829** [0.796, 0.862] vs **0.871** for full fine-tuning. The paired difference is
  **+0.042** [+0.014, +0.061] in favour of full fine-tuning (validation: 0.821 ± 0.011 vs 0.848 ± 0.004 over 3 seeds). LoRA
  still beats the best classical model (0.794) and BERT-base (0.826). At batch size 1, the unmerged adapter is slightly
  *slower* at inference (7.9 vs 5.8 ms).
* **Why the gap:** within the fixed 5-epoch budget, the best learning rate was the top of the pre-declared grid (1e-3) and every
  LoRA run peaked in its last epoch (Figure 33, left). That suggested under-training, which the sensitivity runs below check.
* **Trade-off.** With a 110 M-parameter model and about 5,000 training clauses, full fine-tuning is cheap anyway (94 s on one
  GPU), so LoRA's savings matter little here. They would matter with many tasks or clients (one small adapter each), with
  larger models, or on GPUs with little memory.
"""

INTERP["lora_sensitivity"] = r"""
**Interpretation (`results/lora_sensitivity.csv`).** The gap is **not mainly under-training**:
* With 10 epochs at lr 1e-3, LoRA peaks at validation macro-F1 0.826 in epoch 7 and then declines. That is no better than the
  0.830 of the tested 5-epoch run: the longer schedule changes the learning-rate decay, and the extra epochs over-fit rather
  than help.
* With lr 3e-3 training diverges (macro-F1 about 0.006: the model predicts essentially one class).

So the remaining gap of about 0.02 macro-F1 on validation (0.04 on test) between LoRA r16 on query/value and full fine-tuning
looks like a **capacity limit of this LoRA configuration**: only 0.56 % of the weights adapt, and the feed-forward layers are
frozen. It is not a too-short schedule. More target modules or a higher rank would be the next things to try (not done).
"""

INTERP["demo_task1"] = r"""
Both models give the obvious type to the first three clauses (both with probabilities of 0.97 or more) and both read them as typical. The bare fragment "South Dakota" is uncertain for both
models and is flagged as unusual — the kind of fragment the anomaly scorer flagged among its top test flags. The demo shows how a
reviewer would use the outputs together: two independent opinions on the type, and a warning when a clause does not look like its
type. (Exact probabilities are printed above; the flags are proxy-evaluated only, not human-validated.)
"""
