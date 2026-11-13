# Interpretation paragraphs for notebooks/STUDENTID_Task2.ipynb, written after looking at the real outputs (2026-10-22).

INTERP["windows"] = r"""
**Interpretation (`results/task2_window_choice.csv`).** Contracts have a median of 6,642 word-pieces and up to 78,958, so
every contract needs many windows. With 128-token windows, 10 % of the gold spans are longer than a window and only 80 % fit
completely inside one; with **256 / 128** only 1.7 % are longer than a window and **95.7 %** fit completely inside at least one
window; larger windows (384, 510) raise this to 98–99.5 %. Training time per epoch is almost the same for 256, 384 and 510
(0.70–0.77 min: fewer but longer windows), while memory grows from 3.2 to 4.7 GB. We use **256 / 128**, the setting fixed in
the design brief before these statistics were computed: it keeps almost all spans inside one window, and the rest are still
covered token by token, because every token receives averaged probabilities from the (two) windows that contain it. Larger
windows are a cheap option for future work (more context per prediction).
"""

INTERP["labels"] = r"""
**Interpretation.** 910 of the 4,670 neighbouring same-category span pairs (19.5 %) touch or overlap, which is why BEGIN
targets are needed (Figure 19 shows two touching *Audit Rights* spans: the INSIDE row is one continuous block, only the BEGIN
row shows where the second clause starts). Positive targets are very rare — 0.43 % of the (token, category) INSIDE targets in
training — so the model has to be trained against a strong pull towards "never inside".
"""

INTERP["oracle"] = r"""
**Interpretation (`results/task2_oracle.json`).** With INSIDE + BEGIN, decoding the gold labels gives macro-F1 **0.998** at IoU 0.5
on test (0.995 at exact token boundaries): the formulation and the evaluation code are essentially lossless (the small loss
comes from rare cases such as two spans of one category sharing a token). With INSIDE only, the ceiling falls to **0.932**
(0.870 exact), because touching clauses are merged into one span. So BEGIN markers raise the achievable score by about 0.07,
before any model is trained.
"""

INTERP["baseline"] = r"""
**Interpretation (`results/task2_baseline_search.csv`, `results/task2_segment_oracle.json`).** The heuristic segmentation alone
caps the baseline at macro-F1 **0.859** on test even with a perfect classifier: neighbouring clauses of one category are merged
into one span and segment edges do not always coincide with clause edges. The search over 108 settings (C up to 100, thresholds
0.1–0.9; the first, narrower grid had its optimum on the edge and was extended) selected C = 30 without class weights and a
**low decision threshold of 0.2** (validation macro-F1 0.432; C = 10–100 within 0.002). A low threshold is needed because each
category's segments are rare, so the classifiers' probabilities for positive segments are small; balanced class weights raise
recall but produce too many false positives (best 0.32).
"""

INTERP["tagger_training"] = r"""
**Interpretation (Figure 21, `results/task2_search.csv`).**
* **Variant choice (pre-declared rule):** plain BCE (validation macro-F1 0.458 after 4 epochs) beat focal loss (0.397) and
  positive-weighted BCE (0.279). Positive weighting makes the model predict "inside" far too often (macro recall 0.65 but precision
  0.19), which the span metric punishes; focal loss learns faster in epoch 1 but levels off lower. Negative-window sampling alone
  already handles the imbalance well enough for plain BCE.
* **Epochs:** plain BCE was still improving at epoch 4, so the rule recorded before the other variants were seen
  retrained it for 8 epochs: seed 42 rose to **0.483** (best epoch 6), and 8 epochs were adopted. With seeds 43 and 44 the 8-epoch
  recipe scores **0.489 ± 0.006** on validation (4 epochs: 0.440 ± 0.016).
* **Curves:** the score is near zero after epoch 1 (the model first learns that almost everything is outside), jumps in epochs 2–3,
  and flattens from epoch 6 while the training loss keeps falling — further epochs would mostly fit the training contracts.
* One run takes about 7 minutes for 8 epochs (46 s per epoch plus 5 s for decoding the validation contracts; peak 3.3 GB).
"""

INTERP["thresholds"] = r"""
**Interpretation (`results/task2_threshold_search.csv`, Figure 22, `results/task2_thresholds.json`).** Tuning the decoder on
validation raises macro-F1 from 0.483 (defaults) to **0.565**: θ_inside = 0.6 and **min_tokens = 10** — dropping very short
fragments removes many spurious pieces — while θ_begin makes no difference (0.3, 0.5 and 0.7 give identical scores, because the
BEGIN probabilities are almost always either close to 0 or close to 1; 0.3 was kept by the tie-break order). **Per-category
thresholds were not adopted:** in the split-half check they helped in one direction (0.506 → 0.527) but hurt in the other
(0.608 → 0.572), the typical sign of over-fitting a handful of validation spans per category.
"""

INTERP["test"] = r"""
**Interpretation of the test results (each model scored once; `results/task2_test_summary.csv`).**
* **The Legal-BERT tagger reaches macro-F1 0.523 at IoU 0.5** (95 % CI 0.490–0.560; micro-F1 0.575, macro precision 0.60, recall
  0.49) against **0.394** (0.367–0.431) for the TF-IDF segment baseline. **Objective O4 (first part) is met:** the paired
  contract-level bootstrap difference is **+0.129, CI [+0.099, +0.155]**.
* **Far below the ceilings:** the formulation allows 0.998 and the baseline's segmentation 0.859, so the gap is the models, not the
  format. Exact boundaries are much harder (tagger 0.300): the model usually finds the right region but not the annotator's exact
  first and last word.
* **No false alarms on clause-free contracts:** in the 9 test contracts without any of the 27 clause types, neither model predicts a
  single span.
* **Per category (Figure 23):** formulaic clauses are found well — Governing Law 0.88, Expiration Date 0.83, Renewal Term 0.73,
  License Grant 0.70 — while Competitive Restriction Exception (0.23), Non-Disparagement (0.29), Rofr/Rofo/Rofn (0.30) and
  Post-Termination Services (0.34) are hard. The tagger beats the baseline in 23 of the 27 categories (equal on Non-Transferable
  License; worse on IP Ownership Assignment, No-Solicit Of Employees and, by 0.001, Governing Law); the baseline finds no Competitive
  Restriction Exception or Source Code Escrow span at all.
* **Subgroups (`results/task2_subgroups.csv`):** both models do best on short contracts (tagger 0.63 vs 0.50 on long ones — more
  text means more places for false positives and for missed clauses) and on frequent categories (0.62 for the most frequent third
  vs 0.44 for the rarest third).
* **Budget:** training 7.2 min for the test run (8 epochs) and 39 min for the whole plan including variants and seeds; inference
  0.063 s per contract on the GPU (about 1 minute per 1,000 contracts).
"""

INTERP["published"] = r"""
**Interpretation (`results/task2_cuad_comparison.csv`).** On the CUAD-style evaluation our tagger's AUPR is **52.4 %** (baseline
40.2 %), numerically above every model in the paper's Table 2 (best: RoBERTa-large 48.2 %, DeBERTa-xlarge 47.8 %). **This does
not mean it is better**, for three reasons:
1. **The curves cover different recall ranges.** The paper's QA models output many candidate answers per category with their
   probabilities, so the confidence sweep reaches high recall (DeBERTa-xlarge: 44.0 % precision at 80 % recall, 17.8 % at 90 %).
   Our decoder outputs a fixed set of spans, so our curve stops at the recall those spans reach — **58.1 %** for the tagger (47.9 %
   for the baseline) — and precision at 80 % / 90 % recall is therefore **0**. Our AUPR is high because precision is high over the
   part of the curve we cover (Figure 24), not because we find more clauses. Even the low-threshold candidate set of section 7.1
   (AUPR 58.7 %) stops at **69.0 %** recall and still has 0 precision at 80 % recall.
2. **Different category sets:** 27 vs 41 categories; several of the 14 we exclude (e.g. Parties, Document Name, dates) are easy,
   others (e.g. Most Favored Nation, Uncapped Liability) are rare and hard.
3. **Different models and training:** a windowed multi-label tagger on Legal-BERT-base trained on 343 contracts (65 held out for
   validation) vs extractive QA models up to DeBERTa-xlarge trained on all non-test contracts.
**Per category**, the paper gives only a bar chart (Figure 4), so we compare ranks: the Spearman correlation between the paper's
order of categories by AUPR and our tagger's per-category AUPR is **0.66** over the 27 shared categories. Largely the same
categories are easy (Governing Law, Expiration Date, Anti-Assignment, Cap On Liability) and hard (Competitive Restriction Exception,
Change Of Control, Post-Termination Services, Rofr/Rofo/Rofn, IP Ownership Assignment) for both approaches, which suggests that much
of the difficulty lies in the clauses themselves. The clearest disagreement is Covenant Not To Sue: the hardest category in the
paper's figure, but mid-table for our tagger (AUPR 0.64).
"""

INTERP["taxonomy"] = r"""
**Interpretation (`results/task2_error_taxonomy.csv`, Figure 25).** Of the tagger's 798 missed gold spans, **464 are simply
missed** (no prediction of that category nearby), **157 are merged** (one prediction covers several touching gold spans of the same
category — CUAD annotators often highlight each sentence of a clause separately, the model predicts the whole clause), 82 are found
under a **wrong category**, 70 have a **boundary that is too short** and 21 are split or too long; only 11 involve gold spans longer
than one 256-token window, so the window size is not a main cause. Of the 507 extra predictions, **310 are spurious** (they overlap
no gold span — some are genuine clauses the annotators did not mark, see 8.2), 139 are the counterpart of boundary, split and merge
errors and 58 are a wrong category. Compared with the baseline, the tagger misses far fewer spans (464 vs 688) and makes fewer
spurious predictions (310 vs 418), but it merges touching spans more often (157 vs 108).
"""

INTERP["examples"] = r"""
**Interpretation of the ten examples.**
1. *Governing Law* — found exactly (IoU 1.00, confidence 0.96): a formulaic clause.
2. *Cap On Liability* — the classic "IN NO EVENT SHALL ANY PARTY BE LIABLE…" exclusion is found exactly, but two clauses saying that
   no party guarantees profitability ("no indemnification shall arise…") are missed: unusual wording for a liability cap.
3. *License Grant* — the annotated grant is found exactly, plus an almost identical grant paragraph earlier in the same contract that
   is not annotated: counted as a false positive although it reads like a genuine licence grant (incomplete annotation).
4. *Non-Transferable License* — the right clause, but the prediction stops after 338 of 832 characters (IoU 0.41 < 0.5): a boundary
   error that the IoU criterion counts as both a miss and a false positive.
5. *Exclusivity* — one clause found exactly; a second prediction is only the fragment "PPI hereby grants EKR the exclusive" (the
   rest of the sentence fell below the inside threshold); the "appoints … as the exclusive distributor" clause is missed.
6. *Change Of Control* — the model marks only the sentence that defines a change of control as an assignment (IoU 0.14), while the
   annotator highlighted the whole long anti-assignment section: a disagreement about granularity.
7. *Liquidated Damages* — two of three clauses found exactly; the early-termination-fee clause is missed, and a definition of
   "Estimated Remaining Value" is predicted instead (related wording, wrong place).
8. *Renewal Term* — the clause is split into two predictions; the first just reaches IoU 0.50 and counts as correct, the second is a
   false positive, and a capitalised "intent not to renew" notice elsewhere is a further false positive.
9. *Audit Rights* — found exactly.
10. *Competitive Restriction Exception* — neither "Notwithstanding … shall not apply" exception is found: short, redacted ("[***]")
    clauses whose meaning depends on the clause they modify.
"""

INTERP["errors"] = r"""
**Causes of the five errors (checked by reading the excerpts).**
1. **Missed (License Grant):** "Such grant by PPI shall include the right of EKR to market the Product…" continues the previous
   grant sentence without the usual cue words ("hereby grants", "license"); the model found the neighbouring grants but not this
   follow-on sentence.
2. **Boundary too short (Minimum Commitment):** the gold span starts at the heading "Guaranteed Minimum Royalty during the Second
   Additional Term" and includes the payment table; the prediction starts one sentence later and stops before the table of amounts,
   which has no clause-like wording (IoU < 0.5).
3. **Split (Non-Compete):** one non-compete sentence is predicted as two pieces with confidence 0.65, just above the 0.6 threshold —
   tokens in the middle ("or otherwise authorize the use of her name, likeness or image in connection") fell below it.
4. **Wrong category (Liquidated Damages instead of Cap On Liability):** inside a long Cap On Liability clause, the text about an
   "Additional Commission Amount" payable on termination reads like a liquidated-damages payment; the confusable money-on-termination
   wording misleads the model.
5. **Merged (Insurance):** four touching gold spans (the annotators highlighted each sentence of the insurance clause separately) are
   predicted as one long span (confidence 0.97). The BEGIN probabilities did not fire at the sentence starts, because in many
   training contracts the same kind of clause is annotated as one span.
**Common patterns:** annotation granularity (merge/split and long-section vs single-sentence highlighting), boundaries at headings
and tables, follow-on sentences without cue words, and money-related wording shared by Cap On Liability, Liquidated Damages and
Revenue/Profit Sharing.
"""

INTERP["curves_pr"] = r"""
**Interpretation (Figure 24).** Left: on test, the tuned tagger keeps precision above about 85 % up to roughly 45 % recall
(CUAD-style counting) and its curve ends at 58 % recall; the baseline's precision falls steadily and ends at 48 % recall. The
low-threshold candidate list follows the tuned curve almost exactly up to 58 % recall, then continues to 69 % recall while its
precision drops to about 35 %. The crosses mark the paper's precision at 80 % and 90 % recall (41 categories). None of our curves
reaches that region, which is the main reason the AUPR numbers cannot be compared like-for-like. Right: on validation, raising
θ_inside trades recall for precision; precision climbs steeply while recall falls only slowly, which is why the F1 optimum sits
at the top of the range (0.6).
"""

INTERP["demo"] = r"""
The demo runs the saved tagger on a short made-up text in under a second and finds all five clauses, including the double label
*License Grant* + *Non-Transferable License* on the same sentence (multi-label output). On the real test contract, chosen by the
fixed rule above, each extracted clause is checked against the lawyers' spans: the table shows which extractions match a gold span
(IoU ≥ 0.5) and which gold types were not found, so the demo shows both what the tagger gets right and what it misses. On a
contract that contains none of the 27 clause types it returns nothing.
"""

INTERP["conclusion"] = r"""
* **What was built:** clause extraction as multi-label sequence labelling over sliding windows, with INSIDE + BEGIN targets
  (formulation ceiling 0.998), a TF-IDF segment baseline (segmentation ceiling 0.859) and a Legal-BERT tagger.
* **Result:** the tagger reaches **0.523 macro-F1 at IoU 0.5** on the 102 official test contracts (baseline 0.394; difference +0.129
  [+0.099, +0.155], objective O4 met for the baseline comparison), with no false alarms on clause-free contracts, and high precision
  over the recall range it reaches (CUAD-style AUPR 52.4 %, maximum recall 58 %).
* **Limitations:** (1) annotation boundaries are subjective — sentence-by-sentence vs whole-section highlighting explains many
  merge/split and boundary errors, and exact-boundary F1 is only 0.30; (2) only 27 of CUAD's 41 categories; (3) English, US,
  SEC-filed contracts; (4) 256-token windows lose long-range context (a clause whose meaning depends on a distant definition);
  (5) rare categories have 5–27 test spans, so their scores are very uncertain; (6) recall is capped: at the tuned setting the
  curve ends at 58 % recall, and even the low-threshold candidate list (section 7.1) only reaches 69 %, because the model gives
  14 % of the gold spans (validation) no token probability above 0.05 — unlike the QA models of the CUAD paper, we never reach
  80 % recall; (7) a 384/192 window did not help on validation (one seed).
* **Future work:** a model that ranks candidate spans directly (e.g. an extractive-QA head with n-best answers per category) to
  reach higher recall; longer windows or a long-document
  model; a span-merging rule learned from the annotation style; and using Task 1's classifier to re-check the category of each
  extracted span (Part F pipeline).
"""

INTERP["window_sensitivity"] = r"""
**Interpretation (`results/task2_window_sensitivity.csv`; validation only).** Longer windows fit more gold spans completely
(98.4 % vs 95.7 % of spans inside at least one window) at slightly higher cost (460 vs 432 s of training, 4.1 vs 3.3 GB). At the
default thresholds the two window sizes are practically equal (0.485 vs 0.483 macro-F1 at IoU 0.5). After each model's own
threshold tuning the 384/192 model is **0.020 lower** (0.545 vs 0.565). That is a single seed each, and the 256/128 recipe alone
varies with std 0.006 across seeds, so the honest reading is "no evidence that longer windows help". The few extra spans that now
fit in one window are not where the errors are: only 11 of the tagger's 798 missed test spans are longer than a 256-token window
(section 8.1). The pre-declared 256/128 test model stays as it is; no test evaluation was made with 384/192.
"""

INTERP["recall_ceiling"] = r"""
**Interpretation (`results/task2_recall_ceiling.json`, `results/task2_lowthr_test.csv`).**
*Why recall stops near 58 %.* The decoder only turns tokens whose INSIDE probability is at least θ_inside into spans. On
validation, **28.9 %** of the gold spans contain *no* token with probability ≥ 0.6 for their category, so at the tuned setting
they cannot be found at all, whatever confidence cut-off the CUAD sweep uses. Of the remaining spans, some are merged with a
neighbour or found with boundaries too far off for the overlap criterion. Lowering θ_inside raises recall only slowly: CUAD-style
maximum recall goes from 0.607 (θ 0.6) to 0.694 (θ 0.05), and IoU-recall from 0.562 to 0.619, while precision at IoU 0.5 falls
from 0.656 to 0.311. The reason is that the model is *confidently wrong* on many missed spans: **14.1 %** of the validation gold
spans have no token above 0.05. No threshold can recover these; a different model, or more context, would be needed.

*The low-threshold candidate set.* The selection rule picked θ_inside = 0.05, the lowest value of the pre-declared grid. The
length condition never bound (low thresholds produce more, not longer, spans because `min_tokens` = 10 stays in force). The
recall gain flattens at the bottom of the grid (0.689 at θ 0.1 vs 0.694 at 0.05), so extending the grid further would add little.
Evaluated once on test as `t2_legalbert_tagger_lowthr`, the candidate list reaches **69.0 %** maximum recall (tuned: 58.1 %) and a
CUAD-style AUPR of **58.7 %** (tuned: 52.4 %). Precision at 80 % and 90 % recall is still 0: the curve never gets there. Its
macro-F1 at IoU 0.5 when every candidate is kept is only 0.410. That is expected (it is a ranked list for a reviewer, not an
operating point), and it is why **0.523 stays the main result**. In practice the two settings serve different users: the tuned
setting for an automatic first pass with few false alarms, the low-threshold list for a reviewer who would rather check more
candidates than miss a clause.
"""

INTERP["evaluation"] = r"""
**Summary.** The Legal-BERT tagger is the Task 2 result: 0.523 macro-F1 at IoU 0.5 (exact boundaries 0.300), clearly above the
TF-IDF segment baseline (0.394; paired difference +0.129 [+0.099, +0.155]) and well below the label-format ceiling of 0.998, so
the remaining errors come from the model and the annotation conventions, not from the INSIDE+BEGIN format (without BEGIN the
ceiling would already drop to 0.932). The low-threshold row is an analysis of the ranking only (69 % maximum recall, AUPR 58.7 %);
it is not an operating point and its macro-F1 is not a competing result.
"""

INTERP["pipeline"] = r"""
**Interpretation — PART F2 (`results/pipeline_validation_grid.csv`, `results/pipeline_results.json`).**
* **Verification does not help the extractor.** On validation no combination of a < 1 and τ > 0 beats the extractor alone
  (0.565), so the pre-declared rule chose **a = 1, τ = 0**: the tested pipeline keeps every extracted span and equals the
  extractor (test macro-F1 0.523, AUPR 52.4 %, paired difference 0). Re-ranking by the verifier's confidence lowers test AUPR
  (analysis only: 52.4 % at a = 1, 52.2 % at a = 0.5, 48.9 % at a = 0). This is an honest negative result, explained by the
  next point.
* **Error propagation runs the other way:** the classifier is *more* accurate on correctly located predicted spans (macro-F1
  0.959 on 773 spans) than on all gold clause texts (0.871 on 1,457 single-category spans). The spans the tagger locates
  correctly are the typical, easy ones on which tagger and classifier already agree. The classifier has little to add, and
  the hard cases are the ones the tagger misses entirely, which verification cannot recover.
* **Multi-category spans:** 225 test annotations sit on spans with several categories. The extractor finds 111 of them, and none
  is dropped by single-label verification at the chosen setting. A stricter τ would start dropping them, because the
  classifier spreads probability over the span's categories.
* **Cost:** about 0.075 s per contract on the GPU (extraction 0.063 s + verification and flagging 0.012 s), i.e. about
  **$0.017 per 1,000 contracts** at the AWS g6.xlarge price (read 2026-10-22; same basis as the LLM comparison).
* **What the pipeline adds is the review list.** In the demo contract, 5 spans are kept and 1 is flagged as unusual: a
  Termination For Convenience span whose text the classifier does not support (p = 0.001; it reads like an automatic-termination
  clause). That is exactly the kind of item a reviewer should look at first. The flags are proxy-evaluated only, not
  human-validated (Part F1).
"""
