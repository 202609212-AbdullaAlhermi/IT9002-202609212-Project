# Interpretation texts for notebooks/STUDENTID_LLM.ipynb, written after the cells were run; every number is read from
# results/ (llm_validation_summary.csv, llm_constrained_validation.csv, llm_test_comparison.csv, llm_paired_bootstrap.csv,
# llm_per_class.csv, llm_cost.json, llm_models.csv).

INTERP["models"] = r"""
**What we use.** Qwen2.5-1.5B-Instruct (1.54 B parameters, Apache-2.0) and Qwen2.5-3B-Instruct (3.09 B, Qwen Research Licence:
research or evaluation use only, with the required attribution notice above). Both have a 32,768-token context, far more than
our longest prompt (about 2,400 tokens), so no clause or example has to be cut. Both models are from the same family and share
one tokenizer. The comparison between them therefore isolates **model size** (same training recipe, same vocabulary) but does not
compare two model families as the plan with Llama would have (deviation D3). The licence of the 3B model is more restrictive
than Apache-2.0; for a commercial contract-review product it could not be used without a separate licence.
"""

INTERP["validation"] = r"""
**Interpretation (`results/llm_validation_summary.csv`, Figure 26).**
* **Size matters most.** With every prompt version, the 3B model beats the 1.5B model on validation macro-F1: v1 0.522 vs 0.438,
  v2 0.563 vs 0.441, v3 0.417 vs 0.330.
* **Descriptions help a little; the few-shot prompt hurts.** v2 (one-line descriptions) is the best version for **both** models
  and is chosen for the test run. The gain over names only is small for the 1.5B model (+0.004) and larger for the 3B model
  (+0.041). The few-shot prompt v3 is clearly the worst for both (−0.11 and −0.15 against v2). Its predictions collapse onto a few
  labels: on validation the 3B model answers *Non-Compete* for 417 of 1,026 clauses, and the 1.5B model over-uses
  *Anti-Assignment* (200) and *Termination For Convenience* (128). A block of 27 examples (about 1,900 extra tokens) seems to
  distract these small models more than it informs them. Section 6.2 shows that the problem is the *fixed* block, not
  few-shot prompting itself: three *relevant* examples raise the 3B model to 0.694 on validation.
* **Prompt length costs time.** The fixed prompt grows from 216 tokens (v1) to 663 (v2) and 1,895 (v3); GPU time per clause grows
  with it, 0.016 → 0.035 → 0.107 s (1.5B) and 0.028 → 0.069 → 0.200 s (3B).
* **Invalid outputs are rare and almost never ambiguous.** 0.6 % (1.5B) and 4.3 % (3B) of the validation answers are invalid
  with the chosen prompt. Every invalid answer names *no* label at all (none names two). They are plausible legal labels that
  are simply not on our list: "Non-Solicitation Of Employees" (the list says *No-Solicit Of Employees*), "Non-Assignment",
  "Royalty Duration", "Royalty Payment". More than 94 % of the answers match a label exactly, so the parser rarely has to guess.
"""

INTERP["constrained"] = r"""
**Interpretation (`results/llm_constrained_validation.csv`; validation only, not used for any choice).** Forcing the answer to be
one of the 27 labels changes almost nothing: macro-F1 0.441 → 0.439 (1.5B) and 0.563 → 0.568 (3B). Replacing only the invalid
answers by the constrained choice gives 0.440 and 0.571. The constrained choice agrees with the free answer on 98 % of the valid
answers, which also confirms that greedy generation already picks the most likely label. **So invalid outputs explain less than
0.01 of the gap of about 0.3 to the fine-tuned model:** the models really do not know which of our 27 categories a clause
belongs to. They are not failing to format a correct answer.
"""

INTERP["test"] = r"""
**Interpretation (`results/llm_test_comparison.csv`, `results/llm_paired_bootstrap.csv`).** On the 1,444 test clauses the
chosen prompts score **0.451** [0.407, 0.475] (Qwen2.5-1.5B, v2) and **0.556** [0.518, 0.587] (Qwen2.5-3B, v2), close to their
validation scores (0.441 and 0.563), so the prompt choice did not over-fit validation. Invalid answers: 0.6 % and 5.1 %. The
fine-tuned Legal-BERT (0.871) is far better than both: **+0.420** [+0.386, +0.467] over the 1.5B model and **+0.314** [+0.274,
+0.351] over the 3B model. Even the TF-IDF + logistic regression baseline (0.794) beats the 3B model by +0.238 [+0.195, +0.281].
Doubling the model size helps clearly: 3B − 1.5B = **+0.106** [+0.081, +0.144]. In every comparison the confidence interval
excludes 0 (no resample reverses the order). The robustness number over the 24 classes with ≥ 15 test clauses tells the same
story (0.431 and 0.550 vs 0.866 for Legal-BERT).
"""

INTERP["perclass"] = r"""
**Interpretation (`results/llm_per_class.csv`, Figures 27–28).**
* **What prompting handles.** Categories whose names say what the clause says: *Governing Law* (3B: F1 0.98), *Audit Rights*
  (0.92), *Insurance* (0.92), *Renewal Term* (0.88), *Expiration Date* (0.81). Here the 3B model is within about 0.1 of Legal-BERT.
* **What it cannot do.** The categories defined by CUAD's annotation conventions rather than by everyday meaning:
  *Competitive Restriction Exception* (F1 0.00 for the 3B model; it is an *exception to* a restriction, and the model labels it as
  the restriction itself, e.g. *Non-Transferable License*, *Non-Compete* or *Exclusivity*), *Post-Termination Services* (0.08;
  predicted as *Termination For Convenience* or *Non-Transferable License*), *Volume Restriction* (0.20), *Rofr/Rofo/Rofn* (0.23;
  jargon), *Non-Transferable License* (0.17; massively over-predicted: 111 predictions for 19 true clauses).
* **Label semantics.** Several errors are reasonable readings of the label name that differ from CUAD's definition. A cap on
  liability that limits the *time* to bring claims is answered as *Warranty Duration* (21 of 115); a ban on soliciting employees
  is labelled *Non-Compete*; restrictions on assigning the contract become *Non-Transferable License*. A fine-tuned model learns
  CUAD's conventions from 5,315 examples; a prompted model only has the names and one-line descriptions.
* **Descriptions vs names (validation, by class).** Descriptions help most where the name is jargon or misleading: *Rofr/Rofo/Rofn*
  0.17 → 0.40 and *Covenant Not To Sue* 0.00 → 0.27 for the 3B model. They hardly change the easy classes. The 1.5B model over-uses
  *Liquidated Damages* and *Exclusivity* as "catch-all" answers (106 and 115 test predictions for 21 and 44 true clauses).
"""

INTERP["cost"] = r"""
**Interpretation (`results/llm_cost.json`, Figure 29).** On one common basis (measured time × the stated hourly price):
* **Latency at batch size 1:** TF-IDF 0.5 ms (CPU), Legal-BERT 5.8 ms, Qwen2.5-1.5B 126 ms and Qwen2.5-3B 190 ms per clause
  (median; 95th percentile 0.9 / 6.5 / 171 / 257 ms). The LLMs are 20–35 times slower than Legal-BERT because they read a
  ~730-token prompt and then *generate* the answer token by token, while Legal-BERT reads the clause once and outputs 27 scores.
* **Throughput (batched):** about 4,800, 1,000, 28 and 15 clauses per second.
* **Peak GPU memory:** Legal-BERT 0.6 GB at batch 1; the LLMs 7.0 and 11.2 GB at batch 32. The 3B model needs most of a 16 GB card.
* **Cost per 1,000 clauses (batched):** TF-IDF about $0.00001, Legal-BERT $0.0002, Qwen2.5-1.5B $0.008, Qwen2.5-3B $0.015. At
  batch size 1 the figures are $0.00003, $0.0013, $0.028 and $0.043. All of these are tiny in absolute terms: even the 3B model
  classifies 1,000 clauses for under 5 cents. The *relative* gap is large (the 3B model costs about 70 times more per clause than
  Legal-BERT in batch) and grows with volume. On the other side of the ledger, the LLMs need **no training** (Legal-BERT: 94 s on
  the GPU, plus the labelled data and the hyper-parameter search of notebook 1).
* **Caveats:** the price is the AWS on-demand price of a g6.xlarge (NVIDIA L4) and a c7i.xlarge in us-east-1 (official AWS
  EC2 On-Demand pricing page, read on 2026-10-22; Holori and DoiT give the same figures), applied to timings measured on an RTX
  5080 and a desktop CPU. An L4 would likely be slower than the RTX 5080, so the true cloud cost would be higher
  (by the same factor for all GPU models). The API fee is 0 because the models run locally.
"""

INTERP["discussion"] = r"""
On our task the fine-tuned Legal-BERT is better (0.871 vs 0.556 macro-F1), faster (6 vs 190 ms per clause) and cheaper per clause
than the best prompted model. The prompted model is still the better choice in some situations:
1. **No labelled data yet.** (With some labelled examples, retrieving the most similar ones into the prompt helps a lot:
   0.694 vs 0.563 on validation for the 3B model, section 6.2.) Legal-BERT needed 5,315 expert-labelled training clauses (CUAD's annotation effort was very large). A
   prompted model gives a usable first classifier on day one, without any labels: 0.56 macro-F1 overall and above 0.8 F1 on
   categories whose names are self-explanatory, e.g. *Governing Law*, *Insurance*, *Audit Rights*, *Renewal Term*.
2. **A new or changing label set.** Adding a category (e.g. "Data Protection", which the model even produced unprompted) means
   editing one line of the prompt, not re-annotating and re-training.
3. **Rapid prototyping and low volume.** For a few hundred clauses the cost difference is fractions of a cent and no training
   pipeline has to be built or maintained.
4. **Explanations and interaction.** A generative model can be asked *why* it chose a label or to quote the deciding phrase; our
   classifier only outputs scores. (We did not evaluate the quality of such explanations.)
5. **Bootstrapping annotation.** Its suggestions can pre-label clauses for a human annotator, who corrects them; the corrected
   labels then train a fine-tuned model.

The fine-tuned model is preferable as soon as labelled data exist and the label set is stable, the volume is high, latency
matters, or the categories follow conventions that differ from their everyday names (*Competitive Restriction Exception*,
*Post-Termination Services*, *Rofr/Rofo/Rofn*). In a review setting where a missed risk clause is costly, the 0.3 lower
macro-F1 of the prompted models is decisive.
"""

INTERP["conclusion"] = r"""
* **Result:** two small open instruction-tuned models, prompted without training, reach **0.451** (Qwen2.5-1.5B) and **0.556**
  (Qwen2.5-3B) test macro-F1 on the 27-way task, against **0.871** for the fine-tuned Legal-BERT and 0.794 for TF-IDF + logistic
  regression. All differences have bootstrap confidence intervals clear of 0. One-line category descriptions were the best prompt
  for both models; few-shot examples made them worse.
* **Why the gap:** not formatting. Invalid answers are rare and constraining the output to the 27 labels changes macro-F1 by less
  than 0.01. The errors come from CUAD-specific category definitions that the label names and short descriptions do not convey.
* **Limitations:** (1) small models only (1.5 B and 3 B), so this is a **lower bound** for prompting; larger hosted models were not
  tested (no API key, by design); (2) both models are from one family (Llama was not accessible, D3); (3) one fixed label order
  and one example per class in v3 (retrieving similar examples instead, v4, was much better on validation but was not
  tested); (4) the cost figures
  rest on a price assumption for a different GPU; (5) the 3B model's research-only licence limits practical use.
* **Future work:** testing the retrieval few-shot prompt (v4, 0.694 on validation; section 6.2) as a pre-declared model on a
  fresh test set, larger models, chain-of-thought or two-step prompting (first a family, then the category), and LoRA instruction tuning of a small LLM
  on the training clauses.
"""

INTERP["retrieval"] = r"""
**Interpretation (`results/llm_retrieval_v4_validation.csv`; validation only, analysis).** With the v2 descriptions plus the
**3 most similar training clauses**, Qwen2.5-3B reaches **0.694** validation macro-F1: +0.131 over v2 (0.563) and +0.277 over
the fixed few-shot block v3 (0.417). Invalid answers drop to 0.1 %, and the prompt is shorter than v3 (926 vs 1,961 tokens on
average). This explains why v3 hurt. Its 27 examples are mostly irrelevant to the clause, long, and always in the same order, and
the small model latches onto one of them (*Non-Compete* for 417 clauses). In v4 the few examples are the *nearest* ones. The
model then largely copies the nearest neighbour's label: v4's answer equals the label of the most similar training clause for
82 % of clauses, and when the true label is among the three neighbours (82.5 % of clauses) v4 is right 89 % of the time, against
only 9 % otherwise. The retrieval-only baselines (no LLM, same retriever, `results/llm_retrieval_baselines_validation.csv`)
confirm this: the nearest training clause's label alone gives **0.655** macro-F1 and a majority vote of the 3 neighbours
**0.661**, so about **75 %** of the v2 → v4 gain (+0.131) is achieved by retrieval alone. The LLM adds only +0.033 on top, by
choosing among the neighbours' labels with the help of the descriptions. v4 is thus essentially a nearest-neighbour classifier
with the LLM as a judge, and its success says more about the value of labelled examples than about the LLM's own
understanding of the categories. **As decided before this
analysis, the tested model stays v2** (one test evaluation per model); v4 was not run on test, and its validation score is
not a test result.
"""
