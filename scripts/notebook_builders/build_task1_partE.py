# Section 7 of notebooks/STUDENTID_Task1.ipynb: evaluation and analysis (Part C).
# Reads saved results/predictions (plus one cached Legal-BERT confidence pass), so it re-runs in about a minute.
# Executed by build_task1_partA.py (exec), so md(), code() and interp() come from there.

md(r"""
## 7. Evaluation and analysis (Part C)
This section brings every Task 1 model together, justifies the metrics, restates the ablation, analyses the errors of the
best model, restates both published comparisons, summarises the prompted LLMs, checks subgroups and states the limitations.
It reads only saved files (`results/`), except for one cached pass of Legal-BERT to obtain its confidence scores.
""")

md(r"""
### 7.1 Why these metrics: the cost of errors
The intended use is **decision support for a human contract reviewer**. In that setting the two kinds of error cost
different amounts:
* a **false negative** (a risk clause such as *Cap On Liability*, *Change Of Control* or *Anti-Assignment* put in the wrong
  category, or, in Task 2, not found at all) can mean the reviewer never looks at it — the expensive error;
* a **false positive** (a clause wrongly labelled as, say, *Non-Compete*, or a spurious Task 2 span) costs the reviewer a few
  seconds to dismiss.

**Task 1.** **Macro-F1** over the 27 classes is the primary metric (pre-declared): every clause type counts equally, so a
model cannot look good by mastering only the frequent types, and rare risk types matter as much as common ones. Because the
rarest classes have very few test clauses (Source Code Escrow 4), we also report macro-F1 over the **24 classes with ≥15 test
clauses**, contract-level bootstrap CIs, accuracy (what share of all clauses is right — what a reviewer experiences on
average), weighted-F1 (frequency-weighted) and **per-class recall for the risk-bearing classes** (below), because recall is
what protects the reviewer from misses.

**Task 2.** Entity-level precision / recall / F1 are the unit a reviewer sees (a clause found or not). IoU ≥ 0.5 is CUAD's own
matching criterion and tolerates the subjective boundaries of annotators; exact match is reported as a strict secondary
number. Macro-F1 over categories is primary for the same reason as in Task 1. **AUPR and precision at 80 % / 90 % recall**
(CUAD's metrics) describe the trade-off a reviewer actually faces: they want most clauses found (high recall) without
drowning in false hits (precision at high recall).
""")

code(r'''
T1_MODELS = {"majority": "Majority class", "tfidf_complementnb": "TF-IDF + ComplementNB", "tfidf_linearsvc": "TF-IDF + LinearSVC",
             "tfidf_logreg": "TF-IDF + LogReg (best classical)", "bert_base": "BERT-base (fine-tuned, ablation)",
             "legalbert": "Legal-BERT (fine-tuned)", "qwen25_1p5b_v2": "Qwen2.5-1.5B-Instruct (prompt v2)",
             "qwen25_3b_v2": "Qwen2.5-3B-Instruct (prompt v2)"}
PRED = {m: pd.read_csv(C.RESULTS_DIR / "predictions" / f"{m}.csv").set_index("row_id").loc[test.row_id].pred.astype(str).values
        for m in T1_MODELS}
YT = test.category.values; CT = test.contract.values
RISK = ["Cap On Liability", "Change Of Control", "Termination For Convenience", "Anti-Assignment", "Non-Compete", "Liquidated Damages"]
METR = {m: C.compute_metrics(YT, PRED[m], LABELS, m, "test", save=False, robust_labels=ROBUST) for m in T1_MODELS}
risk_tab = pd.DataFrame({T1_MODELS[m]: {c: METR[m]["per_class"][c]["recall"] for c in RISK} for m in T1_MODELS}).T
risk_tab.columns = [f"{c} (n={int((YT == c).sum())})" for c in RISK]
print("Test recall for the risk-bearing classes:")
display(risk_tab.round(3))
risk_tab.to_csv(C.RESULTS_DIR / "eval_risk_class_recall.csv")
''')

md(r"""
**Why these six classes are called "risk-bearing".** The list was fixed in the project plan before any result was produced
(at project set-up). The justification comes from the
**official CUAD category descriptions** (`data/raw/category_descriptions.csv`, downloaded from the CUAD repository by
`scripts/download_cuad.py`): each of the six describes either a limit on what a party can recover or must pay (Cap On Liability,
Liquidated Damages), a party's ability to end or exit the contract (Termination For Convenience, Change Of Control) or a
restriction on what a party may do with the contract or in its business (Anti-Assignment, Non-Compete). A reviewer who misses
such a clause misjudges the exposure, the exit options or the freedom of the client.
""")

code(r'''
cdesc = pd.read_csv(ROOT / "data" / "raw" / "category_descriptions.csv")
cdesc["cat"] = cdesc.iloc[:, 0].str.replace("Category: ", "", regex=False).str.strip().str.lower()
cdesc["desc"] = cdesc.iloc[:, 1].str.replace("Description: ", "", regex=False).str.strip()
WHY = {"Cap On Liability": "limits what can be recovered after a breach", "Liquidated Damages": "fixes an amount payable on breach or termination",
       "Termination For Convenience": "lets a party exit without cause", "Change Of Control": "exit or consent right when ownership changes",
       "Anti-Assignment": "restricts transferring the contract", "Non-Compete": "restricts the party's business activity"}
RJ = pd.DataFrame({"class": RISK, "official CUAD description": [cdesc.set_index("cat").loc[c.lower(), "desc"] for c in RISK],
                   "why it is risk-bearing": [WHY[c] for c in RISK]})
pd.set_option("display.max_colwidth", 220); display(RJ)
RJ.to_csv(C.RESULTS_DIR / "risk_classes_justification.csv", index=False)
''')

md(r"""
### 7.2 One Task 1 comparison table (all models, the same 1,444 test clauses)
Macro-F1 with the corrected contract-level bootstrap (point estimate over all 27 classes, each resample averaged over the
classes present in it; 1,000 resamples of the 93 test contracts). **Cost** = measured seconds per clause × 1,000 / 3,600 ×
the stated hourly price (AWS on-demand, us-east-1, read 2026-10-22: GPU instance g6.xlarge $0.8048/h, CPU instance
c7i.xlarge $0.1785/h; `results/llm_price_sources.json`); **batched cost is the headline column**. Latency = one clause at a
time on the same 100 validation clauses. Training time = the final fit of the selected configuration (searches listed below
the table). The LoRA row is completed in Part F3 (section 9).
""")

code(r'''
inf = json.load(open(C.RESULTS_DIR / "inference_costs.json"))
llm_cost = {r["model"]: r for r in json.load(open(C.RESULTS_DIR / "llm_cost.json"))["rows"]}
prices = json.load(open(C.RESULTS_DIR / "llm_price_sources.json"))["primary"]
GPU_P, CPU_P = prices["g6.xlarge"]["usd_per_hour"], prices["c7i.xlarge"]["usd_per_hour"]
lat_lr = json.load(open(C.RESULTS_DIR / "llm" / "latency_tfidf_logreg.json"))
budget = pd.read_csv(C.RESULTS_DIR / "compute_budget.csv")
first_fit = lambda m: float(budget[(budget.model == m) & (budget.stage == "train_final")].seconds.iloc[0])
tsr = pd.read_csv(C.RESULTS_DIR / "transformer_search.csv").set_index("run_id")
llm_meta = json.load(open(C.RESULTS_DIR / "llm" / "run_meta.json"))
eff = {   # (seconds per clause batched, batch-1 latency ms, price, training seconds, parameters)
    "majority": (inf["majority"]["throughput_seconds_per_clause"], inf["majority"]["median_ms"], CPU_P, 0.0, "27 class counts"),
    "tfidf_complementnb": (inf["tfidf_complementnb"]["throughput_seconds_per_clause"], inf["tfidf_complementnb"]["median_ms"], CPU_P,
                           first_fit("tfidf_complementnb"), "20,000 features x 27"),
    "tfidf_linearsvc": (inf["tfidf_linearsvc"]["throughput_seconds_per_clause"], inf["tfidf_linearsvc"]["median_ms"], CPU_P,
                        first_fit("tfidf_linearsvc"), "41,209 features x 27"),
    "tfidf_logreg": (lat_lr["throughput_seconds_per_clause"], lat_lr["median_ms"], CPU_P, first_fit("tfidf_logreg"), "69,247 features x 27"),
    "bert_base": (inf["bert_base"]["throughput_seconds_per_clause"], inf["bert_base"]["median_ms"], GPU_P, first_fit("bert_base"),
                  f"{tsr.loc['bert_lr5e-05_s42', 'params_million']:.1f} M"),
    "legalbert": (1 / llm_cost["Legal-BERT (fine-tuned)"]["throughput clauses/s (batched)"], llm_cost["Legal-BERT (fine-tuned)"]["latency median ms (batch 1)"],
                  GPU_P, first_fit("legalbert"), f"{tsr.loc['legalbert_lr5e-05_s42', 'params_million']:.1f} M"),
    "qwen25_1p5b_v2": (1 / llm_cost["Qwen2.5-1.5B-Instruct (v2)"]["throughput clauses/s (batched)"],
                       llm_cost["Qwen2.5-1.5B-Instruct (v2)"]["latency median ms (batch 1)"], GPU_P, 0.0,
                       f"{llm_meta['qwen25_1p5b_v2_test']['n_params'] / 1e9:.2f} B"),
    "qwen25_3b_v2": (1 / llm_cost["Qwen2.5-3B-Instruct (v2)"]["throughput clauses/s (batched)"],
                     llm_cost["Qwen2.5-3B-Instruct (v2)"]["latency median ms (batch 1)"], GPU_P, 0.0,
                     f"{llm_meta['qwen25_3b_v2_test']['n_params'] / 1e9:.2f} B")}
rows = []
for m, name in T1_MODELS.items():
    ci = C.bootstrap_ci(YT, PRED[m], CT, labels=LABELS)
    spc, lat, price, tr_s, params = eff[m]
    rows.append({"model": name, "key": m, "test macro-F1": METR[m]["macro_f1"], "95% CI": f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]",
                 "macro-F1 (24 robust classes)": METR[m]["macro_f1_robust"], "accuracy": METR[m]["accuracy"],
                 "weighted-F1": METR[m]["weighted_f1"], "training time (s, final fit)": tr_s,
                 "cost per 1,000 clauses USD (batched)": spc * 1000 / 3600 * price, "latency ms (batch 1)": lat, "parameters": params})
lora_file = C.RESULTS_DIR / "predictions" / "legalbert_lora.csv"
if lora_file.exists():
    PRED["legalbert_lora"] = pd.read_csv(lora_file).set_index("row_id").loc[test.row_id].pred.astype(str).values
    METR["legalbert_lora"] = C.compute_metrics(YT, PRED["legalbert_lora"], LABELS, "legalbert_lora", "test", save=False, robust_labels=ROBUST)
    ci = C.bootstrap_ci(YT, PRED["legalbert_lora"], CT, labels=LABELS)
    li = json.load(open(C.RESULTS_DIR / "lora_inference.json")); lc_ = json.load(open(C.RESULTS_DIR / "lora_choice.json"))
    lsr_ = pd.read_csv(C.RESULTS_DIR / "transformer_lora_search.csv").set_index("run_id")
    rows.append({"model": "Legal-BERT + LoRA (Part F3)", "key": "legalbert_lora", "test macro-F1": METR["legalbert_lora"]["macro_f1"],
                 "95% CI": f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]", "macro-F1 (24 robust classes)": METR["legalbert_lora"]["macro_f1_robust"],
                 "accuracy": METR["legalbert_lora"]["accuracy"], "weighted-F1": METR["legalbert_lora"]["weighted_f1"],
                 "training time (s, final fit)": float(lsr_.loc[lc_["test_run"], "run_seconds"]),
                 "cost per 1,000 clauses USD (batched)": li["throughput_seconds_per_clause"] * 1000 / 3600 * GPU_P,
                 "latency ms (batch 1)": li["median_ms"], "parameters": f"{lsr_.trainable_params.iloc[0] / 1e6:.2f} M trainable of 109.5 M"})
else:
    rows.append({"model": "Legal-BERT + LoRA (Part F3)", "key": "legalbert_lora", "95% CI": "to be filled in Part F3"})
T1_TABLE = pd.DataFrame(rows)
T1_TABLE.to_csv(C.RESULTS_DIR / "task1_comparison_table.csv", index=False)
with pd.option_context("display.float_format", lambda x: f"{x:.4g}"):
    display(T1_TABLE.drop(columns="key"))
search = budget[budget.stage.isin(["grid_search", "all_runs"]) & budget.model.isin(["classical_search", "transformer_plan"])].drop_duplicates("model")
print("Model-selection searches (validation only, not in the table):",
      "; ".join(f"{r.model} {r.seconds:,.0f} s ({r.notes[:45]})" for r in search.itertuples()))
''')

code(r'''
# Paired contract-level bootstraps for the key comparisons. Comparisons computed in earlier stages are RE-USED from their
# files (same function, same seed); only LogReg vs majority is new here.
tc = json.load(open(C.RESULTS_DIR / "transformer_comparison.json"))
lpb = pd.read_csv(C.RESULTS_DIR / "llm_paired_bootstrap.csv")
def from_llm(a, b):
    r = lpb[(lpb.A.str.startswith(a)) & (lpb.B.str.startswith(b))].iloc[0]
    return {"diff": r["macro-F1 A - B"], "ci_low": r.ci_low, "ci_high": r.ci_high, "p_a_not_better": r["share of resamples A not better"]}
new = C.paired_bootstrap(YT, PRED["tfidf_logreg"], PRED["majority"], CT, labels=LABELS)
PAIRS = pd.DataFrame([
    {"comparison": "Legal-BERT - TF-IDF LogReg", **{k: tc["H1_legalbert_vs_best_classical"]["all_27"][k] for k in ["diff", "ci_low", "ci_high", "p_a_not_better"]},
     "source": "results/transformer_comparison.json"},
    {"comparison": "TF-IDF LogReg - majority", **{k: new[k] for k in ["diff", "ci_low", "ci_high", "p_a_not_better"]}, "source": "computed here"},
    {"comparison": "Legal-BERT - BERT-base (ablation)", **{k: tc["ablation_legalbert_vs_bert_base"]["all_27"][k] for k in ["diff", "ci_low", "ci_high", "p_a_not_better"]},
     "source": "results/transformer_comparison.json"},
    {"comparison": "Legal-BERT - Qwen2.5-1.5B (v2)", **from_llm("Legal-BERT", "Qwen2.5-1.5B"), "source": "results/llm_paired_bootstrap.csv"},
    {"comparison": "Legal-BERT - Qwen2.5-3B (v2)", **from_llm("Legal-BERT", "Qwen2.5-3B"), "source": "results/llm_paired_bootstrap.csv"},
    {"comparison": "Qwen2.5-3B - Qwen2.5-1.5B", **from_llm("Qwen2.5-3B", "Qwen2.5-1.5B"), "source": "results/llm_paired_bootstrap.csv"}])
PAIRS["95% CI"] = [f"[{l:+.3f}, {h:+.3f}]" for l, h in zip(PAIRS.ci_low, PAIRS.ci_high)]
display(PAIRS[["comparison", "diff", "95% CI", "p_a_not_better", "source"]].round(4))
PAIRS.to_csv(C.RESULTS_DIR / "task1_paired_bootstraps.csv", index=False)
''')

code(r'''
# Confusion matrices of all 8 models (normalised by true class) - appendix grid; Legal-BERT's full-size matrix is Figure 17
ABBR = {l: "".join(w[0] for w in re.split(r"[\s/-]+", l) if w)[:4].upper() for l in LABELS}
ABBR.update({"Non-Compete": "NCMP", "Non-Disparagement": "NDIS", "Non-Transferable License": "NTL", "No-Solicit Of Employees": "NSOE",
             "Competitive Restriction Exception": "CRE", "Change Of Control": "COC", "Cap On Liability": "CAP", "Covenant Not To Sue": "CNTS",
             "Rofr/Rofo/Rofn": "ROFR", "Revenue/Profit Sharing": "RPS", "Renewal Term": "RENW", "Insurance": "INS", "Exclusivity": "EXCL",
             "Expiration Date": "EXPD", "Ip Ownership Assignment": "IPOA", "Joint Ip Ownership": "JIPO", "License Grant": "LIC",
             "Liquidated Damages": "LIQD", "Minimum Commitment": "MINC", "Post-Termination Services": "PTS", "Source Code Escrow": "SCE",
             "Termination For Convenience": "TFC", "Volume Restriction": "VOLR", "Warranty Duration": "WARD", "Anti-Assignment": "ANTA",
             "Audit Rights": "AUDR", "Governing Law": "GOVL"})
assert len(set(ABBR.values())) == len(LABELS)
from sklearn.metrics import confusion_matrix
fig, axes = plt.subplots(2, 4, figsize=(20, 12))
for ax, m in zip(axes.ravel(), T1_MODELS):
    cmx = confusion_matrix(YT, PRED[m], labels=LABELS).astype(float)
    cmx = cmx / cmx.sum(axis=1, keepdims=True)
    ax.imshow(cmx, cmap=PL.BLUES, vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels([ABBR[l] for l in LABELS], rotation=90, fontsize=5.5)
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels([ABBR[l] for l in LABELS], fontsize=5.5)
    ax.grid(False); ax.set_title(f"{T1_MODELS[m]}\nmacro-F1 {METR[m]['macro_f1']:.3f}", fontsize=9)
fig.supxlabel("predicted class", fontsize=10); fig.supylabel("true class", fontsize=10)
fig.suptitle("Figure 31. Normalised test confusion matrices of all eight Task 1 models (diagonal = recall; abbreviations in "
             "results/label_abbreviations.csv; invalid LLM answers are not shown as a column)", fontweight="bold", fontsize=11, y=1.0)
fig.tight_layout(h_pad=3.0); PL.savefig(fig, 31, "cm_grid_all_models"); plt.show()
pd.Series(ABBR, name="abbreviation").rename_axis("class").to_csv(C.RESULTS_DIR / "label_abbreviations.csv")
''')

interp("eval_table")

md(r"""
### 7.3 Task 2 summary table (from the Task 2 notebook)
""")

code(r'''
t2 = pd.read_csv(C.RESULTS_DIR / "task2_test_summary.csv")
low = pd.read_csv(C.RESULTS_DIR / "task2_lowthr_test.csv")
t2c = json.load(open(C.RESULTS_DIR / "task2_test_comparison.json"))
cmp2 = pd.read_csv(C.RESULTS_DIR / "task2_cuad_comparison.csv")
mr = dict(zip(cmp2.model, cmp2["max recall reached %"]))
T2_TABLE = pd.DataFrame([
    {"model": "TF-IDF segment baseline", **t2.iloc[0][["macro-F1 @ IoU0.5", "95% CI", "micro-F1 @ IoU0.5", "exact macro-F1", "token micro-F1",
                                                       "CUAD-style AUPR (micro)", "P@80%R"]].to_dict(), "max recall": mr["TF-IDF segment baseline (ours)"] / 100},
    {"model": "Legal-BERT tagger (main result, tuned operating point)", **t2.iloc[1][["macro-F1 @ IoU0.5", "95% CI", "micro-F1 @ IoU0.5",
                                     "exact macro-F1", "token micro-F1", "CUAD-style AUPR (micro)", "P@80%R"]].to_dict(),
     "max recall": mr["Legal-BERT tagger (ours)"] / 100},
    {"model": "ANALYSIS ONLY - tagger low-threshold candidate list (theta_i 0.05; ranking metrics only)",
     "macro-F1 @ IoU0.5": low.iloc[1]["macro-F1 @ IoU0.5 (all spans kept)"], "95% CI": "n/a (not an operating point)",
     "CUAD-style AUPR (micro)": low.iloc[1]["CUAD-style AUPR %"] / 100, "P@80%R": low.iloc[1]["P@80%R %"] / 100,
     "max recall": low.iloc[1]["max recall %"] / 100}])
display(T2_TABLE.round(3))
print(f"Tagger - baseline (macro-F1 @ IoU0.5): {t2c['paired_tagger_vs_baseline']['diff']:+.3f} "
      f"[{t2c['paired_tagger_vs_baseline']['ci_low']:+.3f}, {t2c['paired_tagger_vs_baseline']['ci_high']:+.3f}]")
T2_TABLE.to_csv(C.RESULTS_DIR / "task2_final_table.csv", index=False)
''')

md(r"""
### 7.4 Ablation (restated): general-domain vs legal-domain pre-training
**The single isolated decision:** the pre-trained checkpoint — `bert-base-uncased` (general English) vs
`nlpaueb/legal-bert-base-uncased` (legal English, its own legal vocabulary). Everything else was held constant: architecture and
size, recipe (lr 5e-5 chosen for both on validation, batch 16, 5 epochs, warm-up, weight decay, max length 320, bf16), seeds,
split, metric and the single test evaluation. Because Legal-BERT was pre-trained from scratch with its own vocabulary, corpus
and vocabulary change together (token counts below).
""")

code(r'''
abl = pd.read_csv(C.RESULTS_DIR / "ablation.csv")
display(abl[["model", "val_macro_f1_3seeds_mean", "val_macro_f1_3seeds_std", "test_macro_f1", "test_macro_f1_robust", "test_accuracy",
             "mean_tokens_per_clause", "test_macro_f1_diff_95ci"]].round(4))
lr_s = pd.read_csv(C.RESULTS_DIR / "transformer_lr_sensitivity.csv", index_col=0)
ep_s = json.load(open(C.RESULTS_DIR / "transformer_epoch_sensitivity.json"))
print("Sensitivity (validation only, seed 42): macro-F1 by learning rate"); display(lr_s.round(4))
print(f"BERT-base with 8 instead of 5 epochs: {ep_s['bert_8_epochs_best']:.4f} (best epoch {ep_s['bert_8_epochs_best_epoch']}); "
      f"validation gap to Legal-BERT {ep_s['gap_5_epochs']:+.4f} -> {ep_s['gap_vs_bert_8_epochs']:+.4f}")
ABL_SUM = {"isolated_decision": "general-domain (bert-base-uncased) vs legal-domain (legal-bert-base-uncased) pre-training",
           "test_diff_macro_f1": tc["ablation_legalbert_vs_bert_base"]["all_27"], "test_diff_robust_24": tc["ablation_legalbert_vs_bert_base"]["robust_24"],
           "lr_sensitivity": lr_s.reset_index().to_dict(orient="records"), "epoch_sensitivity": ep_s}
json.dump(ABL_SUM, open(C.RESULTS_DIR / "ablation_summary.json", "w"), indent=2, default=float)
''')

interp("eval_ablation")

md(r"""
### 7.5 Error analysis of the best model (Legal-BERT)
Confidence = the softmax probability of the predicted class. It is obtained once by running the saved test model again
(predictions identical to the ones scored on test) and cached in `results/legalbert_test_confidence.csv`; the same pass
labels the 557 held-out multi-label clauses.
""")

code(r'''
conf_file = C.RESULTS_DIR / "legalbert_test_confidence.csv"; amb_file = C.RESULTS_DIR / "legalbert_ambiguous_predictions.csv"
AMB = C.load_ambiguous()
if not (conf_file.exists() and amb_file.exists()):
    from transformers import AutoModelForSequenceClassification
    from src.preprocess import preprocess_for_transformer
    from src.finetune import RECIPE as FT
    d = ROOT / "models" / "legalbert_lr5e-05_s42" / "best"
    tok_lb = AutoTokenizer.from_pretrained(d); lb = AutoModelForSequenceClassification.from_pretrained(d).to("cuda").eval()
    def probs(texts):
        out = []
        for i in range(0, len(texts), 64):
            enc = tok_lb([preprocess_for_transformer(t) for t in texts[i:i + 64]], truncation=True, max_length=FT["max_length"],
                         padding=True, return_tensors="pt").to("cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(torch.softmax(lb(**enc).logits.float(), -1).cpu().numpy())
        return np.concatenate(out)
    order = [lb.config.id2label[i] for i in range(len(LABELS))]
    Pt = probs(list(test.text))
    pd.DataFrame({"row_id": test.row_id, "pred": [order[i] for i in Pt.argmax(1)], "confidence": Pt.max(1),
                  "p_true": Pt[np.arange(len(Pt)), [order.index(t) for t in test.category]]}).to_csv(conf_file, index=False)
    Pa = probs(list(AMB.text))
    pd.DataFrame({"contract": AMB.contract, "start": AMB.start, "pred": [order[i] for i in Pa.argmax(1)], "confidence": Pa.max(1)}).to_csv(amb_file, index=False)
    del lb; torch.cuda.empty_cache()
CONF = pd.read_csv(conf_file).set_index("row_id").loc[test.row_id]
assert (CONF.pred.values == PRED["legalbert"]).all(), "cached confidences do not belong to the tested predictions"
ERR = test.assign(pred=CONF.pred.values, confidence=CONF.confidence.values, p_true=CONF.p_true.values)
ERR = ERR[ERR.pred != ERR.category]
print(f"{len(ERR)} of {len(test)} test clauses misclassified ({100 * len(ERR) / len(test):.1f} %). Mean confidence: errors "
      f"{ERR.confidence.mean():.3f}, correct {CONF.confidence.values[CONF.pred.values == YT].mean():.3f}; "
      f"{(ERR.confidence < 0.9).mean() * 100:.0f} % of the errors have confidence < 0.9 vs "
      f"{(CONF.confidence.values[CONF.pred.values == YT] < 0.9).mean() * 100:.0f} % of the correct predictions.")
''')

code(r'''
# Seven misclassified test clauses: causes verified by reading each full clause (see the text column)
CAUSES = {
    7223: ("misleading keyword in a very short clause", "Only 10 words; 'insurance carriers' dominates, but the clause is about receiving commissions (a revenue share), not about insurance cover."),
    3708: ("overlapping legal functions (top confusion pair)", "An assignment restriction triggered by a merger or sale of assets: CUAD labels it Change Of Control, while its wording is that of an anti-assignment clause."),
    5543: ("annotation-span fragment", "The span starts mid-sentence ('other Party with notice of the transfer or assignment.'); the change-of-control trigger lies outside the 9 annotated words."),
    2952: ("multi-label reality", "The clause grants an exclusive licence: in substance it is both Exclusivity and License Grant, and only one label can be right in single-label Task 1."),
    1367: ("jargon category with overlapping surface words", "A right of first negotiation about renewing the agreement; 'expiration of the term' and 'renewed' are typical Renewal Term words."),
    3733: ("misleading verb in a very short clause", "'The Contract Term shall extend for a period of fifty (50) years' states the initial term (Expiration Date), but 'extend' is typical of renewal clauses."),
    5075: ("one annotated span covering two functions", "A warranty followed by an exclusive-remedy sentence and 'in no event will the Company be liable for incidental or consequential damages'; the model picks the liability limitation."),
}
EX = ERR.set_index("row_id").loc[list(CAUSES)]
EXT = pd.DataFrame({"row_id": EX.index, "excerpt": [t[:300] + ("..." if len(t) > 300 else "") for t in EX.text], "words": EX.n_words.values,
                    "true": EX.category.values, "predicted": EX.pred.values, "confidence": EX.confidence.round(3).values,
                    "P(true class)": EX.p_true.round(3).values, "cause": [CAUSES[r][0] for r in EX.index],
                    "verified by reading the clause": [CAUSES[r][1] for r in EX.index]})
pd.set_option("display.max_colwidth", 200)
display(EXT)
EXT.to_csv(C.RESULTS_DIR / "error_examples_task1.csv", index=False)
EXT.drop(columns="excerpt").assign(full_text=EX.text.values).to_csv(C.RESULTS_DIR / "error_examples_task1_fulltext.csv", index=False)   # report appendix
''')

code(r'''
# Common patterns: confused pairs (vs the hypotheses of section 2.2), clause length, contracts, ambiguous clauses
pairs_t = ERR.groupby(["category", "pred"]).size().sort_values(ascending=False).rename("test clauses").reset_index()
pairs_t["share of true class"] = [n / (YT == c).sum() for c, n in zip(pairs_t.category, pairs_t["test clauses"])]
print("Most frequent confusions of Legal-BERT on test:"); display(pairs_t.head(10).round(3))
pairs_t.to_csv(C.RESULTS_DIR / "legalbert_test_confusions.csv", index=False)
hyp = pd.read_csv(C.RESULTS_DIR / "confusion_hypotheses.csv")
hyp["Legal-BERT test errors (true -> hypothesised)"] = [int(((ERR.category == a) & (ERR.pred == b)).sum()) for a, b in zip(hyp["new class"], hyp["likely confused with"])]
hyp["true clauses"] = [int((YT == a).sum()) for a in hyp["new class"]]
print("Confusion hypotheses written before training (section 2.2) checked on Legal-BERT's test errors:")
display(hyp[["new class", "likely confused with", "Legal-BERT test errors (true -> hypothesised)", "true clauses"]])
hyp.to_csv(C.RESULTS_DIR / "confusion_hypotheses_legalbert_test.csv", index=False)

terc = pd.qcut(test.n_words, 3, labels=["short", "medium", "long"])
by_len = pd.DataFrame({"words": [f"{test.n_words[terc == t].min()}-{test.n_words[terc == t].max()}" for t in ["short", "medium", "long"]],
                       "clauses": [int((terc == t).sum()) for t in ["short", "medium", "long"]],
                       "error rate": [float((PRED["legalbert"][terc == t] != YT[terc == t]).mean()) for t in ["short", "medium", "long"]]},
                      index=["short", "medium", "long"])
print("Legal-BERT error rate by clause-length tercile:"); display(by_len.round(3))
per_c = ERR.contract.value_counts()
print(f"Errors are spread over {per_c.size} of the {test.contract.nunique()} test contracts; the 5 contracts with most errors hold "
      f"{per_c.head(5).sum()} of {len(ERR)} errors ({100 * per_c.head(5).sum() / len(ERR):.0f} %).")

AMBP = pd.read_csv(amb_file)
import ast
valid = AMB.all_labels.apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else list(s))
hit = np.array([p in v for p, v in zip(AMBP.pred, valid)])
amb_tab = pd.DataFrame({"clauses": AMB.groupby("split").size(), "predicted one of its valid labels": pd.Series(hit).groupby(AMB.split.values).mean()})
amb_tab.loc["all"] = [len(AMB), hit.mean()]
print("The 557 held-out multi-label clauses (never used for training or the main score):"); display(amb_tab.round(3))
combos = valid.apply(lambda v: " + ".join(sorted(v))).value_counts().head(5)
print("Most frequent label combinations among them:", combos.to_dict())
amb_tab.to_csv(C.RESULTS_DIR / "ambiguous_check.csv")
json.dump({"n_errors": int(len(ERR)), "error_rate_by_length": by_len.to_dict(orient="index"), "errors_top5_contracts": int(per_c.head(5).sum()),
           "contracts_with_errors": int(per_c.size), "ambiguous_hit_rate": float(hit.mean()), "ambiguous_top_combinations": combos.to_dict()},
          open(C.RESULTS_DIR / "error_patterns_task1.json", "w"), indent=2, default=float)
''')

interp("eval_errors")

md(r"""
**Task 2 error analysis** lives in the Task 2 notebook: section 8.1 (error taxonomy over all 102 test contracts, Figure 25;
per-category table `results/task2_error_taxonomy_per_category.csv`), section 8.2 (ten gold vs predicted examples) and
section 8.3 (five concrete incorrect or low-quality extractions of the tagger, each with its cause).
""")

md(r"""
### 7.6 Published comparisons (restated with their caveats)
""")

code(r'''
lc = pd.read_csv(C.RESULTS_DIR / "ledgar_comparison.csv")
cc = pd.read_csv(C.RESULTS_DIR / "task2_cuad_comparison.csv")
pub_rows = []
for _, r in lc.iterrows():
    nm = {"tfidf_ledgar": "TF-IDF + linear SVM", "legalbert_ledgar": "Legal-BERT"}[r.model]
    pub_rows.append({"task": "Task 1 - LEDGAR test (identical dataset)", "model": nm, "run by": "ours",
                     "result": f"µ-F1 {r['ours test µ-F1']:.2f} {r['µ-F1 95% CI']}, m-F1 {r['ours test m-F1']:.2f} {r['m-F1 95% CI']}", "source": "results/ledgar_comparison.csv"})
    pub_rows.append({"task": "Task 1 - LEDGAR test (identical dataset)", "model": nm, "run by": "paper",
                     "result": f"µ-F1 {r['paper µ-F1']}, m-F1 {r['paper m-F1']}", "source": "LexGLUE (Chalkidis et al., 2022) Table 3, p. 4317"})
for _, r in cc.iterrows():
    pub_rows.append({"task": "Task 2 - CUAD official test contracts (equivalent, not identical: 27 vs 41 categories)", "model": r.model,
                     "run by": "paper" if r.source.startswith("CUAD paper") else "ours",
                     "result": f"AUPR {r['AUPR %']:.1f}, P@80%R {r['P@80%R %']:.1f}, P@90%R {r['P@90%R %']:.1f}", "source": r.source})
PUB = pd.DataFrame(pub_rows)
display(PUB)
PUB.to_csv(C.RESULTS_DIR / "published_comparisons.csv", index=False)
''')

interp("eval_published")

md(r"""
### 7.7 Prompted LLMs: summary for the report (details in `STUDENTID_LLM.ipynb`)
""")

code(r'''
vs = pd.read_csv(C.RESULTS_DIR / "llm_validation_summary.csv")
LLM_SUM = vs[["model", "version", "macro-F1", "invalid %", "mean prompt tokens", "GPU s per clause"]].rename(columns={"macro-F1": "validation macro-F1"})
LLM_SUM["chosen (validation)"] = [v == json.load(open(C.RESULTS_DIR / "llm_choice.json"))[k]["version"] for k, v in zip(vs.key, vs.version)]
display(LLM_SUM.round(4))
r4 = json.load(open(C.RESULTS_DIR / "llm_retrieval_v4_summary.json"))
print(f"Analysis only (validation): retrieval few-shot v4 for the 3B model {r4['v4_val_macro_f1']:.3f}; nearest-neighbour label without LLM "
      f"{r4['nn1_val_macro_f1']:.3f}; majority of 3 neighbours {r4['nn3_vote_val_macro_f1']:.3f}.")
print("Output conversion: exact -> normalised -> unique longest contained label -> INVALID (counted wrong); 23 hand-made tests pass.")
LLM_SUM.to_csv(C.RESULTS_DIR / "llm_summary_for_report.csv", index=False)
''')

md(r"""
### 7.8 Subgroups: clause length and contract family (best model, with the best classical model for contrast)
**Contract family** is derived from the contract title by keyword rules applied in this order: *franchise*; *licence / IP*
(license, licensing, intellectual property); *distribution / supply* (distribut-, supply, reseller, manufactur-); *marketing /
promotion* (endorsement, sponsorship, promotion, co-branding, marketing, agency, affiliate); *development / collaboration*
(development, joint venture, collaboration, alliance, cooperation); *services* (service, outsourc-, consult-, hosting,
maintenance, transportation); otherwise *other*. Only families with ≥ 50 test clauses are reported; smaller ones are merged into
*other*.
""")

code(r'''
FAMILY_RULES = [("franchise", ["franchise"]), ("licence / IP", ["license", "licensing", "intellectual property"]),
                ("distribution / supply", ["distribut", "supply", "reseller", "manufactur"]),
                ("marketing / promotion", ["endorsement", "sponsorship", "promotion", "co-branding", "marketing", "agency", "affiliate"]),
                ("development / collaboration", ["development", "joint venture", "collaboration", "alliance", "cooperation"]),
                ("services", ["service", "outsourc", "consult", "hosting", "maintenance", "transportation"])]
def family(title):
    t = title.lower()
    for fam, keys in FAMILY_RULES:
        if any(k in t for k in keys):
            return fam
    return "other"
fam = test.contract.map(family)
small = fam.value_counts()[lambda s: s < 50].index
fam = fam.where(~fam.isin(small), "other")
if (fam == "other").sum() < 50:      # 'other' itself too small to report -> excluded from the subgroup table
    print(f"'other' family has only {(fam == 'other').sum()} test clauses (< 50) and is not reported.")
    fam = fam.where(fam != "other", None)
sub_rows = []
for kind, groups in [("clause length", terc), ("contract family", fam)]:
    for g in (["short", "medium", "long"] if kind == "clause length" else fam.dropna().value_counts().index):
        msk = (groups == g).values
        for m in ["legalbert", "tfidf_logreg"]:
            mm = C.compute_metrics(YT[msk], PRED[m][msk], LABELS, "x", "test", save=False)
            present = sorted(set(YT[msk]))
            sub_rows.append({"subgroup type": kind, "subgroup": g, "clauses": int(msk.sum()), "contracts": int(test.contract[msk].nunique()),
                             "model": T1_MODELS[m], "accuracy": mm["accuracy"],
                             "macro-F1 (classes present)": f1_score(YT[msk], PRED[m][msk], labels=present, average="macro", zero_division=0)})
SUB = pd.DataFrame(sub_rows)
display(SUB.pivot_table(index=["subgroup type", "subgroup", "clauses", "contracts"], columns="model", values=["accuracy", "macro-F1 (classes present)"]).round(3))
SUB.to_csv(C.RESULTS_DIR / "subgroups_task1.csv", index=False)
print("Task 2 subgroups (contract-length terciles, category frequency) are in the Task 2 notebook, section 6 (results/task2_subgroups.csv).")
''')

interp("eval_subgroups")

md(r"""
### 7.9 Limitations and responsible use
""")
interp("eval_limitations")

md(r"""
### 7.10 Objectives O1–O6
The objectives table is at the end of the notebook (section 10), after Part F, because objective O6 depends on the anomaly
flagging results of section 8.
""")
