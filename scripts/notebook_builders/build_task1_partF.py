# Sections 8-9 of notebooks/STUDENTID_Task1.ipynb: PART F — EXTRA CHALLENGING WORK (F1 anomaly flagging, F3 LoRA).
# Executed by build_task1_partA.py (exec), so md(), code() and interp() come from there.

md(r"""
## 8. PART F — EXTRA CHALLENGING WORK: F1 anomaly flagging ("clauses that read unusually for their type")
**Goal (tutor's suggestion; the reviewer-support promise of the problem statement).** For a clause *x* and a type *c* — the
type a review tool believes the clause has (the gold type for evaluation, the predicted type in deployment) — a score
`score(x, c)` is high when *x* is unlike typical TRAIN clauses of type *c*. A clause is **flagged** when its score exceeds a
per-type threshold set on **validation** so that **5 %** of genuine validation clauses of that type are flagged.

**Two families of scorers (fit on TRAIN only), kept apart on purpose:**
* **"Unusual wording" family** — how far the clause is from the training clauses of type *c*:
  (a) **TF-IDF kNN**: 1 − mean cosine similarity to the 5 most similar TRAIN clauses of type *c* (word vectors);
  (b) **embedding kNN** with the **fine-tuned** Legal-BERT (seed-42 test model; mean-pooled last layer), and the same with the
  **non-fine-tuned** Legal-BERT as a contrast (does fine-tuning make the space type-aware?);
  (c) **class-conditional Mahalanobis distance** on PCA-reduced (50-d) fine-tuned embeddings with a Ledoit-Wolf shrinkage
  covariance per type (a kNN looks at the nearest neighbours; Mahalanobis at the whole shape of the type's cloud).
* **"Doubt about the type" family** — (d) **1 − p(c | x)** from the Task 1 Legal-BERT. This is *not* "unusual wording": a clause
  can be worded very unusually and still be clearly of type *c* (p ≈ 1), and a perfectly normal clause can be doubtful because
  it sits between two types. It is reported separately so the conclusion is not just "the classifier detects wrong labels".

**Selection rule (fixed before looking at results).** The headline scorer is the **"unusual wording" scorer with the best mean
validation AUROC over the two injected-pair proxies** (random and hard wrong types). The doubt score is evaluated with the same
proxies as a separate family. Only the headline scorer is scored on test through the test-once helper
(`anomaly_<scorer>`); every other test number in this section is **analysis only** (no choice depends on it).

**No gold anomaly labels exist in CUAD.** Evaluation uses proxies — **P1** injected wrong-type pairs (random and *hard* =
one of the 3 types most confused with the true type on validation), **P2** Legal-BERT's own test errors, **P3** the 557
multi-type clauses — plus an optional blind human review sheet. Representations are computed once by
`python scripts/anomaly_features.py` (cached in `models/anomaly_cache/`, about 20 s on the GPU).
""")

code(r'''
%%needs_models models/legalbert_lr5e-05_s42/best
from src import anomaly as AN
importlib.reload(AN)
from sklearn.feature_extraction.text import TfidfVectorizer as _TV
feat_file = ROOT / "models" / "anomaly_cache" / "features.npz"
if not feat_file.exists():
    subprocess.run([sys.executable, "scripts/anomaly_features.py"], check=True)
FE = np.load(feat_file)
assert list(FE["labels"]) == LABELS and (FE["row_id"] == df.row_id.values).all()
POS = {s: np.flatnonzero(df.split.values == s) for s in ["train", "validation", "test"]}
TRN = df.iloc[POS["train"]]
tf_vec = _TV(sublinear_tf=True, min_df=2).fit(TRN.text)
X_TF = tf_vec.transform(df.text)
AMB_TF = tf_vec.transform(AMB.text)
SCORERS = {"tfidf_knn": AN.KNNScorer(X_TF[POS["train"]], TRN.category.values, LABELS, sparse=True),
           "legalbert_ft_knn": AN.KNNScorer(FE["emb_ft"][POS["train"]], TRN.category.values, LABELS),
           "legalbert_base_knn": AN.KNNScorer(FE["emb_base"][POS["train"]], TRN.category.values, LABELS),
           "mahalanobis_ft": AN.MahalanobisScorer(FE["emb_ft"][POS["train"]], TRN.category.values, LABELS)}
REPR = {"tfidf_knn": (X_TF, AMB_TF), "legalbert_ft_knn": (FE["emb_ft"], FE["amb_emb_ft"]),
        "legalbert_base_knn": (FE["emb_base"], FE["amb_emb_base"]), "mahalanobis_ft": (FE["emb_ft"], FE["amb_emb_ft"])}
FAMILY = {"tfidf_knn": "unusual wording", "legalbert_ft_knn": "unusual wording", "legalbert_base_knn": "unusual wording",
          "mahalanobis_ft": "unusual wording", "doubt": "doubt about the type"}
def score_pairs(name, positions, claimed, ambiguous=False):
    """score(x, claimed type) for clauses given by their row positions in df (or in AMB if ambiguous=True)."""
    if name == "doubt":
        P = FE["amb_probs"] if ambiguous else FE["probs"]
        return AN.doubt_score(P[positions], np.asarray(claimed), LABELS)
    X = REPR[name][1] if ambiguous else REPR[name][0]
    return SCORERS[name].score(X[positions], np.asarray(claimed))

# hard alternatives: the 3 types each type is most often confused with on VALIDATION (Legal-BERT), ties/gaps filled by
# the most similar types by TF-IDF centroid cosine on TRAIN
vpred = pd.read_csv(C.RESULTS_DIR / "predictions" / "val_legalbert_lr5e-05_s42.csv")
Xtr = X_TF[POS["train"]]
cent = {c: (lambda m: m / np.linalg.norm(m))(np.asarray(Xtr[(TRN.category == c).values].mean(0)).ravel()) for c in LABELS}
csim = pd.DataFrame([[cent[a] @ cent[b] for b in LABELS] for a in LABELS], index=LABELS, columns=LABELS)
HARD = AN.hard_types(vpred.true, vpred.pred, LABELS, csim)
json.dump(HARD, open(C.RESULTS_DIR / "anomaly_hard_types.json", "w"), indent=2)
print("Hard alternatives (examples):", {c: HARD[c] for c in ["Change Of Control", "Non-Transferable License", "Exclusivity"]})

VALC = df.iloc[POS["validation"]].copy(); VALC.index = POS["validation"]
val_rows = []
for kind, hm in [("P1 random", None), ("P1 hard", HARD)]:
    pr = AN.make_pairs(VALC, LABELS, hm, seed=42)
    for nm in list(SCORERS) + ["doubt"]:
        r = AN.auc_ci(pr.injected.values, score_pairs(nm, pr.pos.values, pr.claimed.values), pr.contract.values)
        val_rows.append({"scorer": nm, "family": FAMILY[nm], "proxy": kind, "AUROC": r["auroc"], "AUROC 95% CI": r["auroc_ci"],
                         "AUPR": r["aupr"], "pairs": r["n"]})
AVAL = pd.DataFrame(val_rows)
AVAL.to_csv(C.RESULTS_DIR / "anomaly_validation.csv", index=False)
display(AVAL.assign(**{"AUROC 95% CI": AVAL["AUROC 95% CI"].map(lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]")}).round(3))
mean_auc = AVAL.groupby("scorer").AUROC.mean()
wording = [s for s in SCORERS]
HEAD = mean_auc[wording].idxmax()
json.dump({"rule": "best mean validation AUROC (P1 random + P1 hard) among the unusual-wording scorers", "mean_auroc": mean_auc.to_dict(),
           "headline": HEAD, "doubt_family_mean_auroc": float(mean_auc["doubt"])}, open(C.RESULTS_DIR / "anomaly_choice.json", "w"), indent=2)
print(f"Headline scorer (chosen on validation within the 'unusual wording' family): {HEAD} (mean AUROC {mean_auc[HEAD]:.3f}); "
      f"doubt score 1 - p(type): {mean_auc['doubt']:.3f} (separate family)")
''')

code(r'''
%%needs_models models/legalbert_lr5e-05_s42/best
# Per-type thresholds at a 5 % flag rate on genuine VALIDATION clauses, and the flag-rate sweep (validation, P1 pairs)
gen_val = {nm: score_pairs(nm, VALC.index.values, VALC.category.values) for nm in list(SCORERS) + ["doubt"]}
THR = {nm: AN.thresholds_at(gen_val[nm], VALC.category.values, LABELS, 0.05) for nm in gen_val}
pr_r = AN.make_pairs(VALC, LABELS, None, seed=42); pr_h = AN.make_pairs(VALC, LABELS, HARD, seed=42)
sweep = []
for rate in [0.01, 0.02, 0.05, 0.10, 0.20]:
    for nm in gen_val:
        th = AN.thresholds_at(gen_val[nm], VALC.category.values, LABELS, rate)
        for kind, pr in [("random", pr_r), ("hard", pr_h)]:
            inj = pr[pr.injected == 1]
            sweep.append({"flag rate on genuine validation clauses": rate, "scorer": nm, "wrong type": kind,
                          "recall of injected pairs": AN.flagged(score_pairs(nm, inj.pos.values, inj.claimed.values), inj.claimed.values, th).mean()})
SW = pd.DataFrame(sweep); SW.to_csv(C.RESULTS_DIR / "anomaly_flag_rate_sweep.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
cols = {"tfidf_knn": PL.AQUA, "legalbert_ft_knn": PL.BLUE, "legalbert_base_knn": PL.YELLOW, "mahalanobis_ft": PL.VIOLET, "doubt": PL.ORANGE}
for nm in gen_val:
    for kind, ls in [("random", "-"), ("hard", "--")]:
        g = SW[(SW.scorer == nm) & (SW["wrong type"] == kind)]
        axes[0].plot(100 * g["flag rate on genuine validation clauses"], 100 * g["recall of injected pairs"], ls, marker="o", color=cols[nm],
                     label=f"{nm} ({kind})" if kind == "random" or nm in (HEAD, "doubt") else None)
axes[0].axvline(5, color=PL.INK_2, linestyle=":", linewidth=1)
axes[0].set_xlabel("flag rate on genuine validation clauses (%)"); axes[0].set_ylabel("injected wrong-type pairs flagged (%)")
axes[0].set_title("Recall of injected pairs vs flag rate\n(solid = random, dashed = hard wrong type)", fontsize=10); axes[0].legend(fontsize=7)
s_gen = gen_val[HEAD]; s_inj = score_pairs(HEAD, pr_h[pr_h.injected == 1].pos.values, pr_h[pr_h.injected == 1].claimed.values)
bins = np.linspace(min(s_gen.min(), s_inj.min()), max(s_gen.max(), s_inj.max()), 40)
axes[1].hist(s_gen, bins=bins, alpha=0.6, color=PL.BLUE, label="genuine (clause, true type)")
axes[1].hist(s_inj, bins=bins, alpha=0.6, color=PL.ORANGE, label="injected (clause, hard wrong type)")
axes[1].set_xlabel(f"{HEAD} score"); axes[1].set_ylabel("validation pairs"); axes[1].legend(fontsize=8)
axes[1].set_title(f"Score distributions, headline scorer\n({HEAD})", fontsize=10)
fig.suptitle("Figure 32. PART F1 anomaly flagging: flag-rate sweep and score distributions (validation)", fontweight="bold", y=1.03, fontsize=11)
PL.savefig(fig, 32, "anomaly_validation"); plt.show()
''')

code(r'''
%%needs_models models/legalbert_lr5e-05_s42/best
# TEST: P1 (random / hard), P2 (Legal-BERT's own errors, scored against the PREDICTED type), P3 (557 multi-type clauses)
TESTC = df.iloc[POS["test"]].copy(); TESTC.index = POS["test"]
assert (TESTC.row_id.values == test.row_id.values).all()
LB_PRED = PRED["legalbert"]
pt_r = AN.make_pairs(TESTC, LABELS, None, seed=42); pt_h = AN.make_pairs(TESTC, LABELS, HARD, seed=42)
gen_test = {nm: score_pairs(nm, TESTC.index.values, TESTC.category.values) for nm in gen_val}
amb_pairs = pd.DataFrame([(i, c, AMB.contract.iloc[i]) for i, v in enumerate(valid) for c in v], columns=["apos", "claimed", "contract"])
rows, TEST_SCORES = [], {}
for nm in gen_val:
    res = {}
    for kind, pr in [("P1 random", pt_r), ("P1 hard", pt_h)]:
        s = score_pairs(nm, pr.pos.values, pr.claimed.values)
        r = AN.auc_ci(pr.injected.values, s, pr.contract.values)
        inj = pr.injected.values == 1
        r["recall_at_5pct"] = float(AN.flagged(s[inj], pr.claimed.values[inj], THR[nm]).mean())
        r["flag_rate_genuine"] = float(AN.flagged(s[~inj], pr.claimed.values[~inj], THR[nm]).mean())
        res[kind] = r
        if kind == "P1 random":
            TEST_SCORES[nm] = pr.assign(score=s)
    s2 = score_pairs(nm, TESTC.index.values, LB_PRED)
    err = (LB_PRED != TESTC.category.values).astype(int)
    r2 = AN.auc_ci(err, s2, TESTC.contract.values)
    r2["recall_at_5pct"] = float(AN.flagged(s2[err == 1], LB_PRED[err == 1], THR[nm]).mean())
    r2["flag_rate_genuine"] = float(AN.flagged(s2[err == 0], LB_PRED[err == 0], THR[nm]).mean())
    res["P2 own errors"] = r2
    s3 = score_pairs(nm, amb_pairs.apos.values, amb_pairs.claimed.values, ambiguous=True)
    y3 = np.r_[np.ones(len(s3)), np.zeros(len(gen_test[nm]))]
    r3 = AN.auc_ci(y3, np.r_[s3, gen_test[nm]], np.r_[amb_pairs.contract.values, TESTC.contract.values])
    r3["recall_at_5pct"] = float(AN.flagged(s3, amb_pairs.claimed.values, THR[nm]).mean())
    r3["flag_rate_genuine"] = float(AN.flagged(gen_test[nm], TESTC.category.values, THR[nm]).mean())
    res["P3 multi-type clauses"] = r3
    for k, r in res.items():
        rows.append({"scorer": nm, "family": FAMILY[nm], "status": "HEADLINE (test-once)" if nm == HEAD else "analysis only", "proxy": k,
                     "AUROC": r["auroc"], "AUROC 95% CI": f"[{r['auroc_ci'][0]:.3f}, {r['auroc_ci'][1]:.3f}]", "AUPR": r["aupr"],
                     "recall at 5 % operating point": r["recall_at_5pct"], "flag rate on genuine / correct": r["flag_rate_genuine"],
                     "positives / n": f"{r['positives']} / {r['n']}"})
    if nm == HEAD:
        HEAD_RES = res
ATEST = pd.DataFrame(rows); ATEST.to_csv(C.RESULTS_DIR / "anomaly_test.csv", index=False)
display(ATEST.round(3))
cfg_an = {"scorer": HEAD, "k": 5, "fit_on": "train", "threshold": "per type, 95th percentile of genuine validation scores (5 % flag rate)",
          "selection": "best mean validation AUROC (P1 random + hard) among the unusual-wording scorers"}
C.test_once_scores(f"anomaly_{HEAD}", cfg_an, {k: {kk: vv for kk, vv in v.items()} for k, v in HEAD_RES.items()},
                   TEST_SCORES[HEAD][["row_id", "contract", "true", "claimed", "injected", "score"]],
                   {"macro_f1": HEAD_RES["P1 random"]["auroc"], "accuracy": HEAD_RES["P1 hard"]["auroc"],
                    "weighted_f1": HEAD_RES["P1 random"]["recall_at_5pct"]},
                   reason="Part F1 anomaly scorer: macro_f1 column = AUROC P1 random, accuracy column = AUROC P1 hard, "
                          "weighted_f1 column = recall of injected pairs at the 5 % operating point")
''')

interp("anomaly_results")

code(r'''
%%needs_models models/legalbert_lr5e-05_s42/best
# Is the headline score just clause length? Spearman correlation with the number of words (genuine test pairs)
from scipy.stats import spearmanr
TESTC["head_score"] = gen_test[HEAD]; TESTC["doubt_score"] = gen_test["doubt"]
TESTC["flag"] = AN.flagged(gen_test[HEAD], TESTC.category.values, THR[HEAD])
TESTC["excess"] = gen_test[HEAD] - np.array([THR[HEAD][c] for c in TESTC.category])
len_corr = {nm: float(spearmanr(gen_test[nm], TESTC.n_words).correlation) for nm in gen_test}
print("Spearman correlation of each score with clause length (words), genuine test clauses:", {k: round(v, 3) for k, v in len_corr.items()})
q1, q3 = TESTC.n_words.quantile([0.25, 0.75])
normal = TESTC[TESTC.flag & TESTC.n_words.between(q1, q3)].sort_values("excess", ascending=False)
print(f"Test flag rate of the headline scorer: {TESTC.flag.mean():.3f} (5 % on validation by construction). "
      f"{len(normal)} of {TESTC.flag.sum()} flagged clauses have a normal length ({q1:.0f}-{q3:.0f} words, the middle 50 %). Examples:")
display(normal.head(5)[["row_id", "category", "n_words", "text"]].assign(text=lambda d: d.text.str[:220]))
json.dump({"spearman_with_length": len_corr, "test_flag_rate": float(TESTC.flag.mean()), "flagged": int(TESTC.flag.sum()),
           "flagged_normal_length": int(len(normal))}, open(C.RESULTS_DIR / "anomaly_length_check.json", "w"), indent=2)
''')

code(r'''
%%needs_models models/legalbert_lr5e-05_s42/best
# Blind review sheet (optional, for the student to fill in) of the top flags and 30 random unflagged clauses
fl = TESTC[TESTC.flag].sort_values("excess", ascending=False)
groups_ = {c: list(g.sort_values("excess", ascending=False).index) for c, g in fl.groupby("category")}
TOP = []
while len(TOP) < 30 and any(groups_.values()):                  # round robin over types -> spread over many types
    for c in sorted(groups_, key=lambda c: -fl.loc[groups_[c][0], "excess"] if groups_[c] else 0):
        if groups_[c]:
            TOP.append(groups_[c].pop(0))
        if len(TOP) >= 30:
            break
rng_ = np.random.default_rng(42)
unfl = TESTC[~TESTC.flag].index.values
RAND = list(rng_.choice(unfl, 30, replace=False))
sheet_file, key_file = C.RESULTS_DIR / "anomaly_review_sheet.csv", C.RESULTS_DIR / "anomaly_review_key.csv"
filled = sheet_file.exists() and pd.read_csv(sheet_file).unusual_for_type.notna().any()
if not filled:
    items = pd.DataFrame({"pos": TOP + RAND, "group": ["flagged"] * 30 + ["random unflagged"] * 30}).sample(frac=1, random_state=7).reset_index(drop=True)
    items["id"] = [f"R{i + 1:02d}" for i in range(len(items))]
    sheet = pd.DataFrame({"id": items.id, "type": TESTC.loc[items.pos, "category"].values, "clause_text": TESTC.loc[items.pos, "text"].values,
                          "unusual_for_type": "", "reason": ""})
    sheet.to_csv(sheet_file, index=False)
    items.assign(row_id=TESTC.loc[items.pos, "row_id"].values)[["id", "row_id", "group"]].to_csv(key_file, index=False)
    print(f"Review sheet written: {sheet_file.name} ({len(sheet)} clauses, {sheet.type.nunique()} types, flagged clauses from "
          f"{TESTC.loc[TOP, 'category'].nunique()} types). Labels are EMPTY -> flags are proxy-evaluated only, NOT human-validated; "
          "O6 human part: not assessed.")
else:
    sh = pd.read_csv(sheet_file).merge(pd.read_csv(key_file), on="id")
    yes = sh.unusual_for_type.str.lower().eq("yes")
    print("Filled review sheet found: yes-rate flagged", yes[sh.group == "flagged"].mean(), "vs random", yes[sh.group != "flagged"].mean())
''')

interp("anomaly_inspection")


# =====================================================================================
md(r"""
## 9. PART F — EXTRA CHALLENGING WORK: F3 LoRA vs full fine-tuning (Task 1)
**Question.** Can parameter-efficient fine-tuning (LoRA) match full fine-tuning of Legal-BERT on our 27 classes, and what does
it save? **Set-up (identical to section 5 except the trainable parameters):** same data, split, recipe (batch 16, 5 epochs,
warm-up 10 %, weight decay 0.01, max length 320, bf16), seeds (42, 43, 44) and metric. LoRA (Hu et al., 2022) freezes the
pre-trained weights and learns small low-rank updates *W + BA* (rank 16, scaling α = 32, dropout 0.1) for the attention
**query** and **value** matrices; the classification head is trained fully. Learning rate from {1e-4, 3e-4, 1e-3} on
validation (seed 42; rule fixed before the runs: best validation macro-F1, ties → smaller lr); seeds 43 and 44 at that lr; the
test model is seed 42 (`legalbert_lora`, evaluated once: `python scripts/train_lora.py --test`).
""")

code(r'''
lsr = pd.read_csv(C.RESULTS_DIR / "transformer_lora_search.csv")
lch = json.load(open(C.RESULTS_DIR / "lora_choice.json"))
llog_ = pd.read_csv(C.RESULTS_DIR / "transformer_lora_log.csv")
display(lsr[["run_id", "lr", "seed", "best_epoch", "val_macro_f1", "mean_epoch_seconds", "run_seconds", "peak_gpu_mem_gb", "trainable_params"]].round(4))
print(f"Chosen lr: {lch['best_lr']:.0e} (top of the grid; every run's best epoch is its last one). Validation macro-F1 over 3 seeds: "
      f"{lch['val_macro_f1_mean']:.4f} ± {lch['val_macro_f1_std']:.4f}")
# the single test evaluation (re-check on notebook re-runs; the first evaluation was made by scripts/train_lora.py --test)
if not C.already_evaluated("legalbert_lora"):
    subprocess.run([sys.executable, "scripts/train_lora.py", "--test"], check=True)
lora_pred = pd.read_csv(C.RESULTS_DIR / "predictions" / "legalbert_lora.csv").set_index("row_id").loc[test.row_id].pred.astype(str).values
full_runs = tsr[(tsr.index.str.startswith("legalbert_lr5e-05")) & (~tsr.class_weighted)]
size_mb = lambda p: sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
full_dir = ROOT / "models" / "legalbert_lr5e-05_s42" / "best"; lora_dir = ROOT / "models" / lch["test_run"] / "best"
lm = C.compute_metrics(YT, lora_pred, LABELS, "legalbert_lora", "test", save=False, robust_labels=ROBUST)
ci_l = C.bootstrap_ci(YT, lora_pred, CT, labels=LABELS)
pb_lf = C.paired_bootstrap(YT, PRED["legalbert"], lora_pred, CT, labels=LABELS)
lora_inf = json.load(open(C.RESULTS_DIR / "lora_inference.json"))
LCOMP = pd.DataFrame([
    {"method": "full fine-tuning", "trainable parameters": int(full_runs.params_million.iloc[0] * 1e6), "share trainable %": 100.0,
     "time per epoch (s)": float(full_runs.mean_epoch_seconds.mean()), "total training (s, final run)": float(tsr.loc["legalbert_lr5e-05_s42", "run_seconds"]),
     "peak GPU memory (GB)": float(full_runs.peak_gpu_mem_gb.mean()),
     "saved weights (MB)": size_mb(full_dir) if full_dir.exists() else np.nan,
     "validation macro-F1 (3 seeds)": f"{full_runs.val_macro_f1.mean():.4f} ± {full_runs.val_macro_f1.std():.4f}",
     "test macro-F1": METR["legalbert"]["macro_f1"], "test 95% CI": T1_TABLE.set_index("key").loc["legalbert", "95% CI"],
     "latency ms (batch 1)": eff["legalbert"][1]},
    {"method": "LoRA r16 (query, value) + head", "trainable parameters": int(lsr.trainable_params.iloc[0]),
     "share trainable %": 100 * lsr.trainable_params.iloc[0] / (full_runs.params_million.iloc[0] * 1e6),
     "time per epoch (s)": float(lsr[lsr.lr == lch["best_lr"]].mean_epoch_seconds.mean()),
     "total training (s, final run)": float(lsr.set_index("run_id").loc[lch["test_run"], "run_seconds"]),
     "peak GPU memory (GB)": float(lsr[lsr.lr == lch["best_lr"]].peak_gpu_mem_gb.mean()),
     "saved weights (MB)": size_mb(lora_dir) if lora_dir.exists() else np.nan,
     "validation macro-F1 (3 seeds)": f"{lch['val_macro_f1_mean']:.4f} ± {lch['val_macro_f1_std']:.4f}",
     "test macro-F1": lm["macro_f1"], "test 95% CI": f"[{ci_l['ci_low']:.3f}, {ci_l['ci_high']:.3f}]", "latency ms (batch 1)": lora_inf["median_ms"]}])
display(LCOMP.round(3))
LCOMP.to_csv(C.RESULTS_DIR / "lora_vs_full.csv", index=False)
print(f"Full - LoRA on test macro-F1: {pb_lf['diff']:+.4f} [{pb_lf['ci_low']:+.4f}, {pb_lf['ci_high']:+.4f}] (paired contract bootstrap)")
json.dump({"paired_full_minus_lora": pb_lf, "lora_test": {k: lm[k] for k in ["macro_f1", "macro_f1_robust", "accuracy", "weighted_f1"]},
           "lora_ci": ci_l}, open(C.RESULTS_DIR / "lora_results.json", "w"), indent=2, default=float)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4))
for rid, col in [("legalbert_lr1e-04_s42_lora", PL.YELLOW), ("legalbert_lr3e-04_s42_lora", PL.AQUA), ("legalbert_lr1e-03_s42_lora", PL.BLUE)]:
    ep = llog_[(llog_.run_id == rid) & (llog_.kind == "epoch")]
    axes[0].plot(ep.epoch, ep.val_macro_f1, "o-", color=col, label=f"LoRA lr {rid.split('_lr')[1].split('_')[0]}")
fl_log = pd.read_csv(C.RESULTS_DIR / "transformer_log.csv")
ep = fl_log[(fl_log.run_id == "legalbert_lr5e-05_s42") & (fl_log.kind == "epoch")]
axes[0].plot(ep.epoch, ep.val_macro_f1, "s-", color=PL.INK, label="full fine-tuning lr 5e-05")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("validation macro-F1"); axes[0].legend(fontsize=8); axes[0].set_title("Validation macro-F1 per epoch (seed 42)", fontsize=10)
metrics_ = ["time per epoch (s)", "peak GPU memory (GB)", "saved weights (MB)"]
x_ = np.arange(len(metrics_))
for i, (meth, col) in enumerate([("full fine-tuning", PL.INK), ("LoRA r16 (query, value) + head", PL.BLUE)]):
    vals = LCOMP.set_index("method").loc[meth, metrics_].astype(float).values
    b = axes[1].bar(x_ + (i - 0.5) * 0.38, vals, 0.38, color=col, label=meth); axes[1].bar_label(b, fmt="%.1f", fontsize=8)
axes[1].set_yscale("log"); axes[1].set_xticks(x_); axes[1].set_xticklabels(metrics_, fontsize=8); axes[1].legend(fontsize=8)
axes[1].set_title("Cost of training (log scale)", fontsize=10); axes[1].grid(axis="x", visible=False)
fig.suptitle("Figure 33. PART F3: LoRA vs full fine-tuning of Legal-BERT on Task 1", fontweight="bold", y=1.03, fontsize=11)
PL.savefig(fig, 33, "lora_vs_full"); plt.show()
''')

interp("lora")

md(r"""
**LoRA sensitivity (validation only; an extra analysis).** Is the gap to full fine-tuning mainly under-training? Two extra
runs with seed 42 (`python scripts/train_lora.py --sensitivity`; logged in `results/transformer_lora_sensitivity_*.csv`): (a) the
chosen lr 1e-3 for **10** epochs, (b) lr **3e-3** for 5 epochs. The tested LoRA model is unchanged.
""")

code(r'''
lss = pd.read_csv(C.RESULTS_DIR / "transformer_lora_sensitivity_search.csv")
lsl = pd.read_csv(C.RESULTS_DIR / "transformer_lora_sensitivity_log.csv")
base42 = lsr.set_index("run_id").loc[lch["test_run"]]
SENS = pd.concat([pd.DataFrame([{"run": "tested LoRA (lr 1e-3, 5 epochs)", "best epoch": int(base42.best_epoch), "val macro-F1": base42.val_macro_f1}]),
                  pd.DataFrame([{"run": {"legalbert_lr1e-03_s42_sens_e10_lora": "lr 1e-3, 10 epochs", "legalbert_lr3e-03_s42_sens_lora": "lr 3e-3, 5 epochs"}[r.run_id],
                                 "best epoch": int(r.best_epoch), "val macro-F1": r.val_macro_f1} for r in lss.itertuples()])], ignore_index=True)
full42 = float(tsr.loc["legalbert_lr5e-05_s42", "val_macro_f1"])
SENS["gap to full fine-tuning (val, seed 42)"] = full42 - SENS["val macro-F1"]
display(SENS.round(4))
ep10 = lsl[(lsl.run_id == "legalbert_lr1e-03_s42_sens_e10_lora") & (lsl.kind == "epoch")][["epoch", "val_macro_f1"]]
print("lr 1e-3, 10 epochs - validation macro-F1 per epoch:", dict(zip(ep10.epoch, ep10.val_macro_f1.round(3))))
SENS.to_csv(C.RESULTS_DIR / "lora_sensitivity.csv", index=False)
''')

interp("lora_sensitivity")

md(r"""
## 10. Objectives O1–O6 (final verdicts)
""")

code(r'''
%%needs_models models/legalbert_lr5e-05_s42/best
o5 = json.load(open(C.RESULTS_DIR / "objective_O5_legalbert.json"))
OBJ = pd.DataFrame([
    {"objective": "O1", "criterion": "all Task 1 models on the same 1,444 test clauses (93 contracts); Task 2 on the 102 official test contracts",
     "verdict": "met", "evidence": "results/test_eval_log.csv (one entry per model, forced = False); task1_comparison_table.csv; task2_test_summary.csv"},
    {"objective": "O2", "criterion": "best TF-IDF classifier beats the majority baseline by a wide margin",
     "verdict": "met", "evidence": f"TF-IDF LogReg - majority = {new['diff']:+.3f} [{new['ci_low']:+.3f}, {new['ci_high']:+.3f}] (task1_paired_bootstraps.csv)"},
    {"objective": "O3", "criterion": "fine-tuned transformer vs best classical vs two prompted LLMs on macro-F1, cost per 1,000 clauses and latency",
     "verdict": "met", "evidence": "task1_comparison_table.csv, task1_paired_bootstraps.csv, llm_cost.json (second LLM = Qwen2.5-3B, deviation D3)"},
    {"objective": "O4", "criterion": "Task 2: Legal-BERT extractor beats the TF-IDF segment baseline (paired CI above 0) and is compared with CUAD's published AUPR",
     "verdict": "met", "evidence": f"+{t2c['paired_tagger_vs_baseline']['diff']:.3f} [{t2c['paired_tagger_vs_baseline']['ci_low']:+.3f}, "
                                   f"{t2c['paired_tagger_vs_baseline']['ci_high']:+.3f}]; CUAD comparison in Task 2 section 7 (not like-for-like)"},
    {"objective": "O5", "criterion": "(exploratory) recall >= 0.70 for every Task 1 class on test",
     "verdict": "not met", "evidence": "results/objective_O5_legalbert.json (Legal-BERT below 0.70: Non-Transferable License, Change Of Control)"},
    {"objective": "O6", "criterion": "(Part F) best anomaly scorer detects injected wrong-type clauses with AUROC CI above 0.5; human part only if the blind review is done",
     "verdict": ("proxy part: " + ("met" if HEAD_RES["P1 random"]["auroc_ci"][0] > 0.5 else "not met") + "; human part: "
                 + ("see review results" if filled else "not assessed (review sheet not filled)")),
     "evidence": (f"{HEAD}: test AUROC P1 random {HEAD_RES['P1 random']['auroc']:.3f} "
                  f"[{HEAD_RES['P1 random']['auroc_ci'][0]:.3f}, {HEAD_RES['P1 random']['auroc_ci'][1]:.3f}], hard "
                  f"{HEAD_RES['P1 hard']['auroc']:.3f} (results/anomaly_test.csv); flags are proxy-evaluated only, not human-validated")}])
display(OBJ)
OBJ.to_csv(C.RESULTS_DIR / "objectives.csv", index=False)
''')


md(r"""
## 11. DEMO — classify any clause and flag it if it reads unusually
Paste any clause into `MY_CLAUSES`. For the same clauses, **side by side**: the three most likely types with probabilities from
the best classical model (TF-IDF character n-grams + logistic regression, refit here from its saved configuration in a few
seconds) and from the fine-tuned Legal-BERT, plus the Part F1 flag saying whether the clause reads unusually for each model's
predicted type (threshold: 5 % of genuine validation clauses). Everything is loaded from saved files, so this group runs on its own.
**Without `models/`** (e.g. on Colab) the logistic regression still runs live; the Legal-BERT columns and the flags then show
the **saved output of the real run** for clauses the demo has seen before (`results/demo_cache.json`), and the table says so
in its `source` column.
""")

code(r"""
# DEMO setup: refit the best classical model (seconds, no GPU) and, if models/ is present, load the saved Legal-BERT,
# the cached training embeddings and the validation thresholds of the anomaly flag
import time, json
from sklearn.feature_extraction.text import TfidfVectorizer as _TV2
from sklearn.linear_model import LogisticRegression as _LR2
from sklearn.pipeline import make_pipeline as _mp
from src.preprocess import clean_text as _clean, preprocess_for_transformer
from src import common as C
_tr = (df.split == "train").values; _va = (df.split == "validation").values
# best classical model, refit from its saved test-once configuration (identical predictions to the evaluated model)
_cfg = json.loads(pd.read_csv(C.RESULTS_DIR / "test_eval_log.csv").set_index("model").loc["tfidf_logreg", "config"])
demo_lr = _mp(_TV2(analyzer="char_wb", preprocessor=lambda s_: _clean(s_, _cfg["preprocessing"]), lowercase=False, ngram_range=(3, 6),
                   min_df=int(_cfg["min_df"]), max_df=float(_cfg["max_df"]), sublinear_tf=True),
              _LR2(C=float(_cfg["C_or_alpha"]), class_weight="balanced", max_iter=3000))
demo_lr.fit(df[_tr].text, df[_tr].category)
_dir, _feat = ROOT / "models" / "legalbert_lr5e-05_s42" / "best", ROOT / "models" / "anomaly_cache" / "features.npz"
DEMO_LIVE = _dir.exists() and _feat.exists()
if DEMO_LIVE:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from src import anomaly as AN
    from src.finetune import RECIPE as FT
    demo_tok = AutoTokenizer.from_pretrained(_dir)
    demo_model = AutoModelForSequenceClassification.from_pretrained(_dir).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    _fe = np.load(_feat)
    demo_knn = AN.KNNScorer(_fe["emb_ft"][_tr], df.category.values[_tr], LABELS)
    demo_thr = AN.thresholds_at(demo_knn.score(_fe["emb_ft"][_va], df.category.values[_va]), df.category.values[_va], LABELS, 0.05)
    demo_labels = [demo_model.config.id2label[i] for i in range(len(LABELS))]
else:
    print("models/ not found (normal on Colab): the logistic regression runs live; the Legal-BERT columns and the flags show "
          "the SAVED output of the real run where available (results/demo_cache.json).")

def classify(clauses):
    top3 = lambda row, names: "; ".join(f"{names[i]} {row[i]:.2f}" for i in np.argsort(-row)[:3])
    PL_ = demo_lr.predict_proba(clauses); lr_cls = list(demo_lr.classes_)
    lr_top = [top3(r, lr_cls) for r in PL_]
    if DEMO_LIVE:
        dev = next(demo_model.parameters()).device
        enc = demo_tok([preprocess_for_transformer(t) for t in clauses], truncation=True, max_length=FT["max_length"], padding=True,
                       return_tensors="pt").to(dev)
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out = demo_model(**enc, output_hidden_states=True)
        P = torch.softmax(out.logits.float(), -1).cpu().numpy()
        m = enc["attention_mask"].unsqueeze(-1).float(); E = ((out.hidden_states[-1].float() * m).sum(1) / m.sum(1)).cpu().numpy()
        flag = lambda preds: ["UNUSUAL" if sc > demo_thr[pr] else "typical" for sc, pr in zip(demo_knn.score(E, preds), preds)]
        rest = [{"flag (LogReg type)": fl, "Legal-BERT: top 3": top3(r, demo_labels), "flag (Legal-BERT type)": fb}
                for fl, r, fb in zip(flag([lr_cls[i] for i in PL_.argmax(1)]), P, flag([demo_labels[i] for i in P.argmax(1)]))]
        for t, r in zip(clauses, rest):
            C.demo_cache_put("task1", t, r)            # lets the demo show this real output where models/ is absent
        source = ["live"] * len(clauses)
    else:
        na = "n/a (needs models/)"
        saved = [C.demo_cache_get("task1", t) for t in clauses]
        rest = [r or {"flag (LogReg type)": na, "Legal-BERT: top 3": na, "flag (Legal-BERT type)": na} for r in saved]
        source = ["LogReg live; rest SAVED from the real run" if r else "LogReg live; rest not available" for r in saved]
    out_tab = pd.DataFrame([{"clause": t[:90] + ("..." if len(t) > 90 else ""), "TF-IDF + LogReg: top 3": lt, **r, "source": s}
                            for t, lt, r, s in zip(clauses, lr_top, rest, source)])
    with pd.option_context("display.max_colwidth", 120):
        display(out_tab)
    return out_tab
""")

code(r"""
MY_CLAUSES = [
    "Neither party may assign this Agreement without the prior written consent of the other party, except to an Affiliate.",
    "In no event shall either party be liable for any indirect, incidental or consequential damages, and each party's total "
    "liability shall not exceed the fees paid in the twelve months before the claim.",
    "Either party may terminate this Agreement for any reason upon sixty (60) days' prior written notice to the other party.",
    "South Dakota",                                     # a fragment, as found among the top anomaly flags
]
t0 = time.perf_counter(); classify(MY_CLAUSES); print(f"({time.perf_counter() - t0:.2f} s)")
""")

interp("demo_task1")
