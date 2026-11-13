# Section 6 of notebooks/STUDENTID_Task1.ipynb: the same-dataset published comparison on LEDGAR (LexGLUE).
# Executed by build_task1_partA.py (exec), so md(), code() and interp() come from there.

md(r"""
## 6. Published comparison on the same dataset: Legal-BERT on LEDGAR (LexGLUE)
**Why.** Our CUAD Task 1 has no published result with the same 27 classes and split, so we cannot compare our Legal-BERT
directly with a paper. LexGLUE (Chalkidis et al., 2022, ACL) reports Legal-BERT on **LEDGAR**, a contract-provision
classification dataset that is also built from SEC EDGAR filings. Running *our* fine-tuning code on LEDGAR and comparing with
the paper's number checks that our recipe and code are sound. Afterwards we relate LEDGAR to CUAD, carefully.

**Data.** `coastalcph/lex_glue`, config `ledgar`: 100 provision labels, train / validation / test = 60,000 / 10,000 / 10,000,
split chronologically by the authors (train 2016–2017, development 2018, test 2019). It was loaded directly with the installed
`datasets` library (no fallback was needed).

**What the paper reports and how** (read from the paper). Table 3, p. 4317: Legal-BERT
**88.2 µ-F1 / 83.0 m-F1**, BERT 87.6 / 81.8, TFIDF+SVM 87.2 / 82.4. Set-up (§5.1, p. 4316): Adam with initial learning rate
3e-5, batch size 8, up to 20 epochs with early stopping on the development set, fp16, five seeds, and the **test score of the
seed with the best development score** (not the mean).

**Our run** (`scripts/run_ledgar.py`): the same checkpoint (`nlpaueb/legal-bert-base-uncased`), [CLS] + softmax head, lr 3e-5,
batch 8, max length 512, AdamW without weight decay with linear decay and no warm-up (the Hugging Face defaults), gradient
clipping 1.0, bf16, early stopping with patience 3 on validation micro-F1. **Cut, logged before training:** at most
**10 epochs** instead of 20, so that **3 seeds** fit the 3-hour budget even in the worst case (benchmark: 5.4 min per epoch).
We report the test score of the **best-validation seed**, exactly as the paper does, evaluated **once** (`legalbert_ledgar`).
TF-IDF + linear SVM (`tfidf_ledgar`) is tuned on LEDGAR validation over the paper's three hyper-parameters (number of features,
C, loss).
""")

code(r'''
LEDGAR = json.load(open(C.RESULTS_DIR / "ledgar_data.json"))
L_NAMES = LEDGAR["labels"]
print("Source:", LEDGAR["source"])
print("Splits:", LEDGAR["splits"], "| labels:", LEDGAR["n_labels"], "| training clauses per label: min", LEDGAR["train_label_counts_min_max"][0],
      "max", LEDGAR["train_label_counts_min_max"][1])
display(pd.DataFrame(LEDGAR["tokens_legalbert"]).T.round(3).rename_axis("Legal-BERT tokens per provision"))
PAPER = pd.DataFrame([("TFIDF+SVM", 87.2, 82.4), ("BERT", 87.6, 81.8), ("Legal-BERT", 88.2, 83.0)],
                     columns=["model", "test µ-F1", "test m-F1"]).assign(source="LexGLUE Table 3, p. 4317")
display(PAPER)
lbench = json.load(open(C.RESULTS_DIR / "ledgar_benchmark.json"))
print(f"Benchmark: {lbench['ms_per_step_batch8']:.0f} ms per training step (batch 8), {lbench['steps_per_epoch']:,} steps -> "
      f"{lbench['projected_epoch_minutes']:.1f} min per epoch incl. validation; 3 seeds x 20 epochs = "
      f"{lbench['projection_hours']['20_epochs_x_3_seeds']:.1f} h (over the 3-hour budget), 3 seeds x 10 epochs = "
      f"{lbench['projection_hours']['10_epochs_x_3_seeds']:.1f} h -> cut to at most 10 epochs, 3 seeds kept.")
''')

code(r'''
RETRAIN_LEDGAR = False    # True: python scripts/run_ledgar.py --train 42 43 44   (about 2 h on an RTX 5080; run detached)
if RETRAIN_LEDGAR:
    subprocess.run([sys.executable, "scripts/run_ledgar.py", "--train", "42", "43", "44"], check=True)
lsearch = pd.read_csv(C.RESULTS_DIR / "ledgar_search.csv")
llog = pd.read_csv(C.RESULTS_DIR / "ledgar_log.csv")
display(lsearch[["run_id", "seed", "best_epoch", "epochs_run", "val_micro_f1", "val_macro_f1", "mean_epoch_train_seconds",
                 "run_seconds", "peak_gpu_mem_gb"]].round(4))
L_BEST = lsearch.sort_values(["val_micro_f1", "val_macro_f1"], ascending=False, kind="stable").iloc[0]
print(f"Best-validation seed (selection as in the paper): {L_BEST.run_id} -> validation µ-F1 {L_BEST.val_micro_f1:.4f}, m-F1 {L_BEST.val_macro_f1:.4f}")
print(f"Validation over our {len(lsearch)} seeds: µ-F1 {100 * lsearch.val_micro_f1.mean():.1f} ± {100 * lsearch.val_micro_f1.std():.1f}, "
      f"m-F1 {100 * lsearch.val_macro_f1.mean():.1f} ± {100 * lsearch.val_macro_f1.std():.1f} "
      "(paper, Tables 5-6 p. 4327, mean ± std of its 3 best seeds: µ-F1 88.5 ± 0.0, m-F1 82.2 ± 0.3)")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4))
cols = [PL.BLUE, PL.ORANGE, PL.AQUA]
for (rid, g), col in zip(llog.groupby("run_id"), cols):
    st, ep = g[g.kind == "step"], g[g.kind == "epoch"]
    axes[0].plot(st.step / lbench["steps_per_epoch"], st.train_loss.rolling(10, min_periods=1).mean(), color=col, linewidth=1.3, label=rid)
    axes[1].plot(ep.epoch, 100 * ep.val_micro_f1, "o-", color=col, label=f"{rid} µ-F1")
    axes[1].plot(ep.epoch, 100 * ep.val_macro_f1, "s--", color=col, alpha=0.7, label=f"{rid} m-F1")
axes[1].axhline(88.5, color=PL.INK, linestyle=":", linewidth=1); axes[1].text(1, 88.7, "paper dev µ-F1 88.5", fontsize=8)
axes[1].axhline(82.2, color=PL.INK_2, linestyle=":", linewidth=1); axes[1].text(1, 82.4, "paper dev m-F1 82.2", fontsize=8)
axes[0].set_yscale("log"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("training loss (moving mean)"); axes[0].legend(fontsize=8)
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation score (%)"); axes[1].legend(fontsize=7, ncol=2)
axes[0].set_title("Training loss", fontsize=10); axes[1].set_title("Validation µ-F1 and m-F1 per epoch", fontsize=10)
fig.suptitle("Figure 30. Legal-BERT on LEDGAR: training loss and validation scores per epoch (3 seeds)", fontweight="bold", y=1.03, fontsize=11)
PL.savefig(fig, 30, "ledgar_curves"); plt.show()
''')

code(r'''
# TF-IDF + linear SVM: the grid on LEDGAR validation (number of features, C, loss), then a refit of the best configuration
from sklearn.svm import LinearSVC
from datasets import load_dataset
lsvm = pd.read_csv(C.RESULTS_DIR / "ledgar_tfidf_search.csv")
display(lsvm.sort_values("val_micro_f1", ascending=False).head(8).round(4))
SV = lsvm.sort_values(["val_micro_f1", "val_macro_f1"], ascending=False, kind="stable").iloc[0]   # ties (identical configs) -> first in grid order
print(f"Selected on validation: max_features={SV.max_features}, C={SV.C}, loss={SV['loss']} -> val µ-F1 {SV.val_micro_f1:.4f}, m-F1 {SV.val_macro_f1:.4f}")
ld = load_dataset("coastalcph/lex_glue", "ledgar")
LF = {s: pd.DataFrame({"text": ld[s]["text"], "label": ld[s]["label"]}) for s in ["train", "validation", "test"]}
mf = None if SV.max_features == "all" else int(SV.max_features)
with C.timer("tfidf_ledgar", "train_final", notes="refit of the selected TF-IDF + LinearSVC configuration on LEDGAR train"):
    vec_l = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=mf)
    svm_l = LinearSVC(C=float(SV.C), loss=SV["loss"], max_iter=5000, random_state=42).fit(vec_l.fit_transform(LF["train"].text), LF["train"].label)
val_check = f1_score(LF["validation"].label, svm_l.predict(vec_l.transform(LF["validation"].text)), average="micro")
assert abs(val_check - SV.val_micro_f1) < 1e-9, "refit does not reproduce the search score"
''')

code(r'''
# Test evaluation, once per model (test-once helper). For single-label classification micro-F1 = accuracy.
yt_l = np.array([L_NAMES[i] for i in LF["test"].label])
ids_l = np.arange(len(yt_l))
pred_file = ROOT / "models" / "ledgar" / L_BEST.run_id / "test_preds.npy"
if not pred_file.exists() and (ROOT / "models" / "ledgar" / L_BEST.run_id / "best").exists():
    subprocess.run([sys.executable, "scripts/run_ledgar.py", "--predict", L_BEST.run_id], check=True)
if pred_file.exists():
    yp_lb = np.array([L_NAMES[i] for i in np.load(pred_file)])
else:
    yp_lb = pd.read_csv(C.RESULTS_DIR / "predictions" / "legalbert_ledgar.csv").pred.values
yp_sv = np.array([L_NAMES[i] for i in svm_l.predict(vec_l.transform(LF["test"].text))])
cfg_lb = {"run": L_BEST.run_id, "selection": "best validation micro-F1 of 3 seeds (as in the paper)", "recipe": json.loads(L_BEST.recipe),
          "val_micro_f1": float(L_BEST.val_micro_f1), "val_macro_f1": float(L_BEST.val_macro_f1)}
cfg_sv = {"max_features": str(SV.max_features), "C": float(SV.C), "loss": SV["loss"], "ngram_range": "(1, 2)", "sublinear_tf": True,
          "min_df": 2, "val_micro_f1": float(SV.val_micro_f1)}
LRES = {"legalbert_ledgar": C.test_once("legalbert_ledgar", cfg_lb, yt_l, yp_lb, L_NAMES, ids_l, ids_l),
        "tfidf_ledgar": C.test_once("tfidf_ledgar", cfg_sv, yt_l, yp_sv, L_NAMES, ids_l, ids_l)}
micro = lambda a, b: f1_score(a, b, average="micro")
rows = []
for name, pred, paper_row in [("tfidf_ledgar", yp_sv, "TFIDF+SVM"), ("legalbert_ledgar", yp_lb, "Legal-BERT")]:
    ci_mi = C.bootstrap_ci(yt_l, pred, ids_l, metric=micro)                 # plain bootstrap over provisions
    ci_ma = C.bootstrap_ci(yt_l, pred, ids_l, labels=L_NAMES)
    p = PAPER.set_index("model").loc[paper_row]
    rows.append({"model": name, "ours test µ-F1": 100 * ci_mi["point"], "µ-F1 95% CI": f"[{100 * ci_mi['ci_low']:.1f}, {100 * ci_mi['ci_high']:.1f}]",
                 "ours test m-F1": 100 * ci_ma["point"], "m-F1 95% CI": f"[{100 * ci_ma['ci_low']:.1f}, {100 * ci_ma['ci_high']:.1f}]",
                 "paper µ-F1": p["test µ-F1"], "paper m-F1": p["test m-F1"],
                 "difference µ-F1": 100 * ci_mi["point"] - p["test µ-F1"], "difference m-F1": 100 * ci_ma["point"] - p["test m-F1"]})
LCMP = pd.DataFrame(rows); display(LCMP.round(2))
LCMP.to_csv(C.RESULTS_DIR / "ledgar_comparison.csv", index=False)
pb_l = C.paired_bootstrap(yt_l, yp_lb, yp_sv, ids_l, labels=L_NAMES)
print(f"Legal-BERT - TF-IDF on LEDGAR test, m-F1: {100 * pb_l['diff']:+.2f} points, 95 % CI [{100 * pb_l['ci_low']:+.2f}, {100 * pb_l['ci_high']:+.2f}]")
json.dump({"comparison": LCMP.to_dict(orient="records"), "paired_legalbert_minus_tfidf_macro": pb_l,
           "best_seed": L_BEST.run_id, "dev_mean_std": {"micro": [float(lsearch.val_micro_f1.mean()), float(lsearch.val_micro_f1.std())],
                                                         "macro": [float(lsearch.val_macro_f1.mean()), float(lsearch.val_macro_f1.std())]}},
          open(C.RESULTS_DIR / "ledgar_results.json", "w"), indent=2, default=float)
''')

code(r'''
# Compute budget of the LEDGAR runs (training rows are logged by scripts/run_ledgar.py)
budget_now = pd.read_csv(C.RESULTS_DIR / "compute_budget.csv")
display(budget_now[budget_now.model.str.contains("ledgar")][["model", "stage", "seconds", "device", "notes"]])
''')

code(r'''
# Reporting as the paper does: best-dev seed = headline (table above); in addition our validation mean +- std next to Tables 5-6
DEV = pd.DataFrame([{"metric": "validation µ-F1", "ours: mean ± std of 3 seeds": f"{100 * lsearch.val_micro_f1.mean():.1f} ± {100 * lsearch.val_micro_f1.std():.1f}",
                     "ours: best-dev seed": round(100 * L_BEST.val_micro_f1, 2), "paper: mean ± std of its 3 best seeds (Table 5, p. 4327)": "88.5 ± 0.0"},
                    {"metric": "validation m-F1", "ours: mean ± std of 3 seeds": f"{100 * lsearch.val_macro_f1.mean():.1f} ± {100 * lsearch.val_macro_f1.std():.1f}",
                     "ours: best-dev seed": round(100 * L_BEST.val_macro_f1, 2), "paper: mean ± std of its 3 best seeds (Table 5, p. 4327)": "82.2 ± 0.3 (Table 6)"}])
display(DEV); DEV.to_csv(C.RESULTS_DIR / "ledgar_dev_mean_std.csv", index=False)
''')

interp("ledgar_compare")

md(r"""
**Learning-curve check (validation only; an extra analysis).** To test the explanation offered below (fine-tuning helps
more when training data are few), Legal-BERT (seed 42, the same LEDGAR recipe, at most 10 epochs) and TF-IDF + LinearSVC
(C and loss re-tuned on validation, all features) were also trained on a **stratified 5,000-provision subset** of LEDGAR
train — roughly the size of our CUAD training set — and compared on the LEDGAR validation set
(`python scripts/run_ledgar.py --subset 5000`). No test evaluation.
""")

code(r'''
ss = pd.read_csv(C.RESULTS_DIR / "ledgar_subset_search.csv").iloc[0]
st_ = pd.read_csv(C.RESULTS_DIR / "ledgar_subset_tfidf.csv").sort_values(["val_micro_f1", "val_macro_f1"], ascending=False, kind="stable").iloc[0]
LC = pd.DataFrame([
    {"training provisions": 5000, "model": "Legal-BERT", "val µ-F1": ss.val_micro_f1, "val m-F1": ss.val_macro_f1, "note": f"seed 42, best epoch {int(ss.best_epoch)}"},
    {"training provisions": 5000, "model": "TF-IDF + LinearSVC", "val µ-F1": st_.val_micro_f1, "val m-F1": st_.val_macro_f1, "note": f"C={st_.C}, {st_['loss']}"},
    {"training provisions": 60000, "model": "Legal-BERT", "val µ-F1": L_BEST.val_micro_f1, "val m-F1": L_BEST.val_macro_f1, "note": "best-dev seed (42)"},
    {"training provisions": 60000, "model": "TF-IDF + LinearSVC", "val µ-F1": SV.val_micro_f1, "val m-F1": SV.val_macro_f1, "note": "selected configuration"}])
display(LC.round(4))
gap = LC.pivot_table(index="training provisions", columns="model", values=["val µ-F1", "val m-F1"])
gap_tab = pd.DataFrame({"Legal-BERT - TF-IDF, µ-F1 (points)": 100 * (gap["val µ-F1"]["Legal-BERT"] - gap["val µ-F1"]["TF-IDF + LinearSVC"]),
                        "Legal-BERT - TF-IDF, m-F1 (points)": 100 * (gap["val m-F1"]["Legal-BERT"] - gap["val m-F1"]["TF-IDF + LinearSVC"])})
display(gap_tab.round(2))
LC.to_csv(C.RESULTS_DIR / "ledgar_learning_curve.csv", index=False)
gap_tab.to_csv(C.RESULTS_DIR / "ledgar_learning_curve_gap.csv")
''')

code(r'''
# Relating LEDGAR to our CUAD Task 1 result
t1 = {m: C.compute_metrics(test.category, pd.read_csv(C.RESULTS_DIR / "predictions" / f"{m}.csv").set_index("row_id").loc[test.row_id].pred,
                           LABELS, m, "test", save=False) for m in ["legalbert", "tfidf_logreg"]}
rel = pd.DataFrame([
    {"": "classes", "CUAD Task 1 (ours)": len(LABELS), "LEDGAR": LEDGAR["n_labels"]},
    {"": "training clauses", "CUAD Task 1 (ours)": len(train), "LEDGAR": LEDGAR["splits"]["train"]},
    {"": "test clauses", "CUAD Task 1 (ours)": len(test), "LEDGAR": LEDGAR["splits"]["test"]},
    {"": "median training clauses per class", "CUAD Task 1 (ours)": int(train.category.value_counts().median()),
     "LEDGAR": int(pd.Series(LF["train"].label).value_counts().median())},
    {"": "split", "CUAD Task 1 (ours)": "by contract (unseen contracts)", "LEDGAR": "chronological (2016-17 / 2018 / 2019)"},
    {"": "label source", "CUAD Task 1 (ours)": "expert clause annotations", "LEDGAR": "single main topic of each provision (paper, p. 4314)"},
    {"": "Legal-BERT test micro-F1 (= accuracy)", "CUAD Task 1 (ours)": round(100 * t1["legalbert"]["accuracy"], 1),
     "LEDGAR": round(LCMP.set_index("model").loc["legalbert_ledgar", "ours test µ-F1"], 1)},
    {"": "Legal-BERT test macro-F1", "CUAD Task 1 (ours)": round(100 * t1["legalbert"]["macro_f1"], 1),
     "LEDGAR": round(LCMP.set_index("model").loc["legalbert_ledgar", "ours test m-F1"], 1)},
    {"": "best TF-IDF test macro-F1", "CUAD Task 1 (ours)": round(100 * t1["tfidf_logreg"]["macro_f1"], 1),
     "LEDGAR": round(LCMP.set_index("model").loc["tfidf_ledgar", "ours test m-F1"], 1)},
]).set_index("")
display(rel)
rel.to_csv(C.RESULTS_DIR / "ledgar_vs_cuad.csv")
''')

interp("ledgar_relate")
