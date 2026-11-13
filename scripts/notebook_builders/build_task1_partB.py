# Section 4 of the Task 1 notebook (classical baselines, design v2). Exec'd by build_task1_partA.py (uses md/code/interp).

md(r"""
## 4. Classical baselines (Task 1)

> **Progress checkpoint (13 Nov 2026):** sections 2 and 4 hold the dataset and the Task 1 baseline results.

### 4.1 The models and why they suit this data
Our data (section 2): a few thousand **short training clauses** (median about 40 words), **27 classes** with
strong imbalance (about 11 : 1 between the largest and smallest class) and several classes with very few examples,
and TF-IDF features that are **sparse and high-dimensional** (tens of thousands of word and word-pair features —
more features than training examples — but only a few dozen non-zero per clause).

| Model | Why it fits this task | What we expect |
|---|---|---|
| **Majority class** (`DummyClassifier`, most frequent) | The floor: always predicts the largest training class. Anything useful must beat it by far. | macro-F1 close to 0 |
| **Linear SVM** (`LinearSVC`) | In very high-dimensional sparse spaces the classes are close to *linearly separable*; a max-margin linear model with L2 regularisation (parameter C) generalises well even with more features than examples, and trains in about a second. The standard strong baseline for TF-IDF text classification. `class_weight="balanced"` re-weights the loss so rare classes are not ignored. | strongest or joint-strongest |
| **Logistic Regression** (multinomial) | Also linear and L2-regularised, but it outputs **class probabilities**, useful for decision support (route low-confidence clauses to a human first, rank clauses for review). Same `class_weight` option. | similar to the SVM |
| **Complement Naive Bayes** | A generative model that is extremely fast and works with few examples per class; the *complement* variant was designed for imbalanced text classes (it estimates each class from all *other* classes' data, which is larger). Its independence assumption is clearly violated by overlapping word pairs, so it is a useful *contrasting* family. | weaker than the linear models |

All three classifiers use the preprocessing selected in section 3 (`P.FINAL_CLASSICAL_CONFIG`). The search below
varies the TF-IDF settings again (including the n-gram range); the preprocessing study only fixes *how text is
turned into tokens*.
""")

md(r"""
### 4.2 Majority-class baseline
""")

code(r'''
from sklearn.dummy import DummyClassifier

majority = DummyClassifier(strategy="most_frequent").fit(train.text, train.category)
maj_val = C.compute_metrics(val.category, majority.predict(val.text), LABELS, "majority", "validation",
                            robust_labels=ROBUST)
print(f"Majority class = '{majority.classes_[majority.class_prior_.argmax()]}' | validation macro-F1 "
      f"{maj_val['macro_f1']:.4f}, accuracy {maj_val['accuracy']:.4f}")
''')

md(r"""
### 4.3 Systematic hyper-parameter search (validation set only)
**Protocol.** Every combination in the grid below is trained on *train* and scored on *validation* (macro-F1 over
all 27 classes). The test set is not touched. Every configuration is recorded in `results/classical_search.csv` with
its validation macro-F1, robustness macro-F1, accuracy, number of features and fit time.

| Part | Setting | Values | Why it is searched |
|---|---|---|---|
| TF-IDF (word) | `ngram_range` | (1,1), (1,2), (1,3) | word pairs/triples capture legal phrases ("governing law") |
| | `min_df` | 1, 2, 5 | drop features seen in very few clauses (typos, names) |
| | `max_df` | 0.5, 1.0 | drop features in > 50 % of clauses (a data-driven stop-word filter) |
| | `sublinear_tf` | False, True | 1 + log(tf): a word repeated 5 times is not 5 times as informative |
| | `max_features` | all, 20,000 | cap the vocabulary (simpler, faster model) |
| TF-IDF (character) | `char_wb` n-grams | (2,5), (3,6) × `min_df` 2, 5 × `sublinear_tf` | robust to word variants and typos from the PDF conversion |
| LinearSVC | `C` | 0.03, 0.1, 0.3, 1, 3, 10, 30 | regularisation strength (small C = simpler model) |
| | `class_weight` | none, **balanced** | up-weight rare classes (27 imbalanced classes) |
| Logistic Regression | `C` | 1, 3, 10, 30, 100, 300 | regularisation strength |
| | `class_weight` | none, **balanced** | up-weight rare classes |
| Complement NB | `alpha` | 0.003, 0.01, 0.03, 0.1, 0.3, 1 | smoothing of word counts |
| | `norm` | False, True | weight normalisation (part of the complement NB method) |

**Token configuration (added after the preprocessing study, before any tuning).** The study in 3.3 decided the
stop-word step on a near-tie (−0.003, inside the ±0.003 noise band) using an *untuned* classifier, and removing stop
words changes what word pairs can capture. So the word-level grid is run twice: with the study's token configuration
(`study`) and with the identical configuration except that stop words are kept (`study_keep_stopwords`). This uses the
validation set only, like the rest of the search.

That is (72 word settings × 2 token configurations + 8 character settings) × 38 classifier settings = **5,776
trained models**, run in parallel on the CPU.
The grid edges were set wide on purpose; a cell below checks whether any selected value still sits on an edge.
**Selection rule (fixed in advance):** per model, the configuration with the highest validation macro-F1;
configurations within 0.001 of the best are treated as tied, and the one with the fewest features (then the fastest
fit) wins.

`RUN_SEARCH = False` loads the saved search; the outputs shown below are from the real search run.
""")

code(r'''
import time, itertools
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score

RUN_SEARCH = False   # the real search ran once (its output is kept below); True = run all configurations again (~1 h)
SEARCH_CSV = C.RESULTS_DIR / "classical_search.csv"
TOKEN_CONFIGS = {"study": P.FINAL_CLASSICAL_CONFIG,
                 "study_keep_stopwords": {**P.FINAL_CLASSICAL_CONFIG, "stopwords": "none"}}
CFG = P.FINAL_CLASSICAL_CONFIG          # character models use the study's cleaned text

# Pre-process once: word tokens per token configuration, cleaned strings for the character models.
TOK = {name: {"train": [P.tokenize(t, c) for t in train.text], "val": [P.tokenize(t, c) for t in val.text]}
       for name, c in TOKEN_CONFIGS.items()}
TXT = {"train": [P.clean_text(t, CFG) for t in train.text], "val": [P.clean_text(t, CFG) for t in val.text]}

VEC_GRID = (
    [dict(analyzer="word", tok=tk, ngram_range=ng, min_df=mn, max_df=mx, sublinear_tf=sl, max_features=mf)
     for tk, ng, mn, mx, sl, mf in itertools.product(TOKEN_CONFIGS, [(1, 1), (1, 2), (1, 3)], [1, 2, 5], [0.5, 1.0],
                                                      [False, True], [None, 20000])]
    + [dict(analyzer="char_wb", tok="study", ngram_range=ng, min_df=mn, max_df=1.0, sublinear_tf=sl, max_features=None)
       for ng, mn, sl in itertools.product([(2, 5), (3, 6)], [2, 5], [False, True])])
GRID_VALUES = {"LinearSVC": [0.03, 0.1, 0.3, 1, 3, 10, 30], "LogisticRegression": [1, 3, 10, 30, 100, 300],
               "ComplementNB": [0.003, 0.01, 0.03, 0.1, 0.3, 1]}
CLF_GRID = (
    [("LinearSVC", dict(C=c, class_weight=w)) for c in GRID_VALUES["LinearSVC"] for w in [None, "balanced"]]
    + [("LogisticRegression", dict(C=c, class_weight=w)) for c in GRID_VALUES["LogisticRegression"]
       for w in [None, "balanced"]]
    + [("ComplementNB", dict(alpha=a, norm=n)) for a in GRID_VALUES["ComplementNB"] for n in [False, True]])

def identity(x):
    return x

def make_tfidf(v):
    if v["analyzer"] == "word":     # input = our token lists
        return TfidfVectorizer(tokenizer=identity, preprocessor=identity, lowercase=False, token_pattern=None,
                               ngram_range=v["ngram_range"], min_df=v["min_df"], max_df=v["max_df"],
                               sublinear_tf=v["sublinear_tf"], max_features=v["max_features"])
    return TfidfVectorizer(analyzer="char_wb", lowercase=False, ngram_range=v["ngram_range"], min_df=v["min_df"],
                           max_df=v["max_df"], sublinear_tf=v["sublinear_tf"])

def make_clf(name, p):
    if name == "LinearSVC":
        return LinearSVC(C=p["C"], class_weight=p["class_weight"], random_state=42, max_iter=5000)
    if name == "LogisticRegression":
        return LogisticRegression(C=p["C"], class_weight=p["class_weight"], max_iter=3000)
    return ComplementNB(alpha=p["alpha"], norm=p["norm"])

def run_vec_config(v, y_tr, y_va):
    data = TOK[v["tok"]] if v["analyzer"] == "word" else TXT
    vec = make_tfidf(v)
    Xtr = vec.fit_transform(data["train"])
    Xva = vec.transform(data["val"])
    out = []
    for name, p in CLF_GRID:
        t0 = time.perf_counter()
        clf = make_clf(name, p).fit(Xtr, y_tr)
        fit_s = time.perf_counter() - t0
        pred = clf.predict(Xva)
        out.append({"model": name, "analyzer": v["analyzer"], "token_config": v["tok"],
                    "ngram_range": str(v["ngram_range"]),
                    "min_df": v["min_df"], "max_df": v["max_df"], "sublinear_tf": v["sublinear_tf"],
                    "max_features": v["max_features"] or "all",
                    "C_or_alpha": p.get("C", p.get("alpha")),
                    "class_weight_or_norm": str(p.get("class_weight", p.get("norm"))),
                    "n_features": Xtr.shape[1],
                    "val_macro_f1": f1_score(y_va, pred, labels=LABELS, average="macro", zero_division=0),
                    "val_macro_f1_robust": f1_score(y_va, pred, labels=ROBUST, average="macro", zero_division=0),
                    "val_accuracy": accuracy_score(y_va, pred), "fit_seconds": fit_s})
    return out

if RUN_SEARCH:
    n_jobs = max(1, os.cpu_count() - 2)
    with C.timer("classical_search", "grid_search",
                 notes=f"{len(VEC_GRID) * len(CLF_GRID)} configs; {n_jobs} parallel CPU workers; validation only"):
        rows = Parallel(n_jobs=n_jobs)(delayed(run_vec_config)(v, train.category.values, val.category.values)
                                       for v in VEC_GRID)
    search = pd.DataFrame([r for rs in rows for r in rs])
    search.to_csv(SEARCH_CSV, index=False)
else:
    search = pd.read_csv(SEARCH_CSV, keep_default_na=False)   # keep the text "None" (no class weights) as text
print(f"{len(search):,} configurations recorded in {SEARCH_CSV.name} "
      f"(sum of individual fit times: {search.fit_seconds.sum() / 60:.1f} min)")
search.groupby("model").val_macro_f1.describe()[["count", "min", "50%", "max"]].round(4)
''')

code(r'''
# Top 10 configurations per model (validation macro-F1)
pd.set_option("display.max_colwidth", 40)
for name in ["LinearSVC", "LogisticRegression", "ComplementNB"]:
    print(f"\n=== {name}: top 10 of {len(search[search.model == name])} configurations ===")
    display(search[search.model == name].sort_values("val_macro_f1", ascending=False).head(10)
            .drop(columns="model").round(4).reset_index(drop=True))
''')

code(r'''
MODEL_COLORS = {"LinearSVC": PL.BLUE, "LogisticRegression": PL.ORANGE, "ComplementNB": PL.AQUA}
WEIGHT_LABEL = {"None": "no class weights", "balanced": "balanced weights", "False": "norm=False", "True": "norm=True"}
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for ax, name in zip(axes, MODEL_COLORS):
    g = search[search.model == name]
    for w, ls, mk in zip(sorted(g.class_weight_or_norm.unique()), ["-", "--"], ["o", "s"]):
        best = g[g.class_weight_or_norm == w].groupby("C_or_alpha").val_macro_f1.max()
        med = g[g.class_weight_or_norm == w].groupby("C_or_alpha").val_macro_f1.median()
        ax.plot(best.index, best.values, ls, marker=mk, color=MODEL_COLORS[name], linewidth=2, markersize=6,
                label=f"{WEIGHT_LABEL[w]}: best")
        ax.plot(med.index, med.values, ls, color=MODEL_COLORS[name], alpha=0.35, linewidth=1.5,
                label=f"{WEIGHT_LABEL[w]}: median")
    ax.set_xscale("log"); ax.set_title(name)
    ax.set_xlabel("alpha (smoothing)" if name == "ComplementNB" else "C (inverse regularisation)")
    ax.legend(fontsize=7.5, loc="lower right")
axes[0].set_ylabel("Validation macro-F1 (27 classes)")
fig.suptitle("Figure 8. Validation macro-F1 vs regularisation / smoothing (best and median over the 80 TF-IDF settings)",
             fontweight="bold", y=1.03, fontsize=11)
PL.savefig(fig, 8, "classical_search_C"); plt.show()
''')

code(r'''
# Which TF-IDF settings mattered? Best (filled) and median (hollow) validation macro-F1 for each value.
search["features"] = np.where(search.analyzer == "word", "word " + search.ngram_range, "char " + search.ngram_range)
search["tokens"] = np.where(search.analyzer == "char_wb", "char (n/a)",
                            search.token_config.map({"study": "stop words removed", "study_keep_stopwords": "stop words kept"}))
params = [("features", "Features"), ("tokens", "Tokens (word models)"), ("min_df", "min_df"), ("max_df", "max_df"),
          ("sublinear_tf", "sublinear_tf"), ("max_features", "max_features")]
fig, axes = plt.subplots(1, len(params), figsize=(15, 3.8), sharey=True,
                         gridspec_kw={"width_ratios": [2.2, 1.2, 1, 1, 1, 1]})
for ax, (col, title) in zip(axes, params):
    vals = sorted(search[col].astype(str).unique())
    for k, name in enumerate(MODEL_COLORS):
        g = search[search.model == name]
        x = np.arange(len(vals)) + (k - 1) * 0.22
        best = [g[g[col].astype(str) == v].val_macro_f1.max() for v in vals]
        med = [g[g[col].astype(str) == v].val_macro_f1.median() for v in vals]
        ax.scatter(x, best, color=MODEL_COLORS[name], s=40, label=f"{name} best" if col == "features" else None, zorder=3)
        ax.scatter(x, med, facecolors="none", edgecolors=MODEL_COLORS[name], s=40,
                   label=f"{name} median" if col == "features" else None, zorder=3)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(vals, rotation=35, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Validation macro-F1 (27 classes)")
fig.legend(loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.24))
fig.suptitle("Figure 9. Sensitivity of validation macro-F1 to each TF-IDF setting (filled = best, hollow = median)",
             fontweight="bold", y=1.03, fontsize=11)
PL.savefig(fig, 9, "classical_search_sensitivity"); plt.show()
''')

code(r'''
# Selection rule: highest validation macro-F1; within 0.001 of the best -> fewest features, then fastest.
def select(g):
    cand = g[g.val_macro_f1 >= g.val_macro_f1.max() - 0.001]
    return cand.sort_values(["n_features", "fit_seconds"]).iloc[0]

selected = search.groupby("model").apply(select, include_groups=False)
display(selected.drop(columns="features").round(4))
# Edge check: does a selected regularisation / smoothing value sit on the edge of its grid?
for name, row in selected.iterrows():
    grid = GRID_VALUES[name]
    v = float(row.C_or_alpha)
    edge = "ON THE EDGE of the grid" if v in (min(grid), max(grid)) else "inside the grid"
    print(f"{name:>18}: selected {'alpha' if name == 'ComplementNB' else 'C'} = {v:g} -> {edge} {grid}")
''')

interp("search")

md(r"""
### 4.4 Validation confusions: testing the hypotheses of section 2.2
Before the test set is used at all, we look at **which classes the best classical model (by validation macro-F1)
confuses on validation**, and check the confusion hypotheses written down in section 2.2.
""")

code(r'''
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

def build_final(row, name):
    """Raw text -> selected TF-IDF -> selected classifier."""
    mf = None if str(row.max_features) == "all" else int(row.max_features)
    ng = tuple(int(x) for x in row.ngram_range.strip("()").split(","))
    if row.analyzer == "word":
        vec = P.make_vectorizer(TOKEN_CONFIGS[row.token_config], min_df=int(row.min_df), max_df=float(row.max_df),
                                sublinear_tf=bool(row.sublinear_tf), max_features=mf)
        vec.set_params(ngram_range=ng)
        if mf is not None:   # numpy breaks ties between equally frequent words differently on different CPUs (AVX-512
            # vs AVX2); where the max_features cut-off falls inside such a tie, the kept words of the original run are loaded
            key = (f"{row.token_config} {row.ngram_range} min_df={int(row.min_df)} max_df={float(row.max_df)} "
                   f"sublinear_tf={bool(row.sublinear_tf)} max_features={mf}")
            frozen = json.load(open(C.RESULTS_DIR / "frozen_vocabularies.json", encoding="utf-8"))
            if key in frozen:
                vec.set_params(vocabulary=frozen[key], max_features=None)
    else:
        vec = TfidfVectorizer(analyzer="char_wb", preprocessor=lambda s: P.clean_text(s, CFG), lowercase=False,
                              ngram_range=ng, min_df=int(row.min_df), max_df=float(row.max_df),
                              sublinear_tf=bool(row.sublinear_tf))
    cw = row.class_weight_or_norm
    p = ({"alpha": float(row.C_or_alpha), "norm": cw == "True"} if name == "ComplementNB"
         else {"C": float(row.C_or_alpha), "class_weight": None if cw == "None" else cw})
    return make_pipeline(vec, make_clf(name, p))

NAMES = {"LinearSVC": "tfidf_linearsvc", "LogisticRegression": "tfidf_logreg", "ComplementNB": "tfidf_complementnb"}
final_models, val_pred = {}, {}
for model, row in selected.iterrows():
    with C.timer(NAMES[model], "train_final", notes="refit of selected config on train"):
        pipe = build_final(row, model).fit(train.text, train.category)
    final_models[model] = pipe
    val_pred[model] = pipe.predict(val.text)
    m = C.compute_metrics(val.category, val_pred[model], LABELS, NAMES[model], "validation", robust_labels=ROBUST)
    assert abs(m["macro_f1"] - row.val_macro_f1) < 1e-9, "refit does not reproduce the search score"
    print(f"{model:>18}: validation macro-F1 {m['macro_f1']:.4f} (identical to the search), "
          f"robustness macro-F1 {m['macro_f1_robust']:.4f}")

BEST = selected.val_macro_f1.idxmax()              # chosen on VALIDATION, not on test
print(f"\nBest classical model by validation macro-F1: {BEST}")

def top_pairs(y_true, y_pred, k=10):
    cmx = confusion_matrix(y_true, y_pred, labels=LABELS)
    rows_ = [(LABELS[i], LABELS[j], int(cmx[i, j]), cmx[i, j] / cmx[i].sum()) for i in range(len(LABELS))
             for j in range(len(LABELS)) if i != j and cmx[i, j] > 0]
    out = (pd.DataFrame(rows_, columns=["true class", "predicted as", "clauses", "share of true class"])
           .sort_values(["clauses", "share of true class"], ascending=False).head(k).reset_index(drop=True))
    out.index += 1
    return out

val_conf = top_pairs(val.category, val_pred[BEST])
val_conf.to_csv(C.RESULTS_DIR / "top_confusions_validation.csv")
display(val_conf.round(3))

# Hypothesis check: how often is each new class predicted as its hypothesised neighbour (validation)?
cmv = pd.DataFrame(confusion_matrix(val.category, val_pred[BEST], labels=LABELS), index=LABELS, columns=LABELS)
hyp_check = hyp[["new class", "likely confused with"]].copy()
hyp_check["val clauses of new class"] = [int(cmv.loc[n].sum()) for n in hyp_check["new class"]]
hyp_check["predicted as that class"] = [int(cmv.loc[n, e]) for n, e in zip(hyp_check["new class"], hyp_check["likely confused with"])]
hyp_check["and the reverse"] = [int(cmv.loc[e, n]) for n, e in zip(hyp_check["new class"], hyp_check["likely confused with"])]
hyp_check["validation recall of new class"] = [cmv.loc[n, n] / cmv.loc[n].sum() for n in hyp_check["new class"]]
hyp_check["supported?"] = np.where(hyp_check["predicted as that class"] + hyp_check["and the reverse"] > 0, "yes", "no")
hyp_check.to_csv(C.RESULTS_DIR / "confusion_hypotheses_check.csv", index=False)
display(hyp_check.round(3))
''')

interp("valconf")

md(r"""
### 4.5 Final models: refit on train, then one evaluation on test
**Refit on train only.** Each selected configuration was refitted on the training set exactly as in the search
(validation is *not* merged into training). Reasons: (1) the selected hyper-parameters were optimal for a model trained
on the training set, and adding the validation clauses changes the effective regularisation; (2) the fine-tuned
transformer needs validation for choosing the epoch, so every model in the comparison is trained on the same data;
(3) the validation score of each final model then matches the search exactly (checked above).

The final pipelines start from **raw text** (our tokenizer is built into the vectorizer), so exactly the same
preprocessing is applied to training, validation and test text. Only now are the four models scored on the test set —
**once each**, through `C.test_once`, which logs the evaluation in `results/test_eval_log.csv` and refuses a second
evaluation (the helper has no override in its normal path).
""")

code(r'''
test_pred, test_res = {}, {}
test_res["majority"] = C.test_once("majority", {"strategy": "most_frequent"}, test.category,
                                   majority.predict(test.text), LABELS, test.row_id, test.contract, ROBUST)
for model, pipe in final_models.items():
    cfg = {"preprocessing": TOKEN_CONFIGS[selected.loc[model, "token_config"]], **{k: (v if not isinstance(v, (np.integer, np.floating)) else v.item())
                                    for k, v in selected.loc[model].drop(["features"]).items()}}
    test_pred[model] = pipe.predict(test.text)
    test_res[model] = C.test_once(NAMES[model], cfg, test.category, test_pred[model], LABELS,
                                  test.row_id, test.contract, ROBUST)
''')

code(r'''
# Summary table with contract-level bootstrap 95 % confidence intervals on test.
# Point estimates use all 27 classes (or the 24 robustness classes); inside each resample macro-F1 is averaged over the
# classes present in that resample's true labels (section 2.8). This is re-analysis of saved predictions, not a new test run.
budget = pd.read_csv(C.RESULTS_DIR / "compute_budget.csv")
rows = []
for key, name in [("majority", "majority")] + [(m, NAMES[m]) for m in final_models]:
    preds = majority.predict(test.text) if key == "majority" else test_pred[key]
    ci = C.bootstrap_ci(test.category.values, preds, test.contract.values, labels=LABELS, n_boot=1000)
    cir = C.bootstrap_ci(test.category.values, preds, test.contract.values, labels=ROBUST, n_boot=1000)
    vm = maj_val["macro_f1"] if key == "majority" else selected.loc[key, "val_macro_f1"]
    fit = budget[(budget.model == name) & (budget.stage == "train_final")].seconds
    rows.append({"model": name, "val_macro_f1": vm, "test_macro_f1": test_res[key]["macro_f1"],
                 "test_macro_f1_95ci": f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]", "bootstrap_median": ci["boot_median"],
                 f"test_macro_f1_robust_{len(ROBUST)}": test_res[key]["macro_f1_robust"],
                 "robust_95ci": f"[{cir['ci_low']:.3f}, {cir['ci_high']:.3f}]",
                 "test_accuracy": test_res[key]["accuracy"], "test_weighted_f1": test_res[key]["weighted_f1"],
                 "train_seconds": fit.iloc[-1] if len(fit) else np.nan})
summary = pd.DataFrame(rows)
summary.to_csv(C.RESULTS_DIR / "classical_summary.csv", index=False)
display(summary.round(4))

SECOND = selected.val_macro_f1.drop(BEST).idxmax()
pb = C.paired_bootstrap(test.category.values, test_pred[BEST], test_pred[SECOND], test.contract.values,
                        labels=LABELS, n_boot=1000)
print(f"Best classical model (chosen by validation macro-F1): {BEST}.")
print(f"Test macro-F1 difference {BEST} - {SECOND}: {pb['diff']:+.4f}  95 % CI [{pb['ci_low']:+.4f}, {pb['ci_high']:+.4f}]"
      f"  (share of resamples where {BEST} is not better: {pb['p_a_not_better']:.3f})")
gap = test_res[BEST]["macro_f1"] - test_res["majority"]["macro_f1"]
print(f"Objective O2 (best TF-IDF beats majority by >= 0.50 macro-F1 on test): gap = {gap:.3f} -> "
      f"{'MET' if gap >= 0.50 else 'NOT MET'}")
json.dump({"best_classical": NAMES[BEST], "second": NAMES[SECOND], "paired_bootstrap_test": pb,
           "O2_gap": gap}, open(C.RESULTS_DIR / "classical_best.json", "w"), indent=2)
''')

interp("test")

md(r"""
### 4.6 The best classical model in detail
Per-class results with 95 % contract-level bootstrap CIs (rare classes get wide intervals), the confusion matrices,
the most-confused pairs on test, and the most important features per class.
""")

code(r'''
pc = C.per_class_bootstrap_ci(test.category.values, test_pred[BEST], test.contract.values, LABELS, n_boot=1000)
pr = pd.DataFrame(test_res[BEST]["per_class"]).T[["precision", "recall"]]
pc = pr.join(pc).sort_values("f1")
pc["recall < 0.70"] = np.where(pc.recall < 0.70, "*", "")
pc["< 15 test clauses"] = np.where(pc.support < C.MIN_TEST_CLAUSES_ROBUST, "rare", "")
pc["CI"] = np.where(pc.ci_low.isna(), "not estimable (1 contract)", "")
pc.to_csv(C.RESULTS_DIR / "per_class_best_classical.csv")
print(f"{NAMES[BEST]} - per-class results on the test set (sorted by F1)")
display(pc.round(3))

fig, ax = plt.subplots(figsize=(9, 8.5))
yy = np.arange(len(pc))
colors = [PL.ORANGE if s < C.MIN_TEST_CLAUSES_ROBUST else PL.BLUE for s in pc.support]
ok = pc.ci_low.notna().values
ax.errorbar(pc.f1[ok], yy[ok], xerr=[(pc.f1 - pc.ci_low)[ok], (pc.ci_high - pc.f1)[ok]], fmt="none", ecolor="#9a988f", capsize=3)
ax.scatter(pc.f1, yy, c=colors, s=40, zorder=3)
for y_, (c_, r_) in zip(yy, pc.iterrows()):
    if pd.isna(r_.ci_low):
        ax.text(r_.f1 + 0.01, y_, "CI not estimable (all clauses in 1 contract)", va="center", fontsize=7.5, color=PL.INK_2)
ax.set_yticks(yy); ax.set_yticklabels([f"{c} (n={s}, {k} contr.)" for c, s, k in zip(pc.index, pc.support, pc.test_contracts)],
                                      fontsize=8.5)
ax.set_xlabel("Test F1 with 95 % contract-level bootstrap CI"); ax.grid(axis="y", visible=False)
ax.set_title(f"Figure 10. Per-class test F1 of {NAMES[BEST]} (orange = fewer than {C.MIN_TEST_CLAUSES_ROBUST} test clauses)",
             fontsize=11)
PL.savefig(fig, 10, "per_class_f1_best_classical"); plt.show()
''')

code(r'''
SHORT = {"Post-Termination Services": "Post-Term. Services", "Termination For Convenience": "Termination For Conv.",
         "Ip Ownership Assignment": "IP Ownership Assign.", "Revenue/Profit Sharing": "Revenue/Profit Shar.",
         "Competitive Restriction Exception": "Compet. Restr. Exception", "Non-Transferable License": "Non-Transf. License",
         "No-Solicit Of Employees": "No-Solicit Employees", "Joint Ip Ownership": "Joint IP Ownership"}
short = [SHORT.get(l, l) for l in LABELS]

def plot_cm(mat, fmt, title, number, fname, vmax=None):
    fig, ax = plt.subplots(figsize=(13.5, 12))
    im = ax.imshow(mat, cmap=PL.BLUES, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(short, rotation=65, ha="right", fontsize=8)
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.grid(False)
    thresh = (vmax or mat.max()) * 0.55
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if mat[i, j] > 0:
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", fontsize=6,
                        color="white" if mat[i, j] > thresh else PL.INK)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    ax.set_title(title, fontsize=11)
    PL.savefig(fig, number, fname); plt.show()

cm = np.array(test_res[BEST]["confusion_matrix"])
plot_cm(cm / cm.sum(axis=1, keepdims=True), ".2f",
        f"Figure 11. Confusion matrix of {NAMES[BEST]} on test, normalised by true class (diagonal = recall)",
        11, "cm_best_classical_normalised", vmax=1)
plot_cm(cm, "d", f"Figure 12. Confusion matrix of {NAMES[BEST]} on test (clause counts)", 12, "cm_best_classical_counts")
''')

code(r'''
# The 10 most frequent confusions on test (true -> predicted), for comparison with validation (4.4)
test_conf = top_pairs(test.category, test_pred[BEST])
test_conf.to_csv(C.RESULTS_DIR / "top_confusions_best_classical.csv")
display(test_conf.round(3))
''')

code(r'''
# Interpretability: the TF-IDF features with the largest positive weight for each class (preprocessed tokens)
pipe = final_models[BEST]
vec_b, clf_b = pipe[0], pipe[-1]
feat = vec_b.get_feature_names_out()
weights = clf_b.coef_ if hasattr(clf_b, "coef_") else clf_b.feature_log_prob_
top = {cls: ", ".join(feat[np.argsort(weights[k])[::-1][:8]]) for k, cls in enumerate(clf_b.classes_)}
pd.set_option("display.max_colwidth", 140)
display(pd.DataFrame.from_dict(top, orient="index", columns=[f"{NAMES[BEST]}: top 8 character n-gram features"]))
''')

md(r"""
**Readable version: the best word-level Logistic Regression.** Character fragments are hard to read, so we also fit the
Logistic Regression configuration with the highest validation macro-F1 **among word-level settings** (same selection
rule as above, restricted to `analyzer = word`) and show its strongest words. This model is used only for
interpretation; it is not evaluated on the test set.
""")

code(r"""
wl = search[(search.model == "LogisticRegression") & (search.analyzer == "word")]
word_row = select(wl)
print("Best word-level Logistic Regression (validation):",
      {k: (word_row[k].item() if hasattr(word_row[k], "item") else word_row[k])
       for k in ["token_config", "ngram_range", "min_df", "max_df", "sublinear_tf", "max_features",
                 "C_or_alpha", "class_weight_or_norm", "val_macro_f1"]})
word_pipe = build_final(word_row, "LogisticRegression").fit(train.text, train.category)
assert abs(f1_score(val.category, word_pipe.predict(val.text), labels=LABELS, average="macro", zero_division=0)
           - word_row.val_macro_f1) < 1e-9
wfeat = word_pipe[0].get_feature_names_out()
wcoef = word_pipe[-1].coef_
wtop = {cls: ", ".join(wfeat[np.argsort(wcoef[k])[::-1][:10]]) for k, cls in enumerate(word_pipe[-1].classes_)}
pd.DataFrame.from_dict(wtop, orient="index", columns=["word-level Logistic Regression: top 10 words"])
""")

interp("best")

md(r"""
### 4.7 Leakage check with the best classical model
Section 2.1 used a default model. Here the **selected best classical pipeline** is trained and scored on a random
clause-level split and on a contract-level split of the *train + validation* contracts (same held-out size, 5 seeds;
test contracts not used), to measure how much a naive split would overstate its performance.
""")

code(r'''
best_row = selected.loc[BEST]
def leak_fit(seed, kind, tr_i, ho_i):
    m_ = build_final(best_row, BEST).fit(pool.text.iloc[tr_i], pool.category.iloc[tr_i])
    pred = m_.predict(pool.text.iloc[ho_i])
    return {"seed": seed, "split": kind, "held_out": len(ho_i),
            "macro_f1": f1_score(pool.category.iloc[ho_i], pred, labels=LABELS, average="macro", zero_division=0)}

jobs = []
for seed in range(5):
    tr_r, ho_r = train_test_split(np.arange(len(pool)), test_size=0.2, random_state=seed, stratify=pool.category)
    tr_g, ho_g = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(pool, groups=pool.contract))
    jobs += [(seed, "random clause-level", tr_r, ho_r), (seed, "contract-level", tr_g, ho_g)]
rows = Parallel(n_jobs=-1)(delayed(leak_fit)(*j) for j in jobs)   # the 10 fits are independent: run them in parallel
leak_best = pd.DataFrame(rows)
leak_best.to_csv(C.RESULTS_DIR / "leakage_check_best_classical.csv", index=False)
lb_sum = leak_best.groupby("split")[["held_out", "macro_f1"]].agg(["mean", "std"])
display(lb_sum.round(4))
infl = lb_sum.loc["random clause-level", ("macro_f1", "mean")] - lb_sum.loc["contract-level", ("macro_f1", "mean")]
print(f"A random clause-level split overstates the best classical model's macro-F1 by {infl:+.3f} on average.")
''')

interp("leakbest")

md(r"""
### 4.8 What this tells us so far
""")
interp("sofar")
