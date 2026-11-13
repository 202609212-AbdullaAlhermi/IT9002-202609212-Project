"""Task 2 baseline search (VALIDATION only): TF-IDF segment classifier, one binary Logistic Regression per category.

    python scripts/t2_baseline_search.py     -> results/task2_baseline_search.csv, results/task2_baseline_choice.json

Grid: C in {0.3, 1, 3, 10, 30, 100} x class_weight in {None, balanced} x decision threshold in {0.1, ..., 0.9}
(first run used C <= 10 and thresholds 0.3-0.7; its optimum sat on the grid edge, so the grid was extended)
(the threshold needs no refit). TF-IDF: word 1-2-grams, lower-case, sublinear tf, min_df 2, fitted on TRAIN segments.
Selection rule: highest validation macro-F1 at IoU 0.5 (ties -> larger C is NOT preferred: smaller C, then higher threshold).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import common as C, span_metrics as M, task2_baseline as B  # noqa: E402

TFIDF = dict(ngram_range=(1, 2), lowercase=True, sublinear_tf=True, min_df=2)
GRID_C, GRID_W, GRID_T = [0.3, 1, 3, 10, 30, 100], [None, "balanced"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def build(split, cs, sp, cats):
    ci = {c: i for i, c in enumerate(cats)}
    con = list(cs[cs.split == split].contract)
    texts, Y, index = [], [], []
    for c in con:
        t = C.load_contract_text(c); segs = B.segment(t)
        g = sp[sp.contract == c]
        Y.append(B.segment_labels(segs, [(ci[r.category], r.start, r.end) for r in g.itertuples()], len(cats)))
        texts += [t[s:e] for s, e in segs]; index += [(c, s, e) for s, e in segs]
    return con, texts, np.vstack(Y), index


def fit_one(X, y, Cval, w):
    return LogisticRegression(C=Cval, class_weight=w, solver="liblinear", max_iter=2000).fit(X, y)


def spans_from(probs, index, contracts, thresh, cats):
    rows = []
    by_c = {}
    for i, (c, s, e) in enumerate(index):
        by_c.setdefault(c, []).append(i)
    for c in contracts:
        ii = by_c.get(c, [])
        segs = [(index[i][1], index[i][2]) for i in ii]
        p = probs[ii]
        for k, s, e, conf in B.merge_segments(segs, p >= thresh, p):
            rows.append((c, cats[k], s, e, conf))
    return pd.DataFrame(rows, columns=["contract", "category", "start", "end", "confidence"])


if __name__ == "__main__":
    cs, sp = C.load_contract_split(), C.load_task2_spans()
    cats = sorted(sp.category.unique())
    t0 = time.perf_counter()
    _, tr_text, Ytr, _ = build("train", cs, sp, cats)
    va_con, va_text, Yva, va_index = build("validation", cs, sp, cats)
    vec = TfidfVectorizer(**TFIDF).fit(tr_text)
    Xtr, Xva = vec.transform(tr_text), vec.transform(va_text)
    print(f"train segments {Xtr.shape[0]:,}, features {Xtr.shape[1]:,}; validation segments {Xva.shape[0]:,}", flush=True)
    gold_va = sp[sp.split == "validation"]
    rows = []
    with C.timer("t2_tfidf_segments", "grid_search", notes="27 one-vs-rest LogReg x 8 settings x 5 thresholds; validation only"):
        for Cval in GRID_C:
            for w in GRID_W:
                t1 = time.perf_counter()
                models = Parallel(n_jobs=14)(delayed(fit_one)(Xtr, Ytr[:, k], Cval, w) for k in range(len(cats)))
                fit_s = time.perf_counter() - t1
                P = np.column_stack([m.predict_proba(Xva)[:, 1] for m in models])
                for th in GRID_T:
                    pred = spans_from(P, va_index, va_con, th, cats)
                    s = M.summarise(M.count_table(gold_va, pred, va_con, cats), cats)
                    rows.append({"C": Cval, "class_weight": str(w), "threshold": th, "val_macro_f1": s["macro_f1"],
                                 "val_micro_f1": s["micro_f1"], "val_macro_precision": s["macro_precision"],
                                 "val_macro_recall": s["macro_recall"], "fit_seconds": fit_s})
                    print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(C.RESULTS_DIR / "task2_baseline_search.csv", index=False)
    best = res.sort_values(["val_macro_f1", "C", "threshold"], ascending=[False, True, False]).iloc[0]
    choice = {"C": float(best.C), "class_weight": None if best.class_weight == "None" else best.class_weight,
              "threshold": float(best.threshold), "val_macro_f1": float(best.val_macro_f1), "tfidf": {**TFIDF, "ngram_range": [1, 2]},
              "search_seconds": time.perf_counter() - t0}
    (C.RESULTS_DIR / "task2_baseline_choice.json").write_text(json.dumps(choice, indent=2))
    print(choice)
