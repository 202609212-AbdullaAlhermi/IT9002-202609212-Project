"""Evaluation of Task 2 (clause extraction): predicted character spans vs gold character spans, per contract and category.

Two families of metrics:

1. OUR entity-level metrics (primary): for each (contract, category), predicted spans are matched ONE-TO-ONE to gold spans,
   greedily in order of confidence; a pair matches if
     * IoU >= 0.5  - character-level Jaccard |A∩B| / |A∪B| of the two character ranges (CUAD's 0.5 threshold), or
     * exact       - both spans cover exactly the same word-pieces (token-boundary match).
   Unmatched predictions are false positives (also in contracts without any gold span of that category), unmatched gold
   spans are false negatives. Precision / recall / F1 per category, MICRO (pooled counts) and MACRO (mean over categories;
   primary metric = macro-F1 at IoU 0.5).

2. CUAD-style metrics, replicating TheAtticusProject/cuad evaluate.py: word-set Jaccard >= 0.5 on the span TEXTS (after
   removing . , ; : , replacing / by a space and lower-casing); every gold span with at least one matching prediction is a TP
   (not one-to-one), every prediction matching no gold span is an FP; counts pooled over all (contract, category) pairs; the
   confidence threshold is swept over 0.99, 0.98, ..., 0.01, 0.001, 0; precision is made monotone; AUPR = area under that
   curve; precision at 80 % / 90 % recall = processed precision at the first threshold reaching that recall.

Contract-level bootstrap: whole contracts are resampled; in each resample macro-F1 is averaged over the categories that have
at least one gold span in that resample (the same rule as Task 1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics as skm


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def char_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def match(gold: list, preds: list, mode: str = "iou", iou_thresh: float = 0.5, snap=None) -> tuple[int, int, int, list]:
    """gold: list of (start, end); preds: list of (start, end, confidence).
    mode "iou": IoU >= iou_thresh; mode "exact": snap(start, end) equal for both (snap maps a char range to token indices).
    Greedy one-to-one: predictions in descending confidence take the best still-unmatched gold span.
    Returns (tp, fp, fn, list of matched (pred_index, gold_index))."""
    used, pairs = set(), []
    order = sorted(range(len(preds)), key=lambda i: -preds[i][2])
    gsnap = [snap(*g) for g in gold] if mode == "exact" else None
    for i in order:
        p = preds[i]
        best, best_j = None, None
        for j, g in enumerate(gold):
            if j in used:
                continue
            if mode == "exact":
                ok = snap(p[0], p[1]) == gsnap[j]
                score = 1.0 if ok else 0.0
            else:
                score = char_iou((p[0], p[1]), g)
                ok = score >= iou_thresh
            if ok and (best is None or score > best):
                best, best_j = score, j
        if best_j is not None:
            used.add(best_j)
            pairs.append((i, best_j))
    tp = len(pairs)
    return tp, len(preds) - tp, len(gold) - tp, pairs


def count_table(gold_df: pd.DataFrame, pred_df: pd.DataFrame, contracts, categories, mode: str = "iou",
                iou_thresh: float = 0.5, snaps: dict | None = None) -> pd.DataFrame:
    """TP/FP/FN for EVERY (contract, category) pair of the split, including pairs without any gold span.
    gold_df: contract, category, start, end. pred_df: contract, category, start, end, confidence.
    snaps: {contract: function(start, end) -> token range} (needed for mode "exact")."""
    g = {k: list(zip(v.start, v.end)) for k, v in gold_df.groupby(["contract", "category"])}
    p = {k: list(zip(v.start, v.end, v.confidence)) for k, v in pred_df.groupby(["contract", "category"])}
    rows = []
    for c in contracts:
        for cat in categories:
            gl, pl = g.get((c, cat), []), p.get((c, cat), [])
            if not gl and not pl:
                rows.append((c, cat, 0, 0, 0))
                continue
            tp, fp, fn, _ = match(gl, pl, mode, iou_thresh, snaps[c] if snaps else None)
            rows.append((c, cat, tp, fp, fn))
    return pd.DataFrame(rows, columns=["contract", "category", "tp", "fp", "fn"])


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def summarise(counts: pd.DataFrame, categories) -> dict:
    """Per-category P/R/F1 (with gold and predicted span counts), micro and macro over the categories."""
    per = counts.groupby("category")[["tp", "fp", "fn"]].sum().reindex(categories, fill_value=0)
    rows = {}
    for cat, r in per.iterrows():
        p, rc, f = prf(r.tp, r.fp, r.fn)
        rows[cat] = {"precision": p, "recall": rc, "f1": f, "gold": int(r.tp + r.fn), "pred": int(r.tp + r.fp)}
    per_df = pd.DataFrame(rows).T
    mp, mr, mf = prf(per.tp.sum(), per.fp.sum(), per.fn.sum())
    has_gold = per_df.gold > 0
    return {"micro_precision": mp, "micro_recall": mr, "micro_f1": mf,
            "macro_precision": float(per_df.precision[has_gold].mean()), "macro_recall": float(per_df.recall[has_gold].mean()),
            "macro_f1": float(per_df.f1[has_gold].mean()), "n_categories_with_gold": int(has_gold.sum()),
            "per_category": per_df}


# ---------------------------------------------------------------------------
# Contract-level bootstrap on the count tables
# ---------------------------------------------------------------------------
def _macro_f1_from_arrays(tp, fp, fn):
    gold = tp + fn
    p = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    r = np.divide(tp, gold, out=np.zeros_like(tp, dtype=float), where=gold > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return f[gold > 0].mean()


def _resample_sums(counts: pd.DataFrame, categories, n_boot: int, seed: int):
    contracts = counts.contract.unique()
    cube = (counts.set_index(["contract", "category"])[["tp", "fp", "fn"]]
            .reindex(pd.MultiIndex.from_product([contracts, categories])).fillna(0)
            .to_numpy().reshape(len(contracts), len(categories), 3))
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        idx = rng.integers(0, len(contracts), len(contracts))
        yield cube[idx].sum(axis=0)


def bootstrap_macro_f1(counts: pd.DataFrame, categories, n_boot: int = 1000, seed: int = 42) -> dict:
    point = summarise(counts, categories)["macro_f1"]
    scores = np.array([_macro_f1_from_arrays(s[:, 0], s[:, 1], s[:, 2])
                       for s in _resample_sums(counts, categories, n_boot, seed)])
    return {"point": point, "ci_low": float(np.percentile(scores, 2.5)), "ci_high": float(np.percentile(scores, 97.5)),
            "boot_median": float(np.median(scores)), "n_boot": n_boot}


def paired_bootstrap_macro_f1(counts_a: pd.DataFrame, counts_b: pd.DataFrame, categories, n_boot: int = 1000,
                              seed: int = 42) -> dict:
    """macro-F1(A) - macro-F1(B) on the same resampled contracts (both tables must cover the same contracts)."""
    a = list(_resample_sums(counts_a, categories, n_boot, seed))
    b = list(_resample_sums(counts_b, categories, n_boot, seed))
    diffs = np.array([_macro_f1_from_arrays(x[:, 0], x[:, 1], x[:, 2]) - _macro_f1_from_arrays(y[:, 0], y[:, 1], y[:, 2])
                      for x, y in zip(a, b)])
    point = summarise(counts_a, categories)["macro_f1"] - summarise(counts_b, categories)["macro_f1"]
    return {"diff": float(point), "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5)),
            "p_a_not_better": float((diffs <= 0).mean()), "n_boot": n_boot}


# ---------------------------------------------------------------------------
# Token-level F1 on the INSIDE targets
# ---------------------------------------------------------------------------
def token_f1(pred_inside: list, gold_inside: list) -> dict:
    """Lists of (n_tokens, K) boolean arrays, one per contract. Micro over all tokens x categories; macro over categories
    with at least one gold token."""
    P = np.concatenate(pred_inside).astype(bool)
    G = np.concatenate(gold_inside).astype(bool)
    tp, fp, fn = (P & G).sum(0), (P & ~G).sum(0), (~P & G).sum(0)
    micro = prf(tp.sum(), fp.sum(), fn.sum())[2]
    macro = _macro_f1_from_arrays(tp.astype(float), fp.astype(float), fn.astype(float))
    return {"token_micro_f1": float(micro), "token_macro_f1": float(macro)}


# ---------------------------------------------------------------------------
# CUAD-style AUPR (replication of evaluate.py)
# ---------------------------------------------------------------------------
def cuad_words(s: str) -> set:
    for t in [".", ",", ";", ":"]:
        s = s.replace(t, "")
    return set(s.lower().replace("/", " ").split(" "))


def cuad_jaccard(a: str, b: str) -> float:
    A, B = cuad_words(a), cuad_words(b)
    return len(A & B) / len(A | B) if A | B else 0.0


def cuad_counts(gold_texts: dict, pred_texts: dict, keys) -> tuple[int, int, int]:
    """gold_texts / pred_texts: {(contract, category): [text, ...]} (predictions already filtered by confidence)."""
    tp = fp = fn = 0
    for k in keys:
        answers, preds = gold_texts.get(k, []), pred_texts.get(k, [])
        if not answers:
            fp += len(preds)
            continue
        for a in answers:
            if any(cuad_jaccard(a, p) >= 0.5 for p in preds):
                tp += 1
            else:
                fn += 1
        fp += sum(1 for p in preds if not any(cuad_jaccard(a, p) >= 0.5 for a in answers))
    return tp, fp, fn


CUAD_THRESHOLDS = list(np.arange(0.99, 0, -0.01)) + [0.001, 0]


def _process_precisions(precisions):
    best = precisions[::-1]
    for i in range(1, len(best)):
        best[i] = max(best[i - 1], best[i])
    return best[::-1]


def cuad_curve(gold_df: pd.DataFrame, pred_df: pd.DataFrame, text_of, keys) -> dict:
    """Precision-recall sweep exactly as evaluate.py. text_of(contract, start, end) -> span text.
    keys: all (contract, category) pairs that are evaluated."""
    keys = list(keys)
    gold_texts = {k: [text_of(k[0], s, e) for s, e in zip(v.start, v.end)]
                  for k, v in gold_df.groupby(["contract", "category"])}
    pred_texts_all = {k: list(zip([text_of(k[0], s, e) for s, e in zip(v.start, v.end)], v.confidence))
                      for k, v in pred_df.groupby(["contract", "category"])}
    precisions, recalls = [1.0], [0.0]
    for conf in CUAD_THRESHOLDS:
        pt = {k: [t for t, c in v if c > conf and t != ""] for k, v in pred_texts_all.items()}
        tp, fp, fn = cuad_counts(gold_texts, pt, keys)
        precisions.append(tp / (tp + fp) if tp + fp else np.nan)
        recalls.append(tp / (tp + fn) if tp + fn else np.nan)
    proc = _process_precisions(list(precisions))
    aupr = skm.auc(recalls, proc)
    aupr = 0.0 if np.isnan(aupr) else float(aupr)

    def p_at(r_thresh):
        for p, r in zip(proc[1:], recalls[1:]):
            if r >= r_thresh:
                return float(p)
        return 0.0
    return {"aupr": aupr, "prec_at_80_recall": p_at(0.8), "prec_at_90_recall": p_at(0.9),
            "precisions": [float(x) for x in proc], "recalls": [float(x) for x in recalls]}
