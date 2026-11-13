"""PART F1 — anomaly flagging: scores that say how unusual a clause x is for a type c.

Every scorer is fit on TRAIN clauses only and scores (clause, type) pairs:
    KNNScorer      1 - mean cosine similarity to the k most similar TRAIN clauses of type c
                   (on TF-IDF vectors or on Legal-BERT embeddings)                         -> "unusual wording" family
    MahalanobisScorer  distance to the TRAIN clauses of type c in a PCA space, with a
                   shrinkage (Ledoit-Wolf) covariance per type                              -> "unusual wording" family
    doubt          1 - p(c | x) from the fine-tuned Task 1 classifier                        -> "doubt about the type" family
Higher score = more unusual. Thresholds are set per type on genuine VALIDATION clauses (flag rate 5 %).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def l2norm(X):
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


class KNNScorer:
    """score(x, c) = 1 - mean cosine similarity of x to its k nearest TRAIN clauses of type c."""

    def __init__(self, X_train, y_train, labels, k=5, sparse=False):
        self.k, self.labels, self.sparse = k, list(labels), sparse
        y = np.asarray(y_train)
        self.by_class = {c: (X_train[np.flatnonzero(y == c)] if sparse else l2norm(X_train[y == c])) for c in self.labels}

    def score(self, X, claimed):
        claimed = np.asarray(claimed)
        out = np.empty(len(claimed), dtype=np.float64)
        Xq = X if self.sparse else l2norm(X)
        for c in np.unique(claimed):
            idx = np.flatnonzero(claimed == c)
            sims = Xq[idx] @ self.by_class[c].T
            sims = sims.toarray() if hasattr(sims, "toarray") else np.asarray(sims)
            k = min(self.k, sims.shape[1])
            top = -np.sort(-sims, axis=1)[:, :k]
            out[idx] = 1.0 - top.mean(axis=1)
        return out


class MahalanobisScorer:
    """Class-conditional Mahalanobis distance in a PCA space fitted on TRAIN embeddings; Ledoit-Wolf shrinkage covariance
    per type (stable even for types with few training clauses)."""

    def __init__(self, X_train, y_train, labels, n_components=50, seed=42):
        from sklearn.covariance import LedoitWolf
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=n_components, random_state=seed).fit(X_train)
        Z = self.pca.transform(X_train)
        y = np.asarray(y_train)
        self.stats = {}
        for c in labels:
            lw = LedoitWolf().fit(Z[y == c])
            self.stats[c] = (lw.location_, lw.precision_)

    def score(self, X, claimed):
        Z = self.pca.transform(X)
        claimed = np.asarray(claimed)
        out = np.empty(len(claimed))
        for c in np.unique(claimed):
            idx = np.flatnonzero(claimed == c)
            mu, P = self.stats[c]
            d = Z[idx] - mu
            out[idx] = np.sqrt(np.einsum("ij,jk,ik->i", d, P, d))
        return out


def doubt_score(probs, claimed, labels):
    """1 - p(claimed type | x) from the classifier's softmax probabilities (rows aligned with the clauses)."""
    col = {c: i for i, c in enumerate(labels)}
    return 1.0 - probs[np.arange(len(claimed)), [col[c] for c in claimed]]


# ---------------------------------------------------------------------------
# Proxy task P1: genuine vs injected wrong-type pairs
# ---------------------------------------------------------------------------
def make_pairs(clauses: pd.DataFrame, labels, hard_map: dict | None = None, seed=42) -> pd.DataFrame:
    """Each clause contributes its genuine pair (claimed = true type, label 0) and one injected pair (label 1) whose
    claimed type is a random other type (hard_map None) or a random one of the types most confused with the true type."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in clauses.itertuples():
        rows.append({"pos": r.Index, "row_id": r.row_id, "contract": r.contract, "true": r.category, "claimed": r.category, "injected": 0})
        pool = hard_map[r.category] if hard_map else [c for c in labels if c != r.category]
        rows.append({"pos": r.Index, "row_id": r.row_id, "contract": r.contract, "true": r.category,
                     "claimed": pool[rng.integers(len(pool))], "injected": 1})
    return pd.DataFrame(rows)


def hard_types(val_true, val_pred, labels, centroid_sim: pd.DataFrame, n=3) -> dict:
    """For each type: the n types its validation clauses were most often (wrongly) predicted as; ties and missing slots are
    filled with the most similar types by TF-IDF centroid cosine (TRAIN), so every type has exactly n hard alternatives."""
    conf = pd.crosstab(pd.Series(val_true, name="t"), pd.Series(val_pred, name="p"))
    out = {}
    for c in labels:
        cand = []
        if c in conf.index:
            row = conf.loc[c].drop(c, errors="ignore")
            row = row[row > 0]
            sim = centroid_sim.loc[c, row.index]
            cand = list(pd.DataFrame({"n": row, "s": sim}).sort_values(["n", "s"], ascending=False).index)
        for d in centroid_sim.loc[c].drop(c).sort_values(ascending=False).index:
            if len(cand) >= n:
                break
            if d not in cand:
                cand.append(d)
        out[c] = cand[:n]
    return out


# ---------------------------------------------------------------------------
# Metrics with contract-level bootstrap
# ---------------------------------------------------------------------------
def auc_ci(y, s, groups, n_boot=1000, seed=42):
    y, s, groups = np.asarray(y), np.asarray(s), np.asarray(groups)
    point = {"auroc": float(roc_auc_score(y, s)), "aupr": float(average_precision_score(y, s))}
    ug = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in ug}
    rng = np.random.default_rng(seed)
    a, p = [], []
    for _ in range(n_boot):
        idx = np.concatenate([idx_of[g] for g in rng.choice(ug, len(ug), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        a.append(roc_auc_score(y[idx], s[idx])); p.append(average_precision_score(y[idx], s[idx]))
    return {**point, "auroc_ci": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
            "aupr_ci": [float(np.percentile(p, 2.5)), float(np.percentile(p, 97.5))], "positives": int(y.sum()), "n": int(len(y))}


def thresholds_at(scores, types, labels, rate=0.05):
    """Per-type threshold = the (1 - rate) quantile of the scores of genuine clauses of that type."""
    s, t = np.asarray(scores), np.asarray(types)
    return {c: float(np.quantile(s[t == c], 1 - rate)) for c in labels if (t == c).any()}


def flagged(scores, claimed, thr):
    return np.array([sc > thr[c] for sc, c in zip(scores, claimed)])
