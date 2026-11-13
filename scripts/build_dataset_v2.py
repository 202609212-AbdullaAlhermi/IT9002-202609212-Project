"""
Build dataset v2 (tutor feedback of 19 Oct 2026): a BROADER label set for Task 1 and the
span annotations for the new Task 2 (clause extraction).

Source: CUAD v1 (Hendrycks, Burns, Chen & Ball, 2021; NeurIPS Datasets & Benchmarks; CC BY 4.0).

What changed relative to v1 (the first-version builder, not included):
  * MIN_FINAL 150 -> 60 unambiguous clauses per category  => 27 classes instead of 20.
    The 7 added categories sit right next to existing ones, so they add hard distinctions:
    Non-Transferable License (vs License Grant / Anti-Assignment), No-Solicit Of Employees and
    Competitive Restriction Exception (vs Non-Compete / Exclusivity), Liquidated Damages (vs Cap On
    Liability), Joint Ip Ownership (vs Ip Ownership Assignment), Non-Disparagement, Source Code Escrow.
  * Validation hold-out uses 6 folds instead of 8 so rare classes keep >= 7 validation clauses.
  * Task 2 files: every span of every kept category (INCLUDING the multi-category spans that Task 1
    holds out as ambiguous) with character offsets into the original contract text, plus a
    contract -> split map for all 510 contracts (same split for Task 1 and Task 2).

Rules that are unchanged: unit = one expert-highlighted clause span; 4 entity/date categories dropped;
split BY CONTRACT; official CUAD test contracts = test; StratifiedGroupKFold(seed 42) for validation;
exact duplicates of train text removed from validation/test (Task 1 only).

Usage (from the project root):   python scripts/build_dataset_v2.py            -> writes data/processed/_rebuild_v2/
                                 python scripts/build_dataset_v2.py --overwrite -> writes data/processed/ (do this once, at hand-off)
The shipped CSVs in data/processed/ are the source of truth; never overwrite them after results exist.
"""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "CUADv1.json"
OFFICIAL_TEST = ROOT / "data" / "raw" / "cuad_official_test.json"
OUT_DIR = ROOT / "data" / "processed"
if "--overwrite" not in sys.argv:
    OUT_DIR = OUT_DIR / "_rebuild_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NON_CLAUSE = {"Parties", "Document Name", "Agreement Date", "Effective Date"}
MIN_FINAL = 60
N_FOLDS = 6
SEED = 42


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    contracts = json.load(open(RAW, encoding="utf-8"))["data"]
    test_titles = {c["title"] for c in json.load(open(OFFICIAL_TEST, encoding="utf-8"))["data"]}
    stats = {"n_contracts": len(contracts), "n_official_test_contracts": len(test_titles)}

    rows, raw_spans = [], []
    misaligned = 0
    for c in contracts:
        ctx = c["paragraphs"][0]["context"]
        for qa in c["paragraphs"][0]["qas"]:
            cat = re.search(r'"([^"]+)"', qa["question"]).group(1)
            for a in qa.get("answers", []):
                start, text = a["answer_start"], a["text"]
                end = start + len(text)
                if ctx[start:end] != text:
                    misaligned += 1
                rows.append({"contract": c["title"], "category": cat, "start": start, "text": norm(text)})
                raw_spans.append({"contract": c["title"], "category": cat, "start": start, "end": end, "text": text,
                                  "aligned": ctx[start:end] == text})
    stats["spans_total"] = len(rows)
    stats["spans_misaligned_with_context"] = misaligned
    df = pd.DataFrame(rows)
    df = df[~df.category.isin(NON_CLAUSE)]

    # ---- iterative category pruning (same algorithm as v1, new threshold) ----
    kept = set(df.category.unique())
    while True:
        sub = df[df.category.isin(kept)]
        n_cats = sub.groupby(["contract", "start", "text"]).category.transform("nunique")
        usable = sub[n_cats == 1].drop_duplicates(subset=["contract", "start", "text", "category"])
        counts = usable.category.value_counts().reindex(sorted(kept), fill_value=0)
        if not (counts < MIN_FINAL).any():
            break
        kept.discard(counts.idxmin())
    stats["dropped_categories"] = sorted(set(df.category.unique()) - kept)
    stats["kept_categories"] = sorted(kept)
    sub = df[df.category.isin(kept)].copy()
    sub["n_cats"] = sub.groupby(["contract", "start", "text"]).category.transform("nunique")
    ambiguous = sub[sub.n_cats > 1].copy()
    ambiguous["all_labels"] = ambiguous.groupby(["contract", "start", "text"]).category.transform(
        lambda x: "|".join(sorted(set(x))))
    ambiguous = ambiguous.drop_duplicates(subset=["contract", "start", "text"]).drop(columns=["category", "n_cats"])
    d = sub[sub.n_cats == 1].drop(columns="n_cats").drop_duplicates(
        subset=["contract", "start", "text", "category"]).reset_index(drop=True)
    stats["single_label_before_dedup"] = len(d)
    stats["ambiguous_clauses"] = len(ambiguous)

    # ---- contract-level split ----
    d["split"] = "train"
    is_test = d.contract.isin(test_titles)
    d.loc[is_test, "split"] = "test"
    pool = d[~is_test]
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    _, val_pos = next(sgkf.split(pool, pool.category, groups=pool.contract))
    d.loc[pool.index[val_pos], "split"] = "validation"
    contract_split = d.drop_duplicates("contract").set_index("contract").split.to_dict()  # before dedup, so no contract is lost

    # ---- leakage control (Task 1 only) ----
    train_texts = set(d.loc[d.split == "train", "text"].str.lower())
    dup = (d.split != "train") & d.text.str.lower().isin(train_texts)
    stats["duplicates_removed_from_val_test"] = int(dup.sum())
    d = d[~dup].reset_index(drop=True)
    d["n_words"] = d.text.str.split().str.len()

    # ---- full contract -> split map (all 510 contracts; shared by Task 1 and Task 2) ----
    cs_rows = []
    for c in contracts:
        t = c["title"]
        sp = "test" if t in test_titles else contract_split.get(t, "train")
        cs_rows.append({"contract": t, "split": sp, "n_chars": len(c["paragraphs"][0]["context"])})
    cs = pd.DataFrame(cs_rows)

    # ---- Task 2: all spans of kept categories, with character offsets ----
    sp = pd.DataFrame(raw_spans)
    sp = sp[sp.category.isin(kept)].drop_duplicates(subset=["contract", "category", "start", "end"]).copy()
    sp = sp[sp.aligned].drop(columns="aligned")
    sp["n_categories_on_span"] = sp.groupby(["contract", "start", "end"]).category.transform("nunique")
    sp["split"] = sp.contract.map(dict(zip(cs.contract, cs.split)))
    sp = sp.sort_values(["contract", "start", "end", "category"]).reset_index(drop=True)
    cs["n_task2_spans"] = cs.contract.map(sp.contract.value_counts()).fillna(0).astype(int)

    # ---- save ----
    p1 = OUT_DIR / "task1_clauses_v2.csv"
    p1a = OUT_DIR / "task1_ambiguous_v2.csv"
    p2 = OUT_DIR / "task2_spans.csv"
    p3 = OUT_DIR / "contract_split.csv"
    d.to_csv(p1, index=False, quoting=csv.QUOTE_ALL, lineterminator="\n")
    ambiguous["split"] = ambiguous.contract.map(dict(zip(cs.contract, cs.split)))
    ambiguous["n_words"] = ambiguous.text.str.split().str.len()
    ambiguous.to_csv(p1a, index=False, quoting=csv.QUOTE_ALL, lineterminator="\n")
    sp.to_csv(p2, index=False, quoting=csv.QUOTE_ALL, lineterminator="\n")
    cs.to_csv(p3, index=False, quoting=csv.QUOTE_ALL, lineterminator="\n")

    tab = d.pivot_table(index="category", columns="split", values="text", aggfunc="count", fill_value=0)
    tab["total"] = tab.sum(axis=1)
    stats.update({
        "task1_clauses": len(d),
        "task1_split_clauses": d.split.value_counts().to_dict(),
        "task1_split_contracts": d.groupby("split").contract.nunique().to_dict(),
        "all_contracts_per_split": cs.split.value_counts().to_dict(),
        "task2_spans": len(sp),
        "task2_spans_per_split": sp.split.value_counts().to_dict(),
        "task2_multi_category_spans": int((sp.n_categories_on_span > 1).sum()),
        "task2_contracts_with_spans_per_split": sp.groupby("split").contract.nunique().to_dict(),
        "clause_words_median": float(d.n_words.median()),
        "clause_words_p95": float(d.n_words.quantile(.95)),
        "per_class": tab.sort_values("total", ascending=False).to_dict(orient="index"),
        "smallest_test_class": [tab.test.idxmin(), int(tab.test.min())],
        "smallest_validation_class": [tab.validation.idxmin(), int(tab.validation.min())],
        "raw_sha256": {"CUADv1.json": sha256(RAW), "cuad_official_test.json": sha256(OFFICIAL_TEST)},
    })
    (OUT_DIR / "build_stats_v2.json").write_text(json.dumps(stats, indent=1))
    with open(OUT_DIR / "CHECKSUMS_v2.sha256", "w", newline="\n") as f:
        for p in (p1, p1a, p2, p3):
            f.write(f"{sha256(p)}  {p.name}\n")
    print(json.dumps({k: v for k, v in stats.items() if k not in ("per_class", "kept_categories")}, indent=1))
    print("\nPer-class counts:\n", tab.sort_values("total", ascending=False).to_string())
    print("\nSaved to", OUT_DIR)


if __name__ == "__main__":
    main()
