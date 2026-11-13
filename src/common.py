"""Small shared helpers for the IT9002 CUAD project (design v2: Task 1 = 27-class clause classification,
Task 2 = clause extraction).

Everything the notebooks need more than once lives here, kept short on purpose:

    load_project_info()     -> dict from project_info.json (name, ID, URLs)
    load_clauses()          -> the frozen Task 1 clause dataset, task1_clauses_v2.csv (with its `split` column)
    load_ambiguous()        -> the multi-label clauses (error-analysis / anomaly slice only)
    load_task2_spans()      -> Task 2 (span, category) annotations with character offsets
    load_contract_split()   -> all 510 contracts -> train / validation / test (shared by both tasks)
    load_contract_text()    -> the full text of one contract (from data/raw/CUADv1.json, cached)
    load_build_stats()      -> dataset numbers from build_stats_v2.json (so notebooks never retype them)
    verify_checksums()      -> checks data/processed against CHECKSUMS_v2.sha256
    get_labels()/encode()   -> fixed, alphabetical label order and label <-> id mapping
    set_seed()              -> makes Python / NumPy / PyTorch runs repeatable
    timer()                 -> times a block of code and appends it to results/compute_budget.csv
    compute_metrics()       -> macro-F1, accuracy, weighted-F1, per-class P/R/F1, confusion matrix
    save_predictions()      -> results/predictions/<model>.csv  (row_id, contract, true, pred)
    evaluate_on_test()      -> the ONLY way to score a model on the test set (test-once rule, no override)
    force_reevaluate_on_test() -> separate, logged emergency exit (forced=True + reason); used by no notebook
    robust_labels()         -> classes with >= 15 test clauses (robustness macro-F1)
    macro_f1_over()         -> macro-F1 over a fixed label list (for the bootstrap helpers)
    per_class_bootstrap_ci()-> per-class F1 with contract-level 95 % CIs
    test_once()             -> notebook wrapper: evaluates once; on re-runs checks predictions are identical
    bootstrap_ci()          -> 95 % confidence interval by resampling *contracts* (per resample: macro over present classes)
    paired_bootstrap()      -> 95 % CI for the difference between two models (same resamples)
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

# ---------------------------------------------------------------------------
# Paths. ROOT is the project folder (the parent of src/).
# RESULTS_DIR is a module variable so a test can point it somewhere harmless.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
SEED = 42


def load_project_info() -> dict:
    """Student name/ID/date/URLs. Placeholders until the student fills them in."""
    return json.loads((ROOT / "project_info.json").read_text(encoding="utf-8"))


# How to regenerate each model folder, with the time it took on our RTX 5080. The Colab T4 times are ESTIMATES (our time
# x 6-8; a T4 is much slower and has no native bfloat16, so mixed precision is emulated) - they were not measured.
RETRAIN_HINTS = {
    "models/legalbert_lr5e-05_s42": "Task 1 section 5: set RETRAIN = True (or run `python scripts/train_transformer.py`); "
                                    "about 21 min on our RTX 5080, roughly 2-3 h on a Colab T4 (estimate)",
    "models/anomaly_cache": "after the Task 1 Legal-BERT exists: `python scripts/anomaly_features.py`; 18 s on our RTX 5080, "
                            "roughly 2-3 min on a Colab T4 (estimate)",
    "models/task2": "Task 2 section 5: set RETRAIN = True (`python scripts/train_task2.py --plan`, then `--predict RUN_ID "
                    "--split validation` and `--split test`); about 40 min on our RTX 5080, roughly 4-5 h on a Colab T4 "
                    "(estimate; longer than a free Colab session may allow)",
}


def register_needs_models() -> None:
    """Notebook helper: defines the cell magic  %%needs_models <path> [<path> ...] [--saved=results/file.csv ...].

    Model checkpoints and caches (models/) are too large for the submission ZIP and the GitHub repository (and are absent
    on Google Colab). A cell that starts with  %%needs_models models/xyz  runs normally when the paths exist (errors are
    raised as usual). When they are missing it prints SKIPPED, says how to retrain (RETRAIN_HINTS) and shows the outputs this
    cell produced in the real run (tables, figures, text), stored in results/saved_cell_outputs.json by
    scripts/save_cell_outputs.py and found by the cell's code; if a cell has none stored there, the --saved files are shown.
    The outputs of the real run also stay in the submitted notebook.
    """
    from IPython import get_ipython
    from IPython.core.magic import register_cell_magic
    from IPython.display import display

    @register_cell_magic
    def needs_models(line, cell):
        parts = line.split()
        saved = [p.split("=", 1)[1] for p in parts if p.startswith("--saved=")]
        missing = [p for p in parts if not p.startswith("--") and not (ROOT / p).exists()]
        if missing:
            print(f"SKIPPED - this cell needs {', '.join(missing)}, which is not in the submission ZIP or the repository "
                  "(model checkpoints and caches are too large, and Colab starts without them).")
            for p in missing:
                hint = next((h for k, h in RETRAIN_HINTS.items() if p.startswith(k)), None)
                if hint:
                    print(f"  To regenerate {p}: {hint}.")
            outputs = load_saved_cell_outputs().get(cell_key(cell))
            if outputs is not None:
                print("Showing the SAVED output of this cell from the real run (results/saved_cell_outputs.json):\n")
                replay_outputs(outputs)
                return
            for f in saved:
                print(f"Showing the SAVED output of the real run instead ({f}):")
                display(pd.read_csv(ROOT / f))
            return
        res = get_ipython().run_cell(cell, store_history=False)
        if not res.success:
            raise RuntimeError("the cell above raised an error (see the traceback)")


# Saved outputs of the %%needs_models cells (written by scripts/save_cell_outputs.py from the executed notebooks). Key = hash
# of the cell's code (without the %%needs_models line), so an edited cell no longer matches and is simply skipped.
SAVED_CELL_OUTPUTS = RESULTS_DIR / "saved_cell_outputs.json"
_REPLAY_MIME = ("text/html", "text/markdown", "image/png", "image/jpeg", "image/svg+xml", "text/plain")


def cell_key(code: str) -> str:
    return hashlib.sha256(code.strip().replace("\r\n", "\n").encode("utf-8")).hexdigest()[:16]


def load_saved_cell_outputs() -> dict:
    return json.loads(SAVED_CELL_OUTPUTS.read_text(encoding="utf-8")) if SAVED_CELL_OUTPUTS.exists() else {}


def replay_outputs(outputs: list) -> None:
    """Show stored notebook outputs again: printed text, tables (HTML) and figures (PNG). stderr (warnings, progress bars)
    is left out."""
    import sys
    from IPython.display import publish_display_data
    for o in outputs:
        if o["output_type"] == "stream" and o.get("name") == "stdout":
            sys.stdout.write(o["text"] if isinstance(o["text"], str) else "".join(o["text"]))
        elif o["output_type"] in ("display_data", "execute_result"):
            data = {k: (v if isinstance(v, str) else "".join(v)) for k, v in o.get("data", {}).items() if k in _REPLAY_MIME}
            if data:
                publish_display_data(data, o.get("metadata", {}))


# Saved demo outputs: when a DEMO runs with the real models, its result for each input text is stored here, so the same
# DEMO can still show (and label as saved) the real output where the models are absent, e.g. on Colab.
DEMO_CACHE = RESULTS_DIR / "demo_cache.json"


def _demo_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def demo_cache_put(kind: str, text: str, value) -> None:
    d = json.loads(DEMO_CACHE.read_text(encoding="utf-8")) if DEMO_CACHE.exists() else {}
    d.setdefault(kind, {})[_demo_key(text)] = value
    DEMO_CACHE.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")


def demo_cache_get(kind: str, text: str):
    d = json.loads(DEMO_CACHE.read_text(encoding="utf-8")) if DEMO_CACHE.exists() else {}
    return d.get(kind, {}).get(_demo_key(text))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
TASK1_FILE = "task1_clauses_v2.csv"
AMBIGUOUS_FILE = "task1_ambiguous_v2.csv"
TASK2_FILE = "task2_spans.csv"
SPLIT_FILE = "contract_split.csv"
CHECKSUM_FILE = "CHECKSUMS_v2.sha256"
BUILD_STATS_FILE = "build_stats_v2.json"
RAW_CUAD = ROOT / "data" / "raw" / "CUADv1.json"


def load_clauses(path: Path | None = None) -> pd.DataFrame:
    """Load the frozen Task 1 clause dataset (design v2).

    Columns: contract, category, start, text, split ('train'/'validation'/'test'), n_words.
    We add `row_id` = the row's position in the CSV, so every prediction file can be
    traced back to exactly one clause.
    """
    df = pd.read_csv(path or DATA_DIR / TASK1_FILE)
    df.insert(0, "row_id", np.arange(len(df)))
    return df


def load_ambiguous(path: Path | None = None) -> pd.DataFrame:
    """Clauses with more than one kept label. Never used for training or the main score."""
    df = pd.read_csv(path or DATA_DIR / AMBIGUOUS_FILE)
    df["all_labels"] = df["all_labels"].str.split("|")
    return df


def load_task2_spans(path: Path | None = None) -> pd.DataFrame:
    """Task 2 annotations: contract, category, start, end (character offsets), text, n_categories_on_span, split."""
    return pd.read_csv(path or DATA_DIR / TASK2_FILE)


def load_contract_split(path: Path | None = None) -> pd.DataFrame:
    """All 510 contracts with their split (the same split is used by Task 1 and Task 2), n_chars, n_task2_spans."""
    return pd.read_csv(path or DATA_DIR / SPLIT_FILE)


_CONTRACT_TEXT: dict | None = None


def load_contract_text(title: str) -> str:
    """Full text of one contract. Reads data/raw/CUADv1.json once (lazily) and keeps it in memory."""
    global _CONTRACT_TEXT
    if _CONTRACT_TEXT is None:
        data = json.loads(RAW_CUAD.read_text(encoding="utf-8"))["data"]
        _CONTRACT_TEXT = {c["title"]: c["paragraphs"][0]["context"] for c in data}
    return _CONTRACT_TEXT[title]


def load_build_stats() -> dict:
    """Every dataset number (curation steps, counts) as computed by scripts/build_dataset_v2.py."""
    return json.loads((DATA_DIR / BUILD_STATS_FILE).read_text(encoding="utf-8"))


def verify_checksums(data_dir: Path = DATA_DIR) -> bool:
    """Compare each file's SHA-256 with CHECKSUMS_v2.sha256. Raises if any file differs."""
    for line in (data_dir / CHECKSUM_FILE).read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split()
        actual = hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {name}: expected {expected}, got {actual}")
        print(f"OK  {name}")
    return True


# SHA-256 of the four frozen CSVs (= data/processed/CHECKSUMS_v2.sha256), kept in the code as well so that data fetched
# from GitHub or rebuilt from the official CUAD release can be verified even when data/processed/ does not exist yet.
FROZEN_SHA256 = {
    "task1_clauses_v2.csv": "40a5d80a8dbf1e5de918b5068cba44d31570260c9c99c625fc5985f6079bbd64",
    "task1_ambiguous_v2.csv": "ca0eb1f7af61ed8771d1cf3c3f6e6c7f5e3f5e540b5376e609701c42fb720ae5",
    "task2_spans.csv": "9d1ced01675024a241fa7ac6db859a0b3555c2b6c6008c5637dcf53eb6fe6b7f",
    "contract_split.csv": "ba7c17ec83db6c8eec17c9f1c4f0ae417181a444964c5395266c165d21dd1036",
}


def ensure_data(github_url: str | None = None) -> None:
    """Make sure the frozen dataset and the raw CUAD file are present, then verify them.

    Order: (1) data/processed/ already there -> use it; (2) else, if the project's GitHub URL is known, download the frozen
    CSVs from the repository; (3) else rebuild them from the official CUAD release (scripts/download_cuad.py, then
    scripts/build_dataset_v2.py into _rebuild_v2/, copied over). In every case the four CSVs must match FROZEN_SHA256.
    """
    import shutil
    import subprocess
    import sys
    import urllib.request

    needed = list(FROZEN_SHA256) + [BUILD_STATS_FILE]
    if not RAW_CUAD.exists():                       # official release: needed for contract texts (Task 2) and the rebuild
        subprocess.run([sys.executable, str(ROOT / "scripts" / "download_cuad.py")], check=True, cwd=ROOT)
    if not all((DATA_DIR / n).exists() for n in needed):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        fetched = False
        if github_url and github_url.startswith("https://github.com/"):
            base = github_url.rstrip("/").removesuffix(".git").replace("https://github.com/", "https://raw.githubusercontent.com/")
            print("Downloading the frozen CSVs from", github_url)
            try:
                for n in needed:
                    urllib.request.urlretrieve(f"{base}/main/data/processed/{n}", DATA_DIR / n)
                fetched = True
            except OSError as e:                     # e.g. HTTP 404: the repository is private (raw files need a login)
                print(f"  could not download them ({e}); rebuilding instead")
        if not fetched:
            print("No data/processed/ (and no reachable repository): rebuilding the dataset from the official CUAD release "
                  "(scripts/build_dataset_v2.py, about a minute)")
            subprocess.run([sys.executable, str(ROOT / "scripts" / "build_dataset_v2.py")], check=True, cwd=ROOT,
                           capture_output=True)
            for n in needed:
                shutil.copy2(DATA_DIR / "_rebuild_v2" / n, DATA_DIR / n)
        (DATA_DIR / CHECKSUM_FILE).write_text("".join(f"{h}  {n}\n" for n, h in FROZEN_SHA256.items()))
    for n, h in FROZEN_SHA256.items():
        actual = hashlib.sha256((DATA_DIR / n).read_bytes()).hexdigest()
        if actual != h:
            raise ValueError(f"{n} differs from the frozen version (expected {h[:12]}..., got {actual[:12]}...). If it was "
                             "rebuilt, check that the pinned scikit-learn/pandas versions of requirements.txt are installed.")
        print(f"OK  {n}  (SHA-256 matches the frozen file)")


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def get_labels(df: pd.DataFrame) -> list[str]:
    """The class names found in the data (27 in design v2), in a fixed alphabetical order, used everywhere."""
    return sorted(df["category"].unique())


MIN_TEST_CLAUSES_ROBUST = 15


def robust_labels(df: pd.DataFrame, min_test: int = MIN_TEST_CLAUSES_ROBUST) -> list[str]:
    """Classes with at least `min_test` TEST clauses. Fixed from the class counts alone (never from results);
    used for the robustness macro-F1. The primary metric always uses all classes."""
    counts = df[df.split == "test"].category.value_counts()
    return sorted(counts[counts >= min_test].index)


def encode(labels: list[str]) -> tuple[dict, dict]:
    """Return (label -> id, id -> label) dictionaries for the fixed label order."""
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    """Fix every random number generator we use so results can be repeated."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _device_name(device: str) -> str:
    """'CPU' -> the CPU description; 'GPU' -> the GPU's name (e.g. 'NVIDIA GeForce RTX 5080')."""
    if device.upper() == "GPU":
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU requested but not available"
    import platform

    return f"CPU ({platform.processor() or platform.machine()}, {os.cpu_count()} threads)"


@contextmanager
def timer(model: str, stage: str, notes: str = "", device: str = "CPU"):
    """Time a block and append one row to results/compute_budget.csv.

    Usage:
        with timer("linear_svc", "grid_search", notes="72 configs"):            # CPU job
            ...
        with timer("legal_bert", "fine_tune", notes="seed 42", device="GPU"):   # GPU job
            ...
    """
    start = time.perf_counter()
    yield
    seconds = time.perf_counter() - start
    path = RESULTS_DIR / "compute_budget.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "model", "stage", "seconds", "device", "notes"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), model, stage,
                    round(seconds, 2), _device_name(device), notes])
    print(f"[timer] {model} / {stage}: {seconds:.1f} s")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, labels: list[str], model_name: str, split: str,
                    save: bool = True, robust_labels: list[str] | None = None) -> dict:
    """All Task 1 metrics for one model on one split.

    Macro-F1 is the primary metric: every clause type counts equally, so a model cannot
    score well by only getting the common types right. Predictions that are not one of
    the 20 labels (possible for the LLM) simply count as wrong for every class.

    If save=True the result is stored in results/metrics.json under "<model>__<split>".
    """
    y_true, y_pred = list(y_true), list(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    result = {
        "model": model_name,
        "split": split,
        "n": len(y_true),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        # robustness number: macro-F1 over the pre-declared classes with >= 15 test clauses (see robust_labels())
        "macro_f1_robust": (float(f1_score(y_true, y_pred, labels=robust_labels, average="macro", zero_division=0))
                            if robust_labels else None),
        "robust_labels": robust_labels,
        "per_class": {lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                            "support": int(s[i])} for i, lab in enumerate(labels)},
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if save:
        path = RESULTS_DIR / "metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        allm = json.loads(path.read_text()) if path.exists() else {}
        allm[f"{model_name}__{split}"] = result
        path.write_text(json.dumps(allm, indent=2))
    return result


def save_predictions(model_name: str, row_ids, contracts, y_true, y_pred) -> Path:
    """Write results/predictions/<model>.csv with columns row_id, contract, true, pred."""
    path = RESULTS_DIR / "predictions" / f"{model_name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": list(row_ids), "contract": list(contracts),
                  "true": list(y_true), "pred": list(y_pred)}).to_csv(path, index=False)
    return path


def _log_test_evaluation(model_name, config, result, forced: bool, reason: str) -> None:
    log = RESULTS_DIR / "test_eval_log.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log.exists()
    with log.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "model", "config", "macro_f1", "accuracy", "weighted_f1",
                        "forced", "reason"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), model_name,
                    json.dumps(config, sort_keys=True, default=str), round(result["macro_f1"], 4),
                    round(result["accuracy"], 4), round(result["weighted_f1"], 4), forced, reason])


def evaluate_on_test(model_name: str, config: dict, y_true, y_pred, labels: list[str],
                     row_ids=None, contracts=None, robust_labels: list[str] | None = None) -> dict:
    """Score a FINAL model on the test set - once.

    1. Refuses if `model_name` is already in results/test_eval_log.csv (test-once rule). There is no
       override here on purpose (force_reevaluate_on_test is the separate, logged emergency exit).
    2. Computes and saves the metrics (split="test") and, if row_ids/contracts are given,
       the prediction file.
    3. Appends a row to results/test_eval_log.csv: timestamp, model, config, scores, forced=False.
    """
    if already_evaluated(model_name):
        raise RuntimeError(f"'{model_name}' was already evaluated on the test set. The test-once rule "
                           "forbids a second evaluation.")
    result = compute_metrics(y_true, y_pred, labels, model_name, "test", robust_labels=robust_labels)
    if row_ids is not None and contracts is not None:
        save_predictions(model_name, row_ids, contracts, y_true, y_pred)
    _log_test_evaluation(model_name, config, result, forced=False, reason="")
    print(f"[test] {model_name}: macro-F1 = {result['macro_f1']:.4f}, accuracy = {result['accuracy']:.4f}")
    return result


def force_reevaluate_on_test(model_name: str, config: dict, y_true, y_pred, labels: list[str], reason: str,
                             row_ids=None, contracts=None, robust_labels: list[str] | None = None) -> dict:
    """EMERGENCY EXIT, not used by any notebook. Re-scores a model that is already in the test log, e.g. after
    a crash corrupted its saved files. The row is logged with forced=True and the written reason, and every
    stage review must report any forced row. Never use it to try a changed model: give that a new name."""
    if not reason.strip():
        raise ValueError("A forced re-evaluation needs a written `reason`.")
    result = compute_metrics(y_true, y_pred, labels, model_name, "test", robust_labels=robust_labels)
    if row_ids is not None and contracts is not None:
        save_predictions(model_name, row_ids, contracts, y_true, y_pred)
    _log_test_evaluation(model_name, config, result, forced=True, reason=reason)
    print(f"[test] FORCED re-evaluation of {model_name} logged (reason: {reason})")
    return result


def already_evaluated(model_name: str) -> bool:
    log = RESULTS_DIR / "test_eval_log.csv"
    return log.exists() and model_name in pd.read_csv(log)["model"].astype(str).tolist()


def test_once(model_name: str, config: dict, y_true, y_pred, labels: list[str], row_ids, contracts,
              robust_labels: list[str] | None = None) -> dict:
    """What the notebooks call. First time: the real test evaluation (evaluate_on_test).
    On a re-run of the notebook: NO new evaluation is logged; the saved predictions are loaded and the
    fresh predictions must be identical (proves the frozen model was re-run, not changed after testing)."""
    if not already_evaluated(model_name):
        return evaluate_on_test(model_name, config, y_true, y_pred, labels, row_ids, contracts, robust_labels)
    saved = pd.read_csv(RESULTS_DIR / "predictions" / f"{model_name}.csv")
    if list(saved["pred"].astype(str)) != [str(p) for p in y_pred]:
        raise RuntimeError(f"'{model_name}' predictions differ from the ones evaluated on test earlier. "
                           "The model must not be changed after its test evaluation.")
    print(f"[test] {model_name}: already evaluated once (see test_eval_log.csv); re-run reproduced the "
          "identical predictions, saved metrics re-used.")
    return json.loads((RESULTS_DIR / "metrics.json").read_text())[f"{model_name}__test"]


# ---------------------------------------------------------------------------
# Contract-level bootstrap
# Clauses from the same contract are not independent (same drafting style), so we
# resample whole contracts with replacement, not individual clauses.
# ---------------------------------------------------------------------------
def _macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def macro_f1_over(labels: list[str]):
    """A macro-F1 function over a FIXED label list, for the bootstrap helpers (so every resample averages over
    the same classes)."""
    def metric(y_true, y_pred):
        return f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return metric


def per_class_bootstrap_ci(y_true, y_pred, contracts, labels: list[str], n_boot: int = 1000,
                           seed: int = SEED) -> pd.DataFrame:
    """Per-class F1 with a 95 % contract-level bootstrap CI (classes with few test clauses get wide intervals).
    A resample in which the class does not occur in the true labels is skipped for that class. If all clauses of a
    class come from a single contract, the CI is not estimable and is returned as NaN."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    point = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    boots, present = [], []
    t, p, k = _label_codes(y_true, labels), _label_codes(y_pred, labels), len(labels)
    for i in _contract_resamples(contracts, n_boot, seed):   # fast exact path (same numbers as f1_score)
        f1, true_sum = _f1_per_class_codes(t[i], p[i], k)
        boots.append(f1)
        present.append(true_sum > 0)
    boots = np.where(np.array(present), np.array(boots), np.nan)   # a class absent from a resample is skipped there
    support = [int((y_true == lab).sum()) for lab in labels]
    contracts = np.asarray(contracts)
    n_contracts = [len(set(contracts[y_true == lab])) for lab in labels]
    out = pd.DataFrame({"f1": point, "ci_low": np.nanpercentile(boots, 2.5, axis=0),
                        "ci_high": np.nanpercentile(boots, 97.5, axis=0),
                        "resamples_used": np.sum(~np.isnan(boots), axis=0), "support": support,
                        "test_contracts": n_contracts}, index=labels)
    # with all clauses of a class in ONE contract, resampling contracts cannot show any uncertainty
    out.loc[out.test_contracts < 2, ["ci_low", "ci_high"]] = np.nan
    return out


def _contract_resamples(contracts, n_boot: int, seed: int):
    """Yield arrays of clause indices, one per bootstrap resample of contracts."""
    contracts = np.asarray(contracts)
    _, inverse, counts = np.unique(contracts, return_inverse=True, return_counts=True)
    groups = np.split(np.argsort(inverse, kind="stable"), np.cumsum(counts)[:-1])   # = flatnonzero(contracts == c), sorted c
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        picked = rng.integers(0, len(groups), size=len(groups))
        yield np.concatenate([groups[i] for i in picked])


def macro_f1_present(labels: list[str]):
    """Macro-F1 averaged over the classes of `labels` that occur in this sample's TRUE labels.

    Used inside bootstrap resamples: a rare class (e.g. 4 test clauses) is often missing from a resample of
    contracts; scoring it as F1 = 0 there would pull every resample down and bias the CI downwards."""
    def metric(y_true, y_pred):
        present = [lab for lab in labels if lab in set(y_true)]
        return f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
    return metric


def _label_codes(y, labels: list[str]) -> np.ndarray:
    """Integer code of each value's position in `labels` (len(labels) = not one of the labels)."""
    idx = {lab: i for i, lab in enumerate(labels)}
    return np.fromiter((idx.get(v, len(labels)) for v in np.asarray(y).tolist()), dtype=np.int64, count=len(y))


def _f1_per_class_codes(t: np.ndarray, p: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-class F1 (zero_division=0) from label codes, with the same arithmetic as sklearn's f1_score
    (F1 = 2 tp / (true + predicted)); used inside the bootstrap loops, where it is ~100x faster than calling
    f1_score 1,000 times. Returns (f1 per class, true count per class)."""
    tp = np.bincount(t[t == p], minlength=k + 1)[:k].astype(np.float64)
    true_sum = np.bincount(t, minlength=k + 1)[:k].astype(np.float64)
    pred_sum = np.bincount(p, minlength=k + 1)[:k].astype(np.float64)
    denom = 1 * true_sum + pred_sum
    f1 = np.zeros(k)
    nz = denom != 0
    f1[nz] = (2 * tp[nz]) / denom[nz]
    return f1, true_sum


def _macro_present_codes(t: np.ndarray, p: np.ndarray, k: int) -> float:
    """macro_f1_present() on label codes: macro-F1 over the classes present in this sample's true labels."""
    f1, true_sum = _f1_per_class_codes(t, p, k)
    return float(np.nanmean(f1[true_sum > 0]))


def bootstrap_ci(y_true, y_pred, contracts, labels: list[str] | None = None, metric=None, n_boot: int = 1000,
                 seed: int = SEED) -> dict:
    """Point estimate and 95 % percentile confidence interval, resampling CONTRACTS.

    With `labels` (the normal use): point estimate = macro-F1 over ALL `labels`; in each resample macro-F1 is
    averaged over the classes present in that resample's true labels (see macro_f1_present).
    Without `labels`: `metric` (default plain macro-F1) is used for both."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    point_m = macro_f1_over(labels) if labels is not None else (metric or _macro_f1)
    boot_m = macro_f1_present(labels) if labels is not None else (metric or _macro_f1)
    if labels is not None:   # fast exact path (same numbers as boot_m)
        t, p, k = _label_codes(y_true, labels), _label_codes(y_pred, labels), len(labels)
        scores = np.array([_macro_present_codes(t[idx], p[idx], k) for idx in _contract_resamples(contracts, n_boot, seed)])
    else:
        scores = np.array([boot_m(y_true[idx], y_pred[idx]) for idx in _contract_resamples(contracts, n_boot, seed)])
    return {"point": float(point_m(y_true, y_pred)), "ci_low": float(np.percentile(scores, 2.5)),
            "ci_high": float(np.percentile(scores, 97.5)), "boot_median": float(np.median(scores)), "n_boot": n_boot}


def paired_bootstrap(y_true, pred_a, pred_b, contracts, labels: list[str] | None = None, metric=None,
                     n_boot: int = 1000, seed: int = SEED) -> dict:
    """Difference metric(A) - metric(B) on the SAME resampled contracts (same class-presence rule as bootstrap_ci).

    `p_a_not_better` is the share of resamples where A did not beat B; a small value
    (e.g. < 0.05) means the advantage of A is unlikely to be luck of the test contracts.
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    point_m = macro_f1_over(labels) if labels is not None else (metric or _macro_f1)
    boot_m = macro_f1_present(labels) if labels is not None else (metric or _macro_f1)
    if labels is not None:   # fast exact path (same numbers as boot_m)
        t, a, b, k = _label_codes(y_true, labels), _label_codes(pred_a, labels), _label_codes(pred_b, labels), len(labels)
        diffs = np.array([_macro_present_codes(t[i], a[i], k) - _macro_present_codes(t[i], b[i], k)
                          for i in _contract_resamples(contracts, n_boot, seed)])
    else:
        diffs = np.array([boot_m(y_true[i], pred_a[i]) - boot_m(y_true[i], pred_b[i])
                          for i in _contract_resamples(contracts, n_boot, seed)])
    return {"diff": float(point_m(y_true, pred_a) - point_m(y_true, pred_b)),
            "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5)),
            "boot_median": float(np.median(diffs)), "p_a_not_better": float((diffs <= 0).mean()), "n_boot": n_boot}


def test_once_spans(model_name: str, config: dict, result: dict, pred_df: pd.DataFrame) -> dict:
    """Test-once logger for Task 2 (span extraction). `result` must contain macro_f1 (primary, entity-level @ IoU 0.5),
    micro_f1 and token_micro_f1; `pred_df` (contract, category, start, end, confidence) is saved to
    results/task2/<model>_test_spans.csv. First call: logs the evaluation (forced=False). Later calls (notebook re-runs):
    nothing is logged; the new predictions must equal the saved ones, otherwise an error is raised."""
    path = RESULTS_DIR / "task2" / f"{model_name}_test_spans.csv"
    metrics_path = RESULTS_DIR / "task2" / f"{model_name}_test_metrics.json"
    if already_evaluated(model_name):
        saved = pd.read_csv(path)
        cols = ["contract", "category", "start", "end"]
        same = (len(saved) == len(pred_df) and (saved[cols].reset_index(drop=True)
                                                 == pred_df[cols].reset_index(drop=True)).all().all())
        if not same:
            raise RuntimeError(f"'{model_name}' predictions differ from the ones evaluated on test earlier.")
        print(f"[test] {model_name}: already evaluated once (see test_eval_log.csv); re-run reproduced the identical "
              "spans, saved metrics re-used.")
        return json.loads(metrics_path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(path, index=False)
    metrics_path.write_text(json.dumps(result, indent=2, default=float))
    _log_test_evaluation(model_name, config, {"macro_f1": result["macro_f1"], "accuracy": result["micro_f1"],
                                              "weighted_f1": result["token_micro_f1"]}, forced=False,
                         reason="Task 2: accuracy column = entity micro-F1@IoU0.5, weighted_f1 column = token micro-F1")
    print(f"[test] {model_name}: macro-F1@IoU0.5 = {result['macro_f1']:.4f}, micro-F1 = {result['micro_f1']:.4f}")
    return result


def test_once_scores(model_name: str, config: dict, result: dict, scores_df: pd.DataFrame, log_values: dict, reason: str) -> dict:
    """Test-once logger for score-type outputs (Part F1 anomaly scorers). `scores_df` (one row per scored test item) is saved
    to results/partf/<model>_test_scores.csv and `result` to ..._test_metrics.json. `log_values` maps the three numeric
    columns of test_eval_log.csv (macro_f1, accuracy, weighted_f1) to the metrics stored there; `reason` says what they
    mean. Later calls (notebook re-runs) log nothing; the scores must equal the saved ones (to 1e-6)."""
    path = RESULTS_DIR / "partf" / f"{model_name}_test_scores.csv"
    metrics_path = RESULTS_DIR / "partf" / f"{model_name}_test_metrics.json"
    if already_evaluated(model_name):
        saved = pd.read_csv(path)
        same = len(saved) == len(scores_df) and np.allclose(saved["score"].to_numpy(), scores_df["score"].to_numpy(), atol=1e-6)
        if not same:
            raise RuntimeError(f"'{model_name}' scores differ from the ones evaluated on test earlier.")
        print(f"[test] {model_name}: already evaluated once (see test_eval_log.csv); re-run reproduced the identical scores.")
        return json.loads(metrics_path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(path, index=False)
    metrics_path.write_text(json.dumps(result, indent=2, default=float))
    _log_test_evaluation(model_name, config, log_values, forced=False, reason=reason)
    print(f"[test] {model_name}: " + ", ".join(f"{k} = {v:.4f}" for k, v in log_values.items()))
    return result
