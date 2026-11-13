"""LEDGAR (LexGLUE) — the same-dataset published comparison for Task 1 (Chalkidis et al., 2022; numbers and set-up
from the paper). Results are analysed in the Task 1 notebook, section 6.

    python scripts/run_ledgar.py --data                 # load LEDGAR, save split sizes and token-length statistics
    python scripts/run_ledgar.py --benchmark            # 30 training steps -> projected time per epoch / per seed
    python scripts/run_ledgar.py --train 42 43 44       # Legal-BERT, one run per seed (resumable, early stopping)
    python scripts/run_ledgar.py --predict RUN_ID       # test-set predictions of the selected run (scored once in the notebook)
    python scripts/run_ledgar.py --tfidf                # TF-IDF + linear SVM grid on LEDGAR validation (CPU)

Recipe = the paper's set-up as far as it is stated (p. 4316): Adam(W) with initial lr 3e-5 (linear decay, no warm-up, no
weight decay = the Hugging Face Trainer defaults), batch size 8, early stopping on the development set (patience 3 on
micro-F1; neither is stated in the paper), max length 512, mixed precision (we use bf16; fp16 in the paper), [CLS] +
softmax head. CUT (decided and logged before training): at most 10 epochs instead of the paper's 20, so that 3 seeds fit the
3-hour budget in the worst case (benchmark: 5.4 min per epoch -> 3 x 10 epochs = 2.7 h; 3 x 20 epochs = 5.4 h).
"""
import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import common as C  # noqa: E402

CKPT = "nlpaueb/legal-bert-base-uncased"
OUT = ROOT / "models" / "ledgar"
RECIPE = dict(max_length=512, batch_size=8, lr=3e-5, max_epochs=10, patience=3, monitor="val_micro_f1",
              weight_decay=0.0, warmup_ratio=0.0, max_grad_norm=1.0, eval_batch_size=64)
LOG, SEARCH = C.RESULTS_DIR / "ledgar_log.csv", C.RESULTS_DIR / "ledgar_search.csv"


def load():
    from datasets import load_dataset
    d = load_dataset("coastalcph/lex_glue", "ledgar")
    names = d["train"].features["label"].names
    frames = {s: pd.DataFrame({"text": d[s]["text"], "label": d[s]["label"]}) for s in ["train", "validation", "test"]}
    return frames, names


def seed_everything(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def data_stats():
    from transformers import AutoTokenizer
    frames, names = load()
    tok = AutoTokenizer.from_pretrained(CKPT)
    n_tok = {s: np.array([len(x) for x in tok(list(f.text), add_special_tokens=True).input_ids]) for s, f in frames.items()}
    stats = {"source": "Hugging Face datasets coastalcph/lex_glue, config ledgar (loaded with datasets "
                       f"{__import__('datasets').__version__}, default revision)",
             "splits": {s: len(f) for s, f in frames.items()}, "n_labels": len(names), "labels": names,
             "tokens_legalbert": {s: {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": int(v.max()),
                                      "share_over_512": float((v > 512).mean())} for s, v in n_tok.items()},
             "train_label_counts_min_max": [int(frames["train"].label.value_counts().min()), int(frames["train"].label.value_counts().max())],
             "test_labels_present": int(frames["test"].label.nunique())}
    (C.RESULTS_DIR / "ledgar_data.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "labels"}, indent=2))


def encode(tok, texts, labels):
    enc = tok(list(texts), truncation=True, max_length=RECIPE["max_length"])
    return [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i], "labels": int(labels[i])}
            for i in range(len(texts))]


def loader(items, tok, bs, shuffle, seed=0):
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding
    g = torch.Generator().manual_seed(seed)
    return DataLoader(items, batch_size=bs, shuffle=shuffle, generator=g if shuffle else None, collate_fn=DataCollatorWithPadding(tok))


def predict(model, dl, device):
    import torch
    model.eval(); preds = []
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(device) for k, v in b.items()}; b.pop("labels")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds.append(model(**b).logits.float().argmax(-1).cpu())
    return torch.cat(preds).numpy()


def scores(y, p, n_labels):
    from sklearn.metrics import f1_score
    return {"micro_f1": float(f1_score(y, p, average="micro")),
            "macro_f1": float(f1_score(y, p, labels=list(range(n_labels)), average="macro", zero_division=0))}


def benchmark():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    frames, names = load()
    tok = AutoTokenizer.from_pretrained(CKPT)
    tr = frames["train"].sample(8 * 40, random_state=0)
    items = encode(tok, tr.text, tr.label.values)
    model = AutoModelForSequenceClassification.from_pretrained(CKPT, num_labels=len(names)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=RECIPE["lr"], weight_decay=0.0)
    dl = loader(items, tok, RECIPE["batch_size"], True)
    torch.cuda.reset_peak_memory_stats()
    for i, b in enumerate(dl):
        if i == 10:
            torch.cuda.synchronize(); t0 = time.perf_counter()
        b = {k: v.cuda() for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**b).loss
        loss.backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize(); ms = (time.perf_counter() - t0) / (len(dl) - 10) * 1000
    va = encode(tok, frames["validation"].text[:2000], frames["validation"].label.values[:2000])
    t0 = time.perf_counter(); predict(model, loader(va, tok, RECIPE["eval_batch_size"], False), "cuda"); torch.cuda.synchronize()
    eval_s = (time.perf_counter() - t0) / 2000 * len(frames["validation"])
    steps = -(-len(frames["train"]) // RECIPE["batch_size"])
    epoch_min = (ms * steps / 1000 + eval_s) / 60
    res = {"ms_per_step_batch8": ms, "steps_per_epoch": steps, "val_eval_seconds": eval_s, "projected_epoch_minutes": epoch_min,
           "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
           "projection_hours": {f"{e}_epochs_x_{s}_seeds": epoch_min * e * s / 60 for e in (6, 8, 10, 20) for s in (1, 3)}}
    (C.RESULTS_DIR / "ledgar_benchmark.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def run_id(seed, tag=""):
    return f"legalbert_ledgar{tag}_s{seed}"


def subset(frames, n=5000, seed=42):
    """Stratified n-provision subset of LEDGAR train (learning-curve check, validation only)."""
    from sklearn.model_selection import train_test_split
    tr = frames["train"]
    sub, _ = train_test_split(tr, train_size=n, stratify=tr.label, random_state=seed)
    return sub.reset_index(drop=True)


def train(seed, recipe=RECIPE, n_train=None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    rid = run_id(seed, f"_sub{n_train // 1000}k" if n_train else "")
    if SEARCH.exists() and rid in pd.read_csv(SEARCH).run_id.values:
        print(f"[{rid}] already finished - skipped"); return
    frames, names = load()
    if n_train:
        frames["train"] = subset(frames, n_train)
    tok = AutoTokenizer.from_pretrained(CKPT)
    tr = encode(tok, frames["train"].text, frames["train"].label.values)
    va = encode(tok, frames["validation"].text, frames["validation"].label.values)
    yv = frames["validation"].label.values
    seed_everything(seed)
    device = torch.device("cuda")
    model = AutoModelForSequenceClassification.from_pretrained(CKPT, num_labels=len(names)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"])
    steps = -(-len(tr) // recipe["batch_size"])
    sched = get_linear_schedule_with_warmup(opt, int(recipe["warmup_ratio"] * steps * recipe["max_epochs"]), steps * recipe["max_epochs"])
    out = OUT / rid; last = out / "last"
    start, best, bad, epoch_secs = 1, {"score": -1.0}, 0, []
    if (last / "state.pt").exists():
        st = torch.load(last / "state.pt", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        random.setstate(st["rng_py"]); np.random.set_state(st["rng_np"]); torch.set_rng_state(st["rng_t"]); torch.cuda.set_rng_state_all(st["rng_c"])
        start, best, bad, epoch_secs = st["epoch"] + 1, st["best"], st["bad"], st["epoch_secs"]
        print(f"[{rid}] resuming after epoch {st['epoch']}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t_run = time.perf_counter()
    vdl = loader(va, tok, recipe["eval_batch_size"], False)
    for epoch in range(start, recipe["max_epochs"] + 1):
        model.train(); rows = []; t0 = time.perf_counter()
        for step, b in enumerate(loader(tr, tok, recipe["batch_size"], True, seed=seed * 1000 + epoch)):
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["max_grad_norm"])
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if step % 200 == 0:
                rows.append({"run_id": rid, "kind": "step", "epoch": epoch, "step": (epoch - 1) * steps + step + 1, "train_loss": loss.item()})
        torch.cuda.synchronize(); tr_s = time.perf_counter() - t0; epoch_secs.append(tr_s)
        pv = predict(model, vdl, device); sc = scores(yv, pv, len(names))
        rows.append({"run_id": rid, "kind": "epoch", "epoch": epoch, "step": epoch * steps,
                     "train_loss": float(np.mean([r["train_loss"] for r in rows])), "val_micro_f1": sc["micro_f1"],
                     "val_macro_f1": sc["macro_f1"], "epoch_train_seconds": tr_s})
        pd.DataFrame(rows).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
        score = sc[recipe["monitor"].replace("val_", "")]
        print(f"[{rid}] epoch {epoch}: loss {rows[-1]['train_loss']:.4f} | val micro-F1 {sc['micro_f1']:.4f} macro-F1 {sc['macro_f1']:.4f} | {tr_s:.0f} s", flush=True)
        if score > best["score"]:
            best = {"score": score, "epoch": epoch, **sc}; bad = 0
            model.save_pretrained(out / "best"); tok.save_pretrained(out / "best"); np.save(out / "val_preds.npy", pv)
        else:
            bad += 1
        last.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch, "best": best,
                    "bad": bad, "epoch_secs": epoch_secs, "rng_py": random.getstate(), "rng_np": np.random.get_state(),
                    "rng_t": torch.get_rng_state(), "rng_c": torch.cuda.get_rng_state_all()}, last / "state.pt")
        if bad >= recipe["patience"]:
            print(f"[{rid}] early stop: no improvement of {recipe['monitor']} for {recipe['patience']} epochs", flush=True)
            break
    summary = {"run_id": rid, "seed": seed, "n_train": len(tr), "best_epoch": best["epoch"], "epochs_run": epoch, "val_micro_f1": best["micro_f1"],
               "val_macro_f1": best["macro_f1"], "mean_epoch_train_seconds": float(np.mean(epoch_secs)),
               "run_seconds": time.perf_counter() - t_run, "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
               "recipe": json.dumps(recipe)}
    pd.DataFrame([summary]).to_csv(SEARCH, mode="a", header=not SEARCH.exists(), index=False)
    shutil.rmtree(last, ignore_errors=True)


def predict_test(rid):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    frames, names = load()
    tok = AutoTokenizer.from_pretrained(OUT / rid / "best")
    model = AutoModelForSequenceClassification.from_pretrained(OUT / rid / "best").cuda()
    te = encode(tok, frames["test"].text, frames["test"].label.values)
    t0 = time.perf_counter(); p = predict(model, loader(te, tok, RECIPE["eval_batch_size"], False), "cuda"); torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    np.save(OUT / rid / "test_preds.npy", p)
    (OUT / rid / "test_predict_seconds.json").write_text(json.dumps({"seconds": secs, "n": len(p)}))
    print(f"test predictions written ({len(p)} provisions, {secs:.1f} s)")


def tfidf():
    """Grid as described in the paper (number of features, C, loss), tuned on LEDGAR validation micro-F1 (the paper's
    first metric); word uni+bigrams, sublinear tf. Every configuration is logged."""
    from joblib import Parallel, delayed
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    frames, names = load()
    tr, va = frames["train"], frames["validation"]
    grid_f = [10_000, 50_000, 200_000, None]
    grid_c = [0.1, 0.3, 1.0, 3.0]
    grid_loss = ["hinge", "squared_hinge"]
    def one(mf):
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=mf)
        Xtr = vec.fit_transform(tr.text); Xva = vec.transform(va.text)
        rows = []
        for c in grid_c:
            for loss in grid_loss:
                t0 = time.perf_counter()
                clf = LinearSVC(C=c, loss=loss, max_iter=5000, random_state=42).fit(Xtr, tr.label)
                sc = scores(va.label.values, clf.predict(Xva), len(names))
                rows.append({"max_features": mf or "all", "n_features": Xtr.shape[1], "C": c, "loss": loss,
                             "val_micro_f1": sc["micro_f1"], "val_macro_f1": sc["macro_f1"], "fit_seconds": time.perf_counter() - t0})
        return rows
    with C.timer("tfidf_ledgar", "grid_search", notes=f"{len(grid_f) * len(grid_c) * len(grid_loss)} configs; validation only"):
        res = Parallel(n_jobs=4)(delayed(one)(mf) for mf in grid_f)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(C.RESULTS_DIR / "ledgar_tfidf_search.csv", index=False)
    print(df.sort_values("val_micro_f1", ascending=False).head(8).to_string())


def learning_curve(n=5000):
    """Extra analysis (validation only): Legal-BERT (seed 42, same recipe) and TF-IDF + LinearSVC trained on a
    stratified 5k subset of LEDGAR train. Logged separately from the main runs."""
    global LOG, SEARCH
    LOG, SEARCH = C.RESULTS_DIR / "ledgar_subset_log.csv", C.RESULTS_DIR / "ledgar_subset_search.csv"
    with C.timer(f"legalbert_ledgar_sub{n // 1000}k_s42", "train", notes=f"stratified {n}-provision subset of LEDGAR train; validation only", device="GPU"):
        train(42, n_train=n)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    frames, names = load()
    sub = subset(frames, n)
    rows = []
    with C.timer(f"tfidf_ledgar_sub{n // 1000}k", "grid_search", notes=f"stratified {n}-provision subset; C x loss grid, all features; validation only"):
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
        Xtr = vec.fit_transform(sub.text); Xva = vec.transform(frames["validation"].text)
        for c in [0.3, 1.0, 3.0, 10.0]:
            for loss in ["hinge", "squared_hinge"]:
                clf = LinearSVC(C=c, loss=loss, max_iter=5000, random_state=42).fit(Xtr, sub.label)
                rows.append({"n_train": n, "n_features": Xtr.shape[1], "C": c, "loss": loss,
                             **{f"val_{k}": v for k, v in scores(frames["validation"].label.values, clf.predict(Xva), len(names)).items()}})
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "ledgar_subset_tfidf.csv", index=False)
    print(pd.DataFrame(rows).sort_values("val_micro_f1", ascending=False).head(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="store_true"); ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--train", nargs="+", type=int); ap.add_argument("--predict"); ap.add_argument("--tfidf", action="store_true")
    ap.add_argument("--subset", type=int)
    a = ap.parse_args()
    if a.data:
        data_stats()
    if a.benchmark:
        benchmark()
    if a.train:
        for s in a.train:
            with C.timer(run_id(s), "train", notes=f"LEDGAR, lr 3e-5, batch 8, max_len 512, early stopping (patience {RECIPE['patience']})", device="GPU"):
                train(s)
    if a.predict:
        predict_test(a.predict)
    if a.tfidf:
        tfidf()
    if a.subset:
        learning_curve(a.subset)
