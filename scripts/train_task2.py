"""Task 2 tagger: Legal-BERT (base checkpoint nlpaueb/legal-bert-base-uncased, NOT the Task 1 model) with a 54-output
token-classification head (INSIDE + BEGIN for each of the 27 categories), trained with multi-label BCE on sliding windows.

    python scripts/train_task2.py --benchmark                 20-step timing for several window sizes -> projection
    python scripts/train_task2.py --plan                      the whole pre-declared plan (variants, then seeds)
    python scripts/train_task2.py --predict RUN_ID --split test    per-token probabilities of a finished run

Plan (decision rules fixed here, before any run):
  1. three loss variants, seed 42, 4 epochs each, VALIDATION only:
       bce_plain - plain BCE;  bce_posw - BCE with pos_weight = clip(sqrt(#neg / #pos), 1, 30) per output;
       focal     - sigmoid focal loss (gamma = 2).
     All variants use negative-window sampling: every window containing a gold span + 20 % of the other windows,
     re-sampled every epoch. Epoch selection and variant selection: validation macro-F1 at IoU 0.5, decoded with the
     default thresholds (theta_inside = theta_begin = 0.5, min_tokens = 1).
  2. seeds 43 and 44 for the chosen variant (validation mean +- std). The test model is seed 42 of the chosen variant.
Safe to interrupt: each run resumes from its last finished epoch; finished runs are skipped.
"""
import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transformers import (AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup,  # noqa: E402
                          logging as hf_logging)

from src import common as C  # noqa: E402
from src import span_metrics as M  # noqa: E402
from src import task2_data as D  # noqa: E402

hf_logging.set_verbosity_error()
CKPT = "nlpaueb/legal-bert-base-uncased"
OUT = ROOT / "models" / "task2"
CACHE = ROOT / "models" / "task2_cache" / "corpus_legalbert.pkl"
RECIPE = dict(window=256, stride=128, batch_size=16, lr=5e-5, epochs=4, warmup_ratio=0.1, weight_decay=0.01,
              max_grad_norm=1.0, neg_fraction=0.2, eval_batch_size=64)
VARIANTS = ["bce_plain", "bce_posw", "focal"]
SEEDS = [42, 43, 44]
LOG, SEARCH = C.RESULTS_DIR / "task2_log.csv", C.RESULTS_DIR / "task2_search.csv"


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data(tok):
    cs, spans = C.load_contract_split(), C.load_task2_spans()
    cats = sorted(spans.category.unique())
    corpus = D.load_corpus(tok, CACHE, list(cs.contract), C.load_contract_text)
    cidx = {c: i for i, c in enumerate(cats)}
    labels = {}
    for c in cs.contract:
        g = spans[spans.contract == c]
        labels[c] = D.token_labels(corpus[c]["offsets"], [(cidx[r.category], r.start, r.end) for r in g.itertuples()],
                                   len(cats))
    return cs, spans, cats, corpus, labels


def windows_of(contracts, corpus, recipe):
    return [(c, s, e) for c in contracts for s, e in D.make_windows(len(corpus[c]["ids"]), recipe["window"], recipe["stride"])]


def collate(batch, corpus, labels, tok, n_out):
    L = max(e - s for _, s, e in batch) + 2
    ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    y = torch.zeros((len(batch), L, n_out), dtype=torch.float)
    mask = torch.zeros((len(batch), L), dtype=torch.bool)
    for i, (c, s, e) in enumerate(batch):
        seq = [tok.cls_token_id] + corpus[c]["ids"][s:e] + [tok.sep_token_id]
        ids[i, :len(seq)] = torch.tensor(seq); att[i, :len(seq)] = 1
        if labels is not None:
            y[i, 1:1 + e - s] = torch.from_numpy(labels[c][s:e].astype(np.float32))
        mask[i, 1:1 + e - s] = True          # loss only on content tokens ([CLS], [SEP], padding masked)
    return ids, att, y, mask


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def make_loss(variant, pos_weight):
    bce = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight if variant == "bce_posw" else None)

    def loss_fn(logits, y, mask):
        if variant == "focal":
            p = torch.sigmoid(logits)
            ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
            pt = p * y + (1 - p) * (1 - y)
            l = ce * (1 - pt) ** 2
        else:
            l = bce(logits, y)
        return l[mask].mean()
    return loss_fn


# ---------------------------------------------------------------------------
# Inference: per-token probabilities for whole contracts
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_contracts(model, contracts, corpus, tok, recipe, n_out, device):
    model.eval()
    out = {}
    for c in contracts:
        n = len(corpus[c]["ids"])
        wins = D.make_windows(n, recipe["window"], recipe["stride"])
        probs = []
        for i in range(0, len(wins), recipe["eval_batch_size"]):
            chunk = [(c, s, e) for s, e in wins[i:i + recipe["eval_batch_size"]]]
            ids, att, _, mask = collate(chunk, corpus, None, tok, n_out)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits.float()
            p = torch.sigmoid(logits).cpu().numpy()
            for j, (_, s, e) in enumerate(chunk):
                probs.append(p[j, 1:1 + e - s])
        out[c] = D.average_window_probs(n, wins, probs, n_out).astype(np.float16)
    return out


def evaluate_probs(probs, contracts, corpus, spans, cats, theta_i=0.5, theta_b=0.5, min_tokens=1):
    rows = []
    for c in contracts:
        for k, s, e, conf in D.decode(probs[c].astype(np.float32), corpus[c]["offsets"], len(cats), theta_i, theta_b,
                                      min_tokens):
            rows.append((c, cats[k], s, e, conf))
    pred = pd.DataFrame(rows, columns=["contract", "category", "start", "end", "confidence"])
    gold = spans[spans.contract.isin(contracts)]
    return M.summarise(M.count_table(gold, pred, contracts, cats), cats), pred


# ---------------------------------------------------------------------------
# One training run
# ---------------------------------------------------------------------------
def run_id(variant, seed, epochs=RECIPE["epochs"], window=RECIPE["window"]):
    return (f"t2_{variant}_s{seed}" + ("" if epochs == RECIPE["epochs"] else f"_e{epochs}")
            + ("" if window == RECIPE["window"] else f"_w{window}"))


def train_run(variant, seed, data, tok, recipe=RECIPE, keep_weights=True):
    cs, spans, cats, corpus, labels = data
    rid = run_id(variant, seed, recipe["epochs"], recipe["window"])
    if SEARCH.exists() and rid in pd.read_csv(SEARCH).run_id.values:
        print(f"[{rid}] already finished - skipped"); return
    device = torch.device("cuda")
    seed_everything(seed)
    n_out = 2 * len(cats)
    train_c = list(cs[cs.split == "train"].contract); val_c = list(cs[cs.split == "validation"].contract)
    all_win = windows_of(train_c, corpus, recipe)
    pos_win = [w for w in all_win if labels[w[0]][w[1]:w[2]].any()]
    neg_win = [w for w in all_win if not labels[w[0]][w[1]:w[2]].any()]
    Y = np.concatenate([labels[c] for c in train_c])
    pos = Y.sum(0).astype(np.float64); neg = len(Y) - pos
    pos_weight = torch.tensor(np.clip(np.sqrt(neg / np.maximum(pos, 1)), 1, 30), dtype=torch.float, device=device)
    n_neg = int(recipe["neg_fraction"] * len(neg_win))
    steps_per_epoch = -(-(len(pos_win) + n_neg) // recipe["batch_size"])

    model = AutoModelForTokenClassification.from_pretrained(CKPT, num_labels=n_out).to(device)
    no_decay = ("bias", "LayerNorm.weight")
    groups = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
               "weight_decay": recipe["weight_decay"]},
              {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=recipe["lr"])
    total = steps_per_epoch * recipe["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(recipe["warmup_ratio"] * total), total)
    loss_fn = make_loss(variant, pos_weight)

    out_dir = OUT / rid; last = out_dir / "last"
    start_epoch, best = 1, {"val_macro_f1": -1.0}
    if (last / "state.pt").exists():
        st = torch.load(last / "state.pt", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        random.setstate(st["rng_py"]); np.random.set_state(st["rng_np"]); torch.set_rng_state(st["rng_t"])
        torch.cuda.set_rng_state_all(st["rng_c"]); start_epoch, best = st["epoch"] + 1, st["best"]
        print(f"[{rid}] resuming after epoch {st['epoch']}")
    torch.cuda.reset_peak_memory_stats()
    t_run = time.perf_counter(); epoch_secs = []
    print(f"[{rid}] {len(pos_win)} windows with spans + {n_neg} of {len(neg_win)} span-free windows per epoch, "
          f"{steps_per_epoch} steps/epoch", flush=True)
    for epoch in range(start_epoch, recipe["epochs"] + 1):
        model.train()
        rng = np.random.default_rng(seed * 100 + epoch)
        ep = pos_win + [neg_win[i] for i in rng.choice(len(neg_win), n_neg, replace=False)]
        order = rng.permutation(len(ep))
        rows, t0 = [], time.perf_counter()
        for step in range(steps_per_epoch):
            batch = [ep[i] for i in order[step * recipe["batch_size"]:(step + 1) * recipe["batch_size"]]]
            ids, att, y, mask = collate(batch, corpus, labels, tok, n_out)
            ids, att, y, mask = ids.to(device), att.to(device), y.to(device), mask.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=att).logits
            loss = loss_fn(logits.float(), y, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["max_grad_norm"])
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if step % 20 == 0:
                rows.append({"run_id": rid, "kind": "step", "epoch": epoch,
                             "step": (epoch - 1) * steps_per_epoch + step + 1, "train_loss": loss.item(),
                             "lr": sched.get_last_lr()[0]})
        torch.cuda.synchronize(); train_s = time.perf_counter() - t0; epoch_secs.append(train_s)
        t1 = time.perf_counter()
        vprobs = predict_contracts(model, val_c, corpus, tok, recipe, n_out, device)
        val_s = time.perf_counter() - t1
        summ, _ = evaluate_probs(vprobs, val_c, corpus, spans, cats)
        rows.append({"run_id": rid, "kind": "epoch", "epoch": epoch, "step": epoch * steps_per_epoch,
                     "train_loss": float(np.mean([r["train_loss"] for r in rows if r["kind"] == "step"])),
                     "val_macro_f1": summ["macro_f1"], "val_micro_f1": summ["micro_f1"],
                     "val_macro_precision": summ["macro_precision"], "val_macro_recall": summ["macro_recall"],
                     "epoch_train_seconds": train_s, "val_predict_seconds": val_s})
        print(f"[{rid}] epoch {epoch}: train loss {rows[-1]['train_loss']:.4f} | val macro-F1@IoU0.5 {summ['macro_f1']:.4f} "
              f"(P {summ['macro_precision']:.3f} R {summ['macro_recall']:.3f}) | {train_s:.0f} s + val {val_s:.0f} s", flush=True)
        if summ["macro_f1"] > best["val_macro_f1"]:
            best = {"val_macro_f1": summ["macro_f1"], "val_micro_f1": summ["micro_f1"], "epoch": epoch}
            if keep_weights:
                model.save_pretrained(out_dir / "best"); tok.save_pretrained(out_dir / "best")
                np.savez_compressed(out_dir / "val_probs.npz", **{f"c{i}": vprobs[c] for i, c in enumerate(val_c)})
        pd.DataFrame(rows).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
        last.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch,
                    "best": best, "rng_py": random.getstate(), "rng_np": np.random.get_state(),
                    "rng_t": torch.get_rng_state(), "rng_c": torch.cuda.get_rng_state_all()}, last / "state.pt")
    summary = {"run_id": rid, "variant": variant, "seed": seed, "best_epoch": best["epoch"],
               "val_macro_f1": best["val_macro_f1"], "val_micro_f1": best["val_micro_f1"], "epochs": recipe["epochs"],
               "windows_with_spans": len(pos_win), "span_free_windows_sampled": n_neg, "steps_per_epoch": steps_per_epoch,
               "mean_epoch_train_seconds": float(np.mean(epoch_secs)) if epoch_secs else None,
               "run_seconds": time.perf_counter() - t_run, "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
               "recipe": json.dumps(recipe)}
    pd.DataFrame([summary]).to_csv(SEARCH, mode="a", header=not SEARCH.exists(), index=False)
    shutil.rmtree(last, ignore_errors=True)
    del model, opt; torch.cuda.empty_cache()


def run_plan(data, tok):
    for v in VARIANTS:
        train_run(v, 42, data, tok)
    s = pd.read_csv(SEARCH)
    s42 = s[s.seed == 42]
    best_v = s42.sort_values("val_macro_f1", ascending=False).iloc[0].variant
    for seed in SEEDS[1:]:
        train_run(best_v, seed, data, tok, keep_weights=False)
    s = pd.read_csv(SEARCH)
    sel = s[s.variant == best_v]
    choice = {"variant": best_v, "test_run": run_id(best_v, 42), "val_macro_f1_seed42": float(sel[sel.seed == 42].val_macro_f1.iloc[0]),
              "val_macro_f1_mean": float(sel.val_macro_f1.mean()), "val_macro_f1_std": float(sel.val_macro_f1.std()),
              "n_seeds": int(len(sel))}
    (C.RESULTS_DIR / "task2_choice.json").write_text(json.dumps(choice, indent=2))
    for d in OUT.glob("t2_*"):                      # keep weights of the test run only
        if d.name != choice["test_run"]:
            shutil.rmtree(d / "best", ignore_errors=True)
    print(choice)


def extend_epochs(data, tok, epochs=8):
    """Rule recorded before the other variants were seen: if the chosen variant's best epoch is its
    last epoch, retrain it for `epochs` epochs (seed 42, validation only); the budget with the higher validation
    macro-F1 becomes the test recipe and seeds 43/44 are trained with that budget."""
    choice = json.loads((C.RESULTS_DIR / "task2_choice.json").read_text())
    s = pd.read_csv(SEARCH)
    base = s[s.run_id == choice["test_run"]].iloc[0]
    if int(base.best_epoch) < int(base.epochs):
        print("best epoch is not the last one - rule not triggered"); choice["epoch_extension"] = "not triggered"
    else:
        rec = {**RECIPE, "epochs": epochs}
        train_run(choice["variant"], 42, data, tok, recipe=rec)
        s = pd.read_csv(SEARCH)
        ext = s[s.run_id == run_id(choice["variant"], 42, epochs)].iloc[0]
        better = ext.val_macro_f1 > base.val_macro_f1
        choice["epoch_extension"] = {"triggered": True, "epochs": epochs, "val_macro_f1_4_epochs": float(base.val_macro_f1),
                                     "val_macro_f1_extended": float(ext.val_macro_f1), "adopted": bool(better)}
        if better:
            for seed in SEEDS[1:]:
                train_run(choice["variant"], seed, data, tok, recipe=rec, keep_weights=False)
            s = pd.read_csv(SEARCH)
            sel = s[(s.variant == choice["variant"]) & (s.epochs == epochs)]
            choice.update({"test_run": run_id(choice["variant"], 42, epochs), "epochs": epochs,
                           "val_macro_f1_seed42": float(ext.val_macro_f1), "val_macro_f1_mean": float(sel.val_macro_f1.mean()),
                           "val_macro_f1_std": float(sel.val_macro_f1.std()), "n_seeds": int(len(sel))})
    (C.RESULTS_DIR / "task2_choice.json").write_text(json.dumps(choice, indent=2))
    for d in OUT.glob("t2_*"):
        if d.name != choice["test_run"]:
            shutil.rmtree(d / "best", ignore_errors=True)
    print(choice)


def benchmark(data, tok):
    cs, spans, cats, corpus, labels = data
    n_out = 2 * len(cats)
    device = torch.device("cuda")
    train_c = list(cs[cs.split == "train"].contract)
    res = {}
    for win, stride in [(128, 64), (256, 128), (384, 192), (510, 255)]:
        rec = {**RECIPE, "window": win, "stride": stride}
        all_win = windows_of(train_c, corpus, rec)
        pos_win = [w for w in all_win if labels[w[0]][w[1]:w[2]].any()]
        n_epoch = len(pos_win) + int(RECIPE["neg_fraction"] * (len(all_win) - len(pos_win)))
        model = AutoModelForTokenClassification.from_pretrained(CKPT, num_labels=n_out).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
        loss_fn = make_loss("bce_plain", None)
        torch.cuda.reset_peak_memory_stats()
        bs = 16
        for step in range(25):
            if step == 5:
                torch.cuda.synchronize(); t0 = time.perf_counter()
            ids, att, y, mask = collate(pos_win[step * bs:(step + 1) * bs], corpus, labels, tok, n_out)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits
            loss = loss_fn(logits.float(), y.to(device), mask.to(device)); loss.backward(); opt.step(); opt.zero_grad()
        torch.cuda.synchronize(); ms = (time.perf_counter() - t0) / 20 * 1000
        res[f"{win}/{stride}"] = {"ms_per_step_batch16": ms, "train_windows_total": len(all_win), "windows_with_spans": len(pos_win),
                                  "windows_per_epoch": n_epoch, "projected_epoch_minutes": ms * n_epoch / bs / 1000 / 60,
                                  "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
        del model, opt; torch.cuda.empty_cache()
        print(win, stride, res[f"{win}/{stride}"], flush=True)
    r = res["256/128"]
    res["projection_256_128"] = {"runs": len(VARIANTS) + 2, "epochs": RECIPE["epochs"],
                                 "projected_training_hours": r["projected_epoch_minutes"] * RECIPE["epochs"] * (len(VARIANTS) + 2) / 60}
    (C.RESULTS_DIR / "task2_benchmark.json").write_text(json.dumps(res, indent=2))
    print(res["projection_256_128"])


def window_sensitivity(data, tok, window=384, stride=192):
    """Extra analysis (validation only, never evaluated on test): the chosen recipe (bce_plain, 8 epochs, seed 42)
    re-trained with 384/192 windows. Logged to separate files so the model-selection tables are unchanged."""
    global LOG, SEARCH
    LOG = C.RESULTS_DIR / "task2_window_sensitivity_log.csv"
    SEARCH = C.RESULTS_DIR / "task2_window_sensitivity_search.csv"
    choice = json.loads((C.RESULTS_DIR / "task2_choice.json").read_text())
    rec = {**RECIPE, "epochs": int(choice["epochs"]), "window": window, "stride": stride}
    train_run(choice["variant"], 42, data, tok, recipe=rec, keep_weights=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true"); ap.add_argument("--plan", action="store_true")
    ap.add_argument("--predict"); ap.add_argument("--split", default="test"); ap.add_argument("--extend", action="store_true")
    ap.add_argument("--window384", action="store_true")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(CKPT)
    data = load_data(tok)
    if a.benchmark:
        benchmark(data, tok)
    elif a.plan:
        with C.timer("t2_tagger_plan", "all_runs", notes="3 loss variants + 2 extra seeds; validation only", device="GPU"):
            run_plan(data, tok)
    elif a.extend:
        with C.timer("t2_tagger_plan", "epoch_extension", notes="rule: chosen variant re-trained with 8 epochs if best epoch was last; validation only", device="GPU"):
            extend_epochs(data, tok)
    elif a.window384:
        with C.timer("t2_window_sensitivity", "train_384_192", notes="bce_plain, 8 epochs, seed 42, window 384/192; validation only", device="GPU"):
            window_sensitivity(data, tok)
    elif a.predict:
        cs = data[0]
        contracts = list(cs[cs.split == a.split].contract)
        model = AutoModelForTokenClassification.from_pretrained(OUT / a.predict / "best").to("cuda")
        t0 = time.perf_counter()
        probs = predict_contracts(model, contracts, data[3], tok, RECIPE, 2 * len(data[2]), torch.device("cuda"))
        secs = time.perf_counter() - t0
        np.savez_compressed(OUT / a.predict / f"{a.split}_probs.npz", **{f"c{i}": probs[c] for i, c in enumerate(contracts)})
        (OUT / a.predict / f"{a.split}_predict_seconds.json").write_text(json.dumps(
            {"contracts": len(contracts), "seconds": secs, "seconds_per_contract": secs / len(contracts)}))
        print(f"predicted {len(contracts)} contracts in {secs:.1f} s")
