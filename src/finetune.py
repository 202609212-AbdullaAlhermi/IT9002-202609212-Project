"""Fine-tuning a BERT-style encoder for clause classification - plain PyTorch, one function per step.

The recipe (every choice is explained in the Task 1 notebook, section 5):
    AdamW, weight decay 0.01 (not on biases / LayerNorm), linear warm-up over the first 10 % of steps then linear
    decay to 0, gradient clipping at 1.0, batch size 16, bfloat16 mixed precision on the GPU, dynamic padding
    (each batch is padded only to its longest clause), max_length 320, fixed seed.

Long-run safety: after every epoch a resume checkpoint (model + optimiser + scheduler + RNG state) is written to
models/<run_id>/last/. If the run is interrupted, calling train_run() again continues from the next epoch.
The resume checkpoint is deleted when the run finishes; only the best epoch's weights are kept (if asked).

Logs (appended once per completed epoch, so an interrupted epoch never leaves half-written rows):
    results/transformer_log.csv     per-step training loss and per-epoch validation loss / macro-F1 / accuracy
    results/transformer_search.csv  one summary row per finished run
"""
from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          get_linear_schedule_with_warmup)

from . import common as C
from .preprocess import preprocess_for_transformer

MODELS_DIR = C.ROOT / "models"
CHECKPOINTS = {"legalbert": "nlpaueb/legal-bert-base-uncased", "bert": "bert-base-uncased"}

def _max_length_from_results(default: int = 320) -> int:
    """max_length chosen in the Task 1 notebook (section 3.5) from the 99th percentile of the training clauses'
    token lengths under BOTH tokenizers (Legal-BERT and BERT-base), saved in results/token_lengths.json."""
    f = C.RESULTS_DIR / "token_lengths.json"
    return int(json.loads(f.read_text())["max_len"]) if f.exists() else default


RECIPE = dict(max_length=_max_length_from_results(), batch_size=16, epochs=5, warmup_ratio=0.10, weight_decay=0.01, max_grad_norm=1.0,
              eval_batch_size=64)


def run_id(model_key: str, lr: float, seed: int, class_weighted: bool = False) -> str:
    return f"{model_key}_lr{lr:.0e}_s{seed}" + ("_cw" if class_weighted else "")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data: tokenise once; the collator pads each batch to its own longest clause (dynamic padding)
# ---------------------------------------------------------------------------
def encode(tokenizer, texts, labels, label2id, max_length):
    enc = tokenizer([preprocess_for_transformer(t) for t in texts], truncation=True, max_length=max_length)
    return [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
             "labels": label2id[labels[i]]} for i in range(len(texts))]


def make_loader(items, tokenizer, batch_size, shuffle, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(items, batch_size=batch_size, shuffle=shuffle, generator=gen if shuffle else None,
                      collate_fn=DataCollatorWithPadding(tokenizer))


# ---------------------------------------------------------------------------
# Evaluation: loss, predictions and timing on one split
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model, loader, device, loss_fn=None):
    model.eval()
    losses, preds, n = [], [], 0
    t0 = time.perf_counter()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(**batch).logits.float()
        if loss_fn is not None:
            losses.append(loss_fn(logits, labels).item() * len(labels))
        preds.append(logits.argmax(-1).cpu())
        n += len(labels)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    return (sum(losses) / n if losses else None), torch.cat(preds).numpy(), seconds


# ---------------------------------------------------------------------------
# One complete fine-tuning run
# ---------------------------------------------------------------------------
def train_run(model_key: str, lr: float, seed: int, train_df: pd.DataFrame, val_df: pd.DataFrame,
              labels: list[str], class_weighted: bool = False, keep_best_weights: bool = False,
              recipe: dict = RECIPE, tag: str = "", lora: dict | None = None) -> dict:
    """tag (e.g. "_sens_e8"): a sensitivity run - its run_id gets the suffix and it is logged to
    results/transformer_sensitivity_{log,search}.csv, so the main runs, logs and test models are never touched.
    lora (Part F3): dict(r, alpha, dropout, target_modules) -> only LoRA adapters on those attention matrices and the
    classification head are trained (peft); logged to results/transformer_lora_{log,search}.csv; run_id gets "_lora"."""
    rid = run_id(model_key, lr, seed, class_weighted) + tag + ("_lora" if lora else "")
    out_dir = MODELS_DIR / rid
    kind = ("transformer_lora_sensitivity" if tag else "transformer_lora") if lora else ("transformer_sensitivity" if tag else "transformer")
    log_csv, search_csv = C.RESULTS_DIR / f"{kind}_log.csv", C.RESULTS_DIR / f"{kind}_search.csv"
    if search_csv.exists() and rid in pd.read_csv(search_csv).run_id.values:
        print(f"[{rid}] already finished - skipped (see transformer_search.csv)")
        return pd.read_csv(search_csv).set_index("run_id").loc[rid].to_dict()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)
    label2id = {lab: i for i, lab in enumerate(labels)}
    tok = AutoTokenizer.from_pretrained(CHECKPOINTS[model_key])
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINTS[model_key], id2label=dict(enumerate(labels)), label2id=label2id).to(device)
    if lora:
        from peft import LoraConfig, TaskType, get_peft_model
        model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=lora["r"], lora_alpha=lora["alpha"],
                                                 lora_dropout=lora["dropout"], target_modules=lora["target_modules"]))
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    tr_items = encode(tok, list(train_df.text), list(train_df.category), label2id, recipe["max_length"])
    va_items = encode(tok, list(val_df.text), list(val_df.category), label2id, recipe["max_length"])
    va_loader = make_loader(va_items, tok, recipe["eval_batch_size"], shuffle=False)

    # Optional class weights (inverse frequency, normalised to mean 1) - only kept if validation shows a benefit.
    weight = None
    if class_weighted:
        counts = train_df.category.value_counts().reindex(labels).values
        weight = torch.tensor(counts.sum() / (len(labels) * counts), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    eval_loss_fn = torch.nn.CrossEntropyLoss()  # validation loss always unweighted, so runs are comparable

    # AdamW: no weight decay on biases and LayerNorm weights (standard BERT practice).
    no_decay = ("bias", "LayerNorm.weight")
    groups = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(k in n for k in no_decay)],
               "weight_decay": recipe["weight_decay"]},
              {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in no_decay)],
               "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(groups, lr=lr)
    steps_per_epoch = -(-len(tr_items) // recipe["batch_size"])
    total_steps = steps_per_epoch * recipe["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(recipe["warmup_ratio"] * total_steps), total_steps)

    # ---- resume if an earlier run of this configuration was interrupted ----
    start_epoch, best = 1, {"val_macro_f1": -1.0}
    last = out_dir / "last"
    if (last / "state.pt").exists():
        state = torch.load(last / "state.pt", weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["rng_python"]); np.random.set_state(state["rng_numpy"])
        torch.set_rng_state(state["rng_torch"]); torch.cuda.set_rng_state_all(state["rng_cuda"])
        start_epoch, best = state["epoch"] + 1, state["best"]
        print(f"[{rid}] resuming after epoch {state['epoch']}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    run_start = time.perf_counter()
    epoch_times = []
    for epoch in range(start_epoch, recipe["epochs"] + 1):
        model.train()
        # a new, reproducible shuffle each epoch
        loader = make_loader(tr_items, tok, recipe["batch_size"], shuffle=True, seed=seed * 1000 + epoch)
        rows, t0 = [], time.perf_counter()
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            y = batch.pop("labels")
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["max_grad_norm"])
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            rows.append({"run_id": rid, "kind": "step", "epoch": epoch, "step": (epoch - 1) * steps_per_epoch + i + 1,
                         "train_loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_seconds = time.perf_counter() - t0
        epoch_times.append(train_seconds)

        val_loss, val_pred, _ = predict(model, va_loader, device, eval_loss_fn)
        y_val = [label2id[c] for c in val_df.category]
        m_f1 = f1_score(y_val, val_pred, average="macro", labels=list(range(len(labels))), zero_division=0)
        acc = accuracy_score(y_val, val_pred)
        rows.append({"run_id": rid, "kind": "epoch", "epoch": epoch, "step": epoch * steps_per_epoch,
                     "train_loss": float(np.mean([r["train_loss"] for r in rows])), "val_loss": val_loss,
                     "val_macro_f1": m_f1, "val_accuracy": acc, "epoch_train_seconds": train_seconds})
        print(f"[{rid}] epoch {epoch}: train loss {rows[-1]['train_loss']:.4f} | val loss {val_loss:.4f} | "
              f"val macro-F1 {m_f1:.4f} | acc {acc:.4f} | {train_seconds:.0f} s")

        if m_f1 > best["val_macro_f1"]:
            best = {"val_macro_f1": m_f1, "val_accuracy": acc, "val_loss": val_loss, "epoch": epoch,
                    "val_pred": [labels[p] for p in val_pred]}
            if keep_best_weights:
                model.save_pretrained(out_dir / "best"); tok.save_pretrained(out_dir / "best")

        # write logs for the finished epoch, then the resume checkpoint
        pd.DataFrame(rows).to_csv(log_csv, mode="a", header=not log_csv.exists(), index=False)
        last.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch, "best": best,
                    "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
                    "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all()},
                   last / "state.pt")

    summary = {"run_id": rid, "model": model_key, "checkpoint": CHECKPOINTS[model_key], "lr": lr, "seed": seed,
               "class_weighted": class_weighted, "best_epoch": best["epoch"],
               "val_macro_f1": best["val_macro_f1"], "val_accuracy": best["val_accuracy"],
               "val_loss_at_best": best["val_loss"], "epochs": recipe["epochs"],
               "mean_epoch_seconds": float(np.mean(epoch_times)) if epoch_times else None,
               "run_seconds": time.perf_counter() - run_start,
               "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
               "params_million": sum(p.numel() for p in model.parameters()) / 1e6,
               "trainable_params": n_trainable,
               "saved_weights_mb": (sum(f.stat().st_size for f in (out_dir / "best").rglob("*") if f.is_file()) / 1e6
                                    if keep_best_weights and (out_dir / "best").exists() else None),
               "lora": json.dumps(lora) if lora else "",
               "device": C._device_name("GPU" if device.type == "cuda" else "CPU"),
               "recipe": json.dumps(recipe)}
    pd.DataFrame([summary]).to_csv(search_csv, mode="a", header=not search_csv.exists(), index=False)
    pd.DataFrame({"row_id": val_df.row_id.values, "true": val_df.category.values, "pred": best["val_pred"]}).to_csv(
        C.RESULTS_DIR / "predictions" / f"val_{rid}.csv", index=False)
    shutil.rmtree(last, ignore_errors=True)  # resume checkpoint no longer needed
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def load_best(rid: str, device=None):
    """Load the saved best-epoch weights of a run (models/<run_id>/best)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = MODELS_DIR / rid / "best"
    return AutoTokenizer.from_pretrained(path), AutoModelForSequenceClassification.from_pretrained(path).to(device)


# ---------------------------------------------------------------------------
# The full plan (main model + ablation). Every decision rule is fixed here, before any run:
#   1. learning rate: {2e-5, 3e-5, 5e-5}, seed 42, for BOTH checkpoints (validation only);
#      the LR used for BOTH final models = the best LR of Legal-BERT (identical recipe for the ablation).
#   2. seeds 43 and 44 at that LR for both checkpoints -> mean +- std over 3 seeds (validation).
#   3. class-weighted loss for Legal-BERT with the same 3 seeds; adopted only if the mean validation gain over the
#      unweighted runs is >= 0.003 AND larger than the seed spread (max - min) of the unweighted runs.
#      If adopted, BERT-base is also trained class-weighted with the same 3 seeds (identical recipe).
#   4. the models evaluated on test are PRE-DECLARED: seed 42 of the chosen recipe (never the best seed).
# Every call to train_run() skips runs that already finished, so the plan can simply be re-started.
# ---------------------------------------------------------------------------
LR_GRID = [2e-5, 3e-5, 5e-5]
SEEDS = [42, 43, 44]


def run_full_plan(train_df, val_df, labels) -> dict:
    for key in ["legalbert", "bert"]:
        for lr in LR_GRID:
            train_run(key, lr, 42, train_df, val_df, labels, keep_best_weights=True)
    search = pd.read_csv(C.RESULTS_DIR / "transformer_search.csv")
    lb = search[(search.model == "legalbert") & (search.seed == 42) & (~search.class_weighted)]
    best_lr = float(lb.sort_values(["val_macro_f1", "lr"], ascending=[False, True]).iloc[0].lr)

    for key in ["legalbert", "bert"]:
        for seed in SEEDS[1:]:
            train_run(key, best_lr, seed, train_df, val_df, labels)
    for seed in SEEDS:
        train_run("legalbert", best_lr, seed, train_df, val_df, labels, class_weighted=True, keep_best_weights=(seed == 42))

    search = pd.read_csv(C.RESULTS_DIR / "transformer_search.csv")
    lbs = search[(search.model == "legalbert") & (search.lr == best_lr)]
    base, cw = lbs[~lbs.class_weighted].val_macro_f1, lbs[lbs.class_weighted].val_macro_f1
    gain, spread = float(cw.mean() - base.mean()), float(base.max() - base.min())
    use_cw = bool(gain >= 0.003 and gain > spread)
    if use_cw:  # identical recipe for the ablation model
        for seed in SEEDS:
            train_run("bert", best_lr, seed, train_df, val_df, labels, class_weighted=True, keep_best_weights=(seed == 42))
    choice = {"best_lr": best_lr, "seed_for_test": 42,
              "unweighted_val_mean": float(base.mean()), "unweighted_val_std": float(base.std()),
              "class_weighted_val_mean": float(cw.mean()), "class_weighted_val_std": float(cw.std()),
              "class_weighted_gain": gain, "seed_spread": spread, "use_class_weights": use_cw,
              "test_runs": {k: run_id(k, best_lr, 42, use_cw) for k in ["legalbert", "bert"]}}
    (C.RESULTS_DIR / "transformer_choice.json").write_text(json.dumps(choice, indent=2))

    # keep only the weights of the two pre-declared test models (each ~440 MB)
    for d in MODELS_DIR.glob("*"):
        if d.is_dir() and d.name not in choice["test_runs"].values():
            shutil.rmtree(d / "best", ignore_errors=True)
    return choice
