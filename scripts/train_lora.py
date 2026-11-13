"""PART F3 — LoRA vs full fine-tuning of Legal-BERT on Task 1. The training runs are validation only; the single test
evaluation (legalbert_lora) is `--test`, and the Task 1 notebook (section 9) re-checks it through the test-once helper.

    python scripts/train_lora.py            # lr search {1e-4, 3e-4, 1e-3} with seed 42, then seeds 43/44 at the best lr
    python scripts/train_lora.py --test     # the ONE test evaluation of the chosen run (legalbert_lora) + inference timing;
                                            # on a re-run it only checks that the predictions are identical

Same data, split, recipe (batch 16, 5 epochs, warm-up 10 %, weight decay 0.01, max length 320, bf16), seeds and metric
as the full fine-tune (src/finetune.py); LoRA rank 16, alpha 32, dropout 0.1 on the attention query/value matrices, the
classification head trainable. Rule fixed before the runs: the lr with the best seed-42 validation macro-F1 (ties -> the
smaller lr); the test model is seed 42 at that lr.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import common as C  # noqa: E402
from src import finetune as F  # noqa: E402

LORA = dict(r=16, alpha=32, dropout=0.1, target_modules=["query", "value"])
LR_GRID = [1e-4, 3e-4, 1e-3]

def test_eval():
    """Test-once evaluation of the pre-declared LoRA run (seed 42 at the chosen lr) + batch-1 latency on the same 100
    validation clauses as the other models and batched throughput on test -> results/lora_inference.json."""
    import time
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from src.preprocess import preprocess_for_transformer
    df = C.load_clauses()
    labels = sorted(df.category.unique())
    test, val = df[df.split == "test"], df[df.split == "validation"].reset_index(drop=True)
    choice = json.loads((C.RESULTS_DIR / "lora_choice.json").read_text())
    d = ROOT / "models" / choice["test_run"] / "best"
    tok = AutoTokenizer.from_pretrained(d)
    base = AutoModelForSequenceClassification.from_pretrained(F.CHECKPOINTS["legalbert"], num_labels=len(labels),
                                                              id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)})
    model = PeftModel.from_pretrained(base, d).cuda().eval()
    def predict(texts, bs):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                enc = tok([preprocess_for_transformer(t) for t in texts[i:i + bs]], truncation=True, max_length=F.RECIPE["max_length"],
                          padding=True, return_tensors="pt").to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out.append(model(**enc).logits.float().argmax(-1).cpu().numpy())
        torch.cuda.synchronize()
        return np.concatenate(out)
    predict(list(test.text[:64]), 64)                                        # warm-up
    t0 = time.perf_counter(); pred = [labels[i] for i in predict(list(test.text), 64)]; tput = (time.perf_counter() - t0) / len(test)
    sample = list(val.sample(100, random_state=42).text); secs = []
    for t in sample:
        t0 = time.perf_counter(); predict([t], 1); secs.append(time.perf_counter() - t0)
    cfg = {"run": choice["test_run"], "lora": choice["lora"], "lr": choice["best_lr"], "recipe": F.RECIPE,
           "val_macro_f1_seed42": choice["val_macro_f1_seed42"]}
    res = C.test_once("legalbert_lora", cfg, test.category, pred, labels, test.row_id, test.contract, C.robust_labels(df))
    inf = {"median_ms": 1000 * float(np.median(secs)), "p95_ms": 1000 * float(np.percentile(secs, 95)),
           "throughput_seconds_per_clause": tput, "note": "adapter not merged into the base weights"}
    f = C.RESULTS_DIR / "lora_inference.json"
    if not f.exists():
        f.write_text(json.dumps(inf, indent=2))
    print(res["macro_f1"], inf)


def sensitivity():
    """Extra analysis (validation only; the tested LoRA model is unchanged): seed 42, (a) lr 1e-3 for 10 epochs,
    (b) lr 3e-3 for 5 epochs. Logged to results/transformer_lora_sensitivity_{log,search}.csv."""
    df = C.load_clauses()
    labels = sorted(df.category.unique())
    train, val = df[df.split == "train"], df[df.split == "validation"]
    with C.timer("legalbert_lora_sensitivity", "all_runs", notes="LoRA lr 1e-3 x 10 epochs and lr 3e-3 x 5 epochs, seed 42; validation only", device="GPU"):
        F.train_run("legalbert", 1e-3, 42, train, val, labels, recipe={**F.RECIPE, "epochs": 10}, tag="_sens_e10", lora=LORA)
        F.train_run("legalbert", 3e-3, 42, train, val, labels, tag="_sens", lora=LORA)


if __name__ == "__main__" and "--test" in sys.argv:
    test_eval()
elif __name__ == "__main__" and "--sensitivity" in sys.argv:
    sensitivity()
elif __name__ == "__main__":
    df = C.load_clauses()
    labels = sorted(df.category.unique())
    train, val = df[df.split == "train"], df[df.split == "validation"]
    with C.timer("legalbert_lora_plan", "all_runs", notes="LoRA r16/a32 q,v: lr search x3 (seed 42) + 2 extra seeds; validation only", device="GPU"):
        for lr in LR_GRID:
            F.train_run("legalbert", lr, 42, train, val, labels, keep_best_weights=True, lora=LORA)
        s = pd.read_csv(C.RESULTS_DIR / "transformer_lora_search.csv")
        best_lr = float(s[s.seed == 42].sort_values(["val_macro_f1", "lr"], ascending=[False, True]).iloc[0].lr)
        for seed in [43, 44]:
            F.train_run("legalbert", best_lr, seed, train, val, labels, lora=LORA)
    s = pd.read_csv(C.RESULTS_DIR / "transformer_lora_search.csv")
    sel = s[s.lr == best_lr]
    choice = {"lora": LORA, "lr_grid": LR_GRID, "best_lr": best_lr, "test_run": F.run_id("legalbert", best_lr, 42) + "_lora",
              "val_macro_f1_seed42": float(sel[sel.seed == 42].val_macro_f1.iloc[0]),
              "val_macro_f1_mean": float(sel.val_macro_f1.mean()), "val_macro_f1_std": float(sel.val_macro_f1.std())}
    (C.RESULTS_DIR / "lora_choice.json").write_text(json.dumps(choice, indent=2))
    print(choice)
