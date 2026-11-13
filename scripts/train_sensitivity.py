"""Sensitivity checks for the Task 1 transformers (VALIDATION ONLY; no test evaluation; test models untouched).

    python scripts/train_sensitivity.py

1. Learning rate above the pre-declared grid: lr 7e-5 and 1e-4, seed 42, Legal-BERT and BERT-base, same recipe.
2. Epochs: BERT-base, lr 5e-5, seed 42, 8 epochs (the controlled ablation itself stays at 5 epochs).
Logs: results/transformer_sensitivity_search.csv and results/transformer_sensitivity_log.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import common as C, finetune as F  # noqa: E402

if __name__ == "__main__":
    df = C.load_clauses()
    labels = C.get_labels(df)
    tr, va = df[df.split == "train"], df[df.split == "validation"]
    with C.timer("transformer_sensitivity", "all_runs", notes="lr 7e-5/1e-4 x 2 models + BERT-base 8 epochs; validation only",
                 device="GPU"):
        for key in ["legalbert", "bert"]:
            for lr in [7e-5, 1e-4]:
                F.train_run(key, lr, 42, tr, va, labels, tag="_sens")
        F.train_run("bert", 5e-5, 42, tr, va, labels, recipe={**F.RECIPE, "epochs": 8}, tag="_sens_e8")
