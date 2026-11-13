"""Run the whole fine-tuning plan (Legal-BERT + BERT-base ablation) outside the notebook, logging to a file.

    python scripts/train_transformer.py            (from the project folder; ~20 min on an RTX 5080)

Safe to interrupt and restart: finished runs are skipped and an interrupted run resumes from its last epoch.
The plan and its decision rules are in src/finetune.py (run_full_plan).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import common as C, finetune as F  # noqa: E402

if __name__ == "__main__":
    C.set_seed(42)
    df = C.load_clauses()
    labels = C.get_labels(df)
    with C.timer("transformer_plan", "all_runs", notes="LR search x2 models + 2 extra seeds x2 + class-weight run",
                 device="GPU"):
        choice = F.run_full_plan(df[df.split == "train"], df[df.split == "validation"], labels)
    print(choice)
