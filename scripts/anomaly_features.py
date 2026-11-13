"""PART F1 — compute once (GPU) the clause representations the anomaly scorers need, cached in models/anomaly_cache/:
mean-pooled last-layer embeddings of the fine-tuned Task 1 Legal-BERT (seed-42 test model) and of the NON-fine-tuned
Legal-BERT checkpoint, and the fine-tuned model's softmax probabilities, for every Task 1 clause (train / validation /
test) and the 557 held-out ambiguous clauses.

    python scripts/anomaly_features.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from src import common as C  # noqa: E402
from src.finetune import RECIPE  # noqa: E402
from src.preprocess import preprocess_for_transformer  # noqa: E402

OUT = ROOT / "models" / "anomaly_cache" / "features.npz"


@torch.no_grad()
def encode(model, tok, texts, with_logits):
    embs, probs = [], []
    for i in range(0, len(texts), 64):
        enc = tok([preprocess_for_transformer(t) for t in texts[i:i + 64]], truncation=True, max_length=RECIPE["max_length"],
                  padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if with_logits:
                out = model(**enc, output_hidden_states=True)
                h = out.hidden_states[-1].float(); probs.append(torch.softmax(out.logits.float(), -1).cpu().numpy())
            else:
                h = model(**enc).last_hidden_state.float()
        m = enc["attention_mask"].unsqueeze(-1).float()
        embs.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    return np.concatenate(embs), (np.concatenate(probs) if probs else None)


if __name__ == "__main__":
    t0 = time.perf_counter()
    df, amb = C.load_clauses(), C.load_ambiguous()
    ft_dir = ROOT / "models" / "legalbert_lr5e-05_s42" / "best"
    tok = AutoTokenizer.from_pretrained(ft_dir)
    ft = AutoModelForSequenceClassification.from_pretrained(ft_dir).cuda().eval()
    labels = [ft.config.id2label[i] for i in range(ft.config.num_labels)]
    e_ft, p_ft = encode(ft, tok, list(df.text), True)
    ea_ft, pa_ft = encode(ft, tok, list(amb.text), True)
    del ft; torch.cuda.empty_cache()
    base = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased").cuda().eval()
    e_b, _ = encode(base, tok, list(df.text), False)
    ea_b, _ = encode(base, tok, list(amb.text), False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, row_id=df.row_id.values, emb_ft=e_ft, emb_base=e_b, probs=p_ft, amb_emb_ft=ea_ft, amb_emb_base=ea_b,
                        amb_probs=pa_ft, labels=np.array(labels))
    print(f"features for {len(df)} + {len(amb)} clauses written in {time.perf_counter() - t0:.0f} s")
