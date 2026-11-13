"""Long runs of the prompted-LLM comparison (Task 1, 27 classes). Raw outputs are written incrementally to
results/llm/<model>_<version>_<split>.csv so a crash can resume; parsing, selection and scoring happen in
the LLM notebook.

    python scripts/run_llm.py --benchmark                      # 20 validation clauses per model x version -> projection
    python scripts/run_llm.py --validation                     # all 3 prompt versions x 2 models on full VALIDATION
    python scripts/run_llm.py --test qwen25_1p5b v2            # the version chosen on validation, on TEST (notebook calls it)
    python scripts/run_llm.py --latency qwen25_1p5b v2         # batch-1 latency on 100 validation clauses + peak memory
    python scripts/run_llm.py --constrained qwen25_1p5b v2     # optional: constrained label scoring on VALIDATION
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src import common as C  # noqa: E402
from src import llm as L  # noqa: E402

MODELS = {"qwen25_1p5b": "Qwen/Qwen2.5-1.5B-Instruct", "qwen25_3b": "Qwen/Qwen2.5-3B-Instruct"}
BATCH = {"v1": 32, "v2": 32, "v3": 8}
OUT = C.RESULTS_DIR / "llm"


def setup():
    df = C.load_clauses()
    labels = sorted(df.category.unique())
    ex_file = C.RESULTS_DIR / "llm_fewshot_examples.csv"
    if ex_file.exists():
        ex = pd.read_csv(ex_file)
    else:
        ex = L.pick_examples(df[df.split == "train"], labels)
        ex.to_csv(ex_file, index=False)
    examples = dict(zip(ex.category, ex.text))
    return df, labels, examples


def prompts_for(runner, df, labels, examples, version, split):
    sub = df[df.split == split]
    return {int(r.row_id): runner.render(L.build_messages(version, r.text, labels, examples)) for r in sub.itertuples()}


def save_prompt_templates(labels, examples):
    """results/llm_prompts.json: the exact chat messages of every version with {CLAUSE} as placeholder."""
    f = C.RESULTS_DIR / "llm_prompts.json"
    if not f.exists():
        f.write_text(json.dumps({v: {"description": L.VERSIONS[v], "messages": L.build_messages(v, "{CLAUSE}", labels, examples)}
                                 for v in L.VERSIONS}, indent=2))


def benchmark(df, labels, examples):
    val = df[df.split == "validation"]
    res = {}
    for key, mid in MODELS.items():
        r = L.Runner(mid)
        for v in L.VERSIONS:
            pr = prompts_for(r, val.head(20 + BATCH[v]), labels, examples, v, "validation")
            ids = list(pr)
            r.generate([pr[i] for i in ids[:BATCH[v]]])                   # warm-up
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for i in range(BATCH[v], len(ids), BATCH[v]):
                r.generate([pr[j] for j in ids[i:i + BATCH[v]]])
            torch.cuda.synchronize()
            spc = (time.perf_counter() - t0) / (len(ids) - BATCH[v])
            res[f"{key}_{v}"] = {"seconds_per_clause": spc, "mean_prompt_tokens": float(np.mean([r.n_tokens(pr[i]) for i in ids])),
                                 "batch_size": BATCH[v]}
            print(key, v, res[f"{key}_{v}"], flush=True)
        r.free()
    n_val, n_test = int((df.split == "validation").sum()), int((df.split == "test").sum())
    proj = sum(x["seconds_per_clause"] for x in res.values()) * n_val
    proj += max(res[f"{k}_{v}"]["seconds_per_clause"] for k in MODELS for v in L.VERSIONS) * n_test * len(MODELS)
    res["projection"] = {"validation_all_versions_minutes": sum(x["seconds_per_clause"] for k, x in res.items()
                                                                if k != "projection") * n_val / 60,
                         "total_incl_test_worst_case_minutes": proj / 60}
    (C.RESULTS_DIR / "llm_benchmark.json").write_text(json.dumps(res, indent=2))
    print(res["projection"])


def run_split(df, labels, examples, split, only=None):
    OUT.mkdir(parents=True, exist_ok=True)
    for key, mid in MODELS.items():
        if only and key != only[0]:
            continue
        r = L.Runner(mid)
        torch.cuda.reset_peak_memory_stats()
        for v in L.VERSIONS:
            if only and v != only[1]:
                continue
            f = OUT / f"{key}_{v}_{split}.csv"
            pr = prompts_for(r, df, labels, examples, v, split)
            if f.exists() and len(pd.read_csv(f)) == len(pr):
                print(f"{f.name} complete - skipped"); continue
            print(f"[{key} {v} {split}] {len(pr)} clauses", flush=True)
            with C.timer(f"{key}_{v}", f"inference_{split}_gpu", notes=f"{len(pr)} clauses, greedy, batch {BATCH[v]} (shrinks for long prompts), bf16",
                         device="GPU"):
                r.run(pr, f, BATCH[v])
            meta = OUT / "run_meta.json"
            m = json.loads(meta.read_text()) if meta.exists() else {}
            m[f"{key}_{v}_{split}"] = {"peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "n_params": r.n_params,
                                       "model_id": mid, "batch_size": BATCH[v]}
            meta.write_text(json.dumps(m, indent=2))
        r.free()


def retrieval(df, labels, key="qwen25_3b"):
    """Validation-only analysis: prompt v4 = v2 + the 3 nearest TRAIN clauses."""
    OUT.mkdir(parents=True, exist_ok=True)
    ret = L.Retriever(df[df.split == "train"])
    val = df[df.split == "validation"]
    r = L.Runner(MODELS[key])
    pr = {int(x.row_id): r.render(L.build_messages("v4", x.text, labels, ret.neighbours(x.text))) for x in val.itertuples()}
    nb = pd.DataFrame([{"row_id": int(x.row_id), "neighbour_labels": "|".join(l for _, l in ret.neighbours(x.text))} for x in val.itertuples()])
    nb.to_csv(OUT / f"{key}_v4_validation_neighbours.csv", index=False)
    f = OUT / f"{key}_v4_validation.csv"
    torch.cuda.reset_peak_memory_stats()
    with C.timer(f"{key}_v4", "inference_validation_gpu", notes=f"{len(pr)} clauses, retrieval few-shot (analysis only), batch 16; "
                 "GPU shared with the LEDGAR training job", device="GPU"):
        r.run(pr, f, 16)
    meta = OUT / "run_meta.json"
    m = json.loads(meta.read_text())
    m[f"{key}_v4_validation"] = {"peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "n_params": r.n_params,
                                 "model_id": MODELS[key], "batch_size": 16, "note": "GPU shared with another job"}
    meta.write_text(json.dumps(m, indent=2))
    r.free()


def latency(df, labels, examples, key, v, n=100):
    r = L.Runner(MODELS[key])
    val = df[df.split == "validation"].sample(n, random_state=42)
    pr = list(prompts_for(r, val, labels, examples, v, "validation").values())
    torch.cuda.reset_peak_memory_stats()
    secs = r.latency(pr)
    res = {"model": key, "version": v, "n_clauses": n, "median_ms": float(np.median(secs) * 1000),
           "p95_ms": float(np.percentile(secs, 95) * 1000), "mean_ms": float(secs.mean() * 1000),
           "peak_gpu_mem_gb_batch1": torch.cuda.max_memory_allocated() / 1e9, "device": torch.cuda.get_device_name(0),
           "mean_prompt_tokens": float(np.mean([r.n_tokens(p) for p in pr]))}
    (OUT / f"latency_{key}_{v}.json").write_text(json.dumps(res, indent=2))
    print(res)
    r.free()


def constrained(df, labels, examples, key, v):
    r = L.Runner(MODELS[key])
    pr = prompts_for(r, df, labels, examples, v, "validation")
    f = OUT / f"{key}_{v}_validation_constrained.csv"
    done = set(pd.read_csv(f).row_id) if f.exists() else set()
    rows, t0 = [], time.perf_counter()
    for i, (rid, p) in enumerate(pr.items()):
        if rid in done:
            continue
        s = r.label_scores(p, labels)
        rows.append({"row_id": rid, "pred": labels[int(np.argmax(s))], **{f"lp_{l}": x for l, x in zip(labels, s)}})
        if len(rows) == 50:
            pd.DataFrame(rows).to_csv(f, mode="a", header=not f.exists(), index=False); rows = []
            print(f"  {i + 1}/{len(pr)} {time.perf_counter() - t0:.0f} s", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(f, mode="a", header=not f.exists(), index=False)
    secs = time.perf_counter() - t0
    m = OUT / "run_meta.json"
    meta = json.loads(m.read_text()) if m.exists() else {}
    meta[f"{key}_{v}_validation_constrained"] = {"seconds": secs, "clauses": len(pr)}
    m.write_text(json.dumps(meta, indent=2))
    r.free()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true"); ap.add_argument("--validation", action="store_true")
    ap.add_argument("--test", nargs=2); ap.add_argument("--latency", nargs=2); ap.add_argument("--constrained", nargs=2)
    ap.add_argument("--retrieval", action="store_true")
    a = ap.parse_args()
    df, labels, examples = setup()
    save_prompt_templates(labels, examples)
    if a.benchmark:
        benchmark(df, labels, examples)
    if a.validation:
        run_split(df, labels, examples, "validation")
    if a.test:
        run_split(df, labels, examples, "test", only=a.test)
    if a.latency:
        latency(df, labels, examples, *a.latency)
    if a.retrieval:
        retrieval(df, labels)
    if a.constrained:
        constrained(df, labels, examples, *a.constrained)
