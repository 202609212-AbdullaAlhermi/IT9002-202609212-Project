"""Generate notebooks/STUDENTID_LLM.ipynb (Task 1 by prompting two local instruction-tuned LLMs).

    python scripts/notebook_builders/build_llm.py      (outputs are cleared; execute the notebook afterwards)
"""
from pathlib import Path

import sys
import nbformat as nbf
sys.path.insert(0, str(Path(__file__).parent))
from colab_cell import COLAB_MD, COLAB_CODE  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks" / "STUDENTID_LLM.ipynb"
INTERP = {}
interp_file = Path(__file__).with_name("llm_interp.py")
if interp_file.exists():
    exec(interp_file.read_text(encoding="utf-8"), {"INTERP": INTERP})
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


def interp(key, default="*Interpretation is written after the cell above has been run.*"):
    md(INTERP.get(key, default))


# =====================================================================================
md(r"""
# IT9002 Natural Language Processing — Individual Project
## Notebook 3 of 3: Task 1 by prompting two local LLMs (zero-shot and few-shot)

**Dataset.** CUAD v1 — Contract Understanding Atticus Dataset (Hendrycks, Burns, Chen & Ball, 2021; CC BY 4.0).
Official page: https://www.atticusprojectai.org/cuad · data and code: https://github.com/TheAtticusProject/cuad

**Question.** Can a small instruction-tuned language model, with no training at all, classify contract clauses into our 27
categories — and how does it compare with the fine-tuned Legal-BERT of notebook 1 on macro-F1, cost and latency?

**Models (both run locally on the GPU with the `transformers` library, bf16, each with its own chat template):**
- primary — `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0);
- second — `Qwen/Qwen2.5-3B-Instruct` (Qwen Research Licence). The planned second model, `meta-llama/Llama-3.2-3B-Instruct`,
  is gated and was not accessible from this machine; the fallback is deviation **D3** in the report.

**Important framing.** These are *small open* models (1.5 B and 3 B parameters) used without any training. Their scores are
a **lower bound for what prompting can do**, not a statement about large hosted LLMs.

**Protocol (test set stays clean).** Three prompt versions per model are run on the full **validation** set; the version
with the best validation macro-F1 is chosen per model and run **once** on the 1,444 test clauses through the test-once
helper (`results/test_eval_log.csv`). Decoding is greedy (deterministic).

### How to run
- Top to bottom. The long generation runs are done by `scripts/run_llm.py` and saved in `results/llm/`; with
  `RERUN = False` (default) the notebook parses the saved raw outputs, so it re-runs in a few minutes. The test outputs and
  latency measurements are produced by the same script on the first run (or when missing).
- Google Colab: set the GitHub URL in the first cell; a GPU runtime is needed for the model cells.
""")

md(COLAB_MD)
code(COLAB_CODE)

md(r"""
## 0. Setup
""")

code(r'''
import sys, os, json, platform, subprocess, importlib, time, warnings, inspect
from pathlib import Path
IN_COLAB = "google.colab" in sys.modules or os.environ.get("IT9002_SIMULATE_COLAB") == "1"
# The project folder = the first folder, going upwards, that contains project_info.json (works from notebooks/, from the
# project folder and on Colab, where the cell above has moved into the project).
ROOT = next(d for d in [Path.cwd(), *Path.cwd().parents] if (d / "project_info.json").exists())
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", message="IProgress not found")

from src import common as C, llm as L, plotting as PL
for mod in (C, L, PL):
    importlib.reload(mod)
import numpy as np, pandas as pd, torch, transformers, sklearn
import matplotlib.pyplot as plt
transformers.logging.set_verbosity_error()
C.set_seed(42); PL.setup_style()
INFO = C.load_project_info()
print(f"Student: {INFO['STUDENT_NAME']} | ID: {INFO['STUDENT_ID']} | {INFO['COURSE_CODE']} {INFO['COURSE_NAME']}")
print("Dataset:", INFO["DATASET_URL"], "|", INFO["DATASET_CODE_URL"])
print(f"Python {platform.python_version()} | torch {torch.__version__} | transformers {transformers.__version__} | "
      f"scikit-learn {sklearn.__version__} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

C.ensure_data(INFO["GITHUB_URL"])   # frozen CSVs + raw CUAD: present, from GitHub, or rebuilt; checksums verified
C.register_needs_models()
df = C.load_clauses()
LABELS = sorted(df.category.unique()); ROBUST = C.robust_labels(df)
train, val, test = (df[df.split == s].reset_index(drop=True) for s in ["train", "validation", "test"])
MODELS = {"qwen25_1p5b": "Qwen/Qwen2.5-1.5B-Instruct", "qwen25_3b": "Qwen/Qwen2.5-3B-Instruct"}
NICE = {"qwen25_1p5b": "Qwen2.5-1.5B-Instruct", "qwen25_3b": "Qwen2.5-3B-Instruct"}
VERSIONS = list(L.VERSIONS)
LLM_DIR = C.RESULTS_DIR / "llm"
print(f"{len(LABELS)} classes | train {len(train):,} / validation {len(val):,} / test {len(test):,} clauses")
''')

# =====================================================================================
md(r"""
## 1. The two models: identity, size, licence and context length
Read from the files actually loaded (model card front matter, `config.json`, and the parameter count of the loaded
weights, saved by `scripts/run_llm.py` in `results/llm/run_meta.json`). First, the access check for the planned Llama model
(configuration file only; no token is ever requested or stored).
""")

code(r'''
from huggingface_hub import hf_hub_download
from transformers import AutoConfig
try:
    AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    print("Llama-3.2-3B-Instruct: accessible")
except Exception as e:
    print(f"Llama-3.2-3B-Instruct: NOT accessible here ({type(e).__name__}) -> second model = Qwen2.5-3B-Instruct (deviation D3)")

meta = json.load(open(LLM_DIR / "run_meta.json"))
rows = []
for key, mid in MODELS.items():
    card = Path(hf_hub_download(mid, "README.md")).read_text(encoding="utf-8")
    front = card.split("---")[1]
    lic = dict(l.split(": ", 1) for l in front.strip().splitlines() if ": " in l)
    cfg = AutoConfig.from_pretrained(mid)
    rev = Path(hf_hub_download(mid, "config.json")).parent.name
    n_params = next(v["n_params"] for k, v in meta.items() if k.startswith(key + "_v"))
    rows.append({"model": mid, "revision": rev[:7], "parameters": f"{n_params / 1e9:.2f} B",
                 "licence (model card)": lic.get("license_name", lic.get("license")),
                 "context length (max_position_embeddings)": cfg.max_position_embeddings,
                 "layers": cfg.num_hidden_layers, "hidden size": cfg.hidden_size})
MODEL_TABLE = pd.DataFrame(rows); display(MODEL_TABLE)
MODEL_TABLE.to_csv(C.RESULTS_DIR / "llm_models.csv", index=False)
print('Qwen2.5-3B-Instruct licence notice: "Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) '
      'Alibaba Cloud. All Rights Reserved." (non-commercial: research or evaluation use only).')
''')

interp("models")

# =====================================================================================
md(r"""
## 2. Prompt versions
Every prompt has a fixed **system message** (role: contract lawyer), the list of the 27 category names (alphabetical),
the clause between triple quotes (never truncated) and the instruction *"Answer with exactly one category name from the
list, copied exactly, and nothing else."* The three versions differ in how much the model is told about the categories:

| version | content |
|---|---|
| v1 | zero-shot, category **names only** |
| v2 | zero-shot, names + a **one-line description** of each category (adapted and shortened from the official CUAD category descriptions, `data/raw/category_descriptions.csv`) |
| v3 | **few-shot**: names + **one TRAIN clause per category** as an example |

**Few-shot example rule (deterministic, TRAIN only):** for each category, the training clause of 15–80 words (so no
example has to be cut) whose TF-IDF vector is closest (cosine) to the category's TF-IDF centroid — its most *typical*
short clause. The exact prompts are saved in `results/llm_prompts.json` (with `{CLAUSE}` as placeholder).
""")

code(r'''
EX = pd.read_csv(C.RESULTS_DIR / "llm_fewshot_examples.csv")
EXAMPLES = dict(zip(EX.category, EX.text))
assert set(EX.row_id) <= set(train.row_id), "few-shot examples must come from TRAIN"
display(EX[["category", "n_words", "cosine_to_centroid", "text"]].assign(text=EX.text.str[:110] + "...").round(3))
PROMPTS = json.load(open(C.RESULTS_DIR / "llm_prompts.json"))
for v in ["v1", "v2"]:
    print(f"===== {v}: {PROMPTS[v]['description']} =====")
    for m in PROMPTS[v]["messages"]:
        print(f"[{m['role']}]\n{m['content']}\n")
v3_user = PROMPTS["v3"]["messages"][1]["content"]
print("===== v3 (user message, middle part shortened) =====")
print(v3_user[:1400], "\n[...]\n", v3_user[-500:])
''')

code(r'''
# Prompt length in tokens (both models share the same Qwen2.5 tokenizer) and its effect on speed
from transformers import AutoTokenizer
toks = {k: AutoTokenizer.from_pretrained(m) for k, m in MODELS.items()}
assert toks["qwen25_1p5b"].get_vocab() == toks["qwen25_3b"].get_vocab(), "different tokenizers"
tk = toks["qwen25_1p5b"]
def render(v, clause):
    return tk.apply_chat_template(L.build_messages(v, clause, LABELS, EXAMPLES), tokenize=False, add_generation_prompt=True)
clause_tokens = np.array([len(tk(t, add_special_tokens=False).input_ids) for t in val.text])
tok_rows = []
for v in VERSIONS:
    fixed = len(tk(render(v, ""), add_special_tokens=False).input_ids)
    tok_rows.append({"version": v, "fixed prompt tokens (without the clause)": fixed,
                     "mean prompt tokens per clause (validation)": fixed + clause_tokens.mean(),
                     "median": fixed + np.median(clause_tokens), "max": fixed + clause_tokens.max()})
TOKENS = pd.DataFrame(tok_rows).set_index("version")
display(TOKENS.round(1))
''')

# =====================================================================================
md(r"""
## 3. Output-to-label conversion (one documented function)
The model answers in free text, so its answer must be converted into one of the 27 labels. `L.parse_label` applies four
rules in order: **exact** match of the first line (after removing a leading "Category:" and quotes) → **normalised** match
(case and punctuation ignored, "intellectual property" = "ip") → **contained**: label names found as whole words anywhere
in the answer, where an occurrence inside a longer one is discarded (the longest name wins), accepted only if all remaining
occurrences name **one** label → otherwise **INVALID**, counted as wrong for every class (it can never be a correct
prediction). Label names that contain or overlap other names (e.g. *Joint Ip Ownership* / *Ip Ownership Assignment*,
*License Grant* / *Non-Transferable License*, *Non-Compete* / *Competitive Restriction Exception*) are handled by the
"one label only" rule and tested on hand-made cases below (`tests/test_llm_parser.py`).
""")

code(r'''
print(inspect.getsource(L.parse_label))
sys.path.insert(0, str(ROOT / "tests"))
import test_llm_parser as TP
importlib.reload(TP)
cases = pd.DataFrame([{"model answer": o, "expected": e, "parsed": L.parse_label(o, LABELS)[0], "rule": L.parse_label(o, LABELS)[1],
                       "ok": L.parse_label(o, LABELS) == (e, r)} for o, e, r in TP.CASES])
display(cases)
TP.test_cases(); TP.test_longest_name_wins()
print(f"All {len(cases)} hand-made cases + 3 longest-name cases pass.")
''')

# =====================================================================================
md(r"""
## 4. Running the prompts on validation
Before the full runs, 20 validation clauses were timed per model and version (`python scripts/run_llm.py --benchmark`)
to project the total time. The full runs use batches with **left padding** (32 clauses per batch for v1/v2, 8 for the
long few-shot prompt), sorted by prompt length, **greedy decoding** (`do_sample=False`; the models' default sampling
settings — temperature 0.7, top-p 0.8 and a repetition penalty — are switched off), at most 24 new tokens, and write every
batch to `results/llm/<model>_<version>_validation.csv` so a crash can resume.
""")

code(r'''
RERUN = False     # True: python scripts/run_llm.py --validation   (about 10 min on an RTX 5080)
if RERUN:
    subprocess.run([sys.executable, "scripts/run_llm.py", "--validation"], check=True)
bench = json.load(open(C.RESULTS_DIR / "llm_benchmark.json"))
display(pd.DataFrame({k: v for k, v in bench.items() if k != "projection"}).T.round(4))
print(f"Projection: all versions on validation {bench['projection']['validation_all_versions_minutes']:.1f} min; "
      f"incl. the two test runs (worst case) {bench['projection']['total_incl_test_worst_case_minutes']:.1f} min -> no cuts needed.")
''')

md(r"""
## 5. Validation results and choice of the prompt version
""")

code(r'''
def load_raw(key, v, split):
    sub = val if split == "validation" else test
    raw = pd.read_csv(LLM_DIR / f"{key}_{v}_{split}.csv").drop_duplicates("row_id", keep="last").set_index("row_id")
    raw = raw.loc[sub.row_id]
    parsed = [L.parse_label(o, LABELS) for o in raw.output.fillna("").astype(str)]
    return raw, [p for p, _ in parsed], [r for _, r in parsed]

def gpu_seconds(raw):
    return float((raw.batch_seconds / raw.batch_size).sum())

VAL_PRED, VAL_RAW, VAL_RULE, val_rows = {}, {}, {}, []
for key in MODELS:
    for v in VERSIONS:
        raw, pred, rule = load_raw(key, v, "validation")
        VAL_PRED[(key, v)], VAL_RAW[(key, v)], VAL_RULE[(key, v)] = pred, raw, rule
        m = C.compute_metrics(val.category, pred, LABELS, f"{key}_{v}", "validation", robust_labels=ROBUST)
        pd.DataFrame({"row_id": val.row_id, "contract": val.contract, "true": val.category, "pred": pred,
                      "raw_output": raw.output.values, "rule": rule}).to_csv(C.RESULTS_DIR / "predictions" / f"{key}_{v}_validation.csv", index=False)
        rc = pd.Series(rule).value_counts(normalize=True)
        val_rows.append({"model": NICE[key], "key": key, "version": v, "macro-F1": m["macro_f1"], "macro-F1 (24 robust classes)": m["macro_f1_robust"],
                         "accuracy": m["accuracy"], "invalid %": 100 * (np.array(pred) == L.INVALID).mean(),
                         "exact %": 100 * rc.get("exact", 0), "normalised %": 100 * rc.get("normalised", 0), "contained %": 100 * rc.get("contained", 0),
                         "mean prompt tokens": float(raw.prompt_tokens.mean()), "GPU s per clause": gpu_seconds(raw) / len(raw)})
VAL_TAB = pd.DataFrame(val_rows)
VAL_TAB.to_csv(C.RESULTS_DIR / "llm_validation_summary.csv", index=False)
display(VAL_TAB.drop(columns="key").round(4))

CHOICE = {}
for key in MODELS:
    t = VAL_TAB[VAL_TAB.key == key].sort_values("macro-F1", ascending=False)
    CHOICE[key] = {"version": t.iloc[0].version, "val_macro_f1": float(t.iloc[0]["macro-F1"]),
                   "runner_up": t.iloc[1].version, "runner_up_val_macro_f1": float(t.iloc[1]["macro-F1"])}
json.dump(CHOICE, open(C.RESULTS_DIR / "llm_choice.json", "w"), indent=2)
print("Chosen on VALIDATION:", {NICE[k]: c["version"] for k, c in CHOICE.items()})
''')

code(r'''
lb_val = float(pd.read_csv(C.RESULTS_DIR / "transformer_search.csv").set_index("run_id").loc["legalbert_lr5e-05_s42", "val_macro_f1"])
cl_val = json.loads(pd.read_csv(C.RESULTS_DIR / "test_eval_log.csv").set_index("model").loc["tfidf_logreg", "config"])["val_macro_f1"]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
w = 0.38; x = np.arange(len(VERSIONS))
for i, (key, col) in enumerate([("qwen25_1p5b", PL.BLUE), ("qwen25_3b", PL.ORANGE)]):
    t = VAL_TAB[VAL_TAB.key == key].set_index("version").loc[VERSIONS]
    b = axes[0].bar(x + (i - 0.5) * w, t["macro-F1"], w, color=col, label=NICE[key])
    axes[0].bar_label(b, fmt="%.3f", fontsize=8)
    b2 = axes[1].bar(x + (i - 0.5) * w, t["invalid %"], w, color=col, label=NICE[key])
    axes[1].bar_label(b2, fmt="%.1f", fontsize=8)
axes[0].axhline(lb_val, color=PL.INK, linestyle="--", linewidth=1); axes[0].text(2.45, lb_val + 0.01, f"Legal-BERT {lb_val:.3f}", ha="right", fontsize=8)
axes[0].axhline(cl_val, color=PL.INK_2, linestyle=":", linewidth=1); axes[0].text(2.45, cl_val + 0.01, f"TF-IDF LogReg {cl_val:.3f}", ha="right", fontsize=8)
labels_x = ["v1\nnames only", "v2\nnames + descriptions", "v3\nnames + 1 example/class"]
for ax in axes:
    ax.set_xticks(x); ax.set_xticklabels(labels_x, fontsize=8); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("validation macro-F1 (27 classes)"); axes[0].set_ylim(0, 1); axes[0].legend(fontsize=8, loc="upper left")
axes[1].set_ylabel("invalid outputs (% of validation clauses)"); axes[1].legend(fontsize=8)
axes[0].set_title("Validation macro-F1 by prompt version", fontsize=10); axes[1].set_title("Invalid-output rate by prompt version", fontsize=10)
fig.suptitle("Figure 26. Prompted LLMs on validation: macro-F1 and invalid-output rate per prompt version", fontweight="bold", y=1.03, fontsize=11)
PL.savefig(fig, 26, "llm_validation_versions"); plt.show()
''')

code(r'''
# What do the invalid outputs look like? (validation)
for key in MODELS:
    for v in VERSIONS:
        raw, rule = VAL_RAW[(key, v)], pd.Series(VAL_RULE[(key, v)], index=VAL_RAW[(key, v)].index)
        bad = raw[rule.values == "ambiguous"].output.tolist() + raw[rule.values == "no label"].output.tolist()
        if bad:
            top = pd.Series([b.strip()[:70] for b in bad]).value_counts().head(5)
            print(f"{NICE[key]} {v}: {len(bad)} invalid ({(rule == 'ambiguous').sum()} name two labels, {(rule == 'no label').sum()} name none). Most frequent:")
            for t_, n_ in top.items():
                print(f"     {n_:>3} x  {t_!r}")
''')

interp("validation")

# =====================================================================================
md(r"""
## 6. Optional analysis (VALIDATION ONLY): constrained label scoring
*Clearly an extra analysis; it is not used to choose anything and it is not run on test.* How much of the gap to the
fine-tuned model is caused by **invalid or off-list answers** rather than by not knowing the category? Instead of letting
the model write freely, we compute for every clause the log-probability that the model's answer is **exactly** each of the
27 label names followed by its end-of-turn token (`L.Runner.label_scores`; the prompt is encoded once and shared by the 27
candidates) and pick the most probable label. This can never be invalid. We also report a *hybrid*: free generation, with
only the invalid answers replaced by the constrained choice. Run with the chosen prompt of each model
(`python scripts/run_llm.py --constrained <model> <version>`).
""")

code(r'''
cons_rows = []
for key in MODELS:
    v = CHOICE[key]["version"]
    f = LLM_DIR / f"{key}_{v}_validation_constrained.csv"
    if not f.exists():
        print(f"{f.name} missing - run scripts/run_llm.py --constrained {key} {v}"); continue
    cons = pd.read_csv(f).drop_duplicates("row_id").set_index("row_id").loc[val.row_id]
    free = np.array(VAL_PRED[(key, v)])
    hybrid = np.where(free == L.INVALID, cons.pred.values, free)
    for label, pred in [("free generation (as in the main table)", free), ("constrained label scoring", cons.pred.values),
                        ("hybrid: free, invalid -> constrained", hybrid)]:
        cons_rows.append({"model": NICE[key], "version": v, "decoding": label,
                          "macro-F1": C.compute_metrics(val.category, pred, LABELS, "x", "validation", save=False)["macro_f1"],
                          "accuracy": float((pred == val.category.values).mean()), "invalid %": 100 * float((pred == L.INVALID).mean())})
    agree = float((cons.pred.values[free != L.INVALID] == free[free != L.INVALID]).mean())
    print(f"{NICE[key]} {v}: constrained choice = free answer on {100 * agree:.1f} % of the valid free answers; "
          f"{json.load(open(LLM_DIR / 'run_meta.json'))[f'{key}_{v}_validation_constrained']['seconds']:.0f} s for {len(val)} clauses")
CONS = pd.DataFrame(cons_rows)
if len(CONS):
    CONS.to_csv(C.RESULTS_DIR / "llm_constrained_validation.csv", index=False)
    display(CONS.round(4))
''')

interp("constrained")

md(r"""
### 6.2 Optional analysis (VALIDATION ONLY): retrieval few-shot prompt v4 (Qwen2.5-3B only)
*An extra analysis, not used for any choice and never run on test.* Why did the fixed few-shot
block (v3) hurt? v3 shows one *typical* example of every category — 27 examples, about 1,900 tokens, mostly unrelated to the
clause at hand. Prompt **v4** keeps the v2 descriptions and adds only the **3 most similar TRAIN clauses** (TF-IDF cosine,
`L.Retriever`; each shortened to its first 80 words) with their labels, so the examples are few, short and relevant.
Run with `python scripts/run_llm.py --retrieval` (the GPU was shared with the LEDGAR training job during this run, so its
speed is not comparable). **The tested model stays v2** whatever the result, because the prompt choice was made before.
""")

code(r'''
v4f = LLM_DIR / "qwen25_3b_v4_validation.csv"
if v4f.exists():
    raw4, pred4, rule4 = load_raw("qwen25_3b", "v4", "validation")
    m4 = C.compute_metrics(val.category, pred4, LABELS, "qwen25_3b_v4", "validation", robust_labels=ROBUST)
    pd.DataFrame({"row_id": val.row_id, "contract": val.contract, "true": val.category, "pred": pred4, "raw_output": raw4.output.values,
                  "rule": rule4}).to_csv(C.RESULTS_DIR / "predictions" / "qwen25_3b_v4_validation.csv", index=False)
    nb4 = pd.read_csv(LLM_DIR / "qwen25_3b_v4_validation_neighbours.csv").set_index("row_id").loc[val.row_id]
    nb_lab = nb4.neighbour_labels.str.split("|")
    top1 = np.array([l[0] for l in nb_lab]); anyk = np.array([t in l for t, l in zip(val.category, nb_lab)])
    rows4 = [r for r in VAL_TAB[VAL_TAB.key == "qwen25_3b"].to_dict(orient="records")]
    rows4.append({"model": NICE["qwen25_3b"], "key": "qwen25_3b", "version": "v4 (analysis)", "macro-F1": m4["macro_f1"],
                  "macro-F1 (24 robust classes)": m4["macro_f1_robust"], "accuracy": m4["accuracy"],
                  "invalid %": 100 * float((np.array(pred4) == L.INVALID).mean()), "mean prompt tokens": float(raw4.prompt_tokens.mean())})
    V4 = pd.DataFrame(rows4)[["version", "macro-F1", "macro-F1 (24 robust classes)", "accuracy", "invalid %", "mean prompt tokens"]]
    V4.to_csv(C.RESULTS_DIR / "llm_retrieval_v4_validation.csv", index=False)
    display(V4.round(4))
    agree_top1 = float((np.array(pred4) == top1).mean())
    print(f"Retrieval quality: the nearest training clause has the right label for {100 * (top1 == val.category.values).mean():.1f} % of "
          f"validation clauses; one of the 3 neighbours has it for {100 * anyk.mean():.1f} %.")
    print(f"The v4 answer equals the label of the nearest neighbour for {100 * agree_top1:.1f} % of clauses; when the true label is "
          f"among the neighbours v4 is right for {100 * (np.array(pred4)[anyk] == val.category.values[anyk]).mean():.1f} %, otherwise for "
          f"{100 * (np.array(pred4)[~anyk] == val.category.values[~anyk]).mean():.1f} %.")
    print("Most predicted labels v3:", pd.Series(VAL_PRED[("qwen25_3b", "v3")]).value_counts().head(3).to_dict(),
          "| v4:", pd.Series(pred4).value_counts().head(3).to_dict())
    json.dump({"v4_val_macro_f1": m4["macro_f1"], "v2_val_macro_f1": CHOICE["qwen25_3b"]["val_macro_f1"],
               "nearest_label_correct": float((top1 == val.category.values).mean()), "true_label_in_top3": float(anyk.mean()),
               "v4_equals_nearest_label": agree_top1, "tested_model_unchanged": "qwen25_3b_v2"},
              open(C.RESULTS_DIR / "llm_retrieval_v4_summary.json", "w"), indent=2)
else:
    print("v4 outputs not found - run scripts/run_llm.py --retrieval")
''')

code(r'''
# How much of v4 is retrieval? Two baselines WITHOUT any LLM, using exactly the same retriever and neighbours (validation only)
if v4f.exists():
    nn1 = np.array([l[0] for l in nb_lab])
    def vote(labels):                                   # majority of the 3 neighbours; no majority -> nearest neighbour
        c = pd.Series(labels).value_counts()
        return c.index[0] if c.iloc[0] > 1 else labels[0]
    nn3 = np.array([vote(l) for l in nb_lab])
    rows_nn = []
    for label, pred in [("v2 (LLM, descriptions only)", np.array(VAL_PRED[("qwen25_3b", "v2")])),
                        ("nearest-neighbour label (no LLM)", nn1), ("majority of 3 neighbours (no LLM)", nn3),
                        ("v4 (LLM + 3 neighbours)", np.array(pred4))]:
        m_ = C.compute_metrics(val.category, pred, LABELS, "x", "validation", save=False, robust_labels=ROBUST)
        rows_nn.append({"method": label, "validation macro-F1": m_["macro_f1"], "macro-F1 (24 robust)": m_["macro_f1_robust"],
                        "accuracy": m_["accuracy"]})
    NN = pd.DataFrame(rows_nn); display(NN.round(4))
    NN.to_csv(C.RESULTS_DIR / "llm_retrieval_baselines_validation.csv", index=False)
    f = dict(zip(NN.method, NN["validation macro-F1"]))
    best_nn = max(f["nearest-neighbour label (no LLM)"], f["majority of 3 neighbours (no LLM)"])
    gain = f["v4 (LLM + 3 neighbours)"] - f["v2 (LLM, descriptions only)"]
    share = (best_nn - f["v2 (LLM, descriptions only)"]) / gain
    print(f"v2 -> v4 gain: {gain:+.3f}. The better retrieval-only baseline reaches {best_nn:.3f}, i.e. {100 * share:.0f} % of the v2->v4 gain "
          f"without any LLM; the LLM adds {f['v4 (LLM + 3 neighbours)'] - best_nn:+.3f} on top of retrieval alone.")
    s4 = json.load(open(C.RESULTS_DIR / "llm_retrieval_v4_summary.json"))
    s4.update({"nn1_val_macro_f1": f["nearest-neighbour label (no LLM)"], "nn3_vote_val_macro_f1": f["majority of 3 neighbours (no LLM)"],
               "share_of_v4_gain_from_retrieval": share})
    json.dump(s4, open(C.RESULTS_DIR / "llm_retrieval_v4_summary.json", "w"), indent=2)
''')

interp("retrieval")

# =====================================================================================
md(r"""
## 7. Test evaluation (once per model)
The chosen version of each model is run on the 1,444 test clauses (`scripts/run_llm.py --test`, only if its outputs are not
yet saved) and scored **once** through `C.test_once` (model names `qwen25_1p5b_<version>`, `qwen25_3b_<version>`).
Confidence intervals resample **contracts** (93 test contracts; point estimate over all 27 classes, each resample averages
over the classes present in it). Paired bootstraps compare models on the same resampled contracts.
""")

code(r'''
TEST_PRED, TEST_RES, TEST_RAW = {}, {}, {}
for key in MODELS:
    v = CHOICE[key]["version"]
    f = LLM_DIR / f"{key}_{v}_test.csv"
    if not f.exists() or pd.read_csv(f).row_id.nunique() < len(test):
        subprocess.run([sys.executable, "scripts/run_llm.py", "--test", key, v], check=True)
    raw, pred, rule = load_raw(key, v, "test")
    name = f"{key}_{v}"
    cfg = {"model_id": MODELS[key], "prompt_version": v, "prompt": L.VERSIONS[v], "decoding": "greedy, max_new_tokens 24",
           "selected_on": "validation macro-F1", "val_macro_f1": CHOICE[key]["val_macro_f1"], "parser": "src/llm.py parse_label"}
    TEST_RES[key] = C.test_once(name, cfg, test.category, pred, LABELS, test.row_id, test.contract, ROBUST)
    TEST_PRED[key], TEST_RAW[key] = np.array(pred), raw
    print(f"{NICE[key]} ({v}): test macro-F1 {TEST_RES[key]['macro_f1']:.4f} | invalid {100 * (np.array(pred) == L.INVALID).mean():.1f} %")
''')

code(r'''
def load_pred(name):
    p = pd.read_csv(C.RESULTS_DIR / "predictions" / f"{name}.csv").set_index("row_id").loc[test.row_id]
    return p.pred.astype(str).values
REF = {"Legal-BERT (fine-tuned)": load_pred("legalbert"), "TF-IDF + LogReg (best classical)": load_pred("tfidf_logreg"),
       "BERT-base (fine-tuned)": load_pred("bert_base")}
ALL = {**REF, **{f"{NICE[k]} ({CHOICE[k]['version']})": TEST_PRED[k] for k in MODELS}}
yt = test.category.values
rows = []
for name, pred in ALL.items():
    m = C.compute_metrics(yt, pred, LABELS, "x", "test", save=False, robust_labels=ROBUST)
    ci = C.bootstrap_ci(yt, pred, test.contract.values, labels=LABELS)
    rows.append({"model": name, "macro-F1": m["macro_f1"], "95% CI": f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]",
                 "macro-F1 (24 robust classes)": m["macro_f1_robust"], "accuracy": m["accuracy"], "weighted-F1": m["weighted_f1"],
                 "invalid %": 100 * float((pred == L.INVALID).mean())})
TEST_TAB = pd.DataFrame(rows); display(TEST_TAB.round(4))
q15, q3 = [f"{NICE[k]} ({CHOICE[k]['version']})" for k in MODELS]
pairs = [("Legal-BERT (fine-tuned)", q15), ("Legal-BERT (fine-tuned)", q3), (q3, q15), ("TF-IDF + LogReg (best classical)", q3)]
PB = []
for a, b in pairs:
    r = C.paired_bootstrap(yt, ALL[a], ALL[b], test.contract.values, labels=LABELS)
    PB.append({"A": a, "B": b, "macro-F1 A - B": r["diff"], "95% CI": f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]",
               "share of resamples A not better": r["p_a_not_better"], **{k: r[k] for k in ["ci_low", "ci_high"]}})
PB = pd.DataFrame(PB); display(PB.drop(columns=["ci_low", "ci_high"]).round(4))
TEST_TAB.to_csv(C.RESULTS_DIR / "llm_test_comparison.csv", index=False)
PB.to_csv(C.RESULTS_DIR / "llm_paired_bootstrap.csv", index=False)
''')

interp("test")

# =====================================================================================
md(r"""
## 8. Which classes do the prompted models handle, and which do they confuse?
""")

code(r'''
pc = pd.DataFrame(index=LABELS)
pc["test clauses"] = test.category.value_counts().reindex(LABELS)
pc["Legal-BERT F1"] = pd.Series(C.compute_metrics(yt, REF["Legal-BERT (fine-tuned)"], LABELS, "x", "test", save=False)["per_class"]).map(lambda d: d["f1"])
for key in MODELS:
    m = TEST_RES[key]["per_class"]
    pc[f"{NICE[key]} F1"] = [m[l]["f1"] for l in LABELS]
    pc[f"{NICE[key]} recall"] = [m[l]["recall"] for l in LABELS]
    ci = C.per_class_bootstrap_ci(yt, TEST_PRED[key], test.contract.values, LABELS)
    pc[f"{NICE[key]} F1 95% CI"] = [f"[{ci.loc[l, 'ci_low']:.2f}, {ci.loc[l, 'ci_high']:.2f}]" if ci.loc[l, "ci_low"] == ci.loc[l, "ci_low"]
                                    else "not estimable" for l in LABELS]
best_key = max(MODELS, key=lambda k: TEST_RES[k]["macro_f1"])
pc = pc.sort_values(f"{NICE[best_key]} F1")
pc.to_csv(C.RESULTS_DIR / "llm_per_class.csv")
display(pc.round(3))
''')

code(r'''
SHORT = {"Post-Termination Services": "Post-Term. Services", "Termination For Convenience": "Termination For Conv.",
         "Ip Ownership Assignment": "IP Ownership Assign.", "Revenue/Profit Sharing": "Revenue/Profit Shar.",
         "Competitive Restriction Exception": "Compet. Restr. Exception", "Non-Transferable License": "Non-Transf. License",
         "No-Solicit Of Employees": "No-Solicit Employees", "Joint Ip Ownership": "Joint IP Ownership"}
short = [SHORT.get(l, l) for l in LABELS] + ["INVALID"]
def plot_cm_llm(pred, title, number, fname):
    cmx = pd.crosstab(pd.Categorical(yt, LABELS), pd.Categorical(pred, LABELS + [L.INVALID]), dropna=False).values
    mat = cmx / cmx.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(mat, cmap=PL.BLUES, vmin=0, vmax=1)
    ax.set_xticks(range(len(short))); ax.set_xticklabels(short, rotation=65, ha="right", fontsize=8)
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(short[:-1], fontsize=8)
    ax.set_xlabel("Predicted class (last column = invalid answer)"); ax.set_ylabel("True class"); ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] > 0:
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if mat[i, j] > 0.55 else PL.INK)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); ax.set_title(title, fontsize=11)
    PL.savefig(fig, number, fname); plt.show()
    return pd.DataFrame(cmx, index=LABELS, columns=LABELS + [L.INVALID])
CM = {}
for key, num in [("qwen25_1p5b", 27), ("qwen25_3b", 28)]:
    CM[key] = plot_cm_llm(TEST_PRED[key], f"Figure {num}. Confusion matrix of {NICE[key]} ({CHOICE[key]['version']}) on test, "
                          "normalised by true class (diagonal = recall)", num, f"cm_{key}_normalised")
''')

code(r'''
# Most frequent confusions of each model on test, and where the predictions pile up
for key in MODELS:
    cmx = CM[key]
    pairs_ = [(t, p, int(cmx.loc[t, p])) for t in LABELS for p in LABELS + [L.INVALID] if t != p and cmx.loc[t, p] > 0]
    top = pd.DataFrame(pairs_, columns=["true", "predicted", "clauses"]).sort_values("clauses", ascending=False).head(8)
    print(f"--- {NICE[key]} ({CHOICE[key]['version']}): most frequent confusions ---"); display(top.reset_index(drop=True))
    over = pd.Series(TEST_PRED[key]).value_counts().head(5)
    print("Most predicted labels:", {k: f"{v} predicted / {int((yt == k).sum())} true" for k, v in over.items()})
''')

code(r'''
# How much do descriptions / few-shot examples help per class? (validation, both models: F1 per version)
rows = []
for key in MODELS:
    for v in VERSIONS:
        per = C.compute_metrics(val.category, VAL_PRED[(key, v)], LABELS, "x", "validation", save=False)["per_class"]
        rows += [{"model": NICE[key], "version": v, "class": l, "F1": per[l]["f1"]} for l in LABELS]
pv = pd.DataFrame(rows).pivot_table(index="class", columns=["model", "version"], values="F1")
JARGON = ["Rofr/Rofo/Rofn", "Competitive Restriction Exception", "Volume Restriction", "Covenant Not To Sue", "Post-Termination Services",
          "Liquidated Damages", "Change Of Control", "Governing Law", "Insurance", "Audit Rights"]
display(pv.loc[JARGON].round(3))
pv.to_csv(C.RESULTS_DIR / "llm_per_class_by_version_validation.csv")
''')

interp("perclass")

# =====================================================================================
md(r"""
## 9. Latency, throughput, memory and cost per 1,000 clauses
All timings were measured on this machine (RTX 5080 16 GB for GPU models; the CPU for TF-IDF). **Latency** = one clause at a
time (batch size 1), on the same 100 validation clauses for every model (median and 95th percentile). **Throughput** = the
batched test run. **Cost per 1,000 clauses** = GPU (or CPU) seconds per clause × 1,000 / 3,600 × an hourly price.

**Price assumption (stated, not measured).** Primary source: Amazon Web Services, *Amazon EC2 On-Demand Pricing*
(https://aws.amazon.com/ec2/pricing/on-demand/), Linux, US East (N. Virginia), read on **2026-10-22** (values taken from the
page's published price file, publication date 2026-10-15): **g6.xlarge** (1 × NVIDIA L4, 4 vCPU, 16 GiB) **$0.8048 per hour**
for every GPU model, and **c7i.xlarge** (4 vCPU, 8 GiB) **$0.1785 per hour** for TF-IDF. Cross-checks, also read on 2026-10-22:
Holori AWS price calculator (https://calculator.holori.com/aws/ec2/g6.xlarge/us-east-1 and …/c7i.xlarge/us-east-1, pages
"last updated Oct 21, 2026") and DoiT (https://www.doit.com/compute/spot/us-east-1/g6.xlarge) give the same figures. All
sources are recorded in `results/llm_price_sources.json`. API fee = 0 (local models). **Assumption:** the L4 is a different
GPU from the RTX 5080 we measured on, so the cost is an order-of-magnitude comparison on **one common basis**, not a quote.
""")

code(r'''
PRICES = json.load(open(C.RESULTS_DIR / "llm_price_sources.json"))
GPU_PRICE, CPU_PRICE = PRICES["primary"]["g6.xlarge"]["usd_per_hour"], PRICES["primary"]["c7i.xlarge"]["usd_per_hour"]
PRICE_SOURCE = {"primary": PRICES["primary"]["url"], "gpu_instance": "g6.xlarge (1x NVIDIA L4)", "cpu_instance": "c7i.xlarge",
                "read_on": PRICES["read_on"], "cross_checks": [c["url"] for c in PRICES["cross_checks"]]}
print(f"Prices (AWS on-demand, us-east-1, read {PRICES['read_on']}): GPU ${GPU_PRICE}/h, CPU ${CPU_PRICE}/h")
LAT_SAMPLE = val.sample(100, random_state=42)          # the same 100 clauses as scripts/run_llm.py --latency
for key in MODELS:
    v = CHOICE[key]["version"]
    if not (LLM_DIR / f"latency_{key}_{v}.json").exists():
        subprocess.run([sys.executable, "scripts/run_llm.py", "--latency", key, v], check=True)
LAT = {key: json.load(open(LLM_DIR / f"latency_{key}_{CHOICE[key]['version']}.json")) for key in MODELS}

# Legal-BERT (Task 1 test model): batch-1 latency on the same clauses + batched throughput from the logged test inference
from transformers import AutoModelForSequenceClassification
from src.preprocess import preprocess_for_transformer
from src.finetune import RECIPE as FT_RECIPE
lb_dir = ROOT / "models" / "legalbert_lr5e-05_s42" / "best"
budget = pd.read_csv(C.RESULTS_DIR / "compute_budget.csv")
lb_tp = budget[(budget.model == "legalbert") & (budget.stage == "inference_test_gpu")].seconds.iloc[0] / len(test)
if not (LLM_DIR / "latency_legalbert.json").exists() and lb_dir.exists() and torch.cuda.is_available():   # measured once
    lb_tok = AutoTokenizer.from_pretrained(lb_dir); lb = AutoModelForSequenceClassification.from_pretrained(lb_dir).to("cuda").eval()
    torch.cuda.reset_peak_memory_stats(); secs = []
    with torch.no_grad():
        for i, t in enumerate([LAT_SAMPLE.text.iloc[0]] + list(LAT_SAMPLE.text)):
            enc = lb_tok(preprocess_for_transformer(t), truncation=True, max_length=FT_RECIPE["max_length"], return_tensors="pt").to("cuda")
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lb(**enc).logits.argmax(-1).item()
            torch.cuda.synchronize()
            if i:
                secs.append(time.perf_counter() - t0)
    LAT_LB = {"median_ms": 1000 * float(np.median(secs)), "p95_ms": 1000 * float(np.percentile(secs, 95)),
              "peak_gpu_mem_gb_batch1": torch.cuda.max_memory_allocated() / 1e9}
    del lb; torch.cuda.empty_cache()
    json.dump(LAT_LB, open(LLM_DIR / "latency_legalbert.json", "w"), indent=2)
else:
    LAT_LB = json.load(open(LLM_DIR / "latency_legalbert.json"))

# TF-IDF + LogReg (best classical): rebuilt with the selected configuration, checked against its saved test predictions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from src import preprocess as P
cl_cfg = json.loads(pd.read_csv(C.RESULTS_DIR / "test_eval_log.csv").set_index("model").loc["tfidf_logreg", "config"])
pipe = make_pipeline(TfidfVectorizer(analyzer="char_wb", preprocessor=lambda s: P.clean_text(s, cl_cfg["preprocessing"]), lowercase=False,
                                     ngram_range=(3, 6), min_df=int(cl_cfg["min_df"]), max_df=float(cl_cfg["max_df"]), sublinear_tf=True),
                     LogisticRegression(C=float(cl_cfg["C_or_alpha"]), class_weight="balanced", max_iter=3000))
pipe.fit(train.text, train.category)
t0 = time.perf_counter(); cl_test_pred = pipe.predict(test.text); tp_now = (time.perf_counter() - t0) / len(test)
print("Rebuilt TF-IDF + LogReg reproduces its saved test predictions:", bool((cl_test_pred == REF["TF-IDF + LogReg (best classical)"]).all()))
cl_file = LLM_DIR / "latency_tfidf_logreg.json"
if not cl_file.exists():                          # measured once, then re-used so a re-run does not change the reported numbers
    secs = []
    for t in LAT_SAMPLE.text:
        t0 = time.perf_counter(); pipe.predict([t]); secs.append(time.perf_counter() - t0)
    json.dump({"median_ms": 1000 * float(np.median(secs)), "p95_ms": 1000 * float(np.percentile(secs, 95)),
               "throughput_seconds_per_clause": tp_now, "device": "CPU"}, open(cl_file, "w"), indent=2)
LAT_CL = json.load(open(cl_file)); cl_tp = LAT_CL["throughput_seconds_per_clause"]
''')

code(r'''
meta = json.load(open(LLM_DIR / "run_meta.json"))
train_s = {r.model: r.seconds for r in budget[budget.stage == "train_final"].drop_duplicates("model").itertuples()}
rows = []
def add(model, kind, tput_s, lat, peak, train_seconds, price, f1, params):
    # headline: batched cost; batch-1 cost and latency are extra columns
    rows.append({"model": model, "device": kind, "macro-F1 (test)": f1,
                 "cost per 1,000 clauses USD (batched)": tput_s * 1000 / 3600 * price,
                 "throughput clauses/s (batched)": 1 / tput_s,
                 "cost per 1,000 clauses USD (batch 1)": lat["median_ms"] / 1000 * 1000 / 3600 * price,
                 "latency median ms (batch 1)": lat["median_ms"], "latency p95 ms (batch 1)": lat["p95_ms"],
                 "peak GPU memory GB": peak, "training seconds (one-off)": train_seconds, "parameters": params})
f1 = dict(zip(TEST_TAB.model, TEST_TAB["macro-F1"]))
add("TF-IDF + LogReg (best classical)", "CPU", cl_tp, LAT_CL, 0.0, train_s.get("tfidf_logreg"), CPU_PRICE, f1["TF-IDF + LogReg (best classical)"], "69,247 features x 27")
add("Legal-BERT (fine-tuned)", "GPU", lb_tp, LAT_LB, LAT_LB["peak_gpu_mem_gb_batch1"], train_s.get("legalbert"), GPU_PRICE,
    f1["Legal-BERT (fine-tuned)"], "110 M")
for key in MODELS:
    v = CHOICE[key]["version"]; raw = TEST_RAW[key]
    add(f"{NICE[key]} ({v})", "GPU", gpu_seconds(raw) / len(raw), LAT[key], meta[f"{key}_{v}_test"]["peak_gpu_mem_gb"], 0.0, GPU_PRICE,
        f1[f"{NICE[key]} ({v})"], f"{meta[f'{key}_{v}_test']['n_params'] / 1e9:.2f} B")
COST = pd.DataFrame(rows)
with pd.option_context("display.float_format", lambda x: f"{x:.6g}"):
    display(COST)
json.dump({"price_assumption": {"gpu_usd_per_hour": GPU_PRICE, "cpu_usd_per_hour": CPU_PRICE, **PRICE_SOURCE,
                                "note": "timings measured on an RTX 5080 (GPU) / local CPU; price of a comparable cloud instance applied"},
           "api_fee": 0, "rows": COST.to_dict(orient="records")}, open(C.RESULTS_DIR / "llm_cost.json", "w"), indent=2, default=float)
# log the latency measurements once in the compute budget
if not (budget.stage == "latency_batch1").any():
    new = [{"timestamp": pd.Timestamp.now().isoformat(timespec="seconds"), "model": r["model"], "stage": "latency_batch1",
            "seconds": round(r["latency median ms (batch 1)"] / 1000 * 100, 3), "device": r["device"],
            "notes": f"100 validation clauses at batch 1 (approx. = median x 100); median {r['latency median ms (batch 1)']:.1f} ms, "
                     f"p95 {r['latency p95 ms (batch 1)']:.1f} ms"} for r in rows]
    pd.concat([budget, pd.DataFrame(new)], ignore_index=True).to_csv(C.RESULTS_DIR / "compute_budget.csv", index=False)
''')

code(r'''
fig, ax = plt.subplots(figsize=(8.5, 4.6))
cols = [PL.AQUA, PL.INK, PL.BLUE, PL.ORANGE]
for (_, r), col in zip(COST.iterrows(), cols):
    ax.scatter(r["cost per 1,000 clauses USD (batched)"], r["macro-F1 (test)"], s=90, color=col, zorder=3)
    ax.annotate(f"{r['model']}\n{r['latency median ms (batch 1)']:.0f} ms/clause at batch 1", (r["cost per 1,000 clauses USD (batched)"], r["macro-F1 (test)"]),
                textcoords="offset points", xytext=(8, -4), fontsize=8)
ax.set_xscale("log"); ax.set_xlim(COST["cost per 1,000 clauses USD (batched)"].min() / 2, COST["cost per 1,000 clauses USD (batched)"].max() * 12)
ax.set_xlabel("cost per 1,000 clauses, USD (batched; log scale; price assumption in the text)")
ax.set_ylabel("test macro-F1 (27 classes)"); ax.set_ylim(0, 1)
ax.set_title("Figure 29. Test macro-F1 vs inference cost per 1,000 clauses", fontsize=11)
PL.savefig(fig, 29, "llm_cost_vs_f1"); plt.show()
''')

interp("cost")

# =====================================================================================
md(r"""
## 10. When would a prompted model be preferred over the fine-tuned one?
""")
interp("discussion")

# =====================================================================================
md(r"""
## 11. DEMO — classify clauses with the chosen prompt of each model
Four clauses: one written for the demo (paste any clause into `MY_CLAUSE`) and three validation clauses. Each model is
loaded once, answers with its chosen prompt version, and the answer is converted by `L.parse_label`. On Colab the two
models are downloaded from Hugging Face the first time (about 9 GB together). Without a GPU the cell shows the saved answers
of the real run instead and says so.
""")

code(r'''
# DEMO setup: the chosen prompt versions and the few-shot examples, loaded from saved files (lets this group run on its own)
CHOICE = json.load(open(C.RESULTS_DIR / "llm_choice.json"))
EX = pd.read_csv(C.RESULTS_DIR / "llm_fewshot_examples.csv"); EXAMPLES = dict(zip(EX.category, EX.text))
''')

code(r'''
MY_CLAUSE = ("Neither party shall, during the term of this Agreement and for one year thereafter, directly or indirectly "
             "solicit for employment any employee of the other party without its prior written consent.")
demo = pd.concat([pd.DataFrame([{"text": MY_CLAUSE, "category": "(written for the demo)"}]),
                  val.groupby("category").head(1).sample(3, random_state=7)[["text", "category"]]], ignore_index=True)
if torch.cuda.is_available():
    for key in MODELS:
        v = CHOICE[key]["version"]; t0 = time.perf_counter()
        r = L.Runner(MODELS[key])
        outs = r.generate([r.render(L.build_messages(v, t, LABELS, EXAMPLES)) for t in demo.text])
        demo[f"{NICE[key]} ({v})"] = [f"{L.parse_label(o, LABELS)[0]}   <- '{o.strip()[:40]}'" for o in outs]
        for t, ans in zip(demo.text, demo[f"{NICE[key]} ({v})"]):
            C.demo_cache_put(f"llm_{key}_{v}", t, ans)     # lets the demo show this real output without a GPU
        r.free()
        print(f"{NICE[key]}: loaded and answered 4 clauses in {time.perf_counter() - t0:.1f} s")
    pd.set_option("display.max_colwidth", 90)
    display(demo.assign(text=demo.text.str[:160] + "..."))
else:
    print("No GPU available: showing the SAVED answers of the real run (results/demo_cache.json); "
          "on Colab choose Runtime -> Change runtime type -> GPU to run the models live.")
    for key in MODELS:
        v = CHOICE[key]["version"]
        demo[f"{NICE[key]} ({v}, saved)"] = [C.demo_cache_get(f"llm_{key}_{v}", t) or "n/a (not run before)" for t in demo.text]
    pd.set_option("display.max_colwidth", 90)
    display(demo.assign(text=demo.text.str[:160] + "..."))
''')

# =====================================================================================
md(r"""
## 12. Conclusion and limitations of the prompted-LLM comparison
""")
interp("conclusion")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
OUT.parent.mkdir(exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
