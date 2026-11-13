"""Generate notebooks/STUDENTID_Task2.ipynb (Task 2: clause extraction, Option A sequence labelling).

    python scripts/notebook_builders/build_task2.py      (outputs are cleared; execute the notebook afterwards)
"""
from pathlib import Path

import sys
import nbformat as nbf
sys.path.insert(0, str(Path(__file__).parent))
from colab_cell import COLAB_MD, COLAB_CODE  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks" / "STUDENTID_Task2.ipynb"
INTERP = {}
interp_file = Path(__file__).with_name("task2_interp.py")
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
## Notebook 2 of 3: Task 2 — Clause extraction from whole contracts (Option A, sequence labelling)

**Student:** see the setup cell (reads `project_info.json`) · **Course:** IT9002 Natural Language Processing

**Dataset:** Contract Understanding Atticus Dataset (CUAD) v1 — https://www.atticusprojectai.org/cuad
(data and code: https://github.com/TheAtticusProject/cuad), licence CC BY 4.0.

| Section | Content | Rubric |
|---|---|---|
| 1 | Purpose and link to Task 1 | Task 2 |
| 2 | Data in the required format: windows, token labels, decoder, **oracle check** | Task 2 data; Part A preprocessing |
| 3 | Evaluation code (entity-level P/R/F1, CUAD-style AUPR) and its tests | Task 2 evaluation; Part C metrics |
| 4 | Baseline: TF-IDF segment classifier | Task 2 model |
| 5 | Main model: Legal-BERT token tagger | Task 2 model |
| 6 | Test evaluation (once per model) | Task 2 evaluation; objective O4 |
| 7 | Published comparison with the CUAD paper | Part C |
| 8 | Representative outputs and error analysis | Task 2 outputs; Part C error analysis |
| 9 | DEMO: extract clauses from any contract text | video |
| 10 | Conclusion and limitations | Part C |

### How to run
Locally: open the notebook from the project folder with the `.venv` environment and *Run All*. On Google Colab the setup
cell downloads the code and data from GitHub. Heavy steps have switches (`RETRAIN`, `RUN_SEARCH`): with `False` the saved
results of the real run are used, so the notebook re-runs in minutes; the tagger's test predictions need the saved model
weights (`models/`, not in the repository) — without them the saved predicted spans are loaded and the notebook says so.
""")

md(COLAB_MD)
code(COLAB_CODE)

md(r"""
## 0. Setup
""")

code(r'''
import sys, os, json, platform, subprocess, importlib, re, time, warnings, contextlib, io
from pathlib import Path
IN_COLAB = "google.colab" in sys.modules or os.environ.get("IT9002_SIMULATE_COLAB") == "1"
# The project folder = the first folder, going upwards, that contains project_info.json (works from notebooks/, from the
# project folder and on Colab, where the cell above has moved into the project).
ROOT = next(d for d in [Path.cwd(), *Path.cwd().parents] if (d / "project_info.json").exists())
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", message="IProgress not found")

from src import common as C, task2_data as D, span_metrics as M, task2_baseline as B, plotting as PL
for mod in (C, D, M, B, PL):
    importlib.reload(mod)
import numpy as np, pandas as pd, torch, transformers, sklearn
import matplotlib.pyplot as plt
transformers.logging.set_verbosity_error()
from transformers import AutoTokenizer
C.set_seed(42); PL.setup_style()
INFO = C.load_project_info()
C.register_needs_models()   # defines %%needs_models: cells that need models/ (not in the ZIP) skip cleanly if it is missing
print(f"Student: {INFO['STUDENT_NAME']} | ID: {INFO['STUDENT_ID']} | {INFO['COURSE_CODE']} {INFO['COURSE_NAME']}")
print("Dataset:", INFO["DATASET_URL"], "|", INFO["DATASET_CODE_URL"])
print(f"Python {platform.python_version()} | torch {torch.__version__} | transformers {transformers.__version__} | "
      f"scikit-learn {sklearn.__version__} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

C.ensure_data(INFO["GITHUB_URL"])   # frozen CSVs + raw CUAD: present, from GitHub, or rebuilt; checksums verified
cs, spans = C.load_contract_split(), C.load_task2_spans()
CATS = sorted(spans.category.unique()); K = len(CATS); CIDX = {c: i for i, c in enumerate(CATS)}
SPLIT = {s: list(cs[cs.split == s].contract) for s in ["train", "validation", "test"]}
TEXT = {c: C.load_contract_text(c) for c in cs.contract}
text_of = lambda c, s, e: TEXT[c][s:e]
CKPT = "nlpaueb/legal-bert-base-uncased"
tok = AutoTokenizer.from_pretrained(CKPT)
corpus = D.load_corpus(tok, ROOT / "models" / "task2_cache" / "corpus_legalbert.pkl", list(cs.contract), C.load_contract_text)
print(f"{len(cs)} contracts | {len(spans):,} (span, category) annotations | {K} categories | fast tokenizer: {tok.is_fast}")
''')

# =====================================================================================
md(r"""
## 1. Purpose and link to Task 1
**Task 1** classifies a clause that someone has already located ("what type is this clause?"). **Task 2** finds the clauses
inside a whole contract ("where are the clauses of each type?"). A reviewer needs both: first find the risk clauses, then
check them. Task 2 uses the **same corpus, the same 27 categories and the same contract split** (`contract_split.csv`) as
Task 1, so no contract seen in training by one task is tested by the other. Unlike Task 1, it keeps the clauses that carry
several categories at once (extraction is multi-label by nature), and it is evaluated on **all 102 official CUAD test
contracts**, including the 9 that contain no clause of the 27 categories (there a model can only make false positives).
Part F later chains the tasks: extract (Task 2) → confirm the type (Task 1) → flag unusual wording.

Task 2 is **Option A (sequence labelling)** of the assessment brief; the approved proposal named Option D (topic modelling).
The change follows the tutor's feedback (deviation D2 in the report).
""")

# =====================================================================================
md(r"""
## 2. Data in the required format

### 2.1 Contracts and gold spans
Each annotation in `task2_spans.csv` is a character range in the original contract text. The cell checks that every
range reproduces the annotated text exactly (so no preprocessing has shifted any offset) and summarises the splits.
""")

code(r'''
assert all(TEXT[r.contract][r.start:r.end] == r.text for r in spans.itertuples()), "an offset does not match the text"
print(f"All {len(spans):,} gold spans match the contract text at their character offsets.")
summary = pd.DataFrame({
    "contracts": cs.split.value_counts(),
    "contracts with >= 1 span": spans.groupby("split").contract.nunique(),
    "annotations": spans.split.value_counts(),
    "distinct spans": spans.drop_duplicates(["contract", "start", "end"]).groupby("split").size(),
    "multi-category annotations": spans[spans.n_categories_on_span > 1].split.value_counts(),
}).loc[["train", "validation", "test"]]
summary.loc["total"] = summary.sum()
display(summary)
SPAN_FREE_TEST = sorted(set(SPLIT["test"]) - set(spans[spans.split == "test"].contract))
print(f"Test contracts without any span of the 27 categories: {len(SPAN_FREE_TEST)}")
span_chars = spans.end - spans.start
print(f"Span length in characters: median {span_chars.median():.0f}, 95th percentile {span_chars.quantile(.95):.0f}, max {span_chars.max():,}")
''')

md(r"""
### 2.2 Tokenisation and sliding windows
The Legal-BERT *fast* tokenizer returns, for every word-piece, its character range in the original text (offset mapping),
so labels can be placed exactly without changing the text. Contracts are far longer than the model's 512-token limit, so
they are cut into overlapping windows. We compare four window / stride settings on how many windows they create, how many
gold spans are longer than a window, how many spans fit completely inside at least one window, and the measured training
speed (20-step benchmark, `results/task2_benchmark.json`).
""")

code(r'''
ntok = pd.Series({c: len(v["ids"]) for c, v in corpus.items()})
print(f"Word-piece tokens per contract: median {ntok.median():,.0f}, 95th percentile {ntok.quantile(.95):,.0f}, max {ntok.max():,}")
span_tok = []
for r in spans.drop_duplicates(["contract", "start", "end"]).itertuples():
    rng = D.span_token_range(corpus[r.contract]["offsets"], r.start, r.end)
    span_tok.append((r.contract, rng[0], rng[1]))
span_tok = pd.DataFrame(span_tok, columns=["contract", "t0", "t1"]); span_tok["n"] = span_tok.t1 - span_tok.t0 + 1
bench = json.load(open(C.RESULTS_DIR / "task2_benchmark.json"))
rows = []
for win, stride in [(128, 64), (256, 128), (384, 192), (510, 255)]:
    n_win = {s: sum(len(D.make_windows(ntok[c], win, stride)) for c in SPLIT[s]) for s in SPLIT}
    inside = []   # a span is inside a window iff the last window starting at or before its first token ends after it
    for c, g in span_tok.groupby("contract", sort=False):
        w = np.array(D.make_windows(int(ntok[c]), win, stride))
        i = np.searchsorted(w[:, 0], g.t0.values, side="right") - 1
        inside.append((i >= 0) & (g.t1.values < w[np.maximum(i, 0), 1]))
    share_in = np.concatenate(inside).mean()
    b = bench[f"{win}/{stride}"]
    rows.append({"window/stride": f"{win}/{stride}", **{f"windows {s}": n_win[s] for s in SPLIT},
                 "spans longer than a window %": 100 * (span_tok.n > win).mean(),
                 "spans fully inside >= 1 window %": 100 * share_in,
                 "ms per step (batch 16)": b["ms_per_step_batch16"], "projected minutes per epoch": b["projected_epoch_minutes"],
                 "peak GPU GB": b["peak_gpu_mem_gb"]})
win_tab = pd.DataFrame(rows).set_index("window/stride")
win_tab.to_csv(C.RESULTS_DIR / "task2_window_choice.csv")
display(win_tab.round(2))
print(f"Gold span length in word-pieces: median {span_tok.n.median():.0f}, 95th percentile {span_tok.n.quantile(.95):.0f}, max {span_tok.n.max():,}")
''')

interp("windows")

md(r"""
### 2.3 Label format: INSIDE and BEGIN for each category (54 targets per token)
For every word-piece and every category there are two yes/no targets: **INSIDE** (the word-piece overlaps a gold span of
that category by at least one character) and **BEGIN** (it is the first word-piece of such a span). Because a span can carry
several categories, several targets can be "yes" at once (multi-label). [CLS], [SEP] and padding are masked out of the
loss. The BEGIN targets are needed because clauses of the same category often **touch** each other: with only INSIDE
targets, two touching clauses would look like one long clause. The cell counts how often that happens.
""")

code(r'''
d = spans.drop_duplicates(["contract", "category", "start", "end"]).sort_values(["contract", "category", "start"])
d["prev_end"] = d.groupby(["contract", "category"]).end.shift()
nb = d.dropna(subset=["prev_end"])
touch = (nb.start - nb.prev_end) <= 1
print(f"Neighbouring same-category span pairs: {len(nb):,}; touching or overlapping (gap <= 1 character): {touch.sum():,} "
      f"({100 * touch.mean():.1f} %)")

LABELS = {c: D.token_labels(corpus[c]["offsets"], [(CIDX[r.category], r.start, r.end) for r in spans[spans.contract == c].itertuples()], K)
          for c in cs.contract}
pos_share = np.concatenate([LABELS[c] for c in SPLIT["train"]])[:, :K].mean()
print(f"Share of (token, category) INSIDE targets that are positive in train: {100 * pos_share:.2f} % -> heavy imbalance")

# A real example with two touching spans of one category: text and token-label matrix
ex = nb[touch].iloc[0]
c0, cat0 = ex.contract, ex.category
s0, e0 = int(ex.prev_end) - 160, int(ex.start) + 160
offs = corpus[c0]["offsets"]; tsel = np.flatnonzero((offs[:, 1] > s0) & (offs[:, 0] < e0))
toks = tok.convert_ids_to_tokens([corpus[c0]["ids"][i] for i in tsel])
used = [k for k in range(K) if LABELS[c0][tsel, k].any()]
print(f"\nContract: {c0[:70]}\nCategory with two touching spans: {cat0}\nExcerpt: {TEXT[c0][s0:e0]!r}")
mat = np.vstack([LABELS[c0][tsel][:, used].T, LABELS[c0][tsel][:, [K + k for k in used]].T])
fig, ax = plt.subplots(figsize=(13, 0.5 * len(mat) + 2))
ax.imshow(mat, aspect="auto", cmap=PL.BLUES, vmin=0, vmax=1)
ax.set_yticks(range(len(mat))); ax.set_yticklabels([f"INSIDE {CATS[k]}" for k in used] + [f"BEGIN {CATS[k]}" for k in used], fontsize=8)
step = max(1, len(toks) // 40)
ax.set_xticks(range(0, len(toks), step)); ax.set_xticklabels(toks[::step], rotation=70, fontsize=7); ax.grid(False)
ax.set_title(f"Figure 19. Token-label matrix for a contract excerpt with two touching '{cat0}' spans", fontsize=11)
PL.savefig(fig, 19, "t2_label_matrix"); plt.show()
''')

interp("labels")

md(r"""
### 2.4 Decoder: from per-token probabilities back to character spans
`D.decode` turns the model's per-token probabilities into spans, separately for each category: a span **starts** at a
token whose INSIDE probability is ≥ θ_inside when the previous token is outside, or when its BEGIN probability is ≥ θ_begin;
it **continues** while INSIDE ≥ θ_inside and **ends** before the next start. Its confidence is the mean INSIDE probability
of its tokens; spans shorter than `min_tokens` are dropped. The thresholds are tuned on validation (section 5). Probabilities
from overlapping windows are averaged per token before decoding (`D.average_window_probs`).
""")

md(r"""
### 2.5 ORACLE CHECK: is the label format lossless enough? (before any training)
We feed the **gold** token labels (probability 1 for every positive target) through the decoder and score the resulting
spans against the gold spans with the evaluation code of section 3 — once with INSIDE + BEGIN and once with INSIDE only.
This is the ceiling of the formulation: a perfect tagger could not score higher. It also tests the evaluation code
(a perfect prediction must score close to 1).
""")

code(r'''
SNAPS = {c: (lambda o: (lambda s, e: D.span_token_range(o, s, e)))(corpus[c]["offsets"]) for c in cs.contract}

def spans_df(rows):
    return pd.DataFrame(rows, columns=["contract", "category", "start", "end", "confidence"])

def decode_contracts(probs_of, contracts, ti=0.5, tb=0.5, mt=1, use_begin=True):
    rows = []
    for c in contracts:
        for k, s, e, conf in D.decode(probs_of(c), corpus[c]["offsets"], K, ti, tb, mt, use_begin):
            rows.append((c, CATS[k], s, e, conf))
    return spans_df(rows)

oracle = {}
for split in ["validation", "test"]:
    gold = spans[spans.split == split]
    for use_begin in [True, False]:
        pred = decode_contracts(lambda c: LABELS[c].astype(np.float32), SPLIT[split], use_begin=use_begin)
        for mode in ["iou", "exact"]:
            s = M.summarise(M.count_table(gold, pred, SPLIT[split], CATS, mode=mode, snaps=SNAPS), CATS)
            oracle[f"{split} | {'INSIDE+BEGIN' if use_begin else 'INSIDE only'} | {'IoU>=0.5' if mode == 'iou' else 'exact'}"] = \
                {"macro_f1": s["macro_f1"], "micro_f1": s["micro_f1"]}
oracle_tab = pd.DataFrame(oracle).T
json.dump(oracle, open(C.RESULTS_DIR / "task2_oracle.json", "w"), indent=2)
display(oracle_tab.round(4))
''')

interp("oracle")

# =====================================================================================
md(r"""
## 3. Evaluation code (`src/span_metrics.py`)
**Our entity-level metrics (primary).** For each (contract, category), predicted spans are matched **one-to-one** to gold
spans, greedily in order of confidence. A pair matches at **IoU ≥ 0.5** (character-level Jaccard of the two ranges, CUAD's
threshold) or, for the strict variant, when both cover **exactly the same word-pieces**. Unmatched predictions are false
positives — also in contracts that have no span of that category — and unmatched gold spans are false negatives. We report
precision, recall and F1 per category, micro (pooled counts) and **macro over the 27 categories (primary: macro-F1 at IoU
0.5)**, plus token-level F1 on the INSIDE targets.

**CUAD-style metrics** replicate the official `evaluate.py` (read from https://github.com/TheAtticusProject/cuad): word-set
Jaccard ≥ 0.5 on the span *texts* (punctuation `. , ; :` removed, lower-cased); a gold span is a TP if *any* prediction matches
it (not one-to-one); a prediction matching no gold span is an FP; counts pooled over all (contract, category) pairs; the
confidence threshold is swept from 0.99 to 0; precision is made monotone; **AUPR** is the area under that curve and
**precision at 80 % / 90 % recall** is read from it.

**Differences from CUAD's evaluation, stated explicitly:** (1) our primary metric is one-to-one entity F1 at a tuned threshold,
CUAD reports AUPR over a threshold sweep; (2) our IoU is character-based, CUAD's Jaccard is word-set based (we compute both);
(3) CUAD evaluates 41 categories, we evaluate 27; (4) CUAD's confidence is a QA answer probability, ours is the mean INSIDE
probability of a decoded span.

**Uncertainty:** contract-level bootstrap (1,000 resamples of whole contracts); in each resample macro-F1 is averaged over
the categories with at least one gold span in that resample (the same rule as in Task 1).

The unit tests below check the metrics and the decoder on tiny hand-made examples.
""")

code(r'''
sys.path.insert(0, str(ROOT / "tests"))
import test_task2
importlib.reload(test_task2)
for name in [n for n in dir(test_task2) if n.startswith("test_")]:
    getattr(test_task2, name)()
    print("passed:", name)
''')

# =====================================================================================
md(r"""
## 4. Baseline: TF-IDF segment classifier (`t2_tfidf_segments`)
Each contract is cut into **segments** with a documented rule (`src/task2_baseline.py`: line breaks; sentence or item ends
followed by a new sentence; inline enumerations such as "(a) "). A segment is labelled positive for a category if at least
50 % of its characters lie inside a gold span of that category (multi-label). One TF-IDF (word 1–2-grams, sublinear tf,
min_df 2) feeds 27 binary Logistic Regressions, one per category. Positive segments that are neighbours are merged into
one span (confidence = mean probability). Hyper-parameters (C, class weights, decision threshold) were searched on
**validation** (`scripts/t2_baseline_search.py`, all settings in `results/task2_baseline_search.csv`).

Its **oracle ceiling** — perfect segment classification, merged the same way — shows how much the segmentation itself costs.
""")

code(r'''
SEGS = {c: B.segment(TEXT[c]) for c in cs.contract}
seg_len = pd.Series([e - s for c in SEGS for s, e in SEGS[c]])
print({s: f"{sum(len(SEGS[c]) for c in SPLIT[s]):,} segments" for s in SPLIT},
      f"| segment length median {seg_len.median():.0f} chars, 95th percentile {seg_len.quantile(.95):.0f}")
SEG_Y = {c: B.segment_labels(SEGS[c], [(CIDX[r.category], r.start, r.end) for r in spans[spans.contract == c].itertuples()], K)
         for c in cs.contract}
seg_oracle = {}
for split in ["validation", "test"]:
    rows = [(c, CATS[k], s, e, conf) for c in SPLIT[split]
            for k, s, e, conf in B.merge_segments(SEGS[c], SEG_Y[c], SEG_Y[c].astype(float))]
    sm = M.summarise(M.count_table(spans[spans.split == split], spans_df(rows), SPLIT[split], CATS), CATS)
    seg_oracle[split] = {"macro_f1": sm["macro_f1"], "micro_f1": sm["micro_f1"]}
json.dump(seg_oracle, open(C.RESULTS_DIR / "task2_segment_oracle.json", "w"), indent=2)
print("Segmentation ORACLE ceiling (macro-F1 / micro-F1 at IoU 0.5):",
      {k: (round(v["macro_f1"], 4), round(v["micro_f1"], 4)) for k, v in seg_oracle.items()})
''')

code(r'''
RUN_SEARCH = False      # True re-runs scripts/t2_baseline_search.py (about 2 min on the CPU)
if RUN_SEARCH:
    subprocess.run([sys.executable, "scripts/t2_baseline_search.py"], check=True)
bsearch = pd.read_csv(C.RESULTS_DIR / "task2_baseline_search.csv", keep_default_na=False)
bchoice = json.load(open(C.RESULTS_DIR / "task2_baseline_choice.json"))
print(f"{len(bsearch)} settings searched on validation; selected: C = {bchoice['C']:g}, class_weight = {bchoice['class_weight']}, "
      f"threshold = {bchoice['threshold']} (validation macro-F1 {bchoice['val_macro_f1']:.4f})")
display(bsearch.sort_values("val_macro_f1", ascending=False).head(10).round(4))
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
for ax, w in zip(axes, ["None", "balanced"]):
    g = bsearch[bsearch.class_weight == w]
    for k, (Cval, gg) in enumerate(g.groupby("C")):
        ax.plot(gg.threshold, gg.val_macro_f1, "o-", markersize=4, linewidth=1.6,
                color=[PL.AQUA, PL.BLUE, PL.VIOLET, PL.ORANGE, PL.RED, PL.GREEN][k], label=f"C = {Cval:g}")
    ax.set_title("no class weights" if w == "None" else "balanced class weights"); ax.set_xlabel("decision threshold")
    ax.legend(fontsize=7.5)
axes[0].set_ylabel("validation macro-F1 @ IoU 0.5")
fig.suptitle("Figure 20. TF-IDF segment baseline: validation search over C, class weights and threshold", fontweight="bold", y=1.03)
PL.savefig(fig, 20, "t2_baseline_search"); plt.show()
''')

code(r'''
# Refit the selected baseline on the TRAIN segments; check that it reproduces the validation score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed

def seg_matrix(contracts):
    texts, index = [], []
    for c in contracts:
        texts += [TEXT[c][s:e] for s, e in SEGS[c]]; index += [(c, s, e) for s, e in SEGS[c]]
    return texts, index

tf_cfg = {**bchoice["tfidf"], "ngram_range": tuple(bchoice["tfidf"]["ngram_range"])}
tr_text, _ = seg_matrix(SPLIT["train"])
Ytr = np.vstack([SEG_Y[c] for c in SPLIT["train"]])
with C.timer("t2_tfidf_segments", "train_final", notes="TF-IDF + 27 one-vs-rest LogReg, selected config, train segments"):
    bvec = TfidfVectorizer(**tf_cfg).fit(tr_text)
    Xtr = bvec.transform(tr_text)
    bmodels = Parallel(n_jobs=14)(delayed(lambda k: LogisticRegression(C=bchoice["C"], class_weight=bchoice["class_weight"],
                                                                       solver="liblinear", max_iter=2000).fit(Xtr, Ytr[:, k]))(k)
                                  for k in range(K))

def baseline_predict(contracts, threshold):
    texts, index = seg_matrix(contracts)
    X = bvec.transform(texts)   # vectorise once, then score with each of the 27 one-vs-rest models
    P = np.column_stack([m.predict_proba(X)[:, 1] for m in bmodels])
    rows, pos = [], 0
    for c in contracts:
        n = len(SEGS[c]); p = P[pos:pos + n]; pos += n
        rows += [(c, CATS[k], s, e, conf) for k, s, e, conf in B.merge_segments(SEGS[c], p >= threshold, p)]
    return spans_df(rows), P

bval_pred, _ = baseline_predict(SPLIT["validation"], bchoice["threshold"])
bval = M.summarise(M.count_table(spans[spans.split == "validation"], bval_pred, SPLIT["validation"], CATS), CATS)
print(f"Baseline validation macro-F1 @ IoU 0.5 after refit: {bval['macro_f1']:.4f} (search: {bchoice['val_macro_f1']:.4f})")
assert abs(bval["macro_f1"] - bchoice["val_macro_f1"]) < 1e-9
''')

interp("baseline")

# =====================================================================================
md(r"""
## 5. Main model: Legal-BERT token tagger (`t2_legalbert_tagger`)
**Model.** The *base* checkpoint `nlpaueb/legal-bert-base-uncased` (not the Task 1 fine-tuned model, which is reused in
Part F) with a token-classification head of **54 sigmoid outputs** (INSIDE + BEGIN × 27 categories), trained with our own
multi-label binary cross-entropy on 256-token windows (stride 128; section 2.2), masked on [CLS]/[SEP]/padding.
Recipe: AdamW (lr 5e-5, weight decay 0.01), 10 % warm-up and linear decay, batch 16, 4 epochs, bf16, gradient clipping 1.0.

**Imbalance.** Positive targets are rare (section 2.3). Two levers: (1) **negative-window sampling** — every window
containing a gold span plus a random 20 % of the span-free windows, re-sampled every epoch; (2) the **loss**, chosen among
three documented variants **on validation**: plain BCE; BCE with a per-output positive weight
`clip(sqrt(#negatives / #positives), 1, 30)`; and focal loss (γ = 2), which down-weights easy tokens.

**Rules fixed before training** (`scripts/train_task2.py`): each variant is trained with seed 42; the epoch and the variant
are chosen by validation macro-F1 at IoU 0.5 decoded with default thresholds (0.5 / 0.5 / 1 token); the chosen variant is
repeated with seeds 43 and 44 (validation mean ± std); the **test model is seed 42** of the chosen variant.
""")

code(r'''
RETRAIN = False     # True runs: python scripts/train_task2.py --plan  (about 25 min on an RTX 5080)
if RETRAIN:
    subprocess.run([sys.executable, "scripts/train_task2.py", "--plan"], check=True)
b256 = bench["256/128"]
print(f"Benchmark (window 256 / stride 128): {b256['ms_per_step_batch16']:.0f} ms per step, {b256['windows_per_epoch']:,} windows "
      f"per epoch -> {b256['projected_epoch_minutes']:.1f} min per epoch; whole plan projected "
      f"{bench['projection_256_128']['projected_training_hours'] * 60:.0f} min (training steps only) -> no cuts needed.")
tsearch = pd.read_csv(C.RESULTS_DIR / "task2_search.csv")
tlog = pd.read_csv(C.RESULTS_DIR / "task2_log.csv")
tchoice = json.load(open(C.RESULTS_DIR / "task2_choice.json"))
display(tsearch[["run_id", "variant", "seed", "best_epoch", "val_macro_f1", "val_micro_f1", "steps_per_epoch",
                 "mean_epoch_train_seconds", "run_seconds", "peak_gpu_mem_gb"]].round(4))
print("Chosen:", tchoice)
''')

code(r'''
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
vcol = {"bce_plain": PL.BLUE, "bce_posw": PL.ORANGE, "focal": PL.AQUA}
ts = tsearch.set_index("run_id")
for rid in tsearch.run_id:
    var, seed, n_ep = ts.loc[rid, "variant"], ts.loc[rid, "seed"], int(ts.loc[rid, "epochs"])
    st = tlog[(tlog.run_id == rid) & (tlog.kind == "step")]; ep = tlog[(tlog.run_id == rid) & (tlog.kind == "epoch")]
    spe = ts.loc[rid, "steps_per_epoch"]
    col = PL.VIOLET if n_ep == 8 else vcol[var]
    ls = "-" if seed == 42 else "--"
    if seed == 42:
        axes[0].plot(st.step / spe, st.train_loss.rolling(10, min_periods=1).mean(), ls, color=col, linewidth=1.6,
                     label=f"{var}, {n_ep} epochs")
    axes[1].plot(ep.epoch, ep.val_macro_f1, ls, marker="o", markersize=4, color=col, linewidth=2 if seed == 42 else 1.2,
                 alpha=1 if seed == 42 else 0.55, label=f"{var}, {n_ep} epochs, seed {seed}")
axes[0].set_yscale("log"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("training loss (10-point moving mean)")
axes[0].set_title("Training loss (seed 42)"); axes[0].legend(fontsize=8)
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation macro-F1 @ IoU 0.5 (default thresholds)")
axes[1].set_title("Validation score per epoch"); axes[1].legend(fontsize=7.5)
fig.suptitle("Figure 21. Legal-BERT tagger: training loss and validation macro-F1 per epoch for the three loss variants and extra seeds",
             fontweight="bold", y=1.03, fontsize=11)
PL.savefig(fig, 21, "t2_tagger_curves"); plt.show()
display(tlog[tlog.kind == "epoch"][["run_id", "epoch", "train_loss", "val_macro_f1", "val_macro_precision", "val_macro_recall",
                                    "epoch_train_seconds", "val_predict_seconds"]].round(4))
''')

interp("tagger_training")

md(r"""
### 5.1 Decoding thresholds (validation only)
The saved validation probabilities of the test run are decoded with a grid of thresholds: θ_inside ∈ {0.2, …, 0.8},
θ_begin ∈ {0.3, 0.5, 0.7} and `min_tokens` ∈ {1, 3, 5, 10}; the setting with the highest validation macro-F1 at IoU 0.5 is
used on test (one global setting). **Per-category thresholds** could adapt to each category but can easily over-fit the
small number of validation spans per category; we only adopt them if they help on validation contracts that were *not* used
to tune them (split-half check: tune on one half of the validation contracts, score on the other half, both ways round).
""")

code(r'''
%%needs_models models/task2
TEST_RUN = tchoice["test_run"]
RUN_DIR = ROOT / "models" / "task2" / TEST_RUN
vz = np.load(RUN_DIR / "val_probs.npz")
VPROBS = {c: vz[f"c{i}"].astype(np.float32) for i, c in enumerate(SPLIT["validation"])}
gold_val = spans[spans.split == "validation"]
grid = []
for ti in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    for tb in [0.3, 0.5, 0.7]:
        for mt in [1, 3, 5, 10]:
            pv = decode_contracts(VPROBS.get, SPLIT["validation"], ti, tb, mt)
            sm = M.summarise(M.count_table(gold_val, pv, SPLIT["validation"], CATS), CATS)
            grid.append({"theta_inside": ti, "theta_begin": tb, "min_tokens": mt, "val_macro_f1": sm["macro_f1"],
                         "val_micro_f1": sm["micro_f1"], "val_macro_precision": sm["macro_precision"], "val_macro_recall": sm["macro_recall"]})
grid = pd.DataFrame(grid); grid.to_csv(C.RESULTS_DIR / "task2_threshold_search.csv", index=False)
best_t = grid.sort_values(["val_macro_f1", "min_tokens"], ascending=[False, True]).iloc[0]
TH = {"theta_inside": float(best_t.theta_inside), "theta_begin": float(best_t.theta_begin), "min_tokens": int(best_t.min_tokens)}
default = grid[(grid.theta_inside == 0.5) & (grid.theta_begin == 0.5) & (grid.min_tokens == 1)].val_macro_f1.iloc[0]
print(f"Selected global thresholds: {TH} -> validation macro-F1 {best_t.val_macro_f1:.4f} (defaults 0.5/0.5/1: {default:.4f})")
display(grid.sort_values("val_macro_f1", ascending=False).head(8).round(4))

pivot = grid[grid.min_tokens == TH["min_tokens"]].pivot_table(index="theta_begin", columns="theta_inside", values="val_macro_f1")
fig, ax = plt.subplots(figsize=(8, 2.8))
im = ax.imshow(pivot.values, cmap=PL.BLUES, aspect="auto")
ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns); ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", fontsize=8,
                color="white" if pivot.values[i, j] > pivot.values.mean() else PL.INK)
ax.set_xlabel("theta_inside"); ax.set_ylabel("theta_begin"); ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.03)
ax.set_title(f"Figure 22. Validation macro-F1 @ IoU 0.5 of the tagger by decoding thresholds (min_tokens = {TH['min_tokens']})", fontsize=11)
PL.savefig(fig, 22, "t2_threshold_grid"); plt.show()
''')

code(r'''
%%needs_models models/task2
# Split-half check for per-category inside thresholds (validation only)
rng = np.random.default_rng(42)
vc = np.array(SPLIT["validation"]); perm = rng.permutation(len(vc)); halves = [list(vc[perm[:len(vc) // 2]]), list(vc[perm[len(vc) // 2:]])]
def per_cat_f1(contracts, ti_map):
    rows = []
    for c in contracts:
        for k in range(K):
            p = VPROBS[c]
            for _, s, e, conf in D.decode(np.c_[p[:, [k]], p[:, [K + k]]], corpus[c]["offsets"], 1, ti_map[k], TH["theta_begin"], TH["min_tokens"]):
                rows.append((c, CATS[k], s, e, conf))
    return M.summarise(M.count_table(gold_val[gold_val.contract.isin(contracts)], spans_df(rows), contracts, CATS), CATS)
def single_cat_f1(contracts, k, t):
    rows = [(c, CATS[k], s, e, conf) for c in contracts
            for _, s, e, conf in D.decode(np.c_[VPROBS[c][:, [k]], VPROBS[c][:, [K + k]]], corpus[c]["offsets"], 1, t,
                                          TH["theta_begin"], TH["min_tokens"])]
    cnt = M.count_table(gold_val[gold_val.contract.isin(contracts) & (gold_val.category == CATS[k])], spans_df(rows), contracts, [CATS[k]])
    return M.prf(cnt.tp.sum(), cnt.fp.sum(), cnt.fn.sum())[2]
check = []
for a, b in [(0, 1), (1, 0)]:
    tuned = {}
    for k in range(K):
        n_spans = (gold_val[gold_val.contract.isin(halves[a])].category == CATS[k]).sum()
        if n_spans < 10:
            tuned[k] = TH["theta_inside"]; continue
        best_k = max([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], key=lambda t: single_cat_f1(halves[a], k, t))
        tuned[k] = best_k
    glob = per_cat_f1(halves[b], {j: TH["theta_inside"] for j in range(K)})["macro_f1"]
    perc = per_cat_f1(halves[b], tuned)["macro_f1"]
    n_tuned = int(sum((gold_val[gold_val.contract.isin(halves[a])].category == CATS[k]).sum() >= 10 for k in range(K)))
    check.append({"tuned on half": a, "scored on half": b, "global macro-F1": glob, "per-category macro-F1": perc,
                  "categories with >= 10 spans (tuned)": n_tuned})
check = pd.DataFrame(check); display(check.round(4))
USE_PER_CAT = bool((check["per-category macro-F1"] > check["global macro-F1"]).all())
json.dump({**TH, "per_category_split_half": check.to_dict(orient="records"), "use_per_category": USE_PER_CAT},
          open(C.RESULTS_DIR / "task2_thresholds.json", "w"), indent=2, default=float)
print("Per-category thresholds adopted:", USE_PER_CAT, "(only if they help on the held-out half in BOTH directions)")
''')

interp("thresholds")

md(r"""
### 5.2 Window-size sensitivity (validation only)
The test model uses 256-token windows with stride 128, as fixed in the design. A longer window sees more context and
fits more gold spans completely (section 2.2), so one extra tagger was trained with **384/192 windows** — same recipe,
loss, 8 epochs and seed 42 (`python scripts/train_task2.py --window384`, logged in
`results/task2_window_sensitivity_*.csv`). It is compared with the 256/128 model **on validation only**, at the default
thresholds and after re-tuning its own thresholds on the same grid as section 5.1. It is **never evaluated on test**;
the test model stays 256/128 whatever the result (decided before training it).
""")

code(r'''
W384_RUN = "t2_bce_plain_s42_e8_w384"
w_search = pd.read_csv(C.RESULTS_DIR / "task2_window_sensitivity_search.csv")
wz_file = ROOT / "models" / "task2" / W384_RUN / "val_probs.npz"
wchoice = pd.read_csv(C.RESULTS_DIR / "task2_window_choice.csv").set_index("window/stride")
def tune_on_val(probs):
    best = None
    for ti in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for tb in [0.3, 0.5, 0.7]:
            for mt in [1, 3, 5, 10]:
                sm = M.summarise(M.count_table(gold_val, decode_contracts(probs.get, SPLIT["validation"], ti, tb, mt),
                                               SPLIT["validation"], CATS), CATS)
                if best is None or sm["macro_f1"] > best[0] + 1e-12:
                    best = (sm["macro_f1"], ti, tb, mt, sm["macro_precision"], sm["macro_recall"])
    return best
if wz_file.exists():
    wz = np.load(wz_file); W384_PROBS = {c: wz[f"c{i}"].astype(np.float32) for i, c in enumerate(SPLIT["validation"])}
    rows = []
    for label, probs, run in [("256/128 (test model)", VPROBS, TEST_RUN), ("384/192", W384_PROBS, W384_RUN)]:
        d = M.summarise(M.count_table(gold_val, decode_contracts(probs.get, SPLIT["validation"]), SPLIT["validation"], CATS), CATS)
        t = tune_on_val(probs)
        meta = (tsearch if run == TEST_RUN else w_search).set_index("run_id").loc[run]
        key = label.split(" ")[0]
        rows.append({"window/stride": label, "best epoch": int(meta.best_epoch),
                     "val macro-F1 (default thresholds)": d["macro_f1"],
                     "val macro-F1 (own tuned thresholds)": t[0], "tuned (theta_i, theta_b, min_tokens)": f"{t[1]}, {t[2]}, {t[3]}",
                     "macro P / R (tuned)": f"{t[4]:.3f} / {t[5]:.3f}",
                     "gold spans fully inside >= 1 window %": float(wchoice.loc[key, "spans fully inside >= 1 window %"]),
                     "training seconds": float(meta.run_seconds), "peak GPU GB": float(meta.peak_gpu_mem_gb)})
    wtab = pd.DataFrame(rows)
    wtab.to_csv(C.RESULTS_DIR / "task2_window_sensitivity.csv", index=False)
    display(wtab.round(4))
    gain = wtab.iloc[1]["val macro-F1 (own tuned thresholds)"] - wtab.iloc[0]["val macro-F1 (own tuned thresholds)"]
    print(f"384/192 minus 256/128 on validation (each with its own tuned thresholds): {gain:+.4f} macro-F1 @ IoU 0.5 "
          f"(one seed each; the seed-to-seed spread of the 256/128 recipe is std {tchoice['val_macro_f1_std']:.4f}). "
          "The test model stays 256/128.")
else:
    print("384/192 validation probabilities not found (models/ is not in the repository); the saved table is shown instead.")
    display(pd.read_csv(C.RESULTS_DIR / "task2_window_sensitivity.csv").round(4))
''')

interp("window_sensitivity")

# =====================================================================================
md(r"""
## 6. Test evaluation (once per model)
Both models are now applied to the **102 official test contracts** with the settings fixed on validation and scored once
each through `C.test_once_spans`, which logs the evaluation in `results/test_eval_log.csv` and refuses a second one.
""")

code(r'''
%%needs_models models/task2
gold_test = spans[spans.split == "test"]
# --- baseline ---
btest_pred, _ = baseline_predict(SPLIT["test"], bchoice["threshold"])
# --- tagger: per-token test probabilities (computed once with the saved weights, then cached in models/) ---
tp_file = RUN_DIR / "test_probs.npz"
if not tp_file.exists() and (RUN_DIR / "best").exists():
    subprocess.run([sys.executable, "scripts/train_task2.py", "--predict", TEST_RUN, "--split", "test"], check=True)
if tp_file.exists():
    tz = np.load(tp_file); TPROBS = {c: tz[f"c{i}"].astype(np.float32) for i, c in enumerate(SPLIT["test"])}
    ttest_pred = decode_contracts(TPROBS.get, SPLIT["test"], TH["theta_inside"], TH["theta_begin"], TH["min_tokens"])
else:
    print("Tagger weights and cached probabilities not found: loading the saved test spans instead.")
    TPROBS = None
    ttest_pred = pd.read_csv(C.RESULTS_DIR / "task2" / "t2_legalbert_tagger_test_spans.csv")

def full_metrics(pred, probs=None, seg_pos=None):
    out = {}
    cnt = M.count_table(gold_test, pred, SPLIT["test"], CATS)
    s = M.summarise(cnt, CATS); se = M.summarise(M.count_table(gold_test, pred, SPLIT["test"], CATS, mode="exact", snaps=SNAPS), CATS)
    out.update({k: s[k] for k in ["macro_f1", "micro_f1", "macro_precision", "macro_recall", "micro_precision", "micro_recall"]})
    out.update({"exact_macro_f1": se["macro_f1"], "exact_micro_f1": se["micro_f1"]})
    # token-level F1 on INSIDE targets
    pred_in = []
    for c in SPLIT["test"]:
        P = np.zeros((len(corpus[c]["offsets"]), K), dtype=bool)
        for r in pred[pred.contract == c].itertuples():
            rng_ = D.span_token_range(corpus[c]["offsets"], r.start, r.end)
            if rng_:
                P[rng_[0]:rng_[1] + 1, CIDX[r.category]] = True
        pred_in.append(P)
    out.update(M.token_f1(pred_in, [LABELS[c][:, :K] for c in SPLIT["test"]]))
    keys = [(c, k) for c in SPLIT["test"] for k in CATS]
    cur = M.cuad_curve(gold_test, pred, text_of, keys)
    out.update({"cuad_micro_aupr": cur["aupr"], "cuad_prec_at_80_recall": cur["prec_at_80_recall"],
                "cuad_prec_at_90_recall": cur["prec_at_90_recall"]})
    per_cat_aupr = {k: M.cuad_curve(gold_test[gold_test.category == k], pred[pred.category == k], text_of,
                                    [(c, k) for c in SPLIT["test"]])["aupr"] for k in CATS}
    out["cuad_macro_aupr"] = float(np.mean(list(per_cat_aupr.values())))
    free = pred[pred.contract.isin(SPAN_FREE_TEST)]
    out.update({"fp_in_span_free_contracts": int(len(free)), "span_free_contracts_with_any_prediction": int(free.contract.nunique())})
    return out, cnt, s, cur, per_cat_aupr

RES, CNT, SUMM, CURVE, PCAUPR = {}, {}, {}, {}, {}
for name, pred in [("t2_tfidf_segments", btest_pred), ("t2_legalbert_tagger", ttest_pred)]:
    RES[name], CNT[name], SUMM[name], CURVE[name], PCAUPR[name] = full_metrics(pred)
cfg_b = {**bchoice, "segmenter": "src/task2_baseline.py"}
cfg_t = {"run": TEST_RUN, "variant": tchoice["variant"], "thresholds": TH, "window": 256, "stride": 128, "base_checkpoint": CKPT}
C.test_once_spans("t2_tfidf_segments", cfg_b, RES["t2_tfidf_segments"], btest_pred)
C.test_once_spans("t2_legalbert_tagger", cfg_t, RES["t2_legalbert_tagger"], ttest_pred)
''')

code(r'''
%%needs_models models/task2
# Computational budget of the two Task 2 models -> results/compute_budget.csv (logged once)
t0 = time.perf_counter(); baseline_predict(SPLIT["test"], bchoice["threshold"]); b_inf = time.perf_counter() - t0
run = ts.loc[TEST_RUN]
tagger_inf = json.load(open(RUN_DIR / "test_predict_seconds.json")) if (RUN_DIR / "test_predict_seconds.json").exists() else None
budget = pd.read_csv(C.RESULTS_DIR / "compute_budget.csv")
new_rows = []
if not ((budget.model == "t2_legalbert_tagger") & (budget.stage == "train_final")).any():
    new_rows.append({"model": "t2_legalbert_tagger", "stage": "train_final", "seconds": round(float(run.run_seconds), 2),
                     "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                     "notes": f"{int(run.epochs)} epochs x {int(run.steps_per_epoch)} steps (batch 16, window 256) incl. validation decoding; "
                              f"peak {run.peak_gpu_mem_gb:.1f} GB"})
    if tagger_inf:
        new_rows.append({"model": "t2_legalbert_tagger", "stage": "inference_test_gpu", "seconds": round(tagger_inf["seconds"], 2),
                         "device": torch.cuda.get_device_name(0), "notes": f"{tagger_inf['contracts']} test contracts, all windows, bf16"})
    new_rows.append({"model": "t2_tfidf_segments", "stage": "inference_test_cpu", "seconds": round(b_inf, 2), "device": "CPU",
                     "notes": f"{len(SPLIT['test'])} test contracts: segmentation + TF-IDF + 27 LogReg + merging"})
    budget = pd.concat([budget, pd.DataFrame([{**r, "timestamp": pd.Timestamp.now().isoformat(timespec="seconds")} for r in new_rows])],
                       ignore_index=True)
    budget.to_csv(C.RESULTS_DIR / "compute_budget.csv", index=False)
bt = budget[budget.model.isin(["t2_legalbert_tagger", "t2_tfidf_segments", "t2_tagger_plan"])]
display(bt[["model", "stage", "seconds", "device", "notes"]])
if tagger_inf:
    print(f"Tagger inference: {tagger_inf['seconds_per_contract']:.3f} s per contract on the GPU -> "
          f"{tagger_inf['seconds_per_contract'] * 1000 / 60:.1f} min per 1,000 contracts; "
          f"baseline: {b_inf / len(SPLIT['test']):.3f} s per contract on the CPU (this re-run)")
''')

code(r'''
%%needs_models models/task2
ci = {n: M.bootstrap_macro_f1(CNT[n], CATS) for n in CNT}
pb = M.paired_bootstrap_macro_f1(CNT["t2_legalbert_tagger"], CNT["t2_tfidf_segments"], CATS)
oracle_test = oracle["test | INSIDE+BEGIN | IoU>=0.5"]["macro_f1"]
rows = []
for label, n in [("TF-IDF segment baseline", "t2_tfidf_segments"), ("Legal-BERT tagger", "t2_legalbert_tagger")]:
    r = RES[n]
    rows.append({"model": label, "macro-F1 @ IoU0.5": r["macro_f1"], "95% CI": f"[{ci[n]['ci_low']:.3f}, {ci[n]['ci_high']:.3f}]",
                 "micro-F1 @ IoU0.5": r["micro_f1"], "macro P": r["macro_precision"], "macro R": r["macro_recall"],
                 "exact macro-F1": r["exact_macro_f1"], "token micro-F1": r["token_micro_f1"],
                 "CUAD-style AUPR (micro)": r["cuad_micro_aupr"], "P@80%R": r["cuad_prec_at_80_recall"], "P@90%R": r["cuad_prec_at_90_recall"],
                 "CUAD-style macro AUPR": r["cuad_macro_aupr"], "FP in 9 span-free contracts": r["fp_in_span_free_contracts"]})
rows.append({"model": "oracle ceilings (gold labels)", "macro-F1 @ IoU0.5": oracle_test,
             "95% CI": f"segments: {seg_oracle['test']['macro_f1']:.3f}"})
t2_table = pd.DataFrame(rows)
t2_table.to_csv(C.RESULTS_DIR / "task2_test_summary.csv", index=False)
display(t2_table.round(4))
print(f"Objective O4: tagger - baseline macro-F1 @ IoU0.5 = {pb['diff']:+.4f}, 95 % CI [{pb['ci_low']:+.4f}, {pb['ci_high']:+.4f}] -> "
      f"{'MET (CI above 0)' if pb['ci_low'] > 0 else 'NOT met'} for the first part of O4 (the CUAD comparison is in section 7)")
json.dump({"ci": ci, "paired_tagger_vs_baseline": pb, "oracle_test_macro_f1": oracle_test, "segment_oracle_test": seg_oracle["test"]},
          open(C.RESULTS_DIR / "task2_test_comparison.json", "w"), indent=2)
''')

code(r'''
%%needs_models models/task2
# Per-category F1 with bootstrap CIs (resamples without the category skipped), both models
def per_cat_ci(cnt, n_boot=1000, seed=42):
    contracts = cnt.contract.unique()
    cube = (cnt.set_index(["contract", "category"])[["tp", "fp", "fn"]]
            .reindex(pd.MultiIndex.from_product([contracts, CATS])).fillna(0).to_numpy().reshape(len(contracts), K, 3))
    rng_ = np.random.default_rng(seed); boots = []
    for _ in range(n_boot):
        s = cube[rng_.integers(0, len(contracts), len(contracts))].sum(0)
        tp, fp, fn = s[:, 0], s[:, 1], s[:, 2]
        p = np.divide(tp, tp + fp, out=np.zeros(K), where=tp + fp > 0); r = np.divide(tp, tp + fn, out=np.zeros(K), where=tp + fn > 0)
        f = np.divide(2 * p * r, p + r, out=np.zeros(K), where=p + r > 0)
        boots.append(np.where(tp + fn > 0, f, np.nan))
    b = np.array(boots)
    return np.nanpercentile(b, 2.5, axis=0), np.nanpercentile(b, 97.5, axis=0)
pc = pd.DataFrame({"gold test spans": SUMM["t2_legalbert_tagger"]["per_category"].gold,
                   "train spans": spans[spans.split == "train"].category.value_counts().reindex(CATS),
                   "baseline F1": SUMM["t2_tfidf_segments"]["per_category"].f1, "tagger F1": SUMM["t2_legalbert_tagger"]["per_category"].f1,
                   "tagger P": SUMM["t2_legalbert_tagger"]["per_category"].precision, "tagger R": SUMM["t2_legalbert_tagger"]["per_category"].recall,
                   "tagger CUAD AUPR": pd.Series(PCAUPR["t2_legalbert_tagger"])})
lo, hi = per_cat_ci(CNT["t2_legalbert_tagger"])
pc["tagger 95% CI"] = [f"[{l:.2f}, {h:.2f}]" for l, h in zip(lo, hi)]
pc = pc.sort_values("tagger F1")
pc.to_csv(C.RESULTS_DIR / "task2_per_category.csv")
display(pc.round(3))
fig, ax = plt.subplots(figsize=(10, 9))
yy = np.arange(len(pc))
ax.hlines(yy, pc[["baseline F1", "tagger F1"]].min(axis=1), pc[["baseline F1", "tagger F1"]].max(axis=1), color="#cfcdc6", linewidth=2)
ax.scatter(pc["baseline F1"], yy, color=PL.AQUA, marker="D", s=40, label="TF-IDF segment baseline", zorder=3)
ax.scatter(pc["tagger F1"], yy, color=PL.BLUE, s=46, label="Legal-BERT tagger", zorder=3)
ax.set_yticks(yy); ax.set_yticklabels([f"{c} (n={n})" for c, n in zip(pc.index, pc["gold test spans"])], fontsize=8.5)
ax.set_xlabel("Test entity F1 @ IoU 0.5"); ax.legend(loc="lower right"); ax.grid(axis="y", visible=False)
ax.set_title("Figure 23. Task 2 per-category test F1: TF-IDF segment baseline vs Legal-BERT tagger", fontsize=11)
PL.savefig(fig, 23, "t2_per_category_f1"); plt.show()
''')

code(r'''
%%needs_models models/task2
# Subgroups: contract length terciles and category-frequency terciles (tagger)
L = pd.Series({c: len(TEXT[c]) for c in SPLIT["test"]})
terc = pd.qcut(L, 3, labels=["short", "medium", "long"])
sub = []
for name in CNT:
    for t in ["short", "medium", "long"]:
        cc = list(L[terc == t].index)
        s = M.summarise(CNT[name][CNT[name].contract.isin(cc)], CATS)
        sub.append({"model": name, "subgroup": f"contract length: {t} ({L[terc == t].min():,}-{L[terc == t].max():,} chars, {len(cc)} contracts)",
                    "macro_f1": s["macro_f1"], "micro_f1": s["micro_f1"]})
    freq = spans[spans.split == "train"].category.value_counts().reindex(CATS)
    ft = pd.qcut(freq.rank(method="first"), 3, labels=["rare", "medium", "frequent"])
    per = SUMM[name]["per_category"]
    for t in ["rare", "medium", "frequent"]:
        cats_t = list(ft[ft == t].index)
        sub.append({"model": name, "subgroup": f"category frequency: {t} ({freq[cats_t].min()}-{freq[cats_t].max()} train spans)",
                    "macro_f1": float(per.loc[cats_t, "f1"].mean()), "micro_f1": np.nan})
sub = pd.DataFrame(sub); sub.to_csv(C.RESULTS_DIR / "task2_subgroups.csv", index=False)
display(sub.pivot_table(index="subgroup", columns="model", values="macro_f1").round(3))
''')

interp("test")

# =====================================================================================
md(r"""
## 7. Published comparison: the CUAD paper
The CUAD paper (Hendrycks et al., 2021) reports results on the same official test contracts. The numbers below were read
from the paper itself (Table 2, page 6; PDF from the NeurIPS Datasets & Benchmarks proceedings)
and are typed here from that table; our numbers come from `results/`. The paper gives
per-category AUPR only as a bar chart without values (Figure 4, page 7), so per-category comparison is by **rank only**.
""")

md(r"""
### 7.1 Why recall stops near 58 %, and a low-threshold candidate set
The CUAD-style sweep ranks the *decoded* spans by their confidence (the mean INSIDE probability over the span's tokens).
At the tuned operating point (θ_inside = 0.6) every decoded span already has confidence ≥ 0.6 and tokens below 0.6 never
form a span, so lowering the confidence cut-off in the sweep cannot add new candidates: the curve ends at the recall of
the decoded set. To see the trade-off, the validation probabilities are decoded at θ_inside from 0.05 to 0.6 (θ_begin and
`min_tokens` fixed at their tuned values).

**Selection rule for the low-threshold set (fixed before looking at the sweep; validation only):** the θ_inside with the
highest **validation CUAD-style maximum recall**, among the values whose median predicted span length is at most twice the
median gold span length (a sanity condition: a very low threshold glues neighbouring clauses into very long spans, which is
not a usable candidate list); ties go to the higher θ. The resulting candidate set, ranked by confidence, is evaluated
**once on test** as a separate model `t2_legalbert_tagger_lowthr` for the threshold-free (AUPR / precision-at-recall)
analysis. **The main operating-point result of the tagger stays the 0.523 of section 6.**
""")

code(r'''
%%needs_models models/task2
val_keys = [(c, k) for c in SPLIT["validation"] for k in CATS]
gold_len_med = float((gold_val.end - gold_val.start).median())
sweep, VCURVE = [], {}
for ti in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]:
    pv = decode_contracts(VPROBS.get, SPLIT["validation"], ti, TH["theta_begin"], TH["min_tokens"])
    sm = M.summarise(M.count_table(gold_val, pv, SPLIT["validation"], CATS), CATS)
    cur = M.cuad_curve(gold_val, pv, text_of, val_keys); VCURVE[ti] = cur
    sweep.append({"theta_inside": ti, "predicted spans": len(pv), "median predicted length (chars)": float((pv.end - pv.start).median()),
                  "IoU0.5 micro precision": sm["micro_precision"], "IoU0.5 micro recall": sm["micro_recall"],
                  "IoU0.5 macro-F1": sm["macro_f1"], "IoU0.5 macro precision": sm["macro_precision"], "IoU0.5 macro recall": sm["macro_recall"], "CUAD-style max recall": max(r for r in cur["recalls"] if r == r),
                  "CUAD-style AUPR": cur["aupr"], "CUAD-style P@80%R": cur["prec_at_80_recall"]})
sweep = pd.DataFrame(sweep)
sweep["sensible (median length <= 2 x gold median)"] = sweep["median predicted length (chars)"] <= 2 * gold_len_med
print(f"Median gold span length on validation: {gold_len_med:.0f} characters")
display(sweep.round(4))
ok = sweep[sweep["sensible (median length <= 2 x gold median)"]]
best_low = ok.sort_values(["CUAD-style max recall", "theta_inside"], ascending=[False, False]).iloc[0]
TI_LOW = float(best_low.theta_inside)
print(f"Selected low threshold: theta_inside = {TI_LOW} -> validation CUAD-style max recall {best_low['CUAD-style max recall']:.3f} "
      f"(tuned 0.6: {sweep.set_index('theta_inside').loc[0.6, 'CUAD-style max recall']:.3f}), AUPR {best_low['CUAD-style AUPR']:.3f}")

# Why the tuned decoder misses spans: the highest INSIDE probability of the gold category inside each validation gold span
mx = []
for r in gold_val.itertuples():
    rr = D.span_token_range(corpus[r.contract]["offsets"], r.start, r.end)
    if rr:
        mx.append(float(VPROBS[r.contract][rr[0]:rr[1] + 1, CIDX[r.category]].max()))
mx = np.array(mx)
peak = pd.DataFrame({"highest INSIDE probability in the gold span is below": [0.6, 0.3, 0.1, 0.05],
                     "share of validation gold spans": [float((mx < t).mean()) for t in [0.6, 0.3, 0.1, 0.05]]})
display(peak.round(3))
json.dump({"selection_rule": "max validation CUAD-style max recall s.t. median predicted length <= 2 x median gold length; ties -> higher theta",
           "theta_inside_low": TI_LOW, "theta_begin": TH["theta_begin"], "min_tokens": TH["min_tokens"], "gold_median_length_chars": gold_len_med,
           "sweep": sweep.to_dict(orient="records"), "gold_span_peak_probability": peak.to_dict(orient="records")},
          open(C.RESULTS_DIR / "task2_recall_ceiling.json", "w"), indent=2, default=float)
sweep.to_csv(C.RESULTS_DIR / "task2_recall_ceiling_sweep.csv", index=False)
# Appendix table (report): macro-averaged values of the same sweep (Figure 24 shows the micro values)
macro_tab = sweep[["theta_inside", "IoU0.5 macro precision", "IoU0.5 macro recall", "IoU0.5 macro-F1"]]
macro_tab.to_csv(C.RESULTS_DIR / "task2_recall_ceiling_macro_appendix.csv", index=False)
print("Appendix table - macro-averaged precision / recall / F1 at IoU 0.5 by theta_inside (validation):")
display(macro_tab.round(4))
''')

code(r'''
%%needs_models models/task2
# The low-threshold candidate set on TEST (evaluated once; threshold-free analysis only)
if TPROBS is not None:
    lowthr_pred = decode_contracts(TPROBS.get, SPLIT["test"], TI_LOW, TH["theta_begin"], TH["min_tokens"])
else:
    lowthr_pred = pd.read_csv(C.RESULTS_DIR / "task2" / "t2_legalbert_tagger_lowthr_test_spans.csv")
n = "t2_legalbert_tagger_lowthr"
RES[n], CNT[n], SUMM[n], CURVE[n], PCAUPR[n] = full_metrics(lowthr_pred)
cfg_low = {"run": TEST_RUN, "theta_inside": TI_LOW, "theta_begin": TH["theta_begin"], "min_tokens": TH["min_tokens"],
           "purpose": "threshold-free analysis (AUPR, precision at recall); same weights and probabilities as t2_legalbert_tagger",
           "selection": "validation only, rule in section 7.1"}
C.test_once_spans(n, cfg_low, RES[n], lowthr_pred)
low_tab = pd.DataFrame([{"model": lab, "decoded spans": len(pred), "CUAD-style AUPR %": 100 * RES[m]["cuad_micro_aupr"],
                         "P@80%R %": 100 * RES[m]["cuad_prec_at_80_recall"], "P@90%R %": 100 * RES[m]["cuad_prec_at_90_recall"],
                         "max recall %": 100 * max(r for r in CURVE[m]["recalls"] if r == r),
                         "macro-F1 @ IoU0.5 (all spans kept)": RES[m]["macro_f1"]}
                        for lab, m, pred in [("tagger, tuned operating point (theta_i 0.6)", "t2_legalbert_tagger", ttest_pred),
                                             (f"tagger, low-threshold candidates (theta_i {TI_LOW})", n, lowthr_pred)]])
low_tab.to_csv(C.RESULTS_DIR / "task2_lowthr_test.csv", index=False)
display(low_tab.round(3))
''')

interp("recall_ceiling")

code(r'''
%%needs_models models/task2
paper = pd.DataFrame([("BERT-base", 32.4, 8.2, 0.0), ("RoBERTa-base", 42.6, 31.1, 0.0),
                      ("RoBERTa-base + Contracts Pretraining", 45.2, 34.1, 0.0), ("RoBERTa-large", 48.2, 38.1, 0.0),
                      ("DeBERTa-xlarge (best in paper)", 47.8, 44.0, 17.8)],
                     columns=["model", "AUPR %", "P@80%R %", "P@90%R %"]).assign(source="CUAD paper, Table 2 p. 6 (41 categories)")
ours = pd.DataFrame([(lab, 100 * RES[n]["cuad_micro_aupr"], 100 * RES[n]["cuad_prec_at_80_recall"], 100 * RES[n]["cuad_prec_at_90_recall"])
                     for lab, n in [("TF-IDF segment baseline (ours)", "t2_tfidf_segments"), ("Legal-BERT tagger (ours)", "t2_legalbert_tagger"),
                                    (f"Legal-BERT tagger, low-threshold candidates (ours, theta_i {TI_LOW})", "t2_legalbert_tagger_lowthr")]],
                    columns=["model", "AUPR %", "P@80%R %", "P@90%R %"]).assign(source="ours, CUAD-style evaluation (27 categories)")
cmp = pd.concat([paper, ours], ignore_index=True)
cmp["max recall reached %"] = [np.nan] * len(paper) + [100 * max(r for r in CURVE[n]["recalls"] if r == r)
                                                       for n in ["t2_tfidf_segments", "t2_legalbert_tagger", "t2_legalbert_tagger_lowthr"]]
cmp.to_csv(C.RESULTS_DIR / "task2_cuad_comparison.csv", index=False)
display(cmp.round(1))
print("Our CUAD-style curves stop at the recall reached by the decoded spans (last column); the paper's QA models produce "
      "n-best candidate answers and reach higher recall (they report precision at 90 % recall). The low-threshold row is our "
      "closest analogue of an n-best list. Not like-for-like: 27 vs 41 categories, windowed tagger vs extractive QA.")

# Rank comparison per category: paper's Figure 4 order (lowest to highest AUPR, DeBERTa-xlarge) vs our tagger's AUPR
paper_order = ["Covenant Not To Sue", "Ip Ownership Assignment", "Rofr/Rofo/Rofn", "Most Favored Nation", "Post-Termination Services",
               "Change Of Control", "Warranty Duration", "Competitive Restriction Exception", "Affiliate License-Licensor",
               "Unlimited/All-You-Can-Eat-License", "Non-Compete", "Joint Ip Ownership", "Minimum Commitment", "Non-Disparagement",
               "Affiliate License-Licensee", "Liquidated Damages", "Volume Restriction", "Third Party Beneficiary", "Uncapped Liability",
               "Non-Transferable License", "No-Solicit Of Customers", "Notice Period To Terminate Renewal", "Termination For Convenience",
               "Effective Date", "Revenue/Profit Sharing", "Audit Rights", "Renewal Term", "Exclusivity", "Irrevocable Or Perpetual License",
               "Source Code Escrow", "Insurance", "License Grant", "Cap On Liability", "No-Solicit Of Employees", "Anti-Assignment",
               "Expiration Date", "Agreement Date", "Parties", "Document Name", "Governing Law"]
shared = [c for c in paper_order if c in CATS]
from scipy.stats import spearmanr
paper_rank = pd.Series(range(len(shared)), index=shared)
our_aupr = pd.Series(PCAUPR["t2_legalbert_tagger"])[shared]
rho = spearmanr(paper_rank, our_aupr).correlation
print(f"{len(shared)} of our 27 categories appear in the paper's Figure 4. Spearman rank correlation between the paper's "
      f"category order and our tagger's per-category AUPR: {rho:.2f}")
json.dump({"shared_categories": len(shared), "spearman_rank_vs_paper_figure4": rho}, open(C.RESULTS_DIR / "task2_cuad_rank.json", "w"), indent=2)
''')

code(r'''
%%needs_models models/task2
# Figure 24: CUAD-style precision-recall curves on test (both tagger candidate sets and the baseline) and the validation sweep
fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
for name, col, ls, lab in [("t2_tfidf_segments", PL.AQUA, "-", "TF-IDF segment baseline"),
                           ("t2_legalbert_tagger", PL.BLUE, "-", "Legal-BERT tagger, tuned (theta_i 0.6)"),
                           ("t2_legalbert_tagger_lowthr", PL.VIOLET, "--", f"Legal-BERT tagger, low threshold (theta_i {TI_LOW})")]:
    cur = CURVE[name]
    axes[0].plot(np.array(cur["recalls"]) * 100, np.array(cur["precisions"]) * 100, ls, color=col, linewidth=2,
                 label=f"{lab}: AUPR {100 * cur['aupr']:.1f}")
for _, r in paper.iterrows():
    if r["model"].startswith(("RoBERTa-large", "DeBERTa")):
        axes[0].scatter([80, 90], [r["P@80%R %"], r["P@90%R %"]], marker="x", s=40, color=PL.INK_2)
        axes[0].annotate(r["model"].split(" (")[0] + " (paper)", (80, r["P@80%R %"]), textcoords="offset points", xytext=(-5, 4),
                         ha="right", fontsize=7)          # label to the left of the point, so it stays inside the left panel
axes[0].axvline(80, color=PL.INK_2, linestyle=":", linewidth=1)
axes[0].set_xlabel("recall (%)"); axes[0].set_ylabel("precision (%)"); axes[0].legend(fontsize=7.5, loc="lower left")
axes[0].set_title("Test, CUAD-style sweep (x = paper's P@80/90%R, 41 categories)", fontsize=10)
axes[1].plot(sweep.theta_inside, sweep["IoU0.5 micro precision"], "s-", color=PL.ORANGE, label="precision (IoU 0.5, micro)")
axes[1].plot(sweep.theta_inside, sweep["IoU0.5 micro recall"], "o-", color=PL.BLUE, label="recall (IoU 0.5, micro)")
axes[1].plot(sweep.theta_inside, sweep["CUAD-style max recall"], "^--", color=PL.VIOLET, label="CUAD-style max recall")
axes[1].axvline(TI_LOW, color=PL.VIOLET, linestyle=":", linewidth=1); axes[1].axvline(TH["theta_inside"], color=PL.BLUE, linestyle=":", linewidth=1)
axes[1].set_xlabel("theta_inside"); axes[1].set_ylabel("validation score"); axes[1].legend(fontsize=8)
axes[1].set_title("Validation: precision and recall vs the inside threshold", fontsize=10)
fig.subplots_adjust(wspace=0.28)
fig.suptitle("Figure 24. Precision-recall trade-off of the Task 2 models", fontweight="bold", y=1.03)
PL.savefig(fig, 24, "t2_precision_recall"); plt.show()
''')

interp("curves_pr")

interp("published")

# =====================================================================================
md(r"""
## 8. Representative outputs and error analysis
### 8.1 Error taxonomy over the whole test set (tagger)
Every unmatched gold span and every unmatched prediction is assigned to one cause, checked against the data:
**missed** (no prediction of that category overlaps it), **boundary too short / too long** (a same-category prediction
overlaps but IoU < 0.5), **split** (several predictions cover one gold span), **merged** (one prediction covers several gold
spans), **wrong category** (a prediction of another category matches it at IoU ≥ 0.5), **spurious** (a prediction overlapping
no gold span of any category; counted separately in the span-free contracts), **multi-category span partly found** (the gold
span carries several categories and only some were found), and **long span cut by the window** (gold span longer than one
window, 256 tokens).
""")

code(r'''
%%needs_models models/task2
def taxonomy(pred, name):
    rows = []
    g_all = gold_test
    for c in SPLIT["test"]:
        gc, pcn = g_all[g_all.contract == c], pred[pred.contract == c]
        offs = corpus[c]["offsets"]
        for k in CATS:
            gl = list(zip(gc[gc.category == k].start, gc[gc.category == k].end, gc[gc.category == k].n_categories_on_span))
            pl = list(zip(pcn[pcn.category == k].start, pcn[pcn.category == k].end, pcn[pcn.category == k].confidence))
            tp, fp, fn, pairs = M.match([(s, e) for s, e, _ in gl], pl)
            mg = {j for _, j in pairs}; mp = {i for i, _ in pairs}
            for j, (s, e, ncat) in enumerate(gl):
                if j in mg:
                    continue
                ov = [p for p in pl if min(p[1], e) > max(p[0], s)]
                other = pcn[(pcn.category != k)]
                wrong = any(M.char_iou((r.start, r.end), (s, e)) >= 0.5 for r in other.itertuples())
                rng_ = D.span_token_range(offs, s, e); long_ = rng_ and (rng_[1] - rng_[0] + 1) > 256
                if len(ov) >= 2:
                    cause = "split"
                elif len(ov) == 1:
                    p = ov[0]
                    covers = sum(1 for s2, e2, _ in gl if min(p[1], e2) > max(p[0], s2))
                    cause = "merged" if covers >= 2 else ("boundary too short" if (p[1] - p[0]) < (e - s) else "boundary too long")
                elif wrong:
                    cause = "wrong category"
                elif ncat > 1:
                    cause = "multi-category span partly found" if any(
                        M.char_iou((r.start, r.end), (s, e)) >= 0.5 for r in pcn.itertuples()) else "missed"
                else:
                    cause = "missed"
                if long_ and cause in ("missed", "split", "boundary too short"):
                    cause += " (long span > window)"
                rows.append({"model": name, "contract": c, "category": k, "kind": "false negative", "cause": cause, "start": s, "end": e,
                             "confidence": np.nan})
            for i, p in enumerate(pl):
                if i in mp:
                    continue
                same = any(min(p[1], e) > max(p[0], s) for s, e, _ in gl)
                anyg = any(min(p[1], r.end) > max(p[0], r.start) for r in gc.itertuples())
                cause = ("boundary / split / merge (same category)" if same else
                         "wrong category" if anyg else
                         "spurious (span-free contract)" if c in SPAN_FREE_TEST else "spurious")
                rows.append({"model": name, "contract": c, "category": k, "kind": "false positive", "cause": cause, "start": p[0], "end": p[1],
                             "confidence": p[2]})
    return pd.DataFrame(rows)
ERR = pd.concat([taxonomy(btest_pred, "t2_tfidf_segments"), taxonomy(ttest_pred, "t2_legalbert_tagger")], ignore_index=True)
ERR.to_csv(C.RESULTS_DIR / "task2_error_taxonomy.csv", index=False)
tax = ERR.groupby(["model", "kind", "cause"]).size().unstack(0).fillna(0).astype(int)
display(tax)
tt = ERR[ERR.model == "t2_legalbert_tagger"].groupby(["kind", "cause"]).size().sort_values()
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.barh([f"{k}: {c}" for k, c in tt.index], tt.values, color=[PL.ORANGE if k == "false positive" else PL.BLUE for k, _ in tt.index])
for i, v in enumerate(tt.values):
    ax.text(v + 1, i, str(v), va="center", fontsize=8)
ax.set_xlabel("number of errors on the test set"); ax.grid(axis="y", visible=False)
ax.set_title("Figure 25. Error taxonomy of the Legal-BERT tagger on the 102 test contracts (blue = missed gold, orange = extra predictions)", fontsize=10)
PL.savefig(fig, 25, "t2_error_taxonomy"); plt.show()
# Per-category breakdown for the report appendix (overall counts go in the report body)
ptax = (ERR[ERR.model == "t2_legalbert_tagger"].assign(cause=lambda d: d.kind.map({"false negative": "FN", "false positive": "FP"}) + ": " + d.cause)
        .pivot_table(index="category", columns="cause", values="start", aggfunc="count", fill_value=0))
ptax["total errors"] = ptax.sum(axis=1)
ptax = ptax.sort_values("total errors", ascending=False)
ptax.to_csv(C.RESULTS_DIR / "task2_error_taxonomy_per_category.csv")
print("Per-category error counts of the tagger (report appendix): results/task2_error_taxonomy_per_category.csv")
display(ptax.head(10))
''')

interp("taxonomy")

md(r"""
### 8.2 Representative outputs: gold vs predicted spans
Ten test excerpts of varied categories. For each, the gold span(s) and the tagger's prediction(s) of that category in the
same region are shown with their confidence (texts shortened).
""")

code(r'''
%%needs_models models/task2
def show(c, k, width=230):
    g = gold_test[(gold_test.contract == c) & (gold_test.category == k)]
    p = ttest_pred[(ttest_pred.contract == c) & (ttest_pred.category == k)]
    print(f"=== {k} | {c[:70]}")
    for r in g.itertuples():
        print(f"  GOLD [{r.start}-{r.end}]: {TEXT[c][r.start:r.end][:width]!r}")
    for r in p.itertuples():
        best = max([M.char_iou((r.start, r.end), (x.start, x.end)) for x in g.itertuples()], default=0)
        print(f"  PRED [{r.start}-{r.end}] conf {r.confidence:.2f} IoU {best:.2f}: {TEXT[c][r.start:r.end][:width]!r}")
    if p.empty:
        print("  PRED: none")
rng_ = np.random.default_rng(7)
cats_show = ["Governing Law", "Cap On Liability", "License Grant", "Non-Transferable License", "Exclusivity", "Change Of Control",
             "Liquidated Damages", "Renewal Term", "Audit Rights", "Competitive Restriction Exception"]
for k in cats_show:
    cands = gold_test[gold_test.category == k].contract.unique()
    show(cands[rng_.integers(len(cands))], k)
''')

interp("examples")

md(r"""
### 8.3 Five incorrect or low-quality outputs of the best model, with their causes
Selected from the taxonomy above (one of each frequent error type, plus a spurious prediction in a span-free contract);
the cause stated under each was checked by reading the excerpt.
""")

code(r'''
%%needs_models models/task2
tag_err = ERR[ERR.model == "t2_legalbert_tagger"]
picks = []
for kind, cause in [("false negative", "missed"), ("false negative", "boundary too short"), ("false negative", "split"),
                    ("false positive", "wrong category"), ("false positive", "spurious (span-free contract)"),
                    ("false negative", "merged")]:
    cand = tag_err[(tag_err.kind == kind) & tag_err.cause.str.startswith(cause)]
    if len(cand):
        picks.append(cand.sort_values(["category", "start"]).iloc[len(cand) // 2])
for i, r in enumerate(picks, 1):
    c, k = r.contract, r.category
    print(f"--- Error {i}: {r.kind}, cause = {r.cause}, category = {k}, contract = {c[:60]}")
    lo_, hi_ = max(0, int(r.start) - 120), int(r.end) + 120
    print(f"  excerpt: {TEXT[c][lo_:hi_][:600]!r}")
    g = gold_test[(gold_test.contract == c) & (gold_test.start < hi_) & (gold_test.end > lo_)]
    for x in g.itertuples():
        print(f"  gold   : {x.category} [{x.start}-{x.end}] {TEXT[c][x.start:x.end][:200]!r}")
    p = ttest_pred[(ttest_pred.contract == c) & (ttest_pred.start < hi_) & (ttest_pred.end > lo_)]
    for x in p.itertuples():
        print(f"  pred   : {x.category} [{x.start}-{x.end}] conf {x.confidence:.2f} {TEXT[c][x.start:x.end][:200]!r}")
# The cause of each error, checked by reading the excerpts above (interpretation below); saved for the report's table
VERIFIED = {"missed": "A follow-on sentence of a grant without the usual cue words (\"hereby grants\", \"license\"); the neighbouring "
                      "grant sentences were found.",
            "boundary too short": "The gold span starts at the heading and includes the payment table; the prediction starts one "
                                  "sentence later and stops before the table of amounts, so the overlap stays below IoU 0.5.",
            "split": "One non-compete sentence is predicted as two pieces: the middle tokens fell just below the inside threshold.",
            "wrong category": "Inside a long Cap On Liability clause, an extra commission payable on termination reads like a "
                              "liquidated-damages payment (money-on-termination wording).",
            "merged": "The annotators highlighted each sentence of the insurance clause separately; the tagger predicts one long span "
                      "because the BEGIN markers did not fire at the sentence starts."}
rows = []
for i, r in enumerate(picks, 1):
    key = next(k for k in VERIFIED if r.cause.startswith(k))
    span = re.sub(r"\s+", " ", TEXT[r.contract][int(r.start):int(r.end)]).strip()
    rows.append({"error": i, "category": r.category, "error type": f"{r.kind}: {r.cause}", "contract": r.contract,
                 "start": int(r.start), "end": int(r.end), "excerpt": span[:297] + ("..." if len(span) > 300 else span[297:]),
                 "verified cause": VERIFIED[key]})
pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "task2_error_examples.csv", index=False)
print(f"saved {len(rows)} examples -> results/task2_error_examples.csv")
''')

interp("errors")

code(r'''
%%needs_models models/task2
free = ttest_pred[ttest_pred.contract.isin(SPAN_FREE_TEST)]
print(f"Span-free test contracts: {len(SPAN_FREE_TEST)}; the tagger predicts nothing in {len(SPAN_FREE_TEST) - free.contract.nunique()} of them "
      f"and {len(free)} spurious spans in the other {free.contract.nunique()} (categories: {free.category.value_counts().to_dict()})")
''')


# =====================================================================================
md(r"""
## 9. DEMO — extract clauses from any contract text
`extract_clauses(text_or_title)` takes **any pasted contract text** (or the title of a CUAD contract), runs the saved tagger
with the validation-tuned thresholds and prints the clauses it finds, with category and confidence. If the model weights are
not present (e.g. on Colab without training), it says so and shows the **saved output of the real run**: the saved test
predictions for a CUAD test contract, or the saved live output for a text the demo has run before (`results/demo_cache.json`).
""")

code(r'''
# DEMO setup: everything the demo needs, loaded from saved files, so this group also runs on its own after the setup cell
RUN_DIR = ROOT / "models" / "task2" / json.load(open(C.RESULTS_DIR / "task2_choice.json"))["test_run"]
TH = json.load(open(C.RESULTS_DIR / "task2_thresholds.json"))
ttest_pred = pd.read_csv(C.RESULTS_DIR / "task2" / "t2_legalbert_tagger_test_spans.csv")
''')

code(r'''
_DEMO = {}
def extract_clauses(text_or_title, max_chars=220, show=True):
    if text_or_title in TEXT:
        title, text = text_or_title, TEXT[text_or_title]
    else:
        title, text = None, text_or_title
    t0 = time.perf_counter()
    if (RUN_DIR / "best").exists():
        if "model" not in _DEMO:
            from transformers import AutoModelForTokenClassification
            _DEMO["model"] = AutoModelForTokenClassification.from_pretrained(RUN_DIR / "best").to("cuda" if torch.cuda.is_available() else "cpu").eval()
        model = _DEMO["model"]; dev = next(model.parameters()).device
        ids, offs = D.tokenize_contract(text, tok)
        wins = D.make_windows(len(ids), 256, 128); probs = []
        with torch.no_grad():
            for s, e in wins:
                x = torch.tensor([[tok.cls_token_id] + ids[s:e] + [tok.sep_token_id]], device=dev)
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    p = torch.sigmoid(model(input_ids=x).logits.float())[0, 1:-1].cpu().numpy()
                probs.append(p)
        P = D.average_window_probs(len(ids), wins, probs, 2 * K)
        found = [(CATS[k], int(s), int(e), float(conf)) for k, s, e, conf in D.decode(P, offs, K, TH["theta_inside"], TH["theta_begin"], TH["min_tokens"])]
        src = "live model"
        C.demo_cache_put("task2", text, found)          # lets the demo show this real output where models/ is absent
    elif title is not None and title in set(ttest_pred.contract):
        found = [(r.category, r.start, r.end, r.confidence) for r in ttest_pred[ttest_pred.contract == title].itertuples()]
        src = "SAVED test predictions of the real run (model weights not available)"
    elif C.demo_cache_get("task2", text) is not None:
        found = [tuple(f) for f in C.demo_cache_get("task2", text)]
        src = "SAVED output of the live model from the real run (model weights not available)"
    else:
        print("Model weights not available and the text is not a CUAD test contract: nothing to show."); return [], "none"
    found = sorted(found, key=lambda x: x[1])
    if show:
        print(f"{len(found)} clauses found in {time.perf_counter() - t0:.1f} s ({src}); {len(text):,} characters")
        for cat, s, e, conf in found:
            print(f"  [{cat}] conf {conf:.2f}: {text[s:e][:max_chars]!r}")
    return found, src

sample = ("This Agreement shall be governed by and construed in accordance with the laws of the State of New York. "
          "Neither party may assign this Agreement without the prior written consent of the other party. "
          "In no event shall either party's aggregate liability under this Agreement exceed the fees paid in the twelve months "
          "preceding the claim. Licensor hereby grants to Licensee a non-exclusive, non-transferable license to use the Software.")
_ = extract_clauses(sample)
''')

code(r'''
# A real, unseen test contract, chosen by a fixed rule (never used for any decision): among the test contracts under
# 40,000 characters, the one with the most distinct gold clause types (ties: the shorter contract)
gold_t = spans[spans.split == "test"]
n_types = gold_t.groupby("contract").category.nunique()
cands = [c for c in SPLIT["test"] if len(TEXT[c]) < 40_000 and c in n_types.index]
demo_c = min(cands, key=lambda c: (-n_types[c], len(TEXT[c])))
gold_c = gold_t[gold_t.contract == demo_c]
found, src = extract_clauses(demo_c, show=False)
match = lambda k, s, e: any(M.char_iou((s, e), (a, b)) >= 0.5 for a, b in zip(gold_c[gold_c.category == k].start, gold_c[gold_c.category == k].end))
rows = [{"type": k, "confidence": round(conf, 2), "excerpt": re.sub(r"\s+", " ", TEXT[demo_c][s:e]).strip()[:110] + "...",
         "matches a gold span (IoU >= 0.5)": "yes" if match(k, s, e) else "no"} for k, s, e, conf in found]
print(f"Test contract: {demo_c[:80]} ({len(TEXT[demo_c]):,} characters; {len(gold_c)} gold spans of {gold_c.category.nunique()} types)")
print(f"{len(found)} clauses extracted ({src}); {sum(r['matches a gold span (IoU >= 0.5)'] == 'yes' for r in rows)} of them match a gold span")
with pd.option_context("display.max_colwidth", 120):
    display(pd.DataFrame(rows))
missed = sorted(set(gold_c.category) - {r["type"] for r in rows if r["matches a gold span (IoU >= 0.5)"] == "yes"})
print("Gold types without a matching extraction:", ", ".join(missed) if missed else "none")
# second case: a test contract with none of the 27 clause types (the shortest one)
empty_c = min(SPLIT["test"], key=lambda c: len(TEXT[c]))
print(f"A contract with none of the 27 types ({empty_c[:45]}..., {len(gold_t[gold_t.contract == empty_c])} gold spans): "
      f"{len(extract_clauses(empty_c, show=False)[0])} clauses returned")
''')

interp("demo")

md(r"""
## 10. Conclusion and limitations of Task 2
""")
interp("conclusion")

# =====================================================================================
md(r"""
## 11. Evaluation summary (Part C)
**Metrics and the cost of errors.** A reviewer who uses the extractor loses most when a clause is *not found* (it may never be
read); a spurious span costs a few seconds to dismiss. We therefore report entity-level recall and precision separately, use
CUAD's IoU ≥ 0.5 criterion (annotation boundaries are subjective — exact match is the strict secondary number), take
**macro-F1 over the 27 categories** as primary (rare risk categories count as much as frequent ones), and add CUAD-style
**AUPR and precision at 80 % / 90 % recall**, which describe the trade-off a reviewer faces when they want most clauses found.
The full justification for both tasks is in `STUDENTID_Task1.ipynb`, section 7.1.

**Second (small) ablation — INSIDE-only vs INSIDE+BEGIN labels.** The oracle check of section 2.5 isolates the BEGIN markers:
decoding the gold labels with BEGIN gives a ceiling of 0.998 macro-F1 at IoU 0.5, without BEGIN 0.932 — the BEGIN markers are
needed because touching same-category clauses would otherwise merge.
""")

code(r'''
%%needs_models models/task2
oracle_all = json.load(open(C.RESULTS_DIR / "task2_oracle.json"))
low = pd.read_csv(C.RESULTS_DIR / "task2_lowthr_test.csv")
T2_SUM = pd.DataFrame([
    {"model": "TF-IDF segment baseline", "macro-F1 @ IoU0.5": RES["t2_tfidf_segments"]["macro_f1"], "exact macro-F1": RES["t2_tfidf_segments"]["exact_macro_f1"],
     "CUAD-style AUPR": RES["t2_tfidf_segments"]["cuad_micro_aupr"], "max recall": max(r for r in CURVE["t2_tfidf_segments"]["recalls"] if r == r)},
    {"model": "Legal-BERT tagger (main result)", "macro-F1 @ IoU0.5": RES["t2_legalbert_tagger"]["macro_f1"], "exact macro-F1": RES["t2_legalbert_tagger"]["exact_macro_f1"],
     "CUAD-style AUPR": RES["t2_legalbert_tagger"]["cuad_micro_aupr"], "max recall": max(r for r in CURVE["t2_legalbert_tagger"]["recalls"] if r == r)},
    {"model": "ANALYSIS ONLY: low-threshold candidate list (ranking metrics)", "macro-F1 @ IoU0.5": RES["t2_legalbert_tagger_lowthr"]["macro_f1"],
     "exact macro-F1": RES["t2_legalbert_tagger_lowthr"]["exact_macro_f1"], "CUAD-style AUPR": RES["t2_legalbert_tagger_lowthr"]["cuad_micro_aupr"],
     "max recall": max(r for r in CURVE["t2_legalbert_tagger_lowthr"]["recalls"] if r == r)},
    {"model": "oracle ceiling, INSIDE+BEGIN (gold labels)", "macro-F1 @ IoU0.5": oracle_all["test | INSIDE+BEGIN | IoU>=0.5"]["macro_f1"]},
    {"model": "oracle ceiling, INSIDE only (gold labels)", "macro-F1 @ IoU0.5": oracle_all["test | INSIDE only | IoU>=0.5"]["macro_f1"]}])
display(T2_SUM.round(3))
T2_SUM.to_csv(C.RESULTS_DIR / "task2_evaluation_summary.csv", index=False)
print("Error analysis: section 8.1 (taxonomy, Figure 25), 8.2 (ten examples), 8.3 (five concrete errors with causes). "
      "Subgroups: section 6 (results/task2_subgroups.csv). Published comparison: section 7 (not like-for-like).")
''')

interp("evaluation")

# =====================================================================================
md(r"""
## 12. PART F — EXTRA CHALLENGING WORK: F2 cross-task pipeline (extract → verify → flag → review list)
**Idea.** Chain the project's models into the tool a reviewer would use on a new contract:
1. **extract** — the Task 2 tagger proposes spans with a category and a confidence *q* (mean INSIDE probability);
2. **verify** — the Task 1 Legal-BERT reads each span's text and gives *p* = p(category | span text); the combined confidence is
   **q^a · p^(1−a)** (a = 1 is the extractor alone, a = 0 trusts only the classifier), and spans below a threshold τ are dropped;
3. **flag** — the Part F1 headline anomaly scorer (`results/anomaly_choice.json`; thresholds at a 5 % flag rate on genuine
   validation clauses, recomputed here from the same cached representations) marks spans that read unusually for their
   predicted type;
4. **review list** — the kept spans of a contract, unusual ones first, then by combined confidence.

**Choice on validation only:** a ∈ {0, 0.25, 0.5, 0.75, 1} and τ ∈ {0, 0.1, …, 0.9} by validation macro-F1 at IoU 0.5 (ties →
larger a, then smaller τ, i.e. the simpler pipeline). The test contracts are scored **once** (`pipeline_extract_verify`).
""")

code(r'''
%%needs_models models/legalbert_lr5e-05_s42/best models/task2
from transformers import AutoModelForSequenceClassification
from src.preprocess import preprocess_for_transformer
from src.finetune import RECIPE as FT
from src import anomaly as AN
importlib.reload(AN)
CLS_DIR = ROOT / "models" / "legalbert_lr5e-05_s42" / "best"
cls_tok = AutoTokenizer.from_pretrained(CLS_DIR)
cls = AutoModelForSequenceClassification.from_pretrained(CLS_DIR).to("cuda").eval()
assert [cls.config.id2label[i] for i in range(K)] == CATS
def verify(texts):
    """Task 1 Legal-BERT on span texts -> (softmax probabilities, mean-pooled last-layer embeddings)."""
    P, E = [], []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = cls_tok([preprocess_for_transformer(t) for t in texts[i:i + 64]], truncation=True, max_length=FT["max_length"],
                          padding=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = cls(**enc, output_hidden_states=True)
            m = enc["attention_mask"].unsqueeze(-1).float(); h = out.hidden_states[-1].float()
            P.append(torch.softmax(out.logits.float(), -1).cpu().numpy()); E.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    return np.concatenate(P), np.concatenate(E)

# anomaly scorer of Part F1 (fit on TRAIN clause embeddings; thresholds from genuine VALIDATION clauses)
an_choice = json.load(open(C.RESULTS_DIR / "anomaly_choice.json"))
assert an_choice["headline"] == "legalbert_ft_knn", "the pipeline implements the embedding-kNN headline scorer"
FE = np.load(ROOT / "models" / "anomaly_cache" / "features.npz")
t1 = C.load_clauses()
KNN = AN.KNNScorer(FE["emb_ft"][(t1.split == "train").values], t1[t1.split == "train"].category.values, CATS)
vmask = (t1.split == "validation").values
AN_THR = AN.thresholds_at(KNN.score(FE["emb_ft"][vmask], t1[vmask].category.values), t1[vmask].category.values, CATS, 0.05)

def enrich(pred):
    """Add q, p(category | span text) and the anomaly score (against the predicted category) to each predicted span."""
    t0 = time.perf_counter()
    texts = [TEXT[r.contract][r.start:r.end] for r in pred.itertuples()]
    P, E = verify(texts) if len(texts) else (np.zeros((0, K)), np.zeros((0, 768)))
    out = pred.assign(q=pred.confidence.values, p=P[np.arange(len(pred)), [CIDX[c] for c in pred.category]],
                      top_category=[CATS[i] for i in P.argmax(1)],
                      anomaly=KNN.score(E, pred.category.values) if len(pred) else [])
    out["unusual"] = AN.flagged(out.anomaly.values, out.category.values, AN_THR)
    return out, time.perf_counter() - t0
torch.cuda.synchronize()
VE, v_sec = enrich(decode_contracts(VPROBS.get, SPLIT["validation"], TH["theta_inside"], TH["theta_begin"], TH["min_tokens"]))
def apply(E_, a, tau):
    comb = E_.q.values ** a * E_.p.values ** (1 - a)
    keep = comb >= tau
    return E_[keep].assign(confidence=comb[keep])
grid_p = []
for a in [0, 0.25, 0.5, 0.75, 1.0]:
    for tau in np.round(np.arange(0, 1.0, 0.1), 1):
        pv = apply(VE, a, tau)
        sm = M.summarise(M.count_table(gold_val, pv[["contract", "category", "start", "end", "confidence"]], SPLIT["validation"], CATS), CATS)
        grid_p.append({"a": a, "tau": tau, "val macro-F1 @ IoU0.5": sm["macro_f1"], "val macro P": sm["macro_precision"], "val macro R": sm["macro_recall"],
                       "kept spans": len(pv)})
GP = pd.DataFrame(grid_p); GP.to_csv(C.RESULTS_DIR / "pipeline_validation_grid.csv", index=False)
best_p = GP.sort_values(["val macro-F1 @ IoU0.5", "a", "tau"], ascending=[False, False, True]).iloc[0]
A_, TAU_ = float(best_p.a), float(best_p.tau)
base_v = GP[(GP.a == 1.0) & (GP.tau == 0.0)].iloc[0]["val macro-F1 @ IoU0.5"]
print(f"Chosen on validation: a = {A_}, tau = {TAU_} -> macro-F1 {best_p['val macro-F1 @ IoU0.5']:.4f} (extractor alone: {base_v:.4f})")
display(GP.pivot_table(index="a", columns="tau", values="val macro-F1 @ IoU0.5").round(3))
json.dump({"a": A_, "tau": TAU_, "val_macro_f1": float(best_p["val macro-F1 @ IoU0.5"]), "val_macro_f1_extractor_alone": float(base_v),
           "anomaly_scorer": an_choice["headline"]}, open(C.RESULTS_DIR / "pipeline_choice.json", "w"), indent=2)
''')

code(r'''
%%needs_models models/legalbert_lr5e-05_s42/best models/task2
# TEST (once): pipeline with the chosen (a, tau), compared with the extractor alone (= section 6 result)
torch.cuda.synchronize(); TE, t_sec = enrich(ttest_pred)
pipe_pred = apply(TE, A_, TAU_)
pipe_spans = pipe_pred[["contract", "category", "start", "end", "confidence"]].reset_index(drop=True)
RES["pipeline"], CNT["pipeline"], SUMM["pipeline"], CURVE["pipeline"], PCAUPR["pipeline"] = full_metrics(pipe_spans)
C.test_once_spans("pipeline_extract_verify", {"a": A_, "tau": TAU_, "extractor": TEST_RUN, "verifier": "legalbert_lr5e-05_s42",
                                              "anomaly": an_choice["headline"], "thresholds_extractor": TH}, RES["pipeline"], pipe_spans)
PCMP = pd.DataFrame([{"system": lab, "macro-F1 @ IoU0.5": RES[n]["macro_f1"], "macro P": RES[n]["macro_precision"], "macro R": RES[n]["macro_recall"],
                      "micro-F1": RES[n]["micro_f1"], "CUAD-style AUPR": RES[n]["cuad_micro_aupr"],
                      "max recall": max(r for r in CURVE[n]["recalls"] if r == r)}
                     for lab, n in [("extractor alone (t2_legalbert_tagger)", "t2_legalbert_tagger"), (f"extract + verify (a={A_}, tau={TAU_})", "pipeline")]])
pb_pipe = M.paired_bootstrap_macro_f1(CNT["pipeline"], CNT["t2_legalbert_tagger"], CATS)
display(PCMP.round(4))
print(f"Pipeline - extractor alone (macro-F1 @ IoU0.5): {pb_pipe['diff']:+.4f} [{pb_pipe['ci_low']:+.4f}, {pb_pipe['ci_high']:+.4f}]")
PCMP.to_csv(C.RESULTS_DIR / "pipeline_test_comparison.csv", index=False)
# ANALYSIS ONLY (no choice depends on it): ranking by the combined confidence for each a, keeping every span (tau = 0), so
# macro-F1 is unchanged and only the confidence ranking (CUAD-style AUPR) shows what verification does
keys_t = [(c, k) for c in SPLIT["test"] for k in CATS]
AR = pd.DataFrame([{"a (1 = extractor alone)": a_, "CUAD-style AUPR on test": M.cuad_curve(gold_test, apply(TE, a_, 0.0)[["contract", "category", "start", "end", "confidence"]],
                                                                                         text_of, keys_t)["aupr"]} for a_ in [0, 0.25, 0.5, 0.75, 1.0]])
print("Analysis only - test AUPR when spans are ranked by q^a * p^(1-a):"); display(AR.round(4))
AR.to_csv(C.RESULTS_DIR / "pipeline_aupr_by_a_test_analysis.csv", index=False)

# Error propagation: Task 1 classifier on GOLD clause texts vs on CORRECTLY LOCATED predicted spans (single-category gold spans)
gt1 = gold_test[gold_test.n_categories_on_span == 1].reset_index(drop=True)
Pg, _ = verify([TEXT[r.contract][r.start:r.end] for r in gt1.itertuples()])
gold_pred = [CATS[i] for i in Pg.argmax(1)]
loc = []
for r in TE.itertuples():
    g = gt1[(gt1.contract == r.contract) & (gt1.category == r.category)]
    if len(g) and max(M.char_iou((r.start, r.end), (a, b)) for a, b in zip(g.start, g.end)) >= 0.5:
        loc.append((r.category, r.top_category))
loc = pd.DataFrame(loc, columns=["true", "pred"])
from sklearn.metrics import f1_score as _f1
EP = pd.DataFrame([{"input to the Task 1 classifier": "gold clause texts (single-category test spans)", "spans": len(gt1),
                    "macro-F1": _f1(gt1.category, gold_pred, labels=CATS, average="macro", zero_division=0),
                    "accuracy": float(np.mean(np.array(gold_pred) == gt1.category.values))},
                   {"input to the Task 1 classifier": "correctly located predicted spans (IoU >= 0.5)", "spans": len(loc),
                    "macro-F1": _f1(loc.true, loc.pred, labels=CATS, average="macro", zero_division=0),
                    "accuracy": float((loc.true == loc.pred).mean())}])
display(EP.round(4))
# multi-category gold spans: a single-label verifier spreads probability over the span's categories
gm = gold_test[gold_test.n_categories_on_span > 1]
hurt, matched = 0, 0
for r in gm.itertuples():
    cand = TE[(TE.contract == r.contract) & (TE.category == r.category)]
    hit = [(M.char_iou((c.start, c.end), (r.start, r.end)), c) for c in cand.itertuples()]
    hit = [c for iou, c in hit if iou >= 0.5]
    if hit:
        matched += 1
        hurt += int(all(c.q ** A_ * c.p ** (1 - A_) < TAU_ for c in hit))
print(f"Multi-category gold (span, category) annotations in test: {len(gm)}; found by the extractor: {matched}; "
      f"dropped by single-label verification: {hurt}")
lat = (json.load(open(RUN_DIR / "test_predict_seconds.json"))["seconds"] + t_sec) / len(SPLIT["test"])
prices = json.load(open(C.RESULTS_DIR / "llm_price_sources.json"))["primary"]
cost = lat * 1000 / 3600 * prices["g6.xlarge"]["usd_per_hour"]
print(f"End-to-end latency per contract on the GPU: extractor {json.load(open(RUN_DIR / 'test_predict_seconds.json'))['seconds'] / len(SPLIT['test']):.3f} s "
      f"+ verify/flag {t_sec / len(SPLIT['test']):.3f} s = {lat:.3f} s -> cost per 1,000 contracts ${cost:.4f} (AWS g6.xlarge price, read 2026-10-22)")
json.dump({"test": PCMP.to_dict(orient="records"), "paired_pipeline_minus_extractor": pb_pipe, "error_propagation": EP.to_dict(orient="records"),
           "multi_category": {"annotations": int(len(gm)), "found": matched, "dropped_by_verification": hurt},
           "latency_seconds_per_contract": lat, "cost_per_1000_contracts_usd": cost, "flagged_share_of_kept_spans": float(pipe_pred.unusual.mean())},
          open(C.RESULTS_DIR / "pipeline_results.json", "w"), indent=2, default=float)
''')

interp("pipeline")

md(r"""
### 12.1 DEMO — the review list for one unseen test contract
Without `models/` (e.g. on Colab) the cell shows the saved review list of the real run (`results/task2_review_demo.csv`) and says so.
""")

code(r'''
%%needs_models models/legalbert_lr5e-05_s42/best models/task2 --saved=results/task2_review_demo.csv
lens = pd.Series({c: len(TEXT[c]) for c in SPLIT["test"]})
demo_c = lens.sort_values().index[len(lens) // 2]          # the median-length test contract (never used for any choice)
t0 = time.perf_counter()
dp = decode_contracts(TPROBS.get, [demo_c], TH["theta_inside"], TH["theta_begin"], TH["min_tokens"]) if TPROBS is not None else ttest_pred[ttest_pred.contract == demo_c]
de, _ = enrich(dp)
de = de.assign(combined=de.q ** A_ * de.p ** (1 - A_))
de["verified"] = de.combined >= TAU_
review = de[de.verified].sort_values(["unusual", "combined"], ascending=[False, False])
gold_c = gold_test[gold_test.contract == demo_c]
review = review.assign(clause=[TEXT[demo_c][s:e][:140].replace("\n", " ") + "..." for s, e in zip(review.start, review.end)],
                       in_gold=[any(M.char_iou((s, e), (a, b)) >= 0.5 for a, b in zip(gold_c[gold_c.category == k].start, gold_c[gold_c.category == k].end))
                                for s, e, k in zip(review.start, review.end, review.category)])
print(f"Contract: {demo_c[:90]} ({len(TEXT[demo_c]):,} characters); {len(dp)} extracted spans, {len(review)} kept after verification, "
      f"{int(review.unusual.sum())} flagged unusual; built in {time.perf_counter() - t0:.2f} s (probabilities cached)")
pd.set_option("display.max_colwidth", 150)
display(review[["category", "clause", "q", "p", "combined", "verified", "unusual", "in_gold"]].round(3))
review[["category", "clause", "q", "p", "combined", "verified", "unusual", "in_gold"]].round(3).to_csv(
    C.RESULTS_DIR / "task2_review_demo.csv", index=False)      # shown instead when models/ is absent (e.g. on Colab)
print(f"Gold annotations in this contract: {len(gold_c)} (categories: {sorted(gold_c.category.unique())})")
''')

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}, "colab": {"provenance": []}}
nbf.write(nb, OUT)
print("wrote", OUT, len(cells), "cells")
