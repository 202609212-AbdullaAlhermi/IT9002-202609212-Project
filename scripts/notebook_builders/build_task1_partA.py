"""Generate notebooks/STUDENTID_Task1.ipynb (design v2). This file writes sections 0-3 and then runs the builders
for section 4 (build_task1_partB.py) and section 5 (build_task1_partC.py).

    python scripts/notebook_builders/build_task1_partA.py      (outputs are cleared; execute the notebook afterwards)
"""
from pathlib import Path

import sys
import nbformat as nbf
sys.path.insert(0, str(Path(__file__).parent))
from colab_cell import COLAB_MD, COLAB_CODE  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]   # project folder
OUT = ROOT / "notebooks" / "STUDENTID_Task1.ipynb"
INCLUDE_SECTION5 = True    # section 5 (transformer + ablation)
INTERP = {}  # interpretation paragraphs, written after looking at the real outputs
interp_file = Path(__file__).with_name("task1_interp.py")
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
## Notebook 1 of 3: Task 1 — Clause-type classification of commercial contracts (CUAD, 27 classes)

**Student:** see the setup cell below, which reads `project_info.json` · **Course:** IT9002 Natural Language Processing

**Dataset:** Contract Understanding Atticus Dataset (CUAD) v1 — https://www.atticusprojectai.org/cuad
(data and code: https://github.com/TheAtticusProject/cuad), licence CC BY 4.0.

| Section | Content | Rubric |
|---|---|---|
| 0 | How to run, setup | — |
| 1 | Problem statement and measurable objectives | Part A, problem statement |
| 2 | Dataset: leakage check and confusion hypotheses first, then source, curation, split, distributions | Part A, dataset |
| 3 | Preprocessing study (classical models), transformer input, Task 2 input | Part A, preprocessing |
| 4 | Classical baselines: majority class + three TF-IDF classifiers | Part B, Task 1 classical |
| 5 | Fine-tuned transformer (Legal-BERT) and the controlled ablation (BERT-base) | Part B, Task 1 transformer; Part C ablation |

Task 2 (clause extraction) has its own notebook (`STUDENTID_Task2.ipynb`), and the prompted LLMs are in `STUDENTID_LLM.ipynb`.

### How to run
* **Locally:** open this notebook from the project folder with the project's Python environment (`.venv`)
  and choose *Run All*. The data is already in `data/processed/`.
* **Google Colab:** *File → Open notebook → GitHub*, paste the repository URL, open this notebook and choose
  *Runtime → Run all*. The setup cell detects Colab, downloads the project code and data from GitHub and
  checks the data files against their SHA-256 checksums before anything else runs.
* Heavy steps have a switch (`RUN_SEARCH`, `RETRAIN`). With `False` the notebook loads the saved results of the real
  run, so it can be re-run in a few minutes. With `True` everything is recomputed.
""")

md(COLAB_MD)
code(COLAB_CODE)

md(r"""
## 0. Setup

**What these cells do:** find the project folder (on Colab the cell above has fetched it), make our helper code in
`src/` importable, fix the random seed (42), print the software and hardware we use (the report must state them),
and load the frozen dataset after checking its checksums.
""")

code(r'''
import sys, os, json, platform, subprocess, importlib, re, warnings, contextlib, io
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules or os.environ.get("IT9002_SIMULATE_COLAB") == "1"
# The project folder = the first folder, going upwards, that contains project_info.json (works from notebooks/, from the
# project folder and on Colab, where the cell above has moved into the project).
ROOT = next(d for d in [Path.cwd(), *Path.cwd().parents] if (d / "project_info.json").exists())
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# ---- 2. Our helper modules (src/common.py, src/preprocess.py, src/plotting.py) ----
from src import common as C
from src import preprocess as P
from src import plotting as PL
importlib.reload(C); importlib.reload(P); importlib.reload(PL)

INFO = C.load_project_info()
C.register_needs_models()   # defines %%needs_models: cells that need models/ (not in the ZIP) skip cleanly if it is missing
print(f"Student: {INFO['STUDENT_NAME']}  |  ID: {INFO['STUDENT_ID']}  |  {INFO['COURSE_CODE']} {INFO['COURSE_NAME']}")
print("Dataset:", INFO["DATASET_URL"], "|", INFO["DATASET_CODE_URL"])

# ---- 3. Reproducibility ----
C.set_seed(42)
PL.setup_style()

# ---- 4. Software and hardware (the report must state these) ----
warnings.filterwarnings("ignore", message="IProgress not found")
import numpy as np, pandas as pd, sklearn, matplotlib, nltk, torch, transformers
transformers.logging.set_verbosity_error()   # hides "sequence longer than 512" notices while we only COUNT tokens
print(f"Python {platform.python_version()} on {platform.platform()}  |  Colab: {IN_COLAB}")
print(f"numpy {np.__version__} | pandas {pd.__version__} | scikit-learn {sklearn.__version__} | "
      f"nltk {nltk.__version__} | torch {torch.__version__} | transformers {transformers.__version__}")
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU only)")
P.ensure_nltk_data()   # stop-word list + WordNet; downloads them once if missing (also behind Colab's proxy)
''')

code(r'''
# ---- 5. Data: use the frozen files; if missing, fetch or rebuild them, then verify the checksums ----
PROCESSED = ROOT / "data" / "processed"
# Frozen CSVs: already here, or downloaded from the project's GitHub repository, or rebuilt from the official CUAD release
# (scripts/download_cuad.py + scripts/build_dataset_v2.py); in every case verified against the frozen SHA-256 checksums.
C.ensure_data(INFO["GITHUB_URL"])
df = C.load_clauses()
STATS = C.load_build_stats()               # every dataset number quoted below comes from here or from the CSVs
LABELS = C.get_labels(df)
ROBUST = C.robust_labels(df)               # classes with >= 15 test clauses (robustness number, see 2.8)
train, val, test = (df[df.split == s] for s in ["train", "validation", "test"])
print(f"{len(df):,} clauses | {len(LABELS)} classes | train {len(train):,} / validation {len(val):,} / test {len(test):,}")
df.head(3)
''')

# =====================================================================================
md(r"""
## 1. Problem statement

**Domain.** Legal and policy documents — commercial contract review. The corpus consists of commercial contracts
filed with the U.S. Securities and Exchange Commission (SEC).

**The problem.** Commercial contracts are long (section 2.4 measures them: thousands of words each), yet a reviewer's
job is to find a small number of risk-bearing clauses — for example a cap on liability, an audit right, a
change-of-control provision, the governing law — check each one against the organisation's playbook, and notice
wording that is unusual. Recognising *what legal function a passage performs* and *where it is* is slow, repetitive
expert work.

**Questions addressed — two tasks that build on each other.**
1. *Task 1 (this notebook): what type is this clause?* Given the text of one clause, can a model recognise which of
   the expert-defined clause types it is, and how do classical TF-IDF classifiers, a fine-tuned transformer and two
   prompted LLMs compare in accuracy, cost and latency?
2. *Task 2 (`STUDENTID_Task2.ipynb`): where in the contract are the clauses of each type?* Given a whole contract, find
   the spans that contain a clause of each type (clause extraction as sequence labelling).

Task 1 assumes the clause has already been located; Task 2 locates it. Part F chains them: extract a clause
(Task 2), confirm its type (Task 1), and flag it if it reads unusually for that type.

**Stakeholders.** In-house counsel; contract-management and compliance teams; legal-technology tools (clause search,
contract-review checklists) that support them.

**Intended use.** *Decision support for a human reviewer* — find the clause, name its type, point out unusual wording
— **not** legal advice. Every prediction is meant to be checked by a person. This matters for the choice of metric:
a clause type that the model systematically misses is costlier than a false flag that a reviewer dismisses in seconds,
and rare clause types matter as much as common ones. The primary metric is therefore **macro-F1** (the unweighted mean
of the per-class F1 scores), backed by per-class recall.

**Why clause level for Task 1.** Classifying whole contracts by type was rejected by the tutor (too few contracts per
type, the type is usually in the title, contracts are far longer than a transformer's input window). One clause is
short (section 2.9), so every model can read the whole input.

### Measurable objectives
These replace objectives O1–O4 of the approved proposal form after the tutor's feedback (deviation D6
in the report). Each will be marked **met / not met** at the end, using only saved result files.

| ID | Objective | Measured by | Met if … |
|---|---|---|---|
| **O1** | All Task 1 models are scored on the **same held-out test set** (the test clauses of the 93 official CUAD test contracts that contain a single-label clause) with macro-F1; Task 2 models on all 102 official test contracts | `results/test_eval_log.csv` | each model has exactly one test evaluation, on the same clauses / contracts |
| **O2** | The best TF-IDF classifier beats the majority-class baseline **by a wide margin** | test macro-F1 | gap ≥ **0.50** macro-F1 (only reachable by reading the clauses) |
| **O3** | The fine-tuned transformer is compared with the best classical model and with **two** prompted LLMs on **macro-F1, cost per 1,000 clauses and latency** | comparison table, `results/compute_budget.csv` | all three quantities reported; *H1: Legal-BERT > best TF-IDF* is supported only if the paired contract-level bootstrap 95 % CI of the macro-F1 difference lies above 0 |
| **O4** | Task 2: the Legal-BERT extractor beats the TF-IDF segment baseline on macro entity-F1 at IoU ≥ 0.5, and is compared with the CUAD paper's published AUPR on the same test split | Task 2 notebook | paired contract-level bootstrap CI above 0; comparison reported |
| **O5** | (self-imposed, exploratory) **no clause type is systematically missed** | per-class test recall of the best Task 1 model | recall ≥ **0.70** for every class (expected to fail for some rare classes; reported honestly) |
| **O6** | (Part F) the best anomaly scorer detects injected wrong-type clauses | AUROC with 95 % CI | CI above 0.5; the human part is "not assessed" unless the student fills the blind-review sheet |
""")

# =====================================================================================
md(r"""
## 2. Dataset

The section opens with the two checks that shape everything after it: **why the data is split by contract**
(a measured leakage check) and **which clause types we expect to be confused** (hypotheses written down before any
model is trained, tested in section 4). The descriptive part (source, curation, split sizes, distributions) follows.

### 2.1 Why split by contract? A data-driven leakage check
**Hypothesis:** with a random *clause-level* split, near-identical clauses from the same contract (or the same
template) land on both sides, so the held-out score is inflated.

**Method (test contracts never touched):** we take only the train + validation contracts and, five times
(seeds 0–4), hold out 20 % in two ways — (a) a random clause-level split (stratified by class), (b) a contract-level
split (`GroupShuffleSplit`). For each held-out clause we find its most similar training clause (cosine similarity of
TF-IDF vectors) and count **near-duplicates (similarity ≥ 0.9)** and **nearest neighbours from the same contract**;
and we train the same default TF-IDF + LinearSVC on each split and score macro-F1 on the held-out part.
""")

code(r'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

pool = pd.concat([train, val]).reset_index(drop=True)           # NO test contracts in this experiment
Xpool = TfidfVectorizer().fit_transform(pool.text)              # rows are L2-normalised -> dot product = cosine

rows = []
for seed in range(5):
    tr_r, ho_r = train_test_split(np.arange(len(pool)), test_size=0.2, random_state=seed, stratify=pool.category)
    tr_g, ho_g = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                      .split(pool, groups=pool.contract))
    for kind, tr_i, ho_i in [("random clause-level", tr_r, ho_r), ("contract-level", tr_g, ho_g)]:
        sim = (Xpool[ho_i] @ Xpool[tr_i].T).toarray()
        best = sim.argmax(axis=1)
        same_contract = (pool.contract.values[tr_i][best] == pool.contract.values[ho_i]).mean()
        vec = TfidfVectorizer().fit(pool.text.iloc[tr_i])
        clf = LinearSVC(random_state=42).fit(vec.transform(pool.text.iloc[tr_i]), pool.category.iloc[tr_i])
        pred = clf.predict(vec.transform(pool.text.iloc[ho_i]))
        rows.append({"seed": seed, "split": kind, "held_out": len(ho_i),
                     "near_duplicate_share": (sim.max(axis=1) >= 0.9).mean(),
                     "nearest_neighbour_same_contract": same_contract,
                     "macro_f1": f1_score(pool.category.iloc[ho_i], pred, labels=LABELS, average="macro",
                                          zero_division=0)})
leak = pd.DataFrame(rows)
leak.to_csv(C.RESULTS_DIR / "leakage_check.csv", index=False)
leak_sum = leak.groupby("split")[["held_out", "near_duplicate_share", "nearest_neighbour_same_contract", "macro_f1"]].agg(["mean", "std"])
display(leak_sum.round(3))
''')

code(r'''
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
order = ["random clause-level", "contract-level"]
cols = [PL.ORANGE, PL.BLUE]
for ax, (col, title, scale) in zip(axes, [
        ("near_duplicate_share", "Held-out clauses with a\nnear-duplicate in train (%)", 100),
        ("nearest_neighbour_same_contract", "Nearest training clause comes\nfrom the same contract (%)", 100),
        ("macro_f1", "Held-out macro-F1\n(TF-IDF + LinearSVC)", 1)]):
    m = leak.groupby("split")[col].mean().loc[order] * scale
    s = leak.groupby("split")[col].std().loc[order] * scale
    ax.bar(order, m, yerr=s, color=cols, width=0.6, capsize=4, edgecolor="white")
    for i, (v, e) in enumerate(zip(m, s)):
        ax.text(i, v + e, f"{v:.3f}" if scale == 1 else f"{v:.1f}", ha="center", va="bottom", fontsize=9,
                color=PL.INK)
    ax.margins(y=0.12)
    ax.set_title(title, fontsize=10); ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelsize=8)
axes[2].set_ylim(0.6, 1.0)
fig.suptitle("Figure 1. Leakage check on train+validation contracts: clause-level vs contract-level split "
             "(mean ± sd over 5 seeds)", fontweight="bold", fontsize=11, y=1.04)
PL.savefig(fig, 1, "leakage_check"); plt.show()
''')

interp("leakage")

md(r"""
### 2.2 New classes and the confusions we expect (hypotheses, written before training)
Following the tutor's feedback, the label set grew from the first design to 27 classes by lowering the minimum class
size (section 2.5). The added classes sit right next to existing ones in meaning, which is exactly what makes the
task harder. We write down **which confusions we expect** now, and test them in section 4 against the confusion
pairs of the best classical model on **validation**.

As a first data check (training clauses only), we measure how similar each new class is to its hypothesised
neighbours: the cosine similarity between the classes' average TF-IDF vectors (centroids). A high similarity means a
bag-of-words model will find the pair hard.
""")

code(r'''
NEW_CLASSES = ["Competitive Restriction Exception", "Joint Ip Ownership", "Liquidated Damages",
               "No-Solicit Of Employees", "Non-Disparagement", "Non-Transferable License", "Source Code Escrow"]
assert set(NEW_CLASSES) <= set(LABELS)
HYPOTHESES = [  # (new class, existing class it is likely to be confused with, why)
    ("Non-Transferable License", "License Grant", "both describe the licence that is granted"),
    ("Non-Transferable License", "Anti-Assignment", "both restrict transfer / assignment"),
    ("No-Solicit Of Employees", "Non-Compete", "both are post-contract restrictive covenants"),
    ("Competitive Restriction Exception", "Exclusivity", "an exception is written inside the exclusivity clause"),
    ("Competitive Restriction Exception", "Non-Compete", "an exception is written inside the non-compete clause"),
    ("Liquidated Damages", "Cap On Liability", "both fix an amount payable on breach"),
    ("Joint Ip Ownership", "Ip Ownership Assignment", "both decide who owns intellectual property"),
    ("Source Code Escrow", "License Grant", "escrow release usually triggers a licence to the code"),
    ("Non-Disparagement", "Covenant Not To Sue", "both restrict what one party may say or do against the other"),
]
vec = TfidfVectorizer(sublinear_tf=True, min_df=2).fit(train.text)
cent = {lab: np.asarray(vec.transform(train.text[train.category == lab]).mean(axis=0)).ravel() for lab in LABELS}
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
hyp = pd.DataFrame([(n, e, why, cos(cent[n], cent[e]),
                     np.median([cos(cent[n], cent[o]) for o in LABELS if o != n]))
                    for n, e, why in HYPOTHESES],
                   columns=["new class", "likely confused with", "why", "centroid cosine", "median cosine to all other classes"])
hyp.to_csv(C.RESULTS_DIR / "confusion_hypotheses.csv", index=False)
pd.set_option("display.max_colwidth", 70)
display(hyp.round(3))

# Most similar class for every new class (data, not hypothesis)
nearest = {n: max((o for o in LABELS if o != n), key=lambda o: cos(cent[n], cent[o])) for n in NEW_CLASSES}
print("Nearest class by TF-IDF centroid (training clauses):")
for n, o in nearest.items():
    print(f"  {n:<35} -> {o:<28} cosine {cos(cent[n], cent[o]):.3f}")

fig, ax = plt.subplots(figsize=(9, 4.2))
lab = [f"{n}  vs  {e}" for n, e in zip(hyp["new class"], hyp["likely confused with"])][::-1]
yy = np.arange(len(lab))
ax.barh(yy, hyp["centroid cosine"][::-1], color=PL.BLUE, height=0.6, label="hypothesised pair")
ax.scatter(hyp["median cosine to all other classes"][::-1], yy, color=PL.ORANGE, marker="|", s=300, linewidths=2.5,
           zorder=3, label="median similarity of the new class to all other classes")
ax.set_yticks(yy); ax.set_yticklabels(lab, fontsize=8.5); ax.set_xlim(0, 1)
ax.set_xlabel("Cosine similarity of TF-IDF class centroids (training clauses)"); ax.grid(axis="y", visible=False)
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
ax.set_title("Figure 2. Confusion hypotheses for the new classes: similarity to the expected neighbour", fontsize=11)
PL.savefig(fig, 2, "confusion_hypotheses"); plt.show()
''')

interp("hypotheses")

md(r"""
### 2.3 Source, licence, size and modality

| Property | Value |
|---|---|
| Name | Contract Understanding Atticus Dataset (CUAD) v1 |
| Citation | Hendrycks, D., Burns, C., Chen, A., & Ball, S. (2021). CUAD: An expert-annotated NLP dataset for legal contract review. *Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, 1.* |
| Paper | https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/6ea9ab1baa0efb9e19094440c317e21b-Abstract-round1.html |
| Dataset page / data | https://www.atticusprojectai.org/cuad · https://github.com/TheAtticusProject/cuad (`data.zip`) |
| Licence | CC BY 4.0 (free to use, including academic work, with attribution) |
| Modality | Text (English) |
| Original size and our Task 1 / Task 2 data | printed by the cell below from `build_stats_v2.json` and the CSVs |
""")

code(r'''
spans2 = C.load_task2_spans()
csplit = C.load_contract_split()
raw_contracts = json.load(open(C.RAW_CUAD, encoding="utf-8"))["data"]
all_cats = {re.search(r'"([^"]+)"', q["question"]).group(1) for c in raw_contracts for q in c["paragraphs"][0]["qas"]}
print(f"CUAD v1: {STATS['n_contracts']} contracts ({STATS['n_official_test_contracts']} official test contracts), "
      f"{len(all_cats)} clause categories, {STATS['spans_total']:,} expert-labelled spans")
print(f"Task 1: {STATS['task1_clauses']:,} single-label clauses in {len(STATS['kept_categories'])} classes")
print(f"Task 2: {len(spans2):,} (span, category) annotations on "
      f"{spans2.drop_duplicates(['contract', 'start', 'end']).shape[0]:,} distinct spans, in {csplit.contract.nunique()} contracts")
print("Keys available in build_stats_v2.json:", ", ".join(sorted(STATS)))
''')

md(r"""
### 2.4 How clauses are extracted from the original files
CUAD is distributed in the SQuAD question-answering format: each contract is one long text, and for each of the
41 categories there is a "question" (e.g. *Highlight the parts (if any) of this contract related to "Governing
Law"…*) whose "answers" are the character spans that the lawyers highlighted. **Each highlighted span becomes one
clause in Task 1, labelled with the category in the question; Task 2 keeps the span's character offsets in the full
contract.** The cell below shows one real example.
""")

code(r'''
c0 = raw_contracts[0]
qa = next(q for q in c0["paragraphs"][0]["qas"] if q["answers"])
print("Contract title     :", c0["title"])
print("Contract length    :", f'{len(c0["paragraphs"][0]["context"].split()):,} words')
print("Question (category):", qa["question"][:160], "...")
print("First answer span  :", repr(qa["answers"][0]["text"][:200]))
print("answer_start       :", qa["answers"][0]["answer_start"], "(character offset in the contract)")
contract_words = [len(c["paragraphs"][0]["context"].split()) for c in raw_contracts]
print(f"\nWhole contracts: median {int(np.median(contract_words)):,} words, max {max(contract_words):,} "
      "-> far too long for a 512-token transformer; one clause is not.")
''')

md(r"""
### 2.5 Curation log (datasheet style)
The dataset was built once by `scripts/build_dataset_v2.py` and frozen. The cell below reads the curation counts it
saved (`build_stats_v2.json`), re-runs the build script into `data/processed/_rebuild_v2/` (never over the frozen
files), and checks that the rebuilt data is identical to the frozen data.

Curation decisions and why:
1. **Drop 4 entity/date categories** (Parties, Document Name, Agreement Date, Effective Date): their spans are short
   names or dates, not clauses, so "classifying" them would be a different task (entity recognition).
2. **Minimum class size — at least 60 unambiguous clauses per category**, applied *iteratively* (drop the weakest
   category, recompute, repeat), because dropping a category can turn an ambiguous span back into an unambiguous one.
   The threshold is lower than in the first design (which kept fewer, larger classes) because the tutor asked for a
   broader range of categories; 60 still leaves every class with training examples and at least a few test clauses.
3. **Hold out ambiguous clauses from Task 1** (section 2.6).
4. **Whitespace normalisation** of the clause text (line breaks and runs of spaces collapsed; nothing else changed).
   Task 2 keeps the original contract text and character offsets untouched.
5. **Remove exact duplicates of training text from validation/test**, so no model can score by recognising
   boilerplate it has already seen verbatim.
""")

code(r'''
ENTITY_CATS = ["Parties", "Document Name", "Agreement Date", "Effective Date"]
def span_count(cats):
    return sum(len(q["answers"]) for c in raw_contracts for q in c["paragraphs"][0]["qas"]
               if re.search(r'"([^"]+)"', q["question"]).group(1) in cats)
n_after_entity = span_count(all_cats - set(ENTITY_CATS))
n_kept_spans = span_count(set(STATS["kept_categories"]))
steps = pd.DataFrame([
    ("All expert-labelled spans in CUAD v1", STATS["spans_total"], len(all_cats)),
    ("Drop the 4 entity/date categories", n_after_entity, len(all_cats) - len(ENTITY_CATS)),
    (f"Iterative rule: keep categories with >= 60 unambiguous clauses ({len(STATS['dropped_categories'])} dropped)",
     n_kept_spans, len(STATS["kept_categories"])),
    ("Keep single-label clauses (ambiguous multi-label clauses held out)", STATS["single_label_before_dedup"],
     len(STATS["kept_categories"])),
    ("Remove val/test clauses whose exact text occurs in train", STATS["task1_clauses"], len(STATS["kept_categories"])),
], columns=["Curation step (Task 1)", "Clauses / spans", "Categories"])
display(steps)
print(f"Ambiguous multi-label clauses held out of Task 1: {STATS['ambiguous_clauses']} | "
      f"duplicates removed from validation/test: {STATS['duplicates_removed_from_val_test']} | "
      f"spans misaligned with the contract text: {STATS['spans_misaligned_with_context']}")
print("Entity/date categories dropped:", ", ".join(ENTITY_CATS))
print(f"Categories dropped by the >= 60 rule ({len(STATS['dropped_categories'])}):", ", ".join(STATS["dropped_categories"]))

with contextlib.redirect_stdout(io.StringIO()):          # reproducibility check: rebuild into _rebuild_v2/
    subprocess.run([sys.executable, "scripts/build_dataset_v2.py"], check=True, capture_output=True)
for name in [C.TASK1_FILE, C.AMBIGUOUS_FILE, C.TASK2_FILE, C.SPLIT_FILE]:
    same = pd.read_csv(PROCESSED / name).equals(pd.read_csv(PROCESSED / "_rebuild_v2" / name))
    print(f"Rebuilt {name} identical to the frozen file: {same}")
assert sorted(STATS["kept_categories"]) == LABELS
''')

interp("curation")

md(r"""
### 2.6 The ambiguous (multi-label) clauses — held out of the main score
""")

code(r'''
amb = C.load_ambiguous()
n_labels = amb.all_labels.str.len()
print(f"Ambiguous clauses: {len(amb):,} (highlighted under 2 or more of the {len(LABELS)} kept categories)")
print("Number of valid labels per ambiguous clause:", n_labels.value_counts().sort_index().to_dict())
print("By split:", amb.split.value_counts().to_dict())
pairs = amb.all_labels.map(lambda l: " + ".join(sorted(l))).value_counts().head(8)
display(pairs.rename("ambiguous clauses").to_frame())
''')

interp("ambiguous")

md(r"""
### 2.7 Train / validation / test split (shared by Task 1 and Task 2)
* **Test** = CUAD's **official test contracts** (102). Using them keeps the evaluation comparable with published CUAD
  work. Task 1 uses the ones that still contain a single-label clause after curation; Task 2 uses all 102, so false
  positives on contracts without any kept clause also count.
* **Validation** = a hold-out of the remaining (official training) contracts chosen with
  `StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)`: *grouped* so that a contract is entirely in one
  split, *stratified* so that every class keeps roughly the same share (6 folds instead of more, so that even the
  rarest class keeps several validation clauses). All tuning uses validation only.
* **Train** = everything else.
* **The split is by contract, never by clause** (section 2.1 measured why), and the *same* contract split
  (`contract_split.csv`) is used by both tasks, so no contract seen in training by one task is tested by the other.
""")

code(r'''
split_tab = pd.DataFrame({
    "Task 1 clauses": df.split.value_counts(),
    "Task 1 contracts": df.groupby("split").contract.nunique(),
    "Task 2 annotations": spans2.split.value_counts(),
    "all contracts (both tasks)": csplit.split.value_counts(),
}).loc[["train", "validation", "test"]]
split_tab.loc["total"] = split_tab.sum()
display(split_tab)

# Integrity checks (they raise an error if anything is wrong)
sp = csplit.set_index("contract").split
for name, d in [("Task 1", df), ("ambiguous", amb), ("Task 2", spans2)]:
    assert d.groupby("contract").split.nunique().max() == 1, f"{name}: a contract appears in two splits"
    assert (d.contract.map(sp) == d.split).all(), f"{name}: split differs from contract_split.csv"
assert not set(train.text.str.lower()) & set(pd.concat([val, test]).text.str.lower()), "duplicate text across splits"
print("OK: every file follows contract_split.csv; no contract in two splits; no val/test clause text occurs in train.")
''')

md(r"""
### 2.8 Label distribution, rare classes and the pre-declared metrics
""")

code(r'''
counts = df.pivot_table(index="category", columns="split", values="text", aggfunc="count")[["train", "validation", "test"]]
counts["total"] = counts.sum(axis=1)
counts = counts.sort_values("total")
share = 100 * counts[["train", "validation", "test"]] / counts[["train", "validation", "test"]].sum()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 9), sharey=True, gridspec_kw={"width_ratios": [1.6, 1]})
left = np.zeros(len(counts))
for s in ["train", "validation", "test"]:
    ax1.barh(counts.index, counts[s], left=left, color=PL.SPLIT_COLORS[s], label=s, height=0.72,
             edgecolor="white", linewidth=1)
    left += counts[s].values
for y, (lab, tot) in enumerate(counts.total.items()):
    ax1.text(tot + 6, y, f"{tot}" + ("  (new)" if lab in NEW_CLASSES else ""), va="center", fontsize=8, color=PL.INK_2)
ax1.set_xlabel("Number of clauses"); ax1.set_title("Clauses per class (stacked by split)")
ax1.legend(loc="lower right"); ax1.grid(axis="y", visible=False)
for s, m in zip(["train", "validation", "test"], ["o", "s", "D"]):
    ax2.scatter(share[s], share.index, color=PL.SPLIT_COLORS[s], marker=m, s=36, label=s, zorder=3)
ax2.set_xlabel("Share of the split's clauses (%)"); ax2.set_title("Class share within each split")
ax2.legend(loc="lower right"); ax2.grid(axis="y", visible=False)
fig.suptitle(f"Figure 3. Label distribution of the {len(LABELS)} clause types ({len(df):,} clauses)",
             fontweight="bold", y=1.0)
fig.tight_layout(); PL.savefig(fig, 3, "label_distribution"); plt.show()

imbalance = counts.total.max() / counts.total.min()
print(f"Largest class: {counts.total.idxmax()} ({counts.total.max()}) | smallest: {counts.total.idxmin()} "
      f"({counts.total.min()}) | imbalance ratio {imbalance:.1f}:1")
rare = counts.test[counts.test < C.MIN_TEST_CLAUSES_ROBUST].sort_values()
print(f"Classes with fewer than {C.MIN_TEST_CLAUSES_ROBUST} test clauses: {rare.to_dict()}")
print(f"Robustness set = the {len(ROBUST)} classes with >= {C.MIN_TEST_CLAUSES_ROBUST} test clauses")
''')

code(r'''
# Class table with counts (all numbers computed from the CSV)
class_tab = counts[["train", "validation", "test", "total"]].sort_values("total", ascending=False).astype(int)
class_tab["% of all"] = (100 * class_tab.total / class_tab.total.sum()).round(1)
class_tab["new class"] = ["yes" if c in NEW_CLASSES else "" for c in class_tab.index]
class_tab["in robustness set"] = ["yes" if c in ROBUST else "no" for c in class_tab.index]
class_tab
''')

md(r"""
**Pre-declared metrics (fixed here, before any model is trained):**
* **Primary metric: macro-F1 over all 27 classes.** Every class counts equally, including the rare ones.
* **Robustness number: macro-F1 over the classes with at least 15 test clauses** (the robustness set above). A class
  with 4–12 test clauses can move its F1 by a large step with a single clause, so this second number shows whether a
  conclusion depends on those few clauses.
* **Per-class F1 with 95 % contract-level bootstrap confidence intervals** for every class, so the uncertainty of the
  rare classes is visible.
* **How the bootstrap treats rare classes.** Each resample draws whole contracts with replacement, so a rare class
  (e.g. Source Code Escrow, whose 4 test clauses all come from one contract) is often missing from a resample; in each
  resample macro-F1 is therefore averaged over the classes present in that resample's true labels, while the point
  estimate is always over all 27 classes. For a per-class CI, resamples without that class are skipped, and a class whose
  clauses all come from one contract gets no CI ("not estimable").
* **The class set is never changed after seeing results.** No class is dropped, merged or re-weighted in the
  evaluation because it scores badly; the robustness set is fixed by test-set *counts* only, which are known before
  any model is run.
""")

interp("labels")

md(r"""
### 2.9 Clause length
""")

code(r'''
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(df.n_words.clip(upper=300), bins=60, color=PL.BLUE, edgecolor="white", linewidth=0.6)
med, p95, mx = df.n_words.median(), df.n_words.quantile(0.95), df.n_words.max()
for v, lab, col in [(med, f"median {med:.0f}", PL.INK), (p95, f"95th percentile {p95:.0f}", PL.ORANGE)]:
    ax.axvline(v, color=col, linestyle="--", linewidth=1.5)
    ax.text(v + 3, ax.get_ylim()[1] * 0.9, lab, color=col, fontsize=9)
ax.set_xlabel("Clause length in words (values above 300 shown at 300)"); ax.set_ylabel("Number of clauses")
ax.set_title("Figure 4. Clause length distribution (all splits)")
PL.savefig(fig, 4, "clause_length_words"); plt.show()
print(f"words per clause: median {med:.0f}, 95th percentile {p95:.0f}, max {mx}, "
      f"share <= 128 words: {100 * (df.n_words <= 128).mean():.1f} %")
display(df.groupby("category").n_words.median().sort_values().rename("median words").to_frame().T)
''')

interp("length")

md(r"""
### 2.10 Example clauses (two per class)
Two training clauses per class — the two whose length is closest to the class median — so the reader can see what the
labels mean in practice (shortened to 200 characters).
""")

code(r'''
pd.set_option("display.max_colwidth", 210)
ex = (train.assign(gap=lambda d: (d.n_words - d.groupby("category").n_words.transform("median")).abs())
           .sort_values(["category", "gap", "row_id"]).groupby("category").head(2))
ex[["category", "n_words", "text"]].assign(text=lambda d: d.text.str.slice(0, 200)).reset_index(drop=True)
''')

# =====================================================================================
md(r"""
## 3. Preprocessing

### 3.1 Approach: preprocessing as an experiment
Each candidate step is treated as a **hypothesis about the data** and tested on the 27-class data, not applied by habit
and not carried over from the first design.

* **Fixed test bench:** default TF-IDF features + `LinearSVC(C=1)`; trained on *train*, scored on **validation**
  (macro-F1 over all classes). The test set is never used. The classifier is deliberately left untuned so that
  differences come from the preprocessing alone (the classifiers are tuned in section 4).
* **Order (greedy, forward):** steps are tried one at a time — characters → case → cross-references → numbers →
  symbols → stop words → word forms → n-grams. Each candidate is compared with the pipeline built so far.
* **Decision rule, fixed before running:** small differences on one validation set are noise. An option is
  **adopted if it improves validation macro-F1 by ≥ 0.003**. If no option clears that bar, the options within ±0.003
  are treated as equivalent and the one with the **smallest vocabulary** (the simpler model) is kept. Options worse by
  more than 0.003 are rejected.
* **Uncertainty:** every change gets a 95 % confidence interval from a *contract-level paired bootstrap* on
  validation (1,000 resamples of the validation contracts; macro-F1 in each resample over the classes present in it,
  see 2.8), so the reader can see which effects are real.

| Step | Hypothesis | Options |
|---|---|---|
| 1 Unicode | typographic characters (curly quotes, bullets, zero-width spaces) split identical words | off / on |
| 2 Lower-casing | "Licensee" and "licensee" should be one feature — but ALL-CAPS text may itself signal certain clauses (e.g. liability caps) | off / on |
| 3 Cross-references | "Section 12.3(b)" numbers are contract-specific noise; the *presence* of a reference may matter | keep / replace with `secref` |
| 4 Numbers | exact values ("30", "5,000,000") are rare features; the presence of a number matters (durations, caps) | keep / replace with `num` |
| 5 Symbols | "$" and "%" signal money and percentages but are deleted by the default tokenizer | drop / keep as words |
| 6 Stop words | function words add noise — but "not", "no", "until", "without", "only", "may" carry legal meaning | none / full NLTK list / NLTK minus legal words |
| 7 Word forms | merging "terminate/terminates/termination" pools evidence | none / Porter stemming / WordNet lemmatisation |
| 8 N-grams | word pairs ("governing law", "change of control") are more specific than single words | 1 / 1–2 / 1–3 |
""")

md(r"""
### 3.2 What each step does to a real clause (before → after)
For each step we show one training clause that the step changes, with the step switched off (before) and on (after).
""")

code(r'''
DEMO_STEPS = [("unicode", True), ("lowercase", True), ("section_refs", True), ("numbers", True),
              ("symbols", True), ("stopwords", "nltk_legal"), ("stem", "porter"), ("stem", "lemma")]
base = dict(P.BASELINE_CONFIG)
short = train[train.n_words.between(15, 45)].sort_values("row_id")
for key, value in DEMO_STEPS:
    on = {**base, key: value}
    for text in short.text:
        before, after = P.tokenize(text, base), P.tokenize(text, on)
        if before != after:
            break
    print(f"--- step: {key} = {value}")
    print("  text  :", text[:200])
    print("  before:", " ".join(before)[:200])
    print("  after :", " ".join(after)[:200], "\n")
''')

md(r"""
### 3.3 The preprocessing study (validation set only)
""")

code(r'''
def run_config(cfg):
    """Fit TF-IDF + LinearSVC(C=1) on train with preprocessing `cfg`; return validation predictions and stats."""
    vec = P.make_vectorizer(cfg)
    Xtr = vec.fit_transform(train.text)
    clf = LinearSVC(C=1.0, random_state=42).fit(Xtr, train.category)
    pred = clf.predict(vec.transform(val.text))
    n_tok = np.mean([len(P.tokenize(t, cfg)) for t in train.text])
    return pred, Xtr.shape[1], n_tok

def macro(pred):
    return f1_score(val.category, pred, labels=LABELS, average="macro", zero_division=0)

STUDY = [  # (step, config key, options in the order they are tried; the first option = "off")
    ("1 Unicode normalisation", "unicode", [False, True]),
    ("2 Lower-casing", "lowercase", [False, True]),
    ("3 Cross-references -> secref", "section_refs", [False, True]),
    ("4 Numbers -> num", "numbers", [False, True]),
    ("5 Keep $ and % as words", "symbols", [False, True]),
    ("6 Stop-word removal", "stopwords", ["none", "nltk", "nltk_legal"]),
    ("7 Stemming / lemmatisation", "stem", ["none", "porter", "lemma"]),
    ("8 N-gram range (max n)", "ngram_max", [1, 2, 3]),
]
TOL = 0.003
print(f"Validation set: {len(val):,} clauses from {val.contract.nunique()} contracts")

with C.timer("preprocessing_study", "greedy_forward_selection", notes="TF-IDF + LinearSVC(C=1), validation only"):
    current = dict(P.BASELINE_CONFIG)
    cur_pred, cur_vocab, cur_tok = run_config(current)
    records = [{"step": "0 Baseline", "option": "raw text, default tokenizer", "macro_f1": macro(cur_pred),
                "delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "vocab": cur_vocab, "mean_tokens": cur_tok,
                "decision": "baseline", "config": json.dumps(current)}]
    for step, key, options in STUDY:
        cands = []
        for opt in options:
            cfg = {**current, key: opt}
            if opt == current[key]:
                pred, vocab, tok = cur_pred, cur_vocab, cur_tok
            else:
                pred, vocab, tok = run_config(cfg)
            pb = C.paired_bootstrap(val.category, pred, cur_pred, val.contract, labels=LABELS, n_boot=1000)
            cands.append(dict(step=step, option=str(opt), opt=opt, cfg=cfg, pred=pred, macro_f1=macro(pred),
                              delta=pb["diff"], ci_low=pb["ci_low"], ci_high=pb["ci_high"], vocab=vocab,
                              mean_tokens=tok, is_current=(opt == current[key])))
        # --- the pre-declared decision rule ---
        best = max(cands, key=lambda c: c["delta"])
        if best["delta"] >= TOL:
            chosen, why = best, f"adopted: +{best['delta']:.3f} >= {TOL}"
        else:
            ties = [c for c in cands if abs(c["delta"]) < TOL]
            chosen = min(ties, key=lambda c: c["vocab"])
            why = ("no option better by >= 0.003; kept the smallest vocabulary among equivalent options")
        for c in cands:
            if c["is_current"] and c is not chosen:
                dec = "replaced"
            elif c is chosen:
                dec = "KEPT (" + why + ")"
            elif c["delta"] <= -TOL:
                dec = f"rejected: {c['delta']:+.3f}"
            else:
                dec = "not chosen (equivalent)"
            records.append({k: c[k] for k in ["step", "option", "macro_f1", "delta", "ci_low", "ci_high", "vocab",
                                              "mean_tokens"]} | {"decision": dec, "config": json.dumps(c["cfg"])})
        current, cur_pred, cur_vocab, cur_tok = chosen["cfg"], chosen["pred"], chosen["vocab"], chosen["mean_tokens"]

study = pd.DataFrame(records)
study.to_csv(C.RESULTS_DIR / "preprocessing_study.csv", index=False)
SELECTED_CONFIG = current
pd.set_option("display.max_colwidth", 90)
study.drop(columns="config").round(4)
''')

code(r'''
plot = study[(study.step != "0 Baseline") & ~study.decision.isin(["replaced"])].copy()
plot = plot[~((plot.delta == 0) & (plot.ci_low == 0) & (plot.ci_high == 0))]   # the reference itself (change = 0 by definition)
plot["label"] = plot.step.str.replace(r"^\d ", "", regex=True) + ": " + plot.option
plot["kept"] = plot.decision.str.startswith("KEPT")
plot = plot.iloc[::-1]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7), sharey=True, gridspec_kw={"width_ratios": [1.5, 1]})
y = np.arange(len(plot))
colors = [PL.BLUE if k else "#b8b6af" for k in plot.kept]
ax1.errorbar(plot.delta, y, xerr=[plot.delta - plot.ci_low, plot.ci_high - plot.delta], fmt="none",
             ecolor="#9a988f", capsize=3, linewidth=1)
ax1.scatter(plot.delta, y, c=colors, s=46, zorder=3)
ax1.axvline(0, color=PL.INK_2, linewidth=1)
for x in (-TOL, TOL):
    ax1.axvline(x, color=PL.ORANGE, linewidth=1, linestyle=":")
ax1.set_yticks(y); ax1.set_yticklabels(plot.label, fontsize=8.5)
ax1.set_xlabel("Change in validation macro-F1 vs the pipeline so far (95 % contract bootstrap CI)")
ax1.set_title("Each option tried, vs the pipeline so far (blue = adopted)")
ax2.barh(y, plot.vocab / 1000, color=colors, height=0.65)
ax2.set_xlabel("Vocabulary size (thousand features)"); ax2.set_title("Feature count")
ax2.grid(axis="y", visible=False)
fig.suptitle("Figure 5. Preprocessing study: greedy forward selection on the validation set", fontweight="bold", y=1.0)
fig.tight_layout(); PL.savefig(fig, 5, "preprocessing_study"); plt.show()
''')

code(r'''
# Final pipeline vs the raw baseline, and consistency with src/preprocess.py
base_pred, base_vocab, base_tok = run_config(P.BASELINE_CONFIG)
fin_pred, fin_vocab, fin_tok = run_config(SELECTED_CONFIG)
pb = C.paired_bootstrap(val.category, fin_pred, base_pred, val.contract, labels=LABELS, n_boot=1000)
print("Selected configuration:", SELECTED_CONFIG)
print(f"Validation macro-F1: baseline {macro(base_pred):.4f} -> selected pipeline {macro(fin_pred):.4f} "
      f"(diff {pb['diff']:+.4f}, 95 % CI [{pb['ci_low']:+.4f}, {pb['ci_high']:+.4f}])")
print(f"Vocabulary: {base_vocab:,} -> {fin_vocab:,} features | mean tokens per clause: {base_tok:.1f} -> {fin_tok:.1f}")
json.dump({"selected_config": SELECTED_CONFIG, "baseline_val_macro_f1": macro(base_pred),
           "selected_val_macro_f1": macro(fin_pred), "paired_bootstrap": pb,
           "vocab_baseline": base_vocab, "vocab_selected": fin_vocab},
          open(C.RESULTS_DIR / "preprocessing_selected.json", "w"), indent=2)
assert P.FINAL_CLASSICAL_CONFIG == SELECTED_CONFIG, "src/preprocess.py FINAL_CLASSICAL_CONFIG is out of date"
print("OK: src/preprocess.py FINAL_CLASSICAL_CONFIG matches the configuration selected by the study.")
''')

interp("study")

md(r"""
### 3.4 Before → after on five sample clauses (final classical pipeline)
This is the view used in the demo video: the raw clause, the cleaned text, and the tokens the TF-IDF model sees.
""")

code(r'''
samples = (train[train.n_words.between(20, 60)]
           .groupby("category").head(1).sort_values("row_id")
           .loc[lambda d: d.category.isin(["Cap On Liability", "Renewal Term", "Liquidated Damages",
                                           "Non-Transferable License", "Anti-Assignment"])])
for _, r in samples.iterrows():
    toks = P.tokenize(r.text, P.FINAL_CLASSICAL_CONFIG)
    print(f"[{r.category}]  ({r.n_words} words -> {len(toks)} tokens)")
    print("  BEFORE :", r.text[:260])
    print("  CLEANED:", P.clean_text(r.text, P.FINAL_CLASSICAL_CONFIG)[:260])
    print("  TOKENS :", toks[:30], "\n")
''')

md(r"""
### 3.5 Preprocessing for the transformer (Legal-BERT) — deliberately light
A fine-tuned transformer needs very different input from a bag-of-words model:
* **No stop-word removal, no stemming:** the model reads words *in context*; "shall **not** assign" and
  "shall assign" must stay different, and its attention needs the full sentence.
* **Casing follows the checkpoint:** `nlpaueb/legal-bert-base-uncased` (and the ablation model `bert-base-uncased`)
  are *uncased* — their tokenizer lower-cases internally. We pass the text as it is.
* **Subword tokenizer instead of our word tokenizer:** rare words are split into known pieces, so there is no
  out-of-vocabulary problem and no need for number or cross-reference placeholders.
* We only apply **Unicode normalisation** (`P.preprocess_for_transformer`) so that typographic characters do not
  become unknown tokens.
* **Maximum sequence length** is chosen from the *training* clauses' token lengths: the larger of the two
  tokenizers' 99th percentiles, rounded up to a multiple of 32, so that the Legal-BERT model and the BERT-base
  ablation model use the **same** `max_length` and almost no clause is truncated by either.
""")

code(r'''
from transformers import AutoTokenizer
TOK_NAMES = {"Legal-BERT": "nlpaueb/legal-bert-base-uncased", "BERT-base": "bert-base-uncased"}
toks = {k: AutoTokenizer.from_pretrained(v) for k, v in TOK_NAMES.items()}
texts = df.text.map(P.preprocess_for_transformer)
for k, t in toks.items():
    df[f"tok_{k}"] = [len(ids) for ids in t(list(texts), add_special_tokens=True)["input_ids"]]

trn = df[df.split == "train"]
p99 = max(trn["tok_Legal-BERT"].quantile(0.99), trn["tok_BERT-base"].quantile(0.99))   # the longer of the two
MAX_LEN = int(min(512, np.ceil(p99 / 32) * 32))          # round up to a multiple of 32; same value for both models
tok_stats = {k: {"median": float(trn[f"tok_{k}"].median()), "p95": float(trn[f"tok_{k}"].quantile(.95)),
                 "p99": float(trn[f"tok_{k}"].quantile(.99)), "max": int(df[f"tok_{k}"].max()),
                 "mean": float(df[f"tok_{k}"].mean()),
                 "truncated_at_max_len_%": {s: float(100 * (df.loc[df.split == s, f"tok_{k}"] > MAX_LEN).mean())
                                            for s in ["train", "validation", "test"]}}
             for k in toks}
ratio = df["tok_BERT-base"].sum() / df["tok_Legal-BERT"].sum()
json.dump({"max_len": MAX_LEN, "chosen_from": "max of the 99th percentiles of the train token lengths under both tokenizers, rounded up to x32",
           "stats": tok_stats, "bert_to_legalbert_token_ratio": ratio},
          open(C.RESULTS_DIR / "token_lengths.json", "w"), indent=2)
display(pd.DataFrame({k: {**{m: v for m, v in s.items() if m != "truncated_at_max_len_%"},
                          **{f"truncated {sp} %": v for sp, v in s["truncated_at_max_len_%"].items()}}
                      for k, s in tok_stats.items()}).round(2))
print(f"Chosen max_length = {MAX_LEN} tokens.  BERT-base needs {100 * (ratio - 1):.1f} % more tokens than "
      "Legal-BERT for the same clauses (its vocabulary was not built from legal text).")
''')

code(r'''
fig, ax = plt.subplots(figsize=(9, 4))
bins = np.arange(0, 400, 8)
for k, col in [("Legal-BERT", PL.BLUE), ("BERT-base", PL.ORANGE)]:
    ax.hist(df[f"tok_{k}"].clip(upper=392), bins=bins, histtype="step", linewidth=2, color=col, label=k)
ax.axvline(MAX_LEN, color=PL.INK, linestyle="--", linewidth=1.5)
ax.text(MAX_LEN + 4, ax.get_ylim()[1] * 0.85, f"max_length = {MAX_LEN}", fontsize=9)
ax.set_xlabel("Clause length in subword tokens incl. [CLS]/[SEP] (values above 392 shown at 392)")
ax.set_ylabel("Number of clauses"); ax.legend()
ax.set_title("Figure 6. Clause length in tokens: Legal-BERT vs BERT-base vocabulary")
PL.savefig(fig, 6, "clause_length_tokens"); plt.show()

example = "The Licensee shall indemnify the Licensor against any infringement of the sublicensed trademarks."
for k, t in toks.items():
    print(f"{k:>10}: {t.tokenize(example)}")
''')

interp("transformer")

md(r"""
### 3.6 Input preparation for Task 2 (clause extraction) — full contracts, offsets untouched
Task 2 works on **whole contracts**, and its labels are **character offsets** into the original contract text. Two
consequences, measured below:
* **No text normalisation that changes characters.** Even Unicode normalisation or whitespace collapsing would shift
  every later offset and misplace the gold spans. The tokenizer's offset mapping (each word-piece knows its character
  range) is used instead, so labels stay exact.
* **Sliding windows.** A contract is far longer than the 512-token limit, so it is cut into overlapping windows of
  about 256 word-pieces with a stride of about 128; predictions from overlapping windows are averaged. The cell
  counts how many windows that produces. The full Task 2 data preparation (token labels, oracle check) is in
  `STUDENTID_Task2.ipynb`.
""")

code(r'''
texts_full = {t: C.load_contract_text(t) for t in csplit.contract}
n_chars = pd.Series({t: len(x) for t, x in texts_full.items()})
assert (n_chars.reindex(csplit.contract).values == csplit.n_chars.values).all()
import unicodedata
changed_nfkc = sum(len(unicodedata.normalize("NFKC", x)) != len(x) for x in texts_full.values())
changed_ws = sum(len(re.sub(r"\s+", " ", x)) != len(x) for x in texts_full.values())
lb = toks["Legal-BERT"]
n_tokens = pd.Series({t: len(lb(x, add_special_tokens=False)["input_ids"]) for t, x in texts_full.items()})
WIN, STRIDE = 256, 128
n_windows = np.maximum(1, np.ceil((n_tokens - WIN) / STRIDE).clip(lower=0) + 1)
t2prep = {"contracts": int(len(n_chars)), "chars_median": float(n_chars.median()), "chars_p95": float(n_chars.quantile(.95)),
          "chars_max": int(n_chars.max()), "legalbert_tokens_median": float(n_tokens.median()),
          "legalbert_tokens_max": int(n_tokens.max()), "windows_total": int(n_windows.sum()),
          "contracts_where_NFKC_changes_length": int(changed_nfkc),
          "contracts_where_whitespace_collapse_changes_length": int(changed_ws), "window": WIN, "stride": STRIDE}
json.dump(t2prep, open(C.RESULTS_DIR / "task2_input_stats.json", "w"), indent=2)
display(pd.Series(t2prep).to_frame("value"))
''')

interp("task2prep")

md(r"""
### 3.7 The representation decisions that did matter: character n-grams and class weighting
The cleaning steps above made no measurable difference, but two **feature / weighting decisions** did. They are
measured here on the **same untuned bench** as the study (LinearSVC, C = 1, train → validation) so the comparison is
controlled: only one thing changes at a time.

* **Word tokens vs character n-grams.** Word features: the study's token configuration (section 3.3), single words.
  Character features: `char_wb` 3–6-character pieces taken inside word boundaries, from the same cleaned text. Character
  pieces match word stems and fragments ("assign" in *assignment, assigned, reassign*) without any stemming rule and are
  robust to hyphenation and typing variants in the filings — the hypothesis is that this suits short legal clauses.
* **No class weights vs `class_weight="balanced"`.** Balanced weighting gives every class the same total weight in the
  loss, so a class with 42 training clauses is not drowned by one with 449. The hypothesis: with a 10.7 : 1 imbalance,
  unweighted models under-predict the rare classes, which hurts macro-F1.

Section 4 then tunes both choices inside the full grid; this controlled 2 × 2 is the before/after evidence.
""")

code(r"""
def bench(features, weight):
    if features == "word tokens (study config)":
        vec = P.make_vectorizer(P.FINAL_CLASSICAL_CONFIG)
    else:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), lowercase=False,
                              preprocessor=lambda t: P.clean_text(t, P.FINAL_CLASSICAL_CONFIG))
    Xtr = vec.fit_transform(train.text)
    clf = LinearSVC(C=1.0, class_weight=weight, random_state=42).fit(Xtr, train.category)
    return clf.predict(vec.transform(val.text)), Xtr.shape[1]

feat_rows, feat_preds = [], {}
for features in ["word tokens (study config)", "char_wb 3-6 n-grams"]:
    for weight in [None, "balanced"]:
        pred, vocab = bench(features, weight)
        feat_preds[(features, weight)] = pred
        feat_rows.append({"features": features, "class_weight": str(weight), "vocab": vocab,
                          "val_macro_f1": macro(pred),
                          "val_macro_f1_robust": f1_score(val.category, pred, labels=ROBUST, average="macro", zero_division=0)})
feat = pd.DataFrame(feat_rows)
ref = feat_preds[("word tokens (study config)", None)]
for k, row in feat.iterrows():
    pb = C.paired_bootstrap(val.category, feat_preds[(row.features, None if row.class_weight == "None" else row.class_weight)],
                            ref, val.contract, labels=LABELS, n_boot=1000)
    feat.loc[k, ["delta_vs_word_unweighted", "ci_low", "ci_high"]] = [pb["diff"], pb["ci_low"], pb["ci_high"]]
feat.to_csv(C.RESULTS_DIR / "feature_decisions.csv", index=False)
display(feat.round(4))

fig, ax = plt.subplots(figsize=(8, 3.6))
lab = [f"{f.split(' (')[0]}\n{'balanced' if w == 'balanced' else 'no weights'}" for f, w in zip(feat.features, feat.class_weight)]
cols = [PL.BLUE if "char" in f else "#b8b6af" for f in feat.features]
ax.bar(range(len(feat)), feat.val_macro_f1, color=cols, width=0.6, edgecolor=[PL.ORANGE if w == "balanced" else "white" for w in feat.class_weight], linewidth=2)
for i_, v_ in enumerate(feat.val_macro_f1):
    ax.text(i_, v_ + 0.002, f"{v_:.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(range(len(feat))); ax.set_xticklabels(lab, fontsize=8.5)
ax.set_ylim(feat.val_macro_f1.min() - 0.03, feat.val_macro_f1.max() + 0.02); ax.grid(axis="x", visible=False)
ax.set_ylabel("Validation macro-F1 (27 classes)")
ax.set_title("Figure 7. Feature and weighting decisions on the untuned bench (LinearSVC, C = 1)", fontsize=11)
PL.savefig(fig, 7, "feature_decisions"); plt.show()
""")

code(r"""
# One clause shown both ways: what the word model and the character model actually see
r = train[(train.category == "Anti-Assignment") & train.n_words.between(15, 25)].sort_values("row_id").iloc[0]
cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), lowercase=False,
                     preprocessor=lambda t: P.clean_text(t, P.FINAL_CLASSICAL_CONFIG)).build_analyzer()
chars = cv(r.text)
print(f"[{r.category}] {r.text}\n")
print(f"WORD tokens ({len(P.tokenize(r.text, P.FINAL_CLASSICAL_CONFIG))}):", P.tokenize(r.text, P.FINAL_CLASSICAL_CONFIG))
print(f"\nCHARACTER n-grams ({len(chars)}), first 40:", chars[:40])
""")

interp("featuredec")

md(r"""
### 3.8 Summary: one corpus, three input pipelines

| | Classical TF-IDF models (Task 1) | Transformer (Task 1: Legal-BERT / BERT) | Task 2 extractor |
|---|---|---|---|
| Function | `P.tokenize(text, P.FINAL_CLASSICAL_CONFIG)` | `P.preprocess_for_transformer` + checkpoint tokenizer | raw contract text + tokenizer offset mapping |
| Chosen by | the validation study in 3.3 (Figure 5) and 3.7 (Figure 7); tuned in section 4 | the model's pre-training (uncased WordPiece) | the need to keep exact character offsets |
| Stop words | decided by the study | never removed | never removed |
| Word forms | decided by the study | subword pieces | subword pieces |
| Input length | whole clause | max_length from the 99th percentile (Figure 6) | sliding windows over the whole contract |

All use the same corpus and the same contract-level split. The code lives in `src/preprocess.py`; the study results
are in `results/preprocessing_study.csv` and `results/preprocessing_selected.json`.
""")

import os as _os
partB = Path(__file__).with_name("build_task1_partB.py")
if partB.exists() and not _os.environ.get("ONLY_SECTIONS_0_3"):   # ONLY_SECTIONS_0_3=1: quick pre-run of the study
    exec(partB.read_text(encoding="utf-8"))
partC = Path(__file__).with_name("build_task1_partC.py")
if INCLUDE_SECTION5 and partC.exists():
    exec(partC.read_text(encoding="utf-8"))
partD = Path(__file__).with_name("build_task1_partD.py")      # section 6: LEDGAR published comparison
if INCLUDE_SECTION5 and partD.exists():
    exec(partD.read_text(encoding="utf-8"))
partE = Path(__file__).with_name("build_task1_partE.py")      # section 7: evaluation and analysis
if INCLUDE_SECTION5 and partE.exists():
    exec(partE.read_text(encoding="utf-8"))
partF = Path(__file__).with_name("build_task1_partF.py")      # sections 8-10: PART F (F1 anomaly flagging, F3 LoRA) + objectives
if INCLUDE_SECTION5 and partF.exists():
    exec(partF.read_text(encoding="utf-8"))

md(r"""
---
*Next: the prompted LLMs are compared in `STUDENTID_LLM.ipynb`, Task 2 (clause extraction) is in `STUDENTID_Task2.ipynb`,
and the full evaluation and error analysis (Part C) bring all Task 1 models together in one comparison table.*
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"},
                  "colab": {"provenance": []}}
OUT.parent.mkdir(exist_ok=True)
nbf.write(nb, OUT)
print("wrote", OUT, len(cells), "cells")
