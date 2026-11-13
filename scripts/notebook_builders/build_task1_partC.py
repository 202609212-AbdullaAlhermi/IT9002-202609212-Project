# Section 5 of the Task 1 notebook (design v2): fine-tuned Legal-BERT + controlled ablation with BERT-base.
# Exec'd by build_task1_partA.py (uses md/code/interp).

md(r"""
## 5. Fine-tuned transformer: Legal-BERT (and the controlled ablation with BERT-base)

### 5.1 Why this checkpoint
**`nlpaueb/legal-bert-base-uncased`** (Chalkidis et al., 2020, *Findings of EMNLP*; model licence CC BY-SA 4.0):
* **Domain match.** It was pre-trained from scratch on about 12 GB of English legal text — legislation, court cases
  and **76,366 US contracts from SEC EDGAR**, the same kind of document as CUAD. Its WordPiece vocabulary was built
  from that legal text, so legal terms such as "licensee" or "indemnify" are single tokens (section 3.5).
* **Size fits the compute.** BERT-base architecture: 12 layers, 768 hidden units, about 110 M parameters. On our
  RTX 5080 (16 GB) one epoch over the training clauses takes seconds (benchmark below), so we can afford a
  learning-rate search, three seeds, a class-weighting test and the ablation. The same model would also train on a
  free Colab GPU.
* **Input window covers the clauses.** The 512-token limit is far above typical clause length; we use the
  `max_length` chosen in section 3.5 from both tokenizers, which truncates fewer than 1 % of clauses.
* **Honesty note (Honesty log):** Legal-BERT's pre-training corpus contains EDGAR contracts, and CUAD contracts are
  EDGAR filings too, so some CUAD *text* may have been seen during pre-training. This is unlabelled text, so no label
  can leak, but memorised boilerplate could help a little. We cannot check the overlap; it is stated as a limitation.

**Ablation model:** `bert-base-uncased` (Devlin et al., 2019) — identical architecture and size, pre-trained on general
English (books and Wikipedia). Legal-BERT-base was pre-trained **from scratch on legal text with its own legal
vocabulary**, so the ablation changes the pre-training corpus **and** the vocabulary together; it measures their
combined effect and cannot separate them.
""")

md(r"""
### 5.2 The fine-tuning recipe (identical for both models)

| Setting | Value | Why |
|---|---|---|
| Optimiser | AdamW, weight decay 0.01 (none on biases / LayerNorm) | standard for BERT fine-tuning; decoupled weight decay regularises without disturbing Adam's step sizes |
| Learning rate | searched over {2e-5, 3e-5, 5e-5} on validation | the range recommended for BERT fine-tuning by Devlin et al. (2019) |
| Schedule | linear warm-up over the first 10 % of steps, then linear decay to 0 | warm-up avoids large, destructive updates while the new classification layer is still random |
| Batch size | 16 | recommended range (16/32); more parameter updates per epoch on a small dataset |
| Epochs | 5, best epoch chosen by validation macro-F1 | Devlin et al. suggest 2–4; one extra epoch lets the curves show whether the model over-fits |
| Gradient clipping | max norm 1.0 | protects against occasional exploding gradients |
| Precision | bfloat16 mixed precision on the GPU | about 2× faster, half the memory; bfloat16 needs no loss scaling |
| Padding | dynamic (each batch padded to its own longest clause), `max_length` from section 3.5 | most clauses are short, so padding every clause to the maximum would waste most of the compute |
| Loss | cross-entropy; a class-weighted variant (inverse class frequency) is tested with 3 seeds | 27 imbalanced classes; adopted only if it clearly helps (rule below) |
| Seeds | 42, 43, 44 → mean ± std on validation | the test model is **pre-declared**: seed 42 — never the best seed |
| Selection rules | LR = best Legal-BERT validation macro-F1 (seed 42); BERT-base uses the **same** LR; class weights adopted only if the 3-seed mean gain is ≥ 0.003 **and** larger than the seed spread (max − min) of the unweighted runs | fixed in `src/finetune.py` (`run_full_plan`) before any run |

The training loop is plain PyTorch in `src/finetune.py` (function `train_run`, about 100 commented lines). After every
epoch it writes a resume checkpoint (model, optimiser, scheduler, random-number state), so an interrupted run continues
from the last finished epoch.
""")

code(r'''
import inspect, torch, logging
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"          # keep the outputs readable
transformers.utils.logging.disable_progress_bar()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
from src import finetune as F
importlib.reload(F)
print("Recipe used by every run:", F.RECIPE)
# The core of one training step, printed from src/finetune.py so the notebook shows the real code:
src_lines = inspect.getsource(F.train_run).splitlines()
start = next(i for i, l in enumerate(src_lines) if "for i, batch in enumerate(loader)" in l)
print("\n".join(src_lines[start:start + 12]))
''')

md(r"""
### 5.3 Benchmark before training (measure first)
Twenty training steps at the batch size we use, timed on this machine, to project the cost of the whole plan before
starting it. (Without a GPU, e.g. on Colab's CPU runtime, this cell shows the saved measurement instead.)
""")

code(r'''
from transformers import AutoModelForSequenceClassification
BENCH_JSON = C.RESULTS_DIR / "transformer_benchmark.json"
if torch.cuda.is_available() and not BENCH_JSON.exists():
    bench = {}
    for key, ckpt in F.CHECKPOINTS.items():
        tok = AutoTokenizer.from_pretrained(ckpt)
        mdl = AutoModelForSequenceClassification.from_pretrained(ckpt, id2label=dict(enumerate(LABELS))).cuda()
        items = F.encode(tok, list(train.text[:16 * 25]), list(train.category[:16 * 25]),
                         {l: i for i, l in enumerate(LABELS)}, F.RECIPE["max_length"])
        loader = F.make_loader(items, tok, 16, shuffle=False)
        opt = torch.optim.AdamW(mdl.parameters(), lr=3e-5)
        torch.cuda.reset_peak_memory_stats()
        for i, b in enumerate(loader):
            if i == 5:
                torch.cuda.synchronize(); t0 = time.perf_counter()
            b = {k: v.cuda() for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = mdl(**b).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0); opt.step(); opt.zero_grad()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 20 * 1000
        steps = -(-len(train) // 16)
        bench[key] = {"ms_per_step": ms, "steps_per_epoch": steps, "projected_epoch_seconds": ms * steps / 1000,
                      "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "gpu": torch.cuda.get_device_name(0)}
        del mdl, opt; torch.cuda.empty_cache()
    n_runs, epochs = 13, F.RECIPE["epochs"]      # 6 LR runs + 4 extra seeds + 3 class-weighted (+3 if adopted)
    bench["projection"] = {"runs": n_runs, "runs_if_class_weights_adopted": n_runs + 3, "epochs_per_run": epochs,
                           "projected_total_minutes": n_runs * epochs * bench["legalbert"]["projected_epoch_seconds"] / 60}
    json.dump(bench, open(BENCH_JSON, "w"), indent=2)
bench = json.load(open(BENCH_JSON))
for key in F.CHECKPOINTS:
    b = bench[key]
    print(f"{key:>9}: {b['ms_per_step']:.0f} ms per step (batch 16) -> {b['projected_epoch_seconds']:.0f} s per epoch, "
          f"peak GPU memory {b['peak_gpu_mem_gb']:.1f} GB on {b['gpu']}")
pj = bench["projection"]
print(f"Projected plan: {pj['runs']} runs (up to {pj['runs_if_class_weights_adopted']}) x {pj['epochs_per_run']} epochs ≈ "
      f"{pj['projected_total_minutes']:.0f} min of training steps (+ validation) -> far below the 6-hour limit, so the "
      "whole plan runs locally.")
''')

md(r"""
### 5.4 Training all runs
The plan (all decision rules fixed in advance in `F.run_full_plan`):
1. learning-rate search {2e-5, 3e-5, 5e-5} with seed 42 for **both** checkpoints (6 runs);
2. seeds 43 and 44 at the chosen learning rate for both checkpoints (4 runs);
3. class-weighted Legal-BERT with seeds 42, 43, 44 (3 runs); if adopted, the same for BERT-base (3 more runs).

The real run was executed with `python scripts/train_transformer.py` (same function, run in the background with a log
file). `RETRAIN = False` loads its logs; `RETRAIN = True` re-runs everything (about 20–25 min on an RTX 5080).
""")

code(r'''
RETRAIN = False
if RETRAIN:
    with C.timer("transformer_plan", "all_runs", notes="re-run from notebook", device="GPU"):
        choice = F.run_full_plan(train, val, LABELS)
choice = json.load(open(C.RESULTS_DIR / "transformer_choice.json"))
tsearch = pd.read_csv(C.RESULTS_DIR / "transformer_search.csv")
tlog = pd.read_csv(C.RESULTS_DIR / "transformer_log.csv")
cols = ["run_id", "model", "lr", "seed", "class_weighted", "best_epoch", "val_macro_f1", "val_accuracy",
        "mean_epoch_seconds", "run_seconds", "peak_gpu_mem_gb"]
display(tsearch[cols].round(4).assign(lr=tsearch.lr.map("{:.0e}".format)))
print("Decisions:", {k: (f"{v:.0e}" if k == "best_lr" else round(v, 4) if isinstance(v, float) else v)
                     for k, v in choice.items() if k != "test_runs"})
print("Pre-declared test models:", choice["test_runs"])
''')

md(r"""
### 5.5 Training and validation curves
""")

code(r'''
TEST_RUNS = choice["test_runs"]
fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex="col")
for r, (key, name, col) in enumerate([("legalbert", "Legal-BERT", PL.BLUE), ("bert", "BERT-base", PL.ORANGE)]):
    rid = TEST_RUNS[key]
    st = tlog[(tlog.run_id == rid) & (tlog.kind == "step")]
    ep = tlog[(tlog.run_id == rid) & (tlog.kind == "epoch")]
    spe = int(ep.step.iloc[0])
    ax = axes[r, 0]
    ax.plot(st.step / spe, st.train_loss, color=col, alpha=0.18, linewidth=0.8, label="training loss (each step)")
    ax.plot(st.step / spe, st.train_loss.rolling(25, min_periods=1).mean(), color=col, linewidth=2,
            label="training loss (25-step mean)")
    ax.plot(ep.epoch, ep.val_loss, "s--", color=PL.INK, markersize=7, linewidth=1.5, label="validation loss")
    ax.set_ylabel(f"{name}\ncross-entropy loss"); ax.set_yscale("log"); ax.legend(fontsize=8)
    ax = axes[r, 1]
    ax.plot(ep.epoch, ep.val_macro_f1, "o-", color=col, linewidth=2, markersize=7, label="validation macro-F1")
    ax.plot(ep.epoch, ep.val_accuracy, "s:", color=col, alpha=0.6, linewidth=1.5, label="validation accuracy")
    be = ep.loc[ep.val_macro_f1.idxmax()]
    ax.scatter([be.epoch], [be.val_macro_f1], s=180, facecolors="none", edgecolors=PL.INK, linewidths=1.5, zorder=4,
               label=f"selected epoch {int(be.epoch)}")
    ax.set_ylabel("score"); ax.legend(fontsize=8, loc="lower right")
    ax = axes[r, 2]
    ax.plot(st.step / spe, st.lr * 1e5, color=col, linewidth=2)
    ax.set_ylabel("learning rate (×1e-5)")
for ax in axes[1]:
    ax.set_xlabel("epoch")
axes[0, 0].set_title("Loss"); axes[0, 1].set_title("Validation scores"); axes[0, 2].set_title("Learning-rate schedule")
fig.suptitle(f"Figure 13. Training and validation curves of the two pre-declared test runs "
             f"(lr {choice['best_lr']:.0e}, seed 42)", fontweight="bold", y=1.0)
fig.tight_layout(); PL.savefig(fig, 13, "transformer_curves"); plt.show()
display(tlog[(tlog.run_id.isin(TEST_RUNS.values())) & (tlog.kind == "epoch")]
        [["run_id", "epoch", "train_loss", "val_loss", "val_macro_f1", "val_accuracy", "epoch_train_seconds"]].round(4))
''')

interp("curves")

code(r'''
# Learning-rate search, seed variation and the class-weighted runs (validation macro-F1 per epoch)
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, (key, name) in zip(axes, [("legalbert", "Legal-BERT"), ("bert", "BERT-base")]):
    for _, r in tsearch[tsearch.model == key].iterrows():
        ep = tlog[(tlog.run_id == r.run_id) & (tlog.kind == "epoch")]
        if r.class_weighted:
            style, color, lw, alpha = ":", PL.RED, 1.3, 0.8
        elif r.seed == 42:
            style, color, lw, alpha = "o-", {2e-5: PL.AQUA, 3e-5: PL.BLUE, 5e-5: PL.VIOLET}[r.lr], 2, 1
        else:
            style, color, lw, alpha = "--", PL.BLUE, 1.3, 0.55
        ax.plot(ep.epoch, ep.val_macro_f1, style, color=color, alpha=alpha, linewidth=lw, markersize=5,
                label=f"lr {r.lr:.0e}, seed {r.seed}" + (", class-weighted" if r.class_weighted else ""))
    ax.set_title(name); ax.set_xlabel("epoch"); ax.legend(fontsize=7, loc="lower right")
axes[0].set_ylabel("validation macro-F1 (27 classes)")
fig.suptitle("Figure 14. Learning-rate search (solid, seed 42), extra seeds (dashed) and class-weighted runs (dotted)",
             fontweight="bold", y=1.03)
PL.savefig(fig, 14, "transformer_lr_seeds"); plt.show()

seed_tab = (tsearch[(tsearch.lr == choice["best_lr"])]
            .groupby(["model", "class_weighted"]).val_macro_f1.agg(["mean", "std", "min", "max", "count"]))
print(f"Validation macro-F1 over 3 seeds at lr {choice['best_lr']:.0e} (best epoch of each run):")
display(seed_tab.round(4))
seed_tab.to_csv(C.RESULTS_DIR / "transformer_seeds.csv")
print(f"Class-weighted loss: 3-seed mean gain {choice['class_weighted_gain']:+.4f} vs seed spread "
      f"{choice['seed_spread']:.4f} -> adopted: {choice['use_class_weights']}")
''')

interp("lrseeds")

md(r"""
### 5.6 Test evaluation (once) and computational budget
The two pre-declared runs (seed 42, chosen recipe, best validation epoch) are now scored on the test set once each. If
the saved weights are not available (e.g. on Colab without re-training), the saved test predictions are loaded instead.
We also measure inference speed on the GPU and on the CPU, and the model size on disk.
""")

code(r'''
TEST_NAMES = {"legalbert": "legalbert", "bert": "bert_base"}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label2id = {l: i for i, l in enumerate(LABELS)}
budget_json = C.RESULTS_DIR / "transformer_budget.json"
budget_info = json.load(open(budget_json)) if budget_json.exists() else {}
tf_test_pred = {}
for key, name in TEST_NAMES.items():
    rid = TEST_RUNS[key]
    wdir = F.MODELS_DIR / rid / "best"
    if wdir.exists():
        tok, mdl = F.load_best(rid, device)
        items = F.encode(tok, list(test.text), list(test.category), label2id, F.RECIPE["max_length"])
        _, pred_ids, gpu_s = F.predict(mdl, F.make_loader(items, tok, 64, shuffle=False), device)
        preds = [LABELS[i] for i in pred_ids]
        if name not in budget_info:      # timings of the real test run are kept, not overwritten by re-runs
            torch.set_num_threads(os.cpu_count())
            cpu_mdl = mdl.float().to("cpu")
            _, _, cpu_s = F.predict(cpu_mdl, F.make_loader(items[:200], tok, 16, shuffle=False), torch.device("cpu"))
            size_mb = sum(f.stat().st_size for f in wdir.glob("*") if f.is_file()) / 1e6
            run = tsearch.set_index("run_id").loc[rid]
            budget_info[name] = {
                "run_id": rid, "params_million": float(run.params_million), "best_epoch": int(run.best_epoch),
                "train_seconds_run": float(run.run_seconds), "mean_epoch_seconds": float(run.mean_epoch_seconds),
                "peak_gpu_mem_gb_train": float(run.peak_gpu_mem_gb), "model_size_mb": size_mb,
                "gpu_test_seconds": gpu_s, "gpu_clauses_per_second": len(test) / gpu_s,
                "gpu_ms_per_clause": 1000 * gpu_s / len(test),
                "cpu_clauses_per_second": 200 / cpu_s, "cpu_ms_per_clause": 1000 * cpu_s / 200,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
            del cpu_mdl
        del mdl; torch.cuda.empty_cache()
        tf_test_pred[key] = preds
        C.test_once(name, {"run_id": rid, "recipe": F.RECIPE, "lr": choice["best_lr"], "seed": 42,
                           "class_weighted": choice["use_class_weights"],
                           "best_epoch": int(tsearch.set_index("run_id").loc[rid].best_epoch)},
                    test.category, preds, LABELS, test.row_id, test.contract, ROBUST)
    else:
        print(f"{name}: weights not found ({wdir}); using the saved test predictions.")
        tf_test_pred[key] = pd.read_csv(C.RESULTS_DIR / "predictions" / f"{name}.csv").pred.tolist()
json.dump(budget_info, open(budget_json, "w"), indent=2)

btab = pd.DataFrame(budget_info).T[["params_million", "best_epoch", "mean_epoch_seconds", "train_seconds_run",
                                    "peak_gpu_mem_gb_train", "model_size_mb", "gpu_clauses_per_second",
                                    "gpu_ms_per_clause", "cpu_clauses_per_second", "cpu_ms_per_clause"]]
display(btab.astype(float).round(2))
''')

code(r'''
# Log the budget of the two test models in results/compute_budget.csv (once)
budget = pd.read_csv(C.RESULTS_DIR / "compute_budget.csv")
for name, b in budget_info.items():
    if not ((budget.model == name) & (budget.stage == "train_final")).any():
        new = pd.DataFrame([
            {"timestamp": pd.Timestamp.now().isoformat(timespec="seconds"), "model": name, "stage": "train_final",
             "seconds": round(b["train_seconds_run"], 2), "device": b["gpu"],
             "notes": f"5 epochs incl. validation; best epoch {b['best_epoch']}; peak {b['peak_gpu_mem_gb_train']:.1f} GB"},
            {"timestamp": pd.Timestamp.now().isoformat(timespec="seconds"), "model": name, "stage": "inference_test_gpu",
             "seconds": round(b["gpu_test_seconds"], 2), "device": b["gpu"],
             "notes": f"{len(test):,} test clauses, batch 64, bf16"}])
        budget = pd.concat([budget, new], ignore_index=True)
budget.to_csv(C.RESULTS_DIR / "compute_budget.csv", index=False)
plan_rows = budget[budget.model == "transformer_plan"]
if len(plan_rows):
    print(f"Whole transformer plan ({len(tsearch)} runs, both checkpoints): {plan_rows.seconds.iloc[0] / 60:.1f} min "
          f"wall-clock on {plan_rows.device.iloc[0]}")
''')

md(r"""
### 5.7 Results: Legal-BERT vs BERT-base vs the best classical model
Confidence intervals and paired differences use the contract-level bootstrap of section 2.8 (point estimate over all 27
classes; in each resample, macro-F1 over the classes present). The best classical model is the one selected on
validation in section 4 (`results/classical_best.json`).
""")

code(r'''
best_classical = json.load(open(C.RESULTS_DIR / "classical_best.json"))["best_classical"]
METRICS = json.load(open(C.RESULTS_DIR / "metrics.json"))
cls_pred = pd.read_csv(C.RESULTS_DIR / "predictions" / f"{best_classical}.csv").pred.values
y, grp = test.category.values, test.contract.values
PRED = {best_classical: cls_pred, "bert_base": np.array(tf_test_pred["bert"]), "legalbert": np.array(tf_test_pred["legalbert"])}
rows = []
for name, label in [(best_classical, "best classical (TF-IDF char n-grams + LogReg)"), ("bert_base", "BERT-base (ablation)"),
                    ("legalbert", "Legal-BERT (main)")]:
    ci = C.bootstrap_ci(y, PRED[name], grp, labels=LABELS, n_boot=1000)
    cir = C.bootstrap_ci(y, PRED[name], grp, labels=ROBUST, n_boot=1000)
    m = METRICS[f"{name}__test"]
    rows.append({"model": name, "description": label, "test_macro_f1": m["macro_f1"],
                 "95% CI": f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]",
                 f"macro_f1_robust_{len(ROBUST)}": m["macro_f1_robust"], "robust 95% CI": f"[{cir['ci_low']:.3f}, {cir['ci_high']:.3f}]",
                 "test_accuracy": m["accuracy"], "test_weighted_f1": m["weighted_f1"]})
cmp_tab = pd.DataFrame(rows)
cmp_tab.to_csv(C.RESULTS_DIR / "transformer_vs_classical.csv", index=False)
display(cmp_tab.round(4))

pairs = {"H1_legalbert_vs_best_classical": ("legalbert", best_classical),
         "ablation_legalbert_vs_bert_base": ("legalbert", "bert_base"),
         "bert_base_vs_best_classical": ("bert_base", best_classical)}
comparison = {}
for key, (a, b) in pairs.items():
    comparison[key] = {"all_27": C.paired_bootstrap(y, PRED[a], PRED[b], grp, labels=LABELS, n_boot=1000),
                       f"robust_{len(ROBUST)}": C.paired_bootstrap(y, PRED[a], PRED[b], grp, labels=ROBUST, n_boot=1000)}
    for which, pb in comparison[key].items():
        print(f"{a} - {b} ({which}): {pb['diff']:+.4f}, 95 % CI [{pb['ci_low']:+.4f}, {pb['ci_high']:+.4f}], "
              f"share of resamples where {a} is not better: {pb['p_a_not_better']:.3f}")
h1 = comparison["H1_legalbert_vs_best_classical"]["all_27"]
print(f"\nH1 (objective O3): Legal-BERT > best TF-IDF -> {'SUPPORTED' if h1['ci_low'] > 0 else 'NOT supported'} "
      "(the 95 % CI of the difference must lie above 0)")
json.dump(comparison, open(C.RESULTS_DIR / "transformer_comparison.json", "w"), indent=2)
''')

code(r'''
# Controlled ablation table -> results/ablation.csv
tl = json.load(open(C.RESULTS_DIR / "token_lengths.json"))
seeds = pd.read_csv(C.RESULTS_DIR / "transformer_seeds.csv")
seeds = seeds[seeds.class_weighted == choice["use_class_weights"]].set_index("model")
abl = comparison["ablation_legalbert_vs_bert_base"]["all_27"]
abl_rows = []
for key, name, tk in [("legalbert", "Legal-BERT", "Legal-BERT"), ("bert", "BERT-base", "BERT-base")]:
    r42 = tsearch.set_index("run_id").loc[TEST_RUNS[key]]
    m = METRICS[f"{TEST_NAMES[key]}__test"]
    abl_rows.append({"model": name, "checkpoint": F.CHECKPOINTS[key], "lr": f"{choice['best_lr']:.0e}",
                     "class_weighted": choice["use_class_weights"],
                     "val_macro_f1_seed42": r42.val_macro_f1, "val_macro_f1_3seeds_mean": seeds.loc[key, "mean"],
                     "val_macro_f1_3seeds_std": seeds.loc[key, "std"], "test_macro_f1": m["macro_f1"],
                     "test_macro_f1_robust": m["macro_f1_robust"], "test_accuracy": m["accuracy"],
                     "mean_tokens_per_clause": tl["stats"][tk]["mean"]})
ablation = pd.DataFrame(abl_rows)
diff_row = {"model": "difference (Legal-BERT - BERT-base)",
            "val_macro_f1_seed42": ablation.val_macro_f1_seed42[0] - ablation.val_macro_f1_seed42[1],
            "val_macro_f1_3seeds_mean": ablation.val_macro_f1_3seeds_mean[0] - ablation.val_macro_f1_3seeds_mean[1],
            "test_macro_f1": abl["diff"], "test_macro_f1_diff_95ci": f"[{abl['ci_low']:.4f}, {abl['ci_high']:.4f}]",
            "test_macro_f1_robust": ablation.test_macro_f1_robust[0] - ablation.test_macro_f1_robust[1],
            "test_accuracy": ablation.test_accuracy[0] - ablation.test_accuracy[1],
            "mean_tokens_per_clause": ablation.mean_tokens_per_clause[0] - ablation.mean_tokens_per_clause[1]}
ablation = pd.concat([ablation, pd.DataFrame([diff_row])], ignore_index=True)
ablation.to_csv(C.RESULTS_DIR / "ablation.csv", index=False)
bert_own = tsearch[(tsearch.model == "bert") & (tsearch.seed == 42) & (~tsearch.class_weighted)].sort_values(
    "val_macro_f1", ascending=False).iloc[0]
print(f"Sensitivity check: BERT-base's own best LR on validation is {bert_own.lr:.0e} (val macro-F1 "
      f"{bert_own.val_macro_f1:.4f}); at the shared LR {choice['best_lr']:.0e} it scores "
      f"{tsearch[(tsearch.model == 'bert') & (tsearch.seed == 42) & (tsearch.lr == choice['best_lr']) & (~tsearch.class_weighted)].val_macro_f1.iloc[0]:.4f}.")
ablation.round(4)
''')

interp("results5")

md(r"""
### 5.8 Per-class analysis (27 classes)
Per-class test F1 with contract-level bootstrap CIs for Legal-BERT, BERT-base and the best classical model; a focus
table for the 7 classes that are new in the 27-class design and for Exclusivity (the class with the lowest recall so
far); objective O5 (recall ≥ 0.70 for every class); and the effect of class size on F1.
""")

code(r'''
pcs = {}
for name in [best_classical, "bert_base", "legalbert"]:
    pc_ = C.per_class_bootstrap_ci(y, PRED[name], grp, LABELS, n_boot=1000)
    pc_["recall"] = pd.DataFrame(METRICS[f"{name}__test"]["per_class"]).T.recall
    pcs[name] = pc_
pcf = pd.DataFrame({n: pcs[n].f1 for n in pcs}).rename(columns={best_classical: "best classical", "bert_base": "BERT-base",
                                                                  "legalbert": "Legal-BERT"})
pcf["Legal-BERT CI"] = [("n/a (1 contract)" if pd.isna(l) else f"[{l:.2f}, {h:.2f}]")
                        for l, h in zip(pcs["legalbert"].ci_low, pcs["legalbert"].ci_high)]
pcf["Legal-BERT recall"] = pcs["legalbert"].recall
pcf["Legal-BERT - classical"] = pcf["Legal-BERT"] - pcf["best classical"]
pcf["Legal-BERT - BERT-base"] = pcf["Legal-BERT"] - pcf["BERT-base"]
pcf["test clauses"] = pcs["legalbert"].support
pcf["new class"] = ["yes" if c in NEW_CLASSES else "" for c in pcf.index]
pcf = pcf.sort_values("Legal-BERT")
pcf.to_csv(C.RESULTS_DIR / "per_class_f1_models.csv")
display(pcf.round(3))

fig, ax = plt.subplots(figsize=(10, 9))
yy = np.arange(len(pcf))
ax.hlines(yy, pcf[["best classical", "BERT-base", "Legal-BERT"]].min(axis=1),
          pcf[["best classical", "BERT-base", "Legal-BERT"]].max(axis=1), color="#cfcdc6", linewidth=2, zorder=1)
for lab, col, mk in [("best classical", PL.AQUA, "D"), ("BERT-base", PL.ORANGE, "s"), ("Legal-BERT", PL.BLUE, "o")]:
    ax.scatter(pcf[lab], yy, color=col, marker=mk, s=46, label=lab, zorder=3)
ax.set_yticks(yy)
ax.set_yticklabels([f"{c}{' *' if c in NEW_CLASSES else ''} (n={n})" for c, n in zip(pcf.index, pcf["test clauses"])], fontsize=8.5)
ax.set_xlabel("Test F1 per class"); ax.legend(loc="lower right"); ax.grid(axis="y", visible=False)
ax.set_title("Figure 15. Per-class test F1: best classical model, BERT-base and Legal-BERT (* = new class)", fontsize=11)
PL.savefig(fig, 15, "per_class_f1_models"); plt.show()
''')

code(r'''
# Focus: the 7 new classes and Exclusivity
focus = NEW_CLASSES + ["Exclusivity"]
foc = pd.DataFrame({
    "test clauses": pcs["legalbert"].support[focus], "test contracts": pcs["legalbert"].test_contracts[focus],
    "classical F1": pcs[best_classical].f1[focus], "BERT-base F1": pcs["bert_base"].f1[focus],
    "Legal-BERT F1": pcs["legalbert"].f1[focus],
    "Legal-BERT 95% CI": [("n/a (1 contract)" if pd.isna(l) else f"[{l:.2f}, {h:.2f}]")
                          for l, h in zip(pcs["legalbert"].ci_low[focus], pcs["legalbert"].ci_high[focus])],
    "classical recall": pcs[best_classical].recall[focus], "Legal-BERT recall": pcs["legalbert"].recall[focus]})
foc["best of the three"] = foc[["classical F1", "BERT-base F1", "Legal-BERT F1"]].idxmax(axis=1).str.replace(" F1", "")
foc.to_csv(C.RESULTS_DIR / "per_class_focus_new_classes.csv")
display(foc.round(3))

def confusions_of(name, cls):
    t = pd.Series(PRED[name][y == cls]).value_counts()
    return ", ".join(f"{k} {v}" for k, v in t.items() if k != cls) or "none"
print("Where the clauses of each focus class go (Legal-BERT, test):")
for c in focus:
    print(f"  {c:<34} -> {confusions_of('legalbert', c)}")
''')

code(r'''
# Objective O5 (recall >= 0.70 for every class) and the effect of class size
lb_rec = pcs["legalbert"].recall
below = lb_rec[lb_rec < 0.70].sort_values()
print(f"Legal-BERT classes with test recall < 0.70: {len(below)} -> " + (", ".join(f"{k} {v:.2f}" for k, v in below.items()) or "none"))
print("Objective O5 for Legal-BERT:", "MET" if below.empty else "NOT MET")
json.dump({"model": "legalbert", "classes_below_0.70_recall": below.round(4).to_dict(), "O5_met": bool(below.empty)},
          open(C.RESULTS_DIR / "objective_O5_legalbert.json", "w"), indent=2)

ntrain = train.category.value_counts()
fig, ax = plt.subplots(figsize=(8, 5))
for name, col, mk in [(best_classical, PL.AQUA, "D"), ("legalbert", PL.BLUE, "o")]:
    ax.scatter(ntrain[LABELS], pcs[name].f1[LABELS], color=col, marker=mk, s=40, label=name, alpha=0.85)
for c in focus:
    ax.annotate(c, (ntrain[c], pcs["legalbert"].f1[c]), fontsize=7, xytext=(4, -3), textcoords="offset points")
ax.set_xscale("log"); ax.set_xlabel("Training clauses of the class (log scale)"); ax.set_ylabel("Test F1")
ax.legend(loc="lower right")
from scipy.stats import spearmanr
rho = {n: spearmanr(ntrain[LABELS], pcs[n].f1[LABELS]).correlation for n in [best_classical, "legalbert"]}
ax.set_title(f"Figure 16. Class size vs test F1 (Spearman rho: classical {rho[best_classical]:.2f}, Legal-BERT {rho['legalbert']:.2f})",
             fontsize=10.5)
PL.savefig(fig, 16, "class_size_vs_f1"); plt.show()
json.dump(rho, open(C.RESULTS_DIR / "class_size_vs_f1_spearman.json", "w"), indent=2)
''')

code(r'''
for key, name, num, title in [("legalbert", "legalbert", 17, "Legal-BERT"), ("bert", "bert_base", 18, "BERT-base")]:
    cmx = np.array(METRICS[f"{name}__test"]["confusion_matrix"])
    plot_cm(cmx / cmx.sum(axis=1, keepdims=True), ".2f",
            f"Figure {num}. Confusion matrix of {title} on test, normalised by true class (diagonal = recall)",
            num, f"cm_{name}_normalised", vmax=1)
''')

interp("perclass")

md(r"""
### 5.9 Sensitivity checks (validation only)
Two questions remained after the main runs: is the best learning rate perhaps *above* the pre-declared grid (both models
were best at its top value, 5e-5), and would BERT-base close the gap with more epochs (it was still improving at epoch 5)?
Both are answered on the **validation set only**, with seed 42 and otherwise the identical recipe; the test models, the
test evaluations and the controlled ablation (5 epochs for both models) are unchanged. The runs were made with
`python scripts/train_sensitivity.py` and logged separately (`results/transformer_sensitivity_*.csv`).
""")

code(r"""
sens = pd.read_csv(C.RESULTS_DIR / "transformer_sensitivity_search.csv")
main42 = tsearch[(tsearch.seed == 42) & (~tsearch.class_weighted)]
lr_rows = pd.concat([main42[["model", "lr", "val_macro_f1"]].assign(source="main grid"),
                     sens[sens.epochs == 5][["model", "lr", "val_macro_f1"]].assign(source="sensitivity")])
lr_tab = lr_rows.pivot_table(index="lr", columns="model", values="val_macro_f1").rename(columns={"bert": "BERT-base", "legalbert": "Legal-BERT"})
lr_tab.index = [f"{v:.0e}" + (" (above grid)" if v > 5e-5 else "") for v in lr_tab.index]
print("Validation macro-F1 by learning rate (seed 42, 5 epochs):")
display(lr_tab.round(4))
lr_tab.to_csv(C.RESULTS_DIR / "transformer_lr_sensitivity.csv")

e8 = sens[sens.run_id == "bert_lr5e-05_s42_sens_e8"].iloc[0]
lb42 = tsearch.set_index("run_id").loc[TEST_RUNS["legalbert"]].val_macro_f1
bb42 = tsearch.set_index("run_id").loc[TEST_RUNS["bert"]].val_macro_f1
slog = pd.read_csv(C.RESULTS_DIR / "transformer_sensitivity_log.csv")
curve = slog[(slog.run_id == e8.run_id) & (slog.kind == "epoch")][["epoch", "val_loss", "val_macro_f1"]]
print("BERT-base, lr 5e-5, seed 42, trained for 8 epochs (validation):")
display(curve.round(4).set_index("epoch").T)
epoch_sens = {"bert_5_epochs": float(bb42), "bert_8_epochs_best": float(e8.val_macro_f1), "bert_8_epochs_best_epoch": int(e8.best_epoch),
              "legalbert_5_epochs": float(lb42), "gap_5_epochs": float(lb42 - bb42), "gap_vs_bert_8_epochs": float(lb42 - e8.val_macro_f1)}
json.dump(epoch_sens, open(C.RESULTS_DIR / "transformer_epoch_sensitivity.json", "w"), indent=2)
print(f"Validation gap Legal-BERT - BERT-base: {epoch_sens['gap_5_epochs']:+.4f} with 5 epochs -> "
      f"{epoch_sens['gap_vs_bert_8_epochs']:+.4f} if BERT-base gets 8 epochs")
""")

interp("sensitivity")
