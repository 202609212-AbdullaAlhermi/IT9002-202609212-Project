# IT9002 NLP Individual Project — clause classification and clause extraction on CUAD

Student: Abdulla Alhermi · ID: 202609212 · IT9002 Natural Language Processing, Bahrain Polytechnic

The report, video and supporting documents are in the Moodle submission.

## Notebooks
- `notebooks/202609212_Task1.ipynb` — Task 1, clause-type classification (27 classes): data, preprocessing, classical models,
  Legal-BERT and the BERT-base ablation, the LEDGAR comparison, evaluation, Part F1 (anomaly flagging), Part F3 (LoRA), DEMO.
- `notebooks/202609212_Task2.ipynb` — Task 2, clause extraction from whole contracts: baseline, Legal-BERT tagger, CUAD
  comparison, error analysis, Part F2 (extract → verify → flag pipeline), DEMO.
- `notebooks/202609212_LLM.ipynb` — Task 1 by prompting two local LLMs: prompt versions, output parser, cost and latency, DEMO.

The notebooks are saved with the outputs of the real run. Heavy steps have switches (`RETRAIN = False`, `RERUN = False`,
`RUN_SEARCH = False`) that load saved results, so a re-run takes minutes.

## Run locally
1. Python 3.11 with an NVIDIA GPU for the transformer parts (tested on Windows 11, RTX 5080 16 GB, torch 2.11 + CUDA 12.8).
   Install the CUDA build of PyTorch first (https://pytorch.org/get-started/locally/), then:
   ```
   pip install -r requirements.txt
   ```
2. Open a notebook from the `notebooks/` folder and run it top to bottom. The first setup cell downloads the official CUAD
   release into `data/raw/` and checks the frozen CSVs in `data/processed/` against their SHA-256 checksums.
3. The trained models (`models/`, about 4 GB) are not in the repository. Cells that need them start with
   `%%needs_models <folder>` and print **SKIPPED** with how to retrain them (e.g. set `RETRAIN = True`, or run
   `scripts/train_transformer.py`, `scripts/train_task2.py`, `scripts/anomaly_features.py`), then show the outputs those
   cells produced in the real run (tables, figures and text, labelled SAVED; stored in `results/saved_cell_outputs.json` by
   `scripts/save_cell_outputs.py`); every other cell runs, and the DEMO sections do the same where a model is missing.

## Run on Google Colab
Each notebook starts with a **"Run on Google Colab"** cell (it does nothing on your own computer).
1. Open the notebook in Colab (File → Open notebook → GitHub, or upload the `.ipynb`).
2. Runtime → Change runtime type → **GPU** (a T4 is enough).
3. Runtime → **Run all**. The first cell gets the project (a clone of this repository, or the submission ZIP from Google Drive,
   `MyDrive/IT9002_202609212.zip`; change `ZIP_PATH` if it is elsewhere) and installs the pinned packages while keeping Colab's
   own torch. If it asks for it, choose Runtime → Restart session and Run all again.
4. As locally, cells that need the trained models print **SKIPPED** and show their saved real-run outputs; the LLM DEMO
   downloads the two Qwen models (about 9 GB).

The repository is public, so Colab clones it directly; the ZIP route is the fallback if it cannot be reached.

## Data
- **CUAD v1** — Hendrycks, Burns, Chen & Ball (2021), NeurIPS Datasets and Benchmarks. Official page:
  https://www.atticusprojectai.org/cuad · data and code: https://github.com/TheAtticusProject/cuad · licence **CC BY 4.0**.
  The raw CUAD file is not in this repository (`scripts/download_cuad.py` downloads it); `data/processed/` holds the frozen
  processed CSVs, reproducible with `scripts/build_dataset_v2.py` and verified against `data/processed/CHECKSUMS_v2.sha256`.
- **LEDGAR** (LexGLUE, `coastalcph/lex_glue`, config `ledgar`) — https://huggingface.co/datasets/coastalcph/lex_glue · licence
  **CC BY 4.0** (dataset card); downloaded on demand by `scripts/run_ledgar.py`.

## Model licences
- `nlpaueb/legal-bert-base-uncased` — CC BY-SA 4.0 · `bert-base-uncased` — Apache-2.0
- `Qwen/Qwen2.5-1.5B-Instruct` — Apache-2.0 · `Qwen/Qwen2.5-3B-Instruct` — Qwen Research Licence (research or evaluation use
  only; "Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved.")
