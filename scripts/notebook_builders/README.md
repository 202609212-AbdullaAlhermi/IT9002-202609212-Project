# Notebook builders (Task 1)

`notebooks/STUDENTID_Task1.ipynb` is generated from these files, so every cell can be read and reviewed as code:

- `build_task1_partA.py` — sections 0–3 (setup, problem, dataset, preprocessing); it `exec`s the other parts
- `build_task1_partB.py` — section 4 (classical baselines)
- `build_task1_partC.py` — section 5 (fine-tuned transformer + ablation)
- `task1_interp.py` — the interpretation paragraphs (written after looking at the real outputs)

Run `python scripts/notebook_builders/build_task1_partA.py` to rewrite the notebook (outputs are cleared), then execute it with
`jupyter nbconvert --to notebook --execute --inplace notebooks/STUDENTID_Task1.ipynb`.

Other builders: `build_task1_partD.py` (section 6, LEDGAR), `build_task1_partE.py` (section 7, evaluation),
`build_task1_partF.py` (sections 8-11, Part F, objectives, DEMO), `build_task2.py` + `task2_interp.py` (Task 2 notebook),
`build_llm.py` + `llm_interp.py` (LLM notebook). The executed notebooks are saved as `<StudentID>_*.ipynb`.
