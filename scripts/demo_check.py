"""Run the DEMO cell groups of the three notebooks headlessly: for each notebook, the setup cell(s) at the top plus
every code cell under the "## ... DEMO ..." heading (up to the next "## " heading) are executed in a fresh kernel and timed.

    python scripts/demo_check.py          -> prints a summary and writes results/demo_check.json; exit code 1 on any error
"""
import json
import sys
import time
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB = {"Task1": 3, "Task2": 2, "LLM": 2}   # setup code cells (incl. the Colab cell)
SID = json.loads((ROOT / "project_info.json").read_text(encoding="utf-8"))["STUDENT_ID"]


def demo_cells(nb):
    cells, inside = [], False
    for c in nb.cells:
        if c.cell_type == "markdown" and c.source.lstrip().startswith("## "):
            inside = "DEMO" in c.source.splitlines()[0]
            continue
        if inside and c.cell_type == "code":
            cells.append(c)
    return cells


results, ok = {}, True
for task, n_setup in NB.items():
    # the executed notebooks are <StudentID>_<task>.ipynb; the notebook builders write STUDENTID_<task>.ipynb
    name = next(n for n in (f"{SID}_{task}.ipynb", f"STUDENTID_{task}.ipynb") if (ROOT / "notebooks" / n).exists())
    nb = nbf.read(ROOT / "notebooks" / name, 4)
    setup = [c for c in nb.cells if c.cell_type == "code"][:n_setup]
    demo = demo_cells(nb)
    run = nbf.v4.new_notebook(cells=[nbf.v4.new_code_cell(c.source) for c in setup + demo], metadata=nb.metadata)
    client = NotebookClient(run, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "notebooks")}})
    t0 = time.perf_counter()
    try:
        client.execute()
        err = None
    except Exception as e:                                  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[-400:]}"
        ok = False
    total = time.perf_counter() - t0
    # time of the demo cells alone, from the execution metadata
    demo_secs = 0.0
    for c in run.cells[n_setup:]:
        md = c.get("metadata", {}).get("execution", {})
        if "iopub.execute_input" in md and "shell.execute_reply" in md:
            from datetime import datetime
            f = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
            demo_secs += (f(md["shell.execute_reply"]) - f(md["iopub.execute_input"])).total_seconds()
    results[name] = {"setup_cells": n_setup, "demo_cells": len(demo), "total_seconds_incl_setup": round(total, 1),
                     "demo_cells_seconds": round(demo_secs, 1), "error": err}
    print(f"{name}: {len(demo)} demo cells, demo {demo_secs:.1f} s (total incl. setup and kernel start {total:.1f} s)"
          + (f"  ERROR {err}" if err else "  OK"))
(ROOT / "results" / "demo_check.json").write_text(json.dumps(results, indent=2))
sys.exit(0 if ok else 1)
