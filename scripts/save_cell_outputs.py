"""Store the real-run outputs of every %%needs_models cell in results/saved_cell_outputs.json.

The trained models (models/) are not in the repository, so on Google Colab these cells are skipped. With this file the
skip message is followed by the outputs the cell produced in the real run (tables, figures, printed text), which the
notebooks label as SAVED. Run this after executing the notebooks with the models present:

    python scripts/save_cell_outputs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import SAVED_CELL_OUTPUTS, cell_key  # noqa: E402

NOTEBOOKS = ["202609212_Task1.ipynb", "202609212_Task2.ipynb", "202609212_LLM.ipynb"]
KEEP = ("text/html", "text/markdown", "image/png", "image/jpeg", "image/svg+xml", "text/plain")


def main() -> None:
    saved = {}
    for name in NOTEBOOKS:
        nb = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
        n = 0
        for c in nb["cells"]:
            src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            if c["cell_type"] != "code" or not src.startswith("%%needs_models"):
                continue
            body = src.split("\n", 1)[1] if "\n" in src else ""
            outs = []
            for o in c.get("outputs", []):
                if o["output_type"] == "stream" and o.get("name") == "stdout":
                    text = "".join(o["text"]) if isinstance(o["text"], list) else o["text"]
                    if text.startswith("SKIPPED - this cell needs"):   # an output of a skipped run is not a real result
                        outs = []
                        break
                    outs.append({"output_type": "stream", "name": "stdout", "text": text})
                elif o["output_type"] in ("display_data", "execute_result"):
                    data = {k: v for k, v in o.get("data", {}).items() if k in KEEP}
                    if data:
                        outs.append({"output_type": "display_data", "data": data, "metadata": o.get("metadata", {})})
            if outs:
                saved[cell_key(body)] = outs
                n += 1
        print(f"{name}: {n} cells with saved outputs")
    SAVED_CELL_OUTPUTS.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
    print(f"-> {SAVED_CELL_OUTPUTS.relative_to(ROOT)} ({SAVED_CELL_OUTPUTS.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
