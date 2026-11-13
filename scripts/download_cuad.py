"""
Download the official CUAD v1 release and extract the two files this project needs.

Source : https://github.com/TheAtticusProject/cuad  (data.zip)   |   licence: CC BY 4.0
Output : data/raw/CUADv1.json            all 510 contracts, SQuAD-style, 41 question categories
         data/raw/cuad_official_test.json  the official CUAD test contracts (102)
         data/raw/category_descriptions.csv  expert definitions of the 41 categories

Run from the project root:  python scripts/download_cuad.py
Needed whenever data/raw/ is empty: the raw data is never in the submission ZIP or the repository (the notebooks call this
script automatically through src/common.py ensure_data()).
"""
import io
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BASE = "https://raw.githubusercontent.com/TheAtticusProject/cuad/main/"


def fetch(name: str) -> bytes:
    with urllib.request.urlopen(BASE + name, timeout=120) as r:
        return r.read()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(io.BytesIO(fetch("data.zip")))
    (RAW / "CUADv1.json").write_bytes(z.read("CUADv1.json"))
    (RAW / "cuad_official_test.json").write_bytes(z.read("test.json"))
    (RAW / "category_descriptions.csv").write_bytes(fetch("category_descriptions.csv"))
    print("saved:", *sorted(p.name for p in RAW.iterdir()))


if __name__ == "__main__":
    main()
