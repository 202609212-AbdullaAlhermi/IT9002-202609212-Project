"""The "Run on Google Colab" cell, shared by the three notebook builders (inserted right after each notebook's title).

On your own computer the cell does nothing. On Colab it gets the project (git clone of the repository if its URL is known,
otherwise the submission ZIP from Google Drive), installs the pinned requirements (keeping Colab's own CUDA torch and
Jupyter), and moves into the project's notebooks/ folder; the normal setup cell then finds the project root and fetches or
rebuilds the data.

IT9002_SIMULATE_COLAB=1 makes the cell take the Colab path on a normal computer for testing:
no Drive mount, the ZIP path and the /content folder come from IT9002_COLAB_ZIP / IT9002_COLAB_BASE, and pip only does a
dry run so the local environment is not changed.
"""

COLAB_MD = r"""
## ▶ Run on Google Colab (skip this on your own computer)
**For markers using Google Colab.** Runtime → Change runtime type → **GPU** (T4 is enough), then Runtime → **Run all**.
The cell below gets the project — from the GitHub repository if its URL is set, otherwise from the submission ZIP uploaded
to Google Drive (default `MyDrive/IT9002_202609212.zip`; change `ZIP_PATH` if it is elsewhere) — installs the pinned
packages and moves into the project folder. On your own computer it does nothing. Colab starts without the trained models
(`models/`, too large to share), so cells that need them print **SKIPPED** with retraining instructions and then show the
outputs those cells produced in the real run (tables, figures, text, labelled SAVED); the DEMO sections do the same.
"""

COLAB_CODE = r'''
# ==== RUN ON GOOGLE COLAB - does nothing on your own computer ====
import os, sys, subprocess, zipfile
from pathlib import Path
SIMULATE = os.environ.get("IT9002_SIMULATE_COLAB") == "1"         # local test of the Colab path
IN_COLAB = "google.colab" in sys.modules or SIMULATE
GITHUB_URL = "[GitHub repository URL]"                             # the project's GitHub repository (as in project_info.json)
ZIP_PATH = "/content/drive/MyDrive/IT9002_202609212.zip"           # used when the repository cannot be cloned

def find_root(start):
    """The project folder = the first folder, going upwards, that contains project_info.json."""
    return next((d for d in [start, *start.parents] if (d / "project_info.json").exists()), None)

if IN_COLAB:
    base = Path(os.environ.get("IT9002_COLAB_BASE", "/content"))
    root = find_root(Path.cwd())
    if root is None and GITHUB_URL.startswith("https://github.com/"):            # a) clone the repository
        clone = base / "IT9002_project"
        if not clone.exists():
            # no password prompt: a private repository simply fails here and the ZIP route below is used instead
            r = subprocess.run(["git", "-c", "credential.helper=", "clone", "--depth", "1", GITHUB_URL, str(clone)],
                               env=dict(os.environ, GIT_TERMINAL_PROMPT="0"), capture_output=True, text=True)
            if r.returncode != 0:
                print("Could not clone the repository (it may be private); using the submission ZIP from Google Drive.")
        root = clone if (clone / "project_info.json").exists() else None
    if root is None:                                                             # b) unzip the submission ZIP from Drive
        if not SIMULATE:
            from google.colab import drive
            drive.mount("/content/drive")
        zp = Path(os.environ.get("IT9002_COLAB_ZIP", ZIP_PATH))
        assert zp.exists(), f"Upload the submission ZIP to Google Drive and set ZIP_PATH (not found: {zp})"
        with zipfile.ZipFile(zp) as z:
            root = base / z.namelist()[0].split("/")[0]
            if not root.exists():
                z.extractall(base)
    os.chdir(root / "notebooks")
    # pinned requirements, but keep Colab's own CUDA build of torch and its Jupyter packages
    keep = ("torch", "jupyterlab", "ipykernel", "nbformat", "nbconvert", "nbclient")
    reqs = [l.strip() for l in (root / "requirements.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#") and not l.split("==")[0].strip().lower().startswith(keep)]
    (root / "requirements_colab.txt").write_text("\n".join(reqs) + "\n")
    loaded = {m: sys.modules[m].__version__ for m in ("numpy", "pandas", "scipy", "sklearn") if m in sys.modules}
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(root / "requirements_colab.txt")]
                   + (["--dry-run"] if SIMULATE else []), check=True)
    pins = dict(l.split("==") for l in reqs if "==" in l)
    stale = [m for m, v in loaded.items() if pins.get({"sklearn": "scikit-learn"}.get(m, m), v) != v]
    if stale and not SIMULATE:
        raise RuntimeError(f"New versions of {stale} were installed: Runtime -> Restart session, then Run all again.")
    from importlib.metadata import version
    print(f"Colab: project folder {root} | pinned requirements {'checked (dry run)' if SIMULATE else 'installed'} | "
          f"torch {version('torch')} kept")
else:
    print("Not on Colab: nothing to do here.")
'''
