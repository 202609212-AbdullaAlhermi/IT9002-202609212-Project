"""One consistent look for every figure, and one function that saves it as figures/figNN_name.png.

Colours: a colour-blind-checked categorical palette (fixed order, never cycled) and a
single-hue blue ramp for heatmaps / confusion matrices.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

# Categorical slots, used in this order.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
SPLIT_COLORS = {"train": BLUE, "validation": ORANGE, "test": AQUA}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# Sequential blue ramp (light = low, dark = high) for heatmaps.
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "legend.frameon": False,
    })


def savefig(fig, number: int, name: str) -> Path:
    """Save as figures/fig<NN>_<name>.png (NN = the figure number used in the report)."""
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / f"fig{number:02d}_{name}.png"
    fig.savefig(path)
    return path
