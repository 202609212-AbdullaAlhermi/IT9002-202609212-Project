"""Tiny hand-made tests for the Task 2 decoder and span metrics.   Run:  python tests/test_task2.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import span_metrics as M  # noqa: E402
from src import task2_data as D  # noqa: E402


def test_iou_and_match():
    assert M.char_iou((0, 10), (5, 15)) == 5 / 15
    assert M.char_iou((0, 10), (20, 30)) == 0
    # one gold, two predictions: only the more confident overlapping one is a TP (one-to-one)
    tp, fp, fn, _ = M.match([(0, 100)], [(0, 90, 0.9), (10, 100, 0.8)])
    assert (tp, fp, fn) == (1, 1, 0)
    # IoU below 0.5 is no match
    assert M.match([(0, 100)], [(0, 40, 0.9)])[:3] == (0, 1, 1)
    # exact mode uses the snapping function
    snap = lambda s, e: (s // 10, (e - 1) // 10)
    assert M.match([(0, 100)], [(3, 98, 0.5)], mode="exact", snap=snap)[:3] == (1, 0, 0)
    assert M.match([(0, 100)], [(12, 98, 0.5)], mode="exact", snap=snap)[:3] == (0, 1, 1)


def test_labels_and_decoder_roundtrip():
    # 10 tokens of 5 characters each; two touching spans of category 0 and one span of category 1
    offsets = np.array([(5 * i, 5 * i + 4) for i in range(10)])
    spans = [(0, 0, 14), (0, 15, 29), (1, 35, 49)]
    y = D.token_labels(offsets, spans, n_categories=2)
    assert y[:, 0].tolist() == [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]          # INSIDE cat 0
    assert y[:, 2].tolist() == [1, 0, 0, 1, 0, 0, 0, 0, 0, 0]          # BEGIN cat 0: two starts
    dec = D.decode(y.astype(float), offsets, 2)
    assert [(c, s, e) for c, s, e, _ in dec] == [(0, 0, 14), (0, 15, 29), (1, 35, 49)]
    # without BEGIN the two touching spans merge into one
    dec2 = D.decode(y.astype(float), offsets, 2, use_begin=False)
    assert [(c, s, e) for c, s, e, _ in dec2] == [(0, 0, 29), (1, 35, 49)]
    # min_tokens drops short spans
    assert len(D.decode(y.astype(float), offsets, 2, min_tokens=4)) == 0


def test_windows_and_averaging():
    w = D.make_windows(10, 4, 2)
    assert w[0] == (0, 4) and w[-1] == (6, 10) and all(e - s == 4 for s, e in w)
    p = D.average_window_probs(6, [(0, 4), (2, 6)], [np.ones((4, 1)), np.zeros((4, 1))], 1)
    assert p[:, 0].tolist() == [1, 1, 0.5, 0.5, 0, 0]


def test_counts_summary_and_cuad():
    gold = pd.DataFrame({"contract": ["a", "a"], "category": ["X", "Y"], "start": [0, 50], "end": [40, 90]})
    pred = pd.DataFrame({"contract": ["a", "b"], "category": ["X", "X"], "start": [0, 0], "end": [38, 10],
                         "confidence": [0.9, 0.4]})
    counts = M.count_table(gold, pred, ["a", "b"], ["X", "Y"])
    s = M.summarise(counts, ["X", "Y"])
    # X: 1 TP, 1 FP (in contract b, which has no gold X); Y: 1 FN
    assert s["per_category"].loc["X", "precision"] == 0.5 and s["per_category"].loc["Y", "recall"] == 0
    assert abs(s["macro_f1"] - (2 / 3 + 0) / 2) < 1e-9
    texts = {"a": "The licensee shall not assign this agreement. " + "x" * 100, "b": "y" * 50}
    text_of = lambda c, st, e: texts[c][st:e]
    cur = M.cuad_curve(gold, pred, text_of, [(c, k) for c in ["a", "b"] for k in ["X", "Y"]])
    assert 0 <= cur["aupr"] <= 1 and cur["recalls"][0] == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)
