"""Task 2 baseline: a TF-IDF segment classifier.

    segment(text)                    -> list of (start, end) character ranges (sentences / numbered sub-clauses)
    segment_labels(segs, spans, K)   -> (n_segments, K) 0/1: segment positive for a category if >= 50 % of its characters
                                        lie inside a gold span of that category
    merge_segments(segs, pos, conf)  -> spans: consecutive positive segments of the same category merged into one span

Segmentation heuristic (documented, no learning): the contract text is cut
  * at line breaks (CUAD texts keep the line structure of the filing),
  * after ". ", "; " or ": " when the next character starts a new sentence or item (capital letter, digit, "(" or a quote),
  * before inline enumerations such as "(a)", "(ii)", "1." that follow a space.
Segments shorter than 3 non-space characters are dropped; leading/trailing whitespace is trimmed (offsets stay exact).
"""
from __future__ import annotations

import re

import numpy as np

_BOUNDARY = re.compile(
    r"\n+"                                             # line breaks
    r"|(?<=[.;:])\s+(?=[A-Z0-9(\"'“])"                 # sentence / item end followed by a new start
    r"|\s+(?=\((?:[a-z]|[ivx]{1,4}|\d{1,2})\)\s)"      # inline enumeration "(a) ", "(ii) ", "(3) "
)


def segment(text: str) -> list[tuple[int, int]]:
    segs, pos = [], 0
    for m in _BOUNDARY.finditer(text):
        segs.append((pos, m.start())); pos = m.end()
    segs.append((pos, len(text)))
    out = []
    for s, e in segs:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if len(text[s:e].replace(" ", "")) >= 3:
            out.append((s, e))
    return out


def segment_labels(segs, spans, n_categories: int) -> np.ndarray:
    """spans: iterable of (category index, start, end). Positive if >= 50 % of the segment's characters are covered by
    the union of that category's gold spans."""
    y = np.zeros((len(segs), n_categories), dtype=np.uint8)
    if not segs:
        return y
    starts = np.array([s for s, _ in segs]); ends = np.array([e for _, e in segs]); lens = ends - starts
    for c in range(n_categories):
        cs = sorted((s, e) for k, s, e in spans if k == c)
        if not cs:
            continue
        covered = np.zeros(len(segs))
        for s, e in cs:                                   # (overlapping gold spans of one category are rare)
            covered += np.clip(np.minimum(ends, e) - np.maximum(starts, s), 0, None)
        y[:, c] = (np.minimum(covered, lens) >= 0.5 * lens).astype(np.uint8)
    return y


def merge_segments(segs, positive: np.ndarray, conf: np.ndarray):
    """positive, conf: (n_segments, K). Returns list of (category, start, end, confidence) where runs of consecutive
    positive segments of one category become one span (confidence = mean probability of its segments)."""
    out = []
    for c in range(positive.shape[1]):
        idx = np.flatnonzero(positive[:, c])
        if len(idx) == 0:
            continue
        run = [idx[0]]
        for i in idx[1:]:
            if i == run[-1] + 1:
                run.append(i)
            else:
                out.append((c, segs[run[0]][0], segs[run[-1]][1], float(conf[run, c].mean()))); run = [i]
        out.append((c, segs[run[0]][0], segs[run[-1]][1], float(conf[run, c].mean())))
    return out
