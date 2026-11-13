"""Task 2 (clause extraction) data preparation: whole contracts -> word-piece tokens with character offsets ->
sliding windows -> 54 binary targets per token (INSIDE and BEGIN for each of the 27 categories) -> and back to spans.

    tokenize_contract(text, tok)        -> token character offsets (no text normalisation: offsets stay exact)
    token_labels(offsets, spans, cats)  -> (n_tokens, 2 * n_categories) uint8: [:, :K] INSIDE, [:, K:] BEGIN
    make_windows(n_tokens, win, stride) -> list of (start, end) token ranges covering the contract
    decode(probs, offsets, ...)         -> list of (category index, char_start, char_end, confidence)
    average_window_probs(...)           -> per-token probabilities averaged over the overlapping windows

Plain-language summary: a contract is far longer than a transformer can read at once, so it is cut into overlapping
windows of word-pieces. For every word-piece the model answers 54 yes/no questions: "is this word-piece inside a clause of
category c?" and "does a clause of category c START here?". The BEGIN answers are needed because two clauses of the same
category often touch each other; without them the decoder would merge them into one long span.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Tokenisation with character offsets
# ---------------------------------------------------------------------------
def tokenize_contract(text: str, tokenizer) -> tuple[list[int], np.ndarray]:
    """Word-piece ids and (n_tokens, 2) character offsets of the WHOLE contract (no special tokens, no truncation).
    The text is passed unchanged, so every offset points into the original contract string."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    return enc["input_ids"], np.asarray(enc["offset_mapping"], dtype=np.int64)


def make_windows(n_tokens: int, window: int, stride: int) -> list[tuple[int, int]]:
    """Token ranges [start, end) of overlapping windows. `window` counts content tokens (the model input adds [CLS] and
    [SEP]); consecutive windows start `stride` tokens apart, so neighbouring windows overlap by window - stride tokens."""
    if n_tokens <= window:
        return [(0, n_tokens)]
    starts = list(range(0, n_tokens - window, stride)) + [n_tokens - window]
    return [(s, s + window) for s in sorted(set(starts))]


# ---------------------------------------------------------------------------
# Labels: INSIDE and BEGIN per category
# ---------------------------------------------------------------------------
def token_labels(offsets: np.ndarray, spans, n_categories: int) -> np.ndarray:
    """spans: iterable of (category index, char_start, char_end).
    INSIDE[t, c] = 1 if token t overlaps a gold span of category c by at least one character.
    BEGIN[t, c]  = 1 if token t is the first token that overlaps that span."""
    y = np.zeros((len(offsets), 2 * n_categories), dtype=np.uint8)
    tok_start, tok_end = offsets[:, 0], offsets[:, 1]
    for c, s, e in spans:
        idx = np.flatnonzero((tok_end > s) & (tok_start < e))
        if len(idx) == 0:
            continue
        y[idx, c] = 1
        y[idx[0], n_categories + c] = 1
    return y


def span_token_range(offsets: np.ndarray, start: int, end: int) -> tuple[int, int] | None:
    """First and last token index overlapping the character range (None if no token overlaps)."""
    idx = np.flatnonzero((offsets[:, 1] > start) & (offsets[:, 0] < end))
    return (int(idx[0]), int(idx[-1])) if len(idx) else None


# ---------------------------------------------------------------------------
# Window probabilities -> one probability per token
# ---------------------------------------------------------------------------
def average_window_probs(n_tokens: int, windows, window_probs, n_out: int) -> np.ndarray:
    """window_probs[i]: (end - start, n_out) probabilities for the content tokens of window i. Tokens covered by
    several windows get the mean of their predictions."""
    total = np.zeros((n_tokens, n_out), dtype=np.float32)
    count = np.zeros((n_tokens, 1), dtype=np.float32)
    for (s, e), p in zip(windows, window_probs):
        total[s:e] += p
        count[s:e] += 1
    return total / np.maximum(count, 1)


# ---------------------------------------------------------------------------
# Decoder: per-token probabilities -> character spans
# ---------------------------------------------------------------------------
def decode(probs: np.ndarray, offsets: np.ndarray, n_categories: int, theta_inside: float = 0.5,
           theta_begin: float = 0.5, min_tokens: int = 1, use_begin: bool = True):
    """probs: (n_tokens, 2K) with INSIDE in [:, :K] and BEGIN in [:, K:].
    For each category: a span starts at a token with p_inside >= theta_inside when the previous token is outside, or
    (if use_begin) when p_begin >= theta_begin; it continues while p_inside >= theta_inside and ends before the next start.
    Confidence = mean p_inside over the span's tokens. Spans with fewer than min_tokens tokens are dropped.
    Returns a list of (category, char_start, char_end, confidence)."""
    K = n_categories
    out = []
    inside = probs[:, :K] >= theta_inside
    begin = (probs[:, K:] >= theta_begin) if use_begin else np.zeros_like(inside)
    for c in range(K):
        ins = inside[:, c]
        if not ins.any():
            continue
        starts = ins & (np.r_[True, ~ins[:-1]] | begin[:, c])
        start_idx = np.flatnonzero(starts)
        for k, s in enumerate(start_idx):
            # the span runs until the token before the next start or the first token that is not inside
            stop = start_idx[k + 1] if k + 1 < len(start_idx) else len(ins)
            outside = np.flatnonzero(~ins[s:stop])
            e = s + (outside[0] if len(outside) else stop - s)   # exclusive end token
            if e - s < min_tokens:
                continue
            out.append((c, int(offsets[s, 0]), int(offsets[e - 1, 1]), float(probs[s:e, c].mean())))
    return out


# ---------------------------------------------------------------------------
# The whole corpus, tokenised once and cached (models/ is gitignored; the cache is rebuilt if missing)
# ---------------------------------------------------------------------------
def load_corpus(tokenizer, cache_path, contracts, text_of) -> dict:
    """{contract: {"ids": list[int], "offsets": (n, 2) array}} for the given contracts."""
    import pickle
    from pathlib import Path

    cache_path = Path(cache_path)
    if cache_path.exists():
        corpus = pickle.loads(cache_path.read_bytes())
        if set(contracts) <= set(corpus):
            return corpus
    corpus = {}
    for c in contracts:
        ids, offs = tokenize_contract(text_of(c), tokenizer)
        corpus[c] = {"ids": ids, "offsets": offs}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(pickle.dumps(corpus))
    return corpus
