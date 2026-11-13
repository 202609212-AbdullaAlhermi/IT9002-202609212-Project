"""Prompted-LLM helpers for Task 1 (27-way clause classification): prompts, output-to-label conversion, batched greedy
generation with incremental saving, and constrained label scoring.

Used by scripts/run_llm.py (the long runs) and the LLM notebook (analysis, test evaluation, demo).
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# One-line category descriptions (v2), adapted and shortened from the official CUAD category descriptions
# (data/raw/category_descriptions.csv, fetched by scripts/download_cuad.py; CC BY 4.0).
# ---------------------------------------------------------------------------
DESCRIPTIONS = {
    "Anti-Assignment": "consent or notice is required if the contract is assigned to a third party.",
    "Audit Rights": "a party may audit the books, records or premises of the other party to check compliance.",
    "Cap On Liability": "liability for breach is capped, or the time to bring claims is limited.",
    "Change Of Control": "a party may terminate, or consent/notice is required, if the other party undergoes a merger, "
                         "acquisition or other change of control.",
    "Competitive Restriction Exception": "an exception or carve-out to a non-compete, exclusivity or no-solicit restriction.",
    "Covenant Not To Sue": "a party may not contest the other party's IP ownership or bring certain claims against it.",
    "Exclusivity": "an exclusive dealing commitment, e.g. buying all requirements from one party or not selling to others.",
    "Expiration Date": "the date or period at which the initial term of the contract ends.",
    "Governing Law": "which state's or country's law governs the contract.",
    "Insurance": "a party must maintain insurance for the benefit of the other party.",
    "Ip Ownership Assignment": "intellectual property created by one party becomes the property of the other party.",
    "Joint Ip Ownership": "intellectual property is jointly or co-owned by the parties.",
    "License Grant": "one party grants the other a license (e.g. to use, make or sell) to its IP or products.",
    "Liquidated Damages": "a fixed amount (liquidated damages) or a termination fee is payable on breach or termination.",
    "Minimum Commitment": "a minimum order, quantity, amount or number of units must be bought per period.",
    "No-Solicit Of Employees": "a party may not solicit or hire the other party's employees or contractors.",
    "Non-Compete": "a party may not compete with the other party or operate in a certain field or territory.",
    "Non-Disparagement": "a party must not disparage or speak negatively about the other party.",
    "Non-Transferable License": "the licensee may not transfer, assign or sublicense the license granted.",
    "Post-Termination Services": "obligations that continue after termination or expiry (transition, wind-down, last buy, "
                                 "payments).",
    "Renewal Term": "the contract renews or extends automatically or by notice after the initial term.",
    "Revenue/Profit Sharing": "a party must share revenue or profit (e.g. royalties, a percentage of sales) with the other.",
    "Rofr/Rofo/Rofn": "a right of first refusal, first offer or first negotiation to buy, license or distribute something.",
    "Source Code Escrow": "source code must be deposited with an escrow agent and released on events such as bankruptcy.",
    "Termination For Convenience": "a party may terminate without cause, simply by giving notice.",
    "Volume Restriction": "a fee increase or consent is required if usage exceeds a certain volume or threshold.",
    "Warranty Duration": "how long a warranty against defects or errors lasts.",
}

SYSTEM = "You are an expert contract lawyer who classifies clauses from commercial contracts."
INSTRUCTION = "Answer with exactly one category name from the list, copied exactly, and nothing else."
VERSIONS = {"v1": "zero-shot, category names only", "v2": "zero-shot, names + one-line descriptions",
            "v3": "few-shot, names + one TRAIN example per category"}
# Extra analysis version (validation only, never selected or run on test).
ANALYSIS_VERSIONS = {"v4": "retrieval few-shot: v2 + the 3 most similar TRAIN clauses (TF-IDF cosine) with their labels"}


def build_messages(version: str, clause: str, labels: list[str], examples: dict | None = None) -> list[dict]:
    """Chat messages for one clause. The clause itself is never truncated.
    v3: `examples` = {category: example text}; v4: `examples` = [(text, category), ...] (the retrieved neighbours)."""
    if version in ("v2", "v4"):
        cat_list = "\n".join(f"- {l}: {DESCRIPTIONS[l]}" for l in labels)
    else:
        cat_list = "\n".join(f"- {l}" for l in labels)
    parts = [f"Classify the contract clause below into exactly one of these {len(labels)} categories:\n{cat_list}"]
    if version == "v3":
        ex = "\n\n".join(f'Clause: """{examples[l]}"""\nCategory: {l}' for l in labels)
        parts.append(f"Examples (one per category):\n\n{ex}")
    if version == "v4":
        ex = "\n\n".join(f'Clause: """{t}"""\nCategory: {l}' for t, l in examples)
        parts.append(f"The most similar clauses from the training data, with their categories:\n\n{ex}")
    parts.append(f'Clause: """{clause}"""')
    parts.append(INSTRUCTION)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def pick_examples(train: pd.DataFrame, labels: list[str], min_words: int = 15, max_words: int = 80) -> pd.DataFrame:
    """Deterministic few-shot rule: for each category, the TRAIN clause (15-80 words, so nothing has to be truncated)
    whose TF-IDF vector is closest (cosine) to the category's TF-IDF centroid, i.e. its most 'typical' short clause."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2)
    X = vec.fit_transform(train.text)                       # rows are L2-normalised
    rows = []
    for l in labels:
        idx = np.flatnonzero((train.category == l).to_numpy())
        centroid = np.asarray(X[idx].mean(axis=0)).ravel()
        centroid /= np.linalg.norm(centroid) or 1.0
        ok = [i for i in idx if min_words <= train.n_words.iloc[i] <= max_words]
        sims = X[ok] @ centroid
        best = ok[int(np.argmax(sims))]
        rows.append({"category": l, "row_id": int(train.row_id.iloc[best]), "contract": train.contract.iloc[best],
                     "n_words": int(train.n_words.iloc[best]), "cosine_to_centroid": float(sims.max()),
                     "text": " ".join(train.text.iloc[best].split())})
    return pd.DataFrame(rows)


class Retriever:
    """v4: the k TRAIN clauses most similar to a clause (TF-IDF cosine, same vectoriser settings as pick_examples);
    each neighbour is shortened to its first `max_words` words so that three examples cannot make the prompt explode."""

    def __init__(self, train: pd.DataFrame, k: int = 3, max_words: int = 80):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(sublinear_tf=True, min_df=2)
        self.X = self.vec.fit_transform(train.text)
        self.texts = [" ".join(t.split()[:max_words]) + (" ..." if len(t.split()) > max_words else "") for t in train.text]
        self.labels = list(train.category)
        self.k = k

    def neighbours(self, clause: str) -> list[tuple[str, str]]:
        sims = (self.X @ self.vec.transform([clause]).T).toarray().ravel()
        top = np.argsort(-sims, kind="stable")[: self.k]
        return [(self.texts[i], self.labels[i]) for i in top]


# ---------------------------------------------------------------------------
# Output-to-label conversion (ONE function, documented and unit-tested in tests/test_llm_parser.py)
# ---------------------------------------------------------------------------
INVALID = "INVALID"
_ALIASES = {"Rofr/Rofo/Rofn": ["rofr", "rofo", "rofn", "right of first refusal", "right of first offer",
                               "right of first negotiation"]}


def normalise(s: str) -> str:
    """Lower case, 'intellectual property' -> 'ip', every non-alphanumeric character -> one space."""
    s = s.lower().replace("intellectual property", "ip")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def parse_label(output: str, labels: list[str]) -> tuple[str, str]:
    """Convert a raw model answer into one of `labels`. Returns (label or INVALID, rule used).

    1. exact      - the first line, after removing a leading 'Category:'/'Answer:' and surrounding quotes, * or a final
                    full stop, is exactly a label name;
    2. normalised - the same line equals a label name ignoring case and punctuation (normalise());
    3. contained  - label names (and the Rofr/Rofo/Rofn spelled-out aliases) found as whole words anywhere in the
                    normalised answer; an occurrence lying inside a longer occurrence is discarded (longest name wins);
                    if the remaining occurrences all name ONE label, that label;
    4. otherwise INVALID (no label found, or two different labels named) - counted as wrong for every class.
    """
    raw = (output or "").strip()
    first = raw.splitlines()[0].strip() if raw else ""
    cand = re.sub(r"^(the\s+)?(category|answer|label)\s*(is)?\s*[:\-]?\s*", "", first, flags=re.I).strip().strip("\"'*`").strip()
    cand = cand.rstrip(".").strip().strip("\"'*`").strip()
    if cand in labels:
        return cand, "exact"
    norm = {normalise(l): l for l in labels}
    if normalise(cand) in norm:
        return norm[normalise(cand)], "normalised"
    text = f" {normalise(raw)} "
    occ = []
    for l in labels:
        for form in [normalise(l)] + _ALIASES.get(l, []):
            for m in re.finditer(r"(?<= )" + re.escape(form) + r"(?= )", text):
                occ.append((m.start(), m.end(), l))
    keep = [o for o in occ if not any(p[0] <= o[0] and o[1] <= p[1] and (p[1] - p[0]) > (o[1] - o[0]) for p in occ)]
    found = {o[2] for o in keep}
    if len(found) == 1:
        return found.pop(), "contained"
    return INVALID, ("ambiguous" if found else "no label")


# ---------------------------------------------------------------------------
# Model runner: bf16 on the GPU, chat template, left padding, pure greedy decoding
# ---------------------------------------------------------------------------
class Runner:
    def __init__(self, model_id: str, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        self.torch, self.model_id, self.device = torch, model_id, device
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        # bfloat16 (used for all reported runs); GPUs without native bfloat16 (e.g. a Colab T4) fall back to float16
        native_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16 if native_bf16 else torch.float16
                                                          ).to(device).eval()
        # The models' own generation_config samples (temperature 0.7, top-p 0.8) and applies a repetition penalty;
        # all of that is switched off: pure greedy decoding, so results are deterministic.
        self.gen = GenerationConfig(do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                                    max_new_tokens=24, pad_token_id=self.tok.pad_token_id,
                                    eos_token_id=self.model.generation_config.eos_token_id)
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def render(self, messages: list[dict]) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def n_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False).input_ids)

    def generate(self, prompts: list[str]) -> list[str]:
        torch = self.torch
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, generation_config=self.gen)
        return self.tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)

    def run(self, prompts: dict, out_csv: Path, batch_size: int, max_batch_tokens: int = 60_000, log_every: int = 10) -> pd.DataFrame:
        """prompts: {row_id: prompt text}. Writes row_id, output, prompt_tokens to out_csv after every batch (resumable).
        Prompts are sorted by length so a batch has little padding; batch size shrinks for very long prompts."""
        torch = self.torch
        out_csv = Path(out_csv)
        done = set(pd.read_csv(out_csv).row_id) if out_csv.exists() else set()
        todo = sorted([r for r in prompts if r not in done], key=lambda r: -len(prompts[r]))
        ntok = {r: self.n_tokens(prompts[r]) for r in todo}
        i, b = 0, 0
        t_start = time.perf_counter()
        while i < len(todo):
            bs = max(1, min(batch_size, max_batch_tokens // max(ntok[todo[i]], 1)))
            ids = todo[i:i + bs]
            t0 = time.perf_counter()
            outs = self.generate([prompts[r] for r in ids])
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            pd.DataFrame({"row_id": ids, "output": outs, "prompt_tokens": [ntok[r] for r in ids],
                          "batch_size": len(ids), "batch_seconds": dt}).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
            i += bs; b += 1
            if b % log_every == 0:
                el = time.perf_counter() - t_start
                print(f"  {i}/{len(todo)} clauses, {el:.0f} s elapsed", flush=True)
        return pd.read_csv(out_csv)

    def latency(self, prompts: list[str]) -> np.ndarray:
        """Seconds per clause at batch size 1 (one warm-up call first)."""
        torch = self.torch
        self.generate(prompts[:1])
        secs = []
        for p in prompts:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            self.generate([p]); torch.cuda.synchronize()
            secs.append(time.perf_counter() - t0)
        return np.array(secs)

    def label_scores(self, prompt: str, labels: list[str]) -> np.ndarray:
        """Constrained label scoring: log P(label tokens + end-of-turn token | prompt) for every label, i.e. the log
        probability that the model answers exactly this label and stops. The prompt is encoded once and its key/value
        cache is shared by the 27 candidate answers."""
        torch = self.torch
        tok, model = self.tok, self.model
        eos = tok.convert_tokens_to_ids("<|im_end|>")
        cands = [tok(l, add_special_tokens=False).input_ids + [eos] for l in labels]
        L = max(len(c) for c in cands)
        with torch.no_grad():
            p_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            out = model(input_ids=p_ids, use_cache=True)
            first_lp = torch.log_softmax(out.logits[0, -1].float(), -1)
            cache = out.past_key_values
            cache.batch_repeat_interleave(len(labels))
            ids = torch.full((len(labels), L), tok.pad_token_id, dtype=torch.long, device=self.device)
            for j, c in enumerate(cands):
                ids[j, :len(c)] = torch.tensor(c)
            # right padding after each candidate is harmless: causal attention means it never affects earlier positions
            att = torch.ones((len(labels), p_ids.shape[1] + L), dtype=torch.long, device=self.device)
            lp = torch.log_softmax(model(input_ids=ids, attention_mask=att, past_key_values=cache).logits.float(), -1)
        scores = []
        for j, c in enumerate(cands):
            s = first_lp[c[0]].item() + sum(lp[j, t - 1, c[t]].item() for t in range(1, len(c)))
            scores.append(s)
        return np.array(scores)

    def free(self):
        import gc
        del self.model
        gc.collect(); self.torch.cuda.empty_cache()
