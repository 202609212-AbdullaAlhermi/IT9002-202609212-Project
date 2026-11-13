"""Text preprocessing for the three model families.

Every step is a small, separate function so the Task 1 notebook can switch each one on
or off and MEASURE its effect on validation macro-F1 (the preprocessing study, Part A).

    classical models (TF-IDF)  -> tokenize(text, cfg)           cfg = a dict of on/off switches
    transformer (Legal-BERT)   -> preprocess_for_transformer()  light touch: the subword tokenizer does the rest
    topic models (LDA / NMF)   -> preprocess_for_topics()       only for the optional Part F item F4 (not the main Task 2)

FINAL_CLASSICAL_CONFIG holds the steps that the preprocessing study kept. The notebook
re-derives that choice from results/preprocessing_study.csv and checks that it matches.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# ---------------------------------------------------------------------------
# Step 1 - Unicode normalisation
# The clauses were copied from SEC filings (PDF/HTML), so a few contain typographic
# characters: curly quotes, long dashes, bullets, zero-width spaces, soft hyphens.
# NFKC folds compatibility characters (e.g. ligatures) and the table maps the rest.
# ---------------------------------------------------------------------------
_CHAR_MAP = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',   # curly quotes
    "–": "-", "—": "-",                                   # en / em dash
    "…": "...",                                                # ellipsis
    "•": " ", "●": " ", "·": " ",                    # bullets
    "​": "", "­": "", "﻿": "",                       # zero-width / soft hyphen / BOM
    " ": " ",                                                  # non-breaking space
})


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text).translate(_CHAR_MAP)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Step 3 - cross-references such as "Section 12.3(b)", "Article VIII", "Exhibit A"
# The NUMBER is specific to one contract's layout and carries no meaning about the
# clause type, so the whole reference becomes one placeholder word.
# ---------------------------------------------------------------------------
SECTION_REF = re.compile(
    r"\b(?:sections?|articles?|paragraphs?|clauses?|exhibits?|schedules?|annex(?:es)?)\s+"
    r"(?:\d+(?:\.\d+)*(?:\s*\([a-z0-9]{1,4}\))*|[ivxlc]+\b|[a-z]\b)"
    r"|§+\s*\d+(?:\.\d+)*(?:\s*\([a-z0-9]{1,4}\))*",
    flags=re.IGNORECASE,
)


def replace_section_refs(text: str) -> str:
    return SECTION_REF.sub(" secref ", text)


# ---------------------------------------------------------------------------
# Step 4 - numbers. "thirty (30) days", "$5,000,000", "12.5%". The exact value rarely
# decides the clause TYPE, but the presence of a number often does (durations, caps).
# Every number becomes the placeholder word "num".
# ---------------------------------------------------------------------------
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def replace_numbers(text: str) -> str:
    return NUMBER.sub(" num ", text)


# ---------------------------------------------------------------------------
# Step 5 - symbols. The default TF-IDF tokenizer throws away all punctuation, including
# "$" and "%", which signal money and percentages (Cap On Liability, Revenue/Profit
# Sharing, Minimum Commitment). This step turns them into words so they survive.
# ---------------------------------------------------------------------------
def keep_symbols(text: str) -> str:
    return text.replace("$", " dollarsign ").replace("%", " percentsign ")


# Tokens = runs of 2+ letters/digits (same rule as scikit-learn's default TF-IDF tokenizer).
TOKEN = re.compile(r"(?u)\b\w\w+\b")

# ---------------------------------------------------------------------------
# Step 6 - stop words. Standard English lists delete words that change a clause's legal
# meaning: "not", "no", "nor" (negation), "until", "unless", "without" (conditions),
# "only", "any", "all", "each" (scope). The "legal" variant keeps those.
# ---------------------------------------------------------------------------
LEGAL_KEEP = {
    "not", "no", "nor", "unless", "until", "without", "only", "any", "all", "each", "both",
    "against", "before", "after", "under", "during", "above", "below", "off", "out", "further",
    "same", "other", "such", "own", "more", "most", "few", "shall", "may", "will", "must",
    "should", "can", "cannot", "won", "don", "doesn", "didn", "isn", "aren", "wasn", "weren",
    "hasn", "haven", "hadn", "shouldn", "wouldn", "couldn", "mustn", "needn", "mightn",
}


# ---------------------------------------------------------------------------
# NLTK data (stop-word list, WordNet). Downloaded once if missing. Newer NLTK refuses downloads
# through a web proxy (Google Colab uses one); the source is NLTK's own fixed data index, so we
# allow it, and fall back to fetching the official zip directly if NLTK's downloader still fails.
# ---------------------------------------------------------------------------
_NLTK_PACKAGES = {"stopwords": "corpora/stopwords", "wordnet": "corpora/wordnet", "omw-1.4": "corpora/omw-1.4"}
_NLTK_ZIP_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/{}.zip"


def ensure_nltk_data(packages=("stopwords", "wordnet", "omw-1.4")) -> None:
    import os
    import nltk

    for pkg in packages:
        path = _NLTK_PACKAGES[pkg]
        try:
            nltk.data.find(path)
            continue
        except LookupError:
            pass
        os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")
        try:
            nltk.download(pkg, quiet=True)
            nltk.data.find(path)
            continue
        except Exception:
            pass
        import io, urllib.request, zipfile  # direct fallback: the official NLTK data zip
        from pathlib import Path
        target = Path.home() / "nltk_data" / "corpora"
        target.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(_NLTK_ZIP_URL.format(pkg), timeout=60) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(target)
        if str(Path.home() / "nltk_data") not in nltk.data.path:
            nltk.data.path.append(str(Path.home() / "nltk_data"))
        nltk.data.find(path)


@lru_cache(maxsize=None)
def _stopword_set(kind: str) -> frozenset:
    ensure_nltk_data(("stopwords",))
    from nltk.corpus import stopwords

    base = set(stopwords.words("english"))
    if kind == "nltk":
        return frozenset(base)
    if kind == "nltk_legal":
        return frozenset(base - LEGAL_KEEP)
    raise ValueError(f"unknown stop-word list: {kind}")


# ---------------------------------------------------------------------------
# Step 7 - stemming (crude suffix chopping: "terminates" -> "termin") vs lemmatisation
# (dictionary form: "terminates" -> "terminate"). Both merge word forms into one feature.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _porter():
    from nltk.stem import PorterStemmer

    return PorterStemmer()


@lru_cache(maxsize=None)
def _lemmatizer():
    ensure_nltk_data(("wordnet", "omw-1.4"))
    from nltk.stem import WordNetLemmatizer

    return WordNetLemmatizer()


@lru_cache(maxsize=200_000)
def _stem(word: str) -> str:
    return _porter().stem(word, to_lowercase=False)


@lru_cache(maxsize=200_000)
def _lemma(word: str) -> str:
    # WordNet only knows lower-case words; keep the original casing otherwise.
    lem = _lemmatizer().lemmatize(word.lower())
    return lem if word.islower() else word


# ---------------------------------------------------------------------------
# The configurable pipeline used by the classical models
# ---------------------------------------------------------------------------
BASELINE_CONFIG = {
    "unicode": False,      # step 1
    "lowercase": False,    # step 2
    "section_refs": False, # step 3
    "numbers": False,      # step 4
    "symbols": False,      # step 5
    "stopwords": "none",   # step 6: "none" | "nltk" | "nltk_legal"
    "stem": "none",        # step 7: "none" | "porter" | "lemma"
    "ngram_max": 1,        # step 8: 1 = single words, 2 = words + word pairs, 3 = up to triples
}

# Selected by the preprocessing study on the design-v2 (27-class) data (Task 1 notebook,
# section 3.3; results/preprocessing_study.csv); the notebook asserts that this dict matches the study's choice, so it cannot silently go stale.
FINAL_CLASSICAL_CONFIG = {
    "unicode": False,      # all values chosen by the pre-declared rule on the 27-class validation set;
    "lowercase": False,    # no step improved macro-F1 by >= 0.003, so most were decided by the
    "section_refs": True,  # "smallest vocabulary among equivalent options" tie-break (see the notebook, 3.3)
    "numbers": False,
    "symbols": False,
    "stopwords": "nltk",   # tie-break choice at -0.003 (inside the noise band); section 4 re-tests "none" in the tuned grid
    "stem": "none",
    "ngram_max": 1,
}


def clean_text(text: str, cfg: dict) -> str:
    """Apply the string-level steps (1-5) that are switched on in `cfg`."""
    if cfg.get("unicode"):
        text = normalize_unicode(text)
    if cfg.get("lowercase"):
        text = text.lower()
    if cfg.get("section_refs"):
        text = replace_section_refs(text)
    if cfg.get("numbers"):
        text = replace_numbers(text)
    if cfg.get("symbols"):
        text = keep_symbols(text)
    return normalize_whitespace(text)


def tokenize(text: str, cfg: dict) -> list[str]:
    """Clean the text, split it into words, then apply the token-level steps (6-7)."""
    tokens = TOKEN.findall(clean_text(text, cfg))
    if cfg.get("stopwords", "none") != "none":
        stop = _stopword_set(cfg["stopwords"])
        tokens = [t for t in tokens if t.lower() not in stop]
    if cfg.get("stem") == "porter":
        tokens = [_stem(t) for t in tokens]
    elif cfg.get("stem") == "lemma":
        tokens = [_lemma(t) for t in tokens]
    return tokens


def make_vectorizer(cfg: dict, **tfidf_kwargs):
    """A TfidfVectorizer that uses our tokenizer, so the SAME preprocessing is applied to
    training, validation and test text (no chance of preprocessing them differently)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(
        tokenizer=lambda s: tokenize(s, cfg),
        lowercase=False,          # lower-casing is controlled by cfg, not by sklearn
        token_pattern=None,
        ngram_range=(1, cfg.get("ngram_max", 1)),
        **tfidf_kwargs,
    )


# ---------------------------------------------------------------------------
# Transformer: almost nothing. The checkpoint's own WordPiece tokenizer lower-cases
# (both Legal-BERT and bert-base are "uncased"), splits rare words into known pieces and
# needs stop words and word order to understand the sentence. We only repair characters.
# ---------------------------------------------------------------------------
def preprocess_for_transformer(text: str) -> str:
    return normalize_whitespace(normalize_unicode(text))


# ---------------------------------------------------------------------------
# Topic models: bag-of-words models with no notion of grammar, so function words and
# numbers would dominate every topic. Lower-case, drop numbers / references, remove stop
# words (here the FULL list - negation does not define a topic), lemmatise, keep words
# of 3+ letters. Used only if the optional Part F item F4 (topic modelling) is done.
# ---------------------------------------------------------------------------
def preprocess_for_topics(text: str, extra_stopwords: frozenset | set = frozenset()) -> list[str]:
    text = replace_section_refs(normalize_unicode(text).lower()).replace("secref", " ")
    stop = _stopword_set("nltk") | set(extra_stopwords)
    words = re.findall(r"[a-z]{3,}", text)
    return [w for w in (_lemma(w) for w in words if w not in stop) if w not in stop and len(w) > 2]
