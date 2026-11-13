"""Hand-made tricky cases for src/llm.parse_label (run: python -m pytest tests/test_llm_parser.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm import DESCRIPTIONS, INVALID, parse_label  # noqa: E402

LABELS = sorted(DESCRIPTIONS)

CASES = [
    # (model output, expected label, expected rule)
    ("Joint Ip Ownership", "Joint Ip Ownership", "exact"),
    ("Category: Ip Ownership Assignment", "Ip Ownership Assignment", "exact"),
    ('"Non-Compete".', "Non-Compete", "exact"),
    ("**License Grant**", "License Grant", "exact"),
    ("joint IP ownership.", "Joint Ip Ownership", "normalised"),
    ("Intellectual Property Ownership Assignment", "Ip Ownership Assignment", "normalised"),
    ("termination for convenience", "Termination For Convenience", "normalised"),
    ("Rofr Rofo Rofn", "Rofr/Rofo/Rofn", "normalised"),
    ("The clause is a Non-Transferable License clause.", "Non-Transferable License", "contained"),
    ("This is an anti-assignment clause", "Anti-Assignment", "contained"),
    ("Right of first refusal", "Rofr/Rofo/Rofn", "contained"),
    ("ROFR", "Rofr/Rofo/Rofn", "contained"),
    # first line wins before any searching of the explanation
    ("Exclusivity\nUnlike a Non-Compete, this clause ...", "Exclusivity", "exact"),
    # overlapping but different names -> two labels named -> INVALID
    ("Joint Ip Ownership Assignment", INVALID, "ambiguous"),
    ("Non-Transferable License Grant", INVALID, "ambiguous"),
    ("Competitive Restriction Exception to the Non-Compete clause", INVALID, "ambiguous"),
    ("License Grant or Non-Transferable License", INVALID, "ambiguous"),
    # a CUAD category that is NOT in our 27 classes, and empty / chatty outputs
    ("No-Solicit Of Customers", INVALID, "no label"),
    ("", INVALID, "no label"),
    ("I cannot determine the category.", INVALID, "no label"),
]


def test_cases():
    for out, label, rule in CASES:
        assert parse_label(out, LABELS) == (label, rule), (out, parse_label(out, LABELS))


def test_longest_name_wins():
    # synthetic label set where one name is contained in another: the occurrence inside the longer one is dropped
    labels = ["Ownership", "Joint Ip Ownership", "Exclusivity"]
    assert parse_label("It provides for joint IP ownership between the parties", labels) == ("Joint Ip Ownership", "contained")
    assert parse_label("ownership of the results", labels) == ("Ownership", "contained")
    assert parse_label("Joint IP ownership and exclusivity", labels) == (INVALID, "ambiguous")
