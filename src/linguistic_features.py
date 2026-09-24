"""
Linguistic Features Extraction Module - Phase 4
AI-Based Sentiment Analysis System

Extracts lightweight, interpretable linguistic features from original sentence text.
Preserves original sentences and does not modify cleaned datasets.

Features extracted:
  Group A - Basic text statistics:
    1. word_count: Number of whitespace-separated words in sentence.
    2. character_count: Total number of characters including whitespace.
    3. sentence_length: Number of characters excluding leading/trailing whitespace.
  Group B - Punctuation and capitalization:
    4. exclamation_count: Total occurrences of '!' character.
    5. question_count: Total occurrences of '?' character.
    6. uppercase_ratio: Uppercase letters / total alphabetic letters (0.0 if none).
    7. uppercase_word_count: Words of length >= 2 consisting entirely of uppercase letters.
  Group C - Sentiment linguistic cues:
    8. emoji_count: Count of emoji characters (via lightweight `emoji` library).
    9. negation_count: Occurrences of designated negation words / contractions.
    10. intensifier_count: Occurrences of designated degree adverbs / intensifiers.
    11. contrast_word_count: Occurrences of designated contrast / discourse markers.
  Group D - Visual and repetition cues:
    12. repeated_character_count: Sequences of 3+ identical consecutive characters (e.g. soooo, !!!).
"""

import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import emoji

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "outputs" / "processed"
PHASE4_DIR = BASE_DIR / "outputs" / "phase4"
PHASE4_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature Names & Groups ────────────────────────────────────────────────────
FEATURE_NAMES = [
    "word_count",
    "character_count",
    "sentence_length",
    "exclamation_count",
    "question_count",
    "uppercase_ratio",
    "uppercase_word_count",
    "emoji_count",
    "negation_count",
    "intensifier_count",
    "contrast_word_count",
    "repeated_character_count",
]

FEATURE_GROUPS = {
    "Group A": ["word_count", "character_count", "sentence_length"],
    "Group B": ["exclamation_count", "question_count", "uppercase_ratio", "uppercase_word_count"],
    "Group C": ["negation_count", "intensifier_count", "contrast_word_count"],
    "Group D": ["emoji_count", "repeated_character_count"],
}

# ── Word Lists ────────────────────────────────────────────────────────────────
NEGATIONS = [
    "not", "no", "never", "neither", "nor", "don't", "doesn't", "didn't",
    "isn't", "aren't", "wasn't", "weren't", "can't", "couldn't", "won't",
    "wouldn't", "shouldn't", "haven't", "hasn't", "hadn't"
]

INTENSIFIERS = [
    "very", "really", "extremely", "highly", "so", "too", "incredibly",
    "absolutely", "completely", "totally", "quite"
]

CONTRAST_WORDS = [
    "but", "however", "although", "though", "yet", "nevertheless",
    "nonetheless", "whereas"
]

# ── Compiled Patterns ─────────────────────────────────────────────────────────
def _build_word_regex(words: List[str]) -> re.Pattern:
    """Build word-boundary regex supporting standard & curly apostrophes."""
    escaped = []
    # Sort longest first to avoid prefix shadowing
    for w in sorted(words, key=len, reverse=True):
        esc = re.escape(w).replace(r"\'", r"['’\u2019]")
        escaped.append(esc)
    pattern = r"\b(?:" + "|".join(escaped) + r")\b"
    return re.compile(pattern, re.IGNORECASE)

NEGATION_REGEX = _build_word_regex(NEGATIONS)
INTENSIFIER_REGEX = _build_word_regex(INTENSIFIERS)
CONTRAST_REGEX = _build_word_regex(CONTRAST_WORDS)
REPEATED_CHAR_REGEX = re.compile(r"(.)\1{2,}")


def extract_features_from_text(text: str) -> Dict[str, float]:
    """
    Extract all 12 linguistic features from a single raw sentence text.
    Preserves all punctuation, casing, emojis, and repetition.
    """
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    # Group A: Basic text statistics
    char_count = len(text)
    words = text.split()
    word_cnt = len(words)
    sent_len = len(text.strip())

    # Group B: Punctuation & capitalization
    excl_cnt = text.count("!")
    quest_cnt = text.count("?")

    alpha_chars = [c for c in text if c.isalpha()]
    total_alpha = len(alpha_chars)
    if total_alpha > 0:
        upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / total_alpha
    else:
        upper_ratio = 0.0

    upper_word_cnt = sum(1 for w in words if w.isupper() and any(c.isalpha() for c in w) and len(w) >= 2)

    # Group C: Sentiment cues
    neg_cnt = len(NEGATION_REGEX.findall(text))
    intens_cnt = len(INTENSIFIER_REGEX.findall(text))
    contrast_cnt = len(CONTRAST_REGEX.findall(text))

    # Group D: Visual & repetition cues
    emoji_cnt = emoji.emoji_count(text)
    rep_char_cnt = len(list(REPEATED_CHAR_REGEX.finditer(text)))

    return {
        "word_count": float(word_cnt),
        "character_count": float(char_count),
        "sentence_length": float(sent_len),
        "exclamation_count": float(excl_cnt),
        "question_count": float(quest_cnt),
        "uppercase_ratio": float(round(upper_ratio, 6)),
        "uppercase_word_count": float(upper_word_cnt),
        "emoji_count": float(emoji_cnt),
        "negation_count": float(neg_cnt),
        "intensifier_count": float(intens_cnt),
        "contrast_word_count": float(contrast_cnt),
        "repeated_character_count": float(rep_char_cnt),
    }


def extract_features_df(df: pd.DataFrame, text_col: str = "sentence") -> pd.DataFrame:
    """
    Extract linguistic features for every row in a DataFrame.
    Returns a new DataFrame containing metadata ('sentence', 'label' if present)
    plus all 12 feature columns.
    """
    records = []
    for text in df[text_col]:
        records.append(extract_features_from_text(text))

    features_df = pd.DataFrame(records, index=df.index)

    # Build output dataframe retaining identification columns
    cols_to_keep = [c for c in ["sentence", "label"] if c in df.columns]
    result_df = pd.concat([df[cols_to_keep], features_df], axis=1)
    return result_df


def generate_and_save_feature_matrices() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract linguistic feature matrices for train, validation, and test datasets.
    Saves outputs to outputs/phase4/linguistic_features_{train,val,test}.csv.
    """
    print("[INFO] Loading cleaned datasets from outputs/processed/ ...")
    train = pd.read_csv(PROC_DIR / "train_clean.csv")
    val = pd.read_csv(PROC_DIR / "val_clean.csv")
    test = pd.read_csv(PROC_DIR / "test_clean.csv")

    print(f"  Train : {len(train):,} rows")
    print(f"  Val   : {len(val):,} rows")
    print(f"  Test  : {len(test):,} rows")

    print("\n[INFO] Extracting linguistic features for Train ...")
    train_feat = extract_features_df(train)
    train_path = PHASE4_DIR / "linguistic_features_train.csv"
    train_feat.to_csv(train_path, index=False, encoding="utf-8")
    print(f"  Saved: {train_path} ({len(train_feat):,} rows, {train_feat.shape[1]} cols)")

    print("\n[INFO] Extracting linguistic features for Validation ...")
    val_feat = extract_features_df(val)
    val_path = PHASE4_DIR / "linguistic_features_val.csv"
    val_feat.to_csv(val_path, index=False, encoding="utf-8")
    print(f"  Saved: {val_path} ({len(val_feat):,} rows, {val_feat.shape[1]} cols)")

    print("\n[INFO] Extracting linguistic features for Test ...")
    test_feat = extract_features_df(test)
    test_path = PHASE4_DIR / "linguistic_features_test.csv"
    test_feat.to_csv(test_path, index=False, encoding="utf-8")
    print(f"  Saved: {test_path} ({len(test_feat):,} rows, {test_feat.shape[1]} cols)")

    return train_feat, val_feat, test_feat


def analyze_linguistic_features(train_feat: pd.DataFrame, out_file: Path = None) -> str:
    """
    Calculate statistical distributions (mean, std, min, max) and class-wise
    averages across sentiment labels. Writes report to outputs/phase4/linguistic_feature_analysis.txt.
    """
    if out_file is None:
        out_file = PHASE4_DIR / "linguistic_feature_analysis.txt"

    lines = [
        "=" * 78,
        "PHASE 4 - LINGUISTIC FEATURE STATISTICAL ANALYSIS",
        "AI-Based Sentiment Analysis System",
        "=" * 78,
        "",
        "Dataset Source: outputs/phase4/linguistic_features_train.csv (Training split only)",
        f"Total Samples : {len(train_feat):,}",
        "",
        "Feature Definitions & Grouping:",
        "  Group A (Basic text statistics):",
        "    - word_count              : Whitespace-split token count",
        "    - character_count         : Total length including whitespace",
        "    - sentence_length         : Stripped string character length",
        "  Group B (Punctuation & capitalization):",
        "    - exclamation_count       : Count of '!' characters",
        "    - question_count          : Count of '?' characters",
        "    - uppercase_ratio         : Uppercase alphabetic chars / Total alphabetic chars",
        "    - uppercase_word_count    : Fully uppercase words with length >= 2",
        "  Group C (Sentiment linguistic cues):",
        "    - negation_count          : Occurrences of designated negation words",
        "    - intensifier_count       : Occurrences of designated intensifier adverbs",
        "    - contrast_word_count     : Occurrences of designated contrast discourse markers",
        "  Group D (Visual & repetition cues):",
        "    - emoji_count             : Unicode emoji symbols count",
        "    - repeated_character_count: Occurrences of 3+ consecutive identical characters",
        "",
        "-" * 78,
        f"{'OVERALL TRAINING SET FEATURE SUMMARY':^78}",
        "-" * 78,
        f"{'Feature':<26} {'Mean':>10} {'Std':>10} {'Min':>8} {'Max':>10} {'Non-Zero %':>12}",
        "-" * 78,
    ]

    for feat in FEATURE_NAMES:
        col = train_feat[feat]
        mean_v = col.mean()
        std_v = col.std()
        min_v = col.min()
        max_v = col.max()
        nz_pct = (col > 0).mean() * 100
        lines.append(f"{feat:<26} {mean_v:>10.4f} {std_v:>10.4f} {min_v:>8.1f} {max_v:>10.1f} {nz_pct:>11.2f}%")

    lines.append("-" * 78)
    lines.append("")

    # Class-wise averages
    if "label" in train_feat.columns:
        label_order = ["negative", "neutral", "positive"]
        lines.extend([
            "-" * 78,
            f"{'CLASS-WISE FEATURE MEANS (TRAINING SET)':^78}",
            "-" * 78,
            f"{'Feature':<26} {'Negative':>14} {'Neutral':>14} {'Positive':>14}",
            "-" * 78,
        ])
        grouped = train_feat.groupby("label")[FEATURE_NAMES].mean()
        for feat in FEATURE_NAMES:
            neg_m = grouped.loc["negative", feat] if "negative" in grouped.index else 0.0
            neu_m = grouped.loc["neutral", feat] if "neutral" in grouped.index else 0.0
            pos_m = grouped.loc["positive", feat] if "positive" in grouped.index else 0.0
            lines.append(f"{feat:<26} {neg_m:>14.4f} {neu_m:>14.4f} {pos_m:>14.4f}")
        lines.append("-" * 78)
        lines.append("")

    # Observational summary & cautions
    lines.extend([
        "[CRITICAL EXPERIMENTAL OBSERVATION]",
        "  1. The differences in means across sentiment classes are empirical observations,",
        "     NOT proof that these features will improve downstream classification performance.",
        "  2. TF-IDF features already capture vocabulary and n-grams corresponding to words",
        "     like 'not', 'very', 'never', etc. Whether explicit feature counts provide supplementary",
        "     orthogonal signal or redundant noise must be determined by controlled experiment.",
        "  3. Features such as uppercase_ratio, exclamation_count, and emoji_count carry surface",
        "     expressive information not directly encoded in standard lowercased TF-IDF.",
        "  4. Feature scaling must be strictly fitted on training data only to avoid leakage.",
        "=" * 78,
    ])

    report_text = "\n".join(lines)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[INFO] Feature analysis saved: {out_file}")
    return report_text


if __name__ == "__main__":
    train_f, val_f, test_f = generate_and_save_feature_matrices()
    analysis = analyze_linguistic_features(train_f)
    print("\nFeature Analysis Summary Preview:")
    for l in analysis.splitlines()[:35]:
        print(l)
