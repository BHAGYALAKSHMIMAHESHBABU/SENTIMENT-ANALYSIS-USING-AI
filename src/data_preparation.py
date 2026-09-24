"""
Data Preparation Script - Phase 2
AI-Based Sentiment Analysis System

Loads raw CSVs, cleans duplicates, flags conflicting labels, handles cross-split
overlap, and saves derived working datasets to outputs/processed/.

IMPORTANT: Original files in data/ are NEVER modified.
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Ensure UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = BASE_DIR / "outputs" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_LABELS = {"negative", "neutral", "positive"}

# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD RAW DATA
# ──────────────────────────────────────────────────────────────────────────────

def load_raw():
    train = pd.read_csv(DATA_DIR / "train_all.csv")
    val   = pd.read_csv(DATA_DIR / "val_all.csv")
    test  = pd.read_csv(DATA_DIR / "test_all.csv")
    return train, val, test

# ──────────────────────────────────────────────────────────────────────────────
# 2. RECORD ORIGINAL SIZES
# ──────────────────────────────────────────────────────────────────────────────

def record_sizes(train, val, test, label="ORIGINAL"):
    print(f"\n[{label} SIZES]")
    print(f"  Train : {len(train):,} rows")
    print(f"  Val   : {len(val):,} rows")
    print(f"  Test  : {len(test):,} rows")
    return {"train": len(train), "val": len(val), "test": len(test)}

# ──────────────────────────────────────────────────────────────────────────────
# 3. DUPLICATE HANDLING
# ──────────────────────────────────────────────────────────────────────────────

def handle_duplicates(df, split_name, report_lines):
    """
    Step A: Remove exact duplicate rows (all columns identical).
    Step B: For sentences with the SAME label -> keep one copy.
    Step C: For sentences with DIFFERENT labels -> flag, do NOT auto-remove.
    """
    # Step A: exact duplicate rows
    exact_dup_mask = df.duplicated()
    n_exact = int(exact_dup_mask.sum())
    df = df[~exact_dup_mask].copy()
    report_lines.append(f"  [{split_name}] Exact duplicate rows removed: {n_exact}")

    # Step B & C: sentence-level duplicates
    sentence_label_counts = df.groupby("sentence")["label"].nunique()
    conflict_sentences = sentence_label_counts[sentence_label_counts > 1].index.tolist()

    # Report conflicts - do not auto-resolve
    report_lines.append(f"  [{split_name}] Intra-split conflicting label sentences: {len(conflict_sentences)}")
    for s in conflict_sentences:
        labels = df[df["sentence"] == s]["label"].tolist()
        report_lines.append(f"    ! CONFLICT: \"{s}\" -> labels: {labels}")
        report_lines.append(f"    ! ACTION: Retained as-is. NOT auto-resolved.")

    # Remove same-label sentence duplicates (keep first occurrence only)
    before_dedup = len(df)
    df_no_conflict = df[~df["sentence"].isin(conflict_sentences)]
    df_conflict_kept = df[df["sentence"].isin(conflict_sentences)]
    df_no_conflict = df_no_conflict.drop_duplicates(subset=["sentence"], keep="first")
    df = pd.concat([df_no_conflict, df_conflict_kept], ignore_index=True)
    n_same_removed = before_dedup - len(df)
    report_lines.append(f"  [{split_name}] Same-label duplicate sentences removed: {n_same_removed}")
    report_lines.append(f"  [{split_name}] Rows after deduplication: {len(df):,}")

    return df, conflict_sentences

# ──────────────────────────────────────────────────────────────────────────────
# 4. CROSS-SPLIT OVERLAP HANDLING
# ──────────────────────────────────────────────────────────────────────────────

def handle_cross_split_overlap(train, val, test, report_lines):
    """Remove train sentences that appear in test to prevent data leakage."""
    test_sentences = set(test["sentence"].astype(str))
    val_sentences  = set(val["sentence"].astype(str))

    # Train vs Test
    train_test_overlap = train[train["sentence"].astype(str).isin(test_sentences)]
    report_lines.append(f"\n[CROSS-SPLIT OVERLAP]")
    report_lines.append(f"  Train vs Test overlapping sentences: {len(train_test_overlap)}")
    for _, row in train_test_overlap.iterrows():
        test_row = test[test["sentence"] == row["sentence"]]
        test_label = test_row["label"].values[0] if len(test_row) > 0 else "N/A"
        match = "(SAME)" if row["label"] == test_label else "(DIFFERENT!)"
        report_lines.append(f"    Sentence: \"{row['sentence']}\"")
        report_lines.append(f"    Train label: {row['label']}  |  Test label: {test_label}  {match}")
        if row["sentence"] == "I could not finish it.":
            report_lines.append(
                "    *** NOTE: 'I could not finish it.' is labeled 'neutral' in train "
                "but 'negative' in test. Label correctness NOT auto-decided. Flagged for human review. ***"
            )
        report_lines.append(f"    ACTION: Removed from training set.")

    train_clean = train[~train["sentence"].astype(str).isin(test_sentences)].copy()
    n_removed = len(train) - len(train_clean)
    report_lines.append(f"  Train rows removed due to test overlap: {n_removed}")

    # Train vs Val (informational)
    train_val_overlap = train_clean[train_clean["sentence"].astype(str).isin(val_sentences)]
    report_lines.append(f"  Train vs Val overlapping sentences (informational): {len(train_val_overlap)}")
    report_lines.append(f"  NOTE: Train/Val overlap kept in train (val is not held-out test set).")

    # Val vs Test
    val_test_overlap = len(set(val["sentence"].astype(str)).intersection(test_sentences))
    report_lines.append(f"  Val vs Test overlapping sentences: {val_test_overlap}")

    return train_clean

# ──────────────────────────────────────────────────────────────────────────────
# 5. LABEL VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

def validate_labels(df, split_name, report_lines):
    unknown = df[~df["label"].isin(EXPECTED_LABELS)]
    if len(unknown) > 0:
        report_lines.append(f"  [{split_name}] Unknown labels: {unknown['label'].unique().tolist()}")
    else:
        report_lines.append(f"  [{split_name}] All labels valid: {sorted(df['label'].unique())}")

# ──────────────────────────────────────────────────────────────────────────────
# 6. SAVE CLEANED DATASETS
# ──────────────────────────────────────────────────────────────────────────────

def save_clean(df, name):
    out_path = PROCESSED_DIR / name
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Saved: {out_path}  ({len(df):,} rows)")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run_data_preparation():
    report_lines = ["=" * 70, "PHASE 2 - DATA PREPARATION REPORT", "=" * 70]

    print("[INFO] Loading raw datasets ...")
    train_raw, val_raw, test_raw = load_raw()
    orig_sizes = record_sizes(train_raw, val_raw, test_raw, "ORIGINAL")

    report_lines.append("\n[ORIGINAL DATASET SIZES]")
    report_lines.append(f"  Train : {orig_sizes['train']:,}")
    report_lines.append(f"  Val   : {orig_sizes['val']:,}")
    report_lines.append(f"  Test  : {orig_sizes['test']:,}")

    print("\n[INFO] Handling duplicates in train ...")
    report_lines.append("\n[DUPLICATE HANDLING]")
    train_clean, train_conflicts = handle_duplicates(train_raw.copy(), "TRAIN", report_lines)

    print("[INFO] Handling duplicates in val ...")
    val_clean, val_conflicts = handle_duplicates(val_raw.copy(), "VAL", report_lines)

    test_dup = int(test_raw.duplicated().sum())
    report_lines.append(f"  [TEST] Exact duplicate rows (informational): {test_dup} - test left untouched.")

    print("[INFO] Handling cross-split overlap (train vs test) ...")
    train_clean = handle_cross_split_overlap(train_clean, val_clean, test_raw.copy(), report_lines)

    print("\n[INFO] Validating labels ...")
    report_lines.append("\n[LABEL VALIDATION]")
    validate_labels(train_clean, "TRAIN", report_lines)
    validate_labels(val_clean,   "VAL",   report_lines)
    validate_labels(test_raw,    "TEST",  report_lines)

    clean_sizes = {"train": len(train_clean), "val": len(val_clean), "test": len(test_raw)}
    report_lines.append("\n[CLEANED DATASET SIZES]")
    for split, n in clean_sizes.items():
        orig = orig_sizes[split]
        removed = orig - n
        report_lines.append(f"  {split.upper():<5}: {n:,}  (original: {orig:,}, removed: {removed:,})")

    print("\n[INFO] Saving cleaned datasets ...")
    report_lines.append("\n[SAVED FILES]")
    save_clean(train_clean, "train_clean.csv")
    save_clean(val_clean,   "val_clean.csv")
    save_clean(test_raw,    "test_clean.csv")

    record_sizes(train_clean, val_clean, test_raw, "CLEANED")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    return train_clean, val_clean, test_raw, report_lines, clean_sizes, orig_sizes

if __name__ == "__main__":
    run_data_preparation()
