"""
Phase 3 - Step 1: Fix Train/Val Overlap
Remove "Hey Arnold !" from train_clean.csv (same neutral label in both splits).
Overwrites train_clean.csv ONLY. Does NOT touch val_clean.csv, test_clean.csv, or raw data.
"""
import pandas as pd
from pathlib import Path

PROC = Path("outputs/processed")

train = pd.read_csv(PROC / "train_clean.csv")
val   = pd.read_csv(PROC / "val_clean.csv")
test  = pd.read_csv(PROC / "test_clean.csv")

before = len(train)
val_sentences = set(val["sentence"].astype(str))
train_fixed = train[~train["sentence"].astype(str).isin(val_sentences)].copy()
removed = before - len(train_fixed)

print(f"Train before : {before:,}")
print(f"Sentences removed (train/val overlap): {removed}")
print(f"Train after  : {len(train_fixed):,}")
print(f"Val unchanged: {len(val):,}")
print(f"Test unchanged: {len(test):,}")

# Verify
assert len(set(train_fixed["sentence"]).intersection(val_sentences)) == 0, "Overlap still present!"
assert len(train_fixed) == before - removed
print("\nVerification: Train/Val overlap is now 0.")

train_fixed.to_csv(PROC / "train_clean.csv", index=False, encoding="utf-8")
print(f"Saved: {PROC / 'train_clean.csv'}")
