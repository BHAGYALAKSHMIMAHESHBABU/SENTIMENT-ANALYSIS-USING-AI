"""
Text Preprocessing Module - Phase 2
AI-Based Sentiment Analysis System

Provides a reusable preprocessing function for the baseline TF-IDF model.
Does NOT modify the original 'sentence' column - operates on copies.

Preservation rules (DO NOT remove):
  - negation words (not, no, never, cannot)
  - emojis
  - punctuation (TF-IDF handles tokenization)
  - exclamation/question marks
  - capitalization patterns (converted to lowercase for baseline only)
  - repeated characters
  - intensifiers
"""

import re
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def preprocess_for_tfidf(text: str) -> str:
    """
    Lightweight preprocessing for the TF-IDF baseline model.

    What this does:
      1. Converts to lowercase (needed for vocabulary consistency in TF-IDF).
      2. Normalises runs of whitespace to a single space.
      3. Strips leading/trailing whitespace.

    What this deliberately does NOT do:
      - Remove punctuation (TF-IDF's token_pattern handles this)
      - Remove negation words (not, no, never, cannot)
      - Remove emojis (they carry sentiment signal)
      - Remove repeated characters (e.g. 'soooo')
      - Remove exclamation/question marks
      - Remove stop words aggressively
      - Perform stemming or lemmatization

    The original 'sentence' column is NEVER modified.
    Use a separate column (e.g. 'processed_text') to store the result.
    """
    if not isinstance(text, str):
        text = str(text)

    # Step 1: lowercase
    text = text.lower()

    # Step 2: normalise whitespace only
    text = re.sub(r"\s+", " ", text).strip()

    return text


def apply_preprocessing(df, text_col="sentence", new_col="processed_text"):
    """
    Apply preprocess_for_tfidf to a DataFrame column.
    Returns a NEW DataFrame with the additional column.
    Original DataFrame is unchanged.
    """
    import pandas as pd
    result = df.copy()
    result[new_col] = result[text_col].apply(preprocess_for_tfidf)
    return result


if __name__ == "__main__":
    # Quick smoke test
    samples = [
        "GOOD food!!! soooo delicious",
        "not good at all",
        "cannot believe this happened",
        "😍 amazing place",
        "The service was okay.",
        "EVERY SINGLE ITEM WAS INEDIBLE.",
    ]
    print("Preprocessing smoke test:")
    for s in samples:
        print(f"  Original : {s!r}")
        print(f"  Processed: {preprocess_for_tfidf(s)!r}")
        print()
