"""
Baseline Model Script - Phase 2
AI-Based Sentiment Analysis System

Pipeline:
  1. Load cleaned datasets from outputs/processed/
  2. Apply text preprocessing (lowercase + whitespace normalise)
  3. Fit TF-IDF vectorizer on TRAIN only, transform val + test
  4. Train LogisticRegression (class_weight='balanced') baseline
  5. Evaluate on validation and test sets
  6. Save model, vectorizer, predictions, metrics, confusion matrix chart
  7. Write phase2_report.txt

IMPORTANT: No deep learning, no BERT, no LSTM/GRU/CNN.
"""

import sys
import os
import pickle
import warnings
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Ensure UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

BASE_DIR   = Path(__file__).resolve().parent.parent
PROC_DIR   = BASE_DIR / "outputs" / "processed"
MODELS_DIR = BASE_DIR / "models"
OUT_DIR    = BASE_DIR / "outputs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
LABEL_ORDER  = ["negative", "neutral", "positive"]

# ──────────────────────────────────────────────────────────────────────────────
# PREPROCESSING (inline, consistent with text_preprocessing.py)
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ──────────────────────────────────────────────────────────────────────────────
# LOAD CLEANED DATA
# ──────────────────────────────────────────────────────────────────────────────

def load_cleaned():
    train = pd.read_csv(PROC_DIR / "train_clean.csv")
    val   = pd.read_csv(PROC_DIR / "val_clean.csv")
    test  = pd.read_csv(PROC_DIR / "test_clean.csv")
    return train, val, test

# ──────────────────────────────────────────────────────────────────────────────
# EVALUATE
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred, y_proba, split_name, report_lines):
    acc       = accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average="macro",    labels=LABEL_ORDER, zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=LABEL_ORDER, zero_division=0)
    macro_p   = precision_score(y_true, y_pred, average="macro",    labels=LABEL_ORDER, zero_division=0)
    macro_r   = recall_score(y_true, y_pred, average="macro",       labels=LABEL_ORDER, zero_division=0)
    cr        = classification_report(y_true, y_pred, labels=LABEL_ORDER, digits=4, zero_division=0)
    cm        = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)

    report_lines.append(f"\n[{split_name} EVALUATION]")
    report_lines.append(f"  Accuracy         : {acc:.4f}  ({acc*100:.2f}%)")
    report_lines.append(f"  Macro F1         : {macro_f1:.4f}")
    report_lines.append(f"  Weighted F1      : {weighted_f1:.4f}")
    report_lines.append(f"  Macro Precision  : {macro_p:.4f}")
    report_lines.append(f"  Macro Recall     : {macro_r:.4f}")
    report_lines.append(f"\n  Per-class Report:\n{cr}")
    report_lines.append(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    report_lines.append(f"  Labels order: {LABEL_ORDER}")
    for i, row in enumerate(cm):
        report_lines.append(f"    {LABEL_ORDER[i]:<10}: {row.tolist()}")

    print(f"\n  [{split_name}] Accuracy={acc:.4f}  Macro-F1={macro_f1:.4f}  Weighted-F1={weighted_f1:.4f}")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "classification_report": cr,
        "confusion_matrix": cm,
    }

# ──────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX CHART
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(val_cm, test_cm, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmaps_data = [
        (val_cm,  "Validation Set Confusion Matrix", axes[0]),
        (test_cm, "Test Set Confusion Matrix",       axes[1]),
    ]
    for cm, title, ax in cmaps_data:
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax, shrink=0.85)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ticks = range(len(LABEL_ORDER))
        ax.set_xticks(list(ticks))
        ax.set_yticks(list(ticks))
        ax.set_xticklabels([l.capitalize() for l in LABEL_ORDER], fontsize=10, rotation=15)
        ax.set_yticklabels([l.capitalize() for l in LABEL_ORDER], fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("Actual Label", fontsize=10)
        thresh = cm.max() / 2.0
        for i in range(len(LABEL_ORDER)):
            for j in range(len(LABEL_ORDER)):
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=12, fontweight="bold", color=color)

    plt.suptitle("TF-IDF + Logistic Regression Baseline — Confusion Matrices",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix chart saved: {out_path}")

# ──────────────────────────────────────────────────────────────────────────────
# SAVE PREDICTIONS
# ──────────────────────────────────────────────────────────────────────────────

def save_predictions(df, y_pred, y_proba, split_name, classifier, out_path):
    pred_df = df[["sentence", "label"]].copy().rename(columns={"label": "actual_label"})
    pred_df["predicted_label"] = y_pred
    pred_df["correct"] = pred_df["actual_label"] == pred_df["predicted_label"]
    # Probabilities for each class
    classes = list(classifier.classes_)
    for label in LABEL_ORDER:
        if label in classes:
            idx = classes.index(label)
            pred_df[f"{label}_probability"] = y_proba[:, idx]
        else:
            pred_df[f"{label}_probability"] = 0.0
    pred_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Predictions saved: {out_path}  ({len(pred_df):,} rows)")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run_baseline():
    report_lines = [
        "=" * 70,
        "PHASE 2 - BASELINE MODEL REPORT",
        "AI-Based Sentiment Analysis System",
        "=" * 70,
    ]

    # ── Environment info ──────────────────────────────────────────────────────
    import sklearn, scipy, numpy as np
    report_lines.append(f"\n[ENVIRONMENT]")
    report_lines.append(f"  Python      : {sys.version.split()[0]}")
    report_lines.append(f"  scikit-learn: {sklearn.__version__}")
    report_lines.append(f"  scipy       : {scipy.__version__}")
    report_lines.append(f"  numpy       : {np.__version__}")
    report_lines.append(f"  pandas      : {pd.__version__}")

    # ── Load cleaned data ─────────────────────────────────────────────────────
    print("[INFO] Loading cleaned datasets ...")
    train, val, test = load_cleaned()
    report_lines.append(f"\n[CLEANED DATASET SIZES USED FOR MODELING]")
    report_lines.append(f"  Train : {len(train):,}")
    report_lines.append(f"  Val   : {len(val):,}")
    report_lines.append(f"  Test  : {len(test):,}")
    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    print("[INFO] Applying text preprocessing ...")
    train["processed_text"] = train["sentence"].apply(preprocess)
    val["processed_text"]   = val["sentence"].apply(preprocess)
    test["processed_text"]  = test["sentence"].apply(preprocess)

    X_train = train["processed_text"].values
    y_train = train["label"].values
    X_val   = val["processed_text"].values
    y_val   = val["label"].values
    X_test  = test["processed_text"].values
    y_test  = test["label"].values

    # ── TF-IDF ────────────────────────────────────────────────────────────────
    tfidf_config = {
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "min_df": 2,
        "max_df": 0.95,
        "strip_accents": "unicode",
        "analyzer": "word",
    }
    print(f"\n[INFO] Fitting TF-IDF vectorizer on training data only ...")
    vectorizer = TfidfVectorizer(**tfidf_config)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf   = vectorizer.transform(X_val)
    X_test_tfidf  = vectorizer.transform(X_test)

    n_features = X_train_tfidf.shape[1]
    report_lines.append(f"\n[TF-IDF CONFIGURATION]")
    for k, v in tfidf_config.items():
        report_lines.append(f"  {k:<20}: {v}")
    report_lines.append(f"  {'Vocabulary size':<20}: {n_features:,} features")
    report_lines.append(f"  Fitted on train only. Val + Test are transformed only.")
    print(f"  TF-IDF vocabulary size: {n_features:,} features")

    # Save vectorizer
    vec_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"  Vectorizer saved: {vec_path}")
    report_lines.append(f"  Saved: models/tfidf_vectorizer.pkl")

    # ── Logistic Regression ───────────────────────────────────────────────────
    lr_config = {
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
        "solver": "lbfgs",
        "multi_class": "multinomial",
        "C": 1.0,
    }
    print(f"\n[INFO] Training LogisticRegression (class_weight='balanced') ...")
    clf = LogisticRegression(**lr_config)
    clf.fit(X_train_tfidf, y_train)

    report_lines.append(f"\n[LOGISTIC REGRESSION CONFIGURATION]")
    for k, v in lr_config.items():
        report_lines.append(f"  {k:<20}: {v}")
    report_lines.append(f"  Classes             : {list(clf.classes_)}")
    report_lines.append(f"  random_state        : {RANDOM_STATE}")

    # Save model
    model_path = MODELS_DIR / "logistic_regression_baseline.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"  Model saved: {model_path}")
    report_lines.append(f"  Saved: models/logistic_regression_baseline.pkl")

    # ── Evaluate on Validation ────────────────────────────────────────────────
    print("\n[INFO] Evaluating on validation set ...")
    y_val_pred  = clf.predict(X_val_tfidf)
    y_val_proba = clf.predict_proba(X_val_tfidf)
    val_metrics = evaluate(y_val, y_val_pred, y_val_proba, "VALIDATION", report_lines)

    # ── Evaluate on Test ──────────────────────────────────────────────────────
    print("\n[INFO] Evaluating on test set ...")
    y_test_pred  = clf.predict(X_test_tfidf)
    y_test_proba = clf.predict_proba(X_test_tfidf)
    test_metrics = evaluate(y_test, y_test_pred, y_test_proba, "TEST", report_lines)

    # ── Confusion Matrix Chart ────────────────────────────────────────────────
    cm_path = OUT_DIR / "baseline_confusion_matrix.png"
    plot_confusion_matrices(val_metrics["confusion_matrix"],
                            test_metrics["confusion_matrix"],
                            cm_path)
    report_lines.append(f"\n[CONFUSION MATRIX]")
    report_lines.append(f"  Saved: outputs/baseline_confusion_matrix.png")

    # ── Save Predictions ──────────────────────────────────────────────────────
    val_pred_path  = OUT_DIR / "baseline_predictions_val.csv"
    test_pred_path = OUT_DIR / "baseline_predictions.csv"
    save_predictions(val,  y_val_pred,  y_val_proba,  "val",  clf, val_pred_path)
    save_predictions(test, y_test_pred, y_test_proba, "test", clf, test_pred_path)
    report_lines.append(f"\n[PREDICTIONS SAVED]")
    report_lines.append(f"  Val  predictions: outputs/baseline_predictions_val.csv")
    report_lines.append(f"  Test predictions: outputs/baseline_predictions.csv")
    report_lines.append(f"  Columns: sentence, actual_label, predicted_label, correct, "
                        f"negative_probability, neutral_probability, positive_probability")

    # ── Warnings / Limitations ────────────────────────────────────────────────
    report_lines.append(f"\n[WARNINGS AND LIMITATIONS]")
    report_lines.append(f"  1. Train set has 1 conflicting-label sentence ('The service was okay.') "
                        f"retained without resolution.")
    report_lines.append(f"  2. 3 train/test overlapping sentences removed from train "
                        f"(including 1 with a label conflict: 'I could not finish it.').")
    report_lines.append(f"  3. TF-IDF discards word order and emoji features; "
                        f"linguistic features will be used in later phases.")
    report_lines.append(f"  4. Logistic Regression with class_weight='balanced' compensates "
                        f"for the moderate Neutral-skew in the training set.")
    report_lines.append(f"  5. This is a BASELINE model only. Do not treat as final.")
    report_lines.append(f"\n[SAVED MODEL PATHS]")
    report_lines.append(f"  models/tfidf_vectorizer.pkl")
    report_lines.append(f"  models/logistic_regression_baseline.pkl")
    report_lines.append("=" * 70)

    # ── Write phase2_report.txt ───────────────────────────────────────────────
    report_text = "\n".join(report_lines)
    report_path = OUT_DIR / "phase2_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[INFO] Phase 2 report written: {report_path}")
    print(report_text)

    return val_metrics, test_metrics

if __name__ == "__main__":
    run_baseline()
