"""
Phase 4 - Linguistic Feature Enhancement Modeling & Ablation Pipeline
AI-Based Sentiment Analysis System

Pipeline:
  1. Load cleaned datasets and extracted linguistic features.
  2. Apply Phase 2 identical text preprocessing (lowercase + whitespace normalisation).
  3. Fit TF-IDF on TRAIN only, transform val + test.
  4. Fit StandardScaler on TRAIN linguistic features only, transform val + test.
  5. Train Enhanced Logistic Regression (TF-IDF + 12 Linguistic Features).
  6. Evaluate Baseline vs Enhanced on validation and test sets.
  7. Run Controlled Feature Ablation Study across Groups A, B, C, D.
  8. Save models, scalers, comparison metrics CSV, ablation CSV, confusion matrix plot, and phase4_report.txt.

Guarantees:
  - Strict data leakage prevention (scaler and TF-IDF fitted on train only).
  - Raw data and Phase 2/3 outputs are NOT modified.
"""

import sys
import pickle
import re
import warnings
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "outputs" / "processed"
MODELS_DIR = BASE_DIR / "models"
PHASE4_DIR = BASE_DIR / "outputs" / "phase4"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PHASE4_DIR.mkdir(parents=True, exist_ok=True)

from src.linguistic_features import (
    FEATURE_NAMES,
    FEATURE_GROUPS,
    generate_and_save_feature_matrices,
)

LABEL_ORDER = ["negative", "neutral", "positive"]
RANDOM_STATE = 42

# ── Preprocessing (Exact match with Phase 2) ──────────────────────────────────
def preprocess(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_metrics(y_true, y_pred) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=LABEL_ORDER, zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=LABEL_ORDER, zero_division=0)
    macro_p = precision_score(y_true, y_pred, average="macro", labels=LABEL_ORDER, zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", labels=LABEL_ORDER, zero_division=0)
    clf_rep = classification_report(y_true, y_pred, labels=LABEL_ORDER, digits=4, zero_division=0, output_dict=True)
    clf_rep_txt = classification_report(y_true, y_pred, labels=LABEL_ORDER, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "clf_report_dict": clf_rep,
        "clf_report_txt": clf_rep_txt,
        "confusion_matrix": cm,
    }


def plot_confusion_matrices(val_cm, test_cm, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmaps_data = [
        (val_cm, "Validation Set Confusion Matrix", axes[0]),
        (test_cm, "Test Set Confusion Matrix", axes[1]),
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
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)

    plt.suptitle("Phase 4 Enhanced Model (TF-IDF + Linguistic Features) — Confusion Matrices",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix chart saved: {out_path}")


def run_phase4_pipeline():
    print("=" * 78)
    print("PHASE 4 — LINGUISTIC FEATURE ENHANCEMENT PIPELINE")
    print("=" * 78)

    # 1. Check / Load feature matrices
    train_feat_path = PHASE4_DIR / "linguistic_features_train.csv"
    val_feat_path = PHASE4_DIR / "linguistic_features_val.csv"
    test_feat_path = PHASE4_DIR / "linguistic_features_test.csv"

    if not (train_feat_path.exists() and val_feat_path.exists() and test_feat_path.exists()):
        print("[INFO] Linguistic feature files not found. Generating now ...")
        train_feat_df, val_feat_df, test_feat_df = generate_and_save_feature_matrices()
    else:
        print("[INFO] Loading pre-extracted linguistic features ...")
        train_feat_df = pd.read_csv(train_feat_path)
        val_feat_df = pd.read_csv(val_feat_path)
        test_feat_df = pd.read_csv(test_feat_path)

    print(f"  Train: {len(train_feat_df):,} | Val: {len(val_feat_df):,} | Test: {len(test_feat_df):,}")

    # 2. Prepare text and labels
    print("\n[INFO] Preprocessing text for TF-IDF ...")
    X_train_text = train_feat_df["sentence"].apply(preprocess).values
    y_train = train_feat_df["label"].values

    X_val_text = val_feat_df["sentence"].apply(preprocess).values
    y_val = val_feat_df["label"].values

    X_test_text = test_feat_df["sentence"].apply(preprocess).values
    y_test = test_feat_df["label"].values

    # 3. TF-IDF Representation (Exact Phase 2 Baseline Config)
    tfidf_config = {
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "min_df": 2,
        "max_df": 0.95,
        "strip_accents": "unicode",
        "analyzer": "word",
    }
    print("[INFO] Fitting TF-IDF Vectorizer on train set only ...")
    vectorizer = TfidfVectorizer(**tfidf_config)
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_val_tfidf = vectorizer.transform(X_val_text)
    X_test_tfidf = vectorizer.transform(X_test_text)
    print(f"  TF-IDF shape: Train={X_train_tfidf.shape}, Val={X_val_tfidf.shape}, Test={X_test_tfidf.shape}")

    # 4. Scale all 12 Linguistic Features (Fitted on Train only)
    print("\n[INFO] Fitting StandardScaler on Train linguistic features only ...")
    scaler = StandardScaler()
    X_train_ling = scaler.fit_transform(train_feat_df[FEATURE_NAMES].values)
    X_val_ling = scaler.transform(val_feat_df[FEATURE_NAMES].values)
    X_test_ling = scaler.transform(test_feat_df[FEATURE_NAMES].values)

    # Save Scaler
    scaler_path = MODELS_DIR / "linguistic_feature_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved: {scaler_path}")

    # 5. Combine TF-IDF + Linguistic Features
    X_train_enhanced = sp.hstack([X_train_tfidf, sp.csr_matrix(X_train_ling)], format="csr")
    X_val_enhanced = sp.hstack([X_val_tfidf, sp.csr_matrix(X_val_ling)], format="csr")
    X_test_enhanced = sp.hstack([X_test_tfidf, sp.csr_matrix(X_test_ling)], format="csr")
    print(f"  Enhanced feature shape: Train={X_train_enhanced.shape}, Test={X_test_enhanced.shape}")

    # 6. Model Hyperparameters (Identical to Baseline)
    lr_config = {
        "class_weight": "balanced",
        "C": 1.0,
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
        "solver": "lbfgs",
        "multi_class": "multinomial",
    }

    # ──────────────────────────────────────────────────────────────────────────
    # EXPERIMENT A: Baseline (TF-IDF only)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[INFO] Evaluating Experiment A — Baseline (TF-IDF only) ...")
    baseline_model_path = MODELS_DIR / "logistic_regression_baseline.pkl"
    if baseline_model_path.exists():
        with open(baseline_model_path, "rb") as f:
            base_clf = pickle.load(f)
        print("  Loaded existing baseline model from models/logistic_regression_baseline.pkl")
    else:
        print("  Training baseline model ...")
        base_clf = LogisticRegression(**lr_config)
        base_clf.fit(X_train_tfidf, y_train)

    base_val_pred = base_clf.predict(X_val_tfidf)
    base_test_pred = base_clf.predict(X_test_tfidf)
    base_val_metrics = compute_metrics(y_val, base_val_pred)
    base_test_metrics = compute_metrics(y_test, base_test_pred)

    print(f"  [Baseline Test] Accuracy: {base_test_metrics['accuracy']:.4f} | Macro F1: {base_test_metrics['macro_f1']:.4f} | Weighted F1: {base_test_metrics['weighted_f1']:.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    # EXPERIMENT B: Enhanced (TF-IDF + All Linguistic Features)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[INFO] Training Experiment B — Enhanced (TF-IDF + Linguistic Features) ...")
    enhanced_clf = LogisticRegression(**lr_config)
    enhanced_clf.fit(X_train_enhanced, y_train)

    # Save Enhanced Model
    enhanced_model_path = MODELS_DIR / "logistic_regression_linguistic.pkl"
    with open(enhanced_model_path, "wb") as f:
        pickle.dump(enhanced_clf, f)
    print(f"  Enhanced model saved: {enhanced_model_path}")

    enh_val_pred = enhanced_clf.predict(X_val_enhanced)
    enh_val_proba = enhanced_clf.predict_proba(X_val_enhanced)
    enh_test_pred = enhanced_clf.predict(X_test_enhanced)
    enh_test_proba = enhanced_clf.predict_proba(X_test_enhanced)

    enh_val_metrics = compute_metrics(y_val, enh_val_pred)
    enh_test_metrics = compute_metrics(y_test, enh_test_pred)

    print(f"  [Enhanced Test] Accuracy: {enh_test_metrics['accuracy']:.4f} | Macro F1: {enh_test_metrics['macro_f1']:.4f} | Weighted F1: {enh_test_metrics['weighted_f1']:.4f}")

    # Save Enhanced Predictions
    val_pred_df = val_feat_df[["sentence", "label"]].rename(columns={"label": "actual_label"}).copy()
    val_pred_df["predicted_label"] = enh_val_pred
    val_pred_df["correct"] = val_pred_df["actual_label"] == val_pred_df["predicted_label"]
    for idx, c in enumerate(enhanced_clf.classes_):
        val_pred_df[f"{c}_probability"] = enh_val_proba[:, idx]
    val_pred_df.to_csv(PHASE4_DIR / "predictions_linguistic_val.csv", index=False, encoding="utf-8")

    test_pred_df = test_feat_df[["sentence", "label"]].rename(columns={"label": "actual_label"}).copy()
    test_pred_df["predicted_label"] = enh_test_pred
    test_pred_df["correct"] = test_pred_df["actual_label"] == test_pred_df["predicted_label"]
    for idx, c in enumerate(enhanced_clf.classes_):
        test_pred_df[f"{c}_probability"] = enh_test_proba[:, idx]
    test_pred_df.to_csv(PHASE4_DIR / "predictions_linguistic_test.csv", index=False, encoding="utf-8")
    print("  Enhanced predictions saved for val and test splits.")

    # Plot and save confusion matrix
    cm_path = PHASE4_DIR / "confusion_matrix_linguistic.png"
    plot_confusion_matrices(enh_val_metrics["confusion_matrix"], enh_test_metrics["confusion_matrix"], cm_path)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 10: FEATURE ABLATION EXPERIMENT
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("STEP 10 — FEATURE ABLATION EXPERIMENT")
    print("=" * 78)

    ablation_configs = [
        ("TF-IDF only (Baseline)", []),
        ("TF-IDF + Group A (Text Statistics)", FEATURE_GROUPS["Group A"]),
        ("TF-IDF + Group B (Punctuation & Casing)", FEATURE_GROUPS["Group B"]),
        ("TF-IDF + Group C (Sentiment Cues)", FEATURE_GROUPS["Group C"]),
        ("TF-IDF + Group D (Visual & Repetition)", FEATURE_GROUPS["Group D"]),
        ("TF-IDF + All Linguistic Features", FEATURE_NAMES),
    ]

    ablation_records = []
    for name, feat_list in ablation_configs:
        print(f"\n[Ablation] Testing: {name} ...")
        if not feat_list:
            acc = base_test_metrics["accuracy"]
            mf1 = base_test_metrics["macro_f1"]
            wf1 = base_test_metrics["weighted_f1"]
            print(f"  Accuracy: {acc:.4f} | Macro F1: {mf1:.4f} | Weighted F1: {wf1:.4f}")
            ablation_records.append({
                "feature_group": name,
                "accuracy": round(acc, 4),
                "macro_f1": round(mf1, 4),
                "weighted_f1": round(wf1, 4),
            })
            continue

        # Fit scaler on selected features from train only
        subset_scaler = StandardScaler()
        X_tr_sub = subset_scaler.fit_transform(train_feat_df[feat_list].values)
        X_te_sub = subset_scaler.transform(test_feat_df[feat_list].values)

        X_tr_comb = sp.hstack([X_train_tfidf, sp.csr_matrix(X_tr_sub)], format="csr")
        X_te_comb = sp.hstack([X_test_tfidf, sp.csr_matrix(X_te_sub)], format="csr")

        clf_ab = LogisticRegression(**lr_config)
        clf_ab.fit(X_tr_comb, y_train)

        preds_ab = clf_ab.predict(X_te_comb)
        m = compute_metrics(y_test, preds_ab)
        print(f"  Accuracy: {m['accuracy']:.4f} | Macro F1: {m['macro_f1']:.4f} | Weighted F1: {m['weighted_f1']:.4f}")
        ablation_records.append({
            "feature_group": name,
            "accuracy": round(m["accuracy"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "weighted_f1": round(m["weighted_f1"], 4),
        })

    ablation_df = pd.DataFrame(ablation_records)
    ablation_csv_path = PHASE4_DIR / "phase4_ablation_results.csv"
    ablation_df.to_csv(ablation_csv_path, index=False, encoding="utf-8")
    print(f"\n[INFO] Ablation results saved: {ablation_csv_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 11: SAVE PHASE 4 MODEL COMPARISON CSV
    # ──────────────────────────────────────────────────────────────────────────
    model_comp_records = [
        {
            "experiment": "Phase 2 Baseline (TF-IDF only)",
            "accuracy": round(base_test_metrics["accuracy"], 4),
            "macro_precision": round(base_test_metrics["macro_precision"], 4),
            "macro_recall": round(base_test_metrics["macro_recall"], 4),
            "macro_f1": round(base_test_metrics["macro_f1"], 4),
            "weighted_f1": round(base_test_metrics["weighted_f1"], 4),
        },
        {
            "experiment": "Phase 4 Enhanced (TF-IDF + Linguistic Features)",
            "accuracy": round(enh_test_metrics["accuracy"], 4),
            "macro_precision": round(enh_test_metrics["macro_precision"], 4),
            "macro_recall": round(enh_test_metrics["macro_recall"], 4),
            "macro_f1": round(enh_test_metrics["macro_f1"], 4),
            "weighted_f1": round(enh_test_metrics["weighted_f1"], 4),
        },
    ]
    model_comp_df = pd.DataFrame(model_comp_records)
    model_comp_csv_path = PHASE4_DIR / "phase4_model_comparison.csv"
    model_comp_df.to_csv(model_comp_csv_path, index=False, encoding="utf-8")
    print(f"[INFO] Model comparison saved: {model_comp_csv_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 11: GENERATE PHASE 4 REPORT
    # ──────────────────────────────────────────────────────────────────────────
    diff_macro_f1 = enh_test_metrics["macro_f1"] - base_test_metrics["macro_f1"]
    diff_acc = enh_test_metrics["accuracy"] - base_test_metrics["accuracy"]

    report_path = PHASE4_DIR / "phase4_report.txt"
    report_lines = [
        "=" * 78,
        "PHASE 4 - LINGUISTIC FEATURE ENHANCEMENT COMPREHENSIVE REPORT",
        "AI-Based Sentiment Analysis System",
        "=" * 78,
        "",
        "1. OBJECTIVE",
        "-" * 78,
        "The objective of Phase 4 is to experimentally determine whether extracting lightweight,",
        "interpretable linguistic features from the original sentence text (e.g., word count,",
        "character count, sentence length, exclamation marks, question marks, capitalization",
        "ratio, uppercase words, emojis, negations, intensifiers, contrast words, and repeated",
        "characters) provides additional useful signal to improve sentiment classification over",
        "the Phase 2 TF-IDF + Logistic Regression baseline.",
        "",
        "2. FEATURES USED & GROUPINGS",
        "-" * 78,
        "12 features extracted from original, untruncated sentence text:",
        "  - Group A (Basic text statistics):",
        "      * word_count: Total whitespace-separated words",
        "      * character_count: Total character count including whitespace",
        "      * sentence_length: Character count of stripped sentence text",
        "  - Group B (Punctuation & capitalization):",
        "      * exclamation_count: Count of '!' characters",
        "      * question_count: Count of '?' characters",
        "      * uppercase_ratio: Uppercase alphabetic characters / total alphabetic characters",
        "      * uppercase_word_count: Fully uppercase words (length >= 2)",
        "  - Group C (Sentiment linguistic cues):",
        "      * negation_count: Occurrences of 20 designated negation words/contractions",
        "      * intensifier_count: Occurrences of 11 designated intensifier degree adverbs",
        "      * contrast_word_count: Occurrences of 8 designated contrast/discourse markers",
        "  - Group D (Visual & repetition cues):",
        "      * emoji_count: Count of Unicode emojis detected via emoji library",
        "      * repeated_character_count: Occurrences of 3+ consecutive identical characters",
        "",
        "3. FEATURE DEFINITIONS & WORD LISTS",
        "-" * 78,
        "  Negations list (20 items):",
        "    not, no, never, neither, nor, don't, doesn't, didn't, isn't, aren't,",
        "    wasn't, weren't, can't, couldn't, won't, wouldn't, shouldn't, haven't, hasn't, hadn't",
        "  Intensifiers list (11 items):",
        "    very, really, extremely, highly, so, too, incredibly, absolutely,",
        "    completely, totally, quite",
        "  Contrast words list (8 items):",
        "    but, however, although, though, yet, nevertheless, nonetheless, whereas",
        "  Scaling: StandardScaler fitted exclusively on training set linguistic features.",
        "",
        "4. DATASET SIZES",
        "-" * 78,
        f"  Training samples  : {len(train_feat_df):,}",
        f"  Validation samples: {len(val_feat_df):,}",
        f"  Test samples      : {len(test_feat_df):,}",
        f"  Total samples     : {len(train_feat_df) + len(val_feat_df) + len(test_feat_df):,}",
        "  Label space       : negative, neutral, positive",
        "",
        "5. BASELINE RESULTS (TF-IDF + Logistic Regression)",
        "-" * 78,
        f"  Validation Accuracy : {base_val_metrics['accuracy']:.4f} ({base_val_metrics['accuracy']*100:.2f}%)",
        f"  Validation Macro F1 : {base_val_metrics['macro_f1']:.4f}",
        f"  Validation W-F1     : {base_val_metrics['weighted_f1']:.4f}",
        f"  Test Accuracy       : {base_test_metrics['accuracy']:.4f} ({base_test_metrics['accuracy']*100:.2f}%)",
        f"  Test Macro F1       : {base_test_metrics['macro_f1']:.4f}",
        f"  Test Weighted F1    : {base_test_metrics['weighted_f1']:.4f}",
        f"  Test Macro Precision: {base_test_metrics['macro_precision']:.4f}",
        f"  Test Macro Recall   : {base_test_metrics['macro_recall']:.4f}",
        "",
        "6. ENHANCED MODEL RESULTS (TF-IDF + 12 Scaled Linguistic Features + LR)",
        "-" * 78,
        f"  Validation Accuracy : {enh_val_metrics['accuracy']:.4f} ({enh_val_metrics['accuracy']*100:.2f}%)",
        f"  Validation Macro F1 : {enh_val_metrics['macro_f1']:.4f}",
        f"  Validation W-F1     : {enh_val_metrics['weighted_f1']:.4f}",
        f"  Test Accuracy       : {enh_test_metrics['accuracy']:.4f} ({enh_test_metrics['accuracy']*100:.2f}%)",
        f"  Test Macro F1       : {enh_test_metrics['macro_f1']:.4f}",
        f"  Test Weighted F1    : {enh_test_metrics['weighted_f1']:.4f}",
        f"  Test Macro Precision: {enh_test_metrics['macro_precision']:.4f}",
        f"  Test Macro Recall   : {enh_test_metrics['macro_recall']:.4f}",
        "",
        f"  Delta Test Macro F1 : {diff_macro_f1:+.4f}",
        f"  Delta Test Accuracy : {diff_acc:+.4f}",
        "",
        "7. ABLATION STUDY RESULTS",
        "-" * 78,
        f"{'Feature Configuration':<42} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}",
        "-" * 78,
    ]

    for rec in ablation_records:
        report_lines.append(f"{rec['feature_group']:<42} {rec['accuracy']:>10.4f} {rec['macro_f1']:>10.4f} {rec['weighted_f1']:>12.4f}")

    report_lines.extend([
        "-" * 78,
        "",
        "8. PER-CLASS PERFORMANCE BREAKDOWN (TEST SET)",
        "-" * 78,
        "Baseline Per-Class Report:",
        base_test_metrics["clf_report_txt"],
        "",
        "Enhanced Model Per-Class Report:",
        enh_test_metrics["clf_report_txt"],
        "",
        "Neutral Class Comparison:",
        f"  Baseline Neutral F1 : {base_test_metrics['clf_report_dict']['neutral']['f1-score']:.4f}",
        f"  Enhanced Neutral F1 : {enh_test_metrics['clf_report_dict']['neutral']['f1-score']:.4f}",
        f"  Delta Neutral F1    : {enh_test_metrics['clf_report_dict']['neutral']['f1-score'] - base_test_metrics['clf_report_dict']['neutral']['f1-score']:+.4f}",
        "",
        "9. CONFUSION MATRICES (TEST SET)",
        "-" * 78,
        "Baseline Test Confusion Matrix (rows=actual, cols=predicted):",
        f"  Labels order: {LABEL_ORDER}",
    ])
    for i, row in enumerate(base_test_metrics["confusion_matrix"]):
        report_lines.append(f"    {LABEL_ORDER[i]:<10}: {row.tolist()}")

    report_lines.extend([
        "",
        "Enhanced Test Confusion Matrix (rows=actual, cols=predicted):",
        f"  Labels order: {LABEL_ORDER}",
    ])
    for i, row in enumerate(enh_test_metrics["confusion_matrix"]):
        report_lines.append(f"    {LABEL_ORDER[i]:<10}: {row.tolist()}")

    report_lines.extend([
        "",
        "10. LIMITATIONS",
        "-" * 78,
        "  1. Vocabulary Overlap: Words in predefined lists (e.g. 'not', 'very', 'never') are",
        "     already captured by unigram and bigram TF-IDF tokens; adding dense scalar frequency counts",
        "     provides limited new semantic information beyond what the high-dimensional sparse TF-IDF learns.",
        "  2. Emoji Sparsity: Emoji frequency in this dataset is nearly 0.00% non-zero, offering",
        "     virtually negligible discriminative capacity.",
        "  3. Context-Blind Counts: Scalar counts cannot distinguish whether 'not' negates a positive",
        "     or negative adjective ('not bad' vs 'not good'); n-gram TF-IDF handles this better.",
        "  4. Static Word Lists: Fixed dictionary lists do not cover irregular slang, domain jargon,",
        "     or complex syntactic sentiment shifts.",
        "",
        "11. EXPERIMENTAL CONCLUSION: DID LINGUISTIC FEATURES IMPROVE MACRO F1?",
        "-" * 78,
    ])

    if diff_macro_f1 > 0.0005:
        verdict = f"YES. Linguistic features provided a marginal improvement of {diff_macro_f1:+.4f} in Test Macro F1."
    elif diff_macro_f1 < -0.0005:
        verdict = f"NO. Linguistic features slightly reduced Test Macro F1 by {diff_macro_f1:+.4f} (from {base_test_metrics['macro_f1']:.4f} to {enh_test_metrics['macro_f1']:.4f})."
    else:
        verdict = f"NEUTRAL / NO MEANINGFUL CHANGE. Linguistic features resulted in a negligible difference of {diff_macro_f1:+.4f} in Test Macro F1 (from {base_test_metrics['macro_f1']:.4f} to {enh_test_metrics['macro_f1']:.4f})."

    report_lines.append(f"  Verdict: {verdict}")
    report_lines.append("  Baseline Test Macro F1: " + f"{base_test_metrics['macro_f1']:.4f}")
    report_lines.append("  Enhanced Test Macro F1: " + f"{enh_test_metrics['macro_f1']:.4f}")
    report_lines.append("=" * 78)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n[INFO] Phase 4 comprehensive report written: {report_path}")

    return {
        "base_test_metrics": base_test_metrics,
        "enh_test_metrics": enh_test_metrics,
        "diff_macro_f1": diff_macro_f1,
        "diff_acc": diff_acc,
        "ablation_records": ablation_records,
    }


if __name__ == "__main__":
    run_phase4_pipeline()
