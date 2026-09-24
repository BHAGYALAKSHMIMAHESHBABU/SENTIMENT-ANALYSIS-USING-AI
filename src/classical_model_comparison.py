"""
Phase 3 - Classical Model Comparison
AI-Based Sentiment Analysis System

Compares:
  1. Logistic Regression  (Phase 2 baseline - results loaded, NOT retrained)
  2. Multinomial Naive Bayes  (newly trained)
  3. Linear SVM               (newly trained)

All three classifiers use the SAME TF-IDF representation:
  - Fitted on train_clean.csv ONLY
  - Same hyperparameters as Phase 2
  - Val and test are transform-only

Outputs saved to: outputs/phase3/
Models saved to : models/

IMPORTANT:
  - Phase 2 outputs are NOT overwritten.
  - Raw data/ files are NOT modified.
  - LinearSVC has no predict_proba; decision scores are saved instead.
"""

import sys
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).resolve().parent.parent
PROC      = BASE / "outputs" / "processed"
MODELS    = BASE / "models"
P3_OUT    = BASE / "outputs" / "phase3"
P3_OUT.mkdir(parents=True, exist_ok=True)

LABEL_ORDER  = ["negative", "neutral", "positive"]
RANDOM_STATE = 42

# ── Text preprocessing (must match Phase 2 exactly) ───────────────────────────
def preprocess(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Load cleaned data ──────────────────────────────────────────────────────────
def load_data():
    train = pd.read_csv(PROC / "train_clean.csv")
    val   = pd.read_csv(PROC / "val_clean.csv")
    test  = pd.read_csv(PROC / "test_clean.csv")
    return train, val, test

# ── Evaluate helper ────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    return {
        "accuracy"        : accuracy_score(y_true, y_pred),
        "macro_f1"        : f1_score(y_true, y_pred, average="macro",    labels=LABEL_ORDER, zero_division=0),
        "weighted_f1"     : f1_score(y_true, y_pred, average="weighted", labels=LABEL_ORDER, zero_division=0),
        "macro_precision" : precision_score(y_true, y_pred, average="macro",    labels=LABEL_ORDER, zero_division=0),
        "macro_recall"    : recall_score(y_true, y_pred, average="macro",       labels=LABEL_ORDER, zero_division=0),
        "clf_report"      : classification_report(y_true, y_pred, labels=LABEL_ORDER, digits=4, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABEL_ORDER),
    }

# ── Confusion matrix chart ─────────────────────────────────────────────────────
def plot_cm(val_cm, test_cm, title_prefix, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for cm, title, ax in [
        (val_cm,  f"{title_prefix} — Validation", axes[0]),
        (test_cm, f"{title_prefix} — Test",       axes[1]),
    ]:
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax, shrink=0.85)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xticks(range(len(LABEL_ORDER)))
        ax.set_yticks(range(len(LABEL_ORDER)))
        ax.set_xticklabels([l.capitalize() for l in LABEL_ORDER], fontsize=9, rotation=15)
        ax.set_yticklabels([l.capitalize() for l in LABEL_ORDER], fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual",    fontsize=9)
        thresh = cm.max() / 2.0
        for i in range(len(LABEL_ORDER)):
            for j in range(len(LABEL_ORDER)):
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=12, fontweight="bold", color=color)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved: {out_path.name}")

# ── Save classification report ─────────────────────────────────────────────────
def save_clf_report(val_m, test_m, model_name, out_path):
    lines = [
        "=" * 60,
        f"Classification Report — {model_name}",
        "=" * 60,
        "\n[VALIDATION SET]",
        val_m["clf_report"],
        "\n[TEST SET]",
        test_m["clf_report"],
        "=" * 60,
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Classification report saved: {out_path.name}")

# ── Save predictions CSV ───────────────────────────────────────────────────────
def save_predictions(df, y_pred, scores, score_cols, out_path):
    out = df[["sentence", "label"]].copy().rename(columns={"label": "actual_label"})
    out["predicted_label"] = y_pred
    out["correct"] = out["actual_label"] == out["predicted_label"]
    if scores is not None:
        for i, col in enumerate(score_cols):
            out[col] = scores[:, i] if scores.ndim == 2 else scores
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Predictions saved: {out_path.name}  ({len(out):,} rows)")

# ── Main ───────────────────────────────────────────────────────────────────────
def run():
    report = [
        "=" * 70,
        "PHASE 3 — CLASSICAL MODEL COMPARISON REPORT",
        "AI-Based Sentiment Analysis System",
        "=" * 70,
    ]

    # ── 1. Load data ────────────────────────────────────────────────────────
    print("[INFO] Loading cleaned datasets ...")
    train, val, test = load_data()

    # Verify zero overlap
    tv_overlap = set(train["sentence"]).intersection(set(val["sentence"]))
    tt_overlap = set(train["sentence"]).intersection(set(test["sentence"]))
    report.append(f"\n[DATA LEAKAGE VERIFICATION]")
    report.append(f"  Train/Val overlap : {len(tv_overlap)} sentences")
    report.append(f"  Train/Test overlap: {len(tt_overlap)} sentences")
    assert len(tv_overlap) == 0, f"Train/Val overlap detected: {tv_overlap}"
    assert len(tt_overlap) == 0, f"Train/Test overlap detected: {tt_overlap}"
    report.append(f"  Status: CLEAN — no leakage detected")
    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
    print(f"  Train/Val overlap: {len(tv_overlap)}  Train/Test overlap: {len(tt_overlap)}")

    report.append(f"\n[DATASET SIZES (after Phase 3 train/val fix)]")
    report.append(f"  Train : {len(train):,}")
    report.append(f"  Val   : {len(val):,}")
    report.append(f"  Test  : {len(test):,}")

    # ── 2. Preprocess ───────────────────────────────────────────────────────
    print("[INFO] Applying text preprocessing ...")
    X_train_raw = train["sentence"].apply(preprocess).values
    X_val_raw   = val["sentence"].apply(preprocess).values
    X_test_raw  = test["sentence"].apply(preprocess).values
    y_train = train["label"].values
    y_val   = val["label"].values
    y_test  = test["label"].values

    # ── 3. TF-IDF — refit on updated train (102,076 rows) ──────────────────
    print("[INFO] Fitting TF-IDF on train (Phase 3 train_clean — 102,076 rows) ...")
    tfidf_cfg = dict(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
        analyzer="word",
    )
    vectorizer = TfidfVectorizer(**tfidf_cfg)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_val   = vectorizer.transform(X_val_raw)
    X_test  = vectorizer.transform(X_test_raw)

    n_features = X_train.shape[1]
    print(f"  Vocabulary size: {n_features:,} features")

    # Save updated vectorizer for Phase 3 (keeps Phase 2 vectorizer intact)
    vec_p3_path = MODELS / "tfidf_vectorizer_phase3.pkl"
    with open(vec_p3_path, "wb") as f:
        pickle.dump(vectorizer, f)

    report.append(f"\n[TF-IDF CONFIGURATION (Phase 3)]")
    for k, v in tfidf_cfg.items():
        report.append(f"  {k:<20}: {v}")
    report.append(f"  {'Vocabulary size':<20}: {n_features:,} features")
    report.append(f"  Fitted on train only. Val + Test are transformed only.")
    report.append(f"  Saved: models/tfidf_vectorizer_phase3.pkl")
    report.append(f"  NOTE: Phase 2 tfidf_vectorizer.pkl is UNCHANGED.")

    # ── 4. Phase 2 LR baseline results (load, do NOT retrain) ──────────────
    print("\n[INFO] Loading Phase 2 Logistic Regression baseline results (not retraining) ...")
    LR_PHASE2 = {
        "val": {
            "accuracy": 0.6531, "macro_f1": 0.6529, "weighted_f1": 0.6533,
            "macro_precision": 0.6535, "macro_recall": 0.6540,
        },
        "test": {
            "accuracy": 0.6629, "macro_f1": 0.6611, "weighted_f1": 0.6635,
            "macro_precision": 0.6606, "macro_recall": 0.6625,
        },
    }
    # Per-class test F1 from Phase 2 report
    LR_PER_CLASS_TEST_F1 = {"negative": 0.6546, "neutral": 0.6315, "positive": 0.6973}

    report.append(f"\n{'='*60}")
    report.append(f"MODEL 1: Logistic Regression (Phase 2 Baseline — NOT retrained)")
    report.append(f"  Config: class_weight=balanced, C=1.0, max_iter=1000, solver=lbfgs")
    report.append(f"{'='*60}")
    report.append(f"  [VALIDATION]")
    for k, v in LR_PHASE2["val"].items():
        report.append(f"    {k:<22}: {v:.4f}")
    report.append(f"  [TEST]")
    for k, v in LR_PHASE2["test"].items():
        report.append(f"    {k:<22}: {v:.4f}")
    report.append(f"  Per-class Test F1:")
    for cls, f1 in LR_PER_CLASS_TEST_F1.items():
        report.append(f"    {cls:<12}: {f1:.4f}")
    report.append(f"  Saved: models/logistic_regression_baseline.pkl (Phase 2, UNCHANGED)")

    # ── 5. Multinomial Naive Bayes ──────────────────────────────────────────
    print("\n[INFO] Training Multinomial Naive Bayes ...")
    mnb = MultinomialNB(alpha=1.0)

    # MNB requires non-negative input; TF-IDF with sublinear_tf can produce
    # values that are non-negative by definition (log(1+tf) >= 0, idf > 0).
    # Dense conversion not needed — fit_transform already returns sparse CSR.
    mnb.fit(X_train, y_train)

    mnb_path = MODELS / "multinomial_naive_bayes.pkl"
    with open(mnb_path, "wb") as f:
        pickle.dump(mnb, f)
    print(f"  Model saved: {mnb_path.name}")

    y_val_mnb  = mnb.predict(X_val)
    y_test_mnb = mnb.predict(X_test)
    val_mnb_proba  = mnb.predict_proba(X_val)
    test_mnb_proba = mnb.predict_proba(X_test)

    val_mnb  = compute_metrics(y_val,  y_val_mnb)
    test_mnb = compute_metrics(y_test, y_test_mnb)

    classes_mnb = list(mnb.classes_)
    proba_cols  = [f"{l}_probability" for l in LABEL_ORDER]
    # reorder proba columns to match LABEL_ORDER
    idx_map = [classes_mnb.index(l) for l in LABEL_ORDER]
    val_mnb_proba_ord  = val_mnb_proba[:, idx_map]
    test_mnb_proba_ord = test_mnb_proba[:, idx_map]

    save_predictions(val,  y_val_mnb,  val_mnb_proba_ord,  proba_cols, P3_OUT / "predictions_naive_bayes_val.csv")
    save_predictions(test, y_test_mnb, test_mnb_proba_ord, proba_cols, P3_OUT / "predictions_naive_bayes_test.csv")

    plot_cm(val_mnb["confusion_matrix"], test_mnb["confusion_matrix"],
            "Multinomial Naive Bayes", P3_OUT / "confusion_matrix_naive_bayes.png")
    save_clf_report(val_mnb, test_mnb, "Multinomial Naive Bayes (alpha=1.0)",
                    P3_OUT / "classification_report_naive_bayes.txt")

    report.append(f"\n{'='*60}")
    report.append(f"MODEL 2: Multinomial Naive Bayes")
    report.append(f"  Config: alpha=1.0")
    report.append(f"{'='*60}")
    report.append(f"  [VALIDATION]")
    for k in ["accuracy","macro_f1","weighted_f1","macro_precision","macro_recall"]:
        report.append(f"    {k:<22}: {val_mnb[k]:.4f}")
    report.append(f"  [TEST]")
    for k in ["accuracy","macro_f1","weighted_f1","macro_precision","macro_recall"]:
        report.append(f"    {k:<22}: {test_mnb[k]:.4f}")
    report.append(f"\n  Per-class Test Report:\n{test_mnb['clf_report']}")
    report.append(f"  Confusion Matrix (Test, rows=actual, cols=predicted):")
    for i, row in enumerate(test_mnb["confusion_matrix"]):
        report.append(f"    {LABEL_ORDER[i]:<10}: {row.tolist()}")
    report.append(f"  Saved: models/multinomial_naive_bayes.pkl")

    print(f"  [MNB] Val  Acc={val_mnb['accuracy']:.4f}  MacroF1={val_mnb['macro_f1']:.4f}")
    print(f"  [MNB] Test Acc={test_mnb['accuracy']:.4f}  MacroF1={test_mnb['macro_f1']:.4f}")

    # ── 6. Linear SVM ───────────────────────────────────────────────────────
    print("\n[INFO] Training Linear SVM ...")
    svm = LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE, max_iter=2000)
    svm.fit(X_train, y_train)

    svm_path = MODELS / "linear_svm.pkl"
    with open(svm_path, "wb") as f:
        pickle.dump(svm, f)
    print(f"  Model saved: {svm_path.name}")

    y_val_svm  = svm.predict(X_val)
    y_test_svm = svm.predict(X_test)

    # Decision scores (LinearSVC has no predict_proba by default)
    dec_val  = svm.decision_function(X_val)   # shape (n, 3)
    dec_test = svm.decision_function(X_test)
    dec_cols = [f"{l}_decision_score" for l in svm.classes_]

    save_predictions(val,  y_val_svm,  dec_val,  dec_cols, P3_OUT / "predictions_linear_svm_val.csv")
    save_predictions(test, y_test_svm, dec_test, dec_cols, P3_OUT / "predictions_linear_svm_test.csv")

    val_svm  = compute_metrics(y_val,  y_val_svm)
    test_svm = compute_metrics(y_test, y_test_svm)

    plot_cm(val_svm["confusion_matrix"], test_svm["confusion_matrix"],
            "Linear SVM", P3_OUT / "confusion_matrix_linear_svm.png")
    save_clf_report(val_svm, test_svm, "Linear SVM (C=1.0, class_weight=balanced)",
                    P3_OUT / "classification_report_linear_svm.txt")

    report.append(f"\n{'='*60}")
    report.append(f"MODEL 3: Linear SVM")
    report.append(f"  Config: C=1.0, class_weight=balanced, random_state=42")
    report.append(f"{'='*60}")
    report.append(f"  [VALIDATION]")
    for k in ["accuracy","macro_f1","weighted_f1","macro_precision","macro_recall"]:
        report.append(f"    {k:<22}: {val_svm[k]:.4f}")
    report.append(f"  [TEST]")
    for k in ["accuracy","macro_f1","weighted_f1","macro_precision","macro_recall"]:
        report.append(f"    {k:<22}: {test_svm[k]:.4f}")
    report.append(f"\n  Per-class Test Report:\n{test_svm['clf_report']}")
    report.append(f"  Confusion Matrix (Test, rows=actual, cols=predicted):")
    for i, row in enumerate(test_svm["confusion_matrix"]):
        report.append(f"    {LABEL_ORDER[i]:<10}: {row.tolist()}")
    report.append(f"  NOTE: LinearSVC has no predict_proba; decision scores saved instead.")
    report.append(f"  Saved: models/linear_svm.pkl")

    print(f"  [SVM] Val  Acc={val_svm['accuracy']:.4f}  MacroF1={val_svm['macro_f1']:.4f}")
    print(f"  [SVM] Test Acc={test_svm['accuracy']:.4f}  MacroF1={test_svm['macro_f1']:.4f}")

    # ── 7. Comparison table ─────────────────────────────────────────────────
    print("\n[INFO] Generating comparison table ...")

    # Extract per-class test F1 from classification report dict
    def per_class_f1(report_str):
        out = {}
        for line in report_str.splitlines():
            for label in LABEL_ORDER:
                if line.strip().startswith(label):
                    parts = line.split()
                    try:
                        out[label] = float(parts[3])
                    except (IndexError, ValueError):
                        out[label] = 0.0
        return out

    mnb_pc = per_class_f1(test_mnb["clf_report"])
    svm_pc = per_class_f1(test_svm["clf_report"])

    comparison_rows = [
        {
            "model"              : "Logistic Regression (Phase 2)",
            "val_accuracy"       : LR_PHASE2["val"]["accuracy"],
            "val_macro_f1"       : LR_PHASE2["val"]["macro_f1"],
            "val_weighted_f1"    : LR_PHASE2["val"]["weighted_f1"],
            "val_macro_precision": LR_PHASE2["val"]["macro_precision"],
            "val_macro_recall"   : LR_PHASE2["val"]["macro_recall"],
            "test_accuracy"      : LR_PHASE2["test"]["accuracy"],
            "test_macro_f1"      : LR_PHASE2["test"]["macro_f1"],
            "test_weighted_f1"   : LR_PHASE2["test"]["weighted_f1"],
            "test_macro_precision": LR_PHASE2["test"]["macro_precision"],
            "test_macro_recall"  : LR_PHASE2["test"]["macro_recall"],
            "test_f1_negative"   : LR_PER_CLASS_TEST_F1["negative"],
            "test_f1_neutral"    : LR_PER_CLASS_TEST_F1["neutral"],
            "test_f1_positive"   : LR_PER_CLASS_TEST_F1["positive"],
        },
        {
            "model"              : "Multinomial Naive Bayes",
            "val_accuracy"       : val_mnb["accuracy"],
            "val_macro_f1"       : val_mnb["macro_f1"],
            "val_weighted_f1"    : val_mnb["weighted_f1"],
            "val_macro_precision": val_mnb["macro_precision"],
            "val_macro_recall"   : val_mnb["macro_recall"],
            "test_accuracy"      : test_mnb["accuracy"],
            "test_macro_f1"      : test_mnb["macro_f1"],
            "test_weighted_f1"   : test_mnb["weighted_f1"],
            "test_macro_precision": test_mnb["macro_precision"],
            "test_macro_recall"  : test_mnb["macro_recall"],
            "test_f1_negative"   : mnb_pc.get("negative", 0.0),
            "test_f1_neutral"    : mnb_pc.get("neutral",  0.0),
            "test_f1_positive"   : mnb_pc.get("positive", 0.0),
        },
        {
            "model"              : "Linear SVM",
            "val_accuracy"       : val_svm["accuracy"],
            "val_macro_f1"       : val_svm["macro_f1"],
            "val_weighted_f1"    : val_svm["weighted_f1"],
            "val_macro_precision": val_svm["macro_precision"],
            "val_macro_recall"   : val_svm["macro_recall"],
            "test_accuracy"      : test_svm["accuracy"],
            "test_macro_f1"      : test_svm["macro_f1"],
            "test_weighted_f1"   : test_svm["weighted_f1"],
            "test_macro_precision": test_svm["macro_precision"],
            "test_macro_recall"  : test_svm["macro_recall"],
            "test_f1_negative"   : svm_pc.get("negative", 0.0),
            "test_f1_neutral"    : svm_pc.get("neutral",  0.0),
            "test_f1_positive"   : svm_pc.get("positive", 0.0),
        },
    ]

    cdf = pd.DataFrame(comparison_rows)
    metrics_path = P3_OUT / "model_comparison_metrics.csv"
    cdf.to_csv(metrics_path, index=False, encoding="utf-8")
    print(f"  Comparison metrics saved: {metrics_path.name}")

    # ── 8. Comparison summary in report ────────────────────────────────────
    report.append(f"\n{'='*70}")
    report.append(f"PHASE 3 — MODEL COMPARISON SUMMARY (TEST SET)")
    report.append(f"{'='*70}")
    hdr = f"{'Model':<34} {'Acc':>6} {'MacroF1':>8} {'WtdF1':>7} {'MacroPr':>8} {'MacroRe':>8}"
    sep = "-" * len(hdr)
    report.append(hdr)
    report.append(sep)
    for r in comparison_rows:
        report.append(
            f"{r['model']:<34} {r['test_accuracy']:>6.4f} "
            f"{r['test_macro_f1']:>8.4f} {r['test_weighted_f1']:>7.4f} "
            f"{r['test_macro_precision']:>8.4f} {r['test_macro_recall']:>8.4f}"
        )

    report.append(f"\nPer-class Test F1 (primary diagnostic — Neutral is hardest):")
    hdr2 = f"{'Model':<34} {'Negative':>9} {'Neutral':>8} {'Positive':>9}"
    report.append(hdr2)
    report.append("-" * len(hdr2))
    for r in comparison_rows:
        report.append(
            f"{r['model']:<34} {r['test_f1_negative']:>9.4f} "
            f"{r['test_f1_neutral']:>8.4f} {r['test_f1_positive']:>9.4f}"
        )

    # Rank by test Macro F1
    ranked = sorted(comparison_rows, key=lambda x: x["test_macro_f1"], reverse=True)
    report.append(f"\nRanking by Test Macro F1:")
    for rank, r in enumerate(ranked, 1):
        report.append(f"  {rank}. {r['model']:<34} Macro F1 = {r['test_macro_f1']:.4f}")

    report.append(f"\n[WARNINGS AND NOTES]")
    report.append(f"  1. LR results are from Phase 2 (train had 102,077 rows then). "
                  f"New train has 102,076 (1 row removed for train/val overlap fix).")
    report.append(f"  2. MultinomialNB requires non-negative features; "
                  f"TF-IDF sublinear_tf satisfies this (log(1+tf) >= 0).")
    report.append(f"  3. LinearSVC has no predict_proba. Decision scores are saved instead.")
    report.append(f"  4. Phase 2 outputs and models are UNCHANGED.")
    report.append(f"  5. Raw data/ files are UNCHANGED.")

    report.append(f"\n[PHASE 3 OUTPUT FILES]")
    for f in sorted(P3_OUT.iterdir()):
        report.append(f"  {f.name}")
    report.append(f"  models/multinomial_naive_bayes.pkl")
    report.append(f"  models/linear_svm.pkl")
    report.append(f"  models/tfidf_vectorizer_phase3.pkl")

    report.append("\n" + "=" * 70)

    # Write phase3_report.txt
    report_text = "\n".join(report)
    rpt_path = P3_OUT / "phase3_report.txt"
    rpt_path.write_text(report_text, encoding="utf-8")
    print(f"\n[INFO] Phase 3 report written: {rpt_path}")
    print(report_text)


if __name__ == "__main__":
    run()
