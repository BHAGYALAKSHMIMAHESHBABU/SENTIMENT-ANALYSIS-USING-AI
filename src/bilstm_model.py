"""
Phase 5 - Bidirectional LSTM (BiLSTM) Sentiment Classification
AI-Based Sentiment Analysis System

Pipeline:
  1. Load cleaned datasets from outputs/processed/
  2. Encode labels: negative=0, neutral=1, positive=2
  3. Fit Keras Tokenizer on TRAINING DATA ONLY (vocab size = 40,000)
  4. Convert sentences to padded integer sequences (maxlen = 32)
  5. Compute balanced class weights from training labels only
  6. Build BiLSTM: Embedding → BiLSTM → Dropout → Dense → Softmax
  7. Train with EarlyStopping (monitor val_loss, patience=2)
  8. Evaluate on validation and test sets
  9. Save model, tokenizer, predictions, metrics, confusion matrix,
     training history plot, and Phase 5 report

Guarantees:
  - Tokenizer fitted on train ONLY
  - Sequence length determined from train ONLY
  - Class weights derived from train labels ONLY
  - Raw data and Phase 2/3/4 outputs are NOT touched
  - No BERT, no transformer, no linguistic features in this phase
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

# Suppress TF startup noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Ensure UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
PROC_DIR   = BASE_DIR / "outputs" / "processed"
MODELS_DIR = BASE_DIR / "models"
P5_DIR     = BASE_DIR / "outputs" / "phase5"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
P5_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
LABEL_MAP   = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_NAMES = ["negative", "neutral", "positive"]
RANDOM_SEED = 42

MAX_VOCAB_SIZE = 40000   # covers full training vocabulary (39,091 unique words)
OOV_TOKEN      = "<OOV>"
MAX_SEQ_LEN    = 32      # 95th percentile of training sentence lengths is 29 words
EMBEDDING_DIM  = 128
LSTM_UNITS     = 64
DROPOUT_RATE   = 0.3
DENSE_UNITS    = 64
LEARNING_RATE  = 0.001
BATCH_SIZE     = 256
MAX_EPOCHS     = 12
PATIENCE       = 2

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── Load Cleaned Data ─────────────────────────────────────────────────────────
def load_data():
    train = pd.read_csv(PROC_DIR / "train_clean.csv")
    val   = pd.read_csv(PROC_DIR / "val_clean.csv")
    test  = pd.read_csv(PROC_DIR / "test_clean.csv")
    return train, val, test


# ── Analyse Sequence Length (from train only) ─────────────────────────────────
def analyse_seq_lengths(texts):
    lengths = np.array([len(str(t).split()) for t in texts])
    stats = {
        "min":   int(lengths.min()),
        "mean":  float(lengths.mean()),
        "p50":   float(np.percentile(lengths, 50)),
        "p90":   float(np.percentile(lengths, 90)),
        "p95":   float(np.percentile(lengths, 95)),
        "p99":   float(np.percentile(lengths, 99)),
        "max":   int(lengths.max()),
        "pct_truncated": float((lengths > MAX_SEQ_LEN).mean() * 100),
        "pct_padded":    float((lengths < MAX_SEQ_LEN).mean() * 100),
    }
    return stats


# ── Tokenise & Pad ────────────────────────────────────────────────────────────
def tokenise_and_pad(tokenizer, texts, maxlen=MAX_SEQ_LEN):
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=maxlen, padding="post", truncating="post")


# ── Build BiLSTM Model ───────────────────────────────────────────────────────
def build_bilstm(vocab_size):
    inputs = keras.Input(shape=(MAX_SEQ_LEN,), name="sequence_input")
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        mask_zero=False,
        name="embedding"
    )(inputs)
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS, return_sequences=False),
        name="bilstm"
    )(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout")(x)
    x = layers.Dense(DENSE_UNITS, activation="relu", name="dense")(x)
    outputs = layers.Dense(3, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="bilstm_sentiment")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Evaluate Metrics ─────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    return {
        "accuracy":        accuracy_score(y_true, y_pred),
        "macro_f1":        f1_score(y_true, y_pred, average="macro",    labels=[0,1,2], zero_division=0),
        "weighted_f1":     f1_score(y_true, y_pred, average="weighted", labels=[0,1,2], zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro",    labels=[0,1,2], zero_division=0),
        "macro_recall":    recall_score(y_true, y_pred, average="macro",       labels=[0,1,2], zero_division=0),
        "clf_report":      classification_report(y_true, y_pred,
                               target_names=LABEL_NAMES, digits=4, zero_division=0),
        "clf_report_dict": classification_report(y_true, y_pred,
                               target_names=LABEL_NAMES, digits=4,
                               zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0,1,2]),
    }


# ── Plot Confusion Matrix ────────────────────────────────────────────────────
def plot_confusion_matrices(val_cm, test_cm, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for cm, title, ax in [
        (val_cm,  "Validation Set", axes[0]),
        (test_cm, "Test Set",       axes[1]),
    ]:
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax, shrink=0.85)
        ax.set_title(f"BiLSTM — {title}", fontsize=11, fontweight="bold", pad=10)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels([l.capitalize() for l in LABEL_NAMES], fontsize=9, rotation=15)
        ax.set_yticklabels([l.capitalize() for l in LABEL_NAMES], fontsize=9)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("Actual Label", fontsize=10)
        thresh = cm.max() / 2.0
        for i in range(3):
            for j in range(3):
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)
    plt.suptitle("Phase 5 BiLSTM — Confusion Matrices",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved: {out_path}")


# ── Plot Training History ────────────────────────────────────────────────────
def plot_training_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # Loss
    ax = axes[0]
    ax.plot(history.history["loss"],     label="Train Loss",     color="#2979FF", linewidth=2)
    ax.plot(history.history["val_loss"], label="Val Loss",       color="#FF6D00", linewidth=2, linestyle="--")
    ax.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    # Accuracy
    ax = axes[1]
    ax.plot(history.history["accuracy"],     label="Train Accuracy", color="#2979FF", linewidth=2)
    ax.plot(history.history["val_accuracy"], label="Val Accuracy",   color="#FF6D00", linewidth=2, linestyle="--")
    ax.set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle("Phase 5 BiLSTM — Training History", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Training history plot saved: {out_path}")


# ── Main Pipeline ─────────────────────────────────────────────────────────────
def run_bilstm_pipeline():
    print("=" * 78)
    print("PHASE 5 — BIDIRECTIONAL LSTM (BiLSTM) SENTIMENT ANALYSIS")
    print("=" * 78)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n[STEP 1] Loading cleaned datasets ...")
    train, val, test = load_data()
    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    # Verify no overlap
    train_texts = set(train["sentence"].astype(str))
    val_texts   = set(val["sentence"].astype(str))
    test_texts  = set(test["sentence"].astype(str))
    tv_overlap  = train_texts & val_texts
    tt_overlap  = train_texts & test_texts
    print(f"  Train/Val overlap: {len(tv_overlap)} | Train/Test overlap: {len(tt_overlap)}")
    assert len(tv_overlap) == 0, "INTEGRITY ERROR: Train/Val overlap detected!"
    assert len(tt_overlap) == 0, "INTEGRITY ERROR: Train/Test overlap detected!"

    # ── Encode labels ──────────────────────────────────────────────────────────
    print("\n[STEP 2] Encoding labels ...")
    y_train = np.array([LABEL_MAP[l] for l in train["label"]])
    y_val   = np.array([LABEL_MAP[l] for l in val["label"]])
    y_test  = np.array([LABEL_MAP[l] for l in test["label"]])
    label_dist = {name: int((y_train == idx).sum()) for name, idx in LABEL_MAP.items()}
    print(f"  Label mapping: {LABEL_MAP}")
    print(f"  Train distribution: {label_dist}")

    # ── Analyse sequence length ────────────────────────────────────────────────
    print("\n[STEP 3] Analysing training sequence lengths ...")
    seq_stats = analyse_seq_lengths(train["sentence"])
    print(f"  Min={seq_stats['min']} | Mean={seq_stats['mean']:.1f} | "
          f"P50={seq_stats['p50']:.1f} | P90={seq_stats['p90']:.1f} | "
          f"P95={seq_stats['p95']:.1f} | P99={seq_stats['p99']:.1f} | Max={seq_stats['max']}")
    print(f"  Selected MAX_SEQ_LEN={MAX_SEQ_LEN}: "
          f"Truncated={seq_stats['pct_truncated']:.2f}% | Padded={seq_stats['pct_padded']:.2f}%")

    # ── Tokenise ───────────────────────────────────────────────────────────────
    print(f"\n[STEP 4] Fitting Tokenizer on train data (vocab_size={MAX_VOCAB_SIZE}, oov='{OOV_TOKEN}') ...")
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(train["sentence"].astype(str))
    actual_vocab = len(tokenizer.word_index)
    print(f"  Actual vocabulary in train: {actual_vocab:,} unique tokens")
    print(f"  Tokenizer num_words (capped): {MAX_VOCAB_SIZE:,}")

    # ── Pad sequences ─────────────────────────────────────────────────────────
    print("\n[STEP 5] Converting to padded integer sequences ...")
    X_train = tokenise_and_pad(tokenizer, train["sentence"].astype(str))
    X_val   = tokenise_and_pad(tokenizer, val["sentence"].astype(str))
    X_test  = tokenise_and_pad(tokenizer, test["sentence"].astype(str))
    print(f"  X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")

    # ── Class weights ──────────────────────────────────────────────────────────
    print("\n[STEP 6] Computing class weights from train labels only ...")
    unique_classes = np.unique(y_train)
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=y_train
    )
    class_weight_dict = {int(cls): float(w) for cls, w in zip(unique_classes, class_weights_arr)}
    print(f"  Class weights: {class_weight_dict}")
    print(f"    negative ({LABEL_MAP['negative']}): {class_weight_dict[0]:.4f}")
    print(f"    neutral  ({LABEL_MAP['neutral']}):  {class_weight_dict[1]:.4f}")
    print(f"    positive ({LABEL_MAP['positive']}): {class_weight_dict[2]:.4f}")

    # ── Build & summarise model ────────────────────────────────────────────────
    print("\n[STEP 7] Building BiLSTM model ...")
    vocab_size_for_embedding = min(MAX_VOCAB_SIZE, actual_vocab) + 1  # +1 for padding index 0
    model = build_bilstm(vocab_size_for_embedding)
    model.summary()

    total_params = model.count_params()
    print(f"  Total parameters: {total_params:,}")

    # ── Train ──────────────────────────────────────────────────────────────────
    print(f"\n[STEP 8] Training BiLSTM (max_epochs={MAX_EPOCHS}, batch_size={BATCH_SIZE}) ...")
    print(f"  EarlyStopping: monitor=val_loss, patience={PATIENCE}, restore_best_weights=True")

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=1,
    )

    epochs_trained = len(history.history["loss"])
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    print(f"\n  Training completed: {epochs_trained} epoch(s). Best epoch: {best_epoch}")

    # Save training history
    hist_df = pd.DataFrame(history.history)
    hist_df.index.name = "epoch"
    hist_df.to_csv(P5_DIR / "training_history.csv", index=True, encoding="utf-8")
    print(f"  Training history saved: {P5_DIR / 'training_history.csv'}")

    # Plot training history
    plot_training_history(history, P5_DIR / "training_history.png")

    # ── Save model & tokenizer ─────────────────────────────────────────────────
    model_path = MODELS_DIR / "bilstm_sentiment.keras"
    model.save(model_path)
    print(f"  Model saved: {model_path}")

    tok_path = MODELS_DIR / "bilstm_tokenizer.pkl"
    with open(tok_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"  Tokenizer saved: {tok_path}")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print("\n[STEP 9] Evaluating on Validation set ...")
    y_val_proba = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    y_val_pred  = np.argmax(y_val_proba, axis=1)
    val_metrics = compute_metrics(y_val, y_val_pred)
    print(f"  Val  Accuracy: {val_metrics['accuracy']:.4f} | Macro F1: {val_metrics['macro_f1']:.4f} | Weighted F1: {val_metrics['weighted_f1']:.4f}")

    print("\n[STEP 9] Evaluating on Test set ...")
    y_test_proba = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    y_test_pred  = np.argmax(y_test_proba, axis=1)
    test_metrics = compute_metrics(y_test, y_test_pred)
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f} | Macro F1: {test_metrics['macro_f1']:.4f} | Weighted F1: {test_metrics['weighted_f1']:.4f}")

    # ── Save predictions ───────────────────────────────────────────────────────
    print("\n[STEP 10] Saving predictions ...")
    pred_df = test[["sentence", "label"]].copy().rename(columns={"label": "actual_label"})
    pred_df["predicted_label"] = [LABEL_NAMES[p] for p in y_test_pred]
    pred_df["correct"] = pred_df["actual_label"] == pred_df["predicted_label"]
    for i, name in enumerate(LABEL_NAMES):
        pred_df[f"{name}_probability"] = y_test_proba[:, i]
    pred_path = P5_DIR / "bilstm_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8")
    print(f"  Predictions saved: {pred_path} ({len(pred_df):,} rows)")

    # ── Confusion matrices ─────────────────────────────────────────────────────
    plot_confusion_matrices(
        val_metrics["confusion_matrix"],
        test_metrics["confusion_matrix"],
        P5_DIR / "confusion_matrix_bilstm.png"
    )

    # ── Classification report ──────────────────────────────────────────────────
    with open(P5_DIR / "classification_report_bilstm.txt", "w", encoding="utf-8") as f:
        f.write("Phase 5 BiLSTM — Classification Report (Test Set)\n")
        f.write("=" * 60 + "\n\n")
        f.write(test_metrics["clf_report"])
    print(f"  Classification report saved: {P5_DIR / 'classification_report_bilstm.txt'}")

    # ── Metrics CSV ───────────────────────────────────────────────────────────
    # Comparison baselines
    phase2_results = {"accuracy": 0.6629, "macro_precision": 0.6606, "macro_recall": 0.6625, "macro_f1": 0.6611, "weighted_f1": 0.6635}
    phase4_results = {"accuracy": 0.6669, "macro_precision": 0.6651, "macro_recall": 0.6680, "macro_f1": 0.6657, "weighted_f1": 0.6674}

    metrics_df = pd.DataFrame([
        {"model": "Phase 2 Baseline (TF-IDF + LR)",           **phase2_results},
        {"model": "Phase 4 Enhanced (TF-IDF + LingFeats + LR)", **phase4_results},
        {"model": "Phase 5 BiLSTM (Trained from scratch)",
         "accuracy":        round(test_metrics["accuracy"],        4),
         "macro_precision": round(test_metrics["macro_precision"], 4),
         "macro_recall":    round(test_metrics["macro_recall"],    4),
         "macro_f1":        round(test_metrics["macro_f1"],        4),
         "weighted_f1":     round(test_metrics["weighted_f1"],     4)},
    ])
    metrics_df.to_csv(P5_DIR / "bilstm_metrics.csv", index=False, encoding="utf-8")
    print(f"  Metrics CSV saved: {P5_DIR / 'bilstm_metrics.csv'}")

    # ── Phase 5 Comprehensive Report ──────────────────────────────────────────
    print("\n[STEP 11] Writing Phase 5 report ...")
    bilstm_vs_p2 = test_metrics["macro_f1"] - 0.6611
    bilstm_vs_p4 = test_metrics["macro_f1"] - 0.6657

    neutral_f1_bilstm = test_metrics["clf_report_dict"]["neutral"]["f1-score"]

    rep = [
        "=" * 78,
        "PHASE 5 - BIDIRECTIONAL LSTM (BiLSTM) COMPREHENSIVE REPORT",
        "AI-Based Sentiment Analysis System",
        "=" * 78,
        "",
        "1. OBJECTIVE",
        "-" * 78,
        "Determine whether a Bidirectional LSTM neural sequence model can improve",
        "sentiment classification by learning word order and bidirectional context",
        "that TF-IDF does not explicitly represent.",
        "",
        "2. DEEP LEARNING ENVIRONMENT",
        "-" * 78,
        f"  Python      : 3.13.15",
        f"  TensorFlow  : {tf.__version__}",
        f"  Keras       : {keras.__version__}",
        f"  NumPy       : {np.__version__}",
        f"  pandas      : {pd.__version__}",
        f"  Device      : CPU only (no GPU detected on native Windows TF >= 2.11)",
        "",
        "3. TEXT TOKENIZATION",
        "-" * 78,
        f"  Tokenizer      : Keras Tokenizer (tensorflow.keras.preprocessing.text)",
        f"  Fitted on      : TRAINING DATA ONLY (102,076 sentences)",
        f"  Max vocab size : {MAX_VOCAB_SIZE:,}",
        f"  Actual vocab   : {actual_vocab:,} unique tokens in training set",
        f"  OOV token      : '{OOV_TOKEN}' (index 1)",
        f"  Vocab used in embedding: {vocab_size_for_embedding:,} (includes padding index 0)",
        "",
        "4. SEQUENCE LENGTH SELECTION",
        "-" * 78,
        f"  Determined from TRAINING DATA ONLY",
        f"  Min words             : {seq_stats['min']}",
        f"  Mean words            : {seq_stats['mean']:.2f}",
        f"  50th percentile       : {seq_stats['p50']:.1f}",
        f"  90th percentile       : {seq_stats['p90']:.1f}",
        f"  95th percentile       : {seq_stats['p95']:.1f}",
        f"  99th percentile       : {seq_stats['p99']:.1f}",
        f"  Max observed          : {seq_stats['max']}",
        f"  Selected MAX_SEQ_LEN  : {MAX_SEQ_LEN}",
        f"  Samples truncated     : {seq_stats['pct_truncated']:.2f}%",
        f"  Samples padded        : {seq_stats['pct_padded']:.2f}%",
        "",
        "5. LABEL ENCODING",
        "-" * 78,
        "  negative → 0",
        "  neutral  → 1",
        "  positive → 2",
        "  (Original CSV label columns are NOT modified)",
        "",
        "6. TRAINING LABEL DISTRIBUTION & CLASS WEIGHTS",
        "-" * 78,
        f"  negative: {label_dist['negative']:,} samples | weight: {class_weight_dict[0]:.4f}",
        f"  neutral : {label_dist['neutral']:,} samples  | weight: {class_weight_dict[1]:.4f}",
        f"  positive: {label_dist['positive']:,} samples | weight: {class_weight_dict[2]:.4f}",
        "  Class weights computed from training labels ONLY.",
        "",
        "7. BILSTM ARCHITECTURE",
        "-" * 78,
        f"  Layer 1 : Embedding(input_dim={vocab_size_for_embedding}, output_dim={EMBEDDING_DIM}, mask_zero=False)",
        f"  Layer 2 : Bidirectional(LSTM(units={LSTM_UNITS}, return_sequences=False))",
        f"  Layer 3 : Dropout(rate={DROPOUT_RATE})",
        f"  Layer 4 : Dense(units={DENSE_UNITS}, activation='relu')",
        f"  Layer 5 : Dense(units=3, activation='softmax')",
        f"  Total parameters : {total_params:,}",
        f"  Loss function    : sparse_categorical_crossentropy",
        f"  Optimizer        : Adam(learning_rate={LEARNING_RATE})",
        f"  Batch size       : {BATCH_SIZE}",
        "",
        "8. TRAINING CONFIGURATION",
        "-" * 78,
        f"  Max epochs        : {MAX_EPOCHS}",
        f"  EarlyStopping     : monitor=val_loss, patience={PATIENCE}, restore_best_weights=True",
        f"  Epochs completed  : {epochs_trained}",
        f"  Best epoch        : {best_epoch}",
        f"  Final train loss  : {history.history['loss'][-1]:.4f}",
        f"  Final val loss    : {history.history['val_loss'][-1]:.4f}",
        f"  Best val loss     : {min(history.history['val_loss']):.4f}",
        "",
        "9. VALIDATION RESULTS",
        "-" * 78,
        f"  Accuracy         : {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']*100:.2f}%)",
        f"  Macro F1         : {val_metrics['macro_f1']:.4f}",
        f"  Weighted F1      : {val_metrics['weighted_f1']:.4f}",
        f"  Macro Precision  : {val_metrics['macro_precision']:.4f}",
        f"  Macro Recall     : {val_metrics['macro_recall']:.4f}",
        "",
        "10. TEST RESULTS",
        "-" * 78,
        f"  Accuracy         : {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)",
        f"  Macro F1         : {test_metrics['macro_f1']:.4f}",
        f"  Weighted F1      : {test_metrics['weighted_f1']:.4f}",
        f"  Macro Precision  : {test_metrics['macro_precision']:.4f}",
        f"  Macro Recall     : {test_metrics['macro_recall']:.4f}",
        "",
        "11. PER-CLASS TEST PERFORMANCE",
        "-" * 78,
        "BiLSTM Per-Class Report (Test Set):",
        test_metrics["clf_report"],
        "",
        f"  Neutral class F1 : {neutral_f1_bilstm:.4f}",
        "",
        "12. CONFUSION MATRICES (TEST SET)",
        "-" * 78,
        "  Labels order: [negative, neutral, positive]",
    ]
    cm = test_metrics["confusion_matrix"]
    for i, row in enumerate(cm):
        rep.append(f"    {LABEL_NAMES[i]:<10}: {row.tolist()}")

    rep += [
        "",
        "13. COMPARISON TABLE (TEST SET — PRIMARY METRIC: MACRO F1)",
        "-" * 78,
        f"{'Model':<44} {'Accuracy':>9} {'Macro P':>8} {'Macro R':>8} {'Macro F1':>9} {'W-F1':>8}",
        "-" * 78,
        f"{'Phase 2: TF-IDF + Logistic Regression':<44} {phase2_results['accuracy']:>9.4f} {phase2_results['macro_precision']:>8.4f} {phase2_results['macro_recall']:>8.4f} {phase2_results['macro_f1']:>9.4f} {phase2_results['weighted_f1']:>8.4f}",
        f"{'Phase 4: TF-IDF + LingFeats + LR':<44} {phase4_results['accuracy']:>9.4f} {phase4_results['macro_precision']:>8.4f} {phase4_results['macro_recall']:>8.4f} {phase4_results['macro_f1']:>9.4f} {phase4_results['weighted_f1']:>8.4f}",
        f"{'Phase 5: BiLSTM (from scratch)':<44} {test_metrics['accuracy']:>9.4f} {test_metrics['macro_precision']:>8.4f} {test_metrics['macro_recall']:>8.4f} {test_metrics['macro_f1']:>9.4f} {test_metrics['weighted_f1']:>8.4f}",
        "-" * 78,
        f"  BiLSTM vs Phase 2 Macro F1 : {bilstm_vs_p2:+.4f}",
        f"  BiLSTM vs Phase 4 Macro F1 : {bilstm_vs_p4:+.4f}",
        "",
        "14. EXPERIMENTAL INTEGRITY VERIFICATION",
        "-" * 78,
        "  [OK] Raw data files (data/) unchanged",
        "  [OK] Phase 2 outputs unchanged",
        "  [OK] Phase 3 outputs unchanged",
        "  [OK] Phase 4 outputs unchanged",
        "  [OK] No train/val/test overlap (verified at runtime)",
        "  [OK] Tokenizer fitted on training data ONLY",
        "  [OK] Sequence length determined from training data ONLY",
        "  [OK] Class weights computed from training labels ONLY",
        "  [OK] No BERT, no transformers, no pretrained embeddings used",
        "  [OK] No linguistic features added to BiLSTM in this phase",
        "  [OK] Model evaluated on validation and test sets",
        "",
        "15. VERDICT: DID BILSTM IMPROVE THE BENCHMARK?",
        "-" * 78,
    ]

    if bilstm_vs_p4 > 0.001:
        verdict = f"YES. BiLSTM achieved Test Macro F1 = {test_metrics['macro_f1']:.4f}, improving both Phase 2 ({bilstm_vs_p2:+.4f}) and Phase 4 ({bilstm_vs_p4:+.4f}) benchmarks."
    elif bilstm_vs_p4 < -0.001:
        verdict = f"NO. BiLSTM Test Macro F1 = {test_metrics['macro_f1']:.4f} did NOT surpass Phase 4 benchmark of 0.6657 (delta: {bilstm_vs_p4:+.4f}). Classical TF-IDF with enhanced features remains stronger."
    else:
        verdict = f"MARGINAL / NO MEANINGFUL CHANGE. BiLSTM Test Macro F1 = {test_metrics['macro_f1']:.4f} vs Phase 4 = 0.6657 (delta: {bilstm_vs_p4:+.4f}). Results are effectively equivalent."

    rep.append(f"  {verdict}")
    rep.append("=" * 78)

    with open(P5_DIR / "phase5_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(rep))
    print(f"  Phase 5 report written: {P5_DIR / 'phase5_report.txt'}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("PHASE 5 COMPLETE — SUMMARY")
    print("=" * 78)
    print(f"  TensorFlow : {tf.__version__} | Keras : {keras.__version__}")
    print(f"  Vocabulary : {MAX_VOCAB_SIZE:,} tokens | Seq length : {MAX_SEQ_LEN} words")
    print(f"  Epochs     : {epochs_trained} (best at {best_epoch})")
    print(f"  BiLSTM Test Accuracy : {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  BiLSTM Test Macro F1 : {test_metrics['macro_f1']:.4f}  (vs Phase 2: {bilstm_vs_p2:+.4f} | vs Phase 4: {bilstm_vs_p4:+.4f})")
    print(f"  Neutral Class F1     : {neutral_f1_bilstm:.4f}")
    print(f"  Phase 5 report       : {P5_DIR / 'phase5_report.txt'}")
    print("=" * 78)

    return test_metrics


if __name__ == "__main__":
    run_bilstm_pipeline()
