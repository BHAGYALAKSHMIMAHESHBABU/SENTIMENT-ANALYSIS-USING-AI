"""
Phase 7 — DistilBERT Feasibility Test
======================================
Objective : Verify CPU-trainable keras-hub DistilBERT on ~2,500 samples.
            Measure throughput and extrapolate to full dataset.

Model source: HuggingFace distilbert/distilbert-base-uncased
              Weights loaded via safetensors (no Kaggle credentials needed).

Constraints (fixed by experiment design):
  - Dataset       : train_clean.csv  (stratified 2,500 samples)
  - Validation    : val_clean.csv    (full, unchanged)
  - Test set      : NOT used
  - Backbone      : distil_bert_base_en_uncased (keras-hub architecture)
  - Weights       : distilbert/distilbert-base-uncased from HuggingFace
  - Backend       : TensorFlow / Keras
  - Max seq len   : 128
  - Batch size    : 16
  - Learning rate : 2e-5
  - Epochs        : 1
  - Freeze/unfreeze : NOT applied (all weights trainable)
  - Linguistic features : NOT used
  - Random seed   : 42
"""

import os
import time
import random

# ── Reproducibility & env ─────────────────────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"]        = str(SEED)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"
os.environ["KERAS_BACKEND"]         = "tensorflow"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import keras_hub
from huggingface_hub import hf_hub_download
from safetensors import safe_open

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Constants ─────────────────────────────────────────────────────────────────
N_FEASIBILITY    = 2500
MAX_SEQ_LEN      = 128
BATCH_SIZE       = 16
LEARNING_RATE    = 2e-5
EPOCHS           = 1
HF_MODEL_ID      = "distilbert/distilbert-base-uncased"
LABEL_MAP        = {"negative": 0, "neutral": 1, "positive": 2}
NUM_CLASSES      = 3
FULL_TRAIN_SIZE  = 102_076

# DistilBERT-base-uncased architecture hyperparameters
DISTILBERT_VOCAB_SIZE      = 30522
DISTILBERT_NUM_LAYERS      = 6
DISTILBERT_NUM_HEADS       = 12
DISTILBERT_HIDDEN_DIM      = 768
DISTILBERT_INTERMEDIATE_DIM = 3072
DISTILBERT_DROPOUT         = 0.1
DISTILBERT_MAX_SEQ_LEN     = 512

TRAIN_PATH = os.path.join("outputs", "processed", "train_clean.csv")
VAL_PATH   = os.path.join("outputs", "processed", "val_clean.csv")
OUTPUT_DIR = os.path.join("outputs", "phase7")
os.makedirs(OUTPUT_DIR, exist_ok=True)
REPORT_PATH = os.path.join(OUTPUT_DIR, "phase7_feasibility_report.txt")

# ── RAM helper ────────────────────────────────────────────────────────────────
def get_ram_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except ImportError:
        return None

def fmt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"

# ── Step 1: Data loading ──────────────────────────────────────────────────────
print("=" * 70)
print("Phase 7 — DistilBERT CPU Feasibility Test")
print("=" * 70)
print(f"\n[1/7] Loading data ...")

train_df_full = pd.read_csv(TRAIN_PATH)
val_df        = pd.read_csv(VAL_PATH)

# Stratified sample per class (avoids pandas groupby key-drop bug)
_label_counts = train_df_full["label"].value_counts()
_parts = []
for _lbl, _cnt in _label_counts.items():
    _n = max(1, int(round(N_FEASIBILITY * _cnt / len(train_df_full))))
    _parts.append(
        train_df_full[train_df_full["label"] == _lbl]
        .sample(n=min(_n, _cnt), random_state=SEED)
    )

train_df = (
    pd.concat(_parts, ignore_index=True)
    .sample(frac=1, random_state=SEED)
    .reset_index(drop=True)
    .iloc[:N_FEASIBILITY]
    .reset_index(drop=True)
)

print(f"    Training samples (feasibility): {len(train_df)}")
print(f"    Validation samples            : {len(val_df)}")
print(f"    Training label distribution   : {train_df['label'].value_counts().to_dict()}")
print(f"    Validation label distribution : {val_df['label'].value_counts().to_dict()}")

train_texts  = train_df["sentence"].astype(str).tolist()
train_labels = train_df["label"].map(LABEL_MAP).values.astype(np.int32)
val_texts    = val_df["sentence"].astype(str).tolist()
val_labels   = val_df["label"].map(LABEL_MAP).values.astype(np.int32)

# ── Step 2: Download / locate HF files ───────────────────────────────────────
print(f"\n[2/7] Fetching DistilBERT files from HuggingFace ({HF_MODEL_ID}) ...")
ram_before_load = get_ram_mb()
load_start = time.time()

vocab_path      = hf_hub_download(repo_id=HF_MODEL_ID, filename="vocab.txt")
weights_path    = hf_hub_download(repo_id=HF_MODEL_ID, filename="model.safetensors")

print(f"    Vocab  : {vocab_path}")
print(f"    Weights: {weights_path}")

# ── Step 3: Build tokenizer ───────────────────────────────────────────────────
print(f"\n[3/7] Building WordPiece tokenizer (max_seq_len={MAX_SEQ_LEN}) ...")
tokenizer = keras_hub.models.BertTokenizer(vocabulary=vocab_path)

preprocessor = keras_hub.models.BertTextClassifierPreprocessor(
    tokenizer=tokenizer,
    sequence_length=MAX_SEQ_LEN,
)

# ── Step 4: Build backbone + classifier ──────────────────────────────────────
print(f"\n[4/7] Building DistilBERT backbone and loading pretrained weights ...")

backbone = keras_hub.models.DistilBertBackbone(
    vocabulary_size=DISTILBERT_VOCAB_SIZE,
    num_layers=DISTILBERT_NUM_LAYERS,
    num_heads=DISTILBERT_NUM_HEADS,
    hidden_dim=DISTILBERT_HIDDEN_DIM,
    intermediate_dim=DISTILBERT_INTERMEDIATE_DIM,
    dropout=DISTILBERT_DROPOUT,
    max_sequence_length=DISTILBERT_MAX_SEQ_LEN,
)

# ── Weight mapping: HF safetensors → keras-hub layer names ───────────────────
# Load the safetensors file and map weights to the backbone
print("    Loading pretrained weights from safetensors ...")

hf_to_keras = {}
with safe_open(weights_path, framework="np") as f:
    hf_keys = list(f.keys())
    hf_tensors = {k: f.get_tensor(k) for k in hf_keys}

print(f"    HF weight keys ({len(hf_keys)} total): {hf_keys[:6]} ...")

# Build a mapping from HF key → keras variable
# DistilBERT HF name pattern: distilbert.embeddings.*, distilbert.transformer.layer.N.*
keras_var_map = {v.name: v for v in backbone.variables}
keras_var_names = list(keras_var_map.keys())

def map_hf_to_keras(hf_key):
    """Map a HuggingFace DistilBERT weight name to a keras-hub variable name."""
    k = hf_key
    # Remove top-level prefix
    k = k.replace("distilbert.", "")

    # Embedding layer
    k = k.replace("embeddings.word_embeddings.weight",
                   "token_and_position_embedding/token_embedding/embeddings")
    k = k.replace("embeddings.position_embeddings.weight",
                   "token_and_position_embedding/position_embedding/embeddings")
    k = k.replace("embeddings.LayerNorm.weight", "embeddings_layer_norm/gamma")
    k = k.replace("embeddings.LayerNorm.bias",   "embeddings_layer_norm/beta")

    # Transformer layers
    import re
    def replace_layer(m):
        n = m.group(1)
        rest = m.group(2)
        rest = rest.replace("attention.q_lin.weight", f"transformer_layer_{n}/multi_head_attention/query/kernel")
        rest = rest.replace("attention.q_lin.bias",   f"transformer_layer_{n}/multi_head_attention/query/bias")
        rest = rest.replace("attention.k_lin.weight", f"transformer_layer_{n}/multi_head_attention/key/kernel")
        rest = rest.replace("attention.k_lin.bias",   f"transformer_layer_{n}/multi_head_attention/key/bias")
        rest = rest.replace("attention.v_lin.weight", f"transformer_layer_{n}/multi_head_attention/value/kernel")
        rest = rest.replace("attention.v_lin.bias",   f"transformer_layer_{n}/multi_head_attention/value/bias")
        rest = rest.replace("attention.out_lin.weight", f"transformer_layer_{n}/multi_head_attention/attention_output/kernel")
        rest = rest.replace("attention.out_lin.bias",   f"transformer_layer_{n}/multi_head_attention/attention_output/bias")
        rest = rest.replace("sa_layer_norm.weight", f"transformer_layer_{n}/layer_norm_1/gamma")
        rest = rest.replace("sa_layer_norm.bias",   f"transformer_layer_{n}/layer_norm_1/beta")
        rest = rest.replace("ffn.lin1.weight", f"transformer_layer_{n}/feedforward_intermediate_dense/kernel")
        rest = rest.replace("ffn.lin1.bias",   f"transformer_layer_{n}/feedforward_intermediate_dense/bias")
        rest = rest.replace("ffn.lin2.weight", f"transformer_layer_{n}/feedforward_output_dense/kernel")
        rest = rest.replace("ffn.lin2.bias",   f"transformer_layer_{n}/feedforward_output_dense/bias")
        rest = rest.replace("output_layer_norm.weight", f"transformer_layer_{n}/layer_norm_2/gamma")
        rest = rest.replace("output_layer_norm.bias",   f"transformer_layer_{n}/layer_norm_2/beta")
        return rest
    k = re.sub(r"transformer\.layer\.(\d+)\.(.*)", replace_layer, k)
    return k

# Apply weights
matched = 0
skipped = []
for hf_key, tensor in hf_tensors.items():
    keras_name = map_hf_to_keras(hf_key)
    # Search for matching keras variable
    matched_var = None
    for var_name, var in keras_var_map.items():
        # Match on the suffix of the variable name
        if keras_name in var_name or var_name.endswith(keras_name):
            matched_var = var
            break
    if matched_var is not None:
        # Transpose Dense kernels: HF stores [out, in], Keras expects [in, out]
        if "kernel" in keras_name and tensor.ndim == 2:
            tensor = tensor.T
        # Reshape attention kernels: [heads*head_dim, hidden] → handled by transposing
        matched_var.assign(tensor)
        matched += 1
    else:
        skipped.append(hf_key)

print(f"    Weights matched and loaded: {matched}/{len(hf_keys)}")
if skipped:
    print(f"    Skipped (not matched): {skipped}")

load_elapsed = time.time() - load_start
ram_after_load = get_ram_mb()
print(f"    Model + weights loaded in {load_elapsed:.1f}s")

# ── Step 5: Build classifier on top of backbone ───────────────────────────────
print(f"\n[5/7] Building classifier head ...")

# Use BertTextClassifier which works with DistilBert backbone (same interface)
classifier = keras_hub.models.BertTextClassifier(
    backbone=backbone,
    num_classes=NUM_CLASSES,
    preprocessor=preprocessor,
)

total_params     = classifier.count_params()
trainable_params = sum(np.prod(v.shape) for v in classifier.trainable_weights)

print(f"    Total parameters    : {total_params:,}")
print(f"    Trainable parameters: {trainable_params:,}")
print(f"    Max sequence length : {MAX_SEQ_LEN}")
print(f"    Batch size          : {BATCH_SIZE}")
print(f"    Learning rate       : {LEARNING_RATE}")
print(f"    Epochs              : {EPOCHS}")
print(f"    Freeze/unfreeze     : NOT applied")

classifier.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# ── Step 6: Build tf.data pipelines ──────────────────────────────────────────
print(f"\n[6/7] Building tf.data pipelines ...")

train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_texts, train_labels))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
val_dataset = (
    tf.data.Dataset.from_tensor_slices((val_texts, val_labels))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

train_steps = -(-len(train_texts) // BATCH_SIZE)   # ceiling division
val_steps   = -(-len(val_texts)   // BATCH_SIZE)

print(f"    Train batches : {train_steps}")
print(f"    Val batches   : {val_steps}")

# ── Step 7: Train ─────────────────────────────────────────────────────────────
print(f"\n[7/7] Training for {EPOCHS} epoch on {len(train_texts)} samples ...")
ram_before_train = get_ram_mb()
train_start = time.time()

history = classifier.fit(
    train_dataset,
    epochs=EPOCHS,
    verbose=1,
)

train_elapsed    = time.time() - train_start
ram_after_train  = get_ram_mb()
train_loss       = history.history["loss"][-1]
train_accuracy   = history.history["accuracy"][-1]
throughput       = len(train_texts) / train_elapsed

print(f"\n    Completed in {train_elapsed:.1f}s | {throughput:.2f} samples/sec")

# ── Validation ────────────────────────────────────────────────────────────────
print(f"\nRunning validation on {len(val_texts)} samples ...")
val_start  = time.time()
val_logits = classifier.predict(val_dataset, verbose=1)
val_elapsed = time.time() - val_start

val_preds = np.argmax(val_logits, axis=1)

from sklearn.metrics import accuracy_score, f1_score, classification_report

val_accuracy = accuracy_score(val_labels, val_preds)
val_macro_f1 = f1_score(val_labels, val_preds, average="macro")
val_report   = classification_report(
    val_labels, val_preds,
    target_names=["negative", "neutral", "positive"],
    digits=4,
)

# ── Time extrapolation ────────────────────────────────────────────────────────
full_1ep_sec = FULL_TRAIN_SIZE / throughput
full_3ep_sec = full_1ep_sec * 3

# ── Report ────────────────────────────────────────────────────────────────────
lines = []
lines.append("=" * 70)
lines.append("PHASE 7 — DISTILBERT CPU FEASIBILITY REPORT")
lines.append("=" * 70)
lines.append("")
lines.append("EXPERIMENT CONFIGURATION")
lines.append("-" * 70)
lines.append(f"  Model source        : HuggingFace {HF_MODEL_ID}")
lines.append(f"  Architecture        : distil_bert_base_en_uncased (keras-hub)")
lines.append(f"  Backend             : TensorFlow {tf.__version__} / Keras {keras.__version__}")
lines.append(f"  keras-hub version   : {keras_hub.__version__}")
lines.append(f"  Max sequence length : {MAX_SEQ_LEN}")
lines.append(f"  Batch size          : {BATCH_SIZE}")
lines.append(f"  Learning rate       : {LEARNING_RATE}")
lines.append(f"  Epochs run          : {EPOCHS}")
lines.append(f"  Freeze / unfreeze   : NOT applied (all weights trainable)")
lines.append(f"  Linguistic features : NOT used")
lines.append(f"  Random seed         : {SEED}")
lines.append(f"  Training samples    : {len(train_df)}")
lines.append(f"  Validation samples  : {len(val_df)}")
lines.append(f"  Test set used       : NO")
lines.append(f"  Weights matched     : {matched}/{len(hf_keys)}")
lines.append("")
lines.append("MODEL SIZE")
lines.append("-" * 70)
lines.append(f"  Total parameters    : {total_params:,}")
lines.append(f"  Trainable params    : {trainable_params:,}")
lines.append("")
lines.append("TIMING & THROUGHPUT")
lines.append("-" * 70)
lines.append(f"  Model load time     : {load_elapsed:.1f}s")
lines.append(f"  Training time       : {train_elapsed:.1f}s  ({fmt_time(train_elapsed)})")
lines.append(f"  Validation time     : {val_elapsed:.1f}s  ({fmt_time(val_elapsed)})")
lines.append(f"  Throughput          : {throughput:.2f} samples/sec")
lines.append(f"  CPU training status : COMPLETED SUCCESSFULLY")
lines.append("")
lines.append("RAM USAGE (approximate)")
lines.append("-" * 70)
if ram_before_load is not None:
    lines.append(f"  Before model load   : {ram_before_load:.0f} MB")
    lines.append(f"  After model load    : {ram_after_load:.0f} MB")
    lines.append(f"  Model footprint     : +{ram_after_load - ram_before_load:.0f} MB")
    if ram_after_train is not None:
        lines.append(f"  After training      : {ram_after_train:.0f} MB")
else:
    lines.append("  psutil not available — RAM not measured")
lines.append("")
lines.append("TRAINING RESULTS (2,500-sample feasibility — NOT comparable to Phase 4)")
lines.append("-" * 70)
lines.append(f"  Training Loss       : {train_loss:.4f}")
lines.append(f"  Training Accuracy   : {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
lines.append("")
lines.append("VALIDATION RESULTS (full val set — feasibility indicator only)")
lines.append("-" * 70)
lines.append(f"  Validation Accuracy : {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
lines.append(f"  Validation Macro F1 : {val_macro_f1:.4f}")
lines.append("")
lines.append("Per-class breakdown (validation):")
lines.append(val_report)
lines.append("")
lines.append("FULL-DATASET TIME EXTRAPOLATION")
lines.append("-" * 70)
lines.append(f"  Full training set size : {FULL_TRAIN_SIZE:,} samples")
lines.append(f"  Measured throughput    : {throughput:.2f} samples/sec")
lines.append(f"  Estimated 1 epoch      : {fmt_time(full_1ep_sec)}  ({full_1ep_sec:.0f}s)")
lines.append(f"  Estimated 3 epochs     : {fmt_time(full_3ep_sec)}  ({full_3ep_sec:.0f}s)")
lines.append("")
lines.append("NOTE: Extrapolation assumes linear scaling; actual time may vary")
lines.append("      due to batch boundary effects and system load variation.")
lines.append("")
lines.append("PHASE 4 BENCHMARK (for reference — DO NOT compare directly)")
lines.append("-" * 70)
lines.append("  Test Accuracy : 66.69%  |  Test Macro F1 : 0.6657")
lines.append("  (Phase 4 used full training set; feasibility uses 2,500 samples)")
lines.append("")
lines.append("=" * 70)
lines.append("END OF FEASIBILITY REPORT")
lines.append("=" * 70)

report_text = "\n".join(lines)
print("\n" + report_text)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"\nReport saved to: {REPORT_PATH}")
