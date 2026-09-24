"""
Prediction Service Layer.
Abstracts model loading, text preprocessing, feature extraction, and inference.
Designed for extensibility so other AI models can be plugged in without changing the API contract.
"""

from abc import ABC, abstractmethod
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import scipy.sparse as sp

from api.config import MODELS_DIR, MODEL_NAME, API_VERSION, MODEL_MACRO_F1, MODEL_TRAINING_SAMPLES

# Ensure workspace root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Reuse existing preprocessing & feature extraction modules
from src.text_preprocessing import preprocess_for_tfidf
from src.linguistic_features import extract_features_from_text, FEATURE_NAMES


class BasePredictionService(ABC):
    """Abstract interface defining the prediction service contract."""

    @abstractmethod
    def predict(self, text: str) -> Dict[str, Any]:
        """Generate prediction for a single text input."""
        pass

    @abstractmethod
    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Generate predictions for a list of text inputs."""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True if model and preprocessors are loaded and ready."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata describing the active model."""
        pass


class Phase4SentimentService(BasePredictionService):
    """
    Concrete sentiment analysis service utilizing the Phase 4 production model:
    TF-IDF (123,493 features) + 12 Linguistic Features + Logistic Regression.
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.tfidf_path = self.models_dir / "tfidf_vectorizer.pkl"
        self.scaler_path = self.models_dir / "linguistic_feature_scaler.pkl"
        self.model_path = self.models_dir / "logistic_regression_linguistic.pkl"

        self.tfidf = None
        self.scaler = None
        self.model = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads saved artifacts from disk. Strictly read-only; no training."""
        missing = []
        for p, name in [
            (self.tfidf_path, "TF-IDF Vectorizer"),
            (self.scaler_path, "Linguistic Feature Scaler"),
            (self.model_path, "Logistic Regression Model"),
        ]:
            if not p.exists():
                missing.append(f"{name} ({p.name})")

        if missing:
            raise FileNotFoundError(
                f"Missing required model artifact(s): {', '.join(missing)} in {self.models_dir}"
            )

        with open(self.tfidf_path, "rb") as f:
            self.tfidf = pickle.load(f)
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

    def is_ready(self) -> bool:
        return (
            self.tfidf is not None
            and self.scaler is not None
            and self.model is not None
        )

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": MODEL_NAME,
            "version": API_VERSION,
            "classes": ["negative", "neutral", "positive"],
            "macro_f1": MODEL_MACRO_F1,
            "training_samples": MODEL_TRAINING_SAMPLES,
            "status": "ready" if self.is_ready() else "not_ready",
        }

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Executes the Phase 4 inference pipeline:
        1. Preprocess text for TF-IDF (lowercase + whitespace normalization).
        2. Transform with TF-IDF vectorizer.
        3. Extract 12 interpretable linguistic features from raw text.
        4. Scale linguistic features with training StandardScaler.
        5. Concatenate sparse TF-IDF and dense scaled features.
        6. Predict with Logistic Regression (inference only).
        """
        clean_text = preprocess_for_tfidf(text)
        tfidf_vec = self.tfidf.transform([clean_text])

        ling_dict = extract_features_from_text(text)
        ling_vec = np.array([[ling_dict[name] for name in FEATURE_NAMES]])
        scaled_ling = self.scaler.transform(ling_vec)

        X_comb = sp.hstack([tfidf_vec, sp.csr_matrix(scaled_ling)], format="csr")

        pred = self.model.predict(X_comb)[0]
        probas = self.model.predict_proba(X_comb)[0]

        classes = list(self.model.classes_)
        prob_dict = {c: float(p) for c, p in zip(classes, probas)}

        conf = float(np.max(probas))
        return {
            "sentiment": str(pred),
            "confidence": round(conf, 4),
            "probabilities": {
                "negative": round(prob_dict.get("negative", 0.0), 4),
                "neutral": round(prob_dict.get("neutral", 0.0), 4),
                "positive": round(prob_dict.get("positive", 0.0), 4),
            },
            "input_text": text,
        }

    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Executes vectorized batch inference for maximum throughput.
        """
        if not texts:
            return []

        clean_texts = [preprocess_for_tfidf(t) for t in texts]
        tfidf_mat = self.tfidf.transform(clean_texts)

        feat_rows = []
        for t in texts:
            f_dict = extract_features_from_text(t)
            feat_rows.append([f_dict[col] for col in FEATURE_NAMES])
        feat_mat = np.array(feat_rows)
        scaled_feat = self.scaler.transform(feat_mat)

        X_comb = sp.hstack([tfidf_mat, sp.csr_matrix(scaled_feat)], format="csr")
        preds = self.model.predict(X_comb)
        probas = self.model.predict_proba(X_comb)

        classes = list(self.model.classes_)
        neg_idx = classes.index("negative") if "negative" in classes else 0
        neu_idx = classes.index("neutral") if "neutral" in classes else 1
        pos_idx = classes.index("positive") if "positive" in classes else 2

        results = []
        for i, text in enumerate(texts):
            p = probas[i]
            conf = float(np.max(p))
            results.append({
                "sentiment": str(preds[i]),
                "confidence": round(conf, 4),
                "probabilities": {
                    "negative": round(float(p[neg_idx]), 4),
                    "neutral": round(float(p[neu_idx]), 4),
                    "positive": round(float(p[pos_idx]), 4),
                },
                "input_text": text,
            })
        return results


# Global singleton instance for easy dependency injection
_service_instance: BasePredictionService = None


def get_prediction_service() -> BasePredictionService:
    global _service_instance
    if _service_instance is None:
        _service_instance = Phase4SentimentService()
    return _service_instance
