"""
Configuration settings for the AI Sentiment Analysis REST API.
Can be overridden using environment variables or a .env file.
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(BASE_DIR / "models")))

# API Metadata
API_TITLE = os.getenv("API_TITLE", "AI-Based Sentiment Analysis API")
API_DESCRIPTION = os.getenv(
    "API_DESCRIPTION",
    "Production REST API for three-class sentiment analysis using TF-IDF, "
    "12 interpretable linguistic features, and Logistic Regression (Phase 4 Benchmark)."
)
API_VERSION = os.getenv("API_VERSION", "1.0.0")

# Model Info
MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "Phase 4 TF-IDF + Linguistic Features + Logistic Regression"
)
MODEL_MACRO_F1 = 0.6657
MODEL_TRAINING_SAMPLES = 102076

# Operational Limits
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "10000"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "100"))

# Server Binding
HOST = os.getenv("API_HOST", "127.0.0.1")
PORT = int(os.getenv("API_PORT", "8000"))

# CORS Configuration (allows browser extension, web apps, and local testing)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
