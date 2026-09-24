# AI Sentiment Analysis — REST API Documentation

## Overview

The FastAPI backend exposes a sentiment analysis inference API backed by the
Phase 4 production model (TF-IDF + 12 Linguistic Features + Logistic Regression).

**Base URL:** `http://127.0.0.1:8000`
**Interactive Docs:** `http://127.0.0.1:8000/docs`

---

## Quick Start

```bash
# 1. Activate the virtual environment
.venv\Scripts\activate

# 2. Start the server
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

---

## Endpoints

### GET /health
Returns the operational status of the service.

**Response 200:**
```json
{
  "status": "ok",
  "model": "Phase4-TF-IDF-Linguistic-LR",
  "version": "1.0.0"
}
```

---

### POST /predict
Analyze sentiment for a single text input.

**Request body:**
```json
{
  "text": "I absolutely loved this product!"
}
```

**Response 200:**
```json
{
  "sentiment": "positive",
  "confidence": 0.912,
  "probabilities": {
    "negative": 0.031,
    "neutral": 0.057,
    "positive": 0.912
  },
  "input_text": "I absolutely loved this product!"
}
```

**Validation:**
- `text` must be a non-empty string (1–10,000 characters)
- Whitespace-only input returns HTTP 422

---

### POST /batch_predict
Analyze sentiment for a list of texts (up to 100 items).

**Request body:**
```json
{
  "texts": [
    "Great quality!",
    "Terrible experience.",
    "Package arrived on time."
  ]
}
```

**Response 200:**
```json
{
  "total": 3,
  "predictions": [
    { "sentiment": "positive", "confidence": 0.88, "probabilities": {...}, "input_text": "Great quality!" },
    { "sentiment": "negative", "confidence": 0.79, "probabilities": {...}, "input_text": "Terrible experience." },
    { "sentiment": "neutral",  "confidence": 0.65, "probabilities": {...}, "input_text": "Package arrived on time." }
  ]
}
```

---

## Error Codes

| Code | Meaning |
|------|---------|
| 200  | Success |
| 422  | Validation error (empty/missing text) |
| 500  | Internal inference error |
| 503  | Service unavailable (model not loaded) |

---

## Running Tests

```bash
# Start the API server first, then:
pytest tests/test_api.py -v
```

---

## Model Information

| Component | Details |
|-----------|---------|
| Algorithm | Logistic Regression |
| Features | TF-IDF (10,000 terms) + 12 Linguistic Features |
| Classes | positive / neutral / negative |
| Training set | 102,076 samples |
| Macro F1 | 0.6557 |
| Best Phase | Phase 4 (highest overall performance) |

---

## CORS

The API allows cross-origin requests from `localhost` and browser extensions.
For production, restrict `CORS_ORIGINS` in the `.env` file.
