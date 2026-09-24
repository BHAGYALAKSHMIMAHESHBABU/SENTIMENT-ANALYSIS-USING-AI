"""
tests/test_api.py
Automated tests for the AI Sentiment Analysis FastAPI backend.
Run: pytest tests/test_api.py -v
Server must be running: uvicorn api.main:app --reload
"""

import pytest
import requests

BASE_URL = "http://127.0.0.1:8000"


# ─── Health check ──────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self):
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        assert resp.status_code == 200

    def test_health_schema(self):
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        data = resp.json()
        assert "status" in data
        assert data["status"] == "ok"
        assert "model" in data
        assert "version" in data

    def test_root_endpoint(self):
        resp = requests.get(f"{BASE_URL}/", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert "service" in data
        assert "endpoints" in data


# ─── Single prediction ─────────────────────────────────────────────────────────

class TestPredict:
    def test_positive_review(self):
        payload = {"text": "I absolutely loved this product! It is amazing."}
        resp = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert data["sentiment"] in ("positive", "neutral", "negative")
        assert 0.0 <= data["confidence"] <= 1.0
        probs = data["probabilities"]
        assert abs(probs["positive"] + probs["neutral"] + probs["negative"] - 1.0) < 0.01

    def test_negative_review(self):
        payload = {"text": "This product is terrible and completely broke after one day."}
        resp = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert data["sentiment"] in ("positive", "neutral", "negative")

    def test_neutral_review(self):
        payload = {"text": "The package arrived today."}
        resp = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert data["sentiment"] in ("positive", "neutral", "negative")

    def test_input_text_echoed(self):
        payload = {"text": "Great service."}
        resp = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
        data = resp.json()
        assert data.get("input_text") is not None

    def test_whitespace_only_rejected(self):
        payload = {"text": "   "}
        resp = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
        assert resp.status_code == 422

    def test_empty_string_rejected(self):
        payload = {"text": ""}
        resp = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
        assert resp.status_code == 422

    def test_missing_text_field_rejected(self):
        resp = requests.post(f"{BASE_URL}/predict", json={}, timeout=10)
        assert resp.status_code == 422

    def test_long_text_accepted(self):
        long_text = "This product is great. " * 200
        payload = {"text": long_text[:9000]}
        resp = requests.post(f"{BASE_URL}/predict", json=payload, timeout=15)
        assert resp.status_code == 200


# ─── Batch prediction ──────────────────────────────────────────────────────────

class TestBatchPredict:
    def test_batch_basic(self):
        payload = {
            "texts": [
                "I loved this product!",
                "Terrible quality, very disappointed.",
                "The item arrived on time."
            ]
        }
        resp = requests.post(f"{BASE_URL}/batch_predict", json=payload, timeout=15)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert len(data["predictions"]) == 3

    def test_batch_single_item(self):
        payload = {"texts": ["Works perfectly."]}
        resp = requests.post(f"{BASE_URL}/batch_predict", json=payload, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1

    def test_batch_all_sentiments_schema(self):
        payload = {"texts": ["Great!", "Terrible!", "Okay."]}
        resp = requests.post(f"{BASE_URL}/batch_predict", json=payload, timeout=15)
        data = resp.json()
        for pred in data["predictions"]:
            assert pred["sentiment"] in ("positive", "neutral", "negative")
            assert 0.0 <= pred["confidence"] <= 1.0

    def test_batch_empty_list_rejected(self):
        payload = {"texts": []}
        resp = requests.post(f"{BASE_URL}/batch_predict", json=payload, timeout=10)
        assert resp.status_code == 422

    def test_batch_empty_item_rejected(self):
        payload = {"texts": ["Good product", ""]}
        resp = requests.post(f"{BASE_URL}/batch_predict", json=payload, timeout=10)
        assert resp.status_code == 422


# ─── CORS headers ──────────────────────────────────────────────────────────────

class TestCORS:
    def test_cors_preflight(self):
        resp = requests.options(
            f"{BASE_URL}/predict",
            headers={
                "Origin": "chrome-extension://test",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
            timeout=5,
        )
        # Should not block the preflight
        assert resp.status_code in (200, 204)
