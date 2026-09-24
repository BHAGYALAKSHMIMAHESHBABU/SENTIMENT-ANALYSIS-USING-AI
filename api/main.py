"""
FastAPI REST API Server for the AI Sentiment Analysis Platform.
Exposes health check, single review prediction, and batch prediction endpoints.
Strictly inference only.
"""

from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    CORS_ORIGINS,
    MODEL_NAME,
)
from api.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
)
from api.service import get_prediction_service, BasePredictionService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eagerly load the model service on server startup."""
    try:
        service = get_prediction_service()
        if not service.is_ready():
            print("[WARNING] Prediction service initialized but not in ready state.")
        else:
            print(f"[INFO] Prediction service ready: {MODEL_NAME}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize prediction service: {e}")
    yield


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware to enable browser extension and frontend cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root() -> Dict[str, Any]:
    """Root endpoint providing service overview and documentation links."""
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "POST /predict",
            "batch_predict": "POST /batch_predict",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health() -> HealthResponse:
    """Health check endpoint returning operational status and active model."""
    try:
        service: BasePredictionService = get_prediction_service()
        if not service.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction model artifacts are not ready.",
            )
        return HealthResponse(
            status="ok",
            model=MODEL_NAME,
            version=API_VERSION,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(exc)}",
        )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(payload: PredictRequest) -> PredictResponse:
    """
    Generate sentiment prediction and probability distribution for a single text.
    Uses the Phase 4 production model (TF-IDF + 12 Linguistic Features + Logistic Regression).
    """
    try:
        service = get_prediction_service()
        result = service.predict(payload.text)
        return PredictResponse(**result)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}",
        )


@app.post("/batch_predict", response_model=BatchPredictResponse, tags=["Inference"])
async def batch_predict(payload: BatchPredictRequest) -> BatchPredictResponse:
    """
    Generate sentiment predictions for a batch of input texts.
    Vectorized for high throughput.
    """
    try:
        service = get_prediction_service()
        results = service.batch_predict(payload.texts)
        return BatchPredictResponse(
            total=len(results),
            predictions=[PredictResponse(**r) for r in results],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(exc)}",
        )


if __name__ == "__main__":
    import uvicorn
    from api.config import HOST, PORT
    uvicorn.run("api.main:app", host=HOST, port=PORT, reload=True)
