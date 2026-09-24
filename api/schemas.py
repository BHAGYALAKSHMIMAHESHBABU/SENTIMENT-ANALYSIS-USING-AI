"""
Pydantic request and response schemas for the Sentiment Analysis REST API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class SentimentProbabilities(BaseModel):
    negative: float = Field(..., ge=0.0, le=1.0, description="Probability of negative sentiment")
    neutral: float = Field(..., ge=0.0, le=1.0, description="Probability of neutral sentiment")
    positive: float = Field(..., ge=0.0, le=1.0, description="Probability of positive sentiment")


class HealthResponse(BaseModel):
    status: str = Field("ok", description="API operational status")
    model: str = Field(..., description="Name of the underlying prediction model")
    version: str = Field(..., description="API version")


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Raw input sentence or review text to analyze",
        examples=["The product quality is excellent."]
    )

    @field_validator("text")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Input text cannot be empty or only whitespace.")
        return trimmed


class PredictResponse(BaseModel):
    sentiment: str = Field(..., description="Predicted sentiment class (positive, neutral, negative)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the winning class")
    probabilities: SentimentProbabilities = Field(..., description="Class probability distribution")
    input_text: Optional[str] = Field(None, description="Echo of input text analyzed")


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of sentences or reviews to analyze",
        examples=[[
            "The product quality is excellent.",
            "This item is terrible and broke immediately.",
            "The package arrived today."
        ]]
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("The 'texts' list cannot be empty.")
        validated = []
        for i, item in enumerate(v):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"Text item at index {i} cannot be empty or only whitespace.")
            validated.append(item.strip())
        return validated


class BatchPredictResponse(BaseModel):
    total: int = Field(..., description="Total number of items processed")
    predictions: List[PredictResponse] = Field(..., description="Predictions for each input text")
