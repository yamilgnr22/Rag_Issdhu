from collections.abc import Sequence

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from app.config import get_settings


class RerankRequest(BaseModel):
    query: str
    documents: list[str] = Field(default_factory=list)
    top_n: int = Field(default=10, ge=1)
    model: str | None = Field(default=None)


class RerankResult(BaseModel):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    model: str
    device: str
    results: list[RerankResult]


class HealthResponse(BaseModel):
    status: str
    device: str
    default_model: str
    cached_models: list[str]


class RerankerRegistry:
    def __init__(self, *, device: str, trust_remote_code: bool) -> None:
        self.device = device
        self.trust_remote_code = trust_remote_code
        self._models: dict[str, CrossEncoder] = {}

    def get(self, model_name: str) -> CrossEncoder:
        model = self._models.get(model_name)
        if model is None:
            model = CrossEncoder(
                model_name,
                device=self.device,
                trust_remote_code=self.trust_remote_code,
            )
            self._models[model_name] = model
        return model

    @property
    def cached_models(self) -> list[str]:
        return sorted(self._models.keys())


def _resolve_device(preference: str) -> str:
    if preference == "cuda":
        return "cuda"
    if preference == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _rank_scores(*, scores: Sequence[float], top_n: int) -> list[RerankResult]:
    indexed = sorted(
        enumerate(scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return [
        RerankResult(index=index, relevance_score=float(score))
        for index, score in indexed[:top_n]
    ]


settings = get_settings()
resolved_device = _resolve_device(settings.reranker_device)
registry = RerankerRegistry(
    device=resolved_device,
    trust_remote_code=settings.reranker_trust_remote_code,
)

app = FastAPI(
    title="Rag_Issdhu Reranker API",
    version="0.1.0",
    description="Local reranker microservice for retrieval reranking",
)


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=resolved_device,
        default_model=settings.reranker_default_model,
        cached_models=registry.cached_models,
    )


@app.post("/rerank", response_model=RerankResponse)
def rerank(request: RerankRequest) -> RerankResponse:
    if not request.documents:
        raise HTTPException(status_code=400, detail="documents must not be empty")
    if len(request.documents) > settings.reranker_max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"documents exceeds reranker_max_batch_size={settings.reranker_max_batch_size}",
        )

    model_name = request.model or settings.reranker_default_model
    try:
        model = registry.get(model_name)
        scores = model.predict(
            [(request.query, document) for document in request.documents],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"reranker_load_or_predict_failed: {exc}") from exc

    results = _rank_scores(scores=scores, top_n=min(request.top_n, len(request.documents)))
    return RerankResponse(
        model=model_name,
        device=resolved_device,
        results=results,
    )
