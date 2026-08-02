import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import Base, engine, ensure_runtime_schema, get_db_session
from app.models.contracts import (
    ACL,
    ChunkContract,
    ChunkingRequest,
    ChunkingResponse,
    DocumentVersionListResponse,
    FeedbackRequest,
    IngestionFilesystemRequest,
    IngestionResponse,
    QueryRequest,
    QueryResponse,
    RetrievalEvalRequest,
    RetrievalEvalResponse,
    ReindexRequest,
    ReindexResponse,
    VersionDeleteResponse,
    VersionStatusUpdateRequest,
    VersionStatusUpdateResponse,
)
from app.services.chunking import ChunkingService
from app.services.ingestion import IngestionService
from app.services.indexing import IndexingService
from app.services.retrieval_evaluation import RetrievalEvaluationService
from app.services.retrieval import RetrievalService


settings = get_settings()

# Sin esta configuracion, los warning de degradacion silenciosa (rerank caido,
# embeddings en hash, indice de otro perfil) no llegan a ninguna parte.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    ensure_runtime_schema()
    status = _rerank_backend_status()
    if status["configured"] and not status["reachable"]:
        logger.error(
            "rerank_backend_unreachable mode=%s url=%s: el sistema respondera con el "
            "rerank heuristico (~0.20 peor en hit@1) hasta que el servicio vuelva",
            settings.rerank_mode,
            status.get("url"),
        )
    else:
        logger.info("rerank_backend mode=%s reachable=%s", settings.rerank_mode, status["reachable"])
    yield


def _rerank_backend_status() -> dict[str, object]:
    """Comprueba de verdad el backend de rerank en lugar de asumir que responde."""
    mode = settings.rerank_mode
    if mode == "off":
        return {"configured": False, "reachable": False, "reason": "rerank_mode_off"}
    if mode == "cohere":
        configured = bool(settings.rerank_cohere_api_key and settings.rerank_cohere_model)
        return {"configured": configured, "reachable": configured, "url": settings.rerank_cohere_base_url}

    base_url = (settings.rerank_local_base_url or "").rstrip("/")
    if not base_url or not settings.rerank_local_model:
        return {"configured": False, "reachable": False, "reason": "local_unconfigured"}
    try:
        with httpx.Client(timeout=3.0) as client:
            response = client.get(f"{base_url}/health")
        return {
            "configured": True,
            "reachable": response.status_code == 200,
            "url": base_url,
            "detail": response.json() if response.status_code == 200 else None,
        }
    except Exception as exc:
        return {"configured": True, "reachable": False, "url": base_url, "reason": type(exc).__name__}


app = FastAPI(
    title="Rag_Issdhu API",
    version="0.3.1",
    description="API bootstrap with domain-aware ingestion and Phase 2 chunking implemented",
    lifespan=lifespan,
)


def get_ingestion_service() -> IngestionService:
    return IngestionService(settings)


def get_chunking_service() -> ChunkingService:
    return ChunkingService(settings)


def get_indexing_service() -> IndexingService:
    return IndexingService(settings)


def get_retrieval_service() -> RetrievalService:
    return RetrievalService(settings)


def get_retrieval_evaluation_service() -> RetrievalEvaluationService:
    return RetrievalEvaluationService(settings)


@app.get("/health")
def healthcheck() -> dict[str, object]:
    return {
        "status": "ok",
        "profile": settings.app_profile,
        "database_url": settings.database_url,
        "storage_backend": settings.storage_backend,
        "services": {
            "postgres": settings.postgres_host,
            "minio": settings.minio_endpoint,
            "opensearch": settings.opensearch_url,
            "qdrant": settings.qdrant_url,
        },
        "retrieval_indexing": {
            "opensearch_index": settings.resolved_opensearch_index_name,
            "qdrant_collection": settings.resolved_qdrant_collection_name,
            "embedding_profile": settings.embedding_profile_slug,
            "rerank_mode": settings.rerank_mode,
            # Estado real, no la configuracion declarada: el modo puede decir
            # "local" mientras el servicio esta caido y cada consulta degrada.
            "rerank_backend": _rerank_backend_status(),
        },
    }


@app.post("/documents:ingest", response_model=IngestionResponse)
async def ingest_document(
    file: UploadFile = File(...),
    source: str = Form(default="upload"),
    source_authority: str = Form(default="primary"),
    language: str | None = Form(default=None),
    version_status: str = Form(default="active"),
    acl_users: str = Form(default=""),
    acl_groups: str = Form(default=""),
    db: Session = Depends(get_db_session),
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestionResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    acl = ACL(
        users=[value.strip() for value in acl_users.split(",") if value.strip()],
        groups=[value.strip() for value in acl_groups.split(",") if value.strip()],
    )
    return service.ingest_upload(
        db=db,
        filename=file.filename or "upload.bin",
        content=content,
        source=source,
        source_authority=source_authority,
        language=language,
        version_status=version_status,
        acl=acl,
    )


@app.post("/documents:ingest/filesystem", response_model=IngestionResponse)
def ingest_document_from_filesystem(
    request: IngestionFilesystemRequest,
    db: Session = Depends(get_db_session),
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestionResponse:
    try:
        return service.ingest_from_filesystem(db=db, request=request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/documents/manual-review")
def manual_review_queue(
    db: Session = Depends(get_db_session),
    service: IngestionService = Depends(get_ingestion_service),
) -> dict[str, object]:
    return {"items": service.list_manual_review(db)}


@app.get("/documents", response_model=DocumentVersionListResponse)
def list_documents(
    limit: int = 50,
    version_status: str | None = None,
    document_class: str | None = None,
    q: str | None = None,
    db: Session = Depends(get_db_session),
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentVersionListResponse:
    return service.list_versions(
        db,
        limit=limit,
        version_status=version_status,
        document_class=document_class,
        q=q,
    )


@app.post("/documents/{version_id}/status", response_model=VersionStatusUpdateResponse)
def update_document_version_status(
    version_id: str,
    request: VersionStatusUpdateRequest,
    db: Session = Depends(get_db_session),
    service: IngestionService = Depends(get_ingestion_service),
) -> VersionStatusUpdateResponse:
    try:
        return service.update_version_status(
            db=db,
            version_id=version_id,
            version_status=request.version_status,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/documents/{version_id}", response_model=VersionDeleteResponse)
def delete_document_version(
    version_id: str,
    db: Session = Depends(get_db_session),
    service: IngestionService = Depends(get_ingestion_service),
) -> VersionDeleteResponse:
    try:
        return service.delete_version(db=db, version_id=version_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/documents/{version_id}/chunk", response_model=ChunkingResponse)
def chunk_document_version(
    version_id: str,
    request: ChunkingRequest,
    db: Session = Depends(get_db_session),
    service: ChunkingService = Depends(get_chunking_service),
) -> ChunkingResponse:
    try:
        return service.chunk_version(db=db, version_id=version_id, request=request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/documents/{version_id}/chunks", response_model=list[ChunkContract])
def list_document_chunks(
    version_id: str,
    db: Session = Depends(get_db_session),
    service: ChunkingService = Depends(get_chunking_service),
) -> list[ChunkContract]:
    chunks = service.list_chunks(db=db, version_id=version_id)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"No chunks found for version: {version_id}")
    return chunks


@app.post("/documents/reindex", response_model=ReindexResponse)
def reindex_documents(
    request: ReindexRequest | None = None,
    db: Session = Depends(get_db_session),
    service: IndexingService = Depends(get_indexing_service),
) -> ReindexResponse:
    try:
        return service.reindex(db=db, request=request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    service: RetrievalService = Depends(get_retrieval_service),
) -> QueryResponse:
    try:
        return service.query(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/eval/retrieval", response_model=RetrievalEvalResponse)
def evaluate_retrieval(
    request: RetrievalEvalRequest | None = None,
    db: Session = Depends(get_db_session),
    service: RetrievalEvaluationService = Depends(get_retrieval_evaluation_service),
) -> RetrievalEvalResponse:
    try:
        return service.evaluate(db=db, request=request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/feedback")
def feedback_placeholder(request: FeedbackRequest) -> dict[str, object]:
    return {
        "status": "accepted",
        "message": "Feedback endpoint placeholder created.",
        "payload": request.model_dump(),
    }
