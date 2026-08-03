from functools import lru_cache
from pathlib import Path
import re
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_profile: Literal["dev-gpu", "prod-cpu"] = Field(default="dev-gpu")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    database_url: str = Field(default="sqlite:///./.data/catalog.db")
    storage_backend: Literal["localfs", "minio"] = Field(default="localfs")
    local_storage_path: str = Field(default=".data/object_store")
    ocr_enabled: bool = Field(default=False)
    ocr_language: str = Field(default="spa+eng")
    # Control de calidad de la extraccion. El unico criterio previo era "menos de
    # 50 caracteres", asi que 106.000 caracteres de OCR degradado entraron al
    # indice marcados como extraidos. Umbrales calibrados sobre este corpus: los
    # documentos nativos tienen 0,02-0,05% de tokens pegados y 9-14% de palabras
    # acentuadas; el Reglamento escaneado tenia 18,5% y 0,14%.
    extraction_quality_gate_enabled: bool = Field(default=True)
    extraction_max_glued_ratio: float = Field(default=0.02)
    extraction_min_accent_ratio: float = Field(default=0.05)
    extraction_quality_min_words: int = Field(default=200)
    # Integridad: proporcion entre lo extraido y lo que el PDF realmente
    # contiene, medido con pdfplumber sobre una muestra de paginas. Un texto
    # puede ser impecable y aun asi estar incompleto: NIIF para PYMES paso el
    # control de calidad con 0,0004 de tokens pegados mientras le faltaba el 34%
    # del documento, porque docling descarto paginas por falta de memoria.
    extraction_min_coverage_ratio: float = Field(default=0.8)
    # Conversion por lotes de paginas. docling acumula memoria a lo largo del
    # documento y a partir de cierto punto falla con std::bad_alloc, descartando
    # paginas en silencio: NIIF para PYMES (276 pag) perdio 107 en una corrida y
    # 3 en otra. Convertir por tramos acota el pico de memoria. 0 desactiva.
    extraction_batch_pages: int = Field(default=40)
    extraction_batch_threshold_pages: int = Field(default=60)

    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="rag_issdhu")
    postgres_user: str = Field(default="rag")
    postgres_password: str = Field(default="rag")

    minio_endpoint: str = Field(default="localhost:9000")
    minio_access_key: str = Field(default="minioadmin")
    minio_secret_key: str = Field(default="minioadmin")
    minio_bucket: str = Field(default="documents")
    minio_secure: bool = Field(default=False)
    chunk_target_tokens: int = Field(default=600)
    chunk_min_tokens: int = Field(default=400)
    chunk_max_tokens: int = Field(default=800)
    chunk_overlap_ratio: float = Field(default=0.2)
    classifier_mode: Literal["heuristic", "local_llm", "cloud_llm", "hybrid"] = Field(default="hybrid")
    classifier_deterministic_short_circuit: float = Field(default=0.93)
    classifier_llm_min_confidence: float = Field(default=0.6)
    classifier_llm_base_url: str | None = Field(default=None)
    classifier_llm_api_key: str | None = Field(default=None)
    classifier_llm_model: str | None = Field(default=None)
    classifier_llm_timeout: float = Field(default=20.0)
    classifier_cloud_base_url: str | None = Field(default=None)
    classifier_cloud_api_key: str | None = Field(default=None)
    classifier_cloud_model: str | None = Field(default=None)
    classifier_cloud_timeout: float | None = Field(default=None)
    classifier_local_base_url: str | None = Field(default=None)
    classifier_local_api_key: str | None = Field(default=None)
    classifier_local_model: str | None = Field(default=None)
    classifier_local_timeout: float | None = Field(default=None)
    query_planner_mode: Literal["local_llm", "cloud_llm", "hybrid"] = Field(default="cloud_llm")
    query_planner_min_confidence: float = Field(default=0.55)
    query_strategy_classifier_enabled: bool = Field(default=True)
    query_strategy_classifier_fixture_path: str = Field(
        default="apps/api/app/evals/retrieval_eval_strategy_train_cases.json"
    )
    query_strategy_classifier_min_confidence: float = Field(default=0.5)
    query_router_classifier_enabled: bool = Field(default=True)
    query_router_classifier_fixture_path: str = Field(
        default="apps/api/app/evals/retrieval_eval_cases.json"
    )
    query_router_classifier_min_confidence: float = Field(default=0.45)
    # Por encima de este umbral la clase predicha filtra de forma excluyente; por
    # debajo se usa solo como preferencia, para que un error de routing no deje
    # fuera de la busqueda al documento correcto.
    query_router_hard_filter_confidence: float = Field(default=0.95)
    branch_centrality_scorer_enabled: bool = Field(default=True)
    branch_centrality_fixture_path: str = Field(
        default="apps/api/app/evals/answer_form_train_seed_cases.json"
    )
    branch_centrality_positive_threshold: float = Field(default=0.5)
    evidence_role_classifier_enabled: bool = Field(default=True)
    evidence_role_classifier_fixture_path: str = Field(
        default="apps/api/app/evals/evidence_role_train_cases.json"
    )
    query_evidence_intent_classifier_enabled: bool = Field(default=True)
    query_evidence_intent_classifier_fixture_path: str = Field(
        default="apps/api/app/evals/query_evidence_intent_train_cases.json"
    )
    query_evidence_intent_classifier_min_confidence: float = Field(default=0.55)
    query_planner_cloud_base_url: str | None = Field(default=None)
    query_planner_cloud_api_key: str | None = Field(default=None)
    query_planner_cloud_model: str | None = Field(default="gpt-5-mini")
    query_planner_cloud_timeout: float | None = Field(default=20.0)
    query_planner_local_base_url: str | None = Field(default=None)
    query_planner_local_api_key: str | None = Field(default=None)
    query_planner_local_model: str | None = Field(default=None)
    query_planner_local_timeout: float | None = Field(default=20.0)
    answer_synthesis_mode: Literal["off", "local_llm", "cloud_llm", "hybrid"] = Field(default="cloud_llm")
    # Corte por evidencia debil: por debajo de este valor el sistema se abstiene
    # sin llamar al LLM. DESACTIVADO (0.0) por defecto tras medirlo: la confianza
    # de casos contestables baja hasta 0.03 y la de preguntas sin respuesta sube
    # hasta 0.95, asi que no existe umbral que separe ambos grupos. Con 0.5 se
    # perdian 2 respuestas correctas de 30. Ver docs/fase0_baseline_honesto.md.
    answer_min_confidence: float = Field(default=0.0)
    # Caracteres de cada pasaje que se envian al LLM. Con el valor anterior (650)
    # solo el 52% de los chunks cabia entero y el resto se reducia a un unico
    # fragmento: articulos que enumeran requisitos llegaban cortados justo antes
    # de la lista. Con 1500 cabe el 77,5% y el prompt queda en ~3000 tokens.
    answer_evidence_char_limit: int = Field(default=1500)
    answer_synthesis_cloud_base_url: str | None = Field(default=None)
    answer_synthesis_cloud_api_key: str | None = Field(default=None)
    answer_synthesis_cloud_model: str | None = Field(default=None)
    answer_synthesis_cloud_timeout: float | None = Field(default=20.0)
    answer_synthesis_local_base_url: str | None = Field(default=None)
    answer_synthesis_local_api_key: str | None = Field(default=None)
    answer_synthesis_local_model: str | None = Field(default=None)
    answer_synthesis_local_timeout: float | None = Field(default=20.0)
    chunk_summary_context_mode: Literal["off", "local_llm", "cloud_llm", "hybrid"] = Field(
        default="cloud_llm"
    )
    chunk_summary_context_max_input_chars: int = Field(default=1800)
    chunk_summary_context_cloud_base_url: str | None = Field(default=None)
    chunk_summary_context_cloud_api_key: str | None = Field(default=None)
    chunk_summary_context_cloud_model: str | None = Field(default=None)
    chunk_summary_context_cloud_timeout: float | None = Field(default=20.0)
    chunk_summary_context_local_base_url: str | None = Field(default=None)
    chunk_summary_context_local_api_key: str | None = Field(default=None)
    chunk_summary_context_local_model: str | None = Field(default=None)
    chunk_summary_context_local_timeout: float | None = Field(default=20.0)

    embedding_mode: Literal["hash", "local_openai", "cloud_openai", "hybrid"] = Field(
        default="hybrid"
    )
    embedding_batch_size: int = Field(default=32)
    embedding_vector_size: int = Field(default=256)
    embedding_profiled_index_names: bool = Field(default=True)
    embedding_profile_name: str | None = Field(default=None)
    embedding_cloud_base_url: str | None = Field(default=None)
    embedding_cloud_api_key: str | None = Field(default=None)
    embedding_cloud_model: str | None = Field(default="text-embedding-3-small")
    embedding_cloud_timeout: float | None = Field(default=20.0)
    embedding_local_base_url: str | None = Field(default=None)
    embedding_local_api_key: str | None = Field(default=None)
    embedding_local_model: str | None = Field(default=None)
    embedding_local_timeout: float | None = Field(default=20.0)
    retrieval_dense_top_k: int = Field(default=12)
    retrieval_sparse_top_k: int = Field(default=12)
    retrieval_fused_top_k: int = Field(default=8)
    retrieval_rrf_k: int = Field(default=60)
    # Etapas que reordenan despues del reranker. Se exponen como flags para poder
    # medir su aporte por separado contra el fixture honesto; el default conserva
    # el comportamiento historico.
    # Control de acceso por documento. Las ACL se indexaban por chunk desde el
    # principio, pero ninguna busqueda las aplicaba: cualquier usuario recuperaba
    # cualquier documento (hallazgo C1 de la auditoria).
    acl_enforcement_enabled: bool = Field(default=True)
    # Documento sin acl_users ni acl_groups = visible para todos. El corpus
    # actual esta asi, de modo que con True nada deja de funcionar y las ACL solo
    # restringen a partir de que se declaran. Poner en False para exigir ACL
    # explicita en todo documento (deny by default).
    acl_open_when_unrestricted: bool = Field(default=True)
    retrieval_visibility_policy_enabled: bool = Field(default=True)
    retrieval_diversify_enabled: bool = Field(default=True)
    retrieval_answer_prioritization_enabled: bool = Field(default=True)
    rerank_mode: Literal["off", "local", "cohere"] = Field(default="local")
    rerank_top_n: int = Field(default=24)
    rerank_local_base_url: str | None = Field(default=None)
    rerank_local_api_key: str | None = Field(default=None)
    rerank_local_model: str | None = Field(default="BAAI/bge-reranker-v2-m3")
    rerank_local_timeout: float | None = Field(default=20.0)
    rerank_cohere_base_url: str = Field(default="https://api.cohere.com/v2")
    rerank_cohere_api_key: str | None = Field(default=None)
    rerank_cohere_model: str | None = Field(default="rerank-v3.5")
    rerank_cohere_timeout: float | None = Field(default=20.0)

    opensearch_url: str = Field(default="http://localhost:9200")
    opensearch_index_name: str = Field(default="rag_chunks_v1")
    opensearch_branch_index_name: str = Field(default="rag_branches_v1")
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection_name: str = Field(default="rag_chunks_v1")
    qdrant_branch_collection_name: str = Field(default="rag_branches_v1")

    @property
    def data_root(self) -> Path:
        root = Path(".data")
        root.mkdir(parents=True, exist_ok=True)
        return root

    @property
    def local_storage_root(self) -> Path:
        root = Path(self.local_storage_path)
        root.mkdir(parents=True, exist_ok=True)
        return root

    @property
    def embedding_profile_slug(self) -> str:
        if self.embedding_profile_name:
            return self._slugify(self.embedding_profile_name)

        model_name = self._preferred_embedding_model_name()
        if model_name:
            return self._slugify(model_name)

        return f"hash-{int(self.embedding_vector_size)}"

    @property
    def resolved_opensearch_index_name(self) -> str:
        if not self.embedding_profiled_index_names:
            return self.opensearch_index_name
        return f"{self.opensearch_index_name}__{self.embedding_profile_slug}"

    @property
    def resolved_qdrant_collection_name(self) -> str:
        if not self.embedding_profiled_index_names:
            return self.qdrant_collection_name
        return f"{self.qdrant_collection_name}__{self.embedding_profile_slug}"

    @property
    def resolved_opensearch_branch_index_name(self) -> str:
        if not self.embedding_profiled_index_names:
            return self.opensearch_branch_index_name
        return f"{self.opensearch_branch_index_name}__{self.embedding_profile_slug}"

    @property
    def resolved_qdrant_branch_collection_name(self) -> str:
        if not self.embedding_profiled_index_names:
            return self.qdrant_branch_collection_name
        return f"{self.qdrant_branch_collection_name}__{self.embedding_profile_slug}"

    def _preferred_embedding_model_name(self) -> str | None:
        if self.embedding_mode == "local_openai":
            return self.embedding_local_model or self.classifier_local_model
        if self.embedding_mode == "cloud_openai":
            return self.embedding_cloud_model or self.classifier_cloud_model or self.classifier_llm_model
        if self.embedding_mode == "hybrid":
            return (
                self.embedding_local_model
                or self.embedding_cloud_model
                or self.classifier_local_model
                or self.classifier_cloud_model
                or self.classifier_llm_model
            )
        return None

    def _slugify(self, value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
        return slug or "default"


@lru_cache
def get_settings() -> Settings:
    return Settings()
