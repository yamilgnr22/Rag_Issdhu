from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


ClassificationSource = Literal[
    "deterministic",
    "llm_fallback",
    "llm_cloud",
    "llm_local",
    "safe_fallback",
]

DocumentClass = Literal["legal_normative", "technical_standard", "general_document"]
DocumentFamily = Literal["normative", "general"]
ChunkingStrategy = Literal[
    "normative_hierarchical",
    "technical_structural",
    "generic_structural",
    "generic_freeform",
]
EmbeddingSource = Literal["hash", "local_openai", "cloud_openai"]
ReindexStatus = Literal["indexed", "skipped", "error"]
QueryStrategy = Literal["focal", "multi_branch", "global"]


class ACL(BaseModel):
    users: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)


class DocumentContract(BaseModel):
    document_id: str
    source: str
    source_authority: Literal["primary", "secondary", "derived"]
    mime_type: str
    language: str
    acl: ACL
    created_at: datetime


class VersionContract(BaseModel):
    version_id: str
    document_id: str
    checksum: str
    version_status: Literal["active", "superseded", "draft", "deprecated"]
    created_at: datetime


class VersionStatusUpdateRequest(BaseModel):
    version_status: Literal["active", "superseded", "draft", "deprecated"]


class VersionStatusUpdateResponse(BaseModel):
    version_id: str
    document_id: str
    previous_status: Literal["active", "superseded", "draft", "deprecated"]
    version_status: Literal["active", "superseded", "draft", "deprecated"]


class VersionDeleteResponse(BaseModel):
    version_id: str
    document_id: str
    deleted_chunks: int
    deleted_blocks: int
    deleted_indexes: bool
    deleted_storage_prefixes: list[str] = Field(default_factory=list)
    deleted_document: bool = False
    warnings: list[str] = Field(default_factory=list)


class DocumentVersionListItem(BaseModel):
    document_id: str
    version_id: str
    version_status: Literal["active", "superseded", "draft", "deprecated"]
    document_class: DocumentClass | None = None
    document_family: DocumentFamily | None = None
    storage_uri: str
    artifact_uri: str | None = None
    extraction_status: str
    chunking_strategy: ChunkingStrategy | None = None
    classification_source: ClassificationSource | None = None
    classification_confidence: float | None = None
    review_required: bool
    latest_for_document: bool
    created_at: datetime


class DocumentVersionListResponse(BaseModel):
    total: int
    items: list[DocumentVersionListItem] = Field(default_factory=list)


class BlockContract(BaseModel):
    block_id: str
    document_id: str
    version_id: str
    block_type: str
    text: str
    page: int
    section_path: list[str]


class ChunkContract(BaseModel):
    chunk_id: str
    document_id: str
    version_id: str
    block_refs: list[str]
    text: str
    metadata: dict[str, object]
    acl: ACL
    index_status: Literal["pending", "indexed", "error"]
    indexed_profile: str | None = None


class CitationContract(BaseModel):
    label: str
    document_id: str
    version_id: str
    chunk_id: str


class IngestionFilesystemRequest(BaseModel):
    path: str
    source: str = "filesystem"
    source_authority: Literal["primary", "secondary", "derived"] = "primary"
    language: str | None = None
    acl_users: list[str] = Field(default_factory=list)
    acl_groups: list[str] = Field(default_factory=list)
    version_status: Literal["active", "superseded", "draft", "deprecated"] = "active"


class IngestionResponse(BaseModel):
    document: DocumentContract
    version: VersionContract
    storage_uri: str
    artifact_uri: str | None = None
    extraction_status: Literal["extracted", "manual_review", "failed"]
    review_required: bool
    review_reason: str | None = None
    blocks_count: int = 0
    text_length: int = 0
    document_class: DocumentClass | None = None
    document_family: DocumentFamily | None = None
    chunking_strategy: ChunkingStrategy | None = None
    classification_source: ClassificationSource | None = None
    classification_confidence: float | None = None
    classification_reason_short: str | None = None


class ChunkingRequest(BaseModel):
    force: bool = False
    target_tokens: int | None = None
    min_tokens: int | None = None
    max_tokens: int | None = None
    overlap_ratio: float | None = None


class ChunkingResponse(BaseModel):
    document_id: str
    version_id: str
    chunks_count: int
    table_chunks_count: int
    artifact_uri: str | None = None
    chunk_window: dict[str, float | int]
    document_class: DocumentClass
    document_family: DocumentFamily
    chunking_strategy: ChunkingStrategy
    classification_source: ClassificationSource
    classification_confidence: float
    classification_reason_short: str | None = None


class ReindexRequest(BaseModel):
    version_ids: list[str] = Field(default_factory=list)
    force: bool = False
    limit: int = 25
    document_class: DocumentClass | None = None
    refresh: bool = True


class ReindexVersionResult(BaseModel):
    version_id: str
    document_id: str
    document_class: DocumentClass | None = None
    document_family: DocumentFamily | None = None
    status: ReindexStatus
    chunks_indexed: int = 0
    reason: str | None = None


class ReindexResponse(BaseModel):
    processed_versions: int
    indexed_versions: int
    skipped_versions: int
    failed_versions: int
    indexed_chunks: int
    qdrant_collection: str
    opensearch_index: str
    embedding_source: EmbeddingSource
    embedding_profile: str
    results: list[ReindexVersionResult]


class QueryRequest(BaseModel):
    question: str
    user_identity: str
    # Grupos/areas del usuario. El control de acceso cruza estos valores contra
    # acl_groups de cada chunk; sin ellos el usuario solo ve documentos sin
    # restriccion y los que lo nombren explicitamente en acl_users.
    user_groups: list[str] = Field(default_factory=list)
    filters: dict[str, object] = Field(default_factory=dict)
    conversation_context: list[dict[str, object]] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationContract]
    confidence: float
    applied_filters: dict[str, object]
    rewrites_applied: list[str]
    decision_trace: list[str]
    fallback_reason: str | None
    source_conflict: bool


class RetrievalEvalTargetSelector(BaseModel):
    version_id: str | None = None
    document_id: str | None = None
    storage_uri_contains: str | None = None
    document_class: DocumentClass | None = None
    document_family: DocumentFamily | None = None


class RetrievalEvalExpectation(BaseModel):
    document_class: DocumentClass
    document_family: DocumentFamily | None = None
    target: RetrievalEvalTargetSelector
    targets: list[RetrievalEvalTargetSelector] = Field(default_factory=list)
    target_match_mode: Literal["any", "all"] = "any"
    pass_rank: int = 3
    must_contain_all: list[str] = Field(default_factory=list)
    should_contain_any: list[str] = Field(default_factory=list)
    expected_label_hints: list[str] = Field(default_factory=list)


class RetrievalEvalCase(BaseModel):
    case_id: str
    suite_id: str
    question: str
    strategy: QueryStrategy | None = None
    filters: dict[str, object] = Field(default_factory=dict)
    expectation: RetrievalEvalExpectation
    notes: str | None = None


class RetrievalEvalRequest(BaseModel):
    suite_ids: list[str] = Field(default_factory=list)
    case_ids: list[str] = Field(default_factory=list)
    fixture_path: str | None = None
    fixture_paths: list[str] = Field(default_factory=list)
    include_passed: bool = True
    user_identity: str = "eval@local"


class RetrievalEvalHitPreview(BaseModel):
    rank: int
    document_id: str
    version_id: str
    chunk_id: str
    section_path: list[str] = Field(default_factory=list)
    chunk_kind: str | None = None
    preview: str


class RetrievalEvalCaseResult(BaseModel):
    case_id: str
    suite_id: str
    question: str
    expected_strategy: QueryStrategy | None = None
    planner_strategy: QueryStrategy | None = None
    planner_ok: bool
    expected_document_class: DocumentClass
    expected_document_family: DocumentFamily | None = None
    target_match_mode: Literal["any", "all"] = "any"
    expected_version_id: str | None = None
    expected_document_id: str | None = None
    expected_version_ids: list[str] = Field(default_factory=list)
    expected_document_ids: list[str] = Field(default_factory=list)
    router_query_type: str
    router_ok: bool
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    visible_hit_at_k: bool
    visible_top_k: int
    reciprocal_rank: float
    citation_precision_at_3: float
    required_terms_ok: bool
    optional_terms_ok: bool
    visible_required_terms_ok: bool
    visible_optional_terms_ok: bool
    label_terms_ok: bool
    visible_label_terms_ok: bool
    passed: bool
    fallback_reason: str | None = None
    decision_trace: list[str] = Field(default_factory=list)
    top_hits: list[RetrievalEvalHitPreview] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RetrievalEvalSuiteSummary(BaseModel):
    suite_id: str
    total_cases: int
    passed_cases: int
    planner_accuracy: float
    router_accuracy: float
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mean_reciprocal_rank: float
    citation_precision_at_3: float
    no_result_rate: float


class RetrievalEvalResponse(BaseModel):
    fixture_path: str
    fixture_paths: list[str] = Field(default_factory=list)
    total_cases: int
    passed_cases: int
    planner_accuracy: float
    router_accuracy: float
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mean_reciprocal_rank: float
    citation_precision_at_3: float
    no_result_rate: float
    by_suite: list[RetrievalEvalSuiteSummary]
    results: list[RetrievalEvalCaseResult]


class FeedbackRequest(BaseModel):
    query_id: str | None = None
    useful: bool
    comment: str | None = None
