import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import httpx
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.contracts import (
    DocumentClass,
    ReindexRequest,
    ReindexResponse,
    ReindexVersionResult,
)
from app.models.database import BlockRecord, ChunkRecord, DocumentRecord, VersionRecord
from app.services.chunk_summary_context import ChunkSummaryContextService
from app.services.chunking import CanonicalUnit, ChunkingService
from app.services.document_classification import DocumentClassification
from app.services.embedding_gateway import EmbeddingGateway, TOKEN_PATTERN
from app.services.evidence_role_classifier import EvidenceRoleClassifier, VALID_EVIDENCE_ROLES


FOCUS_STOPWORDS = {
    "a",
    "al",
    "con",
    "de",
    "del",
    "el",
    "en",
    "es",
    "la",
    "las",
    "lo",
    "los",
    "para",
    "por",
    "que",
    "se",
    "su",
    "sus",
    "un",
    "una",
    "y",
}


class QdrantIndexClient:
    def __init__(self, settings, *, collection_name: str | None = None) -> None:
        self.base_url = settings.qdrant_url.rstrip("/")
        self.collection_name = collection_name or settings.resolved_qdrant_collection_name

    def ensure_collection(self, *, vector_size: int) -> None:
        with httpx.Client(timeout=20.0) as client:
            response = client.get(f"{self.base_url}/collections/{self.collection_name}")
            if response.status_code == 404:
                create = client.put(
                    f"{self.base_url}/collections/{self.collection_name}",
                    json={
                        "vectors": {
                            "size": vector_size,
                            "distance": "Cosine",
                        }
                    },
                )
                create.raise_for_status()
                return

            response.raise_for_status()
            payload = response.json()
            vectors = payload.get("result", {}).get("config", {}).get("params", {}).get("vectors", {})
            existing_size = None
            if isinstance(vectors, dict) and "size" in vectors:
                existing_size = vectors.get("size")
            elif isinstance(vectors, dict) and "default" in vectors:
                existing_size = vectors.get("default", {}).get("size")
            if existing_size is None:
                raise RuntimeError("Could not determine Qdrant vector size for existing collection.")
            if int(existing_size) != int(vector_size):
                raise RuntimeError(
                    f"Qdrant collection '{self.collection_name}' uses vector size {existing_size}, "
                    f"but the embedding provider returned {vector_size}."
                )

    def upsert_points(self, *, points: list[dict[str, object]]) -> None:
        with httpx.Client(timeout=60.0) as client:
            response = client.put(
                f"{self.base_url}/collections/{self.collection_name}/points",
                params={"wait": "true"},
                json={"points": points},
            )
            response.raise_for_status()

    def delete_by_version_id(self, *, version_id: str) -> None:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{self.base_url}/collections/{self.collection_name}/points/delete",
                params={"wait": "true"},
                json={
                    "filter": {
                        "must": [
                            {
                                "key": "version_id",
                                "match": {"value": version_id},
                            }
                        ]
                    }
                },
            )
            response.raise_for_status()


class OpenSearchIndexClient:
    def __init__(
        self,
        settings,
        *,
        index_name: str | None = None,
        id_field: str = "chunk_id",
        properties: dict[str, object] | None = None,
    ) -> None:
        self.base_url = settings.opensearch_url.rstrip("/")
        self.index_name = index_name or settings.resolved_opensearch_index_name
        self.id_field = id_field
        self.properties = properties or {}

    def ensure_index(self) -> None:
        with httpx.Client(timeout=20.0) as client:
            response = client.head(f"{self.base_url}/{self.index_name}")
            if response.status_code == 200:
                return
            if response.status_code not in {404, 405}:
                response.raise_for_status()

            create = client.put(
                f"{self.base_url}/{self.index_name}",
                json={
                    "settings": {
                        "index": {
                            "number_of_shards": 1,
                            "number_of_replicas": 0,
                        }
                    },
                    "mappings": {
                        "properties": {
                            **self.properties,
                        }
                    },
                },
            )
            if create.status_code not in {200, 201}:
                create.raise_for_status()

    def bulk_upsert(self, *, documents: list[dict[str, object]], refresh: bool) -> None:
        lines: list[str] = []
        for document in documents:
            lines.append(
                json.dumps({"index": {"_index": self.index_name, "_id": document[self.id_field]}})
            )
            lines.append(json.dumps(document, ensure_ascii=True))
        body = "\n".join(lines) + "\n"

        headers = {"Content-Type": "application/x-ndjson"}
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{self.base_url}/_bulk",
                params={"refresh": str(refresh).lower()},
                headers=headers,
                content=body.encode("utf-8"),
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("errors") is True:
                raise RuntimeError("OpenSearch bulk indexing returned item errors.")

    def delete_by_version_id(self, *, version_id: str, refresh: bool = True) -> None:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{self.base_url}/{self.index_name}/_delete_by_query",
                params={"refresh": str(refresh).lower()},
                json={"query": {"term": {"version_id": version_id}}},
            )
            response.raise_for_status()


class IndexingService:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.embeddings = EmbeddingGateway(settings)
        self.chunk_summary_context = ChunkSummaryContextService(settings)
        self.evidence_role_classifier = EvidenceRoleClassifier(settings)
        self.chunking = ChunkingService(settings)
        self.qdrant = QdrantIndexClient(
            settings,
            collection_name=settings.resolved_qdrant_collection_name,
        )
        self.branch_qdrant = QdrantIndexClient(
            settings,
            collection_name=settings.resolved_qdrant_branch_collection_name,
        )
        self.opensearch = OpenSearchIndexClient(
            settings,
            index_name=settings.resolved_opensearch_index_name,
            id_field="chunk_id",
            properties=self._chunk_index_properties(),
        )
        self.branch_opensearch = OpenSearchIndexClient(
            settings,
            index_name=settings.resolved_opensearch_branch_index_name,
            id_field="branch_id",
            properties=self._branch_index_properties(),
        )

    def reindex(self, *, db: Session, request: ReindexRequest | None = None) -> ReindexResponse:
        request = request or ReindexRequest()
        targets = self._select_targets(db=db, request=request)
        results: list[ReindexVersionResult] = []
        indexed_versions = 0
        skipped_versions = 0
        failed_versions = 0
        indexed_chunks = 0
        embedding_source = self.embeddings.resolve_source()
        embedding_profile = self.settings.embedding_profile_slug

        for version, document in targets:
            blocks = (
                db.query(BlockRecord)
                .filter(BlockRecord.version_id == version.version_id)
                .order_by(BlockRecord.block_order.asc())
                .all()
            )
            chunks = (
                db.query(ChunkRecord)
                .filter(ChunkRecord.version_id == version.version_id)
                .order_by(ChunkRecord.chunk_order.asc())
                .all()
            )
            if not chunks:
                results.append(
                    ReindexVersionResult(
                        version_id=version.version_id,
                        document_id=version.document_id,
                        document_class=version.document_class,
                        document_family=version.document_family,
                        status="skipped",
                        reason="No chunks available for indexing.",
                    )
                )
                skipped_versions += 1
                continue

            if not request.force and all(
                chunk.index_status == "indexed" and (chunk.indexed_profile or "") == embedding_profile
                for chunk in chunks
            ):
                results.append(
                    ReindexVersionResult(
                        version_id=version.version_id,
                        document_id=version.document_id,
                        document_class=version.document_class,
                        document_family=version.document_family,
                        status="skipped",
                        reason=f"All chunks already indexed for embedding profile '{embedding_profile}'.",
                    )
                )
                skipped_versions += 1
                continue

            try:
                canonical_units = self._build_canonical_units_for_version(
                    blocks=blocks,
                    version=version,
                )
                index_documents = [
                    self._build_index_document(document=document, version=version, chunk=chunk)
                    for chunk in chunks
                ]
                self.evidence_role_classifier.annotate_documents(index_documents)
                branch_leads = self._annotate_branch_leads(
                    document_id=version.document_id,
                    version_id=version.version_id,
                    chunk_documents=index_documents,
                )
                branch_documents = self._build_branch_documents(
                    document=document,
                    version=version,
                    chunk_documents=index_documents,
                    branch_leads=branch_leads,
                    canonical_units=canonical_units,
                )
                self._apply_branch_role_profiles(
                    branch_documents=branch_documents,
                    chunk_documents=index_documents,
                )
                texts_to_embed = [self._embedding_text(item) for item in index_documents]
                branch_texts = [self._embedding_text(item) for item in branch_documents]
                vectors, source = self.embeddings.embed_texts(texts_to_embed)
                embedding_source = source
                if not vectors:
                    raise RuntimeError("Embedding provider returned no vectors.")
                branch_vectors: list[list[float]] = []
                if branch_texts:
                    branch_vectors, branch_source = self.embeddings.embed_texts(branch_texts)
                    embedding_source = branch_source

                self.qdrant.ensure_collection(vector_size=len(vectors[0]))
                if branch_vectors:
                    self.branch_qdrant.ensure_collection(vector_size=len(branch_vectors[0]))
                self.opensearch.ensure_index()
                if branch_documents:
                    self.branch_opensearch.ensure_index()
                self.qdrant.upsert_points(
                    points=[
                        {
                            "id": self._qdrant_point_id(str(doc["chunk_id"])),
                            "vector": vector,
                            "payload": doc,
                        }
                        for doc, vector in zip(index_documents, vectors, strict=True)
                    ]
                )
                if branch_documents and branch_vectors:
                    self.branch_qdrant.upsert_points(
                        points=[
                            {
                                "id": self._qdrant_point_id(str(doc["branch_id"])),
                                "vector": vector,
                                "payload": doc,
                            }
                            for doc, vector in zip(branch_documents, branch_vectors, strict=True)
                        ]
                    )
                self.opensearch.bulk_upsert(documents=index_documents, refresh=request.refresh)
                if branch_documents:
                    self.branch_opensearch.bulk_upsert(documents=branch_documents, refresh=request.refresh)

                for chunk in chunks:
                    chunk.index_status = "indexed"
                    chunk.indexed_profile = embedding_profile

                indexed_versions += 1
                indexed_chunks += len(chunks)
                results.append(
                    ReindexVersionResult(
                        version_id=version.version_id,
                        document_id=version.document_id,
                        document_class=version.document_class,
                        document_family=version.document_family,
                        status="indexed",
                        chunks_indexed=len(chunks),
                    )
                )
            except Exception as exc:
                for chunk in chunks:
                    chunk.index_status = "error"
                failed_versions += 1
                results.append(
                    ReindexVersionResult(
                        version_id=version.version_id,
                        document_id=version.document_id,
                        document_class=version.document_class,
                        document_family=version.document_family,
                        status="error",
                        reason=self._truncate_reason(str(exc)),
                    )
                )

        db.commit()
        return ReindexResponse(
            processed_versions=len(targets),
            indexed_versions=indexed_versions,
            skipped_versions=skipped_versions,
            failed_versions=failed_versions,
            indexed_chunks=indexed_chunks,
            qdrant_collection=self.settings.resolved_qdrant_collection_name,
            opensearch_index=self.settings.resolved_opensearch_index_name,
            embedding_source=embedding_source,
            embedding_profile=embedding_profile,
            results=results,
        )

    def _select_targets(
        self,
        *,
        db: Session,
        request: ReindexRequest,
    ) -> list[tuple[VersionRecord, DocumentRecord]]:
        current_profile = self.settings.embedding_profile_slug
        version_ids: list[str]
        if request.version_ids:
            version_ids = request.version_ids
        else:
            query = (
                db.query(ChunkRecord.version_id)
                .join(VersionRecord, ChunkRecord.version_id == VersionRecord.version_id)
                .distinct()
            )
            if not request.force:
                query = query.filter(
                    or_(
                        ChunkRecord.index_status != "indexed",
                        ChunkRecord.indexed_profile.is_(None),
                        ChunkRecord.indexed_profile != current_profile,
                    )
                )
            if request.document_class is not None:
                query = query.filter(VersionRecord.document_class == request.document_class)
            version_ids = [
                row[0]
                for row in query.order_by(VersionRecord.created_at.desc())
                .limit(max(1, request.limit))
                .all()
            ]

        if not version_ids:
            return []

        records = (
            db.query(VersionRecord, DocumentRecord)
            .join(DocumentRecord, VersionRecord.document_id == DocumentRecord.document_id)
            .filter(VersionRecord.version_id.in_(version_ids))
            .all()
        )
        by_id = {version.version_id: (version, document) for version, document in records}
        ordered = [by_id[version_id] for version_id in version_ids if version_id in by_id]
        if request.document_class is None:
            return ordered
        return [
            (version, document)
            for version, document in ordered
            if version.document_class == request.document_class
        ]

    def _build_canonical_units_for_version(
        self,
        *,
        blocks: list[BlockRecord],
        version: VersionRecord,
    ) -> list[CanonicalUnit]:
        if not blocks:
            return []
        classification = self._version_classification(version)
        return self.chunking.build_canonical_units(
            blocks=blocks,
            classification=classification,
        )

    def _version_classification(self, version: VersionRecord) -> DocumentClassification:
        document_class = str(version.document_class or "general_document")
        document_family = str(version.document_family or ("normative" if document_class != "general_document" else "general"))
        chunking_strategy = str(
            version.chunking_strategy
            or (
                "technical_structural"
                if document_class == "technical_standard"
                else "normative_hierarchical"
                if document_class == "legal_normative"
                else "generic_structural"
            )
        )
        classification_source = str(version.classification_source or "safe_fallback")
        confidence = float(version.classification_confidence or 0.0)
        reason_short = str(version.classification_reason or "Persisted version classification.")
        return DocumentClassification(
            document_class=document_class,
            document_family=document_family,
            chunking_strategy=chunking_strategy,
            classification_source=classification_source,
            confidence=confidence,
            reason_short=reason_short,
        )

    def _build_index_document(
        self,
        *,
        document: DocumentRecord,
        version: VersionRecord,
        chunk: ChunkRecord,
    ) -> dict[str, object]:
        metadata = chunk.chunk_metadata or {}
        retrieval_context = str(metadata.get("retrieval_context") or "").strip()
        contextualized_text = str(metadata.get("contextualized_text") or "").strip()
        if not contextualized_text:
            contextualized_text = (
                f"{retrieval_context}\n{chunk.text}".strip() if retrieval_context else chunk.text
            )

        hierarchy_path = self._coerce_path(
            metadata.get("hierarchy_path") or metadata.get("section_path") or []
        )
        parent_path = hierarchy_path[:-1]
        path_text = " > ".join(hierarchy_path)
        approx_tokens = int(metadata.get("approx_tokens") or self._approx_tokens(chunk.text))
        page_start = int(metadata.get("page_start") or 1)
        page_end = int(metadata.get("page_end") or page_start)
        document_title = self._document_title(version.storage_uri)
        canonical_label, unit_type, unit_number = self._derive_unit_metadata(
            document_class=str(version.document_class or ""),
            hierarchy_path=hierarchy_path,
            metadata=metadata,
        )
        unit_topic = self._extract_unit_topic(
            canonical_label=canonical_label,
            unit_type=unit_type,
            hierarchy_path=hierarchy_path,
        )
        unit_key = self._unit_key(unit_type=unit_type, unit_number=unit_number)
        reference_variants = self._reference_variants(
            canonical_label=canonical_label,
            unit_type=unit_type,
            unit_number=unit_number,
            hierarchy_path=hierarchy_path,
            unit_topic=unit_topic,
        )
        sample_questions = self._build_sample_questions(
            document_title=document_title,
            document_class=str(version.document_class or ""),
            canonical_label=canonical_label,
            unit_type=unit_type,
            unit_number=unit_number,
            parent_path=parent_path,
            unit_topic=unit_topic,
        )
        chunk_summary_context = self.chunk_summary_context.build_context(
            document_title=document_title,
            document_class=str(version.document_class or ""),
            canonical_label=canonical_label,
            unit_type=unit_type,
            unit_number=unit_number,
            path_text=path_text,
            retrieval_context=retrieval_context,
            raw_text=chunk.text,
            unit_topic=unit_topic,
        )
        focus_terms = self._focus_terms(
            hierarchy_path=hierarchy_path,
            retrieval_context=retrieval_context,
            raw_text=chunk.text,
            supplemental_texts=[chunk_summary_context],
        )
        branch_keys = self._branch_keys_for_path(
            document_id=version.document_id,
            version_id=version.version_id,
            hierarchy_path=hierarchy_path,
        )
        boilerplate_profile = self._boilerplate_profile(
            hierarchy_path=hierarchy_path,
            canonical_label=canonical_label,
            unit_type=unit_type,
            raw_text=chunk.text,
            chunk_kind=str(metadata.get("chunk_kind", "unknown")),
        )

        index_document: dict[str, object] = {
            "chunk_id": chunk.chunk_id,
            "document_id": version.document_id,
            "version_id": version.version_id,
            "version_status": version.version_status,
            "document_class": version.document_class,
            "document_family": version.document_family,
            "chunking_strategy": version.chunking_strategy,
            "classification_source": version.classification_source,
            "chunk_kind": metadata.get("chunk_kind", "unknown"),
            "mime_type": document.mime_type,
            "source": document.source,
            "source_authority": document.source_authority,
            "section_path": hierarchy_path,
            "hierarchy_path": hierarchy_path,
            "parent_path": parent_path,
            "path_text": path_text,
            "retrieval_context": retrieval_context,
            "contextualized_text": contextualized_text,
            "chunk_summary_context": chunk_summary_context,
            "raw_text": chunk.text,
            "block_refs": chunk.block_refs,
            "acl_users": list((document.acl or {}).get("users", [])),
            "acl_groups": list((document.acl or {}).get("groups", [])),
            "chunk_order": chunk.chunk_order,
            "approx_tokens": approx_tokens,
            "page_start": page_start,
            "page_end": page_end,
            "document_title": document_title,
            "canonical_label": canonical_label,
            "unit_type": unit_type,
            "unit_number": unit_number,
            "unit_key": unit_key,
            "unit_topic": unit_topic,
            "reference_variants": reference_variants,
            "reference_variants_text": " | ".join(reference_variants),
            "focus_terms": focus_terms,
            "focus_terms_text": " | ".join(focus_terms),
            "sample_questions": sample_questions,
            "sample_questions_text": " | ".join(sample_questions),
            "branch_summary_context": "",
            "branch_keywords": [],
            "branch_keywords_text": "",
            "branch_keys": branch_keys,
            "branch_lead_keys": [],
            "is_branch_lead": False,
            "is_boilerplate": boilerplate_profile["is_boilerplate"],
            "boilerplate_score": boilerplate_profile["boilerplate_score"],
            "boilerplate_reasons": boilerplate_profile["boilerplate_reasons"],
        }

        for label in (
            "book_label",
            "title_label",
            "chapter_label",
            "article_label",
            "section_label",
            "subsection_label",
        ):
            if metadata.get(label):
                index_document[label] = metadata[label]
        return index_document

    def _build_branch_documents(
        self,
        *,
        document: DocumentRecord,
        version: VersionRecord,
        chunk_documents: list[dict[str, object]],
        branch_leads: dict[str, dict[str, str]],
        canonical_units: list[CanonicalUnit],
    ) -> list[dict[str, object]]:
        aggregates: dict[str, dict[str, object]] = {}
        document_title = self._document_title(version.storage_uri)

        for unit in canonical_units:
            branch_id = self._branch_id(
                document_id=version.document_id,
                version_id=version.version_id,
                hierarchy_path=unit.hierarchy_path,
            )
            canonical_label, unit_type, unit_number = self._derive_unit_metadata(
                document_class=str(version.document_class or ""),
                hierarchy_path=unit.hierarchy_path,
                metadata={"section_path": unit.hierarchy_path},
            )
            unit_topic = self._extract_unit_topic(
                canonical_label=canonical_label,
                unit_type=unit_type,
                hierarchy_path=unit.hierarchy_path,
            )
            unit_key = self._unit_key(unit_type=unit_type, unit_number=unit_number)
            reference_variants = self._reference_variants(
                canonical_label=canonical_label,
                unit_type=unit_type,
                unit_number=unit_number,
                hierarchy_path=unit.hierarchy_path,
                unit_topic=unit_topic,
            )
            sample_questions = self._build_sample_questions(
                document_title=document_title,
                document_class=str(version.document_class or ""),
                canonical_label=canonical_label,
                unit_type=unit_type,
                unit_number=unit_number,
                parent_path=unit.parent_path,
                unit_topic=unit_topic,
            )
            child_counter = Counter(label for label in unit.child_labels if str(label).strip())
            source_block_types = Counter(
                block_type for block_type in unit.support_block_types if str(block_type).strip()
            )
            aggregates[branch_id] = {
                "branch_id": branch_id,
                "branch_key": branch_id,
                "parent_branch_key": self._branch_id(
                    document_id=version.document_id,
                    version_id=version.version_id,
                    hierarchy_path=unit.parent_path,
                )
                if unit.parent_path
                else "",
                "document_id": version.document_id,
                "version_id": version.version_id,
                "version_status": version.version_status,
                "document_class": version.document_class,
                "document_family": version.document_family,
                "chunking_strategy": version.chunking_strategy,
                "classification_source": version.classification_source,
                "mime_type": document.mime_type,
                "source": document.source,
                "source_authority": document.source_authority,
                "hierarchy_path": list(unit.hierarchy_path),
                "parent_path": list(unit.parent_path),
                "path_text": " > ".join(unit.hierarchy_path),
                "document_title": document_title,
                "canonical_label": canonical_label,
                "unit_type": unit_type,
                "unit_number": unit_number,
                "unit_key": unit_key,
                "unit_topic": unit_topic,
                "reference_variants": reference_variants,
                "sample_questions": set(sample_questions),
                "child_labels": child_counter,
                "chunk_kinds": Counter(),
                "source_block_types": source_block_types,
                "child_chunk_ids": [],
                "source_block_ids": list(unit.support_block_ids),
                "snippets": list(unit.representative_snippets),
                "acl_users": set((document.acl or {}).get("users", [])),
                "acl_groups": set((document.acl or {}).get("groups", [])),
                "chunk_order": int(unit.order),
                "page_start": int(unit.page_start),
                "page_end": int(unit.page_end),
                "inferred_only": bool(unit.inferred_only),
            }

        for chunk_document in chunk_documents:
            hierarchy_path = self._coerce_path(chunk_document.get("hierarchy_path"))
            if not hierarchy_path:
                continue

            for depth in range(1, len(hierarchy_path) + 1):
                branch_path = hierarchy_path[:depth]
                branch_id = self._branch_id(
                    document_id=version.document_id,
                    version_id=version.version_id,
                    hierarchy_path=branch_path,
                )
                aggregate = aggregates.get(branch_id)
                if aggregate is None:
                    canonical_label, unit_type, unit_number = self._derive_unit_metadata(
                        document_class=str(version.document_class or ""),
                        hierarchy_path=branch_path,
                        metadata={"section_path": branch_path},
                    )
                    unit_topic = self._extract_unit_topic(
                        canonical_label=canonical_label,
                        unit_type=unit_type,
                        hierarchy_path=branch_path,
                    )
                    unit_key = self._unit_key(unit_type=unit_type, unit_number=unit_number)
                    reference_variants = self._reference_variants(
                        canonical_label=canonical_label,
                        unit_type=unit_type,
                        unit_number=unit_number,
                        hierarchy_path=branch_path,
                        unit_topic=unit_topic,
                    )
                    sample_questions = self._build_sample_questions(
                        document_title=document_title,
                        document_class=str(version.document_class or ""),
                        canonical_label=canonical_label,
                        unit_type=unit_type,
                        unit_number=unit_number,
                        parent_path=branch_path[:-1],
                        unit_topic=unit_topic,
                    )
                    aggregate = {
                        "branch_id": branch_id,
                        "branch_key": branch_id,
                        "parent_branch_key": self._branch_id(
                            document_id=version.document_id,
                            version_id=version.version_id,
                            hierarchy_path=branch_path[:-1],
                        )
                        if branch_path[:-1]
                        else "",
                        "document_id": version.document_id,
                        "version_id": version.version_id,
                        "version_status": version.version_status,
                        "document_class": version.document_class,
                        "document_family": version.document_family,
                        "chunking_strategy": version.chunking_strategy,
                        "classification_source": version.classification_source,
                        "mime_type": document.mime_type,
                        "source": document.source,
                        "source_authority": document.source_authority,
                        "hierarchy_path": branch_path,
                        "parent_path": branch_path[:-1],
                        "path_text": " > ".join(branch_path),
                        "document_title": document_title,
                        "canonical_label": canonical_label,
                        "unit_type": unit_type,
                        "unit_number": unit_number,
                        "unit_key": unit_key,
                        "unit_topic": unit_topic,
                        "reference_variants": reference_variants,
                        "sample_questions": set(sample_questions),
                        "child_labels": Counter(),
                        "chunk_kinds": Counter(),
                        "source_block_types": Counter(),
                        "child_chunk_ids": [],
                        "source_block_ids": [],
                        "snippets": [],
                        "acl_users": set((document.acl or {}).get("users", [])),
                        "acl_groups": set((document.acl or {}).get("groups", [])),
                        "chunk_order": int(chunk_document.get("chunk_order") or 0),
                        "page_start": int(chunk_document.get("page_start") or 1),
                        "page_end": int(chunk_document.get("page_end") or int(chunk_document.get("page_start") or 1)),
                        "inferred_only": False,
                    }
                    aggregates[branch_id] = aggregate

                next_label = hierarchy_path[depth] if depth < len(hierarchy_path) else None
                if next_label:
                    aggregate["child_labels"][str(next_label)] += 1
                aggregate["chunk_kinds"][str(chunk_document.get("chunk_kind") or "unknown")] += 1
                aggregate["child_chunk_ids"].append(str(chunk_document["chunk_id"]))
                aggregate["inferred_only"] = False
                aggregate["chunk_order"] = min(
                    int(aggregate["chunk_order"]),
                    int(chunk_document.get("chunk_order") or 0),
                )
                aggregate["page_start"] = min(
                    int(aggregate["page_start"]),
                    int(chunk_document.get("page_start") or int(aggregate["page_start"])),
                )
                aggregate["page_end"] = max(
                    int(aggregate["page_end"]),
                    int(chunk_document.get("page_end") or int(aggregate["page_end"])),
                )

                snippet = self._representative_snippet(chunk_document)
                snippets = aggregate["snippets"]
                if snippet and snippet not in snippets and len(snippets) < 3:
                    snippets.append(snippet)

        branch_documents: list[dict[str, object]] = []
        for aggregate in aggregates.values():
            sample_questions = sorted(aggregate["sample_questions"])
            child_labels = [
                label for label, _ in aggregate["child_labels"].most_common(4) if label.strip()
            ]
            snippets = list(aggregate["snippets"])
            branch_summary = self._branch_summary(
                document_title=document_title,
                path=aggregate["hierarchy_path"],
                canonical_label=str(aggregate["canonical_label"] or ""),
                unit_type=str(aggregate["unit_type"] or ""),
                unit_topic=str(aggregate["unit_topic"] or ""),
                child_labels=child_labels,
                snippets=snippets,
            )
            branch_summary_context = self._branch_summary_context(
                document_title=document_title,
                path=aggregate["hierarchy_path"],
                canonical_label=str(aggregate["canonical_label"] or ""),
                unit_type=str(aggregate["unit_type"] or ""),
                unit_number=str(aggregate["unit_number"] or ""),
                unit_topic=str(aggregate["unit_topic"] or ""),
                child_labels=child_labels,
                snippets=snippets,
            )
            branch_keywords = self._branch_keywords(
                hierarchy_path=list(aggregate["hierarchy_path"]),
                canonical_label=str(aggregate["canonical_label"] or ""),
                unit_type=str(aggregate["unit_type"] or ""),
                unit_topic=str(aggregate["unit_topic"] or ""),
                child_labels=child_labels,
                snippets=snippets,
            )
            retrieval_context = self._branch_retrieval_context(
                document_title=document_title,
                path=aggregate["hierarchy_path"],
                canonical_label=str(aggregate["canonical_label"] or ""),
                unit_type=str(aggregate["unit_type"] or ""),
                unit_number=str(aggregate["unit_number"] or ""),
                unit_topic=str(aggregate["unit_topic"] or ""),
                child_labels=child_labels,
            )
            chunk_summary_context = "\n".join(
                part
                for part in (
                    self._document_title_short(document_title),
                    branch_summary_context,
                )
                if part
            )
            focus_terms = self._focus_terms(
                hierarchy_path=list(aggregate["hierarchy_path"]),
                retrieval_context=retrieval_context,
                raw_text=" ".join(snippets),
                supplemental_texts=[chunk_summary_context, " | ".join(branch_keywords), *child_labels],
            )
            contextualized_text = "\n".join(
                part
                for part in (
                    retrieval_context,
                    chunk_summary_context,
                    "Palabras clave: " + ", ".join(branch_keywords[:10]) if branch_keywords else "",
                    branch_summary,
                )
                if part
            )
            boilerplate_profile = self._boilerplate_profile(
                hierarchy_path=list(aggregate["hierarchy_path"]),
                canonical_label=str(aggregate["canonical_label"] or ""),
                unit_type=str(aggregate["unit_type"] or ""),
                raw_text=branch_summary,
                child_labels=child_labels,
                source_block_types=aggregate["source_block_types"],
            )
            branch_primary_kind = self._branch_primary_kind(
                aggregate["chunk_kinds"],
                unit_type=str(aggregate["unit_type"] or ""),
                source_block_types=aggregate["source_block_types"],
            )
            branch_documents.append(
                {
                    "branch_id": aggregate["branch_id"],
                    "branch_key": aggregate["branch_key"],
                    "parent_branch_key": aggregate["parent_branch_key"],
                    "chunk_id": aggregate["branch_id"],
                    "document_id": aggregate["document_id"],
                    "version_id": aggregate["version_id"],
                    "version_status": aggregate["version_status"],
                    "document_class": aggregate["document_class"],
                    "document_family": aggregate["document_family"],
                    "chunking_strategy": aggregate["chunking_strategy"],
                    "classification_source": aggregate["classification_source"],
                    "chunk_kind": branch_primary_kind,
                    "branch_primary_kind": branch_primary_kind,
                    "mime_type": aggregate["mime_type"],
                    "source": aggregate["source"],
                    "source_authority": aggregate["source_authority"],
                    "section_path": aggregate["hierarchy_path"],
                    "hierarchy_path": aggregate["hierarchy_path"],
                    "parent_path": aggregate["parent_path"],
                    "path_text": aggregate["path_text"],
                    "retrieval_context": retrieval_context,
                    "contextualized_text": contextualized_text,
                    "chunk_summary_context": chunk_summary_context,
                    "branch_summary_context": branch_summary_context,
                    "branch_keywords": branch_keywords,
                    "branch_keywords_text": " | ".join(branch_keywords),
                    "raw_text": branch_summary,
                    "block_refs": aggregate["child_chunk_ids"],
                    "source_block_refs": aggregate["source_block_ids"][:48],
                    "source_block_types": sorted(aggregate["source_block_types"].keys()),
                    "acl_users": sorted(aggregate["acl_users"]),
                    "acl_groups": sorted(aggregate["acl_groups"]),
                    "chunk_order": aggregate["chunk_order"],
                    "approx_tokens": self._approx_tokens(branch_summary),
                    "document_title": aggregate["document_title"],
                    "canonical_label": aggregate["canonical_label"],
                    "unit_type": aggregate["unit_type"],
                    "unit_number": aggregate["unit_number"],
                    "unit_key": aggregate["unit_key"],
                    "unit_topic": aggregate["unit_topic"],
                    "reference_variants": aggregate["reference_variants"],
                    "reference_variants_text": " | ".join(aggregate["reference_variants"]),
                    "focus_terms": focus_terms,
                    "focus_terms_text": " | ".join(focus_terms),
                    "sample_questions": sample_questions,
                    "sample_questions_text": " | ".join(sample_questions),
                    "lead_chunk_id": branch_leads.get(str(aggregate["branch_id"]), {}).get("chunk_id"),
                    "lead_chunk_kind": branch_leads.get(str(aggregate["branch_id"]), {}).get("chunk_kind"),
                    "branch_depth": len(aggregate["hierarchy_path"]),
                    "child_count": len(set(aggregate["child_chunk_ids"])),
                    "page_start": aggregate["page_start"],
                    "page_end": aggregate["page_end"],
                    "inferred_only": aggregate["inferred_only"],
                    "is_boilerplate": boilerplate_profile["is_boilerplate"],
                    "boilerplate_score": boilerplate_profile["boilerplate_score"],
                    "boilerplate_reasons": boilerplate_profile["boilerplate_reasons"],
                }
            )

        branch_documents.sort(key=lambda item: (int(item["chunk_order"]), int(item["branch_depth"])))
        return branch_documents

    def _chunk_index_properties(self) -> dict[str, object]:
        return {
            "chunk_id": {"type": "keyword"},
            "document_id": {"type": "keyword"},
            "version_id": {"type": "keyword"},
            "version_status": {"type": "keyword"},
            "document_class": {"type": "keyword"},
            "document_family": {"type": "keyword"},
            "chunking_strategy": {"type": "keyword"},
            "classification_source": {"type": "keyword"},
            "chunk_kind": {"type": "keyword"},
            "mime_type": {"type": "keyword"},
            "source": {"type": "keyword"},
            "source_authority": {"type": "keyword"},
            "section_path": {"type": "keyword"},
            "hierarchy_path": {"type": "keyword"},
            "parent_path": {"type": "keyword"},
            "path_text": {"type": "text"},
            "retrieval_context": {"type": "text"},
            "contextualized_text": {"type": "text"},
            "chunk_summary_context": {"type": "text"},
            "branch_summary_context": {"type": "text"},
            "raw_text": {"type": "text"},
            "block_refs": {"type": "keyword"},
            "source_block_refs": {"type": "keyword"},
            "source_block_types": {"type": "keyword"},
            "acl_users": {"type": "keyword"},
            "acl_groups": {"type": "keyword"},
            "chunk_order": {"type": "integer"},
            "approx_tokens": {"type": "integer"},
            "page_start": {"type": "integer"},
            "page_end": {"type": "integer"},
            "book_label": {"type": "keyword"},
            "title_label": {"type": "keyword"},
            "chapter_label": {"type": "keyword"},
            "article_label": {"type": "keyword"},
            "section_label": {"type": "keyword"},
            "subsection_label": {"type": "keyword"},
            "document_title": {"type": "text"},
            "canonical_label": {"type": "text"},
            "unit_type": {"type": "keyword"},
            "unit_number": {"type": "keyword"},
            "unit_key": {"type": "keyword"},
            "unit_topic": {"type": "text"},
            "evidence_role": {"type": "keyword"},
            "evidence_role_confidence": {"type": "float"},
            "evidence_role_substantive": {"type": "float"},
            "evidence_role_reference": {"type": "float"},
            "evidence_role_editorial": {"type": "float"},
            "reference_variants": {"type": "text"},
            "reference_variants_text": {"type": "text"},
            "focus_terms": {"type": "keyword"},
            "focus_terms_text": {"type": "text"},
            "branch_keywords": {"type": "keyword"},
            "branch_keywords_text": {"type": "text"},
            "sample_questions": {"type": "text"},
            "sample_questions_text": {"type": "text"},
            "branch_keys": {"type": "keyword"},
            "branch_lead_keys": {"type": "keyword"},
            "is_branch_lead": {"type": "boolean"},
            "inferred_only": {"type": "boolean"},
            "is_boilerplate": {"type": "boolean"},
            "boilerplate_score": {"type": "float"},
            "boilerplate_reasons": {"type": "keyword"},
        }

    def _branch_index_properties(self) -> dict[str, object]:
        return {
            **self._chunk_index_properties(),
            "branch_id": {"type": "keyword"},
            "branch_key": {"type": "keyword"},
            "parent_branch_key": {"type": "keyword"},
            "branch_primary_kind": {"type": "keyword"},
            "lead_chunk_id": {"type": "keyword"},
            "lead_chunk_kind": {"type": "keyword"},
            "branch_depth": {"type": "integer"},
            "child_count": {"type": "integer"},
        }

    def _document_title(self, storage_uri: str) -> str:
        return Path(storage_uri or "").name or "documento"

    def _derive_unit_metadata(
        self,
        *,
        document_class: str,
        hierarchy_path: list[str],
        metadata: dict[str, object],
    ) -> tuple[str, str, str]:
        inferred = self._infer_structural_labels(hierarchy_path)
        article_label = str(metadata.get("article_label") or inferred["article_label"] or "").strip()
        section_label = str(metadata.get("section_label") or inferred["section_label"] or "").strip()
        subsection_label = str(
            metadata.get("subsection_label") or inferred["subsection_label"] or ""
        ).strip()
        chapter_label = str(metadata.get("chapter_label") or inferred["chapter_label"] or "").strip()
        title_label = str(metadata.get("title_label") or inferred["title_label"] or "").strip()
        book_label = str(metadata.get("book_label") or inferred["book_label"] or "").strip()
        recommendation_label = str(inferred["recommendation_label"] or "").strip()
        fallback = (hierarchy_path[-1] if hierarchy_path else "Documento").strip()

        if article_label:
            return article_label, "article", self._extract_label_number(article_label)

        if recommendation_label:
            return recommendation_label, "recommendation", self._extract_label_number(recommendation_label)

        subsection_number = self._extract_label_number(subsection_label)

        if section_label and re.match(r"^secci[oó]n\s+\d+", section_label, re.IGNORECASE):
            return section_label, "section", self._extract_label_number(section_label)

        if re.search(r"\banexo\b|\bap[eé]ndice\b", fallback, re.IGNORECASE):
            return fallback, "annex", self._extract_label_number(fallback)
        if re.match(r"^cap[ií]tulo\b", chapter_label or fallback, re.IGNORECASE):
            label = chapter_label or fallback
            return label, "chapter", self._extract_label_number(label)
        if re.match(r"^t[ií]tulo\b", title_label or fallback, re.IGNORECASE):
            label = title_label or fallback
            return label, "title", self._extract_label_number(label)
        if re.match(r"^libro\b", book_label or fallback, re.IGNORECASE):
            label = book_label or fallback
            return label, "book", self._extract_label_number(label)
        if subsection_label:
            return subsection_label, "subsection", subsection_number
        if section_label:
            return section_label, "section_group", self._extract_label_number(section_label)
        return fallback, "document", self._extract_label_number(fallback)

    def _build_sample_questions(
        self,
        *,
        document_title: str,
        document_class: str,
        canonical_label: str,
        unit_type: str,
        unit_number: str,
        parent_path: list[str],
        unit_topic: str,
    ) -> list[str]:
        questions: list[str] = []
        short_title = self._document_title_short(document_title)
        parent_label = parent_path[-1] if parent_path else ""
        lookup_noun = self._lookup_unit_noun(unit_type)

        def add(question: str) -> None:
            cleaned = " ".join((question or "").split()).strip()
            if cleaned and cleaned not in questions:
                questions.append(cleaned)

        if unit_type == "article":
            add(f"Que establece {canonical_label} de {short_title}?")
            add(f"Que dice {canonical_label}?")
        elif unit_type == "recommendation":
            label = canonical_label or f"Recomendacion {unit_number}".strip()
            add(f"Que exige {label} de {short_title}?")
            add(f"Como se evalua tecnicamente {label}?")
        elif unit_type == "section":
            add(f"Que establece {canonical_label} de {short_title}?")
            add(f"Que dice {canonical_label}?")
        elif unit_type == "subsection":
            add(f"Que dice {canonical_label} en {short_title}?")
            if parent_label:
                add(f"Que establece {canonical_label} de {parent_label}?")
        elif unit_type == "annex":
            add(f"Que contiene {canonical_label} de {short_title}?")
            add(f"Que dice {canonical_label}?")
        else:
            add(f"Que establece {canonical_label} de {short_title}?")
            add(f"Que regula {canonical_label}?")

        if unit_topic:
            if lookup_noun:
                add(f"Que {lookup_noun} aborda {unit_topic}?")
            add(f"Donde se regula {unit_topic} en {short_title}?")
            if parent_label and unit_type in {"section", "subsection", "recommendation"}:
                add(f"Que trata {unit_topic} dentro de {parent_label}?")

        if document_class == "technical_standard" and parent_label and unit_type in {"section", "subsection"}:
            add(f"Que trata {canonical_label} dentro de {parent_label}?")

        return questions[:4]

    def _unit_key(self, *, unit_type: str, unit_number: str) -> str:
        if unit_type and unit_number:
            return f"{unit_type}:{unit_number.upper()}"
        return unit_type or "document"

    def _reference_variants(
        self,
        *,
        canonical_label: str,
        unit_type: str,
        unit_number: str,
        hierarchy_path: list[str],
        unit_topic: str,
    ) -> list[str]:
        variants: list[str] = []

        def add(value: str) -> None:
            cleaned = " ".join((value or "").split()).strip()
            if cleaned and cleaned not in variants:
                variants.append(cleaned)

        add(canonical_label)
        add(unit_topic)
        for label in hierarchy_path[-2:]:
            add(label)

        if unit_type == "article" and unit_number:
            add(f"Arto. {unit_number}")
            add(f"Art. {unit_number}")
            add(f"Articulo {unit_number}")
        elif unit_type == "section" and unit_number:
            add(f"Sección {unit_number}")
            add(f"Seccion {unit_number}")
        elif unit_type == "recommendation" and unit_number:
            add(f"Recomendación {unit_number}")
            add(f"Recomendacion {unit_number}")
            add(f"R. {unit_number}")
        elif unit_type == "annex" and unit_number:
            add(f"Anexo {unit_number}")
            add(f"Apéndice {unit_number}")
            add(f"Apendice {unit_number}")
        elif unit_type == "chapter" and unit_number:
            add(f"Capítulo {unit_number}")
            add(f"Capitulo {unit_number}")
        elif unit_type == "title" and unit_number:
            add(f"Título {unit_number}")
            add(f"Titulo {unit_number}")
        elif unit_type == "book" and unit_number:
            add(f"Libro {unit_number}")
        elif unit_type == "paragraph" and unit_number:
            add(f"Párrafo {unit_number}")
            add(f"Parrafo {unit_number}")

        return variants[:8]

    def _infer_structural_labels(self, hierarchy_path: list[str]) -> dict[str, str]:
        labels = {
            "book_label": "",
            "title_label": "",
            "chapter_label": "",
            "article_label": "",
            "section_label": "",
            "subsection_label": "",
            "recommendation_label": "",
        }
        path = self._coerce_path(hierarchy_path)
        used: set[str] = set()

        for label in path:
            normalized = self._normalize(label)
            if not labels["book_label"] and re.match(r"^libro\b", normalized):
                labels["book_label"] = label
                used.add(label)
                continue
            if not labels["title_label"] and re.match(r"^titulo\b", normalized):
                labels["title_label"] = label
                used.add(label)
                continue
            if not labels["chapter_label"] and re.match(r"^capitulo\b", normalized):
                labels["chapter_label"] = label
                used.add(label)
                continue
            if not labels["article_label"] and re.match(
                r"^(?:art\.?|arto\.?|articulo)\s*\d+",
                normalized,
            ):
                labels["article_label"] = label
                used.add(label)
                continue
            if not labels["section_label"] and re.match(r"^seccion\s+\d+", normalized):
                labels["section_label"] = label
                used.add(label)
                continue
            if (
                not labels["recommendation_label"]
                and re.match(r"^(?:recomendacion\s+\d+|\d+(?:\.\d+)?\s*[\.\-])", normalized)
            ):
                labels["recommendation_label"] = label
                used.add(label)

        if len(path) > 1 and not labels["subsection_label"]:
            for label in reversed(path):
                if label not in used:
                    labels["subsection_label"] = label
                    break

        return labels

    def _extract_unit_topic(
        self,
        *,
        canonical_label: str,
        unit_type: str,
        hierarchy_path: list[str],
    ) -> str:
        candidates: list[str] = []
        if canonical_label:
            candidates.append(canonical_label)
        for label in reversed(self._coerce_path(hierarchy_path)):
            if label and label not in candidates:
                candidates.append(label)

        for candidate in candidates:
            topic = self._strip_structural_prefix(candidate, unit_type)
            cleaned = " ".join((topic or "").replace("*", " ").split()).strip(" -:;,.")
            if not cleaned:
                continue
            if unit_type in {"subsection", "document"}:
                if len(cleaned.split()) >= 2:
                    return cleaned
                continue
            if self._normalize(cleaned) == self._normalize(candidate):
                continue
            if len(cleaned.split()) >= 2:
                return cleaned
        return ""

    def _strip_structural_prefix(self, label: str, unit_type: str) -> str:
        text = " ".join((label or "").split()).strip()
        if not text:
            return ""

        patterns = {
            "article": r"^(?:art\.?|arto\.?|articulo)\s*\d+(?:\.\d+)?\s*",
            "section": r"^secci[oó]n\s+\d+(?:\.\d+)?\s*",
            "recommendation": r"^(?:recomendaci[oó]n\s+\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*[\.\-])\s*",
            "annex": r"^(?:anexo|ap[eé]ndice)\s+[ivxlcdm\d]+\s*",
            "chapter": r"^cap[ií]tulo\s+[ivxlcdm\d]+\s*",
            "title": r"^t[ií]tulo\s+[ivxlcdm\d]+\s*",
            "book": r"^libro\s+[ivxlcdm\d]+\s*",
            "paragraph": r"^p[aá]rrafo\s+\d+(?:\.\d+)?\s*",
        }
        pattern = patterns.get(unit_type)
        if pattern:
            stripped = re.sub(pattern, "", text, flags=re.IGNORECASE).strip(" -:;,.")
            if stripped:
                return stripped
        return text

    def _lookup_unit_noun(self, unit_type: str) -> str:
        return {
            "article": "articulo",
            "section": "seccion",
            "subsection": "apartado",
            "recommendation": "recomendacion",
            "annex": "anexo",
            "chapter": "capitulo",
            "title": "titulo",
            "book": "libro",
            "paragraph": "parrafo",
        }.get(unit_type, "apartado")

    def _focus_terms(
        self,
        *,
        hierarchy_path: list[str],
        retrieval_context: str,
        raw_text: str,
        supplemental_texts: list[str] | None = None,
        limit: int = 12,
    ) -> list[str]:
        frequencies: Counter[str] = Counter()
        for text in [*hierarchy_path, retrieval_context, *(supplemental_texts or []), raw_text[:1200]]:
            for token in TOKEN_PATTERN.findall(self._normalize(text)):
                if token in FOCUS_STOPWORDS or token.isdigit() or len(token) < 4:
                    continue
                frequencies[token] += 1

        ordered = [
            token
            for token, _ in frequencies.most_common(limit * 2)
            if token not in FOCUS_STOPWORDS
        ]
        return ordered[:limit]

    def _document_title_short(self, document_title: str) -> str:
        title = Path(document_title or "").stem
        return re.sub(r"[-_]+", " ", title).strip() or "el documento"

    def _embedding_text(self, payload: dict[str, object]) -> str:
        summary_context = str(payload.get("chunk_summary_context") or "").strip()
        branch_summary_context = str(payload.get("branch_summary_context") or "").strip()
        branch_keywords_text = str(payload.get("branch_keywords_text") or "").strip()
        contextualized_text = str(payload.get("contextualized_text") or "").strip()
        raw_text = str(payload.get("raw_text") or "").strip()
        return "\n".join(
            part
            for part in (
                summary_context,
                branch_summary_context,
                branch_keywords_text,
                contextualized_text,
                raw_text,
            )
            if part
        ).strip()

    def _extract_label_number(self, label: str) -> str:
        text = (label or "").strip()
        match = re.search(r"\b(\d+)\b", text)
        if match:
            return match.group(1)
        roman = re.search(r"\b([IVXLCM]+)\b", text, re.IGNORECASE)
        if roman:
            return roman.group(1).upper()
        return ""

    def _branch_keys_for_path(
        self,
        *,
        document_id: str,
        version_id: str,
        hierarchy_path: list[str],
    ) -> list[str]:
        keys: list[str] = []
        for depth in range(1, len(hierarchy_path) + 1):
            keys.append(
                self._branch_id(
                    document_id=document_id,
                    version_id=version_id,
                    hierarchy_path=hierarchy_path[:depth],
                )
            )
        return keys

    def _annotate_branch_leads(
        self,
        *,
        document_id: str,
        version_id: str,
        chunk_documents: list[dict[str, object]],
    ) -> dict[str, dict[str, str]]:
        best_by_branch: dict[str, tuple[tuple[int, int, int], str, str]] = {}
        by_chunk_id = {
            str(document["chunk_id"]): document for document in chunk_documents
        }

        for chunk_document in chunk_documents:
            hierarchy_path = self._coerce_path(chunk_document.get("hierarchy_path"))
            if not hierarchy_path:
                continue
            chunk_id = str(chunk_document["chunk_id"])
            chunk_kind = str(chunk_document.get("chunk_kind") or "unknown")
            chunk_order = int(chunk_document.get("chunk_order") or 0)

            for depth in range(1, len(hierarchy_path) + 1):
                branch_id = self._branch_id(
                    document_id=document_id,
                    version_id=version_id,
                    hierarchy_path=hierarchy_path[:depth],
                )
                score = (
                    self._branch_lead_kind_priority(chunk_kind),
                    len(hierarchy_path) - depth,
                    chunk_order,
                )
                current = best_by_branch.get(branch_id)
                if current is None or score < current[0]:
                    best_by_branch[branch_id] = (score, chunk_id, chunk_kind)

        branch_leads: dict[str, dict[str, str]] = {}
        for branch_id, (_, chunk_id, chunk_kind) in best_by_branch.items():
            branch_leads[branch_id] = {"chunk_id": chunk_id, "chunk_kind": chunk_kind}
            chunk_document = by_chunk_id.get(chunk_id)
            if chunk_document is None:
                continue
            branch_lead_keys = list(chunk_document.get("branch_lead_keys") or [])
            if branch_id not in branch_lead_keys:
                branch_lead_keys.append(branch_id)
            chunk_document["branch_lead_keys"] = branch_lead_keys
            chunk_document["is_branch_lead"] = True

        return branch_leads

    def _branch_lead_kind_priority(self, chunk_kind: str) -> int:
        priorities = {
            "section": 0,
            "subsection": 1,
            "article": 1,
            "unknown": 2,
            "preamble": 3,
            "table": 4,
            "reference_table": 5,
        }
        return priorities.get((chunk_kind or "").lower(), 2)

    def _branch_id(
        self,
        *,
        document_id: str,
        version_id: str,
        hierarchy_path: list[str],
    ) -> str:
        branch_seed = " > ".join(self._coerce_path(hierarchy_path))
        return str(uuid5(NAMESPACE_URL, f"rag-issdhu:branch:{document_id}:{version_id}:{branch_seed}"))

    def _representative_snippet(self, chunk_document: dict[str, object]) -> str:
        chunk_kind = str(chunk_document.get("chunk_kind") or "").lower()
        if chunk_kind in {"table", "reference_table"}:
            return ""
        return self._compress_text(str(chunk_document.get("raw_text") or ""), 280)

    def _branch_summary(
        self,
        *,
        document_title: str,
        path: list[str],
        canonical_label: str,
        unit_type: str,
        unit_topic: str,
        child_labels: list[str],
        snippets: list[str],
    ) -> str:
        parts = [
            f"Documento: {document_title}.",
            f"Ruta: {' > '.join(path)}." if path else "",
            f"Unidad: {canonical_label} ({unit_type})." if canonical_label else "",
            f"Tema principal: {unit_topic}." if unit_topic else "",
            f"Subtemas: {'; '.join(child_labels)}." if child_labels else "",
            f"Contenido representativo: {' '.join(snippets)}" if snippets else "",
        ]
        return " ".join(part for part in parts if part).strip()

    def _branch_summary_context(
        self,
        *,
        document_title: str,
        path: list[str],
        canonical_label: str,
        unit_type: str,
        unit_number: str,
        unit_topic: str,
        child_labels: list[str],
        snippets: list[str],
    ) -> str:
        short_title = self._document_title_short(document_title)
        parts = [
            f"Documento {short_title}.",
            f"Ruta estructural {' > '.join(path)}." if path else "",
            f"Unidad central {canonical_label} ({unit_type})." if canonical_label else "",
            f"Numero {unit_number}." if unit_number else "",
            f"Tema principal {unit_topic}." if unit_topic else "",
            f"Subtemas visibles: {', '.join(child_labels[:4])}." if child_labels else "",
        ]
        lead_snippet = " ".join(snippets[:2]).strip()
        if lead_snippet:
            parts.append(f"Contenido representativo: {self._compress_text(lead_snippet, 320)}")
        return " ".join(part for part in parts if part).strip()

    def _branch_retrieval_context(
        self,
        *,
        document_title: str,
        path: list[str],
        canonical_label: str,
        unit_type: str,
        unit_number: str,
        unit_topic: str,
        child_labels: list[str],
    ) -> str:
        parts = [
            f"Documento {document_title}",
            f"rama {' > '.join(path)}" if path else "",
            f"unidad {canonical_label}" if canonical_label else "",
            f"tipo {unit_type}" if unit_type else "",
            f"numero {unit_number}" if unit_number else "",
            f"tema {unit_topic}" if unit_topic else "",
            f"subtemas {', '.join(child_labels)}" if child_labels else "",
        ]
        return "; ".join(part for part in parts if part).strip()

    def _branch_keywords(
        self,
        *,
        hierarchy_path: list[str],
        canonical_label: str,
        unit_type: str,
        unit_topic: str,
        child_labels: list[str],
        snippets: list[str],
        limit: int = 12,
    ) -> list[str]:
        blocked = {
            "art",
            "arto",
            "articulo",
            "capitulo",
            "titulo",
            "libro",
            "seccion",
            "recomendacion",
            "anexo",
            "documento",
            unit_type.strip().lower(),
        }
        frequencies: Counter[str] = Counter()
        for text in [*hierarchy_path, canonical_label, unit_topic, *child_labels, *snippets[:2]]:
            for token in TOKEN_PATTERN.findall(self._normalize(text)):
                if (
                    token in FOCUS_STOPWORDS
                    or token in blocked
                    or token.isdigit()
                    or len(token) < 4
                    or re.fullmatch(r"[ivxlcdm]+", token)
                ):
                    continue
                frequencies[token] += 1
        return [token for token, _ in frequencies.most_common(limit)]

    def _boilerplate_profile(
        self,
        *,
        hierarchy_path: list[str],
        canonical_label: str,
        unit_type: str,
        raw_text: str,
        chunk_kind: str = "",
        child_labels: list[str] | None = None,
        source_block_types: Counter[str] | None = None,
    ) -> dict[str, object]:
        normalized_path = self._normalize(" > ".join(hierarchy_path))
        normalized_label = self._normalize(canonical_label)
        normalized_text = self._normalize(" ".join((raw_text or "").split()))
        marker_probe_text = normalized_text.replace("contenido representativo", " ")
        reasons: list[str] = []
        score = 0.0

        marker_hits = [
            marker
            for marker in (
                "indice",
                "contenido",
                "tabla de contenido",
                "sumario",
                "portada",
            )
            if marker in normalized_path or marker in normalized_label or marker in marker_probe_text[:600]
        ]
        if marker_hits:
            score += 0.9
            reasons.append("marker:" + ",".join(marker_hits[:3]))

        if chunk_kind in {"reference_table"}:
            score += 0.65
            reasons.append("chunk_kind:reference_table")

        if source_block_types and source_block_types.get("reference_table"):
            score += 0.55
            reasons.append("source_block_type:reference_table")

        if unit_type in {"document", "title"} and len(hierarchy_path) <= 1 and len(normalized_text) < 320:
            score += 0.35
            reasons.append("short_top_level")

        if re.search(r"\.{3,}|\bpag(?:ina)?\b", normalized_text) and len(normalized_text) < 800:
            score += 0.35
            reasons.append("toc_pattern")

        if child_labels and len(child_labels) >= 8 and len(normalized_text) < 700:
            score += 0.2
            reasons.append("dense_child_listing")

        score = round(min(1.0, score), 3)
        return {
            "is_boilerplate": score >= 0.75,
            "boilerplate_score": score,
            "boilerplate_reasons": reasons,
        }

    def _branch_primary_kind(
        self,
        counter: Counter[str],
        *,
        unit_type: str = "",
        source_block_types: Counter[str] | None = None,
    ) -> str:
        preferred_order = ("section", "subsection", "article", "preamble", "reference_table", "table", "unknown")
        for kind in preferred_order:
            if counter.get(kind):
                return kind
        if unit_type in {"section", "section_group", "subsection"}:
            return "section"
        if unit_type == "article":
            return "article"
        if source_block_types:
            if source_block_types.get("reference_table"):
                return "reference_table"
            if source_block_types.get("table"):
                return "table"
            if source_block_types.get("preamble"):
                return "preamble"
            if source_block_types.get("section"):
                return "section"
        return "unknown"

    def _apply_branch_role_profiles(
        self,
        *,
        branch_documents: list[dict[str, object]],
        chunk_documents: list[dict[str, object]],
    ) -> None:
        if not branch_documents or not chunk_documents:
            return

        chunk_scores = {
            str(document["chunk_id"]): {
                role: float(document.get(f"evidence_role_{role}") or 0.0)
                for role in VALID_EVIDENCE_ROLES
            }
            for document in chunk_documents
        }

        default_role = self.evidence_role_classifier.default_role()
        for branch_document in branch_documents:
            block_refs = [str(value) for value in (branch_document.get("block_refs") or []) if str(value).strip()]
            role_totals = {role: 0.0 for role in VALID_EVIDENCE_ROLES}
            matched = 0
            for block_ref in block_refs:
                scores = chunk_scores.get(block_ref)
                if not scores:
                    continue
                matched += 1
                for role, value in scores.items():
                    role_totals[role] += value

            if matched:
                averaged = {
                    role: round(role_totals[role] / matched, 4)
                    for role in VALID_EVIDENCE_ROLES
                }
                dominant_role = max(averaged.items(), key=lambda item: item[1])[0]
                confidence = round(float(averaged[dominant_role]), 2)
            else:
                averaged = {role: 0.0 for role in VALID_EVIDENCE_ROLES}
                dominant_role = default_role
                confidence = 0.0

            branch_document["evidence_role"] = dominant_role
            branch_document["evidence_role_confidence"] = confidence
            for role in VALID_EVIDENCE_ROLES:
                branch_document[f"evidence_role_{role}"] = averaged.get(role, 0.0)

    def _compress_text(self, text: str, limit: int) -> str:
        normalized = " ".join((text or "").split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."

    def _normalize(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text or "")
        return "".join(char for char in normalized if not unicodedata.combining(char)).lower()

    def _coerce_path(self, raw_path: object) -> list[str]:
        if isinstance(raw_path, list):
            return [str(item).strip() for item in raw_path if str(item).strip()]
        if raw_path is None:
            return []
        text = str(raw_path).strip()
        return [text] if text else []

    def _approx_tokens(self, text: str) -> int:
        return max(1, math.ceil(len((text or "").split()) * 1.2))

    def _truncate_reason(self, reason: str) -> str:
        reason = (reason or "").strip()
        return reason if len(reason) <= 240 else f"{reason[:237]}..."

    def _qdrant_point_id(self, chunk_id: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"rag-issdhu:{chunk_id}"))
