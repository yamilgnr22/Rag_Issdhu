import hashlib
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy.orm import Session

from app.models.contracts import (
    ACL,
    BlockContract,
    DocumentContract,
    DocumentVersionListResponse,
    IngestionFilesystemRequest,
    IngestionResponse,
    DocumentVersionListItem,
    VersionDeleteResponse,
    VersionStatusUpdateResponse,
    VersionContract,
)
from app.models.database import BlockRecord, ChunkRecord, DocumentRecord, VersionRecord
from app.services.document_classification import DocumentClassifier
from app.services.document_isolation import DocumentIsolationService
from app.services.extractors import ExtractionPipeline
from app.services.indexing import OpenSearchIndexClient, QdrantIndexClient
from app.services.storage import build_object_store


class IngestionService:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.storage = build_object_store(settings)
        self.extractor = ExtractionPipeline(settings)
        self.isolator = DocumentIsolationService(settings)
        self.classifier = DocumentClassifier(settings)
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
        )
        self.branch_opensearch = OpenSearchIndexClient(
            settings,
            index_name=settings.resolved_opensearch_branch_index_name,
            id_field="branch_id",
        )

    def ingest_upload(
        self,
        *,
        db: Session,
        filename: str,
        content: bytes,
        source: str,
        source_authority: str,
        language: str | None,
        version_status: str,
        acl: ACL,
    ) -> IngestionResponse:
        return self._ingest(
            db=db,
            filename=filename,
            content=content,
            source=source,
            source_authority=source_authority,
            language=language,
            version_status=version_status,
            acl=acl,
        )

    def ingest_from_filesystem(self, db: Session, request: IngestionFilesystemRequest) -> IngestionResponse:
        path = Path(request.path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"File not found: {request.path}")
        return self._ingest(
            db=db,
            filename=path.name,
            content=path.read_bytes(),
            source=request.source,
            source_authority=request.source_authority,
            language=request.language,
            version_status=request.version_status,
            acl=ACL(users=request.acl_users, groups=request.acl_groups),
        )

    def list_manual_review(self, db: Session) -> list[dict[str, object]]:
        items = (
            db.query(VersionRecord)
            .filter(VersionRecord.review_required.is_(True))
            .order_by(VersionRecord.created_at.desc())
            .all()
        )
        return [
            {
                "document_id": item.document_id,
                "version_id": item.version_id,
                "extraction_status": item.extraction_status,
                "review_reason": item.review_reason,
                "document_class": item.document_class,
                "document_family": item.document_family,
                "chunking_strategy": item.chunking_strategy,
                "classification_source": item.classification_source,
                "storage_uri": item.storage_uri,
                "created_at": item.created_at.isoformat(),
            }
            for item in items
        ]

    def list_versions(
        self,
        db: Session,
        *,
        limit: int = 50,
        version_status: str | None = None,
        document_class: str | None = None,
        q: str | None = None,
    ) -> DocumentVersionListResponse:
        query = (
            db.query(VersionRecord, DocumentRecord)
            .join(DocumentRecord, VersionRecord.document_id == DocumentRecord.document_id)
            .order_by(VersionRecord.created_at.desc())
        )
        if version_status:
            query = query.filter(VersionRecord.version_status == version_status)
        if document_class:
            query = query.filter(VersionRecord.document_class == document_class)
        if q:
            term = f"%{q.strip()}%"
            query = query.filter(
                (VersionRecord.storage_uri.ilike(term))
                | (VersionRecord.version_id.ilike(term))
                | (VersionRecord.document_id.ilike(term))
            )

        rows = query.limit(max(1, min(limit, 500))).all()
        items = [
            DocumentVersionListItem(
                document_id=version.document_id,
                version_id=version.version_id,
                version_status=version.version_status,
                document_class=version.document_class,
                document_family=version.document_family,
                storage_uri=version.storage_uri,
                artifact_uri=version.artifact_uri,
                extraction_status=version.extraction_status,
                chunking_strategy=version.chunking_strategy,
                classification_source=version.classification_source,
                classification_confidence=version.classification_confidence,
                review_required=bool(version.review_required),
                latest_for_document=document.latest_version_id == version.version_id,
                created_at=version.created_at,
            )
            for version, document in rows
        ]
        return DocumentVersionListResponse(total=len(items), items=items)

    def update_version_status(
        self,
        *,
        db: Session,
        version_id: str,
        version_status: str,
    ) -> VersionStatusUpdateResponse:
        version = db.query(VersionRecord).filter(VersionRecord.version_id == version_id).one_or_none()
        if version is None:
            raise ValueError(f"Version not found: {version_id}")

        document = (
            db.query(DocumentRecord)
            .filter(DocumentRecord.document_id == version.document_id)
            .one_or_none()
        )
        if document is None:
            raise ValueError(f"Document not found for version: {version_id}")

        previous_status = version.version_status
        version.version_status = version_status

        if version_status == "active":
            document.latest_version_id = version.version_id
        elif document.latest_version_id == version.version_id:
            replacement = (
                db.query(VersionRecord)
                .filter(
                    VersionRecord.document_id == version.document_id,
                    VersionRecord.version_status == "active",
                    VersionRecord.version_id != version.version_id,
                )
                .order_by(VersionRecord.created_at.desc())
                .first()
            )
            document.latest_version_id = replacement.version_id if replacement is not None else None

        db.commit()

        return VersionStatusUpdateResponse(
            version_id=version.version_id,
            document_id=version.document_id,
            previous_status=previous_status,
            version_status=version.version_status,
        )

    def delete_version(
        self,
        *,
        db: Session,
        version_id: str,
    ) -> VersionDeleteResponse:
        version = db.query(VersionRecord).filter(VersionRecord.version_id == version_id).one_or_none()
        if version is None:
            raise ValueError(f"Version not found: {version_id}")

        document = (
            db.query(DocumentRecord)
            .filter(DocumentRecord.document_id == version.document_id)
            .one_or_none()
        )
        if document is None:
            raise ValueError(f"Document not found for version: {version_id}")

        deleted_prefixes = [
            f"documents/{version.document_id}/{version.version_id}",
            f"artifacts/{version.document_id}/{version.version_id}",
        ]
        warnings: list[str] = []
        deleted_indexes = True
        for label, action in (
            ("qdrant_chunks", lambda: self.qdrant.delete_by_version_id(version_id=version.version_id)),
            ("qdrant_branches", lambda: self.branch_qdrant.delete_by_version_id(version_id=version.version_id)),
            ("opensearch_chunks", lambda: self.opensearch.delete_by_version_id(version_id=version.version_id)),
            ("opensearch_branches", lambda: self.branch_opensearch.delete_by_version_id(version_id=version.version_id)),
        ):
            try:
                action()
            except Exception as exc:
                deleted_indexes = False
                warnings.append(f"{label}:{type(exc).__name__}")

        for prefix in deleted_prefixes:
            try:
                self.storage.delete_prefix(prefix)
            except Exception as exc:
                warnings.append(f"storage:{prefix}:{type(exc).__name__}")

        deleted_blocks = (
            db.query(BlockRecord).filter(BlockRecord.version_id == version.version_id).delete()
        )
        deleted_chunks = (
            db.query(ChunkRecord).filter(ChunkRecord.version_id == version.version_id).delete()
        )
        db.delete(version)
        db.flush()

        remaining_versions = (
            db.query(VersionRecord)
            .filter(VersionRecord.document_id == document.document_id)
            .order_by(VersionRecord.created_at.desc())
            .all()
        )

        deleted_document = False
        if not remaining_versions:
            db.delete(document)
            deleted_document = True
        else:
            replacement = next(
                (item for item in remaining_versions if item.version_status == "active"),
                None,
            )
            document.latest_version_id = replacement.version_id if replacement is not None else None

        db.commit()

        return VersionDeleteResponse(
            version_id=version.version_id,
            document_id=version.document_id,
            deleted_chunks=int(deleted_chunks),
            deleted_blocks=int(deleted_blocks),
            deleted_indexes=deleted_indexes,
            deleted_storage_prefixes=deleted_prefixes,
            deleted_document=deleted_document,
            warnings=warnings,
        )

    def _ingest(
        self,
        *,
        db: Session,
        filename: str,
        content: bytes,
        source: str,
        source_authority: str,
        language: str | None,
        version_status: str,
        acl: ACL,
    ) -> IngestionResponse:
        now = datetime.now(timezone.utc)
        document_id = f"doc_{uuid4().hex[:12]}"
        version_id = f"v_{uuid4().hex[:12]}"
        checksum = hashlib.sha256(content).hexdigest()
        mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        final_language = language or "und"

        storage_object = self.storage.save_bytes(
            f"documents/{document_id}/{version_id}/{filename}",
            content,
        )
        extraction = self.extractor.extract(filename, content)
        extraction = self.isolator.isolate(filename=filename, extraction=extraction)
        classification = self.classifier.classify(
            filename=filename,
            mime_type=mime_type,
            blocks=extraction.blocks,
        )

        blocks: list[BlockContract] = []
        for order, block in enumerate(extraction.blocks, start=1):
            blocks.append(
                BlockContract(
                    block_id=f"b_{uuid4().hex[:12]}",
                    document_id=document_id,
                    version_id=version_id,
                    block_type=block.block_type,
                    text=block.text,
                    page=block.page,
                    section_path=block.section_path,
                )
            )

        artifact_uri = None
        if blocks:
            artifact = self.storage.save_json(
                f"artifacts/{document_id}/{version_id}/blocks.json",
                [block.model_dump(mode="json") for block in blocks],
            )
            artifact_uri = artifact.uri

        document_contract = DocumentContract(
            document_id=document_id,
            source=source,
            source_authority=source_authority,
            mime_type=mime_type,
            language=final_language,
            acl=acl,
            created_at=now,
        )
        version_contract = VersionContract(
            version_id=version_id,
            document_id=document_id,
            checksum=f"sha256:{checksum}",
            version_status=version_status,
            created_at=now,
        )

        db_document = DocumentRecord(
            document_id=document_id,
            source=source,
            source_authority=source_authority,
            mime_type=mime_type,
            language=final_language,
            document_class=classification.document_class,
            document_family=classification.document_family,
            acl=acl.model_dump(),
            created_at=now.replace(tzinfo=None),
            latest_version_id=version_id,
        )
        db_version = VersionRecord(
            version_id=version_id,
            document_id=document_id,
            checksum=version_contract.checksum,
            version_status=version_status,
            storage_uri=storage_object.uri,
            artifact_uri=artifact_uri,
            extraction_status="manual_review" if extraction.review_required else "extracted",
            review_required=extraction.review_required,
            review_reason=extraction.review_reason,
            extraction_backend=extraction.backend,
            extraction_quality=extraction.quality,
            text_length=extraction.text_length,
            document_class=classification.document_class,
            document_family=classification.document_family,
            chunking_strategy=classification.chunking_strategy,
            classification_source=classification.classification_source,
            classification_confidence=classification.confidence,
            classification_reason=classification.reason_short,
            created_at=now.replace(tzinfo=None),
        )
        db.add(db_document)
        db.add(db_version)
        for order, block in enumerate(blocks, start=1):
            db.add(
                BlockRecord(
                    block_id=block.block_id,
                    document_id=block.document_id,
                    version_id=block.version_id,
                    block_type=block.block_type,
                    text=block.text,
                    page=block.page,
                    section_path=block.section_path,
                    block_order=order,
                )
            )
        db.commit()

        return IngestionResponse(
            document=document_contract,
            version=version_contract,
            storage_uri=storage_object.uri,
            artifact_uri=artifact_uri,
            extraction_status="manual_review" if extraction.review_required else "extracted",
            review_required=extraction.review_required,
            review_reason=extraction.review_reason,
            blocks_count=len(blocks),
            text_length=extraction.text_length,
            document_class=classification.document_class,
            document_family=classification.document_family,
            chunking_strategy=classification.chunking_strategy,
            classification_source=classification.classification_source,
            classification_confidence=classification.confidence,
            classification_reason_short=classification.reason_short,
        )
