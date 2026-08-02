from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    source: Mapped[str] = mapped_column(String(64))
    source_authority: Mapped[str] = mapped_column(String(32))
    mime_type: Mapped[str] = mapped_column(String(255))
    language: Mapped[str] = mapped_column(String(16))
    document_class: Mapped[str | None] = mapped_column(String(32), nullable=True)
    document_family: Mapped[str | None] = mapped_column(String(32), nullable=True)
    acl: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    latest_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    versions: Mapped[list["VersionRecord"]] = relationship(back_populates="document")


class VersionRecord(Base):
    __tablename__ = "versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.document_id"), index=True)
    checksum: Mapped[str] = mapped_column(String(128), index=True)
    version_status: Mapped[str] = mapped_column(String(32))
    storage_uri: Mapped[str] = mapped_column(Text)
    artifact_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    extraction_status: Mapped[str] = mapped_column(String(32))
    review_required: Mapped[bool] = mapped_column(Boolean, default=False)
    review_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    extraction_backend: Mapped[str | None] = mapped_column(String(64), nullable=True)
    extraction_quality: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    text_length: Mapped[int] = mapped_column(Integer, default=0)
    document_class: Mapped[str | None] = mapped_column(String(32), nullable=True)
    document_family: Mapped[str | None] = mapped_column(String(32), nullable=True)
    chunking_strategy: Mapped[str | None] = mapped_column(String(64), nullable=True)
    classification_source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    classification_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    classification_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    document: Mapped[DocumentRecord] = relationship(back_populates="versions")
    blocks: Mapped[list["BlockRecord"]] = relationship(back_populates="version")
    chunks: Mapped[list["ChunkRecord"]] = relationship(back_populates="version")


class BlockRecord(Base):
    __tablename__ = "blocks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    block_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    document_id: Mapped[str] = mapped_column(String(64), index=True)
    version_id: Mapped[str] = mapped_column(ForeignKey("versions.version_id"), index=True)
    block_type: Mapped[str] = mapped_column(String(64))
    text: Mapped[str] = mapped_column(Text)
    page: Mapped[int] = mapped_column(Integer, default=1)
    section_path: Mapped[list[str]] = mapped_column(JSON)
    block_order: Mapped[int] = mapped_column(Integer, default=0)

    version: Mapped[VersionRecord] = relationship(back_populates="blocks")


class ChunkRecord(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    document_id: Mapped[str] = mapped_column(String(64), index=True)
    version_id: Mapped[str] = mapped_column(ForeignKey("versions.version_id"), index=True)
    block_refs: Mapped[list[str]] = mapped_column(JSON)
    text: Mapped[str] = mapped_column(Text)
    chunk_metadata: Mapped[dict] = mapped_column("metadata", JSON)
    acl: Mapped[dict] = mapped_column(JSON)
    index_status: Mapped[str] = mapped_column(String(32), default="pending")
    indexed_profile: Mapped[str | None] = mapped_column(String(128), nullable=True)
    chunk_order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    version: Mapped[VersionRecord] = relationship(back_populates="chunks")
