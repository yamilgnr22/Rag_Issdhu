import re
from collections import Counter
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from sqlalchemy.orm import Session

from app.models.contracts import ACL, ChunkContract, ChunkingRequest, ChunkingResponse
from app.models.database import BlockRecord, ChunkRecord, DocumentRecord, VersionRecord
from app.services.document_classification import DocumentClassifier, DocumentClassification
from app.services.storage import build_object_store


HEADING_PATTERNS = [
    re.compile(r"^CAP[IÍ]TULO\s+[IVXLC0-9]+", re.IGNORECASE),
    re.compile(r"^ARTO\.\s*\d+", re.IGNORECASE),
    re.compile(r"^\d+\)\s+[A-ZÁÉÍÓÚÑ].{0,120}$"),
    re.compile(r"^[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ0-9\s\-/(),.]{8,}$"),
]
PAGE_NUMBER_PATTERN = re.compile(r"^(P[ÁA]GINA\s+\d+\s+DE\s+\d+|\d+)$", re.IGNORECASE)
TABLE_ROW_PATTERN = re.compile(r"^(?:[A-ZÁÉÍÓÚÑa-záéíóúñ0-9$().,%/-]+\s{2,})+[A-ZÁÉÍÓÚÑa-záéíóúñ0-9$().,%/-]+$")
RANGE_ROW_PATTERN = re.compile(
    r"^(DE\s+[\d.,]+\s+A\s+[\d.,]+.*\d[\d.,]*|[A-ZÁÉÍÓÚÑa-záéíóúñ\s]+\([\w\s]+\)\s+\([\w\s]+\))$",
    re.IGNORECASE,
)


@dataclass
class ChunkWindow:
    target_tokens: int
    min_tokens: int
    max_tokens: int
    overlap_ratio: float


@dataclass
class SourceBlock:
    block_id: str
    block_type: str
    text: str
    page: int
    section_path: list[str]


@dataclass
class SectionContext:
    document: str | None = None
    book: str | None = None
    title: str | None = None
    chapter: str | None = None
    article: str | None = None


@dataclass
class CanonicalUnit:
    hierarchy_path: list[str]
    parent_path: list[str]
    child_labels: list[str]
    support_block_ids: list[str]
    support_block_types: list[str]
    representative_snippets: list[str]
    page_start: int
    page_end: int
    order: int
    inferred_only: bool


@dataclass
class CanonicalUnitCandidate:
    hierarchy_path: list[str]
    block_id: str
    block_type: str
    page: int
    snippet: str


class ChunkingService:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.storage = build_object_store(settings)
        self.classifier = DocumentClassifier(settings)

    def chunk_version(
        self,
        *,
        db: Session,
        version_id: str,
        request: ChunkingRequest | None = None,
    ) -> ChunkingResponse:
        request = request or ChunkingRequest()
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

        blocks = (
            db.query(BlockRecord)
            .filter(BlockRecord.version_id == version_id)
            .order_by(BlockRecord.block_order.asc())
            .all()
        )
        if not blocks:
            raise ValueError(f"No extracted blocks available for version: {version_id}")

        classification = self._resolve_classification(
            db=db,
            version=version,
            document=document,
            blocks=blocks,
        )
        existing = (
            db.query(ChunkRecord)
            .filter(ChunkRecord.version_id == version_id)
            .order_by(ChunkRecord.chunk_order.asc())
            .all()
        )
        if existing and not request.force:
            db.commit()
            return self._build_response_from_existing(
                existing, version.document_id, version_id, classification
            )
        window = self._resolve_window(request)
        acl = ACL(**document.acl)
        prepared_blocks = self._prepare_blocks(blocks, classification)
        chunks = self._build_chunks(
            prepared_blocks,
            version.document_id,
            version_id,
            acl,
            window,
            classification,
        )
        if classification.chunking_strategy in {"generic_structural", "generic_freeform"}:
            chunks = self._merge_small_chunks(chunks, window)

        db.query(ChunkRecord).filter(ChunkRecord.version_id == version_id).delete()
        for order, chunk in enumerate(chunks, start=1):
            db.add(
                ChunkRecord(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    version_id=chunk.version_id,
                    block_refs=chunk.block_refs,
                    text=chunk.text,
                    chunk_metadata=chunk.metadata,
                    acl=chunk.acl.model_dump(),
                    index_status=chunk.index_status,
                    chunk_order=order,
                )
            )

        artifact = self.storage.save_json(
            f"artifacts/{version.document_id}/{version_id}/chunks.json",
            [chunk.model_dump(mode="json") for chunk in chunks],
        )
        db.commit()

        return ChunkingResponse(
            document_id=version.document_id,
            version_id=version_id,
            chunks_count=len(chunks),
            table_chunks_count=sum(1 for chunk in chunks if chunk.metadata.get("table_chunk") is True),
            artifact_uri=artifact.uri,
            chunk_window={
                "target_tokens": window.target_tokens,
                "min_tokens": window.min_tokens,
                "max_tokens": window.max_tokens,
                "overlap_ratio": window.overlap_ratio,
            },
            document_class=classification.document_class,
            document_family=classification.document_family,
            chunking_strategy=classification.chunking_strategy,
            classification_source=classification.classification_source,
            classification_confidence=classification.confidence,
            classification_reason_short=classification.reason_short,
        )

    def list_chunks(self, *, db: Session, version_id: str) -> list[ChunkContract]:
        records = (
            db.query(ChunkRecord)
            .filter(ChunkRecord.version_id == version_id)
            .order_by(ChunkRecord.chunk_order.asc())
            .all()
        )
        return [
            ChunkContract(
                chunk_id=record.chunk_id,
                document_id=record.document_id,
                version_id=record.version_id,
                block_refs=record.block_refs,
                text=record.text,
                metadata=record.chunk_metadata,
                acl=ACL(**record.acl),
                index_status=record.index_status,
            )
            for record in records
        ]

    def build_canonical_units(
        self,
        *,
        blocks: list[BlockRecord],
        classification: DocumentClassification,
    ) -> list[CanonicalUnit]:
        if not blocks:
            return []
        prepared_blocks = self._prepare_blocks(blocks, classification)
        return self._build_canonical_unit_inventory(prepared_blocks, classification)

    def _build_response_from_existing(
        self,
        existing: list[ChunkRecord],
        document_id: str,
        version_id: str,
        classification: DocumentClassification,
    ) -> ChunkingResponse:
        return ChunkingResponse(
            document_id=document_id,
            version_id=version_id,
            chunks_count=len(existing),
            table_chunks_count=sum(
                1 for chunk in existing if chunk.chunk_metadata.get("table_chunk") is True
            ),
            artifact_uri=f"{self.storage.base_prefix}/artifacts/{document_id}/{version_id}/chunks.json",
            chunk_window={
                "target_tokens": self.settings.chunk_target_tokens,
                "min_tokens": self.settings.chunk_min_tokens,
                "max_tokens": self.settings.chunk_max_tokens,
                "overlap_ratio": self.settings.chunk_overlap_ratio,
            },
            document_class=classification.document_class,
            document_family=classification.document_family,
            chunking_strategy=classification.chunking_strategy,
            classification_source=classification.classification_source,
            classification_confidence=classification.confidence,
            classification_reason_short=classification.reason_short,
        )

    def _resolve_classification(
        self,
        *,
        db: Session,
        version: VersionRecord,
        document: DocumentRecord,
        blocks: list[BlockRecord],
    ) -> DocumentClassification:
        if (
            getattr(version, "document_class", None)
            and version.document_family
            and version.chunking_strategy
            and version.classification_source
            and version.classification_confidence is not None
        ):
            document_class = version.document_class
            document_family = self._normalize_document_family(version.document_family)
            chunking_strategy = self._normalize_chunking_strategy(version.chunking_strategy)
            return DocumentClassification(
                document_class=document_class,
                document_family=document_family,
                chunking_strategy=chunking_strategy,
                classification_source=version.classification_source,
                confidence=round(float(version.classification_confidence), 2),
                reason_short=version.classification_reason or "Persisted ingestion classification.",
            )

        classification = self.classifier.classify(
            filename=self._filename_from_storage_uri(version.storage_uri),
            mime_type=document.mime_type,
            blocks=blocks,
        )
        version.document_class = classification.document_class
        version.document_family = classification.document_family
        version.chunking_strategy = classification.chunking_strategy
        version.classification_source = classification.classification_source
        version.classification_confidence = classification.confidence
        version.classification_reason = classification.reason_short
        document.document_class = classification.document_class
        document.document_family = classification.document_family
        db.flush()
        return classification

    def _resolve_window(self, request: ChunkingRequest) -> ChunkWindow:
        target_tokens = request.target_tokens or self.settings.chunk_target_tokens
        min_tokens = request.min_tokens or self.settings.chunk_min_tokens
        max_tokens = request.max_tokens or self.settings.chunk_max_tokens
        overlap_ratio = request.overlap_ratio
        if overlap_ratio is None:
            overlap_ratio = self.settings.chunk_overlap_ratio

        if min_tokens <= 0 or max_tokens <= 0 or target_tokens <= 0:
            raise ValueError("Chunk token settings must be positive.")
        if min_tokens > target_tokens or target_tokens > max_tokens:
            raise ValueError("Chunk token settings must satisfy min <= target <= max.")
        if overlap_ratio < 0 or overlap_ratio >= 1:
            raise ValueError("Chunk overlap ratio must be in [0, 1).")

        return ChunkWindow(
            target_tokens=target_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            overlap_ratio=overlap_ratio,
        )

    def _prepare_blocks(
        self,
        blocks: list[BlockRecord],
        classification: DocumentClassification,
    ) -> list[SourceBlock]:
        boilerplate_lines = self._detect_boilerplate_lines(blocks)
        if classification.chunking_strategy == "normative_hierarchical":
            return self._prepare_normative_blocks(blocks, boilerplate_lines)
        if classification.chunking_strategy == "technical_structural":
            return self._prepare_technical_blocks(blocks, boilerplate_lines)
        return self._prepare_general_blocks(blocks, boilerplate_lines, classification)

    def _prepare_normative_blocks(
        self,
        blocks: list[BlockRecord],
        boilerplate_lines: set[str],
    ) -> list[SourceBlock]:
        prepared: list[SourceBlock] = []
        context = SectionContext()
        for block in blocks:
            cleaned_text = self._clean_block_text(block.text, boilerplate_lines)
            cleaned_text = self._normalize_normative_ocr_text(cleaned_text, block.block_type)
            if not cleaned_text:
                continue

            if block.block_type == "table":
                table_block = SourceBlock(
                    block_id=block.block_id,
                    block_type="table",
                    text=cleaned_text,
                    page=block.page,
                    section_path=block.section_path or [f"Page {block.page}"],
                )
                table_block.section_path = self._merge_with_context(table_block.section_path, context, block.page)
                prepared.append(table_block)
                continue

            split_blocks = self._split_block_into_sections(block, cleaned_text)
            for split_block in split_blocks:
                split_block.section_path = self._merge_with_context(
                    split_block.section_path, context, split_block.page
                )
                context = self._advance_context(context, split_block.section_path)
                prepared.append(split_block)
        return prepared

    def _prepare_technical_blocks(
        self,
        blocks: list[BlockRecord],
        boilerplate_lines: set[str],
    ) -> list[SourceBlock]:
        prepared: list[SourceBlock] = []
        primary_section: str | None = None
        body_started = False

        for block in blocks:
            cleaned_text = self._clean_block_text(block.text, boilerplate_lines)
            cleaned_text = self._normalize_technical_text(cleaned_text, block.block_type)
            if not cleaned_text:
                continue

            if block.block_type == "table":
                prepared.append(
                    SourceBlock(
                        block_id=block.block_id,
                        block_type="table",
                        text=cleaned_text,
                        page=block.page,
                        section_path=block.section_path or [f"Page {block.page}"],
                    )
                )
                continue

            lines = self._non_empty_lines(cleaned_text)
            if not lines:
                continue

            current_lines: list[str] = []
            current_path: list[str] = ["Document"]
            current_block_type = "section"

            def flush_current() -> None:
                if not current_lines:
                    return
                segment_text = "\n".join(current_lines).strip()
                segment_type = self._technical_block_type(
                    segment_text,
                    current_path,
                    body_started=body_started,
                    current_block_type=current_block_type,
                )
                prepared.append(
                    SourceBlock(
                        block_id=block.block_id,
                        block_type=segment_type,
                        text=segment_text,
                        page=block.page,
                        section_path=current_path,
                    )
                )

            for line in lines:
                if self._looks_like_markdown_table_line(line):
                    if current_lines and current_block_type != "table":
                        flush_current()
                        current_lines = []
                    current_block_type = "table"
                    current_lines.append(line)
                    continue

                if current_block_type == "table" and current_lines:
                    flush_current()
                    current_lines = []
                    current_block_type = "section"

                heading = self._technical_heading(line)
                if heading and current_lines:
                    flush_current()
                    current_lines = [line]
                elif heading:
                    current_lines = [line]
                else:
                    current_lines.append(line)
                    continue

                if heading is None:
                    continue

                label, level = heading
                if level == 1:
                    primary_section = label
                    current_path = [label]
                    if self._is_primary_technical_body_heading(label):
                        body_started = True
                elif primary_section:
                    current_path = [primary_section, label]
                else:
                    current_path = [label]
                current_block_type = "section"

            flush_current()

        return prepared

    def _prepare_general_blocks(
        self,
        blocks: list[BlockRecord],
        boilerplate_lines: set[str],
        classification: DocumentClassification,
    ) -> list[SourceBlock]:
        if classification.chunking_strategy == "generic_structural":
            return self._prepare_general_structured_blocks(blocks, boilerplate_lines)

        prepared: list[SourceBlock] = []
        for block in blocks:
            cleaned_text = self._clean_block_text(block.text, boilerplate_lines)
            if not cleaned_text:
                continue

            if block.block_type == "table":
                prepared.append(
                    SourceBlock(
                        block_id=block.block_id,
                        block_type="table",
                        text=cleaned_text,
                        page=block.page,
                        section_path=block.section_path or [f"Page {block.page}"],
                    )
                )
                continue

            for segment_text in self._generic_segments(cleaned_text, classification.chunking_strategy):
                if not segment_text:
                    continue
                prepared.append(
                    SourceBlock(
                        block_id=block.block_id,
                        block_type="section",
                        text=segment_text,
                        page=block.page,
                        section_path=self._generic_section_path(segment_text, block.page, block.section_path),
                    )
                )
        return prepared

    def _prepare_general_structured_blocks(
        self,
        blocks: list[BlockRecord],
        boilerplate_lines: set[str],
    ) -> list[SourceBlock]:
        prepared: list[SourceBlock] = []
        primary_section: str | None = None
        secondary_section: str | None = None
        decision_mode = False
        decision_item_number: int | None = None

        for block in blocks:
            cleaned_text = self._clean_block_text(block.text, boilerplate_lines)
            cleaned_text = self._normalize_general_structured_text(cleaned_text, block.block_type)
            if not cleaned_text:
                continue

            lines = self._strip_repeated_general_heading_lines(self._non_empty_lines(cleaned_text))
            if not lines:
                continue

            current_lines: list[str] = []
            current_path: list[str] = ["Document"]
            current_block_type = "section"

            def flush_current() -> None:
                if not current_lines:
                    return
                prepared.append(
                    SourceBlock(
                        block_id=block.block_id,
                        block_type=current_block_type,
                        text="\n".join(current_lines).strip(),
                        page=block.page,
                        section_path=current_path,
                    )
                )

            for line in lines:
                if self._looks_like_markdown_table_line(line):
                    if current_lines and current_block_type != "table":
                        flush_current()
                        current_lines = []
                    current_block_type = "table"
                    current_lines.append(line)
                    continue

                if current_block_type == "table" and current_lines:
                    flush_current()
                    current_lines = []
                    current_block_type = "section"

                heading = self._general_heading(line)
                if decision_mode:
                    decision_item = self._general_decision_item_heading(
                        line,
                        previous_number=decision_item_number,
                    )
                    if decision_item:
                        heading = (decision_item, 3)
                if heading and current_lines:
                    flush_current()
                    current_lines = [line]
                elif heading:
                    current_lines = [line]
                else:
                    current_lines.append(line)
                    continue

                label, level = heading
                if level == 1:
                    primary_section = label
                    secondary_section = None
                    decision_mode = False
                    decision_item_number = None
                    current_path = [label]
                elif level == 2:
                    secondary_section = label
                    decision_mode = self._is_general_decision_heading(label)
                    decision_item_number = None
                    if primary_section:
                        current_path = [primary_section, label]
                    else:
                        current_path = [label]
                elif level == 3:
                    if decision_mode:
                        decision_item_number = self._general_heading_number(label)
                    if primary_section and secondary_section:
                        current_path = [primary_section, secondary_section, label]
                    elif secondary_section:
                        current_path = [secondary_section, label]
                    elif primary_section:
                        current_path = [primary_section, label]
                    else:
                        current_path = [label]
                elif primary_section:
                    current_path = [primary_section, label]
                else:
                    current_path = [label]
                current_block_type = "section"

            flush_current()

        return prepared

    def _build_canonical_unit_inventory(
        self,
        prepared_blocks: list[SourceBlock],
        classification: DocumentClassification,
    ) -> list[CanonicalUnit]:
        if not prepared_blocks:
            return []

        inventory: dict[tuple[str, ...], dict[str, object]] = {}
        order_lookup = {block.block_id: index for index, block in enumerate(prepared_blocks)}

        def ensure_unit(
            hierarchy_path: list[str],
            *,
            inferred: bool,
            block: SourceBlock | None = None,
            snippet: str = "",
        ) -> dict[str, object]:
            cleaned_path = [label.strip() for label in hierarchy_path if label and label.strip()]
            if not cleaned_path:
                return {}
            key = tuple(self._normalize_line(label) for label in cleaned_path)
            unit = inventory.get(key)
            if unit is None:
                order = (
                    order_lookup.get(block.block_id, len(prepared_blocks))
                    if block is not None
                    else len(prepared_blocks)
                )
                page = block.page if block is not None else 1
                unit = {
                    "hierarchy_path": cleaned_path,
                    "parent_path": cleaned_path[:-1],
                    "child_labels": Counter(),
                    "support_block_ids": [],
                    "support_block_types": [],
                    "representative_snippets": [],
                    "page_start": page,
                    "page_end": page,
                    "order": order,
                    "inferred_only": inferred,
                }
                inventory[key] = unit
            else:
                unit["inferred_only"] = bool(unit["inferred_only"]) and inferred

            if block is not None:
                unit["page_start"] = min(int(unit["page_start"]), int(block.page))
                unit["page_end"] = max(int(unit["page_end"]), int(block.page))
                unit["order"] = min(int(unit["order"]), order_lookup.get(block.block_id, int(unit["order"])))
                support_ids = unit["support_block_ids"]
                if block.block_id not in support_ids:
                    support_ids.append(block.block_id)
                support_types = unit["support_block_types"]
                if block.block_type not in support_types:
                    support_types.append(block.block_type)

            if snippet:
                snippets = unit["representative_snippets"]
                if snippet not in snippets and len(snippets) < 4:
                    snippets.append(snippet)

            return unit

        for block in prepared_blocks:
            path = [label.strip() for label in (block.section_path or []) if label and label.strip()]
            if not path:
                continue
            for depth in range(1, len(path) + 1):
                unit = ensure_unit(path[:depth], inferred=False, block=block)
                if depth < len(path):
                    unit["child_labels"][path[depth]] += 1
                snippet = self._canonical_unit_snippet(block) if depth == len(path) else ""
                if snippet:
                    ensure_unit(path[:depth], inferred=False, block=block, snippet=snippet)

        for block in prepared_blocks:
            for candidate in self._infer_canonical_unit_candidates(block, classification):
                ensure_unit(
                    candidate.hierarchy_path,
                    inferred=True,
                    block=block,
                    snippet=candidate.snippet,
                )

        units: list[CanonicalUnit] = []
        for item in inventory.values():
            child_labels = [
                label for label, _ in item["child_labels"].most_common(6) if str(label).strip()
            ]
            units.append(
                CanonicalUnit(
                    hierarchy_path=list(item["hierarchy_path"]),
                    parent_path=list(item["parent_path"]),
                    child_labels=child_labels,
                    support_block_ids=list(item["support_block_ids"]),
                    support_block_types=list(item["support_block_types"]),
                    representative_snippets=list(item["representative_snippets"]),
                    page_start=int(item["page_start"]),
                    page_end=int(item["page_end"]),
                    order=int(item["order"]),
                    inferred_only=bool(item["inferred_only"]),
                )
            )

        units.sort(key=lambda unit: (unit.order, len(unit.hierarchy_path)))
        return units

    def _canonical_unit_snippet(self, block: SourceBlock) -> str:
        if block.block_type in {"table", "reference_table"}:
            return ""
        normalized = " ".join((block.text or "").split())
        if len(normalized) <= 260:
            return normalized
        return normalized[:257].rstrip() + "..."

    def _infer_canonical_unit_candidates(
        self,
        block: SourceBlock,
        classification: DocumentClassification,
    ) -> list[CanonicalUnitCandidate]:
        if classification.document_family != "normative":
            return []
        if block.block_type not in {"table", "reference_table"}:
            return []

        parent_path = self._canonical_inference_parent_path(block.section_path, classification)
        if not parent_path:
            return []

        candidates: list[CanonicalUnitCandidate] = []
        seen: set[tuple[str, ...]] = set()
        for line in self._non_empty_lines(block.text):
            if not self._looks_like_markdown_table_line(line):
                continue
            cells = self._markdown_table_cells(line)
            label = self._canonical_label_from_reference_cells(cells, classification)
            if not label:
                continue
            hierarchy_path = [*parent_path, label]
            key = tuple(self._normalize_line(item) for item in hierarchy_path)
            if key in seen:
                continue
            seen.add(key)
            snippet = f"{label}. Fuente estructural: {' | '.join(cells[:3])}".strip()
            candidates.append(
                CanonicalUnitCandidate(
                    hierarchy_path=hierarchy_path,
                    block_id=block.block_id,
                    block_type=block.block_type,
                    page=block.page,
                    snippet=snippet[:320],
                )
            )
        return candidates

    def _canonical_inference_parent_path(
        self,
        section_path: list[str],
        classification: DocumentClassification,
    ) -> list[str]:
        cleaned = [label.strip() for label in (section_path or []) if label and label.strip()]
        if not cleaned:
            return []
        if classification.document_class == "technical_standard":
            return cleaned[:1]
        if len(cleaned) > 1:
            return cleaned[:-1]
        return cleaned

    def _markdown_table_cells(self, line: str) -> list[str]:
        stripped = line.strip().strip("|")
        if not stripped:
            return []
        cells = [
            " ".join(cell.replace("`", " ").replace("*", " ").split()).strip(" -:;,.")
            for cell in stripped.split("|")
        ]
        return [cell for cell in cells if cell]

    def _canonical_label_from_reference_cells(
        self,
        cells: list[str],
        classification: DocumentClassification,
    ) -> str | None:
        if len(cells) < 2:
            return None
        first = cells[0].strip()
        second = cells[1].strip()
        first_label = self._structural_label_candidate(first)
        second_label = self._structural_label_candidate(second)

        if first_label:
            topic = self._reference_title_text(second)
            if topic and topic.lower() not in first_label.lower():
                return f"{first_label} {topic}".strip()
            return first_label
        if second_label:
            return second_label

        if (
            classification.document_class == "technical_standard"
            and re.fullmatch(r"\d+(?:\.\d+)?", first)
            and self._is_reference_title_text(second)
        ):
            return f"Sección {first} {self._reference_title_text(second)}".strip()

        return None

    def _structural_label_candidate(self, text: str) -> str | None:
        candidate = " ".join((text or "").replace("*", " ").split()).strip(" -:;,.")
        if not candidate:
            return None
        if re.fullmatch(r"[:-]+", candidate):
            return None
        if re.match(
            r"^(?:SECCI[OÓ]N|SECTION|AP[EÉ]NDICE|APPENDIX|ANEXO|RECOMENDACI[OÓ]N|CAP[IÍ]TULO|T[ÍI]TULO|LIBRO|P[ÁA]RRAFO|ART(?:\.|O\.)?|ARTICULO)\s+[A-Z0-9IVXLCM]+",
            candidate,
            re.IGNORECASE,
        ):
            return candidate[:160]
        return None

    def _is_reference_title_text(self, text: str) -> bool:
        candidate = self._reference_title_text(text)
        if not candidate:
            return False
        normalized = self._normalize_line(candidate)
        if normalized in {
            "NUMERO",
            "NUMERO ANTERIOR",
            "FUENTES",
            "FUENTE",
            "SOURCE",
            "SOURCES",
            "PAGINA",
            "PAGE",
            "SECCION DE LA NIIF PARA LAS PYMES",
        }:
            return False
        if len(candidate.split()) < 2:
            return False
        return any(char.isalpha() for char in candidate)

    def _reference_title_text(self, text: str) -> str:
        candidate = " ".join((text or "").replace("*", " ").split()).strip(" -:;,.")
        candidate = re.sub(r"\s{2,}", " ", candidate)
        return candidate[:140]

    def _normalize_general_structured_text(self, text: str, block_type: str) -> str:
        if not text.strip():
            return text

        normalized = text.replace("<!-- image -->", "\n")
        normalized = re.sub(r"\r\n?", "\n", normalized)
        if block_type == "docling_markdown":
            normalized = re.sub(r"(?m)^\s*(#{1,6})\s*", r"\1 ", normalized)
        normalized = re.sub(r"[ \t]{2,}", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    def _strip_repeated_general_heading_lines(self, lines: list[str]) -> list[str]:
        heading_candidates = [self._repeated_heading_key(line) for line in lines]
        repeated = Counter(key for key in heading_candidates if key)
        kept: dict[str, int] = {}
        output: list[str] = []
        for line in lines:
            key = self._repeated_heading_key(line)
            if key and repeated.get(key, 0) >= 3:
                kept[key] = kept.get(key, 0) + 1
                if kept[key] > 1:
                    continue
            output.append(line)
        return output

    def _repeated_heading_key(self, line: str) -> str | None:
        stripped = line.strip()
        if not stripped:
            return None
        candidate = re.sub(r"^#+\s*", "", stripped)
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if len(candidate) < 8 or len(candidate) > 120:
            return None
        if candidate.isupper():
            return self._normalize_line(candidate)
        if re.match(r"^INSTITUTO\b", candidate, re.IGNORECASE):
            return self._normalize_line(candidate)
        return None

    def _general_heading(self, text: str) -> tuple[str, int] | None:
        raw = text.strip()
        if not raw:
            return None

        markdown_match = re.match(r"^(#{1,6})\s*(.+)$", raw)
        candidate = markdown_match.group(2) if markdown_match else raw
        candidate = re.sub(r"\s+", " ", candidate).strip(" -:;")
        if not candidate:
            return None
        if re.match(r"^(PROUESTA|PROPUESTA|ACUERDO|CONCLUSI[Ã“O]N|RECOMENDACI[Ã“O]N)\s*:", candidate, re.IGNORECASE):
            return (candidate.split(":", 1)[0][:120], 2)

        if re.match(r"^(DESARROLLO|AGENDA|ORDEN DEL D[IÍ]A|ACUERDOS?|PROPUESTAS?|RESULTADOS?|RESUMEN|ANTECEDENTES|CONSIDERACIONES)\b", candidate, re.IGNORECASE):
            return (candidate[:120].rstrip(" :"), 1)
        if re.match(r"^(NO HABIENDO OTRO ASUNTO|SE LEVANTA LA SESI[Ã“O]N|CIERRE|CLAUSURA)\b", candidate, re.IGNORECASE):
            return ("Cierre", 2)
        if re.match(r"^\d+(?:\.\d+)*[.)]\s+", candidate):
            return (candidate[:120], 1)
        if re.match(r"^[IVXLCM]+\.\s+", candidate, re.IGNORECASE):
            return (candidate[:120], 1)
        if re.match(r"^(PROUESTA|PROPUESTA|ACUERDO|CONCLUSI[ÓO]N|RECOMENDACI[ÓO]N)\s*:", candidate, re.IGNORECASE):
            return (candidate.split(":", 1)[0][:120], 2)
        if re.match(r"^\d+(?:\.\d+)+\.?\s+", candidate):
            return (candidate[:120], 1)
        if markdown_match and self._generic_heading_label(raw):
            label = self._generic_heading_label(raw)
            return (label, 1)
        if markdown_match and re.match(r"^(?:[A-ZÁÉÍÓÚÑ][^|]{1,120})$", candidate) and len(candidate.split()) <= 18:
            return (candidate[:120], 1)
        return None

    def _general_decision_item_heading(self, text: str, *, previous_number: int | None) -> str | None:
        raw = text.strip()
        if not raw:
            return None

        candidate = re.sub(r"^#+\s*", "", raw)
        candidate = re.sub(r"\s+", " ", candidate).strip(" -:;")
        match = re.match(r"^(\d+)([.)])\s+(.+)$", candidate)
        if not match:
            return None

        number = int(match.group(1))
        if previous_number is None:
            if number != 1:
                return None
        elif number != previous_number + 1:
            return None

        suffix = match.group(3).strip()
        if ":" in suffix:
            suffix = suffix.split(":", 1)[0].strip()
        suffix = re.sub(r"\s+", " ", suffix)
        if len(suffix) > 90:
            suffix = f"{suffix[:87].rstrip()}..."
        return f"{number}{match.group(2)} {suffix}".strip()

    def _is_general_decision_heading(self, label: str) -> bool:
        candidate = label.strip()
        return bool(
            re.match(
                r"^(PROUESTA|PROPUESTA|ACUERDO|ACUERDOS|CONCLUSI[Ã“O]N|RECOMENDACI[Ã“O]N)\b",
                candidate,
                re.IGNORECASE,
            )
        )

    def _general_heading_number(self, label: str) -> int | None:
        match = re.match(r"^(\d+)", label.strip())
        if not match:
            return None
        return int(match.group(1))

    def _normalize_technical_text(self, text: str, block_type: str) -> str:
        if not text.strip():
            return text

        normalized = text.replace("<!-- image -->", "\n")
        normalized = re.sub(r"\r\n?", "\n", normalized)
        if block_type == "docling_markdown":
            normalized = re.sub(r"(?m)^\s*(#{1,6})\s*", r"\1 ", normalized)

        normalized = re.sub(
            r"(?i)\b(secci[oó]n|section|ap[eé]ndice|appendix|pregunta|question|ejemplo|example|nota|m[oó]dulo|modulo)\s*([A-Z]?\d+(?:\.\d+)*)",
            r"\1 \2",
            normalized,
        )
        normalized = re.sub(r"(?<=[a-záéíóúñ0-9])(?=[A-ZÁÉÍÓÚÑ])", " ", normalized)
        normalized = re.sub(r"(?<=[?!:;])(?=[A-ZÁÉÍÓÚÑ])", " ", normalized)
        normalized = re.sub(r"[ \t]{2,}", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    def _technical_heading(self, text: str) -> tuple[str, int] | None:
        raw = text.strip()
        if not raw:
            return None

        markdown_match = re.match(r"^(#{1,6})\s*(.+)$", raw)
        candidate = markdown_match.group(2) if markdown_match else raw
        candidate = re.sub(r"\s+", " ", candidate).strip(" -:;")
        if not candidate:
            return None

        strong_section = re.match(
            r"^(SECCI[ÓO]N|SECTION|AP[EÉ]NDICE|APPENDIX|M[ÓO]DULO|MODULO)\b",
            candidate,
            re.IGNORECASE,
        )
        strong_subsection = re.match(
            r"^(EJEMPLO|EXAMPLE|ESTUDIO DE CASO|CASE STUDY|PREGUNTA|QUESTION|NOTA|FIGURA|TABLA|RESPUESTAS?|C[ÁA]LCULO|RECONCILIACI[ÓO]N)\b",
            candidate,
            re.IGNORECASE,
        )

        if markdown_match:
            level = 1 if strong_section or candidate.isupper() else 2
            if strong_subsection:
                level = 2
            return (candidate[:120], level)

        if strong_section:
            return (candidate[:120], 1)
        if candidate.isupper() and len(candidate.split()) <= 12:
            return (candidate[:120], 1)
        if strong_subsection:
            return (candidate[:120], 2)
        return None

    def _looks_like_markdown_table_line(self, line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2

    def _is_primary_technical_body_heading(self, label: str) -> bool:
        candidate = label.strip()
        if re.match(r"^(SECCI[OÓ]N|SECTION)\s+\d+", candidate, re.IGNORECASE):
            return True
        if candidate.isupper() and not re.match(r"^(COPYRIGHT|ISBN|CONTENIDO|CONTENTS)\b", candidate, re.IGNORECASE):
            return True
        return False

    def _technical_block_type(
        self,
        text: str,
        section_path: list[str],
        *,
        body_started: bool,
        current_block_type: str,
    ) -> str:
        if current_block_type == "table":
            if self._is_reference_table_text(text):
                return "reference_table"
            return "table"

        normalized = self._normalize_line(text[:1200])
        top_label = section_path[0] if section_path else "Document"
        if self._is_technical_preamble_text(normalized, top_label=top_label, body_started=body_started):
            return "preamble"
        return "section"

    def _is_reference_table_text(self, text: str) -> bool:
        lines = self._non_empty_lines(text)
        if len(lines) < 3:
            return False
        table_lines = sum(1 for line in lines if self._looks_like_markdown_table_line(line))
        page_rows = sum(1 for line in lines if re.search(r"\|\s*\d+\s*\|?$", line))
        return table_lines >= 3 and page_rows >= 2

    def _is_technical_preamble_text(
        self,
        normalized_text: str,
        *,
        top_label: str,
        body_started: bool,
    ) -> bool:
        if "COPYRIGHT" in normalized_text or "RESERVADOS TODOS LOS DERECHOS" in normalized_text or "ISBN" in normalized_text:
            return True
        if body_started:
            return False
        normalized_label = self._normalize_line(top_label)
        if any(token in normalized_label for token in ("NORMA", "MODULO", "MÓDULO")):
            return True
        return False

    def _generic_segments(self, text: str, chunking_strategy: str) -> list[str]:
        if chunking_strategy == "generic_structural":
            lines = self._non_empty_lines(text)
            if not lines:
                return []

            sections: list[list[str]] = []
            current: list[str] = []
            for line in lines:
                heading = self._generic_heading_label(line)
                if heading and current:
                    sections.append(current)
                    current = [line]
                else:
                    current.append(line)
            if current:
                sections.append(current)

            segments = ["\n".join(section).strip() for section in sections if any(part.strip() for part in section)]
            if len(segments) > 1:
                return segments

            paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
            if len(paragraphs) > 1:
                return paragraphs
            return [text.strip()]

        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        if len(paragraphs) > 1:
            return paragraphs

        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
        if len(sentences) > 1:
            return sentences
        return [text.strip()]

    def _generic_section_path(
        self,
        segment_text: str,
        page: int,
        fallback: list[str] | None,
    ) -> list[str]:
        lines = self._non_empty_lines(segment_text)
        labels: list[str] = []
        for line in lines[:3]:
            generic_heading = self._generic_heading_label(line)
            if generic_heading:
                labels.append(generic_heading)
                continue
            if self._is_section_boundary(line):
                heading = self._compact_section_label(line)
                if heading:
                    labels.append(heading)
        if labels:
            return list(dict.fromkeys(labels[:2]))

        if fallback:
            compacted = [self._compact_section_label(label) or label for label in fallback if label]
            compacted = [label for label in compacted if label]
            if compacted:
                return list(dict.fromkeys(compacted[:2]))

        if lines:
            heading = self._generic_heading_label(lines[0]) or self._compact_section_label(lines[0])
            if heading:
                return [heading]
        return [f"Page {page}"]

    def _filename_from_storage_uri(self, storage_uri: str) -> str:
        return Path(storage_uri).name or storage_uri

    def _derive_document_class(
        self,
        stored_family: str | None,
        stored_strategy: str | None,
    ) -> str:
        strategy = self._normalize_chunking_strategy(stored_strategy)
        if strategy == "normative_hierarchical":
            return "legal_normative"
        if strategy == "technical_structural":
            return "technical_standard"
        if stored_family == "normative":
            return "legal_normative"
        return "general_document"

    def _normalize_document_family(self, stored_family: str | None) -> str:
        if stored_family == "normative":
            return "normative"
        return "general"

    def _normalize_chunking_strategy(self, stored_strategy: str | None) -> str:
        if stored_strategy == "normative_hierarchical":
            return "normative_hierarchical"
        if stored_strategy == "technical_structural":
            return "technical_structural"
        if stored_strategy == "generic_freeform":
            return "generic_freeform"
        return "generic_structural"

    def _detect_boilerplate_lines(self, blocks: list[BlockRecord]) -> set[str]:
        page_blocks = [block for block in blocks if block.block_type == "page"]
        if len(page_blocks) < 3:
            return set()

        first_lines: Counter[str] = Counter()
        last_lines: Counter[str] = Counter()
        for block in page_blocks:
            lines = self._non_empty_lines(block.text)
            for line in lines[:3]:
                first_lines[self._normalize_line(line)] += 1
            for line in lines[-3:]:
                last_lines[self._normalize_line(line)] += 1

        threshold = max(3, ceil(len(page_blocks) * 0.4))
        repeated = {
            line
            for line, count in (first_lines + last_lines).items()
            if count >= threshold and not PAGE_NUMBER_PATTERN.match(line)
        }
        return repeated

    def _clean_block_text(self, text: str, boilerplate_lines: set[str]) -> str:
        cleaned_lines: list[str] = []
        for line in self._non_empty_lines(text):
            normalized = self._normalize_line(line)
            if PAGE_NUMBER_PATTERN.match(normalized):
                continue
            if normalized in boilerplate_lines:
                continue
            cleaned_lines.append(line.strip())
        return "\n".join(cleaned_lines).strip()

    def _normalize_normative_ocr_text(self, text: str, block_type: str) -> str:
        if not text.strip():
            return text

        normalized = text
        if block_type == "docling_markdown":
            normalized = normalized.replace("<!-- image -->", "\n")
            normalized = re.sub(r"(?m)^\s*#+\s*", "", normalized)
        return self._normalize_normative_clean(normalized)

        # Clean normative normalization path kept in plain Unicode to avoid
        # mojibake-sensitive regexes in older experimental rules below.
        normalized = re.sub(r"(?i)\b(Libro)\s*([IVXLC0-9]+)\b", r"\1 \2", normalized)
        normalized = re.sub(r"(?i)\b(T[ií]tulo)\s*([IVXLC0-9]+)\b", r"\1 \2", normalized)
        normalized = re.sub(r"(?i)\b(Cap[ií]tulo)\s*([IVXLC0-9]+)\b", r"\1 \2", normalized)
        normalized = re.sub(r"(?i)\b(Art(?:[ií]culo|o\.?))\s*([0-9]+)\s*([A-ZÁÉÍÓÚÑ])", r"\1 \2 \3", normalized)
        normalized = re.sub(r"(?i)\b(Art(?:[ií]culo|o\.?))\s*([0-9]+)\b", r"\1 \2", normalized)
        normalized = re.sub(r"(?i)\b(Ley)\s*N[°ºO.]?\s*\.?\s*([0-9]+)\b", r"\1 N°. \2", normalized)
        normalized = re.sub(
            r"(?<!\n)(?=(?:Ley\s+N[°ºO.]|Libro\s+[IVXLC0-9]+|T[ií]tulo\s+[IVXLC0-9]+|Cap[ií]tulo\s+[IVXLC0-9]+|(?:Art(?:[ií]culo|o\.?))\s*\d+))",
            "\n",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(r"(?i)\bdelaley\b", "de la ley", normalized)
        normalized = re.sub(r"(?i)\bInstituciony\b", "Institucion y", normalized)
        normalized = self._extract_normative_body(normalized)
        normalized = self._light_normative_lexical_cleanup(normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

        normalized = re.sub(r"(?i)\b(Libro)\s*([IVXLC0-9]+)\b", r"\1 \2", normalized)
        normalized = re.sub(r"(?i)\b(T[ií]tulo)\s*([IVXLC0-9]+)\b", r"\1 \2", normalized)

        normalized = re.sub(r"(?i)\b(Cap[ií]tulo)\s*([IVXLC0-9]+)\b", r"\1 \2", normalized)
        normalized = re.sub(r"(?i)\b(Art(?:[ií]culo|o\.?))\s*([0-9]+)\s*([A-ZÁÉÍÓÚÑ])", r"\1 \2 \3", normalized)
        normalized = re.sub(r"(?i)\b(Art(?:[ií]culo|o\.?))\s*([0-9]+)\b", r"\1 \2", normalized)
        normalized = re.sub(r"(?i)\b(Ley)\s*N[°ºO.]?\s*\.?\s*([0-9]+)\b", r"\1 N°. \2", normalized)

        normalized = re.sub(
            r"(?<!\n)(?=(?:Ley\s+N[°ºO.]|Cap[ií]tulo\s+[IVXLC0-9]+|(?:Art(?:[ií]culo|o\.?))\s*\d+))",
            "\n",
            normalized,
            flags=re.IGNORECASE,
        )

        normalized = re.sub(r"(?i)\bdelaley\b", "de la ley", normalized)
        normalized = re.sub(
            r"(?<!\n)(?=(?:Libro\s+[IVXLC0-9]+|T[ií]tulo\s+[IVXLC0-9]+))",
            "\n",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(r"(?i)\bInstituciony\b", "Institucion y", normalized)
        normalized = self._extract_normative_body(normalized)
        normalized = self._light_normative_lexical_cleanup(normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    def _normalize_normative_clean(self, text: str) -> str:
        normalized = text
        normalized = re.sub(r"(?i)\b(Libro)\s*([IVXLC0-9]+)\b", r"\1 \2", normalized)
        normalized = re.sub(r"(?i)\b(T[i\u00ed]tulo)\s*([IVXLC0-9]+)\b", r"\1 \2", normalized)
        normalized = re.sub(r"(?i)\b(Cap[i\u00ed]tulo)\s*([IVXLC0-9]+)\b", r"\1 \2", normalized)
        normalized = re.sub(r"(?i)\b(Art(?:[i\u00ed]culo|o)?\.?)\s*([0-9]+)\s*([A-Z\u00c1\u00c9\u00cd\u00d3\u00da\u00d1])", r"\1 \2 \3", normalized)
        normalized = re.sub(r"(?i)\b(Art(?:[i\u00ed]culo|o)?\.?)\s*([0-9]+)\b", r"\1 \2", normalized)
        normalized = re.sub(r"(?i)\b(Ley)\s*N[\u00b0\u00baO.]?\s*\.?\s*([0-9]+)\b", r"\1 N. \2", normalized)
        normalized = re.sub(
            r"(?<!\n)(?=(?:Ley\s+N[\u00b0\u00baO.]|Libro\s+[IVXLC0-9]+|T[i\u00ed]tulo\s+[IVXLC0-9]+|Cap[i\u00ed]tulo\s+[IVXLC0-9]+|(?:Art(?:[i\u00ed]culo|o)?\.?)\s*\d+(?:\s+(?:bis|ter|quater|quinquies))?(?=\s*(?:[.:\-]|\s+[A-Z\u00c1\u00c9\u00cd\u00d3\u00da\u00d1(]))))",
            "\n",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(r"(?i)\bdelaley\b", "de la ley", normalized)
        normalized = re.sub(r"(?i)\bInstituciony\b", "Institucion y", normalized)
        normalized = self._extract_normative_body_clean(normalized)
        normalized = self._light_normative_lexical_cleanup(normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()
        normalized = re.sub(r"(?i)\b(Ley)\s*N[\u00b0\u00baO.]?\s*\.?\s*([0-9]+)\b", r"\1 N°. \2", normalized)
        normalized = re.sub(
            r"(?<!\n)(?=(?:Ley\s+N[\u00b0\u00baO.]|Libro\s+[IVXLC0-9]+|T[i\u00ed]tulo\s+[IVXLC0-9]+|Cap[i\u00ed]tulo\s+[IVXLC0-9]+|(?:Art(?:[i\u00ed]culo|o\.?))\s*\d+))",
            "\n",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(r"(?i)\bdelaley\b", "de la ley", normalized)
        normalized = re.sub(r"(?i)\bInstituciony\b", "Institucion y", normalized)
        normalized = self._extract_normative_body(normalized)
        normalized = self._light_normative_lexical_cleanup(normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    def _extract_normative_body_clean(self, text: str) -> str:
        preferred_markers = (
            r"(?is)HA\s*DICTADO.*?(Ley\s+N(?:\.|°|º|O)?\s*\.?\s*\d+)",
            r"(?is)La\s+siguiente:.*?(Ley\s+N(?:\.|°|º|O)?\s*\.?\s*\d+)",
        )
        for pattern in preferred_markers:
            match = re.search(pattern, text)
            if match and match.start(1) <= 1500:
                return text[match.start(1):]

        law_matches = list(re.finditer(r"(?im)^\s*Ley\s+N(?:\.|°|º|O)?\s*\.?\s*\d+", text))
        candidates: list[int] = []
        for pattern in (
            r"(?im)^\s*Ley\s+N(?:\.|°|º|O)?\s*\.?\s*\d+",
            r"(?im)^\s*Libro\s+(?:[IVXLC0-9]+|PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|SEPTIMO|S[ÉE]PTIMO|OCTAVO|NOVENO|D[ÉE]CIMO)",
            r"(?im)^\s*T[i\u00ed]tulo\s+(?:[IVXLC0-9]+|PRELIMINAR|PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|SEPTIMO|S[ÉE]PTIMO|OCTAVO|NOVENO|D[ÉE]CIMO)",
            r"(?im)^\s*Cap[i\u00ed]tulo\s+[IVXLC0-9]+",
            r"(?im)^\s*(?:Art(?:[i\u00ed]culo|o)?\.?)\s*1\b",
        ):
            match = re.search(pattern, text)
            if match:
                candidates.append(match.start())

        if len(law_matches) >= 2:
            first_law = law_matches[0].start()
            last_law = law_matches[-1].start()
            first_candidate = min(candidates) if candidates else first_law
            if first_candidate - first_law <= 500:
                return text[first_candidate:]
            prefix = text[:last_law].upper()
            if any(marker in prefix for marker in ("SUMARIO", "LA GACETA", "LAGACETA", "DIARIO OFICIAL", "CASA DE GOBIERNO")):
                return text[last_law:]
            return text[first_candidate:]

        if not candidates:
            return text

        start = min(candidates)
        prefix = text[:start].upper()
        if start > 300 and any(marker in prefix for marker in ("SUMARIO", "LAGACETA", "DIARIO OFICIAL", "CASA DE GOBIERNO")):
            return text[start:]
        return text

    def _extract_normative_body(self, text: str) -> str:
        preferred_markers = (
            r"(?is)HADICTADO.*?(Ley\s+N[Â°ÂºO.]?\s*\.?\s*\d+)",
            r"(?is)La\s+siguiente:.*?(Ley\s+N[Â°ÂºO.]?\s*\.?\s*\d+)",
        )
        for pattern in preferred_markers:
            match = re.search(pattern, text)
            if match and match.start(1) <= 1500:
                return text[match.start(1):]

        law_matches = list(re.finditer(r"(?im)^\s*Ley\s+N[Â°ÂºO.]?\s*\.?\s*\d+", text))
        candidates: list[int] = []
        for pattern in (
            r"(?im)^\s*Ley\s+N[°ºO.]?\s*\.?\s*\d+",
            r"(?im)^\s*Cap[ií]tulo\s+[IVXLC0-9]+",
            r"(?im)^\s*(?:Art(?:[ií]culo|o)?\.?)\s*1\b",
        ):
            match = re.search(pattern, text)
            if match:
                candidates.append(match.start())

        if not candidates:
            return text

        start = min(candidates)
        prefix = text[:start].upper()
        if start > 300 and any(marker in prefix for marker in ("SUMARIO", "LAGACETA", "DIARIO OFICIAL", "CASA DE GOBIERNO")):
            return text[start:]
        return text

    def _light_normative_lexical_cleanup(self, text: str) -> str:
        cleaned = text
        cleaned = re.sub(r"(?<=[a-záéíóúñ0-9])(?=[A-ZÁÉÍÓÚÑ])", " ", cleaned)

        phrase_replacements = {
            "Objeto delaley": "Objeto de la ley",
            "Objetodelaley": "Objeto de la ley",
            "dePersonalidadJuridica": "de Personalidad Juridica",
            "ConsejoDirectivo": "Consejo Directivo",
            "DireccionEjecutiva": "Direccion Ejecutiva",
            "Auditoria Interna": "Auditoria Interna",
            "Seguridad Socialy Desarrollo Humano": "Seguridad Social y Desarrollo Humano",
            "de laSeguridad Social": "de la Seguridad Social",
            "deSeguridad Social": "de Seguridad Social",
            "lapresente ley": "la presente ley",
            "Conformealodispuestoen": "Conforme a lo dispuesto en",
            "deSeguridadSocialyDesarrolloHumano": "de Seguridad Social y Desarrollo Humano",
            "InstitutoDeSeguridadSocialyDesarrolloHumano": "Instituto de Seguridad Social y Desarrollo Humano",
            "Consejo Directivoseran": "Consejo Directivo seran",
        }
        for source, target in phrase_replacements.items():
            cleaned = cleaned.replace(source, target)

        cleaned = re.sub(r"\bSocialy\b", "Social y", cleaned)
        cleaned = re.sub(r"\bdelInstituto\b", "del Instituto", cleaned)
        cleaned = re.sub(r"\bInstitutoocon\b", "Instituto o con", cleaned)
        cleaned = re.sub(r"\bdeSeguridad\b", "de Seguridad", cleaned)
        cleaned = re.sub(r"\blaSeguridad\b", "la Seguridad", cleaned)
        cleaned = re.sub(r"\blaDireccion\b", "la Direccion", cleaned)
        cleaned = re.sub(r"\blaAuditoria\b", "la Auditoria", cleaned)
        cleaned = re.sub(r"\bdelaley\b", "de la ley", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\blapresente\b", "la presente", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bdePersonalidad\b", "de Personalidad", cleaned)
        cleaned = re.sub(r"\bJuridicaConforme\b", "Juridica Conforme", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r" *\n *", "\n", cleaned)
        return cleaned

    def _split_block_into_sections(self, block: BlockRecord, text: str) -> list[SourceBlock]:
        lines = self._non_empty_lines(text)
        if not lines:
            return []

        segments: list[list[str]] = []
        current: list[str] = []
        for line in lines:
            if self._is_section_boundary(line) and current:
                segments.append(current)
                current = [line]
            else:
                current.append(line)
        if current:
            segments.append(current)

        output: list[SourceBlock] = []
        for segment in segments:
            segment_text = "\n".join(segment).strip()
            if not segment_text:
                continue
            block_type = "table" if self._looks_like_table(segment) else "section"
            section_path = self._derive_section_path(block.page, segment, block.section_path)
            output.append(
                SourceBlock(
                    block_id=block.block_id,
                    block_type=block_type,
                    text=segment_text,
                    page=block.page,
                    section_path=section_path,
                )
            )
        return output

    def _is_section_boundary(self, line: str) -> bool:
        candidate = re.sub(r"^#+\s*", "", line.strip())
        if not candidate:
            return False
        if re.match(r"^LIBRO\b", candidate, re.IGNORECASE):
            return self._is_book_heading(candidate)
        if re.match(r"^T[IÍ]TULO\b", candidate, re.IGNORECASE):
            return self._is_title_heading(candidate)
        if re.match(r"^CAP[IÍ]TULO\b", candidate, re.IGNORECASE):
            return self._is_chapter_heading(candidate)
        if re.match(r"^ART\.\s*\d", candidate, re.IGNORECASE):
            return self._is_article_heading(candidate)
        if re.match(r"^(?:ARTO\.|ART[IÍ]CULO)\b", candidate, re.IGNORECASE):
            return self._is_article_heading(candidate)
        if re.match(r"^(LIBRO\s*(?:[IVXLC0-9]+|PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|SEPTIMO|S[Ã‰E]PTIMO|OCTAVO|NOVENO|D[Ã‰E]CIMO))", candidate, re.IGNORECASE):
            return True
        if re.match(r"^(T[IÍ]TULO\s*[IVXLC0-9]+)", candidate, re.IGNORECASE):
            return True
        if re.match(r"^[IVXLCM]+\s*$", candidate, re.IGNORECASE):
            return True
        if re.match(r"^(CAP[IÍ]TULO\s*[IVXLC0-9]+)", candidate, re.IGNORECASE):
            return True
        if self._is_article_heading(candidate):
            return True
        return any(pattern.match(candidate) for pattern in HEADING_PATTERNS)

    def _looks_like_table(self, segment: list[str]) -> bool:
        table_like_lines = sum(1 for line in segment if TABLE_ROW_PATTERN.match(line))
        numeric_rows = sum(1 for line in segment if re.search(r"\d", line) and re.search(r"\s{2,}", line))
        range_rows = sum(1 for line in segment if RANGE_ROW_PATTERN.match(line))
        header_rows = sum(1 for line in segment[:3] if "RANGO" in line.upper() or "MONTO" in line.upper())
        return table_like_lines >= 2 or numeric_rows >= 3 or range_rows >= 3 or (header_rows >= 1 and range_rows >= 2)

    def _derive_section_path(
        self,
        page: int,
        segment: list[str],
        fallback: list[str] | None,
    ) -> list[str]:
        headings = [self._compact_section_label(line) for line in segment[:4] if self._is_section_boundary(line)]
        headings = [heading for heading in headings if heading]
        if headings:
            return list(dict.fromkeys(headings[:2]))
        return fallback or [f"Page {page}"]

    def _generic_heading_label(self, text: str) -> str | None:
        candidate = re.sub(r"^#+\s*", "", text.strip())
        candidate = re.sub(r"\s+", " ", candidate)
        if not candidate:
            return None
        if len(candidate) > 80:
            return None
        if re.match(r"^(ACTA|SESI[OÓ]N|REUNI[OÓ]N|AGENDA|DESARROLLO|ACUERDOS?|ANEXOS?|DOCUMENTOS? DEL CLIENTE)\b", candidate, re.IGNORECASE):
            return candidate[:72].rstrip(" -:;.")
        if re.match(r"^(ESTADO DE RESULTADOS|ESTADO DE SITUACION FINANCIERA|GIRO DEL NEGOCIO|DIRECCION DEL NEGOCIO)\b", candidate, re.IGNORECASE):
            return candidate[:72].rstrip(" -:;.")
        if re.match(r"^(OBJETIVO|ALCANCE|RESPONSABILIDADES|PROCEDIMIENTO|DEFINICIONES|ANEXO)\b", candidate, re.IGNORECASE):
            return candidate.title()
        if re.match(r"^(PASO|STEP)\s+\d+[:). -]?", candidate, re.IGNORECASE):
            return candidate[:48].rstrip(" -:;.")
        if candidate.isupper() and len(candidate.split()) <= 10:
            return candidate
        if candidate.istitle() and len(candidate.split()) <= 8:
            return candidate
        return None

    def _build_chunks(
        self,
        blocks: list[SourceBlock],
        document_id: str,
        version_id: str,
        acl: ACL,
        window: ChunkWindow,
        classification: DocumentClassification,
    ) -> list[ChunkContract]:
        if classification.chunking_strategy == "normative_hierarchical":
            return self._build_normative_chunks(
                blocks,
                document_id,
                version_id,
                acl,
                window,
                classification,
            )
        if classification.chunking_strategy == "technical_structural":
            return self._build_technical_chunks(
                blocks,
                document_id,
                version_id,
                acl,
                window,
                classification,
            )

        chunks: list[ChunkContract] = []
        current: list[SourceBlock] = []

        for block in blocks:
            if block.block_type == "table":
                if current:
                    chunks.extend(self._flush_text_buffer(current, document_id, version_id, acl, classification))
                    current = []
                chunks.extend(self._build_table_chunks(block, document_id, version_id, acl, window, classification))
                continue

            if current and self._should_flush_on_new_section(current, block, window, classification):
                chunks.extend(self._flush_text_buffer(current, document_id, version_id, acl, classification))
                current = []

            block_tokens = self._estimate_tokens(block.text)
            if block_tokens > window.max_tokens:
                if current:
                    chunks.extend(self._flush_text_buffer(current, document_id, version_id, acl, classification))
                    current = []
                chunks.extend(self._split_single_block(block, document_id, version_id, acl, window, classification))
                continue

            projected_tokens = self._blocks_tokens(current) + block_tokens
            if current and projected_tokens > window.max_tokens:
                flushed_chunk, overlap_blocks = self._flush_with_overlap(
                    current, document_id, version_id, acl, window, classification
                )
                chunks.append(flushed_chunk)
                current = self._seed_next_buffer(overlap_blocks, block, window.max_tokens)
            else:
                current.append(block)

        if current:
            chunks.extend(self._flush_text_buffer(current, document_id, version_id, acl, classification))

        return chunks

    def _build_technical_chunks(
        self,
        blocks: list[SourceBlock],
        document_id: str,
        version_id: str,
        acl: ACL,
        window: ChunkWindow,
        classification: DocumentClassification,
    ) -> list[ChunkContract]:
        chunks: list[ChunkContract] = []
        current: list[SourceBlock] = []

        def flush_current() -> None:
            nonlocal current
            if not current:
                return
            chunks.extend(
                self._flush_technical_buffer(
                    current,
                    document_id,
                    version_id,
                    acl,
                    window,
                    classification,
                )
            )
            current = []

        for block in blocks:
            if block.block_type in {"table", "reference_table"}:
                flush_current()
                chunks.extend(
                    self._build_technical_table_chunks(
                        block,
                        document_id,
                        version_id,
                        acl,
                        window,
                        classification,
                    )
                )
                continue

            if current and current[-1].block_type != block.block_type:
                flush_current()

            if current and current[-1].section_path != block.section_path:
                flush_current()

            block_tokens = self._estimate_tokens(block.text)
            if block_tokens > window.max_tokens:
                flush_current()
                chunks.extend(
                    self._split_single_block(
                        block,
                        document_id,
                        version_id,
                        acl,
                        window,
                        classification,
                    )
                )
                continue

            current.append(block)

        flush_current()
        chunks = self._compact_technical_editorial_chunks(chunks, window)
        return self._compact_technical_heading_chunks(chunks, window)

    def _flush_technical_buffer(
        self,
        blocks: list[SourceBlock],
        document_id: str,
        version_id: str,
        acl: ACL,
        window: ChunkWindow,
        classification: DocumentClassification,
    ) -> list[ChunkContract]:
        if not blocks:
            return []
        text = self._join_block_texts(blocks)
        if not text.strip():
            return []

        block_types = {block.block_type for block in blocks}
        extra_metadata: dict[str, object] | None = None
        if block_types == {"preamble"}:
            extra_metadata = {"chunk_kind": "preamble"}

        if self._estimate_tokens(text) <= window.max_tokens:
            return [
                self._make_chunk(
                    blocks,
                    document_id,
                    version_id,
                    acl,
                    table_chunk=False,
                    classification=classification,
                    extra_metadata=extra_metadata,
                )
            ]

        units = self._technical_text_units(text)
        packed = self._pack_text_units(
            units,
            window.max_tokens,
            allow_overlap=True,
            overlap_ratio=window.overlap_ratio,
        )
        return [
            self._make_chunk(
                blocks,
                document_id,
                version_id,
                acl,
                table_chunk=False,
                classification=classification,
                override_text="\n\n".join(chunk_units),
                extra_metadata={
                    **(extra_metadata or {}),
                    "split_parts": len(packed),
                    "split_part": index,
                },
            )
            for index, chunk_units in enumerate(packed, start=1)
        ]

    def _build_technical_table_chunks(
        self,
        block: SourceBlock,
        document_id: str,
        version_id: str,
        acl: ACL,
        window: ChunkWindow,
        classification: DocumentClassification,
    ) -> list[ChunkContract]:
        lines = self._non_empty_lines(block.text)
        if not lines:
            return []

        header = lines[0]
        body = lines[1:] or [header]
        body_chunks = self._pack_text_units(body, window.max_tokens, allow_overlap=False)
        kind = "reference_table" if block.block_type == "reference_table" else "table"
        chunks: list[ChunkContract] = []
        for part_index, units in enumerate(body_chunks, start=1):
            text = "\n".join([header, *units]) if units and units[0] != header else "\n".join(units)
            chunks.append(
                self._make_chunk(
                    [block],
                    document_id,
                    version_id,
                    acl,
                    table_chunk=True,
                    classification=classification,
                    override_text=text,
                    extra_metadata={
                        "chunk_kind": kind,
                        "table_part": part_index,
                        "table_rows": len(units),
                    },
                )
            )
        return chunks

    def _compact_technical_editorial_chunks(
        self,
        chunks: list[ChunkContract],
        window: ChunkWindow,
    ) -> list[ChunkContract]:
        if len(chunks) < 2:
            return chunks

        body_start = next(
            (
                index
                for index, chunk in enumerate(chunks)
                if self._is_technical_body_chunk(chunk)
            ),
            None,
        )
        if body_start is None or body_start <= 1:
            return chunks

        prefix = chunks[:body_start]
        suffix = chunks[body_start:]
        compacted: list[ChunkContract] = []
        current_group: list[ChunkContract] = []
        current_tokens = 0

        for chunk in prefix:
            approx = int(chunk.metadata.get("approx_tokens", 0))
            if current_group and current_tokens + approx > window.max_tokens:
                compacted.append(self._merge_technical_chunk_group(current_group))
                current_group = [chunk]
                current_tokens = approx
            else:
                current_group.append(chunk)
                current_tokens += approx

        if current_group:
            compacted.append(self._merge_technical_chunk_group(current_group))

        return compacted + suffix

    def _is_technical_body_chunk(self, chunk: ChunkContract) -> bool:
        path = chunk.metadata.get("section_path", [])
        if not path:
            return False
        top_label = str(path[0]).strip()
        return bool(re.match(r"^(Secci[oó]n|Section)\s+\d+\b", top_label, re.IGNORECASE))

    def _merge_technical_chunk_group(self, group: list[ChunkContract]) -> ChunkContract:
        if len(group) == 1:
            return group[0]

        first = group[0]
        combined_text = "\n\n".join(chunk.text.strip() for chunk in group if chunk.text.strip()).strip()
        combined_path = self._combined_technical_editorial_path(group)
        block_types = list(
            dict.fromkeys(
                block_type
                for chunk in group
                for block_type in chunk.metadata.get("block_types", [])
            )
        )
        kinds = [str(chunk.metadata.get("chunk_kind") or "") for chunk in group]
        section_only_small = all(
            kind != "section" or int(chunk.metadata.get("approx_tokens", 0)) <= 80
            for kind, chunk in zip(kinds, group, strict=False)
        )
        has_tableish = any(kind in {"table", "reference_table"} for kind in kinds)
        merged_kind = "reference_table" if has_tableish and section_only_small else "preamble"
        table_chunk = merged_kind == "reference_table"

        section_label = combined_path[0] if combined_path else None
        subsection_label = combined_path[1] if len(combined_path) > 1 else None
        retrieval_context = " > ".join(combined_path)

        metadata = {
            "page_start": min(int(chunk.metadata.get("page_start", 0)) for chunk in group),
            "page_end": max(int(chunk.metadata.get("page_end", 0)) for chunk in group),
            "section_path": combined_path,
            "block_types": block_types,
            "approx_tokens": self._estimate_tokens(combined_text),
            "source_block_count": len(
                {
                    ref
                    for chunk in group
                    for ref in chunk.block_refs
                }
            ),
            "table_chunk": table_chunk,
            "merged_editorial_prefix": True,
            "document_class": first.metadata.get("document_class"),
            "document_family": first.metadata.get("document_family"),
            "chunking_strategy": first.metadata.get("chunking_strategy"),
            "classification_source": first.metadata.get("classification_source"),
            "book_label": None,
            "title_label": None,
            "chapter_label": None,
            "article_label": None,
            "section_label": section_label,
            "subsection_label": subsection_label,
            "hierarchy_path": combined_path,
            "retrieval_context": retrieval_context,
            "contextualized_text": f"{retrieval_context}\n{combined_text}".strip(),
            "chunk_kind": merged_kind,
        }

        return ChunkContract(
            chunk_id=f"c_{uuid4().hex[:12]}",
            document_id=first.document_id,
            version_id=first.version_id,
            block_refs=list(dict.fromkeys(ref for chunk in group for ref in chunk.block_refs)),
            text=combined_text,
            metadata=metadata,
            acl=first.acl,
            index_status="pending",
        )

    def _combined_technical_editorial_path(self, group: list[ChunkContract]) -> list[str]:
        preferred = next(
            (
                chunk.metadata.get("section_path", [])
                for chunk in group
                if chunk.metadata.get("chunk_kind") in {"reference_table", "table"}
                and chunk.metadata.get("section_path")
            ),
            None,
        )
        if preferred:
            return preferred
        return group[0].metadata.get("section_path", []) or ["Document"]

    def _compact_technical_heading_chunks(
        self,
        chunks: list[ChunkContract],
        window: ChunkWindow,
    ) -> list[ChunkContract]:
        if len(chunks) < 2:
            return chunks

        compacted: list[ChunkContract] = []
        index = 0
        while index < len(chunks):
            current = chunks[index]
            if index < len(chunks) - 1 and self._is_small_heading_only_technical_chunk(current):
                next_chunk = chunks[index + 1]
                if self._can_merge_technical_heading(current, next_chunk, window):
                    compacted.append(self._merge_technical_heading_pair(current, next_chunk))
                    index += 2
                    continue
            compacted.append(current)
            index += 1

        return compacted

    def _is_small_heading_only_technical_chunk(self, chunk: ChunkContract) -> bool:
        if chunk.metadata.get("document_class") != "technical_standard":
            return False
        if chunk.metadata.get("chunk_kind") != "section":
            return False
        if int(chunk.metadata.get("approx_tokens", 0)) > 24:
            return False
        text = chunk.text.strip()
        lines = self._non_empty_lines(text)
        if len(lines) != 1:
            return False
        return lines[0].startswith("## ")

    def _can_merge_technical_heading(
        self,
        heading_chunk: ChunkContract,
        next_chunk: ChunkContract,
        window: ChunkWindow,
    ) -> bool:
        if next_chunk.metadata.get("document_class") != "technical_standard":
            return False
        if next_chunk.metadata.get("chunk_kind") not in {"section", "table"}:
            return False

        heading_path = heading_chunk.metadata.get("section_path", [])
        next_path = next_chunk.metadata.get("section_path", [])
        if not heading_path or not next_path:
            return False

        combined_tokens = int(heading_chunk.metadata.get("approx_tokens", 0)) + int(
            next_chunk.metadata.get("approx_tokens", 0)
        )
        if combined_tokens > window.max_tokens:
            return False

        if len(heading_path) == 1:
            return next_path[0] == heading_path[0]

        return (
            len(next_path) >= 2
            and next_path[0] == heading_path[0]
        )

    def _merge_technical_heading_pair(
        self,
        heading_chunk: ChunkContract,
        next_chunk: ChunkContract,
    ) -> ChunkContract:
        heading_path = heading_chunk.metadata.get("section_path", [])
        next_path = next_chunk.metadata.get("section_path", [])

        if len(heading_path) == 1 and len(next_path) >= 2 and next_path[0] == heading_path[0]:
            combined_path = next_path
        else:
            combined_path = heading_path or next_path

        combined_text = f"{heading_chunk.text.strip()}\n\n{next_chunk.text.strip()}".strip()
        block_types = list(
            dict.fromkeys(
                [*heading_chunk.metadata.get("block_types", []), *next_chunk.metadata.get("block_types", [])]
            )
        )
        retrieval_context = " > ".join(combined_path)

        metadata = {
            "page_start": min(
                int(heading_chunk.metadata.get("page_start", 0)),
                int(next_chunk.metadata.get("page_start", 0)),
            ),
            "page_end": max(
                int(heading_chunk.metadata.get("page_end", 0)),
                int(next_chunk.metadata.get("page_end", 0)),
            ),
            "section_path": combined_path,
            "block_types": block_types,
            "approx_tokens": self._estimate_tokens(combined_text),
            "source_block_count": len({*heading_chunk.block_refs, *next_chunk.block_refs}),
            "table_chunk": bool(next_chunk.metadata.get("table_chunk")),
            "merged_heading_chunk": True,
            "document_class": next_chunk.metadata.get("document_class"),
            "document_family": next_chunk.metadata.get("document_family"),
            "chunking_strategy": next_chunk.metadata.get("chunking_strategy"),
            "classification_source": next_chunk.metadata.get("classification_source"),
            "book_label": None,
            "title_label": None,
            "chapter_label": None,
            "article_label": None,
            "section_label": combined_path[0] if combined_path else None,
            "subsection_label": combined_path[1] if len(combined_path) > 1 else None,
            "hierarchy_path": combined_path,
            "retrieval_context": retrieval_context,
            "contextualized_text": f"{retrieval_context}\n{combined_text}".strip(),
            "chunk_kind": next_chunk.metadata.get("chunk_kind", "section"),
        }

        return ChunkContract(
            chunk_id=f"c_{uuid4().hex[:12]}",
            document_id=heading_chunk.document_id,
            version_id=heading_chunk.version_id,
            block_refs=list(dict.fromkeys([*heading_chunk.block_refs, *next_chunk.block_refs])),
            text=combined_text,
            metadata=metadata,
            acl=heading_chunk.acl,
            index_status="pending",
        )

    def _build_normative_chunks(
        self,
        blocks: list[SourceBlock],
        document_id: str,
        version_id: str,
        acl: ACL,
        window: ChunkWindow,
        classification: DocumentClassification,
    ) -> list[ChunkContract]:
        chunks: list[ChunkContract] = []
        preamble_blocks: list[SourceBlock] = []
        current_article_blocks: list[SourceBlock] = []
        current_article: str | None = None
        saw_article = False

        for block in blocks:
            article = self._article_label(block.section_path)

            if block.block_type == "table":
                if current_article_blocks:
                    chunks.extend(
                        self._flush_normative_article(
                            current_article_blocks,
                            document_id,
                            version_id,
                            acl,
                            window,
                            classification,
                        )
                    )
                    current_article_blocks = []
                    current_article = None
                chunks.extend(self._build_table_chunks(block, document_id, version_id, acl, window, classification))
                continue

            if article:
                if not saw_article and preamble_blocks:
                    chunks.extend(
                        self._flush_preamble(
                            preamble_blocks,
                            document_id,
                            version_id,
                            acl,
                            classification,
                        )
                    )
                    preamble_blocks = []
                saw_article = True
                if current_article and article != current_article:
                    chunks.extend(
                        self._flush_normative_article(
                            current_article_blocks,
                            document_id,
                            version_id,
                            acl,
                            window,
                            classification,
                        )
                    )
                    current_article_blocks = [block]
                    current_article = article
                else:
                    current_article_blocks.append(block)
                    current_article = article
                continue

            if not saw_article:
                preamble_blocks.append(block)
                continue

            if current_article_blocks and not self._is_heading_only_block(block.text):
                current_article_blocks.append(block)

        if preamble_blocks:
            chunks.extend(
                self._flush_preamble(
                    preamble_blocks,
                    document_id,
                    version_id,
                    acl,
                    classification,
                )
            )
        if current_article_blocks:
            chunks.extend(
                self._flush_normative_article(
                    current_article_blocks,
                    document_id,
                    version_id,
                    acl,
                    window,
                    classification,
                )
            )
        return chunks

    def _flush_preamble(
        self,
        blocks: list[SourceBlock],
        document_id: str,
        version_id: str,
        acl: ACL,
        classification: DocumentClassification,
    ) -> list[ChunkContract]:
        text = self._join_block_texts(blocks)
        if not text.strip():
            return []
        return [
            self._make_chunk(
                blocks,
                document_id,
                version_id,
                acl,
                table_chunk=False,
                classification=classification,
                extra_metadata={"chunk_kind": "preamble"},
            )
        ]

    def _flush_normative_article(
        self,
        blocks: list[SourceBlock],
        document_id: str,
        version_id: str,
        acl: ACL,
        window: ChunkWindow,
        classification: DocumentClassification,
    ) -> list[ChunkContract]:
        if not blocks:
            return []
        text = self._join_block_texts(blocks)
        if self._estimate_tokens(text) <= window.max_tokens:
            return [
                self._make_chunk(
                    blocks,
                    document_id,
                    version_id,
                    acl,
                    table_chunk=False,
                    classification=classification,
                    extra_metadata={"chunk_kind": "article"},
                )
            ]

        units = self._normative_text_units(text, window.max_tokens)
        packed = self._pack_text_units(
            units,
            window.max_tokens,
            allow_overlap=True,
            overlap_ratio=window.overlap_ratio,
        )
        return [
            self._make_chunk(
                blocks,
                document_id,
                version_id,
                acl,
                table_chunk=False,
                classification=classification,
                override_text="\n\n".join(chunk_units),
                extra_metadata={
                    "chunk_kind": "article_subchunk",
                    "split_parts": len(packed),
                    "split_part": index,
                },
            )
            for index, chunk_units in enumerate(packed, start=1)
        ]

    def _should_flush_on_new_section(
        self,
        current: list[SourceBlock],
        next_block: SourceBlock,
        window: ChunkWindow,
        classification: DocumentClassification,
    ) -> bool:
        current_tokens = self._blocks_tokens(current)
        current_path = current[-1].section_path or []
        next_path = next_block.section_path or []
        if classification.chunking_strategy == "generic_structural":
            if self._should_flush_general_structural(current_path, next_path, current_tokens, window):
                return True
        current_section = current_path[0] if current_path else ""
        next_section = next_path[0] if next_path else ""
        if not current_section or not next_section or current_section == next_section:
            current_article = self._article_label(current_path)
            next_article = self._article_label(next_path)
            if current_article and next_article and current_article != next_article:
                return True
            if not current_article and next_article and current_tokens >= 80:
                return True
            return False

        current_article = self._article_label(current_path)
        next_article = self._article_label(next_path)
        if current_article and next_article and current_article != next_article:
            return True
        if not current_article and next_article and current_tokens >= 80:
            return True

        current_chapter = self._chapter_label(current_path)
        next_chapter = self._chapter_label(next_path)
        if current_chapter and next_chapter and current_chapter != next_chapter:
            if current_article:
                return True
            return current_tokens >= max(120, window.min_tokens // 3)

        if self._label_priority(next_section) >= 4 and current_tokens >= max(120, window.min_tokens // 3):
            return True
        if current_tokens < window.min_tokens:
            return False
        if self._label_priority(next_section) >= 3 and self._label_priority(current_section) >= 3:
            return True
        if self._label_priority(next_section) > self._label_priority(current_section):
            return True
        return False

    def _should_flush_general_structural(
        self,
        current_path: list[str],
        next_path: list[str],
        current_tokens: int,
        window: ChunkWindow,
    ) -> bool:
        if not current_path or not next_path or current_path == next_path:
            return False

        current_leaf = current_path[-1]
        next_leaf = next_path[-1]
        same_parent = len(current_path) >= 2 and len(next_path) >= 2 and current_path[:-1] == next_path[:-1]
        prefix_transition = (
            len(current_path) != len(next_path)
            and (next_path[: len(current_path)] == current_path or current_path[: len(next_path)] == next_path)
        )
        minimum_flush = max(40, window.min_tokens // 10)
        prefix_flush = max(40, window.min_tokens // 10)

        if same_parent and current_leaf != next_leaf:
            return True
        if prefix_transition and current_tokens >= prefix_flush:
            return True

        if current_path[0] != next_path[0] and (len(current_path) > 1 or current_tokens >= minimum_flush):
            return True

        return False

    def _flush_text_buffer(
        self,
        blocks: list[SourceBlock],
        document_id: str,
        version_id: str,
        acl: ACL,
        classification: DocumentClassification,
    ) -> list[ChunkContract]:
        return [
            self._make_chunk(
                blocks,
                document_id,
                version_id,
                acl,
                table_chunk=False,
                classification=classification,
            )
        ]

    def _flush_with_overlap(
        self,
        blocks: list[SourceBlock],
        document_id: str,
        version_id: str,
        acl: ACL,
        window: ChunkWindow,
        classification: DocumentClassification,
    ) -> tuple[ChunkContract, list[SourceBlock]]:
        chunk = self._make_chunk(
            blocks,
            document_id,
            version_id,
            acl,
            table_chunk=False,
            classification=classification,
        )
        overlap_blocks: list[SourceBlock] = []
        if window.overlap_ratio > 0:
            overlap_blocks = self._select_overlap_blocks(
                blocks,
                ceil(window.target_tokens * window.overlap_ratio),
                classification,
            )
        return chunk, overlap_blocks

    def _build_table_chunks(
        self,
        block: SourceBlock,
        document_id: str,
        version_id: str,
        acl: ACL,
        window: ChunkWindow,
        classification: DocumentClassification,
    ) -> list[ChunkContract]:
        lines = self._non_empty_lines(block.text)
        if not lines:
            return []

        header = lines[0]
        body = lines[1:] or [header]
        body_chunks = self._pack_text_units(body, window.max_tokens, allow_overlap=False)
        chunks: list[ChunkContract] = []
        for part_index, units in enumerate(body_chunks, start=1):
            text = "\n".join([header, *units]) if units and units[0] != header else "\n".join(units)
            chunks.append(
                self._make_chunk(
                    [block],
                    document_id,
                    version_id,
                    acl,
                    table_chunk=True,
                    classification=classification,
                    override_text=text,
                    extra_metadata={"table_part": part_index, "table_rows": len(units)},
                )
            )
        return chunks

    def _split_single_block(
        self,
        block: SourceBlock,
        document_id: str,
        version_id: str,
        acl: ACL,
        window: ChunkWindow,
        classification: DocumentClassification,
    ) -> list[ChunkContract]:
        if classification.chunking_strategy == "technical_structural":
            units = self._technical_text_units(block.text)
        else:
            units = self._text_units(block.text)
        packed = self._pack_text_units(
            units,
            window.max_tokens,
            allow_overlap=True,
            overlap_ratio=window.overlap_ratio,
        )
        return [
            self._make_chunk(
                [block],
                document_id,
                version_id,
                acl,
                table_chunk=False,
                classification=classification,
                override_text="\n\n".join(chunk_units),
                extra_metadata={"split_block": True, "split_parts": len(packed), "split_part": index},
            )
            for index, chunk_units in enumerate(packed, start=1)
        ]

    def _technical_text_units(self, text: str) -> list[str]:
        lines = self._non_empty_lines(text)
        if not lines:
            return []

        segments: list[list[str]] = []
        current: list[str] = []
        current_mode = "text"

        def flush() -> None:
            nonlocal current
            if current:
                segments.append(current)
                current = []

        for line in lines:
            if self._looks_like_markdown_table_line(line):
                if current_mode != "table":
                    flush()
                    current_mode = "table"
                current.append(line)
                continue

            if current_mode == "table":
                flush()
                current_mode = "text"

            heading = self._technical_heading(line)
            if heading and current:
                flush()
            current.append(line)

        flush()

        if len(segments) > 1:
            return ["\n".join(segment).strip() for segment in segments if any(part.strip() for part in segment)]

        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        if len(paragraphs) > 1:
            return paragraphs

        return self._text_units(text)

    def _pack_text_units(
        self,
        units: list[str],
        max_tokens: int,
        *,
        allow_overlap: bool,
        overlap_ratio: float = 0.0,
    ) -> list[list[str]]:
        packed: list[list[str]] = []
        current: list[str] = []
        for unit in units:
            for expanded_unit in self._explode_oversized_unit(unit, max_tokens):
                unit_tokens = self._estimate_tokens(expanded_unit)
                current_tokens = self._estimate_tokens("\n\n".join(current)) if current else 0
                if current and current_tokens + unit_tokens > max_tokens:
                    packed.append(current)
                    if allow_overlap and overlap_ratio > 0:
                        current = self._select_overlap_units(current, ceil(max_tokens * overlap_ratio))
                        while current and self._estimate_tokens("\n\n".join([*current, expanded_unit])) > max_tokens:
                            current.pop(0)
                    else:
                        current = []
                current.append(expanded_unit)
        if current:
            packed.append(current)
        return packed

    def _select_overlap_blocks(
        self,
        blocks: list[SourceBlock],
        token_budget: int,
        classification: DocumentClassification,
    ) -> list[SourceBlock]:
        overlap: list[SourceBlock] = []
        running = 0
        anchor = blocks[-1] if blocks else None
        for block in reversed(blocks):
            if (
                classification.chunking_strategy == "normative_hierarchical"
                and anchor is not None
                and not self._same_normative_unit(block.section_path, anchor.section_path)
            ):
                break
            overlap.insert(0, block)
            running += self._estimate_tokens(block.text)
            if running >= token_budget:
                break
        return overlap

    def _select_overlap_units(self, units: list[str], token_budget: int) -> list[str]:
        overlap: list[str] = []
        running = 0
        for unit in reversed(units):
            overlap.insert(0, unit)
            running += self._estimate_tokens(unit)
            if running >= token_budget:
                break
        return overlap

    def _seed_next_buffer(
        self,
        overlap_blocks: list[SourceBlock],
        next_block: SourceBlock,
        max_tokens: int,
    ) -> list[SourceBlock]:
        seeded = [*overlap_blocks, next_block]
        while len(seeded) > 1 and self._blocks_tokens(seeded) > max_tokens:
            seeded.pop(0)
        return seeded

    def _make_chunk(
        self,
        blocks: list[SourceBlock],
        document_id: str,
        version_id: str,
        acl: ACL,
        *,
        table_chunk: bool,
        classification: DocumentClassification,
        override_text: str | None = None,
        extra_metadata: dict[str, object] | None = None,
    ) -> ChunkContract:
        text = override_text or self._join_block_texts(blocks)
        block_types = list(dict.fromkeys(block.block_type for block in blocks))
        section_path = self._best_section_path(blocks)
        book_label = self._book_label(section_path)
        title_label = self._title_label(section_path)
        chapter_label = self._chapter_label(section_path)
        article_label = self._article_label(section_path)
        section_label = self._section_label(section_path)
        subsection_label = self._subsection_label(section_path)
        if classification.document_class == "technical_standard":
            section_label = section_path[0] if section_path else None
            subsection_label = section_path[1] if len(section_path) > 1 else None
            article_label = None
            chapter_label = None
        elif classification.document_family != "normative":
            section_label = section_label or (section_path[0] if section_path else None)
            subsection_label = subsection_label or (section_path[1] if len(section_path) > 1 else None)
        if table_chunk:
            chunk_kind = "table"
        elif classification.document_class == "legal_normative":
            chunk_kind = "article"
        elif classification.chunking_strategy in {"technical_structural", "generic_structural"}:
            chunk_kind = "section"
        else:
            chunk_kind = "freeform"
        if "preamble" in block_types:
            chunk_kind = "preamble"
        elif "reference_table" in block_types:
            chunk_kind = "reference_table"
        retrieval_context = " > ".join(section_path)
        metadata = {
            "page_start": min(block.page for block in blocks),
            "page_end": max(block.page for block in blocks),
            "section_path": section_path,
            "block_types": block_types,
            "approx_tokens": self._estimate_tokens(text),
            "source_block_count": len({block.block_id for block in blocks}),
            "table_chunk": table_chunk,
            "document_class": classification.document_class,
            "document_family": classification.document_family,
            "chunking_strategy": classification.chunking_strategy,
            "classification_source": classification.classification_source,
            "book_label": book_label,
            "title_label": title_label,
            "chapter_label": chapter_label,
            "article_label": article_label,
            "section_label": section_label,
            "subsection_label": subsection_label,
            "hierarchy_path": section_path,
            "retrieval_context": retrieval_context,
            "contextualized_text": f"{retrieval_context}\n{text}" if retrieval_context else text,
            "chunk_kind": chunk_kind,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return ChunkContract(
            chunk_id=f"c_{uuid4().hex[:12]}",
            document_id=document_id,
            version_id=version_id,
            block_refs=list(dict.fromkeys(block.block_id for block in blocks)),
            text=text,
            metadata=metadata,
            acl=acl,
            index_status="pending",
        )

    def _merge_small_chunks(self, chunks: list[ChunkContract], window: ChunkWindow) -> list[ChunkContract]:
        if len(chunks) < 2:
            return chunks

        merged: list[ChunkContract] = []
        minimum = max(200, int(window.min_tokens * 0.75))
        index = 0
        while index < len(chunks):
            chunk = chunks[index]
            approx = int(chunk.metadata.get("approx_tokens", 0))
            if (
                approx >= minimum
                or chunk.metadata.get("table_chunk") is True
                or index == len(chunks) - 1
            ):
                merged.append(chunk)
                index += 1
                continue

            next_chunk = chunks[index + 1]
            if next_chunk.metadata.get("table_chunk") is True:
                merged.append(chunk)
                index += 1
                continue
            if (
                chunk.metadata.get("chunking_strategy") == "generic_structural"
                and not self._can_merge_small_general_structured_chunks(
                    chunk,
                    next_chunk,
                    minimum=minimum,
                )
            ):
                merged.append(chunk)
                index += 1
                continue

            combined_text = f"{chunk.text}\n\n{next_chunk.text}".strip()
            combined_tokens = self._estimate_tokens(combined_text)
            if combined_tokens > window.max_tokens + ceil(window.target_tokens * window.overlap_ratio):
                merged.append(chunk)
                index += 1
                continue

            left_path = chunk.metadata.get("section_path", [])
            right_path = next_chunk.metadata.get("section_path", [])
            normative_merge = False
            article_span_label: str | None = None
            if chunk.metadata.get("chunking_strategy") == "normative_hierarchical":
                if self._same_normative_unit(left_path, right_path):
                    normative_merge = True
                elif self._can_merge_small_normative_articles(
                    chunk,
                    next_chunk,
                    minimum=minimum,
                    target_tokens=window.target_tokens,
                ):
                    normative_merge = True
                    article_span_label = self._article_span_label(left_path, right_path)
                else:
                    merged.append(chunk)
                    index += 1
                    continue

            combined_path = self._choose_combined_path(left_path, right_path)
            if normative_merge and article_span_label:
                chapter = self._chapter_label(combined_path) or self._chapter_label(left_path) or self._chapter_label(right_path)
                combined_path = [chapter, article_span_label] if chapter else [article_span_label]
            merged.append(
                ChunkContract(
                    chunk_id=f"c_{uuid4().hex[:12]}",
                    document_id=chunk.document_id,
                    version_id=chunk.version_id,
                    block_refs=list(dict.fromkeys([*chunk.block_refs, *next_chunk.block_refs])),
                    text=combined_text,
                    metadata={
                        "page_start": min(chunk.metadata["page_start"], next_chunk.metadata["page_start"]),
                        "page_end": max(chunk.metadata["page_end"], next_chunk.metadata["page_end"]),
                        "section_path": combined_path,
                        "block_types": list(
                            dict.fromkeys([*chunk.metadata["block_types"], *next_chunk.metadata["block_types"]])
                        ),
                        "approx_tokens": combined_tokens,
                        "source_block_count": len(
                            {*(chunk.block_refs), *(next_chunk.block_refs)}
                        ),
                        "table_chunk": False,
                        "merged_small_chunk": True,
                        "document_class": chunk.metadata.get("document_class"),
                        "document_family": chunk.metadata.get("document_family"),
                        "chunking_strategy": chunk.metadata.get("chunking_strategy"),
                        "classification_source": chunk.metadata.get("classification_source"),
                        "book_label": chunk.metadata.get("book_label") or next_chunk.metadata.get("book_label"),
                        "title_label": chunk.metadata.get("title_label") or next_chunk.metadata.get("title_label"),
                        "chapter_label": chunk.metadata.get("chapter_label") or next_chunk.metadata.get("chapter_label"),
                        "article_label": chunk.metadata.get("article_label") or next_chunk.metadata.get("article_label"),
                        "section_label": chunk.metadata.get("section_label") or next_chunk.metadata.get("section_label"),
                        "subsection_label": chunk.metadata.get("subsection_label") or next_chunk.metadata.get("subsection_label"),
                        "hierarchy_path": combined_path,
                        "retrieval_context": " > ".join(combined_path),
                        "contextualized_text": f"{' > '.join(combined_path)}\n{combined_text}".strip(),
                        "chunk_kind": chunk.metadata.get("chunk_kind") or next_chunk.metadata.get("chunk_kind"),
                    },
                    acl=chunk.acl,
                    index_status="pending",
                )
            )
            index += 2
        return merged

    def _can_merge_small_general_structured_chunks(
        self,
        chunk: ChunkContract,
        next_chunk: ChunkContract,
        *,
        minimum: int,
    ) -> bool:
        left_path = chunk.metadata.get("section_path", [])
        right_path = next_chunk.metadata.get("section_path", [])
        if not left_path or not right_path:
            return False
        if left_path == right_path:
            return True

        left_tokens = int(chunk.metadata.get("approx_tokens", 0))
        left_is_prefix = len(left_path) < len(right_path) and right_path[: len(left_path)] == left_path
        if left_is_prefix and left_tokens <= max(60, minimum // 3):
            return True

        return False

    def _choose_combined_path(self, left: list[str], right: list[str]) -> list[str]:
        return max([left, right], key=self._section_path_rank)

    def _same_normative_unit(self, left: list[str], right: list[str]) -> bool:
        left_article = self._article_label(left)
        right_article = self._article_label(right)
        if left_article or right_article:
            return bool(left_article and right_article and left_article == right_article)
        left_chapter = self._chapter_label(left)
        right_chapter = self._chapter_label(right)
        if left_chapter or right_chapter:
            return bool(left_chapter and right_chapter and left_chapter == right_chapter)
        return left == right

    def _book_label(self, labels: list[str]) -> str | None:
        return next((label for label in labels if self._label_kind(label) == "book"), None)

    def _title_label(self, labels: list[str]) -> str | None:
        return next((label for label in labels if self._label_kind(label) == "title"), None)

    def _chapter_label(self, labels: list[str]) -> str | None:
        return next((label for label in labels if self._label_kind(label) == "chapter"), None)

    def _article_label(self, labels: list[str]) -> str | None:
        return next((label for label in labels if self._label_kind(label) == "article"), None)

    def _section_label(self, labels: list[str]) -> str | None:
        return next((label for label in labels if self._label_kind(label) == "section"), None)

    def _subsection_label(self, labels: list[str]) -> str | None:
        roman = next((label for label in labels if self._label_kind(label) == "roman"), None)
        if roman:
            return roman
        return next((label for label in labels if self._label_kind(label) == "item"), None)

    def _can_merge_small_normative_articles(
        self,
        left_chunk: ChunkContract,
        right_chunk: ChunkContract,
        *,
        minimum: int,
        target_tokens: int,
    ) -> bool:
        left_path = left_chunk.metadata.get("section_path", [])
        right_path = right_chunk.metadata.get("section_path", [])
        left_article = self._article_label(left_path)
        right_article = self._article_label(right_path)
        if not (left_article and right_article):
            return False
        if (self._chapter_label(left_path) or "") != (self._chapter_label(right_path) or ""):
            return False
        if not self._are_consecutive_articles(left_article, right_article):
            return False

        left_tokens = int(left_chunk.metadata.get("approx_tokens", 0))
        right_tokens = int(right_chunk.metadata.get("approx_tokens", 0))
        combined = left_tokens + right_tokens
        return (
            left_tokens < minimum
            and right_tokens < max(minimum, 260)
            and combined <= max(target_tokens, minimum + 160)
        )

    def _are_consecutive_articles(self, left_article: str, right_article: str) -> bool:
        left_number = self._article_number(left_article)
        right_number = self._article_number(right_article)
        return left_number is not None and right_number is not None and right_number == left_number + 1

    def _article_number(self, article_label: str | None) -> int | None:
        if not article_label:
            return None
        match = re.search(r"(\d+)", article_label)
        return int(match.group(1)) if match else None

    def _article_span_label(self, left_path: list[str], right_path: list[str]) -> str | None:
        left_article = self._article_label(left_path)
        right_article = self._article_label(right_path)
        left_number = self._article_number(left_article)
        right_number = self._article_number(right_article)
        if left_number is None or right_number is None:
            return None
        if left_number == right_number:
            return left_article
        return f"Artos. {left_number}-{right_number}"

    def _best_section_path(self, blocks: list[SourceBlock]) -> list[str]:
        ranked = [block.section_path for block in blocks if block.section_path]
        article_ranked = [path for path in ranked if self._article_label(path)]
        if article_ranked:
            return max(article_ranked, key=lambda path: (len(path), self._section_path_rank(path)))
        if ranked:
            return max(ranked, key=lambda path: (len(path), self._section_path_rank(path)))
        return ["Document"]

    def _join_block_texts(self, blocks: Iterable[SourceBlock]) -> str:
        return "\n\n".join(block.text.strip() for block in blocks if block.text.strip())

    def _blocks_tokens(self, blocks: list[SourceBlock]) -> int:
        return sum(self._estimate_tokens(block.text) for block in blocks)

    def _is_heading_only_block(self, text: str) -> bool:
        lines = self._non_empty_lines(text)
        if not lines:
            return True
        if len(lines) > 4:
            return False
        return all(
            self._is_section_boundary(line)
            or (line.isupper() and len(line.split()) <= 12)
            for line in lines
        )

    def _text_units(self, text: str) -> list[str]:
        article_splits = self._split_on_boundaries(text)
        if len(article_splits) > 1:
            return article_splits

        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        if len(paragraphs) > 1:
            return paragraphs

        lines = [part.strip() for part in text.splitlines() if part.strip()]
        if len(lines) > 1:
            return lines

        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
        return sentences or [text.strip()]

    def _normative_text_units(self, text: str, max_tokens: int) -> list[str]:
        stripped = text.strip()
        if not stripped:
            return []

        outline_units = self._split_normative_outline(self._non_empty_lines(stripped))
        units = outline_units if len(outline_units) > 1 else self._text_units(stripped)

        expanded: list[str] = []
        for unit in units:
            expanded.extend(self._explode_oversized_unit(unit, max_tokens, normative=True))
        return expanded or [stripped]

    def _split_normative_outline(self, lines: list[str]) -> list[str]:
        if not lines:
            return []

        boundary_patterns = [
            re.compile(r"^\d+\)\s*"),
            re.compile(r"^[A-Za-zÁÉÍÓÚÑáéíóúñ]\)\s*"),
            re.compile(r"^(?:[ivxlcdm]+)\)\s*", re.IGNORECASE),
            re.compile(r"^[A-ZÁÉÍÓÚÑ][^:]{1,80}:\s*$"),
        ]
        for pattern in boundary_patterns:
            matches = sum(1 for line in lines if pattern.match(line))
            if matches >= 2:
                return self._split_lines_on_pattern(lines, pattern)
        return ["\n".join(lines).strip()]

    def _split_lines_on_pattern(self, lines: list[str], pattern: re.Pattern[str]) -> list[str]:
        segments: list[list[str]] = []
        current: list[str] = []
        for line in lines:
            if pattern.match(line) and current:
                segments.append(current)
                current = [line]
            else:
                current.append(line)
        if current:
            segments.append(current)
        return ["\n".join(segment).strip() for segment in segments if any(part.strip() for part in segment)]

    def _explode_oversized_unit(
        self,
        unit: str,
        max_tokens: int,
        *,
        normative: bool = False,
    ) -> list[str]:
        stripped = unit.strip()
        if not stripped:
            return []
        if self._estimate_tokens(stripped) <= max_tokens:
            return [stripped]

        if normative:
            outline_units = self._split_normative_outline(self._non_empty_lines(stripped))
            if len(outline_units) > 1:
                exploded: list[str] = []
                for outline_unit in outline_units:
                    exploded.extend(
                        self._explode_oversized_unit(outline_unit, max_tokens, normative=True)
                    )
                return exploded

        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", stripped) if part.strip()]
        if len(paragraphs) > 1:
            return [
                "\n\n".join(chunk_units)
                for chunk_units in self._pack_text_units(paragraphs, max_tokens, allow_overlap=False)
            ]

        lines = [part.strip() for part in stripped.splitlines() if part.strip()]
        if len(lines) > 1:
            return [
                "\n".join(chunk_units)
                for chunk_units in self._pack_text_units(lines, max_tokens, allow_overlap=False)
            ]

        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", stripped) if part.strip()]
        if len(sentences) > 1:
            return [
                " ".join(chunk_units)
                for chunk_units in self._pack_text_units(sentences, max_tokens, allow_overlap=False)
            ]

        return self._word_window_units(stripped, max_tokens)

    def _word_window_units(self, text: str, max_tokens: int) -> list[str]:
        words = text.split()
        if not words:
            return []

        windows: list[str] = []
        current: list[str] = []
        for word in words:
            candidate = " ".join([*current, word]).strip()
            if current and self._estimate_tokens(candidate) > max_tokens:
                windows.append(" ".join(current).strip())
                current = [word]
            else:
                current.append(word)
        if current:
            windows.append(" ".join(current).strip())
        return windows

    def _split_on_boundaries(self, text: str) -> list[str]:
        lines = self._non_empty_lines(text)
        if not lines:
            return []
        segments: list[list[str]] = []
        current: list[str] = []
        for line in lines:
            if self._is_section_boundary(line) and current:
                segments.append(current)
                current = [line]
            else:
                current.append(line)
        if current:
            segments.append(current)
        return ["\n".join(segment).strip() for segment in segments if any(part.strip() for part in segment)]

    def _compact_section_label(self, text: str) -> str | None:
        candidate = re.sub(r"^#+\s*", "", text.strip())
        candidate = re.sub(r"\s+", " ", candidate)
        if not candidate:
            return None
        if re.match(r"^LIBRO\b", candidate, re.IGNORECASE) and not self._is_book_heading(candidate):
            return None
        if re.match(r"^T[IÍ]TULO\b", candidate, re.IGNORECASE) and not self._is_title_heading(candidate):
            return None
        if re.match(r"^CAP[IÍ]TULO\b", candidate, re.IGNORECASE) and not self._is_chapter_heading(candidate):
            return None
        if re.match(r"^ART\.\s*\d", candidate, re.IGNORECASE) and not self._is_article_heading(candidate):
            return None
        if re.match(r"^(?:ARTO\.|ART[IÍ]CULO)\b", candidate, re.IGNORECASE) and not self._is_article_heading(candidate):
            return None

        book_match = re.match(r"^(LIBRO\s*[IVXLC0-9]+)", candidate, re.IGNORECASE)
        if book_match:
            return book_match.group(1).upper()

        title_match = re.match(r"^(T[IÍ]TULO\s*[IVXLC0-9]+)", candidate, re.IGNORECASE)
        if title_match:
            return title_match.group(1).upper()

        chapter_match = re.match(r"^(CAP[IÍ]TULO\s*[IVXLC0-9]+)", candidate, re.IGNORECASE)
        if chapter_match:
            return chapter_match.group(1).upper().replace("Í", "I")

        article_match = (
            re.match(r"^ART(?:[IÍ]CULO|O)?\.?\s*(\d+)", candidate, re.IGNORECASE)
            if self._is_article_heading(candidate)
            else None
        )
        if article_match:
            return f"Arto. {article_match.group(1)}"

        item_match = re.match(r"^(\d+)\)", candidate)
        if item_match:
            return f"Item {item_match.group(1)}"

        roman_match = re.match(r"^([IVXLCM]+)$", candidate, re.IGNORECASE)
        if roman_match:
            return roman_match.group(1).upper()

        if re.match(r"^(REGLAMENTO|NORMATIVA|POL[IÍ]TICA|MANUAL)\b", candidate, re.IGNORECASE):
            words = candidate.split()
            return " ".join(words[:5]).rstrip(" -,:;")

        if candidate.isupper():
            return candidate[:72].rstrip(" -,:;")

        if len(candidate) > 72:
            return f"{candidate[:69].rstrip()}..."

        return candidate

    def _is_article_heading(self, text: str) -> bool:
        candidate = re.sub(r"^#+\s*", "", text.strip())
        match = re.match(
            r"^ART(?:[IÍ]CULO|O)?\.?\s*(\d+)(.*)$",
            candidate,
            re.IGNORECASE,
        )
        if not match:
            return False

        remainder = match.group(2).strip()
        if not remainder:
            return True

        remainder = re.sub(
            r"^(?:bis|ter|quater|quinquies)\b\.?\s*",
            "",
            remainder,
            flags=re.IGNORECASE,
        )
        if not remainder:
            return True

        if remainder[0] in ".:-":
            return True

        return bool(re.match(r"^[A-ZÁÉÍÓÚÑ(]", remainder))

    def _is_book_heading(self, text: str) -> bool:
        return self._is_normative_outline_heading(
            text,
            r"^LIBRO\s*(?:[IVXLC0-9]+|PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|SEPTIMO|S[EÉ]PTIMO|OCTAVO|NOVENO|D[EÉ]CIMO)\b",
        )

    def _is_title_heading(self, text: str) -> bool:
        return self._is_normative_outline_heading(
            text,
            r"^T[IÍ]TULO\s*(?:[IVXLC0-9]+|PRELIMINAR|PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|SEPTIMO|S[EÉ]PTIMO|OCTAVO|NOVENO|D[EÉ]CIMO)\b",
        )

    def _is_chapter_heading(self, text: str) -> bool:
        return self._is_normative_outline_heading(
            text,
            r"^CAP[IÍ]TULO\s*(?:[IVXLC0-9]+|ÚNICO|UNICO|PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|SEPTIMO|S[EÉ]PTIMO|OCTAVO|NOVENO|D[EÉ]CIMO)\b",
        )

    def _is_normative_outline_heading(self, text: str, pattern: str) -> bool:
        candidate = re.sub(r"^#+\s*", "", text.strip())
        match = re.match(pattern, candidate, re.IGNORECASE)
        if not match:
            return False

        remainder = candidate[match.end():].strip()
        if not remainder:
            return True
        if remainder[0] in ".:-":
            return True

        return bool(re.match(r"^[A-ZÁÉÍÓÚÑ(]", remainder))

    def _merge_with_context(
        self,
        local_labels: list[str] | None,
        context: SectionContext,
        page: int,
    ) -> list[str]:
        labels = [label for label in (local_labels or []) if label]
        if not labels:
            labels = [f"Page {page}"]

        book = next((label for label in labels if self._label_kind(label) == "book"), None)
        title = next((label for label in labels if self._label_kind(label) == "title"), None)
        chapter = next((label for label in labels if self._label_kind(label) == "chapter"), None)
        article = next((label for label in labels if self._label_kind(label) == "article"), None)
        item = next((label for label in labels if self._label_kind(label) == "item"), None)
        document = next((label for label in labels if self._label_kind(label) == "document"), None)
        roman = next((label for label in labels if self._label_kind(label) == "roman"), None)
        section = next((label for label in labels if self._label_kind(label) == "section"), None)

        merged: list[str] = []
        if book:
            merged.append(book)
        elif context.book:
            merged.append(context.book)

        if title:
            merged.append(title)
        elif context.title:
            merged.append(context.title)

        if chapter:
            merged.append(chapter)
        elif context.chapter:
            merged.append(context.chapter)

        if article:
            merged.append(article)
        elif context.article and not any((book, title, chapter)):
            merged.append(context.article)
        elif roman and not article:
            merged.append(roman)
        elif section and not article:
            merged.append(section)
        elif item and not chapter and not article:
            if context.article:
                merged.append(context.article)
            elif not merged:
                merged.append(item)

        if not merged:
            if document:
                merged.append(document)
            elif context.document:
                merged.append(context.document)
            else:
                merged.append(labels[0])
        return list(dict.fromkeys(merged[:4]))

    def _advance_context(self, context: SectionContext, labels: list[str]) -> SectionContext:
        next_context = SectionContext(
            document=context.document,
            book=context.book,
            title=context.title,
            chapter=context.chapter,
            article=context.article,
        )
        for label in labels:
            kind = self._label_kind(label)
            if kind == "document":
                next_context.document = label
            elif kind == "book":
                next_context.book = label
                next_context.title = None
                next_context.chapter = None
                next_context.article = None
            elif kind == "title":
                next_context.title = label
                next_context.chapter = None
                next_context.article = None
            elif kind == "chapter":
                next_context.chapter = label
                next_context.article = None
            elif kind == "article":
                next_context.article = label
        return next_context

    def _section_path_rank(self, labels: list[str]) -> tuple[int, int]:
        priorities = [self._label_priority(label) for label in labels]
        return (max(priorities, default=0), sum(priorities))

    def _label_priority(self, label: str) -> int:
        kind = self._label_kind(label)
        priorities = {
            "page": 0,
            "item": 1,
            "document": 2,
            "section": 2,
            "roman": 2,
            "article": 3,
            "chapter": 4,
            "title": 5,
            "book": 6,
        }
        return priorities.get(kind, 1)

    def _label_kind(self, label: str) -> str:
        candidate = label.strip()
        if candidate.startswith("Page "):
            return "page"
        if candidate.startswith("Item "):
            return "item"
        if candidate.upper().startswith("LIBRO "):
            return "book"
        if candidate.upper().startswith("TITULO "):
            return "title"
        if candidate.upper().startswith("CAPITULO "):
            return "chapter"
        if candidate.startswith("Arto. "):
            return "article"
        if re.fullmatch(r"[IVXLCM]+", candidate, re.IGNORECASE):
            return "roman"
        if re.match(
            r"^(SECCI[OÓ]N|SECTION|AP[EÉ]NDICE|APPENDIX|M[OÓ]DULO|MODULO|EJEMPLO|EXAMPLE|ESTUDIO DE CASO|CASE STUDY|PREGUNTA|QUESTION|NOTA|FIGURA|TABLA|RESPUESTAS?|C[ÁA]LCULO|RECONCILIACI[OÓ]N)\b",
            candidate,
            re.IGNORECASE,
        ):
            return "section"
        if re.match(r"^(ACTA|SESI[OÓ]N|REUNI[OÓ]N|AGENDA|DESARROLLO|ACUERDOS?|ANEXOS?|ESTADO DE RESULTADOS|ESTADO DE SITUACION FINANCIERA|GIRO DEL NEGOCIO|DIRECCION DEL NEGOCIO|DOCUMENTOS? DEL CLIENTE)\b", candidate, re.IGNORECASE):
            return "section"
        return "document"

    def _estimate_tokens(self, text: str) -> int:
        units = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        return max(1, ceil(len(units) * 1.1))

    def _non_empty_lines(self, text: str) -> list[str]:
        return [line.strip() for line in text.splitlines() if line.strip()]

    def _normalize_line(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().upper())
