import io
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path

import openpyxl
import pdfplumber
from bs4 import BeautifulSoup
from docx import Document as DocxDocument

from app.config import Settings


@dataclass
class ExtractionBlock:
    block_type: str
    text: str
    page: int
    section_path: list[str]


@dataclass
class ExtractionResult:
    blocks: list[ExtractionBlock]
    backend: str
    review_required: bool
    review_reason: str | None = None

    @property
    def text_length(self) -> int:
        return sum(len(block.text) for block in self.blocks)


class DoclingExtractor:
    def extract(self, filename: str, content: bytes) -> ExtractionResult | None:
        try:
            from docling.document_converter import DocumentConverter
        except Exception:
            return None

        suffix = Path(filename).suffix.lower()
        if suffix not in {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".png", ".jpg", ".jpeg"}:
            return None

        temp_dir = Path(".data") / "docling_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"docling_{Path(filename).name}"
        temp_path.write_bytes(content)

        try:
            converter = DocumentConverter()
            result = converter.convert(str(temp_path))
            document = result.document
            markdown = document.export_to_markdown().strip()
            if not markdown:
                return None
            review_required = len(markdown) < 50
            return ExtractionResult(
                blocks=[ExtractionBlock("docling_markdown", markdown, 1, ["Document"])],
                backend="docling",
                review_required=review_required,
                review_reason="docling_extracted_too_little_text" if review_required else None,
            )
        except Exception:
            return None
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass


class FallbackExtractor:
    MIN_TEXT_LENGTH = 50

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def extract(self, filename: str, content: bytes, mime_type: str | None = None) -> ExtractionResult:
        suffix = Path(filename).suffix.lower()
        mime_type = mime_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"

        if suffix in {".txt", ".md", ".json"} or mime_type.startswith("text/"):
            text = self._decode_text(content)
            return self._finalize([ExtractionBlock("text", text, 1, ["Document"])], "plain_text")

        if suffix == ".pdf" or mime_type == "application/pdf":
            return self._extract_pdf(content)

        if suffix == ".docx":
            return self._extract_docx(content)

        if suffix in {".html", ".htm"} or mime_type == "text/html":
            return self._extract_html(content)

        if suffix == ".xlsx":
            return self._extract_xlsx(content)

        return ExtractionResult([], "unsupported", True, f"unsupported_format:{suffix or mime_type}")

    def _decode_text(self, content: bytes) -> str:
        for encoding in ("utf-8", "cp1252", "latin-1"):
            try:
                return content.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
        return ""

    def _extract_pdf(self, content: bytes) -> ExtractionResult:
        blocks: list[ExtractionBlock] = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    blocks.append(
                        ExtractionBlock(
                            "page",
                            re.sub(r"\s+\n", "\n", text),
                            page_number,
                            [f"Page {page_number}"],
                        )
                    )
        if not blocks:
            return ExtractionResult([], "pdfplumber", True, "pdf_no_text_detected_manual_review")
        return self._finalize(blocks, "pdfplumber")

    def _extract_docx(self, content: bytes) -> ExtractionResult:
        doc = DocxDocument(io.BytesIO(content))
        blocks: list[ExtractionBlock] = []
        current_heading = "Document"
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            if para.style and para.style.name.lower().startswith("heading"):
                current_heading = text
                blocks.append(ExtractionBlock("heading", text, 1, [current_heading]))
            else:
                blocks.append(ExtractionBlock("paragraph", text, 1, [current_heading]))
        for table_index, table in enumerate(doc.tables, start=1):
            rows = []
            for row in table.rows:
                values = [cell.text.strip() for cell in row.cells]
                if any(values):
                    rows.append("\t".join(values))
            if rows:
                blocks.append(
                    ExtractionBlock("table", "\n".join(rows), 1, [current_heading, f"Table {table_index}"])
                )
        return self._finalize(blocks, "python-docx")

    def _extract_html(self, content: bytes) -> ExtractionResult:
        soup = BeautifulSoup(content, "html.parser")
        blocks: list[ExtractionBlock] = []
        current_heading = "Document"
        for tag in soup.find_all(["h1", "h2", "h3", "p", "li", "table"]):
            text = tag.get_text(" ", strip=True)
            if not text:
                continue
            if tag.name in {"h1", "h2", "h3"}:
                current_heading = text
                blocks.append(ExtractionBlock("heading", text, 1, [current_heading]))
            elif tag.name == "table":
                blocks.append(ExtractionBlock("table", text, 1, [current_heading, "Table"]))
            else:
                block_type = "paragraph" if tag.name == "p" else "list_item"
                blocks.append(ExtractionBlock(block_type, text, 1, [current_heading]))
        return self._finalize(blocks, "beautifulsoup4")

    def _extract_xlsx(self, content: bytes) -> ExtractionResult:
        workbook = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        blocks: list[ExtractionBlock] = []
        for sheet in workbook.worksheets:
            rows = []
            for row in sheet.iter_rows(values_only=True):
                values = [str(value).strip() for value in row if value is not None and str(value).strip()]
                if values:
                    rows.append("\t".join(values))
            if rows:
                blocks.append(ExtractionBlock("table", "\n".join(rows), 1, [sheet.title]))
        return self._finalize(blocks, "openpyxl")

    def _finalize(self, blocks: list[ExtractionBlock], backend: str) -> ExtractionResult:
        cleaned = [block for block in blocks if block.text.strip()]
        text_length = sum(len(block.text) for block in cleaned)
        review_required = text_length < self.MIN_TEXT_LENGTH
        return ExtractionResult(
            blocks=cleaned,
            backend=backend,
            review_required=review_required,
            review_reason="extracted_text_below_threshold" if review_required else None,
        )


class ExtractionPipeline:
    def __init__(self, settings: Settings) -> None:
        self.docling = DoclingExtractor()
        self.fallback = FallbackExtractor(settings)

    def extract(self, filename: str, content: bytes) -> ExtractionResult:
        docling_result = self.docling.extract(filename, content)
        if docling_result and docling_result.blocks:
            return docling_result
        return self.fallback.extract(filename, content)

