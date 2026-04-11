import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx


VALID_CLASSES = {"legal_normative", "technical_standard", "general_document"}


@dataclass
class DocumentClassification:
    document_class: str
    document_family: str
    chunking_strategy: str
    classification_source: str
    confidence: float
    reason_short: str


@dataclass
class LLMProviderConfig:
    provider_name: str
    base_url: str
    model: str
    api_key: str | None
    timeout: float


class DocumentClassifier:
    def __init__(self, settings) -> None:
        self.settings = settings

    def classify(self, *, filename: str, mime_type: str, blocks: Iterable[object]) -> DocumentClassification:
        sample_text, heading_count = self._build_sample(blocks)
        deterministic = self._deterministic_classification(
            filename=filename,
            mime_type=mime_type,
            sample_text=sample_text,
            heading_count=heading_count,
        )
        safe_strategy = self._default_general_strategy(sample_text, heading_count)
        mode = self.settings.classifier_mode

        if mode == "heuristic":
            if deterministic is not None:
                return deterministic
            return self._safe_fallback(
                safe_strategy=safe_strategy,
                reason_short="Heuristic mode could not classify confidently.",
            )

        if mode == "local_llm":
            local = self._classify_with_provider(
                provider=self._provider_config("local"),
                filename=filename,
                mime_type=mime_type,
                sample_text=sample_text,
                heading_count=heading_count,
            )
            if local is not None:
                return local
            if deterministic is not None:
                return deterministic
            return self._safe_fallback(
                safe_strategy=safe_strategy,
                reason_short="Local classifier unavailable or low confidence.",
            )

        if mode == "cloud_llm":
            cloud = self._classify_with_provider(
                provider=self._provider_config("cloud"),
                filename=filename,
                mime_type=mime_type,
                sample_text=sample_text,
                heading_count=heading_count,
            )
            if cloud is not None:
                return cloud
            if deterministic is not None:
                return deterministic
            return self._safe_fallback(
                safe_strategy=safe_strategy,
                reason_short="Cloud classifier unavailable or low confidence.",
            )

        if deterministic is not None and deterministic.confidence >= self.settings.classifier_deterministic_short_circuit:
            return deterministic

        for provider_name in ("local", "cloud"):
            provider_result = self._classify_with_provider(
                provider=self._provider_config(provider_name),
                filename=filename,
                mime_type=mime_type,
                sample_text=sample_text,
                heading_count=heading_count,
            )
            if provider_result is not None:
                return provider_result

        if deterministic is not None:
            return deterministic

        return self._safe_fallback(
            safe_strategy=safe_strategy,
            reason_short="Hybrid classifier fell back to safe general mode.",
        )

    def _build_sample(self, blocks: Iterable[object]) -> tuple[str, int]:
        text_parts: list[str] = []
        heading_count = 0
        for block in blocks:
            text = getattr(block, "text", "")
            if text:
                text_parts.append(text)
            if getattr(block, "block_type", "") in {"heading", "table"}:
                heading_count += 1
        return "\n".join(text_parts[:8])[:12000], heading_count

    def _deterministic_classification(
        self,
        *,
        filename: str,
        mime_type: str,
        sample_text: str,
        heading_count: int,
    ) -> DocumentClassification | None:
        haystack = f"{Path(filename).name}\n{sample_text}".lower()

        meeting_primary_patterns = [
            r"\bacta\b",
            r"\bsesi[oó]n\b",
            r"\breuni[oó]n\b",
            r"\bpuntos?\s+de\s+agenda\b",
            r"\borden\s+del\s+d[ií]a\b",
            r"\blectura\s+de\s+acuerdos?\s+anteriores\b",
        ]
        meeting_secondary_patterns = [
            r"\bconsejo\s+directivo\b",
            r"\breuni[oó]n\s+ordinaria\b",
            r"\breuni[oó]n\s+extraordinaria\b",
        ]
        meeting_primary_hits = self._keyword_hits(haystack, meeting_primary_patterns)
        meeting_secondary_hits = self._keyword_hits(haystack, meeting_secondary_patterns)
        if meeting_primary_hits >= 2 or (meeting_primary_hits >= 1 and meeting_secondary_hits >= 1):
            return self._build_result(
                document_class="general_document",
                classification_source="deterministic",
                confidence=0.94,
                reason_short="Meeting/minutes markers override incidental normative references.",
                sample_text=sample_text,
                heading_count=heading_count,
                forced_strategy="generic_structural",
            )

        technical_patterns = [
            r"\bniif\b",
            r"\bifrs\b",
            r"\bnic\b",
            r"\bnorma\s+de\s+contabilidad\b",
            r"\bm[oó]dulo\s+educativo\b",
            r"\bestudio\s+de\s+caso\b",
            r"\bpreguntas?\s+de\s+autoevaluaci[oó]n\b",
            r"\bconsejo\s+de\s+normas\s+internacionales\s+de\s+contabilidad\b",
            r"\bfundaci[oó]n\s+ifrs\b",
            r"\bsecci[oó]n\s+\d+\b",
        ]
        technical_hits = self._keyword_hits(haystack, technical_patterns)
        if technical_hits >= 3:
            return self._build_result(
                document_class="technical_standard",
                classification_source="deterministic",
                confidence=min(0.97, 0.68 + (technical_hits * 0.04)),
                reason_short=f"Technical standard markers favored technical_standard (hits={technical_hits}).",
                sample_text=sample_text,
                heading_count=heading_count,
            )

        legal_score = 0
        general_structured_score = 0
        general_freeform_score = 0

        legal_keywords = [
            "ley",
            "código",
            "codigo",
            "reglamento",
            "normativa",
            "resolución",
            "resolucion",
            "decreto",
            "arto.",
            "artículo",
            "articulo",
            "capítulo",
            "capitulo",
            "título",
            "titulo",
            "libro",
        ]
        structured_keywords = [
            "manual",
            "procedimiento",
            "instructivo",
            "política",
            "politica",
            "guía",
            "guia",
            "objetivo",
            "alcance",
            "responsabilidades",
            "anexo",
            "agenda",
        ]

        legal_score += sum(2 for keyword in legal_keywords if keyword in haystack)
        general_structured_score += sum(2 for keyword in structured_keywords if keyword in haystack)

        article_hits = len(re.findall(r"\bart(?:[íi]culo|o)?\.?\s+\d+", haystack))
        chapter_hits = len(re.findall(r"\bcap[íi]tulo\s+[ivxlc0-9]+", haystack))
        title_hits = len(re.findall(r"\bt[íi]tulo\s+[ivxlc0-9]+|\bt[íi]tulo\s+preliminar", haystack))
        book_hits = len(re.findall(r"\blibro\s+[ivxlc0-9]+|\blibro\s+(?:primero|segundo|tercero|cuarto)", haystack))
        item_hits = len(re.findall(r"(?m)^\d+\)", sample_text))
        uppercase_headings = len(
            [
                line
                for line in sample_text.splitlines()[:40]
                if line.strip() and line.strip() == line.strip().upper() and len(line.strip()) > 6
            ]
        )

        legal_score += min(8, article_hits * 2)
        legal_score += min(4, chapter_hits * 2)
        legal_score += min(3, title_hits)
        legal_score += min(2, book_hits)
        general_structured_score += min(4, heading_count) + min(3, uppercase_headings)
        if mime_type.startswith("text/") and heading_count == 0:
            general_freeform_score += 2
        if not article_hits and not chapter_hits and heading_count == 0:
            general_freeform_score += 3
        if item_hits >= 4 and article_hits == 0:
            general_structured_score += 2

        scores = {
            "legal_normative": legal_score,
            "general_structured": general_structured_score,
            "general_freeform": general_freeform_score,
        }
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_label, top_score = ranked[0]
        second_score = ranked[1][1]

        if top_score < 3:
            return None
        if top_score - second_score < 2 and top_score < 7:
            return None

        confidence = min(0.97, 0.55 + ((top_score - second_score) * 0.08) + (top_score * 0.02))
        if top_label == "legal_normative":
            return self._build_result(
                document_class="legal_normative",
                classification_source="deterministic",
                confidence=round(confidence, 2),
                reason_short=f"Deterministic markers favored legal_normative (score {top_score} vs {second_score}).",
                sample_text=sample_text,
                heading_count=heading_count,
            )

        forced_strategy = "generic_structural" if top_label == "general_structured" else "generic_freeform"
        return self._build_result(
            document_class="general_document",
            classification_source="deterministic",
            confidence=round(confidence, 2),
            reason_short=f"Deterministic markers favored {top_label} (score {top_score} vs {second_score}).",
            sample_text=sample_text,
            heading_count=heading_count,
            forced_strategy=forced_strategy,
        )

    def _provider_config(self, provider_name: str) -> LLMProviderConfig | None:
        if provider_name == "cloud":
            base_url = self.settings.classifier_cloud_base_url or self.settings.classifier_llm_base_url
            api_key = self.settings.classifier_cloud_api_key or self.settings.classifier_llm_api_key
            model = self.settings.classifier_cloud_model or self.settings.classifier_llm_model
            timeout = self.settings.classifier_cloud_timeout or self.settings.classifier_llm_timeout
        else:
            base_url = self.settings.classifier_local_base_url
            api_key = self.settings.classifier_local_api_key
            model = self.settings.classifier_local_model
            timeout = self.settings.classifier_local_timeout or self.settings.classifier_llm_timeout

        if not base_url or not model:
            return None
        return LLMProviderConfig(
            provider_name=provider_name,
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
        )

    def _classify_with_provider(
        self,
        *,
        provider: LLMProviderConfig | None,
        filename: str,
        mime_type: str,
        sample_text: str,
        heading_count: int,
    ) -> DocumentClassification | None:
        if provider is None:
            return None

        prompt = (
            "Clasifica el documento en una sola clase operativa: legal_normative, "
            "technical_standard o general_document. Responde solo JSON con keys "
            "document_class, confidence, reason_short.\n\n"
            "Definiciones:\n"
            "- legal_normative: leyes, códigos, reglamentos, decretos, resoluciones y normas jurídicas con artículos/capítulos/libros.\n"
            "- technical_standard: estándares técnicos oficiales o cuasi oficiales estructurados por secciones, ejemplos, apéndices o requisitos técnicos, por ejemplo NIIF/NIC/IFRS.\n"
            "- general_document: actas, minutas, certificaciones, manuales, reportes, guías y documentos generales no normativos.\n\n"
            f"filename: {filename}\n"
            f"mime_type: {mime_type}\n"
            f"heading_count: {heading_count}\n"
            f"sample_text:\n{sample_text[:6000]}"
        )

        headers = {"Content-Type": "application/json"}
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"

        try:
            with httpx.Client(timeout=provider.timeout) as client:
                response = client.post(
                    f"{provider.base_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    json={
                        "model": provider.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You classify documents for chunking strategy selection. "
                                    "Return strict JSON only."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return None

        try:
            content = payload["choices"][0]["message"]["content"]
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            data = json.loads(match.group(0) if match else content)
            document_class = data["document_class"]
            confidence = float(data.get("confidence", 0.0))
            if document_class not in VALID_CLASSES:
                return None
            if confidence < self.settings.classifier_llm_min_confidence:
                return None
            return self._build_result(
                document_class=document_class,
                classification_source=f"llm_{provider.provider_name}",
                confidence=round(confidence, 2),
                reason_short=str(data.get("reason_short", f"{provider.provider_name} LLM classification."))[:180],
                sample_text=sample_text,
                heading_count=heading_count,
            )
        except Exception:
            return None

    def _safe_fallback(self, *, safe_strategy: str, reason_short: str) -> DocumentClassification:
        return self._build_result(
            document_class="general_document",
            classification_source="safe_fallback",
            confidence=0.45,
            reason_short=reason_short[:180],
            sample_text="",
            heading_count=0,
            forced_strategy=safe_strategy,
        )

    def _build_result(
        self,
        *,
        document_class: str,
        classification_source: str,
        confidence: float,
        reason_short: str,
        sample_text: str,
        heading_count: int,
        forced_strategy: str | None = None,
    ) -> DocumentClassification:
        family_map = {
            "legal_normative": "normative",
            "technical_standard": "normative",
            "general_document": "general",
        }
        if document_class == "legal_normative":
            strategy = "normative_hierarchical"
        elif document_class == "technical_standard":
            strategy = "technical_structural"
        else:
            strategy = forced_strategy or self._default_general_strategy(sample_text, heading_count)

        return DocumentClassification(
            document_class=document_class,
            document_family=family_map[document_class],
            chunking_strategy=strategy,
            classification_source=classification_source,
            confidence=confidence,
            reason_short=reason_short,
        )

    def _default_general_strategy(self, sample_text: str, heading_count: int) -> str:
        haystack = sample_text.lower()
        structured_markers = [
            r"\bacta\b",
            r"\bsesi[oó]n\b",
            r"\breuni[oó]n\b",
            r"\bmanual\b",
            r"\bprocedimiento\b",
            r"\bobjetivo\b",
            r"\balcance\b",
            r"\bresponsabilidades\b",
            r"\banexo\b",
            r"\bagenda\b",
            r"\bestado de resultados\b",
            r"\bestado de situaci[oó]n financiera\b",
        ]
        uppercase_headings = len(
            [
                line
                for line in sample_text.splitlines()[:40]
                if line.strip() and line.strip() == line.strip().upper() and len(line.strip()) > 6
            ]
        )
        if heading_count > 0 or uppercase_headings >= 2 or self._keyword_hits(haystack, structured_markers) >= 2:
            return "generic_structural"
        return "generic_freeform"

    def _keyword_hits(self, haystack: str, patterns: list[str]) -> int:
        return sum(1 for pattern in patterns if re.search(pattern, haystack, re.IGNORECASE))
