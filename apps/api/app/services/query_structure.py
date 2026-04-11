import re
import unicodedata
from dataclasses import dataclass

from app.services.query_routing import QueryRoute


STOPWORDS = {
    "a",
    "al",
    "ante",
    "como",
    "con",
    "cual",
    "cuales",
    "cuando",
    "de",
    "del",
    "donde",
    "el",
    "ella",
    "ellas",
    "ellos",
    "en",
    "entre",
    "es",
    "esa",
    "ese",
    "esta",
    "este",
    "estas",
    "estos",
    "la",
    "las",
    "lo",
    "los",
    "mas",
    "menos",
    "para",
    "pero",
    "por",
    "que",
    "respecto",
    "segun",
    "si",
    "sin",
    "sobre",
    "solo",
    "su",
    "sus",
    "un",
    "una",
    "uno",
    "unas",
    "unos",
    "y",
}
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
REFERENCE_PATTERNS = (
    ("article", r"\b(?:art\.?|arto\.?|articulo)\s*(\d+(?:\.\d+)?)\b"),
    ("section", r"\bseccion\s*(\d+(?:\.\d+)?)\b"),
    ("recommendation", r"\brecomendacion\s*(\d+(?:\.\d+)?)\b"),
    ("recommendation", r"\br\.?\s*(\d+(?:\.\d+)?)\b"),
    ("annex", r"\banexo\s+([ivxlcdm]+|\d+)\b"),
    ("chapter", r"\bcapitulo\s+([ivxlcdm]+|\d+)\b"),
    ("title", r"\btitulo\s+([ivxlcdm]+|\d+)\b"),
    ("book", r"\blibro\s+([ivxlcdm]+|\d+)\b"),
    ("paragraph", r"\bparr?afo\s*(\d+(?:\.\d+)?)\b"),
    ("law", r"\bley\s*(\d+)\b"),
)


@dataclass
class QueryReference:
    kind: str
    value: str
    label: str


@dataclass
class QueryStructure:
    normalized_question: str
    references: list[QueryReference]
    document_markers: list[str]
    topic_terms: list[str]
    comparison_clauses: list[str]
    search_queries: list[str]
    expansion_queries: list[str]
    decision_trace: list[str]


class QueryStructureService:
    def parse(self, *, question: str, route: QueryRoute) -> QueryStructure:
        normalized = self._normalize(question)
        references = self._extract_references(normalized)
        document_markers = self._extract_document_markers(normalized, references)
        topic_terms = self._extract_topic_terms(normalized, references, document_markers)
        comparison_clauses = self._extract_compare_subqueries(normalized, route)
        search_queries = self._build_search_queries(
            references=references,
            document_markers=document_markers,
            topic_terms=topic_terms,
        )
        expansion_queries = list(comparison_clauses)
        decision_trace = [
            f"structure:refs={len(references)}",
            f"structure:markers={len(document_markers)}",
            f"structure:topics={len(topic_terms)}",
            f"structure:clauses={len(comparison_clauses)}",
        ]
        return QueryStructure(
            normalized_question=normalized,
            references=references,
            document_markers=document_markers,
            topic_terms=topic_terms,
            comparison_clauses=comparison_clauses,
            search_queries=search_queries,
            expansion_queries=expansion_queries,
            decision_trace=decision_trace,
        )

    def _build_search_queries(
        self,
        *,
        references: list[QueryReference],
        document_markers: list[str],
        topic_terms: list[str],
    ) -> list[str]:
        queries: list[str] = []
        marker_text = " ".join(document_markers[:2]).strip()
        topic_text = " ".join(topic_terms[:5]).strip()

        def add(query: str) -> None:
            cleaned = " ".join((query or "").split()).strip()
            if cleaned and cleaned not in queries:
                queries.append(cleaned)

        if marker_text and topic_text:
            add(f"{marker_text} {topic_text}")

        for reference in references:
            label = reference.label
            if marker_text and topic_text:
                add(f"{marker_text} {label} {topic_text}")
            if marker_text:
                add(f"{marker_text} {label}")
            if topic_text:
                add(f"{label} {topic_text}")
            add(label)

        if marker_text:
            for reference in references[:2]:
                if reference.kind in {"article", "section", "recommendation", "annex", "paragraph"}:
                    add(f"{reference.kind} {reference.value} {marker_text}")

        return queries[:8]

    def _extract_references(self, normalized_question: str) -> list[QueryReference]:
        references: list[QueryReference] = []
        seen: set[tuple[str, str]] = set()
        for kind, pattern in REFERENCE_PATTERNS:
            for match in re.finditer(pattern, normalized_question):
                value = match.group(1).upper()
                key = (kind, value)
                if key in seen:
                    continue
                seen.add(key)
                references.append(
                    QueryReference(
                        kind=kind,
                        value=value,
                        label=self._reference_label(kind, value),
                    )
                )
        return references

    def _extract_document_markers(
        self,
        normalized_question: str,
        references: list[QueryReference],
    ) -> list[str]:
        markers: list[str] = []

        def add(marker: str) -> None:
            cleaned = " ".join((marker or "").split()).strip()
            if cleaned and cleaned not in markers:
                markers.append(cleaned)

        if "niif para las pymes" in normalized_question:
            add("niif para las pymes")
        elif "niif" in normalized_question and "pymes" in normalized_question:
            add("niif para las pymes")
        elif "niif" in normalized_question:
            add("niif")

        if "gafi" in normalized_question:
            add("gafi")
        if "metodologia" in normalized_question:
            add("metodologia")
        if "codigo del trabajo" in normalized_question:
            add("codigo del trabajo")
        elif "codigo" in normalized_question and "trabajo" in normalized_question:
            add("codigo del trabajo")
        if "nic 39" in normalized_question:
            add("nic 39")
        if "apnfd" in normalized_question:
            add("apnfd")

        for reference in references:
            if reference.kind == "law":
                add(reference.label.lower())

        return markers[:4]

    def _extract_topic_terms(
        self,
        normalized_question: str,
        references: list[QueryReference],
        document_markers: list[str],
    ) -> list[str]:
        excluded = {
            "art",
            "arto",
            "articulo",
            "seccion",
            "recomendacion",
            "anexo",
            "capitulo",
            "titulo",
            "libro",
            "parrafo",
            "ley",
            "codigo",
            "niif",
            "nic",
            "gafi",
            "metodologia",
        }
        excluded.update(token for marker in document_markers for token in TOKEN_PATTERN.findall(marker))
        excluded.update(token for ref in references for token in TOKEN_PATTERN.findall(ref.label.lower()))

        terms: list[str] = []
        for token in TOKEN_PATTERN.findall(normalized_question):
            if token in STOPWORDS or token in excluded or token.isdigit() or len(token) < 3:
                continue
            if token not in terms:
                terms.append(token)
        return terms[:8]

    def _extract_compare_subqueries(
        self,
        normalized_question: str,
        route: QueryRoute,
    ) -> list[str]:
        if not self._is_comparison_query(normalized_question):
            return []
        if "technical_standard" not in route.target_classes and route.query_type != "technical_standard":
            return []

        match = re.match(r"^(que\s+(?:documento|norma|parte))\s+(.+?)\s+y\s+cual\s+(.+)$", normalized_question)
        if match:
            subject = match.group(1)
            first = f"{subject} {match.group(2).strip(' ?')}"
            second = f"{subject} {match.group(3).strip(' ?')}"
            return [first, second]

        match = re.match(r"^(que\s+diferencia\s+hay\s+entre)\s+(.+?)\s+y\s+(.+)$", normalized_question)
        if match:
            return [match.group(2).strip(" ?"), match.group(3).strip(" ?")]

        return []

    def _is_comparison_query(self, normalized_question: str) -> bool:
        return bool(
            " y cual " in normalized_question
            or normalized_question.startswith("que documento")
            or normalized_question.startswith("que norma")
            or " diferencia " in normalized_question
        )

    def _reference_label(self, kind: str, value: str) -> str:
        if kind == "article":
            return f"Arto. {value}"
        if kind == "section":
            return f"Seccion {value}"
        if kind == "recommendation":
            return f"Recomendacion {value}"
        if kind == "annex":
            return f"Anexo {value}"
        if kind == "chapter":
            return f"Capitulo {value}"
        if kind == "title":
            return f"Titulo {value}"
        if kind == "book":
            return f"Libro {value}"
        if kind == "paragraph":
            return f"Parrafo {value}"
        if kind == "law":
            return f"Ley {value}"
        return value

    def _normalize(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text or "")
        return " ".join(
            "".join(char for char in normalized if not unicodedata.combining(char)).lower().split()
        )
