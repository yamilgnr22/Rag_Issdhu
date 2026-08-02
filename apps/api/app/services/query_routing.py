from dataclasses import dataclass
import re
import unicodedata

from app.models.contracts import DocumentClass
from app.services.query_router_classifier import QueryRouterClassifier


@dataclass
class QueryRoute:
    query_type: str
    target_classes: list[DocumentClass]
    decision_trace: list[str]
    confidence: float = 1.0
    # Cuando es False, target_classes se usa como preferencia y no como filtro
    # excluyente: el router acierta 27/30, pero su confianza no separa aciertos de
    # errores, y un error excluia por completo el documento correcto.
    hard_class_filter: bool = True


class QueryRoutingService:
    def __init__(self, settings=None) -> None:
        self.settings = settings
        self.classifier = QueryRouterClassifier(settings) if settings is not None else None

    def route(self, *, question: str, filters: dict[str, object]) -> QueryRoute:
        explicit_class = self._coerce_explicit_class(filters.get("document_class"))
        if explicit_class is not None:
            return QueryRoute(
                query_type=explicit_class,
                target_classes=[explicit_class],
                decision_trace=[f"router:explicit_document_class:{explicit_class}"],
            )

        explicit_family = self._coerce_explicit_family(filters.get("document_family"))
        if explicit_family == "general":
            return QueryRoute(
                query_type="general_document",
                target_classes=["general_document"],
                decision_trace=["router:explicit_document_family:general"],
            )
        if explicit_family == "normative":
            return QueryRoute(
                query_type="normative",
                target_classes=["legal_normative", "technical_standard"],
                decision_trace=["router:explicit_document_family:normative"],
            )

        heuristic_route = self._heuristic_route(question)
        if heuristic_route is not None:
            return heuristic_route

        if self.classifier is not None:
            classification = self.classifier.classify(question=question)
            if classification is not None:
                threshold = float(
                    getattr(self.settings, "query_router_hard_filter_confidence", 1.0)
                    if self.settings is not None
                    else 1.0
                )
                hard = classification.confidence >= threshold
                trace = list(classification.decision_trace)
                trace.append(f"router:class_filter={'hard' if hard else 'soft'}")
                return QueryRoute(
                    query_type=classification.query_type,
                    target_classes=classification.target_classes,
                    decision_trace=trace,
                    confidence=classification.confidence,
                    hard_class_filter=hard,
                )

        return QueryRoute(
            query_type="all_domains",
            target_classes=["legal_normative", "technical_standard", "general_document"],
            decision_trace=["router:provider=all_domains_fallback", "router:all_domains"],
        )

    def _coerce_explicit_class(self, value: object) -> DocumentClass | None:
        if value in {"legal_normative", "technical_standard", "general_document"}:
            return value
        return None

    def _coerce_explicit_family(self, value: object) -> str | None:
        if value in {"normative", "general"}:
            return str(value)
        return None

    def _heuristic_route(self, question: str) -> QueryRoute | None:
        normalized = self._normalize(question)
        if re.search(r"\b(?:art\.?|arto\.?|articulo)\s*\d+(?:\.\d+)?\b", normalized):
            return QueryRoute(
                query_type="legal_normative",
                target_classes=["legal_normative"],
                decision_trace=["router:heuristic=article_reference", "router:legal_normative"],
            )
        if "codigo del trabajo" in normalized or re.search(r"\bley\s+\d+\b", normalized):
            return QueryRoute(
                query_type="legal_normative",
                target_classes=["legal_normative"],
                decision_trace=["router:heuristic=legal_marker", "router:legal_normative"],
            )
        if any(marker in normalized for marker in ("niif", "gafi", "metodologia", "recomendacion")):
            return QueryRoute(
                query_type="technical_standard",
                target_classes=["technical_standard"],
                decision_trace=["router:heuristic=technical_marker", "router:technical_standard"],
            )
        return None

    def _normalize(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text or "")
        return "".join(char for char in normalized if not unicodedata.combining(char)).lower()
