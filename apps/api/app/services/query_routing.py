from dataclasses import dataclass

from app.models.contracts import DocumentClass
from app.services.query_router_classifier import QueryRouterClassifier


@dataclass
class QueryRoute:
    query_type: str
    target_classes: list[DocumentClass]
    decision_trace: list[str]


class QueryRoutingService:
    def __init__(self, settings=None) -> None:
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

        if self.classifier is not None:
            classification = self.classifier.classify(question=question)
            if classification is not None:
                return QueryRoute(
                    query_type=classification.query_type,
                    target_classes=classification.target_classes,
                    decision_trace=list(classification.decision_trace),
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
