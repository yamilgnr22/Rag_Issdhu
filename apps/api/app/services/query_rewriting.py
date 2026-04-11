from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.services.query_structure import QueryStructure
from app.services.query_routing import QueryRoute

if TYPE_CHECKING:
    from app.services.query_planning import QueryPlan


@dataclass
class QueryRewritePlan:
    search_queries: list[str]
    expansion_queries: list[str]
    applied: list[str]
    decision_trace: list[str]


class QueryRewriteService:
    def build(
        self,
        *,
        question: str,
        route: QueryRoute,
        structure: QueryStructure | None = None,
        plan: QueryPlan | None = None,
    ) -> QueryRewritePlan:
        normalized = self._normalize(question)
        search_queries: list[str] = []
        expansion_queries: list[str] = []
        applied: list[str] = []

        def add_search(query: str, label: str) -> None:
            if query and self._normalize(query) != normalized and query not in search_queries:
                search_queries.append(query)
                applied.append(label)

        def add_expand(query: str, label: str) -> None:
            if query and self._normalize(query) != normalized and query not in expansion_queries:
                expansion_queries.append(query)
                applied.append(label)

        if structure is not None:
            for query in structure.search_queries:
                add_search(query, "structure:query")
            for query in structure.expansion_queries:
                add_expand(query, "structure:decompose")

        if plan is not None:
            for query in plan.decomposition_queries:
                add_expand(query, "planner:decompose")
            if structure is not None and plan.topics:
                marker_text = " ".join(structure.document_markers[:2]).strip()
                target_text = " ".join([*plan.target_units[:2], *plan.target_numbers[:2]]).strip()
                topic_text = " ".join(plan.topics[:5]).strip()
                planner_focus = " ".join(part for part in (marker_text, target_text, topic_text) if part).strip()
                add_search(planner_focus, "planner:focus")

        for clause in self._extract_compare_subqueries(normalized, route):
            add_expand(clause, "comparison:split")

        return QueryRewritePlan(
            search_queries=search_queries,
            expansion_queries=expansion_queries,
            applied=applied,
            decision_trace=[
                f"rewrite:queries={len(search_queries)}",
                f"rewrite:expansions={len(expansion_queries)}",
            ],
        )

    def _extract_compare_subqueries(self, normalized_question: str, route: QueryRoute) -> list[str]:
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

    def _normalize(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text or "")
        return " ".join(
            "".join(char for char in normalized if not unicodedata.combining(char)).lower().split()
        )
