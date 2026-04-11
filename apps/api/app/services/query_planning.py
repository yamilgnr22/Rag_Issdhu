import json
from dataclasses import dataclass

import httpx
import re

from app.services.query_strategy_classifier import (
    QueryStrategyClassifier,
    StrategyClassification,
)
from app.services.query_routing import QueryRoute
from app.services.query_structure import QueryStructure


VALID_STRATEGIES = {"focal", "multi_branch", "global"}
VALID_ANSWER_STYLES = {"single_branch", "grouped_by_branch", "summary"}
VALID_UNIT_TYPES = {
    "article",
    "section",
    "recommendation",
    "annex",
    "chapter",
    "title",
    "book",
    "paragraph",
    "branch",
}


@dataclass
class QueryPlannerProviderConfig:
    provider_name: str
    base_url: str
    model: str
    api_key: str | None
    timeout: float


@dataclass
class QueryPlan:
    strategy: str
    answer_style: str
    same_branch_preferred: bool
    diversity_required: bool
    target_units: list[str]
    target_numbers: list[str]
    topics: list[str]
    decomposition_queries: list[str]
    confidence: float
    source: str
    reason_short: str
    decision_trace: list[str]


class QueryPlanningService:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.strategy_classifier = QueryStrategyClassifier(settings)

    def plan(self, *, question: str, route: QueryRoute, structure: QueryStructure) -> QueryPlan:
        prototype_plan = self._plan_with_strategy_classifier(
            question=question,
            route=route,
            structure=structure,
        )
        if prototype_plan is not None:
            return prototype_plan

        mode = self.settings.query_planner_mode
        provider_order = {
            "local_llm": ["local"],
            "cloud_llm": ["cloud"],
            "hybrid": ["local", "cloud"],
        }[mode]

        for provider_name in provider_order:
            plan = self._plan_with_provider(
                provider=self._provider_config(provider_name),
                question=question,
                route=route,
                structure=structure,
            )
            if plan is not None:
                return plan

        return self._fallback_plan(structure=structure)

    def _plan_with_strategy_classifier(
        self,
        *,
        question: str,
        route: QueryRoute,
        structure: QueryStructure,
    ) -> QueryPlan | None:
        classification = self.strategy_classifier.classify(
            question=question,
            route=route,
            structure=structure,
        )
        if classification is None:
            return None
        return self._build_plan_from_classification(
            classification=classification,
            structure=structure,
        )

    def _provider_config(self, provider_name: str) -> QueryPlannerProviderConfig | None:
        if provider_name == "cloud":
            base_url = (
                self.settings.query_planner_cloud_base_url
                or self.settings.classifier_cloud_base_url
                or self.settings.classifier_llm_base_url
            )
            api_key = self.settings.query_planner_cloud_api_key or self.settings.classifier_cloud_api_key
            model = (
                self.settings.query_planner_cloud_model
                or self.settings.classifier_cloud_model
                or self.settings.classifier_llm_model
            )
            timeout = self.settings.query_planner_cloud_timeout or self.settings.classifier_cloud_timeout
        else:
            base_url = self.settings.query_planner_local_base_url or self.settings.classifier_local_base_url
            api_key = self.settings.query_planner_local_api_key or self.settings.classifier_local_api_key
            model = self.settings.query_planner_local_model or self.settings.classifier_local_model
            timeout = self.settings.query_planner_local_timeout or self.settings.classifier_local_timeout

        if not base_url or not model:
            return None
        return QueryPlannerProviderConfig(
            provider_name=provider_name,
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout or 20.0,
        )

    def _plan_with_provider(
        self,
        *,
        provider: QueryPlannerProviderConfig | None,
        question: str,
        route: QueryRoute,
        structure: QueryStructure,
    ) -> QueryPlan | None:
        if provider is None:
            return None

        prompt = self._planning_prompt(question=question, route=route, structure=structure)
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
                                    "You are a retrieval query planner for a structured RAG system. "
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
            strategy = str(data["strategy"]).strip().lower()
            answer_style = str(data.get("answer_style", "")).strip().lower()
            confidence = float(data.get("confidence", 0.0))
            if strategy not in VALID_STRATEGIES:
                return None
            if answer_style not in VALID_ANSWER_STYLES:
                answer_style = self._default_answer_style(strategy)
            if confidence < self.settings.query_planner_min_confidence:
                return None

            target_units = self._sanitize_string_list(data.get("target_units"), allowed=VALID_UNIT_TYPES)
            target_numbers = self._sanitize_string_list(data.get("target_numbers"))
            topics = self._sanitize_string_list(data.get("topics"))
            decomposition_queries = self._sanitize_string_list(data.get("decomposition_queries"), limit=3)

            same_branch_preferred = bool(data.get("same_branch_preferred", strategy == "focal"))
            diversity_required = bool(data.get("diversity_required", strategy != "focal"))
            if strategy == "focal":
                diversity_required = False
                same_branch_preferred = True
            elif strategy in {"multi_branch", "global"}:
                diversity_required = True
                same_branch_preferred = False

            reason_short = str(data.get("reason_short", f"Planner selected {strategy}.")).strip()[:180]
            return QueryPlan(
                strategy=strategy,
                answer_style=answer_style,
                same_branch_preferred=same_branch_preferred,
                diversity_required=diversity_required,
                target_units=target_units,
                target_numbers=target_numbers,
                topics=topics,
                decomposition_queries=decomposition_queries,
                confidence=round(confidence, 2),
                source=f"llm_{provider.provider_name}",
                reason_short=reason_short,
                decision_trace=[
                    f"planner:provider={provider.provider_name}",
                    f"planner:strategy={strategy}",
                    f"planner:confidence={round(confidence, 2)}",
                    f"planner:answer_style={answer_style}",
                    f"planner:decompose={len(decomposition_queries)}",
                ],
            )
        except Exception:
            return None

    def _fallback_plan(self, *, structure: QueryStructure) -> QueryPlan:
        strategy = self.strategy_classifier.default_strategy()
        answer_style = self._default_answer_style(strategy)
        decomposition_queries = list(structure.expansion_queries[:3]) if strategy != "focal" else []
        target_units = self._dedupe([ref.kind for ref in structure.references])[:3]
        target_numbers = self._dedupe([ref.value for ref in structure.references])[:4]
        topics = list(structure.topic_terms[:6])
        return QueryPlan(
            strategy=strategy,
            answer_style=answer_style,
            same_branch_preferred=strategy == "focal",
            diversity_required=strategy != "focal",
            target_units=target_units,
            target_numbers=target_numbers,
            topics=topics,
            decomposition_queries=decomposition_queries,
            confidence=0.34,
            source="prior_fallback",
            reason_short="Planner fallback based on the learned class prior from routing training data.",
            decision_trace=[
                "planner:provider=prior_fallback",
                f"planner:strategy={strategy}",
                "planner:confidence=0.34",
                f"planner:answer_style={answer_style}",
                f"planner:decompose={len(decomposition_queries)}",
            ],
        )

    def _build_plan_from_classification(
        self,
        *,
        classification: StrategyClassification,
        structure: QueryStructure,
    ) -> QueryPlan:
        strategy = classification.strategy
        answer_style = self._default_answer_style(strategy)
        decomposition_queries = list(structure.expansion_queries[:3]) if strategy != "focal" else []
        target_units = self._dedupe([reference.kind for reference in structure.references])[:3]
        target_numbers = self._dedupe([reference.value for reference in structure.references])[:4]
        topics = list(structure.topic_terms[:6])
        return QueryPlan(
            strategy=strategy,
            answer_style=answer_style,
            same_branch_preferred=strategy == "focal",
            diversity_required=strategy != "focal",
            target_units=target_units,
            target_numbers=target_numbers,
            topics=topics,
            decomposition_queries=decomposition_queries,
            confidence=classification.confidence,
            source=classification.source,
            reason_short=classification.reason_short,
            decision_trace=classification.decision_trace + [
                f"planner:answer_style={answer_style}",
                f"planner:decompose={len(decomposition_queries)}",
            ],
        )

    def _planning_prompt(self, *, question: str, route: QueryRoute, structure: QueryStructure) -> str:
        references = ", ".join(f"{ref.kind}:{ref.value}" for ref in structure.references) or "none"
        markers = ", ".join(structure.document_markers) or "none"
        topics = ", ".join(structure.topic_terms) or "none"
        clauses = " | ".join(structure.comparison_clauses) or "none"
        return (
            "Clasifica la pregunta en una estrategia de retrieval para un sistema RAG estructurado.\n"
            "Devuelve solo JSON con las llaves:\n"
            "strategy, confidence, reason_short, answer_style, same_branch_preferred, "
            "diversity_required, target_units, target_numbers, topics, decomposition_queries.\n\n"
            "Reglas:\n"
            "- strategy debe ser exactamente uno de: focal, multi_branch, global.\n"
            "- focal: la respuesta probablemente vive en una sola unidad o en una region pequena y contigua.\n"
            "- multi_branch: la respuesta probablemente esta repartida en varias ramas/unidades y requiere cobertura diversa.\n"
            "- global: la respuesta requiere una vista de alto nivel o sintesis sobre gran parte del documento o del corpus.\n"
            "- answer_style debe ser: single_branch, grouped_by_branch o summary.\n"
            "- target_units usa solo: article, section, recommendation, annex, chapter, title, book, paragraph, branch.\n"
            "- decomposition_queries debe tener 0 a 3 subconsultas cortas. Para focal normalmente [].\n"
            "- No inventes numeros o unidades que no esten apoyados por la pregunta.\n\n"
            f"question: {question}\n"
            f"route_query_type: {route.query_type}\n"
            f"route_target_classes: {', '.join(route.target_classes)}\n"
            f"references: {references}\n"
            f"document_markers: {markers}\n"
            f"topic_terms: {topics}\n"
            f"comparison_clauses: {clauses}\n"
        )

    def _sanitize_string_list(
        self,
        values: object,
        *,
        allowed: set[str] | None = None,
        limit: int = 8,
    ) -> list[str]:
        if isinstance(values, str):
            raw_values = [values]
        elif isinstance(values, list):
            raw_values = [str(item) for item in values]
        else:
            return []

        sanitized: list[str] = []
        seen: set[str] = set()
        for item in raw_values:
            value = " ".join((item or "").strip().lower().split())
            if not value or value in seen:
                continue
            if allowed is not None and value not in allowed:
                continue
            seen.add(value)
            sanitized.append(value[:120])
            if len(sanitized) >= limit:
                break
        return sanitized

    def _dedupe(self, values: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = " ".join((value or "").strip().lower().split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    def _default_answer_style(self, strategy: str) -> str:
        if strategy == "global":
            return "summary"
        if strategy == "multi_branch":
            return "grouped_by_branch"
        return "single_branch"
