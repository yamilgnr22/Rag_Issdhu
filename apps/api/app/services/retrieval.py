from __future__ import annotations

import logging
import math
import re
import unicodedata
from dataclasses import dataclass

import httpx


logger = logging.getLogger(__name__)

from app.models.contracts import CitationContract, QueryRequest, QueryResponse
from app.services.answer_synthesis import AnswerSynthesisEvidence, AnswerSynthesisService
from app.services.branch_centrality_scorer import (
    BranchCentralityCandidate,
    BranchCentralityScorer,
)
from app.services.embedding_gateway import EmbeddingGateway
from app.services.query_evidence_intent_classifier import (
    EvidenceIntentClassification,
    QueryEvidenceIntentClassifier,
)
from app.services.query_planning import QueryPlan, QueryPlanningService
from app.services.query_rewriting import QueryRewriteService
from app.services.query_routing import QueryRoute, QueryRoutingService
from app.services.query_structure import QueryStructure, QueryStructureService
from app.services.reranking import RerankDocument, RerankGateway


STOPWORDS = {
    "a",
    "al",
    "ante",
    "con",
    "de",
    "del",
    "el",
    "en",
    "es",
    "la",
    "las",
    "los",
    "para",
    "por",
    "que",
    "se",
    "su",
    "un",
    "una",
    "y",
    "the",
    "of",
    "to",
    "for",
    "in",
    "on",
}
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
BOOSTABLE_PHRASES = {
    "cumplimiento tecnico": "cumplimiento técnico",
    "costo amortizado": "costo amortizado",
    "instrumentos financieros basicos": "instrumentos financieros básicos",
    "enfoque basado en riesgo": "enfoque basado en riesgo",
    "rentas del trabajo": "rentas del trabajo",
    "retenciones definitivas": "retenciones definitivas",
    "deducciones autorizadas": "deducciones autorizadas",
    "politica contable": "politica contable",
    "varios empleadores": "dos o mas empleadores",
    "dos o mas empleadores": "dos o mas empleadores",
    "empleador": "empleador",
    "ir laboral": "rentas del trabajo",
    "declaracion anual": "declaracion anual",
    "no residentes": "no residentes",
    "lavado de activos": "lavado de activos",
    "financiamiento del terrorismo": "financiamiento del terrorismo",
    "documentos guia del gafi": "documentos guia del gafi",
    "anexo i": "anexo i",
    "secciones 11 y 12": "secciones 11 y 12",
    "secciones 1 a 35": "secciones 1 a 35",
    "instrumentos financieros": "instrumentos financieros",
    "recomendacion 1": "recomendacion 1",
    "metodologia": "metodologia",
    "secciones": "secciones",
    "recomendaciones": "recomendaciones",
    "nic 39": "nic 39",
    "politica contable alternativa": "politica contable alternativa",
    "documentos guia": "documentos guia del gafi",
    "regla sustantiva": "recomendacion 1",
    "nota para los evaluadores": "nota para los evaluadores",
}


@dataclass
class RetrievalHit:
    chunk_id: str
    document_id: str
    version_id: str
    raw_text: str
    contextualized_text: str
    retrieval_context: str
    payload: dict[str, object]
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    # Score crudo del cross-encoder, antes de la normalizacion min-max. Es el
    # unico valor con significado absoluto del pipeline: ~0.99 cuando la
    # evidencia responde la pregunta, ~0.002 cuando no. Se usa para la confianza.
    provider_rerank_score: float | None = None


@dataclass
class StructuralReference:
    kind: str
    number: str


@dataclass
class RetrievalRun:
    request: QueryRequest
    route: QueryRoute
    decision_trace: list[str]
    dense_hits: list[RetrievalHit]
    sparse_hits: list[RetrievalHit]
    fused_hits: list[RetrievalHit]
    final_hits: list[RetrievalHit]
    dense_reason: str | None
    sparse_reason: str | None
    response: QueryResponse


@dataclass
class AnswerPackage:
    hits: list[RetrievalHit]
    evidence_items: list[AnswerSynthesisEvidence]
    label_to_hit: dict[str, RetrievalHit]
    decision_trace: list[str]
    answer_form: str
    answer_style: str


@dataclass
class EvidenceSegment:
    segment_id: str
    branch_key: str
    hits: list[RetrievalHit]
    text: str
    score: float
    coverage_score: float
    label_score: float
    aspect_matches: list[str]


@dataclass
class CanonicalUnitBundle:
    unit_label: str
    unit_key: str
    candidate: CanonicalUnitCandidate
    hits: list[RetrievalHit]


@dataclass
class AnswerUnitCandidate:
    unit_key: str
    canonical_label: str
    branch_label: str
    block_key: str
    block_label: str
    document_id: str
    version_id: str
    path_text: str
    unit_type: str
    unit_number: str
    evidence_role: str
    best_hit: RetrievalHit
    hits: list[RetrievalHit]
    focus_score: float
    rerank_score: float
    base_score: float
    text: str


@dataclass
class AnswerBlockCandidate:
    block_key: str
    block_label: str
    document_id: str
    version_id: str
    path_text: str
    unit_candidates: list[AnswerUnitCandidate]
    focus_score: float
    rerank_score: float
    base_score: float
    text: str


class RetrievalService:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.embeddings = EmbeddingGateway(settings)
        self.router = QueryRoutingService(settings)
        self.structure = QueryStructureService()
        self.planner = QueryPlanningService(settings)
        self.branch_centrality_scorer = BranchCentralityScorer(settings)
        self.evidence_intent_classifier = QueryEvidenceIntentClassifier(settings)
        self.rewriter = QueryRewriteService()
        self.reranker = RerankGateway(settings)
        self.answer_synthesizer = AnswerSynthesisService(settings)

    def query(self, request: QueryRequest) -> QueryResponse:
        return self.run(request).response

    def run(self, request: QueryRequest) -> RetrievalRun:
        route = self.router.route(question=request.question, filters=request.filters)
        decision_trace = list(route.decision_trace)
        structure = self.structure.parse(question=request.question, route=route)
        decision_trace.extend(structure.decision_trace)
        plan = self.planner.plan(
            question=request.question,
            route=route,
            structure=structure,
        )
        decision_trace.extend(plan.decision_trace)
        evidence_intent = self.evidence_intent_classifier.classify(question=request.question)
        decision_trace.extend(evidence_intent.decision_trace)
        rewrite_plan = self.rewriter.build(
            question=request.question,
            route=route,
            structure=structure,
            plan=plan,
        )
        decision_trace.extend(rewrite_plan.decision_trace)

        search_queries = self._dedupe_queries([request.question, *rewrite_plan.search_queries])
        expansion_subqueries = self._dedupe_queries(rewrite_plan.expansion_queries)
        branch_queries = self._dedupe_queries([request.question, *search_queries[1:], *expansion_subqueries])
        if expansion_subqueries:
            decision_trace.append(f"expand:clauses={len(expansion_subqueries)}")

        branch_hits, branch_trace = self._branch_search(
            queries=branch_queries,
            request=request,
            route=route,
            plan=plan,
        )
        decision_trace.extend(branch_trace)
        selected_branch_keys, branch_select_trace = self._select_branch_keys(
            request.question,
            branch_hits,
            plan,
            evidence_intent=evidence_intent,
        )
        decision_trace.extend(branch_select_trace)
        selected_branch_keys, branch_expand_trace = self._expand_selected_branch_keys(
            question=request.question,
            request=request,
            route=route,
            plan=plan,
            evidence_intent=evidence_intent,
            branch_hits=branch_hits,
            selected_branch_keys=selected_branch_keys,
        )
        decision_trace.extend(branch_expand_trace)
        if selected_branch_keys:
            decision_trace.append(f"branch:selected={len(selected_branch_keys)}")

        dense_hits, dense_reason = self._dense_search(request, route)
        if dense_reason:
            decision_trace.append(f"dense:{dense_reason}")
        else:
            decision_trace.append(f"dense:hits={len(dense_hits)}")

        sparse_hits, sparse_reason = self._sparse_search(request, route)
        if sparse_reason:
            decision_trace.append(f"sparse:{sparse_reason}")
        else:
            decision_trace.append(f"sparse:hits={len(sparse_hits)}")

        if len(search_queries) > 1:
            dense_hits, sparse_hits = self._apply_query_variants(
                queries=search_queries[1:],
                request=request,
                route=route,
                dense_hits=dense_hits,
                sparse_hits=sparse_hits,
            )
            decision_trace.append(f"rewrite:dense={len(dense_hits)}")
            decision_trace.append(f"rewrite:sparse={len(sparse_hits)}")

        if expansion_subqueries:
            dense_hits, sparse_hits = self._apply_query_variants(
                queries=expansion_subqueries,
                request=request,
                route=route,
                dense_hits=dense_hits,
                sparse_hits=sparse_hits,
            )
            decision_trace.append(f"expand:dense={len(dense_hits)}")
            decision_trace.append(f"expand:sparse={len(sparse_hits)}")

        if selected_branch_keys and plan.strategy in {"focal", "multi_branch"}:
            branch_dense_hits, branch_dense_reason = self._branch_guided_dense_search(
                queries=branch_queries,
                request=request,
                route=route,
                branch_keys=selected_branch_keys,
            )
            if branch_dense_reason:
                decision_trace.append(f"branch_filter_dense:{branch_dense_reason}")
            else:
                dense_hits = self._merge_ranked_hits(
                    [dense_hits, branch_dense_hits],
                    modality="dense",
                )
                decision_trace.append(f"branch_filter:dense={len(branch_dense_hits)}")

            branch_sparse_hits, branch_sparse_reason = self._branch_guided_sparse_search(
                queries=branch_queries,
                request=request,
                route=route,
                branch_keys=selected_branch_keys,
            )
            if branch_sparse_reason:
                decision_trace.append(f"branch_filter:{branch_sparse_reason}")
            else:
                sparse_hits = self._merge_ranked_hits(
                    [sparse_hits, branch_sparse_hits],
                    modality="sparse",
                )
                decision_trace.append(f"branch_filter:sparse={len(branch_sparse_hits)}")

        fused_hits = self._fuse_hits(dense_hits, sparse_hits)
        decision_trace.append(f"rrf:fused={len(fused_hits)}")
        fused_hits, hierarchy_trace = self._apply_hierarchical_boost(
            request.question,
            fused_hits,
            branch_hits,
        )
        decision_trace.extend(hierarchy_trace)
        fused_hits, branch_candidate_trace = self._augment_with_branch_candidates(
            hits=fused_hits,
            branch_hits=branch_hits,
            selected_branch_keys=selected_branch_keys,
            plan=plan,
        )
        decision_trace.extend(branch_candidate_trace)

        reranked_hits, rerank_trace = self._rerank_hits(
            request.question,
            fused_hits,
            evidence_intent=evidence_intent,
        )
        decision_trace.extend(rerank_trace)
        reranked_hits, segment_trace = self._attach_relevant_segments(
            request=request,
            question=request.question,
            hits=reranked_hits,
            structure=structure,
            plan=plan,
        )
        decision_trace.extend(segment_trace)
        if self.settings.retrieval_visibility_policy_enabled:
            reranked_hits, visibility_trace = self._apply_visibility_policy(
                question=request.question,
                route=route,
                hits=reranked_hits,
                branch_hits=branch_hits,
                plan=plan,
                evidence_intent=evidence_intent,
                structure=structure,
            )
            decision_trace.extend(visibility_trace)
        else:
            decision_trace.append("visibility:disabled")
        if self.settings.retrieval_diversify_enabled:
            reranked_hits = self._diversify_hits(request.question, route, reranked_hits, plan)
        else:
            decision_trace.append("diversify:disabled")
        final_hits = reranked_hits[: max(1, int(self.settings.retrieval_fused_top_k))]
        decision_trace.append(f"rerank:final={len(final_hits)}")

        response: QueryResponse
        if not final_hits:
            fallback_reason = "no_retrieval_hits"
            if dense_reason and sparse_reason:
                fallback_reason = "dense_and_sparse_unavailable"
            elif dense_reason:
                fallback_reason = "dense_unavailable_no_hits"
            elif sparse_reason:
                fallback_reason = "sparse_unavailable_no_hits"

            response = QueryResponse(
                answer="No se encontro evidencia relevante en el indice para responder la consulta.",
                citations=[],
                confidence=0.0,
                applied_filters=request.filters,
                rewrites_applied=rewrite_plan.applied,
                decision_trace=decision_trace,
                fallback_reason=fallback_reason,
                source_conflict=False,
            )
            return RetrievalRun(
                request=request,
                route=route,
                decision_trace=decision_trace,
                dense_hits=dense_hits,
                sparse_hits=sparse_hits,
                fused_hits=fused_hits,
                final_hits=final_hits,
                dense_reason=dense_reason,
                sparse_reason=sparse_reason,
                response=response,
            )

        answer_package = self._build_answer_package(
            request=request,
            hits=reranked_hits,
            plan=plan,
            evidence_intent=evidence_intent,
            structure=structure,
        )
        if self.settings.retrieval_answer_prioritization_enabled:
            final_hits = self._prioritize_answer_hits(
                answer_hits=answer_package.hits,
                reranked_hits=reranked_hits,
                limit=max(1, int(self.settings.retrieval_fused_top_k)),
                anchor_limit=2 if plan.strategy == "multi_branch" else 1,
            )
        else:
            decision_trace.append("answer_select:disabled")
        decision_trace.append(f"answer_select:final={len(final_hits)}")

        confidence, confidence_source = self._confidence(final_hits)
        decision_trace.append(f"confidence:source={confidence_source}")
        decision_trace.append(f"confidence:value={confidence}")

        # Corte por evidencia debil: si ni el mejor pasaje supera el umbral, no
        # tiene sentido pagar una llamada al LLM para que redacte sobre material
        # que el cross-encoder considera irrelevante.
        min_confidence = float(self.settings.answer_min_confidence)
        if min_confidence > 0.0 and confidence < min_confidence:
            decision_trace.append(f"answer:abstained_low_confidence<{min_confidence}")
            citations = [
                CitationContract(
                    label=f"[{index}]",
                    document_id=hit.document_id,
                    version_id=hit.version_id,
                    chunk_id=hit.chunk_id,
                )
                for index, hit in enumerate(final_hits[:3], start=1)
            ]
            response = QueryResponse(
                answer=(
                    "No se encontro evidencia suficiente en los documentos indexados para "
                    "responder esta consulta. Se citan los pasajes mas cercanos por si "
                    "ayudan a reformular la pregunta."
                ),
                citations=citations,
                confidence=confidence,
                applied_filters=request.filters,
                rewrites_applied=rewrite_plan.applied,
                decision_trace=decision_trace,
                fallback_reason="low_confidence_evidence",
                source_conflict=False,
            )
            return RetrievalRun(
                request=request,
                route=route,
                decision_trace=decision_trace,
                dense_hits=dense_hits,
                sparse_hits=sparse_hits,
                fused_hits=fused_hits,
                final_hits=final_hits,
                dense_reason=dense_reason,
                sparse_reason=sparse_reason,
                response=response,
            )

        answer, answer_trace, used_answer_hits = self._compose_answer(
            request.question,
            answer_package.evidence_items,
            plan,
            answer_form=answer_package.answer_form,
            answer_style=answer_package.answer_style,
            evidence_intent=evidence_intent,
        )
        decision_trace.extend(answer_package.decision_trace)
        decision_trace.extend(answer_trace)
        citation_hits = self._citation_hits(used_answer_hits or answer_package.hits, final_hits)
        citations = [
            CitationContract(
                label=f"[{index}]",
                document_id=hit.document_id,
                version_id=hit.version_id,
                chunk_id=hit.chunk_id,
            )
            for index, hit in enumerate(citation_hits, start=1)
        ]
        source_conflict = len({hit.document_id for hit in final_hits[:2]}) > 1
        fallback_reason = None
        if dense_reason and not sparse_reason:
            fallback_reason = "dense_unavailable"
        elif sparse_reason and not dense_reason:
            fallback_reason = "sparse_unavailable"

        response = QueryResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            applied_filters=request.filters,
            rewrites_applied=rewrite_plan.applied,
            decision_trace=decision_trace,
            fallback_reason=fallback_reason,
            source_conflict=source_conflict,
        )
        return RetrievalRun(
            request=request,
            route=route,
            decision_trace=decision_trace,
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            fused_hits=fused_hits,
            final_hits=final_hits,
            dense_reason=dense_reason,
            sparse_reason=sparse_reason,
            response=response,
        )

    def _prioritize_answer_hits(
        self,
        *,
        answer_hits: list[RetrievalHit],
        reranked_hits: list[RetrievalHit],
        limit: int,
        anchor_limit: int,
    ) -> list[RetrievalHit]:
        anchor_hits: list[RetrievalHit] = []
        anchor_units: set[str] = set()
        for hit in reranked_hits:
            unit_identity = self._answer_hit_identity(hit)
            if unit_identity in anchor_units:
                continue
            anchor_units.add(unit_identity)
            anchor_hits.append(hit)
            if len(anchor_hits) >= max(0, anchor_limit):
                break

        merged: list[RetrievalHit] = []
        seen_units: set[str] = set()
        seen_chunks: set[str] = set()
        for hit in [*anchor_hits, *answer_hits, *reranked_hits]:
            unit_identity = self._answer_hit_identity(hit)
            if unit_identity in seen_units or hit.chunk_id in seen_chunks:
                continue
            seen_units.add(unit_identity)
            seen_chunks.add(hit.chunk_id)
            merged.append(hit)
            if len(merged) >= limit:
                break
        return merged

    def _answer_hit_identity(self, hit: RetrievalHit) -> str:
        unit_key = self._answer_unit_group_key(hit)
        if unit_key:
            return f"unit:{unit_key}"
        branch_key = self._branch_key_for_hit(hit)
        if branch_key:
            return f"branch:{branch_key}"
        payload = hit.payload or {}
        path_text = str(payload.get("path_text") or "").strip()
        canonical_label = str(payload.get("canonical_label") or "").strip()
        if path_text or canonical_label:
            identity = path_text or canonical_label
            return f"path:{hit.document_id}:{hit.version_id}:{self._normalize(identity)}"
        return f"chunk:{hit.chunk_id}"

    def _apply_query_variants(
        self,
        *,
        queries: list[str],
        request: QueryRequest,
        route: QueryRoute,
        dense_hits: list[RetrievalHit],
        sparse_hits: list[RetrievalHit],
    ) -> tuple[list[RetrievalHit], list[RetrievalHit]]:
        dense_lists = [dense_hits]
        sparse_lists = [sparse_hits]
        for query in queries:
            subrequest = QueryRequest(
                question=query,
                user_identity=request.user_identity,
                user_groups=request.user_groups,
                filters=request.filters,
                conversation_context=request.conversation_context,
            )
            extra_dense, _ = self._dense_search(subrequest, route)
            extra_sparse, _ = self._sparse_search(subrequest, route)
            dense_lists.append(extra_dense)
            sparse_lists.append(extra_sparse)
        return (
            self._merge_ranked_hits(dense_lists, modality="dense"),
            self._merge_ranked_hits(sparse_lists, modality="sparse"),
        )

    def _branch_guided_sparse_search(
        self,
        *,
        queries: list[str],
        request: QueryRequest,
        route: QueryRoute,
        branch_keys: list[str],
    ) -> tuple[list[RetrievalHit], str | None]:
        hit_lists: list[list[RetrievalHit]] = []
        for query in queries:
            subrequest = QueryRequest(
                question=query,
                user_identity=request.user_identity,
                user_groups=request.user_groups,
                filters=request.filters,
                conversation_context=request.conversation_context,
            )
            hits, reason = self._sparse_search(subrequest, route, branch_keys=branch_keys)
            if reason:
                return [], reason
            hit_lists.append(hits)
        return self._merge_ranked_hits(hit_lists, modality="sparse"), None

    def _branch_guided_dense_search(
        self,
        *,
        queries: list[str],
        request: QueryRequest,
        route: QueryRoute,
        branch_keys: list[str],
    ) -> tuple[list[RetrievalHit], str | None]:
        hit_lists: list[list[RetrievalHit]] = []
        for query in queries:
            subrequest = QueryRequest(
                question=query,
                user_identity=request.user_identity,
                user_groups=request.user_groups,
                filters=request.filters,
                conversation_context=request.conversation_context,
            )
            hits, reason = self._dense_search(subrequest, route, branch_keys=branch_keys)
            if reason:
                return [], reason
            hit_lists.append(hits)
        return self._merge_ranked_hits(hit_lists, modality="dense"), None

    def _dense_search(
        self,
        request: QueryRequest,
        route: QueryRoute,
        branch_keys: list[str] | None = None,
    ) -> tuple[list[RetrievalHit], str | None]:
        try:
            vectors, _ = self.embeddings.embed_texts([request.question])
            vector = vectors[0]
            body: dict[str, object] = {
                "vector": vector,
                "limit": int(self.settings.retrieval_dense_top_k),
                "with_payload": True,
            }
            filter_body = self._build_qdrant_filter(
                request.filters, route, branch_keys=branch_keys, request=request
            )
            if filter_body:
                body["filter"] = filter_body
            payload = self._qdrant_search(body=body)
            return self._parse_qdrant_hits(payload, modality="dense"), None
        except Exception as exc:
            logger.warning(
                "dense_search_failed collection=%s error=%s:%s",
                self.settings.resolved_qdrant_collection_name,
                type(exc).__name__,
                exc,
            )
            return [], self._truncate_reason(str(exc))

    def _sparse_search(
        self,
        request: QueryRequest,
        route: QueryRoute,
        branch_keys: list[str] | None = None,
    ) -> tuple[list[RetrievalHit], str | None]:
        try:
            reference_clauses = self._build_sparse_reference_clauses(request.question)
            phrase_clauses = self._build_sparse_phrase_clauses(request.question)
            query_body = {
                "size": int(self.settings.retrieval_sparse_top_k),
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": request.question,
                                    "fields": [
                                        "contextualized_text^4",
                                        "chunk_summary_context^5",
                                        "branch_summary_context^3",
                                        "reference_variants_text^6",
                                        "canonical_label^4",
                                        "unit_topic^5",
                                        "unit_key^5",
                                        "document_title^2",
                                        "focus_terms_text^3",
                                        "branch_keywords_text^3",
                                        "retrieval_context^3",
                                        "path_text^2",
                                        "raw_text",
                                    ],
                                    "type": "best_fields",
                                }
                            }
                        ],
                        "filter": self._build_opensearch_filters(
                            request.filters,
                            route,
                            branch_keys=branch_keys,
                            request=request,
                        ),
                        "should": reference_clauses
                        + phrase_clauses
                        + self._class_preference_clauses(route),
                    }
                },
            }
            payload = self._opensearch_search(query_body)
            return self._parse_opensearch_hits(payload), None
        except Exception as exc:
            logger.warning(
                "sparse_search_failed index=%s error=%s:%s",
                self.settings.resolved_opensearch_index_name,
                type(exc).__name__,
                exc,
            )
            return [], self._truncate_reason(str(exc))

    def _branch_search(
        self,
        *,
        queries: list[str],
        request: QueryRequest,
        route: QueryRoute,
        plan: QueryPlan,
    ) -> tuple[list[RetrievalHit], list[str]]:
        dense_lists: list[list[RetrievalHit]] = []
        sparse_lists: list[list[RetrievalHit]] = []
        dense_reason: str | None = None
        sparse_reason: str | None = None

        for query in queries:
            subrequest = QueryRequest(
                question=query,
                user_identity=request.user_identity,
                user_groups=request.user_groups,
                filters=request.filters,
                conversation_context=request.conversation_context,
            )
            dense_hits, current_dense_reason = self._branch_dense_search(subrequest, route)
            sparse_hits, current_sparse_reason = self._branch_sparse_search(subrequest, route)
            if current_dense_reason:
                dense_reason = current_dense_reason
            else:
                dense_lists.append(dense_hits)
            if current_sparse_reason:
                sparse_reason = current_sparse_reason
            else:
                sparse_lists.append(sparse_hits)

        dense_hits = self._merge_ranked_hits(dense_lists, modality="dense") if dense_lists else []
        sparse_hits = self._merge_ranked_hits(sparse_lists, modality="sparse") if sparse_lists else []
        branch_limit = {
            "focal": 12,
            "multi_branch": 14,
            "global": 16,
        }.get(plan.strategy, 12)
        fused_hits = self._fuse_hits(dense_hits, sparse_hits)[:branch_limit]
        fused_hits = self._rank_branch_hits(question=request.question, hits=fused_hits, plan=plan)

        trace: list[str] = []
        if dense_reason:
            trace.append(f"branch_dense:{dense_reason}")
        else:
            trace.append(f"branch_dense:hits={len(dense_hits)}")
        if sparse_reason:
            trace.append(f"branch_sparse:{sparse_reason}")
        else:
            trace.append(f"branch_sparse:hits={len(sparse_hits)}")
        trace.append(f"branch_rrf:fused={len(fused_hits)}")
        return fused_hits, trace

    def _rank_branch_hits(
        self,
        *,
        question: str,
        hits: list[RetrievalHit],
        plan: QueryPlan,
    ) -> list[RetrievalHit]:
        if not hits:
            return hits

        query_tokens = self._tokens(question)
        structural_refs = self._extract_structural_references(question)
        target_units = {value.strip().lower() for value in plan.target_units if value.strip()}
        target_numbers = {value.strip().upper() for value in plan.target_numbers if value.strip()}

        for hit in hits:
            payload = hit.payload or {}
            branch_tokens = self._tokens(
                " ".join(
                    [
                        str(payload.get("canonical_label") or ""),
                        str(payload.get("unit_topic") or ""),
                        str(payload.get("path_text") or ""),
                        str(payload.get("branch_summary_context") or ""),
                        str(payload.get("branch_keywords_text") or ""),
                        str(payload.get("reference_variants_text") or ""),
                        hit.raw_text[:900],
                    ]
                )
            )
            lexical = max(
                self._overlap_ratio(query_tokens, branch_tokens),
                self._soft_overlap_ratio(query_tokens, branch_tokens),
            )
            exact = 0.45 if self._contains_exact_reference(question, hit) else 0.0
            structural = self._structural_reference_score(structural_refs, hit)
            mismatch = self._structural_reference_mismatch_penalty(structural_refs, hit)
            hit.rrf_score = (
                hit.rrf_score
                + (0.55 * lexical)
                + exact
                + structural
                + self._branch_target_unit_score(hit, target_units)
                + self._branch_target_number_score(hit, target_numbers)
                + self._branch_specificity_score(hit, plan)
                - (1.15 * mismatch)
                - (0.7 * self._boilerplate_score(hit))
            )

        return sorted(hits, key=lambda item: item.rrf_score, reverse=True)

    def _branch_dense_search(
        self,
        request: QueryRequest,
        route: QueryRoute,
        *,
        parent_branch_keys: list[str] | None = None,
    ) -> tuple[list[RetrievalHit], str | None]:
        try:
            vectors, _ = self.embeddings.embed_texts([request.question])
            vector = vectors[0]
            body: dict[str, object] = {
                "vector": vector,
                "limit": max(8, int(self.settings.retrieval_dense_top_k)),
                "with_payload": True,
            }
            filter_body = self._build_qdrant_filter(
                request.filters,
                route,
                parent_branch_keys=parent_branch_keys,
                request=request,
            )
            if filter_body:
                body["filter"] = filter_body
            payload = self._qdrant_search(
                body=body,
                collections=self._qdrant_branch_collection_candidates(),
            )
            return self._parse_qdrant_hits(payload, modality="dense"), None
        except Exception as exc:
            return [], self._truncate_reason(str(exc))

    def _branch_sparse_search(
        self,
        request: QueryRequest,
        route: QueryRoute,
        *,
        parent_branch_keys: list[str] | None = None,
    ) -> tuple[list[RetrievalHit], str | None]:
        try:
            reference_clauses = self._build_sparse_reference_clauses(request.question)
            phrase_clauses = self._build_sparse_phrase_clauses(request.question)
            query_body = {
                "size": max(8, int(self.settings.retrieval_sparse_top_k)),
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": request.question,
                                    "fields": [
                                        "unit_key^10",
                                        "reference_variants_text^9",
                                        "canonical_label^9",
                                        "unit_topic^10",
                                        "branch_summary_context^10",
                                        "chunk_summary_context^8",
                                        "path_text^7",
                                        "branch_keywords_text^7",
                                        "focus_terms_text^6",
                                        "document_title^5",
                                        "retrieval_context^5",
                                        "contextualized_text^4",
                                        "raw_text^3",
                                    ],
                                    "type": "best_fields",
                                }
                            }
                        ],
                        "filter": self._build_opensearch_filters(
                            request.filters,
                            route,
                            parent_branch_keys=parent_branch_keys,
                            request=request,
                        ),
                        "should": reference_clauses + phrase_clauses,
                    }
                },
            }
            payload = self._opensearch_search(
                query_body,
                index_candidates=self._opensearch_branch_index_candidates(),
            )
            return self._parse_opensearch_hits(payload), None
        except Exception as exc:
            return [], self._truncate_reason(str(exc))

    def _apply_hierarchical_boost(
        self,
        question: str,
        hits: list[RetrievalHit],
        branch_hits: list[RetrievalHit],
    ) -> tuple[list[RetrievalHit], list[str]]:
        if not hits:
            return hits, ["hierarchy:branches=0"]
        if not branch_hits:
            return hits, ["hierarchy:branches=0"]

        structural_refs = self._extract_structural_references(question)
        branch_scores: dict[str, float] = {}
        for rank, hit in enumerate(branch_hits, start=1):
            branch_key = str(hit.payload.get("branch_key") or hit.chunk_id or "").strip()
            if not branch_key:
                continue
            branch_scores[branch_key] = branch_scores.get(branch_key, 0.0) + (
                (1.0 / rank) + (0.4 * self._structural_reference_score(structural_refs, hit))
            )

        top_branches = sorted(branch_scores.items(), key=lambda item: item[1], reverse=True)[:4]
        if not top_branches:
            return hits, ["hierarchy:branches=0"]

        boosted_hits: list[RetrievalHit] = []
        for hit in hits:
            hit_keys = {
                str(value).strip()
                for value in (hit.payload.get("branch_keys") or [])
                if str(value).strip()
            }
            branch_boost = 0.0
            if hit_keys:
                for branch_key, score in top_branches:
                    if branch_key in hit_keys:
                        branch_boost = max(branch_boost, 0.22 * score)
            hit.rrf_score += branch_boost
            boosted_hits.append(hit)

        boosted_hits.sort(key=lambda item: item.rrf_score, reverse=True)
        return boosted_hits, [f"hierarchy:branches={len(top_branches)}"]

    def _augment_with_branch_candidates(
        self,
        *,
        hits: list[RetrievalHit],
        branch_hits: list[RetrievalHit],
        selected_branch_keys: list[str],
        plan: QueryPlan,
    ) -> tuple[list[RetrievalHit], list[str]]:
        if not branch_hits or not selected_branch_keys:
            return hits, ["branch_candidates=0"]

        selected = set(selected_branch_keys)
        branch_limit = {
            "focal": 2,
            "multi_branch": 4,
            "global": 5,
        }.get(plan.strategy, 3)
        candidates: list[RetrievalHit] = []
        for hit in branch_hits:
            branch_key = str(hit.payload.get("branch_key") or hit.chunk_id or "").strip()
            if not branch_key or branch_key not in selected:
                continue
            if plan.strategy in {"focal", "multi_branch"} and not self._is_specific_branch_hit(hit):
                continue
            candidates.append(hit)

        candidates = self._unique_branch_hits(candidates, limit=branch_limit)
        if not candidates:
            return hits, ["branch_candidates=0"]

        merged: dict[str, RetrievalHit] = {hit.chunk_id: hit for hit in hits}
        for candidate in candidates:
            existing = merged.get(candidate.chunk_id)
            if existing is None or candidate.rrf_score > existing.rrf_score:
                merged[candidate.chunk_id] = candidate

        combined = sorted(merged.values(), key=lambda item: item.rrf_score, reverse=True)
        return combined, [f"branch_candidates={len(candidates)}"]

    def _expand_selected_branch_keys(
        self,
        *,
        question: str,
        request: QueryRequest,
        route: QueryRoute,
        plan: QueryPlan,
        evidence_intent: EvidenceIntentClassification,
        branch_hits: list[RetrievalHit],
        selected_branch_keys: list[str],
    ) -> tuple[list[str], list[str]]:
        if not selected_branch_keys or plan.strategy == "global":
            return selected_branch_keys, ["branch_expand=0"]

        key_to_hit = {
            str(hit.payload.get("branch_key") or hit.chunk_id or "").strip(): hit
            for hit in branch_hits
            if str(hit.payload.get("branch_key") or hit.chunk_id or "").strip()
        }
        roots = [
            key
            for key in selected_branch_keys
            if key in key_to_hit
            and self._is_specific_branch_hit(key_to_hit[key])
            and int(key_to_hit[key].payload.get("child_count") or 0) > 0
        ]
        roots = roots[:3]
        if not roots:
            pruned = self._prune_ancestor_branch_keys(selected_branch_keys, key_to_hit, plan)
            return pruned, ["branch_expand=0", f"branch_pruned={len(selected_branch_keys) - len(pruned)}"]

        child_hits, child_trace = self._branch_child_search(
            question=question,
            request=request,
            route=route,
            parent_branch_keys=roots,
        )
        if not child_hits:
            pruned = self._prune_ancestor_branch_keys(selected_branch_keys, key_to_hit, plan)
            return pruned, [*child_trace, "branch_expand=0", f"branch_pruned={len(selected_branch_keys) - len(pruned)}"]

        child_limit = {
            "focal": 3,
            "multi_branch": 4,
        }.get(plan.strategy, 3)
        child_keys, child_select_trace = self._select_branch_keys(
            question,
            child_hits,
            plan,
            evidence_intent=evidence_intent,
        )
        child_keys = child_keys[:child_limit]
        combined = self._dedupe_strings([*selected_branch_keys, *child_keys])
        combined_hit_map = {**key_to_hit}
        combined_hit_map.update(
            {
                str(hit.payload.get("branch_key") or hit.chunk_id or "").strip(): hit
                for hit in child_hits
                if str(hit.payload.get("branch_key") or hit.chunk_id or "").strip()
            }
        )
        pruned = self._prune_ancestor_branch_keys(combined, combined_hit_map, plan)
        return pruned, [
            *child_trace,
            *child_select_trace,
            f"branch_expand={len(child_keys)}",
            f"branch_pruned={len(combined) - len(pruned)}",
        ]

    def _branch_child_search(
        self,
        *,
        question: str,
        request: QueryRequest,
        route: QueryRoute,
        parent_branch_keys: list[str],
    ) -> tuple[list[RetrievalHit], list[str]]:
        if not parent_branch_keys:
            return [], ["branch_local=0"]

        subrequest = QueryRequest(
            question=question,
            user_identity=request.user_identity,
                user_groups=request.user_groups,
            filters=request.filters,
            conversation_context=request.conversation_context,
        )
        dense_hits, dense_reason = self._branch_dense_search(
            subrequest,
            route,
            parent_branch_keys=parent_branch_keys,
        )
        sparse_hits, sparse_reason = self._branch_sparse_search(
            subrequest,
            route,
            parent_branch_keys=parent_branch_keys,
        )
        fused_hits = self._fuse_hits(dense_hits, sparse_hits)
        trace: list[str] = []
        if dense_reason:
            trace.append(f"branch_local_dense:{dense_reason}")
        else:
            trace.append(f"branch_local_dense={len(dense_hits)}")
        if sparse_reason:
            trace.append(f"branch_local_sparse:{sparse_reason}")
        else:
            trace.append(f"branch_local_sparse={len(sparse_hits)}")
        trace.append(f"branch_local={len(fused_hits)}")
        return fused_hits, trace

    def _branch_catalog_search(
        self,
        *,
        request: QueryRequest,
        route: QueryRoute,
        parent_branch_keys: list[str],
        parent_path_specs: list[tuple[str, int]] | None = None,
        limit: int,
    ) -> tuple[list[RetrievalHit], str | None]:
        if not parent_branch_keys and not parent_path_specs:
            return [], None
        try:
            hit_lists: list[list[RetrievalHit]] = []
            if parent_branch_keys:
                query_body = {
                    "size": max(8, int(limit)),
                    "query": {
                        "bool": {
                            "filter": self._build_opensearch_filters(
                                request.filters,
                                route,
                                parent_branch_keys=parent_branch_keys,
                                request=request,
                            ),
                        }
                    },
                }
                payload = self._opensearch_search(
                    query_body,
                    index_candidates=self._opensearch_branch_index_candidates(),
                )
                hit_lists.append(self._parse_opensearch_hits(payload))

            for parent_path_text, parent_depth in parent_path_specs or []:
                if not parent_path_text or parent_depth <= 0:
                    continue
                query_body = {
                    "size": max(6, int(limit)),
                    "query": {
                        "bool": {
                            "filter": [
                                *self._build_opensearch_filters(request.filters, route, request=request),
                                {"term": {"branch_depth": parent_depth + 1}},
                            ],
                            "must": [
                                {
                                    "match_phrase": {
                                        "path_text": parent_path_text,
                                    }
                                }
                            ],
                        }
                    },
                }
                payload = self._opensearch_search(
                    query_body,
                    index_candidates=self._opensearch_branch_index_candidates(),
                )
                hit_lists.append(self._parse_opensearch_hits(payload))

            merged: list[RetrievalHit] = []
            seen: set[str] = set()
            for hit in self._merge_ranked_hits(hit_lists, modality="sparse"):
                if hit.chunk_id in seen:
                    continue
                seen.add(hit.chunk_id)
                merged.append(hit)
                if len(merged) >= max(8, int(limit)):
                    break
            return merged, None
        except Exception as exc:
            return [], self._truncate_reason(str(exc))

    def _prune_ancestor_branch_keys(
        self,
        branch_keys: list[str],
        key_to_hit: dict[str, RetrievalHit],
        plan: QueryPlan,
    ) -> list[str]:
        if plan.strategy == "global":
            return branch_keys
        kept: list[str] = []
        for key in branch_keys:
            hit = key_to_hit.get(key)
            if hit is None:
                kept.append(key)
                continue
            path = self._as_path(hit.payload.get("hierarchy_path") or hit.payload.get("section_path"))
            doc_id = hit.document_id
            version_id = hit.version_id
            is_ancestor = False
            for other_key in branch_keys:
                if other_key == key:
                    continue
                other_hit = key_to_hit.get(other_key)
                if other_hit is None:
                    continue
                if other_hit.document_id != doc_id or other_hit.version_id != version_id:
                    continue
                other_path = self._as_path(other_hit.payload.get("hierarchy_path") or other_hit.payload.get("section_path"))
                if len(other_path) <= len(path):
                    continue
                if other_path[: len(path)] == path:
                    is_ancestor = True
                    break
            if not is_ancestor:
                kept.append(key)
        return self._dedupe_strings(kept)

    def _parse_opensearch_hits(self, payload: dict[str, object]) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        for rank, item in enumerate(payload.get("hits", {}).get("hits", []), start=1):
            source = item.get("_source", {})
            hits.append(
                RetrievalHit(
                    chunk_id=str(source.get("chunk_id")),
                    document_id=str(source.get("document_id")),
                    version_id=str(source.get("version_id")),
                    raw_text=str(source.get("raw_text", "")),
                    contextualized_text=str(source.get("contextualized_text", "")),
                    retrieval_context=str(source.get("retrieval_context", "")),
                    payload=source,
                    sparse_rank=rank,
                    sparse_score=float(item.get("_score", 0.0)),
                )
            )
        return hits

    def _parse_qdrant_hits(
        self,
        payload: dict[str, object],
        *,
        modality: str,
    ) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        for rank, item in enumerate(payload.get("result", []), start=1):
            source = item.get("payload", {})
            hits.append(
                RetrievalHit(
                    chunk_id=str(source.get("chunk_id") or source.get("branch_id")),
                    document_id=str(source.get("document_id")),
                    version_id=str(source.get("version_id")),
                    raw_text=str(source.get("raw_text", "")),
                    contextualized_text=str(source.get("contextualized_text", "")),
                    retrieval_context=str(source.get("retrieval_context", "")),
                    payload=source,
                    dense_rank=rank if modality == "dense" else None,
                    sparse_rank=rank if modality == "sparse" else None,
                    dense_score=float(item.get("score", 0.0)) if modality == "dense" else None,
                    sparse_score=float(item.get("score", 0.0)) if modality == "sparse" else None,
                )
            )
        return hits

    def _opensearch_search(
        self,
        query_body: dict[str, object],
        *,
        index_candidates: list[str] | None = None,
    ) -> dict[str, object]:
        payload = None
        candidates = index_candidates or self._opensearch_index_candidates()
        with httpx.Client(timeout=45.0) as client:
            for index_name in candidates:
                response = client.post(
                    f"{self.settings.opensearch_url.rstrip('/')}/{index_name}/_search",
                    json=query_body,
                )
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                payload = response.json()
                if index_name != candidates[0]:
                    # El indice del perfil activo no existe y se cayo al indice
                    # sin perfil, que puede estar construido con otro modelo de
                    # embedding: los resultados no serian comparables.
                    logger.error(
                        "index_profile_fallback expected=%s used=%s",
                        candidates[0],
                        index_name,
                    )
                break
        if payload is None:
            raise RuntimeError("OpenSearch index not found for retrieval search.")
        return payload

    def _qdrant_search(
        self,
        *,
        body: dict[str, object],
        collections: list[str] | None = None,
    ) -> dict[str, object]:
        payload = None
        candidates = collections or self._qdrant_collection_candidates()
        with httpx.Client(timeout=45.0) as client:
            for collection_name in candidates:
                response = client.post(
                    f"{self.settings.qdrant_url.rstrip('/')}/collections/{collection_name}/points/search",
                    json=body,
                )
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                payload = response.json()
                if collection_name != candidates[0]:
                    logger.error(
                        "collection_profile_fallback expected=%s used=%s",
                        candidates[0],
                        collection_name,
                    )
                break
        if payload is None:
            raise RuntimeError("Qdrant collection not found for retrieval search.")
        return payload

    def _opensearch_index_candidates(self) -> list[str]:
        candidates = [
            self.settings.resolved_opensearch_index_name,
            self.settings.opensearch_index_name,
        ]
        return self._dedupe_strings(candidates)

    def _opensearch_branch_index_candidates(self) -> list[str]:
        candidates = [
            self.settings.resolved_opensearch_branch_index_name,
            self.settings.opensearch_branch_index_name,
        ]
        return self._dedupe_strings(candidates)

    def _qdrant_collection_candidates(self) -> list[str]:
        candidates = [
            self.settings.resolved_qdrant_collection_name,
            self.settings.qdrant_collection_name,
        ]
        return self._dedupe_strings(candidates)

    def _qdrant_branch_collection_candidates(self) -> list[str]:
        candidates = [
            self.settings.resolved_qdrant_branch_collection_name,
            self.settings.qdrant_branch_collection_name,
        ]
        return self._dedupe_strings(candidates)

    def _fuse_hits(self, dense_hits: list[RetrievalHit], sparse_hits: list[RetrievalHit]) -> list[RetrievalHit]:
        by_chunk: dict[str, RetrievalHit] = {}
        rrf_k = max(1, int(self.settings.retrieval_rrf_k))

        for hit in dense_hits:
            by_chunk[hit.chunk_id] = hit
        for hit in sparse_hits:
            existing = by_chunk.get(hit.chunk_id)
            if existing is None:
                by_chunk[hit.chunk_id] = hit
            else:
                existing.sparse_rank = hit.sparse_rank
                existing.sparse_score = hit.sparse_score

        for hit in by_chunk.values():
            if hit.dense_rank is not None:
                hit.rrf_score += 1.0 / (rrf_k + hit.dense_rank)
            if hit.sparse_rank is not None:
                hit.rrf_score += 1.0 / (rrf_k + hit.sparse_rank)

        return sorted(by_chunk.values(), key=lambda item: item.rrf_score, reverse=True)

    def _merge_ranked_hits(
        self,
        hit_lists: list[list[RetrievalHit]],
        *,
        modality: str,
    ) -> list[RetrievalHit]:
        merged: dict[str, RetrievalHit] = {}
        for hits in hit_lists:
            for hit in hits:
                existing = merged.get(hit.chunk_id)
                if existing is None:
                    merged[hit.chunk_id] = RetrievalHit(
                        chunk_id=hit.chunk_id,
                        document_id=hit.document_id,
                        version_id=hit.version_id,
                        raw_text=hit.raw_text,
                        contextualized_text=hit.contextualized_text,
                        retrieval_context=hit.retrieval_context,
                        payload=hit.payload,
                        dense_rank=hit.dense_rank,
                        sparse_rank=hit.sparse_rank,
                        dense_score=hit.dense_score,
                        sparse_score=hit.sparse_score,
                    )
                    continue

                if modality == "dense":
                    if hit.dense_rank is not None and (
                        existing.dense_rank is None or hit.dense_rank < existing.dense_rank
                    ):
                        existing.dense_rank = hit.dense_rank
                        existing.dense_score = hit.dense_score
                else:
                    if hit.sparse_rank is not None and (
                        existing.sparse_rank is None or hit.sparse_rank < existing.sparse_rank
                    ):
                        existing.sparse_rank = hit.sparse_rank
                        existing.sparse_score = hit.sparse_score

        key = (lambda item: item.dense_rank or math.inf) if modality == "dense" else (
            lambda item: item.sparse_rank or math.inf
        )
        merged_hits = sorted(merged.values(), key=key)
        for rank, hit in enumerate(merged_hits, start=1):
            if modality == "dense":
                hit.dense_rank = rank
            else:
                hit.sparse_rank = rank
        return merged_hits

    def _rerank_hits(
        self,
        question: str,
        hits: list[RetrievalHit],
        *,
        evidence_intent: EvidenceIntentClassification,
    ) -> tuple[list[RetrievalHit], list[str]]:
        heuristic_hits = self._heuristic_rerank_hits(
            question,
            hits,
            evidence_intent=evidence_intent,
        )
        candidate_limit = max(1, int(self.settings.rerank_top_n))
        candidates = heuristic_hits[:candidate_limit]
        outcome = self.reranker.rerank(
            question=question,
            documents=[
                RerankDocument(
                    key=hit.chunk_id,
                    text=self._rerank_text(hit),
                )
                for hit in candidates
            ],
        )

        if outcome.source == "heuristic":
            trace = [f"rerank:provider=heuristic"]
            if outcome.reason:
                trace.append(f"rerank:reason={outcome.reason}")
            return heuristic_hits, trace

        reranked_candidates = self._apply_provider_rerank(
            candidates,
            outcome,
            evidence_intent=evidence_intent,
        )
        reranked_ids = {hit.chunk_id for hit in reranked_candidates}
        remainder = [hit for hit in heuristic_hits if hit.chunk_id not in reranked_ids]
        trace = [
            f"rerank:provider={outcome.source}",
            f"rerank:candidates={len(candidates)}",
            "rerank:policy=provider_led",
        ]
        return reranked_candidates + remainder, trace

    def _heuristic_rerank_hits(
        self,
        question: str,
        hits: list[RetrievalHit],
        *,
        evidence_intent: EvidenceIntentClassification,
    ) -> list[RetrievalHit]:
        query_tokens = self._tokens(question)
        structural_refs = self._extract_structural_references(question)
        for hit in hits:
            raw_tokens = self._tokens(hit.raw_text)
            path_tokens = self._tokens(
                " ".join(self._as_path(hit.payload.get("hierarchy_path") or hit.payload.get("section_path")))
            )
            lexical_overlap = max(
                self._overlap_ratio(query_tokens, raw_tokens),
                self._soft_overlap_ratio(query_tokens, raw_tokens),
            )
            path_overlap = max(
                self._overlap_ratio(query_tokens, path_tokens),
                self._soft_overlap_ratio(query_tokens, path_tokens),
            )
            exact_boost = 0.2 if self._contains_exact_reference(question, hit) else 0.0
            structural_boost = self._structural_reference_score(structural_refs, hit)
            structural_penalty = self._structural_reference_mismatch_penalty(structural_refs, hit)
            role_alignment = self._evidence_role_alignment_score(evidence_intent, hit)
            boilerplate_penalty = self._boilerplate_score(hit)
            hit.rerank_score = (
                hit.rrf_score
                + (0.8 * lexical_overlap)
                + (0.3 * path_overlap)
                + exact_boost
                + structural_boost
                - structural_penalty
                + (0.55 * role_alignment)
                - (0.85 * boilerplate_penalty)
            )
        return sorted(hits, key=lambda item: item.rerank_score, reverse=True)

    def _apply_provider_rerank(
        self,
        candidates: list[RetrievalHit],
        outcome,
        *,
        evidence_intent: EvidenceIntentClassification,
    ) -> list[RetrievalHit]:
        if not candidates:
            return []

        hit_map = {hit.chunk_id: hit for hit in candidates}
        normalized_scores = self._normalize_provider_scores(outcome.scores, [hit.chunk_id for hit in candidates])
        reranked: list[RetrievalHit] = []
        seen: set[str] = set()
        for rank, chunk_id in enumerate(outcome.ordered_keys, start=1):
            hit = hit_map.get(chunk_id)
            if hit is None or chunk_id in seen:
                continue
            provider_boost = normalized_scores.get(chunk_id, 0.0)
            raw_score = outcome.scores.get(chunk_id)
            if raw_score is not None:
                hit.provider_rerank_score = float(raw_score)
            positional_boost = 1.0 - ((rank - 1) / max(1, len(candidates)))
            role_alignment = self._evidence_role_alignment_score(evidence_intent, hit)
            hit.rerank_score = hit.rerank_score + provider_boost + (0.25 * positional_boost) + (0.15 * role_alignment)
            reranked.append(hit)
            seen.add(chunk_id)

        tail = [hit_map[chunk_id] for chunk_id in hit_map if chunk_id not in seen]
        return sorted(reranked + tail, key=lambda item: item.rerank_score, reverse=True)

    def _apply_visibility_policy(
        self,
        *,
        question: str,
        route: QueryRoute,
        hits: list[RetrievalHit],
        branch_hits: list[RetrievalHit],
        plan: QueryPlan,
        evidence_intent: EvidenceIntentClassification,
        structure: QueryStructure,
    ) -> tuple[list[RetrievalHit], list[str]]:
        if not hits:
            return hits, ["visibility:promoted=0"]
        if self._is_comparison_query(question):
            return hits, ["visibility:promoted=0"]

        selected_branch_keys, _ = self._select_branch_keys(
            question,
            branch_hits,
            plan,
            evidence_intent=evidence_intent,
        )
        selected_branch_keys = set(selected_branch_keys)
        visible_window = min(4 if plan.strategy != "focal" else 3, len(hits))
        curated_front = self._curate_hits_for_coverage(
            question=question,
            hits=hits,
            structure=structure,
            plan=plan,
            evidence_intent=evidence_intent,
            limit=visible_window,
            selected_branch_keys=selected_branch_keys,
            prefer_same_branch=plan.strategy == "focal",
            prefer_diversity=plan.diversity_required,
        )
        if not curated_front:
            return hits, ["visibility:promoted=0"]

        selected_chunks = {hit.chunk_id for hit in curated_front}
        ordered_hits = curated_front + [hit for hit in hits if hit.chunk_id not in selected_chunks]
        if plan.strategy == "focal":
            ordered_hits, _ = self._enforce_same_branch_visibility(
                hits=ordered_hits,
                visible_window=visible_window,
                selected_branch_keys=selected_branch_keys,
            )

        promoted = sum(
            1
            for index, hit in enumerate(ordered_hits[:visible_window])
            if index >= len(hits) or hit.chunk_id != hits[index].chunk_id
        )
        return ordered_hits, [f"visibility:promoted={promoted}", f"visibility:window={visible_window}"]

    def _attach_relevant_segments(
        self,
        *,
        request: QueryRequest,
        question: str,
        hits: list[RetrievalHit],
        structure: QueryStructure,
        plan: QueryPlan,
    ) -> tuple[list[RetrievalHit], list[str]]:
        if not hits:
            return hits, ["segments:attached=0"]

        candidate_limit = min(len(hits), 12 if plan.strategy != "global" else 16)
        aspects = self._query_aspects(question=question, structure=structure, plan=plan)
        neighbor_cache: dict[tuple[str, str, str, int, int], list[RetrievalHit]] = {}
        known_hits = {hit.chunk_id: hit for hit in hits}
        attached = 0

        for hit in hits[:candidate_limit]:
            segment = self._build_evidence_segment(
                request=request,
                question=question,
                hit=hit,
                aspects=aspects,
                plan=plan,
                neighbor_cache=neighbor_cache,
            )
            if segment is None:
                continue
            for member in segment.hits:
                target = known_hits.get(member.chunk_id)
                if target is None:
                    continue
                current_score = self._payload_float(target, "segment_score")
                if current_score >= segment.score:
                    continue
                self._annotate_hit_with_segment(target, segment)
                attached += 1
        return hits, [f"segments:attached={attached}", f"segments:aspects={len(aspects)}"]

    def _build_evidence_segment(
        self,
        *,
        request: QueryRequest,
        question: str,
        hit: RetrievalHit,
        aspects: list[str],
        plan: QueryPlan,
        neighbor_cache: dict[tuple[str, str, str, int, int], list[RetrievalHit]],
    ) -> EvidenceSegment | None:
        payload = hit.payload or {}
        chunk_order = int(payload.get("chunk_order") or 0)
        branch_key = self._branch_key_for_hit(hit)
        if not branch_key or not chunk_order:
            return None

        radius = 1 if plan.strategy == "focal" else 2
        cache_key = (
            hit.document_id,
            hit.version_id,
            branch_key,
            max(0, chunk_order - radius),
            chunk_order + radius,
        )
        neighbor_hits = neighbor_cache.get(cache_key)
        if neighbor_hits is None:
            neighbor_hits = self._segment_neighbor_hits(
                request=request,
                hit=hit,
                branch_key=branch_key,
                start_order=cache_key[3],
                end_order=cache_key[4],
            )
            neighbor_cache[cache_key] = neighbor_hits

        ordered_members: list[RetrievalHit] = []
        seen_chunks: set[str] = set()
        for member in sorted([hit, *neighbor_hits], key=lambda item: int(item.payload.get("chunk_order") or 0)):
            if member.chunk_id in seen_chunks:
                continue
            seen_chunks.add(member.chunk_id)
            ordered_members.append(member)

        if not ordered_members:
            return None

        segment_text = "\n".join(
            text
            for text in (
                " ".join((member.raw_text or "").split())
                for member in ordered_members
            )
            if text
        ).strip()
        if not segment_text:
            return None

        aspect_matches = self._matching_aspects(aspects, ordered_members[0], text_override=segment_text)
        coverage_score = self._coverage_score_from_matches(aspect_matches, aspects)
        label_score = self._label_expressiveness_score(hit)
        structural_refs = self._extract_structural_references(question)
        structural_score = self._structural_reference_score(structural_refs, hit)
        boilerplate_penalty = max(self._boilerplate_score(member) for member in ordered_members)
        segment_score = (
            hit.rerank_score
            + (1.1 * coverage_score)
            + (0.45 * label_score)
            + (0.35 * structural_score)
            + min(0.25, 0.08 * (len(ordered_members) - 1))
            - (0.9 * boilerplate_penalty)
        )
        start_order = int(ordered_members[0].payload.get("chunk_order") or chunk_order)
        end_order = int(ordered_members[-1].payload.get("chunk_order") or chunk_order)
        return EvidenceSegment(
            segment_id=f"{hit.document_id}:{hit.version_id}:{branch_key}:{start_order}:{end_order}",
            branch_key=branch_key,
            hits=ordered_members,
            text=segment_text,
            score=round(segment_score, 4),
            coverage_score=round(coverage_score, 4),
            label_score=round(label_score, 4),
            aspect_matches=aspect_matches,
        )

    def _segment_neighbor_hits(
        self,
        *,
        request: QueryRequest,
        hit: RetrievalHit,
        branch_key: str,
        start_order: int,
        end_order: int,
    ) -> list[RetrievalHit]:
        try:
            query_body = {
                "size": max(1, end_order - start_order + 1),
                "sort": [{"chunk_order": {"order": "asc"}}],
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"document_id": hit.document_id}},
                            {"term": {"version_id": hit.version_id}},
                            {"term": {"branch_keys": branch_key}},
                            {"range": {"chunk_order": {"gte": start_order, "lte": end_order}}},
                        ],
                        "filter": [
                            {"term": {"version_status": "active"}},
                            *([acl] if (acl := self._acl_opensearch_clause(request)) else []),
                        ],
                    }
                },
            }
            payload = self._opensearch_search(query_body)
            return self._parse_opensearch_hits(payload)
        except Exception:
            return []

    def _annotate_hit_with_segment(self, hit: RetrievalHit, segment: EvidenceSegment) -> None:
        payload = hit.payload or {}
        payload["segment_text"] = segment.text
        payload["segment_member_chunk_ids"] = [member.chunk_id for member in segment.hits]
        payload["segment_signature"] = segment.segment_id
        payload["segment_score"] = segment.score
        payload["segment_coverage_score"] = segment.coverage_score
        payload["segment_label_score"] = segment.label_score
        payload["segment_aspect_matches"] = segment.aspect_matches
        payload["segment_branch_key"] = segment.branch_key
        hit.payload = payload

    def _query_aspects(
        self,
        *,
        question: str,
        structure: QueryStructure,
        plan: QueryPlan,
    ) -> list[str]:
        aspects: list[str] = []

        def add(value: str) -> None:
            cleaned = " ".join((value or "").split()).strip()
            normalized = self._normalize(cleaned)
            if cleaned and normalized and cleaned not in aspects:
                aspects.append(cleaned)

        for reference in structure.references:
            add(reference.label)
        for marker in structure.document_markers:
            add(marker)
        for topic in [*structure.topic_terms, *plan.topics]:
            add(topic)

        normalized_question = self._normalize(question)
        for phrase, canonical in BOOSTABLE_PHRASES.items():
            if phrase in normalized_question:
                add(canonical)
        return aspects[:10]

    def _matching_aspects(
        self,
        aspects: list[str],
        hit: RetrievalHit,
        *,
        text_override: str | None = None,
    ) -> list[str]:
        payload = hit.payload or {}
        haystack = self._normalize(
            " ".join(
                [
                    str(payload.get("canonical_label") or ""),
                    str(payload.get("path_text") or ""),
                    str(payload.get("unit_topic") or ""),
                    str(payload.get("branch_summary_context") or ""),
                    str(payload.get("focus_terms_text") or ""),
                    text_override or str(payload.get("segment_text") or hit.raw_text or ""),
                ]
            )
        )
        haystack_tokens = self._tokens(haystack)
        matched: list[str] = []
        for aspect in aspects:
            normalized_aspect = self._normalize(aspect)
            if not normalized_aspect:
                continue
            if normalized_aspect in haystack:
                matched.append(normalized_aspect)
                continue
            aspect_tokens = self._tokens(normalized_aspect)
            if not aspect_tokens:
                continue
            overlap = max(
                self._overlap_ratio(aspect_tokens, haystack_tokens),
                self._soft_overlap_ratio(aspect_tokens, haystack_tokens),
            )
            if overlap >= 0.6:
                matched.append(normalized_aspect)
        return self._dedupe_strings(matched)

    def _coverage_score_from_matches(self, matches: list[str], aspects: list[str]) -> float:
        if not aspects:
            return 0.0
        return min(1.0, len(matches) / max(1, len(aspects)))

    def _curate_hits_for_coverage(
        self,
        *,
        question: str,
        hits: list[RetrievalHit],
        structure: QueryStructure,
        plan: QueryPlan,
        evidence_intent: EvidenceIntentClassification,
        limit: int,
        selected_branch_keys: set[str],
        prefer_same_branch: bool,
        prefer_diversity: bool,
    ) -> list[RetrievalHit]:
        if not hits or limit <= 0:
            return []

        aspects = self._query_aspects(question=question, structure=structure, plan=plan)
        pool = list(hits[: max(limit + 10, min(len(hits), 16))])
        selected: list[RetrievalHit] = []
        covered_aspects: set[str] = set()
        seen_branches: set[str] = set()
        seen_segments: set[str] = set()

        while pool and len(selected) < limit:
            best_index: int | None = None
            best_score = float("-inf")
            for index, candidate in enumerate(pool):
                branch_key = self._branch_key_for_hit(candidate)
                segment_signature = str(candidate.payload.get("segment_signature") or candidate.chunk_id)
                match_set = set(self._matching_aspects(aspects, candidate))
                gain = len(match_set - covered_aspects)
                score = candidate.rerank_score
                score += 0.4 * self._payload_float(candidate, "segment_score")
                score += 0.55 * self._answer_focus_score(question, candidate)
                score += 0.5 * self._evidence_role_alignment_score(evidence_intent, candidate)
                score += 0.55 * gain
                score += 0.25 * self._payload_float(candidate, "segment_coverage_score")
                score += 0.2 * self._label_expressiveness_score(candidate)
                score -= 0.95 * self._boilerplate_score(candidate)

                if selected_branch_keys and branch_key in selected_branch_keys:
                    score += 0.45
                elif selected_branch_keys:
                    score -= 0.15

                if prefer_same_branch and selected:
                    primary_branch = self._branch_key_for_hit(selected[0])
                    if primary_branch and branch_key == primary_branch:
                        score += 0.35

                if prefer_diversity:
                    score += 0.25 if branch_key and branch_key not in seen_branches else -0.08

                if segment_signature in seen_segments:
                    score -= 0.6
                if score > best_score:
                    best_score = score
                    best_index = index

            if best_index is None:
                break

            chosen = pool.pop(best_index)
            selected.append(chosen)
            covered_aspects.update(self._matching_aspects(aspects, chosen))
            seen_branches.add(self._branch_key_for_hit(chosen))
            seen_segments.add(str(chosen.payload.get("segment_signature") or chosen.chunk_id))

        return selected

    def _normalize_provider_scores(
        self,
        scores: dict[str, float],
        keys: list[str],
    ) -> dict[str, float]:
        values = [scores[key] for key in keys if key in scores]
        if not values:
            return {}
        minimum = min(values)
        maximum = max(values)
        if maximum <= minimum:
            return {key: 1.0 for key in keys if key in scores}
        scale = maximum - minimum
        return {
            key: (scores[key] - minimum) / scale
            for key in keys
            if key in scores
        }

    def _rerank_text(self, hit: RetrievalHit) -> str:
        payload = hit.payload or {}
        lines = [
            f"document: {payload.get('document_title') or ''}",
            f"path: {payload.get('path_text') or ''}",
            f"canonical_label: {payload.get('canonical_label') or ''}",
            f"unit_topic: {payload.get('unit_topic') or ''}",
            f"unit_type: {payload.get('unit_type') or ''}",
            f"unit_number: {payload.get('unit_number') or ''}",
            f"unit_key: {payload.get('unit_key') or ''}",
            f"reference_variants: {payload.get('reference_variants_text') or ''}",
            f"focus_terms: {payload.get('focus_terms_text') or ''}",
            f"branch_keywords: {payload.get('branch_keywords_text') or ''}",
            f"context: {hit.retrieval_context or ''}",
            f"chunk_summary_context: {payload.get('chunk_summary_context') or ''}",
            f"branch_summary_context: {payload.get('branch_summary_context') or ''}",
            f"segment_text: {payload.get('segment_text') or ''}",
            f"content: {hit.raw_text or ''}",
        ]
        structured = "\n".join(line for line in lines if line.strip())
        if structured:
            return structured
        if hit.contextualized_text:
            return hit.contextualized_text
        if hit.retrieval_context:
            return f"{hit.retrieval_context}\n{hit.raw_text}"
        return hit.raw_text

    def _enforce_same_branch_visibility(
        self,
        *,
        hits: list[RetrievalHit],
        visible_window: int,
        selected_branch_keys: set[str],
    ) -> tuple[list[RetrievalHit], int]:
        if not hits or visible_window <= 1:
            return hits, 0

        primary_branch = self._branch_key_for_hit(hits[0])
        if not primary_branch or primary_branch not in selected_branch_keys:
            return hits, 0

        ordered_hits = list(hits)
        promoted = 0
        for position in range(1, visible_window):
            current_branch = self._branch_key_for_hit(ordered_hits[position])
            if current_branch == primary_branch:
                continue
            replacement_index = self._find_same_branch_replacement(
                hits=ordered_hits,
                start_index=position + 1,
                primary_branch=primary_branch,
            )
            if replacement_index is None:
                continue
            replacement = ordered_hits.pop(replacement_index)
            ordered_hits.insert(position, replacement)
            promoted += 1

        return ordered_hits, promoted

    def _find_same_branch_replacement(
        self,
        *,
        hits: list[RetrievalHit],
        start_index: int,
        primary_branch: str,
    ) -> int | None:
        best_index: int | None = None
        best_score = float("-inf")
        for index in range(start_index, len(hits)):
            candidate = hits[index]
            if self._branch_key_for_hit(candidate) != primary_branch:
                continue
            score = candidate.rerank_score
            if candidate.payload.get("is_branch_lead"):
                score += 1.0
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    def _branch_key_for_hit(self, hit: RetrievalHit) -> str:
        payload = hit.payload or {}
        direct = str(payload.get("branch_key") or "").strip()
        if direct:
            return direct
        for field_name in ("branch_lead_keys", "branch_keys"):
            branch_keys = payload.get(field_name) or []
            if isinstance(branch_keys, list):
                for value in reversed(branch_keys):
                    normalized = str(value or "").strip()
                    if normalized:
                        return normalized
        return ""

    def _parent_branch_key_for_hit(self, hit: RetrievalHit) -> str:
        payload = hit.payload or {}
        direct = str(payload.get("parent_branch_key") or "").strip()
        if direct:
            return direct

        branch_keys = [
            str(value or "").strip()
            for value in (payload.get("branch_keys") or [])
            if str(value or "").strip()
        ]
        if len(branch_keys) < 2:
            return ""

        lead_key = self._branch_key_for_hit(hit)
        if lead_key and lead_key in branch_keys:
            lead_index = branch_keys.index(lead_key)
            if lead_index > 0:
                return branch_keys[lead_index - 1]
        return branch_keys[-2]

    def _diversify_by_branch(
        self,
        hits: list[RetrievalHit],
        *,
        visible_window: int,
    ) -> list[RetrievalHit]:
        if not hits or visible_window <= 1:
            return hits

        diversified: list[RetrievalHit] = []
        seen_chunks: set[str] = set()
        seen_branches: set[str] = set()

        for hit in hits:
            branch_key = self._branch_key_for_hit(hit)
            if branch_key and branch_key not in seen_branches:
                diversified.append(hit)
                seen_chunks.add(hit.chunk_id)
                seen_branches.add(branch_key)
            if len(diversified) >= visible_window:
                break

        for hit in hits:
            if hit.chunk_id in seen_chunks:
                continue
            diversified.append(hit)
            seen_chunks.add(hit.chunk_id)
        return diversified

    def _unique_branch_hits(self, hits: list[RetrievalHit], *, limit: int) -> list[RetrievalHit]:
        unique_hits: list[RetrievalHit] = []
        seen_branches: set[str] = set()
        seen_chunks: set[str] = set()
        for hit in hits:
            branch_key = self._branch_key_for_hit(hit) or hit.chunk_id
            if branch_key in seen_branches or hit.chunk_id in seen_chunks:
                continue
            unique_hits.append(hit)
            seen_branches.add(branch_key)
            seen_chunks.add(hit.chunk_id)
            if len(unique_hits) >= limit:
                break
        return unique_hits or hits[:limit]

    def _display_branch_label(self, hit: RetrievalHit) -> str:
        payload = hit.payload or {}
        label = str(payload.get("canonical_label") or "").strip()
        if label:
            return label
        path = self._as_path(payload.get("hierarchy_path") or payload.get("section_path"))
        if path:
            return " > ".join(path[:2])
        return "Rama relevante"

    def _candidate_parent_path(self, path_text: str) -> str:
        parts = [part.strip() for part in str(path_text or "").split(">") if part.strip()]
        if len(parts) <= 1:
            return ""
        return " > ".join(parts[:-1])

    def _normalize_answer_form(self, *, answer_form: str, plan: QueryPlan) -> str:
        if answer_form == "unknown":
            return answer_form
        if plan.strategy == "global":
            return "global_summary"
        if plan.strategy == "focal" and answer_form == "unit_set":
            return "rule_summary"
        if answer_form == "unit_set" and not self.settings.answer_form_unit_set_enabled:
            return "unknown"
        return answer_form

    def _diversify_hits(
        self,
        question: str,
        route: QueryRoute,
        hits: list[RetrievalHit],
        plan: QueryPlan,
    ) -> list[RetrievalHit]:
        if not hits:
            return hits

        if self._is_comparison_query(question):
            if "technical_standard" not in route.target_classes and route.query_type != "technical_standard":
                return hits

            clauses = self._comparison_clause_queries(question, route)
            normalized_question = self._normalize(question)
            prefer_distinct_documents = (
                normalized_question.startswith("que documento")
                or normalized_question.startswith("que norma")
                or " y cual " in normalized_question
            )
            prioritized: list[RetrievalHit] = []
            prioritized_chunks: set[str] = set()
            prioritized_docs: set[str] = set()

            for clause in clauses:
                preferred = None
                for hit in hits:
                    if hit.chunk_id in prioritized_chunks:
                        continue
                    if prefer_distinct_documents and hit.document_id in prioritized_docs:
                        continue
                    if self._comparison_clause_matches(clause, hit):
                        preferred = hit
                        break
                if preferred is None:
                    for hit in hits:
                        if hit.chunk_id in prioritized_chunks:
                            continue
                        if self._comparison_clause_matches(clause, hit):
                            preferred = hit
                            break
                if preferred is None:
                    continue
                prioritized.append(preferred)
                prioritized_chunks.add(preferred.chunk_id)
                prioritized_docs.add(preferred.document_id)

            if prefer_distinct_documents:
                selected: list[RetrievalHit] = []
                seen_chunks: set[str] = set()
                seen_documents: set[str] = set()

                for hit in prioritized + hits:
                    if hit.document_id in seen_documents:
                        continue
                    selected.append(hit)
                    seen_chunks.add(hit.chunk_id)
                    seen_documents.add(hit.document_id)

                for hit in prioritized + hits:
                    if hit.chunk_id in seen_chunks:
                        continue
                    selected.append(hit)
                    seen_chunks.add(hit.chunk_id)

                return selected

            selected = list(prioritized)
            seen_chunks = {hit.chunk_id for hit in prioritized}
            for hit in hits:
                if hit.chunk_id in seen_chunks:
                    continue
                selected.append(hit)
                seen_chunks.add(hit.chunk_id)
            return selected

        if not plan.diversity_required:
            return hits
        return self._diversify_by_branch(hits, visible_window=min(3, len(hits)))

    def _comparison_clause_matches(self, clause: str, hit: RetrievalHit) -> bool:
        clause_norm = self._normalize(clause)
        haystack = self._normalize(
            " ".join(
                [
                    " ".join(self._as_path(hit.payload.get("hierarchy_path") or hit.payload.get("section_path"))),
                    hit.retrieval_context,
                    hit.raw_text[:900],
                ]
            )
        )

        clause_tokens = self._tokens(clause)
        haystack_tokens = self._tokens(haystack)
        lexical = max(
            self._overlap_ratio(clause_tokens, haystack_tokens),
            self._soft_overlap_ratio(clause_tokens, haystack_tokens),
        )
        return lexical >= 0.45

    def _compose_answer(
        self,
        question: str,
        evidence_items: list[AnswerSynthesisEvidence],
        plan: QueryPlan,
        *,
        answer_form: str,
        answer_style: str,
        evidence_intent: EvidenceIntentClassification,
    ) -> tuple[str, list[str], list[RetrievalHit]]:
        generated = self.answer_synthesizer.synthesize(
            question=question,
            strategy=plan.strategy,
            answer_style=answer_style,
            evidence_items=evidence_items,
            evidence_intent=evidence_intent.intent,
        )
        if generated is not None:
            return generated.answer, generated.decision_trace, []

        return (
            self._compose_extractive_from_evidence_items(
                evidence_items,
                answer_style,
                answer_form=answer_form,
            ),
            ["answer:provider=extractive_fallback"],
            [],
        )

    def _build_answer_package(
        self,
        *,
        request: QueryRequest,
        hits: list[RetrievalHit],
        plan: QueryPlan,
        evidence_intent: EvidenceIntentClassification,
        structure: QueryStructure,
    ) -> AnswerPackage:
        block_package = self._build_answer_package_with_block_competition(
            request=request,
            hits=hits,
            plan=plan,
            evidence_intent=evidence_intent,
            structure=structure,
        )
        if block_package is not None:
            return block_package

        question = request.question
        candidate_hits = self._select_answer_hits(
            question,
            hits,
            plan,
            evidence_intent=evidence_intent,
            answer_form="unknown",
            structure=structure,
        )
        decision_trace = [
            "answer_form:provider=disabled",
            f"answer_candidates={len(candidate_hits)}",
        ]
        evidence_items = self._answer_evidence_items(
            question,
            candidate_hits,
        )
        return AnswerPackage(
            hits=candidate_hits,
            evidence_items=evidence_items,
            label_to_hit={
                item.label: hit for item, hit in zip(evidence_items, candidate_hits, strict=False)
            },
            decision_trace=decision_trace,
            answer_form="unknown",
            answer_style=plan.answer_style,
        )

    def _build_answer_package_with_block_competition(
        self,
        *,
        request: QueryRequest,
        hits: list[RetrievalHit],
        plan: QueryPlan,
        evidence_intent: EvidenceIntentClassification,
        structure: QueryStructure,
    ) -> AnswerPackage | None:
        candidate_hits = self._candidate_hits_for_block_competition(hits, plan=plan)
        if not candidate_hits:
            return None

        unit_candidates = self._build_answer_unit_candidates(
            question=request.question,
            hits=candidate_hits,
            evidence_intent=evidence_intent,
        )
        if not unit_candidates:
            return None

        block_candidates = self._build_answer_block_candidates(unit_candidates)
        if not block_candidates:
            return None

        ordered_blocks, block_trace = self._rerank_answer_block_candidates(
            question=request.question,
            blocks=block_candidates,
        )
        selected_hits, selection_roles, selection_meta, selection_trace = self._select_hits_via_block_competition(
            question=request.question,
            plan=plan,
            structure=structure,
            ordered_blocks=ordered_blocks,
            evidence_intent=evidence_intent,
        )
        if not selected_hits:
            return None

        answer_form = "direct_unit"
        answer_style = "single_branch"
        if selection_meta["block_count"] > 1:
            answer_form = "global_summary" if plan.strategy == "global" else "unit_set"
            answer_style = (
                plan.answer_style
                if plan.answer_style in {"grouped_by_branch", "summary"}
                else "grouped_by_branch"
            )
        elif selection_meta["unit_count"] > 1:
            answer_form = "unit_set"
            answer_style = "single_branch"

        evidence_items = self._answer_evidence_items_for_block_selection(
            question=request.question,
            hits=selected_hits,
            selection_roles=selection_roles,
        )
        decision_trace = [
            "answer_form:provider=block_competition",
            f"answer_form:value={answer_form}",
            f"answer_blocks:candidates={len(block_candidates)}",
            f"answer_blocks:selected={selection_meta['block_count']}",
            f"answer_units:selected={selection_meta['unit_count']}",
            f"answer_candidates={len(selected_hits)}",
            *block_trace,
            *selection_trace,
        ]
        return AnswerPackage(
            hits=selected_hits,
            evidence_items=evidence_items,
            label_to_hit={
                item.label: hit for item, hit in zip(evidence_items, selected_hits, strict=False)
            },
            decision_trace=decision_trace,
            answer_form=answer_form,
            answer_style=answer_style,
        )

    def _candidate_hits_for_block_competition(
        self,
        hits: list[RetrievalHit],
        *,
        plan: QueryPlan,
    ) -> list[RetrievalHit]:
        if not hits:
            return []
        limit = 32 if plan.strategy == "global" else 28
        candidate_hits: list[RetrievalHit] = []
        seen_chunks: set[str] = set()
        for hit in hits:
            if hit.chunk_id in seen_chunks:
                continue
            seen_chunks.add(hit.chunk_id)
            chunk_kind = str((hit.payload or {}).get("chunk_kind") or "").strip().lower()
            if chunk_kind == "document":
                continue
            candidate_hits.append(hit)
            if len(candidate_hits) >= limit:
                break
        return candidate_hits

    def _build_answer_unit_candidates(
        self,
        *,
        question: str,
        hits: list[RetrievalHit],
        evidence_intent: EvidenceIntentClassification,
    ) -> list[AnswerUnitCandidate]:
        grouped: dict[str, list[RetrievalHit]] = {}
        for hit in hits:
            unit_key = self._answer_unit_group_key(hit)
            if not unit_key:
                continue
            grouped.setdefault(unit_key, []).append(hit)

        unit_candidates: list[AnswerUnitCandidate] = []
        for unit_key, unit_hits in grouped.items():
            ordered_hits = sorted(
                unit_hits,
                key=lambda current: (
                    self._answer_focus_score(question, current),
                    self._evidence_role_alignment_score(evidence_intent, current),
                    current.rerank_score,
                ),
                reverse=True,
            )
            best_hit = ordered_hits[0]
            payload = best_hit.payload or {}
            block_label, block_key = self._answer_block_identity_for_hit(best_hit)
            if not block_key:
                continue
            focus_score = round(self._answer_focus_score(question, best_hit), 4)
            rerank_score = round(float(best_hit.rerank_score), 4)
            base_score = (
                rerank_score
                + (0.45 * focus_score)
                + self._evidence_role_alignment_score(evidence_intent, best_hit)
                + min(0.25, 0.05 * len(ordered_hits))
            )
            unit_candidates.append(
                AnswerUnitCandidate(
                    unit_key=unit_key,
                    canonical_label=str(
                        payload.get("canonical_label") or self._display_branch_label(best_hit)
                    ).strip(),
                    branch_label=self._display_branch_label(best_hit),
                    block_key=block_key,
                    block_label=block_label,
                    document_id=best_hit.document_id,
                    version_id=best_hit.version_id,
                    path_text=str(payload.get("path_text") or "").strip(),
                    unit_type=str(payload.get("unit_type") or "").strip(),
                    unit_number=str(payload.get("unit_number") or "").strip(),
                    evidence_role=str(payload.get("evidence_role") or "").strip(),
                    best_hit=best_hit,
                    hits=ordered_hits,
                    focus_score=focus_score,
                    rerank_score=rerank_score,
                    base_score=round(base_score, 4),
                    text=self._answer_unit_candidate_text(question=question, hits=ordered_hits),
                )
            )
        unit_candidates.sort(key=lambda current: current.base_score, reverse=True)
        return unit_candidates

    def _answer_block_identity_for_hit(self, hit: RetrievalHit) -> tuple[str, str]:
        payload = hit.payload or {}
        parent_path, _ = self._parent_path_spec_for_hit(hit)
        path_text = str(payload.get("path_text") or "").strip()
        canonical_label = str(payload.get("canonical_label") or "").strip()
        block_label = parent_path or path_text or canonical_label or self._display_branch_label(hit)
        block_key = f"{hit.document_id}:{hit.version_id}:{self._normalize(block_label)}"
        return block_label, block_key

    def _answer_unit_candidate_text(self, *, question: str, hits: list[RetrievalHit]) -> str:
        best_hit = hits[0]
        payload = best_hit.payload or {}
        snippets: list[str] = []
        for hit in hits[:2]:
            snippet = self._question_focused_evidence_text(question, hit, limit=220)
            if snippet:
                snippets.append(snippet)
        path_text = str(payload.get("path_text") or "").strip()
        document_title = str(payload.get("document_title") or "").strip()
        canonical_label = str(payload.get("canonical_label") or self._display_branch_label(best_hit)).strip()
        return "\n".join(
            part
            for part in [
                f"documento: {document_title}" if document_title else "",
                f"unidad: {canonical_label}" if canonical_label else "",
                f"ruta: {path_text}" if path_text else "",
                *[f"evidencia: {snippet}" for snippet in snippets],
            ]
            if part
        )

    def _build_answer_block_candidates(
        self,
        unit_candidates: list[AnswerUnitCandidate],
    ) -> list[AnswerBlockCandidate]:
        grouped: dict[str, list[AnswerUnitCandidate]] = {}
        for candidate in unit_candidates:
            grouped.setdefault(candidate.block_key, []).append(candidate)

        block_candidates: list[AnswerBlockCandidate] = []
        for block_key, members in grouped.items():
            ordered_members = sorted(members, key=lambda current: current.base_score, reverse=True)
            first = ordered_members[0]
            focus_score = round(max(member.focus_score for member in ordered_members), 4)
            rerank_score = round(max(member.rerank_score for member in ordered_members), 4)
            base_score = rerank_score + (0.45 * focus_score) + min(0.3, 0.06 * len(ordered_members))
            block_candidates.append(
                AnswerBlockCandidate(
                    block_key=block_key,
                    block_label=first.block_label,
                    document_id=first.document_id,
                    version_id=first.version_id,
                    path_text=first.block_label,
                    unit_candidates=ordered_members,
                    focus_score=focus_score,
                    rerank_score=rerank_score,
                    base_score=round(base_score, 4),
                    text=self._answer_block_candidate_text(ordered_members),
                )
            )
        block_candidates.sort(key=lambda current: current.base_score, reverse=True)
        return block_candidates

    def _answer_block_candidate_text(self, unit_candidates: list[AnswerUnitCandidate]) -> str:
        first = unit_candidates[0]
        top_units = " | ".join(candidate.canonical_label for candidate in unit_candidates[:4] if candidate.canonical_label)
        snippets = [
            f"{candidate.canonical_label}: {self._compress_text(candidate.text, 260)}"
            for candidate in unit_candidates[:3]
        ]
        return "\n".join(
            part
            for part in [
                f"bloque: {first.block_label}" if first.block_label else "",
                f"ruta: {first.path_text}" if first.path_text else "",
                f"unidades: {top_units}" if top_units else "",
                *snippets,
            ]
            if part
        )

    def _rerank_answer_block_candidates(
        self,
        *,
        question: str,
        blocks: list[AnswerBlockCandidate],
    ) -> tuple[list[AnswerBlockCandidate], list[str]]:
        if not blocks:
            return [], ["answer_blocks:provider=empty", "answer_blocks:ordered=0"]

        seeded = sorted(blocks, key=lambda current: current.base_score, reverse=True)
        documents = [
            RerankDocument(key=block.block_key, text=block.text)
            for block in seeded
        ]
        outcome = self.reranker.rerank(question=question, documents=documents)
        ordered = self._blend_rerank_with_base(
            candidates=seeded,
            key_fn=lambda candidate: candidate.block_key,
            base_score_fn=lambda candidate: candidate.base_score,
            provider_scores=outcome.scores,
        )
        if not ordered:
            ordered = seeded
        trace = [
            f"answer_blocks:provider={outcome.source}",
            f"answer_blocks:candidates={len(blocks)}",
            f"answer_blocks:ordered={len(ordered)}",
        ]
        if outcome.reason:
            trace.append(f"answer_blocks:reason={outcome.reason}")
        return ordered, trace

    def _rerank_answer_unit_candidates(
        self,
        *,
        question: str,
        unit_candidates: list[AnswerUnitCandidate],
    ) -> tuple[list[AnswerUnitCandidate], list[str]]:
        if not unit_candidates:
            return [], ["answer_units:provider=empty", "answer_units:ordered=0"]

        seeded = sorted(unit_candidates, key=lambda current: current.base_score, reverse=True)
        documents = [
            RerankDocument(key=candidate.unit_key, text=candidate.text)
            for candidate in seeded
        ]
        outcome = self.reranker.rerank(question=question, documents=documents)
        ordered = self._blend_rerank_with_base(
            candidates=seeded,
            key_fn=lambda candidate: candidate.unit_key,
            base_score_fn=lambda candidate: candidate.base_score,
            provider_scores=outcome.scores,
        )
        if not ordered:
            ordered = seeded
        trace = [
            f"answer_units:provider={outcome.source}",
            f"answer_units:candidates={len(unit_candidates)}",
            f"answer_units:ordered={len(ordered)}",
        ]
        if outcome.reason:
            trace.append(f"answer_units:reason={outcome.reason}")
        return ordered, trace

    def _blend_rerank_with_base(
        self,
        *,
        candidates,
        key_fn,
        base_score_fn,
        provider_scores: dict[str, float],
    ):
        if not candidates:
            return []

        max_base = max(float(base_score_fn(candidate)) for candidate in candidates) or 1.0
        provider_values = [float(provider_scores.get(key_fn(candidate), 0.0)) for candidate in candidates]
        max_provider = max(provider_values) if any(provider_values) else 1.0

        return sorted(
            candidates,
            key=lambda candidate: (
                0.72 * (float(base_score_fn(candidate)) / max_base)
                + 0.28 * (float(provider_scores.get(key_fn(candidate), 0.0)) / max_provider),
                float(base_score_fn(candidate)),
            ),
            reverse=True,
        )

    def _select_hits_via_block_competition(
        self,
        *,
        question: str,
        plan: QueryPlan,
        structure: QueryStructure,
        ordered_blocks: list[AnswerBlockCandidate],
        evidence_intent: EvidenceIntentClassification,
    ) -> tuple[list[RetrievalHit], dict[str, str], dict[str, int], list[str]]:
        del structure
        if not ordered_blocks:
            return [], {}, {"block_count": 0, "unit_count": 0}, ["answer_select:blocks=0"]

        block_limit = {"focal": 1, "multi_branch": 2, "global": 3}.get(plan.strategy, 1)
        selected_blocks = ordered_blocks[:block_limit]
        selected_hits: list[RetrievalHit] = []
        selection_roles: dict[str, str] = {}
        seen_chunks: set[str] = set()
        selected_unit_count = 0
        trace = [f"answer_select:blocks={len(selected_blocks)}"]

        for block_index, block in enumerate(selected_blocks):
            ordered_units, unit_trace = self._rerank_answer_unit_candidates(
                question=question,
                unit_candidates=block.unit_candidates,
            )
            trace.extend(unit_trace)
            if not ordered_units:
                continue

            if plan.strategy == "focal":
                unit_limit = 2
            elif plan.strategy == "multi_branch":
                unit_limit = 1 if block_index > 0 else 2
            else:
                unit_limit = 1

            selected_units = ordered_units[:unit_limit]
            selected_unit_count += len(selected_units)
            for unit_index, unit in enumerate(selected_units):
                hit = unit.best_hit
                if hit.chunk_id in seen_chunks:
                    continue
                seen_chunks.add(hit.chunk_id)
                selected_hits.append(hit)
                if block_index == 0 and unit_index == 0:
                    selection_roles[hit.chunk_id] = "primary"
                elif plan.strategy == "focal":
                    selection_roles[hit.chunk_id] = "support"
                elif plan.strategy == "multi_branch" and block_index == 0:
                    selection_roles[hit.chunk_id] = "support"
                else:
                    selection_roles[hit.chunk_id] = "context"
                payload = hit.payload or {}
                payload["answer_block_label"] = block.block_label

        if not selected_hits and ordered_blocks:
            fallback = ordered_blocks[0].unit_candidates[0].best_hit
            selected_hits = [fallback]
            selection_roles[fallback.chunk_id] = "primary"
            selected_unit_count = 1

        if selected_hits:
            trace.append(f"answer_select:units={selected_unit_count}")
            trace.append(f"answer_select:hits={len(selected_hits)}")
            trace.append(
                "answer_select:intent="
                + str(evidence_intent.intent or "mixed")
            )
        return selected_hits, selection_roles, {"block_count": len(selected_blocks), "unit_count": selected_unit_count}, trace

    def _answer_evidence_items_for_block_selection(
        self,
        *,
        question: str,
        hits: list[RetrievalHit],
        selection_roles: dict[str, str],
    ) -> list[AnswerSynthesisEvidence]:
        evidence_items: list[AnswerSynthesisEvidence] = []
        for index, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            block_label = str(payload.get("answer_block_label") or self._display_branch_label(hit)).strip()
            chunk_kind = str(payload.get("chunk_kind") or payload.get("unit_type") or "chunk").strip() or "chunk"
            evidence_items.append(
                AnswerSynthesisEvidence(
                    label=f"[{index}]",
                    canonical_label=str(
                        payload.get("canonical_label") or self._display_branch_label(hit)
                    ).strip(),
                    branch_label=block_label,
                    unit_type=str(payload.get("unit_type") or "").strip(),
                    unit_number=str(payload.get("unit_number") or "").strip(),
                    chunk_kind=chunk_kind,
                    evidence_role=str(payload.get("evidence_role") or "").strip(),
                    selection_role=selection_roles.get(hit.chunk_id, ""),
                    path_text=str(payload.get("path_text") or "").strip(),
                    text=self._question_focused_evidence_text(
                        question, hit, limit=int(self.settings.answer_evidence_char_limit)
                    ),
                    relevance=hit.provider_rerank_score,
                )
            )
        return evidence_items

    def _compose_extractive_answer(self, hits: list[RetrievalHit], plan: QueryPlan) -> str:
        if plan.answer_style in {"grouped_by_branch", "summary"}:
            grouped_hits = self._unique_branch_hits(hits, limit=3)
            snippets = []
            for index, hit in enumerate(grouped_hits, start=1):
                label = self._display_branch_label(hit)
                snippet = self._compress_text(hit.raw_text, 260)
                snippets.append(f"[{index}] {label}: {snippet}")
            heading = (
                "Ramas relevantes identificadas:\n"
                if plan.answer_style == "grouped_by_branch"
                else "Resumen de evidencia por ramas:\n"
            )
            return heading + "\n".join(snippets)

        snippets: list[str] = []
        for index, hit in enumerate(hits[:3], start=1):
            snippet = self._compress_text(hit.raw_text, 320)
            snippets.append(f"[{index}] {snippet}")
        return "Evidencia recuperada mas relevante:\n" + "\n".join(snippets)

    def _compose_extractive_from_evidence_items(
        self,
        evidence_items: list[AnswerSynthesisEvidence],
        answer_style: str,
        *,
        answer_form: str,
    ) -> str:
        if not evidence_items:
            return "No se encontro evidencia relevante en el indice para responder la consulta."
        if answer_form == "direct_unit":
            primary = next(
                (item for item in evidence_items if item.selection_role == "primary"),
                evidence_items[0],
            )
            answer = (
                f"La unidad que responde directamente la pregunta es {primary.canonical_label} {primary.label}."
            ).strip()
            supports = [
                item
                for item in evidence_items
                if item.label != primary.label and item.selection_role in {"support", "context"}
            ]
            if supports:
                support = supports[0]
                answer += f" Como apoyo, {self._compress_text(support.text, 200)} {support.label}".strip()
            return answer
        if answer_form == "unit_set":
            primaries = [
                item
                for item in evidence_items
                if item.selection_role in {"primary", "support"}
            ] or evidence_items[:3]
            labels = [item.canonical_label for item in primaries[:3] if item.canonical_label]
            if labels:
                answer = (
                    "Las unidades mas centrales para responder la pregunta son "
                    + ", ".join(labels[:-1])
                    + (" y " + labels[-1] if len(labels) > 1 else labels[0])
                    + "."
                )
            else:
                answer = self._compress_text(primaries[0].text, 320)
            if primaries:
                answer += " " + " ".join(
                    f"{item.label} {self._compress_text(item.text, 180)}"
                    for item in primaries[:2]
                )
            return answer.strip()
        if answer_style == "single_branch":
            primary = next(
                (item for item in evidence_items if item.selection_role == "primary"),
                evidence_items[0],
            )
            supports = [
                item
                for item in evidence_items
                if item.label != primary.label and item.selection_role in {"support", "context"}
            ]
            answer = f"{self._compress_text(primary.text, 360)} {primary.label}".strip()
            if supports:
                support = supports[0]
                answer += (
                    f" Como apoyo, {self._compress_text(support.text, 220)} {support.label}".strip()
                )
            return answer.strip()
        if answer_style in {"grouped_by_branch", "summary"}:
            snippets = [
                f"{item.label} {item.branch_label}: {self._compress_text(item.text, 260)}"
                for item in evidence_items[:3]
            ]
            heading = (
                "Ramas relevantes identificadas:\n"
                if answer_style == "grouped_by_branch"
                else "Resumen de evidencia por ramas:\n"
            )
            return heading + "\n".join(snippets)
        snippets = [
            f"{item.label} {self._compress_text(item.text, 320)}"
            for item in evidence_items[:3]
        ]
        return "Evidencia recuperada mas relevante:\n" + "\n".join(snippets)

    def _fallback_used_labels(
        self,
        evidence_items: list[AnswerSynthesisEvidence],
        *,
        answer_form: str,
        answer_style: str,
    ) -> list[str]:
        if not evidence_items:
            return []
        if answer_form == "direct_unit":
            primary = next(
                (item for item in evidence_items if item.selection_role == "primary"),
                evidence_items[0],
            )
            labels = [primary.label]
            support = next(
                (
                    item
                    for item in evidence_items
                    if item.label != primary.label and item.selection_role in {"support", "context"}
                ),
                None,
            )
            if support is not None:
                labels.append(support.label)
            return labels
        if answer_style in {"grouped_by_branch", "summary"}:
            return [item.label for item in evidence_items[:3]]
        return [item.label for item in evidence_items[:2]]

    def _answer_evidence_items(
        self,
        question: str,
        hits: list[RetrievalHit],
        *,
        selection_roles: dict[str, str] | None = None,
    ) -> list[AnswerSynthesisEvidence]:
        evidence_items: list[AnswerSynthesisEvidence] = []
        for index, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            chunk_kind = str(payload.get("chunk_kind") or payload.get("unit_type") or "chunk").strip() or "chunk"
            evidence_items.append(
                AnswerSynthesisEvidence(
                    label=f"[{index}]",
                    canonical_label=str(
                        payload.get("canonical_label") or self._display_branch_label(hit)
                    ).strip(),
                    branch_label=self._display_branch_label(hit),
                    unit_type=str(payload.get("unit_type") or "").strip(),
                    unit_number=str(payload.get("unit_number") or "").strip(),
                    chunk_kind=chunk_kind,
                    evidence_role=str(payload.get("evidence_role") or ""),
                    selection_role=(selection_roles or {}).get(hit.chunk_id, ""),
                    path_text=str(payload.get("path_text") or ""),
                    text=self._question_focused_evidence_text(
                        question, hit, limit=int(self.settings.answer_evidence_char_limit)
                    ),
                    relevance=hit.provider_rerank_score,
                )
            )
        return evidence_items

    def _build_canonical_unit_bundles(
        self,
        question: str,
        hits: list[RetrievalHit],
        *,
        evidence_intent: EvidenceIntentClassification,
        retrieved_examples: list[RetrievedAnswerUnitExample],
    ) -> list[CanonicalUnitBundle]:
        grouped: dict[str, list[RetrievalHit]] = {}
        for hit in hits:
            unit_key = self._answer_unit_group_key(hit)
            if not unit_key:
                continue
            grouped.setdefault(unit_key, []).append(hit)

        bundles: list[CanonicalUnitBundle] = []
        provisional: list[tuple[float, CanonicalUnitBundle]] = []
        for group_key, unit_hits in grouped.items():
            ordered_hits = sorted(
                unit_hits,
                key=lambda current: (
                    self._answer_focus_score(question, current),
                    self._evidence_role_alignment_score(evidence_intent, current),
                    current.rerank_score,
                ),
                reverse=True,
            )
            best_hit = ordered_hits[0]
            payload = best_hit.payload or {}
            representative_text = self._question_focused_evidence_text(question, best_hit, limit=340)
            support_texts: list[str] = []
            for support_hit in ordered_hits[1:3]:
                support_text = self._question_focused_evidence_text(question, support_hit, limit=220)
                if support_text and support_text != representative_text:
                    support_texts.append(support_text)

            candidate = CanonicalUnitCandidate(
                unit_label="",
                unit_key=group_key,
                canonical_label=str(payload.get("canonical_label") or self._display_branch_label(best_hit)).strip(),
                branch_label=self._display_branch_label(best_hit),
                unit_type=str(payload.get("unit_type") or "").strip(),
                unit_number=str(payload.get("unit_number") or "").strip(),
                unit_topic=str(payload.get("unit_topic") or "").strip(),
                path_text=str(payload.get("path_text") or "").strip(),
                evidence_role=str(payload.get("evidence_role") or "").strip(),
                representative_text=representative_text,
                support_texts=support_texts,
                focus_score=round(self._answer_focus_score(question, best_hit), 4),
                rerank_score=round(float(best_hit.rerank_score), 4),
                hit_count=len(ordered_hits),
                example_prior=self.answer_unit_examples.match_score(
                    candidate_label=str(payload.get("canonical_label") or self._display_branch_label(best_hit)),
                    unit_type=str(payload.get("unit_type") or ""),
                    unit_number=str(payload.get("unit_number") or ""),
                    examples=retrieved_examples,
                ),
            )
            bundle = CanonicalUnitBundle(
                unit_label="",
                unit_key=candidate.unit_key,
                candidate=candidate,
                hits=ordered_hits,
            )
            provisional.append(
                (
                    self._canonical_unit_bundle_score(
                        bundle=bundle,
                        evidence_intent=evidence_intent,
                    ),
                    bundle,
                )
            )

        provisional.sort(key=lambda item: item[0], reverse=True)
        for index, (_, bundle) in enumerate(provisional, start=1):
            unit_label = f"U{index}"
            bundle.unit_label = unit_label
            bundle.candidate.unit_label = unit_label
            bundles.append(bundle)
        return bundles

    def _answer_unit_group_key(self, hit: RetrievalHit) -> str:
        payload = hit.payload or {}
        unit_key = str(payload.get("unit_key") or "").strip()
        if unit_key:
            return f"{hit.document_id}:{hit.version_id}:{unit_key}"
        branch_key = str(payload.get("branch_key") or "").strip()
        if branch_key:
            return f"{hit.document_id}:{hit.version_id}:{branch_key}"
        canonical_label = str(payload.get("canonical_label") or "").strip()
        if canonical_label:
            return f"{hit.document_id}:{hit.version_id}:{self._normalize(canonical_label)}"
        return hit.chunk_id

    def _canonical_unit_bundle_score(
        self,
        *,
        bundle: CanonicalUnitBundle,
        evidence_intent: EvidenceIntentClassification,
    ) -> float:
        if not bundle.hits:
            return 0.0
        best_hit = bundle.hits[0]
        return (
            (0.08 * bundle.candidate.example_prior)
            + (0.35 * bundle.candidate.focus_score)
            + self._evidence_role_alignment_score(evidence_intent, best_hit)
            + best_hit.rerank_score
            + min(0.4, 0.08 * len(bundle.hits))
        )

    def _select_units_with_centrality(
        self,
        *,
        answer_form: str,
        bundles: list[CanonicalUnitBundle],
        centrality_result,
    ) -> CanonicalUnitSelectionResult | None:
        if answer_form not in {"direct_unit", "unit_set", "rule_summary"}:
            return None

        bundle_map = {bundle.unit_label: bundle for bundle in bundles}
        scored = [
            prediction
            for prediction in centrality_result.predictions
            if prediction.unit_label in bundle_map
        ]
        if not scored:
            return None

        scored.sort(
            key=lambda prediction: self._unit_selection_score(
                prediction=prediction,
                bundle=bundle_map[prediction.unit_label],
            ),
            reverse=True,
        )

        assignments: list[CanonicalUnitAssignment] = []
        render_style = "single_branch"
        if answer_form == "unit_set":
            selected = [
                prediction
                for prediction in scored
                if self._unit_selection_score(
                    prediction=prediction,
                    bundle=bundle_map[prediction.unit_label],
                )
                >= 0.45
            ][:6]
            minimum_units = min(2, len(scored))
            if len(selected) < minimum_units:
                selected = scored[: min(6, max(minimum_units, len(selected)))]
            assignments = [
                CanonicalUnitAssignment(
                    unit_label=prediction.unit_label,
                    selection_role="primary",
                )
                for prediction in selected
            ]
            render_style = "grouped_by_branch"
        else:
            top_prediction = scored[0]
            assignments = [
                CanonicalUnitAssignment(
                    unit_label=top_prediction.unit_label,
                    selection_role="primary",
                )
            ]
            render_style = "single_branch"

        if not assignments:
            return None

        return CanonicalUnitSelectionResult(
            assignments=assignments,
            render_style=render_style,
            source="centrality_linear",
            decision_trace=[
                "unit_select:provider=centrality_linear",
                f"unit_select:answer_form={answer_form}",
                f"unit_select:candidates={len(scored)}",
                f"unit_select:selected={len(assignments)}",
                "unit_select:roles="
                + ",".join(
                    f"{role}:{sum(1 for item in assignments if item.selection_role == role)}"
                    for role in ("primary", "support", "context")
                ),
                f"unit_select:render_style={render_style}",
            ],
        )

    def _unit_selection_score(
        self,
        *,
        prediction,
        bundle: CanonicalUnitBundle,
    ) -> float:
        return (
            (0.9 * float(prediction.probability))
            + (0.07 * bundle.candidate.focus_score)
            + (0.03 * max(0.0, bundle.candidate.rerank_score))
        )

    def _build_unit_set_block_candidates(
        self,
        bundles: list[CanonicalUnitBundle],
    ) -> list[UnitSetBlockCandidate]:
        grouped: dict[str, dict[str, object]] = {}
        for bundle in bundles:
            if not bundle.hits:
                continue
            best_hit = bundle.hits[0]
            parent_branch_key = ""
            parent_path = ""
            for hit in bundle.hits:
                if not parent_branch_key:
                    parent_branch_key = self._parent_branch_key_for_hit(hit)
                if not parent_path:
                    parent_path = self._parent_path_spec_for_hit(hit)[0]
                if parent_branch_key and parent_path:
                    break
            if not parent_path:
                parent_path = self._candidate_parent_path(bundle.candidate.path_text)
            block_identity = parent_path or parent_branch_key
            if not block_identity:
                continue
            block_key = f"{best_hit.document_id}:{best_hit.version_id}:{block_identity}"
            bucket = grouped.setdefault(
                block_key,
                {
                    "block_label": parent_path or parent_branch_key or block_identity,
                    "parent_path": parent_path,
                    "parent_branch_key": parent_branch_key,
                    "document_id": best_hit.document_id,
                    "version_id": best_hit.version_id,
                    "unit_type": str(bundle.candidate.unit_type or "").strip(),
                    "member_unit_labels": [],
                    "member_unit_keys": [],
                    "member_canonical_labels": [],
                    "member_topics": [],
                    "member_count": 0,
                    "hit_count": 0,
                    "focus_scores": [],
                    "rerank_scores": [],
                    "representative_text": "",
                },
            )
            bucket["member_unit_labels"].append(bundle.unit_label)
            bucket["member_unit_keys"].append(bundle.unit_key)
            bucket["member_canonical_labels"].append(bundle.candidate.canonical_label)
            if bundle.candidate.unit_topic:
                bucket["member_topics"].append(bundle.candidate.unit_topic)
            bucket["member_count"] = int(bucket["member_count"]) + 1
            bucket["hit_count"] = int(bucket["hit_count"]) + len(bundle.hits)
            bucket["focus_scores"].append(float(bundle.candidate.focus_score))
            bucket["rerank_scores"].append(float(bundle.candidate.rerank_score))
            if not bucket["representative_text"]:
                bucket["representative_text"] = bundle.candidate.representative_text

        candidates: list[UnitSetBlockCandidate] = []
        for block_key, bucket in grouped.items():
            member_unit_labels = self._dedupe_strings(bucket["member_unit_labels"])
            if len(member_unit_labels) < 2:
                continue
            member_unit_keys = self._dedupe_strings(bucket["member_unit_keys"])
            member_canonical_labels = self._dedupe_strings(bucket["member_canonical_labels"])
            member_topics = self._dedupe_strings(bucket["member_topics"])
            focus_scores = [float(value) for value in bucket["focus_scores"]]
            rerank_scores = [float(value) for value in bucket["rerank_scores"]]
            candidates.append(
                UnitSetBlockCandidate(
                    block_key=block_key,
                    block_label=str(bucket["block_label"]).strip(),
                    parent_path=str(bucket["parent_path"]).strip(),
                    parent_branch_key=str(bucket["parent_branch_key"]).strip(),
                    document_id=str(bucket["document_id"]).strip(),
                    version_id=str(bucket["version_id"]).strip(),
                    unit_type=str(bucket["unit_type"]).strip(),
                    member_count=int(bucket["member_count"]),
                    hit_count=int(bucket["hit_count"]),
                    focus_score=round(max(focus_scores) if focus_scores else 0.0, 4),
                    rerank_score=round(max(rerank_scores) if rerank_scores else 0.0, 4),
                    member_unit_labels=member_unit_labels,
                    member_unit_keys=member_unit_keys,
                    member_canonical_labels=member_canonical_labels,
                    member_topics_text=" | ".join(member_topics[:6]),
                    representative_text=str(bucket["representative_text"]).strip()[:340],
                )
            )
        candidates.sort(
            key=lambda candidate: (
                candidate.focus_score,
                candidate.member_count,
                candidate.hit_count,
            ),
            reverse=True,
        )
        return candidates

    def _hits_for_selected_units(
        self,
        *,
        question: str,
        hits: list[RetrievalHit],
        bundles: list[CanonicalUnitBundle],
        selection: CanonicalUnitSelectionResult,
        answer_form: str,
    ) -> tuple[list[RetrievalHit], dict[str, str]]:
        bundle_map = {bundle.unit_label: bundle for bundle in bundles}
        role_map: dict[str, str] = {}
        selected: list[RetrievalHit] = []
        seen_chunks: set[str] = set()

        for assignment in selection.assignments:
            bundle = bundle_map.get(assignment.unit_label)
            if bundle is None:
                continue
            budget = self._unit_hit_budget(
                answer_form=answer_form,
                selection_role=assignment.selection_role,
            )
            candidate_hits = self._hits_for_unit_bundle(
                question=question,
                all_hits=hits,
                bundle=bundle,
            )
            for hit in candidate_hits[:budget]:
                if hit.chunk_id in seen_chunks:
                    continue
                seen_chunks.add(hit.chunk_id)
                selected.append(hit)
                role_map[hit.chunk_id] = assignment.selection_role

        return selected, role_map

    def _hits_for_unit_bundle(
        self,
        *,
        question: str,
        all_hits: list[RetrievalHit],
        bundle: CanonicalUnitBundle,
    ) -> list[RetrievalHit]:
        matching_hits = [
            hit for hit in all_hits if self._answer_unit_group_key(hit) == bundle.unit_key
        ]
        if not matching_hits:
            matching_hits = list(bundle.hits)
        return sorted(
            matching_hits,
            key=lambda current: (
                self._answer_focus_score(question, current),
                current.rerank_score,
            ),
            reverse=True,
        )

    def _unit_hit_budget(self, *, answer_form: str, selection_role: str) -> int:
        if answer_form == "unit_set":
            return 1 if selection_role == "primary" else 1
        if answer_form == "direct_unit":
            return 2 if selection_role == "primary" else 1
        if answer_form == "rule_summary":
            return 2 if selection_role == "primary" else 1
        if answer_form == "global_summary":
            return 1
        return 1

    def _parent_branch_label(self, hit: RetrievalHit) -> str:
        payload = hit.payload or {}
        parent_branch_key = str(payload.get("parent_branch_key") or "").strip()
        if not parent_branch_key:
            return ""
        path = self._as_path(payload.get("hierarchy_path") or payload.get("section_path"))
        if len(path) >= 2:
            return " > ".join(path[:-1])
        return parent_branch_key

    def _parent_path_spec_for_hit(self, hit: RetrievalHit) -> tuple[str, int]:
        payload = hit.payload or {}
        parent_path = self._as_path(payload.get("parent_path"))
        if not parent_path:
            hierarchy_path = self._as_path(payload.get("hierarchy_path") or payload.get("section_path"))
            parent_path = hierarchy_path[:-1]
        if not parent_path:
            return "", 0
        return " > ".join(parent_path), len(parent_path)

    def _hit_parent_identity(self, hit: RetrievalHit) -> str:
        parent_branch_key = self._parent_branch_key_for_hit(hit)
        if parent_branch_key:
            return parent_branch_key
        parent_path_text, _ = self._parent_path_spec_for_hit(hit)
        return parent_path_text

    def _augment_unit_set_candidates_with_siblings(
        self,
        *,
        request: QueryRequest,
        route: QueryRoute,
        question: str,
        candidate_hits: list[RetrievalHit],
        all_hits: list[RetrievalHit],
        plan: QueryPlan,
        evidence_intent: EvidenceIntentClassification,
        answer_form: str,
    ) -> tuple[list[RetrievalHit], list[str]]:
        del plan
        if answer_form != "unit_set" or not candidate_hits:
            return candidate_hits, ["unit_set_expand:parents=0", "unit_set_expand:sibling_branches=0", "unit_set_expand:leaf_hits=0"]

        seed_hits = self._rank_answer_hits(
            question,
            self._unique_branch_hits(candidate_hits, limit=8),
            evidence_intent=evidence_intent,
        )
        unit_type_counts: dict[str, int] = {}
        for hit in seed_hits:
            unit_type = str((hit.payload or {}).get("unit_type") or "").strip().lower()
            if not unit_type or unit_type in {"document", "unknown"}:
                continue
            unit_type_counts[unit_type] = unit_type_counts.get(unit_type, 0) + 1
        preferred_unit_types: set[str] = set()
        if unit_type_counts:
            max_count = max(unit_type_counts.values())
            preferred_unit_types = {
                unit_type
                for unit_type, count in unit_type_counts.items()
                if count == max_count
            }
        parent_candidates: dict[str, dict[str, object]] = {}
        existing_branch_keys = {
            self._branch_key_for_hit(hit)
            for hit in candidate_hits
            if self._branch_key_for_hit(hit)
        }
        for hit in seed_hits:
            if not self._is_specific_branch_hit(hit):
                continue
            parent_path_spec = self._parent_path_spec_for_hit(hit)
            if parent_path_spec[0]:
                parent_identity = parent_path_spec[0]
            else:
                parent_identity = ""
            parent_branch_key = self._parent_branch_key_for_hit(hit)
            if parent_branch_key:
                parent_identity = parent_branch_key or parent_identity
            if not parent_identity:
                continue
            current = parent_candidates.setdefault(
                parent_identity,
                {
                    "branch_key": "",
                    "path_spec": ("", 0),
                    "score": 0.0,
                },
            )
            if parent_branch_key and not current["branch_key"]:
                current["branch_key"] = parent_branch_key
            if parent_path_spec[0] and not current["path_spec"][0]:
                current["path_spec"] = parent_path_spec
            current["score"] = float(current["score"]) + self._answer_focus_score(question, hit) + max(0.0, hit.rerank_score)
        selected_parent_entries = sorted(
            parent_candidates.items(),
            key=lambda item: float(item[1]["score"]),
            reverse=True,
        )[:2]
        parent_branch_keys = self._dedupe_strings(
            [str(item[1]["branch_key"]) for item in selected_parent_entries if str(item[1]["branch_key"]).strip()]
        )
        parent_path_specs = [
            item[1]["path_spec"]
            for item in selected_parent_entries
            if item[1]["path_spec"][0]
        ]
        if not parent_branch_keys and not parent_path_specs:
            return candidate_hits, ["unit_set_expand:parents=0", "unit_set_expand:sibling_branches=0", "unit_set_expand:leaf_hits=0"]
        parent_priority = {
            str(identity): (len(selected_parent_entries) - index)
            for index, (identity, _) in enumerate(selected_parent_entries)
        }
        selected_parent_identities = [str(identity) for identity, _ in selected_parent_entries]

        question_child_hits, child_trace = self._branch_child_search(
            question=question,
            request=request,
            route=route,
            parent_branch_keys=parent_branch_keys,
        )
        catalog_hits, catalog_reason = self._branch_catalog_search(
            request=request,
            route=route,
            parent_branch_keys=parent_branch_keys,
            parent_path_specs=parent_path_specs,
            limit=24,
        )

        sibling_branch_hits: list[RetrievalHit] = []
        seen_branch_keys: set[str] = set()
        for hit in [*question_child_hits, *catalog_hits]:
            branch_key = self._branch_key_for_hit(hit)
            if not branch_key or branch_key in existing_branch_keys or branch_key in seen_branch_keys:
                continue
            if not self._is_specific_branch_hit(hit):
                continue
            unit_type = str((hit.payload or {}).get("unit_type") or "").strip().lower()
            if preferred_unit_types and unit_type and unit_type not in preferred_unit_types:
                continue
            seen_branch_keys.add(branch_key)
            sibling_branch_hits.append(hit)
        sibling_branch_hits.sort(
            key=lambda hit: (
                parent_priority.get(
                    self._parent_branch_key_for_hit(hit) or self._parent_path_spec_for_hit(hit)[0],
                    0,
                ),
                self._answer_focus_score(question, hit),
                str((hit.payload or {}).get("path_text") or ""),
            ),
            reverse=True,
        )
        sibling_branch_hits = sibling_branch_hits[:24]
        sibling_branch_keys = self._dedupe_strings(
            [self._branch_key_for_hit(hit) for hit in sibling_branch_hits if self._branch_key_for_hit(hit)]
        )

        sibling_dense_hits: list[RetrievalHit] = []
        sibling_sparse_hits: list[RetrievalHit] = []
        dense_reason: str | None = None
        sparse_reason: str | None = None
        if sibling_branch_keys:
            sibling_dense_hits, dense_reason = self._branch_guided_dense_search(
                queries=[question],
                request=request,
                route=route,
                branch_keys=sibling_branch_keys,
            )
            sibling_sparse_hits, sparse_reason = self._branch_guided_sparse_search(
                queries=[question],
                request=request,
                route=route,
                branch_keys=sibling_branch_keys,
            )
        sibling_leaf_hits = self._fuse_hits(sibling_dense_hits, sibling_sparse_hits)
        sibling_key_set = set(sibling_branch_keys)
        same_sibling_hits = [
            hit
            for hit in all_hits
            if self._branch_key_for_hit(hit) in sibling_key_set
        ]

        merged: list[RetrievalHit] = []
        seen_chunks: set[str] = set()
        for hit in [*candidate_hits, *same_sibling_hits, *sibling_branch_hits, *sibling_leaf_hits]:
            if hit.chunk_id in seen_chunks:
                continue
            seen_chunks.add(hit.chunk_id)
            merged.append(hit)

        ranked = self._rank_answer_hits(
            question,
            merged,
            evidence_intent=evidence_intent,
        )
        limit = max(24, min(40, len(candidate_hits) + len(sibling_branch_hits) + len(sibling_leaf_hits)))
        preserved_parent_block_hits = [
            hit
            for hit in merged
            if selected_parent_identities
            and self._hit_parent_identity(hit) in set(selected_parent_identities)
            and (
                not preferred_unit_types
                or str((hit.payload or {}).get("unit_type") or "").strip().lower() in preferred_unit_types
            )
        ]
        preserved_parent_block_hits = sorted(
            preserved_parent_block_hits,
            key=lambda hit: (
                parent_priority.get(self._hit_parent_identity(hit), 0),
                str((hit.payload or {}).get("path_text") or ""),
                self._answer_focus_score(question, hit),
                hit.rerank_score,
            ),
            reverse=True,
        )[:14]
        augmented_hits: list[RetrievalHit] = []
        preserved_seen: set[str] = set()
        for hit in [*candidate_hits, *preserved_parent_block_hits, *sibling_branch_hits, *sibling_leaf_hits, *ranked]:
            if hit.chunk_id in preserved_seen:
                continue
            preserved_seen.add(hit.chunk_id)
            augmented_hits.append(hit)
            if len(augmented_hits) >= limit:
                break

        trace = [f"unit_set_expand:parents={len(parent_branch_keys) + len(parent_path_specs)}"]
        trace.extend(child_trace)
        if catalog_reason:
            trace.append(f"unit_set_catalog:{catalog_reason}")
        else:
            trace.append(f"unit_set_catalog={len(catalog_hits)}")
        if dense_reason:
            trace.append(f"unit_set_leaf_dense:{dense_reason}")
        else:
            trace.append(f"unit_set_leaf_dense={len(sibling_dense_hits)}")
        if sparse_reason:
            trace.append(f"unit_set_leaf_sparse:{sparse_reason}")
        else:
            trace.append(f"unit_set_leaf_sparse={len(sibling_sparse_hits)}")
        trace.extend(
            [
                f"unit_set_expand:preserved_block={len(preserved_parent_block_hits)}",
                f"unit_set_expand:sibling_branches={len(sibling_branch_keys)}",
                f"unit_set_expand:leaf_hits={len(sibling_leaf_hits)}",
            ]
        )
        return augmented_hits, trace

    def _select_answer_hits(
        self,
        question: str,
        hits: list[RetrievalHit],
        plan: QueryPlan,
        *,
        evidence_intent: EvidenceIntentClassification,
        answer_form: str,
        structure: QueryStructure,
    ) -> list[RetrievalHit]:
        if answer_form == "unit_set":
            candidate_hits = self._unique_branch_hits(hits, limit=24)
        elif answer_form == "direct_unit":
            candidate_hits = hits[:12]
        elif answer_form == "global_summary":
            candidate_hits = self._unique_branch_hits(hits, limit=10)
        elif plan.answer_style in {"grouped_by_branch", "summary"}:
            candidate_hits = self._unique_branch_hits(hits, limit=10)
        else:
            candidate_hits = hits[:10]
        ranked_hits = self._rank_answer_hits(
            question,
            candidate_hits,
            evidence_intent=evidence_intent,
        )
        curated = self._curate_hits_for_coverage(
            question=question,
            hits=ranked_hits,
            structure=structure,
            plan=plan,
            evidence_intent=evidence_intent,
            limit=min(len(ranked_hits), 6 if plan.strategy == "focal" else 8),
            selected_branch_keys=set(),
            prefer_same_branch=plan.same_branch_preferred,
            prefer_diversity=plan.diversity_required,
        )
        if not curated:
            return ranked_hits
        selected_chunks = {hit.chunk_id for hit in curated}
        return curated + [hit for hit in ranked_hits if hit.chunk_id not in selected_chunks]

    def _augment_answer_candidates_with_example_units(
        self,
        *,
        request: QueryRequest,
        route: QueryRoute,
        question: str,
        answer_form: str,
        candidate_hits: list[RetrievalHit],
        all_hits: list[RetrievalHit],
        retrieved_examples: list[RetrievedAnswerUnitExample],
        evidence_intent: EvidenceIntentClassification,
    ) -> tuple[list[RetrievalHit], list[str]]:
        if answer_form not in {"direct_unit", "unit_set", "rule_summary"}:
            return candidate_hits, ["unit_examples:augment_queries=0", "unit_examples:augment_hits=0"]
        if not retrieved_examples:
            return candidate_hits, ["unit_examples:augment_queries=0", "unit_examples:augment_hits=0"]

        unit_queries = self._example_unit_queries(retrieved_examples, answer_form=answer_form)
        if not unit_queries:
            return candidate_hits, ["unit_examples:augment_queries=0", "unit_examples:augment_hits=0"]

        augmented_hits: list[RetrievalHit] = []
        for unit_query in unit_queries:
            for hit in self._search_hits_for_example_unit(
                request=request,
                route=route,
                unit_query=unit_query,
            ):
                augmented_hits.append(hit)

        if not augmented_hits:
            return candidate_hits, [
                f"unit_examples:augment_queries={len(unit_queries)}",
                "unit_examples:augment_hits=0",
            ]

        merged: list[RetrievalHit] = []
        seen_chunks: set[str] = set()
        for hit in [*candidate_hits, *all_hits, *augmented_hits]:
            if hit.chunk_id in seen_chunks:
                continue
            seen_chunks.add(hit.chunk_id)
            merged.append(hit)

        ranked = self._rank_answer_hits(
            question,
            merged,
            evidence_intent=evidence_intent,
        )
        limit = max(len(candidate_hits), 24 if answer_form == "unit_set" else 12)
        preserved: list[RetrievalHit] = []
        seen_chunks: set[str] = set()
        for hit in [*augmented_hits, *ranked]:
            if hit.chunk_id in seen_chunks:
                continue
            seen_chunks.add(hit.chunk_id)
            preserved.append(hit)
            if len(preserved) >= limit:
                break
        return preserved, [
            f"unit_examples:augment_queries={len(unit_queries)}",
            f"unit_examples:augment_hits={len(augmented_hits)}",
        ]

    def _example_unit_queries(
        self,
        retrieved_examples: list[RetrievedAnswerUnitExample],
        *,
        answer_form: str,
    ) -> list[str]:
        limit = {"direct_unit": 2, "rule_summary": 3, "unit_set": 8}.get(answer_form, 0)
        if limit <= 0:
            return []
        example_pool = retrieved_examples[:2]
        if (
            answer_form == "unit_set"
            and retrieved_examples
            and float(retrieved_examples[0].similarity) >= 0.9
        ):
            example_pool = retrieved_examples[:1]
        queries: list[str] = []
        seen: set[str] = set()
        for example in example_pool:
            ordered_units = list(example.expected_units)
            if example.primary_unit:
                ordered_units = [example.primary_unit, *ordered_units]
            for unit in ordered_units:
                query = str(unit).strip()
                normalized = self._normalize(query)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                queries.append(query)
                if len(queries) >= limit:
                    return queries
        return queries

    def _search_hits_for_example_unit(
        self,
        *,
        request: QueryRequest,
        route: QueryRoute,
        unit_query: str,
    ) -> list[RetrievalHit]:
        probe_request = QueryRequest(
            question=unit_query,
            user_identity=request.user_identity,
                user_groups=request.user_groups,
            filters=dict(request.filters),
            conversation_context=list(request.conversation_context),
        )
        dense_hits, _ = self._dense_search(probe_request, route)
        sparse_hits, _ = self._sparse_search(probe_request, route)
        fused_hits = self._fuse_hits(dense_hits, sparse_hits)
        refs = self._extract_structural_references(unit_query)
        matched: list[RetrievalHit] = []
        seen_chunks: set[str] = set()
        for hit in fused_hits:
            if hit.chunk_id in seen_chunks:
                continue
            structural_score = self._structural_reference_score(refs, hit)
            mismatch_penalty = self._structural_reference_mismatch_penalty(refs, hit)
            if structural_score <= 0.0 or mismatch_penalty >= structural_score:
                continue
            hit.rerank_score = max(hit.rerank_score, hit.rrf_score + structural_score - mismatch_penalty)
            matched.append(hit)
            seen_chunks.add(hit.chunk_id)
            if len(matched) >= 2:
                break
        return matched

    def _citation_hits(
        self,
        answer_hits: list[RetrievalHit],
        final_hits: list[RetrievalHit],
    ) -> list[RetrievalHit]:
        ordered: list[RetrievalHit] = []
        seen: set[str] = set()
        for hit in [*answer_hits, *final_hits]:
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            ordered.append(hit)
        return ordered

    def _hits_for_used_labels(
        self,
        label_to_hit: dict[str, RetrievalHit],
        used_labels: list[str],
    ) -> list[RetrievalHit]:
        if not used_labels:
            return []
        selected: list[RetrievalHit] = []
        seen_chunks: set[str] = set()
        for label in used_labels:
            hit = label_to_hit.get(label)
            if hit is None or hit.chunk_id in seen_chunks:
                continue
            selected.append(hit)
            seen_chunks.add(hit.chunk_id)
        return selected

    def _confidence(self, hits: list[RetrievalHit]) -> tuple[float, str]:
        """Confianza basada en el score crudo del cross-encoder.

        La formula anterior (0.35 + rerank_score*3) saturaba en 0.95 casi
        siempre, incluso cuando la respuesta era "no hay evidencia". El score del
        proveedor sí discrimina: ~0.99 cuando la evidencia responde, ~0.002
        cuando no. Se toma el maximo de los hits finales, no el del primero,
        porque las etapas de reordenamiento pueden no dejar arriba al mejor.
        """
        if not hits:
            return 0.0, "no_hits"

        provider_scores = [
            hit.provider_rerank_score
            for hit in hits
            if hit.provider_rerank_score is not None
        ]
        if provider_scores:
            best = max(provider_scores)
            return round(min(0.95, max(0.01, float(best))), 2), "provider_rerank"

        # Sin proveedor de rerank (modo off o fallback heuristico) no hay señal
        # con significado absoluto: se conserva la formula historica.
        top_score = hits[0].rerank_score
        return round(min(0.95, 0.35 + top_score * 3.0), 2), "heuristic_fallback"

    def _acl_qdrant_clause(self, request: QueryRequest) -> dict[str, object] | None:
        """Un chunk es visible si nombra al usuario, si alcanza a alguno de sus
        grupos, o si no declara restriccion alguna."""
        if not self.settings.acl_enforcement_enabled:
            return None
        alternatives: list[dict[str, object]] = []
        identity = (request.user_identity or "").strip()
        if identity:
            alternatives.append({"key": "acl_users", "match": {"any": [identity]}})
        groups = [str(group).strip() for group in (request.user_groups or []) if str(group).strip()]
        if groups:
            alternatives.append({"key": "acl_groups", "match": {"any": groups}})
        if self.settings.acl_open_when_unrestricted:
            alternatives.append(
                {
                    "must": [
                        {"is_empty": {"key": "acl_users"}},
                        {"is_empty": {"key": "acl_groups"}},
                    ]
                }
            )
        if not alternatives:
            # Sin identidad ni grupos y sin apertura por defecto: no ver nada es
            # la respuesta correcta, no ver todo.
            return {"must": [{"key": "acl_users", "match": {"any": ["__deny_all__"]}}]}
        # Filtro anidado: en Qdrant un `should` ya exige al menos una coincidencia,
        # y `min_should` no es valido en esta posicion (error 400).
        return {"should": alternatives}

    def _build_qdrant_filter(
        self,
        filters: dict[str, object],
        route: QueryRoute,
        branch_keys: list[str] | None = None,
        parent_branch_keys: list[str] | None = None,
        request: QueryRequest | None = None,
    ) -> dict[str, object] | None:
        must: list[dict[str, object]] = [
            {"key": "version_status", "match": {"value": "active"}},
        ]
        if request is not None:
            acl_clause = self._acl_qdrant_clause(request)
            if acl_clause is not None:
                must.append(acl_clause)
        should: list[dict[str, object]] = []
        class_values = route.target_classes
        if class_values and getattr(route, "hard_class_filter", True):
            if len(class_values) == 1:
                must.append({"key": "document_class", "match": {"value": class_values[0]}})
            else:
                should.extend(
                    {"key": "document_class", "match": {"value": class_value}}
                    for class_value in class_values
                )

        for key in ("document_id", "version_id", "document_family", "chunk_kind"):
            value = filters.get(key)
            if value:
                must.append({"key": key, "match": {"value": value}})
        if branch_keys:
            must.append({"key": "branch_keys", "match": {"any": branch_keys}})
        if parent_branch_keys:
            must.append({"key": "parent_branch_key", "match": {"any": parent_branch_keys}})

        filter_body: dict[str, object] = {"must": must}
        if should:
            filter_body["should"] = should
        return filter_body

    def _class_preference_clauses(self, route: QueryRoute) -> list[dict[str, object]]:
        """Cuando el filtro de clase es suave, la clase predicha se degrada a
        preferencia: suma score sin excluir al resto del corpus."""
        if getattr(route, "hard_class_filter", True):
            return []
        class_values = route.target_classes
        if not class_values:
            return []
        return [{"terms": {"document_class": list(class_values), "boost": 2.0}}]

    def _acl_opensearch_clause(self, request: QueryRequest) -> dict[str, object] | None:
        if not self.settings.acl_enforcement_enabled:
            return None
        alternatives: list[dict[str, object]] = []
        identity = (request.user_identity or "").strip()
        if identity:
            alternatives.append({"terms": {"acl_users": [identity]}})
        groups = [str(group).strip() for group in (request.user_groups or []) if str(group).strip()]
        if groups:
            alternatives.append({"terms": {"acl_groups": groups}})
        if self.settings.acl_open_when_unrestricted:
            # Un array vacio no indexa el campo, asi que "sin ACL" equivale a que
            # ninguno de los dos campos exista.
            alternatives.append(
                {
                    "bool": {
                        "must_not": [
                            {"exists": {"field": "acl_users"}},
                            {"exists": {"field": "acl_groups"}},
                        ]
                    }
                }
            )
        if not alternatives:
            return {"bool": {"must_not": [{"match_all": {}}]}}
        return {"bool": {"should": alternatives, "minimum_should_match": 1}}

    def _build_opensearch_filters(
        self,
        filters: dict[str, object],
        route: QueryRoute,
        branch_keys: list[str] | None = None,
        parent_branch_keys: list[str] | None = None,
        request: QueryRequest | None = None,
    ) -> list[dict[str, object]]:
        clauses: list[dict[str, object]] = [{"term": {"version_status": "active"}}]
        if request is not None:
            acl_clause = self._acl_opensearch_clause(request)
            if acl_clause is not None:
                clauses.append(acl_clause)
        class_values = route.target_classes
        if class_values and getattr(route, "hard_class_filter", True):
            clauses.append({"terms": {"document_class": class_values}})
        if branch_keys:
            clauses.append({"terms": {"branch_keys": branch_keys}})
        if parent_branch_keys:
            clauses.append({"terms": {"parent_branch_key": parent_branch_keys}})

        for key in ("document_id", "version_id", "document_family", "chunk_kind"):
            value = filters.get(key)
            if value:
                clauses.append({"term": {key: value}})
        return clauses

    def _tokens(self, text: str) -> set[str]:
        normalized = self._normalize(text)
        return {
            token
            for token in TOKEN_PATTERN.findall(normalized)
            if token not in STOPWORDS and len(token) > 1
        }

    def _normalize(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text or "")
        return "".join(char for char in normalized if not unicodedata.combining(char)).lower()

    def _overlap_ratio(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left)

    def _soft_overlap_ratio(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        matches = 0
        for token in left:
            if any(self._token_matches(token, candidate) for candidate in right):
                matches += 1
        return matches / len(left)

    def _token_matches(self, left: str, right: str) -> bool:
        if left == right:
            return True
        if min(len(left), len(right)) < 5:
            return False
        return left.startswith(right) or right.startswith(left)

    def _contains_exact_reference(self, question: str, hit: RetrievalHit) -> bool:
        question_norm = self._normalize(question)
        payload_text = self._normalize(
            " ".join(
                [
                    hit.retrieval_context,
                    " ".join(self._as_path(hit.payload.get("hierarchy_path") or hit.payload.get("section_path"))),
                    str(hit.payload.get("article_label") or ""),
                    str(hit.payload.get("section_label") or ""),
                    str(hit.payload.get("subsection_label") or ""),
                    hit.raw_text[:400],
                ]
            )
        )
        references = re.findall(
            r"(art\.?\s*\d+|arto\.?\s*\d+|articulo\s+\d+|seccion\s+\d+|recomendacion\s+\d+|r\.?\s*\d+|ley\s+\d+)",
            question_norm,
        )
        return any(reference in payload_text for reference in references)

    def _find_visibility_replacement(
        self,
        *,
        question: str,
        route: QueryRoute,
        hits: list[RetrievalHit],
        start_index: int,
        selected_branch_keys: set[str],
        evidence_intent: EvidenceIntentClassification,
    ) -> int | None:
        best_index: int | None = None
        best_score = float("-inf")
        search_window = min(len(hits), max(start_index + 8, 8))
        for index in range(start_index, search_window):
            candidate = hits[index]
            score = self._visibility_candidate_score(
                question=question,
                route=route,
                hit=candidate,
                selected_branch_keys=selected_branch_keys,
                position=index,
                evidence_intent=evidence_intent,
            )
            if score <= 0:
                continue
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    def _visibility_candidate_score(
        self,
        *,
        question: str,
        route: QueryRoute,
        hit: RetrievalHit,
        selected_branch_keys: set[str],
        position: int,
        evidence_intent: EvidenceIntentClassification,
    ) -> float:
        del question
        del route
        score = 0.0
        branch_lead_keys = {
            str(value).strip()
            for value in (hit.payload.get("branch_lead_keys") or [])
            if str(value).strip()
        }
        branch_keys = {
            str(value).strip()
            for value in (hit.payload.get("branch_keys") or [])
            if str(value).strip()
        }
        if branch_lead_keys & selected_branch_keys:
            score += 2.5
        elif branch_keys & selected_branch_keys:
            score += 1.2

        score += 0.9 * self._evidence_role_alignment_score(evidence_intent, hit)
        score += 0.25 * self._payload_float(hit, "segment_score")
        score += max(0.0, 0.4 - (0.03 * max(0, position - 1)))
        score += min(0.6, max(0.0, hit.rerank_score))
        score -= 0.9 * self._boilerplate_score(hit)

        return score

    def _evidence_role_scores(self, hit: RetrievalHit) -> dict[str, float]:
        payload = hit.payload or {}
        scores = {
            "substantive": float(payload.get("evidence_role_substantive") or 0.0),
            "reference": float(payload.get("evidence_role_reference") or 0.0),
            "editorial": float(payload.get("evidence_role_editorial") or 0.0),
        }
        if sum(scores.values()) > 0:
            return scores

        role = str(payload.get("evidence_role") or "").strip().lower()
        if role in scores:
            scores[role] = float(payload.get("evidence_role_confidence") or 1.0)
        return scores

    def _evidence_role_alignment_score(
        self,
        intent: EvidenceIntentClassification,
        hit: RetrievalHit,
    ) -> float:
        scores = self._evidence_role_scores(hit)
        if intent.intent == "substantive":
            target = scores.get("substantive", 0.0)
            competing = max(scores.get("reference", 0.0), scores.get("editorial", 0.0))
            return max(-1.0, min(1.0, target - competing))
        if intent.intent == "reference":
            target = scores.get("reference", 0.0)
            competing = max(scores.get("substantive", 0.0), scores.get("editorial", 0.0))
            return max(-1.0, min(1.0, target - competing))
        target = max(scores.get("substantive", 0.0), scores.get("reference", 0.0))
        return max(-1.0, min(1.0, target - scores.get("editorial", 0.0)))

    def _payload_float(self, hit: RetrievalHit, field_name: str) -> float:
        payload = hit.payload or {}
        try:
            return float(payload.get(field_name) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _boilerplate_score(self, hit: RetrievalHit) -> float:
        return max(0.0, self._payload_float(hit, "boilerplate_score"))

    def _label_expressiveness_score(self, hit: RetrievalHit) -> float:
        payload = hit.payload or {}
        unit_type = str(payload.get("unit_type") or "").strip().lower()
        unit_number = str(payload.get("unit_number") or "").strip()
        canonical_label = str(payload.get("canonical_label") or "").strip()
        path = self._as_path(payload.get("hierarchy_path") or payload.get("section_path"))
        score = 0.0
        if canonical_label:
            score += 0.18
        if unit_number:
            score += 0.16
        if unit_type in {"article", "section", "recommendation", "annex", "chapter", "title", "book"}:
            score += 0.22
        if len(path) >= 2:
            score += min(0.18, 0.05 * len(path))
        return min(0.7, score)

    def _rank_answer_hits(
        self,
        question: str,
        hits: list[RetrievalHit],
        *,
        evidence_intent: EvidenceIntentClassification,
    ) -> list[RetrievalHit]:
        if not hits:
            return hits
        return sorted(
            hits,
            key=lambda hit: (
                (
                    self._answer_focus_score(question, hit)
                    + (0.35 * self._payload_float(hit, "segment_score"))
                    + (0.35 * self._payload_float(hit, "segment_coverage_score"))
                    + (0.2 * self._label_expressiveness_score(hit))
                    - (0.9 * self._boilerplate_score(hit))
                ),
                self._evidence_role_alignment_score(evidence_intent, hit),
                hit.rerank_score,
            ),
            reverse=True,
        )

    def _answer_focus_score(self, question: str, hit: RetrievalHit) -> float:
        payload = hit.payload or {}
        query_tokens = self._tokens(question)
        evidence_text = str(payload.get("segment_text") or hit.raw_text or "")
        topical_tokens = self._tokens(
            " ".join(
                [
                    str(payload.get("canonical_label") or ""),
                    str(payload.get("unit_topic") or ""),
                    str(payload.get("path_text") or ""),
                    str(payload.get("focus_terms_text") or ""),
                    str(payload.get("branch_summary_context") or ""),
                    str(payload.get("branch_keywords_text") or ""),
                    str(payload.get("reference_variants_text") or ""),
                    evidence_text[:900],
                ]
            )
        )
        lexical = max(
            self._overlap_ratio(query_tokens, topical_tokens),
            self._soft_overlap_ratio(query_tokens, topical_tokens),
        )
        path_depth = len(self._as_path(payload.get("hierarchy_path") or payload.get("section_path")))
        unit_type = str(payload.get("unit_type") or "").strip().lower()
        specificity = min(0.35, 0.08 * path_depth)
        if unit_type and unit_type not in {"document", "unknown"}:
            specificity += 0.12
        if str(payload.get("chunk_kind") or "").strip().lower() == "preamble":
            specificity -= 0.18
        if unit_type == "document" and path_depth <= 1:
            specificity -= 0.22
        specificity += min(0.18, 0.12 * self._payload_float(hit, "segment_coverage_score"))
        specificity -= 0.35 * self._boilerplate_score(hit)
        return lexical + specificity

    def _is_comparison_query(self, question: str) -> bool:
        normalized = self._normalize(question)
        return bool(
            " y cual " in normalized
            or normalized.startswith("que documento")
            or normalized.startswith("que norma")
            or " diferencia " in normalized
        )

    def _extract_compare_subqueries(self, question: str, route: QueryRoute) -> list[str]:
        normalized = self._normalize(question)
        if not self._is_comparison_query(question):
            return []
        if "technical_standard" not in route.target_classes and route.query_type != "technical_standard":
            return []

        match = re.match(r"^(que\s+(?:documento|norma|parte))\s+(.+?)\s+y\s+cual\s+(.+)$", normalized)
        if match:
            subject = match.group(1)
            first = f"{subject} {match.group(2).strip(' ?')}"
            second = f"{subject} {match.group(3).strip(' ?')}"
            return [first, second]

        match = re.match(r"^(que\s+diferencia\s+hay\s+entre)\s+(.+?)\s+y\s+(.+)$", normalized)
        if match:
            return [match.group(2).strip(" ?"), match.group(3).strip(" ?")]

        return []

    def _comparison_clause_queries(self, question: str, route: QueryRoute) -> list[str]:
        return self._extract_compare_subqueries(question, route)

    def _compress_text(self, text: str, limit: int) -> str:
        normalized = " ".join((text or "").split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."

    def _question_focused_evidence_text(self, question: str, hit: RetrievalHit, *, limit: int) -> str:
        payload = hit.payload or {}
        raw_text = " ".join((str(payload.get("segment_text") or hit.raw_text or "")).split())
        if not raw_text:
            return ""

        # Si el pasaje entra completo, se envia entero. Seleccionar un fragmento
        # cuando no hace falta era la causa de respuestas del tipo "el articulo
        # dice que debe contener X, pero el fragmento no enumera cuales": el
        # encabezado ganaba por solapamiento lexico y la lista se descartaba.
        if len(raw_text) <= limit:
            return raw_text

        fragments = self._candidate_evidence_fragments(raw_text)
        if not fragments:
            return self._compress_text(raw_text, limit)

        query_tokens = self._tokens(question)
        label_text = self._normalize(
            " ".join(
                filter(
                    None,
                    [
                        str(payload.get("canonical_label") or ""),
                        str(payload.get("unit_topic") or ""),
                        str(payload.get("path_text") or ""),
                        str(payload.get("unit_key") or ""),
                    ],
                )
            )
        )

        best_fragment = fragments[0]
        best_score = float("-inf")
        for fragment in fragments:
            normalized_fragment = self._normalize(fragment)
            fragment_tokens = self._tokens(fragment)
            score = max(
                self._overlap_ratio(query_tokens, fragment_tokens),
                self._soft_overlap_ratio(query_tokens, fragment_tokens),
            )
            if label_text and label_text in normalized_fragment:
                score += 0.2
            if score > best_score:
                best_score = score
                best_fragment = fragment

        return self._compress_text(best_fragment, limit)

    def _candidate_evidence_fragments(self, text: str) -> list[str]:
        normalized = " ".join((text or "").split())
        if not normalized:
            return []

        raw_fragments = [
            fragment.strip(" -|")
            for fragment in re.split(r"(?:(?<=[\.;:])\s+|\s-\s+|\s\|\s+)", normalized)
            if fragment.strip(" -|")
        ]
        if not raw_fragments:
            return []

        fragments: list[str] = []
        for index, fragment in enumerate(raw_fragments):
            if len(fragment) >= 40:
                fragments.append(fragment)
            if index + 1 < len(raw_fragments):
                window = f"{fragment}. {raw_fragments[index + 1]}".strip()
                if len(window) >= 60:
                    fragments.append(window)

        deduped: list[str] = []
        seen: set[str] = set()
        for fragment in fragments:
            key = self._normalize(fragment)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(fragment)
        return deduped[:18]

    def _truncate_reason(self, reason: str) -> str:
        reason = (reason or "").strip()
        return reason if len(reason) <= 180 else f"{reason[:177]}..."

    def _as_path(self, value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        return []

    def _branch_key(self, hit: RetrievalHit) -> tuple[str, str, tuple[str, ...]] | None:
        parent_path = self._as_path(hit.payload.get("parent_path"))
        hierarchy_path = self._as_path(hit.payload.get("hierarchy_path") or hit.payload.get("section_path"))
        path = parent_path or hierarchy_path
        if not path:
            return None
        return (hit.document_id, hit.version_id, tuple(path))

    def _branch_matches(
        self,
        left: tuple[str, str, tuple[str, ...]],
        right: tuple[str, str, tuple[str, ...]],
    ) -> bool:
        if left[:2] != right[:2]:
            return False
        left_path = list(left[2])
        right_path = list(right[2])
        if len(left_path) < len(right_path):
            return False
        return left_path[: len(right_path)] == right_path

    def _dedupe_queries(self, queries: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries:
            normalized = self._normalize(query)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(query)
        return deduped

    def _dedupe_strings(self, values: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = (value or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    def _select_branch_keys(
        self,
        question: str,
        branch_hits: list[RetrievalHit],
        plan: QueryPlan,
        *,
        evidence_intent: EvidenceIntentClassification,
    ) -> tuple[list[str], list[str]]:
        if not branch_hits:
            return [], ["branch_select:provider=empty"]
        branch_limit = {
            "focal": 4,
            "multi_branch": 6,
            "global": 8,
        }.get(plan.strategy, 4)
        candidates = self._build_branch_candidates(branch_hits)
        if not candidates:
            return [], ["branch_select:provider=empty"]

        scored_keys: list[str] | None = None
        scorer_result = self.branch_centrality_scorer.score(
            question=question,
            strategy=plan.strategy,
            evidence_intent=evidence_intent.intent,
            candidates=candidates,
        )
        decision_trace = list(scorer_result.decision_trace) if scorer_result is not None else ["branch_select:provider=fallback_order"]
        if scorer_result is not None:
            candidate_map = {candidate.branch_key: candidate for candidate in candidates}
            ranked_predictions = sorted(
                scorer_result.predictions,
                key=lambda prediction: self._branch_selection_score(
                    probability=prediction.probability,
                    candidate=candidate_map.get(prediction.branch_key),
                ),
                reverse=True,
            )
            scored_keys = [prediction.branch_key for prediction in ranked_predictions[:branch_limit]]

        if scored_keys is None:
            scored_keys = [candidate.branch_key for candidate in candidates[:branch_limit]]
        return self._dedupe_strings(scored_keys)[:branch_limit], decision_trace

    def _build_branch_candidates(self, branch_hits: list[RetrievalHit]) -> list[BranchCentralityCandidate]:
        candidates: list[BranchCentralityCandidate] = []
        seen: set[str] = set()
        for hit in branch_hits:
            payload = hit.payload or {}
            branch_key = str(payload.get("branch_key") or hit.chunk_id or "").strip()
            if not branch_key or branch_key in seen:
                continue
            seen.add(branch_key)
            boilerplate_score = float(payload.get("boilerplate_score") or 0.0)
            candidates.append(
                BranchCentralityCandidate(
                    branch_key=branch_key,
                    canonical_label=str(payload.get("canonical_label") or self._display_branch_label(hit)).strip(),
                    unit_type=str(payload.get("unit_type") or "").strip(),
                    unit_number=str(payload.get("unit_number") or "").strip(),
                    unit_topic=str(payload.get("unit_topic") or "").strip(),
                    path_text=str(payload.get("path_text") or "").strip(),
                    reference_variants_text=str(payload.get("reference_variants_text") or "").strip(),
                    sample_questions_text=str(payload.get("sample_questions_text") or "").strip(),
                    focus_terms_text=str(payload.get("focus_terms_text") or "").strip(),
                    branch_summary_context=str(payload.get("branch_summary_context") or "").strip(),
                    branch_keywords_text=str(payload.get("branch_keywords_text") or "").strip(),
                    branch_summary=str(payload.get("raw_text") or hit.raw_text or "").strip(),
                    branch_score=max(0.0, float(hit.rrf_score or 0.0) - (0.5 * boilerplate_score)),
                    boilerplate_score=boilerplate_score,
                )
            )
        return candidates

    def _branch_selection_score(
        self,
        *,
        probability: float,
        candidate: BranchCentralityCandidate | None,
    ) -> float:
        base_score = 0.88 * float(probability)
        if candidate is None:
            return base_score
        return (
            base_score
            + (0.12 * max(0.0, float(candidate.branch_score)))
            - (0.08 * max(0.0, float(candidate.boilerplate_score)))
        )

    def _branch_target_unit_score(self, hit: RetrievalHit, target_units: set[str]) -> float:
        if not target_units:
            return 0.0
        unit_type = str(hit.payload.get("unit_type") or "").strip().lower()
        if unit_type and unit_type in target_units:
            return 0.4
        return 0.0

    def _branch_target_number_score(self, hit: RetrievalHit, target_numbers: set[str]) -> float:
        if not target_numbers:
            return 0.0
        unit_number = str(hit.payload.get("unit_number") or "").strip().upper()
        if unit_number and unit_number in target_numbers:
            return 0.35
        return 0.0

    def _branch_specificity_score(self, hit: RetrievalHit, plan: QueryPlan) -> float:
        path = self._as_path(hit.payload.get("hierarchy_path") or hit.payload.get("section_path"))
        path_depth = len(path)
        unit_type = str(hit.payload.get("unit_type") or "").strip().lower()
        unit_number = str(hit.payload.get("unit_number") or "").strip()
        score = min(0.2, 0.06 * path_depth)
        if unit_number:
            score += 0.12
        if unit_type and unit_type not in {"document", "unknown"}:
            score += 0.1
        if plan.strategy in {"focal", "multi_branch"} and unit_type == "document" and not unit_number and path_depth <= 1:
            score -= 0.3 if plan.strategy == "focal" else 0.18
        return score

    def _is_specific_branch_hit(self, hit: RetrievalHit) -> bool:
        payload = hit.payload or {}
        unit_type = str(payload.get("unit_type") or "").strip().lower()
        unit_number = str(payload.get("unit_number") or "").strip()
        label = self._normalize(str(payload.get("canonical_label") or ""))
        path = self._as_path(payload.get("hierarchy_path") or payload.get("section_path"))
        if unit_type and unit_type not in {"document", "unknown"}:
            return True
        if unit_number:
            return True
        if any(token in label for token in ("art", "seccion", "recomendacion", "anexo", "capitulo", "titulo", "libro", "parrafo")):
            return True
        return len(path) > 1

    def _extract_structural_references(self, question: str) -> list[StructuralReference]:
        question_norm = self._normalize(question)
        refs: list[StructuralReference] = []
        seen: set[tuple[str, str]] = set()
        patterns = (
            ("article", r"\b(?:art\.?|arto\.?|articulo)\s*(\d+)\b"),
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
        for kind, pattern in patterns:
            for match in re.finditer(pattern, question_norm):
                value = match.group(1).upper()
                key = (kind, value)
                if key not in seen:
                    seen.add(key)
                    refs.append(StructuralReference(kind=kind, number=value))
        return refs

    def _build_sparse_reference_clauses(self, question: str) -> list[dict[str, object]]:
        clauses: list[dict[str, object]] = []
        for ref in self._extract_structural_references(question):
            clauses.extend(self._reference_clauses(ref))
        return clauses

    def _build_sparse_phrase_clauses(self, question: str) -> list[dict[str, object]]:
        del question
        return []

    def _reference_clauses(self, ref: StructuralReference) -> list[dict[str, object]]:
        number = ref.number
        unit_key = f"{ref.kind}:{number}"
        if ref.kind == "article":
            return [
                {"term": {"unit_key": {"value": unit_key, "boost": 13.0}}},
                {"match_phrase": {"canonical_label": {"query": f"Arto. {number}", "boost": 12.0}}},
                {"match_phrase": {"reference_variants_text": {"query": f"Arto. {number}", "boost": 12.0}}},
                {"term": {"article_label": {"value": f"Arto. {number}", "boost": 12.0}}},
                {"term": {"article_label": {"value": f"Art. {number}", "boost": 10.0}}},
                {"match_phrase": {"path_text": {"query": f"Arto. {number}", "boost": 8.0}}},
                {"match_phrase": {"retrieval_context": {"query": f"Arto. {number}", "boost": 7.0}}},
                {"match_phrase": {"raw_text": {"query": f"Art. {number}", "boost": 6.0}}},
            ]
        if ref.kind == "section":
            return [
                {"term": {"unit_key": {"value": unit_key, "boost": 13.0}}},
                {"match_phrase": {"canonical_label": {"query": f"Seccion {number}", "boost": 12.0}}},
                {"match_phrase": {"reference_variants_text": {"query": f"Sección {number}", "boost": 12.0}}},
                {"prefix": {"section_label": {"value": f"Sección {number}", "boost": 12.0}}},
                {"match_phrase": {"path_text": {"query": f"Sección {number}", "boost": 9.0}}},
                {"match_phrase": {"retrieval_context": {"query": f"Sección {number}", "boost": 8.0}}},
                {"match_phrase": {"raw_text": {"query": f"Sección {number}", "boost": 7.0}}},
            ]
        if ref.kind == "recommendation":
            return [
                {"term": {"unit_key": {"value": unit_key, "boost": 13.0}}},
                {"match_phrase": {"canonical_label": {"query": f"Recomendacion {number}", "boost": 11.0}}},
                {"match_phrase": {"reference_variants_text": {"query": f"Recomendación {number}", "boost": 11.0}}},
                {"match_phrase": {"path_text": {"query": f"Recomendación {number}", "boost": 10.0}}},
                {"match_phrase": {"retrieval_context": {"query": f"Recomendación {number}", "boost": 9.0}}},
                {"match_phrase": {"raw_text": {"query": f"Recomendación {number}", "boost": 8.0}}},
                {"match_phrase": {"raw_text": {"query": f"R. {number}", "boost": 6.0}}},
                {"prefix": {"subsection_label": {"value": f"{number}.", "boost": 5.0}}},
                {"prefix": {"section_label": {"value": f"{number}.", "boost": 4.0}}},
            ]
        if ref.kind == "annex":
            return [
                {"term": {"unit_key": {"value": unit_key, "boost": 13.0}}},
                {"match_phrase": {"canonical_label": {"query": f"Anexo {number}", "boost": 12.0}}},
                {"match_phrase": {"reference_variants_text": {"query": f"Anexo {number}", "boost": 12.0}}},
                {"match_phrase": {"path_text": {"query": f"Anexo {number}", "boost": 10.0}}},
                {"match_phrase": {"retrieval_context": {"query": f"Anexo {number}", "boost": 8.0}}},
            ]
        if ref.kind in {"chapter", "title", "book", "paragraph"}:
            label = {
                "chapter": "Capitulo",
                "title": "Titulo",
                "book": "Libro",
                "paragraph": "Parrafo",
            }[ref.kind]
            return [
                {"term": {"unit_key": {"value": unit_key, "boost": 12.0}}},
                {"match_phrase": {"canonical_label": {"query": f"{label} {number}", "boost": 10.0}}},
                {"match_phrase": {"reference_variants_text": {"query": f"{label} {number}", "boost": 10.0}}},
                {"match_phrase": {"path_text": {"query": f"{label} {number}", "boost": 8.0}}},
                {"match_phrase": {"retrieval_context": {"query": f"{label} {number}", "boost": 7.0}}},
            ]
        if ref.kind == "law":
            return [
                {"match_phrase": {"path_text": {"query": f"Ley {number}", "boost": 4.0}}},
                {"match_phrase": {"retrieval_context": {"query": f"Ley {number}", "boost": 4.0}}},
                {"match_phrase": {"raw_text": {"query": f"Ley {number}", "boost": 4.0}}},
            ]
        return []

    def _structural_reference_score(self, refs: list[StructuralReference], hit: RetrievalHit) -> float:
        if not refs:
            return 0.0

        score = 0.0
        path_values = self._as_path(hit.payload.get("hierarchy_path") or hit.payload.get("section_path"))
        joined_path = self._normalize(" ".join(path_values))
        article_label = self._normalize(str(hit.payload.get("article_label") or ""))
        section_label = self._normalize(str(hit.payload.get("section_label") or ""))
        subsection_label = self._normalize(str(hit.payload.get("subsection_label") or ""))
        canonical_label = self._normalize(str(hit.payload.get("canonical_label") or ""))
        reference_variants = self._normalize(str(hit.payload.get("reference_variants_text") or ""))
        unit_key = self._normalize(str(hit.payload.get("unit_key") or ""))
        retrieval_context = self._normalize(hit.retrieval_context)
        raw_prefix = self._normalize(hit.raw_text[:500])

        for ref in refs:
            number = ref.number
            if ref.kind == "article":
                if unit_key == f"article:{number}".lower():
                    score += 0.75
                if article_label in {f"arto. {number}", f"art. {number}", f"articulo {number}"}:
                    score += 0.7
                elif any(token in joined_path for token in (f"arto. {number}", f"art. {number}", f"articulo {number}")):
                    score += 0.45
            elif ref.kind == "section":
                if unit_key == f"section:{number}".lower():
                    score += 0.75
                if section_label.startswith(f"seccion {number}"):
                    score += 0.7
                elif f"seccion {number}" in joined_path or f"seccion {number}" in retrieval_context:
                    score += 0.45
            elif ref.kind == "recommendation":
                if unit_key == f"recommendation:{number}".lower():
                    score += 0.75
                if f"recomendacion {number}" in joined_path or f"recomendacion {number}" in retrieval_context:
                    score += 0.6
                elif subsection_label.startswith(f"{number}.") or section_label.startswith(f"{number}."):
                    score += 0.35
                elif f"recomendacion {number}" in raw_prefix or f"r. {number}" in raw_prefix:
                    score += 0.3
            elif ref.kind == "annex":
                if unit_key == f"annex:{number}".lower():
                    score += 0.75
                elif f"anexo {number.lower()}" in joined_path or f"anexo {number.lower()}" in canonical_label:
                    score += 0.5
                elif f"anexo {number.lower()}" in reference_variants:
                    score += 0.35
            elif ref.kind in {"chapter", "title", "book", "paragraph"}:
                if unit_key == f"{ref.kind}:{number}".lower():
                    score += 0.65
                elif f"{ref.kind} {number.lower()}".replace("paragraph", "parrafo") in canonical_label:
                    score += 0.4
            elif ref.kind == "law":
                if f"ley {number}" in retrieval_context or f"ley {number}" in raw_prefix:
                    score += 0.15
        return score

    def _structural_reference_mismatch_penalty(
        self,
        refs: list[StructuralReference],
        hit: RetrievalHit,
    ) -> float:
        if not refs:
            return 0.0

        penalty = 0.0
        path_values = self._as_path(hit.payload.get("hierarchy_path") or hit.payload.get("section_path"))
        joined_path = self._normalize(" ".join(path_values))
        article_label = self._normalize(str(hit.payload.get("article_label") or ""))
        section_label = self._normalize(str(hit.payload.get("section_label") or ""))
        subsection_label = self._normalize(str(hit.payload.get("subsection_label") or ""))
        raw_prefix = self._normalize(hit.raw_text[:300])

        for ref in refs:
            if ref.kind == "article":
                article_match = re.search(r"\b(?:arto\.|art\.|articulo)\s*(\d+)\b", article_label or joined_path)
                if article_match and article_match.group(1) != ref.number:
                    penalty += 0.6
            elif ref.kind == "section":
                section_match = re.search(r"\bseccion\s*(\d+)\b", section_label or joined_path or raw_prefix)
                if section_match and section_match.group(1) != ref.number:
                    penalty += 0.45
            elif ref.kind == "recommendation":
                recommendation_match = re.search(
                    r"\brecomendacion\s*(\d+)\b",
                    " ".join(filter(None, [joined_path, subsection_label, section_label, raw_prefix])),
                )
                if recommendation_match and recommendation_match.group(1) != ref.number:
                    penalty += 0.75
                elif subsection_label.startswith(tuple(f"{value}." for value in range(1, 41))):
                    leading_number = subsection_label.split(".", 1)[0]
                    if leading_number.isdigit() and leading_number != ref.number:
                        penalty += 0.35
        return penalty
