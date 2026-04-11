import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from sqlalchemy.orm import Session

from app.models.contracts import (
    QueryRequest,
    RetrievalEvalCase,
    RetrievalEvalCaseResult,
    RetrievalEvalExpectation,
    RetrievalEvalHitPreview,
    RetrievalEvalRequest,
    RetrievalEvalResponse,
    RetrievalEvalSuiteSummary,
    RetrievalEvalTargetSelector,
)
from app.models.database import VersionRecord
from app.services.retrieval import RetrievalRun, RetrievalService


class RetrievalEvaluationService:
    def __init__(self, settings, retrieval_service: RetrievalService | None = None) -> None:
        self.settings = settings
        self.retrieval = retrieval_service or RetrievalService(settings)

    def evaluate(
        self,
        *,
        db: Session,
        request: RetrievalEvalRequest | None = None,
    ) -> RetrievalEvalResponse:
        request = request or RetrievalEvalRequest()
        fixture_paths = self._resolve_fixture_paths(request)
        cases: list[RetrievalEvalCase] = []
        for fixture_path in fixture_paths:
            cases.extend(self._load_cases(fixture_path))
        cases = self._filter_cases(cases, request=request)

        results: list[RetrievalEvalCaseResult] = []
        for case in cases:
            results.append(self._evaluate_case(db=db, case=case, user_identity=request.user_identity))

        summary_results = results
        visible_results = results if request.include_passed else [item for item in results if not item.passed]

        return RetrievalEvalResponse(
            fixture_path=str(fixture_paths[0]),
            fixture_paths=[str(path) for path in fixture_paths],
            total_cases=len(summary_results),
            passed_cases=sum(1 for item in summary_results if item.passed),
            planner_accuracy=self._ratio(sum(1 for item in summary_results if item.planner_ok), len(summary_results)),
            router_accuracy=self._ratio(sum(1 for item in summary_results if item.router_ok), len(summary_results)),
            hit_at_1=self._ratio(sum(1 for item in summary_results if item.hit_at_1), len(summary_results)),
            hit_at_3=self._ratio(sum(1 for item in summary_results if item.hit_at_3), len(summary_results)),
            hit_at_5=self._ratio(sum(1 for item in summary_results if item.hit_at_5), len(summary_results)),
            mean_reciprocal_rank=round(
                sum(item.reciprocal_rank for item in summary_results) / len(summary_results), 3
            )
            if summary_results
            else 0.0,
            citation_precision_at_3=round(
                sum(item.citation_precision_at_3 for item in summary_results) / len(summary_results), 3
            )
            if summary_results
            else 0.0,
            no_result_rate=self._ratio(
                sum(1 for item in summary_results if item.fallback_reason == "no_retrieval_hits"),
                len(summary_results),
            ),
            by_suite=self._build_suite_summaries(summary_results),
            results=visible_results,
        )

    def _evaluate_case(
        self,
        *,
        db: Session,
        case: RetrievalEvalCase,
        user_identity: str,
    ) -> RetrievalEvalCaseResult:
        notes: list[str] = []
        selectors = self._selectors(case.expectation)
        resolved_targets = [
            self._resolve_target_version(db=db, selector=selector, notes=notes) for selector in selectors
        ]
        resolved_targets = [item for item in resolved_targets if item is not None]

        request = QueryRequest(
            question=case.question,
            user_identity=user_identity,
            filters=case.filters,
            conversation_context=[],
        )
        run = self.retrieval.run(request)

        expected_version_ids = [item.version_id for item in resolved_targets]
        expected_document_ids = [item.document_id for item in resolved_targets]
        if not expected_version_ids and case.expectation.target.version_id:
            expected_version_ids = [case.expectation.target.version_id]
        if not expected_document_ids and case.expectation.target.document_id:
            expected_document_ids = [case.expectation.target.document_id]
        expected_version_id = expected_version_ids[0] if expected_version_ids else None
        expected_document_id = expected_document_ids[0] if expected_document_ids else None
        expected_class = case.expectation.document_class
        planner_strategy = self._extract_planner_strategy(run)
        planner_ok = case.strategy is None or planner_strategy == case.strategy
        router_ok = expected_class in run.route.target_classes

        reciprocal_rank = self._reciprocal_rank(run=run, expected_version_ids=expected_version_ids)
        hit_at_1 = self._hit_at_k(
            run=run,
            expected_version_ids=expected_version_ids,
            k=1,
            mode=case.expectation.target_match_mode,
        )
        hit_at_3 = self._hit_at_k(
            run=run,
            expected_version_ids=expected_version_ids,
            k=3,
            mode=case.expectation.target_match_mode,
        )
        hit_at_5 = self._hit_at_k(
            run=run,
            expected_version_ids=expected_version_ids,
            k=5,
            mode=case.expectation.target_match_mode,
        )
        citation_precision_at_3 = self._citation_precision_at_k(
            run=run,
            expected_version_ids=expected_version_ids,
            k=3,
        )

        evaluation_top_k = max(1, case.expectation.pass_rank)
        visible_top_k = min(3, evaluation_top_k)
        evidence_haystack = self._aggregate_top_hit_text(run=run, top_k=evaluation_top_k)
        visible_evidence_haystack = self._aggregate_top_hit_text(run=run, top_k=visible_top_k)
        label_hints = self._expected_label_hints(case)
        label_haystack = self._aggregate_hit_labels(run=run, top_k=evaluation_top_k)
        visible_label_haystack = self._aggregate_hit_labels(run=run, top_k=visible_top_k)
        required_terms_ok = self._contains_all(
            evidence_haystack,
            case.expectation.must_contain_all,
        )
        optional_terms_ok = self._contains_any(
            evidence_haystack,
            case.expectation.should_contain_any,
        )
        visible_required_terms_ok = self._contains_all(
            visible_evidence_haystack,
            case.expectation.must_contain_all,
        )
        visible_optional_terms_ok = self._contains_any(
            visible_evidence_haystack,
            case.expectation.should_contain_any,
        )
        label_terms_ok = self._contains_any(label_haystack, label_hints)
        visible_label_terms_ok = self._contains_any(visible_label_haystack, label_hints)
        visible_hit_at_k = self._hit_at_k(
            run=run,
            expected_version_ids=expected_version_ids,
            k=visible_top_k,
            mode=case.expectation.target_match_mode,
        )

        if not resolved_targets:
            notes.append("target_version_unresolved")

        if not planner_ok:
            notes.append("planner_strategy_mismatch")
        if not router_ok:
            notes.append("router_mismatch")
        if expected_version_ids and not self._hit_at_k(
            run=run,
            expected_version_ids=expected_version_ids,
            k=evaluation_top_k,
            mode=case.expectation.target_match_mode,
        ):
            notes.append(
                f"expected_targets_missing_top_{evaluation_top_k}"
            )
        if expected_version_ids and not visible_hit_at_k:
            notes.append(f"visible_targets_missing_top_{visible_top_k}")
        if not required_terms_ok:
            notes.append("required_terms_missing")
        if case.expectation.should_contain_any and not optional_terms_ok:
            notes.append("optional_terms_missing")
        if not visible_required_terms_ok:
            notes.append(f"visible_required_terms_missing_top_{visible_top_k}")
        if case.expectation.should_contain_any and not visible_optional_terms_ok:
            notes.append(f"visible_optional_terms_missing_top_{visible_top_k}")
        if label_hints and not label_terms_ok:
            notes.append("expected_label_missing")
        if label_hints and not visible_label_terms_ok:
            notes.append(f"visible_expected_label_missing_top_{visible_top_k}")

        passed = (
            planner_ok
            and
            router_ok
            and self._hit_at_k(
                run=run,
                expected_version_ids=expected_version_ids,
                k=evaluation_top_k,
                mode=case.expectation.target_match_mode,
            )
            and required_terms_ok
            and optional_terms_ok
            and visible_hit_at_k
            and visible_required_terms_ok
            and visible_optional_terms_ok
            and label_terms_ok
            and visible_label_terms_ok
        )

        return RetrievalEvalCaseResult(
            case_id=case.case_id,
            suite_id=case.suite_id,
            question=case.question,
            expected_strategy=case.strategy,
            planner_strategy=planner_strategy,
            planner_ok=planner_ok,
            expected_document_class=expected_class,
            expected_document_family=case.expectation.document_family,
            target_match_mode=case.expectation.target_match_mode,
            expected_version_id=expected_version_id,
            expected_document_id=expected_document_id,
            expected_version_ids=expected_version_ids,
            expected_document_ids=expected_document_ids,
            router_query_type=run.route.query_type,
            router_ok=router_ok,
            hit_at_1=hit_at_1,
            hit_at_3=hit_at_3,
            hit_at_5=hit_at_5,
            visible_hit_at_k=visible_hit_at_k,
            visible_top_k=visible_top_k,
            reciprocal_rank=round(reciprocal_rank, 3),
            citation_precision_at_3=round(citation_precision_at_3, 3),
            required_terms_ok=required_terms_ok,
            optional_terms_ok=optional_terms_ok,
            visible_required_terms_ok=visible_required_terms_ok,
            visible_optional_terms_ok=visible_optional_terms_ok,
            label_terms_ok=label_terms_ok,
            visible_label_terms_ok=visible_label_terms_ok,
            passed=passed,
            fallback_reason=run.response.fallback_reason,
            decision_trace=run.decision_trace,
            top_hits=self._build_hit_previews(run),
            notes=notes,
        )

    def _resolve_fixture_paths(self, request: RetrievalEvalRequest) -> list[Path]:
        raw_paths = list(request.fixture_paths)
        if request.fixture_path:
            raw_paths.insert(0, request.fixture_path)
        if not raw_paths:
            raw_paths = [str(Path(__file__).resolve().parents[1] / "evals" / "retrieval_eval_cases.json")]

        resolved: list[Path] = []
        seen: set[str] = set()
        for raw_path in raw_paths:
            path = Path(raw_path).expanduser().resolve()
            normalized = str(path)
            if normalized in seen:
                continue
            seen.add(normalized)
            resolved.append(path)
        return resolved

    def _load_cases(self, fixture_path: Path) -> list[RetrievalEvalCase]:
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        cases = payload.get("cases", payload)
        return [RetrievalEvalCase.model_validate(item) for item in cases]

    def _filter_cases(
        self,
        cases: list[RetrievalEvalCase],
        *,
        request: RetrievalEvalRequest,
    ) -> list[RetrievalEvalCase]:
        filtered = cases
        if request.suite_ids:
            allowed = set(request.suite_ids)
            filtered = [case for case in filtered if case.suite_id in allowed]
        if request.case_ids:
            allowed = set(request.case_ids)
            filtered = [case for case in filtered if case.case_id in allowed]
        return filtered

    def _resolve_target_version(
        self,
        *,
        db: Session,
        selector: RetrievalEvalTargetSelector,
        notes: list[str],
    ) -> VersionRecord | None:
        query = db.query(VersionRecord).filter(VersionRecord.version_status == "active")

        if selector.version_id:
            query = query.filter(VersionRecord.version_id == selector.version_id)
        if selector.document_id:
            query = query.filter(VersionRecord.document_id == selector.document_id)
        if selector.storage_uri_contains:
            query = query.filter(VersionRecord.storage_uri.contains(selector.storage_uri_contains))
        if selector.document_class:
            query = query.filter(VersionRecord.document_class == selector.document_class)
        if selector.document_family:
            query = query.filter(VersionRecord.document_family == selector.document_family)

        rows = query.order_by(VersionRecord.created_at.desc()).limit(2).all()
        if not rows:
            return None
        if len(rows) > 1:
            notes.append("multiple_target_versions_resolved_using_latest")
        return rows[0]

    def _selectors(self, expectation: RetrievalEvalExpectation) -> list[RetrievalEvalTargetSelector]:
        if expectation.targets:
            return expectation.targets
        return [expectation.target]

    def _reciprocal_rank(self, *, run: RetrievalRun, expected_version_ids: list[str]) -> float:
        if not expected_version_ids:
            return 0.0
        for rank, hit in enumerate(run.final_hits, start=1):
            if hit.version_id in expected_version_ids:
                return 1.0 / rank
        return 0.0

    def _hit_at_k(
        self,
        *,
        run: RetrievalRun,
        expected_version_ids: list[str],
        k: int,
        mode: str = "any",
    ) -> bool:
        if not expected_version_ids:
            return False
        observed = {hit.version_id for hit in run.final_hits[:k]}
        expected = set(expected_version_ids)
        if mode == "all":
            return expected.issubset(observed)
        return any(version_id in observed for version_id in expected)

    def _citation_precision_at_k(
        self,
        *,
        run: RetrievalRun,
        expected_version_ids: list[str],
        k: int,
    ) -> float:
        if not expected_version_ids:
            return 0.0
        top_hits = run.final_hits[:k]
        if not top_hits:
            return 0.0
        expected = set(expected_version_ids)
        matches = sum(1 for hit in top_hits if hit.version_id in expected)
        return matches / len(top_hits)

    def _aggregate_top_hit_text(self, *, run: RetrievalRun, top_k: int) -> str:
        parts: list[str] = []
        for hit in run.final_hits[:top_k]:
            path = " > ".join(self._as_path(hit.payload.get("hierarchy_path") or hit.payload.get("section_path")))
            parts.append(path)
            parts.append(hit.retrieval_context)
            parts.append(hit.raw_text)
        return self._normalize(" ".join(parts))

    def _aggregate_hit_labels(self, *, run: RetrievalRun, top_k: int) -> str:
        parts: list[str] = []
        for hit in run.final_hits[:top_k]:
            payload = hit.payload or {}
            parts.extend(
                [
                    " > ".join(self._as_path(payload.get("hierarchy_path") or payload.get("section_path"))),
                    str(payload.get("canonical_label") or ""),
                    str(payload.get("reference_variants_text") or ""),
                    str(payload.get("path_text") or ""),
                    hit.retrieval_context,
                    hit.raw_text[:350],
                ]
            )
        return self._normalize(" ".join(parts))

    def _expected_label_hints(self, case: RetrievalEvalCase) -> list[str]:
        hints: list[str] = []

        def add(value: str) -> None:
            normalized = self._normalize(value)
            if normalized and normalized not in hints:
                hints.append(normalized)

        for hint in case.expectation.expected_label_hints:
            add(hint)

        note = (case.notes or "").strip()
        if note.lower().startswith("expected_label:"):
            raw_label = note.split(":", 1)[1].strip()
            for segment in [part.strip() for part in raw_label.split(">") if part.strip()]:
                self._expand_label_segment(segment, add)

        return hints

    def _expand_label_segment(self, segment: str, add) -> None:
        add(segment)
        normalized = self._normalize(segment)

        article_range = re.search(r"\barts?\.?\s*(\d+)\s*-\s*(\d+)\b", normalized)
        if article_range:
            start = int(article_range.group(1))
            end = int(article_range.group(2))
            for number in range(start, end + 1):
                add(f"Arto. {number}")
                add(f"Art. {number}")
            return

        article_match = re.search(r"\b(?:arts?\.?|arto\.?|articulo)\s*(\d+)\b", normalized)
        if article_match:
            number = article_match.group(1)
            add(f"Arto. {number}")
            add(f"Art. {number}")
            add(f"Articulo {number}")
            return

        section_match = re.search(r"\bseccion\s*(\d+(?:\.\d+)?)\b", normalized)
        if section_match:
            number = section_match.group(1)
            add(f"Seccion {number}")
            add(f"Sección {number}")

        recommendation_match = re.search(r"\brecomendacion\s*(\d+(?:\.\d+)?)\b", normalized)
        if recommendation_match:
            number = recommendation_match.group(1)
            add(f"Recomendacion {number}")
            add(f"Recomendación {number}")
            add(f"R. {number}")

        annex_match = re.search(r"\banexo\s+([ivxlcdm]+|\d+)\b", normalized)
        if annex_match:
            value = annex_match.group(1).upper()
            add(f"Anexo {value}")

        paragraph_match = re.search(r"\b(\d+\.\d+)\b", normalized)
        if paragraph_match:
            add(paragraph_match.group(1))

    def _contains_all(self, haystack: str, needles: list[str]) -> bool:
        if not needles:
            return True
        return all(self._normalize(needle) in haystack for needle in needles)

    def _contains_any(self, haystack: str, needles: list[str]) -> bool:
        if not needles:
            return True
        return any(self._normalize(needle) in haystack for needle in needles)

    def _build_hit_previews(self, run: RetrievalRun) -> list[RetrievalEvalHitPreview]:
        previews: list[RetrievalEvalHitPreview] = []
        for rank, hit in enumerate(run.final_hits[:5], start=1):
            previews.append(
                RetrievalEvalHitPreview(
                    rank=rank,
                    document_id=hit.document_id,
                    version_id=hit.version_id,
                    chunk_id=hit.chunk_id,
                    section_path=self._as_path(hit.payload.get("hierarchy_path") or hit.payload.get("section_path")),
                    chunk_kind=str(hit.payload.get("chunk_kind")) if hit.payload.get("chunk_kind") else None,
                    preview=self._compress(hit.raw_text, 220),
                )
            )
        return previews

    def _build_suite_summaries(self, results: list[RetrievalEvalCaseResult]) -> list[RetrievalEvalSuiteSummary]:
        grouped: dict[str, list[RetrievalEvalCaseResult]] = defaultdict(list)
        for result in results:
            grouped[result.suite_id].append(result)

        summaries: list[RetrievalEvalSuiteSummary] = []
        for suite_id in sorted(grouped):
            suite_results = grouped[suite_id]
            total = len(suite_results)
            summaries.append(
                RetrievalEvalSuiteSummary(
                    suite_id=suite_id,
                    total_cases=total,
                    passed_cases=sum(1 for item in suite_results if item.passed),
                    planner_accuracy=self._ratio(sum(1 for item in suite_results if item.planner_ok), total),
                    router_accuracy=self._ratio(sum(1 for item in suite_results if item.router_ok), total),
                    hit_at_1=self._ratio(sum(1 for item in suite_results if item.hit_at_1), total),
                    hit_at_3=self._ratio(sum(1 for item in suite_results if item.hit_at_3), total),
                    hit_at_5=self._ratio(sum(1 for item in suite_results if item.hit_at_5), total),
                    mean_reciprocal_rank=round(
                        sum(item.reciprocal_rank for item in suite_results) / total, 3
                    )
                    if total
                    else 0.0,
                    citation_precision_at_3=round(
                        sum(item.citation_precision_at_3 for item in suite_results) / total, 3
                    )
                    if total
                    else 0.0,
                    no_result_rate=self._ratio(
                        sum(1 for item in suite_results if item.fallback_reason == "no_retrieval_hits"),
                        total,
                    ),
                )
            )
        return summaries

    def _ratio(self, numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return round(numerator / denominator, 3)

    def _as_path(self, value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        return []

    def _normalize(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text or "")
        return " ".join(
            "".join(char for char in normalized if not unicodedata.combining(char)).lower().split()
        )

    def _extract_planner_strategy(self, run: RetrievalRun) -> str | None:
        for item in run.decision_trace:
            if item.startswith("planner:strategy="):
                strategy = item.split("=", 1)[1].strip()
                return strategy or None
        return None

    def _compress(self, text: str, limit: int) -> str:
        compact = " ".join((text or "").split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."
