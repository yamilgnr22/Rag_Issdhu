from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.config import get_settings
from app.models.contracts import QueryRequest
from app.services.retrieval import RetrievalRun, RetrievalService


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return " ".join(
        "".join(char for char in normalized if not unicodedata.combining(char)).lower().split()
    )


def _compress(text: str, limit: int = 220) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 3)


def _extract_planner_strategy(run: RetrievalRun) -> str | None:
    for item in run.decision_trace:
        if item.startswith("planner:strategy="):
            strategy = item.split("=", 1)[1].strip()
            return strategy or None
    return None


def _expand_expected_unit_hints(value: str) -> set[str]:
    hints: set[str] = set()

    def add(raw: str) -> None:
        normalized = _normalize(raw)
        if normalized:
            hints.add(normalized)

    add(value)

    normalized = _normalize(value)

    article_match = re.search(r"\b(?:arts?\.?|arto\.?|articulo)\s*(\d+(?:\.\d+)?)\b", normalized)
    if article_match:
        number = article_match.group(1)
        add(f"Arto. {number}")
        add(f"Art. {number}")
        add(f"Articulo {number}")

    section_match = re.search(r"\bseccion\s*(\d+(?:\.\d+)?)\b", normalized)
    if section_match:
        number = section_match.group(1)
        add(f"Seccion {number}")
        add(f"Seccion {number} ")

    recommendation_match = re.search(r"\brecomendacion\s*(\d+(?:\.\d+)?)\b", normalized)
    if recommendation_match:
        number = recommendation_match.group(1)
        add(f"Recomendacion {number}")
        add(f"R. {number}")

    paragraph_match = re.search(r"\b(\d+\.\d+)\b", normalized)
    if paragraph_match:
        add(paragraph_match.group(1))

    return hints


def _hit_haystack(hit) -> str:
    payload = hit.payload or {}
    parts = [
        str(payload.get("canonical_label") or ""),
        str(payload.get("path_text") or ""),
        " > ".join(str(item) for item in (payload.get("hierarchy_path") or payload.get("section_path") or [])),
        str(payload.get("reference_variants_text") or ""),
        hit.retrieval_context or "",
        hit.raw_text[:500],
    ]
    return _normalize(" ".join(parts))


def _matched_units_for_hit(hit, candidate_units: list[str]) -> list[str]:
    haystack = _hit_haystack(hit)
    matched: list[str] = []
    for expected in candidate_units:
        hints = _expand_expected_unit_hints(expected)
        if any(hint and hint in haystack for hint in hints):
            matched.append(expected)
    return matched


def _block_matches_hit(hit, expected_block: str) -> bool:
    if not expected_block:
        return False
    payload = hit.payload or {}
    path_text = str(payload.get("path_text") or "")
    path_normalized = _normalize(path_text)
    block_normalized = _normalize(expected_block)
    if not block_normalized or not path_normalized:
        return False
    return block_normalized in path_normalized or path_normalized.startswith(block_normalized)


def _top_hit_preview(hit, rank: int) -> dict[str, object]:
    payload = hit.payload or {}
    return {
        "rank": rank,
        "chunk_id": hit.chunk_id,
        "version_id": hit.version_id,
        "canonical_label": payload.get("canonical_label"),
        "path_text": payload.get("path_text"),
        "chunk_kind": payload.get("chunk_kind"),
        "rerank_score": round(float(hit.rerank_score), 4),
        "segment_score": payload.get("segment_score"),
        "preview": _compress(hit.raw_text, 220),
    }


def _load_cases(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
    return rows


def _filter_cases(
    cases: list[dict[str, object]],
    *,
    corpora: set[str],
    case_ids: set[str],
    limit: int | None,
) -> list[dict[str, object]]:
    filtered = list(cases)
    if corpora:
        filtered = [case for case in filtered if str(case.get("corpus")) in corpora]
    if case_ids:
        filtered = [case for case in filtered if str(case.get("id")) in case_ids]
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def _evaluate_case(service: RetrievalService, case: dict[str, object], *, top_k: int, user_identity: str) -> dict[str, object]:
    question = str(case["question"])
    expected_units = [str(item) for item in case.get("expected_units") or []]
    acceptable_units = [str(item) for item in case.get("acceptable_units") or []]
    expected_block = str(case.get("expected_block") or "")
    expected_index_block = str(case.get("expected_index_block") or "")
    block_match_target = expected_index_block or expected_block

    started_at = time.perf_counter()
    run = service.run(
        QueryRequest(
            question=question,
            user_identity=user_identity,
            filters={},
            conversation_context=[],
        )
    )
    elapsed_seconds = round(time.perf_counter() - started_at, 3)

    matched_unit_ranks: dict[str, int] = {}
    acceptable_unit_ranks: dict[str, int] = {}
    block_match_ranks: list[int] = []
    previews: list[dict[str, object]] = []

    for rank, hit in enumerate(run.final_hits[:top_k], start=1):
        previews.append(_top_hit_preview(hit, rank))
        for expected in _matched_units_for_hit(hit, expected_units):
            matched_unit_ranks.setdefault(expected, rank)
        for acceptable in _matched_units_for_hit(hit, acceptable_units):
            acceptable_unit_ranks.setdefault(acceptable, rank)
        if block_match_target and _block_matches_hit(hit, block_match_target):
            block_match_ranks.append(rank)

    strict_first_unit_rank = min(matched_unit_ranks.values()) if matched_unit_ranks else None
    acceptable_first_unit_rank = min(acceptable_unit_ranks.values()) if acceptable_unit_ranks else None
    rank_candidates = [
        rank for rank in (strict_first_unit_rank, acceptable_first_unit_rank) if rank is not None
    ]
    first_unit_rank = min(rank_candidates) if rank_candidates else None
    reciprocal_rank = round(1.0 / first_unit_rank, 3) if first_unit_rank else 0.0

    strict_unit_hit_at_1 = strict_first_unit_rank == 1
    strict_unit_hit_at_3 = strict_first_unit_rank is not None and strict_first_unit_rank <= 3
    strict_unit_hit_at_5 = strict_first_unit_rank is not None and strict_first_unit_rank <= 5

    acceptable_unit_hit_at_1 = acceptable_first_unit_rank == 1
    acceptable_unit_hit_at_3 = acceptable_first_unit_rank is not None and acceptable_first_unit_rank <= 3
    acceptable_unit_hit_at_5 = acceptable_first_unit_rank is not None and acceptable_first_unit_rank <= 5

    unit_hit_at_1 = first_unit_rank == 1
    unit_hit_at_3 = first_unit_rank is not None and first_unit_rank <= 3
    unit_hit_at_5 = first_unit_rank is not None and first_unit_rank <= 5

    matched_units_at_5 = [unit for unit, rank in matched_unit_ranks.items() if rank <= 5]
    acceptable_matches_at_5 = [unit for unit, rank in acceptable_unit_ranks.items() if rank <= 5]
    expected_unit_recall_at_5 = (
        round(len(matched_units_at_5) / len(expected_units), 3) if expected_units else 0.0
    )
    full_unit_match_at_5 = bool(expected_units) and len(matched_units_at_5) == len(expected_units)

    first_block_rank = min(block_match_ranks) if block_match_ranks else None
    block_hit_at_1 = first_block_rank == 1
    block_hit_at_3 = first_block_rank is not None and first_block_rank <= 3
    block_hit_at_5 = first_block_rank is not None and first_block_rank <= 5

    granularity = str(case.get("expected_granularity") or "")
    if granularity == "single_unit":
        passed = unit_hit_at_3
    else:
        passed = full_unit_match_at_5 or block_hit_at_3

    return {
        "id": case["id"],
        "question": question,
        "corpus": case["corpus"],
        "version_id": case["version_id"],
        "expected_granularity": granularity,
        "expected_units": expected_units,
        "acceptable_units": acceptable_units,
        "expected_block": expected_block,
        "expected_index_block": expected_index_block,
        "block_match_target": block_match_target,
        "review_status": case.get("review_status"),
        "planner_strategy": _extract_planner_strategy(run),
        "router_query_type": run.route.query_type,
        "fallback_reason": run.response.fallback_reason,
        "confidence": run.response.confidence,
        "elapsed_seconds": elapsed_seconds,
        "strict_unit_hit_at_1": strict_unit_hit_at_1,
        "strict_unit_hit_at_3": strict_unit_hit_at_3,
        "strict_unit_hit_at_5": strict_unit_hit_at_5,
        "acceptable_unit_hit_at_1": acceptable_unit_hit_at_1,
        "acceptable_unit_hit_at_3": acceptable_unit_hit_at_3,
        "acceptable_unit_hit_at_5": acceptable_unit_hit_at_5,
        "unit_hit_at_1": unit_hit_at_1,
        "unit_hit_at_3": unit_hit_at_3,
        "unit_hit_at_5": unit_hit_at_5,
        "expected_unit_recall_at_5": expected_unit_recall_at_5,
        "full_unit_match_at_5": full_unit_match_at_5,
        "first_unit_rank": first_unit_rank,
        "strict_first_unit_rank": strict_first_unit_rank,
        "acceptable_first_unit_rank": acceptable_first_unit_rank,
        "reciprocal_rank": reciprocal_rank,
        "block_hit_at_1": block_hit_at_1,
        "block_hit_at_3": block_hit_at_3,
        "block_hit_at_5": block_hit_at_5,
        "first_block_rank": first_block_rank,
        "passed": passed,
        "answer_preview": _compress(run.response.answer, 260),
        "matched_units_at_5": matched_units_at_5,
        "acceptable_matches_at_5": acceptable_matches_at_5,
        "decision_trace_tail": run.decision_trace[-20:],
        "top_hits": previews,
    }


def _summarize(results: list[dict[str, object]]) -> dict[str, object]:
    total = len(results)
    with_block = [item for item in results if item.get("block_match_target")]
    with_acceptable = [item for item in results if item.get("acceptable_units")]

    summary = {
        "total_cases": total,
        "passed_cases": sum(1 for item in results if item["passed"]),
        "strict_unit_hit_at_1": _ratio(sum(1 for item in results if item["strict_unit_hit_at_1"]), total),
        "strict_unit_hit_at_3": _ratio(sum(1 for item in results if item["strict_unit_hit_at_3"]), total),
        "strict_unit_hit_at_5": _ratio(sum(1 for item in results if item["strict_unit_hit_at_5"]), total),
        "unit_hit_at_1": _ratio(sum(1 for item in results if item["unit_hit_at_1"]), total),
        "unit_hit_at_3": _ratio(sum(1 for item in results if item["unit_hit_at_3"]), total),
        "unit_hit_at_5": _ratio(sum(1 for item in results if item["unit_hit_at_5"]), total),
        "acceptable_unit_hit_at_1": _ratio(
            sum(1 for item in with_acceptable if item["acceptable_unit_hit_at_1"]),
            len(with_acceptable),
        ),
        "acceptable_unit_hit_at_3": _ratio(
            sum(1 for item in with_acceptable if item["acceptable_unit_hit_at_3"]),
            len(with_acceptable),
        ),
        "acceptable_unit_hit_at_5": _ratio(
            sum(1 for item in with_acceptable if item["acceptable_unit_hit_at_5"]),
            len(with_acceptable),
        ),
        "full_unit_match_at_5": _ratio(sum(1 for item in results if item["full_unit_match_at_5"]), total),
        "mean_expected_unit_recall_at_5": round(
            sum(item["expected_unit_recall_at_5"] for item in results) / total, 3
        )
        if total
        else 0.0,
        "mean_reciprocal_rank": round(sum(item["reciprocal_rank"] for item in results) / total, 3)
        if total
        else 0.0,
        "block_hit_at_1": _ratio(sum(1 for item in with_block if item["block_hit_at_1"]), len(with_block)),
        "block_hit_at_3": _ratio(sum(1 for item in with_block if item["block_hit_at_3"]), len(with_block)),
        "block_hit_at_5": _ratio(sum(1 for item in with_block if item["block_hit_at_5"]), len(with_block)),
        "no_result_rate": _ratio(sum(1 for item in results if item["fallback_reason"] == "no_retrieval_hits"), total),
        "by_corpus": [],
    }

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in results:
        grouped[str(item["corpus"])].append(item)

    by_corpus: list[dict[str, object]] = []
    for corpus in sorted(grouped):
        rows = grouped[corpus]
        rows_with_block = [item for item in rows if item.get("block_match_target")]
        rows_with_acceptable = [item for item in rows if item.get("acceptable_units")]
        by_corpus.append(
            {
                "corpus": corpus,
                "total_cases": len(rows),
                "passed_cases": sum(1 for item in rows if item["passed"]),
                "strict_unit_hit_at_1": _ratio(
                    sum(1 for item in rows if item["strict_unit_hit_at_1"]),
                    len(rows),
                ),
                "strict_unit_hit_at_3": _ratio(
                    sum(1 for item in rows if item["strict_unit_hit_at_3"]),
                    len(rows),
                ),
                "strict_unit_hit_at_5": _ratio(
                    sum(1 for item in rows if item["strict_unit_hit_at_5"]),
                    len(rows),
                ),
                "unit_hit_at_1": _ratio(sum(1 for item in rows if item["unit_hit_at_1"]), len(rows)),
                "unit_hit_at_3": _ratio(sum(1 for item in rows if item["unit_hit_at_3"]), len(rows)),
                "unit_hit_at_5": _ratio(sum(1 for item in rows if item["unit_hit_at_5"]), len(rows)),
                "acceptable_unit_hit_at_3": _ratio(
                    sum(1 for item in rows_with_acceptable if item["acceptable_unit_hit_at_3"]),
                    len(rows_with_acceptable),
                ),
                "full_unit_match_at_5": _ratio(
                    sum(1 for item in rows if item["full_unit_match_at_5"]),
                    len(rows),
                ),
                "mean_expected_unit_recall_at_5": round(
                    sum(item["expected_unit_recall_at_5"] for item in rows) / len(rows), 3
                )
                if rows
                else 0.0,
                "mean_reciprocal_rank": round(
                    sum(item["reciprocal_rank"] for item in rows) / len(rows), 3
                )
                if rows
                else 0.0,
                "block_hit_at_3": _ratio(
                    sum(1 for item in rows_with_block if item["block_hit_at_3"]),
                    len(rows_with_block),
                ),
            }
        )

    summary["by_corpus"] = by_corpus
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline retrieval evaluation against queries gold JSONL.")
    parser.add_argument(
        "--input",
        default=str(Path(".data/training/queries_gold_v1_proposed.jsonl")),
        help="Path to the queries gold JSONL fixture.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(".data/training/eval_runs")),
        help="Directory where summary/detail outputs will be written.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of cases.")
    parser.add_argument(
        "--case-ids",
        default="",
        help="Comma-separated case ids to evaluate, for example c001,c031.",
    )
    parser.add_argument(
        "--corpora",
        default="",
        help="Comma-separated corpus names to evaluate, for example codigo_del_trabajo,ley_822.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k window used for metrics and previews.")
    parser.add_argument("--user-identity", default="eval@local", help="Synthetic user identity for evaluation queries.")
    parser.add_argument(
        "--answer-synthesis-mode",
        choices=["off", "inherit"],
        default="off",
        help="Disable answer synthesis by default to benchmark retrieval faster.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    case_ids = {item.strip() for item in args.case_ids.split(",") if item.strip()}
    corpora = {item.strip() for item in args.corpora.split(",") if item.strip()}

    cases = _filter_cases(_load_cases(input_path), corpora=corpora, case_ids=case_ids, limit=args.limit)
    if not cases:
        raise SystemExit("No evaluation cases matched the provided filters.")

    settings = get_settings()
    if args.answer_synthesis_mode == "off":
        settings.answer_synthesis_mode = "off"
    service = RetrievalService(settings)

    started_at = time.perf_counter()
    results = [_evaluate_case(service, case, top_k=args.top_k, user_identity=args.user_identity) for case in cases]
    total_elapsed = round(time.perf_counter() - started_at, 3)

    summary = _summarize(results)
    summary.update(
        {
            "fixture_path": str(input_path),
            "top_k": args.top_k,
            "answer_synthesis_mode": settings.answer_synthesis_mode,
            "total_elapsed_seconds": total_elapsed,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{input_path.stem}__offline_eval__{stamp}"
    summary_path = output_dir / f"{base_name}__summary.json"
    detail_path = output_dir / f"{base_name}__details.jsonl"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    detail_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in results) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"summary_path": str(summary_path), "detail_path": str(detail_path)}, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
