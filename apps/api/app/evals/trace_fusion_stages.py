"""Rastrea la posición de la unidad esperada a lo largo de TODO el pipeline.

Complementa a trace_reorder_stages.py, que solo cubre de rerank en adelante.
Aquí interesa el tramo anterior: denso, sparse, variantes de consulta y fusión
RRF. Sirve para responder una pregunta concreta: si el recuperador denso trae el
documento correcto en primer lugar, ¿en qué etapa se pierde?

No modifica el pipeline: envuelve métodos en tiempo de ejecución.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.config import get_settings
from app.models.contracts import QueryRequest
from app.services.retrieval import RetrievalService


# Orden aproximado de ejecucion en RetrievalService.run
STAGES = (
    "_dense_search",
    "_sparse_search",
    "_apply_query_variants",
    "_fuse_hits",
    "_apply_hierarchical_boost",
    "_augment_with_branch_candidates",
    "_rerank_hits",
    "_attach_relevant_segments",
    "_apply_visibility_policy",
    "_diversify_hits",
    "_prioritize_answer_hits",
)


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return " ".join(
        "".join(char for char in normalized if not unicodedata.combining(char)).lower().split()
    )


def _label(hit) -> str:
    payload = hit.payload or {}
    return str(payload.get("canonical_label") or payload.get("path_text") or hit.chunk_id)[:24]


def _matches(hit, expected_units: list[str]) -> bool:
    payload = hit.payload or {}
    haystack = _normalize(
        " ".join(
            [
                str(payload.get("canonical_label") or ""),
                str(payload.get("path_text") or ""),
                str(payload.get("reference_variants_text") or ""),
            ]
        )
    )
    for expected in expected_units:
        normalized = _normalize(expected)
        variants = {normalized}
        number = re.search(r"(\d+)", normalized)
        if number:
            digits = number.group(1)
            variants |= {f"arto. {digits}", f"art. {digits}", f"articulo {digits}"}
        if any(variant in haystack for variant in variants):
            return True
    return False


def _rank_of(hits: list, expected_units: list[str]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if _matches(hit, expected_units):
            return rank
    return None


def _fusion_detail(hits: list, expected_units: list[str], top: int) -> list[dict[str, object]]:
    rows = []
    for rank, hit in enumerate(hits[:top], start=1):
        rows.append(
            {
                "pos": rank,
                "label": _label(hit),
                "dense_rank": hit.dense_rank,
                "sparse_rank": hit.sparse_rank,
                "rrf": round(float(hit.rrf_score), 5),
                "is_target": _matches(hit, expected_units),
            }
        )
    return rows


def _instrument(service: RetrievalService, expected: list[str], top: int) -> list[dict[str, object]]:
    timeline: list[dict[str, object]] = []

    def record(name: str, hits: list, extra: dict[str, object] | None = None) -> None:
        entry = {"stage": name, "n": len(hits), "target_rank": _rank_of(hits, expected)}
        if extra:
            entry.update(extra)
        timeline.append(entry)

    def wrap(name: str):
        original = getattr(service, name)

        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            if name == "_apply_query_variants" and isinstance(result, tuple) and len(result) == 2:
                record("_apply_query_variants[dense]", result[0])
                record("_apply_query_variants[sparse]", result[1])
                return result
            hits = result[0] if isinstance(result, tuple) else result
            if isinstance(hits, list):
                extra = None
                if name == "_fuse_hits":
                    extra = {"detail": _fusion_detail(hits, expected, top)}
                record(name, hits, extra)
            return result

        setattr(service, name, wrapped)

    for stage in STAGES:
        if hasattr(service, stage):
            wrap(stage)
    return timeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace target rank across the whole retrieval pipeline.")
    parser.add_argument("--input", default=str(Path("apps/api/app/evals/queries_honest_v1.jsonl")))
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--top", type=int, default=6)
    parser.add_argument("--output-dir", default=str(Path(".data/training/eval_runs_honest")))
    args = parser.parse_args()

    wanted = {item.strip() for item in args.case_ids.split(",") if item.strip()}
    cases = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if wanted:
        cases = [case for case in cases if str(case.get("id")) in wanted]

    settings = get_settings()
    settings.answer_synthesis_mode = "off"

    report = []
    for case in cases:
        service = RetrievalService(settings)
        expected = [str(item) for item in case.get("expected_units") or []]
        timeline = _instrument(service, expected, args.top)
        service.run(
            QueryRequest(
                question=str(case["question"]),
                user_identity="eval@local",
                filters={},
                conversation_context=[],
            )
        )
        report.append({"id": case["id"], "question": case["question"], "expected": expected, "timeline": timeline})

        print("=" * 80)
        print(f"{case['id']}  esperado={expected}")
        print(f"  Q: {case['question']}")
        previous = None
        for step in timeline:
            rank = step["target_rank"]
            mark = ""
            if previous is not None and previous is not None and rank is None:
                mark = "   <<< SE PIERDE AQUI"
            elif previous is not None and rank is not None and rank > previous:
                mark = f"   <<< baja {previous}->{rank}"
            print(f"    {step['stage']:34} n={step['n']:<4} target={str(rank):<6}{mark}")
            if step.get("detail"):
                for row in step["detail"]:
                    flag = "  <-- TARGET" if row["is_target"] else ""
                    print(
                        f"         {row['pos']}. {row['label']:24} dense={str(row['dense_rank']):>4}"
                        f" sparse={str(row['sparse_rank']):>4} rrf={row['rrf']:.5f}{flag}"
                    )
            previous = rank

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"fusion_trace__{stamp}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(json.dumps({"trace_path": str(out_path), "cases": len(report)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
