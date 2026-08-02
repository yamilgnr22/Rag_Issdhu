"""Traza el orden de los hits después de cada etapa post-rerank.

No modifica el pipeline: envuelve los métodos de RetrievalService en tiempo de
ejecución y registra el top-N tras cada etapa, junto con la posición de la unidad
esperada. Sirve para atribuir el desorden a una etapa concreta antes de tocar
nada.
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


STAGES = (
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
    return str(payload.get("canonical_label") or payload.get("path_text") or hit.chunk_id)[:26]


def _target_rank(hits: list, expected_units: list[str]) -> int | None:
    """Posición 1-indexada de la primera unidad esperada, o None."""
    for rank, hit in enumerate(hits, start=1):
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
            number = re.search(r"(\d+)", normalized)
            variants = {normalized}
            if number:
                digits = number.group(1)
                variants |= {f"arto. {digits}", f"art. {digits}", f"articulo {digits}"}
            if any(variant in haystack for variant in variants):
                return rank
    return None


def _snapshot(hits: list, expected_units: list[str], top: int) -> dict[str, object]:
    window = hits[:top]
    scores = [round(float(hit.rerank_score), 3) for hit in window]
    return {
        "labels": [_label(hit) for hit in window],
        "scores": scores,
        "sorted_by_score": scores == sorted(scores, reverse=True),
        "target_rank": _target_rank(hits, expected_units),
        "n": len(hits),
    }


def _instrument(service: RetrievalService, expected_units: list[str], top: int) -> list[dict[str, object]]:
    timeline: list[dict[str, object]] = []

    def wrap(name: str):
        original = getattr(service, name)

        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            hits = result[0] if isinstance(result, tuple) else result
            if isinstance(hits, list):
                timeline.append({"stage": name, **_snapshot(hits, expected_units, top)})
            return result

        setattr(service, name, wrapped)

    for stage in STAGES:
        wrap(stage)
    return timeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace hit ordering across post-rerank stages.")
    parser.add_argument("--input", default=str(Path("apps/api/app/evals/queries_honest_v1.jsonl")))
    parser.add_argument("--case-ids", default="", help="Comma-separated case ids; empty means all.")
    parser.add_argument("--top", type=int, default=5)
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
    settings.answer_synthesis_mode = "off"  # el orden no depende de la sintesis

    report: list[dict[str, object]] = []
    for case in cases:
        service = RetrievalService(settings)
        expected_units = [str(item) for item in case.get("expected_units") or []]
        timeline = _instrument(service, expected_units, args.top)
        service.run(
            QueryRequest(
                question=str(case["question"]),
                user_identity="eval@local",
                filters={},
                conversation_context=[],
            )
        )
        report.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected_units": expected_units,
                "timeline": timeline,
            }
        )

        print("=" * 78)
        print(f"{case['id']}  esperado={expected_units}")
        print(f"  Q: {case['question']}")
        previous_rank = None
        for step in timeline:
            flag = "" if step["sorted_by_score"] else "  <-- DESORDENADO"
            rank = step["target_rank"]
            movement = ""
            if previous_rank is not None and rank != previous_rank:
                movement = f"  [target {previous_rank} -> {rank}]"
            previous_rank = rank
            print(f"  {step['stage']:28} target_rank={str(rank):>4}{flag}{movement}")
            print(f"       {list(zip(step['labels'], step['scores']))}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"reorder_trace__{stamp}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(json.dumps({"trace_path": str(out_path), "cases": len(report)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
