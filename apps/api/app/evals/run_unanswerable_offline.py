"""Mide si el sistema se abstiene cuando el corpus no contiene la respuesta.

El benchmark de retrieval existente solo mide si el chunk esperado aparece en el
top-k. No mide el fallo opuesto: inventar una respuesta cuando no hay evidencia.
Este runner cubre ese hueco.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.config import get_settings
from app.models.contracts import QueryRequest
from app.services.retrieval import RetrievalService


ABSTENTION_MARKERS = (
    "no se encontro evidencia",
    "no hay evidencia",
    "no se especifica",
    "no especifica",
    "la evidencia no",
    "evidencia proporcionada no",
    "no contiene informacion",
    "no se menciona",
    "no aparece",
    "no permite determinar",
    "no es posible determinar",
    "insuficiente",
    # Formas verbales observadas en respuestas reales del sistema: la abstencion
    # se expresa negando el sujeto de la pregunta, no con una formula fija.
    "no establece",
    "no indica",
    "no indican",
    "no incorpora",
    "no identifica",
    "no menciona",
    "no incluye",
    "tampoco menciona",
    "es parcial",
    "informacion disponible es parcial",
)


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return " ".join(
        "".join(char for char in normalized if not unicodedata.combining(char)).lower().split()
    )


def _looks_like_abstention(answer: str) -> bool:
    normalized = _normalize(answer)
    return any(marker in normalized for marker in ABSTENTION_MARKERS)


def _load_cases(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate abstention on unanswerable questions.")
    parser.add_argument("--input", default=str(Path("apps/api/app/evals/queries_honest_unanswerable_v1.jsonl")))
    parser.add_argument("--output-dir", default=str(Path(".data/training/eval_runs_honest")))
    parser.add_argument("--user-identity", default="eval@local")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = _load_cases(input_path)
    service = RetrievalService(get_settings())

    results: list[dict[str, object]] = []
    for case in cases:
        started_at = time.perf_counter()
        run = service.run(
            QueryRequest(
                question=str(case["question"]),
                user_identity=args.user_identity,
                filters={},
                conversation_context=[],
            )
        )
        answer = run.response.answer
        results.append(
            {
                "id": case["id"],
                "question": case["question"],
                "notes": case.get("notes"),
                "abstained": _looks_like_abstention(answer),
                "fallback_reason": run.response.fallback_reason,
                "confidence": run.response.confidence,
                "citations": len(run.response.citations),
                "hits": len(run.final_hits),
                "top_labels": [
                    str((hit.payload or {}).get("canonical_label") or "") for hit in run.final_hits[:3]
                ],
                "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                "answer": " ".join(answer.split()),
            }
        )

    total = len(results) or 1
    abstained = sum(1 for item in results if item["abstained"])
    summary = {
        "fixture_path": str(input_path),
        "total_cases": len(results),
        "abstained_cases": abstained,
        "abstention_rate": round(abstained / total, 3),
        "hallucination_risk_cases": [item["id"] for item in results if not item["abstained"]],
        "mean_confidence_when_not_abstained": round(
            sum(float(item["confidence"]) for item in results if not item["abstained"])
            / max(1, len(results) - abstained),
            3,
        ),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"unanswerable__{stamp}__summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(out_path)}, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
