"""Consulta el RAG desde la terminal, sin levantar la API.

Uso:
    python -m app.tools.ask "cuantos dias de vacaciones me tocan?"
    python -m app.tools.ask            # modo interactivo
    python -m app.tools.ask "..." --grupos rrhh,contabilidad
    python -m app.tools.ask "..." --trace        # muestra el decision_trace
    python -m app.tools.ask "..." --sin-respuesta  # solo recuperacion, sin LLM
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.config import get_settings
from app.models.contracts import QueryRequest
from app.services.retrieval import RetrievalService


def _corta(texto: str, limite: int) -> str:
    compacto = " ".join((texto or "").split())
    return compacto if len(compacto) <= limite else compacto[: limite - 1] + "…"


def preguntar(service, pregunta: str, *, usuario: str, grupos: list[str], trace: bool) -> None:
    inicio = time.perf_counter()
    run = service.run(
        QueryRequest(
            question=pregunta,
            user_identity=usuario,
            user_groups=grupos,
            filters={},
            conversation_context=[],
        )
    )
    elapsed = time.perf_counter() - inicio
    respuesta = run.response

    print()
    print("=" * 78)
    print(f"RESPUESTA  ({elapsed:.1f}s, confianza {respuesta.confidence})")
    print("=" * 78)
    print(respuesta.answer)

    if run.final_hits:
        print()
        print("FUENTES:")
        for indice, hit in enumerate(run.final_hits[:5], start=1):
            payload = hit.payload or {}
            etiqueta = payload.get("canonical_label") or payload.get("path_text") or hit.chunk_id
            documento = payload.get("document_title") or hit.version_id
            marca = "*" if any(c.chunk_id == hit.chunk_id for c in respuesta.citations) else " "
            print(f" {marca}[{indice}] {str(etiqueta)[:34]:36} {str(documento)[:30]}")
            print(f"      {_corta(hit.raw_text, 150)}")
        if any(True for _ in respuesta.citations):
            print("  (* = citada en la respuesta)")

    if respuesta.fallback_reason:
        print()
        print(f"AVISO: {respuesta.fallback_reason}")

    if trace:
        print()
        print("DECISION TRACE:")
        for paso in run.decision_trace:
            print(f"   {paso}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Preguntarle al RAG desde la terminal.")
    parser.add_argument("pregunta", nargs="?", help="Si se omite, entra en modo interactivo.")
    parser.add_argument("--usuario", default="cli@local")
    parser.add_argument("--grupos", default="", help="Grupos/areas separados por coma, para las ACL.")
    parser.add_argument("--trace", action="store_true", help="Mostrar el decision_trace completo.")
    parser.add_argument(
        "--sin-respuesta",
        action="store_true",
        help="Solo recuperacion: no llama al LLM de sintesis (mas rapido y sin costo).",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.sin_respuesta:
        settings.answer_synthesis_mode = "off"

    grupos = [g.strip() for g in args.grupos.split(",") if g.strip()]

    print(f"modelo de sintesis: {settings.answer_synthesis_mode} | rerank: {settings.rerank_mode}", end="")
    print(f" | grupos: {grupos or 'ninguno'}")
    service = RetrievalService(settings)

    if args.pregunta:
        preguntar(service, args.pregunta, usuario=args.usuario, grupos=grupos, trace=args.trace)
        return 0

    print("Modo interactivo. Escribi tu pregunta (o 'salir').")
    while True:
        try:
            pregunta = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not pregunta:
            continue
        if pregunta.lower() in {"salir", "exit", "quit"}:
            return 0
        try:
            preguntar(service, pregunta, usuario=args.usuario, grupos=grupos, trace=args.trace)
        except Exception as exc:  # una consulta fallida no debe cerrar la sesion
            print(f"ERROR: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
