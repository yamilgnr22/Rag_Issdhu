"""Reingesta una version de documento y retira la anterior del indice.

Nunca retira la version vieja hasta comprobar que la nueva quedo realmente
escrita en los indices. La primera version de este script no lo hacia: cuando
la indexacion fallo (body de Qdrant por encima del limite, HTTP 400), retiro la
vieja igualmente y el corpus quedo sin ese documento.

Uso:
    python -m app.tools.reingest_document --old-version v_xxx [--label NIIF]
    python -m app.tools.reingest_document --old-version v_xxx --index-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import httpx

from app.config import get_settings
from app.db import Base, SessionLocal, engine, ensure_runtime_schema
from app.models.contracts import ChunkingRequest, IngestionFilesystemRequest, ReindexRequest
from app.models.database import ChunkRecord, VersionRecord
from app.services.chunking import ChunkingService
from app.services.indexing import IndexingService, OpenSearchIndexClient, QdrantIndexClient
from app.services.ingestion import IngestionService


def count_in_opensearch(settings, version_id: str) -> int:
    url = f"{settings.opensearch_url.rstrip('/')}/{settings.resolved_opensearch_index_name}/_count"
    response = httpx.post(url, json={"query": {"term": {"version_id": version_id}}}, timeout=30)
    response.raise_for_status()
    return int(response.json().get("count", 0))


def count_in_qdrant(settings, version_id: str) -> int:
    url = f"{settings.qdrant_url.rstrip('/')}/collections/{settings.resolved_qdrant_collection_name}/points/count"
    body = {"filter": {"must": [{"key": "version_id", "match": {"value": version_id}}]}, "exact": True}
    response = httpx.post(url, json=body, timeout=30)
    response.raise_for_status()
    return int(response.json().get("result", {}).get("count", 0))


def retire_version(settings, db, old_version: str) -> None:
    """Saca la version del indice dejando sus chunks en SQLite, de modo que se
    pueda restaurar con un reindex --force si hiciera falta."""
    targets = [
        ("qdrant_chunks", QdrantIndexClient(settings, collection_name=settings.resolved_qdrant_collection_name)),
        ("qdrant_branches", QdrantIndexClient(settings, collection_name=settings.resolved_qdrant_branch_collection_name)),
        ("os_chunks", OpenSearchIndexClient(settings, index_name=settings.resolved_opensearch_index_name)),
        ("os_branches", OpenSearchIndexClient(settings, index_name=settings.resolved_opensearch_branch_index_name, id_field="branch_id")),
    ]
    for name, client in targets:
        try:
            client.delete_by_version_id(version_id=old_version)
        except Exception as exc:
            print(f"      aviso {name}: {type(exc).__name__}")

    version = db.query(VersionRecord).filter(VersionRecord.version_id == old_version).one_or_none()
    if version is not None:
        version.version_status = "superseded"
    db.query(ChunkRecord).filter(ChunkRecord.version_id == old_version).update({"index_status": "pending"})
    db.commit()


def main() -> int:
    parser = argparse.ArgumentParser(description="Reingest a document version safely.")
    parser.add_argument("--old-version", required=True, help="Version a reemplazar (se lee su storage_uri).")
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--index-only",
        default="",
        help="Version ya ingerida y chunkeada: solo indexar y retirar la vieja.",
    )
    args = parser.parse_args()

    Base.metadata.create_all(bind=engine)
    ensure_runtime_schema()
    settings = get_settings()
    db = SessionLocal()

    old = db.query(VersionRecord).filter(VersionRecord.version_id == args.old_version).one_or_none()
    if old is None:
        print(f"version no encontrada: {args.old_version}")
        return 1
    print(f"=== {args.label or args.old_version}\n    origen: {old.storage_uri}")

    if args.index_only:
        new_version = args.index_only
        print(f"[1-2/4] omitidos: se reutiliza la version ya chunkeada {new_version}")
    else:
        started = time.perf_counter()
        result = IngestionService(settings).ingest_from_filesystem(
            db=db,
            request=IngestionFilesystemRequest(
                path=old.storage_uri,
                source="upload",
                source_authority="primary",
                language="es",
                version_status="active",
                acl_users=[],
                acl_groups=[],
            ),
        )
        new_version = result.version.version_id
        print(
            f"[1/4] ingesta {time.perf_counter()-started:.0f}s | version={new_version} "
            f"chars={result.text_length:,} review={result.review_required} reason={result.review_reason}"
        )

        started = time.perf_counter()
        chunked = ChunkingService(settings).chunk_version(
            db=db, version_id=new_version, request=ChunkingRequest(force=True)
        )
        print(f"[2/4] chunking {time.perf_counter()-started:.0f}s | chunks={chunked.chunks_count}")

    expected = db.query(ChunkRecord).filter(ChunkRecord.version_id == new_version).count()
    started = time.perf_counter()
    indexed = IndexingService(settings).reindex(
        db=db, request=ReindexRequest(version_ids=[new_version], force=True, refresh=True)
    )
    print(
        f"[3/4] indexado {time.perf_counter()-started:.0f}s | chunks={indexed.indexed_chunks} "
        f"fallidas={indexed.failed_versions}"
    )
    for item in indexed.results:
        if item.status != "indexed":
            print("      ", item.status, (item.reason or "")[:200])

    # Verificacion antes de retirar: sin esto, un fallo de indexacion deja el
    # corpus sin el documento.
    in_os = count_in_opensearch(settings, new_version)
    in_qd = count_in_qdrant(settings, new_version)
    print(f"[4/4] verificacion | esperados={expected} opensearch={in_os} qdrant={in_qd}")

    if indexed.failed_versions or in_os < expected or in_qd < expected:
        print("      ABORTADO: la nueva version no quedo completa en el indice.")
        print(f"      La version anterior {args.old_version} se deja intacta y activa.")
        db.close()
        return 2

    retire_version(settings, db, args.old_version)
    print(f"      {args.old_version} -> superseded y fuera del indice")
    print(f"OK. NUEVA VERSION: {new_version}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
