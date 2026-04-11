# Rag_Issdhu

Base de un RAG documental con monorepo `api + web + contracts + infra`.

## Estrategia actual

La plataforma usa una sola arquitectura RAG, con tres clases operativas:

- `legal_normative`
  - leyes
  - codigos
  - reglamentos
  - resoluciones juridicas
- `technical_standard`
  - NIIF
  - NIC
  - IFRS
  - estandares tecnicos oficiales
- `general_document`
  - actas
  - minutas
  - certificaciones
  - manuales
  - procedimientos
  - reportes

No son dos productos separados.
La especializacion ocurre en:

- clasificacion documental
- chunking
- retrieval
- evaluacion

Familias derivadas:

- `legal_normative` -> `normative`
- `technical_standard` -> `normative`
- `general_document` -> `general`

Fase 2 deja preparados los chunks para retrieval por dominio:

- `legal_normative`
  - jerarquia `LIBRO -> TITULO -> CAPITULO -> ARTICULO`
  - `article_subchunk` solo para articulos largos
- `technical_standard`
  - chunks por seccion/subseccion/ejemplo
- `general_document`
  - `generic_structural` para secciones/bloques tematicos
  - `generic_freeform` para documentos poco estructurados

## Estructura

- `apps/api`: backend FastAPI y contratos runtime
- `apps/web`: UI web en Next.js + React + TypeScript
- `libs/contracts`: esquemas JSON y ejemplos canonicos
- `infra/local`: dependencias locales con Docker Compose
- `docs`: arquitectura, contratos, perfiles y roadmap

## Requisitos

- Python 3.12+
- Node 24+
- npm 11+
- Docker 28+

## Backend local

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .\apps\api
uvicorn app.main:app --reload --app-dir .\apps\api
```

API disponible en `http://localhost:8000`.

## Ingesta local

Por upload:

```powershell
curl.exe -X POST "http://localhost:8000/documents:ingest" `
  -F "file=@C:\ruta\archivo.pdf" `
  -F "source=upload" `
  -F "source_authority=primary" `
  -F "acl_groups=compliance"
```

Por filesystem:

```powershell
curl.exe -X POST "http://localhost:8000/documents:ingest/filesystem" `
  -H "Content-Type: application/json" `
  -d "{\"path\":\"C:\\\\ruta\\\\archivo.pdf\",\"source_authority\":\"primary\"}"
```

La respuesta de ingesta expone:

- `document_class`
- `document_family`
- `chunking_strategy`
- `classification_source`
- `classification_confidence`
- `classification_reason_short`

## Chunking local

```powershell
curl.exe -X POST "http://localhost:8000/documents/v_tu_version/chunk" `
  -H "Content-Type: application/json" `
  -d "{\"force\":true,\"target_tokens\":600,\"min_tokens\":400,\"max_tokens\":800,\"overlap_ratio\":0.2}"
```

La respuesta de chunking expone:

- `document_class`
- `document_family`
- `chunking_strategy`
- `classification_source`
- `classification_confidence`
- `classification_reason_short`

Estrategias activas:

- `normative_hierarchical`
- `technical_structural`
- `generic_structural`
- `generic_freeform`

## Reindex local

Primera etapa de Fase 3:

- embeddings densos sobre `contextualized_text`
- indexacion sparse/BM25 sobre `contextualized_text`
- metadata de filtro persistida por chunk

Antes de reindexar, levanta:

```powershell
docker compose -f .\infra\local\docker-compose.yml up -d
```

Importante:

- `Qdrant` y `OpenSearch` deben tener sus volumenes activos
- si recreaste contenedores y perdiste el indice de OpenSearch, vuelve a correr `POST /documents/reindex`

Reindexar versiones pendientes:

```powershell
curl.exe -X POST "http://localhost:8000/documents/reindex" `
  -H "Content-Type: application/json" `
  -d "{}"
```

Reindexar una version puntual:

```powershell
curl.exe -X POST "http://localhost:8000/documents/reindex" `
  -H "Content-Type: application/json" `
  -d "{\"version_ids\":[\"v_tu_version\"],\"force\":true}"
```

La respuesta expone:

- `processed_versions`
- `indexed_versions`
- `skipped_versions`
- `failed_versions`
- `indexed_chunks`
- `qdrant_collection`
- `opensearch_index`
- `embedding_source`
- `results`

## Query local

La primera version de Fase 3.2 ya hace:

- query routing por dominio
- dense retrieval en Qdrant
- sparse retrieval en OpenSearch
- fusion por RRF
- respuesta minima basada en evidencia recuperada
- endpoint de evaluacion offline reutilizando el retrieval real

Ejemplo:

```powershell
$body = @{
  question = "Que acuerdo se tomo sobre la oferta de compra de la propiedad de Chontales?"
  user_identity = "analyst@example.com"
  filters = @{}
  conversation_context = @()
} | ConvertTo-Json -Depth 4 -Compress

Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/query" `
  -ContentType "application/json" `
  -Body $body
```

Configuracion de retrieval:

- `RETRIEVAL_DENSE_TOP_K`
- `RETRIEVAL_SPARSE_TOP_K`
- `RETRIEVAL_FUSED_TOP_K`
- `RETRIEVAL_RRF_K`
- `RERANK_MODE`
- `RERANK_TOP_N`
- `RERANK_LOCAL_*`
- `RERANK_COHERE_*`

Rerank gateway soportado:

- `off`
  - usa solo el rerank heuristico actual
- `local`
  - usa un endpoint local `POST /rerank`
  - contrato esperado:
    - `query`
    - `documents`
    - `top_n`
    - `model`
- `cohere`
  - usa Cohere Rerank

Modo recomendado actual:

- `cohere`
  - opcion operativa preferida por ahora
  - el gateway local queda disponible para laboratorio y comparacion

Modo recomendado para comparar calidad:

1. correr benchmark con `RERANK_MODE=local`
2. correr benchmark con `RERANK_MODE=cohere`
3. comparar `hit@1`, `MRR`, `citation_precision`

## Reranker local

Para Fase 3 avanzada ya existe un microservicio aparte en:

- `apps/reranker`

Su contrato es:

- `POST /rerank`
  - `query`
  - `documents`
  - `top_n`
  - `model`

Arranque recomendado:

```powershell
py -3.12 -m venv .venv-reranker
.\.venv-reranker\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .\apps\reranker
uvicorn app.main:app --host 0.0.0.0 --port 7997 --app-dir .\apps\reranker
```

Luego configura la API principal:

```env
RERANK_MODE=local
RERANK_LOCAL_BASE_URL=http://localhost:7997
RERANK_LOCAL_MODEL=BAAI/bge-reranker-v2-m3
```

El modelo se descarga al primer arranque del microservicio.

## Evaluacion de retrieval

Fase 3 incluye un benchmark local para no depender solo de inspeccion manual.

- endpoint: `POST /eval/retrieval`
- fixture por defecto:
  - `apps/api/app/evals/retrieval_eval_cases.json`
- fixture hold-out:
  - `apps/api/app/evals/retrieval_eval_holdout_cases.json`
- suites iniciales:
  - `ley_822`
  - `niif_pymes`
  - `gafi_2025`

Ejemplo:

```powershell
$body = @{
  suite_ids = @("ley_822", "niif_pymes", "gafi_2025")
  include_passed = $false
  user_identity = "eval@local"
} | ConvertTo-Json -Depth 4 -Compress

Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/eval/retrieval" `
  -ContentType "application/json" `
  -Body $body
```

La respuesta resume:

- `router_accuracy`
- `hit_at_1`
- `hit_at_3`
- `hit_at_5`

Tambien puedes correr varios fixtures en una sola pasada con `fixture_paths`, por ejemplo:

```json
{
  "fixture_paths": [
    "D:\\Proyectos\\Rag_Issdhu\\apps\\api\\app\\evals\\retrieval_eval_cases.json",
    "D:\\Proyectos\\Rag_Issdhu\\apps\\api\\app\\evals\\retrieval_eval_holdout_cases.json"
  ],
  "include_passed": false,
  "user_identity": "eval@local"
}
```
- `mean_reciprocal_rank`
- `citation_precision_at_3`
- `no_result_rate`
- `by_suite`
- `results`

La clasificacion se hace por un gateway configurable. Modos disponibles:

- `CLASSIFIER_MODE=heuristic`
- `CLASSIFIER_MODE=local_llm`
- `CLASSIFIER_MODE=cloud_llm`
- `CLASSIFIER_MODE=hybrid`

Configuracion recomendada:

- desarrollo: `cloud_llm` o `hybrid`
- produccion cerrada: `local_llm` o `hybrid`

Compatibilidad OpenAI-compatible:

- `CLASSIFIER_LOCAL_BASE_URL`
- `CLASSIFIER_LOCAL_API_KEY`
- `CLASSIFIER_LOCAL_MODEL`
- `CLASSIFIER_CLOUD_BASE_URL`
- `CLASSIFIER_CLOUD_API_KEY`
- `CLASSIFIER_CLOUD_MODEL`

Compatibilidad backward:

- `CLASSIFIER_LLM_BASE_URL`
- `CLASSIFIER_LLM_API_KEY`
- `CLASSIFIER_LLM_MODEL`

En `hybrid`, el orden es:

1. deterministico solo si la confianza es alta
2. `local_llm`
3. `cloud_llm`
4. `safe_fallback`

`classification_source` puede salir como:

- `deterministic`
- `llm_local`
- `llm_cloud`
- `llm_fallback`
- `safe_fallback`

Configuracion de embeddings para Fase 3:

- `EMBEDDING_MODE=hash|local_openai|cloud_openai|hybrid`
- `EMBEDDING_PROFILED_INDEX_NAMES=true|false`
- `EMBEDDING_PROFILE_NAME`
- `EMBEDDING_CLOUD_BASE_URL`
- `EMBEDDING_CLOUD_API_KEY`
- `EMBEDDING_CLOUD_MODEL`
- `EMBEDDING_LOCAL_BASE_URL`
- `EMBEDDING_LOCAL_API_KEY`
- `EMBEDDING_LOCAL_MODEL`

`hybrid` intenta:

1. `local_openai`
2. `cloud_openai`
3. `hash`

Si quieres alternar entre `text-embedding-3-small` y `text-embedding-3-large`
sin mezclar dimensiones en la misma coleccion:

1. cambia `EMBEDDING_CLOUD_MODEL`
2. define `EMBEDDING_PROFILE_NAME`, por ejemplo:
   - `te3-small`
   - `te3-large`
3. vuelve a correr `POST /documents/reindex` con `force=true`

Con `EMBEDDING_PROFILED_INDEX_NAMES=true`, la API deriva nombres separados de
indice/coleccion por perfil de embedding.

Para inspeccionar los chunks generados:

```powershell
curl.exe "http://localhost:8000/documents/v_tu_version/chunks"
```

## Frontend local

```powershell
cd .\apps\web
npm install
npm run dev
```

UI disponible en `http://localhost:3000`.

## Infra local

```powershell
docker compose -f .\infra\local\docker-compose.yml up -d
```

Servicios incluidos:

- PostgreSQL
- MinIO
- OpenSearch
- Qdrant

Fase 1 y Fase 2 usan por defecto SQLite y filesystem storage para no bloquear
el desarrollo local. Si quieres conectar PostgreSQL y MinIO reales, ajusta `.env`.

## Perfiles

- `dev-gpu`
- `prod-cpu`

Ver:

- [docs/architecture.md](D:\Proyectos\Rag_Issdhu\docs\architecture.md)
- [docs/contracts.md](D:\Proyectos\Rag_Issdhu\docs\contracts.md)
- [docs/profiles.md](D:\Proyectos\Rag_Issdhu\docs\profiles.md)
- [docs/roadmap.md](D:\Proyectos\Rag_Issdhu\docs\roadmap.md)
