# Roadmap revisado

## Resumen

El sistema se mantiene como una sola plataforma RAG.
No se divide en dos productos.

Se formalizan tres clases operativas:

- `legal_normative`
- `technical_standard`
- `general_document`

Familias derivadas:

- `legal_normative` -> `normative`
- `technical_standard` -> `normative`
- `general_document` -> `general`

## Fase 1

Objetivo:

- extraer
- normalizar
- clasificar el documento
- persistir la clasificacion

Entregables:

- `document_class`
- `document_family`
- `chunking_strategy`
- `classification_source`
- `classification_confidence`
- `classification_reason`

## Fase 2

Objetivo:

- aplicar chunking especializado por dominio

Rutas oficiales:

- `normative_hierarchical`
- `technical_structural`
- `generic_structural`
- `generic_freeform`

La entrada de Fase 2 es la clasificacion persistida en Fase 1.

Cierre esperado de Fase 2:

- `legal_normative`
  - soporte de `LIBRO`, `TITULO`, `CAPITULO` y `ARTICULO`
  - leaf canonico por articulo
  - `article_subchunk` solo cuando el articulo excede el presupuesto
- `technical_standard`
  - `technical_structural` para normas tecnicas por seccion/subseccion/ejemplo
- `general_document`
  - `generic_structured` para actas, minutas, certificaciones y documentos con headings
  - `generic_freeform` para OCR/plano con estructura pobre
- metadata lista para retrieval:
  - `hierarchy_path`
  - `retrieval_context`
  - `contextualized_text`
  - labels de dominio (`book/title/chapter/article` o `section/subsection`)

## Fase 3

Objetivo:

- indexacion dual
- retrieval por dominio

Diseno recomendado:

- una sola plataforma
- dos retrievers logicos
  - `normative_retriever`
  - `general_retriever`
- un `router_retriever`

Implementacion minima:

- usar `document_class` como particion logica
- reutilizar la misma infraestructura
- primera etapa:
  - `IndexingService`
  - indexacion dense en Qdrant sobre `contextualized_text`
  - indexacion sparse en OpenSearch sobre `contextualized_text`
  - endpoint `POST /documents/reindex`
  - perfiles de embedding intercambiables por configuracion
- segunda etapa:
  - `QueryRoutingService`
  - `RetrievalService`
  - dense retrieval + sparse retrieval
  - fusion RRF
  - respuesta minima basada en evidencia y citas
- tercera etapa minima de validacion:
  - `RetrievalEvaluationService`
  - endpoint `POST /eval/retrieval`
  - suites de benchmark versionadas
  - metricas: `router_accuracy`, `hit@k`, `MRR`, `citation_precision`
- endurecimiento de Fase 3:
  - hold-out benchmark separado del fixture base
  - `QueryRewriteService`
  - retrieval jerarquico ligero por rama
  - comparacion de generalizacion antes de pasar a answer synthesis
- cierre avanzado de Fase 3:
  - `RerankGateway`
  - proveedor local como baseline
  - Cohere como alternativa configurable
  - comparacion por benchmark antes de fijar proveedor productivo

Mapeo inicial de retrieval:

- `legal_normative` -> `normative_retriever`
- `technical_standard` -> `normative_retriever`
- `general_document` -> `general_retriever`

Implementacion posterior si la calidad lo exige:

- indices o colecciones separados por dominio

## Lo que no cambia

- API
- UI
- auth y ACL
- Model Gateway
- PostgreSQL
- MinIO
- Qdrant
- OpenSearch
- observabilidad
- evaluacion offline
