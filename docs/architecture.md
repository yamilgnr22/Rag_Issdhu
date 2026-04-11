# Arquitectura base del monorepo

## Objetivo

Construir una sola plataforma RAG para documentos internos, pero con
tratamiento especializado por dominio documental.

La plataforma no se divide en dos productos separados. Se mantiene una sola:

- API
- UI
- catalogo documental
- observabilidad
- seguridad
- capa de modelos

Lo que se especializa es el pipeline documental y el retrieval.

## Decision de arquitectura

La arquitectura revisada usa una sola plataforma con tres clases operativas:

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

Familias derivadas:

- `legal_normative` -> `normative`
- `technical_standard` -> `normative`
- `general_document` -> `general`

La separacion no se hace a nivel de producto. Se hace a nivel de:

- clasificacion documental
- chunking
- indexacion
- retrieval
- evaluacion

## Mapa del monorepo

- `apps/api`
  - servicio HTTP principal
  - contratos runtime
  - ingesta
  - chunking
  - clasificacion documental
  - persistencia del catalogo
- `apps/web`
  - chat UI
  - consola administrativa minima
- `libs/contracts`
  - esquemas JSON canónicos
  - ejemplos de payloads
- `infra/local`
  - stack local con Docker Compose
- `docs`
  - decisiones de arquitectura
  - contratos
  - perfiles
  - roadmap

## Principios

1. El backend concentra logica de negocio, seguridad y contratos.
2. El frontend no implementa reglas criticas de retrieval ni grounding.
3. La clasificacion documental ocurre en Fase 1 y queda persistida en catalogo.
4. El chunking consume esa clasificacion persistida en Fase 2.
5. El retrieval en Fase 3 debe enrutar por dominio antes de recuperar evidencia.
6. El despliegue productivo no depende de `.venv`; `.venv` solo es para desarrollo local.

## Fase 1: ingesta y clasificacion de dominio

La ingesta debe producir:

- binario original
- bloques extraidos
- metadatos de version
- clasificacion documental persistida

Campos minimos persistidos por version:

- `document_class`
- `document_family`
- `chunking_strategy`
- `classification_source`
- `classification_confidence`
- `classification_reason`

Esto evita reclasificar el mismo documento en cada fase posterior.

## Fase 2: chunking por dominio

El sistema mantiene tres rutas oficiales de chunking:

- `normative_hierarchical`
  - chunk base por articulo
  - subchunk por inciso/numeral si el articulo es largo
  - preambulo y tablas separados
- `technical_structural`
  - chunk base por seccion/subseccion/ejemplo
  - tablas y apendices conservados como unidades propias
- `generic_structural` / `generic_freeform`
  - headings
  - secciones
  - parrafos
  - tablas
  - fallback libre si el documento es poco estructurado

La clasificacion no debe depender de decisiones humanas en el flujo normal.
La clasificacion se hace a traves de un `Classification Gateway` configurable:

- `heuristic`
- `local_llm`
- `cloud_llm`
- `hybrid`

El gateway devuelve siempre el mismo contrato persistido en catalogo:

- `document_class`
- `document_family`
- `chunking_strategy`
- `classification_source`
- `classification_confidence`
- `classification_reason`

`hybrid` es el modo recomendado:

1. reglas deterministicas solo para casos obvios
2. proveedor local si esta configurado
3. proveedor cloud si esta configurado
4. `safe_fallback` a `general_document`

Leaf units cerradas para Fase 2:

- `legal_normative`
  - `LIBRO -> TITULO -> CAPITULO -> ARTICULO`
  - el leaf canonico es el articulo
  - si el articulo es largo se divide en `article_subchunk`
- `technical_standard`
  - `SECCION -> SUBSECCION -> EJEMPLO/APENDICE`
  - el leaf canonico es la seccion o subseccion tecnica
- `general_document`
  - `generic_structured`
  - el leaf es una seccion o bloque tematico
  - `generic_freeform`
  - el leaf es un chunk libre por parrafos/sentencias

Cada chunk conserva:

- `text` como evidencia fuente
- `retrieval_context` como contexto estructural
- `contextualized_text = retrieval_context + text` preparado para Fase 3

## Fase 3: retrieval por dominio

La recomendacion para Fase 3 no es crear dos productos.

La recomendacion es:

- un solo sistema
- dos retrievers logicos
  - `normative_retriever`
  - `general_retriever`
- un router de consulta
- primera entrega vertical:
  - `IndexingService`
  - dense index en Qdrant
  - sparse index en OpenSearch
  - `POST /documents/reindex`
  - nombres de indice/coleccion perfilados por embedding para alternar modelos sin mezclar dimensiones
 - segunda entrega minima:
   - `QueryRoutingService`
   - `RetrievalService`
   - dense + sparse retrieval
   - fusion RRF
   - respuesta extractiva minima con citas
 - capa de calidad opcional dentro de Fase 3:
   - `RerankGateway`
   - modos `off`, `local`, `cohere`
   - baseline heuristico como fallback
   - microservicio local `apps/reranker` para modelos de rerank dedicados
   - benchmark reutilizable para comparar proveedores
 - capa de validacion:
   - `RetrievalEvaluationService`
   - `POST /eval/retrieval`
   - fixtures versionados por suite
   - fixture base + fixture hold-out
   - metricas de routing y retrieval sobre el mismo pipeline productivo
 - endurecimiento de generalizacion:
   - `QueryRewriteService`
   - retrieval jerarquico ligero por `document_id + parent_path`
   - benchmark hold-out separado del fixture de tuning

La particion inicial puede hacerse con `document_class` y filtros.
En esa implementacion inicial, el dominio de retrieval se deriva asi:

- `legal_normative` -> `normative`
- `technical_standard` -> `normative`
- `general_document` -> `general`

Si luego la calidad lo exige, se podran separar colecciones o indices por dominio.

## Infraestructura

La base de infraestructura no cambia:

- PostgreSQL
- MinIO
- Qdrant
- OpenSearch
- Next.js
- FastAPI

La especializacion ocurre sobre el mismo stack, no fuera de el.
