# Contratos canonicos

## Entidades principales

### document

- `document_id`
- `source`
- `source_authority`
- `mime_type`
- `language`
- `document_class`
- `document_family`
- `acl`
- `created_at`

### version

- `version_id`
- `document_id`
- `checksum`
- `version_status`
- `storage_uri`
- `artifact_uri`
- `extraction_status`
- `review_required`
- `review_reason`
- `extraction_backend`
- `text_length`
- `document_class`
- `document_family`
- `chunking_strategy`
- `classification_source`
- `classification_confidence`
- `classification_reason`
- `created_at`

### block

- `block_id`
- `document_id`
- `version_id`
- `block_type`
- `text`
- `page`
- `section_path`

### chunk

- `chunk_id`
- `document_id`
- `version_id`
- `block_refs`
- `text`
- `metadata`
- `acl`
- `index_status`

## Metadata minima de dominio

Cada version debe dejar persistido:

- `document_class`
- `document_family`
- `chunking_strategy`
- `classification_source`
- `classification_confidence`
- `classification_reason`

Valores validos para `classification_source`:

- `deterministic`
- `llm_local`
- `llm_cloud`
- `llm_fallback`
- `safe_fallback`

Interpretacion de dominio para Fase 3:

- `legal_normative` se enruta al dominio normativo legal
- `technical_standard` se enruta al dominio normativo tecnico
- `general_document` se enruta al dominio general

Cada chunk debe exponer al menos:

- `document_class`
- `document_family`
- `chunking_strategy`
- `classification_source`
- `section_path`
- `hierarchy_path`
- `retrieval_context`
- `contextualized_text`
- `chunk_kind`

En `legal_normative`, ademas:

- `book_label`
- `title_label`
- `chapter_label`
- `article_label`

En `technical_standard` y `general_document`, ademas:

- `section_label`
- `subsection_label`

Regla de leaf unit en Fase 2:

- `legal_normative`
  - `chunk_kind = article`
  - `chunk_kind = article_subchunk` solo si el articulo excede el presupuesto
  - `chunk_kind = preamble` para texto introductorio
- `technical_standard`
  - `chunk_kind = section`
- `general_document`
  - `generic_structural`
  - `chunk_kind = section`
  - `generic_freeform`
  - `chunk_kind = freeform`

## Respuesta de ingesta

- `document`
- `version`
- `storage_uri`
- `artifact_uri`
- `extraction_status`
- `review_required`
- `review_reason`
- `blocks_count`
- `text_length`
- `document_class`
- `document_family`
- `chunking_strategy`
- `classification_source`
- `classification_confidence`
- `classification_reason_short`

## Respuesta de chunking

- `document_id`
- `version_id`
- `chunks_count`
- `table_chunks_count`
- `artifact_uri`
- `chunk_window`
- `document_class`
- `document_family`
- `chunking_strategy`
- `classification_source`
- `classification_confidence`
- `classification_reason_short`

## Request de reindex

- `version_ids`
- `force`
- `limit`
- `document_class`
- `refresh`

## Respuesta de reindex

- `processed_versions`
- `indexed_versions`
- `skipped_versions`
- `failed_versions`
- `indexed_chunks`
- `qdrant_collection`
- `opensearch_index`
- `embedding_source`
- `results`

Cada item de `results` expone:

- `version_id`
- `document_id`
- `document_class`
- `document_family`
- `status`
- `chunks_indexed`
- `reason`

## Request de evaluacion de retrieval

- `suite_ids`
- `case_ids`
- `fixture_path`
- `include_passed`
- `user_identity`

## Respuesta de evaluacion de retrieval

- `fixture_path`
- `total_cases`
- `passed_cases`
- `router_accuracy`
- `hit_at_1`
- `hit_at_3`
- `hit_at_5`
- `mean_reciprocal_rank`
- `citation_precision_at_3`
- `no_result_rate`
- `by_suite`
- `results`

Cada item de `results` expone:

- `case_id`
- `suite_id`
- `question`
- `expected_document_class`
- `expected_document_family`
- `expected_version_id`
- `expected_document_id`
- `router_query_type`
- `router_ok`
- `hit_at_1`
- `hit_at_3`
- `hit_at_5`
- `reciprocal_rank`
- `citation_precision_at_3`
- `required_terms_ok`
- `optional_terms_ok`
- `passed`
- `fallback_reason`
- `decision_trace`
- `top_hits`
- `notes`

## Reglas de precedencia

1. `active` tiene prioridad sobre `superseded`, `draft` y `deprecated`.
2. `source_authority` primaria prevalece sobre secundaria.
3. Una contradiccion entre fuentes activas debe senalarse, no fusionarse silenciosamente.
4. La clasificacion de dominio hecha en Fase 1 es la fuente de verdad para Fase 2.
