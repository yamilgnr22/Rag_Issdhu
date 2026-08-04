# Fase 0 — Baseline honesto y análisis de errores

Fecha de ejecución: 2026-08-01. Corpus: 8 documentos, 1810 chunks, perfil de
embedding `openai-3-large-test`. Sistema medido **sin modificar**: el objetivo de
esta fase era obtener un punto de partida creíble, no mejorar nada todavía.

## 1. Por qué hacía falta un baseline nuevo

El benchmark existente reportaba `hit@1 = 0.946` mientras el sistema respondía mal
en uso real. Tres razones explican la contradicción:

1. El clasificador del router se entrena con el mismo fixture con que se evalúa
   (`app/config.py`: `query_router_classifier_fixture_path` apunta a
   `retrieval_eval_cases.json`).
2. Frases literales de las suites de evaluación están codificadas como boosts en
   `BOOSTABLE_PHRASES` (`app/services/retrieval.py`).
2. **Todas** las preguntas del gold set incluyen ya el número de artículo
   ("Que establece el Arto. 45 sobre..."), que es justo el dato que el usuario no
   conoce cuando pregunta.

Además, la última corrida offline usó `answer_synthesis_mode = off`: ninguna
métrica del proyecto había medido nunca la calidad de la **respuesta**, que es el
síntoma reportado.

## 2. Fixtures nuevos

- `.data/training/queries_honest_v1.jsonl` — 30 preguntas en lenguaje natural de
  usuario, **sin número de artículo**, con tildes y sinónimos populares
  ("aguinaldo" por décimo tercer mes, "embarazo" por estado de gravidez).
  Targets verificados contra el texto real de los chunks.
- `.data/training/queries_honest_unanswerable_v1.jsonl` — 8 preguntas sin
  respuesta en el corpus, incluidas trampas (licencia de paternidad, que no
  existe junto a maternidad que sí existe).
- `apps/api/app/evals/run_unanswerable_offline.py` — runner nuevo que mide
  abstención, hueco que el benchmark existente no cubría.

## 3. Baseline

| Métrica | Honesto | Gold viejo | Δ |
|---|---|---|---|
| unit_hit@1 | **0.333** | 0.800 | −0.467 |
| unit_hit@3 | **0.500** | 0.967 | −0.467 |
| unit_hit@5 | 0.633 | 0.967 | −0.334 |
| MRR | **0.437** | 0.872 | −0.435 |
| Casos pasados | 18/30 | 29/30 | |
| Latencia | **41.6 s/consulta** | 34 s | |

Por corpus: `codigo_del_trabajo` 8/12, `reglamento_issdhu` 5/10, `ley_822` 2/4,
`actas_consejo` 3/4 (las actas se miden por bloque, no por unidad).

**El sistema real acierta 1 de cada 3 consultas al primer intento.** El 0.946
previo medía memorización del fixture de entrenamiento.

## 4. Hallazgo principal: el fallo dominante es de ORDEN, no de recuperación

Clasificación de los 30 casos:

- 14 casos: unidad correcta en rank 1 (correcto).
- **9 casos: unidad correcta recuperada pero fuera del primer puesto.**
- 7 casos: unidad correcta ausente del top-5 (fallo de recuperación real).

Es decir, **en 23 de 30 casos (77%) la evidencia correcta sí se recupera**; en 9
de ellos queda mal ordenada. Si el orden fuera perfecto, `hit@1` pasaría de 0.333
a ~0.767 sin tocar embeddings, chunking ni OCR.

Causa mecánica, verificada en el código: después del reranker hay **cuatro etapas
que reordenan** — `_attach_relevant_segments`, `_apply_visibility_policy`,
`_diversify_hits` y `_prioritize_answer_hits`, que construye la lista final como
`[*anchor_hits, *answer_hits, *reranked_hits]`. Resultado medido: **25 de 30 casos
(83%) tienen el top-5 final desordenado respecto a `rerank_score`.**

La generación amplifica el error de orden, porque el prompt instruye "resuelve
primero la respuesta directa con la evidencia primary" y trata la posición como
prioridad. Ejemplos reales de esta corrida:

- `h001` "¿Cuántos días de vacaciones le corresponden a un trabajador?" — el
  Arto. 76, que dice literalmente "quince días", se recupera en **rank 5**, y la
  respuesta es *"los textos proporcionados no especifican cuántos días de
  vacaciones corresponden"*. Tenía la respuesta en el contexto.
- `h002` "Si me despiden sin justificación, ¿qué indemnización me toca?" — el
  Arto. 45 se recupera en rank 3 y el sistema responde sobre indemnización por
  **enfermedad profesional** (Art. 127, rank 1).

## 5. Segundo hallazgo: el Reglamento del ISSDHU es un PDF escaneado con OCR malo

De los 8 documentos del corpus, **el Reglamento Interno es el único sin capa de
texto** (pdfplumber extrae 0 caracteres en las 36 páginas). Docling lo procesó con
su OCR interno y el resultado tiene **18.5% de tokens pegados** frente a 0.02–0.05%
en los documentos nativos: `"Delassiglasyconceptos"`, `"Institutode Seguridad
Social"`, `"depemsioapormuerte"`.

Es el documento propio de la institución y presumiblemente el más consultado.
Ningún control lo detecta: `review_required` solo se activa si el texto extraído
mide menos de 50 caracteres, así que 200.000 caracteres de OCR degradado pasan
como `extracted`. La config `ocr_enabled` / `ocr_language` no se usa en ninguna
parte del código.

## 6. Tercer hallazgo: el reranker configurado no está corriendo

`.env` fija `RERANK_MODE=local` apuntando a `http://localhost:7997`, que **no
responde**. Cada consulta cae en silencio al rerank heurístico
(`reranking.py` devuelve `status="fallback"` sin registrar nada). La API key de
Cohere está configurada y sin usar. Todo el baseline de arriba se midió, por
tanto, **sin reranker neuronal**.

## 7. Abstención: el sistema no alucina, pero su confianza no significa nada

8 de 8 preguntas sin respuesta fueron correctamente abstenidas — resultado
genuinamente bueno, incluso en las trampas (no inventó licencia de paternidad ni
resultados financieros de 2030).

Pero en esos 8 casos el sistema reportó **`confidence = 0.95`** y emitió **8 citas**
mientras decía que no había evidencia. La fórmula es
`min(0.95, 0.35 + rerank_score*3)`, sin calibración: satura en 0.95 casi siempre.

## 8. Qué NO se cambió en esta fase

Nada del sistema. No se tocó configuración, prompt, índices ni fixtures
existentes, para que el baseline refleje el estado real. En particular, la
separación del fixture de entrenamiento del router sigue pendiente: debe hacerse
en Fase 1 y remedirse contra este baseline.

## 9. Primera corrección aplicada: reranker levantado

Se arrancó el microservicio `apps/reranker` en GPU (`BAAI/bge-reranker-v2-m3`,
device `cuda`). No hizo falta instalar nada ni cambiar configuración: `.env` ya
apuntaba a `localhost:7997`, solo que el proceso no estaba corriendo.

Nota para quien retome esto: Ollama **no** sirve como backend de rerank para este
sistema. Devuelve 404 en `/rerank`, `/api/rerank` y `/v1/rerank`, y el modelo
`qwen3-embedding:8b` instalado es un *embedder*, no un reranker.

Remedición sobre el mismo fixture honesto, sin ningún otro cambio:

| Métrica | Sin reranker | Con reranker | Δ |
|---|---|---|---|
| unit_hit@1 | 0.333 | **0.533** | +0.200 |
| unit_hit@3 | 0.500 | **0.700** | +0.200 |
| MRR | 0.437 | **0.611** | +0.174 |
| Casos pasados | 18/30 | **24/30** | +6 |
| Latencia | 41.6 s | **35.6 s** | −6.0 s |

Mejoró también la latencia, porque el reranker acorta el trabajo posterior.
Por corpus: `ley_822` triplicó su `hit@1` (0.25 → 0.75), `codigo_del_trabajo`
pasó a 11/12 y `reglamento_issdhu` a 7/10 — sigue siendo el peor, coherente con
su OCR degradado.

Los dos casos testigo quedaron corregidos:

- `h001` — Arto. 76 pasó de rank 5 a **rank 1**, y la respuesta ahora es correcta:
  *"le corresponden quince días de descanso continuo y remunerado por cada seis
  meses de trabajo ininterrumpido"*.
- `h002` — Arto. 45 pasó de rank 3 a **rank 1**, y ya no responde sobre
  enfermedad profesional.

**El desorden post-rerank sigue presente: 26 de 30 casos** conservan un top-5 que
no respeta `rerank_score`. El reranker mejoró la señal, pero las cuatro etapas
posteriores la siguen degradando. Reparto de fallos actualizado: 5 de
recuperación (evidencia ausente del top-5) y 5 de orden (evidencia recuperada
pero fuera del primer puesto). Techo si se corrigiera solo el orden:
`hit@1` ≈ 0.833.

## 10. Segunda corrección: ablación de las etapas que reordenan

Se instrumentó el pipeline (`app/evals/trace_reorder_stages.py`) para atribuir el
desorden a una etapa concreta antes de tocar nada. Resultado sobre 7 casos:

- `_attach_relevant_segments` — **nunca** altera el orden (solo anota payload).
- `_diversify_hits` — no alteró el orden en ningún caso (solo actúa con
  `plan.diversity_required` o consultas comparativas).
- `_apply_visibility_policy` — **reordena siempre**, y con daño demostrable: en
  `h015` bajó del puesto 1 al 2 un chunk de score 2.308 para promover uno de
  2.14; en `h004` promovió un chunk de score 1.229 por delante de otro de 2.162.
- `_prioritize_answer_hits` — reordena, unas veces a favor y otras en contra.

Se añadieron tres flags en `app/config.py`
(`retrieval_visibility_policy_enabled`, `retrieval_diversify_enabled`,
`retrieval_answer_prioritization_enabled`), con default `True` para conservar el
comportamiento histórico, y se midió cada configuración sobre los 30 casos:

| Métrica | Control | visibility OFF | visibility+prioritize OFF |
|---|---|---|---|
| unit_hit@1 | 0.533 | **0.567** | 0.567 |
| unit_hit@3 | 0.700 | **0.700** | 0.667 |
| MRR | 0.611 | **0.622** | 0.606 |
| Casos pasados | 24 | **24** | 23 |

Conclusión: **desactivar la política de visibilidad mejora** (+0.034 `hit@1`,
+0.066 `strict_hit@1`, y algo más rápido), pero **`_prioritize_answer_hits` sí
aporta** — quitarla degrada `hit@3` y pierde un caso. Se conserva.

Aplicado en `.env`: `RETRIEVAL_VISIBILITY_POLICY_ENABLED=false` (respaldo previo
en `.env.bak_prefase1`).

Corrección honesta a la estimación anterior: el techo de ~0.833 por arreglo de
orden era optimista. La política de visibilidad a veces ayudaba y a veces dañaba,
y en agregado su retirada vale +0.034, no +0.30.

## 11. Estado consolidado

| Métrica | Inicial | +reranker | +visibility OFF | Total |
|---|---|---|---|---|
| unit_hit@1 | 0.333 | 0.533 | **0.567** | +0.234 |
| unit_hit@3 | 0.500 | 0.700 | **0.700** | +0.200 |
| strict_hit@1 | 0.267 | 0.467 | **0.533** | +0.266 |
| MRR | 0.437 | 0.611 | **0.622** | +0.185 |
| Casos pasados | 18/30 | 24/30 | **24/30** | +6 |
| Latencia | 41.6 s | 35.6 s | **31.5 s** | −24% |

Desorden residual: 19/30 (era 26/30). Fallos restantes: **5 de recuperación**
(`h008`, `h013`, `h016`, `h017`, `h023` — la evidencia no llega al top-5) y
**4 de orden** (`h004`, `h007`, `h010`, `h021`).

## 12. Tercera corrección: OCR en español (mejoró el texto, no las métricas)

Causa raíz encontrada: `DoclingExtractor` llamaba a `DocumentConverter()` **sin
opciones**, y el default de Docling 2.81 trae `lang=[]` — OCR sin idioma. Se
corrigió en `app/services/extractors.py`: ahora detecta si el PDF trae capa de
texto (`_has_text_layer`) y, si no, configura Tesseract con los idiomas de
`ocr_language` (config que hasta ahora estaba muerta) y `force_full_page_ocr`.
El backend queda registrado como `docling+ocr[spa+eng]` para trazabilidad.

Calidad del texto del Reglamento, reingerido como `v_bf61dac6fb4b`:

| | OCR anterior | OCR nuevo |
|---|---|---|
| Palabras pegadas | 18,5 % | **0,04 %** |
| Palabras con tilde | 0,14 % | **11,9 %** |
| Artículos con chunk propio | 145 | **151** |
| Caracteres | 106.702 | 140.401 |

Recupera 6 artículos que **no existían en el índice** (20, 70, 79, 90, 108, 111)
y no pierde ninguno. No-regresión verificada: un PDF nativo (acta 208) produce
exactamente los mismos 36.278 caracteres y conserva sus 37 tablas.

**Pero el impacto en las métricas fue nulo o levemente negativo:**

| Métrica | pre-OCR | con OCR |
|---|---|---|
| unit_hit@1 | 0.567 | 0.533 |
| unit_hit@3 | 0.700 | 0.700 |
| Casos pasados | 24/30 | 24/30 |

La diferencia de `hit@1` es **un solo caso** (h015 pasó de rank 1 a 2), dentro del
ruido con n=30. Ninguno de los tres casos que se esperaba arreglar (`h013`,
`h016`, `h017`) mejoró.

### Por qué: la hipótesis era incorrecta

El análisis caso por caso mostró que esos tres fallos **no eran de OCR**:

- **`h013` y `h016` son fallos de ROUTER.** Son los dos únicos casos de todo el
  fixture donde el router clasificó la pregunta como `general_document`
  ("¿A qué edad me puedo pensionar por vejez?", "¿Quién puede modificar el grado
  de discapacidad?"). Como el filtro de clase es restrictivo, **el Reglamento
  queda excluido de la búsqueda** y el sistema recupera actas del Consejo
  Directivo. Ningún arreglo de OCR podía corregirlo: el documento correcto ni
  siquiera entraba en el espacio de búsqueda.
- **`h017` es competencia léxica.** La pregunta ("¿cuánto se descuenta del salario
  para cotizar?") recupera el Arto. 20 ("Salario objeto de cotización en el
  Programa de Afiliación Voluntaria") en vez del Arto. 16 ("Tasa de cotización de
  afiliado activo"). Ambos son defensibles; el target del fixture es discutible.
  Nota: el Arto. 20 es uno de los 6 artículos que el OCR nuevo recuperó, o sea que
  la propia corrección introdujo un competidor legítimo.

### Decisión: se mantiene

Se conserva la reingesta pese a no mover las métricas, por corrección de datos y
no por rendimiento: seis artículos del reglamento vigente simplemente **no
existían** en el índice y cualquier consulta sobre ellos era imposible de
responder; el fixture no los cubre, pero los usuarios sí pueden preguntarlos.
Además, el texto legible es prerrequisito para el analizador español de BM25.

Estado reversible: la versión vieja `v_f007334278f1` quedó marcada `superseded` y
sus vectores se borraron de ambos índices, pero **sus 149 chunks siguen en
SQLite** con `index_status='pending'`. Para revertir: poner `version_status` en
`active` y correr reindex con `force=true` sobre esa versión.

Lección metodológica: la calidad del insumo y la calidad de la respuesta son ejes
distintos. El texto estaba objetivamente roto y arreglarlo era correcto, pero la
causa de esos fallos concretos estaba en otra etapa. Sin el eval, la reingesta se
habría dado por exitosa mirando solo el texto.

## 13. Cuarta corrección: el filtro de clase deja de excluir

Diagnóstico del router sobre los 30 casos: **acierta 27/30**, pero su confianza
**no separa aciertos de errores** — `h013` falla con 0.60 mientras `h008` y `h015`
aciertan con 0.48. Por tanto no existe un umbral que distinga cuándo fiarse; lo
que sí se puede evitar es que un error del router sea irreversible.

Antes, `target_classes` entraba como `must` en Qdrant y como `filter` en
OpenSearch: una clase mal predicha **excluía del espacio de búsqueda** al
documento correcto, y ninguna etapa posterior podía recuperarlo.

Cambio: `QueryRoute` transporta ahora `confidence` y `hard_class_filter`. Por
encima de `query_router_hard_filter_confidence` (default **0.95**) la clase sigue
filtrando de forma excluyente; por debajo se degrada a preferencia — se retira del
filtro y entra como `should` con `boost: 2.0` en la búsqueda sparse
(`_class_preference_clauses`). El `decision_trace` registra
`router:class_filter=hard|soft`.

| Métrica | pre-router | umbral 0.90 | umbral 0.95 |
|---|---|---|---|
| unit_hit@1 | 0.533 | 0.600 | **0.600** |
| unit_hit@3 | 0.700 | 0.767 | **0.767** |
| MRR | 0.611 | 0.678 | **0.678** |
| block_hit@3 | 0.750 | 0.750 | **1.000** |
| Casos pasados | 24/30 | 26/30 | **27/30** |

`h013` y `h016` pasaron de no recuperar la unidad a **rank 1**, y `h029` (actas)
se recuperó al subir el umbral a 0.95. **Ningún caso se degradó**: Código del
Trabajo, Ley 822 y actas mantuvieron o mejoraron sus cifras.

Efecto secundario: la reingesta OCR de la sección 12, que por sí sola no movió las
métricas, se revela útil en combinación — el Reglamento pasó de 7/10 a **9/10**
casos, porque el router ya no excluye el documento y el texto legible sí compite.

## 14. Estado consolidado

| Métrica | Inicial | Final | |
|---|---|---|---|
| unit_hit@1 | 0.333 | **0.600** | +80 % |
| unit_hit@3 | 0.500 | **0.767** | +53 % |
| MRR | 0.437 | **0.678** | +55 % |
| block_hit@3 | 0.750 | **1.000** | |
| Casos pasados | 18/30 | **27/30** | +9 |
| Latencia | 41.6 s | ~36 s | −13 % |

Los tres casos restantes no son fallos de router (los tres se enrutan bien):

- `h008` "¿Qué pasa con mi contrato si la empresa cambia de dueño?" → Arto. 11
  ("sustitución del empleador"): distancia léxica pregunta/texto.
- `h017` → recupera Arto. 20 en vez del Arto. 16; **el target del fixture es
  discutible**, ambos son defendibles.
- `h023` "¿Quiénes están obligados a pagar impuesto sobre la renta?" → Arto. 18.

Los tres apuntan a la misma palanca pendiente: vocabulario. Es el turno del
analizador español con `asciifolding` y stemming en OpenSearch.

## 15. Quinta corrección: confianza basada en el cross-encoder

Diagnóstico previo (sección 7): el sistema reportaba `confidence = 0.95` en las
8 preguntas sin respuesta, mientras decía que no había evidencia. La fórmula
`min(0.95, 0.35 + rerank_score*3)` saturaba casi siempre.

Al rastrear el pipeline apareció la señal que faltaba: el **score crudo del
cross-encoder** sí tiene significado absoluto — ~0.99 cuando la evidencia
responde la pregunta, ~0.002 cuando no. Pero se perdía:
`_normalize_provider_scores` aplica min-max sobre los candidatos, de modo que un
conjunto con scores 0.001–0.002 se normaliza igual que uno con 0.5–0.99.

Cambio: `RetrievalHit` conserva `provider_rerank_score` (el valor crudo, antes de
normalizar) y `_confidence` lo usa, tomando el **máximo de los hits finales** —no
el del primero— porque las etapas de reordenamiento pueden no dejar arriba al
mejor. Si no hay proveedor de rerank (modo `off` o fallback heurístico) se
conserva la fórmula histórica, y el `decision_trace` registra
`confidence:source=provider_rerank|heuristic_fallback`.

### Poder discriminante medido

| Grupo | n | Media | Rango |
|---|---|---|---|
| Acierta en rank 1 | 21 | **0.904** | 0.73 – 0.95 |
| No acierta en rank 1 | 9 | 0.502 | 0.03 – 0.95 |
| Pregunta sin respuesta | 8 | 0.546 | 0.01 – 0.95 |

Separación entre aciertos y preguntas sin respuesta: **+0.358**. Antes, los tres
grupos valían 0.95.

Sobre el umbral: una primera lectura sugería que `confidence < 0.5` era un corte
seguro, porque ninguno de los 21 aciertos **en rank 1** bajaba de 0.73. Esa
lectura era incompleta y la sección 16 la corrige: el sistema también responde
bien con evidencia en rank 2–3, y esos casos sí puntúan bajo.

**Limitación importante y asimétrica:** 5 de las 8 preguntas sin respuesta siguen
dando confianza alta (u004 y u008 en 0.95, u005 en 0.89). El cross-encoder mide
relevancia **temática**, no si el pasaje responde exactamente la pregunta. Por
tanto: confianza baja es señal fiable de evidencia débil; confianza alta **no**
garantiza que la respuesta sea correcta. No debe presentarse como probabilidad de
acierto.

No-regresión verificada: `hit@1`, `hit@3`, MRR y casos pasados idénticos (27/30).

## 16. Sexta corrección: abstención por umbral — probada y descartada

Se implementó el corte derivado de la sección 15: si ni el mejor pasaje supera
`answer_min_confidence`, abstenerse sin llamar al LLM. Y se reforzó el prompt de
síntesis para que la prioridad de la evidencia dependa de la relevancia medida y
no de su posición (`AnswerSynthesisEvidence.relevance` lleva ahora el score crudo
del cross-encoder, y las reglas dicen explícitamente que el orden no indica
importancia).

Resultado en preguntas sin respuesta: la abstención subió de 0.75 a **0.875**, y
3 de 8 casos se resolvieron **sin gastar la llamada al LLM**.

**Pero el corte descarta respuestas correctas.** Sobre los 30 casos contestables
cortó 4, y **2 de ellos (`h004`, `h021`) tenían la respuesta en el top-3**:

| Grupo | Confianzas observadas |
|---|---|
| Contestables con respuesta en top-3 | 0.03, 0.04, 0.64, 0.73 … 0.95 |
| Preguntas sin respuesta | 0.01, 0.08, 0.40, 0.52, 0.57, 0.89, 0.95, 0.95 |

Los rangos se solapan de extremo a extremo: los buenos bajan a 0.03 y los malos
suben a 0.95. **No existe umbral que separe ambos grupos.** El error de la
calibración anterior fue medir solo sobre aciertos en rank 1; el sistema responde
bien también con evidencia en rank 2–3, y ahí el cross-encoder puntúa bajo.

Decisión: `answer_min_confidence` queda en **0.0 (desactivado)** por defecto. El
mecanismo permanece implementado y configurable para quien prefiera un sistema
más conservador, con el coste conocido de perder ~7 % de respuestas correctas.

Sobre el prompt: las respuestas con y sin `relevancia` explícita son de calidad
comparable, sin mejora ni daño medible. Se conserva por ser conceptualmente
correcto —desacopla prioridad de posición— pero **no hay evidencia de que ayude**.

Hallazgo lateral: `h014` sigue respondiendo mal *con el pasaje correcto en rank
1*, porque el chunk del Arto. 56 llega **truncado** ("comienza a listar
beneficiarios..."). Es un problema de chunking, no de prompt ni de recuperación:
una pista para la siguiente iteración.

## 17. Séptima corrección: la evidencia llegaba mutilada al LLM

Al comparar modelos de síntesis apareció que los tres (`gpt-5-mini`, `gpt-5`,
`gpt-5.6-luna`) fallaban en **los mismos casos** y con la misma queja: *"el
fragmento proporcionado no enumera cuáles son"*, *"el pasaje está incompleto"*.
No era el modelo.

Causa: `_question_focused_evidence_text` no truncaba, hacía algo peor. Partía el
chunk por puntuación (`_candidate_evidence_fragments`) y enviaba al LLM **un solo
fragmento**, el de mayor solapamiento léxico con la pregunta. Para el Arto. 20
("El contrato escrito debe contener: a)… b)… c)…") el fragmento ganador era el
encabezado *"debe contener:"* y la lista entera se descartaba. Con el límite de
650 caracteres, solo el **51,9 %** de los chunks cabía completo.

Corrección: si el pasaje entra en el presupuesto se envía **entero**, sin elegir
fragmento; la selección solo actúa cuando el chunk excede el límite. Y el límite
pasa a ser configurable (`answer_evidence_char_limit`, default **1500**), con lo
que cabe el 77,5 % de los chunks y el prompt queda en ~3000 tokens.

| Configuración | Hedging con evidencia correcta | Citas |
|---|---|---|
| mini + límite 650 (inicial) | 8/25 | 8/26 |
| luna + límite 650 | 9/25 | 14/26 |
| **luna + límite 1500** | **2/25** | 12/26 |

Reducción del hedging del **78 %**, con el retrieval intacto (27/30). Ejemplos
reales de la misma pregunta, antes y después:

- *"¿Qué datos debe incluir un contrato escrito?"* — antes: *"el fragmento no
  enumera cuáles son"*; ahora: *"lugar y fecha de celebración; identificación y
  domicilio de las partes…"*.
- *"Si fallece un afiliado, ¿su esposa tiene derecho a pensión?"* — antes: *"la
  evidencia no permite confirmar"*; ahora: *"Sí. La esposa, en calidad de
  cónyuge, tiene derecho a pensión de viudez"*.
- *"¿Cuánto descanso por embarazo y parto?"* — antes: *"no indica cuánto dura"*;
  ahora: *"cuatro semanas antes del parto y ocho después; diez en parto
  múltiple"*.

### Sobre la elección de modelo

Comparados sobre 26 casos (excluyendo 4 que la corrida de `mini` abstuvo por el
corte de confianza, ya desactivado): `gpt-5` citaba mejor (20/26 frente a 8/26 de
mini), pero **la diferencia decisiva no estaba en el modelo sino en el límite**.
Se fija `gpt-5.6-luna` por criterio de costo, con `ANSWER_SYNTHESIS_CLOUD_MODEL`
explícito en `.env` — antes el modelo llegaba heredado del planner por la cadena
de fallbacks, sin que nadie lo eligiera.

También se sube `ANSWER_SYNTHESIS_CLOUD_TIMEOUT` a 90 s: con 20 s un modelo
lento caía al fallback extractivo, y ese fallback tampoco se registraba (ahora
emite `answer_synthesis_failed`).

Advertencia sobre los precios: las cifras por token provienen de fuentes
secundarias; openai.com devolvió 403 y no se pudo contrastar. Conviene
verificarlas en el panel de facturación antes de tomarlas como definitivas.

## 18. Octava corrección: control de acceso aplicado (C1)

Hallazgo C1 de la auditoría inicial, abierto desde el principio: las ACL se
guardaban por documento y se indexaban por chunk (`acl_users`, `acl_groups`),
pero **ninguna búsqueda las aplicaba**. Cualquier `user_identity` recuperaba
cualquier documento. Confirmado como bloqueante: el ISSDHU tendrá áreas con
acceso a distintos conjuntos documentales.

Cambios:

- `QueryRequest` gana `user_groups`. Antes solo llegaba `user_identity`, con lo
  que no había forma de resolver pertenencia a un área.
- `_acl_qdrant_clause` / `_acl_opensearch_clause`: un chunk es visible si nombra
  al usuario en `acl_users`, si alcanza alguno de sus grupos en `acl_groups`, o
  si no declara restricción. Se aplica en las **seis** construcciones de filtro
  (densa, sparse, ramas y catálogo de ramas).
- Los grupos se propagan a las seis subconsultas internas (variantes de
  consulta, búsqueda guiada por rama), donde antes solo viajaba la identidad.
- `acl_enforcement_enabled` (default `true`) y `acl_open_when_unrestricted`
  (default `true`: documento sin ACL = público). El corpus actual tiene ACL
  vacía en los 9 documentos, así que con este default nada deja de funcionar y
  las restricciones aplican solo desde que se declaran. Poner en `false` para
  exigir ACL explícita (deny by default).

### Dos fallos encontrados al verificar, no al escribir

1. **Qdrant devolvía HTTP 400.** `min_should` no es válido en esa posición en
   Qdrant 1.13.4. Como un `should` ya exige al menos una coincidencia, se
   eliminó. Sin la prueba de integración esto habría dejado la búsqueda densa
   caída en silencio.
2. **`_segment_neighbor_hits` consultaba OpenSearch sin ACL.** Trae chunks
   vecinos para armar segmentos; en teoría son del mismo documento ya
   autorizado, pero un control de acceso no debe descansar en un invariante
   implícito. Se propagó `request` por la cadena
   `_attach_relevant_segments` → `_build_evidence_segment` →
   `_segment_neighbor_hits`.

### Verificación de aislamiento

Marcando temporalmente el Reglamento como restringido al grupo `rrhh` en ambos
índices (chunks y ramas):

| Consulta | Sin grupo `rrhh` | Con grupo `rrhh` |
|---|---|---|
| "¿su esposa tiene derecho a pensión?" | **0 chunks** | 8 chunks |
| "¿a qué edad me puedo pensionar?" | **0 chunks** | 8 chunks |
| "¿cuánto se descuenta del salario?" | **0 chunks** | 4 chunks |

Sin el grupo, el sistema responde solo con actas y Código del Trabajo. La ACL de
prueba se revirtió tras verificar.

No-regresión: `hit@1` 0.600, `hit@3` 0.767, MRR 0.678, 27/30 — idénticos.

Pendiente derivado: la ingesta ya acepta `acl_users`/`acl_groups`, pero **no hay
forma de editarlos después**. Para operar de verdad hace falta un endpoint que
cambie la ACL de un documento existente y reindexe sus chunks.

## 19. Fixture ampliado a 150 preguntas: el de 30 subestimaba el sistema

El fixture de 30 se quedó sin resolución (27/30 resueltos, cada caso valía 3,3
puntos) y dejaba fuera NIIF y GAFI, 649 chunks del corpus. Se amplió a 150
conservando las 30 originales y sus ids.

Cobertura: `codigo_del_trabajo` 32, `ley_822` 32, `reglamento_issdhu` 28,
`gafi` 22, `niif_pymes` 21, `actas_consejo` 15. Los 150 targets se verificaron
como alcanzables contra el índice antes de medir.

Las métricas globales del runner mezclan dos poblaciones y no deben leerse
juntas: GAFI y actas se evalúan por **bloque** (sus `unit_hit` son 0 por
diseño). Separadas:

| | Casos | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|---|
| Medidos por unidad | 113 | **74.3 %** | **91.2 %** | 94.7 % | 0.826 |
| Medidos por bloque | 37 | 81.1 % | 94.6 % | 97.3 % | — |

**Global: 138/150 resueltos (92 %).** Latencia 24,4 s/consulta.

Por corpus: `niif_pymes` 21/21, `actas_consejo` 15/15, `reglamento_issdhu`
27/28, `codigo_del_trabajo` 30/32, `gafi` 20/22, `ley_822` 25/32.

### El fixture chico subestimaba

Con 30 preguntas el `hit@1` medía 0.60; con 113 casos de unidad mide **0.743**.
La diferencia no es que el sistema mejorara —no cambió nada entre ambas
medidas— sino que el fixture de 30 estaba sesgado hacia casos difíciles: se
construyó eligiendo deliberadamente sinónimos y brechas de vocabulario para
sondear límites, y esa selección no representa la distribución real de
consultas. Es un recordatorio de que un fixture pequeño no solo tiene ruido:
puede tener sesgo sistemático.

También confirma el valor de la reingesta OCR: `reglamento_issdhu`, el peor
corpus al empezar, ahora resuelve 27 de 28.

### Dónde se concentran los fallos

7 de los 12 fallos son de `ley_822`, y el patrón es consistente: preguntas
conceptuales cuya respuesta está en los artículos iniciales de definiciones
(Art. 2 principios, 5 territorio, 6 valoración, 10 fuente nicaragüense) que
pierden contra artículos operativos de numeración alta. `h065` ("¿qué
principios rigen los tributos?") recupera los artículos 146, 319 y 16.

Los otros cinco ya eran conocidos o son de vocabulario: `h008` y `h017`
(brechas semánticas donde solo el denso acierta), `h037`, `h127` y `h128`
(GAFI, donde la pregunta usa lenguaje llano y el documento término técnico).

## 20. Extracción por lotes: NIIF y GAFI perdían la mitad del documento

`DocumentConverter.convert` acepta `page_range`, así que los PDFs de más de
`extraction_batch_threshold_pages` (60) se convierten en tramos de
`extraction_batch_pages` (40) y se concatena el markdown. Esto acota el pico de
memoria de docling, que era la causa del `std::bad_alloc`.

| Documento | Indexado antes | Con troceo | Páginas perdidas |
|---|---|---|---|
| NIIF para PYMES (276 pág.) | 413.439 | **878.379** | 107 → **0** |
| GAFI (381 pág.) | 519.179 | **1.068.519** | — → **0** |
| Ley 822 (65 pág.) | 550.647 | 556.471 | 0 → 0 |
| Acta 208 (14 pág., no se trocea) | 36.278 | 36.278 | 0 → 0 |

Los dos documentos más largos del corpus **tenían indexada menos de la mitad de
su contenido**. Las 14 secciones ausentes de NIIF (16-29: Arrendamientos,
Provisiones, Ingresos, Beneficios a los Empleados, Deterioro, Impuesto a las
Ganancias) reaparecen con 20 a 68 referencias a sus párrafos cada una.

### Dos fallos que aparecieron al reingerir

**El upsert enviaba todos los puntos en una sola petición.** 654 chunks de 3072
dimensiones son ~25 MB solo de vectores: por encima del límite de body de
Qdrant. La indexación fallaba entera con HTTP 400 **después** de gastar 56
minutos generando los resúmenes por chunk. Con los 362 chunks de la versión
anterior no se llegaba al límite, así que el problema apareció justo al mejorar
la extracción. Ahora se escribe en lotes de 64 en Qdrant y OpenSearch.

**El script de reingesta retiraba la versión vieja sin comprobar la nueva.**
Cuando la indexación falló, retiró la anterior igualmente: ni la vieja ni la
nueva quedaron en el índice, y el corpus estuvo sin NIIF durante horas.
`app/tools/reingest_document.py` cuenta ahora los documentos realmente escritos
en ambos motores y aborta dejando la versión anterior intacta si no coinciden.
Incluye `--index-only` para reanudar sin repetir la extracción.

### Coste operativo que esto deja al descubierto

Un reindexado de 654 chunks tarda ~55 minutos, casi todo en generar un resumen
por chunk con el LLM local. **Los resúmenes no se cachean**: cualquier reintento
los recalcula íntegros, y un fallo al final tira el trabajo entero. Era el
hallazgo M2 de la auditoría inicial; aquí se ve su costo real. Cachearlos por
hash del texto del chunk convertiría un reintento de 55 minutos en uno de
segundos.

## 21. Baseline sobre el corpus completo (161 preguntas)

Primera medición con todo el contenido indexado. El corpus pasó de 1.810 a
**2.469 chunks**: un 36 % no era consultable y ninguna métrica lo mostraba,
porque el fixture no preguntaba por lo que faltaba.

| Métrica | 150 preg. (corpus incompleto) | 161 preg. (completo) |
|---|---|---|
| hit@1 por unidad | 74,3 % | **76,6 %** |
| hit@3 por unidad | 91,2 % | 90,3 % |
| MRR | 0,826 | **0,836** |
| block@3 | 94,6 % | 94,6 % |
| Global | 138/150 (92,0 %) | **147/161 (91,3 %)** |

Por corpus: `actas_consejo` 15/15, `reglamento_issdhu` 27/28, `codigo_del_trabajo`
30/32, `niif_pymes` 30/32, `gafi` 20/22, `ley_822` 25/32.

**9 de las 11 preguntas sobre las secciones recuperadas se responden ahora, todas
en rank 1**: propiedades de inversión, depreciación, arrendamientos, provisiones,
ingresos, deterioro, combinaciones de negocios, pasivo vs. patrimonio y pagos
basados en acciones. Antes ninguna era planteable.

NIIF pasó de 21 a 32 preguntas y aun así mejoró a 30/32 con `hit@1` de 0,875 — el
mejor del corpus junto al Reglamento. El global baja tres décimas solo porque el
fixture incorpora 11 preguntas que antes eran imposibles: es una medición más
honesta sobre un corpus mayor, no un retroceso.

### Los 14 fallos restantes

- **Ley 822 concentra 6** (arts. 2, 6, 10, 33, 41, 45). Patrón consistente:
  preguntas conceptuales cuya respuesta está en los artículos de definiciones
  iniciales, que pierden contra artículos operativos de numeración alta. Es el
  único corpus por debajo del 80 %.
- **Dos de NIIF resisten** pese a estar ahora completas: Sección 28 (Beneficios a
  los Empleados, 39 chunks) y Sección 29 (Impuesto a las Ganancias, 25 chunks).
  Secciones grandes cuyo contenido compite consigo mismo.
- Los otros seis ya eran conocidos: brechas de vocabulario donde solo el
  recuperador denso acierta.

## 22. Estado del plan

Punto de partida: `hit@1` 0.333 sobre 30 preguntas, corpus de 1.810 chunks.
Estado actual: `hit@1` **0.766** sobre 124 casos de unidad, corpus de 2.469
chunks, 147/161 preguntas resueltas.

### Hecho

| # | Intervención | Efecto medido |
|---|---|---|
| 1 | Levantar el reranker (estaba configurado pero muerto) | **+0.200** `hit@1` |
| 2 | Ablación de etapas post-rerank; se retira la política de visibilidad | +0.034 |
| 3 | Filtro de clase del router como preferencia bajo 0.95 de confianza | +0.067, +3 casos |
| 4 | Evidencia completa al LLM (límite 650 → 1500, sin fragmentar) | hedging −78 % |
| 5 | OCR en español para PDFs escaneados | Reglamento: 7/10 → 27/28 |
| 6 | Extracción por lotes de PDFs largos | corpus 1.810 → **2.469 chunks** |
| 7 | ACL aplicadas en las búsquedas (C1 crítico) | aislamiento verificado |
| 8 | Logging de degradación silenciosa (C4 crítico) | 6 puntos ciegos cubiertos |
| 9 | Gate de calidad e integridad en la ingesta | detecta OCR malo y páginas perdidas |
| 10 | Escritura al índice por lotes | corrige HTTP 400 en documentos grandes |
| 11 | Reingesta que verifica antes de retirar | evita dejar el corpus sin un documento |
| 12 | Confianza basada en el cross-encoder | 0.904 al acertar vs 0.546 sin respuesta |
| 13 | Modelo de síntesis elegido explícitamente (`gpt-5.6-luna`) | antes lo heredaba del planner |
| 14 | Fixture honesto: 30 → 161 preguntas, todo el corpus | resolución 3,3 → 0,6 pts/caso |

Probado y descartado con evidencia: el corte automático por umbral de confianza
(descartaba respuestas correctas) y exponer la relevancia en el prompt (sin
efecto medible).

| 15 | Cache de resúmenes por chunk (hallazgo M2) | reindexado **−93 %** |

El cache usa como clave el hash del prompt exacto más el modelo, de modo que
cualquier cambio en el texto del chunk, en los metadatos que entran al prompt o
en el modelo genera una clave distinta: no hace falta versionarlo a mano. Vive
en su propio SQLite (`.data/summary_cache.db`) para no competir por el lock de
la base principal; borrar ese archivo invalida el cache entero.

Medido reindexando dos veces el acta 207 (41 chunks): **231,8 s → 16,4 s**. Una
llamada suelta pasa de 19,3 s a 0,00 s con resultado idéntico.

| 16 | Mezcla ponderada provider/heurístico en el rerank | **+4,0** `hit@1` |
| 17 | Corregir la degradación por VRAM del microservicio reranker | lote de 139 s → **3,3 s** |

### El rerank: qué se probó y qué resultó

Diagnóstico de los 7 fallos de `ley_822`. Se descartaron con datos dos hipótesis
razonables: que los artículos de definiciones perdieran por genéricos, y que el
texto con palabras pegadas los perjudicara (los que fallan tienen 2,3 % de tokens
pegados; los que aciertan, 3,7 %).

La causa real tenía dos partes. El artículo correcto quedaba en la posición 30 de
35 del orden heurístico y `rerank_top_n=24` lo dejaba **fuera del cross-encoder**;
y aunque llegase, el score del proveedor se **sumaba** a un heurístico de mayor
escala (2-3 frente a 0-1), de modo que su opinión se diluía.

Diseño factorial 2×2 sobre las 161 preguntas:

| | 24 + suma | 48 + suma | **24 + mezcla** | 48 + mezcla |
|---|---|---|---|---|
| hit@1 | 76,6 | 76,6 | **80,6** | 80,6 |
| hit@3 | 90,3 | 91,1 | **91,1** | 91,1 |
| MRR | 0,836 | 0,840 | **0,862** | 0,864 |
| block@3 | 94,6 | 91,9 | 91,9 | 91,9 |
| Casos pasados | 147 | 147 | **147** | 147 |
| s/consulta | 25,2 | 29,8 | **23,6** | 24,8 |

**Ampliar los candidatos no aporta nada por sí solo** (hit@1 idéntico con 24 y 48
bajo la fórmula histórica). Lo que aporta los 4 puntos es la mezcla ponderada, y
lo hace igual con 24 candidatos que con 48. Se conserva `rerank_top_n=24` por ser
la celda más barata; la fórmula histórica queda accesible con
`rerank_provider_weight` negativo, para poder repetir la ablación.

Advertencia sobre el alcance: **los casos resueltos no se mueven — 147/161 en las
cuatro celdas**. La mejora es de calidad de ranking (la respuesta correcta aparece
primera más a menudo, lo que para el usuario significa menos lectura), no de
cobertura. Y `block@3` cae 2,7 puntos: se recuperan `h065` y `h060`, se pierde
`h119`.

Nota metodológica: las primeras mediciones de latencia de esta fase (100-205 s por
consulta) estaban contaminadas por la degradación de VRAM del reranker, no por la
configuración. La latencia debe medirse en frío o verificando el estado del
servicio; extrapolar de corridas largas dio tres estimaciones erróneas seguidas.

### Pendiente, por relación impacto/esfuerzo

1. **Reranker como servicio persistente.** Depende de un proceso a mano; cayó
   tres veces en una sesión y cada caída vale −0.20 de `hit@1` en silencio.
3. **Ley 822**: concentra 6 de los 14 fallos. Preguntas conceptuales cuya
   respuesta está en los artículos de definiciones, que pierden contra artículos
   operativos de numeración alta. Único corpus bajo el 80 %.
3. **Endpoint de administración de ACL.** El control de acceso funciona pero no
   se puede administrar: cambiar los permisos de un documento exige reindexarlo
   o tocar los índices a mano.
4. **Modo estricto de embeddings** (C3): hoy la caída a vectores hash se
   registra pero no se bloquea.
5. Separar el fixture de entrenamiento del router del de evaluación y
   reentrenarlo con preguntas en lenguaje natural.
6. NIIF secciones 28 y 29: completas en el índice pero aún no recuperables.
7. Analizador español con `asciifolding` en OpenSearch.
8. Deduplicación por checksum en la ingesta.
9. Ablación de `BOOSTABLE_PHRASES` y poda de los clasificadores sklearn
    entrenados con pocos ejemplos.
10. Revisar los targets marcados `proposed_auto` en el fixture.
11. Exponer la confianza en la UI como señal informativa, nunca como
    probabilidad de acierto.
