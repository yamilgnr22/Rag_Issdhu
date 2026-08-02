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
3. **Todas** las preguntas del gold set incluyen ya el número de artículo
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

## 18. Orden recomendado para lo que queda

1. ~~Levantar el reranker~~ — hecho, +0.20 en `hit@1`.
2. ~~Ablación de etapas post-rerank~~ — hecho, +0.034 adicional.
3. ~~Reingerir el Reglamento con OCR en español~~ — hecho; corrigió el texto pero
   no movió las métricas (ver sección 12).
4. ~~Arreglar el router~~ — hecho por la vía (a): filtro de clase como
   preferencia bajo 0.95 de confianza. +0.067 `hit@1`, +3 casos. Queda pendiente
   la vía (b): separar el fixture de entrenamiento del router del de evaluación
   —pendiente desde la sección 8— y reentrenarlo con preguntas en lenguaje
   natural.
5. Gate de calidad de extracción en la ingesta (paso 3 del plan de ingesta, aún
   sin implementar): hoy `review_required` solo salta bajo 50 caracteres, así que
   el próximo escaneado malo volverá a pasar inadvertido. Umbrales validados:
   tokens pegados < 2 %, palabras acentuadas > 5 %.
6. Analizador español + asciifolding en OpenSearch.
7. Revisar los targets discutibles del fixture (`h017`, y los de `ley_822` y
   actas marcados en su día como `proposed_auto`).
8. ~~Desacoplar la prioridad de la evidencia en el prompt de su posición~~ —
   hecho, sin efecto medible (sección 16).
9. ~~Calibrar o retirar el `confidence`~~ — hecho (sección 15). El corte
   automático se probó y se descartó por solapamiento de rangos (sección 16).
   Queda: exponer la confianza en la UI como señal informativa, **sin**
   presentarla como probabilidad de acierto.
11. ~~Chunks truncados~~ — resuelto (seccion 17): la evidencia se enviaba mutilada al LLM.
    porque el chunk llega cortado. Revisar el corte de artículos largos en
    `normative_hierarchical` y el límite de 650 caracteres de
    `_question_focused_evidence_text`.
10. Dejar el reranker como servicio persistente: hoy depende de un proceso
    arrancado a mano, y si muere el sistema degrada en silencio al heurístico.
