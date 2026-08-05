# Fase 1 — Calidad de las respuestas

Continuación de `fase0_baseline_honesto.md`. Aquel documento cubre la
recuperación; este cubre la generación, que es donde queda margen medible.

## Por qué este plan

El trabajo de la fase anterior llevó la recuperación de `hit@1` 0.333 a 0.806 y
el corpus de 1.810 a 2.469 chunks. Pero las **cuatro últimas mediciones dieron
147/161 casos resueltos, invariable**: con 14 fallos sobre 161 preguntas, el
fixture ya no distingue una mejora real del ruido. Esa veta está agotada.

En cambio, la generación —redactar la respuesta a partir de la evidencia
recuperada— **nunca se midió sistemáticamente**. Se evaluó una sola vez, sobre
30 preguntas y un único eje (hedging), y el modelo actual (`gpt-5.6-luna`) se
eligió por costo, no por calidad demostrada.

Y "responde mal" fue el síntoma original del proyecto.

## Fase 0 — Construir el termómetro de generación

Hoy no existe. Todo lo demás depende de esto.

### 0.1 Corpus de respuestas

Correr las 161 preguntas de `queries_honest_v3.jsonl` con síntesis activa, más
las 8 sin respuesta de `queries_honest_unanswerable_v1.jsonl`.

**Prerrequisito verificado antes de lanzar**: el reranker debe responder en
`/health`. Tres mediciones de la fase anterior salieron mal por correr contra un
servicio degradado.

### 0.2 Rúbrica: cuatro ejes binarios

No "¿qué tan buena es la respuesta?" —eso no se puede auditar ni comparar entre
corridas—, sino cuatro preguntas de sí o no:

| Eje | Pregunta |
|---|---|
| **Dato exacto** | ¿Da la cifra o el plazo concreto, o parafrasea en vago? |
| **Cita correcta** | ¿La etiqueta `[n]` apunta al pasaje que sostiene la afirmación? |
| **Completitud** | Si la evidencia tiene 4 causales, ¿aparecen las 4? |
| **Hedging injustificado** | ¿Dice "la evidencia no especifica" teniendo el dato delante? |

### 0.3 Juez calibrado, no juez a ciegas

LLM-juez para las 161, **validado contra revisión manual de una muestra de 30**.
En la fase anterior una métrica de hedging por palabras clave resultó ruidosa y
llevó a una conclusión falsa; un juez sin calibrar repetiría ese error a escala.

Controles: el juez no sabe qué configuración generó cada respuesta, evalúa cada
eje por separado, y se reporta su acuerdo con la revisión manual.

**Puerta de fase**: baseline por eje + los fallos de generación clasificados por
tipo. Ese reparto decide dónde va el esfuerzo de la fase siguiente.

## Fase 1 — Iterar contra el eval

Regla: **un cambio por corrida**. Se acepta solo si mueve el eval de generación
**sin degradar** el de recuperación. Candidatos, por retorno esperado:

1. **Prompt por tipo de pregunta.** Cifra, lista y definición piden formatos
   distintos; hoy hay un prompt único.
2. **Comparación de modelos en serio.** Luna frente a `gpt-5.4-mini` y `gpt-5`,
   ahora con criterio de calidad medible en vez de solo costo.
3. **Presupuesto y orden de evidencia.** Subir el límite de 650 a 1500 caracteres
   fue la mayor mejora de generación de la fase anterior (−78 % de hedging);
   conviene ver si queda margen.
4. **Abstención con señal compuesta.** La confianza sola no separaba los grupos;
   con datos de generación quizá una señal combinada sí.

## Fase 2 — Usuarios reales (en paralelo)

Es la fuente de mejora más barata: cada fallo real es un caso nuevo del fixture,
y sin sesgo de quien lo escribió.

1. **Autenticación real.** Hoy `user_groups` viene en el request, afirmado por el
   llamador: cualquiera puede declararse del grupo `rrhh`. Los grupos deben
   derivarse del token. Bloqueante para multiusuario.
2. **Endpoint de administración de ACL** (ya diseñado, media jornada).
3. **Latencia.** 25-35 s por consulta es incómodo. Primero perfilar las etapas,
   después optimizar.
4. **Captura de feedback.** `/feedback` existe como placeholder; conectarlo con
   log de consultas y pulgar arriba/abajo.

**Puerta de fase**: 5-10 usuarios piloto y 50+ consultas reales capturadas.

## Fase 3 — Lo que se reabre con más datos

Con 300+ preguntas recuperan sentido: los 5 fallos de `ley_822`, las secciones 28
y 29 de NIIF, la poda de los clasificadores sklearn y `BOOSTABLE_PHRASES`. Hoy
son inmedibles.

## Qué no hacer

- Tocar recuperación mientras se mide generación: configuración congelada.
- Confiar en el LLM-juez sin calibrarlo contra revisión humana.
- Optimizar latencia sin perfil previo.
