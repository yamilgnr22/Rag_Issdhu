import json
import re
from dataclasses import dataclass

import httpx


@dataclass
class AnswerSynthesisProviderConfig:
    provider_name: str
    base_url: str
    model: str
    api_key: str | None
    timeout: float


@dataclass
class AnswerSynthesisEvidence:
    label: str
    canonical_label: str
    branch_label: str
    unit_type: str
    unit_number: str
    chunk_kind: str
    evidence_role: str
    selection_role: str
    path_text: str
    text: str


@dataclass
class AnswerSynthesisResult:
    answer: str
    source: str
    decision_trace: list[str]


class AnswerSynthesisService:
    def __init__(self, settings) -> None:
        self.settings = settings

    def synthesize(
        self,
        *,
        question: str,
        strategy: str,
        answer_style: str,
        evidence_items: list[AnswerSynthesisEvidence],
        evidence_intent: str,
    ) -> AnswerSynthesisResult | None:
        if not evidence_items:
            return None

        mode = self.settings.answer_synthesis_mode
        if mode == "off":
            return None

        provider_order = {
            "local_llm": ["local"],
            "cloud_llm": ["cloud"],
            "hybrid": ["local", "cloud"],
        }[mode]

        for provider_name in provider_order:
            provider = self._provider_config(provider_name)
            if provider is None:
                continue
            result = self._synthesize_with_provider(
                provider=provider,
                question=question,
                strategy=strategy,
                answer_style=answer_style,
                evidence_items=evidence_items,
                evidence_intent=evidence_intent,
            )
            if result is not None:
                return result

        return None

    def _provider_config(self, provider_name: str) -> AnswerSynthesisProviderConfig | None:
        if provider_name == "cloud":
            base_url = (
                self.settings.answer_synthesis_cloud_base_url
                or self.settings.query_planner_cloud_base_url
                or self.settings.classifier_cloud_base_url
                or self.settings.classifier_llm_base_url
            )
            api_key = (
                self.settings.answer_synthesis_cloud_api_key
                or self.settings.query_planner_cloud_api_key
                or self.settings.classifier_cloud_api_key
                or self.settings.classifier_llm_api_key
            )
            model = (
                self.settings.answer_synthesis_cloud_model
                or self.settings.query_planner_cloud_model
                or self.settings.classifier_cloud_model
                or self.settings.classifier_llm_model
            )
            timeout = (
                self.settings.answer_synthesis_cloud_timeout
                or self.settings.query_planner_cloud_timeout
                or self.settings.classifier_cloud_timeout
                or self.settings.classifier_llm_timeout
                or 20.0
            )
        else:
            base_url = (
                self.settings.answer_synthesis_local_base_url
                or self.settings.query_planner_local_base_url
                or self.settings.classifier_local_base_url
            )
            api_key = (
                self.settings.answer_synthesis_local_api_key
                or self.settings.query_planner_local_api_key
                or self.settings.classifier_local_api_key
            )
            model = (
                self.settings.answer_synthesis_local_model
                or self.settings.query_planner_local_model
                or self.settings.classifier_local_model
            )
            timeout = (
                self.settings.answer_synthesis_local_timeout
                or self.settings.query_planner_local_timeout
                or self.settings.classifier_local_timeout
                or 20.0
            )

        if not base_url or not model:
            return None
        return AnswerSynthesisProviderConfig(
            provider_name=provider_name,
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=float(timeout),
        )

    def _synthesize_with_provider(
        self,
        *,
        provider: AnswerSynthesisProviderConfig,
        question: str,
        strategy: str,
        answer_style: str,
        evidence_items: list[AnswerSynthesisEvidence],
        evidence_intent: str,
    ) -> AnswerSynthesisResult | None:
        headers = {"Content-Type": "application/json"}
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"

        prompt = self._build_prompt(
            question=question,
            strategy=strategy,
            answer_style=answer_style,
            evidence_items=evidence_items,
            evidence_intent=evidence_intent,
        )

        try:
            with httpx.Client(timeout=provider.timeout) as client:
                response = client.post(
                    f"{provider.base_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    json={
                        "model": provider.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You synthesize concise grounded answers for a structured RAG system. "
                                    "Return strict JSON only."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return None

        try:
            content = payload["choices"][0]["message"]["content"]
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            parsed_as = "json"
            if match:
                data = json.loads(match.group(0))
            else:
                try:
                    data = json.loads(content)
                except Exception:
                    parsed_as = "plain_text"
                    data = {"answer": " ".join(str(content).split()).strip()}
            answer = self._coerce_answer_payload(data, answer_style)
            if not answer:
                return None
            return AnswerSynthesisResult(
                answer=answer,
                source=f"llm_{provider.provider_name}",
                decision_trace=[
                    f"answer:provider={provider.provider_name}",
                    f"answer:style={answer_style}",
                    f"answer:evidence={len(evidence_items)}",
                    f"answer:format={parsed_as}",
                ],
            )
        except Exception:
            return None

    def _build_prompt(
        self,
        *,
        question: str,
        strategy: str,
        answer_style: str,
        evidence_items: list[AnswerSynthesisEvidence],
        evidence_intent: str,
    ) -> str:
        style_instruction = {
            "single_branch": (
                "Responde en espanol con 1 parrafo corto y, si hace falta, una segunda frase breve. "
                "No listes snippets crudos. Usa citas inline como [1] o [2]."
            ),
            "grouped_by_branch": (
                "Responde en espanol con una lista plana de 2 a 4 bullets. "
                "Cada bullet debe resumir una rama o bloque relevante y cerrar con citas inline."
            ),
            "summary": (
                "Responde en espanol con una sintesis breve de 2 a 4 bullets sobre los temas principales. "
                "Cada bullet debe incluir citas inline."
            ),
        }.get(answer_style, "Responde en espanol con una sintesis breve y citas inline.")

        evidence_blocks = []
        for item in evidence_items:
            evidence_blocks.append(
                "\n".join(
                    [
                        f"{item.label}",
                        f"unidad: {item.canonical_label}",
                        f"rama: {item.branch_label}",
                        f"tipo_unidad: {item.unit_type}",
                        f"numero_unidad: {item.unit_number}",
                        f"tipo: {item.chunk_kind}",
                        f"rol: {item.evidence_role}",
                        f"seleccion: {item.selection_role or 'candidate'}",
                        f"ruta: {item.path_text}",
                        f"texto: {item.text}",
                    ]
                )
            )

        evidence_text = "\n\n".join(evidence_blocks)
        return (
            "Tarea: redacta una respuesta grounded usando solo la evidencia recuperada.\n\n"
            f"Pregunta: {question}\n"
            f"Estrategia: {strategy}\n"
            f"Estilo de respuesta: {answer_style}\n\n"
            f"Intencion de evidencia prioritaria: {evidence_intent}\n\n"
            "Reglas:\n"
            "- No inventes hechos ni cites etiquetas no provistas.\n"
            "- Si la evidencia es parcial, dilo de forma breve.\n"
            "- Usa el rol de evidencia como senal: substantive suele ser mejor que reference o editorial para responder reglas, obligaciones o unidades canonicas.\n"
            "- Usa el rol de seleccion como prioridad: primary > support > context.\n"
            "- Resuelve primero la respuesta directa con la evidencia primary; luego usa support para justificar o precisar.\n"
            "- Usa context solo si agrega marco util sin desviar la respuesta principal.\n"
            "- Si solo hay una evidencia primary, la respuesta debe abrir con esa respuesta directa y no expandirse a materias vecinas salvo que sean necesarias.\n"
            "- Si una evidencia identifica explicitamente una unidad canonica que responde la pregunta (por ejemplo una seccion, articulo o recomendacion), prioriza esa unidad sobre otras evidencias que solo mencionan el tema de paso.\n"
            "- Distingue entre la unidad que regula o aborda directamente el tema y otras unidades que solo la citan o la usan como ejemplo.\n"
            f"- {style_instruction}\n"
            "- Devuelve JSON estricto. Usa preferentemente esta forma:\n"
            '  {"answer_segments":[{"text":"...", "labels":["[1]","[2]"]}]}\n'
            "- No pongas citas dentro de text; el sistema las insertara.\n"
            "- Si no puedes, puedes devolver {\"answer\":\"...\"}, pero la forma preferida es answer_segments.\n\n"
            "Evidencia:\n"
            f"{evidence_text}\n"
        )

    def _coerce_answer_payload(self, data: object, answer_style: str) -> str:
        if isinstance(data, dict):
            structured = self._coerce_answer_segments(data.get("answer_segments"), answer_style)
            if structured:
                return structured
            return self._coerce_answer_text(data.get("answer", ""), answer_style)
        return self._coerce_answer_text(data, answer_style)

    def _coerce_answer_segments(self, value: object, answer_style: str) -> str:
        if not isinstance(value, list):
            return ""
        segments: list[str] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            text = " ".join(str(item.get("text", "")).split()).strip()
            if not text:
                continue
            labels = self._normalize_labels(item.get("labels"))
            if labels:
                label_text = "".join(labels)
                if label_text not in text:
                    text = f"{text} {label_text}".strip()
            segments.append(text)
        if not segments:
            return ""
        if answer_style in {"grouped_by_branch", "summary"}:
            return "\n".join(f"- {segment}" for segment in segments)
        return " ".join(segments)

    def _normalize_labels(self, value: object) -> list[str]:
        labels: list[str] = []
        if isinstance(value, list):
            for item in value:
                label = str(item).strip()
                if re.fullmatch(r"\[\d+\]", label):
                    labels.append(label)
        elif isinstance(value, str):
            label = value.strip()
            if re.fullmatch(r"\[\d+\]", label):
                labels.append(label)
        deduped: list[str] = []
        seen: set[str] = set()
        for label in labels:
            if label in seen:
                continue
            seen.add(label)
            deduped.append(label)
        return deduped

    def _coerce_answer_text(self, value: object, answer_style: str) -> str:
        if isinstance(value, list):
            items = [" ".join(str(item).split()).strip() for item in value if str(item).strip()]
            if not items:
                return ""
            if answer_style in {"grouped_by_branch", "summary"}:
                return "\n".join(f"- {item}" for item in items)
            return " ".join(items)
        return " ".join(str(value).split()).strip()
