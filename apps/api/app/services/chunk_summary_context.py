import hashlib
import json
import logging
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import httpx


logger = logging.getLogger(__name__)

_CACHE_LOCK = threading.Lock()


class SummaryCache:
    """Cache persistente de resumenes por chunk.

    Clave: hash del prompt exacto mas el modelo. Cualquier cambio en el texto del
    chunk, en los metadatos que entran al prompt o en el modelo produce una clave
    distinta, asi que no hace falta versionar el prompt a mano.

    Vive en su propio SQLite para no competir por el lock de la base principal
    durante la indexacion.
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _ensure_schema(self) -> None:
        with _CACHE_LOCK, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS summary_cache (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    @staticmethod
    def build_key(*, prompt: str, model: str) -> str:
        digest = hashlib.sha256()
        digest.update(model.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(prompt.encode("utf-8"))
        return digest.hexdigest()

    def get(self, cache_key: str) -> str | None:
        try:
            with _CACHE_LOCK, self._connect() as connection:
                row = connection.execute(
                    "SELECT summary FROM summary_cache WHERE cache_key = ?", (cache_key,)
                ).fetchone()
            return row[0] if row else None
        except Exception as exc:
            logger.warning("summary_cache_read_failed error=%s:%s", type(exc).__name__, exc)
            return None

    def put(self, cache_key: str, *, model: str, summary: str) -> None:
        try:
            with _CACHE_LOCK, self._connect() as connection:
                connection.execute(
                    "INSERT OR REPLACE INTO summary_cache (cache_key, model, summary) VALUES (?, ?, ?)",
                    (cache_key, model, summary),
                )
        except Exception as exc:
            logger.warning("summary_cache_write_failed error=%s:%s", type(exc).__name__, exc)

    def entries(self) -> int:
        try:
            with _CACHE_LOCK, self._connect() as connection:
                return int(connection.execute("SELECT COUNT(*) FROM summary_cache").fetchone()[0])
        except Exception:
            return 0


@dataclass
class ChunkSummaryProviderConfig:
    provider_name: str
    base_url: str
    model: str
    api_key: str | None
    timeout: float


class ChunkSummaryContextService:
    def __init__(self, settings) -> None:
        self.settings = settings

    def build_context(
        self,
        *,
        document_title: str,
        document_class: str,
        canonical_label: str,
        unit_type: str,
        unit_number: str,
        path_text: str,
        retrieval_context: str,
        raw_text: str,
        unit_topic: str,
    ) -> str:
        summary = self._summary_with_llm(
            document_title=document_title,
            document_class=document_class,
            canonical_label=canonical_label,
            unit_type=unit_type,
            unit_number=unit_number,
            path_text=path_text,
            retrieval_context=retrieval_context,
            raw_text=raw_text,
            unit_topic=unit_topic,
        )
        if not summary:
            summary = self._deterministic_summary(
                canonical_label=canonical_label,
                unit_type=unit_type,
                raw_text=raw_text,
                unit_topic=unit_topic,
            )

        header = self._header(document_title=document_title, path_text=path_text)
        parts = [part.strip() for part in (header, summary) if str(part).strip()]
        return "\n".join(parts).strip()

    def _summary_with_llm(
        self,
        *,
        document_title: str,
        document_class: str,
        canonical_label: str,
        unit_type: str,
        unit_number: str,
        path_text: str,
        retrieval_context: str,
        raw_text: str,
        unit_topic: str,
    ) -> str | None:
        mode = self.settings.chunk_summary_context_mode
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
            summary = self._generate_summary_with_provider(
                provider=provider,
                document_title=document_title,
                document_class=document_class,
                canonical_label=canonical_label,
                unit_type=unit_type,
                unit_number=unit_number,
                path_text=path_text,
                retrieval_context=retrieval_context,
                raw_text=raw_text,
                unit_topic=unit_topic,
            )
            if summary:
                return summary
        return None

    def _provider_config(self, provider_name: str) -> ChunkSummaryProviderConfig | None:
        if provider_name == "cloud":
            base_url = (
                self.settings.chunk_summary_context_cloud_base_url
                or self.settings.answer_synthesis_cloud_base_url
                or self.settings.query_planner_cloud_base_url
                or self.settings.classifier_cloud_base_url
                or self.settings.classifier_llm_base_url
            )
            api_key = (
                self.settings.chunk_summary_context_cloud_api_key
                or self.settings.answer_synthesis_cloud_api_key
                or self.settings.query_planner_cloud_api_key
                or self.settings.classifier_cloud_api_key
                or self.settings.classifier_llm_api_key
            )
            model = (
                self.settings.chunk_summary_context_cloud_model
                or self.settings.answer_synthesis_cloud_model
                or self.settings.query_planner_cloud_model
                or self.settings.classifier_cloud_model
                or self.settings.classifier_llm_model
            )
            timeout = (
                self.settings.chunk_summary_context_cloud_timeout
                or self.settings.answer_synthesis_cloud_timeout
                or self.settings.query_planner_cloud_timeout
                or self.settings.classifier_cloud_timeout
                or self.settings.classifier_llm_timeout
                or 20.0
            )
        else:
            base_url = (
                self.settings.chunk_summary_context_local_base_url
                or self.settings.answer_synthesis_local_base_url
                or self.settings.query_planner_local_base_url
                or self.settings.classifier_local_base_url
            )
            api_key = (
                self.settings.chunk_summary_context_local_api_key
                or self.settings.answer_synthesis_local_api_key
                or self.settings.query_planner_local_api_key
                or self.settings.classifier_local_api_key
            )
            model = (
                self.settings.chunk_summary_context_local_model
                or self.settings.answer_synthesis_local_model
                or self.settings.query_planner_local_model
                or self.settings.classifier_local_model
            )
            timeout = (
                self.settings.chunk_summary_context_local_timeout
                or self.settings.answer_synthesis_local_timeout
                or self.settings.query_planner_local_timeout
                or self.settings.classifier_local_timeout
                or 20.0
            )

        if not base_url or not model:
            return None
        return ChunkSummaryProviderConfig(
            provider_name=provider_name,
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=float(timeout),
        )

    def _generate_summary_with_provider(
        self,
        *,
        provider: ChunkSummaryProviderConfig,
        document_title: str,
        document_class: str,
        canonical_label: str,
        unit_type: str,
        unit_number: str,
        path_text: str,
        retrieval_context: str,
        raw_text: str,
        unit_topic: str,
    ) -> str | None:
        headers = {"Content-Type": "application/json"}
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"

        prompt = self._build_summary_prompt(
            document_title=document_title,
            document_class=document_class,
            canonical_label=canonical_label,
            unit_type=unit_type,
            unit_number=unit_number,
            path_text=path_text,
            retrieval_context=retrieval_context,
            raw_text=raw_text,
            unit_topic=unit_topic,
        )

        cache = self._cache()
        cache_key = None
        if cache is not None:
            cache_key = SummaryCache.build_key(prompt=prompt, model=provider.model)
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

        summary = self._call_provider(provider=provider, headers=headers, prompt=prompt)
        if summary and cache is not None and cache_key is not None:
            cache.put(cache_key, model=provider.model, summary=summary)
        return summary

    def _cache(self) -> SummaryCache | None:
        if not getattr(self.settings, "chunk_summary_cache_enabled", True):
            return None
        if getattr(self, "_cache_instance", None) is None:
            try:
                self._cache_instance = SummaryCache(self.settings.chunk_summary_cache_path)
            except Exception as exc:
                logger.warning("summary_cache_disabled error=%s:%s", type(exc).__name__, exc)
                self._cache_instance = None
        return self._cache_instance

    def _build_summary_prompt(
        self,
        *,
        document_title: str,
        document_class: str,
        canonical_label: str,
        unit_type: str,
        unit_number: str,
        path_text: str,
        retrieval_context: str,
        raw_text: str,
        unit_topic: str,
    ) -> str:
        max_chars = max(600, int(self.settings.chunk_summary_context_max_input_chars))
        return "\n".join(
            [
                "Genera una micro-contextualizacion factual para retrieval.",
                "Devuelve JSON estricto: {\"summary\": \"...\"}.",
                "Reglas:",
                "- 1 sola frase.",
                "- 12 a 35 palabras.",
                "- Debe decir que regula, cubre o contiene este chunk.",
                "- No hagas preguntas.",
                "- No cites etiquetas como [1].",
                "- No inventes informacion fuera del texto.",
                "",
                f"document_title: {document_title}",
                f"document_class: {document_class}",
                f"path_text: {path_text}",
                f"retrieval_context: {retrieval_context}",
                f"canonical_label: {canonical_label}",
                f"unit_type: {unit_type}",
                f"unit_number: {unit_number}",
                f"unit_topic: {unit_topic}",
                f"raw_text: {raw_text[:max_chars]}",
            ]
        )

    def _call_provider(
        self,
        *,
        provider: ChunkSummaryProviderConfig,
        headers: dict[str, str],
        prompt: str,
    ) -> str | None:
        try:
            request_payload = {
                "model": provider.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You generate short factual chunk-context summaries for retrieval. "
                            "Return strict JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 120,
            }
            if provider.provider_name == "local":
                # Ollama thinking models may otherwise emit only reasoning traces.
                request_payload["reasoning_effort"] = "none"
                request_payload["reasoning"] = {"effort": "none"}
                request_payload["response_format"] = {"type": "json_object"}

            with httpx.Client(timeout=provider.timeout) as client:
                response = client.post(
                    f"{provider.base_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=request_payload,
                )
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return None

        try:
            content = payload["choices"][0]["message"]["content"]
            text = self._content_to_text(content)
            summary = self._parse_summary_payload(text)
            return self._normalize_summary(summary)
        except Exception:
            return None

    def _content_to_text(self, content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
                elif item:
                    parts.append(str(item))
            return "\n".join(parts).strip()
        return str(content or "").strip()

    def _parse_summary_payload(self, content: str) -> str:
        text = self._strip_markdown_fences(content)
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                summary = str(
                    data.get("summary")
                    or data.get("chunk_summary_context")
                    or data.get("context")
                    or ""
                ).strip()
                if summary:
                    return summary
            except Exception:
                pass

        line_match = re.search(
            r'(?im)^(?:summary|chunk_summary_context|context)\s*[:=]\s*["\']?(.*?)["\']?\s*$',
            text,
        )
        if line_match:
            return line_match.group(1).strip()

        return text.strip()

    def _strip_markdown_fences(self, content: str) -> str:
        text = (content or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return text.strip()

    def _normalize_summary(self, summary: str) -> str | None:
        cleaned = " ".join((summary or "").split()).strip(" -:;,.")
        if not cleaned or "?" in cleaned:
            return None
        words = cleaned.split()
        if len(words) < 8:
            return None
        if len(words) > 40:
            cleaned = " ".join(words[:40]).rstrip(" -:;,.")
        if cleaned[-1] not in ".:":
            cleaned = f"{cleaned}."
        return cleaned

    def _deterministic_summary(
        self,
        *,
        canonical_label: str,
        unit_type: str,
        raw_text: str,
        unit_topic: str,
    ) -> str:
        if unit_topic:
            topic = " ".join(unit_topic.split()).strip(" -:;,.")
            if topic:
                sentence = f"Este {self._unit_name(unit_type)} regula o desarrolla {topic}."
                normalized = self._normalize_summary(sentence)
                if normalized:
                    return normalized

        sentence = self._first_sentence(raw_text)
        if sentence:
            normalized = self._normalize_summary(sentence)
            if normalized:
                return normalized

        label = canonical_label or "esta unidad"
        fallback = f"Este {self._unit_name(unit_type)} contiene el texto principal de {label}."
        return self._normalize_summary(fallback) or fallback

    def _header(self, *, document_title: str, path_text: str) -> str:
        short_title = re.sub(r"[-_]+", " ", Path(document_title or "").stem).strip() or "documento"
        if path_text:
            if short_title.lower() in path_text.lower():
                return path_text.strip()
            return f"{short_title} > {path_text}".strip()
        return short_title

    def _first_sentence(self, raw_text: str) -> str:
        text = " ".join((raw_text or "").split()).strip()
        if not text:
            return ""
        sentence = re.split(r"(?<=[\.\!\?])\s+", text, maxsplit=1)[0].strip()
        sentence = re.sub(
            r"^(?:art\.?|arto\.?|articulo|artículo|secci[oó]n|recomendaci[oó]n)\s*\d+(?:\.\d+)?\s*[:\.-]?\s*",
            "",
            sentence,
            flags=re.IGNORECASE,
        )
        words = sentence.split()
        if len(words) > 32:
            sentence = " ".join(words[:32]).rstrip(" -:;,.")
        return sentence

    def _unit_name(self, unit_type: str) -> str:
        return {
            "article": "articulo",
            "section": "seccion",
            "subsection": "apartado",
            "recommendation": "recomendacion",
            "annex": "anexo",
            "chapter": "capitulo",
            "title": "titulo",
            "book": "libro",
            "paragraph": "parrafo",
        }.get(unit_type, "apartado")
