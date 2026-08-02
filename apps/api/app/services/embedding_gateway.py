import hashlib
import logging
import math
import re
from dataclasses import dataclass
from typing import Literal

import httpx


logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


@dataclass
class RemoteEmbeddingConfig:
    source: Literal["local_openai", "cloud_openai"]
    base_url: str
    model: str
    api_key: str | None
    timeout: float


class EmbeddingGateway:
    def __init__(self, settings) -> None:
        self.settings = settings

    def resolve_source(self) -> Literal["hash", "local_openai", "cloud_openai"]:
        mode = self.settings.embedding_mode
        if mode == "hash":
            return "hash"
        if mode == "local_openai":
            return "local_openai"
        if mode == "cloud_openai":
            return "cloud_openai"
        local_config = self._local_config()
        if local_config is not None:
            return "local_openai"
        cloud_config = self._cloud_config()
        if cloud_config is not None:
            return "cloud_openai"
        return "hash"

    def embed_texts(
        self,
        texts: list[str],
    ) -> tuple[list[list[float]], Literal["hash", "local_openai", "cloud_openai"]]:
        if not texts:
            return [], self.resolve_source()

        mode = self.settings.embedding_mode
        if mode == "hash":
            return self._embed_hash(texts), "hash"
        if mode == "local_openai":
            config = self._local_config(strict=True)
            return self._embed_remote(texts, config), "local_openai"
        if mode == "cloud_openai":
            config = self._cloud_config(strict=True)
            return self._embed_remote(texts, config), "cloud_openai"

        local_config = self._local_config()
        if local_config is not None:
            try:
                return self._embed_remote(texts, local_config), "local_openai"
            except Exception as exc:
                logger.warning(
                    "embedding_provider_failed source=local_openai model=%s error=%s:%s",
                    local_config.model,
                    type(exc).__name__,
                    exc,
                )

        cloud_config = self._cloud_config()
        if cloud_config is not None:
            try:
                return self._embed_remote(texts, cloud_config), "cloud_openai"
            except Exception as exc:
                logger.warning(
                    "embedding_provider_failed source=cloud_openai model=%s error=%s:%s",
                    cloud_config.model,
                    type(exc).__name__,
                    exc,
                )

        # Fallback a vectores hash: no son comparables con los del indice, asi
        # que la recuperacion queda inservible. Debe verse en los logs.
        logger.error(
            "embedding_degraded_to_hash texts=%d dim=%s "
            "(los vectores hash no son comparables con el indice existente)",
            len(texts),
            self.settings.embedding_vector_size,
        )
        return self._embed_hash(texts), "hash"

    def _embed_remote(self, texts: list[str], config: RemoteEmbeddingConfig) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"

        vectors: list[list[float]] = []
        batch_size = max(1, int(self.settings.embedding_batch_size))
        with httpx.Client(timeout=config.timeout) as client:
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                response = client.post(
                    f"{config.base_url.rstrip('/')}/embeddings",
                    headers=headers,
                    json={"model": config.model, "input": batch},
                )
                response.raise_for_status()
                payload = response.json()
                data = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
                if len(data) != len(batch):
                    raise RuntimeError("Embedding provider returned a mismatched batch size.")
                vectors.extend([item["embedding"] for item in data])
        return vectors

    def _embed_hash(self, texts: list[str]) -> list[list[float]]:
        dim = max(8, int(self.settings.embedding_vector_size))
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * dim
            tokens = TOKEN_PATTERN.findall((text or "").lower())
            if not tokens:
                vector[0] = 1e-6
                vectors.append(vector)
                continue
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                weight = 1.0 + (digest[5] / 255.0)
                vector[index] += sign * weight
            norm = math.sqrt(sum(value * value for value in vector))
            if norm > 0:
                vector = [value / norm for value in vector]
            vectors.append(vector)
        return vectors

    def _cloud_config(self, strict: bool = False) -> RemoteEmbeddingConfig | None:
        base_url = (
            self.settings.embedding_cloud_base_url
            or self.settings.classifier_cloud_base_url
            or self.settings.classifier_llm_base_url
        )
        api_key = (
            self.settings.embedding_cloud_api_key
            or self.settings.classifier_cloud_api_key
            or self.settings.classifier_llm_api_key
        )
        model = self.settings.embedding_cloud_model
        timeout = self.settings.embedding_cloud_timeout or self.settings.classifier_llm_timeout
        if base_url and model:
            return RemoteEmbeddingConfig(
                source="cloud_openai",
                base_url=base_url,
                model=model,
                api_key=api_key,
                timeout=float(timeout or 20.0),
            )
        if strict:
            raise RuntimeError("Cloud embedding provider is not configured.")
        return None

    def _local_config(self, strict: bool = False) -> RemoteEmbeddingConfig | None:
        base_url = self.settings.embedding_local_base_url or self.settings.classifier_local_base_url
        api_key = self.settings.embedding_local_api_key or self.settings.classifier_local_api_key
        model = self.settings.embedding_local_model
        timeout = self.settings.embedding_local_timeout or self.settings.classifier_local_timeout
        if base_url and model:
            return RemoteEmbeddingConfig(
                source="local_openai",
                base_url=base_url,
                model=model,
                api_key=api_key,
                timeout=float(timeout or 20.0),
            )
        if strict:
            raise RuntimeError("Local embedding provider is not configured.")
        return None
