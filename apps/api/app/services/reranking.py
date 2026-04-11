from dataclasses import dataclass

import httpx


@dataclass
class RerankDocument:
    key: str
    text: str


@dataclass
class RerankOutcome:
    ordered_keys: list[str]
    scores: dict[str, float]
    source: str
    status: str
    reason: str | None = None


class RerankGateway:
    def __init__(self, settings) -> None:
        self.settings = settings

    def rerank(
        self,
        *,
        question: str,
        documents: list[RerankDocument],
    ) -> RerankOutcome:
        if not documents:
            return RerankOutcome(
                ordered_keys=[],
                scores={},
                source="heuristic",
                status="skipped",
                reason="no_candidates",
            )

        mode = self.settings.rerank_mode
        if mode == "off":
            return RerankOutcome(
                ordered_keys=[document.key for document in documents],
                scores={},
                source="heuristic",
                status="skipped",
                reason="rerank_mode_off",
            )

        if mode == "local":
            return self._rerank_local(question=question, documents=documents)

        if mode == "cohere":
            return self._rerank_cohere(question=question, documents=documents)

        return RerankOutcome(
            ordered_keys=[document.key for document in documents],
            scores={},
            source="heuristic",
            status="fallback",
            reason=f"unsupported_mode:{mode}",
        )

    def _rerank_local(
        self,
        *,
        question: str,
        documents: list[RerankDocument],
    ) -> RerankOutcome:
        base_url = (self.settings.rerank_local_base_url or "").rstrip("/")
        model = self.settings.rerank_local_model
        timeout = float(self.settings.rerank_local_timeout or 20.0)
        if not base_url or not model:
            return RerankOutcome(
                ordered_keys=[document.key for document in documents],
                scores={},
                source="heuristic",
                status="fallback",
                reason="local_unconfigured",
            )

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.settings.rerank_local_api_key:
            headers["Authorization"] = f"Bearer {self.settings.rerank_local_api_key}"

        payload = {
            "model": model,
            "query": question,
            "documents": [document.text for document in documents],
            "top_n": min(len(documents), int(self.settings.rerank_top_n)),
        }

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(f"{base_url}/rerank", headers=headers, json=payload)
                response.raise_for_status()
            return self._parse_provider_response(
                source="local",
                payload=response.json(),
                documents=documents,
            )
        except Exception as exc:
            return RerankOutcome(
                ordered_keys=[document.key for document in documents],
                scores={},
                source="heuristic",
                status="fallback",
                reason=self._truncate_reason(f"local_error:{exc}"),
            )

    def _rerank_cohere(
        self,
        *,
        question: str,
        documents: list[RerankDocument],
    ) -> RerankOutcome:
        base_url = (self.settings.rerank_cohere_base_url or "https://api.cohere.com/v2").rstrip("/")
        model = self.settings.rerank_cohere_model
        timeout = float(self.settings.rerank_cohere_timeout or 20.0)
        api_key = self.settings.rerank_cohere_api_key
        if not api_key or not model:
            return RerankOutcome(
                ordered_keys=[document.key for document in documents],
                scores={},
                source="heuristic",
                status="fallback",
                reason="cohere_unconfigured",
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "query": question,
            "documents": [document.text for document in documents],
            "top_n": min(len(documents), int(self.settings.rerank_top_n)),
        }

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(f"{base_url}/rerank", headers=headers, json=payload)
                response.raise_for_status()
            return self._parse_provider_response(
                source="cohere",
                payload=response.json(),
                documents=documents,
            )
        except Exception as exc:
            return RerankOutcome(
                ordered_keys=[document.key for document in documents],
                scores={},
                source="heuristic",
                status="fallback",
                reason=self._truncate_reason(f"cohere_error:{exc}"),
            )

    def _parse_provider_response(
        self,
        *,
        source: str,
        payload: dict[str, object],
        documents: list[RerankDocument],
    ) -> RerankOutcome:
        raw_results = payload.get("results") or payload.get("data") or []
        results = raw_results if isinstance(raw_results, list) else []
        ordered_keys: list[str] = []
        scores: dict[str, float] = {}
        for item in results:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            if not isinstance(index, int):
                continue
            if index < 0 or index >= len(documents):
                continue
            key = documents[index].key
            ordered_keys.append(key)
            raw_score = item.get("relevance_score", item.get("score", 0.0))
            try:
                scores[key] = float(raw_score)
            except (TypeError, ValueError):
                scores[key] = 0.0

        if not ordered_keys:
            return RerankOutcome(
                ordered_keys=[document.key for document in documents],
                scores={},
                source="heuristic",
                status="fallback",
                reason=f"{source}_empty_results",
            )

        seen = set(ordered_keys)
        ordered_keys.extend(document.key for document in documents if document.key not in seen)
        return RerankOutcome(
            ordered_keys=ordered_keys,
            scores=scores,
            source=source,
            status="ok",
        )

    def _truncate_reason(self, reason: str, *, limit: int = 120) -> str:
        compact = " ".join((reason or "").split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."
