import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import httpx
import numpy as np
from sklearn.linear_model import LogisticRegression

from app.services.embedding_gateway import EmbeddingGateway


VALID_ANSWER_FORMS: tuple[str, ...] = (
    "direct_unit",
    "unit_set",
    "rule_summary",
    "global_summary",
)
_MODEL_CACHE: dict[tuple[str, int, int, str, str], "BranchCentralityModel"] = {}
_CACHE_LOCK = Lock()


@dataclass
class BranchSeedCase:
    case_id: str
    question: str
    answer_form: str
    source_notes: str
    expected_labels: list[str]


@dataclass
class BranchPrototype:
    branch_key: str
    canonical_label: str
    unit_type: str
    unit_number: str
    context_text: str
    signal_text: str


@dataclass
class BranchCentralityCandidate:
    branch_key: str
    canonical_label: str
    unit_type: str
    unit_number: str
    unit_topic: str
    path_text: str
    reference_variants_text: str
    sample_questions_text: str
    focus_terms_text: str
    branch_summary_context: str
    branch_keywords_text: str
    branch_summary: str
    branch_score: float
    boilerplate_score: float


@dataclass
class BranchCentralityModel:
    weights: np.ndarray
    bias: float
    mean: np.ndarray
    scale: np.ndarray
    training_accuracy: float
    embedding_source: str
    pair_count: int
    positive_count: int


@dataclass
class BranchCentralityPrediction:
    branch_key: str
    probability: float
    is_central: bool


@dataclass
class BranchCentralityResult:
    predictions: list[BranchCentralityPrediction]
    source: str
    decision_trace: list[str]


class BranchCentralityScorer:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.embeddings = EmbeddingGateway(settings)

    def score(
        self,
        *,
        question: str,
        strategy: str,
        evidence_intent: str,
        candidates: list[BranchCentralityCandidate],
    ) -> BranchCentralityResult | None:
        if not self.settings.branch_centrality_scorer_enabled:
            return None
        if not candidates:
            return None

        model = self._load_model()
        if model is None:
            return None

        pair_texts = [
            self._inference_pair_text(
                question=question,
                strategy=strategy,
                evidence_intent=evidence_intent,
                candidate=candidate,
            )
            for candidate in candidates
        ]
        try:
            vectors, embedding_source = self.embeddings.embed_texts(pair_texts)
        except Exception:
            return None
        if not vectors:
            return None
        features = np.asarray(vectors, dtype=np.float64)
        if (
            embedding_source != model.embedding_source
            or features.ndim != 2
            or features.shape[1] != model.mean.shape[0]
        ):
            return None

        features = self._standardize(features, model.mean, model.scale)
        logits = (features @ model.weights) + model.bias
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        threshold = float(self.settings.branch_centrality_positive_threshold)
        predictions = [
            BranchCentralityPrediction(
                branch_key=candidate.branch_key,
                probability=round(float(probability), 4),
                is_central=float(probability) >= threshold,
            )
            for candidate, probability in zip(candidates, probabilities, strict=False)
        ]
        predictions.sort(key=lambda item: item.probability, reverse=True)
        positives = sum(1 for item in predictions if item.is_central)
        return BranchCentralityResult(
            predictions=predictions,
            source="linear_pair_classifier",
            decision_trace=[
                "branch_select:provider=linear_pair_classifier",
                f"branch_select:embedding={embedding_source}",
                f"branch_select:strategy={strategy}",
                f"branch_select:candidates={len(candidates)}",
                f"branch_select:positives={positives}",
                f"branch_select:threshold={round(threshold, 2)}",
                f"branch_select:train_acc={round(model.training_accuracy, 2)}",
                f"branch_select:pairs={model.pair_count}",
            ],
        )

    def _load_model(self) -> BranchCentralityModel | None:
        fixture_path = self._resolve_fixture_path(self.settings.branch_centrality_fixture_path)
        if not fixture_path.exists():
            return None

        stat = fixture_path.stat()
        cache_key = (
            str(fixture_path.resolve()),
            stat.st_mtime_ns,
            stat.st_size,
            self.settings.embedding_profile_slug,
            self.embeddings.resolve_source(),
        )
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        with _CACHE_LOCK:
            cached = _MODEL_CACHE.get(cache_key)
            if cached is not None:
                return cached

            seed_cases = self._load_seed_cases(fixture_path)
            if len(seed_cases) < 6:
                return None
            pair_texts, labels = self._training_pairs_from_mined_candidates(seed_cases)
            if len(pair_texts) < 20 or len(set(labels)) < 2:
                prototypes = self._build_prototypes(seed_cases)
                if len(prototypes) < 6:
                    return None
                pair_texts, labels = self._training_pairs(seed_cases, prototypes)
            if len(pair_texts) < 20 or len(set(labels)) < 2:
                return None

            try:
                vectors, embedding_source = self.embeddings.embed_texts(pair_texts)
            except Exception:
                return None
            if not vectors:
                return None

            features = np.asarray(vectors, dtype=np.float64)
            labels_array = np.asarray(labels, dtype=np.int64)
            model = self._fit_classifier(features=features, labels=labels_array, embedding_source=embedding_source)
            if model is None:
                return None
            _MODEL_CACHE.clear()
            _MODEL_CACHE[cache_key] = model
            return model

    def _load_seed_cases(self, fixture_path: Path) -> list[BranchSeedCase]:
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        raw_cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
        cases: list[BranchSeedCase] = []
        for item in raw_cases:
            if not isinstance(item, dict):
                continue
            answer_form = str(
                item.get("answer_form", "") or item.get("suggested_answer_form", "")
            ).strip().lower()
            question = str(item.get("question", "")).strip()
            if answer_form not in VALID_ANSWER_FORMS or not question:
                continue
            labels: list[str] = []
            expected_units = item.get("expected_units") or item.get("suggested_expected_units") or []
            if isinstance(expected_units, list):
                for value in expected_units:
                    text = str(value).strip()
                    if text:
                        labels.append(text)
            primary_unit = str(
                item.get("primary_unit", "") or item.get("suggested_primary_unit", "")
            ).strip()
            if primary_unit:
                labels.append(primary_unit)
            source_notes = str(item.get("source_notes", "") or item.get("notes", "")).strip()
            label_from_notes = self._expected_label_from_notes(source_notes)
            if label_from_notes:
                labels.append(label_from_notes)
            normalized = self._dedupe_texts(labels)
            if not normalized:
                continue
            cases.append(
                BranchSeedCase(
                    case_id=str(item.get("case_id", "")).strip() or question[:80],
                    question=question,
                    answer_form=answer_form,
                    source_notes=source_notes,
                    expected_labels=normalized,
                )
            )
        return cases

    def _build_prototypes(self, seed_cases: list[BranchSeedCase]) -> dict[str, BranchPrototype]:
        contexts: dict[str, list[str]] = {}
        signals: dict[str, list[str]] = {}
        labels: dict[str, str] = {}
        parsed: dict[str, tuple[str, str]] = {}

        for case in seed_cases:
            for label in case.expected_labels:
                branch_key = self._branch_key(label)
                labels.setdefault(branch_key, label)
                parsed.setdefault(branch_key, self._parse_unit(label))
                if case.source_notes:
                    contexts.setdefault(branch_key, []).append(case.source_notes)
                signals.setdefault(branch_key, []).append(case.question)

        prototypes: dict[str, BranchPrototype] = {}
        for branch_key, label in labels.items():
            unit_type, unit_number = parsed.get(branch_key, ("", ""))
            context_text = " | ".join(self._dedupe_texts(contexts.get(branch_key, []))[:3])
            signal_text = " | ".join(self._dedupe_texts(signals.get(branch_key, []))[:4])
            prototypes[branch_key] = BranchPrototype(
                branch_key=branch_key,
                canonical_label=label,
                unit_type=unit_type,
                unit_number=unit_number,
                context_text=context_text,
                signal_text=signal_text,
            )
        return prototypes

    def _training_pairs_from_mined_candidates(
        self,
        seed_cases: list[BranchSeedCase],
    ) -> tuple[list[str], list[int]]:
        pair_texts: list[str] = []
        labels: list[int] = []
        for case in seed_cases:
            candidates = self._mine_branch_candidates(case.question, case.expected_labels)
            if not candidates:
                continue
            positive_keys = {self._branch_key(value) for value in case.expected_labels}
            case_pairs: list[str] = []
            case_labels: list[int] = []
            for candidate in candidates:
                label = 1 if self._candidate_match_key(candidate) in positive_keys else 0
                case_pairs.append(
                    self._inference_pair_text(
                        question=case.question,
                        strategy=self._strategy_for_answer_form(case.answer_form),
                        evidence_intent="substantive",
                        candidate=candidate,
                    )
                )
                case_labels.append(label)
            if 1 not in case_labels or 0 not in case_labels:
                continue
            pair_texts.extend(case_pairs)
            labels.extend(case_labels)
        return pair_texts, labels

    def _training_pairs(
        self,
        seed_cases: list[BranchSeedCase],
        prototypes: dict[str, BranchPrototype],
    ) -> tuple[list[str], list[int]]:
        pair_texts: list[str] = []
        labels: list[int] = []
        prototype_items = list(prototypes.items())
        for case in seed_cases:
            positive_keys = {self._branch_key(value) for value in case.expected_labels}
            for branch_key, prototype in prototype_items:
                pair_texts.append(
                    self._training_pair_text(
                        question=case.question,
                        answer_form=case.answer_form,
                        prototype=prototype,
                    )
                )
                labels.append(1 if branch_key in positive_keys else 0)
        return pair_texts, labels

    def _training_pair_text(
        self,
        *,
        question: str,
        answer_form: str,
        prototype: BranchPrototype,
    ) -> str:
        lines = [
            f"question: {question.strip()}",
            f"answer_form: {answer_form}",
            f"canonical_label: {prototype.canonical_label}",
        ]
        if prototype.unit_type:
            lines.append(f"unit_type: {prototype.unit_type}")
        if prototype.unit_number:
            lines.append(f"unit_number: {prototype.unit_number}")
        if prototype.context_text:
            lines.append(f"context: {prototype.context_text}")
        if prototype.signal_text:
            lines.append(f"signals: {prototype.signal_text}")
        return "\n".join(lines)

    def _inference_pair_text(
        self,
        *,
        question: str,
        strategy: str,
        evidence_intent: str,
        candidate: BranchCentralityCandidate,
    ) -> str:
        lines = [
            f"question: {question.strip()}",
            f"strategy: {strategy}",
            f"evidence_intent: {evidence_intent}",
            f"canonical_label: {candidate.canonical_label}",
        ]
        if candidate.unit_type:
            lines.append(f"unit_type: {candidate.unit_type}")
        if candidate.unit_number:
            lines.append(f"unit_number: {candidate.unit_number}")
        if candidate.unit_topic:
            lines.append(f"unit_topic: {candidate.unit_topic}")
        if candidate.path_text:
            lines.append(f"path: {candidate.path_text}")
        if candidate.reference_variants_text:
            lines.append(f"reference_variants: {candidate.reference_variants_text}")
        if candidate.focus_terms_text:
            lines.append(f"focus_terms: {candidate.focus_terms_text}")
        if candidate.branch_keywords_text:
            lines.append(f"branch_keywords: {candidate.branch_keywords_text}")
        if candidate.branch_summary_context:
            lines.append(f"summary_context: {candidate.branch_summary_context[:600]}")
        if candidate.branch_summary:
            lines.append(f"summary: {candidate.branch_summary[:900]}")
        lines.append(f"branch_score: {round(candidate.branch_score, 4)}")
        lines.append(f"boilerplate_score: {round(candidate.boilerplate_score, 4)}")
        return "\n".join(lines)

    def _fit_classifier(
        self,
        *,
        features: np.ndarray,
        labels: np.ndarray,
        embedding_source: str,
    ) -> BranchCentralityModel | None:
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        normalized_features = self._standardize(features, mean, scale)

        classifier = LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
        )
        classifier.fit(normalized_features, labels)

        weights = np.asarray(classifier.coef_[0], dtype=np.float64)
        bias = float(classifier.intercept_[0])
        training_accuracy = float(classifier.score(normalized_features, labels))

        return BranchCentralityModel(
            weights=weights,
            bias=bias,
            mean=mean,
            scale=scale,
            training_accuracy=training_accuracy,
            embedding_source=embedding_source,
            pair_count=int(len(labels)),
            positive_count=int(labels.sum()),
        )

    def _mine_branch_candidates(
        self,
        question: str,
        expected_labels: list[str],
    ) -> list[BranchCentralityCandidate]:
        score_map: dict[str, float] = {}
        payload_map: dict[str, dict[str, object]] = {}

        question_dense = self._search_qdrant_branches(question=question, limit=8)
        question_sparse = self._search_opensearch_branches(question=question, limit=12)
        self._accumulate_hits(score_map, payload_map, question_dense, modality="dense")
        self._accumulate_hits(score_map, payload_map, question_sparse, modality="sparse")

        for label in expected_labels:
            label_hits = self._search_opensearch_branches(question=label, limit=4)
            self._accumulate_hits(score_map, payload_map, label_hits, modality="sparse")

        ordered = sorted(score_map.items(), key=lambda item: item[1], reverse=True)[:18]
        candidates: list[BranchCentralityCandidate] = []
        for branch_key, score in ordered:
            payload = payload_map.get(branch_key) or {}
            candidates.append(
                BranchCentralityCandidate(
                    branch_key=branch_key,
                    canonical_label=str(payload.get("canonical_label") or "").strip(),
                    unit_type=str(payload.get("unit_type") or "").strip(),
                    unit_number=str(payload.get("unit_number") or "").strip(),
                    unit_topic=str(payload.get("unit_topic") or "").strip(),
                    path_text=str(payload.get("path_text") or "").strip(),
                    reference_variants_text=str(payload.get("reference_variants_text") or "").strip(),
                    sample_questions_text=str(payload.get("sample_questions_text") or "").strip(),
                    focus_terms_text=str(payload.get("focus_terms_text") or "").strip(),
                    branch_summary_context=str(payload.get("branch_summary_context") or "").strip(),
                    branch_keywords_text=str(payload.get("branch_keywords_text") or "").strip(),
                    branch_summary=str(payload.get("raw_text") or payload.get("contextualized_text") or "").strip(),
                    branch_score=round(float(score), 4),
                    boilerplate_score=round(float(payload.get("boilerplate_score") or 0.0), 4),
                )
            )
        return candidates

    def _accumulate_hits(
        self,
        score_map: dict[str, float],
        payload_map: dict[str, dict[str, object]],
        hits: list[tuple[str, dict[str, object]]],
        *,
        modality: str,
    ) -> None:
        rrf_k = 30.0
        for rank, (branch_key, payload) in enumerate(hits, start=1):
            score_map[branch_key] = score_map.get(branch_key, 0.0) + (1.0 / (rrf_k + rank))
            payload_map.setdefault(branch_key, payload)

    def _search_opensearch_branches(
        self,
        *,
        question: str,
        limit: int,
    ) -> list[tuple[str, dict[str, object]]]:
        query_body = {
            "size": limit,
            "query": {
                "bool": {
                    "must": [
                        {
                                "multi_match": {
                                    "query": question,
                                    "fields": [
                                        "unit_key^10",
                                        "reference_variants_text^9",
                                        "canonical_label^9",
                                        "unit_topic^10",
                                        "branch_summary_context^10",
                                        "path_text^7",
                                        "branch_keywords_text^7",
                                        "focus_terms_text^6",
                                        "document_title^5",
                                        "retrieval_context^5",
                                        "contextualized_text^4",
                                        "raw_text^3",
                                ],
                                "type": "best_fields",
                            }
                        }
                    ],
                    "filter": [{"term": {"version_status": "active"}}],
                }
            },
        }
        try:
            payload = self._opensearch_search(
                index_candidates=[
                    self.settings.resolved_opensearch_branch_index_name,
                    self.settings.opensearch_branch_index_name,
                ],
                query_body=query_body,
            )
        except Exception:
            return []
        return self._parse_opensearch_branch_hits(payload)

    def _search_qdrant_branches(
        self,
        *,
        question: str,
        limit: int,
    ) -> list[tuple[str, dict[str, object]]]:
        try:
            vectors, _ = self.embeddings.embed_texts([question])
            vector = vectors[0]
            payload = self._qdrant_search(
                collections=[
                    self.settings.resolved_qdrant_branch_collection_name,
                    self.settings.qdrant_branch_collection_name,
                ],
                body={
                    "vector": vector,
                    "limit": limit,
                    "with_payload": True,
                    "filter": {"must": [{"key": "version_status", "match": {"value": "active"}}]},
                },
            )
        except Exception:
            return []
        return self._parse_qdrant_branch_hits(payload)

    def _parse_opensearch_branch_hits(self, payload: dict[str, object]) -> list[tuple[str, dict[str, object]]]:
        hits: list[tuple[str, dict[str, object]]] = []
        for item in payload.get("hits", {}).get("hits", []):
            source = item.get("_source", {})
            branch_key = str(source.get("branch_key") or source.get("branch_id") or "").strip()
            if branch_key:
                hits.append((branch_key, source))
        return hits

    def _parse_qdrant_branch_hits(self, payload: dict[str, object]) -> list[tuple[str, dict[str, object]]]:
        hits: list[tuple[str, dict[str, object]]] = []
        for item in payload.get("result", []):
            source = item.get("payload", {})
            branch_key = str(source.get("branch_key") or source.get("branch_id") or "").strip()
            if branch_key:
                hits.append((branch_key, source))
        return hits

    def _opensearch_search(
        self,
        *,
        index_candidates: list[str],
        query_body: dict[str, object],
    ) -> dict[str, object]:
        payload = None
        with httpx.Client(timeout=45.0) as client:
            for index_name in index_candidates:
                response = client.post(
                    f"{self.settings.opensearch_url.rstrip('/')}/{index_name}/_search",
                    json=query_body,
                )
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                payload = response.json()
                break
        if payload is None:
            raise RuntimeError("OpenSearch branch index not found.")
        return payload

    def _qdrant_search(
        self,
        *,
        collections: list[str],
        body: dict[str, object],
    ) -> dict[str, object]:
        payload = None
        with httpx.Client(timeout=45.0) as client:
            for collection_name in collections:
                response = client.post(
                    f"{self.settings.qdrant_url.rstrip('/')}/collections/{collection_name}/points/search",
                    json=body,
                )
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                payload = response.json()
                break
        if payload is None:
            raise RuntimeError("Qdrant branch collection not found.")
        return payload

    def _candidate_match_key(self, candidate: BranchCentralityCandidate) -> str:
        if candidate.unit_type and candidate.unit_number:
            return f"{candidate.unit_type.lower()}:{candidate.unit_number.upper()}"
        return self._branch_key(candidate.canonical_label or candidate.path_text)

    def _strategy_for_answer_form(self, answer_form: str) -> str:
        if answer_form == "global_summary":
            return "global"
        if answer_form == "unit_set":
            return "multi_branch"
        return "focal"

    def _expected_label_from_notes(self, text: str) -> str:
        if not text:
            return ""
        marker = "expected_label:"
        lower = text.lower()
        if marker not in lower:
            return ""
        index = lower.index(marker)
        return text[index + len(marker) :].strip()

    def _branch_key(self, text: str) -> str:
        normalized = self._normalize(text)
        unit_type, unit_number = self._parse_unit(text)
        if unit_type and unit_number:
            return f"{unit_type}:{unit_number}"
        return normalized

    def _parse_unit(self, text: str) -> tuple[str, str]:
        normalized = self._normalize(text)
        patterns = (
            ("article", r"\b(?:art\.?|arto\.?|articulo)\s*(\d+)\b"),
            ("section", r"\bseccion\s*(\d+(?:\.\d+)?)\b"),
            ("recommendation", r"\brecomendacion\s*(\d+(?:\.\d+)?)\b"),
            ("chapter", r"\bcapitulo\s*([ivxlcdm]+|\d+)\b"),
            ("title", r"\btitulo\s*([ivxlcdm]+|\d+)\b"),
            ("annex", r"\banexo\s*([a-z]|\d+)\b"),
            ("paragraph", r"\b(\d+\.\d+)\b"),
        )
        for unit_type, pattern in patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                return unit_type, match.group(1).upper()
        return "", ""

    def _normalize(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text or "")
        return "".join(char for char in normalized if not unicodedata.combining(char)).lower().strip()

    def _dedupe_texts(self, values: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = self._normalize(value)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(value.strip())
        return deduped

    def _standardize(self, features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
        return (features - mean) / scale

    def _resolve_fixture_path(self, fixture_path: str) -> Path:
        path = Path(fixture_path)
        if path.is_absolute():
            return path
        return Path.cwd() / path
