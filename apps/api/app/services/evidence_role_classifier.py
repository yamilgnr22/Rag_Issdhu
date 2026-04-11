from collections import Counter
import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
from sklearn.linear_model import LogisticRegression

from app.services.embedding_gateway import EmbeddingGateway


VALID_EVIDENCE_ROLES: tuple[str, ...] = ("substantive", "reference", "editorial")
_MODEL_CACHE: dict[tuple[str, int, int, str], "EvidenceRoleLinearModel"] = {}
_CACHE_LOCK = Lock()


@dataclass
class EvidenceRoleTrainingExample:
    case_id: str
    role: str
    text: str


@dataclass
class EvidenceRoleLinearModel:
    classes: list[str]
    weights: np.ndarray
    bias: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    training_accuracy: float
    embedding_source: str
    examples: list[EvidenceRoleTrainingExample]


@dataclass
class EvidenceRoleClassification:
    role: str
    confidence: float
    probabilities: dict[str, float]
    source: str


class EvidenceRoleClassifier:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.embeddings = EmbeddingGateway(settings)

    def annotate_documents(self, documents: list[dict[str, object]]) -> None:
        if not self.settings.evidence_role_classifier_enabled or not documents:
            return

        model = self._load_model()
        if model is None:
            return

        texts = [self._serialize_document(document) for document in documents]
        try:
            vectors, _ = self.embeddings.embed_texts(texts)
        except Exception:
            return
        if not vectors:
            return

        features = self._standardize(np.asarray(vectors, dtype=np.float64), model.mean, model.scale)
        probabilities = self._predict_probabilities(features, model.weights, model.bias)

        for document, row in zip(documents, probabilities, strict=True):
            best_index = int(np.argmax(row))
            role = model.classes[best_index]
            confidence = round(float(row[best_index]), 2)
            role_probs = {
                label: round(float(row[index]), 4)
                for index, label in enumerate(model.classes)
            }
            document["evidence_role"] = role
            document["evidence_role_confidence"] = confidence
            for label in VALID_EVIDENCE_ROLES:
                document[f"evidence_role_{label}"] = role_probs.get(label, 0.0)

    def default_role(self) -> str:
        fixture_path = self._resolve_fixture_path(self.settings.evidence_role_classifier_fixture_path)
        if not fixture_path.exists():
            return "substantive"
        examples = self._load_examples(fixture_path)
        if not examples:
            return "substantive"
        counts = Counter(example.role for example in examples)
        return counts.most_common(1)[0][0]

    def _load_model(self) -> EvidenceRoleLinearModel | None:
        fixture_path = self._resolve_fixture_path(self.settings.evidence_role_classifier_fixture_path)
        if not fixture_path.exists():
            return None

        stat = fixture_path.stat()
        cache_key = (
            str(fixture_path.resolve()),
            stat.st_mtime_ns,
            stat.st_size,
            self.settings.embedding_profile_slug,
        )
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        with _CACHE_LOCK:
            cached = _MODEL_CACHE.get(cache_key)
            if cached is not None:
                return cached

            examples = self._load_examples(fixture_path)
            if len(examples) < 6:
                return None

            try:
                vectors, embedding_source = self.embeddings.embed_texts(
                    [example.text for example in examples]
                )
            except Exception:
                return None
            if not vectors:
                return None

            classes = [
                label
                for label in VALID_EVIDENCE_ROLES
                if any(example.role == label for example in examples)
            ]
            if len(classes) < 2:
                return None

            class_to_index = {label: index for index, label in enumerate(classes)}
            features = np.asarray(vectors, dtype=np.float64)
            labels = np.asarray([class_to_index[example.role] for example in examples], dtype=np.int64)
            model = self._fit_classifier(
                features=features,
                labels=labels,
                classes=classes,
                examples=examples,
                embedding_source=embedding_source,
            )
            _MODEL_CACHE.clear()
            _MODEL_CACHE[cache_key] = model
            return model

    def _load_examples(self, fixture_path: Path) -> list[EvidenceRoleTrainingExample]:
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        raw_cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
        examples: list[EvidenceRoleTrainingExample] = []
        for item in raw_cases:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip().lower()
            case_id = str(item.get("case_id", "")).strip() or role
            text = self._serialize_training_example(item)
            if role not in VALID_EVIDENCE_ROLES or not text:
                continue
            examples.append(
                EvidenceRoleTrainingExample(
                    case_id=case_id,
                    role=role,
                    text=text,
                )
            )
        return examples

    def _serialize_training_example(self, item: dict[str, object]) -> str:
        text = str(item.get("text") or "").strip()
        path_text = str(item.get("path_text") or "").strip()
        canonical_label = str(item.get("canonical_label") or "").strip()
        unit_topic = str(item.get("unit_topic") or "").strip()
        chunk_kind = str(item.get("chunk_kind") or "").strip()
        return "\n".join(
            line
            for line in (
                f"path: {path_text}" if path_text else "",
                f"canonical_label: {canonical_label}" if canonical_label else "",
                f"unit_topic: {unit_topic}" if unit_topic else "",
                f"chunk_kind: {chunk_kind}" if chunk_kind else "",
                f"text: {text}" if text else "",
            )
            if line
        )

    def _serialize_document(self, document: dict[str, object]) -> str:
        raw_text = str(document.get("raw_text") or "").strip()
        if len(raw_text) > 1200:
            raw_text = raw_text[:1200]
        sample_questions = document.get("sample_questions") or []
        if isinstance(sample_questions, list):
            sample_questions_text = " | ".join(
                str(item).strip() for item in sample_questions if str(item).strip()
            )
        else:
            sample_questions_text = str(document.get("sample_questions_text") or "").strip()
        return "\n".join(
            line
            for line in (
                f"document: {document.get('document_title') or ''}",
                f"path: {document.get('path_text') or ''}",
                f"canonical_label: {document.get('canonical_label') or ''}",
                f"unit_topic: {document.get('unit_topic') or ''}",
                f"unit_type: {document.get('unit_type') or ''}",
                f"chunk_kind: {document.get('chunk_kind') or ''}",
                f"context: {document.get('retrieval_context') or ''}",
                f"sample_questions: {sample_questions_text}",
                f"text: {raw_text}",
            )
            if line.strip()
        )

    def _fit_classifier(
        self,
        *,
        features: np.ndarray,
        labels: np.ndarray,
        classes: list[str],
        examples: list[EvidenceRoleTrainingExample],
        embedding_source: str,
    ) -> EvidenceRoleLinearModel:
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

        weights = np.asarray(classifier.coef_, dtype=np.float64).T
        bias = np.asarray(classifier.intercept_, dtype=np.float64)
        training_accuracy = float(classifier.score(normalized_features, labels))

        return EvidenceRoleLinearModel(
            classes=classes,
            weights=weights,
            bias=bias,
            mean=mean,
            scale=scale,
            training_accuracy=training_accuracy,
            embedding_source=embedding_source,
            examples=examples,
        )

    def _predict_probabilities(
        self,
        features: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray,
    ) -> np.ndarray:
        logits = (features @ weights) + bias
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def _standardize(
        self,
        features: np.ndarray,
        mean: np.ndarray,
        scale: np.ndarray,
    ) -> np.ndarray:
        return (features - mean) / scale

    def _resolve_fixture_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            return candidate

        repo_root = Path(__file__).resolve().parents[4]
        for base in (Path.cwd(), repo_root):
            resolved = (base / candidate).resolve()
            if resolved.exists():
                return resolved
        return (repo_root / candidate).resolve()
