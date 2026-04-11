import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
from sklearn.linear_model import LogisticRegression

from app.services.embedding_gateway import EmbeddingGateway


VALID_EVIDENCE_INTENTS: tuple[str, ...] = ("substantive", "reference", "mixed")
_MODEL_CACHE: dict[tuple[str, int, int, str], "EvidenceIntentLinearModel"] = {}
_CACHE_LOCK = Lock()


@dataclass
class EvidenceIntentTrainingExample:
    case_id: str
    intent: str
    question: str


@dataclass
class EvidenceIntentLinearModel:
    classes: list[str]
    weights: np.ndarray
    bias: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    training_accuracy: float
    embedding_source: str
    examples: list[EvidenceIntentTrainingExample]


@dataclass
class EvidenceIntentClassification:
    intent: str
    confidence: float
    probabilities: dict[str, float]
    source: str
    decision_trace: list[str]


class QueryEvidenceIntentClassifier:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.embeddings = EmbeddingGateway(settings)

    def classify(self, *, question: str) -> EvidenceIntentClassification:
        if not self.settings.query_evidence_intent_classifier_enabled:
            return self._mixed_result(reason="intent:provider=disabled")

        model = self._load_model()
        if model is None:
            return self._mixed_result(reason="intent:provider=unavailable")

        try:
            vectors, embedding_source = self.embeddings.embed_texts([question.strip()])
        except Exception:
            return self._mixed_result(reason="intent:provider=embedding_error")
        if not vectors:
            return self._mixed_result(reason="intent:provider=embedding_empty")

        features = self._standardize(np.asarray(vectors, dtype=np.float64), model.mean, model.scale)
        probabilities = self._predict_probabilities(features, model.weights, model.bias)[0]
        best_index = int(np.argmax(probabilities))
        sorted_probabilities = np.sort(probabilities)[::-1]
        best_probability = float(probabilities[best_index])
        second_probability = float(sorted_probabilities[1]) if len(sorted_probabilities) > 1 else 0.0
        confidence = round(best_probability, 2)
        margin = round(best_probability - second_probability, 2)
        probabilities_dict = {
            label: round(float(probabilities[index]), 2)
            for index, label in enumerate(model.classes)
        }

        if confidence < self.settings.query_evidence_intent_classifier_min_confidence:
            mixed = self._mixed_result(reason="intent:provider=low_confidence")
            mixed.decision_trace.extend(
                [
                    f"intent:embedding={embedding_source}",
                    f"intent:confidence={confidence}",
                    f"intent:margin={margin}",
                    "intent:class=mixed",
                ]
            )
            return mixed

        intent = model.classes[best_index]
        return EvidenceIntentClassification(
            intent=intent,
            confidence=confidence,
            probabilities=probabilities_dict,
            source="linear_classifier",
            decision_trace=[
                "intent:provider=linear_classifier",
                f"intent:embedding={embedding_source}",
                f"intent:class={intent}",
                f"intent:confidence={confidence}",
                f"intent:margin={margin}",
                "intent:class_probs="
                + ",".join(f"{label}:{probabilities_dict.get(label, 0.0)}" for label in model.classes),
            ],
        )

    def _mixed_result(self, *, reason: str) -> EvidenceIntentClassification:
        return EvidenceIntentClassification(
            intent="mixed",
            confidence=0.0,
            probabilities={"mixed": 1.0, "substantive": 0.0, "reference": 0.0},
            source="fallback",
            decision_trace=[reason],
        )

    def _load_model(self) -> EvidenceIntentLinearModel | None:
        fixture_path = self._resolve_fixture_path(self.settings.query_evidence_intent_classifier_fixture_path)
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
                    [example.question for example in examples]
                )
            except Exception:
                return None
            if not vectors:
                return None

            classes = [
                label
                for label in VALID_EVIDENCE_INTENTS
                if any(example.intent == label for example in examples)
            ]
            if len(classes) < 2:
                return None

            class_to_index = {label: index for index, label in enumerate(classes)}
            features = np.asarray(vectors, dtype=np.float64)
            labels = np.asarray([class_to_index[example.intent] for example in examples], dtype=np.int64)
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

    def _load_examples(self, fixture_path: Path) -> list[EvidenceIntentTrainingExample]:
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        raw_cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
        examples: list[EvidenceIntentTrainingExample] = []
        for item in raw_cases:
            if not isinstance(item, dict):
                continue
            intent = str(item.get("intent", "")).strip().lower()
            question = str(item.get("question", "")).strip()
            case_id = str(item.get("case_id", "")).strip() or question[:80]
            if intent not in VALID_EVIDENCE_INTENTS or not question:
                continue
            examples.append(
                EvidenceIntentTrainingExample(
                    case_id=case_id,
                    intent=intent,
                    question=question,
                )
            )
        return examples

    def _fit_classifier(
        self,
        *,
        features: np.ndarray,
        labels: np.ndarray,
        classes: list[str],
        examples: list[EvidenceIntentTrainingExample],
        embedding_source: str,
    ) -> EvidenceIntentLinearModel:
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

        return EvidenceIntentLinearModel(
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
