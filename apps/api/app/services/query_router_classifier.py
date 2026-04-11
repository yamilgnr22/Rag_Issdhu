import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
from sklearn.linear_model import LogisticRegression

from app.models.contracts import DocumentClass
from app.services.embedding_gateway import EmbeddingGateway


VALID_ROUTER_CLASSES: tuple[DocumentClass, ...] = (
    "legal_normative",
    "technical_standard",
    "general_document",
)
_MODEL_CACHE: dict[tuple[str, int, int, str], "RouterLinearModel"] = {}
_CACHE_LOCK = Lock()


@dataclass
class RouterTrainingExample:
    case_id: str
    question: str
    document_class: DocumentClass


@dataclass
class RouterLinearModel:
    classes: list[DocumentClass]
    weights: np.ndarray
    bias: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    training_accuracy: float
    embedding_source: str
    examples: list[RouterTrainingExample]


@dataclass
class RouterClassification:
    query_type: str
    target_classes: list[DocumentClass]
    confidence: float
    probabilities: dict[str, float]
    source: str
    decision_trace: list[str]


class QueryRouterClassifier:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.embeddings = EmbeddingGateway(settings)

    def classify(self, *, question: str) -> RouterClassification | None:
        if not self.settings.query_router_classifier_enabled:
            return None

        model = self._load_model()
        if model is None:
            return None

        try:
            vectors, embedding_source = self.embeddings.embed_texts([question.strip()])
        except Exception:
            return None
        if not vectors:
            return None

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
        if confidence < self.settings.query_router_classifier_min_confidence:
            return None

        predicted_class = model.classes[best_index]
        return RouterClassification(
            query_type=predicted_class,
            target_classes=[predicted_class],
            confidence=confidence,
            probabilities=probabilities_dict,
            source="linear_classifier",
            decision_trace=[
                "router:provider=linear_classifier",
                f"router:embedding={embedding_source}",
                f"router:class={predicted_class}",
                f"router:confidence={confidence}",
                f"router:margin={margin}",
                f"router:train_acc={round(model.training_accuracy, 2)}",
                "router:class_probs="
                + ",".join(f"{label}:{probabilities_dict.get(label, 0.0)}" for label in model.classes),
                f"router:{predicted_class}",
            ],
        )

    def _load_model(self) -> RouterLinearModel | None:
        fixture_path = self._resolve_fixture_path(self.settings.query_router_classifier_fixture_path)
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
                vectors, embedding_source = self.embeddings.embed_texts([example.question for example in examples])
            except Exception:
                return None
            if not vectors:
                return None

            classes = [
                label
                for label in VALID_ROUTER_CLASSES
                if any(example.document_class == label for example in examples)
            ]
            if len(classes) < 2:
                return None

            class_to_index = {label: index for index, label in enumerate(classes)}
            features = np.asarray(vectors, dtype=np.float64)
            labels = np.asarray(
                [class_to_index[example.document_class] for example in examples],
                dtype=np.int64,
            )
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

    def _load_examples(self, fixture_path: Path) -> list[RouterTrainingExample]:
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        raw_cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
        examples: list[RouterTrainingExample] = []
        for item in raw_cases:
            if not isinstance(item, dict):
                continue
            expectation = item.get("expectation") if isinstance(item.get("expectation"), dict) else {}
            document_class = str(expectation.get("document_class", "")).strip()
            question = str(item.get("question", "")).strip()
            case_id = str(item.get("case_id", "")).strip() or question[:80]
            if document_class not in VALID_ROUTER_CLASSES or not question:
                continue
            examples.append(
                RouterTrainingExample(
                    case_id=case_id,
                    question=question,
                    document_class=document_class,  # type: ignore[arg-type]
                )
            )
        return examples

    def _fit_classifier(
        self,
        *,
        features: np.ndarray,
        labels: np.ndarray,
        classes: list[DocumentClass],
        examples: list[RouterTrainingExample],
        embedding_source: str,
    ) -> RouterLinearModel:
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

        return RouterLinearModel(
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
        if weights.ndim == 2 and weights.shape[1] == 1 and bias.shape[0] == 1:
            logits = (features @ weights).reshape(-1) + float(bias[0])
            positive = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
            negative = 1.0 - positive
            return np.stack([negative, positive], axis=1)
        logits = features @ weights + bias
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def _standardize(self, features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
        return (features - mean) / scale

    def _resolve_fixture_path(self, fixture_path: str) -> Path:
        path = Path(fixture_path)
        if path.is_absolute():
            return path
        return Path.cwd() / path
