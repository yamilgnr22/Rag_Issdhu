from collections import Counter
import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
from sklearn.linear_model import LogisticRegression

from app.models.contracts import QueryStrategy
from app.services.embedding_gateway import EmbeddingGateway
from app.services.query_routing import QueryRoute
from app.services.query_structure import QueryStructure


VALID_STRATEGIES: tuple[QueryStrategy, ...] = ("focal", "multi_branch", "global")
_MODEL_CACHE: dict[tuple[str, int, int, str], "StrategyLinearModel"] = {}
_CACHE_LOCK = Lock()


@dataclass
class StrategyTrainingExample:
    case_id: str
    strategy: QueryStrategy
    question: str


@dataclass
class StrategyLinearModel:
    classes: list[QueryStrategy]
    weights: np.ndarray
    bias: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    training_accuracy: float
    embedding_source: str
    examples: list[StrategyTrainingExample]


@dataclass
class StrategyClassification:
    strategy: QueryStrategy
    confidence: float
    source: str
    reason_short: str
    decision_trace: list[str]


class QueryStrategyClassifier:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.embeddings = EmbeddingGateway(settings)

    def classify(
        self,
        *,
        question: str,
        route: QueryRoute,
        structure: QueryStructure,
    ) -> StrategyClassification | None:
        del route
        del structure

        if not self.settings.query_strategy_classifier_enabled:
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

        if confidence < self.settings.query_strategy_classifier_min_confidence:
            return None

        predicted_strategy = model.classes[best_index]
        class_trace = ",".join(
            f"{label}:{round(float(probabilities[index]), 2)}"
            for index, label in enumerate(model.classes)
        )
        return StrategyClassification(
            strategy=predicted_strategy,
            confidence=confidence,
            source="linear_classifier",
            reason_short=f"Linear query-strategy classifier selected {predicted_strategy}.",
            decision_trace=[
                "planner:provider=linear_classifier",
                f"planner:embedding={embedding_source}",
                f"planner:strategy={predicted_strategy}",
                f"planner:confidence={confidence}",
                f"planner:margin={margin}",
                f"planner:train_acc={round(model.training_accuracy, 2)}",
                f"planner:class_probs={class_trace}",
            ],
        )

    def default_strategy(self) -> QueryStrategy:
        fixture_path = self._resolve_fixture_path(self.settings.query_strategy_classifier_fixture_path)
        if not fixture_path.exists():
            return "focal"
        examples = self._load_examples(fixture_path)
        if not examples:
            return "focal"
        counts = Counter(example.strategy for example in examples)
        return counts.most_common(1)[0][0]

    def _load_model(self) -> StrategyLinearModel | None:
        fixture_path = self._resolve_fixture_path(self.settings.query_strategy_classifier_fixture_path)
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
            if len(examples) < 3:
                return None

            try:
                vectors, embedding_source = self.embeddings.embed_texts(
                    [example.question for example in examples]
                )
            except Exception:
                return None

            if not vectors:
                return None

            classes = [label for label in VALID_STRATEGIES if any(example.strategy == label for example in examples)]
            if len(classes) < 2:
                return None

            class_to_index = {label: index for index, label in enumerate(classes)}
            features = np.asarray(vectors, dtype=np.float64)
            labels = np.asarray([class_to_index[example.strategy] for example in examples], dtype=np.int64)
            model = self._fit_softmax_classifier(
                features=features,
                labels=labels,
                classes=classes,
                examples=examples,
                embedding_source=embedding_source,
            )

            _MODEL_CACHE.clear()
            _MODEL_CACHE[cache_key] = model
            return model

    def _load_examples(self, fixture_path: Path) -> list[StrategyTrainingExample]:
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        raw_cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
        examples: list[StrategyTrainingExample] = []
        for item in raw_cases:
            if not isinstance(item, dict):
                continue
            strategy = str(item.get("strategy", "")).strip().lower()
            question = str(item.get("question", "")).strip()
            case_id = str(item.get("case_id", "")).strip() or question[:80]
            if strategy not in VALID_STRATEGIES or not question:
                continue
            examples.append(
                StrategyTrainingExample(
                    case_id=case_id,
                    strategy=strategy,
                    question=question,
                )
            )
        return examples

    def _fit_softmax_classifier(
        self,
        *,
        features: np.ndarray,
        labels: np.ndarray,
        classes: list[QueryStrategy],
        examples: list[StrategyTrainingExample],
        embedding_source: str,
    ) -> StrategyLinearModel:
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

        return StrategyLinearModel(
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
