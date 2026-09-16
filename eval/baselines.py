"""Baseline systems module for Hiver Customer Support Agent evaluation.

Implements two baseline models to benchmark against the full LLM agent:
1. Trivial Baseline: Majority-class intent prediction + static canned reply + always escalates.
2. Simple Baseline: TF-IDF (1-2 grams) + LogisticRegression intent classifier +
   verbatim top-1 retrieved historical resolution + identical Phase 5 routing rules.
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.router import EscalationReason, route

logger = logging.getLogger(__name__)


class TrivialBaseline:
    """Trivial baseline model.

    Always predicts the empirical majority intent, returns a static canned apology
    linking to Apple Support, and always escalates with LOW_CONFIDENCE.
    """

    def __init__(self, majority_intent: str = "software_update"):
        """Initialize trivial baseline with known majority intent.

        Args:
            majority_intent: Default intent name.
        """
        self.majority_intent = majority_intent

    def predict_intent(self, message: str) -> Dict[str, Any]:
        """Predict majority intent with fixed baseline confidence.

        Args:
            message: Customer tweet.

        Returns:
            Intent dictionary.
        """
        return {
            "intent": self.majority_intent,
            "confidence": 0.50,
            "reasoning": "Trivial baseline constant majority class prediction.",
        }

    def reply(self, message: str, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate static canned apology with support link.

        Args:
            message: Customer tweet.
            retrieved: Retrieved resolution items (unused).

        Returns:
            Reply dictionary.
        """
        return {
            "draft": "We apologize for the inconvenience. For troubleshooting steps, please visit https://support.apple.com.",
            "grounding_used": False,
            "source_thread_ids": [],
            "for_human_review": True,
        }

    def run(
        self,
        customer_message: str,
        retrieved: List[Dict[str, Any]],
        cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute trivial baseline pipeline.

        Args:
            customer_message: Customer tweet.
            retrieved: Retrieved items from FAISS index.
            cfg: Configuration dictionary.

        Returns:
            Combined output dictionary.
        """
        intent_res = self.predict_intent(customer_message)
        routing_res = {
            "action": "escalate",
            "reason": EscalationReason.LOW_CONFIDENCE,
            "decided_by": "rule",
        }
        reply_res = self.reply(customer_message, retrieved)
        return {
            "input": customer_message,
            "intent": intent_res,
            "routing": routing_res,
            "retrieved": retrieved,
            "reply": reply_res,
        }


class SimpleBaseline:
    """Simple baseline model.

    TF-IDF (1-2 grams) + LogisticRegression for intent classification.
    Reply drafter extracts verbatim top-1 retrieved historical resolution.
    Reuses the exact deterministic routing rules from Phase 5.
    """

    def __init__(self, model_path: Optional[Path] = None):
        """Initialize simple baseline.

        Args:
            model_path: Optional path to persisted (vectorizer, classifier) artifact.
        """
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[LogisticRegression] = None
        self.classes: List[str] = []

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def train(self, texts: List[str], labels: List[str], random_state: int = 42) -> None:
        """Train TF-IDF vectorizer and Logistic Regression classifier.

        Trained strictly on non-golden training threads to prevent evaluation leakage.

        Args:
            texts: List of customer message strings.
            labels: Intent labels for training messages.
            random_state: Random seed for reproducibility.
        """
        logger.info("Training SimpleBaseline TF-IDF + LogisticRegression on %d samples...", len(texts))
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
        )
        X = self.vectorizer.fit_transform(texts)
        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            C=1.0,
        )
        self.classifier.fit(X, labels)
        self.classes = list(self.classifier.classes_)
        logger.info("SimpleBaseline trained successfully across %d classes: %s", len(self.classes), self.classes)

    def save(self, path: Path) -> None:
        """Save vectorizer and classifier to disk.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "classifier": self.classifier,
                "classes": self.classes,
            }, f)
        logger.info("Saved SimpleBaseline artifact to %s", path)

    def load(self, path: Path) -> None:
        """Load vectorizer and classifier from disk.

        Args:
            path: Source file path.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.classifier = data["classifier"]
        self.classes = data["classes"]
        logger.info("Loaded SimpleBaseline artifact from %s", path)

    def predict_intent(self, message: str) -> Dict[str, Any]:
        """Predict intent and confidence probability using trained LogisticRegression.

        Args:
            message: Customer tweet.

        Returns:
            Dict with keys 'intent', 'confidence', 'reasoning'.
        """
        if self.vectorizer is None or self.classifier is None:
            return {"intent": "other", "confidence": 0.5, "reasoning": "Untrained model fallback."}

        X = self.vectorizer.transform([message])
        probs = self.classifier.predict_proba(X)[0]
        best_idx = int(np.argmax(probs))
        predicted_class = str(self.classes[best_idx])
        confidence = float(probs[best_idx])

        return {
            "intent": predicted_class,
            "confidence": round(confidence, 4),
            "reasoning": f"TF-IDF LogisticRegression argmax probability {confidence:.3f}.",
        }

    def reply(self, message: str, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Produce reply by taking verbatim top-1 retrieved resolution without LLM.

        Args:
            message: Customer message.
            retrieved: Retrieved historical resolution records.

        Returns:
            Reply dictionary.
        """
        if retrieved:
            top_resolution = str(retrieved[0].get("brand_resolution", "")).strip()
            source_id = str(retrieved[0].get("thread_id", ""))
            # Cap at 280 chars
            if len(top_resolution) > 280:
                top_resolution = top_resolution[:277] + "..."
            return {
                "draft": top_resolution,
                "grounding_used": True,
                "source_thread_ids": [source_id] if source_id else [],
                "for_human_review": False,
            }

        return {
            "draft": "Please reach out to Apple Support directly at https://support.apple.com for help with your device.",
            "grounding_used": False,
            "source_thread_ids": [],
            "for_human_review": True,
        }

    def run(
        self,
        customer_message: str,
        retrieved: List[Dict[str, Any]],
        cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute simple baseline pipeline.

        Reuses exact route() rules from Phase 5.

        Args:
            customer_message: Customer tweet.
            retrieved: Retrieved items from FAISS index.
            cfg: Configuration dictionary.

        Returns:
            Output dictionary.
        """
        intent_res = self.predict_intent(customer_message)
        routing_res = route(
            message=customer_message,
            intent=intent_res["intent"],
            confidence=intent_res["confidence"],
            retrieved=retrieved,
            cfg=cfg,
        )
        reply_res = self.reply(customer_message, retrieved)
        reply_res["for_human_review"] = (routing_res.get("action") == "escalate")

        return {
            "input": customer_message,
            "intent": intent_res,
            "routing": routing_res,
            "retrieved": retrieved,
            "reply": reply_res,
        }


def train_simple_baseline(
    corpus: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> SimpleBaseline:
    """Train and persist SimpleBaseline on resolution corpus.

    Labels training corpus items using taxonomy classifier to create a robust
    multi-class training set without touching the golden evaluation set.

    Args:
        corpus: List of resolution corpus items with customer_message and intent.
        cfg: Configuration dictionary.

    Returns:
        Trained SimpleBaseline instance.
    """
    from src.intents import classify_intent, load_taxonomy

    cache_dir = Path(cfg["data"].get("cache_dir", ".cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "simple_baseline_model.pkl"
    seed = cfg.get("sampling", {}).get("random_seed", 42)

    taxonomy = load_taxonomy(Path(cfg["data"]["taxonomy_json"]))

    # Subsample 2500 training items to keep training fast and balanced
    np.random.seed(seed)
    sampled_corpus = corpus
    if len(corpus) > 2500:
        indices = np.random.choice(len(corpus), size=2500, replace=False)
        sampled_corpus = [corpus[i] for i in indices]

    texts = []
    labels = []
    logger.info("Generating training labels for SimpleBaseline on %d items...", len(sampled_corpus))
    for item in sampled_corpus:
        msg = item["customer_message"]
        clf = classify_intent(msg, taxonomy)
        intent = clf["intent"]
        texts.append(msg)
        labels.append(intent)

    baseline = SimpleBaseline()
    baseline.train(texts, labels, random_state=seed)
    baseline.save(model_path)
    return baseline
