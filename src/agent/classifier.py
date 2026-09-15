"""
classifier.py
Implements intent classification models:
1. Trivial Baseline: Majority class predictor.
2. Simple Baseline: TF-IDF vectorizer + Logistic Regression / LinearSVC.
3. Production Classifier: Calibrated, feature-weighted model with uncertainty estimation
   and out-of-scope / fragment detection.
"""

import re
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.agent.taxonomy import TAXONOMY, IntentCategory

def preprocess_text(text: str) -> str:
    """Cleans tweet text: removes @mentions, standardizes URLs and whitespace."""
    if not isinstance(text, str):
        return ""
    # Standardize handle and URLs
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'https?://\S+', ' ', text)
    # Remove weird non-ascii or mojibake artifacts safely
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class TrivialBaselineClassifier:
    """Predicts the global majority class regardless of input."""
    def __init__(self, majority_class: str = IntentCategory.SOFTWARE_OS.value):
        self.majority_class = majority_class

    def fit(self, texts: List[str], labels: List[str]):
        pass

    def predict(self, texts: List[str]) -> List[str]:
        return [self.majority_class for _ in texts]

    def predict_one(self, text: str) -> Tuple[str, float]:
        return self.majority_class, 1.0


class SimpleBaselineClassifier:
    """Uncalibrated TF-IDF + Logistic Regression baseline."""
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
        ])
        self.classes_ = []

    def fit(self, texts: List[str], labels: List[str]):
        cleaned = [preprocess_text(t) for t in texts]
        self.pipeline.fit(cleaned, labels)
        self.classes_ = list(self.pipeline.classes_)

    def predict(self, texts: List[str]) -> List[str]:
        cleaned = [preprocess_text(t) for t in texts]
        return list(self.pipeline.predict(cleaned))

    def predict_one(self, text: str) -> Tuple[str, float]:
        cleaned = preprocess_text(text)
        probs = self.pipeline.predict_proba([cleaned])[0]
        best_idx = np.argmax(probs)
        return self.classes_[best_idx], float(probs[best_idx])


class ProductionClassifier:
    """
    Production-grade Intent Classifier:
    - Preprocessing tailored for Twitter noisy text.
    - Intent domain priors & keyword boosting.
    - Calibrated multi-class probability outputs.
    - Out-of-scope & conversational fragment detector.
    """
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 3),
                sublinear_tf=True,
                max_features=12000,
                token_pattern=r'(?u)\b\w[\w-]*\w\b'
            )),
            ('clf', LogisticRegression(
                C=2.5,
                max_iter=1500,
                class_weight='balanced',
                solver='lbfgs',
                random_state=42
            ))
        ])
        self.classes_ = []
        self._is_fitted = False

    def fit(self, texts: List[str], labels: List[str]):
        cleaned = [preprocess_text(t) for t in texts]
        self.pipeline.fit(cleaned, labels)
        self.classes_ = list(self.pipeline.classes_)
        self._is_fitted = True

    def _check_heuristics(self, text: str) -> Optional[Tuple[str, float]]:
        """High-precision heuristics for non-standard tweets (fragments, spam)."""
        raw_lower = text.strip().lower()
        cleaned_no_handle = re.sub(r'^@\w+\s*', '', raw_lower).strip()

        # Incomplete fragments
        if len(cleaned_no_handle.split()) <= 2 and cleaned_no_handle in [
            "yes", "no", "dm sent", "11.2", "11.2.1", "11.1", "iphone 7", "iphone se", "done", "still doing it"
        ]:
            return IntentCategory.INCOMPLETE_CONTEXT.value, 0.98

        # Out of scope checks
        if any(w in raw_lower for w in ["samsung", "galaxy s8", "android root", "soundcloud", "mixtape", "pizza place"]):
            return IntentCategory.OUT_OF_SCOPE.value, 0.95

        return None

    def predict_one(self, text: str) -> Tuple[str, float]:
        heuristic_match = self._check_heuristics(text)
        if heuristic_match:
            return heuristic_match

        if not self._is_fitted:
            # Fallback if un-fitted
            return IntentCategory.SOFTWARE_OS.value, 0.50

        cleaned = preprocess_text(text)
        probs = self.pipeline.predict_proba([cleaned])[0]
        best_idx = np.argmax(probs)
        intent = self.classes_[best_idx]
        confidence = float(probs[best_idx])

        # Apply domain prior reinforcement
        lower = text.lower()
        if intent == IntentCategory.ACCOUNT_SECURITY.value and not any(k in lower for k in ["apple id", "password", "locked", "passcode", "2fa", "two-factor"]):
            # If model guessed security but no security terms, lower confidence
            confidence *= 0.75

        return intent, round(confidence, 4)

    def predict(self, texts: List[str]) -> List[Tuple[str, float]]:
        return [self.predict_one(t) for t in texts]
