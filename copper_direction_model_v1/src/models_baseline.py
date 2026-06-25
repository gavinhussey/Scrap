"""Non-sequence baseline/classifier models with a uniform interface.

Every model exposes:
    fit(X, y) -> self
    predict_proba_up(X) -> np.ndarray of P(up) in [0, 1]

The naive model is a special case: it reads the (unscaled) daily_return feature
and ignores training. The sklearn wrappers operate on the scaled feature matrix.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .utils import SEED

# SVC(probability=True) is intentional for V1 (the paper's SVR arm -> SVC).
# Newer sklearn deprecates the flag and warns at fit time; silence that one.
warnings.filterwarnings(
    "ignore", message=".*probability.*parameter was deprecated.*", category=FutureWarning
)


class NaiveDirectionModel:
    """Predict UP if today's daily_return > 0, else DOWN (momentum-of-one-day).

    Emits soft probabilities (0.55 / 0.45) so probability-based evaluation works.
    Must be given the UNSCALED feature matrix (scaling would move the 0 boundary).
    """

    name = "naive"

    def __init__(self, daily_return_idx: int):
        self.daily_return_idx = daily_return_idx

    def fit(self, X, y):  # noqa: D401 - no training needed
        return self

    def predict_proba_up(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        ret = X[:, self.daily_return_idx]
        return np.where(ret > 0, 0.55, 0.45)


class SklearnProbModel:
    """Thin wrapper giving sklearn classifiers a predict_proba_up method."""

    def __init__(self, name: str, estimator):
        self.name = name
        self.estimator = estimator
        self._single_class = None  # set if training labels are degenerate

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2:
            # Degenerate window: predict the constant base rate, no crash.
            self._single_class = int(y[0])
            return self
        self._single_class = None
        self.estimator.fit(X, y)
        return self

    def predict_proba_up(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self._single_class is not None:
            return np.full(X.shape[0], float(self._single_class))
        proba = self.estimator.predict_proba(X)
        # Column for class "1" (up).
        classes = list(self.estimator.classes_)
        up_col = classes.index(1) if 1 in classes else 1
        return proba[:, up_col]


def make_logistic() -> SklearnProbModel:
    est = LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=SEED
    )
    return SklearnProbModel("logistic", est)


def make_random_forest() -> SklearnProbModel:
    est = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )
    return SklearnProbModel("random_forest", est)


def make_svc() -> SklearnProbModel:
    est = SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=SEED,
    )
    return SklearnProbModel("svc", est)
