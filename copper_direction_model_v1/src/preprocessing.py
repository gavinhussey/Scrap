"""Leakage-safe preprocessing.

A ``Preprocessor`` is fit on the TRAIN window only (median imputation + robust
scaling) and then applied to validation/test rows. It is never fit on the full
dataset, and extreme price moves are kept (we describe them with features rather
than overwriting them).
"""
from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


class Preprocessor:
    """Median imputer + RobustScaler, fit on training rows only."""

    def __init__(self, scaler: str = "robust"):
        self.imputer = SimpleImputer(strategy="median")
        if scaler == "standard":
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler()
        else:
            self.scaler = RobustScaler()
        self._fitted = False

    def fit(self, X_train: np.ndarray) -> "Preprocessor":
        X_train = np.asarray(X_train, dtype=float)
        imputed = self.imputer.fit_transform(X_train)
        self.scaler.fit(imputed)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Preprocessor.transform called before fit.")
        X = np.asarray(X, dtype=float)
        return self.scaler.transform(self.imputer.transform(X))

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)
