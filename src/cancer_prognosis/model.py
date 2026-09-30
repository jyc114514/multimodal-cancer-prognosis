"""Small ridge-regularized linear Cox model for a clean reference workflow."""

import numpy as np

from .survival import cox_negative_partial_likelihood_and_gradient


class CoxRidgeModel:
    def __init__(self, alpha=0.1, learning_rate=0.02, epochs=300):
        if alpha < 0 or learning_rate <= 0 or epochs < 1:
            raise ValueError("Invalid optimizer settings.")
        self.alpha = float(alpha)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.coefficients_ = None

    def fit(self, features, durations, events):
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("Training features must be a non-empty matrix.")
        if not np.all(np.isfinite(x)):
            raise ValueError("Preprocess missing and non-finite values before fitting.")
        self.coefficients_ = np.zeros(x.shape[1], dtype=float)
        for _ in range(self.epochs):
            _, gradient = cox_negative_partial_likelihood_and_gradient(
                x, durations, events, self.coefficients_, alpha=self.alpha
            )
            norm = float(np.linalg.norm(gradient))
            if norm > 10.0:
                gradient = gradient * (10.0 / norm)
            self.coefficients_ -= self.learning_rate * gradient
        return self

    def predict_risk(self, features):
        if self.coefficients_ is None:
            raise RuntimeError("Fit the model before prediction.")
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.coefficients_):
            raise ValueError("Prediction features have the wrong shape.")
        if not np.all(np.isfinite(x)):
            raise ValueError("Preprocess missing and non-finite values before prediction.")
        return x @ self.coefficients_
