"""Training-fold-only median imputation and standardization."""

import numpy as np


class TrainFoldPreprocessor:
    def __init__(self):
        self.medians_ = None
        self.means_ = None
        self.scales_ = None

    def fit(self, features):
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("Training features must be a non-empty 2D array.")
        finite = np.isfinite(x)
        medians = []
        for column in range(x.shape[1]):
            observed = x[finite[:, column], column]
            medians.append(float(np.median(observed)) if observed.size else 0.0)
        self.medians_ = np.asarray(medians, dtype=float)
        filled = self._impute(x)
        self.means_ = np.mean(filled, axis=0)
        scales = np.std(filled, axis=0)
        self.scales_ = np.where(scales > 1e-12, scales, 1.0)
        return self

    def _impute(self, features):
        if self.medians_ is None:
            raise RuntimeError("Call fit on training data before transform.")
        x = np.asarray(features, dtype=float).copy()
        if x.ndim != 2 or x.shape[1] != len(self.medians_):
            raise ValueError("Feature matrix has the wrong number of columns.")
        for column, median in enumerate(self.medians_):
            invalid = ~np.isfinite(x[:, column])
            x[invalid, column] = median
        return x

    def transform(self, features):
        if self.means_ is None or self.scales_ is None:
            raise RuntimeError("Call fit on training data before transform.")
        return (self._impute(features) - self.means_) / self.scales_

    def fit_transform(self, features):
        return self.fit(features).transform(features)

    def to_dict(self):
        if self.means_ is None:
            raise RuntimeError("Preprocessor is not fitted.")
        return {
            "medians": self.medians_.tolist(),
            "means": self.means_.tolist(),
            "scales": self.scales_.tolist(),
        }

    @classmethod
    def from_dict(cls, state):
        obj = cls()
        obj.medians_ = np.asarray(state["medians"], dtype=float)
        obj.means_ = np.asarray(state["means"], dtype=float)
        obj.scales_ = np.asarray(state["scales"], dtype=float)
        if not (obj.medians_.shape == obj.means_.shape == obj.scales_.shape):
            raise ValueError("Preprocessor state arrays must have the same shape.")
        if np.any(obj.scales_ <= 0):
            raise ValueError("Preprocessor scales must be positive.")
        return obj
