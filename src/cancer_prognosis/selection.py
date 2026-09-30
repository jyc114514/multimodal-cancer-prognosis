"""Optional unsupervised feature selection fitted on training rows only."""

import numpy as np


class TrainFoldVarianceSelector:
    """Keep the top-k features by variance among finite training values.

    This selector is generic and is not a reconstruction of a historical gene
    panel. Fit a new instance inside each training fold.
    """

    def __init__(self, top_k=100):
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        self.top_k = top_k
        self.indices_ = None
        self.n_features_in_ = None

    def fit(self, features):
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("Training features must be a non-empty 2D array.")
        variances = np.full(x.shape[1], -np.inf, dtype=float)
        for j in range(x.shape[1]):
            observed = x[np.isfinite(x[:, j]), j]
            if observed.size:
                variances[j] = float(np.var(observed))
        order = np.argsort(-variances, kind="stable")
        self.indices_ = order[:min(self.top_k, x.shape[1])]
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, features):
        if self.indices_ is None:
            raise RuntimeError("Call fit on training data before transform.")
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError("Feature matrix has the wrong number of columns.")
        return x[:, self.indices_]

    def fit_transform(self, features):
        return self.fit(features).transform(features)
