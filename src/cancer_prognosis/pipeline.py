"""Fixed-model cross-validation with fold-local preprocessing."""

import numpy as np

from .model import CoxRidgeModel
from .preprocessing import TrainFoldPreprocessor
from .selection import TrainFoldVarianceSelector
from .splits import assert_disjoint_ids, make_group_folds
from .survival import c_index


def run_cross_validation(cohort, n_splits=5, seed=2026, alpha=0.1, learning_rate=0.02, epochs=300, top_variance_k=None):
    cohort.validate()
    folds = make_group_folds(cohort.subject_ids, cohort.events, n_splits=n_splits, seed=seed)
    results = []
    for fold_number, (train_rows, validation_rows) in enumerate(folds):
        assert_disjoint_ids(cohort.subject_ids[train_rows], cohort.subject_ids[validation_rows])
        train_raw = cohort.features[train_rows]
        validation_raw = cohort.features[validation_rows]
        if top_variance_k is not None:
            selector = TrainFoldVarianceSelector(top_k=top_variance_k).fit(train_raw)
            train_raw = selector.transform(train_raw)
            validation_raw = selector.transform(validation_raw)
        preprocessor = TrainFoldPreprocessor()
        train_x = preprocessor.fit_transform(train_raw)
        validation_x = preprocessor.transform(validation_raw)
        model = CoxRidgeModel(alpha=alpha, learning_rate=learning_rate, epochs=epochs)
        model.fit(train_x, cohort.durations[train_rows], cohort.events[train_rows])
        prediction = model.predict_risk(validation_x)
        score = c_index(
            cohort.durations[validation_rows], cohort.events[validation_rows], prediction
        )
        results.append({
            "fold": int(fold_number),
            "n_train": int(len(train_rows)),
            "n_validation": int(len(validation_rows)),
            "n_validation_events": int(np.sum(cohort.events[validation_rows])),
            "c_index": float(score),
        })
    return results
