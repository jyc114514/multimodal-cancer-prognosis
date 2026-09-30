import numpy as np

from cancer_prognosis.data import read_cohort_csv
from cancer_prognosis.pipeline import run_cross_validation
from cancer_prognosis.splits import assert_disjoint_ids
from examples.synthetic_example.generate import write_synthetic_csv


def test_synthetic_training_and_evaluation_smoke(tmp_path):
    train_csv = tmp_path / "synthetic_train.csv"
    evaluation_csv = tmp_path / "synthetic_evaluation.csv"
    write_synthetic_csv(train_csv, n_subjects=75, seed=31, id_prefix="TRAIN")
    write_synthetic_csv(evaluation_csv, n_subjects=35, seed=32, id_prefix="EVAL")
    cohort = read_cohort_csv(train_csv)
    evaluation = read_cohort_csv(evaluation_csv)
    assert_disjoint_ids(cohort.subject_ids, evaluation.subject_ids)
    results = run_cross_validation(
        cohort, n_splits=3, seed=12, alpha=0.2, learning_rate=0.02, epochs=50, top_variance_k=2
    )
    scores = np.asarray([row["c_index"] for row in results])
    assert len(results) == 3
    assert np.all(np.isfinite(scores))
    assert np.all((scores >= 0.0) & (scores <= 1.0))
