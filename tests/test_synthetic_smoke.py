import numpy as np

from cancer_prognosis.data import read_cohort_csv
from cancer_prognosis.pipeline import run_cross_validation
from examples.synthetic_example.generate import write_synthetic_csv


def test_synthetic_training_and_evaluation_smoke(tmp_path):
    input_csv = tmp_path / "synthetic.csv"
    write_synthetic_csv(input_csv, n_subjects=75, seed=31)
    cohort = read_cohort_csv(input_csv)
    results = run_cross_validation(
        cohort, n_splits=3, seed=12, alpha=0.2, learning_rate=0.02, epochs=50, top_variance_k=2
    )
    scores = np.asarray([row["c_index"] for row in results])
    assert len(results) == 3
    assert np.all(np.isfinite(scores))
    assert np.all((scores >= 0.0) & (scores <= 1.0))
