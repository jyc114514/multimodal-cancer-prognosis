import numpy as np
import pytest

from cancer_prognosis.pathology import mean_pool_slide_features


def test_mean_pool_slide_features_keeps_first_seen_subject_order():
    subject_ids, pooled = mean_pool_slide_features(
        ["subject_b", "subject_a", "subject_b"],
        [[1.0, 3.0], [8.0, 4.0], [5.0, 7.0]],
    )
    assert subject_ids.tolist() == ["subject_b", "subject_a"]
    np.testing.assert_allclose(pooled, [[3.0, 5.0], [8.0, 4.0]])


def test_mean_pool_rejects_nonfinite_features():
    with pytest.raises(ValueError, match="finite"):
        mean_pool_slide_features(["subject_a"], [[np.nan, 1.0]])
