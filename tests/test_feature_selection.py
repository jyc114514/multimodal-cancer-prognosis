import numpy as np
import pytest

from cancer_prognosis.selection import TrainFoldVarianceSelector


def test_variance_selector_fits_and_transforms_with_fixed_training_columns():
    train = np.array([[0.0, 10.0, 1.0], [1.0, 10.0, 7.0], [2.0, 10.0, 3.0]])
    selector = TrainFoldVarianceSelector(top_k=1).fit(train)
    assert selector.indices_.tolist() == [2]
    transformed = selector.transform(np.array([[100.0, -5.0, 5.0]]))
    np.testing.assert_allclose(transformed, [[5.0]])


def test_variance_selector_requires_fit():
    with pytest.raises(RuntimeError, match="fit"):
        TrainFoldVarianceSelector().transform(np.ones((1, 2)))
