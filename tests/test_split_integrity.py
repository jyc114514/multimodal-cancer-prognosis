import numpy as np

from cancer_prognosis.splits import assert_disjoint_ids, make_group_folds


def test_repeated_rows_stay_in_one_patient_fold():
    groups = np.asarray(["p1", "p1", "p2", "p3", "p4", "p5", "p6", "p7"])
    events = np.asarray([1, 1, 0, 1, 0, 1, 0, 1])
    folds = make_group_folds(groups, events, n_splits=3, seed=9)
    appearances = {}
    for train_rows, validation_rows in folds:
        assert_disjoint_ids(groups[train_rows], groups[validation_rows])
        for group in set(groups[validation_rows].tolist()):
            appearances[group] = appearances.get(group, 0) + 1
    assert set(appearances) == set(groups.tolist())
    assert all(count == 1 for count in appearances.values())


def test_overlap_guard_rejects_shared_subject():
    try:
        assert_disjoint_ids(["a", "b"], ["b", "c"])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected overlap to be rejected.")
