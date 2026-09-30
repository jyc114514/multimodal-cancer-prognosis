import numpy as np

from cancer_prognosis.splits import assert_disjoint_ids, make_group_folds


def test_repeated_subject_rows_stay_in_one_fold():
    subject_ids = np.asarray(["s1", "s1", "s2", "s3", "s4", "s5", "s6", "s7"])
    events = np.asarray([1, 1, 0, 1, 0, 1, 0, 1])
    folds = make_group_folds(subject_ids, events, n_splits=3, seed=9)
    appearances = {}
    for train_rows, validation_rows in folds:
        assert_disjoint_ids(subject_ids[train_rows], subject_ids[validation_rows])
        for subject in set(subject_ids[validation_rows].tolist()):
            appearances[subject] = appearances.get(subject, 0) + 1
    assert set(appearances) == set(subject_ids.tolist())
    assert all(count == 1 for count in appearances.values())


def test_overlap_guard_rejects_shared_subject_id():
    try:
        assert_disjoint_ids(["a", "b"], ["b", "c"])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected overlap to be rejected.")
