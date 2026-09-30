"""Patient/group-aware splits with explicit overlap checks."""

from typing import Dict, List, Tuple

import numpy as np


def assert_disjoint_ids(left_ids, right_ids):
    left = set(np.asarray(left_ids, dtype=str).tolist())
    right = set(np.asarray(right_ids, dtype=str).tolist())
    if left.intersection(right):
        raise ValueError("Subject groups overlap across partitions.")
    return True


def make_group_folds(group_ids, events, n_splits=5, seed=2026):
    """Return row-index folds while keeping every group in exactly one validation fold."""
    groups = np.asarray(group_ids, dtype=str)
    event_values = np.asarray(events, dtype=int)
    if groups.ndim != 1 or event_values.shape != groups.shape:
        raise ValueError("group_ids and events must be one-dimensional and aligned.")
    if not np.all(np.isin(event_values, [0, 1])):
        raise ValueError("Events must be binary.")
    unique_groups = list(dict.fromkeys(groups.tolist()))
    if n_splits < 2 or len(unique_groups) < n_splits:
        raise ValueError("Need at least n_splits unique groups and n_splits >= 2.")

    group_rows: Dict[str, List[int]] = {}
    group_event: Dict[str, int] = {}
    for row_index, group in enumerate(groups.tolist()):
        group_rows.setdefault(group, []).append(row_index)
        status = int(event_values[row_index])
        if group in group_event and group_event[group] != status:
            raise ValueError("Rows from a group have inconsistent event status.")
        group_event[group] = status

    rng = np.random.default_rng(seed)
    fold_groups: List[List[str]] = [[] for _ in range(n_splits)]
    offset = 0
    for status in (0, 1):
        stratum = [group for group in unique_groups if group_event[group] == status]
        rng.shuffle(stratum)
        for index, group in enumerate(stratum):
            fold_groups[(offset + index) % n_splits].append(group)
        offset = (offset + len(stratum)) % n_splits

    all_rows = np.arange(len(groups), dtype=int)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for validation_group_list in fold_groups:
        validation_group_set = set(validation_group_list)
        validation_mask = np.asarray([group in validation_group_set for group in groups], dtype=bool)
        validation_rows = all_rows[validation_mask]
        train_rows = all_rows[~validation_mask]
        assert_disjoint_ids(groups[train_rows], groups[validation_rows])
        folds.append((train_rows, validation_rows))

    validation_rows = np.concatenate([valid for _, valid in folds])
    if sorted(validation_rows.tolist()) != list(range(len(groups))):
        raise RuntimeError("Each row must appear in exactly one validation fold.")
    return folds
