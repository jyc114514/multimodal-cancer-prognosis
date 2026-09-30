"""Case-level pooling for pre-extracted slide feature vectors.

This is simple arithmetic mean pooling, not learned multiple-instance learning
or whole-slide image feature extraction.
"""

from collections import OrderedDict
import numpy as np


def mean_pool_slide_features(subject_ids, slide_features):
    """Return first-seen subject order and mean-pooled slide features.

    Parameters
    ----------
    subject_ids : sequence
        One grouping key per slide. Keys are used in memory and never logged.
    slide_features : array-like, shape (n_slides, n_features)
        Pre-extracted numeric feature vectors; raw images are not accepted.
    """
    ids = np.asarray(subject_ids, dtype=object)
    x = np.asarray(slide_features, dtype=float)
    if ids.ndim != 1:
        raise ValueError("subject_ids must be one-dimensional.")
    if x.ndim != 2 or x.shape[0] != len(ids):
        raise ValueError("slide_features must have one row per subject ID entry.")
    if len(ids) == 0 or x.shape[1] == 0:
        raise ValueError("At least one slide and one feature are required.")
    if not np.all(np.isfinite(x)):
        raise ValueError("slide_features must be finite; handle missingness explicitly.")

    sums = OrderedDict()
    counts = OrderedDict()
    for subject, row in zip(ids.tolist(), x):
        if subject is None or (isinstance(subject, str) and not subject.strip()):
            raise ValueError("A subject grouping key is blank.")
        try:
            if subject not in sums:
                sums[subject] = np.zeros(x.shape[1], dtype=float)
                counts[subject] = 0
            sums[subject] += row
            counts[subject] += 1
        except TypeError as exc:
            raise ValueError("Subject grouping keys must be hashable scalars.") from exc

    pooled_ids = np.asarray(list(sums.keys()), dtype=str)
    pooled = np.vstack([sums[key] / counts[key] for key in sums])
    return pooled_ids, pooled
