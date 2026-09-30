"""Cohort schema and CSV loading. IDs remain in memory and are never logged."""

import csv
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class SurvivalCohort:
    subject_ids: np.ndarray
    durations: np.ndarray
    events: np.ndarray
    features: np.ndarray
    feature_names: List[str]

    def validate(self):
        self.subject_ids = np.asarray(self.subject_ids, dtype=str)
        self.durations = np.asarray(self.durations, dtype=float)
        self.events = np.asarray(self.events, dtype=int)
        self.features = np.asarray(self.features, dtype=float)

        n = len(self.subject_ids)
        if n == 0:
            raise ValueError("The cohort is empty.")
        if self.subject_ids.ndim != 1:
            raise ValueError("subject_ids must be one-dimensional.")
        if self.durations.shape != (n,) or self.events.shape != (n,):
            raise ValueError("Outcome arrays must have one value per subject.")
        if self.features.ndim != 2 or self.features.shape[0] != n:
            raise ValueError("features must be a two-dimensional subject-by-feature matrix.")
        if self.features.shape[1] != len(self.feature_names):
            raise ValueError("feature_names must match the feature matrix columns.")
        if len(set(self.subject_ids.tolist())) != n:
            raise ValueError("Expected one row per subject; aggregate repeated slides before loading.")
        if not np.all(np.isfinite(self.durations)) or np.any(self.durations <= 0):
            raise ValueError("Durations must be finite and greater than zero.")
        if not np.all(np.isin(self.events, [0, 1])):
            raise ValueError("Events must be coded as 1=observed event and 0=right-censored.")
        if self.features.shape[1] == 0:
            raise ValueError("At least one numeric feature column is required.")
        return self


def _parse_feature(value):
    if value is None or value.strip() == "":
        return np.nan
    parsed = float(value)
    return parsed


def read_cohort_csv(path, id_column="subject_id", time_column="time", event_column="event"):
    """Read a one-row-per-subject table without printing or returning IDs in logs."""
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV header is missing.")
        required = [id_column, time_column, event_column]
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise ValueError("Required cohort columns are missing.")
        feature_names = [name for name in reader.fieldnames if name not in required]
        if not feature_names:
            raise ValueError("At least one numeric feature column is required.")

        subject_ids = []
        durations = []
        events = []
        feature_rows = []
        for row in reader:
            subject_id = (row.get(id_column) or "").strip()
            if not subject_id:
                raise ValueError("A subject ID is blank.")
            try:
                duration = float(row[time_column])
                event = int(row[event_column])
            except (TypeError, ValueError):
                raise ValueError("Outcome columns must contain numeric time and binary event values.")
            subject_ids.append(subject_id)
            durations.append(duration)
            events.append(event)
            try:
                feature_rows.append([_parse_feature(row.get(name)) for name in feature_names])
            except (TypeError, ValueError):
                raise ValueError("Feature columns must be numeric or blank.")

    return SurvivalCohort(
        subject_ids=np.asarray(subject_ids, dtype=str),
        durations=np.asarray(durations, dtype=float),
        events=np.asarray(events, dtype=int),
        features=np.asarray(feature_rows, dtype=float),
        feature_names=feature_names,
    ).validate()
