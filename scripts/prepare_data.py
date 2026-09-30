#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cancer_prognosis.data import read_cohort_csv


def main():
    parser = argparse.ArgumentParser(description="Validate a one-row-per-subject survival table.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--id-column", default="subject_id")
    parser.add_argument("--time-column", default="time")
    parser.add_argument("--event-column", default="event")
    args = parser.parse_args()

    cohort = read_cohort_csv(args.input, args.id_column, args.time_column, args.event_column)
    missing = int(np.sum(~np.isfinite(cohort.features)))
    print(json.dumps({
        "subjects": int(len(cohort.subject_ids)),
        "features": int(cohort.features.shape[1]),
        "events": int(np.sum(cohort.events)),
        "missing_feature_cells": missing,
        "schema_valid": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
