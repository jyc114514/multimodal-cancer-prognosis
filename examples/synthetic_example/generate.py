"""Generate a small synthetic survival table for software smoke checks."""

import argparse
import csv
from pathlib import Path

import numpy as np


def make_synthetic_rows(n_subjects=80, seed=17):
    rng = np.random.default_rng(seed)
    molecular = rng.normal(size=n_subjects)
    image_feature = rng.normal(size=n_subjects)
    clinical_feature = rng.normal(loc=55.0, scale=10.0, size=n_subjects)
    linear_risk = 0.55 * molecular - 0.3 * image_feature + 0.015 * (clinical_feature - 55.0)
    event_time = rng.exponential(scale=np.exp(-linear_risk))
    censor_time = rng.exponential(scale=1.25, size=n_subjects)
    durations = np.minimum(event_time, censor_time) + 0.01
    events = (event_time <= censor_time).astype(int)

    rows = []
    for index in range(n_subjects):
        rows.append([
            "SYNTH_{:04d}".format(index),
            "{:.6f}".format(float(durations[index])),
            str(int(events[index])),
            "{:.6f}".format(float(molecular[index])),
            "{:.6f}".format(float(image_feature[index])),
            "{:.6f}".format(float(clinical_feature[index])),
        ])
    if n_subjects > 6:
        rows[2][4] = ""
    return rows


def write_synthetic_csv(output, n_subjects=80, seed=17):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "patient_id", "time", "event", "molecular_feature",
            "image_feature", "clinical_feature",
        ])
        writer.writerows(make_synthetic_rows(n_subjects, seed))
    return n_subjects


def main():
    parser = argparse.ArgumentParser(description="Write a synthetic-only survival CSV.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--subjects", type=int, default=80)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    count = write_synthetic_csv(args.output, args.subjects, args.seed)
    print("synthetic_rows_written={}".format(count))


if __name__ == "__main__":
    main()
