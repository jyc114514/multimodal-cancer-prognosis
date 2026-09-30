#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cancer_prognosis.data import read_cohort_csv
from cancer_prognosis.pipeline import run_cross_validation


def main():
    parser = argparse.ArgumentParser(description="Run fixed-model patient-grouped CV.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    cohort = read_cohort_csv(args.input_csv)
    rows = run_cross_validation(
        cohort, n_splits=args.folds, seed=args.seed, alpha=args.alpha, epochs=args.epochs
    )
    scores = np.asarray([row["c_index"] for row in rows], dtype=float)
    summary = {
        "folds": rows,
        "mean_fold_c_index": float(np.nanmean(scores)),
        "sd_fold_c_index": float(np.nanstd(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "output_note": "Metrics from synthetic fixtures are software checks only; they are not scientific results.",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
