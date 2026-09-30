#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cancer_prognosis.data import read_cohort_csv
from cancer_prognosis.model import CoxRidgeModel
from cancer_prognosis.preprocessing import TrainFoldPreprocessor


def main():
    parser = argparse.ArgumentParser(description="Fit a fixed Cox model on a training table.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    cohort = read_cohort_csv(args.train_csv)
    preprocessor = TrainFoldPreprocessor()
    x = preprocessor.fit_transform(cohort.features)
    model = CoxRidgeModel(args.alpha, args.learning_rate, args.epochs)
    model.fit(x, cohort.durations, cohort.events)
    payload = {
        "schema_version": 1,
        "feature_names": cohort.feature_names,
        "preprocessor": preprocessor.to_dict(),
        "coefficients": model.coefficients_.tolist(),
        "alpha": model.alpha,
    }
    output = Path(args.model_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "subjects": int(len(cohort.subject_ids)),
        "events": int(np.sum(cohort.events)),
        "model_saved": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
