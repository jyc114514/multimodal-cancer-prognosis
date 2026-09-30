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
from cancer_prognosis.survival import c_index


def main():
    parser = argparse.ArgumentParser(description="Evaluate a frozen model on a separate cohort table.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    cohort = read_cohort_csv(args.input_csv)
    payload = json.loads(Path(args.model).read_text(encoding="utf-8"))
    if cohort.feature_names != payload["feature_names"]:
        raise ValueError("Evaluation feature columns/order do not match the frozen model.")
    preprocessor = TrainFoldPreprocessor.from_dict(payload["preprocessor"])
    model = CoxRidgeModel(alpha=payload["alpha"])
    model.coefficients_ = np.asarray(payload["coefficients"], dtype=float)
    prediction = model.predict_risk(preprocessor.transform(cohort.features))
    score = c_index(cohort.durations, cohort.events, prediction)
    print(json.dumps({
        "subjects": int(len(cohort.subject_ids)),
        "events": int(np.sum(cohort.events)),
        "c_index": score,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
