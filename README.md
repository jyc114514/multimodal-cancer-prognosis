# Multimodal Cancer Prognosis — Clean Methodology Reference

**Research code.** This repository is a newly authored, sanitized reference workflow for retrospective survival-risk analysis. It does not contain patient-level data, raw TCGA files, clinical tables, images, embeddings, model weights, predictions or caches.

The public code represents a corrected, leakage-aware methodology reference. Historical experimental artifacts and patient-level data are not distributed. This repository is **not** the exact source snapshot for any historical Phase4 run and does not reproduce the historical teacher/student or HyperBank/HGNN architecture or results. No performance headline is included.

## Overview

The examples show patient-grouped folds, optional training-fold-only variance selection, training-fold-only imputation and standardization, a regularized Cox model, and censoring-aware ranking evaluation. A small pathology utility mean-pools already-extracted slide feature vectors to a subject-level vector. It does not read whole-slide images, extract pathology embeddings, or implement learned multiple-instance learning.

## Repository structure

- `src/cancer_prognosis/`: cohort schema, patient-grouped splits, optional feature selection, preprocessing, survival model/evaluation, and slide-feature pooling abstraction.
- `scripts/`: prepare a table, train, evaluate and run stability checks.
- `configs/`: example configuration schema.
- `examples/synthetic_example/`: generator for synthetic smoke-test data only.
- `tests/`: lightweight unit and smoke tests.
- `docs/`: methodology, validation, data availability and reproducibility notes.

## Installation

Use Python 3.8 or newer.

~~~
python -m pip install -r requirements.txt
python -m compileall .
pytest -q
~~~

## Usage

Generate a synthetic CSV and inspect its schema:

~~~
python examples/synthetic_example/generate.py --output /tmp/synthetic_survival.csv
python scripts/prepare_data.py --input /tmp/synthetic_survival.csv
~~~

Fit a simple model on a training table, evaluate on a separate table, or run a fixed-model CV smoke workflow:

~~~
python scripts/train.py --train-csv /path/to/train.csv --model-out /tmp/cox_model.json
python scripts/evaluate.py --input-csv /path/to/evaluation.csv --model /tmp/cox_model.json
PYTHONPATH=src python scripts/run_stability_analysis.py --input-csv /tmp/synthetic_survival.csv
~~~

The input table requires a subject key, follow-up time, event indicator and numeric feature columns. Pool repeated slide feature rows to one subject row before loading. `mean_pool_slide_features` accepts pre-extracted numeric vectors only; it is simple mean aggregation, not MIL. `TrainFoldVarianceSelector` is optional and generic; fit a fresh selector inside each training fold. Neither component reconstructs a historical model or selected gene panel.

## Data

No study data are included. Researchers must obtain data from official repositories such as the [NCI Genomic Data Commons](https://portal.gdc.cancer.gov/) and comply with the source data-use terms and any applicable institutional requirements. This repository does not package or download TCGA/CPTAC records. The synthetic generator is only for software smoke checks; its outputs are not scientific results.

## Reproducibility and limitations

Read [methodology](docs/methodology.md), [validation](docs/validation.md), [reproducibility](docs/reproducibility.md), and [data availability](docs/data_availability.md) before adapting the code. All learned feature selection, imputation and scaling must be fitted using training data only. A C-index is a survival-risk ranking metric, not diagnostic accuracy, calibration or clinical utility. The exact historical TCGA endpoint mapping and time origin were not recovered.

## Clinical disclaimer

**Research use only. Not intended for clinical decision-making, diagnosis, screening, treatment selection, or patient management.** This software has not been clinically validated or approved for clinical use.

## License and source identity

No license file is included, and no license grant should be inferred. This newly authored reference must not be presented as the exact source for historical results.
