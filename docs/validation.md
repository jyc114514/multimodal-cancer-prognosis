# Validation

- Define the intended-use population, unit of analysis, time origin, event, and censoring rules before splitting.
- Use subject_id as the patient-level key. Split by subject before feature selection, imputation, scaling, batch correction, or outcome-informed filtering.
- Keep all rows from a subject in one partition; this reference loader then requires one row per subject after aggregation.
- If selecting features, hyperparameters, checkpoints, or thresholds, use an inner validation procedure. Do not report the same data used for selection as an untouched evaluation score.
- Reserve an independent external cohort when making transportability claims. A retrospective random split is internal validation.
- For diagnosis, predefine thresholds and report clinically meaningful sensitivity/specificity, predictive values at relevant prevalence, calibration, subgroup performance, and uncertainty.
- For survival, use censoring-aware discrimination and calibration measures and state the endpoint precisely.
- Synthetic smoke data only test software paths. Synthetic metrics are not scientific evidence.
