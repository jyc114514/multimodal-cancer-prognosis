# Methodology

## Unit of analysis

The loader requires one row per subject. Aggregate repeated slides or specimens with a prespecified rule before using this interface. The split helper can keep repeated group rows together, but the model input table should be defined and audited at the intended analysis grain.

## Preprocessing

Imputation medians, means, and standard deviations are fit on training rows only. Validation or evaluation rows are transformed with the saved training state. Feature selection and any learned batch correction must follow the same rule.

## Model

The reference model is a ridge-regularized linear Cox model with Breslow handling for tied event times. It is intentionally small and is not the historical HyperBank/HGNN or teacher/student model. It is provided to make data flow and survival metrics inspectable.

## Risk score and metric

The linear predictor is a relative risk score. Harrell-style C-index evaluates pairwise ordering for comparable observations. It does not provide calibrated event probabilities or diagnostic operating characteristics.


## Pathology feature input

`mean_pool_slide_features` accepts pre-extracted slide-level numeric features and averages them to one subject-level vector. It does not extract image features or implement learned MIL. If feature selection is needed, `TrainFoldVarianceSelector` is optional and must be fitted on training rows within each fold. It is a generic method, not the historical project's gene panel.
