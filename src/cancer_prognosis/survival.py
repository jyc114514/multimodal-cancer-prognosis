"""Censoring-aware survival metrics and Cox partial likelihood helpers."""

import numpy as np


def c_index(durations, events, risk_scores):
    """Harrell-style concordance; a larger score means higher modeled event risk."""
    time = np.asarray(durations, dtype=float)
    event = np.asarray(events, dtype=bool)
    risk = np.asarray(risk_scores, dtype=float)
    if time.shape != event.shape or time.shape != risk.shape or time.ndim != 1:
        raise ValueError("durations, events, and risk_scores must be aligned 1D arrays.")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(risk)):
        raise ValueError("durations and risk scores must be finite.")

    concordant = 0.0
    comparable = 0
    n = len(time)
    for i in range(n):
        if not event[i]:
            continue
        for j in range(n):
            if time[i] < time[j]:
                comparable += 1
                if risk[i] > risk[j]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    concordant += 0.5
    return float(concordant / comparable) if comparable else float("nan")


def cox_negative_partial_likelihood_and_gradient(features, durations, events, coefficients, alpha=0.0):
    """Breslow-tie negative partial likelihood and gradient for a linear Cox score."""
    x = np.asarray(features, dtype=float)
    time = np.asarray(durations, dtype=float)
    event = np.asarray(events, dtype=bool)
    beta = np.asarray(coefficients, dtype=float)
    if x.ndim != 2 or x.shape[0] != len(time) or time.shape != event.shape:
        raise ValueError("Cox inputs have incompatible shapes.")
    if x.shape[1] != len(beta):
        raise ValueError("Coefficient count must match feature columns.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(time)):
        raise ValueError("Cox inputs must be finite after preprocessing.")
    if not np.any(event):
        raise ValueError("At least one observed event is required.")

    score = x @ beta
    loss = 0.0
    gradient = np.zeros_like(beta)
    event_times = np.unique(time[event])
    for event_time in event_times:
        event_rows = np.flatnonzero(event & (time == event_time))
        risk_rows = np.flatnonzero(time >= event_time)
        risk_scores = score[risk_rows]
        shift = float(np.max(risk_scores))
        weights = np.exp(risk_scores - shift)
        denominator = float(np.sum(weights))
        n_tied = len(event_rows)
        log_denominator = shift + np.log(denominator)
        loss -= float(np.sum(score[event_rows])) - n_tied * log_denominator
        gradient -= np.sum(x[event_rows], axis=0)
        weighted_mean = np.sum(x[risk_rows] * weights[:, None], axis=0) / denominator
        gradient += n_tied * weighted_mean

    n_events = float(np.sum(event))
    loss = loss / n_events + 0.5 * alpha * float(beta @ beta)
    gradient = gradient / n_events + alpha * beta
    return float(loss), gradient
