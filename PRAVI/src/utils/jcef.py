"""Joint cross-fitted evidence fusion utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_2d(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    return array


def _sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def evidence_features(image_evidence, text_evidence, include_synergy):
    """Construct additive or agreement-aware pair-evidence features."""
    image = np.asarray(image_evidence, dtype=np.float64).reshape(-1)
    text = np.asarray(text_evidence, dtype=np.float64).reshape(-1)
    if image.size != text.size:
        raise ValueError("image and text evidence must have equal size")
    columns = [image, text]
    if include_synergy:
        columns.extend((image * text, np.abs(image - text)))
    return np.column_stack(columns)


@dataclass(frozen=True)
class RidgeLogisticCalibrator:
    """Small deterministic ridge-logistic evidence model fitted by Newton steps."""

    mean: np.ndarray
    scale: np.ndarray
    intercept: float
    coefficients: np.ndarray
    iterations: int

    @classmethod
    def fit(
        cls,
        features,
        labels,
        l2=0.01,
        max_iter=50,
        tolerance=1e-8,
    ):
        x = _as_2d(features)
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        if x.shape[0] != y.size or y.size == 0:
            raise ValueError("one non-empty label is required per feature row")
        if not np.all(np.isin(y, (0.0, 1.0))):
            raise ValueError("labels must be binary")
        if np.unique(y).size != 2:
            raise ValueError("both classes are required")

        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-8] = 1.0
        normalized = (x - mean) / scale
        design = np.column_stack((np.ones(y.size), normalized))
        weights = np.zeros(design.shape[1], dtype=np.float64)
        penalty = np.full(design.shape[1], float(l2), dtype=np.float64)
        penalty[0] = 0.0
        identity = np.eye(design.shape[1], dtype=np.float64) * 1e-9

        iterations = 0
        for iterations in range(1, int(max_iter) + 1):
            probability = _sigmoid(design @ weights)
            curvature = np.maximum(probability * (1.0 - probability), 1e-8)
            gradient = (
                design.T @ (probability - y) / y.size + penalty * weights
            )
            hessian = (
                design.T @ (design * curvature[:, None]) / y.size
                + np.diag(penalty)
                + identity
            )
            step = np.linalg.solve(hessian, gradient)
            weights -= step
            if np.linalg.norm(step) <= float(tolerance) * (
                1.0 + np.linalg.norm(weights)
            ):
                break
        return cls(
            mean=mean.astype(np.float32),
            scale=scale.astype(np.float32),
            intercept=float(weights[0]),
            coefficients=weights[1:].astype(np.float32),
            iterations=iterations,
        )

    def decision_function(self, features):
        x = _as_2d(features)
        if x.shape[1] != self.coefficients.size:
            raise ValueError("feature width does not match fitted calibrator")
        normalized = (x - self.mean) / self.scale
        return (
            self.intercept + normalized @ self.coefficients
        ).astype(np.float32)

    def summary(self, feature_names):
        names = list(feature_names)
        if len(names) != self.coefficients.size:
            raise ValueError("one name is required per coefficient")
        return {
            "intercept": self.intercept,
            "iterations": self.iterations,
            "features": {
                name: {
                    "coefficient_standardized": float(coefficient),
                    "mean": float(mean),
                    "scale": float(scale),
                }
                for name, coefficient, mean, scale in zip(
                    names, self.coefficients, self.mean, self.scale
                )
            },
        }


def diagnostic_gate(
    joint_auc,
    delta_fixed,
    delta_fixed_ci_low,
    delta_additive,
    delta_additive_ci_low,
    fold_deltas,
    min_auc=0.6314,
    min_fixed_delta=0.005,
    min_additive_delta=0.002,
):
    """Apply the preregistered JCEF signal gate."""
    fold_deltas = np.asarray(fold_deltas, dtype=np.float64).reshape(-1)
    checks = {
        "absolute_auc": float(joint_auc) >= float(min_auc),
        "fixed_global_delta": float(delta_fixed) >= float(min_fixed_delta),
        "fixed_global_delta_ci": float(delta_fixed_ci_low) > 0.0,
        "learned_additive_delta": float(delta_additive)
        >= float(min_additive_delta),
        "learned_additive_delta_ci": float(delta_additive_ci_low) > 0.0,
        "fold_consistency": bool(
            fold_deltas.size > 0 and np.all(fold_deltas > 0.0)
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_auc": float(min_auc),
            "min_fixed_global_delta": float(min_fixed_delta),
            "min_learned_additive_delta": float(min_additive_delta),
            "paired_ci_low_must_exceed": 0.0,
            "every_fold_delta_must_exceed": 0.0,
        },
    }
