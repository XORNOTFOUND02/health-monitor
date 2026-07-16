"""
Comprehensive evaluation metrics for binary classification.

Computes all standard metrics from sklearn plus additional statistics
useful for medical diagnostics evaluation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Computes comprehensive evaluation metrics for binary classification.
    
    Handles edge cases: empty data, all-same-class, NaN predictions.
    All metrics are computed from actual data, never hardcoded.
    """

    def compute_all(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Compute all evaluation metrics from ground truth and predicted scores.

        Parameters
        ----------
        y_true : np.ndarray
            Binary ground truth labels (0 or 1).
        y_score : np.ndarray
            Predicted probabilities or scores (continuous).
        threshold : float
            Decision threshold for converting scores to binary predictions.

        Returns
        -------
        dict
            Dictionary containing all computed metrics.
        """
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_score = np.asarray(y_score, dtype=np.float64).ravel()

        # Sanitize inputs
        valid_mask = np.isfinite(y_score) & np.isfinite(y_true)
        if not valid_mask.all():
            n_invalid = int(np.sum(~valid_mask))
            logger.warning("Removing %d samples with non-finite values", n_invalid)
            y_true = y_true[valid_mask]
            y_score = y_score[valid_mask]

        if len(y_true) == 0:
            return self._empty_metrics()

        y_pred = (y_score >= threshold).astype(int)

        has_positive = np.sum(y_true) > 0
        has_negative = np.sum(y_true == 0) > 0
        both_classes = has_positive and has_negative

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # Basic metrics
        accuracy = float(accuracy_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        # Specificity (true negative rate)
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        # AUC-ROC and Average Precision
        if both_classes:
            auc_roc = float(roc_auc_score(y_true, y_score))
            average_precision = float(average_precision_score(y_true, y_score))
        else:
            auc_roc = 0.0
            average_precision = 0.0

        # Matthews Correlation Coefficient
        mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0

        # Log loss (requires probabilities, clip to avoid log(0))
        y_score_clipped = np.clip(y_score, 1e-15, 1 - 1e-15)
        try:
            logloss = float(log_loss(y_true, y_score_clipped))
        except ValueError:
            logloss = 0.0

        # Brier score
        brier = float(brier_score_loss(y_true, y_score))

        # Youden's J statistic and optimal threshold from ROC
        if both_classes:
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            youden_index = float(j_scores[optimal_idx])
            optimal_threshold = float(roc_thresholds[optimal_idx])
        else:
            youden_index = 0.0
            optimal_threshold = threshold

        # Detection rate and prevalence
        detection_rate = float(np.mean(y_pred))
        detection_prevalence = float(np.mean(y_true))

        # Balanced accuracy
        balanced_accuracy = float((recall + specificity) / 2)

        # True/false positive/negative rates
        tpr = recall  # same as recall
        fpr_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        tnr = specificity

        return {
            "confusion_matrix": cm.tolist(),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "auc_roc": auc_roc,
            "average_precision": average_precision,
            "matthews_corrcoef": mcc,
            "log_loss": logloss,
            "brier_score": brier,
            "youden_index": youden_index,
            "optimal_threshold": optimal_threshold,
            "detection_rate": detection_rate,
            "detection_prevalence": detection_prevalence,
            "balanced_accuracy": balanced_accuracy,
            "true_positive_rate": tpr,
            "false_positive_rate": fpr_rate,
            "false_negative_rate": fnr,
            "true_negative_rate": tnr,
            "n_samples": len(y_true),
            "n_positive": int(np.sum(y_true)),
            "n_negative": int(np.sum(y_true == 0)),
        }

    def per_threshold_metrics(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        thresholds: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Compute Precision, Recall, F1, FPR, TPR at each threshold.

        Parameters
        ----------
        y_true : np.ndarray
            Binary ground truth labels.
        y_score : np.ndarray
            Predicted probabilities.
        thresholds : list of float, optional
            Thresholds to evaluate. Defaults to 9 evenly spaced thresholds.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: threshold, precision, recall, f1, fpr, tpr
        """
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_score = np.asarray(y_score, dtype=np.float64).ravel()

        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # Remove invalid samples
        valid_mask = np.isfinite(y_score) & np.isfinite(y_true)
        y_true = y_true[valid_mask]
        y_score = y_score[valid_mask]

        if len(y_true) == 0:
            return pd.DataFrame(columns=["threshold", "precision", "recall", "f1", "fpr", "tpr"])

        has_positive = np.sum(y_true) > 0
        has_negative = np.sum(y_true == 0) > 0

        records = []
        for thr in thresholds:
            y_pred = (y_score >= thr).astype(int)

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1_val = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            fpr_val = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            tpr_val = rec  # same as recall

            records.append({
                "threshold": thr,
                "precision": prec,
                "recall": rec,
                "f1": f1_val,
                "fpr": fpr_val,
                "tpr": tpr_val,
            })

        return pd.DataFrame(records)

    def classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> str:
        """
        Generate a formatted text classification report.

        Parameters
        ----------
        y_true : np.ndarray
            Binary ground truth labels.
        y_pred : np.ndarray
            Binary predicted labels.

        Returns
        -------
        str
            Formatted text report similar to sklearn's classification_report.
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        # Sanitize
        valid_mask = np.isfinite(y_true.astype(float)) & np.isfinite(y_pred.astype(float))
        if not valid_mask.all():
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]

        if len(y_true) == 0:
            return "No valid samples for classification report."

        return classification_report(
            y_true,
            y_pred,
            target_names=["Negative", "Positive"],
            digits=4,
            zero_division=0,
        )

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return a dictionary of zero-valued metrics for empty data."""
        return {
            "confusion_matrix": [[0, 0], [0, 0]],
            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "specificity": 0.0,
            "auc_roc": 0.0,
            "average_precision": 0.0,
            "matthews_corrcoef": 0.0,
            "log_loss": 0.0,
            "brier_score": 0.0,
            "youden_index": 0.0,
            "optimal_threshold": 0.5,
            "detection_rate": 0.0,
            "detection_prevalence": 0.0,
            "balanced_accuracy": 0.0,
            "true_positive_rate": 0.0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "true_negative_rate": 0.0,
            "n_samples": 0,
            "n_positive": 0,
            "n_negative": 0,
        }
