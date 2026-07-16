"""
Model evaluation module with medical-relevant metrics.

Provides :class:`ModelEvaluator` for computing classification metrics
that are especially important in clinical/medical decision support:
accuracy, precision, recall, specificity, F1, AUC-ROC, AUC-PR, and
Matthews Correlation Coefficient (MCC).

MCC is particularly valuable for imbalanced medical datasets because it
accounts for all four confusion-matrix quadrants and returns a score in
[-1, +1] where 0 is no better than random guessing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate model performance with medical-relevant metrics.

    Handles both binary (single-condition) and multi-label
    (multi-condition) evaluation scenarios.

    Examples
    --------
    >>> evaluator = ModelEvaluator()
    >>> y_true = np.array([0, 1, 1, 0, 1])
    >>> y_pred = np.array([0, 1, 0, 0, 1])
    >>> y_score = np.array([0.1, 0.9, 0.4, 0.2, 0.8])
    >>> metrics = evaluator.compute_metrics(y_true, y_pred, y_score)
    """

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute a comprehensive set of classification metrics.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth binary labels (0 or 1).
        y_pred : np.ndarray
            Predicted binary labels (0 or 1).
        y_score : np.ndarray, optional
            Prediction probabilities / confidence scores.
            Required for AUC-ROC and AUC-PR.

        Returns
        -------
        dict
            Dictionary containing:

            - ``accuracy`` -- (TP + TN) / total
            - ``precision`` -- TP / (TP + FP)
            - ``recall`` -- TP / (TP + FN) (sensitivity)
            - ``specificity`` -- TN / (TN + FP)
            - ``f1_score`` -- harmonic mean of precision and recall
            - ``auc_roc`` -- area under the ROC curve (if scores provided)
            - ``auc_pr`` -- area under the precision-recall curve
            - ``mcc`` -- Matthews Correlation Coefficient
            - ``prevalence`` -- fraction of positives in ground truth
        """
        y_true = np.asarray(y_true, dtype=np.int64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.int64).ravel()

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Length mismatch: y_true has {len(y_true)} elements, "
                f"y_pred has {len(y_pred)}"
            )

        cm = self.confusion_matrix(y_true, y_pred)
        tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        prevalence = (tp + fn) / total if total > 0 else 0.0

        metrics: Dict[str, float] = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1_score": float(f1),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "prevalence": float(prevalence),
        }

        # AUC metrics (require probability scores)
        if y_score is not None:
            y_score = np.asarray(y_score, dtype=np.float64).ravel()
            if len(y_score) != len(y_true):
                raise ValueError("y_score length must match y_true")

            # Only compute AUC if both classes are present
            unique_true = np.unique(y_true)
            if len(unique_true) == 2:
                metrics["auc_roc"] = float(roc_auc_score(y_true, y_score))
                metrics["auc_pr"] = float(
                    average_precision_score(y_true, y_score)
                )
            else:
                # Degenerate case: only one class present
                metrics["auc_roc"] = 0.0
                metrics["auc_pr"] = 0.0
                logger.warning(
                    "Only one class present in y_true (%s); AUC metrics set to 0.",
                    unique_true.tolist(),
                )

        return metrics

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    def confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, int]:
        """Compute confusion matrix components.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth binary labels.
        y_pred : np.ndarray
            Predicted binary labels.

        Returns
        -------
        dict
            ``{"TP": int, "TN": int, "FP": int, "FN": int}``
        """
        y_true = np.asarray(y_true, dtype=np.int64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.int64).ravel()

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

    # ------------------------------------------------------------------
    # Per-condition report
    # ------------------------------------------------------------------

    def per_condition_report(
        self,
        true_dict: Dict[str, np.ndarray],
        pred_dict: Dict[str, np.ndarray],
        condition_names: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-condition metrics.

        Parameters
        ----------
        true_dict : dict
            ``{condition_name: np.ndarray}`` of ground-truth labels.
        pred_dict : dict
            ``{condition_name: np.ndarray}`` of predicted labels.
        condition_names : list of str
            Names of conditions to evaluate.

        Returns
        -------
        dict
            ``{condition_name: {metric_name: value}}``
        """
        report: Dict[str, Dict[str, float]] = {}
        for cond in condition_names:
            if cond in true_dict and cond in pred_dict:
                y_true = np.asarray(true_dict[cond]).ravel()
                y_pred = np.asarray(pred_dict[cond]).ravel()
                report[cond] = self.compute_metrics(y_true, y_pred)
            else:
                logger.warning(
                    "Condition '%s' missing from true or pred dicts; skipping.",
                    cond,
                )
        return report

    # ------------------------------------------------------------------
    # Pretty-printed report
    # ------------------------------------------------------------------

    def classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        condition_names: Optional[List[str]] = None,
    ) -> str:
        """Return a pretty-printed classification report.

        For multi-label data, pass ``y_true`` and ``y_pred`` as 2-D arrays
        with shape ``(n_samples, n_conditions)`` and provide
        ``condition_names``.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth labels (1-D binary or 2-D multi-label).
        y_pred : np.ndarray
            Predicted labels (same shape as *y_true*).
        condition_names : list of str or None
            Condition names for multi-label case.

        Returns
        -------
        str
            Formatted report string.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_true.ndim == 2 and y_true.shape[1] > 1:
            # Multi-label
            if condition_names is None:
                condition_names = [f"condition_{i}" for i in range(y_true.shape[1])]
            lines = [
                f"{'Condition':<25} {'Prec':>6} {'Rec':>6} {'Spec':>6} "
                f"{'F1':>6} {'MCC':>6} {'AUC-ROC':>8}",
                "-" * 73,
            ]
            for i, cond in enumerate(condition_names):
                m = self.compute_metrics(y_true[:, i], y_pred[:, i])
                lines.append(
                    f"{cond:<25} {m['precision']:6.3f} {m['recall']:6.3f} "
                    f"{m['specificity']:6.3f} {m['f1_score']:6.3f} "
                    f"{m['mcc']:6.3f} {m.get('auc_roc', 0.0):8.3f}"
                )
            return "\n".join(lines)

        # Binary case
        m = self.compute_metrics(y_true.ravel(), y_pred.ravel())
        lines = [
            "Classification Report",
            "=" * 40,
            f"  Accuracy    : {m['accuracy']:.4f}",
            f"  Precision   : {m['precision']:.4f}",
            f"  Recall      : {m['recall']:.4f}  (sensitivity)",
            f"  Specificity : {m['specificity']:.4f}",
            f"  F1 Score    : {m['f1_score']:.4f}",
            f"  MCC         : {m['mcc']:.4f}",
            f"  AUC-ROC     : {m.get('auc_roc', 0.0):.4f}",
            f"  AUC-PR      : {m.get('auc_pr', 0.0):.4f}",
            f"  Prevalence  : {m['prevalence']:.4f}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Optimal threshold
    # ------------------------------------------------------------------

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        metric: str = "f1",
    ) -> float:
        """Find the optimal classification threshold.

        Two methods are supported:

        * ``metric='f1'`` -- maximise the F1 score across all thresholds.
        * ``metric='youden'`` -- maximise Youden's J statistic
          (TPR - FPR), equivalent to maximising the sum of sensitivity
          and specificity.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth binary labels.
        y_score : np.ndarray
            Prediction probabilities.
        metric : str
            ``"f1"`` or ``"youden"``.

        Returns
        -------
        float
            Optimal threshold value.

        Raises
        ------
        ValueError
            If *metric* is unknown or if there is only one unique class.
        """
        y_true = np.asarray(y_true, dtype=np.int64).ravel()
        y_score = np.asarray(y_score, dtype=np.float64).ravel()

        if len(np.unique(y_true)) < 2:
            logger.warning(
                "Only one class in y_true; returning default threshold 0.5."
            )
            return 0.5

        if metric == "f1":
            precision_arr, recall_arr, thresholds = precision_recall_curve(
                y_true, y_score
            )
            # Compute F1 for each threshold
            with np.errstate(divide="ignore", invalid="ignore"):
                f1_scores = (
                    2 * precision_arr * recall_arr / (precision_arr + recall_arr)
                )
            f1_scores = np.nan_to_num(f1_scores)
            best_idx = int(np.argmax(f1_scores))
            # precision_recall_curve returns n+1 thresholds for n precision values
            # The last precision/recall pair corresponds to threshold=0 (not in array)
            if best_idx < len(thresholds):
                return float(thresholds[best_idx])
            return 0.5

        if metric == "youden":
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            j_scores = tpr - fpr
            best_idx = int(np.argmax(j_scores))
            return float(thresholds[best_idx])

        raise ValueError(f"Unknown metric '{metric}'. Use 'f1' or 'youden'.")

    # ------------------------------------------------------------------
    # Multi-condition convenience
    # ------------------------------------------------------------------

    def evaluate_multi_label(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray] = None,
        condition_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate a multi-label prediction matrix.

        Parameters
        ----------
        y_true : np.ndarray
            Shape ``(n_samples, n_conditions)`` -- binary ground truth.
        y_pred : np.ndarray
            Shape ``(n_samples, n_conditions)`` -- binary predictions.
        y_score : np.ndarray, optional
            Shape ``(n_samples, n_conditions)`` -- probability scores.
        condition_names : list of str or None
            Human-readable condition names.

        Returns
        -------
        dict
            ``{condition_name: {metric_name: value}}``
        """
        y_true = np.asarray(y_true, dtype=np.int64)
        y_pred = np.asarray(y_pred, dtype=np.int64)

        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)
            if y_score is not None:
                y_score = np.asarray(y_score, dtype=np.float64).reshape(-1, 1)

        n_conditions = y_true.shape[1]
        if condition_names is None:
            condition_names = [f"condition_{i}" for i in range(n_conditions)]

        results: Dict[str, Dict[str, float]] = {}
        for i, name in enumerate(condition_names):
            scores_i = y_score[:, i] if y_score is not None else None
            results[name] = self.compute_metrics(
                y_true[:, i], y_pred[:, i], scores_i
            )

        return results
