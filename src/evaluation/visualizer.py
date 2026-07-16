"""
Publication-quality evaluation charts for health symptom detection.

Generates 10 chart types with computed statistics embedded directly in each image.
Uses matplotlib with Agg backend for headless rendering and seaborn for styling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Use Agg backend for headless rendering
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)

# Professional medical color scheme
COLOR_PRIMARY = "#1a5276"      # Deep blue
COLOR_SECONDARY = "#148f77"    # Teal
COLOR_ACCENT = "#d4ac0d"       # Gold
COLOR_BG = "#f8f9fa"           # Light gray background
COLOR_TEXT = "#2c3e50"         # Dark text
COLOR_GRID = "#dce1e3"         # Light grid
POSITIVE_COLOR = "#c0392b"     # Red for positive
NEGATIVE_COLOR = "#2980b9"     # Blue for negative
CMAP_HEATMAP = "Blues"         # Blue colormap for confusion matrix


class EvaluationVisualizer:
    """
    Generates publication-quality evaluation charts with embedded statistics.
    
    All chart functions are standalone and can be called independently.
    Each chart saves to a PNG file and returns the matplotlib figure.
    """

    def __init__(self, dpi: int = 150, figsize: Tuple[int, int] = (8, 6)):
        """
        Initialize the visualizer.

        Parameters
        ----------
        dpi : int
            Resolution for individual charts (default 150).
        figsize : tuple
            Figure size in inches (width, height).
        """
        self.dpi = dpi
        self.figsize = figsize
        self._setup_style()

    def _setup_style(self) -> None:
        """Configure seaborn/matplotlib for publication-quality output."""
        sns.set_style("whitegrid")
        plt.rcParams.update({
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "font.family": "sans-serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": COLOR_BG,
        })

    # ------------------------------------------------------------------
    # 1. Confusion Matrix
    # ------------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str | Path,
        title: str = "Confusion Matrix",
    ) -> plt.Figure:
        """
        Plot annotated confusion matrix with counts and percentages.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels.
        y_pred : np.ndarray
            Predicted binary labels.
        save_path : str or Path
            Output file path for the PNG.
        title : str
            Chart title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from .metrics import MetricsCalculator
        calc = MetricsCalculator()
        metrics = calc.compute_all(y_true, y_pred, threshold=0.5)

        cm = np.array(metrics["confusion_matrix"])
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp

        fig, ax = plt.subplots(figsize=self.figsize)

        # Create annotated heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=CMAP_HEATMAP,
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            ax=ax,
            cbar=True,
            linewidths=1,
            linecolor="white",
            annot_kws={"size": 16, "weight": "bold", "color": "white"},
        )

        # Add percentage annotations
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = val / total * 100 if total > 0 else 0
                ax.text(
                    j + 0.5, i + 0.72,
                    f"({pct:.1f}%)",
                    ha="center", va="center",
                    fontsize=10, color="white" if val > total * 0.15 else "black",
                )

        # Title with computed metrics
        subtitle = (
            f"Accuracy: {metrics['accuracy']:.4f}  |  "
            f"Precision: {metrics['precision']:.4f}  |  "
            f"Recall: {metrics['recall']:.4f}  |  "
            f"F1: {metrics['f1']:.4f}"
        )
        ax.set_title(f"{title}\n{subtitle}", pad=15, fontsize=13, color=COLOR_TEXT)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)

        # Add cell labels (TP/TN/FP/FN)
        ax.text(1.5, -0.15, f"TP={tp}", ha="center", fontsize=10, color=POSITIVE_COLOR)
        ax.text(-0.15, 1.5, f"FP={fp}", ha="center", fontsize=10, color=POSITIVE_COLOR, rotation=90)

        fig.tight_layout()
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Saved confusion matrix to %s", save_path)
        return fig

    # ------------------------------------------------------------------
    # 2. ROC Curve
    # ------------------------------------------------------------------

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        save_path: str | Path,
        title: str = "ROC Curve",
    ) -> plt.Figure:
        """
        Plot ROC curve with AUC in legend and Youden's J threshold.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels.
        y_score : np.ndarray
            Predicted probabilities.
        save_path : str or Path
            Output file path for the PNG.
        title : str
            Chart title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score).ravel()

        # Remove invalid
        valid = np.isfinite(y_score) & np.isfinite(y_true)
        y_true, y_score = y_true[valid], y_score[valid]

        fig, ax = plt.subplots(figsize=self.figsize)

        if len(np.unique(y_true)) < 2:
            ax.text(0.5, 0.5, "Insufficient class diversity\nfor ROC curve",
                    ha="center", va="center", fontsize=14, color=COLOR_TEXT)
            ax.set_title(title)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            return fig

        # Compute ROC
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc_val = roc_auc_score(y_true, y_score)

        # Plot ROC curve
        ax.plot(fpr, tpr, color=COLOR_PRIMARY, linewidth=2.5,
                label=f"Model (AUC = {auc_val:.4f})")

        # Diagonal baseline (random classifier)
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5,
                label="Random (AUC = 0.50)")

        # Youden's J statistic point
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        youden_fpr = fpr[optimal_idx]
        youden_tpr = tpr[optimal_idx]
        youden_thr = thresholds[optimal_idx]

        ax.plot(youden_fpr, youden_tpr, "o", color=COLOR_ACCENT, markersize=10,
                label=f"Youden (thr={youden_thr:.3f})")

        # Fill under curve
        ax.fill_between(fpr, tpr, alpha=0.15, color=COLOR_PRIMARY)

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{title}\nAUC = {auc_val:.4f}  |  Youden Threshold = {youden_thr:.4f}",
                     pad=15, fontsize=13, color=COLOR_TEXT)
        ax.legend(loc="lower right", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Saved ROC curve to %s", save_path)
        return fig

    # ------------------------------------------------------------------
    # 3. Precision-Recall Curve
    # ------------------------------------------------------------------

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        save_path: str | Path,
        title: str = "Precision-Recall Curve",
    ) -> plt.Figure:
        """
        Plot Precision-Recall curve with Average Precision.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels.
        y_score : np.ndarray
            Predicted probabilities.
        save_path : str or Path
            Output file path for the PNG.
        title : str
            Chart title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score).ravel()

        valid = np.isfinite(y_score) & np.isfinite(y_true)
        y_true, y_score = y_true[valid], y_score[valid]

        fig, ax = plt.subplots(figsize=self.figsize)

        if len(np.unique(y_true)) < 2:
            ax.text(0.5, 0.5, "Insufficient class diversity\nfor PR curve",
                    ha="center", va="center", fontsize=14, color=COLOR_TEXT)
            ax.set_title(title)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            return fig

        # Compute PR curve
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        # Baseline (positive class prevalence)
        prevalence = np.mean(y_true)

        ax.plot(recall, precision, color=COLOR_SECONDARY, linewidth=2.5,
                label=f"Model (AP = {ap:.4f})")
        ax.axhline(y=prevalence, color="gray", linestyle="--", linewidth=1.5,
                   label=f"Baseline (Prevalence = {prevalence:.4f})")

        ax.fill_between(recall, precision, alpha=0.15, color=COLOR_SECONDARY)

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{title}\nAverage Precision = {ap:.4f}  |  Prevalence = {prevalence:.4f}",
                     pad=15, fontsize=13, color=COLOR_TEXT)
        ax.legend(loc="lower left", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Saved PR curve to %s", save_path)
        return fig

    # ------------------------------------------------------------------
    # 4. Calibration Curve
    # ------------------------------------------------------------------

    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        save_path: str | Path,
        title: str = "Calibration Curve",
    ) -> plt.Figure:
        """
        Plot reliability diagram (calibration curve).

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels.
        y_score : np.ndarray
            Predicted probabilities.
        save_path : str or Path
            Output file path for the PNG.
        title : str
            Chart title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score).ravel()

        valid = np.isfinite(y_score) & np.isfinite(y_true)
        y_true, y_score = y_true[valid], y_score[valid]

        fig, ax = plt.subplots(figsize=self.figsize)

        if len(np.unique(y_true)) < 2 or len(y_true) < 5:
            ax.text(0.5, 0.5, "Insufficient data\nfor calibration curve",
                    ha="center", va="center", fontsize=14, color=COLOR_TEXT)
            ax.set_title(title)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            return fig

        # Compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_score, n_bins=10, strategy="uniform"
        )

        # Brier score
        brier = brier_score_loss(y_true, y_score)

        # Expected Calibration Error (ECE)
        bin_edges = np.linspace(0, 1, 11)
        ece = 0.0
        n_total = len(y_true)
        for i in range(len(mean_predicted_value)):
            bin_mask = (y_score >= bin_edges[i]) & (y_score < bin_edges[i + 1]) if i < len(mean_predicted_value) - 1 else (y_score >= bin_edges[i])
            n_bin = np.sum(bin_mask)
            if n_bin > 0:
                frac_pos = np.mean(y_true[bin_mask])
                ece += abs(frac_pos - mean_predicted_value[i]) * n_bin / n_total

        ax.plot(mean_predicted_value, fraction_of_positives, "s-",
                color=COLOR_PRIMARY, linewidth=2, markersize=8, label="Model")
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.5,
                label="Perfect calibration")

        ax.fill_between([0, 1], [0, 0.02], [0.02, 1], alpha=0.05, color="gray")

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(
            f"{title}\nBrier Score: {brier:.4f}  |  ECE: {ece:.4f}",
            pad=15, fontsize=13, color=COLOR_TEXT,
        )
        ax.legend(loc="upper left", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Saved calibration curve to %s", save_path)
        return fig

    # ------------------------------------------------------------------
    # 5. Feature Importance
    # ------------------------------------------------------------------

    def plot_feature_importance(
        self,
        feature_names: List[str],
        importance_values: np.ndarray,
        save_path: str | Path,
        title: str = "Feature Importance",
        top_n: int = 20,
    ) -> plt.Figure:
        """
        Plot horizontal bar chart of feature importance (top N).

        Parameters
        ----------
        feature_names : list of str
            Feature names.
        importance_values : np.ndarray
            Importance scores.
        save_path : str or Path
            Output file path for the PNG.
        title : str
            Chart title.
        top_n : int
            Number of top features to display.

        Returns
        -------
        matplotlib.figure.Figure
        """
        names = list(feature_names)
        values = np.asarray(importance_values).ravel()

        # Sort and take top N
        sorted_idx = np.argsort(values)[::-1][:top_n]
        top_names = [names[i] for i in sorted_idx]
        top_values = values[sorted_idx]

        # Normalize for display
        max_val = np.max(top_values) if len(top_values) > 0 else 1.0
        top_values_norm = top_values / max_val if max_val > 0 else top_values

        fig, ax = plt.subplots(figsize=(self.figsize[0], max(6, top_n * 0.4)))

        colors = sns.color_palette(CMAP_HEATMAP, n_colors=top_n)[::-1]
        bars = ax.barh(range(top_n), top_values_norm[::-1], color=colors, edgecolor="white", linewidth=0.5)

        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names[::-1], fontsize=9)
        ax.set_xlabel("Normalized Importance (Gain)")
        ax.set_title(f"{title}\nTop {min(top_n, len(names))} Features",
                     pad=15, fontsize=13, color=COLOR_TEXT)

        # Add value labels at bar ends
        for i, (bar, val) in enumerate(zip(bars, top_values[::-1])):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}",
                va="center", fontsize=8, color=COLOR_TEXT,
            )

        ax.set_xlim([0, 1.15])
        ax.invert_yaxis()

        fig.tight_layout()
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Saved feature importance to %s", save_path)
        return fig

    # ------------------------------------------------------------------
    # 6. Threshold Analysis
    # ------------------------------------------------------------------

    def plot_threshold_analysis(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        save_path: str | Path,
        title: str = "Threshold Analysis",
    ) -> plt.Figure:
        """
        Plot Precision, Recall, F1 vs decision threshold.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels.
        y_score : np.ndarray
            Predicted probabilities.
        save_path : str or Path
            Output file path for the PNG.
        title : str
            Chart title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from .metrics import MetricsCalculator
        calc = MetricsCalculator()
        df = calc.per_threshold_metrics(y_true, y_score)

        if df.empty:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "Insufficient data\nfor threshold analysis",
                    ha="center", va="center", fontsize=14)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            return fig

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(df["threshold"], df["precision"], "o-", color=COLOR_PRIMARY,
                linewidth=2, markersize=6, label="Precision")
        ax.plot(df["threshold"], df["recall"], "s-", color=COLOR_SECONDARY,
                linewidth=2, markersize=6, label="Recall")
        ax.plot(df["threshold"], df["f1"], "^-", color=COLOR_ACCENT,
                linewidth=2, markersize=6, label="F1 Score")

        # Find optimal threshold (max F1)
        best_idx = df["f1"].idxmax()
        best_thr = df.loc[best_idx, "threshold"]
        best_f1 = df.loc[best_idx, "f1"]
        best_prec = df.loc[best_idx, "precision"]
        best_rec = df.loc[best_idx, "recall"]

        ax.axvline(x=best_thr, color=POSITIVE_COLOR, linestyle="--", linewidth=1.5,
                   label=f"Best Threshold = {best_thr:.2f}")

        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Score")
        ax.set_title(
            f"{title}\nBest Threshold = {best_thr:.2f}  |  "
            f"P={best_prec:.3f}, R={best_rec:.3f}, F1={best_f1:.3f}",
            pad=15, fontsize=13, color=COLOR_TEXT,
        )
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])

        fig.tight_layout()
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Saved threshold analysis to %s", save_path)
        return fig

    # ------------------------------------------------------------------
    # 7. Radar Chart
    # ------------------------------------------------------------------

    def plot_radar_chart(
        self,
        metrics_dict: Dict[str, float],
        save_path: str | Path,
        title: str = "Model Performance Radar",
        model_groups: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> plt.Figure:
        """
        Plot radar chart comparing metrics.

        Parameters
        ----------
        metrics_dict : dict
            Mapping of metric name to value (0-1 scale).
            Required keys: accuracy, precision, recall, f1, auc_roc
        save_path : str or Path
            Output file path for the PNG.
        title : str
            Chart title.
        model_groups : dict, optional
            Multiple model groups for comparison. Each maps metric->value.

        Returns
        -------
        matplotlib.figure.Figure
        """
        default_metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
        labels = []
        values = []

        for m in default_metrics:
            labels.append(m.replace("_", " ").title())
            values.append(metrics_dict.get(m, 0.0))

        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values_plot = values + [values[0]]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        if model_groups:
            # Plot multiple model groups
            colors = sns.color_palette("husl", n_colors=len(model_groups))
            for (name, mets), color in zip(model_groups.items(), colors):
                group_vals = [mets.get(m, 0.0) for m in default_metrics]
                group_vals_plot = group_vals + [group_vals[0]]
                ax.plot(angles, group_vals_plot, "o-", linewidth=2, label=name, color=color)
                ax.fill(angles, group_vals_plot, alpha=0.1, color=color)
        else:
            ax.plot(angles, values_plot, "o-", linewidth=2.5, color=COLOR_PRIMARY, label="Model")
            ax.fill(angles, values_plot, alpha=0.2, color=COLOR_PRIMARY)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim([0, 1.05])
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
        ax.set_title(title, pad=25, fontsize=14, color=COLOR_TEXT, fontweight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), framealpha=0.9)

        fig.tight_layout()
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Saved radar chart to %s", save_path)
        return fig

    # ------------------------------------------------------------------
    # 8. Learning Curve
    # ------------------------------------------------------------------

    def plot_learning_curve(
        self,
        history_dict: Dict[str, List[float]],
        save_path: str | Path,
        title: str = "Learning Curve",
        early_stopping_round: Optional[int] = None,
    ) -> plt.Figure:
        """
        Plot train loss vs validation loss over boosting rounds.

        Parameters
        ----------
        history_dict : dict
            Must contain 'train_loss' and 'val_loss' lists.
            Optionally 'early_stopping_round'.
        save_path : str or Path
            Output file path for the PNG.
        title : str
            Chart title.
        early_stopping_round : int, optional
            Round at which early stopping occurred.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        train_loss = history_dict.get("train_loss", [])
        val_loss = history_dict.get("val_loss", [])

        if not train_loss and not val_loss:
            ax.text(0.5, 0.5, "No training history available",
                    ha="center", va="center", fontsize=14, color=COLOR_TEXT)
            ax.set_title(title)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            return fig

        rounds = list(range(1, max(len(train_loss), len(val_loss)) + 1))

        if train_loss:
            ax.plot(rounds[:len(train_loss)], train_loss, "-",
                    color=COLOR_PRIMARY, linewidth=2, label="Train Loss")
        if val_loss:
            ax.plot(rounds[:len(val_loss)], val_loss, "-",
                    color=POSITIVE_COLOR, linewidth=2, label="Validation Loss")

        # Mark early stopping
        stop_round = early_stopping_round or history_dict.get("early_stopping_round")
        if stop_round is not None and stop_round > 0:
            min_val = min(val_loss[stop_round - 1], val_loss[min(stop_round, len(val_loss) - 1)]) if val_loss else 0
            ax.axvline(x=stop_round, color=COLOR_ACCENT, linestyle="--", linewidth=1.5)
            ax.plot(stop_round, val_loss[stop_round - 1] if stop_round <= len(val_loss) else 0,
                    "D", color=COLOR_ACCENT, markersize=10,
                    label=f"Early Stop: Round {stop_round}")

        ax.set_xlabel("Boosting Round")
        ax.set_ylabel("Loss (Binary Log-Loss)")
        ax.set_title(f"{title}\nTrain Loss vs Validation Loss",
                     pad=15, fontsize=13, color=COLOR_TEXT)
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Saved learning curve to %s", save_path)
        return fig

    # ------------------------------------------------------------------
    # 9. MCB/DSC (Miscalibration vs Discrimination)
    # ------------------------------------------------------------------

    def plot_mcb_dsc(
        self,
        mcb_dsc_data: Dict[str, Dict[str, float]],
        save_path: str | Path,
        title: str = "Miscalibration vs Discrimination",
    ) -> plt.Figure:
        """
        Plot Miscalibration vs Discrimination scatter with Brier score contours.

        Parameters
        ----------
        mcb_dsc_data : dict
            Maps condition/classifier name to {"mcb": float, "dsc": float}.
            mcb = miscalibration, dsc = discrimination (AUC-based).
        save_path : str or Path
            Output file path for the PNG.
        title : str
            Chart title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if not mcb_dsc_data:
            ax.text(0.5, 0.5, "No MCB/DSC data available",
                    ha="center", va="center", fontsize=14)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            return fig

        # Draw Brier score contour lines
        mcb_range = np.linspace(0, 0.5, 100)
        dsc_range = np.linspace(0, 1, 100)
        MCB, DSC = np.meshgrid(mcb_range, dsc_range)
        # Brier = MCB^2 + DSC * (1-DSC) / 2  (approximation)
        BRIER = MCB**2 + DSC * (1 - DSC) / 2

        contour = ax.contour(MCB, DSC, BRIER, levels=[0.05, 0.1, 0.15, 0.2, 0.25],
                             colors="gray", linewidths=0.8, alpha=0.5)
        ax.clabel(contour, inline=True, fontsize=8, fmt="Brier=%.2f")

        # Plot each condition
        colors = sns.color_palette("husl", n_colors=len(mcb_dsc_data))
        for (name, vals), color in zip(mcb_dsc_data.items(), colors):
            mcb = vals.get("mcb", 0.0)
            dsc = vals.get("dsc", 0.5)
            ax.scatter(mcb, dsc, s=120, color=color, edgecolors="black",
                       linewidth=1, zorder=5, label=name)
            ax.annotate(name, (mcb, dsc), textcoords="offset points",
                       xytext=(8, 5), fontsize=9, color=color)

        ax.set_xlabel("Miscalibration (MCB)")
        ax.set_ylabel("Discrimination (1 - AUC)")
        ax.set_title(title, pad=15, fontsize=14, color=COLOR_TEXT)
        ax.set_xlim([-0.02, 0.52])
        ax.set_ylim([-0.02, 1.02])
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Saved MCB/DSC plot to %s", save_path)
        return fig

    # ------------------------------------------------------------------
    # 10. Evaluation Dashboard (3x3 grid)
    # ------------------------------------------------------------------

    def create_evaluation_dashboard(
        self,
        results_dir: str | Path,
        model_groups_data: Dict[str, Dict[str, Any]],
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        """
        Create a 3x3 dashboard grid of all 9 chart types.

        Parameters
        ----------
        results_dir : str or Path
            Directory containing individual chart PNGs.
        model_groups_data : dict
            Mapping of condition name to its evaluation data dict.
            Each should contain y_true, y_score, feature_names, importance.
        save_path : str or Path, optional
            Output path for dashboard PNG. Defaults to results_dir/dashboard.png.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if save_path is None:
            save_path = Path(results_dir) / "dashboard.png"
        save_path = Path(save_path)

        dashboard_dpi = 200
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)

        # Select first condition for main dashboard
        first_cond = next(iter(model_groups_data)) if model_groups_data else None

        chart_titles = [
            "Confusion Matrix",
            "ROC Curve",
            "Precision-Recall Curve",
            "Calibration Curve",
            "Feature Importance",
            "Threshold Analysis",
            "Radar Chart",
            "Learning Curve",
            "MCB/DSC Analysis",
        ]

        for idx in range(9):
            ax = fig.add_subplot(gs[idx // 3, idx % 3])
            chart_file = Path(results_dir) / "charts" / f"{first_cond}_{chart_titles[idx].lower().replace(' ', '_')}.png"

            if chart_file.exists():
                try:
                    img = plt.imread(str(chart_file))
                    ax.imshow(img)
                    ax.axis("off")
                except Exception:
                    ax.text(0.5, 0.5, f"Could not load\n{chart_titles[idx]}",
                            ha="center", va="center", fontsize=11, color=COLOR_TEXT)
                    ax.set_title(chart_titles[idx], fontsize=12, color=COLOR_TEXT)
            else:
                # Generate placeholder with key metrics
                if first_cond and first_cond in model_groups_data:
                    data = model_groups_data[first_cond]
                    metrics = data.get("metrics", {})
                    ax.text(
                        0.5, 0.5,
                        f"{chart_titles[idx]}\n\n"
                        f"Accuracy: {metrics.get('accuracy', 0):.4f}\n"
                        f"AUC: {metrics.get('auc_roc', 0):.4f}\n"
                        f"F1: {metrics.get('f1', 0):.4f}",
                        ha="center", va="center", fontsize=11,
                        color=COLOR_TEXT, transform=ax.transAxes,
                    )
                else:
                    ax.text(0.5, 0.5, chart_titles[idx],
                            ha="center", va="center", fontsize=11, color=COLOR_TEXT)
                ax.set_title(chart_titles[idx], fontsize=12, color=COLOR_TEXT)

        fig.suptitle(
            "Health Monitor — Evaluation Dashboard",
            fontsize=18, fontweight="bold", color=COLOR_TEXT, y=0.98,
        )

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dashboard_dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Saved evaluation dashboard to %s", save_path)
        return fig
