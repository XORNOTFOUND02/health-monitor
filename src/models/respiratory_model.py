"""
LightGBM binary classifier for respiratory condition detection.

Predicts: low_spo2

Uses SpO2 features, heart-rate features, and frequency-domain features.
While SpO2 < 95% is detectable via simple thresholding, the ML model
captures subtle desaturation patterns and HR-SpO2 coupling that rules miss.

GPU configuration targets an RTX 2050 with 4 GB VRAM:
    device_type='gpu', gpu_use_dp=False, max_bin=63, num_leaves=31
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

from ..config import RANDOM_SEED

logger = logging.getLogger(__name__)

CONDITION_NAME: str = "low_spo2"


class RespiratoryModel:
    """LightGBM binary classifier for respiratory condition detection.

    Internally holds a single ``lgb.Booster`` for *low_spo2*.

    Parameters
    ----------
    use_gpu : bool
        Enable GPU acceleration (default ``True``).
    random_seed : int
        Seed for reproducibility.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        use_gpu: bool = True,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        self.use_gpu = use_gpu
        self.random_seed = random_seed
        self.model: Optional[lgb.Booster] = None
        self.feature_names: Optional[List[str]] = None
        self._train_history: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # LightGBM parameter helpers
    # ------------------------------------------------------------------

    def _base_params(self) -> Dict[str, Any]:
        """Common LightGBM hyper-parameters for respiratory detection."""
        params: Dict[str, Any] = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 31,
            "max_bin": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "verbose": -1,
            "seed": self.random_seed,
            "feature_fraction_seed": self.random_seed,
            "bagging_seed": self.random_seed,
            "is_unbalance": True,
        }

        if self.use_gpu:
            params.update(
                {
                    "device_type": "gpu",
                    "gpu_platform_id": 0,
                    "gpu_device_id": 0,
                    "gpu_use_dp": False,
                }
            )

        return params

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
    ) -> RespiratoryModel:
        """Train the respiratory model.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix, shape ``(n_samples, n_features)``.
        y_train : np.ndarray
            Training labels, shape ``(n_samples,)`` — binary (0/1).
        X_val : np.ndarray
            Validation feature matrix.
        y_val : np.ndarray
            Validation labels.
        feature_names : list[str], optional
            Feature names for importance reporting.
        num_boost_round : int
            Maximum boosting rounds.
        early_stopping_rounds : int
            Patience for early stopping.

        Returns
        -------
        RespiratoryModel
            ``self`` (trained).
        """
        if y_train.ndim != 1:
            raise ValueError(
                f"y_train must be 1-D; got shape {y_train.shape}"
            )

        self.feature_names = feature_names or [
            f"feature_{i}" for i in range(X_train.shape[1])
        ]

        params = self._base_params()
        callbacks = [
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100),
        ]

        dtrain = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=self.feature_names,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            X_val,
            label=y_val,
            feature_name=self.feature_names,
            free_raw_data=False,
        )

        logger.info("Training respiratory model (low_spo2)")

        self.model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=callbacks,
        )

        # Validation metrics
        val_pred = self.model.predict(X_val)
        val_binary = (val_pred >= 0.5).astype(int)

        auc = roc_auc_score(y_val, val_pred) if len(np.unique(y_val)) > 1 else 0.0
        f1 = f1_score(y_val, val_binary, zero_division=0)

        logger.info(
            "  %s — AUC: %.4f  F1: %.4f  best_iter: %d",
            CONDITION_NAME,
            auc,
            f1,
            self.model.best_iteration,
        )

        self._train_history = {
            "auc": float(auc),
            "f1": float(f1),
            "best_iteration": self.model.best_iteration,
            "num_features": X_train.shape[1],
            "num_train_samples": X_train.shape[0],
            "num_val_samples": X_val.shape[0],
        }

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for low_spo2.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            1-D probability array.
        """
        self._check_trained()
        return self.model.predict(X)

    def predict_binary(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels for low_spo2.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        threshold : float
            Decision threshold (default 0.5).

        Returns
        -------
        np.ndarray
            1-D binary label array.
        """
        probs = self.predict(X)
        return (probs >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance (gain-based) for the single booster.

        Returns
        -------
        dict[str, float]
            Mapping from feature name to importance score, sorted
            descending.
        """
        self._check_trained()

        if self.feature_names is None:
            return {}

        imp = self.model.feature_importance(importance_type="gain")
        importance = {fname: float(val) for fname, val in zip(self.feature_names, imp)}
        return dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Evaluate the model on a test set.

        Returns a dictionary with accuracy, precision, recall, F1,
        AUC-ROC, and a confusion matrix.
        """
        self._check_trained()

        y_prob = self.predict(X)
        y_pred = (y_prob >= threshold).astype(int)

        has_positive = np.sum(y) > 0
        has_negative = np.sum(y == 0) > 0
        both_classes = has_positive and has_negative

        cm = confusion_matrix(y, y_pred).tolist()

        report: Dict[str, Any] = {
            "condition": CONDITION_NAME,
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y, y_prob)) if both_classes else 0.0,
            "confusion_matrix": cm,
            "support_true": int(np.sum(y)),
            "support_false": int(np.sum(y == 0)),
        }

        return report

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the model to disk.

        Creates ``<path>.joblib`` (booster) and ``<path>.meta.json``
        (metadata).
        """
        self._check_trained()

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        model_path = dest.with_suffix(".joblib")
        joblib.dump(self.model, model_path)
        logger.info("Saved respiratory model to %s", model_path)

        meta = {
            "feature_names": self.feature_names,
            "condition_name": CONDITION_NAME,
            "train_history": self._train_history,
            "use_gpu": self.use_gpu,
            "random_seed": self.random_seed,
        }
        meta_path = dest.with_suffix(".meta.json")
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        logger.info("Saved respiratory model metadata to %s", meta_path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        use_gpu: bool = False,
    ) -> RespiratoryModel:
        """Load a saved model from disk.

        Parameters
        ----------
        path : str or Path
            Path prefix used during ``save()``.
        use_gpu : bool
            GPU flag for the returned instance.

        Returns
        -------
        RespiratoryModel
        """
        dest = Path(path)
        model_path = dest.with_suffix(".joblib")
        meta_path = dest.with_suffix(".meta.json")

        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)

        instance = cls(use_gpu=use_gpu, random_seed=meta.get("random_seed", RANDOM_SEED))
        instance.model = joblib.load(model_path)
        instance.feature_names = meta.get("feature_names")
        instance._train_history = meta.get("train_history", {})

        logger.info("Loaded respiratory model from %s", model_path)
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_trained(self) -> None:
        if self.model is None:
            raise RuntimeError(
                "RespiratoryModel has not been trained yet. "
                "Call .train() or .load() first."
            )

    def describe(self) -> str:
        """Human-readable summary."""
        lines = [
            "RespiratoryModel",
            f"  Condition   : {CONDITION_NAME}",
            f"  Features    : {len(self.feature_names) if self.feature_names else 'N/A'}",
            f"  GPU         : {self.use_gpu}",
            f"  Trained     : {self.model is not None}",
        ]
        if self._train_history:
            lines.append(
                f"  Metrics     : AUC={self._train_history['auc']:.4f}  "
                f"F1={self._train_history['f1']:.4f}  "
                f"best_iter={self._train_history['best_iteration']}"
            )
        return "\n".join(lines)
