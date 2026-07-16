"""
LightGBM multi-output model for activity-related condition detection.

Predicts: fall_detected, sleep_problem, fatigue

Architecture: Three independent LightGBM binary classifiers (one per condition)
internally managed as a single ActivityModel unit.  Each classifier uses
motion features, HRV features, and cross-sensor features.

Fall detection is primarily rule-based but the ML component reduces false
positives.  Sleep problems require long-term pattern features encoded in
the feature vector.  Fatigue is the hardest condition, relying heavily
on HRV and cross-sensor coupling.

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
)

from ..config import RANDOM_SEED

logger = logging.getLogger(__name__)

# Condition labels handled by this model (order matters for output arrays)
CONDITION_NAMES: List[str] = ["fall_detected", "sleep_problem", "fatigue"]


class ActivityModel:
    """Multi-output LightGBM model for activity-related condition detection.

    Internally holds three independent ``lgb.Booster`` instances — one for
    each of *fall_detected*, *sleep_problem*, and *fatigue*.  Each booster
    is trained as a binary classification task with ``objective='binary'``.

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
        self.models: Dict[str, lgb.Booster] = {}
        self.feature_names: Optional[List[str]] = None
        self._train_history: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # LightGBM parameter helpers
    # ------------------------------------------------------------------

    def _base_params(self) -> Dict[str, Any]:
        """Common LightGBM hyper-parameters for activity conditions."""
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
    ) -> ActivityModel:
        """Train all three activity sub-models.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix, shape ``(n_samples, n_features)``.
        y_train : np.ndarray
            Training labels, shape ``(n_samples, 3)`` where columns
            correspond to ``CONDITION_NAMES``.
        X_val : np.ndarray
            Validation feature matrix.
        y_val : np.ndarray
            Validation labels, shape ``(n_samples, 3)``.
        feature_names : list[str], optional
            Human-readable feature names.
        num_boost_round : int
            Maximum boosting rounds.
        early_stopping_rounds : int
            Patience for early stopping.

        Returns
        -------
        ActivityModel
            ``self`` (trained).
        """
        if y_train.ndim != 2 or y_train.shape[1] != 3:
            raise ValueError(
                f"y_train must have shape (n, 3); got {y_train.shape}"
            )

        self.feature_names = feature_names or [
            f"feature_{i}" for i in range(X_train.shape[1])
        ]

        params = self._base_params()
        callbacks = [
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100),
        ]

        for idx, cond_name in enumerate(CONDITION_NAMES):
            logger.info("Training activity sub-model: %s", cond_name)

            dtrain = lgb.Dataset(
                X_train,
                label=y_train[:, idx],
                feature_name=self.feature_names,
                free_raw_data=False,
            )
            dval = lgb.Dataset(
                X_val,
                label=y_val[:, idx],
                feature_name=self.feature_names,
                free_raw_data=False,
            )

            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                valid_sets=[dval],
                valid_names=["val"],
                callbacks=callbacks,
            )

            self.models[cond_name] = booster

            # Validation metrics
            val_pred = booster.predict(X_val)
            val_binary = (val_pred >= 0.5).astype(int)
            val_label = y_val[:, idx]

            auc = roc_auc_score(val_label, val_pred) if len(np.unique(val_label)) > 1 else 0.0
            f1 = f1_score(val_label, val_binary, zero_division=0)

            logger.info(
                "  %s — AUC: %.4f  F1: %.4f  best_iter: %d",
                cond_name,
                auc,
                f1,
                booster.best_iteration,
            )

            self._train_history[cond_name] = {
                "auc": float(auc),
                "f1": float(f1),
                "best_iteration": booster.best_iteration,
                "num_features": X_train.shape[1],
                "num_train_samples": X_train.shape[0],
                "num_val_samples": X_val.shape[0],
            }

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for all three conditions.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Probability array, shape ``(n_samples, 3)``.
        """
        self._check_trained()

        preds = np.zeros((X.shape[0], len(CONDITION_NAMES)), dtype=np.float64)
        for idx, cond_name in enumerate(CONDITION_NAMES):
            preds[:, idx] = self.models[cond_name].predict(X)

        return preds

    def predict_binary(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels for all three conditions.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        threshold : float
            Decision threshold (default 0.5).

        Returns
        -------
        np.ndarray
            Binary label array, shape ``(n_samples, 3)``.
        """
        probs = self.predict(X)
        return (probs >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> Dict[str, float]:
        """Aggregate feature importance across all three sub-models.

        Uses ``'gain'`` importance and averages across conditions.

        Returns
        -------
        dict[str, float]
            Mapping from feature name to averaged importance score.
        """
        self._check_trained()

        if self.feature_names is None:
            return {}

        importance_sums: Dict[str, float] = {n: 0.0 for n in self.feature_names}
        count = 0

        for cond_name in CONDITION_NAMES:
            booster = self.models[cond_name]
            imp = booster.feature_importance(importance_type="gain")
            for fname, val in zip(self.feature_names, imp):
                importance_sums[fname] += float(val)
            count += 1

        if count > 0:
            for fname in importance_sums:
                importance_sums[fname] /= count

        return dict(sorted(importance_sums.items(), key=lambda kv: kv[1], reverse=True))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Evaluate all three sub-models on a test set.

        Parameters
        ----------
        X : np.ndarray
            Test feature matrix.
        y : np.ndarray
            Test labels, shape ``(n_samples, 3)``.
        threshold : float
            Decision threshold.

        Returns
        -------
        dict
            Per-condition metrics and overall summary.
        """
        self._check_trained()

        preds_prob = self.predict(X)
        preds_bin = (preds_prob >= threshold).astype(int)

        report: Dict[str, Any] = {}
        for idx, cond_name in enumerate(CONDITION_NAMES):
            y_true = y[:, idx]
            y_pred = preds_bin[:, idx]
            y_prob = preds_prob[:, idx]

            has_positive = np.sum(y_true) > 0
            has_negative = np.sum(y_true == 0) > 0
            both_classes = has_positive and has_negative

            report[cond_name] = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "auc_roc": float(roc_auc_score(y_true, y_prob)) if both_classes else 0.0,
                "support_true": int(np.sum(y_true)),
                "support_false": int(np.sum(y_true == 0)),
            }

        flat_true = y.ravel()
        flat_pred = preds_bin.ravel()
        report["overall"] = {
            "accuracy": float(accuracy_score(flat_true, flat_pred)),
            "macro_f1": float(f1_score(flat_true, flat_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(flat_true, flat_pred, average="weighted", zero_division=0)),
        }

        return report

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the model to disk.

        Creates ``<path>.joblib`` (boosters) and ``<path>.meta.json``
        (metadata).
        """
        self._check_trained()

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        model_path = dest.with_suffix(".joblib")
        joblib.dump(self.models, model_path)
        logger.info("Saved activity model to %s", model_path)

        meta = {
            "feature_names": self.feature_names,
            "condition_names": CONDITION_NAMES,
            "train_history": self._train_history,
            "use_gpu": self.use_gpu,
            "random_seed": self.random_seed,
        }
        meta_path = dest.with_suffix(".meta.json")
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        logger.info("Saved activity model metadata to %s", meta_path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        use_gpu: bool = False,
    ) -> ActivityModel:
        """Load a saved model from disk.

        Parameters
        ----------
        path : str or Path
            Path prefix used during ``save()``.
        use_gpu : bool
            GPU flag for the returned instance.

        Returns
        -------
        ActivityModel
        """
        dest = Path(path)
        model_path = dest.with_suffix(".joblib")
        meta_path = dest.with_suffix(".meta.json")

        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)

        instance = cls(use_gpu=use_gpu, random_seed=meta.get("random_seed", RANDOM_SEED))
        instance.models = joblib.load(model_path)
        instance.feature_names = meta.get("feature_names")
        instance._train_history = meta.get("train_history", {})

        logger.info("Loaded activity model from %s", model_path)
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_trained(self) -> None:
        if not self.models:
            raise RuntimeError(
                "ActivityModel has not been trained yet. "
                "Call .train() or .load() first."
            )

    def describe(self) -> str:
        """Human-readable summary."""
        lines = [
            "ActivityModel",
            f"  Conditions  : {CONDITION_NAMES}",
            f"  Sub-models  : {len(self.models)}",
            f"  Features    : {len(self.feature_names) if self.feature_names else 'N/A'}",
            f"  GPU         : {self.use_gpu}",
            f"  Trained     : {bool(self.models)}",
        ]
        if self._train_history:
            for cond, info in self._train_history.items():
                lines.append(
                    f"  {cond}: AUC={info['auc']:.4f}  F1={info['f1']:.4f}  "
                    f"best_iter={info['best_iteration']}"
                )
        return "\n".join(lines)
