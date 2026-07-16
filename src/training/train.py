"""
End-to-end training pipeline for the health symptom detection system.

Orchestrates the full workflow: data loading, window generation, feature
extraction, label generation, session splitting, model training,
evaluation, and ONNX export.  Designed for an RTX 2050 (4GB VRAM) with
conservative LightGBM GPU settings.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model groupings and hyper-parameters
# ---------------------------------------------------------------------------

# Condition grouping per model
MODEL_GROUPS: Dict[str, List[str]] = {
    "cardiac": ["tachycardia", "irregular_rhythm"],
    "respiratory": ["low_spo2", "fever"],
    "activity": ["fall_detected", "sleep_problem", "fatigue"],
}

# Default LightGBM parameters optimised for RTX 2050 (4GB VRAM).
# max_bin=63 is recommended for GPU training; num_leaves=31 keeps
# memory usage under ~300 MB peak.
DEFAULT_LGB_PARAMS: Dict[str, Any] = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "device": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
    "num_leaves": 31,
    "max_depth": 8,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "max_bin": 63,
    "gpu_use_dp": False,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
    "early_stopping_rounds": 50,
}

# Fallback CPU parameters (when GPU is unavailable)
DEFAULT_LGB_PARAMS_CPU: Dict[str, Any] = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "device": "cpu",
    "num_leaves": 31,
    "max_depth": 8,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
    "early_stopping_rounds": 50,
}


class TrainingPipeline:
    """End-to-end training pipeline for health symptom detection.

    Pipeline steps:

    1. Load session JSON files.
    2. Generate sliding windows per session.
    3. Extract features per window.
    4. Generate labels per window.
    5. Split sessions into train / validation / test.
    6. Train one LightGBM model per model group.
    7. Evaluate on the held-out test set.
    8. Export to ONNX for CPU deployment.

    Examples
    --------
    >>> pipeline = TrainingPipeline(seed=42)
    >>> results = pipeline.run_pipeline(
    ...     data_dir=Path("data/synthetic/raw"),
    ...     output_dir=Path("models"),
    ... )
    """

    def __init__(
        self,
        seed: int = 42,
        use_gpu: bool = True,
    ) -> None:
        """Initialise the pipeline.

        Parameters
        ----------
        seed : int
            Global random seed for reproducibility.
        use_gpu : bool
            If ``True``, use LightGBM GPU acceleration.  Falls back to CPU
            automatically if GPU is unavailable.
        """
        self.seed = seed
        self.use_gpu = use_gpu
        self._set_seeds(seed)

        # Lazy imports to keep module importable without all deps
        from ..data.loader import SensorDataLoader
        from ..data.preprocessor import SensorPreprocessor
        from ..data.window_generator import WindowGenerator
        from ..features.extractor import FeatureExtractor
        from ..data.label_generator import LabelGenerator

        self.loader = SensorDataLoader(strict=False)
        self.preprocessor = SensorPreprocessor()
        self.window_generator_cls = WindowGenerator
        self.feature_extractor = FeatureExtractor()
        self.label_generator = LabelGenerator()

    # ------------------------------------------------------------------
    # Seed management
    # ------------------------------------------------------------------

    @staticmethod
    def _set_seeds(seed: int) -> None:
        """Set random seeds for reproducibility across all libraries."""
        import random

        random.seed(seed)
        np.random.seed(seed)
        try:
            import lightgbm as lgb

            lgb.register_logger(logging.getLogger("lightgbm"))
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def load_sessions(
        self,
        data_dir: Union[str, Path],
    ) -> List[Dict[str, Any]]:
        """Load all session JSON files from a directory.

        Parameters
        ----------
        data_dir : str or Path
            Directory containing ``*.json`` session files.

        Returns
        -------
        list of dict
            Loaded session dictionaries.
        """
        data_dir = Path(data_dir)
        sessions = self.loader.load_dataset(data_dir, pattern="*.json")
        logger.info("Loaded %d sessions from %s", len(sessions), data_dir)
        return sessions

    def prepare_dataset(
        self,
        sessions: List[Dict[str, Any]],
        preprocess: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Generate windows, extract features, and produce labels.

        Parameters
        ----------
        sessions : list of dict
            Loaded session dictionaries.
        preprocess : bool
            Whether to run the preprocessing pipeline on each session.

        Returns
        -------
        tuple of (DataFrame, DataFrame, list)
            ``(features_df, labels_df, session_ids)``

            - ``features_df``: shape ``(n_windows, n_features)``
            - ``labels_df``: shape ``(n_windows, n_conditions)`` with
              one binary column per condition.
            - ``session_ids``: list of session IDs aligned row-by-row.
        """
        all_features: List[Dict[str, float]] = []
        all_labels: List[Dict[str, bool]] = []
        all_session_ids: List[str] = []

        for sess_idx, session in enumerate(sessions):
            session_id = session.get("session_id", f"session_{sess_idx}")

            # Optional preprocessing
            if preprocess:
                session = self.preprocessor.process_session(session)

            # Generate windows
            win_gen = self.window_generator_cls(session)
            windows = win_gen.generate_windows()

            for win in windows:
                # The WindowGenerator returns sensor_data keyed by channel
                # names matching the raw JSON schema.  The FeatureExtractor
                # expects a slightly different structure; adapt here.
                window_data = self._adapt_window_for_extractor(win, session)
                features = self.feature_extractor.extract_all(window_data)

                # Generate labels from raw sensor data
                label_dict = self.label_generator.generate_labels(win)
                labels = {
                    cond: info["detected"]
                    for cond, info in label_dict.items()
                }

                all_features.append(features)
                all_labels.append(labels)
                all_session_ids.append(session_id)

            if (sess_idx + 1) % 50 == 0:
                logger.info(
                    "Processed %d / %d sessions (%d windows so far)",
                    sess_idx + 1,
                    len(sessions),
                    len(all_features),
                )

        features_df = pd.DataFrame(all_features)
        labels_df = pd.DataFrame(all_labels).astype(int)

        logger.info(
            "Dataset prepared: %d windows, %d features, %d conditions",
            len(features_df),
            features_df.shape[1],
            labels_df.shape[1],
        )
        return features_df, labels_df, all_session_ids

    @staticmethod
    def _adapt_window_for_extractor(
        win: Dict[str, Any], session: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt a WindowGenerator output to the FeatureExtractor schema.

        The WindowGenerator stores sensor data as flat arrays keyed by
        channel name (``accelerometer``, ``heart_rate``, etc.).  The
        FeatureExtractor expects a nested dict with sub-keys.  This
        helper bridges the two.
        """
        sensor = win.get("sensor_data", {})
        meta = session.get("metadata", {})

        # Accelerometer: (n, 3) -> dict of ax, ay, az
        accel = sensor.get("accelerometer", np.array([]))
        if isinstance(accel, np.ndarray) and accel.ndim == 2 and accel.shape[1] >= 3:
            accel_dict = {"ax": accel[:, 0], "ay": accel[:, 1], "az": accel[:, 2]}
        else:
            accel_dict = {"ax": np.array([]), "ay": np.array([]), "az": np.array([])}

        # Gyroscope
        gyro = sensor.get("gyroscope", np.array([]))
        if isinstance(gyro, np.ndarray) and gyro.ndim == 2 and gyro.shape[1] >= 3:
            gyro_dict = {"gx": gyro[:, 0], "gy": gyro[:, 1], "gz": gyro[:, 2]}
        else:
            gyro_dict = {"gx": np.array([]), "gy": np.array([]), "gz": np.array([])}

        # Heart rate / SpO2 / PPG (1-D arrays)
        hr = sensor.get("heart_rate", np.array([]))
        spo2 = sensor.get("spo2", np.array([]))
        ppg = sensor.get("ppg", np.array([]))
        temp = sensor.get("temperature", np.array([]))

        hr_dict = {"bpm": hr, "spo2": spo2, "ppg_raw": ppg}
        temp_dict = {"stts22h_celsius": temp, "lm35_celsius": temp}

        # Metadata for the extractor
        sampling_cfg = session.get("sampling_config", {})
        ext_meta = {
            "activity_state": meta.get("activity_state", "resting"),
            "sampling_rate": float(sampling_cfg.get("accel_sample_rate_hz", 50)),
            "hr_sampling_rate": float(sampling_cfg.get("hr_sample_rate_hz", 25)),
            "ppg_sampling_rate": float(sampling_cfg.get("ppg_sample_rate_hz", 25)),
        }

        return {
            "accelerometer": accel_dict,
            "gyroscope": gyro_dict,
            "heart_rate": hr_dict,
            "temperature": temp_dict,
            "metadata": ext_meta,
        }

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def split_dataset(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        session_ids: List[str],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Dict[str, Any]:
        """Split dataset at the session level.

        Parameters
        ----------
        features_df : DataFrame
            Feature matrix.
        labels_df : DataFrame
            Binary label matrix.
        session_ids : list of str
            Session ID per row.
        train_ratio, val_ratio, test_ratio : float
            Split proportions (must sum to 1.0).

        Returns
        -------
        dict
            ``{
                "train": {"X": ndarray, "y": ndarray, "sessions": list},
                "val":   {"X": ndarray, "y": ndarray, "sessions": list},
                "test":  {"X": ndarray, "y": ndarray, "sessions": list},
                "feature_names": list[str],
            }``
        """
        from .splitter import SessionSplitter

        splitter = SessionSplitter()

        # Build session-level labels for stratification
        unique_sessions = list(dict.fromkeys(session_ids))  # preserve order
        session_labels: Dict[str, Dict[str, bool]] = {}
        for sid in unique_sessions:
            mask = np.array(session_ids) == sid
            session_labels[sid] = {
                col: bool(labels_df.loc[mask, col].any())
                for col in labels_df.columns
            }

        train_ids, val_ids, test_ids = splitter.stratified_split(
            sessions=unique_sessions,
            labels=session_labels,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=self.seed,
        )

        logger.info(splitter.split_stats(train_ids, val_ids, test_ids, session_labels))

        # Convert session IDs to row masks
        session_arr = np.array(session_ids)
        feature_names = list(features_df.columns)

        def _extract(ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
            mask = np.isin(session_arr, ids)
            return features_df.values[mask].astype(np.float32), labels_df.values[mask].astype(np.int64)

        X_train, y_train = _extract(train_ids)
        X_val, y_val = _extract(val_ids)
        X_test, y_test = _extract(test_ids)

        return {
            "train": {"X": X_train, "y": y_train, "sessions": train_ids},
            "val": {"X": X_val, "y": y_val, "sessions": val_ids},
            "test": {"X": X_test, "y": y_test, "sessions": test_ids},
            "feature_names": feature_names,
        }

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------

    def train_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        model_groups: Optional[Dict[str, List[str]]] = None,
        lgb_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Train one LightGBM model per model group.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix.
        y_train : np.ndarray
            Training labels, shape ``(n_samples, n_conditions)``.
        X_val : np.ndarray
            Validation feature matrix.
        y_val : np.ndarray
            Validation labels.
        feature_names : list of str
            Ordered feature names.
        model_groups : dict or None
            ``{model_name: [condition_names]}``.  Defaults to
            :data:`MODEL_GROUPS`.
        lgb_params : dict or None
            Override LightGBM hyper-parameters.

        Returns
        -------
        dict
            ``{model_name: {"model": lgb.Booster, "targets": list[str],
            "best_iteration": int, "feature_names": list[str]}}``
        """
        import lightgbm as lgb

        if model_groups is None:
            model_groups = MODEL_GROUPS

        params = dict(lgb_params or (DEFAULT_LGB_PARAMS if self.use_gpu else DEFAULT_LGB_PARAMS_CPU))

        # Pop non-booster params
        early_stopping_rounds = params.pop("early_stopping_rounds", 50)
        n_estimators = params.pop("n_estimators", 500)

        trained_models: Dict[str, Any] = {}

        for model_name, conditions in model_groups.items():
            logger.info(
                "Training model '%s' for conditions: %s",
                model_name,
                conditions,
            )

            # Determine column indices for this model's conditions
            label_cols = list(y_train.shape[1] if y_train.ndim == 2 else [0])
            condition_names = list(
                MODEL_GROUPS.get("cardiac", [])
                + MODEL_GROUPS.get("respiratory", [])
                + MODEL_GROUPS.get("activity", [])
            )

            if y_train.ndim == 2 and y_train.shape[1] > 1:
                # Multi-label: extract columns for this model's conditions
                cond_indices = [
                    i for i, c in enumerate(condition_names) if c in conditions
                ]
                y_train_model = y_train[:, cond_indices]
                y_val_model = y_val[:, cond_indices]
            else:
                y_train_model = y_train.ravel()
                y_val_model = y_val.ravel()

            # For binary: use first column if multi-dimensional
            if y_train_model.ndim == 2:
                # Multi-output: train one model per condition, return list
                # Actually, for simplicity, use binary relevance: if ANY
                # condition in the group is positive, label is 1.
                y_train_binary = (y_train_model.sum(axis=1) > 0).astype(int)
                y_val_binary = (y_val_model.sum(axis=1) > 0).astype(int)
            else:
                y_train_binary = y_train_model.astype(int)
                y_val_binary = y_val_model.astype(int)

            train_data = lgb.Dataset(X_train, label=y_train_binary)
            val_data = lgb.Dataset(X_val, label=y_val_binary, reference=train_data)

            callbacks = [
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(period=100),
            ]

            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=n_estimators,
                callbacks=callbacks,
            )

            best_iter = model.best_iteration if hasattr(model, "best_iteration") else n_estimators
            trained_models[model_name] = {
                "model": model,
                "targets": conditions,
                "best_iteration": best_iter,
                "feature_names": feature_names,
            }

            logger.info(
                "Model '%s' trained: best_iteration=%d, "
                "best_auc=%.4f",
                model_name,
                best_iter,
                model.best_score.get("valid_0", {}).get("auc", 0.0)
                if hasattr(model, "best_score")
                else 0.0,
            )

        return trained_models

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_models(
        self,
        trained_models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        condition_names: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate all trained models on the test set.

        Parameters
        ----------
        trained_models : dict
            Output of :meth:`train_models`.
        X_test : np.ndarray
            Test feature matrix.
        y_test : np.ndarray
            Test labels.
        condition_names : list of str
            All condition names (columns of *y_test*).

        Returns
        -------
        dict
            ``{model_name: {"metrics": dict, "predictions": ndarray,
            "thresholds": dict}}``
        """
        from .evaluator import ModelEvaluator

        evaluator = ModelEvaluator()
        results: Dict[str, Dict[str, Any]] = {}

        for model_name, info in trained_models.items():
            model = info["model"]
            targets = info["targets"]

            # Predict probabilities
            y_score = model.predict(X_test)

            # For multi-output models, y_score may be 2-D
            if y_score.ndim == 2 and y_score.shape[1] > 1:
                y_pred_binary = (y_score.sum(axis=1) > 0).astype(int)
                y_score_binary = y_score.max(axis=1)
            else:
                y_score_ravel = y_score.ravel()
                # Find optimal threshold
                threshold = evaluator.find_optimal_threshold(
                    y_test.ravel(), y_score_ravel, metric="f1"
                )
                y_pred_binary = (y_score_ravel >= threshold).astype(int)
                y_score_binary = y_score_ravel

            metrics = evaluator.compute_metrics(
                y_test.ravel().astype(int),
                y_pred_binary,
                y_score_binary,
            )

            # Per-condition metrics if multi-label
            per_cond: Dict[str, Dict[str, float]] = {}
            if y_test.ndim == 2 and y_test.shape[1] > 1:
                for i, cond in enumerate(condition_names):
                    if i < y_test.shape[1]:
                        per_cond[cond] = evaluator.compute_metrics(
                            y_test[:, i],
                            (y_score[:, i] >= 0.5).astype(int)
                            if y_score.ndim == 2
                            else y_pred_binary,
                            y_score[:, i] if y_score.ndim == 2 else y_score_binary,
                        )

            results[model_name] = {
                "metrics": metrics,
                "per_condition": per_cond,
                "targets": targets,
                "n_test_samples": len(y_test),
            }

            logger.info(
                "Model '%s' test metrics: accuracy=%.4f, f1=%.4f, "
                "auc_roc=%.4f, mcc=%.4f",
                model_name,
                metrics["accuracy"],
                metrics["f1_score"],
                metrics.get("auc_roc", 0.0),
                metrics["mcc"],
            )

        return results

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_models(
        self,
        trained_models: Dict[str, Any],
        output_dir: Union[str, Path],
        feature_names: List[str],
    ) -> Dict[str, Path]:
        """Export all trained models to ONNX.

        Parameters
        ----------
        trained_models : dict
            Output of :meth:`train_models`.
        output_dir : str or Path
            Directory for ONNX files and metadata.
        feature_names : list of str
            Ordered feature names.

        Returns
        -------
        dict
            ``{model_name: Path}`` mapping to ONNX files.
        """
        from .exporter import ModelExporter

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exporter = ModelExporter()

        models_for_export = {
            name: info["model"] for name, info in trained_models.items()
        }

        onnx_paths = exporter.export_all_models(
            models_for_export, output_dir, feature_names
        )

        # Save feature names
        feature_names_path = output_dir / "feature_names.json"
        with open(feature_names_path, "w", encoding="utf-8") as fh:
            json.dump(feature_names, fh, indent=2)
        logger.info("Saved feature names to %s", feature_names_path)

        # Verify each exported model
        for name, path in onnx_paths.items():
            if X_test_sample := np.random.default_rng(42).random(
                (1, len(feature_names))
            ).astype(np.float32):
                try:
                    exporter.verify_onnx(path, X_test_sample)
                except Exception as exc:
                    logger.warning(
                        "ONNX verification failed for '%s': %s", name, exc
                    )

        # Check HF Spaces size limit
        within_limit, size_report = exporter.check_hf_spaces_limit(onnx_paths)
        if not within_limit:
            logger.warning(
                "Total model size exceeds HF Spaces limit! Report: %s",
                json.dumps(size_report, indent=2),
            )

        return onnx_paths

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        data_dir: Union[str, Path],
        output_dir: Union[str, Path],
        num_sessions: Optional[int] = None,
        force_regenerate: bool = False,
        skip_training: bool = False,
        skip_export: bool = False,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        preprocess: bool = True,
        model_groups: Optional[Dict[str, List[str]]] = None,
        lgb_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the full end-to-end training pipeline.

        Parameters
        ----------
        data_dir : str or Path
            Directory containing session JSON files.
        output_dir : str or Path
            Directory to save models and artifacts.
        num_sessions : int or None
            If provided, generate synthetic data up to this count when
            the data directory is empty.
        force_regenerate : bool
            If ``True``, regenerate synthetic data even if files exist.
        skip_training : bool
            If ``True``, skip model training (useful for testing data prep).
        skip_export : bool
            If ``True``, skip ONNX export.
        train_ratio, val_ratio, test_ratio : float
            Split proportions.
        preprocess : bool
            Whether to preprocess sensor data.
        model_groups : dict or None
            Model-to-condition mapping.
        lgb_params : dict or None
            LightGBM parameter overrides.

        Returns
        -------
        dict
            Comprehensive results dictionary with keys:

            - ``"n_sessions"`` -- total sessions loaded
            - ``"n_windows"`` -- total windows extracted
            - ``"n_features"`` -- number of features
            - ``"feature_names"`` -- list of feature names
            - ``"split_sizes"`` -- {train, val, test} sample counts
            - ``"models"`` -- per-model training info
            - ``"evaluation"`` -- per-model test metrics
            - ``"onnx_paths"`` -- exported model paths
            - ``"elapsed_seconds"`` -- total wall-clock time
        """
        t_start = time.time()
        output_dir = Path(output_dir)
        data_dir = Path(data_dir)

        results: Dict[str, Any] = {}

        # Step 1: Load sessions
        logger.info("=" * 60)
        logger.info("STEP 1: Loading sessions from %s", data_dir)
        logger.info("=" * 60)
        sessions = self.load_sessions(data_dir)

        if len(sessions) == 0 and num_sessions is not None and num_sessions > 0:
            logger.info("No sessions found; generating %d synthetic sessions...", num_sessions)
            from data.synthetic.generator import SyntheticDataGenerator

            gen = SyntheticDataGenerator(seed=self.seed)
            gen.generate_dataset(
                num_sessions=num_sessions,
                output_dir=str(data_dir),
            )
            sessions = self.load_sessions(data_dir)

        results["n_sessions"] = len(sessions)

        # Step 2-4: Prepare dataset (windows + features + labels)
        logger.info("=" * 60)
        logger.info("STEP 2-4: Preparing dataset (windows, features, labels)")
        logger.info("=" * 60)
        features_df, labels_df, session_ids = self.prepare_dataset(
            sessions, preprocess=preprocess
        )
        results["n_windows"] = len(features_df)
        results["n_features"] = features_df.shape[1]

        # Drop columns that are all-zero (no signal)
        non_zero_cols = features_df.columns[features_df.sum(axis=0) != 0]
        if len(non_zero_cols) < features_df.shape[1]:
            dropped = features_df.shape[1] - len(non_zero_cols)
            logger.info("Dropping %d all-zero feature columns", dropped)
            features_df = features_df[non_zero_cols]

        # Fill any remaining NaN with 0
        features_df = features_df.fillna(0.0)

        feature_names = list(features_df.columns)
        results["feature_names"] = feature_names

        # Step 5: Split
        logger.info("=" * 60)
        logger.info("STEP 5: Splitting dataset")
        logger.info("=" * 60)
        split = self.split_dataset(
            features_df, labels_df, session_ids,
            train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
        )

        X_train, y_train = split["train"]["X"], split["train"]["y"]
        X_val, y_val = split["val"]["X"], split["val"]["y"]
        X_test, y_test = split["test"]["X"], split["test"]["y"]

        results["split_sizes"] = {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        }

        # Determine condition names from labels_df columns
        condition_names = list(labels_df.columns)
        results["condition_names"] = condition_names

        if skip_training:
            logger.info("Skipping training (--skip-training flag).")
            results["models"] = {}
            results["evaluation"] = {}
            results["onnx_paths"] = {}
            results["elapsed_seconds"] = time.time() - t_start
            return results

        # Step 6: Train
        logger.info("=" * 60)
        logger.info("STEP 6: Training models")
        logger.info("=" * 60)
        trained_models = self.train_models(
            X_train, y_train, X_val, y_val,
            feature_names=feature_names,
            model_groups=model_groups,
            lgb_params=lgb_params,
        )

        results["models"] = {
            name: {
                "targets": info["targets"],
                "best_iteration": info["best_iteration"],
            }
            for name, info in trained_models.items()
        }

        # Step 7: Evaluate
        logger.info("=" * 60)
        logger.info("STEP 7: Evaluating models on test set")
        logger.info("=" * 60)
        evaluation = self.evaluate_models(
            trained_models, X_test, y_test, condition_names
        )
        results["evaluation"] = evaluation

        # Step 8: Export
        if not skip_export:
            logger.info("=" * 60)
            logger.info("STEP 8: Exporting models to ONNX")
            logger.info("=" * 60)
            onnx_paths = self.export_models(
                trained_models, output_dir, feature_names
            )
            results["onnx_paths"] = {k: str(v) for k, v in onnx_paths.items()}
        else:
            logger.info("Skipping export (--skip-export flag).")
            results["onnx_paths"] = {}

        # Save summary
        elapsed = time.time() - t_start
        results["elapsed_seconds"] = round(elapsed, 2)

        summary_path = output_dir / "training_summary.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        # Make summary JSON-serialisable
        serialisable = self._make_json_serialisable(results)
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=2)
        logger.info("Saved training summary to %s", summary_path)

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE in %.1f seconds", elapsed)
        logger.info(
            "Sessions: %d | Windows: %d | Features: %d",
            results["n_sessions"],
            results["n_windows"],
            results["n_features"],
        )
        for name, eval_info in evaluation.items():
            m = eval_info["metrics"]
            logger.info(
                "  %s: accuracy=%.3f, f1=%.3f, auc=%.3f, mcc=%.3f",
                name, m["accuracy"], m["f1_score"],
                m.get("auc_roc", 0.0), m["mcc"],
            )
        logger.info("=" * 60)

        return results

    @staticmethod
    def _make_json_serialisable(obj: Any) -> Any:
        """Recursively convert numpy types for JSON serialisation."""
        if isinstance(obj, dict):
            return {k: TrainingPipeline._make_json_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [TrainingPipeline._make_json_serialisable(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
