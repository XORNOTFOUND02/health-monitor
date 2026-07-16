"""
ONNX model exporter for CPU deployment.

Converts trained LightGBM models to ONNX format for efficient CPU
inference on Hugging Face Spaces.  Handles both ``lgb.Booster`` and
scikit-learn-style ``LGBMClassifier``/``LGBMRegressor`` objects.

The exported ONNX models must stay under the HF Spaces 512 MB artifact
limit, so this module also reports model sizes for monitoring.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def _is_lgb_booster(model: Any) -> bool:
    """Return True if *model* is a ``lightgbm.Booster``."""
    try:
        import lightgbm as lgb

        return isinstance(model, lgb.Booster)
    except ImportError:
        return False


def _is_lgb_sklearn_model(model: Any) -> bool:
    """Return True if *model* is an scikit-learn-style LightGBM model."""
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        return isinstance(model, (LGBMClassifier, LGBMRegressor))
    except ImportError:
        return False


class ModelExporter:
    """Export trained LightGBM models to ONNX format.

    The conversion pipeline:

    1. Accept a ``lgb.Booster`` or scikit-learn LightGBM model.
    2. Convert via ``onnxmltools`` with explicit feature-name metadata.
    3. Verify the exported ONNX model by running sample inference with
       ``onnxruntime``.
    4. Report model file sizes (important for HF Spaces deployment).

    Examples
    --------
    >>> exporter = ModelExporter()
    >>> path = exporter.export_to_onnx(
    ...     model, "models/cardiac.onnx", feature_names, "cardiac"
    ... )
    >>> size = exporter.get_model_size(path)
    """

    def export_to_onnx(
        self,
        model: Any,
        output_path: Union[str, Path],
        feature_names: List[str],
        model_name: Optional[str] = None,
    ) -> Path:
        """Export a single LightGBM model to ONNX.

        Parameters
        ----------
        model : lightgbm.Booster or LGBMClassifier/LGBMRegressor
            Trained model to export.
        output_path : str or Path
            Destination path for the ``.onnx`` file.
        feature_names : list of str
            Ordered feature names matching the training data columns.
        model_name : str or None
            Human-readable name for logging.

        Returns
        -------
        pathlib.Path
            Absolute path to the exported ``.onnx`` file.

        Raises
        ------
        ValueError
            If the model type is not supported.
        RuntimeError
            If conversion fails.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        name = model_name or output_path.stem
        n_features = len(feature_names)

        logger.info(
            "Exporting model '%s' to ONNX (%d features) -> %s",
            name,
            n_features,
            output_path,
        )

        # Convert LightGBM Booster to sklearn LGBMClassifier for ONNX compatibility
        try:
            import lightgbm as lgb
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError as exc:
            raise RuntimeError(
                "lightgbm and skl2onnx are required for ONNX export. "
                "Install with: pip install lightgbm skl2onnx"
            ) from exc

        initial_types = [("input", FloatTensorType([None, n_features]))]

        # Ensure we have an sklearn-compatible model
        if _is_lgb_booster(model):
            # Wrap booster inside an LGBMClassifier
            sklearn_model = lgb.LGBMClassifier(n_estimators=1, max_depth=-1, num_leaves=31)
            sklearn_model._Booster = model
            sklearn_model._n_features = n_features
            sklearn_model._n_classes = 2
            sklearn_model._fitted = True
            sklearn_model._class_map = {0: 0, 1: 1}
            sklearn_model._le = None
            model = sklearn_model
        elif _is_lgb_sklearn_model(model):
            pass  # Already in sklearn format
        else:
            raise ValueError(
                f"Unsupported model type for ONNX export: {type(model)}"
            )

        try:
            onnx_model = convert_sklearn(
                model,
                initial_types=initial_types,
                name=name,
                target_opset=12,
            )
        except Exception as exc:
            raise RuntimeError(
                f"ONNX conversion failed for model '{name}': {exc}"
            ) from exc

        # Write to disk
        with open(output_path, "wb") as fh:
            fh.write(onnx_model.SerializeToString())

        size = output_path.stat().st_size
        logger.info(
            "Exported '%s': %d bytes (%.1f KB)",
            name,
            size,
            size / 1024,
        )

        return output_path.resolve()

    def export_all_models(
        self,
        models_dict: Dict[str, Any],
        output_dir: Union[str, Path],
        feature_names: List[str],
    ) -> Dict[str, Path]:
        """Export multiple models to ONNX in a single directory.

        Parameters
        ----------
        models_dict : dict
            ``{model_name: trained_model}`` mapping.
        output_dir : str or Path
            Directory to write ``<model_name>.onnx`` files.
        feature_names : list of str
            Ordered feature names.

        Returns
        -------
        dict
            ``{model_name: Path}`` mapping to exported ONNX files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exported: Dict[str, Path] = {}
        for model_name, model in models_dict.items():
            onnx_path = output_dir / f"{model_name}.onnx"
            try:
                path = self.export_to_onnx(
                    model, onnx_path, feature_names, model_name
                )
                exported[model_name] = path
            except (RuntimeError, ValueError) as exc:
                logger.error(
                    "Failed to export model '%s': %s", model_name, exc
                )
                raise

        # Report total size
        total_bytes = sum(p.stat().st_size for p in exported.values())
        logger.info(
            "Exported %d models, total size: %d bytes (%.1f KB)",
            len(exported),
            total_bytes,
            total_bytes / 1024,
        )

        return exported

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_onnx(
        self,
        onnx_path: Union[str, Path],
        sample_input: np.ndarray,
    ) -> Dict[str, Any]:
        """Verify an ONNX model by running sample inference.

        Parameters
        ----------
        onnx_path : str or Path
            Path to the ``.onnx`` file.
        sample_input : np.ndarray
            Sample feature array of shape ``(n_samples, n_features)`` or
            ``(n_features,)``.

        Returns
        -------
        dict
            Dictionary with keys:

            - ``input_shape`` -- shape of the input passed to the model
            - ``output_shape`` -- shape of the model output
            - ``output_sample`` -- first few values of the output (for logging)
            - ``inference_time_ms`` -- wall-clock time for the inference call
            - ``status`` -- ``"ok"`` or ``"error"``

        Raises
        ------
        RuntimeError
            If ONNX Runtime cannot load the model or inference fails.
        """
        import time

        import onnxruntime as ort

        onnx_path = Path(onnx_path)
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        sample_input = np.asarray(sample_input, dtype=np.float32)
        if sample_input.ndim == 1:
            sample_input = sample_input.reshape(1, -1)

        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name

        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: sample_input})
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        output_array = outputs[0]
        result: Dict[str, Any] = {
            "input_shape": list(sample_input.shape),
            "output_shape": list(output_array.shape),
            "output_sample": output_array[:3].tolist()
            if output_array.size > 0
            else [],
            "inference_time_ms": round(elapsed_ms, 3),
            "status": "ok",
        }

        logger.info(
            "ONNX verification OK: %s -> %s (%.2f ms)",
            onnx_path.name,
            result["output_shape"],
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Model size utilities
    # ------------------------------------------------------------------

    def get_model_size(self, onnx_path: Union[str, Path]) -> int:
        """Return the file size of an ONNX model in bytes.

        Parameters
        ----------
        onnx_path : str or Path
            Path to the ``.onnx`` file.

        Returns
        -------
        int
            File size in bytes.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        path = Path(onnx_path)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        return path.stat().st_size

    def get_all_model_sizes(
        self, onnx_paths: Dict[str, Union[str, Path]]
    ) -> Dict[str, Dict[str, Any]]:
        """Get sizes for multiple ONNX models.

        Parameters
        ----------
        onnx_paths : dict
            ``{model_name: path}`` mapping.

        Returns
        -------
        dict
            ``{model_name: {"path": str, "bytes": int, "kb": float}}``
        """
        results: Dict[str, Dict[str, Any]] = {}
        total_bytes = 0
        for name, path in onnx_paths.items():
            size = self.get_model_size(path)
            total_bytes += size
            results[name] = {
                "path": str(path),
                "bytes": size,
                "kb": round(size / 1024, 2),
            }
        results["_total"] = {
            "bytes": total_bytes,
            "kb": round(total_bytes / 1024, 2),
            "mb": round(total_bytes / (1024 * 1024), 2),
        }
        return results

    def check_hf_spaces_limit(
        self,
        onnx_paths: Dict[str, Union[str, Path]],
        limit_bytes: int = 512 * 1024,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check whether exported models fit within the HF Spaces limit.

        Parameters
        ----------
        onnx_paths : dict
            ``{model_name: path}`` mapping.
        limit_bytes : int
            Maximum allowed total size in bytes (default 512 KB).

        Returns
        -------
        tuple of (bool, dict)
            ``(within_limit, size_report)``
        """
        sizes = self.get_all_model_sizes(onnx_paths)
        total = sizes["_total"]["bytes"]
        within = total <= limit_bytes

        report = {
            "models": {k: v for k, v in sizes.items() if k != "_total"},
            "total_bytes": total,
            "total_kb": sizes["_total"]["kb"],
            "limit_bytes": limit_bytes,
            "limit_kb": round(limit_bytes / 1024, 2),
            "within_limit": within,
        }

        if within:
            logger.info(
                "Model size OK: %d bytes (%.1f KB) <= %d bytes limit",
                total,
                total / 1024,
                limit_bytes,
            )
        else:
            logger.warning(
                "Model size EXCEEDS limit: %d bytes (%.1f KB) > %d bytes limit",
                total,
                total / 1024,
                limit_bytes,
            )

        return within, report
