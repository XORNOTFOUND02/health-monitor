"""
3D interactive visualizations for health symptom detection sensor data and
ML model feature space using Plotly.

Produces zoomable, rotatable 3D figures suitable for embedding in Gradio apps
or standalone HTML export. All functions return ``plotly.graph_objects.Figure``
instances.

Visualizations
--------------
1. **3D Accelerometer Trajectory** -- (ax, ay, az) path colored by time
2. **3D Gyroscope Path** -- (gx, gy, gz) path colored by time
3. **3D Vital Signs Space** -- HR vs SpO2 vs Temperature, colored by status
4. **3D Feature Space (PCA)** -- 162-dim features reduced to 3 PCA components,
   coloured by predicted condition, computed from multiple generated windows
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------
_CONDITION_COLORS: Dict[str, str] = {
    "normal": "#2e7d32",
    "tachycardia": "#e53935",
    "irregular_rhythm": "#d81b60",
    "low_spo2": "#1565c0",
    "fever": "#ef6c00",
    "fall_detected": "#8e24aa",
    "sleep_problem": "#6a1b9a",
    "fatigue": "#00838f",
}

_SEVERITY_COLORS: Dict[str, str] = {
    "normal": "#2e7d32",
    "mild": "#fdd835",
    "moderate": "#ef6c00",
    "severe": "#e53935",
    "critical": "#b71c1c",
}

_TIME_COLORSCALE = "Viridis"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _try_get_array(data: Any, *keys: str) -> np.ndarray:
    """Safely extract a numpy array from a nested dict."""
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, np.array([]))
        else:
            return np.array([])
    if isinstance(current, (list, tuple)):
        current = np.array(current, dtype=np.float64)
    if not isinstance(current, np.ndarray):
        return np.array([])
    return current.ravel()


def _sensor_data_to_arrays(
    sensor_data: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Convert a raw sensor-data dict into flat numpy arrays keyed by axis.

    Handles both the raw format (``{"accelerometer": [[ax, ay, az], ...]}``)
    and the extractor format (``{"accelerometer": {"ax": arr, ...}}``).
    """
    arrays: Dict[str, np.ndarray] = {}

    # -- Accelerometer --
    accel = sensor_data.get("accelerometer", {})
    if isinstance(accel, dict) and "ax" in accel:
        # Extractor format
        arrays["ax"] = _try_get_array(accel, "ax")
        arrays["ay"] = _try_get_array(accel, "ay")
        arrays["az"] = _try_get_array(accel, "az")
    elif isinstance(accel, (list, np.ndarray)):
        # Raw format: list of [ax, ay, az] samples
        arr = np.asarray(accel, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            arrays["ax"] = arr[:, 0]
            arrays["ay"] = arr[:, 1]
            arrays["az"] = arr[:, 2]

    # -- Gyroscope --
    gyro = sensor_data.get("gyroscope", {})
    if isinstance(gyro, dict) and "gx" in gyro:
        arrays["gx"] = _try_get_array(gyro, "gx")
        arrays["gy"] = _try_get_array(gyro, "gy")
        arrays["gz"] = _try_get_array(gyro, "gz")
    elif isinstance(gyro, (list, np.ndarray)):
        arr = np.asarray(gyro, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            arrays["gx"] = arr[:, 0]
            arrays["gy"] = arr[:, 1]
            arrays["gz"] = arr[:, 2]

    # -- Heart rate --
    hr_data = sensor_data.get("heart_rate", {})
    if isinstance(hr_data, dict) and "bpm" in hr_data:
        arrays["hr"] = _try_get_array(hr_data, "bpm")
        arrays["spo2"] = _try_get_array(hr_data, "spo2")
        arrays["ppg"] = _try_get_array(hr_data, "ppg_raw")
    else:
        hr_raw = sensor_data.get("heart_rate", [])
        if isinstance(hr_raw, (list, np.ndarray)):
            arrays["hr"] = np.asarray(hr_raw, dtype=np.float64).ravel()
        spo2_raw = sensor_data.get("spo2", [])
        if isinstance(spo2_raw, (list, np.ndarray)):
            arrays["spo2"] = np.asarray(spo2_raw, dtype=np.float64).ravel()

    # -- Temperature --
    temp_data = sensor_data.get("temperature", {})
    if isinstance(temp_data, dict) and "stts22h_celsius" in temp_data:
        arrays["temp"] = _try_get_array(temp_data, "stts22h_celsius")
    else:
        temp_raw = sensor_data.get("temperature", [])
        if isinstance(temp_raw, (list, np.ndarray)):
            arrays["temp"] = np.asarray(temp_raw, dtype=np.float64).ravel()

    return arrays


# ===================================================================
# 1. 3D Accelerometer Trajectory
# ===================================================================


def plot_accelerometer_3d(
    sensor_data: Dict[str, Any],
    *,
    title: str = "3D Accelerometer Trajectory",
    marker_size: int = 3,
    line_width: int = 2,
    height: int = 550,
) -> go.Figure:
    """Interactive 3D line+scatter plot of accelerometer axes (ax, ay, az).

    Parameters
    ----------
    sensor_data : dict
        Raw sensor data dictionary (extractor format or raw format).
    title : str
        Plot title.
    marker_size : int
        Marker size for scatter points.
    line_width : int
        Width of the trajectory line.

    Returns
    -------
    go.Figure
        An interactive Plotly 3D figure.
    """
    arrays = _sensor_data_to_arrays(sensor_data)
    ax_arr = arrays.get("ax", np.array([]))
    ay_arr = arrays.get("ay", np.array([]))
    az_arr = arrays.get("az", np.array([]))

    if ax_arr.size == 0 or ay_arr.size == 0 or az_arr.size == 0:
        return _empty_figure("No accelerometer data available")

    # Subsample if too many points for performance
    n = len(ax_arr)
    step = max(1, n // 2000)
    if step > 1:
        idx = np.arange(0, n, step)
        ax_arr = ax_arr[idx]
        ay_arr = ay_arr[idx]
        az_arr = az_arr[idx]

    t_norm = np.linspace(0, 1, len(ax_arr))

    fig = go.Figure()

    # Trajectory line
    fig.add_trace(
        go.Scatter3d(
            x=ax_arr,
            y=ay_arr,
            z=az_arr,
            mode="lines+markers",
            marker=dict(
                size=marker_size,
                color=t_norm,
                colorscale=_TIME_COLORSCALE,
                colorbar=dict(title="Time (normalised)", x=0.85),
                showscale=True,
            ),
            line=dict(color="rgba(100,100,100,0.4)", width=line_width),
            name="Accelerometer",
            hovertemplate=(
                "ax: %{x:.2f} g<br>"
                "ay: %{y:.2f} g<br>"
                "az: %{z:.2f} g<br>"
                "<extra></extra>"
            ),
        )
    )

    _apply_3d_layout(
        fig,
        title=title,
        xaxis_title="ax (g)",
        yaxis_title="ay (g)",
        zaxis_title="az (g)",
        height=height,
    )

    return fig


# ===================================================================
# 2. 3D Gyroscope Path
# ===================================================================


def plot_gyroscope_3d(
    sensor_data: Dict[str, Any],
    *,
    title: str = "3D Gyroscope Path",
    marker_size: int = 3,
    line_width: int = 2,
    height: int = 550,
) -> go.Figure:
    """Interactive 3D line+scatter plot of gyroscope axes (gx, gy, gz).

    Parameters are identical to :func:`plot_accelerometer_3d`.
    """
    arrays = _sensor_data_to_arrays(sensor_data)
    gx_arr = arrays.get("gx", np.array([]))
    gy_arr = arrays.get("gy", np.array([]))
    gz_arr = arrays.get("gz", np.array([]))

    if gx_arr.size == 0 or gy_arr.size == 0 or gz_arr.size == 0:
        return _empty_figure("No gyroscope data available")

    n = len(gx_arr)
    step = max(1, n // 2000)
    if step > 1:
        idx = np.arange(0, n, step)
        gx_arr = gx_arr[idx]
        gy_arr = gy_arr[idx]
        gz_arr = gz_arr[idx]

    t_norm = np.linspace(0, 1, len(gx_arr))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=gx_arr,
            y=gy_arr,
            z=gz_arr,
            mode="lines+markers",
            marker=dict(
                size=marker_size,
                color=t_norm,
                colorscale=_TIME_COLORSCALE,
                colorbar=dict(title="Time (normalised)", x=0.85),
                showscale=True,
            ),
            line=dict(color="rgba(100,100,100,0.4)", width=line_width),
            name="Gyroscope",
            hovertemplate=(
                "gx: %{x:.4f} dps<br>"
                "gy: %{y:.4f} dps<br>"
                "gz: %{z:.4f} dps<br>"
                "<extra></extra>"
            ),
        )
    )

    _apply_3d_layout(
        fig,
        title=title,
        xaxis_title="gx (dps)",
        yaxis_title="gy (dps)",
        zaxis_title="gz (dps)",
        height=height,
    )

    return fig


# ===================================================================
# 3. 3D Vital Signs Space
# ===================================================================


def plot_vitals_3d(
    sensor_data: Dict[str, Any],
    predictions: Optional[Dict[str, Any]] = None,
    *,
    title: str = "3D Vital Signs Space",
    marker_size: int = 6,
    height: int = 550,
) -> go.Figure:
    """3D scatter plot of HR, SpO2, and Temperature.

    If *predictions* are provided, points are colored by overall health
    status. Otherwise a default blue is used.

    Parameters
    ----------
    sensor_data : dict
        Raw sensor data dictionary.
    predictions : dict or None
        Inference result dict (from ``_run_inference``) containing
        ``overall_status`` and per-condition probabilities.
    title : str
        Plot title.
    marker_size : int
        Marker size.

    Returns
    -------
    go.Figure
    """
    arrays = _sensor_data_to_arrays(sensor_data)
    hr_arr = arrays.get("hr", np.array([]))
    spo2_arr = arrays.get("spo2", np.array([]))
    temp_arr = arrays.get("temp", np.array([]))

    has_data = (
        hr_arr.size > 0 or spo2_arr.size > 0 or temp_arr.size > 0
    )
    if not has_data:
        return _empty_figure("No vital sign data available")

    # Determine overall status from predictions
    overall_status = "normal"
    if predictions and isinstance(predictions, dict):
        overall_status = predictions.get("overall_status", "normal")

    # Build per-axis arrays, handling different lengths
    # Interpolate to the longest array for pairwise plotting
    max_len = max(
        hr_arr.size if hr_arr.size > 0 else 0,
        spo2_arr.size if spo2_arr.size > 0 else 0,
        temp_arr.size if temp_arr.size > 0 else 0,
    )

    if max_len < 2:
        return _empty_figure("Not enough vital-sign samples (< 2)")

    def _align(arr: np.ndarray, n: int) -> np.ndarray:
        if arr.size == 0:
            return np.full(n, np.nan)
        if arr.size == n:
            return arr
        # Linear interpolation
        x_old = np.linspace(0, 1, arr.size)
        x_new = np.linspace(0, 1, n)
        return np.interp(x_new, x_old, arr)

    hr_align = _align(hr_arr, max_len)
    spo2_align = _align(spo2_arr, max_len)
    temp_align = _align(temp_arr, max_len)

    # Color by status
    status_color = _SEVERITY_COLORS.get(overall_status, "#2e7d32")

    fig = go.Figure()

    # Main scatter
    valid = (
        ~np.isnan(hr_align)
        & ~np.isnan(spo2_align)
        & ~np.isnan(temp_align)
    )
    if valid.sum() > 0:
        fig.add_trace(
            go.Scatter3d(
                x=hr_align[valid],
                y=spo2_align[valid],
                z=temp_align[valid],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=status_color,
                    symbol="circle",
                    line=dict(color="rgba(0,0,0,0.3)", width=1),
                ),
                name=f"Status: {overall_status}",
                hovertemplate=(
                    "HR: %{x:.1f} bpm<br>"
                    "SpO2: %{y:.1f} %<br>"
                    "Temp: %{z:.1f} C<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Add region markers for clinical thresholds
    _add_clinical_thresholds(fig)

    _apply_3d_layout(
        fig,
        title=title,
        xaxis_title="Heart Rate (bpm)",
        yaxis_title="SpO2 (%)",
        zaxis_title="Temperature (C)",
        height=height,
    )

    return fig


def _add_clinical_thresholds(fig: go.Figure) -> None:
    """Add semi-transparent planes for clinical thresholds."""
    # Tachycardia threshold: HR = 100 bpm
    fig.add_trace(
        go.Scatter3d(
            x=[100, 100],
            y=[80, 100],
            z=[35, 42],
            mode="lines",
            line=dict(color="rgba(225, 50, 50, 0.3)", width=2, dash="dash"),
            name="HR > 100 (tachycardia)",
            showlegend=True,
        )
    )
    # Hypoxemia threshold: SpO2 = 94%
    fig.add_trace(
        go.Scatter3d(
            x=[40, 200],
            y=[94, 94],
            z=[35, 42],
            mode="lines",
            line=dict(color="rgba(50, 50, 225, 0.3)", width=2, dash="dash"),
            name="SpO2 < 94% (hypoxemia)",
            showlegend=True,
        )
    )
    # Fever threshold: Temp = 38 C
    fig.add_trace(
        go.Scatter3d(
            x=[40, 200],
            y=[80, 100],
            z=[38, 38],
            mode="lines",
            line=dict(color="rgba(225, 150, 50, 0.3)", width=2, dash="dash"),
            name="Temp > 38 C (fever)",
            showlegend=True,
        )
    )


# ===================================================================
# 4. 3D Feature Space (PCA)
# ===================================================================


def plot_feature_space_3d(
    generator: Any,
    condition: str = "normal",
    severity: str = "moderate",
    n_windows: int = 30,
    feature_extractor: Any = None,
    *,
    title: str = "3D Feature Space (PCA)",
    marker_size: int = 5,
    height: int = 550,
    random_seed: int = 42,
) -> go.Figure:
    """Generate multiple sensor windows, extract features, reduce to 3
    principal components, and plot coloured by predicted condition.

    This function generates *n_windows* of synthetic sensor data for the
    given *condition* / *severity*, runs the feature extractor + predictor
    on each, then applies PCA to project the 162-dimensional feature vectors
    into 3D for visualisation.

    Parameters
    ----------
    generator : SyntheticDataGenerator
        An instance of the synthetic data generator.
    condition : str
        Health condition to simulate (e.g. ``"tachycardia"``, ``"normal"``).
    severity : str
        Severity level (``"mild"``, ``"moderate"``, ``"severe"``).
    n_windows : int
        Number of windows to generate for the feature space.
    feature_extractor : FeatureExtractor or None
        If provided, extracts features from each window.
    predictor : Predictor or None
        If provided, runs inference to get predicted condition labels.
    title : str
        Plot title.
    marker_size : int
        Marker size for scatter points.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    go.Figure
    """
    if feature_extractor is None:
        return _empty_figure("Feature extractor not available")

    windows_data: List[Dict[str, Any]] = []
    labels: List[str] = []
    probabilities: List[float] = []

    for i in range(n_windows):
        try:
            if condition.lower() == "normal":
                session = generator.generate_normal_session(duration_sec=30)
            else:
                session = generator.generate_condition_session(
                    condition, duration_sec=30, severity=severity,
                )
        except Exception as exc:
            logger.warning("Window %d generation failed: %s", i, exc)
            continue

        sensor_data = (
            session.get("windows", [{}])[0]
            .get("sensor_data", {})
        )
        if not sensor_data:
            continue

        windows_data.append(sensor_data)
        labels.append(condition)

    if len(windows_data) < 3:
        return _empty_figure(
            f"Need at least 3 windows (got {len(windows_data)})"
        )

    # Extract features for all windows
    from src.inference.predictor import Predictor
    feature_vectors: List[np.ndarray] = []
    valid_labels: List[str] = []
    feature_names: List[str] = feature_extractor.get_all_feature_names()

    for sd, lbl in zip(windows_data, labels):
        try:
            # Convert to extractor format if needed
            # Normalize using the same logic as the predictor
            normalized = Predictor._normalize_input(sd)
            features = feature_extractor.extract_all(normalized)
            vec = np.array(
                [features.get(name, 0.0) for name in feature_names],
                dtype=np.float64,
            )
            # Replace NaN
            vec[~np.isfinite(vec)] = 0.0
            feature_vectors.append(vec)
            valid_labels.append(lbl)
        except Exception as exc:
            logger.warning("Feature extraction failed for window: %s", exc)
            continue

    if len(feature_vectors) < 3:
        return _empty_figure(
            f"Need at least 3 valid feature vectors (got {len(feature_vectors)})"
        )

    X = np.vstack(feature_vectors)  # shape (n, 162)

    # PCA via SVD
    X_centered = X - X.mean(axis=0)
    try:
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        X_3d = X_centered @ Vt[:3, :].T  # project onto first 3 PCs
    except np.linalg.LinAlgError:
        return _empty_figure("PCA decomposition failed")

    # Explained variance
    var_explained = S[:3] ** 2 / (S ** 2).sum() if S.sum() > 0 else np.zeros(3)
    var_str = (
        f"PC1: {var_explained[0]:.1%}, "
        f"PC2: {var_explained[1]:.1%}, "
        f"PC3: {var_explained[2]:.1%}"
    )

    # Color by condition
    color_map = _CONDITION_COLORS
    colors = [color_map.get(lbl.lower(), "#757575") for lbl in valid_labels]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=X_3d[:, 0],
            y=X_3d[:, 1],
            z=X_3d[:, 2],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=colors,
                symbol="circle",
                line=dict(color="rgba(0,0,0,0.3)", width=1),
            ),
            text=[f"Condition: {lbl}" for lbl in valid_labels],
            hovertemplate=(
                "PC1: %{x:.2f}<br>"
                "PC2: %{y:.2f}<br>"
                "PC3: %{z:.2f}<br>"
                "%{text}<br>"
                "<extra></extra>"
            ),
            name="Feature Space",
            showlegend=False,
        )
    )

    _apply_3d_layout(
        fig,
        title=f"{title}<br><sup>{var_str}</sup>",
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        zaxis_title="Principal Component 3",
        height=height,
    )

    # Add legend for conditions
    for cond, color in color_map.items():
        if cond in set(lbl.lower() for lbl in valid_labels):
            fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",
                    marker=dict(size=8, color=color),
                    name=cond.replace("_", " ").title(),
                    showlegend=True,
                )
            )

    return fig


# ===================================================================
# 5. Combined: All 3D plots in one layout
# ===================================================================


def plot_all_3d(
    sensor_data: Dict[str, Any],
    predictions: Optional[Dict[str, Any]] = None,
    *,
    height: int = 800,
) -> go.Figure:
    """Arrange accelerometer, gyroscope, and vital-signs 3D plots in a
    single subplot figure.

    Parameters
    ----------
    sensor_data : dict
        Raw sensor data dictionary.
    predictions : dict or None
        Inference result dict for status coloring.
    height : int
        Total figure height in pixels.

    Returns
    -------
    go.Figure
    """
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Accelerometer 3D",
            "Gyroscope 3D",
            "Vital Signs 3D",
        ),
        specs=[
            [{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}]
        ],
    )

    # Accelerometer
    accel_fig = plot_accelerometer_3d(sensor_data, marker_size=2)
    if len(accel_fig.data) > 0:
        for trace in accel_fig.data:
            fig.add_trace(trace, row=1, col=1)

    # Gyroscope
    gyro_fig = plot_gyroscope_3d(sensor_data, marker_size=2)
    if len(gyro_fig.data) > 0:
        for trace in gyro_fig.data:
            fig.add_trace(trace, row=1, col=2)

    # Vitals
    vitals_fig = plot_vitals_3d(sensor_data, predictions, marker_size=5)
    if len(vitals_fig.data) > 0:
        for trace in vitals_fig.data:
            fig.add_trace(trace, row=1, col=3)

    fig.update_layout(
        title_text="3D Sensor Data Overview",
        height=height,
        scene=dict(
            xaxis=dict(title="ax (g)", showbackground=True, backgroundcolor="#f8f9fa"),
            yaxis=dict(title="ay (g)", showbackground=True, backgroundcolor="#f8f9fa"),
            zaxis=dict(title="az (g)", showbackground=True, backgroundcolor="#f8f9fa"),
        ),
        scene2=dict(
            xaxis=dict(title="gx (dps)", showbackground=True, backgroundcolor="#f8f9fa"),
            yaxis=dict(title="gy (dps)", showbackground=True, backgroundcolor="#f8f9fa"),
            zaxis=dict(title="gz (dps)", showbackground=True, backgroundcolor="#f8f9fa"),
        ),
        scene3=dict(
            xaxis=dict(title="HR (bpm)", showbackground=True, backgroundcolor="#f8f9fa"),
            yaxis=dict(title="SpO2 (%)", showbackground=True, backgroundcolor="#f8f9fa"),
            zaxis=dict(title="Temp (C)", showbackground=True, backgroundcolor="#f8f9fa"),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig


# ===================================================================
# Internal utilities
# ===================================================================


def _apply_3d_layout(
    fig: go.Figure,
    *,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    zaxis_title: str,
    height: int = 550,
) -> None:
    """Apply consistent 3D layout styling."""
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=14, color="#2c3e50"),
        ),
        height=height,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="white",
        font=dict(color="#2c3e50"),
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
    )
    fig.update_scenes(
        xaxis=dict(
            title=xaxis_title,
            showbackground=True,
            backgroundcolor="#f8f9fa",
            gridcolor="#dce1e3",
        ),
        yaxis=dict(
            title=yaxis_title,
            showbackground=True,
            backgroundcolor="#f8f9fa",
            gridcolor="#dce1e3",
        ),
        zaxis=dict(
            title=zaxis_title,
            showbackground=True,
            backgroundcolor="#f8f9fa",
            gridcolor="#dce1e3",
        ),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5),
        ),
        aspectmode="cube",
    )


def _empty_figure(message: str) -> go.Figure:
    """Return a figure displaying an error message when data is unavailable."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16, color="#757575"),
    )
    fig.update_layout(
        title=dict(text="No Data", x=0.5, xanchor="center"),
        height=400,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="white",
    )
    return fig


# ===================================================================
# Public convenience API
# ===================================================================

__all__ = [
    "plot_accelerometer_3d",
    "plot_gyroscope_3d",
    "plot_vitals_3d",
    "plot_feature_space_3d",
    "plot_all_3d",
]
