# NeuraBand

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green.svg)](https://lightgbm.readthedocs.io/)
[![Gradio](https://img.shields.io/badge/Gradio-4%2B-orange.svg)](https://gradio.app/)
[![Tests](https://img.shields.io/badge/tests-73%20passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/XORNOTFOUND02/health-monitor/pulls)

AI-powered health symptom detection from wearable sensor data. Uses an ensemble of **LightGBM** classifiers with a **rule-based fever engine** to detect 7 health conditions from MPU6500 (accelerometer + gyroscope), HMC5883L (magnetometer), MAX30102 (heart rate, SpO2, PPG), STTS22H, and LM35 sensors.

Designed for **Hugging Face Spaces free CPU tier** (2 vCPU, 16 GB RAM) and trained on **local GPU** (RTX 2050, 4 GB VRAM).

## Table of Contents

- [Features](#features)
- [Detectable Conditions](#detectable-conditions)
- [Architecture](#architecture)
- [Sensor Configuration](#sensor-configuration)
- [Installation](#installation)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Generate Synthetic Data](#generate-synthetic-data)
  - [Train Models](#train-models)
  - [Run Inference](#run-inference)
  - [Evaluate Models](#evaluate-models)
  - [Launch Gradio App](#launch-gradio-app)
- [Evaluation Dashboard](#evaluation-dashboard)
  - [Per-Condition Performance](#per-condition-performance)
  - [Chart Gallery](#chart-gallery)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
  - [Hugging Face Spaces](#hugging-face-spaces)
  - [Docker](#docker)
- [Testing](#testing)
- [Model Performance](#model-performance)
- [Limitations](#limitations)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Features

- **7 health conditions detected** from multimodal sensor data
- **3 specialised LightGBM models** + rule-based fever detection
- **~162 engineered features** across motion, cardiac, respiratory, and temperature domains
- **30-second sliding windows** with 5-second stride for real-time analysis
- **Temporal smoothing** with N-of-M voting to suppress false positives
- **Cooldown mechanism** prevents repetitive alerts for chronic conditions
- **Gradio web interface** for live sensor input and simulated demos
- **Comprehensive evaluation** with publication-quality charts (confusion matrix, ROC, PR, calibration, threshold analysis, feature importance, radar comparison)
- **ONNX export** (experimental) for optimized CPU inference
- **Synthetic data generator** for training and testing without real sensors

## Detectable Conditions

| Condition | Type | Model | Threshold | Description |
|-----------|------|-------|-----------|-------------|
| Tachycardia | Cardiac | LightGBM | HR > 100 bpm | Abnormally high resting heart rate |
| Irregular Rhythm | Cardiac | LightGBM | HRV < 20 ms | Arrhythmia or atrial fibrillation indicators |
| Low SpO2 | Respiratory | LightGBM | SpO2 < 94% | Hypoxemia, respiratory distress |
| Fever | Temperature | Rule Engine | Temp > 38.0 C | Elevated body temperature |
| Fall Detected | Activity | LightGBM | Accel > 3g impact | Sudden impact followed by stillness |
| Sleep Problem | Activity | LightGBM | HRV + motion | Restless sleep or apnea indicators |
| Fatigue | Activity | LightGBM | HRV + activity | Physical or mental exhaustion markers |

## Architecture

```
Sensor Data (JSON)
    |
    v
[Data Preprocessor]
  - Butterworth filter (2 Hz low-pass for accel)
  - Z-score normalization
  - Resampling to consistent rates
    |
    v
[Window Generator]
  - 30-second windows, 5-second stride
  - 50 Hz accelerometer, 25 Hz heart rate
    |
    v
[Feature Extractor]  (162 features total)
  +-- Motion Features (63)  -- accelerometer + gyroscope stats
  +-- Heart Rate Features (14) -- HR mean, std, min, max, gradients
  +-- HRV Features (20) -- SDNN, RMSSD, pNN50, frequency bands
  +-- SpO2 Features (10) -- mean, min, desaturation events
  +-- Temperature Features (9) -- mean, trend, rate of change
  +-- Cross-Sensor Features (15) -- HR-Temp, HR-SpO2 correlations
  +-- Frequency-Domain Features (31) -- FFT, spectral entropy, band power
    |
    v
[Ensemble Models]
  +-- CardiacModel  -> tachycardia, irregular_rhythm
  +-- RespiratoryModel -> low_spo2
  +-- ActivityModel  -> fall_detected, sleep_problem, fatigue
  +-- RuleEngine   -> fever (threshold-based) + data quality
    |
    v
[Temporal Smoother]
  - N-of-M voting (default 3/5 windows)
  - Per-condition cooldown (60 seconds)
    |
    v
[Response Builder]
  - Standardised JSON response
  - Alert categorization (critical / warning / info)
  - Medical disclaimer
```

## Sensor Configuration

| Sensor | Signals | Sampling Rate | Resolution |
|--------|---------|---------------|------------|
| MPU6500 | Accelerometer (ax, ay, az) | 50 Hz | 16-bit |
| MPU6500 | Gyroscope (gx, gy, gz) | 50 Hz | 16-bit |
| MAX30102 | Heart Rate (BPM) | 25 Hz | 18-bit |
| MAX30102 | Photoplethysmogram (PPG) | 25 Hz | 18-bit |
| MAX30102 | Blood Oxygen (SpO2) | 25 Hz | 18-bit |
| STTS22H | Temperature | 1 Hz | 16-bit |
| LM35 | Temperature | 1 Hz | 10-bit |

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/health-monitor.git
cd health-monitor

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install evaluation dependencies (for charts)
pip install matplotlib seaborn
```

## Usage

### Quick Start

```bash
# Generate synthetic data, train models, run inference
make train-quick
```

### Generate Synthetic Data

```bash
# Generate 100 sessions (300 seconds each)
python scripts/generate_synthetic.py \
    --num-sessions 100 \
    --output-dir data/synthetic/raw \
    --seed 42 \
    --include-labels

# Quick generation (10 sessions, 30 seconds each)
make data-quick
```

### Train Models

```bash
# Train with 30 sessions on CPU
make train

# Or specify custom parameters
python scripts/train_pipeline.py \
    --num-sessions 50 \
    --no-gpu \
    --seed 42 \
    --force

# Quick test (10 sessions, 2 boosting rounds)
make train-quick

# Full training (100 sessions, full rounds)
make train-full
```

Training produces these outputs in `models/`:
- `cardiac.joblib` + `cardiac.meta.json`
- `respiratory.joblib` + `respiratory.meta.json`
- `activity.joblib` + `activity.meta.json`
- `feature_names.json` — canonical 162-feature ordering
- `evaluation_report.json` + `evaluation_report.txt`

### Run Inference

```python
from src.inference import Predictor, TemporalSmoother, ResponseBuilder

# Load models
predictor = Predictor(models_dir="models")
smoother = TemporalSmoother()
builder = ResponseBuilder()

# Predict on a single sensor window
window_data = {
    "accelerometer": {"ax": [...], "ay": [...], "az": [...]},
    "gyroscope": {"gx": [...], "gy": [...], "gz": [...]},
    "heart_rate": {"bpm": [...], "spo2": [...], "ppg_raw": [...]},
    "temperature": {"stts22h_celsius": [...], "lm35_celsius": [...]},
    "metadata": {"activity_state": "resting"},
}

raw_preds = predictor.predict(window_data)
smoothed = smoother.update(raw_preds, timestamp=time.time())
quality = predictor._compute_data_quality(predictor._normalize_input(window_data))
response = builder.build_response(smoothed, quality)

print(json.dumps(response, indent=2))
```

### Evaluate Models

```bash
# Comprehensive evaluation with charts
python scripts/evaluate_model.py --sessions 50 --output-dir evaluation_results

# Quick evaluation (5 sessions, no charts)
python scripts/evaluate_model.py --quick --skip-charts

# Custom thresholds
python scripts/evaluate_model.py --thresholds 0.3,0.5,0.7 --sessions 30
```

The evaluation generates:
- **42+ individual PNG charts** (confusion matrix, ROC, PR, calibration, threshold, feature importance per condition)
- **Radar comparison chart** across all conditions
- **MCB-DSC plot** (calibration vs discrimination)
- **Evaluation dashboard** (3x3 grid of key charts)
- **JSON report** with all metrics at multiple thresholds
- **Text summary** with key findings

### Launch Gradio App

```bash
# Run the Gradio web interface locally
make deploy-run
# or: python hf_space/app.py
```

The app provides 4 tabs:
1. **Live Sensor Input** — paste JSON sensor data, click Analyse
2. **Simulated Demo** — select a condition and severity, generate & analyse
3. **About / Disclaimer** — system info and medical disclaimer
4. **Monitor Dashboard** — streaming real-time dashboard

---

## Evaluation Dashboard

The evaluation module generates **publication-quality charts** with statistics computed directly from model predictions.

### Per-Condition Performance

| Condition | Accuracy | Precision | Recall | F1 Score | AUC | Brier Score |
|-----------|----------|-----------|--------|----------|-----|-------------|
| Tachycardia | 0.880 | 0.538 | 1.000 | 0.700 | 0.940 | 0.120 |
| Irregular Rhythm | 0.900 | 0.556 | 0.833 | 0.667 | 0.985 | 0.048 |
| Low SpO2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| Fall Detected | 0.780 | 0.353 | 1.000 | 0.522 | 1.000 | 0.166 |
| Sleep Problem | 0.880 | 0.500 | 1.000 | 0.667 | 0.973 | 0.122 |
| Fatigue | 0.240 | 0.136 | 1.000 | 0.240 | 0.727 | 0.760 |
| Fever | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.008 |

*Metrics at threshold=0.5 on 50 test windows (balanced across 8 conditions). Results are on synthetic data — real-world performance will differ.*

### Chart Gallery

The following chart types are generated for each condition:

| Chart Type | Description | Key Statistic |
|------------|-------------|---------------|
| **Confusion Matrix** | Annotated heatmap with counts and percentages | TP, TN, FP, FN |
| **ROC Curve** | True Positive Rate vs False Positive Rate | AUC score |
| **Precision-Recall Curve** | Precision vs Recall at all thresholds | Average Precision (AP) |
| **Calibration Curve** | Reliability diagram | Brier Score, ECE |
| **Threshold Analysis** | P/R/F1 vs decision threshold | Optimal threshold |
| **Feature Importance** | Top 20 features by gain | Importance weight |
| **Radar Comparison** | Multi-metric across all conditions | 5 metrics |
| **MCB-DSC Plot** | Miscalibration vs Discrimination | Brier decomposition |
| **Dashboard** | 3x3 grid of key charts | Combined view |

Charts are saved to `evaluation_results/charts/` with naming convention: `{model_group}_{condition}_{chart_type}.png`.

---

## Project Structure

```
health-monitor/
├── src/
│   ├── __init__.py
│   ├── config.py                # Central configuration
│   ├── data/
│   │   ├── loader.py             # Sensor data loading + schema validation
│   │   ├── preprocessor.py       # Butterworth filter, normalisation
│   │   ├── window_generator.py   # Sliding window segmentation
│   │   └── label_generator.py    # Rule-based ground truth labelling
│   ├── features/
│   │   ├── base.py               # Base feature extractor class
│   │   ├── extractor.py          # Feature extraction orchestrator
│   │   ├── motion.py             # 63 motion features
│   │   ├── heart_rate.py         # 14 HR features
│   │   ├── hrv.py                # 20 HRV features
│   │   ├── spo2.py               # 10 SpO2 features
│   │   ├── temperature.py        # 9 temperature features
│   │   ├── cross_sensor.py       # 15 cross-sensor features
│   │   └── frequency_domain.py   # 31 frequency-domain features
│   ├── models/
│   │   ├── cardiac_model.py      # Tachycardia + irregular rhythm
│   │   ├── respiratory_model.py  # Low SpO2
│   │   ├── activity_model.py     # Fall, sleep, fatigue
│   │   └── rule_engine.py        # Fever detection + data quality
│   ├── inference/
│   │   ├── __init__.py           # Export Predictor, TemporalSmoother, ResponseBuilder
│   │   ├── predictor.py          # Ensemble inference engine
│   │   ├── temporal_smoother.py  # N-of-M voting + cooldown
│   │   └── response_builder.py   # Standardised JSON response
│   └── evaluation/
│       ├── __init__.py           # Export evaluation classes
│       ├── metrics.py            # MetricsCalculator (25+ metrics)
│       ├── visualizer.py         # 10 chart types (publication-quality)
│       └── report.py             # EvaluationReport orchestrator
├── data/
│   └── synthetic/
│       └── generator.py          # 5 simulator classes
├── models/                       # Trained models + metadata
├── tests/
│   ├── test_features.py          # 47 feature + pipeline tests
│   └── test_inference.py         # 26 inference integration tests
├── hf_space/
│   ├── app.py                    # Gradio web application (1077 lines)
│   ├── requirements.txt          # HF Space dependencies
│   └── README.md                 # HF Space description
├── scripts/
│   ├── generate_synthetic.py     # Data generation CLI
│   ├── train_pipeline.py         # Training pipeline CLI
│   ├── evaluate_model.py         # Evaluation CLI with charts
│   ├── prepare_deploy.py         # Deployment packaging tool
│   ├── e2e_test.py               # End-to-end validation
│   ├── quick_validate.py         # Quick smoke test
│   └── final_validation.py       # Comprehensive validation
├── evaluation_results/           # Generated evaluation charts + reports
├── Makefile                      # 17 automation targets
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Deployment

### Hugging Face Spaces

```bash
# 1. Check deployment readiness
make deploy-check

# 2. Create deployment package
python scripts/prepare_deploy.py --package

# 3. Create a Space at https://huggingface.co/new-space
#    - SDK: Gradio
#    - Space name: health-monitor

# 4. Upload the deploy/ directory contents or manually copy:
#    - hf_space/app.py -> app.py
#    - hf_space/requirements.txt
#    - hf_space/README.md
#    - src/ (entire directory)
#    - models/ (joblib files + feature_names.json)
#    - data/synthetic/generator.py
```

### Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "hf_space/app.py"]
```

---

## Testing

```bash
# Run all tests
make test
# or: python -m pytest tests/ -v

# Run end-to-end validation
make test-e2e

# Quick smoke test
make validate

# Full validation (73 tests + 32 checks)
python -m pytest tests/ -v
python scripts/final_validation.py
```

**73 tests total:**
- `test_features.py` — 47 tests (motion, HR, HRV, SpO2, temperature, cross-sensor, frequency-domain, feature extractor, data pipeline, synthetic generator)
- `test_inference.py` — 26 tests (predictor loading, inference, temporal smoother, response builder, input normalisation)

---

## Model Performance

Performance on **synthetic test data** (30 training sessions, 50 test windows, 162 features, 1000 boosting rounds):

| Model Group | AUC | Macro F1 | Test Accuracy |
|-------------|-----|----------|---------------|
| Cardiac | 0.962 | 0.683 | 0.890 |
| Respiratory | 1.000 | 1.000 | 1.000 |
| Activity | 0.900 | 0.476 | 0.633 |
| Rule Engine (Fever) | 1.000 | 1.000 | 1.000 |

**Key observations:**
- Low SpO2 and Fever achieve perfect scores due to clear synthetic signal patterns
- Tachycardia and Sleep Problem show strong discrimination (AUC > 0.94)
- Fatigue has lower performance due to subtle synthetic signal differences
- All models show perfect recall but varying precision — reflecting class imbalance in the 50-window test set

**Threshold sensitivity** (macro average F1):
| Threshold | 0.3 | 0.5 | 0.7 |
|-----------|-----|-----|-----|
| Macro F1 | 0.656 | 0.685 | 0.735 |

Higher thresholds (0.7) improve F1 by reducing false positives.

---

## Limitations

1. **Synthetic data only** — Models are trained on simulated sensor data. Real-world performance will differ and requires retraining with actual sensor data.
2. **Class imbalance** — The 50-window test set has ~12% positive rate per condition, affecting precision estimates. Increase test sessions for more stable metrics.
3. **ONNX parity** — ONNX export is experimental; LightGBM 4.6.0 vs onnxmltools 1.16.0 shows output shape differences. Use joblib format for production.
4. **Not a medical device** — This system is for research and educational purposes. It is NOT FDA-approved or certified for clinical use. Always consult a physician for health concerns.
5. **Single-window processing** — The current pipeline processes independent windows. A streaming implementation with overlapping windows would provide smoother real-time analysis.
6. **No anomaly detection** — The system only classifies known conditions. Unknown health events may produce false negatives.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

**Medical disclaimer:** This software is provided for research and educational purposes only. It is not a medical device and must not be used for clinical diagnosis, treatment, or patient monitoring without appropriate regulatory approval. Always consult qualified healthcare professionals for medical decisions.

---

## Acknowledgments

- [LightGBM](https://lightgbm.readthedocs.io/) — Gradient boosting framework by Microsoft
- [Gradio](https://gradio.app/) — Web interface for ML models
- [Hugging Face Spaces](https://huggingface.co/spaces) — Free CPU hosting
- [scikit-learn](https://scikit-learn.org/) — Machine learning utilities and metrics
- [Matplotlib](https://matplotlib.org/) + [Seaborn](https://seaborn.pydata.org/) — Publication-quality visualizations
- [MPU6500](https://invensense.tdk.com/products/motion-tracking/6-axis/mpu-6500/) — 6-axis motion tracking
- [MAX30102](https://www.analog.com/en/products/max30102.html) — Pulse oximetry and heart-rate module
- [STTS22H](https://www.st.com/en/mems-and-sensors/stts22h.html) — Digital temperature sensor
- [LM35](https://www.ti.com/product/LM35) — Precision centigrade temperature sensor

---

*Built with OpenCode Zen (DeepSeek V4 Flash + MiMo V2.5)*
