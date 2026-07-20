---
title: NeuraBand
emoji: "\U0001FA7A"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
---

# NeuraBand

AI-powered health symptom detection from wearable sensor data.

<!-- Screenshot placeholder -->
<!-- ![Screenshot](screenshot.png) -->

## How to Use

### Live Sensor Input

1. Paste your wearable sensor data in JSON format into the text box (or upload a `.json` file).
2. Click **Analyse**.
3. View colour-coded results showing detected conditions and their probability.

### Simulated Demo

1. Select a condition from the dropdown (e.g., Tachycardia, Low SpO2, Fever).
2. Choose a severity level (Mild / Moderate / Severe).
3. Click **Generate and Analyse** to see synthetic sensor data and predictions side by side.

### Monitor Dashboard

1. Paste sensor data JSON and click **Analyse and Update**.
2. View the live dashboard and alert history log.

## Detectable Conditions

| Condition | Detection Method | Clinical Basis |
|-----------|------------------|----------------|
| Tachycardia | ML (Cardiac Model) | Heart rate > 100 BPM |
| Irregular Heart Rhythm | ML (Cardiac Model) | RR interval variability |
| Low Blood Oxygen (SpO2) | ML (Respiratory Model) | SpO2 < 95% |
| Fever | Rule-based | Temperature >= 38.0 C |
| Fall Detection | ML (Activity Model) + Rules | Acceleration impact pattern |
| Sleep Problem | ML (Activity Model) | Restless motion + HR patterns |
| Fatigue | ML (Activity Model) | HRV + cross-sensor coupling |

## Sensor Configuration

Designed for a wrist-worn wearable with:

- **Accelerometer**: MPU6500, 3-axis, 50 Hz
- **Gyroscope**: MPU6500, 3-axis, 50 Hz
- **Heart Rate / SpO2 / PPG**: MAX30102, 25 Hz
- **Temperature**: STTS22H (digital) + LM35 (analog), 1 Hz

## Technology

- **ML Models**: LightGBM (scikit-learn compatible)
- **Features**: 162 engineered features (motion, HR, HRV, SpO2, temperature, cross-sensor, frequency domain)
- **Inference**: ~50 ms per 30-second window on CPU
- **Framework**: Python 3.10+, Gradio 4.x

---

> **IMPORTANT MEDICAL DISCLAIMER**
>
> **This application is NOT a medical device and does NOT provide medical
> diagnoses. The outputs are for informational and educational purposes ONLY.**
>
> **Always consult a qualified healthcare professional for medical advice,
> diagnosis, or treatment. Never disregard professional medical advice or
> delay seeking it because of information provided by this application.**
>
> **In case of a medical emergency, call your local emergency services
> immediately.**

---

## Badges

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
