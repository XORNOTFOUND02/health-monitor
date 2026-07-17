---
title: Health Monitor API
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: fastapi
app_file: backend/app.py
pinned: false
---

# Health Monitor API

FastAPI backend for real-time health monitoring inference.

## Deploy on HF Spaces

1. Create a new Space → Docker
2. Upload the entire repo (including `src/`, `models/`, `backend/`)
3. Set `PORT = 7860` env var
4. Space starts automatically at `https://{username}-health-monitor-api.hf.space`

## Local Dev

```bash
cd backend
pip install -r requirements.txt
python app.py  # Runs on port 7860 by default
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API root |
| GET | `/health` | Health check |
| GET | `/conditions` | List detectable conditions |
| POST | `/predict` | Run inference on sensor data |
| GET | `/demo/generate` | Generate demo sensor data |
| GET | `/history` | Get alert history |
| GET | `/config/thresholds` | Get clinical thresholds |
| PUT | `/config/thresholds` | Update a threshold |
| WS | `/ws` | Real-time streaming inference |
