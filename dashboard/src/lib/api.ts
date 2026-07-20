/**
 * API client for the NeuraBand backend.
 * Falls back to mock data when the backend is unreachable.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type FetchOptions = RequestInit & { timeout?: number };

async function fetchApi<T>(path: string, opts: FetchOptions = {}): Promise<T> {
  const { timeout = 10000, ...fetchOpts } = opts;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);

  try {
    const res = await fetch(`${API_BASE}${path}`, {
      ...fetchOpts,
      signal: controller.signal,
      headers: { "Content-Type": "application/json", ...fetchOpts.headers },
    });
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`${res.status}: ${body.slice(0, 200)}`);
    }
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

// ── Types ──

export interface ConditionInfo {
  id: string;
  name: string;
  severity: "critical" | "warning" | "info";
  detection_method: "ML" | "rule";
}

export interface ConditionsResponse {
  conditions: ConditionInfo[];
  thresholds: Record<string, number>;
  window_config: {
    duration_sec: number;
    stride_sec: number;
    sampling_rates: Record<string, number>;
  };
}

export interface Prediction {
  status: string;
  timestamp: string;
  data_quality_score: number;
  predictions: Record<string, {
    detected: boolean;
    probability: number;
    confidence: number;
  }>;
  alert_summary: {
    critical_alerts: string[];
    warnings: string[];
    info: string[];
    total_alerts: number;
  };
  processing_time_ms: number;
}

export interface AlertEntry {
  condition: string;
  name: string;
  severity: string;
  probability: number;
  confidence: number;
  timestamp: string;
}

export interface HistoryResponse {
  alerts: AlertEntry[];
  total: number;
  filter: string;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  model_versions: Record<string, string>;
  conditions: string[];
  uptime_seconds: number;
  total_inferences: number;
  avg_inference_ms: number;
  models_loaded: boolean;
}

export interface DemoData {
  sensor_data: any;
  condition: string;
  severity: string;
}

// ── API Functions ──

export async function getHealth(): Promise<HealthResponse> {
  return fetchApi<HealthResponse>("/health");
}

export async function getConditions(): Promise<ConditionsResponse> {
  return fetchApi<ConditionsResponse>("/conditions");
}

export async function predict(data: any): Promise<Prediction> {
  return fetchApi<Prediction>("/predict", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function generateDemo(condition: string = "normal", severity: number = 1): Promise<DemoData> {
  return fetchApi<DemoData>(`/demo/generate?condition=${condition}&severity=${severity}`);
}

export async function getHistory(limit: number = 50, severity?: string): Promise<HistoryResponse> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (severity) params.set("severity", severity);
  return fetchApi<HistoryResponse>(`/history?${params}`);
}

export async function getThresholds(): Promise<{ thresholds: Record<string, number> }> {
  return fetchApi("/config/thresholds");
}

// ── Mock Data (for when backend is offline) ──

export const MOCK_PREDICTION: Prediction = {
  status: "success",
  timestamp: new Date().toISOString(),
  data_quality_score: 0.94,
  predictions: {
    tachycardia: { detected: false, probability: 0.08, confidence: 0.94 },
    irregular_rhythm: { detected: false, probability: 0.12, confidence: 0.94 },
    low_spo2: { detected: false, probability: 0.05, confidence: 0.94 },
    fever: { detected: true, probability: 0.76, confidence: 0.88 },
    fall_detected: { detected: false, probability: 0.02, confidence: 0.94 },
    sleep_problem: { detected: true, probability: 0.62, confidence: 0.85 },
    fatigue: { detected: false, probability: 0.18, confidence: 0.94 },
  },
  alert_summary: {
    critical_alerts: [],
    warnings: ["fever"],
    info: ["sleep_problem"],
    total_alerts: 2,
  },
  processing_time_ms: 42.3,
};
