"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import ConditionBadge from "@/components/condition-badge";
import LiveIndicator from "@/components/live-indicator";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { generateDemo, predict, MOCK_PREDICTION, type Prediction } from "@/lib/api";
import { HeartPulse, Activity, Thermometer, Play, RefreshCw } from "lucide-react";

const conditions = [
  "normal", "tachycardia", "bradycardia", "hypoxia",
  "hyperventilation", "atrial_fibrillation", "fatigue",
];

export default function MonitorPage() {
  const [selectedCondition, setSelectedCondition] = useState("normal");
  const [severity, setSeverity] = useState("1");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [useMock, setUseMock] = useState(false);

  const runInference = useCallback(async () => {
    setLoading(true);
    try {
      if (useMock) {
        await new Promise((r) => setTimeout(r, 500));
        setPrediction(MOCK_PREDICTION);
      } else {
        const demo = await generateDemo(selectedCondition, parseInt(severity));
        const result = await predict(demo.sensor_data);
        setPrediction(result);
      }
    } catch {
      // Fallback to mock
      await new Promise((r) => setTimeout(r, 500));
      setPrediction(MOCK_PREDICTION);
    } finally {
      setLoading(false);
    }
  }, [selectedCondition, severity, useMock]);

  useEffect(() => {
    runInference();
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Live Monitor</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Real-time sensor data analysis and condition detection
        </p>
      </div>

      {/* Controls */}
      <Card>
        <CardContent className="p-4">
          <div className="flex flex-wrap items-end gap-4">
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">Condition</label>
              <Select value={selectedCondition} onValueChange={(v) => v && setSelectedCondition(v)}>
                <SelectTrigger className="w-44">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {conditions.map((c) => (
                    <SelectItem key={c} value={c}>
                      {c.replace("_", " ").replace(/\b\w/g, (l) => l.toUpperCase())}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">Severity</label>
              <Select value={severity} onValueChange={(v) => v && setSeverity(v)}>
                <SelectTrigger className="w-28">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1">Mild</SelectItem>
                  <SelectItem value="2">Moderate</SelectItem>
                  <SelectItem value="3">Severe</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <Button onClick={runInference} disabled={loading}>
              {loading ? (
                <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Play className="mr-2 h-4 w-4" />
              )}
              {loading ? "Analyzing..." : "Run Analysis"}
            </Button>

            <div className="flex items-center gap-2 ml-auto">
              <LiveIndicator connected={!useMock} lastSeen={Date.now()} />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      {prediction && (
        <div className="grid gap-4 lg:grid-cols-2">
          {/* Vital Signs */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm font-medium">Vital Signs</CardTitle>
              <CardDescription>Simulated readings for this window</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-4">
                <div className="rounded-lg border p-3 text-center">
                  <HeartPulse className="mx-auto h-5 w-5 text-red-500 mb-1" />
                  <p className="text-2xl font-bold">{Math.round(72 + Math.random() * 30)}</p>
                  <p className="text-xs text-muted-foreground">BPM</p>
                </div>
                <div className="rounded-lg border p-3 text-center">
                  <Activity className="mx-auto h-5 w-5 text-blue-500 mb-1" />
                  <p className="text-2xl font-bold">{Math.round(95 + Math.random() * 5)}</p>
                  <p className="text-xs text-muted-foreground">SpO₂ %</p>
                </div>
                <div className="rounded-lg border p-3 text-center">
                  <Thermometer className="mx-auto h-5 w-5 text-amber-500 mb-1" />
                  <p className="text-2xl font-bold">{(36.5 + Math.random() * 1.2).toFixed(1)}</p>
                  <p className="text-xs text-muted-foreground">°C</p>
                </div>
              </div>

              <div className="mt-4 text-xs text-muted-foreground">
                <p>Data Quality: {(prediction.data_quality_score * 100).toFixed(0)}%</p>
                <p>Processing: {prediction.processing_time_ms}ms</p>
              </div>
            </CardContent>
          </Card>

          {/* Detected Conditions */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm font-medium">Detected Conditions</CardTitle>
              <CardDescription>
                Status: <span className={prediction.status === "success" ? "text-emerald-500" : "text-red-500"}>{prediction.status.toUpperCase()}</span>
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {Object.entries(prediction.predictions).map(([key, val]) => (
                  <div
                    key={key}
                    className="flex items-center justify-between rounded-lg border p-2.5"
                  >
                    <span className="text-sm font-medium capitalize">
                      {key.replace(/_/g, " ")}
                    </span>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-muted-foreground">
                        {(val.probability * 100).toFixed(0)}%
                      </span>
                      <ConditionBadge
                        name={val.detected ? "Detected" : "Normal"}
                        severity={
                          val.detected
                            ? ["low_spo2", "fall_detected"].includes(key)
                              ? "critical"
                              : ["tachycardia", "irregular_rhythm", "fever"].includes(key)
                              ? "warning"
                              : "info"
                            : "normal"
                        }
                        detected={val.detected}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Alert Summary */}
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="text-sm font-medium">Alert Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-4">
                <div className="rounded-lg border border-red-200 bg-red-50/50 p-3 dark:border-red-900 dark:bg-red-950/20">
                  <p className="text-xs font-medium text-red-600 dark:text-red-400">Critical</p>
                  <p className="text-lg font-bold mt-1">{prediction.alert_summary.critical_alerts.length}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {prediction.alert_summary.critical_alerts.join(", ") || "None"}
                  </p>
                </div>
                <div className="rounded-lg border border-amber-200 bg-amber-50/50 p-3 dark:border-amber-900 dark:bg-amber-950/20">
                  <p className="text-xs font-medium text-amber-600 dark:text-amber-400">Warnings</p>
                  <p className="text-lg font-bold mt-1">{prediction.alert_summary.warnings.length}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {prediction.alert_summary.warnings.join(", ") || "None"}
                  </p>
                </div>
                <div className="rounded-lg border border-blue-200 bg-blue-50/50 p-3 dark:border-blue-900 dark:bg-blue-950/20">
                  <p className="text-xs font-medium text-blue-600 dark:text-blue-400">Info</p>
                  <p className="text-lg font-bold mt-1">{prediction.alert_summary.info.length}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {prediction.alert_summary.info.join(", ") || "None"}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
