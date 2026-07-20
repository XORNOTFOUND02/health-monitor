"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { getThresholds, getHealth, type HealthResponse } from "@/lib/api";
import { Save, RefreshCw, HeartPulse, Thermometer, Activity, Moon, AlertTriangle } from "lucide-react";

/* ── Human-readable labels for each threshold ────────────── */
const THRESHOLD_META: Record<string, { label: string; description: string; group: string; unit: string; icon?: any }> = {
  tachycardia_bpm: {
    label: "High Heart Rate Alert",
    description: "Heart rate above this value triggers a tachycardia warning",
    group: "Heart Rate",
    unit: "BPM",
    icon: HeartPulse,
  },
  bradycardia_bpm: {
    label: "Low Heart Rate Alert",
    description: "Heart rate below this value triggers a bradycardia warning",
    group: "Heart Rate",
    unit: "BPM",
    icon: HeartPulse,
  },
  low_spo2_threshold: {
    label: "Low Oxygen Warning",
    description: "Blood oxygen below this level triggers a warning",
    group: "Blood Oxygen",
    unit: "%",
    icon: Activity,
  },
  severe_low_spo2: {
    label: "Dangerously Low Oxygen",
    description: "Blood oxygen below this level triggers a critical alert",
    group: "Blood Oxygen",
    unit: "%",
    icon: Activity,
  },
  apnea_spo2_drop_threshold: {
    label: "Oxygen Drop (Apnea)",
    description: "How much oxygen must drop to suspect sleep apnea",
    group: "Blood Oxygen",
    unit: "%",
    icon: Moon,
  },
  fever_temp_c: {
    label: "Fever Threshold",
    description: "Body temperature above this triggers a fever alert",
    group: "Temperature",
    unit: "°C",
    icon: Thermometer,
  },
  low_grade_fever_c: {
    label: "Mild Fever Threshold",
    description: "Body temperature above this triggers a low-grade fever warning",
    group: "Temperature",
    unit: "°C",
    icon: Thermometer,
  },
  hypothermia_temp_c: {
    label: "Low Temperature Alert",
    description: "Body temperature below this triggers a hypothermia warning",
    group: "Temperature",
    unit: "°C",
    icon: Thermometer,
  },
  fall_accel_threshold_g: {
    label: "Fall Impact Force",
    description: "Minimum impact force (in G) to consider a possible fall",
    group: "Fall Detection",
    unit: "G",
    icon: AlertTriangle,
  },
  fall_stillness_duration_sec: {
    label: "Post-Fall Stillness Time",
    description: "How long the person must be still after impact to confirm a fall",
    group: "Fall Detection",
    unit: "sec",
    icon: AlertTriangle,
  },
  fall_stillness_threshold: {
    label: "Stillness Sensitivity",
    description: "How still the body must be to count as motionless (lower = stricter)",
    group: "Fall Detection",
    unit: "",
    icon: AlertTriangle,
  },
  sleep_motion_threshold: {
    label: "Sleep Motion Limit",
    description: "Maximum movement level to still count as sleeping",
    group: "Sleep & Fatigue",
    unit: "",
    icon: Moon,
  },
  fatigue_hrv_rmssd_threshold: {
    label: "Fatigue Heart Signal",
    description: "Heart rate variability below this suggests fatigue",
    group: "Sleep & Fatigue",
    unit: "ms",
    icon: Moon,
  },
  fatigue_resting_hr_elevation: {
    label: "Resting Heart Rate Rise",
    description: "How much resting heart rate must be elevated to suggest fatigue",
    group: "Sleep & Fatigue",
    unit: "BPM",
    icon: Moon,
  },
  irregular_rr_cv_threshold: {
    label: "Irregular Rhythm Sensitivity",
    description: "Heart rhythm variability above this triggers an irregular rhythm warning",
    group: "Heart Rhythm",
    unit: "",
    icon: HeartPulse,
  },
};

const GROUP_ORDER = ["Heart Rate", "Blood Oxygen", "Temperature", "Fall Detection", "Sleep & Fatigue", "Heart Rhythm"];

export default function SettingsPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [thresholds, setThresholds] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(true);
  const [saved, setSaved] = useState(false);
  const [notifications, setNotifications] = useState(true);
  const [criticalAlerts, setCriticalAlerts] = useState(true);
  const [warningAlerts, setWarningAlerts] = useState(true);

  useEffect(() => {
    Promise.all([
      getHealth().catch(() => null),
      getThresholds().catch(() => null),
    ]).then(([h, t]) => {
      setHealth(h);
      if (t?.thresholds) setThresholds(t.thresholds);
      setLoading(false);
    });
  }, []);

  const updateThreshold = (key: string, newValue: string) => {
    setThresholds((prev) => ({ ...prev, [key]: parseFloat(newValue) || prev[key] }));
  };

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  // Group thresholds by category
  const grouped: Record<string, [string, number][]> = {};
  for (const key of GROUP_ORDER) grouped[key] = [];
  for (const [key, value] of Object.entries(thresholds)) {
    const meta = THRESHOLD_META[key];
    const group = meta?.group || "Other";
    if (!grouped[group]) grouped[group] = [];
    grouped[group].push([key, value]);
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Adjust how NeuraBand detects health conditions and sends alerts
        </p>
      </div>

      {/* ── How It Works ─────────────────────────────────── */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium">How Detection Works</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground leading-relaxed">
            NeuraBand continuously analyzes data from your wearable sensors — heart rate, blood oxygen,
            body temperature, movement, and magnetic orientation. It uses trained AI models to detect
            patterns that may indicate a health issue. The settings below let you adjust how sensitive
            each detection is. <strong>When in doubt, keep the defaults.</strong>
          </p>
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* ── Thresholds (takes 2 cols) ─────────────────── */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="text-sm font-medium">Alert Sensitivity</CardTitle>
            <CardDescription>
              Adjust when NeuraBand should warn you. Lower values = more sensitive. Higher = fewer false alarms.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex items-center justify-center py-8">
                <RefreshCw className="h-5 w-5 animate-spin text-muted-foreground" />
              </div>
            ) : (
              <div className="space-y-6">
                {GROUP_ORDER.map((group) => {
                  const items = grouped[group];
                  if (!items || items.length === 0) return null;
                  return (
                    <div key={group}>
                      <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">
                        {group}
                      </h3>
                      <div className="space-y-3">
                        {items.map(([key, value]) => {
                          const meta = THRESHOLD_META[key];
                          const Icon = meta?.icon;
                          return (
                            <div
                              key={key}
                              className="flex items-center gap-4 rounded-lg border p-3"
                            >
                              {Icon && (
                                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-muted">
                                  <Icon className="h-4 w-4 text-muted-foreground" />
                                </div>
                              )}
                              <div className="flex-1 min-w-0">
                                <p className="text-sm font-medium">
                                  {meta?.label || key}
                                </p>
                                <p className="text-xs text-muted-foreground leading-relaxed">
                                  {meta?.description || key}
                                </p>
                              </div>
                              <div className="flex items-center gap-2 shrink-0">
                                <Input
                                  type="number"
                                  value={value}
                                  onChange={(e) => updateThreshold(key, e.target.value)}
                                  className="w-20 h-8 text-sm text-right"
                                  step="0.1"
                                />
                                {meta?.unit && (
                                  <span className="text-xs text-muted-foreground w-8">{meta.unit}</span>
                                )}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                      {group !== "Heart Rhythm" && <Separator className="mt-4" />}
                    </div>
                  );
                })}
                <Button onClick={handleSave} className="mt-2">
                  {saved ? (
                    <>Saved ✓</>
                  ) : (
                    <>
                      <Save className="mr-2 h-4 w-4" />
                      Save Changes
                    </>
                  )}
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* ── Right sidebar ─────────────────────────────── */}
        <div className="space-y-4">
          {/* Notifications */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm font-medium">Alerts & Notifications</CardTitle>
              <CardDescription>Choose which alerts you want to receive</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium">Browser Notifications</p>
                  <p className="text-xs text-muted-foreground">Show pop-up alerts in your browser</p>
                </div>
                <Switch checked={notifications} onCheckedChange={setNotifications} />
              </div>
              <Separator />
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-red-600 dark:text-red-400">Emergency Alerts</p>
                  <p className="text-xs text-muted-foreground">Low oxygen, dangerous falls</p>
                </div>
                <Switch checked={criticalAlerts} onCheckedChange={setCriticalAlerts} />
              </div>
              <Separator />
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-amber-600 dark:text-amber-400">Warning Alerts</p>
                  <p className="text-xs text-muted-foreground">Fast/slow heart rate, fever, irregular rhythm</p>
                </div>
                <Switch checked={warningAlerts} onCheckedChange={setWarningAlerts} />
              </div>
            </CardContent>
          </Card>

          {/* Model Info */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm font-medium">AI Models</CardTitle>
              <CardDescription>The AI engines powering your health detection</CardDescription>
            </CardHeader>
            <CardContent>
              {health?.model_versions ? (
                <div className="space-y-3">
                  {Object.entries(health.model_versions).map(([name, version]) => {
                    const friendlyNames: Record<string, { label: string; desc: string }> = {
                      cardiac: { label: "Heart Monitor", desc: "Detects fast/slow heart rate and irregular rhythm" },
                      respiratory: { label: "Breathing Monitor", desc: "Detects low blood oxygen levels" },
                      activity: { label: "Movement Monitor", desc: "Detects falls, fatigue, and sleep problems" },
                    };
                    const info = friendlyNames[name] || { label: name, desc: "" };
                    return (
                      <div key={name} className="rounded-lg border p-3">
                        <div className="flex items-center justify-between">
                          <p className="text-sm font-medium">{info.label}</p>
                          <span className="text-[10px] font-mono text-muted-foreground bg-muted px-1.5 py-0.5 rounded">
                            v{version}
                          </span>
                        </div>
                        <p className="text-xs text-muted-foreground mt-0.5">{info.desc}</p>
                      </div>
                    );
                  })}
                  <p className="text-[10px] text-muted-foreground text-center pt-1">
                    Status: {health.models_loaded ? "All models active" : "Models offline"}
                  </p>
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">No models loaded</p>
              )}
            </CardContent>
          </Card>

          {/* System Info */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm font-medium">System Info</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Uptime</span>
                <span>{health ? `${Math.floor(health.uptime_seconds / 60)}m` : "—"}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Total Analyses</span>
                <span>{health?.total_inferences ?? "—"}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Avg Speed</span>
                <span>{health ? `${health.avg_inference_ms}ms` : "—"}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Conditions Tracked</span>
                <span>7</span>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
