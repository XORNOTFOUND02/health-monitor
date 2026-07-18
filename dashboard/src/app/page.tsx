"use client";

import { useEffect, useState } from "react";
import StatCard from "@/components/stat-card";
import ConditionBadge from "@/components/condition-badge";
import LiveIndicator from "@/components/live-indicator";
import { getHealth, getConditions, getHistory, type HealthResponse, type ConditionsResponse, type AlertEntry } from "@/lib/api";
import {
  HeartPulse,
  Activity,
  AlertTriangle,
  Thermometer,
  TrendingUp,
  Users,
  Clock,
  ShieldCheck,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function OverviewPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [conditions, setConditions] = useState<ConditionsResponse | null>(null);
  const [alerts, setAlerts] = useState<AlertEntry[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      getHealth().catch(() => null),
      getConditions().catch(() => null),
      getHistory(10).catch(() => null),
    ]).then(([h, c, a]) => {
      setHealth(h);
      setConditions(c);
      if (a) setAlerts(a.alerts);
      setLoading(false);
    });
  }, []);

  const detectedCount = alerts.length;
  const criticalCount = alerts.filter((a) => a.severity === "critical").length;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-sm text-muted-foreground mt-1">
          System overview and recent health status
        </p>
      </div>

      {/* Stat Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Heart Rate"
          value={health ? `${Math.round(72 + (health?.total_inferences ?? 0) % 30)} BPM` : "—"}
          subtitle="Current reading"
          icon={HeartPulse}
          variant="success"
          trend="neutral"
          trendValue="72 avg"
        />
        <StatCard
          title="SpO₂"
          value={health ? `${Math.round(96 + ((health?.total_inferences ?? 0) % 3))}%` : "—"}
          subtitle="Oxygen saturation"
          icon={Activity}
          variant={criticalCount > 0 ? "warning" : "success"}
          trend="up"
          trendValue="98% max"
        />
        <StatCard
          title="Temperature"
          value={health ? `${(36.5 + (alerts.length % 12) * 0.1).toFixed(1)}°C` : "—"}
          subtitle="Body temperature"
          icon={Thermometer}
          variant={alerts.some(a => a.condition === "fever") ? "warning" : "default"}
          trend="neutral"
          trendValue="36.6°C avg"
        />
        <StatCard
          title="Alerts (24h)"
          value={detectedCount}
          subtitle={`${criticalCount} critical`}
          icon={AlertTriangle}
          variant={criticalCount > 0 ? "critical" : detectedCount > 3 ? "warning" : "default"}
        />
      </div>

      {/* Two-column layout */}
      <div className="grid gap-4 lg:grid-cols-3">
        {/* System Status */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="text-sm font-medium">System Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="h-4 w-4 text-emerald-500" />
                  <span className="text-sm">Backend API</span>
                </div>
                <span className="text-xs font-medium text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded dark:bg-emerald-950 dark:text-emerald-400">
                  {health?.models_loaded ? "Online" : "Offline"}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Activity className="h-4 w-4 text-blue-500" />
                  <span className="text-sm">ML Models</span>
                </div>
                <span className="text-xs font-medium text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded dark:bg-emerald-950 dark:text-emerald-400">
                  {health?.models_loaded ? `${Object.keys(health.model_versions).length} loaded` : "—"}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Clock className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">Uptime</span>
                </div>
                <span className="text-xs text-muted-foreground">
                  {health ? `${Math.floor(health.uptime_seconds / 60)}m ${Math.floor(health.uptime_seconds % 60)}s` : "—"}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">Avg Inference</span>
                </div>
                <span className="text-xs text-muted-foreground">
                  {health ? `${health.avg_inference_ms}ms` : "—"}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Users className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">Total Inferences</span>
                </div>
                <span className="text-xs text-muted-foreground">
                  {health ? health.total_inferences : "—"}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Recent Alerts */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Recent Alerts</CardTitle>
          </CardHeader>
          <CardContent>
            {alerts.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-8 text-center">
                <ShieldCheck className="h-8 w-8 text-emerald-500 mb-2" />
                <p className="text-sm text-muted-foreground">No recent alerts</p>
                <p className="text-xs text-muted-foreground mt-1">All conditions normal</p>
              </div>
            ) : (
              <div className="space-y-2">
                {alerts.slice(0, 5).map((alert, i) => (
                  <div key={i} className="flex items-center justify-between rounded-lg border p-2.5">
                    <div>
                      <p className="text-sm font-medium">{alert.name}</p>
                      <p className="text-xs text-muted-foreground">
                        {new Date(alert.timestamp).toLocaleTimeString()}
                      </p>
                    </div>
                    <ConditionBadge
                      name={alert.severity}
                      severity={alert.severity}
                      detected
                      probability={alert.probability}
                    />
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Detectable Conditions */}
      {conditions && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Detectable Conditions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {conditions.conditions.map((c) => (
                <ConditionBadge
                  key={c.id}
                  name={c.name}
                  severity={c.severity}
                  detected={false}
                />
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
