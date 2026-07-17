"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { getThresholds, getHealth, type HealthResponse } from "@/lib/api";
import { Save, RefreshCw } from "lucide-react";

interface Threshold {
  key: string;
  label: string;
  value: number;
  unit: string;
}

export default function SettingsPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [thresholds, setThresholds] = useState<Threshold[]>([]);
  const [loading, setLoading] = useState(true);
  const [saved, setSaved] = useState(false);
  const [notifications, setNotifications] = useState(true);

  useEffect(() => {
    Promise.all([
      getHealth().catch(() => null),
      getThresholds().catch(() => null),
    ]).then(([h, t]) => {
      setHealth(h);
      if (t?.thresholds) {
        const mapped = Object.entries(t.thresholds).map(([key, value]) => ({
          key,
          label: key
            .replace(/_/g, " ")
            .toLowerCase()
            .replace(/\b\w/g, (l) => l.toUpperCase()),
          value: value as number,
          unit: key.includes("bpm") ? "BPM" : key.includes("temp") || key.includes("fever") ? "°C" : key.includes("spo2") ? "%" : key.includes("g") ? "g" : key.includes("sec") ? "s" : key.includes("ms") ? "ms" : "",
        }));
        setThresholds(mapped);
      }
      setLoading(false);
    });
  }, []);

  const updateThreshold = (key: string, newValue: string) => {
    setThresholds((prev) =>
      prev.map((t) =>
        t.key === key ? { ...t, value: parseFloat(newValue) || t.value } : t
      )
    );
  };

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Configure thresholds, notifications, and system preferences
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        {/* Clinical Thresholds */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="text-sm font-medium">Clinical Thresholds</CardTitle>
            <CardDescription>
              Adjust detection thresholds for each condition
            </CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex items-center justify-center py-8">
                <RefreshCw className="h-5 w-5 animate-spin text-muted-foreground" />
              </div>
            ) : (
              <div className="space-y-3">
                {thresholds.map((t) => (
                  <div
                    key={t.key}
                    className="flex items-center justify-between rounded-lg border p-3"
                  >
                    <div>
                      <p className="text-sm font-medium">{t.label}</p>
                      <p className="text-xs text-muted-foreground">{t.key}</p>
                    </div>
                    <div className="flex items-center gap-2">
                      <Input
                        type="number"
                        value={t.value}
                        onChange={(e) => updateThreshold(t.key, e.target.value)}
                        className="w-24 h-8 text-sm text-right"
                        step="0.1"
                      />
                      <span className="text-xs text-muted-foreground w-8">{t.unit}</span>
                    </div>
                  </div>
                ))}
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

        {/* Notifications */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Notifications</CardTitle>
            <CardDescription>Configure alert delivery</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">Browser Notifications</p>
                <p className="text-xs text-muted-foreground">Receive alerts in browser</p>
              </div>
              <Switch checked={notifications} onCheckedChange={setNotifications} />
            </div>
            <Separator />
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">Critical Alerts</p>
                <p className="text-xs text-muted-foreground">Low SpO₂, Fall Detection</p>
              </div>
              <Switch defaultChecked />
            </div>
            <Separator />
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">Warning Alerts</p>
                <p className="text-xs text-muted-foreground">Tachycardia, Fever, Irregular Rhythm</p>
              </div>
              <Switch defaultChecked />
            </div>
          </CardContent>
        </Card>

        {/* Model Info */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Model Versions</CardTitle>
            <CardDescription>Currently deployed ML models</CardDescription>
          </CardHeader>
          <CardContent>
            {health?.model_versions ? (
              <div className="space-y-3">
                {Object.entries(health.model_versions).map(([name, version]) => (
                  <div key={name} className="flex items-center justify-between rounded-lg border p-2.5">
                    <span className="text-sm capitalize">{name}</span>
                    <span className="text-xs font-mono text-muted-foreground">{version}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">No models loaded</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
