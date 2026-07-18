"use client";

import { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";

interface SensorWaveformProps {
  /** Raw sensor data from the demo endpoint */
  sensorData: Record<string, any>;
}

/**
 * Downsample an array by picking every `step`-th element.
 * Keeps the first and last point to preserve the visual range.
 */
function downsample<T>(data: T[], step: number): T[] {
  if (step <= 1) return data;
  const result: T[] = [];
  for (let i = 0; i < data.length; i += step) {
    result.push(data[i]);
  }
  // ensure last point is included
  if (result[result.length - 1] !== data[data.length - 1]) {
    result.push(data[data.length - 1]);
  }
  return result;
}

/* ── Accelerometer ──────────────────────────────────────── */
function AccelChart({ data }: { data: number[][] }) {
  const chartData = useMemo(() => {
    const step = Math.max(1, Math.floor(data.length / 200));
    return downsample(data, step).map(([ax, ay, az], i) => ({
      t: i,
      ax,
      ay,
      az,
    }));
  }, [data]);

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium">MPU6500 — Accelerometer</CardTitle>
        <CardDescription>3-axis acceleration (m/s²) · 50 Hz</CardDescription>
      </CardHeader>
      <CardContent className="p-2 pt-0">
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={chartData} margin={{ top: 5, right: 8, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
            <XAxis dataKey="t" tick={false} axisLine={false} />
            <YAxis width={50} tick={{ fontSize: 10 }} />
            <Tooltip
              contentStyle={{ fontSize: 12 }}
              formatter={(value: number) => value.toFixed(2)}
            />
            <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
            <Line
              type="monotone"
              dataKey="ax"
              stroke="#ef4444"
              name="X"
              dot={false}
              strokeWidth={1}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="ay"
              stroke="#22c55e"
              name="Y"
              dot={false}
              strokeWidth={1}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="az"
              stroke="#3b82f6"
              name="Z"
              dot={false}
              strokeWidth={1}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

/* ── Gyroscope ──────────────────────────────────────────── */
function GyroChart({ data }: { data: number[][] }) {
  const chartData = useMemo(() => {
    const step = Math.max(1, Math.floor(data.length / 200));
    return downsample(data, step).map(([gx, gy, gz], i) => ({
      t: i,
      gx,
      gy,
      gz,
    }));
  }, [data]);

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium">MPU6500 — Gyroscope</CardTitle>
        <CardDescription>3-axis angular velocity (rad/s) · 50 Hz</CardDescription>
      </CardHeader>
      <CardContent className="p-2 pt-0">
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={chartData} margin={{ top: 5, right: 8, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
            <XAxis dataKey="t" tick={false} axisLine={false} />
            <YAxis width={50} tick={{ fontSize: 10 }} />
            <Tooltip
              contentStyle={{ fontSize: 12 }}
              formatter={(value: number) => value.toFixed(4)}
            />
            <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
            <Line
              type="monotone"
              dataKey="gx"
              stroke="#f59e0b"
              name="X"
              dot={false}
              strokeWidth={1}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="gy"
              stroke="#8b5cf6"
              name="Y"
              dot={false}
              strokeWidth={1}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="gz"
              stroke="#ec4899"
              name="Z"
              dot={false}
              strokeWidth={1}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

/* ── Magnetometer ────────────────────────────────────────── */
function MagChart({ data }: { data: number[][] }) {
  const chartData = useMemo(() => {
    const step = Math.max(1, Math.floor(data.length / 200));
    return downsample(data, step).map(([mx, my, mz], i) => ({
      t: i,
      mx,
      my,
      mz,
    }));
  }, [data]);

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium">HMC5883L — Magnetometer</CardTitle>
        <CardDescription>3-axis magnetic field (µT) · 25 Hz</CardDescription>
      </CardHeader>
      <CardContent className="p-2 pt-0">
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={chartData} margin={{ top: 5, right: 8, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
            <XAxis dataKey="t" tick={false} axisLine={false} />
            <YAxis width={50} tick={{ fontSize: 10 }} />
            <Tooltip
              contentStyle={{ fontSize: 12 }}
              formatter={(value: number) => value.toFixed(1)}
            />
            <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
            <Line
              type="monotone"
              dataKey="mx"
              stroke="#14b8a6"
              name="X"
              dot={false}
              strokeWidth={1}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="my"
              stroke="#f97316"
              name="Y"
              dot={false}
              strokeWidth={1}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="mz"
              stroke="#6366f1"
              name="Z"
              dot={false}
              strokeWidth={1}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

/* ── PPG Waveform ────────────────────────────────────────── */
function PpgChart({ data }: { data: number[] }) {
  const chartData = useMemo(() => {
    const step = Math.max(1, Math.floor(data.length / 200));
    return downsample(data, step).map((val, i) => ({ t: i, ppg: val }));
  }, [data]);

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium">MAX30102 — PPG Waveform</CardTitle>
        <CardDescription>Infrared photoplethysmogram · 25 Hz</CardDescription>
      </CardHeader>
      <CardContent className="p-2 pt-0">
        <ResponsiveContainer width="100%" height={150}>
          <LineChart data={chartData} margin={{ top: 5, right: 8, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
            <XAxis dataKey="t" tick={false} axisLine={false} />
            <YAxis width={50} tick={{ fontSize: 10 }} />
            <Tooltip
              contentStyle={{ fontSize: 12 }}
              formatter={(value: number) => value.toFixed(2)}
            />
            <Line
              type="monotone"
              dataKey="ppg"
              stroke="#ef4444"
              name="PPG"
              dot={false}
              strokeWidth={1}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

/* ── Main Export ─────────────────────────────────────────── */
export default function SensorWaveform({ sensorData }: SensorWaveformProps) {
  if (!sensorData) return null;

  const accel = sensorData.accelerometer;
  const gyro = sensorData.gyroscope;
  const mag = sensorData.magnetometer;
  const ppg = sensorData.ppg;

  const hasAny =
    (accel && accel.length > 0) ||
    (gyro && gyro.length > 0) ||
    (mag && mag.length > 0) ||
    (ppg && ppg.length > 0);

  if (!hasAny) return null;

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold tracking-tight">Sensor Waveforms</h2>
        <p className="text-xs text-muted-foreground">
          Raw time-series data from wearable sensors for the current analysis window
        </p>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        {accel && accel.length > 0 && <AccelChart data={accel} />}
        {gyro && gyro.length > 0 && <GyroChart data={gyro} />}
        {mag && mag.length > 0 && <MagChart data={mag} />}
        {ppg && ppg.length > 0 && <PpgChart data={ppg} />}
      </div>
    </div>
  );
}
