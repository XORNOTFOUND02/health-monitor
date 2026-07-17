"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  LineChart,
  Line,
  Legend,
} from "recharts";
import { Download, Calendar } from "lucide-react";

// Mock historical data
const hrData = Array.from({ length: 24 }, (_, i) => ({
  time: `${i}:00`,
  heartRate: Math.round(70 + Math.sin(i / 4) * 15 + Math.random() * 10),
  spo2: Math.round(96 + Math.sin(i / 6) * 2 + Math.random() * 2),
}));

const conditionHistory = [
  { date: "Mon", tachycardia: 2, hypoxia: 0, fever: 1, fatigue: 3 },
  { date: "Tue", tachycardia: 1, hypoxia: 1, fever: 0, fatigue: 2 },
  { date: "Wed", tachycardia: 0, hypoxia: 0, fever: 2, fatigue: 1 },
  { date: "Thu", tachycardia: 3, hypoxia: 0, fever: 0, fatigue: 4 },
  { date: "Fri", tachycardia: 1, hypoxia: 2, fever: 1, fatigue: 2 },
  { date: "Sat", tachycardia: 0, hypoxia: 0, fever: 0, fatigue: 1 },
  { date: "Sun", tachycardia: 1, hypoxia: 0, fever: 0, fatigue: 2 },
];

export default function HistoryPage() {
  const [timeRange, setTimeRange] = useState("24h");

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">History</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Historical data and trends
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex rounded-lg border p-0.5">
            {["24h", "7d", "30d"].map((range) => (
              <Button
                key={range}
                variant={timeRange === range ? "secondary" : "ghost"}
                size="sm"
                className="px-3"
                onClick={() => setTimeRange(range)}
              >
                {range}
              </Button>
            ))}
          </div>
          <Button variant="outline" size="sm">
            <Download className="mr-2 h-4 w-4" />
            Export
          </Button>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        {/* Heart Rate Trend */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Heart Rate & SpO₂</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={hrData}>
                  <defs>
                    <linearGradient id="hrGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="spo2Grad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="time" className="text-xs" tick={{ fontSize: 11 }} />
                  <YAxis className="text-xs" tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Area
                    type="monotone"
                    dataKey="heartRate"
                    stroke="#ef4444"
                    fill="url(#hrGrad)"
                    strokeWidth={2}
                    name="Heart Rate (BPM)"
                  />
                  <Area
                    type="monotone"
                    dataKey="spo2"
                    stroke="#3b82f6"
                    fill="url(#spo2Grad)"
                    strokeWidth={2}
                    name="SpO₂ (%)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Condition Frequency */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Condition Frequency (7 Days)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={conditionHistory}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="date" className="text-xs" tick={{ fontSize: 11 }} />
                  <YAxis className="text-xs" tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="tachycardia" fill="#ef4444" name="Tachycardia" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="hypoxia" fill="#3b82f6" name="Hypoxia" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="fever" fill="#f59e0b" name="Fever" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="fatigue" fill="#8b5cf6" name="Fatigue" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Temperature Trend */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Temperature Trend</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[250px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={Array.from({ length: 24 }, (_, i) => ({
                    time: `${i}:00`,
                    temp: 36.5 + Math.sin(i / 5) * 0.3 + Math.random() * 0.2,
                  }))}
                >
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="time" className="text-xs" tick={{ fontSize: 11 }} />
                  <YAxis domain={[36, 38]} className="text-xs" tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="temp"
                    stroke="#f59e0b"
                    strokeWidth={2}
                    dot={false}
                    name="Temperature (°C)"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Summary Stats */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">7-Day Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex justify-between rounded-lg border p-3">
                <span className="text-sm">Avg Heart Rate</span>
                <span className="text-sm font-semibold">74 BPM</span>
              </div>
              <div className="flex justify-between rounded-lg border p-3">
                <span className="text-sm">Avg SpO₂</span>
                <span className="text-sm font-semibold">97.2%</span>
              </div>
              <div className="flex justify-between rounded-lg border p-3">
                <span className="text-sm">Avg Temperature</span>
                <span className="text-sm font-semibold">36.6°C</span>
              </div>
              <div className="flex justify-between rounded-lg border p-3">
                <span className="text-sm">Total Alerts</span>
                <span className="text-sm font-semibold">28</span>
              </div>
              <div className="flex justify-between rounded-lg border p-3">
                <span className="text-sm">Most Frequent Condition</span>
                <span className="text-sm font-semibold text-amber-600">Fatigue</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
