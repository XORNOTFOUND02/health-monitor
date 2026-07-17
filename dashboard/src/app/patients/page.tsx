"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import ConditionBadge from "@/components/condition-badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Search, Plus, MoreHorizontal } from "lucide-react";

const patients = [
  { id: "P-001", name: "John Doe", device: "ESP32-001", status: "active", lastSeen: "2s ago", hr: 72, spo2: 98, temp: 36.6, alerts: 0 },
  { id: "P-002", name: "Jane Smith", device: "ESP32-002", status: "active", lastSeen: "5s ago", hr: 88, spo2: 95, temp: 37.2, alerts: 1 },
  { id: "P-003", name: "Bob Wilson", device: "ESP32-003", status: "inactive", lastSeen: "2h ago", hr: 0, spo2: 0, temp: 0, alerts: 0 },
  { id: "P-004", name: "Alice Brown", device: "ESP32-004", status: "active", lastSeen: "1s ago", hr: 65, spo2: 99, temp: 36.4, alerts: 0 },
  { id: "P-005", name: "Charlie Davis", device: "ESP32-005", status: "active", lastSeen: "10s ago", hr: 95, spo2: 92, temp: 37.8, alerts: 2 },
];

export default function PatientsPage() {
  const [search, setSearch] = useState("");

  const filtered = patients.filter((p) =>
    p.name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Patients</h1>
          <p className="text-sm text-muted-foreground mt-1">
            {patients.length} registered devices
          </p>
        </div>
        <Button>
          <Plus className="mr-2 h-4 w-4" />
          Add Patient
        </Button>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          placeholder="Search patients..."
          className="pl-9"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      {/* Patient Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {filtered.map((patient) => (
          <Card key={patient.id} className="hover:shadow-md transition-shadow">
            <CardContent className="p-4">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <Avatar>
                    <AvatarFallback className="bg-primary/10 text-primary text-sm">
                      {patient.name.split(" ").map((n) => n[0]).join("")}
                    </AvatarFallback>
                  </Avatar>
                  <div>
                    <p className="text-sm font-medium">{patient.name}</p>
                    <p className="text-xs text-muted-foreground">{patient.device}</p>
                  </div>
                </div>
                <Badge
                  variant={patient.status === "active" ? "default" : "secondary"}
                  className="text-xs"
                >
                  {patient.status}
                </Badge>
              </div>

              <div className="mt-4 grid grid-cols-3 gap-2 text-center">
                <div className="rounded bg-muted/50 p-2">
                  <p className="text-xs text-muted-foreground">HR</p>
                  <p className="text-sm font-semibold">{patient.hr || "—"}</p>
                </div>
                <div className="rounded bg-muted/50 p-2">
                  <p className="text-xs text-muted-foreground">SpO₂</p>
                  <p className="text-sm font-semibold">{patient.spo2 || "—"}</p>
                </div>
                <div className="rounded bg-muted/50 p-2">
                  <p className="text-xs text-muted-foreground">Temp</p>
                  <p className="text-sm font-semibold">{patient.temp || "—"}</p>
                </div>
              </div>

              <div className="mt-3 flex items-center justify-between">
                <span className="text-xs text-muted-foreground">
                  Last seen: {patient.lastSeen}
                </span>
                {patient.alerts > 0 && (
                  <ConditionBadge
                    name={`${patient.alerts} alert`}
                    severity="warning"
                    detected
                  />
                )}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
