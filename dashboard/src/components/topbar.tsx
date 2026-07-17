"use client";

import { useEffect, useState } from "react";
import LiveIndicator from "@/components/live-indicator";
import { getHealth, type HealthResponse } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { RefreshCw } from "lucide-react";

export default function Topbar() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchHealth = async () => {
    try {
      const data = await getHealth();
      setHealth(data);
    } catch {
      setHealth(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="flex h-14 items-center justify-between border-b bg-card px-6">
      <div className="flex items-center gap-4">
        <h1 className="text-sm font-medium text-muted-foreground">
          {health?.models_loaded
            ? `${health.total_inferences} inferences · avg ${health.avg_inference_ms}ms`
            : "Connecting to backend..."}
        </h1>
      </div>

      <div className="flex items-center gap-4">
        <LiveIndicator connected={!!health?.models_loaded} lastSeen={Date.now()} />
        <Button variant="ghost" size="icon" onClick={fetchHealth} disabled={loading}>
          <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
        </Button>
      </div>
    </header>
  );
}

function cn(...classes: (string | boolean | undefined | null)[]): string {
  return classes.filter(Boolean).join(" ");
}
