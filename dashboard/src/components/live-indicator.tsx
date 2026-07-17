"use client";

import { useEffect, useState } from "react";

interface LiveIndicatorProps {
  connected: boolean;
  lastSeen?: number; // unix timestamp
}

export default function LiveIndicator({ connected, lastSeen }: LiveIndicatorProps) {
  const [elapsed, setElapsed] = useState("");

  useEffect(() => {
    if (!lastSeen) return;
    const update = () => {
      const sec = Math.floor((Date.now() - lastSeen) / 1000);
      setElapsed(sec < 60 ? `${sec}s ago` : `${Math.floor(sec / 60)}m ago`);
    };
    update();
    const interval = setInterval(update, 5000);
    return () => clearInterval(interval);
  }, [lastSeen]);

  return (
    <div className="flex items-center gap-2">
      <span className={`relative flex h-3 w-3 ${connected ? "" : "opacity-50"}`}>
        <span
          className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
            connected ? "bg-emerald-400" : "bg-red-400"
          }`}
        />
        <span
          className={`relative inline-flex rounded-full h-3 w-3 ${
            connected ? "bg-emerald-500" : "bg-red-500"
          }`}
        />
      </span>
      <span className="text-xs font-medium text-muted-foreground">
        {connected ? "Live" : "Disconnected"}
        {elapsed && ` · ${elapsed}`}
      </span>
    </div>
  );
}
