"use client";

import { cn } from "@/lib/utils";

const severityStyles: Record<string, string> = {
  critical: "bg-red-100 text-red-800 border-red-300 dark:bg-red-950 dark:text-red-300 dark:border-red-800",
  warning: "bg-amber-100 text-amber-800 border-amber-300 dark:bg-amber-950 dark:text-amber-300 dark:border-amber-800",
  info: "bg-blue-100 text-blue-800 border-blue-300 dark:bg-blue-950 dark:text-blue-300 dark:border-blue-800",
  normal: "bg-emerald-100 text-emerald-800 border-emerald-300 dark:bg-emerald-950 dark:text-emerald-300 dark:border-emerald-800",
};

interface ConditionBadgeProps {
  name: string;
  severity?: string;
  detected?: boolean;
  probability?: number;
}

export default function ConditionBadge({ name, severity = "info", detected, probability }: ConditionBadgeProps) {
  const style = detected ? severityStyles[severity] || severityStyles.info : "bg-muted text-muted-foreground border-muted";

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors",
        style
      )}
    >
      {detected && (
        <span
          className={cn(
            "h-1.5 w-1.5 rounded-full",
            severity === "critical" ? "bg-red-500" :
            severity === "warning" ? "bg-amber-500" :
            "bg-blue-500"
          )}
        />
      )}
      {name}
      {probability !== undefined && (
        <span className="opacity-60 ml-0.5">{(probability * 100).toFixed(0)}%</span>
      )}
    </span>
  );
}
