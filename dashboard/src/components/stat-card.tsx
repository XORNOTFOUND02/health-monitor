"use client";

import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { LucideIcon } from "lucide-react";

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
  trend?: "up" | "down" | "neutral";
  trendValue?: string;
  variant?: "default" | "warning" | "critical" | "success";
}

const variantStyles = {
  default: "border-border",
  warning: "border-amber-400 bg-amber-50/50 dark:bg-amber-950/20",
  critical: "border-red-400 bg-red-50/50 dark:bg-red-950/20",
  success: "border-emerald-400 bg-emerald-50/50 dark:bg-emerald-950/20",
};

export default function StatCard({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  trendValue,
  variant = "default",
}: StatCardProps) {
  return (
    <Card className={cn("transition-colors", variantStyles[variant])}>
      <CardContent className="p-4">
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <p className="text-xs font-medium text-muted-foreground">{title}</p>
            <p className="text-2xl font-bold tracking-tight">{value}</p>
            {subtitle && (
              <p className="text-xs text-muted-foreground">{subtitle}</p>
            )}
          </div>
          <div
            className={cn(
              "flex h-9 w-9 items-center justify-center rounded-lg",
              variant === "default" && "bg-muted",
              variant === "warning" && "bg-amber-100 dark:bg-amber-900",
              variant === "critical" && "bg-red-100 dark:bg-red-900",
              variant === "success" && "bg-emerald-100 dark:bg-emerald-900"
            )}
          >
            <Icon
              className={cn(
                "h-4 w-4",
                variant === "default" && "text-muted-foreground",
                variant === "warning" && "text-amber-600",
                variant === "critical" && "text-red-600",
                variant === "success" && "text-emerald-600"
              )}
            />
          </div>
        </div>
        {trend && trendValue && (
          <div className="mt-2 flex items-center gap-1 text-xs text-muted-foreground">
            <span
              className={cn(
                "font-medium",
                trend === "up" && "text-red-500",
                trend === "down" && "text-emerald-500"
              )}
            >
              {trendValue}
            </span>
            <span>vs last hour</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
