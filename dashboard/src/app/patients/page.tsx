"use client";

import { Card, CardContent } from "@/components/ui/card";
import { Users, Construction } from "lucide-react";

export default function PatientsPage() {
  return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <Card className="w-full max-w-md">
        <CardContent className="flex flex-col items-center justify-center py-16 text-center">
          <div className="flex h-16 w-16 items-center justify-center rounded-full bg-muted mb-6">
            <Construction className="h-8 w-8 text-muted-foreground" />
          </div>
          <h2 className="text-xl font-semibold mb-2">Patients — Coming Soon</h2>
          <p className="text-sm text-muted-foreground max-w-sm">
            Multi-device patient management is in development.
            You&apos;ll be able to register, monitor, and manage multiple
            patients and their ESP32 devices here.
          </p>
          <div className="mt-6 flex items-center gap-2 text-xs text-muted-foreground">
            <Users className="h-3.5 w-3.5" />
            <span>Single device mode active</span>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
