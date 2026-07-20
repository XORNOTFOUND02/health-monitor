"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  HeartPulse,
  Activity,
  Thermometer,
  Moon,
  AlertTriangle,
  Zap,
  Wind,
} from "lucide-react";

const conditions = [
  {
    id: "tachycardia",
    name: "Fast Heart Rate",
    medical: "Tachycardia",
    icon: HeartPulse,
    color: "text-red-500",
    bg: "bg-red-50 dark:bg-red-950/20",
    border: "border-red-200 dark:border-red-900",
    what: "Your heart is beating too fast. A normal heart beats about 60 to 100 times in one minute. If it beats more than 100 times, we call it fast heart rate.",
    why: "This can happen when you are stressed, scared, or have done hard exercise. But sometimes it can mean something is not right inside your body.",
    danger: "If it happens for a long time, your heart gets tired. In serious cases, the heart may not pump blood well.",
    whattodo:
      "Sit down, take slow deep breaths. Drink water. If it does not go back to normal in a few minutes, tell a doctor.",
  },
  {
    id: "bradycardia",
    name: "Slow Heart Rate",
    medical: "Bradycardia",
    icon: HeartPulse,
    color: "text-blue-500",
    bg: "bg-blue-50 dark:bg-blue-950/20",
    border: "border-blue-200 dark:border-blue-900",
    what: "Your heart is beating too slow — less than 60 times in one minute. Athletes and people who sleep well sometimes have slow heart rates, and that is okay.",
    why: "Sometimes the heart's natural wiring gets a problem. Certain medicines or being very tired can also cause this.",
    danger: "When the heart beats too slow, your brain and body do not get enough blood. You may feel dizzy, tired, or even faint.",
    whattodo:
      "If you feel fine, it is probably okay. If you feel dizzy or faint, sit or lie down and see a doctor soon.",
  },
  {
    id: "irregular_rhythm",
    name: "Uneven Heartbeat",
    medical: "Irregular Rhythm",
    icon: HeartPulse,
    color: "text-purple-500",
    bg: "bg-purple-50 dark:bg-purple-950/20",
    border: "border-purple-200 dark:border-purple-900",
    what: "Your heart usually beats like a steady drum — boom, boom, boom. An uneven heartbeat means the beats come at strange, unpredictable times. Sometimes fast, sometimes slow, sometimes skipping.",
    why: "The tiny wires inside your heart that control the beat may not be working perfectly. Stress, caffeine, or getting older can cause this.",
    danger: "Blood can pool inside the heart when the beat is uneven. Over time this can increase the chance of a blood clot or stroke.",
    whattodo:
      "Do not panic. Sit down and relax. If it keeps happening, see a doctor. They may do a simple heart test called an ECG.",
  },
  {
    id: "low_spo2",
    name: "Low Oxygen in Blood",
    medical: "Hypoxia",
    icon: Wind,
    color: "text-cyan-500",
    bg: "bg-cyan-50 dark:bg-cyan-950/20",
    border: "border-cyan-200 dark:border-cyan-900",
    what: "Your blood carries oxygen to every part of your body — your brain, heart, muscles. A normal oxygen level is 95% or higher. If it drops below 95%, we call it low oxygen.",
    why: "A cold, asthma, lung problems, or sleeping problems like apnea (when you stop breathing for a few seconds while sleeping) can cause this.",
    danger: "Your brain needs oxygen the most. If oxygen stays low for too long, you can feel very confused, tired, or in the worst case, pass out.",
    whattodo:
      "Try sitting up straight and taking deep breaths. Open a window for fresh air. If it stays below 92%, get medical help right away.",
  },
  {
    id: "fever",
    name: "Fever",
    medical: "Fever",
    icon: Thermometer,
    color: "text-orange-500",
    bg: "bg-orange-50 dark:bg-orange-950/20",
    border: "border-orange-200 dark:border-orange-900",
    what: "Your normal body temperature is about 36.5°C to 37.5°C. When it goes above 38°C, we call it a fever. Your body is doing this on purpose to fight germs.",
    why: "Most often it is because your body is fighting an infection — like a cold, the flu, or a stomach bug. Sometimes medicines or heat strokes can also raise temperature.",
    danger: "A mild fever is usually fine and goes away in a day or two. But a very high fever (above 39.5°C) can be dangerous, especially for babies and old people.",
    whattodo:
      "Drink lots of water. Rest. Take a warm bath. If the fever is very high or lasts more than 2 days, see a doctor.",
  },
  {
    id: "fall_detected",
    name: "Fall Detected",
    medical: "Fall Detection",
    icon: AlertTriangle,
    color: "text-amber-500",
    bg: "bg-amber-50 dark:bg-amber-950/20",
    border: "border-amber-200 dark:border-amber-900",
    what: "NeuraBand felt a sudden hard hit on your body, and then you stopped moving. This pattern usually means you fell down.",
    why: "Falls can happen because of slippery floors, dizziness, loss of balance, or tripping over something. Old people fall more often because their bones and muscles get weaker.",
    danger: "Falls can cause broken bones, head injuries, or bleeding inside the body. If you cannot get up, it can be very serious.",
    whattodo:
      "If you fell and feel okay, move slowly and check for pain. If you cannot move or feel very hurt, stay still and call for help.",
  },
  {
    id: "sleep_problem",
    name: "Sleep Problem",
    medical: "Sleep Disorder",
    icon: Moon,
    color: "text-indigo-500",
    bg: "bg-indigo-50 dark:bg-indigo-950/20",
    border: "border-indigo-200 dark:border-indigo-900",
    what: "Your body is moving too much while sleeping, or your heart rate and breathing are not calming down the way they should during sleep. Good sleep means your body gets proper rest.",
    why: "Stress, drinking coffee late at night, using your phone before bed, or sleeping at different times every day can cause sleep problems. Some people also have conditions like sleep apnea where breathing stops for a few seconds.",
    danger: "Bad sleep for many days makes you feel very tired, forgetful, and grumpy. It also weakens your immune system and increases the risk of heart disease over time.",
    whattodo:
      "Try to sleep and wake up at the same time every day. Avoid screens before bed. Keep your room dark and cool. If you snore loudly or feel very tired during the day, see a doctor.",
  },
  {
    id: "fatigue",
    name: "Extreme Tiredness",
    medical: "Fatigue",
    icon: Zap,
    color: "text-yellow-500",
    bg: "bg-yellow-50 dark:bg-yellow-950/20",
    border: "border-yellow-200 dark:border-yellow-900",
    what: "This is not just feeling a little sleepy. This is when your body is so tired that even after resting, you still feel completely drained. Your heart rate is a bit higher than normal even when you are not doing anything.",
    why: "Working too hard, not sleeping enough, being sick, or being under a lot of stress for a long time can cause this. Sometimes it is also a sign of your body fighting an illness.",
    danger: "Fatigue makes it hard to think clearly and do daily work. It also increases the chance of accidents — like falling or crashing while driving.",
    whattodo:
      "Take a real break. Sleep well tonight. Eat good food and drink water. If you feel this way for more than a week, talk to a doctor.",
  },
];

export default function InfoPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Health Conditions Guide</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Simple explanations of every health condition NeuraBand can detect. Written so anyone can understand.
        </p>
      </div>

      {/* Quick intro */}
      <Card>
        <CardContent className="p-4">
          <p className="text-sm text-muted-foreground leading-relaxed">
            NeuraBand watches your body signals 24 hours a day using tiny sensors on your wrist. If it
            notices something unusual, it will alert you. Below you can learn about each condition — what
            it means, why it happens, and what you should do. <strong>This is not medical advice. Always
            talk to a doctor if you are worried.</strong>
          </p>
        </CardContent>
      </Card>

      {/* Condition cards */}
      <div className="grid gap-4 md:grid-cols-2">
        {conditions.map((c) => {
          const Icon = c.icon;
          return (
            <Card key={c.id} className={c.border}>
              <CardHeader className="pb-3">
                <div className="flex items-center gap-3">
                  <div className={`flex h-10 w-10 items-center justify-center rounded-lg ${c.bg}`}>
                    <Icon className={`h-5 w-5 ${c.color}`} />
                  </div>
                  <div>
                    <CardTitle className="text-base">{c.name}</CardTitle>
                    <p className="text-xs text-muted-foreground italic">{c.medical}</p>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">
                    What is it?
                  </p>
                  <p className="text-sm leading-relaxed">{c.what}</p>
                </div>
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">
                    Why does it happen?
                  </p>
                  <p className="text-sm leading-relaxed">{c.why}</p>
                </div>
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">
                    Is it dangerous?
                  </p>
                  <p className="text-sm leading-relaxed">{c.danger}</p>
                </div>
                <div className={`rounded-lg p-3 ${c.bg}`}>
                  <p className="text-xs font-semibold uppercase tracking-wider mb-1">
                    What should I do?
                  </p>
                  <p className="text-sm leading-relaxed font-medium">{c.whattodo}</p>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Bottom note */}
      <Card>
        <CardContent className="p-4">
          <p className="text-sm text-muted-foreground leading-relaxed text-center">
            NeuraBand is a health awareness tool — it is <strong>not a doctor</strong>. If you ever feel
            very sick, hurt, or unsure, always call emergency services or visit a hospital right away.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
