"use client";

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

type Theme = "light" | "dark";

interface ThemeContext {
  theme: Theme;
  toggleTheme: () => void;
  setTheme: (t: Theme) => void;
}

const ThemeCtx = createContext<ThemeContext>({
  theme: "light",
  toggleTheme: () => {},
  setTheme: () => {},
});

const STORAGE_KEY = "neuraband-theme";

function getInitialTheme(): Theme {
  if (typeof window === "undefined") return "light";
  const stored = localStorage.getItem(STORAGE_KEY) as Theme | null;
  if (stored === "dark" || stored === "light") return stored;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>("light");

  // Hydrate after mount
  useEffect(() => {
    setThemeState(getInitialTheme());
  }, []);

  const applyTheme = (t: Theme) => {
    const root = document.documentElement;
    if (t === "dark") {
      root.classList.add("dark");
    } else {
      root.classList.remove("dark");
    }
    localStorage.setItem(STORAGE_KEY, t);
  };

  const setTheme = (t: Theme) => {
    setThemeState(t);
    applyTheme(t);
  };

  const toggleTheme = () => {
    setTheme(theme === "dark" ? "light" : "dark");
  };

  // Sync on change
  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  return (
    <ThemeCtx.Provider value={{ theme, toggleTheme, setTheme }}>
      {children}
    </ThemeCtx.Provider>
  );
}

export function useTheme() {
  return useContext(ThemeCtx);
}
