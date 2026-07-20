import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Sidebar from "@/components/sidebar";
import Topbar from "@/components/topbar";
import { ThemeProvider } from "@/hooks/use-theme";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "NeuraBand — Dashboard",
  description: "Real-time AI-powered health monitoring dashboard for wearable sensor data",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <head>
        <script dangerouslySetInnerHTML={{
          __html: `
            (function() {
              try {
                var t = localStorage.getItem("neuraband-theme");
                if (t === "dark" || (!t && window.matchMedia("(prefers-color-scheme: dark)").matches)) {
                  document.documentElement.classList.add("dark");
                }
              } catch(e) {}
            })();
          `
        }} />
      </head>
      <body className="h-full">
        <ThemeProvider>
          <div className="flex h-full">
            <Sidebar />
            <div className="flex flex-1 flex-col">
              <Topbar />
              <main className="flex-1 overflow-y-auto bg-muted/30 p-6">
                {children}
              </main>
            </div>
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
