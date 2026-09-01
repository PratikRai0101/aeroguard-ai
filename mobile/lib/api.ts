// lib/api.ts
// API client for the AeroGuard AI FastAPI backend.

const API_BASE_URL = process.env.EXPO_PUBLIC_API_URL || "http://localhost:8000";

export interface DashboardStats {
  aqi: number;
  status: string;
  category: number;
  color: string;
  emoji: string;
  temp: number;
  hum: number;
  gas: number;
  mode: string;
  trend: string;
  advice: string;
  preventive_measures: string[];
  outdoor_aqi: number | null;
  outdoor_pm25: number | null;
  total_readings: number;
  total_alerts: number;
  last_updated: string | null;
}

export interface Reading {
  time: string;
  temp: number;
  hum: number;
  gas: number;
  aqi: number;
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface ChatResponse {
  answer: string;
  model: string;
  context: string;
}

async function fetchJson<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${path}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers || {}),
    },
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`API error ${response.status}: ${text}`);
  }

  return response.json() as Promise<T>;
}

export async function getStats(): Promise<DashboardStats> {
  return fetchJson<DashboardStats>("/api/stats");
}

export async function getReadings(limit = 50): Promise<Reading[]> {
  return fetchJson<Reading[]>(`/api/readings?limit=${limit}`);
}

export async function sendChat(
  question: string,
  history: ChatMessage[] = []
): Promise<ChatResponse> {
  return fetchJson<ChatResponse>("/api/chat", {
    method: "POST",
    body: JSON.stringify({ question, history }),
  });
}

export function getApiBaseUrl(): string {
  return API_BASE_URL;
}
