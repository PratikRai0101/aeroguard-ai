// lib/api.ts
// API client for the AeroGuard AI FastAPI backend.

const API_BASE_URL = process.env.EXPO_PUBLIC_API_URL || "http://localhost:8000";

export interface Prediction {
  status: number | null;
  label: string;
  confidence: number;
  class_probabilities?: Record<string, number>;
  error?: string;
}

export interface ReliabilityInterval {
  level: number;
  low: number;
  high: number;
  half_width: number;
}

export interface Reliability {
  category_probability: number;
  label: "High" | "Moderate" | "Low" | string;
  interval: ReliabilityInterval;
  residual_std: number;
}

export interface Risk {
  level: "Low" | "Moderate" | "High" | string;
  score: number;
  factors: string[];
  advisory: string;
  disclaimer: string;
  reliability_caveat?: boolean;
}

export interface ShapFactor {
  feature: string;
  description: string;
  value: number;
  unit: string;
  contribution: number;
  direction: "increases" | "decreases" | string;
}

export interface Explanation {
  predicted_class: number;
  predicted_label: string;
  top_factors: ShapFactor[];
  all_factors: ShapFactor[];
  method: string;
}

export interface DashboardStats {
  aqi: number;
  sensor_aqi?: number;
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
  prediction?: Prediction | null;
  future_prediction?: Prediction | null;
  predicted_aqi?: number | null;
  reliability?: Reliability | null;
  risk?: Risk | null;
  explanation?: Explanation | null;
}

export interface Reading {
  time: string;
  temp: number;
  hum: number;
  gas: number;
  aqi: number;
}

export interface ValidationReport {
  dataset?: {
    rows: number;
    start: string;
    end: string;
    temp_unique?: number;
    hum_unique?: number;
    aqi_min?: number;
    aqi_max?: number;
  } | null;
  dataset_source?: string | null;
  calibration?: {
    trained_on?: string;
    metrics?: { n?: number; r2_log?: number; rmse?: number; mae?: number };
    note?: string;
  } | null;
  rf?: { accuracy?: number; classes?: number[] } | null;
  lr?: {
    metrics?: { n?: number; mae?: number; rmse?: number; r2?: number; pearson?: number };
    validation?: {
      regression?: { n?: number; mae?: number; rmse?: number; r2?: number; pearson?: number };
      categories?: {
        n?: number;
        exact_match?: number;
        within_one?: number;
        confusion?: number[][];
      };
    };
    residual_std?: number;
  } | null;
  lstm?: {
    window?: number;
    val_loss?: number;
    val_accuracy?: number;
    train_sequences?: number;
    val_sequences?: number;
  } | null;
  notes?: string[];
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

export async function getValidation(): Promise<ValidationReport> {
  return fetchJson<ValidationReport>("/api/validation");
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
