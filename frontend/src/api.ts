import type { ExplainResponse, PredictionRequest } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE?.trim() || "";

export async function explainInference(payload: PredictionRequest): Promise<ExplainResponse> {
  const response = await fetch(`${API_BASE}/explain`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let detail = "Request failed";
    try {
      const data = (await response.json()) as { detail?: string };
      detail = data.detail || detail;
    } catch {
      // Keep fallback message if response is not JSON.
    }
    throw new Error(detail);
  }

  return (await response.json()) as ExplainResponse;
}
