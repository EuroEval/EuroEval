import { BrokerError, env } from "./protocol.ts";

export async function fetchWithRetry(input: string, init: RequestInit = {}): Promise<Response> {
  // Retrying a POST can repeat a device-code, assignment, or mutation after the
  // server has already applied it.  Only retry methods whose requests are safe
  // to replay; callers that need mutation retries must provide their own CAS.
  const requestMethod = (init.method || "GET").toUpperCase();
  const safe = requestMethod === "GET" || requestMethod === "HEAD" || requestMethod === "OPTIONS";
  if (!safe) return fetch(input, init);
  let response: Response | undefined;
  for (let attempt = 0; attempt < 3; attempt++) {
    try { response = await fetch(input, init); } catch (error) {
      if (attempt === 2) throw error;
      await new Promise((resolve) => setTimeout(resolve, 100 * (attempt + 1))); continue;
    }
    if (![408, 425, 429, 500, 502, 503, 504].includes(response.status) || attempt === 2) return response;
    const retryAfter = response.headers.get("retry-after");
    const seconds = retryAfter && /^\d+$/.test(retryAfter) ? Math.min(30, Number(retryAfter)) : 0;
    await new Promise((resolve) => setTimeout(resolve, Math.max(100 * (attempt + 1), seconds * 1000)));
  }
  return response as Response;
}
