export interface HfModelSuggestion {
  id: string;
  downloads?: number;
}

let activeController: AbortController | null = null;

export async function searchHfModels(
  query: string,
  limit = 20,
): Promise<HfModelSuggestion[]> {
  const q = query.trim();
  if (!q) return [];

  activeController?.abort();
  const controller = new AbortController();
  activeController = controller;

  const url =
    `https://huggingface.co/api/models?search=${encodeURIComponent(q)}` +
    `&limit=${limit}&sort=downloads&direction=-1`;

  try {
    const r = await fetch(url, { signal: controller.signal });
    if (!r.ok) return [];
    const data = (await r.json()) as Array<{ id?: string; modelId?: string; downloads?: number }>;
    return data
      .map((m) => ({ id: m.id ?? m.modelId ?? "", downloads: m.downloads }))
      .filter((m) => m.id);
  } catch (e) {
    if ((e as Error).name === "AbortError") return [];
    return [];
  }
}
