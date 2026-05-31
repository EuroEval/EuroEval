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
    `&limit=${limit * 3}&sort=downloads&direction=-1`;

  try {
    const r = await fetch(url, { signal: controller.signal });
    if (!r.ok) return [];
    const data = (await r.json()) as Array<{ id?: string; modelId?: string; downloads?: number }>;
    return data
      .map((m) => ({ id: m.id ?? m.modelId ?? "", downloads: m.downloads }))
      .filter((m) => m.id)
      .slice(0, limit);
  } catch (e) {
    if ((e as Error).name === "AbortError") return [];
    return [];
  }
}

export async function detectGgufQuants(modelId: string): Promise<string[]> {
  try {
    const r = await fetch(
      `https://huggingface.co/api/models/${encodeURIComponent(modelId)}/tree/main`,
    );
    if (!r.ok) return [];
    const files = (await r.json()) as Array<{ path: string; type: string }>;
    const quants = new Set<string>();
    for (const f of files) {
      if (f.type !== "file") continue;
      const m = f.path.match(/model-[QK]([0-9]_?[A-Z_]+)\.gguf$/i);
      if (m) quants.add(m[1].toUpperCase());
    }
    return Array.from(quants).slice(0, 30);
  } catch {
    return [];
  }
}
