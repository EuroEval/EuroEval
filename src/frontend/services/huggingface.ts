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

/**
 * Detect whether a Hugging Face repo is a GGUF-only model, robustly.
 *
 * The evaluation queue cannot load `.gguf` weights, so we must catch GGUF
 * repos before submission. But many repos ship GGUF quants *alongside*
 * safetensors (e.g. norallm/normistral-11b-warm) and are perfectly runnable
 * from the safetensors — so a repo only counts as GGUF when it has `.gguf`
 * weights and *no* safetensors.
 *
 * The GGUF signal uses three independent indicators so it survives the many
 * ways GGUF repos are laid out (per-quant subfolders, sharded files, names
 * without a quant suffix, etc.):
 *
 *   1. The repo carries the "gguf" tag — Hugging Face sets this automatically
 *      on any repo that contains at least one `.gguf` file.
 *   2. `library_name` is "gguf".
 *   3. Any file in `siblings` ends in `.gguf`. `siblings` lists every file in
 *      the repo recursively (including those nested in subfolders), so this
 *      catches repos even if the tag is somehow missing.
 *
 * Returns `false` on any network/parse error so a transient failure never
 * silently classifies a model as GGUF — the caller treats a non-GGUF result
 * as "submittable"/"show in queue", so we err towards not blocking on errors.
 */
export async function detectGguf(modelId: string): Promise<boolean> {
  // Encode each path segment separately: encodeURIComponent on the whole id
  // would turn the "/" between owner and repo into "%2F", which the model-info
  // endpoint rejects with HTTP 400 — silently defeating detection.
  const encodedId = modelId
    .split("/")
    .map((seg) => encodeURIComponent(seg))
    .join("/");
  try {
    const r = await fetch(`https://huggingface.co/api/models/${encodedId}`);
    if (!r.ok) return false;
    const data = (await r.json()) as {
      tags?: string[];
      library_name?: string;
      siblings?: Array<{ rfilename?: string }>;
    };

    const tags = (data.tags ?? []).map((t) => t.toLowerCase());
    const files = (data.siblings ?? []).map((s) =>
      (s.rfilename ?? "").toLowerCase(),
    );
    const hasGguf =
      tags.includes("gguf") ||
      data.library_name?.toLowerCase() === "gguf" ||
      files.some((f) => f.endsWith(".gguf"));
    const hasSafetensors = files.some((f) => f.endsWith(".safetensors"));

    return hasGguf && !hasSafetensors;
  } catch {
    return false;
  }
}
