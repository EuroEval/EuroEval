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
 * Detect whether a Hugging Face repo is a GGUF model, robustly.
 *
 * The evaluation queue cannot run GGUF models, so we must catch them before
 * submission. Two independent signals are used so the check survives the many
 * ways GGUF repos are laid out (per-quant subfolders, sharded files, names
 * without a quant suffix, etc.):
 *
 *   1. The repo carries the "gguf" tag — Hugging Face sets this automatically
 *      on any repo that contains at least one `.gguf` file. This is the
 *      primary signal and works regardless of file layout.
 *   2. Any file in `siblings` ends in `.gguf`. `siblings` lists every file in
 *      the repo recursively (including those nested in subfolders), so this
 *      catches repos even if the tag is somehow missing.
 *
 * Returns `false` on any network/parse error so a transient failure never
 * silently classifies a GGUF model as submittable — the caller treats a
 * non-GGUF result as "submittable", so we err towards not blocking on errors.
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
    const hasGgufFile = (data.siblings ?? []).some((s) =>
      (s.rfilename ?? "").toLowerCase().endsWith(".gguf"),
    );

    return (
      tags.includes("gguf") ||
      data.library_name?.toLowerCase() === "gguf" ||
      hasGgufFile
    );
  } catch {
    return false;
  }
}
