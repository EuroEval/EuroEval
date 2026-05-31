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

export interface GgufDetection {
  isGguf: boolean;
  quants: string[];
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
 * Returns `{ isGguf: false, quants: [] }` on any network/parse error so a
 * transient failure never silently classifies a GGUF model as submittable —
 * the caller treats a non-GGUF result as "submittable", so we err towards the
 * less destructive default by simply not blocking on errors.
 */
export async function detectGguf(modelId: string): Promise<GgufDetection> {
  try {
    const r = await fetch(
      `https://huggingface.co/api/models/${encodeURIComponent(modelId)}`,
    );
    if (!r.ok) return { isGguf: false, quants: [] };
    const data = (await r.json()) as {
      tags?: string[];
      library_name?: string;
      siblings?: Array<{ rfilename?: string }>;
    };

    const tags = (data.tags ?? []).map((t) => t.toLowerCase());
    const files = (data.siblings ?? [])
      .map((s) => s.rfilename ?? "")
      .filter(Boolean);
    const ggufFiles = files.filter((f) => f.toLowerCase().endsWith(".gguf"));

    const isGguf =
      tags.includes("gguf") ||
      data.library_name?.toLowerCase() === "gguf" ||
      ggufFiles.length > 0;

    return { isGguf, quants: extractQuants(ggufFiles) };
  } catch {
    return { isGguf: false, quants: [] };
  }
}

/**
 * Best-effort extraction of quantisation labels from GGUF file paths, purely
 * for display. GGUF repos either name files `<model>-<QUANT>.gguf` or place
 * shards in a `<QUANT>/` subfolder, so we look at the subfolder first and fall
 * back to parsing the filename. Detection never depends on this succeeding.
 */
function extractQuants(ggufFiles: string[]): string[] {
  const quants = new Set<string>();
  for (const path of ggufFiles) {
    const parts = path.split("/");
    if (parts.length > 1) {
      // Per-quant subfolder, e.g. "UD-Q6_K_XL/model-00001-of-00003.gguf".
      quants.add(parts[0].toUpperCase());
      continue;
    }
    // Flat file, e.g. "Model-Q4_K_M.gguf" or "Model-IQ4_XS.gguf".
    const name = parts[0].replace(/\.gguf$/i, "");
    const m = name.match(/(?:^|[-_.])((?:UD-)?(?:IQ|Q|MXFP|BF|FP)\d[\w.]*)$/i);
    if (m) quants.add(m[1].toUpperCase());
  }
  return Array.from(quants).sort().slice(0, 30);
}
