// Generator for sitemap.xml, robots.txt, and llms.txt.
//
// At `vite build`, writes the three files to the `dist/` directory and
// mirrors every leaderboard CSV under `dist/leaderboards-csv/` so they can
// be downloaded by direct URL.
//
// At `vite dev`, a dev-server middleware serves the same paths on the fly
// so developers can sanity-check them at /sitemap.xml, /robots.txt,
// /llms.txt, and /leaderboards-csv/<stem>.csv.

import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import yaml from "js-yaml";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
// File lives at `src/scripts/`, so the repo root is two levels up.
const REPO_ROOT = path.resolve(__dirname, "../..");
const CONFIG_PATH = path.join(REPO_ROOT, "src/frontend/config.yaml");
const CSV_DIR = path.join(REPO_ROOT, "src/frontend/csv");
const BASE_URL = "https://euroeval.com";

/** Convert a sidebar title to a plain text label (drop HTML and emojis). */
function stripEmoji(s) {
  return s
    .replace(/<[a-z][a-z0-9]*\b(?:[^>'"]|'[^']*'|"[^"]*")*>/gi, "")
    .replace(/[\p{Extended_Pictographic}\p{Regional_Indicator}‍️]/gu, "")
    .trim();
}

function pageSlug(page) {
  if (page.slug) return page.slug;
  const ref = page.path || page.csv || "";
  const file = ref.split("/").pop() || ref;
  return file.replace(/\.(md|csv)$/i, "");
}

function urlFor(sectionIndex, sectionId, slug) {
  if (sectionIndex === 0 && !slug) return `${BASE_URL}/`;
  if (!slug) return `${BASE_URL}/${sectionId}`;
  return `${BASE_URL}/${sectionId}/${slug}`;
}

async function listLeaderboardCsvs() {
  const all = await fs.readdir(CSV_DIR);
  // Only ship the user-facing variants (drop `_simplified` versions).
  return all.filter((f) => f.endsWith(".csv") && !f.includes("_simplified"));
}

async function loadConfig() {
  const text = await fs.readFile(CONFIG_PATH, "utf-8");
  return yaml.load(text);
}

function collectUrls(config) {
  const urls = [];
  config.sections.forEach((section, i) => {
    urls.push(urlFor(i, section.id, undefined));
    if (section.pages) {
      for (const page of section.pages) {
        urls.push(urlFor(i, section.id, pageSlug(page)));
      }
    }
  });
  return urls;
}

function buildSitemap(urls) {
  const lines = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
  ];
  for (const u of urls) {
    lines.push("  <url>");
    lines.push(`    <loc>${u}</loc>`);
    lines.push("  </url>");
  }
  lines.push("</urlset>");
  return lines.join("\n") + "\n";
}

function buildRobots() {
  return [
    "User-agent: *",
    "Allow: /",
    "",
    `Sitemap: ${BASE_URL}/sitemap.xml`,
    "",
  ].join("\n");
}

function buildLlmsIndex(config, csvs) {
  const lines = [
    "# EuroEval",
    "",
    "> The robust European language model benchmark.",
    "",
    "EuroEval evaluates language models (encoders, decoders, encoder-decoders, base and instruction-tuned models) across 30+ European languages with reproducible benchmarks. The Python package is at https://github.com/EuroEval/EuroEval.",
    "",
    "## Documentation",
    "",
  ];
  for (const section of config.sections) {
    lines.push(`### ${stripEmoji(section.title)}`);
    lines.push("");
    if (section.indexPath || section.path) {
      const indexUrl = urlFor(
        config.sections.indexOf(section),
        section.id,
        undefined,
      );
      lines.push(`- [${stripEmoji(section.title)} overview](${indexUrl})`);
    }
    if (section.pages) {
      for (const page of section.pages) {
        const url = urlFor(
          config.sections.indexOf(section),
          section.id,
          pageSlug(page),
        );
        const label = stripEmoji(page.title);
        lines.push(`- [${label}](${url})`);
      }
    }
    lines.push("");
  }
  lines.push("## Leaderboard CSVs");
  lines.push("");
  lines.push("Raw evaluation results in DataWrapper-formatted CSV. Direct download URLs:");
  lines.push("");
  for (const name of csvs.sort()) {
    lines.push(`- [${name}](${BASE_URL}/leaderboards-csv/${name})`);
  }
  lines.push("");
  return lines.join("\n");
}

async function copyCsvsTo(distDir) {
  const target = path.join(distDir, "leaderboards-csv");
  await fs.mkdir(target, { recursive: true });
  for (const name of await fs.readdir(CSV_DIR)) {
    if (!name.endsWith(".csv")) continue;
    await fs.copyFile(path.join(CSV_DIR, name), path.join(target, name));
  }
}

// Monolingual leaderboard stem -> language group label (matches the labels
// used in the model-evaluation-request issue template).
const STEM_TO_GROUP = {
  albanian: "Albanian",
  bosnian:
    "Slavic languages (Belarusian, Bulgarian, Bosnian, Croatian, Czech," +
    " Polish, Serbian, Slovak, Slovenian, Ukrainian)",
  bulgarian:
    "Slavic languages (Belarusian, Bulgarian, Bosnian, Croatian, Czech," +
    " Polish, Serbian, Slovak, Slovenian, Ukrainian)",
  catalan:
    "Romance languages (Catalan, French, Italian, Portuguese, Romanian," +
    " Spanish)",
  croatian:
    "Slavic languages (Belarusian, Bulgarian, Bosnian, Croatian, Czech," +
    " Polish, Serbian, Slovak, Slovenian, Ukrainian)",
  czech:
    "Slavic languages (Belarusian, Bulgarian, Bosnian, Croatian, Czech," +
    " Polish, Serbian, Slovak, Slovenian, Ukrainian)",
  danish:
    "Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)",
  dutch: "West Germanic languages (Dutch, English, German)",
  english: "West Germanic languages (Dutch, English, German)",
  estonian: "Finnic languages (Estonian, Finnish)",
  faroese:
    "Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)",
  finnish: "Finnic languages (Estonian, Finnish)",
  french:
    "Romance languages (Catalan, French, Italian, Portuguese, Romanian," +
    " Spanish)",
  german: "West Germanic languages (Dutch, English, German)",
  greek: "Greek",
  hungarian: "Hungarian",
  icelandic:
    "Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)",
  italian:
    "Romance languages (Catalan, French, Italian, Portuguese, Romanian," +
    " Spanish)",
  latvian: "Baltic languages (Latvian, Lithuanian)",
  lithuanian: "Baltic languages (Latvian, Lithuanian)",
  norwegian:
    "Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)",
  polish:
    "Slavic languages (Belarusian, Bulgarian, Bosnian, Croatian, Czech," +
    " Polish, Serbian, Slovak, Slovenian, Ukrainian)",
  portuguese:
    "Romance languages (Catalan, French, Italian, Portuguese, Romanian," +
    " Spanish)",
  romanian:
    "Romance languages (Catalan, French, Italian, Portuguese, Romanian," +
    " Spanish)",
  serbian:
    "Slavic languages (Belarusian, Bulgarian, Bosnian, Croatian, Czech," +
    " Polish, Serbian, Slovak, Slovenian, Ukrainian)",
  slovak:
    "Slavic languages (Belarusian, Bulgarian, Bosnian, Croatian, Czech," +
    " Polish, Serbian, Slovak, Slovenian, Ukrainian)",
  slovene:
    "Slavic languages (Belarusian, Bulgarian, Bosnian, Croatian, Czech," +
    " Polish, Serbian, Slovak, Slovenian, Ukrainian)",
  spanish:
    "Romance languages (Catalan, French, Italian, Portuguese, Romanian," +
    " Spanish)",
  swedish:
    "Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)",
  ukrainian:
    "Slavic languages (Belarusian, Bulgarian, Bosnian, Croatian, Czech," +
    " Polish, Serbian, Slovak, Slovenian, Ukrainian)",
};

/** Parse a simplified leaderboard CSV and return the set of model ids. */
async function modelsInSimplifiedCsv(filePath) {
  let text;
  try {
    text = await fs.readFile(filePath, "utf-8");
  } catch {
    return new Set();
  }
  const lines = text.split("\n");
  // Skip header.
  const out = new Set();
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    // First column may be quoted. Extract whichever form, then pull the
    // inner anchor text and strip any trailing " (...)" suffix.
    const firstCol = line.startsWith('"')
      ? line.slice(1, line.indexOf('",'))
      : line.slice(0, line.indexOf(","));
    const anchorText =
      firstCol.match(/<a [^>]*>([^<]+)<\/a>/i)?.[1] ?? firstCol;
    const modelId = anchorText.replace(/\s*\([^)]*\)\s*$/, "").trim();
    if (modelId) out.add(modelId);
  }
  return out;
}

/** Build the model id -> array of fully-evaluated language groups map. */
async function buildEvaluatedModelsIndex() {
  // For each monolingual stem we care about, get the set of evaluated
  // models (union of generative + all_models simplified CSVs).
  const stemModels = {};
  for (const stem of Object.keys(STEM_TO_GROUP)) {
    const gen = await modelsInSimplifiedCsv(
      path.join(CSV_DIR, `${stem}_generative_simplified.csv`),
    );
    const all = await modelsInSimplifiedCsv(
      path.join(CSV_DIR, `${stem}_all_models_simplified.csv`),
    );
    stemModels[stem] = new Set([...gen, ...all]);
  }

  // Group -> list of stems contributing to it.
  const groupStems = {};
  for (const [stem, group] of Object.entries(STEM_TO_GROUP)) {
    (groupStems[group] ||= []).push(stem);
  }

  // For each model, a group is "covered" iff the model appears in every
  // stem of the group.
  const allModels = new Set();
  for (const models of Object.values(stemModels)) {
    for (const m of models) allModels.add(m);
  }

  const index = {};
  for (const model of allModels) {
    const covered = [];
    for (const [group, stems] of Object.entries(groupStems)) {
      if (stems.every((s) => stemModels[s].has(model))) covered.push(group);
    }
    if (covered.length > 0) index[model] = covered.sort();
  }
  return index;
}

/** Vite plugin entry point — works in both dev and build. */
export default function seoFilesPlugin() {
  return {
    name: "euroeval:seo-files",

    // Dev: serve the generated files on the fly.
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const url = req.url?.split("?")[0] ?? "";
        try {
          if (url === "/sitemap.xml") {
            const config = await loadConfig();
            res.setHeader("Content-Type", "application/xml; charset=utf-8");
            res.end(buildSitemap(collectUrls(config)));
            return;
          }
          if (url === "/robots.txt") {
            res.setHeader("Content-Type", "text/plain; charset=utf-8");
            res.end(buildRobots());
            return;
          }
          if (url === "/evaluated-models.json") {
            res.setHeader("Content-Type", "application/json; charset=utf-8");
            res.end(JSON.stringify(await buildEvaluatedModelsIndex()));
            return;
          }
          if (url === "/llms.txt") {
            const config = await loadConfig();
            const csvs = await listLeaderboardCsvs();
            res.setHeader("Content-Type", "text/markdown; charset=utf-8");
            res.end(buildLlmsIndex(config, csvs));
            return;
          }
          const csvMatch = url.match(/^\/leaderboards-csv\/([^/]+\.csv)$/);
          if (csvMatch) {
            const filePath = path.join(CSV_DIR, csvMatch[1]);
            try {
              const data = await fs.readFile(filePath);
              res.setHeader("Content-Type", "text/csv; charset=utf-8");
              res.setHeader(
                "Content-Disposition",
                `attachment; filename="${csvMatch[1]}"`,
              );
              res.end(data);
              return;
            } catch {
              res.statusCode = 404;
              res.end("Not found");
              return;
            }
          }
        } catch (err) {
          server.config.logger.error(
            `[seo-files] dev middleware error: ${err}`,
          );
        }
        next();
      });
    },

    // Build: emit files into dist/.
    async closeBundle() {
      const distDir = path.join(REPO_ROOT, "dist");
      try {
        await fs.access(distDir);
      } catch {
        return; // dist not produced (e.g. --watch failed); nothing to do
      }

      const config = await loadConfig();
      const urls = collectUrls(config);
      const csvs = await listLeaderboardCsvs();

      await fs.writeFile(
        path.join(distDir, "sitemap.xml"),
        buildSitemap(urls),
        "utf-8",
      );
      await fs.writeFile(
        path.join(distDir, "robots.txt"),
        buildRobots(),
        "utf-8",
      );
      await fs.writeFile(
        path.join(distDir, "llms.txt"),
        buildLlmsIndex(config, csvs),
        "utf-8",
      );
      await copyCsvsTo(distDir);

      const evaluatedIndex = await buildEvaluatedModelsIndex();
      await fs.writeFile(
        path.join(distDir, "evaluated-models.json"),
        JSON.stringify(evaluatedIndex),
        "utf-8",
      );

      this.info?.(
        `[seo-files] sitemap (${urls.length} URLs), robots.txt, llms.txt, copied ${csvs.length} CSVs, indexed ${Object.keys(evaluatedIndex).length} evaluated models.`,
      );
    },
  };
}
