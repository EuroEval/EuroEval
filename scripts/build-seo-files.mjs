// Build-time generator for sitemap.xml, robots.txt, and llms.txt.
//
// Reads `src/frontend/config.yaml` to enumerate every routable page and
// writes the three SEO/AI-friendly files into Vite's `dist/` directory at
// the end of `vite build`. Also mirrors the bundled CSVs into
// `dist/leaderboards-csv/` so they can be downloaded by direct URL.

import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import yaml from "js-yaml";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..");
const CONFIG_PATH = path.join(REPO_ROOT, "src/frontend/config.yaml");
const CSV_DIR = path.join(REPO_ROOT, "src/frontend/csv");
const BASE_URL = "https://euroeval.com";

/** Convert a sidebar title to a plain text label (drop emojis). */
function stripEmoji(s) {
  return s
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

/** Vite plugin entry point. */
export default function seoFilesPlugin() {
  return {
    name: "euroeval:seo-files",
    apply: "build",
    async closeBundle() {
      const distDir = path.join(REPO_ROOT, "dist");
      try {
        await fs.access(distDir);
      } catch {
        return; // dist not produced (e.g. --watch failed); nothing to do
      }

      const configText = await fs.readFile(CONFIG_PATH, "utf-8");
      const config = yaml.load(configText);

      const urls = [];
      config.sections.forEach((section, i) => {
        urls.push(urlFor(i, section.id, undefined));
        if (section.pages) {
          for (const page of section.pages) {
            urls.push(urlFor(i, section.id, pageSlug(page)));
          }
        }
      });

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

      this.info?.(
        `[seo-files] sitemap (${urls.length} URLs), robots.txt, llms.txt, copied ${csvs.length} CSVs.`,
      );
    },
  };
}
