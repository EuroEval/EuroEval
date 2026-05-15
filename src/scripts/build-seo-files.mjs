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

      this.info?.(
        `[seo-files] sitemap (${urls.length} URLs), robots.txt, llms.txt, copied ${csvs.length} CSVs.`,
      );
    },
  };
}
