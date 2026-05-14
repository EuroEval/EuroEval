import { navConfig, pageSlug } from "@/nav";
import { mdLookup } from "@/markdown";

export interface SearchEntry {
  sectionId: string;
  sectionTitle: string;
  pageTitle: string;
  url: string;
  text: string;
  textLower: string;
  titleLower: string;
}

export interface SearchResult {
  entry: SearchEntry;
  score: number;
  snippet: string;
}

const stripFrontmatter = (text: string): string => {
  if (text.startsWith("---")) {
    const end = text.indexOf("\n---", 3);
    if (end !== -1) return text.slice(end + 4);
  }
  return text;
};

const stripMarkdown = (text: string): string => {
  return text
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`[^`]*`/g, " ")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/[#>*_~]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
};

const buildIndex = (): SearchEntry[] => {
  const lookup = mdLookup;
  const entries: SearchEntry[] = [];

  for (const section of navConfig.sections) {
    const indexMd = section.indexPath || section.path;
    if (indexMd && lookup[indexMd]) {
      const text = stripMarkdown(stripFrontmatter(lookup[indexMd]));
      entries.push({
        sectionId: section.id,
        sectionTitle: section.title,
        pageTitle: section.title,
        url: section.id === navConfig.sections[0].id ? "/" : `/${section.id}`,
        text,
        textLower: text.toLowerCase(),
        titleLower: section.title.toLowerCase(),
      });
    }

    if (section.pages) {
      for (const page of section.pages) {
        if (page.path) {
          const md = lookup[page.path];
          if (!md) continue;
          const text = stripMarkdown(stripFrontmatter(md));
          entries.push({
            sectionId: section.id,
            sectionTitle: section.title,
            pageTitle: page.title,
            url: `/${section.id}/${pageSlug(page)}`,
            text,
            textLower: text.toLowerCase(),
            titleLower: page.title.toLowerCase(),
          });
        } else if (page.csv) {
          // Leaderboard pages: index title only — CSV content is not searched.
          entries.push({
            sectionId: section.id,
            sectionTitle: section.title,
            pageTitle: page.title,
            url: `/${section.id}/${pageSlug(page)}`,
            text: page.title,
            textLower: page.title.toLowerCase(),
            titleLower: page.title.toLowerCase(),
          });
        }
      }
    }
  }

  return entries;
};

const index = buildIndex();

const makeSnippet = (text: string, query: string): string => {
  const lower = text.toLowerCase();
  const idx = lower.indexOf(query.toLowerCase());
  if (idx === -1) return text.slice(0, 140);
  const start = Math.max(0, idx - 50);
  const end = Math.min(text.length, idx + query.length + 90);
  const prefix = start > 0 ? "…" : "";
  const suffix = end < text.length ? "…" : "";
  return prefix + text.slice(start, end) + suffix;
};

export function searchAll(query: string, limit = 8): SearchResult[] {
  const q = query.trim().toLowerCase();
  if (q.length < 2) return [];

  const tokens = q.split(/\s+/).filter(Boolean);
  const results: SearchResult[] = [];

  for (const entry of index) {
    let score = 0;
    let allHit = true;
    for (const t of tokens) {
      const titleHit = entry.titleLower.includes(t);
      const sectionHit = entry.sectionTitle.toLowerCase().includes(t);
      const bodyHit = entry.textLower.includes(t);
      if (titleHit) score += 10;
      if (sectionHit) score += 3;
      if (bodyHit) score += 1;
      if (!titleHit && !sectionHit && !bodyHit) {
        allHit = false;
        break;
      }
    }
    if (!allHit || score === 0) continue;
    results.push({
      entry,
      score,
      snippet: makeSnippet(entry.text, tokens[0]),
    });
  }

  results.sort((a, b) => b.score - a.score);
  return results.slice(0, limit);
}
