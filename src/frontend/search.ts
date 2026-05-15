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

const slugify = (text: string): string =>
  text
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-");

/** Split markdown into per-H2 chunks. Returns the lead text (before the first
 *  H2) plus one entry per H2 section, each with the heading title, slug, and
 *  the body text up to the next H2. */
const splitByH2 = (
  md: string,
): {
  lead: string;
  sections: { title: string; slug: string; body: string }[];
} => {
  const stripped = stripFrontmatter(md);
  const lines = stripped.split("\n");
  let lead: string[] = [];
  const sections: { title: string; slug: string; body: string[] }[] = [];
  let inFence = false;
  for (const line of lines) {
    if (/^```/.test(line)) inFence = !inFence;
    const m = !inFence ? line.match(/^##\s+(.+?)\s*$/) : null;
    if (m) {
      sections.push({ title: m[1].trim(), slug: slugify(m[1]), body: [] });
      continue;
    }
    if (sections.length === 0) lead.push(line);
    else sections[sections.length - 1].body.push(line);
  }
  return {
    lead: lead.join("\n"),
    sections: sections.map((s) => ({
      title: s.title,
      slug: s.slug,
      body: s.body.join("\n"),
    })),
  };
};

const buildIndex = (): SearchEntry[] => {
  const lookup = mdLookup;
  const entries: SearchEntry[] = [];

  const addPageEntries = (
    sectionId: string,
    sectionTitle: string,
    pageTitle: string,
    baseUrl: string,
    md: string,
  ) => {
    const { lead, sections } = splitByH2(md);
    // Page-level entry: body is the lead-in text (before the first H2). If
    // there are no H2s, fall back to the full page.
    const pageText =
      sections.length === 0
        ? stripMarkdown(stripFrontmatter(md))
        : stripMarkdown(lead);
    entries.push({
      sectionId,
      sectionTitle,
      pageTitle,
      url: baseUrl,
      text: pageText,
      textLower: pageText.toLowerCase(),
      titleLower: pageTitle.toLowerCase(),
    });
    // One entry per H2 section, deep-linked to its anchor.
    for (const s of sections) {
      const body = stripMarkdown(s.body);
      const combinedTitle = `${pageTitle} › ${s.title}`;
      entries.push({
        sectionId,
        sectionTitle,
        pageTitle: combinedTitle,
        url: `${baseUrl}#${s.slug}`,
        text: body,
        textLower: body.toLowerCase(),
        titleLower: `${pageTitle.toLowerCase()} ${s.title.toLowerCase()}`,
      });
    }
  };

  for (const section of navConfig.sections) {
    const indexMd = section.indexPath || section.path;
    if (indexMd && lookup[indexMd]) {
      const baseUrl =
        section.id === navConfig.sections[0].id ? "/" : `/${section.id}`;
      addPageEntries(
        section.id,
        section.title,
        section.title,
        baseUrl,
        lookup[indexMd],
      );
    }

    if (section.pages) {
      for (const page of section.pages) {
        if (page.path) {
          const md = lookup[page.path];
          if (!md) continue;
          addPageEntries(
            section.id,
            section.title,
            page.title,
            `/${section.id}/${pageSlug(page)}`,
            md,
          );
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
