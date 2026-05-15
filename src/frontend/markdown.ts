import MarkdownIt from "markdown-it";
import hljs from "highlight.js";

export interface TocItem {
  id: string;
  text: string;
  level: number;
}

export interface RenderedMarkdown {
  html: string;
  toc: TocItem[];
}

const mdModules = import.meta.glob("@/md/**/*.md", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

export const mdLookup: Record<string, string> = {};
for (const fullPath in mdModules) {
  const key = fullPath.replace(/^.*\/md\//, "");
  mdLookup[key] = mdModules[fullPath];
}

// Transform mkdocs-blocks-style tab groups (`/// tab | Title` … `///`) into
// pure-CSS radio-driven tab containers. Consecutive tab blocks form a single
// group; isolated blocks become a one-tab group.
let tabGroupCounter = 0;
const escapeHtml = (s: string): string =>
  s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

const transformTabs = (text: string): string => {
  const lines = text.split("\n");
  const out: string[] = [];
  const openRe = /^\/\/\/\s+tab\s+\|\s+(.+?)\s*$/;
  let i = 0;
  while (i < lines.length) {
    if (!openRe.test(lines[i])) {
      out.push(lines[i]);
      i++;
      continue;
    }
    const tabs: Array<{ title: string; content: string }> = [];
    while (i < lines.length) {
      const m = lines[i].match(openRe);
      if (!m) break;
      const title = m[1];
      i++;
      const contentLines: string[] = [];
      while (i < lines.length && lines[i].trim() !== "///") {
        contentLines.push(lines[i]);
        i++;
      }
      if (i < lines.length) i++; // skip closing ///
      tabs.push({ title, content: contentLines.join("\n").trim() });
      // Look ahead past blank lines for another tab in the same group.
      let j = i;
      while (j < lines.length && lines[j].trim() === "") j++;
      if (j < lines.length && openRe.test(lines[j])) {
        i = j;
      } else {
        break;
      }
    }
    const gid = `md-tabs-${++tabGroupCounter}`;
    out.push(`<div class="md-tabs">`);
    tabs.forEach((t, idx) => {
      const checked = idx === 0 ? " checked" : "";
      out.push(
        `<input type="radio" class="md-tab-radio md-tab-radio-${idx}" name="${gid}" id="${gid}-${idx}"${checked} aria-hidden="true">` +
          `<label class="md-tab-label" for="${gid}-${idx}">${escapeHtml(t.title)}</label>`,
      );
    });
    tabs.forEach((t, idx) => {
      out.push(`<div class="md-tab-pane" data-idx="${idx}">`);
      out.push("");
      out.push(t.content);
      out.push("");
      out.push(`</div>`);
    });
    out.push(`</div>`);
    out.push("");
  }
  return out.join("\n");
};

const stripFrontmatter = (text: string): string => {
  if (text.startsWith("---")) {
    const end = text.indexOf("\n---", 3);
    if (end !== -1) {
      return text.slice(end + 4).replace(/^\s*\n/, "");
    }
  }
  return text;
};

const slugify = (text: string): string =>
  text
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-");

const md: MarkdownIt = new MarkdownIt({
  html: true,
  linkify: true,
  highlight: (code, lang) => {
    const language = (lang || "").replace(/\W/g, "");
    if (language && hljs.getLanguage(language)) {
      const escaped = hljs.highlight(code, {
        language,
        ignoreIllegals: true,
      }).value;
      return `<pre class="code-block"><code class="hljs language-${language}">${escaped}</code></pre>`;
    }
    return `<pre class="code-block"><code class="hljs">${md.utils.escapeHtml(code)}</code></pre>`;
  },
});

// Inject anchor IDs into h1–h3, and collect a TOC list of h2/h3 entries.
md.core.ruler.push("inject-heading-ids", (state) => {
  const usedIds = new Set<string>();
  for (let i = 0; i < state.tokens.length; i++) {
    const token = state.tokens[i];
    if (token.type !== "heading_open") continue;
    const level = parseInt(token.tag.slice(1), 10);
    if (level < 1 || level > 3) continue;
    const inline = state.tokens[i + 1];
    if (!inline || inline.type !== "inline") continue;
    const text = (inline.content || "").trim();
    if (!text) continue;
    let id = slugify(text);
    if (!id) continue;
    let unique = id;
    let n = 1;
    while (usedIds.has(unique)) {
      unique = `${id}-${++n}`;
    }
    usedIds.add(unique);
    token.attrSet("id", unique);
  }
});

const cache = new Map<string, RenderedMarkdown>();

export function renderMarkdown(path: string): RenderedMarkdown | undefined {
  const cached = cache.get(path);
  if (cached) return cached;

  const raw = mdLookup[path];
  if (!raw) return undefined;

  const env: Record<string, unknown> = {};
  const html = md.render(transformTabs(stripFrontmatter(raw)), env);

  // Walk the html to extract heading info for TOC (h2 + h3 only).
  const toc: TocItem[] = [];
  const re = /<h([23])\s+id="([^"]+)"[^>]*>([\s\S]*?)<\/h\1>/g;
  let match: RegExpExecArray | null;
  while ((match = re.exec(html)) !== null) {
    const level = parseInt(match[1], 10);
    const id = match[2];
    // Strip HTML tags from heading inner text.
    const text = match[3].replace(/<[^>]+>/g, "").trim();
    if (text) toc.push({ id, text, level });
  }

  const result: RenderedMarkdown = { html, toc };
  cache.set(path, result);
  return result;
}
