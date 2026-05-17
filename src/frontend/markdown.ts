import MarkdownIt from "markdown-it";
// Use the core build and register only the languages that actually appear in
// our docs, instead of pulling in highlight.js's full ~600KB language pack.
import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import json from "highlight.js/lib/languages/json";
import python from "highlight.js/lib/languages/python";
import yaml from "highlight.js/lib/languages/yaml";

hljs.registerLanguage("bash", bash);
hljs.registerLanguage("json", json);
hljs.registerLanguage("python", python);
hljs.registerLanguage("yaml", yaml);

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
}) as Record<string, () => Promise<string>>;

const mdLoaders: Record<string, () => Promise<string>> = {};
for (const fullPath in mdModules) {
  const key = fullPath.replace(/^.*\/md\//, "");
  mdLoaders[key] = mdModules[fullPath];
}

export const mdKeys: string[] = Object.keys(mdLoaders);

export async function loadRawMarkdown(
  path: string,
): Promise<string | undefined> {
  const loader = mdLoaders[path];
  return loader ? await loader() : undefined;
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
        `<input type="radio" class="md-tab-radio md-tab-radio-${idx}" name="${gid}" id="${gid}-${idx}"${checked} aria-hidden="true" data-tab-title="${escapeHtml(t.title)}">` +
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

// Transform mkdocs-style collapsible admonitions:
//   ??? example          → collapsed <details> with summary "Example"
//   ???+ example         → expanded  <details open>
//   ??? note "My title"  → summary uses the explicit title
// Indented (≥4 spaces) lines below belong to the block; the indent is stripped.
const transformAdmonitions = (text: string): string => {
  const lines = text.split("\n");
  const out: string[] = [];
  const openRe = /^(\?\?\?\+?)\s+(\S+)(?:\s+"([^"]+)")?\s*$/;
  const capitalize = (s: string): string =>
    s.charAt(0).toUpperCase() + s.slice(1);
  let i = 0;
  while (i < lines.length) {
    const m = lines[i].match(openRe);
    if (!m) {
      out.push(lines[i]);
      i++;
      continue;
    }
    const marker = m[1];
    const type = m[2].toLowerCase();
    const title = m[3] ?? capitalize(m[2]);
    const openAttr = marker.endsWith("+") ? " open" : "";
    i++;
    // Skip a single blank line directly after the marker, if present.
    if (i < lines.length && lines[i].trim() === "") i++;
    const contentLines: string[] = [];
    while (i < lines.length) {
      const line = lines[i];
      if (line.trim() === "") {
        contentLines.push("");
        i++;
        continue;
      }
      if (/^ {4}/.test(line)) {
        contentLines.push(line.slice(4));
        i++;
        continue;
      }
      break;
    }
    while (contentLines.length && contentLines[contentLines.length - 1] === "")
      contentLines.pop();
    out.push(
      `<details class="md-admonition md-admonition-${escapeHtml(type)}"${openAttr}>`,
    );
    out.push(
      `<summary class="md-admonition-title">${escapeHtml(title)}</summary>`,
    );
    out.push("");
    out.push(contentLines.join("\n"));
    out.push("");
    out.push(`</details>`);
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

const decodeEntities = (s: string): string =>
  s
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&amp;/g, "&");

/**
 * Trim a rendered heading down to a short label suitable for the right-hand
 * table of contents. Strips tags, decodes entities, and — for API-reference
 * headings like `class Foo(Bar)` or `bar.baz(arg: int) -> bool` — drops the
 * parenthesised signature and any trailing return-type annotation, leaving
 * just the symbol name.
 */
const simplifyHeadingForToc = (rawInner: string): string => {
  // Strip the "source" GitHub link the API generator appends to each heading
  // before we strip generic HTML, so its text content doesn't leak into the
  // TOC label.
  let stripped = rawInner.replace(
    /<a\s+class="api-source-link"[^>]*>[\s\S]*?<\/a>/gi,
    "",
  );
  let text = decodeEntities(stripped.replace(/<[^>]+>/g, "")).trim();
  // Drop a trailing `-> type` annotation if it appears outside the parens.
  text = text.replace(/\s*->.*$/, "");
  // Drop the parenthesised signature, if any.
  const paren = text.indexOf("(");
  if (paren !== -1) text = text.slice(0, paren).trim();
  // Drop a leading `class ` keyword, leaving just the class name.
  text = text.replace(/^class\s+/, "");
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

// Cross-group tab sync: when a user picks a tab in one group (e.g. "Using the
// command line"), every other group on the page that has a tab with the same
// title switches to it too. The preference also persists across pages for the
// session via localStorage.
const TAB_PREF_KEY = "euroeval:preferredTabTitle";
let preferredTabTitle: string | null = null;
try {
  preferredTabTitle = window.localStorage.getItem(TAB_PREF_KEY);
} catch {
  // localStorage may be unavailable; fall back to in-memory only.
}

const setPreferredTabTitle = (title: string): void => {
  preferredTabTitle = title;
  try {
    window.localStorage.setItem(TAB_PREF_KEY, title);
  } catch {
    // ignore
  }
};

const syncAllGroups = (root: ParentNode, title: string): void => {
  const radios = root.querySelectorAll<HTMLInputElement>(
    `.md-tabs > .md-tab-radio[data-tab-title="${CSS.escape(title)}"]`,
  );
  radios.forEach((r) => {
    if (!r.checked) r.checked = true;
  });
};

/**
 * Wire up cross-group tab synchronisation on the given rendered-markdown
 * container. Returns a cleanup function. Call this from a component that
 * renders `renderMarkdown` output via `v-html`.
 */
export function wireTabSync(root: HTMLElement): () => void {
  // Apply the stored preference on mount, if any.
  if (preferredTabTitle) syncAllGroups(root, preferredTabTitle);

  const onChange = (e: Event) => {
    const target = e.target as HTMLElement | null;
    if (!(target instanceof HTMLInputElement)) return;
    if (!target.classList.contains("md-tab-radio")) return;
    const title = target.dataset.tabTitle;
    if (!title) return;
    setPreferredTabTitle(title);
    // Document-wide sync so any other markdown root on screen also follows.
    syncAllGroups(document, title);
  };
  root.addEventListener("change", onChange);
  return () => root.removeEventListener("change", onChange);
}

export async function renderMarkdown(
  path: string,
): Promise<RenderedMarkdown | undefined> {
  const cached = cache.get(path);
  if (cached) return cached;

  const raw = await loadRawMarkdown(path);
  if (!raw) return undefined;

  const env: Record<string, unknown> = {};
  const html = md.render(
    transformTabs(transformAdmonitions(stripFrontmatter(raw))),
    env,
  );

  // Walk the html to extract heading info for TOC (h2 + h3 only).
  const toc: TocItem[] = [];
  const re = /<h([23])\s+id="([^"]+)"[^>]*>([\s\S]*?)<\/h\1>/g;
  let match: RegExpExecArray | null;
  while ((match = re.exec(html)) !== null) {
    const level = parseInt(match[1], 10);
    const id = match[2];
    const text = simplifyHeadingForToc(match[3]);
    if (text) toc.push({ id, text, level });
  }

  const result: RenderedMarkdown = { html, toc };
  cache.set(path, result);
  return result;
}
