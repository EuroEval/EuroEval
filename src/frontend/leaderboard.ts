// Parses the DataWrapper-style leaderboard CSVs into a typed table model.
//
// The CSVs have unusual conventions:
//   * Row 0 is a "task type" grouping header with HTML <span>/<a> tags.
//   * Row 1 is the real column header row, also containing HTML <a> tags.
//   * Cells may contain HTML anchor tags (rendered as a link in the UI).
//   * Numeric cells use a `display@@sortkey` suffix to provide an explicit
//     sort key (e.g. `1.15@@1.15`, `17.1.0.dev0@@170100`).
//   * Score cells use `60.17 ± 1.40 / 72.92 ± 1.18@@60.17` — two metrics
//     (primary / secondary), each with a confidence interval.

export type CellKind =
  | "model"
  | "number"
  | "icon"
  | "score"
  | "version"
  | "text";

export interface ParsedCell {
  /** Raw HTML for display (may contain a single safe <a> tag). */
  html: string;
  /** Plain text representation (for filtering / global search). */
  text: string;
  /** Sort key — numeric when sensible, else the lowercased text. */
  sortKey: number | string;
}

export interface Column {
  key: string;
  title: string;
  titleHtml: string;
  kind: CellKind;
  /** Distinct values across the rows (when small) — for dropdown filters. */
  distinctValues?: string[];
  /** Numeric min/max — for range filters. */
  min?: number;
  max?: number;
}

export interface Row {
  cells: ParsedCell[];
}

export interface LeaderboardTable {
  columns: Column[];
  rows: Row[];
}

// --- CSV parsing (RFC 4180) ------------------------------------------------

export function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;
  let i = 0;

  while (i < text.length) {
    const ch = text[i];

    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 2;
          continue;
        }
        inQuotes = false;
        i++;
        continue;
      }
      field += ch;
      i++;
      continue;
    }

    if (ch === '"') {
      inQuotes = true;
      i++;
      continue;
    }
    if (ch === ",") {
      row.push(field);
      field = "";
      i++;
      continue;
    }
    if (ch === "\r") {
      i++;
      continue;
    }
    if (ch === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
      i++;
      continue;
    }
    field += ch;
    i++;
  }
  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}

// --- Cell helpers ----------------------------------------------------------

const stripTags = (s: string): string =>
  s.replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();

const splitDisplaySort = (
  raw: string,
): { display: string; sort: string | null } => {
  const idx = raw.indexOf("@@");
  if (idx === -1) return { display: raw, sort: null };
  return { display: raw.slice(0, idx), sort: raw.slice(idx + 2) };
};

const ICONS = new Set(["🔍", "🧠", "📝", "🤔", "✓", "✗", "(✓)", "(✗)", "?", ""]);

const NUMERIC_COLS = new Set([
  "rank",
  "parameters",
  "vocabulary",
  "context",
]);

const ICON_COLS = new Set([
  "type",
  "commercial",
  "merge",
  "open",
  "trained from scratch",
]);

const VERSION_SUFFIX = /version$/i;

const parseNumberSafe = (s: string): number | null => {
  if (s === "?" || s === "" || s === "-" || s === "??") return null;
  // Allow comma-grouping just in case.
  const cleaned = s.replace(/,/g, "");
  const n = Number(cleaned);
  return Number.isFinite(n) ? n : null;
};

/**
 * Parse a single cell, given the column's inferred kind.
 */
// Collapse the partial-open-source tick into the standard tick: we no longer
// distinguish "fully open source" from "open-weight only", just open-weight
// vs. closed.
const normalizeIconText = (s: string): string => (s === "(✓)" ? "✓" : s);

export function parseCell(raw: string, kind: CellKind): ParsedCell {
  const { display, sort } = splitDisplaySort(raw);
  let text = stripTags(display);
  if (kind === "icon") text = normalizeIconText(text);

  // Whitelist <a href=...> tags only when rendering HTML; everything else
  // becomes plain text. Inputs come from a build-time CSV under our control,
  // so the risk is limited but we'd rather be safe.
  let safeHtml = sanitizeHtml(display);
  if (kind === "icon") safeHtml = normalizeIconText(safeHtml);

  let sortKey: number | string;
  if (sort !== null) {
    // Legacy `value@@sortkey` syntax (kept for backward compatibility with
    // older CSVs). New CSVs derive the sort key from the displayed value.
    const n = parseNumberSafe(sort);
    sortKey = n !== null ? n : sort.toLowerCase();
  } else if (kind === "number") {
    const n = parseNumberSafe(text);
    sortKey = n !== null ? n : Number.POSITIVE_INFINITY;
  } else if (kind === "score") {
    // Score cells look like "60.17 ± 1.40 / 72.92 ± 1.18" — sort by the
    // leading number, with sentinels going to the bottom.
    const m = text.match(/^-?\d+(?:\.\d+)?/);
    sortKey =
      m && parseNumberSafe(m[0]) !== null
        ? parseNumberSafe(m[0])!
        : Number.POSITIVE_INFINITY;
  } else if (kind === "icon") {
    // Order common icons for sensible sort (✓ > (✓) > ? > (✗) > ✗).
    const order: Record<string, number> = {
      "✓": 5,
      "(✓)": 4,
      "?": 3,
      "(✗)": 2,
      "✗": 1,
      "🤔": 4,
      "📝": 3,
      "🧠": 2,
      "🔍": 1,
    };
    sortKey = order[text] ?? 0;
  } else {
    sortKey = text.toLowerCase();
  }

  return { html: safeHtml, text, sortKey };
}

// Allow <a href="..." target="..."> tags only; strip everything else.
function sanitizeHtml(input: string): string {
  return input
    .replace(/<(?!\/?a(?=>|\s))[^>]*>/gi, "")
    .replace(/<a\s+([^>]*)>/gi, (_match, attrs: string) => {
      // Keep only href and add target/rel.
      const hrefMatch = attrs.match(/href\s*=\s*(['"])([^'"]*)\1/i);
      const href = hrefMatch?.[2];
      if (!href) return "";
      return `<a href="${escapeAttr(href)}" target="_blank" rel="noopener">`;
    });
}

function escapeAttr(s: string): string {
  return s.replace(/"/g, "&quot;").replace(/</g, "&lt;");
}

// --- Column inference -----------------------------------------------------

function inferKind(title: string, sampleValues: string[]): CellKind {
  const t = title.toLowerCase();
  if (t === "model") return "model";
  if (VERSION_SUFFIX.test(t)) return "version";
  if (ICON_COLS.has(t)) return "icon";
  if (NUMERIC_COLS.has(t)) return "number";

  // If all non-empty samples are numeric (after stripping HTML and @@), it's a number.
  // If they look like "x ± e / y ± e", it's a score.
  let numeric = true;
  let scoreLike = false;
  let allEmpty = true;
  for (const v of sampleValues.slice(0, 25)) {
    const text = stripTags(splitDisplaySort(v).display);
    if (!text) continue;
    allEmpty = false;
    if (/\d+(\.\d+)?\s*±/.test(text)) {
      scoreLike = true;
      numeric = false;
      break;
    }
    if (parseNumberSafe(text) === null && text !== "?") {
      numeric = false;
    }
  }
  if (scoreLike) return "score";
  if (numeric && !allEmpty) return "number";
  return "text";
}

// --- Public entry: parse a leaderboard CSV --------------------------------

export function parseLeaderboard(csvText: string): LeaderboardTable {
  const rows = parseCsv(csvText.trim());
  if (rows.length < 2) return { columns: [], rows: [] };

  // Row 0 is a task-type grouping; row 1 is the real header.
  const headerRow = rows[1];
  const dataRows = rows.slice(2);

  // Build column metadata.
  const columns: Column[] = headerRow.map((rawTitle, idx) => {
    let title = stripTags(rawTitle);
    let titleHtml = sanitizeHtml(rawTitle);
    const sampleValues = dataRows.map((r) => r[idx] ?? "");
    const kind = inferKind(title, sampleValues);
    // Rename the "Open" column to "Open-weight" — we no longer distinguish
    // fully open-source from open-weight-only.
    if (title.toLowerCase() === "open") {
      titleHtml = titleHtml.replace(/Open/i, "Open-weight");
      title = "Open-weight";
    }
    return {
      key: title || `col_${idx}`,
      title,
      titleHtml,
      kind,
    };
  });

  // Hide trailing version columns from the visible table — they're a lot of
  // noise and the user can still filter the rank column.
  const visibleColIndexes = columns
    .map((c, i) => ({ c, i }))
    .filter(({ c }) => c.kind !== "version")
    .map(({ i }) => i);

  const visibleColumns = visibleColIndexes.map((i) => columns[i]);

  // Parse cells.
  const parsedRows: Row[] = dataRows
    .filter((r) => r.length > 1 && stripTags(r[0]).trim() !== "")
    .map((r) => ({
      cells: visibleColIndexes.map((i) =>
        parseCell(r[i] ?? "", columns[i].kind),
      ),
    }));

  // Augment columns with min/max + distinct values.
  for (let c = 0; c < visibleColumns.length; c++) {
    const col = visibleColumns[c];
    if (col.kind === "number" || col.kind === "score") {
      let min = Number.POSITIVE_INFINITY;
      let max = Number.NEGATIVE_INFINITY;
      for (const r of parsedRows) {
        const k = r.cells[c].sortKey;
        if (typeof k === "number" && Number.isFinite(k)) {
          if (k < min) min = k;
          if (k > max) max = k;
        }
      }
      if (Number.isFinite(min)) col.min = min;
      if (Number.isFinite(max)) col.max = max;
    }
    if (col.kind === "icon") {
      const distinct = new Set<string>();
      for (const r of parsedRows) {
        const t = r.cells[c].text;
        if (t) distinct.add(t);
      }
      col.distinctValues = Array.from(distinct);
    }
  }

  return { columns: visibleColumns, rows: parsedRows };
}

// --- Build-time CSV lookup -------------------------------------------------

const csvModules = import.meta.glob("@/csv/*.csv", {
  query: "?raw",
  import: "default",
}) as Record<string, () => Promise<string>>;

export const csvKeys: string[] = Object.keys(csvModules).map((k) =>
  k.replace(/^.*\/csv\//, "").replace(/\.csv$/, ""),
);

/** Async-load and parse a leaderboard by stem (e.g. `danish_all`). */
export async function loadLeaderboard(
  stem: string,
): Promise<LeaderboardTable | undefined> {
  const entry = Object.entries(csvModules).find(([path]) =>
    path.endsWith(`/${stem}.csv`),
  );
  if (!entry) return undefined;
  const text = await entry[1]();
  return parseLeaderboard(text);
}

/** Load a leaderboard's raw CSV text (e.g. for a download button). */
export async function loadLeaderboardCsv(
  stem: string,
): Promise<string | undefined> {
  const entry = Object.entries(csvModules).find(([path]) =>
    path.endsWith(`/${stem}.csv`),
  );
  if (!entry) return undefined;
  return await entry[1]();
}
