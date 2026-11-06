/**
 * Split a query into terms. Double-quoted phrases ("a b (c, d)") are kept as a
 * single term; everything else is split on whitespace.
 */
export function parseQueryTerms(query: string): string[] {
  const terms: string[] = [];
  const re = /"([^"]*)"?|(\S+)/g;
  for (const m of query.matchAll(re)) {
    const term = (m[1] ?? m[2] ?? "").trim();
    if (term) terms.push(term);
  }
  return terms;
}

/**
 * Check if text matches a query with OR semantics on whitespace-separated terms.
 * Double-quoted phrases are matched as one term, so labels containing spaces
 * (e.g. `"gpt-4o (zero-shot, val)"`) can be targeted exactly.
 * Empty/whitespace-only queries match everything. Case-insensitive.
 */
export function matchesQuery(text: string, query: string): boolean {
  const terms = parseQueryTerms(query);
  if (!terms.length) return true;
  const lowerText = text.toLowerCase();
  return terms.some((term) => lowerText.includes(term.toLowerCase()));
}
