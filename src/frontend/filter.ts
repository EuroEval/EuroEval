/**
 * Check if text matches a query with OR semantics on whitespace-separated terms.
 * Empty/whitespace-only queries match everything. Case-insensitive.
 */
export function matchesQuery(text: string, query: string): boolean {
  const trimmed = query.trim();
  if (!trimmed) return true;
  const terms = trimmed.split(/\s+/);
  const lowerText = text.toLowerCase();
  return terms.some((term) => lowerText.includes(term.toLowerCase()));
}
