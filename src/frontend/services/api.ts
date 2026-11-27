const configuredOrigin = import.meta.env?.VITE_API_ORIGIN;

function apiOrigin(configuredOrigin: string | undefined): string | undefined {
  if (!configuredOrigin) return undefined;
  const url = new URL(configuredOrigin);
  if (!['http:', 'https:'].includes(url.protocol)) {
    throw new Error("VITE_API_ORIGIN must use http or https.");
  }
  if (
    url.username ||
    url.password ||
    url.pathname !== "/" ||
    url.search ||
    url.hash
  ) {
    throw new Error("VITE_API_ORIGIN must be an origin without path or credentials.");
  }
  return url.origin;
}

export function apiUrl(path: `/api/${string}`, configured: string | undefined = configuredOrigin): string {
  const origin = apiOrigin(configured);
  return origin ? new URL(path, origin).toString() : path;
}
