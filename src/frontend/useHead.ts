import { watchEffect } from "vue";

const DEFAULT_TITLE =
  "EuroEval — The robust European language model benchmark";
const DEFAULT_DESCRIPTION =
  "EuroEval is the standard benchmark for evaluating language models across 30+ European languages. Browse leaderboards by language, inspect per-task scores, and reproduce results with the open-source Python package.";
const BASE_URL = "https://euroeval.com";

interface HeadInput {
  /** Page-specific title. Will be suffixed with " — EuroEval" automatically. */
  title?: string;
  /** Page-specific meta description. */
  description?: string;
  /** Path used to build a canonical URL. Defaults to current location. */
  path?: string;
}

function setMeta(name: string, value: string, attr: "name" | "property" = "name") {
  let el = document.head.querySelector<HTMLMetaElement>(
    `meta[${attr}="${name}"]`,
  );
  if (!el) {
    el = document.createElement("meta");
    el.setAttribute(attr, name);
    document.head.appendChild(el);
  }
  el.setAttribute("content", value);
}

function setLink(rel: string, href: string) {
  let el = document.head.querySelector<HTMLLinkElement>(`link[rel="${rel}"]`);
  if (!el) {
    el = document.createElement("link");
    el.setAttribute("rel", rel);
    document.head.appendChild(el);
  }
  el.setAttribute("href", href);
}

/**
 * Reactively update the document head (title, description, canonical URL,
 * Open Graph + Twitter card mirrors) based on the input getter.
 */
export function useHead(get: () => HeadInput): void {
  watchEffect(() => {
    const { title, description, path } = get();
    const finalTitle = title ? `${title} — EuroEval` : DEFAULT_TITLE;
    const finalDescription = description || DEFAULT_DESCRIPTION;
    const finalUrl =
      BASE_URL + (path ?? window.location.pathname + window.location.search);

    document.title = finalTitle;
    setMeta("description", finalDescription);
    setMeta("og:title", finalTitle, "property");
    setMeta("og:description", finalDescription, "property");
    setMeta("og:url", finalUrl, "property");
    setMeta("twitter:title", finalTitle);
    setMeta("twitter:description", finalDescription);
    setLink("canonical", finalUrl);
  });
}
