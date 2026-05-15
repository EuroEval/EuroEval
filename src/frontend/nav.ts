import yaml from "js-yaml";
import configYaml from "@/config.yaml?raw";

declare const __PACKAGE_VERSION__: string | undefined;

const packageVersion =
  typeof __PACKAGE_VERSION__ !== "undefined" ? __PACKAGE_VERSION__ : "";

export interface NavPage {
  title: string;
  /** Markdown path under src/frontend/md/. Mutually exclusive with `csv`. */
  path?: string;
  /** Leaderboard CSV stem (e.g. `danish`). Mutually exclusive with `path`. */
  csv?: string;
  /** Explicit URL slug. Defaults to the file stem of `path` or `csv`. */
  slug?: string;
  /** Optional group heading shown in the sidebar above this page. */
  group?: string;
}

export interface NavSection {
  id: string;
  title: string;
  path?: string;
  indexPath?: string;
  pages?: NavPage[];
  toc?: boolean;
  /** Render a custom Vue component instead of markdown/CSV. */
  view?: string;
}

export interface GithubInfo {
  repo: string;
  version: string;
}

export interface NavConfig {
  github: GithubInfo;
  sections: NavSection[];
}

const rawConfig = yaml.load(configYaml) as Omit<NavConfig, "github"> & {
  github: { repo: string };
};

export const navConfig: NavConfig = {
  ...rawConfig,
  github: {
    ...rawConfig.github,
    version: packageVersion ? `v${packageVersion}` : "",
  },
};

export const defaultSection = navConfig.sections[0];

export function findSection(id: string | undefined): NavSection | undefined {
  if (!id) return undefined;
  return navConfig.sections.find((s) => s.id === id);
}

export function pageSlug(pageOrPath: NavPage | string): string {
  if (typeof pageOrPath === "string") {
    const file = pageOrPath.split("/").pop() || pageOrPath;
    return file.replace(/\.(md|csv)$/i, "");
  }
  if (pageOrPath.slug) return pageOrPath.slug;
  const ref = pageOrPath.path || pageOrPath.csv || "";
  return pageSlug(ref);
}

export function findPage(
  section: NavSection,
  pageSlugStr: string | undefined,
): NavPage | undefined {
  if (!pageSlugStr || !section.pages) return undefined;
  return section.pages.find((p) => pageSlug(p) === pageSlugStr);
}

export function resolveMdPath(
  sectionId: string | undefined,
  pageSlugStr: string | undefined,
): string | undefined {
  const section = findSection(sectionId) || defaultSection;
  if (!section) return undefined;

  if (!pageSlugStr) {
    return section.indexPath || section.path;
  }

  const match = findPage(section, pageSlugStr);
  return match?.path;
}

export function sectionUrl(section: NavSection): string {
  if (section.id === defaultSection.id) return "/";
  return `/${section.id}`;
}

export function pageUrl(sectionId: string, page: NavPage): string {
  return `/${sectionId}/${pageSlug(page)}`;
}

export interface SideNavGroup {
  title?: string;
  pages: NavPage[];
}

/** Group a section's pages by their `group` field, preserving order. */
export function groupedPages(section: NavSection): SideNavGroup[] {
  if (!section.pages) return [];
  const groups: SideNavGroup[] = [];
  let current: SideNavGroup | null = null;
  for (const page of section.pages) {
    const groupTitle = page.group;
    if (!current || current.title !== groupTitle) {
      current = { title: groupTitle, pages: [] };
      groups.push(current);
    }
    current.pages.push(page);
  }
  return groups;
}
