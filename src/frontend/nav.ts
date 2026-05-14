import yaml from "js-yaml";
import configYaml from "@/config.yaml?raw";

export interface NavPage {
  path: string;
  title: string;
}

export interface NavSection {
  id: string;
  title: string;
  path?: string;
  indexPath?: string;
  pages?: NavPage[];
  toc?: boolean;
}

export interface GithubInfo {
  repo: string;
  version: string;
}

export interface NavConfig {
  github: GithubInfo;
  sections: NavSection[];
}

export const navConfig = yaml.load(configYaml) as NavConfig;

export const defaultSection = navConfig.sections[0];

export function findSection(id: string | undefined): NavSection | undefined {
  if (!id) return undefined;
  return navConfig.sections.find((s) => s.id === id);
}

export function pageSlug(path: string): string {
  // datasets/danish.md -> danish, tasks/speed.md -> speed
  const file = path.split("/").pop() || path;
  return file.replace(/\.md$/i, "");
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

  if (section.pages) {
    const match = section.pages.find((p) => pageSlug(p.path) === pageSlugStr);
    return match?.path;
  }
  return section.path;
}

export function sectionUrl(section: NavSection): string {
  if (section.id === defaultSection.id) return "/";
  return `/${section.id}`;
}

export function pageUrl(sectionId: string, page: NavPage): string {
  return `/${sectionId}/${pageSlug(page.path)}`;
}
