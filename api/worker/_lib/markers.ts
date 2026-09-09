import { GROUPS, PROTOCOL_VERSION, TITLE_PREFIX, VOLUNTEER_MARKER_RE, canonicalJson, removeFinalCredit } from "./protocol.ts";
import type { VolunteerLeaseMarker, VolunteerSubmission } from "./protocol.ts";

export function parseVolunteerMarker(body: string | null): VolunteerLeaseMarker | null {
  if (!body) return null;
  const matches = [...body.matchAll(new RegExp(VOLUNTEER_MARKER_RE.source, "gi"))];
  if (matches.length !== 1) return null;
  try {
    const parsed = JSON.parse(matches[0][1].trim()) as Partial<VolunteerLeaseMarker>;
    if (parsed.protocol_version !== PROTOCOL_VERSION || typeof parsed.coordinator !== "string" ||
        !parsed.coordinator || !["active", "submitted", "accepted", "rejected"].includes(parsed.submission || "") ||
        !Array.isArray(parsed.leases)) return null;
    const leases = parsed.leases.filter((lease): lease is VolunteerLeaseMarker["leases"][number] =>
      !!lease && typeof lease === "object" && typeof lease.lease_id === "string" &&
      typeof lease.language === "string" && typeof lease.worker === "string" &&
      typeof lease.contributor === "string" && typeof lease.expires_at === "string" &&
      !Number.isNaN(Date.parse(lease.expires_at)));
    if (leases.length !== parsed.leases.length || new Set(leases.map((item) => item.lease_id)).size !== leases.length || new Set(leases.map((item) => item.language)).size !== leases.length) return null;
    const submissions = Array.isArray(parsed.submissions) ? parsed.submissions.filter((item): item is VolunteerSubmission =>
      !!item && typeof item === "object" && typeof item.submission_id === "string" && !!item.submission_id &&
      typeof item.language === "string" && !!item.language && typeof item.manifest_path === "string" && !!item.manifest_path &&
      typeof item.submitted_at === "string" && !Number.isNaN(Date.parse(item.submitted_at)) &&
      typeof item.verified_contributor === "string" && !!item.verified_contributor &&
      Number.isSafeInteger(item.result_count) && item.result_count > 0 &&
      ["submitted", "accepted", "rejected"].includes(item.status)) : undefined;
    if (parsed.submissions !== undefined && (submissions === undefined || submissions.length !== parsed.submissions.length || new Set(submissions.map((item) => item.submission_id)).size !== submissions.length)) return null;
    const completed = Array.isArray(parsed.completed_languages) ? parsed.completed_languages.filter((item): item is string => typeof item === "string" && !!item) : undefined;
    if (parsed.completed_languages !== undefined && (completed === undefined || completed.length !== parsed.completed_languages.length)) return null;
    return { protocol_version: PROTOCOL_VERSION, coordinator: parsed.coordinator,
      submission: parsed.submission as VolunteerLeaseMarker["submission"], leases,
      ...(submissions === undefined ? {} : { submissions }), ...(completed === undefined ? {} : { completed_languages: completed }),
      ...(typeof parsed.signature === "string" ? { signature: parsed.signature } : {}) };
  } catch { return null; }
}
export function hasVolunteerMarker(body: string | null): boolean { return !!body?.match(VOLUNTEER_MARKER_RE); }
export function renderVolunteerMarker(marker: VolunteerLeaseMarker): string {
  return `<!-- euroeval-volunteer-worker:v1 ${canonicalJson(marker)} -->`;
}
export function replaceVolunteerMarker(body: string, marker: VolunteerLeaseMarker | null): string {
  const without = removeFinalCredit(body).replace(VOLUNTEER_MARKER_RE, "").replace(/\n{3,}/g, "\n\n").trimEnd();
  return marker ? `${without}\n\n${renderVolunteerMarker(marker)}\n` : `${without}\n`;
}

export function selectedLanguages(body: string | null): string[] {
  if (!body) return [];
  const languages: string[] = [];
  for (const [group, codes] of Object.entries(GROUPS)) {
    const pattern = new RegExp(`-\\s*\\[[xX]\\]\\s*${escapeRegExp(group)}(?:\\s|$)`, "m");
    if (pattern.test(body)) languages.push(...codes);
  }
  return languages;
}

export function claimableLanguages(
  selected: string[], marker: VolunteerLeaseMarker | null, now = Date.now(),
): string[] {
  if (!marker) return selected;
  return selected.filter((language) =>
    !marker.leases.some((item) => item.language === language && Date.parse(item.expires_at) > now) &&
    !(marker.submissions || []).some((item) => item.language === language &&
      ["submitted", "accepted"].includes(item.status)));
}
function escapeRegExp(value: string): string { return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

export function extractModelId(title: string, body: string | null): string | null {
  const match = body?.match(/(?:^|\n)#{1,6}\s*Model ID\s*\n+([^\n]+)/i);
  if (!match && !title.startsWith(`${TITLE_PREFIX} `)) return null;
  const value = (match?.[1] || title.slice(TITLE_PREFIX.length)).trim().replace(/^[`*_]+|[`*_]+$/g, "");
  return /^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+(?::[A-Z0-9_]+)?$/.test(value) ? value : null;
}
