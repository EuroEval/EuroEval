import { BrokerError, REPO, REQUEST_LABEL, env, type Lease } from "./protocol.js";
import { parseVolunteerMarker, replaceVolunteerMarker } from "./markers.js";
import { signVolunteerMarker, verifyVolunteerMarker, type VolunteerLeaseMarker } from "./protocol.js";
import { fetchWithRetry } from "./http.js";


interface GithubIssue { number: number; title: string; body: string | null; state: string; assignees?: Array<{ login: string }>; labels?: Array<{ name?: string }>; pull_request?: unknown; }
export async function github(path: string, init: RequestInit = {}): Promise<any> {
  const token = env("GITHUB_TOKEN");
  const response = await fetchWithRetry(`https://api.github.com${path}`, {
    ...init,
    headers: { accept: "application/vnd.github+json", "x-github-api-version": "2022-11-28", authorization: `Bearer ${token}`, ...(init.headers || {}) },
  });
  if (!response.ok) throw new BrokerError(response.status === 404 ? 404 : 502, `GitHub returned HTTP ${response.status}: ${(await response.text()).slice(0, 300)}`);
  if (response.status === 204) return null;
  return response.json();
}
export async function getIssue(number: number, signal?: AbortSignal): Promise<GithubIssue> { return github(`/repos/${REPO}/issues/${number}`, { signal }) as Promise<GithubIssue>; }
export const fetchIssue = getIssue;

export function hasIssueAssignee(issue: Pick<GithubIssue, "assignees">, login: string): boolean {
  return (issue.assignees || []).some((assignee) => assignee.login.toLowerCase() === login.toLowerCase());
}

export async function requireLeaseOwnership(issue: GithubIssue, lease: Lease): Promise<void> {
  const marker = parseVolunteerMarker(issue.body);
  const markerLease = marker?.leases.find((item) => item.lease_id === lease.lease_id);
  if (!marker || !markerLease || !(await verifyVolunteerMarker(issue.number, marker)) ||
      markerLease.contributor.toLowerCase() !== lease.contributor.toLowerCase() ||
      (markerLease.ownership_mode || "assignee") !== (lease.ownership_mode || "assignee")) {
    throw new BrokerError(409, "The signed issue ownership marker is missing, invalid, or no longer owns this lease.", "lease_ownership_lost");
  }
  if (lease.ownership_mode !== "public") await requireAssignee(issue, lease.contributor);
}

export async function requireAssignee(issue: Pick<GithubIssue, "assignees">, login: string): Promise<void> {
  if (!hasIssueAssignee(issue, login)) {
    throw new BrokerError(409,
      "The lease contributor is no longer assigned to this issue.", "lease_assignment_lost");
  }
}

/** Check GitHub's documented repository-assignee eligibility endpoint. */
export async function assertAssignable(login: string): Promise<void> {
  try {
    await github(`/repos/${REPO}/assignees/${encodeURIComponent(login)}`);
  } catch (error) {
    if (error instanceof BrokerError && error.status === 404) {
      throw new BrokerError(422,        "Your GitHub login cannot be assigned to evaluation issues; check repository access and try again.", "github_login_not_assignable");
    }
    throw new BrokerError(502,
      "Unable to verify whether your GitHub login can be assigned; try again later.",
      "assignee_eligibility_unavailable");
  }
}
export async function listOpenIssues(signal?: AbortSignal): Promise<GithubIssue[]> {
  const result: GithubIssue[] = [];
  for (let page = 1; page <= 10; page++) {
    const chunk = await github(`/repos/${REPO}/issues?state=open&labels=${encodeURIComponent(REQUEST_LABEL)}&per_page=100&page=${page}`, { signal }) as GithubIssue[];
    for (const issue of chunk) if (!issue.pull_request) result.push(issue);
    if (chunk.length < 100) break;
  }
  return result;
}
export async function patchIssue(number: number, body: string, signal?: AbortSignal): Promise<void> {
  await github(`/repos/${REPO}/issues/${number}`, { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ body }), signal });
}
export async function assignIssue(number: number, login: string): Promise<void> {
  await github(`/repos/${REPO}/issues/${number}/assignees`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ assignees: [login] }) });
}

/** Reconcile best-effort public evaluator assignments without changing ownership. */
export async function reconcilePublicAssignments(
  issue: GithubIssue,
  marker: VolunteerLeaseMarker,
  assertOwned: () => Promise<void>,
): Promise<{ issue: GithubIssue; marker: VolunteerLeaseMarker }> {
  const desired = new Map<string, string>();
  const now = Date.now();
  for (const lease of marker.leases) {
    if (lease.ownership_mode === "public" && Date.parse(lease.expires_at) > now) {
      desired.set(lease.contributor.toLowerCase(), lease.contributor);
    }
  }
  for (const submission of marker.submissions || []) {
    if (submission.ownership_mode === "public" && submission.status === "submitted") {
      desired.set(submission.verified_contributor.toLowerCase(), submission.verified_contributor);
    }
  }

  const tracked = new Map((marker.public_assignees || []).map((login) => [login.toLowerCase(), login]));
  let current = issue;
  for (const [key, login] of desired) {
    if (hasIssueAssignee(current, login)) continue;
    try {
      await assertOwned();
      await assignIssue(issue.number, login);
      const observed = await fetchIssue(issue.number);
      if (hasIssueAssignee(observed, login)) {
        tracked.set(key, login);
        current = observed;
      }
    } catch {
      // GitHub silently ignores logins that cannot be assigned; signed lease
      // ownership remains authoritative and the queue projection still names them.
    }
  }
  for (const [key, login] of tracked) {
    if (desired.has(key)) continue;
    if (!hasIssueAssignee(current, login)) {
      tracked.delete(key);
      continue;
    }
    try {
      await assertOwned();
      await unassignIssue(issue.number, login);
      const observed = await fetchIssue(issue.number);
      if (!hasIssueAssignee(observed, login)) tracked.delete(key);
      current = observed;
    } catch {
      // Retain the signed provenance so a later heartbeat/claim can retry cleanup.
    }
  }

  const next = [...tracked.values()];
  const previous = marker.public_assignees || [];
  if (next.length !== previous.length || next.some((login) => !previous.some(
    (item) => item.toLowerCase() === login.toLowerCase(),
  ))) {
    await assertOwned();
    const fresh = await fetchIssue(issue.number);
    const freshMarker = parseVolunteerMarker(fresh.body);
    if (!freshMarker || !(await verifyVolunteerMarker(issue.number, freshMarker))) {
      return { issue: fresh, marker };
    }
    const signed = await signVolunteerMarker(issue.number, {
      ...freshMarker,
      public_assignees: next,
    });
    await patchIssue(issue.number, replaceVolunteerMarker(fresh.body || "", signed));
    current = await fetchIssue(issue.number);
    const patched = parseVolunteerMarker(current.body);
    if (patched && await verifyVolunteerMarker(issue.number, patched)) {
      return { issue: current, marker: patched };
    }
  }
  return { issue: current, marker };
}
export async function unassignIssue(number: number, login: string): Promise<void> {
  await github(`/repos/${REPO}/issues/${number}/assignees`, { method: "DELETE", headers: { "content-type": "application/json" }, body: JSON.stringify({ assignees: [login] }) });
}
export async function addIssueLabel(number: number, label: string): Promise<void> {
  await github(`/repos/${REPO}/issues/${number}/labels`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ labels: [label] }) });
}
export async function removeIssueLabel(number: number, label: string): Promise<void> {
  try { await github(`/repos/${REPO}/issues/${number}/labels/${encodeURIComponent(label)}`, { method: "DELETE" }); }
  catch (error) { if (!(error instanceof BrokerError) || error.status !== 404) throw error; }
}
export async function issueComments(number: number): Promise<Array<{ body?: string }>> {
  return github(`/repos/${REPO}/issues/${number}/comments?per_page=100`) as Promise<Array<{ body?: string }>>;
}
export async function commentIssue(number: number, body: string): Promise<void> {
  await github(`/repos/${REPO}/issues/${number}/comments`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ body }) });
}
