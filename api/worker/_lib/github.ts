import { BrokerError, REPO, REQUEST_LABEL, env } from "./protocol.ts";
import { fetchWithRetry } from "./http.ts";


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
export async function getIssue(number: number): Promise<GithubIssue> { return github(`/repos/${REPO}/issues/${number}`) as Promise<GithubIssue>; }
export const fetchIssue = getIssue;

export function hasIssueAssignee(issue: Pick<GithubIssue, "assignees">, login: string): boolean {
  return (issue.assignees || []).some((assignee) => assignee.login.toLowerCase() === login.toLowerCase());
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
      throw new BrokerError(422,
        "Your GitHub login cannot be assigned to evaluation issues; check repository access and try again.",
        "contributor_not_assignable");
    }
    throw new BrokerError(502,
      "Unable to verify whether your GitHub login can be assigned; try again later.",
      "assignee_eligibility_unavailable");
  }
}
export async function listOpenIssues(): Promise<GithubIssue[]> {
  const result: GithubIssue[] = [];
  for (let page = 1; page <= 10; page++) {
    const chunk = await github(`/repos/${REPO}/issues?state=open&labels=${encodeURIComponent(REQUEST_LABEL)}&per_page=100&page=${page}`) as GithubIssue[];
    for (const issue of chunk) if (!issue.pull_request) result.push(issue);
    if (chunk.length < 100) break;
  }
  return result;
}
export async function patchIssue(number: number, body: string): Promise<void> {
  await github(`/repos/${REPO}/issues/${number}`, { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ body }) });
}
export async function assignIssue(number: number, login: string): Promise<void> {
  await github(`/repos/${REPO}/issues/${number}/assignees`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ assignees: [login] }) });
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
