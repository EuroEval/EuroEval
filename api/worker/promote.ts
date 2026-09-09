import {
  BrokerError, ConfigurationError, PROTOCOL_VERSION, acquireIssueMutex, fetchIssue,
  commentIssue, issueComments, json, method, parseVolunteerMarker, patchIssue,
  promotionSecret, readJson, releaseIssueMutex, requireProtocol, VolunteerLeaseMarker,
} from "./_lib";

export const config = { runtime: "edge" };
const PROMOTION_MARKER = "euroeval-volunteer-promotion:v1";

/** Authenticated, monotonic transition from submitted to accepted/rejected. */
export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    promotionSecret(req);
    const body = await readJson(req, 16 * 1024); requireProtocol(body);
    if (!Number.isSafeInteger(body.issue_number) || typeof body.submission_id !== "string" || !body.submission_id || body.outcome !== "accepted" && body.outcome !== "rejected") {
      throw new BrokerError(400, "issue_number, submission_id, and an accepted/rejected outcome are required.");
    }
    const issueNumber = body.issue_number as number;
    const submissionId = body.submission_id as string;
    const outcome = body.outcome as "accepted" | "rejected";
    const mutex = await acquireIssueMutex(issueNumber);
    if (!mutex) throw new BrokerError(409, "Issue is busy; retry promotion.");
    try {
      const issue = await fetchIssue(issueNumber);
      const marker = parseVolunteerMarker(issue.body);
      if (!marker) throw new BrokerError(409, "The issue ownership marker is missing or malformed.");
      const current = marker.submissions?.find((item) => item.submission_id === submissionId);
      if (!current) throw new BrokerError(404, "Submission is not present on this issue.");
      if (current.status === outcome) return json(200, { protocol_version: PROTOCOL_VERSION, status: outcome, submission_id: submissionId });
      if (current.status !== "submitted" && current.status !== undefined) throw new BrokerError(409, "Submission has already completed a different terminal transition.");
      const submissions = (marker.submissions || []).map((item) => item.submission_id === submissionId ? { ...item, status: outcome } : item);
      const next: VolunteerLeaseMarker = { ...marker, submission: outcome, submissions,
        ...(outcome === "accepted" ? { completed_languages: [...new Set([...(marker.completed_languages || []), current.language])] } : {}) };
      let promotedBody = replaceMarker(issue.body || "", next);
      if (outcome === "accepted" && current.contributor && !/euroeval-volunteer-credit:v1/i.test(promotedBody)) {
        promotedBody += `<!-- euroeval-volunteer-credit:v1 ${JSON.stringify({ immutable: true, winner: current.contributor })} -->\n`;
      }
      await patchIssue(issueNumber, promotedBody);
      const fenced = parseVolunteerMarker((await fetchIssue(issueNumber)).body);
      if (fenced?.submissions?.find((item) => item.submission_id === submissionId)?.status !== outcome) throw new BrokerError(409, "Promotion fence lost.");
      const comments = await issueComments(issueNumber);
      if (!comments.some((item) => item.body?.includes(`${PROMOTION_MARKER} ${submissionId}`))) {
        await commentIssue(issueNumber, `<!-- ${PROMOTION_MARKER} ${submissionId} -->\nCommunity submission **${submissionId}** was **${outcome}**.\n\nManifest: \`${current.manifest_path}\``);
      }
      return json(200, { protocol_version: PROTOCOL_VERSION, status: outcome, submission_id: submissionId, manifest_path: current.manifest_path });
    } finally { await releaseIssueMutex(issueNumber, mutex); }
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { protocol_version: PROTOCOL_VERSION, error: error instanceof Error ? error.message : "Unable to promote submission." });
  }
}

function replaceMarker(body: string, marker: VolunteerLeaseMarker): string {
  const without = body.replace(/<!--[\s\S]*?euroeval-volunteer-worker:v1\s+[\s\S]*?-->/i, "").replace(/\n{3,}/g, "\n\n").trimEnd();
  return `${without}\n\n<!-- euroeval-volunteer-worker:v1 ${JSON.stringify(marker)} -->\n`;
}
