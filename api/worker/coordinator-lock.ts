import {
  BrokerError, ConfigurationError, PROTOCOL_VERSION, acquireIssueMutex, coordinatorSecret,
  json, method, readJson, requireProtocol,
} from "./_lib";

export const config = { runtime: "edge" };

/** Acquire the same short Redis mutex used by volunteer claims/finalisation. */
export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    coordinatorSecret(req);
    const body = await readJson(req, 8 * 1024); requireProtocol(body);
    if (!Number.isSafeInteger(body.issue_number) || (body.issue_number as number) < 1) throw new BrokerError(400, "issue_number must be a positive integer.");
    const issueNumber = body.issue_number as number;
    const token = await acquireIssueMutex(issueNumber);
    if (!token) throw new BrokerError(409, "Issue is busy; retry coordinator lock.");
    return json(200, { protocol_version: PROTOCOL_VERSION, issue_number: issueNumber, token, expires_in: 30 });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { protocol_version: PROTOCOL_VERSION, error: error instanceof Error ? error.message : "Unable to acquire coordinator lock." });
  }
}
