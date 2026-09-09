import {
  BrokerError, ConfigurationError, PROTOCOL_VERSION, coordinatorSecret, json, method,
  readJson, renewIssueMutex, requireProtocol,
} from "./_lib";

export const config = { runtime: "edge" };

/** Renew a coordinator lock only while its opaque token still owns it. */
export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    coordinatorSecret(req);
    const body = await readJson(req, 8 * 1024); requireProtocol(body);
    if (!Number.isSafeInteger(body.issue_number) || (body.issue_number as number) < 1 ||
        typeof body.token !== "string" || !body.token) throw new BrokerError(400, "issue_number and token are required.");
    const issueNumber = body.issue_number as number;
    const renewed = await renewIssueMutex(issueNumber, body.token);
    if (!renewed) throw new BrokerError(409, "Coordinator lock was lost or replaced.");
    return json(200, { protocol_version: PROTOCOL_VERSION, status: "renewed", issue_number: issueNumber, expires_in: 30 });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { protocol_version: PROTOCOL_VERSION, error: error instanceof Error ? error.message : "Unable to renew coordinator lock." });
  }
}
