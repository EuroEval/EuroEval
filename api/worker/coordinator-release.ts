import {
  BrokerError, ConfigurationError, PROTOCOL_VERSION, coordinatorSecret, json, method,
  readJson, releaseIssueMutex, requireProtocol,
} from "./_lib";

export const config = { runtime: "edge" };

/** Release a coordinator lock only when its opaque token still owns the key. */
export default async function handler(req: Request): Promise<Response> {
  const rejected = method(req); if (rejected) return rejected;
  try {
    coordinatorSecret(req);
    const body = await readJson(req, 8 * 1024); requireProtocol(body);
    if (!Number.isSafeInteger(body.issue_number) || (body.issue_number as number) < 1 || typeof body.token !== "string" || !body.token) throw new BrokerError(400, "issue_number and token are required.");
    const issueNumber = body.issue_number as number;
    const token = body.token as string;
    await releaseIssueMutex(issueNumber, token);
    return json(200, { protocol_version: PROTOCOL_VERSION, status: "released", issue_number: issueNumber });
  } catch (error) {
    const status = error instanceof BrokerError ? error.status : error instanceof ConfigurationError ? 503 : 502;
    return json(status, { protocol_version: PROTOCOL_VERSION, error: error instanceof Error ? error.message : "Unable to release coordinator lock." });
  }
}
