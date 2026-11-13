import { spawn } from "node:child_process";
import { execFileSync } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { setTimeout as delay } from "node:timers/promises";

export async function withRedisRestBridge(run) {
  const directory = await mkdtemp(path.join(os.tmpdir(), "euroeval-redis-"));
  const socket = path.join(directory, "redis.sock");
  const server = spawn("/opt/homebrew/bin/redis-server", [
    "--port", "0", "--unixsocket", socket, "--unixsocketperm", "700",
    "--save", "", "--appendonly", "no", "--dir", directory,
  ], { stdio: "ignore" });
  const originalFetch = globalThis.fetch;
  const originalEnv = { ...process.env };
  let lostResponseMatcher;
  try {
    for (let attempt = 0; attempt < 100; attempt += 1) {
      try {
        execFileSync("/opt/homebrew/bin/redis-cli", ["-s", socket, "PING"], { stdio: "ignore" });
        break;
      } catch {
        if (server.exitCode !== null) throw new Error("Ephemeral Redis exited before ready");
        await delay(10);
      }
    }
    process.env.UPSTASH_REDIS_REST_URL = "https://redis.test";
    process.env.UPSTASH_REDIS_REST_TOKEN = "redis-token";
    globalThis.fetch = async (_input, init) => {
      const command = JSON.parse(init.body);
      const result = execFileSync("/opt/homebrew/bin/redis-cli", [
        "--json", "-s", socket, ...command.map(String),
      ], { encoding: "utf8" }).trim();
      if (lostResponseMatcher?.(command)) {
        lostResponseMatcher = undefined;
        throw new Error("simulated lost REST response after Redis commit");
      }
      return Response.json({ result: result ? JSON.parse(result) : null });
    };
    return await run({ loseNextRedisResponse: (matcher) => { lostResponseMatcher = matcher; } });
  } finally {
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
    server.kill("SIGTERM");
    await new Promise((resolve) => {
      if (server.exitCode !== null) resolve();
      else {
        server.once("exit", resolve);
        setTimeout(() => { server.kill("SIGKILL"); resolve(); }, 1000).unref();
      }
    });
    await rm(directory, { recursive: true, force: true });
  }
}
