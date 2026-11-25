import { spawn, execFileSync } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { setTimeout as delay } from "node:timers/promises";

// Execute the real Redis commands used by both the broker and Upstash SDK.
export async function redisFixture(t) {
  const directory = await mkdtemp(path.join(tmpdir(), "ee-api-"));
  const socket = path.join(directory, "redis.sock");
  const server = spawn("redis-server", ["--port", "0", "--unixsocket", socket,
    "--save", "", "--appendonly", "no", "--dir", directory], { stdio: "ignore" });
  let startupError;
  server.on("error", (error) => { startupError = error; });
  t.after(async () => {
    if (server.exitCode === null && !startupError) {
      const exited = new Promise((resolve) => server.once("exit", resolve));
      server.kill("SIGTERM");
      await exited;
    }
    await rm(directory, { recursive: true, force: true });
  });
  const execute = (command) => {
    const output = execFileSync("redis-cli", ["--json", "-s", socket, ...command.map(String)],
      { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim();
    if (output.startsWith("error:")) return { error: JSON.parse(output.slice(6)) };
    return { result: output ? JSON.parse(output) : null };
  };
  for (let attempt = 0; ; attempt++) {
    if (startupError) throw startupError;
    try { execute(["PING"]); break; }
    catch (error) { if (attempt >= 100) throw error; await delay(10); }
  }
  return {
    execute,
    respond(init) {
      const commands = JSON.parse(init.body);
      const encoded = new Headers(init.headers).get("upstash-encoding") === "base64";
      const encode = (value) => Array.isArray(value) ? value.map(encode)
        : typeof value === "string" ? Buffer.from(value).toString("base64") : value;
      const run = (command) => {
        const response = execute(command);
        return encoded && "result" in response ? { result: encode(response.result) } : response;
      };
      return Response.json(Array.isArray(commands[0]) ? commands.map(run) : run(commands));
    },
  };
}
