import { createApiServer, loadEvaluatedModels } from "./server.js";

const port = Number(process.env.PORT ?? 3000);
if (!Number.isInteger(port) || port < 0 || port > 65535) throw new Error("Invalid PORT");
const server = createApiServer({
  evaluatedModels: await loadEvaluatedModels(),
  allowedOrigins: (process.env.API_ALLOWED_ORIGINS ?? "https://euroeval.com")
    .split(",").map((value) => value.trim()).filter(Boolean),
});
server.listen(port, process.env.HOST ?? "127.0.0.1", () => {
  const address = server.address();
  if (address && typeof address !== "string") console.log(`API listening on ${address.address}:${address.port}`);
});
for (const signal of ["SIGINT", "SIGTERM"]) {
  process.once(signal, () => {
    server.close(() => process.exit(0));
    setTimeout(() => server.closeAllConnections(), 10_000).unref();
  });
}
