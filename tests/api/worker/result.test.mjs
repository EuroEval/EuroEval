import test from "node:test";
import assert from "node:assert/strict";
import { fetch as resultFetch } from "../../../api/worker/result.ts";

test("result rejects unsupported HTTP methods", async () => {
  const response = await resultFetch(new Request("https://euroeval.test/api/worker/result", {
    method: "GET",
  }));

  assert.equal(response.status, 405);
  assert.deepEqual(await response.json(), { error: "Method not allowed" });
});
