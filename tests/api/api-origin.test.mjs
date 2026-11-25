import test from "node:test";
import assert from "node:assert/strict";
import { apiUrl } from "../../src/frontend/services/api.ts";

test("frontend API URLs remain same-origin until an API origin is configured", () => {
  for (const path of ["/api/issues?state=open&page=2", "/api/submit-evaluation", "/api/hall-of-fame"]) {
    assert.equal(apiUrl(path), path);
    assert.equal(apiUrl(path, "https://api.euroeval.com"), `https://api.euroeval.com${path}`);
  }
  assert.equal(apiUrl("/api/issues?_=123", "http://127.0.0.1:3000/"), "http://127.0.0.1:3000/api/issues?_=123");
});

test("frontend API configuration rejects credentials, non-HTTP URLs and URL components", () => {
  for (const origin of ["javascript:alert(1)", "ftp://api.example.com", "https://user:password@api.example.com", "https://api.example.com/path", "https://api.example.com?key=value", "https://api.example.com#fragment"]) {
    assert.throws(() => apiUrl("/api/issues", origin));
  }
});
