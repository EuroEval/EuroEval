<!-- markdownlint-disable -->
# Evaluation Queue — Implementation Plan

This document plans the new **Evaluation Queue** feature: a page where visitors can submit Hugging Face models for evaluation and watch the queue progress, backed by GitHub Issues as the "database".

## 0. Decisions (already made)

| Topic | Decision |
| --- | --- |
| Submission flow | Frontend POSTs to a **serverless proxy on Vercel** that creates the issue via the GitHub REST API using `saattrupdan`'s PAT. Visitors do **not** sign in. |
| Queue fetching | Public GitHub REST API directly from the browser, with a **manual refresh button** (rate limit: 60/hr per IP, anonymous). |
| HF model picker | **Autocomplete** via `@huggingface/hub` (`listModels` / `huggingface.co/api/models?search=`). |
| Issue template | **Trim** `model_evaluation_request.yaml` to only **Model ID** + **Evaluation languages**. |
| Subscription | "Subscribe" button → 3-second buffer overlay → opens the GitHub issue in a new tab where the visitor uses GitHub's native Subscribe. |
| Hosting | Vercel (frontend + serverless function in the same project). |

## 1. Repo touchpoints

### Frontend (`src/frontend/`)

- `config.yaml` — add a new section `evaluation-queue` after `leaderboards`.
- `components/` — new component `EvaluationQueueView.vue` (page) + small sub-components:
  - `ModelSubmitForm.vue` (HF autocomplete + language-group checkboxes + submit button).
  - `QueueTable.vue` (rows of model / language group / status / subscribe).
  - `SubscribeRedirectModal.vue` (3-second buffer overlay).
- `components/LeaderboardView.vue` (lines 148–152) — append a link "Submit your own model on the Evaluation Queue page." after the existing help line.
- `router/` — add a route for `/evaluation-queue` (and possibly a slug-less alternative depending on existing routing convention).
- `nav.ts` — no schema change needed; new `NavSection` with a `path` rendering the Vue component instead of markdown. **Likely need a small extension**: today sections render markdown or CSV; introduce a third kind ("custom view") or special-case `id === "evaluation-queue"` in `App.vue` / wherever the section body is dispatched. Prefer a small `view?: string` field on `NavSection` for cleanliness.
- New file `lib/github.ts` — helpers: `listOpenEvalIssues()`, `parseIssueBody(body)` (extract model id + language groups), `issueStatus(issue)` (assignee → "Evaluating" / no assignee → "Waiting"), `submitEvalRequest(payload)` (calls our Vercel function).
- New file `lib/huggingface.ts` — wraps `@huggingface/hub` `listModels` for autocomplete.

### Backend / proxy

- `api/submit-evaluation.ts` (Vercel serverless function, TypeScript).
  - Validates request shape (`model_id: string`, `language_groups: string[]`).
  - Checks HF Hub for model existence (`GET https://huggingface.co/api/models/{model_id}`); 404 → 422 to client.
  - Checks GitHub for an **existing open issue** with the same model id (search API: `repo:EuroEval/EuroEval is:issue is:open label:"model evaluation request" in:title "[MODEL EVALUATION REQUEST] {model_id}"`); if found → 409 with the existing issue URL.
  - Creates a new issue via `POST /repos/EuroEval/EuroEval/issues` using a PAT stored as `GITHUB_TOKEN` env var (fine-grained PAT scoped to the EuroEval repo with **issues: write**).
  - Returns `{ url: string, number: number }`.
  - CORS: allow only the production + preview Vercel domains.
- `vercel.json` — add function config + rewrites (if needed). If we already deploy via Vercel, this just slots in.

### Issue template (`.github/ISSUE_TEMPLATE/model_evaluation_request.yaml`)

- Drop `Model type`, `Model size`, `Merged model` sections, leaving:
  - `Model ID` (input)
  - `Evaluation languages` (checkboxes — keep the same 9 options so they match the values our submission form posts).
- Issue body, when posted by our proxy, will be a markdown block matching what GitHub's form rendering produces so it's parseable both ways.

### Compute-server script (`src/scripts/`)

- New: `process_evaluation_queue.py` — runs on the GPU box.
  - Lists open issues with label `model evaluation request` (`gh api` or `requests`).
  - Filters: **unassigned**, model id exists on HF Hub.
  - Assigns issue to `saattrupdan` (`PATCH /repos/.../issues/{n}` with `assignees: ["saattrupdan"]`).
  - Reads model id + language groups from the issue body (regex-parse the form output).
  - Runs `euroeval` for each requested language group; writes/appends `euroeval_benchmark_results.jsonl`.
  - Posts the **new** result lines back as a comment on the issue, wrapped in <code>```jsonl …```</code>.
  - Idempotent: if comment with jsonl block already exists, skip the run.
  - Logs to stdout for cron/systemd visibility. Loop with sleep, or single-pass + crontab — single-pass is simpler.

### Local laptop script (`src/scripts/`)

- New: `collect_evaluation_results.py` — runs on Dan's laptop.
  - Lists open issues labelled `model evaluation request`, assigned (any assignee).
  - For each issue, scans **only the comments** (not the body), finds the **first fenced ```jsonl block** (regex: `` /```jsonl\s*\n([\s\S]*?)\n``` `` — surrounding prose is fine), and extracts only the lines inside the fence.
  - Concatenates all extracted jsonl lines into `<repo-root>/new_results.jsonl` (overwriting).
  - Invokes `generate_leaderboards.py` (existing) to update CSVs.
  - On success, **closes** each issue (`PATCH state=closed`) with a short comment ("Results merged into the leaderboards. Thanks!").
  - Closing the issue removes it from the queue (the frontend only shows open issues).

## 2. UX layout (page `/evaluation-queue`)

```
─────────────────────────────────────────────────────
 Evaluation Queue
 (intro paragraph: explain that anyone can submit a
  HF model; we run it on our hardware; results land
  in the leaderboards)

 ┌───────────────────────────────────────────────┐
 │  Submit a model                               │
 │  Model ID  [hf autocomplete ▾]                │
 │  Languages  ☐ Baltic  ☐ Finnic  ☐ Romance ... │
 │                                  [ Submit ]   │
 └───────────────────────────────────────────────┘

 Queue                                  [↻ Refresh]
 ┌──────────────────┬─────────────┬────────────┬───┐
 │ Model            │ Language gp │ Status     │   │
 ├──────────────────┼─────────────┼────────────┼───┤
 │ meta-llama/...   │ Scand,Germ. │ Evaluating │ ⇗ │
 │ google/gemma-…   │ Romance     │ Waiting    │ ⇗ │
 └──────────────────┴─────────────┴────────────┴───┘
   (Evaluating rows shown first, then Waiting.)
─────────────────────────────────────────────────────
```

- Model column links to `https://huggingface.co/{model_id}` (same convention as leaderboards).
- "Subscribe" (⇗) opens `SubscribeRedirectModal`: a small overlay with the message *"Taking you to the associated GitHub issue, where you can subscribe…"* + a countdown. After 3s, `window.open(issue.html_url, "_blank")`.
- Sort: status `Evaluating` first, then `Waiting`, then within each group oldest issue first (FIFO).
- Empty state: "🎉 No evaluations in the queue right now."

## 3. Data shapes

**Submission POST → proxy**

```json
{ "model_id": "meta-llama/Llama-3.1-8B", "language_groups": ["Scandinavian languages", "West Germanic languages"] }
```

**Issue body our proxy will post** (same layout as the GitHub form output, so the existing rendering and the laptop script's parser both work):

```
### Model ID

meta-llama/Llama-3.1-8B

### Evaluation languages

- [X] Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)
- [X] West Germanic languages (Dutch, English, German)
- [ ] Baltic languages (Latvian, Lithuanian)
...
```

**Parsing rule**: title is `[MODEL EVALUATION REQUEST] <model_id>`; language groups are the checked lines in the "Evaluation languages" section.

## 4. Live-update behaviour (compromise)

We chose **manual refresh** to stay within the 60/hr anonymous GitHub limit. Implementation:

- `Refresh` button calls `listOpenEvalIssues()`.
- After successful submission, we automatically refresh.
- After a subscribe-redirect, we refresh on tab refocus.
- Optional later: light auto-poll on a 60s timer **only when the tab is visible**, with a fallback to the public unauthenticated rate-limit headers (`X-RateLimit-Remaining`) — defer.

## 5. Security / abuse considerations

- Proxy **must** rate-limit by IP. Use `@upstash/ratelimit` + Upstash Redis (free tier) keyed on the `x-forwarded-for` Vercel header. Budget: **5 submissions / hour / IP**, returning 429 with a `Retry-After` header when exceeded.
- Validate model id format (`^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$`).
- Validate language-group strings against an allow-list (the same 9 strings the form ships with).
- Refuse submission if HF model doesn't exist or is gated/private (proxy hits HF API anonymously; if 401/403/404 → reject).
- Server-side dedupe (existing open issue) is authoritative; client check is purely for UX.
- PAT scope: fine-grained, repo-scoped, **issues: read & write**, no other permissions.

## 6. Build / packaging

- Add `@huggingface/hub` to `package.json` `dependencies`.
- Vercel function uses the `@octokit/rest` (or plain `fetch`) — pick `fetch` to avoid an extra dep.
- `vite.config.js` is fine as-is; the function lives outside `src/`.
- TS types for the new modules live alongside their sources (no global `.d.ts` needed).

## 7. Step-by-step task breakdown

1. **Backend**
   1. Add `api/submit-evaluation.ts` with submit + dedupe + HF-existence check.
   2. Document `GITHUB_TOKEN` env var requirement in README.
2. **Issue template**
   3. Trim `model_evaluation_request.yaml`.
3. **Frontend plumbing**
   4. Extend `NavSection` with optional `view?: string`, render `EvaluationQueueView.vue` for that section.
   5. Add the section to `config.yaml`.
   6. Add the "Submit your own model…" link to `LeaderboardView.vue`.
4. **Frontend components**
   7. `lib/github.ts` (issues listing + parsing).
   8. `lib/huggingface.ts` (autocomplete).
   9. `ModelSubmitForm.vue`.
   10. `QueueTable.vue`.
   11. `SubscribeRedirectModal.vue`.
   12. `EvaluationQueueView.vue` (assembles the above).
5. **Compute-server script**
   13. `src/scripts/process_evaluation_queue.py`.
6. **Local results-merging script**
   14. `src/scripts/collect_evaluation_results.py`.
7. **Polish**
   15. Empty-state, error toasts, loading skeletons.
   16. README updates explaining the two scripts and the env vars.

## 8. Open / deferrable items

- Auto-polling vs manual refresh — start manual; revisit if usage warrants.
- (resolved) Rate-limiting in the proxy — included from the start (Upstash, 5/hr/IP).
- Better submission UX (preview the rendered issue before submitting) — nice-to-have.
- Caching the HF model existence check in the proxy to reduce traffic — defer.
