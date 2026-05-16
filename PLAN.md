# Implementation Plan

Five tasks targeting the server-side evaluation queue runner, HF API caching, and
package docs. Items 1–4 modify `src/scripts/process_evaluation_queue.py` (and
related modules); item 5 is documentation work in `src/frontend/md/python-package.md`.

## Task 1 — Skip models that won't fit on the GPU (safetensors size check)

**Goal:** Before claiming an issue, inspect the model's safetensors header to
estimate VRAM footprint. If it exceeds available GPU memory, skip the issue
**without** mutating its queue status (no assign, no `errored-on` marker), so the
model stays in the queue for a larger GPU / future capacity.

**Where:**

- `src/scripts/process_evaluation_queue.py`
  - Reuse `huggingface_model_info()` (already returns `ModelInfo`).
  - Extend `huggingface_param_count()` (line 238) or add a sibling
    `estimate_model_bytes(info)` that sums per-dtype byte counts from
    `info.safetensors.parameters` (dict of dtype → param-count) using a
    `DTYPE_BYTES` table (`F32: 4`, `BF16/F16: 2`, `F8_*: 1`, `I4: 0.5`, etc.).
  - Add `available_gpu_bytes()` that calls
    `torch.cuda.mem_get_info()` (free, total) for each visible device; pick the
    largest free block. If CUDA is unavailable, treat as "no limit" so non-GPU
    boxes still work.
  - Add `fits_on_gpu(info)` returning a bool. Apply a configurable safety
    multiplier (env var `EUROEVAL_GPU_HEADROOM`, default `1.3`) to account for
    activations / KV cache.
  - In `main()` candidate loop (around line 119), after the HF lookup, call
    `fits_on_gpu(info)` and `continue` if it doesn't fit. Log clearly:
    `#N: skipping -- model needs ~Xgb, GPU has ~Ygb free (will retry later)`.
- No queue-state changes for this skip path (contrast with the error path).

**Edge cases:**

- Models without safetensors metadata → fall through to current behaviour
  (try to run; current sort already deprioritises them via `sys.maxsize`).
- Quantised models (`F8_E4M3`, `INT4`) — table must include them.
- Multi-GPU servers — sum free memory across devices if vLLM tensor-parallel
  is in use; gate behind env var `EUROEVAL_GPU_COUNT` (default 1).

## Task 2 — Single-instance guard (PID lock)

**Goal:** If `process_evaluation_queue.py` is already running, the second
invocation must exit with a non-zero status and a message naming the existing
PID.

**Where:** `src/scripts/process_evaluation_queue.py`, at the top of `main()`.

**Approach:** PID file at `${EUROEVAL_QUEUE_LOCK:-/tmp/euroeval-queue.pid}`.

1. If the file exists, read the PID and check liveness via `os.kill(pid, 0)`.
   - Alive → log `Queue processor already running (PID=<n>). Aborting.` and
     `sys.exit(1)`.
   - Dead (stale lock) → overwrite.
2. Else write current `os.getpid()`.
3. Register `atexit` handler + `signal.SIGTERM`/`SIGINT` handlers to unlink the
   PID file on exit.

Add a small helper `acquire_pid_lock(path)` so the logic is testable. Keep
stdlib-only (no `fcntl.flock` needed — PID-file is sufficient and simpler to
reason about across reboots).

## Task 3 — Robust evaluation-result accounting

**Goal:** Today, `run_euroeval` only treats subprocess non-zero exit as error.
But EuroEval catches most evaluation errors internally and still exits 0 — so
some expected results may be missing from the JSONL even on "success". Detect
this, distinguish OOM from other failures, and act accordingly.

**Where:**

- `src/scripts/process_evaluation_queue.py`, `process_issue()` (line 259).
- May need a small read-only helper in `src/euroeval/` exposing the expected
  benchmark-id list per `(model, language)` combination — or compute it
  locally from `LANGUAGE_GROUP_CODES` + a published task list. Prefer importing
  from `euroeval` (e.g. `euroeval.tasks` / `euroeval.dataset_configs`) so we
  stay in sync as datasets are added.

**Steps:**

1. **Compute expected results.** For each `(language, task/dataset)` the
   evaluation should produce, build the expected set of `(model_id, dataset)`
   keys. Use the same iteration order as `benchmarker.py` so this stays
   correct.
2. **Diff actual vs expected.** After `run_euroeval` returns, parse `new_lines`
   (the freshly appended JSONL records) and collect the set of completed
   dataset names. `missing = expected - actual`.
3. **Classify the failure.**
   - **OOM:** detect via either (a) substrings in captured stderr/stdout —
     `CUDA out of memory`, `torch.cuda.OutOfMemoryError`, `OOM`,
     `c10::OutOfMemoryError`; or (b) a per-record `error_type` field in the
     JSONL once we surface it (see "follow-up" below). Treat OOM as "skip and
     return to queue" — same behaviour as task 1: unassign, **do not** set
     `errored-on`, leave issue body untouched. Log clearly.
   - **Other error:** if missing results are present and the cause is not OOM,
     treat as a real error: post comment with the captured output (or a
     synthetic message listing missing datasets), call `set_errored_marker`,
     `unassign_issue` — mirroring the current non-zero-exit branch.
   - **All results present:** existing happy path — post the JSONL comment.
4. **Combine with current non-zero exit branch.** Refactor so both
   subprocess-failure and missing-result failures funnel through a single
   `_handle_error(number, body, captured_output, *, is_oom: bool)` helper.

**Follow-up (optional, separate PR):** have `benchmarker.py` write a record per
attempted (model, dataset) into the JSONL even on caught errors, with an
`error_type` field. This makes OOM detection robust instead of regex-based.

## Task 4 — Cache HF API calls

**Goal:** Avoid HTTP 429 / rate-limit failures from the Hugging Face Hub by
caching responses to the small set of metadata calls we make per run.

**Hot calls to cache (from research):**

- `process_evaluation_queue.huggingface_model_info()` — called once per open
  issue every cron tick.
- `src/euroeval/benchmark_modules/hf.py::get_model_repo_info()` — called from
  the CLI per model.
- `src/euroeval/custom_dataset_configs.py`, `split_utils.py`, `yaml_config.py`,
  `metrics/pipeline.py` — `HfApi()` / `hf_hub_download` lookups for dataset
  metadata / config files.

**Approach:**

1. Add a small caching layer in a new module
   `src/euroeval/hf_cache.py` exposing:
   - `cached_model_info(model_id, *, ttl=3600) -> ModelInfo | None`
   - `cached_dataset_info(repo_id, *, ttl=3600) -> DatasetInfo | None`
   - `cached_file_exists(repo_id, filename, repo_type, *, ttl=3600) -> bool`
2. Storage: on-disk JSON at
   `${HF_HOME:-~/.cache/huggingface}/euroeval-meta-cache.json` (single file,
   short-lived; key by call signature, value = `{ts, payload}`). Use
   `huggingface_hub.constants.HF_HUB_CACHE` to colocate.
3. In-process memoisation via `functools.lru_cache` on top of the disk layer
   so repeated calls within one run never re-read JSON.
4. Cache **negative** lookups too (model-not-found) with a shorter TTL
   (5 min) so we don't hammer the API for typo'd model IDs.
5. Migration: replace direct `HfApi().model_info(...)` /
   `hf_hub_download(...)` call sites with the cached wrappers. Keep behaviour
   identical for cache miss; verify by running the existing test suite.
6. Add `EUROEVAL_DISABLE_HF_CACHE=1` env var to bypass for debugging.

**Files to edit:**

- New: `src/euroeval/hf_cache.py` + tests under `tests/`.
- `src/scripts/process_evaluation_queue.py:220` (`huggingface_model_info`).
- `src/euroeval/benchmark_modules/hf.py` (`get_model_repo_info`).
- `src/euroeval/custom_dataset_configs.py`, `split_utils.py`,
  `yaml_config.py`, `metrics/pipeline.py` (any reads of file existence /
  dataset info).

## Task 5 — Improve the Python-package documentation page

**Goal:** Make `src/frontend/md/python-package.md` (rendered on the website,
1289 lines today) clear, friendly, and comprehensive for users who want to
evaluate their own models locally.

**Audit first (no edits yet).** Read the current page end-to-end and produce
an outline showing:

- What's missing (gaps vs. the CLI's actual flags from `src/euroeval/cli.py`).
- What's redundant or out of order.
- What's outdated (e.g. references to removed flags, old install commands).

**Proposed target structure:**

1. **At a glance** — what EuroEval is, what it evaluates, which backends.
2. **Installation** — `pip`, `uv`, optional extras (`vllm`, `openai`,
   `anthropic`, etc.), GPU/CPU notes, supported Python versions.
3. **Quickstart** — single command that benchmarks a small HF model on one
   language, with the expected output snippet.
4. **CLI reference** — table of every flag (auto-generated from
   `cli.py` if possible, or hand-written but verified against `--help`).
   Group: model selection, dataset/language selection, generation params,
   output/IO, backend-specific.
5. **Python API** — minimal `Benchmarker(...).benchmark(...)` example, links
   to the auto-generated mkapi reference.
6. **Backends** — one subsection each for HF Transformers, vLLM, OpenAI,
   Anthropic, LiteLLM, local servers. Include required env vars + a working
   example per backend.
7. **Selecting datasets / languages / tasks** — explain the
   language → task → dataset hierarchy with a small diagram.
8. **Output format** — schema of `euroeval_benchmark_results.jsonl`, fields,
   how to submit results to the public leaderboard.
9. **Caching & reproducibility** — HF cache, `--cache-dir`, deterministic
   seeds, what's cached between runs.
10. **Troubleshooting** — OOM, rate-limit, "model not found", quant-model
    gotchas. Each entry: symptom → cause → fix.
11. **FAQ** — keep tightly scoped; link to the main FAQ page for the rest.

**Cross-checking:** for every flag and example, run it locally (or grep
`cli.py`) to confirm it still works. No fabricated CLI flags.

**Out of scope** (note explicitly in the PR description): the rest of the
website pages (`api.md`, `methodology.md`, `faq.md`, `about.md`). Touch only
`python-package.md`.

## Suggested execution order

Tasks 2, 4, 5 are independent and can ship as separate PRs in any order.
Tasks 1 and 3 share helpers (OOM detection, GPU-memory check) and benefit
from being done together. Recommended:

1. **PR 1 — Task 2** (smallest, highest safety value): PID lock.
2. **PR 2 — Task 4**: HF caching (unblocks tasks 1 & 3 from hitting rate
   limits while we iterate).
3. **PR 3 — Tasks 1 + 3**: GPU-fit pre-check + missing-result/OOM handling.
   Single PR because they share the "skip without state change" code path.
4. **PR 4 — Task 5**: docs rewrite (no code risk, easy to review in
   isolation).

## Testing notes

- Tasks 1–3: add unit tests under `tests/scripts/` mocking `HfApi`,
  `subprocess.run`, and `torch.cuda.mem_get_info`. The script currently has
  no tests, so the first PR in this group should also wire up the test scaffold.
- Task 4: tests for cache hit/miss, TTL expiry, negative caching, env-var
  bypass.
- Task 5: build the site locally (`mkdocs serve` per `pyproject.toml`) and
  visually verify rendering before merging.
