# EuroEval Implementation Plan

## Overview

This document outlines the implementation plan for five improvements
to the EuroEval server-side evaluation pipeline and documentation.

---

## Task 1: GPU-Fit Check via Safetensors Headers

### Goal

Before running evaluation, check whether the model can fit on the
server's GPU. If not, skip to the next model without changing the
status of the large model in the queue.

### Current State

- `process_evaluation_queue.py` already calls `HfApi().model_info()`
  to get `safetensors.total` parameter count (line ~232).
- The parameter count is used only for sorting priority, not filtering.
- No GPU memory check is performed before running `euroeval`.

### Implementation Details

#### 1. Determine available GPU memory

- At script startup, query
  `torch.cuda.get_device_properties(i).total_memory` for each GPU
  and sum the available memory.
- Store as `AVAILABLE_GPU_MEMORY_BYTES` constant.

#### 2. Extract model size from safetensors headers

- The `model_info.safetensors` field already gives us `total` param
  count, but we need the actual **disk/memory size** in bytes.
- Use `model_info.safetensors.metadata` or iterate `model_info.safetensors`
  to get dtype + shape info, OR
- More reliably: use `HfApi().model_info(repo_id, files_metadata=True)`
  which returns `siblings` with file sizes, then sum the `.safetensors`
  file sizes.
- Alternative: compute from param count x bytes per param
  (bf16 = 2 bytes, fp32 = 4 bytes) + KV cache overhead estimate.

#### 3. Estimate required memory

- Model weights: `param_count x bytes_per_param`
  (assume bf16 = 2 bytes as default)
- Add overhead multiplier (e.g., 1.5x) to account for activation
  memory, KV cache, etc.
- Compare against `AVAILABLE_GPU_MEMORY_BYTES x gpu_memory_utilization`
  (default 0.8, matching EuroEval default)

#### 4. Skip logic in `process_issue()`

- Before calling `run_euroeval()`, check if estimated model size
  exceeds available GPU memory.
- If it does: log a warning, continue to next candidate (do NOT
  unassign, do NOT post error comment, do NOT change issue status).
- The model stays in the queue for when a larger server is available
  or when the user updates their request.

#### 5. New function to add

```python
def model_fits_gpu(
    model_info: ModelInfo, available_memory_bytes: int
) -> bool:
    """Estimate whether the model can fit in the available GPU memory."""
    # Implementation using safetensors metadata
```

### Files to Modify

- `src/scripts/process_evaluation_queue.py`

---

## Task 2: Prevent Concurrent Script Execution

### Goal

When the evaluation script starts, check if another instance is
already running. If so, abort with an error message that includes the
PID of the existing process.

### Current State

- No concurrency protection exists. The script relies on GitHub issue
  assignment as soft protection, but nothing prevents two instances of
  the script from running simultaneously on the same machine.

### Implementation Details

#### Approach: PID file with stale-check

1. **On startup**, check for a PID file
   (e.g., `~/.euroeval/evaluation_queue.pid`).

2. **If PID file exists:**

   - Read the PID
   - Check if the process is still alive using `os.kill(pid, 0)`
   - If alive: print error message with the PID and exit with code 1
   - If stale (process dead): remove the PID file and continue

3. **If no PID file:**

   - Write current PID to the file
   - Register cleanup handler (`atexit`) to remove the PID file on exit

4. **Error message format:**

```text
Another instance of process_evaluation_queue.py is already running
(PID: 12345).
Aborting. If this is a stale process, remove
~/.euroeval/evaluation_queue.pid and try again.
```

#### Alternative: `fcntl.flock()` for atomic file locking

(more robust, but less portable):

```python
import fcntl

lock_file = open(PID_PATH, 'w')
try:
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    # Read PID from file and report
```

### Recommended Approach

Use the PID file method with `atexit` cleanup. It is simpler, more
portable, and gives clear error messages. The `fcntl` approach can be
added later if needed.

### Files to Modify

- `src/scripts/process_evaluation_queue.py` (add at `main()` entry)

---

## Task 3: Robust Evaluation Result Verification

### Goal

Ensure all expected evaluation results are present in the JSONL file
before posting a GitHub comment. Handle missing results differently
based on error type (OOM vs. other).

### Current State

- The script compares JSONL lines before/after and posts new lines
  as a comment.
- If `run_euroeval()` returns `ok=False`, it posts an error comment
  and marks the issue as errored.
- There is no validation that the expected number of results were
  produced.
- Partial results (e.g., from an OOM mid-run) could be posted as
  successful.

### Implementation Details

#### 1. Calculate expected result count

- Before running, determine how many evaluations should produce
  results:
  - Count of `(dataset, task)` combinations for the given languages
    and model type
  - This can be derived from the language codes passed to `--language`

#### 2. Verify results after run

After `run_euroeval()` completes (even with `ok=True`):

- Read the new lines from JSONL (as already done)
- Count the new result entries
- Compare against expected count

#### 3. OOM detection

- Check the captured stderr/stdout for OOM indicators:
  - `"OutOfMemoryError"`, `"CUDA out of memory"`, `"oom"`,
    `" Killed"` (SIGKILL)
  - `"RuntimeError: CUDA"` patterns
- Function: `def is_oom_error(error_output: str) -> bool`

#### 4. Decision logic

```text
If run succeeded AND all results present:
  -> Post results as success comment (current behavior)

If run succeeded BUT results are missing:
  If OOM detected in logs:
    -> Skip to next model (same as Task 1, no status change)
  Else:
    -> Mark as error, post error comment (current error behavior)

If run failed:
  If OOM detected:
    -> Skip to next model (no status change)
  Else:
    -> Mark as error, post error comment (current behavior)
```

#### 5. Expected result count calculation

- Parse the `--language` flags to determine which datasets/tasks will
  be evaluated
- Use the existing `Benchmarker` or dataset config to count expected
  outputs
- Alternative: count the number of `--language` codes x average
  datasets per language (approximate)
- Better: import the dataset configs and count the exact combinations

### Files to Modify

- `src/scripts/process_evaluation_queue.py`

---

## Task 4: Cache HuggingFace API Calls

### Goal

Cache HF API calls to avoid rate limiting, particularly the model
existence check and other repeated API calls.

### Current State

- `result_processing.py` has an in-memory `Cache` dataclass seeded
  from prior run data
- `process_evaluation_queue.py` makes fresh `HfApi().model_info()`
  calls for each model
- No persistent caching of HF API responses
- Multiple scripts independently call `model_info()` for the same
  models

### Implementation Details

#### Design: Persistent disk cache with TTL

##### 1. Cache location

`~/.euroeval/hf_api_cache/` (JSON file or SQLite DB)

##### 2. Cached endpoints

|Endpoint|Key|TTL|Notes|
|--------|-----|-----|------|
|`model_info(repo_id)`|`repo_id`|24 hours|Most important - used for existence, param count, size|
|`model_info(repo_id, files_metadata=True)`|`repo_id:files`|24 hours|For safetensors size estimation|

##### 3. Cache implementation

```python
class HFCache:
    def __init__(self, cache_dir: str = "~/.euroeval/hf_api_cache"):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[dict]:
        cache_file = self.cache_dir / f"{hash(key)}.json"
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            if time.time() - data["timestamp"] < TTL_SECONDS:
                return data["result"]
            else:
                cache_file.unlink()  # remove expired
        return None

    def set(self, key: str, value: dict):
        cache_file = self.cache_dir / f"{hash(key)}.json"
        cache_file.write_text(
            json.dumps({"timestamp": time.time(), "result": value})
        )
```

##### 4. Integration points

- Wrap `HfApi().model_info()` calls in `process_evaluation_queue.py`
  with cache check
- Wrap `model_info()` calls in `result_processing.py` with cache check
- The cache should be shared across script runs for maximum benefit

##### 5. Rate-limit-friendly behavior

- Add exponential backoff on `429` responses
- Retry up to 3 times with increasing delays (1s, 2s, 4s)
- Log retry attempts

##### 6. Bonus: Cache other HF calls

- Frontend `huggingface.ts` search calls could benefit from caching
- `submit-evaluation.ts` model existence check could use the same
  cache

### Files to Modify

- `src/scripts/process_evaluation_queue.py` (add cache wrapper)
- `src/leaderboards/result_processing.py` (integrate cache)
- New file: `src/leaderboards/hf_cache.py` (shared cache module)

---

## Task 5: Improve Python Package Documentation

### Goal

Improve the documentation of the Python package on the website. It
should be clear, user-friendly, and comprehensive. Target audience:
users who want to evaluate models on their own.

### Current State

- `src/frontend/md/python-package.md` (1,289 lines) covers
  installation, quickstart, custom APIs, local models, offline mode,
  metadata overrides, custom datasets, custom tasks, output format,
  and debugging.
- `src/frontend/md/api.md` contains auto-generated API reference.
- Documentation is comprehensive but could be improved in structure,
  clarity, and user-friendliness.

### Implementation Plan

#### 1. Restructure `python-package.md`

- Add a table of contents at the top
- Reorganize into clearer sections with progressive difficulty:
  - Getting Started (installation, quickstart)
  - Running Evaluations (CLI, Python API, Docker)
  - Configuration (model metadata, GPU settings)
  - Custom Workflows (custom datasets, custom tasks, custom APIs)
  - Advanced Topics (offline mode, debugging, output analysis)
  - Reference (link to auto-generated API docs)

#### 2. Add missing content

- **GPU requirements section**: Explain GPU memory needs, how to
  check if a model fits, and troubleshooting OOM errors
- **Performance tips**: Batch size, context length, parallelism
  settings
- **Common errors and solutions**: OOM, CUDA errors, timeout,
  rate limiting
- **Examples section**: More concrete, copy-paste-ready examples for
  common use cases
- **FAQ integration**: Cross-reference with `faq.md` for
  evaluation-specific questions

#### 3. Improve readability

- Add code comments explaining what each output means
- Include expected output samples for each command
- Add warning/info boxes for common pitfalls
- Use consistent formatting throughout

#### 4. Update `api.md` generation

- Ensure the auto-generated API docs are properly linked from the
  main docs
- Add docstrings to any undocumented public functions/classes
- Consider adding examples in docstrings

#### 5. Add navigation

- Link from `python-package.md` to relevant `datasets/` and `tasks/`
  docs
- Link to leaderboard docs for users who want to understand scoring
- Link to CONTRIBUTING.md for users who want to contribute datasets

### Files to Modify

- `src/frontend/md/python-package.md` (restructure and expand)
- `src/frontend/md/api.md` (potentially update generation)
- Potentially: router/navigation files to ensure docs are discoverable

---

## Implementation Order

Recommended order for implementing these tasks:

1. **Task 2** (PID lock) - Simple, independent, reduces risk of
   concurrent runs
2. **Task 4** (HF cache) - Independent, benefits all other tasks
3. **Task 1** (GPU fit check) - Depends on Task 4 (uses cached
   model info)
4. **Task 3** (Result verification) - Can be implemented alongside
   Task 1
5. **Task 5** (Documentation) - Independent, can be done in parallel

---

## Risk Assessment

|Task|Risk|Mitigation|
|----|----|-----------|
|Task 1|Incorrect size estimation causes false skips|Conservative estimates, log all skip decisions|
|Task 2|Stale PID file blocks execution|Staleness check + clear removal instructions|
|Task 3|Overly strict result count causes false errors|Use approximate counts, log discrepancies|
|Task 4|Cache inconsistency|Clear TTL, invalidation on errors|
|Task 5|Documentation drift|Link to code examples, CI check for broken links|
