<!-- markdownlint-disable -->

# EuroEval — Implementation Plan

This plan addresses 5 improvement tasks for the EuroEval project. Each task includes
an overview, specific implementation details, dependencies, recommended order, and
risks.

---

## Task 1: Model Size Check via Safetensors Headers

### Overview

Before running `euroeval` for a model, check if the model's safetensors metadata
indicates it will fit on the server's GPU. If the model is too large, skip it and
proceed to the next model in the queue without changing the large model's status
(the model remains assigned so it can be retried later with different hardware
or configuration).

### Implementation Details

**File:** `src/scripts/process_evaluation_queue.py`

**New function:** `check_model_gpu_fit(model_id: str) -> bool`

- Use `huggingface_hub.get_safetensors_metadata()` (already imported in
  `src/euroeval/safetensors_utils.py`) to read `metadata.parameter_count`.
- Convert parameter count to estimated GPU memory:
  - FP16 weights: `param_count * 2` bytes
  - Add optimizer state overhead (~2x for AdamW), gradient state (~2x), activations
    (proportional to context length)
  - Simplified rule: `estimated_bytes = param_count * 2 * 4` (FP16 weights +
    optimizer + gradients + activations buffer)
- Compare against a configurable environment variable `SERVER_GPU_MEMORY_BYTES`
  (default: 80 GB for A100-80GB). If `estimated_bytes > SERVER_GPU_MEMORY_BYTES`,
  return `False`.
- Also check `safetensors.total` from `ModelInfo` (line 252–256 of
  `process_evaluation_queue.py` already accesses this).

**Changes to `main()` (line 96–129):**

- After `info = huggingface_model_info(model_id=model_id)` (line 119) and
  `param_count = huggingface_param_count(info=info)` (line 124), add:

```python
if not check_model_gpu_fit(model_id=model_id):
    logger.info(
        f"#{number}: skipping -- model {model_id!r} estimated to be too large "
        f"for server GPU ({param_count:,} params)."
    )
    continue
```

- This skips the model without assigning the issue, so the model stays in the
  "waiting" state for the next queue cycle (when hardware changes or the model
  is re-requested with different settings).

**Configuration:**

- Add `SERVER_GPU_MEMORY_BYTES = int(os.environ.get("SERVER_GPU_MEMORY_BYTES", 80 * 1024 * 1024 * 1024))`
  to the top of the script.
- Add `GPU_MEMORY_PER_PARAM = 4` (bytes per parameter: 2 for FP16 weights + 2 for
  optimizer state as baseline).

**Dependencies:** None. This task is self-contained.

### Risks

- Parameter count is not a perfect proxy for GPU memory (quantization, context
  length, batch size all matter). The estimate should be conservative (overestimate
  rather than under).
- If the server has multiple GPUs, the check should use per-GPU memory.

---

## Task 2: Process Lock / Duplicate Execution Guard

### Overview

Prevent `process_evaluation_queue.py` from running concurrently. If an instance
is already running, abort with an error message containing the PID of the existing
process.

### Implementation Details

**File:** `src/scripts/process_evaluation_queue.py`

**Approach:** File-based lock using a lock file in `/tmp` or the project directory.

**New function:** `acquire_lock() -> None`

```python
LOCK_FILE = Path("/tmp/euroeval_queue.lock")

def acquire_lock() -> None:
    """Acquire an exclusive lock. Abort if already held.

    Raises:
        RuntimeError: If another instance holds the lock.
    """
    if LOCK_FILE.exists():
        existing_pid = int(LOCK_FILE.read_text().strip())
        # Check if the existing process is still alive
        try:
            os.kill(existing_pid, 0)
        except OSError:
            # Process is dead; clean up stale lock
            logger.warning(f"Stale lock file found for PID {existing_pid}. Cleaning up.")
            LOCK_FILE.unlink()
        else:
            raise RuntimeError(
                f"Another instance is already running (PID {existing_pid}). "
                "Aborting."
            )

    LOCK_FILE.write_text(str(os.getpid()))
    logger.debug(f"Lock acquired (PID {os.getpid()}).")
```

**Changes to `main()` (line 80):**

- Add `acquire_lock()` as the first statement in the try block.
- Add cleanup on exit:

```python
try:
    acquire_lock()
    issues = list_open_unassigned_issues()
    # ... rest of main ...
finally:
    try:
        LOCK_FILE.unlink()
    except FileNotFoundError:
        pass
```

**Alternative approach (if file-based locks are unreliable):** Use `fcntl`
file locking (`fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)`). This provides
kernel-level atomic locking and is more robust on Unix systems.

```python
LOCK_FD = None

def acquire_lock() -> None:
    global LOCK_FD
    LOCK_FD = open(LOCK_FILE, "w")
    try:
        fcntl.flock(LOCK_FD, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        existing_pid = int(LOCK_FILE.read_text().strip())
        raise RuntimeError(f"Another instance is already running (PID {existing_pid}).")
    LOCK_FD.write(str(os.getpid()))
    LOCK_FD.flush()
```

**Dependencies:** None. Standalone task.

### Risks

- Stale lock: if the process crashes, the lock file persists. The `os.kill(existing_pid, 0)` check handles this.
- Cross-platform: `fcntl` is Unix-only. The file-based approach works on any platform.
- For the server environment (Linux), `fcntl` is recommended.

---

## Task 3: Evaluation Error Handling with Result Verification

### Overview

Before posting GitHub comments with results, verify that all expected evaluation
results are present in the JSONL file. Handle OOM errors (skip model, proceed to
next) and other errors (set status to "error", post error message).

### Implementation Details

**File:** `src/scripts/process_evaluation_queue.py`

**New function:** `verify_expected_results(model_id: str, groups: list[str], jsonl_lines: list[str]) -> bool`

- For each language group, construct the expected language codes:

```python
def verify_expected_results(
    model_id: str,
    groups: list[str],
    jsonl_lines: list[str],
) -> bool:
    """Verify that all expected results are present in the JSONL output.

    Returns:
        True if all expected results are present, False otherwise.
    """
    expected_languages: list[str] = []
    for g in groups:
        expected_languages.extend(LANGUAGE_GROUP_CODES[g])
    expected_languages = sorted(set(expected_languages))

    # Parse each JSONL line and check for expected model+language combinations
    found_models: set[str] = set()
    found_languages: set[str] = set()

    for line in jsonl_lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        if record.get("model") != model_id:
            continue

        found_models.add(model_id)
        for lang in record.get("languages", []):
            found_languages.add(lang)

    # Check that all expected languages have results
    missing_languages = set(expected_languages) - found_languages
    if missing_languages:
        logger.warning(
            f"#{model_id}: missing results for languages: {sorted(missing_languages)}"
        )
        return False

    return True
```

**Changes to `process_issue()` (line 259):**

- After `ok, error_output = run_euroeval(model_id=model_id, languages=languages)`
  (line 280) and `new_lines = [line for line in after if line not in before]`
  (line 282):

```python
if not ok:
    # Check if the error is an OOM error
    if "CUDA" in error_output and ("out of memory" in error_output.lower() or "OOM" in error_output.upper()):
        logger.warning(
            f"#{number}: OOM error for {model_id!r} -- skipping to next model."
        )
        # Do not assign error marker; model stays unassigned for retry
        # with different hardware or configuration
        return
    # For other errors, proceed with error handling
    version = euroeval_version()
    comment = (
        f"Error encountered during evaluation:\n\n"
        f"```bash\n{error_output or '(no output captured)'}\n```\n\n"
        f"EuroEval version: v{version}\n"
    )
    comment_on_issue(number=number, comment=comment)
    set_errored_marker(number=number, body=issue.get("body"), version=version)
    unassign_issue(number=number)
    logger.info(f"#{number}: marked errored on v{version}, returned to queue.")
    return
```

- After `if not new_lines:` (line 297), add verification:

```python
if not verify_expected_results(model_id=model_id, groups=groups, jsonl_lines=new_lines):
    logger.warning(
        f"#{number}: expected results not fully present for {model_id!r}. "
        "Leaving issue assigned for manual inspection."
    )
    return
```

**New exception class:** In `src/euroeval/exceptions.py`, add:

```python
class OutOfMemoryError(Exception):
    """The evaluation ran out of GPU memory."""

    def __init__(self, message: str = "The evaluation ran out of GPU memory.") -> None:
        self.message = message
        super().__init__(self.message)
```

### Dependencies

- Depends on Task 1 (model size check) to reduce OOM occurrences.
- Independent from Tasks 2, 4, 5.

### Risks

- OOM detection relies on error message parsing, which may vary across PyTorch/vLLM versions.
- If a partial result is produced (e.g., some languages succeed, others OOM), the
  verification will catch it and leave the issue assigned.

---

## Task 4: Cache HF API Calls

### Overview

Cache the HuggingFace model existence check (`HfApi().model_info()` calls) to avoid
rate limiting. Also consider caching other HF API calls that could benefit from
caching.

### Implementation Details

**File:** `src/scripts/process_evaluation_queue.py`

**Approach:** In-memory cache with `functools.lru_cache` or a simple dict, with
TTL-based expiration.

**New function:** `cached_hf_model_info(model_id: str) -> object | None`

```python
from functools import lru_cache
import time

# Cache TTL of 5 minutes (300 seconds) to avoid rate limiting
HF_CACHE_TTL = 300
_hf_cache: dict[str, tuple[object | None, float]] = {}

def cached_hf_model_info(model_id: str) -> object | None:
    """Return HF Hub metadata for ``model_id``, cached to avoid rate limiting.

    Args:
        model_id:
            The Hugging Face repo id to look up.

    Returns:
        The ``ModelInfo`` object returned by ``HfApi``, or None when the
        lookup fails for any reason.
    """
    # Check cache
    if model_id in _hf_cache:
        result, timestamp = _hf_cache[model_id]
        if time.time() - timestamp < HF_CACHE_TTL:
            return result

    # Fetch from HF
    result = huggingface_model_info(model_id=model_id)

    # Store in cache
    _hf_cache[model_id] = (result, time.time())

    return result
```

**Changes to `main()` (line 119):**

- Replace `info = huggingface_model_info(model_id=model_id)` with
  `info = cached_hf_model_info(model_id=model_id)`.

**Additional caching opportunities:**

1. `HfApi().model_info()` calls in `src/euroeval/safetensors_utils.py`
   (`get_num_params_from_safetensors_metadata`) — use the same cache.

2. GitHub API calls in `gh_request()` (line 462): cache by path+params+method.
   Add a `gh_cached_request` wrapper with TTL of 30 seconds for list operations
   (issues listing, comments).

```python
_github_cache: dict[str, tuple[dict | list | None, float]] = {}

def cached_gh_request(
    path: str,
    *,
    method: str = "GET",
    body: dict | None = None,
    params: dict | None = None,
) -> dict | list | None:
    """Cached wrapper around gh_request with 30s TTL."""
    cache_key = f"{method}:{path}:{json.dumps(params, sort_keys=True) if params else ''}"
    if cache_key in _gh_cache:
        result, timestamp = _gh_cache[cache_key]
        if time.time() - timestamp < 30:
            return result

    result = gh_request(path, method=method, body=body, params=params)
    _gh_cache[cache_key] = (result, time.time())
    return result
```

- Replace `gh_request()` calls with `cached_gh_request()` in `list_open_unassigned_issues()`
  and `assign_issue()` (assign is write, so no cache needed).

**Dependencies:** None. Standalone task.

### Risks

- Cache invalidation: if a model is deleted from HF Hub, the cache would return
  a stale result. The TTL handles this (5 min).
- Thread safety: if the script uses multiprocessing, the in-memory cache won't
  work. The current single-threaded design is fine.
- Memory: the cache grows with each model. Add a max-size limit (e.g., 1000 entries)
  using `collections.OrderedDict` with eviction.

---

## Task 5: Improve Python Package Documentation

### Overview

Improve the documentation on the website (in the `site/` directory) to be clear,
user-friendly, and comprehensive. Target audience: users who want to use the
package to evaluate models on their own.

### Implementation Details

**Files:** `site/` directory (generated from mkdocs + mkapi)

**Current state:** The site is auto-generated from docstrings using `mkdocs-material`
+ `mkapi`. The `site/python-package/index.html` shows the "Python Package" section.

**Changes needed:**

1. **Update `src/euroeval/__init__.py` module docstring** (line 1):

```python
"""EuroEval — Benchmark European language models with ease.

EuroEval is a robust benchmarking framework for evaluating large language models
on European languages. It supports 25+ languages across 9 language groups, with
tasks covering translation, summarization, question answering, and more.

Quick start
-----------
>>> import euroeval
>>> results = euroeval.Benchencer().benchmark("meta-llama/Llama-3.1-8B")
>>> print(results)

CLI usage
---------
>>> euroeval --model meta-llama/Llama-3.1-8B --language da --language en

Installation
------------
>>> pip install euroeval[all]  # with generative model support
"""
```

2. **Update `src/euroeval/cli.py` docstrings** — add clear examples for each CLI
   flag.

3. **Update `src/euroeval/benchmarker.py` class docstring** — restructure to include:
   - A "What is Benchmarker?" section
   - A "Quick example" section
   - A "Parameters reference" section

4. **Add a `docs/` directory** with hand-written markdown pages:
   - `docs/getting-started.md` — installation, first benchmark, interpreting results
   - `docs/cli-reference.md` — CLI flags with examples
   - `docs/advanced-usage.md` — custom datasets, custom tasks, model caching
   - `docs/troubleshooting.md` — common errors, GPU memory issues, HF token setup

5. **Update `mkdocs.yml`** (create if not exists) to include the new pages:

```yaml
site_name: EuroEval Documentation
docs_dir: docs
nav:
  - Home: index.md
  - Getting Started: getting-started.md
  - CLI Reference: cli-reference.md
  - Advanced Usage: advanced-usage.md
  - Troubleshooting: troubleshooting.md
  - Python Package: python-package/
```

6. **Update `site/index.html`** — the landing page should have:
   - A clear "Get Started" section with installation command
   - A "Quick Example" code block
   - Links to the full documentation

**Source files to update:**

- `src/euroeval/__init__.py` — module docstring
- `src/euroeval/cli.py` — CLI docstring
- `src/euroeval/benchmarker.py` — Benchmarker class docstring
- `site/python-package/index.html` — regenerate from updated docstrings

**Dependencies:** Depends on Task 3 (error handling docs need the new error types).

### Risks

- Documentation regeneration (`make publish-docs`) must be run after changes.
- The mkapi auto-generation may override manual edits. Hand-written markdown in
  `docs/` takes precedence over auto-generated API docs.

---

## Recommended Implementation Order

| Step | Task | Reason |
|------|------|--------|
| 1 | **Task 2** (Process lock) | Lowest risk, no dependencies. Prevents race conditions during development of other tasks. |
| 2 | **Task 4** (Cache HF API calls) | Reduces rate-limiting issues. Simple to implement, no behavioral changes. |
| 3 | **Task 1** (Model size check) | Reduces OOM errors. Depends on Task 4 for efficient metadata fetching. |
| 4 | **Task 3** (Error handling) | Handles remaining errors gracefully. Depends on Task 1 (fewer OOMs to handle). |
| 5 | **Task 5** (Documentation) | Best done last so docs can reflect all the new features and error handling. |

---

## Cross-Task Dependencies

```
Task 2 (lock) ──────────────────────────────────┐
                                                  │
Task 4 (cache HF) ──────────────────────────────┼─► Task 1 (size check)
                                                  │     (uses cached metadata)
Task 1 ─────────────────────────────────────────┼─► Task 3 (error handling)
                                                  │     (fewer OOMs to handle)
Task 3 ─────────────────────────────────────────┼─► Task 5 (docs)
                                                  │     (docs new error types)
```

---

## Summary of Code Changes

| Task | Files Changed | Lines |
|------|---------------|-------|
| 1 | `src/scripts/process_evaluation_queue.py` | ~30 |
| 2 | `src/scripts/process_evaluation_queue.py` | ~30 |
| 3 | `src/scripts/process_evaluation_queue.py`, `src/euroeval/exceptions.py` | ~50 |
| 4 | `src/scripts/process_evaluation_queue.py`, `src/euroeval/safetensors_utils.py` | ~60 |
| 5 | `src/euroeval/__init__.py`, `src/euroeval/cli.py`, `src/euroeval/benchmarker.py`, `site/` | ~100 |

Total estimated changes: ~270 lines across 7 files.
