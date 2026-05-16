# EuroEval Implementation Plan

This document provides a detailed, actionable implementation plan for five improvements
to the EuroEval project. Each task includes prerequisites, file references, code-level
steps, and testing considerations.

---

## Task 1: GPU Size Check via Safetensors Headers

### Goal

Skip models that won't fit on the server's GPU during the candidate filtering phase,
before `assign_issue()` is called. The skip must be silent — no issue status changes,
no errored marker, no re-queue.

### Files Involved

| Path | Role |
|---|---|
| `src/scripts/process_evaluation_queue.py` | Main script to modify |
| `torch` (runtime) | GPU memory query |
| `huggingface_hub` (runtime) | `get_safetensors_metadata()` |

### Current State

- `huggingface_param_count()` (line 238) reads `info.safetensors.total` from the
  `ModelInfo` object. This is the total safetensors file size in **bytes**, not the
  model parameter count. It's only used for queue sorting.
- The candidate filtering loop runs at lines 119–128.
- `assign_issue()` is called inside `process_issue()` at line 277.

### Implementation Steps

#### Step 1.1: Add a new import at the top of the file

After the existing imports (around line 37), add:

```python
from huggingface_hub import get_safetensors_metadata
```

#### Step 1.2: Implement `estimate_model_size_bytes()`

Add a new function after `huggingface_param_count()` (around line 257):

```python
def estimate_model_size_bytes(model_id: str) -> int | None:
    """Estimate the model's GPU memory requirement in bytes.

    Reads safetensors headers via ``get_safetensors_metadata()`` and sums the
    byte size of every tensor, accounting for dtype:
    - float16 / bfloat16 → 2 bytes per param
    - float32            → 4 bytes per param

    Args:
        model_id: The Hugging Face model id to look up.

    Returns:
        Total model size in bytes, or None if the lookup fails.
    """
    try:
        metadata = get_safetensors_metadata(model_id)
        # ``metadata`` is a dict with a ``total_size`` key (int, bytes).
        # This is the sum of all tensor sizes from safetensors headers.
        return metadata["total_size"]  # type: ignore[union-attr]
    except Exception as e:  # noqa: BLE001
        logger.warning(f"GPU size check failed for {model_id}: {e}")
        return None
```

**Note:** `get_safetensors_metadata()` already returns `total_size` in bytes, which
is the sum of `dtype × num_elements` for every tensor in the safetensors files. This
is exactly the GPU memory needed to load the model weights (in float32 for inference,
but we use the actual dtype sizes from the headers).

#### Step 1.3: Implement `model_fits_on_gpu()`

Add after `estimate_model_size_bytes()`:

```python
def model_fits_on_gpu(model_size_bytes: int) -> bool:
    """Check whether ``model_size_bytes`` fits on the current GPU.

    Compares the model size against available GPU memory.

    Args:
        model_size_bytes: Estimated model weight size in bytes.

    Returns:
        True if the model fits, False otherwise.
    """
    try:
        # Get available memory in bytes
        available_bytes, _ = torch.cuda.mem_get_info(0)
        # Add 20% headroom to avoid edge-case OOM during attention weights
        required_bytes = int(model_size_bytes * 1.2)
        return required_bytes <= available_bytes
    except Exception as e:  # noqa: BLE001
        logger.warning(f"GPU memory check failed: {e}")
        return True  # Assume it fits if we can't check (safe default)
```

#### Step 1.4: Add the GPU check to the candidate filtering loop

In the `main()` function, inside the candidate filtering loop (around line 119),
add the GPU check after the HF model info lookup:

```python
        info = huggingface_model_info(model_id=model_id)
        if info is None:
            logger.info(f"#{number}: skipping -- model {model_id!r} not on HF Hub.")
            continue

        # GPU size check — skip silently if model won't fit
        model_size = estimate_model_size_bytes(model_id=model_id)
        if model_size is not None and not model_fits_on_gpu(model_size_bytes=model_size):
            logger.info(
                f"#{number}: skipping -- model {model_id!r} estimated at "
                f"{model_size / 1e9:.2f} GB, exceeds available GPU memory."
            )
            continue
```

This placement ensures the check happens **after** verifying the model exists on HF
but **before** `assign_issue()` is called (which happens inside `process_issue()`).

#### Step 1.5: Add `import torch` at the top

Add `import torch` alongside the other imports (after line 31) to enable GPU memory
queries.

### Testing Considerations

- Unit test `estimate_model_size_bytes()` with a known small model (e.g.,
  `bart/tiny-random`) and verify the returned bytes match expected tensor sizes.
- Unit test `model_fits_on_gpu()` by mocking `torch.cuda.mem_get_info()` to return
  known values.
- Integration test: run the script against a model that is known to exceed GPU memory
  and verify it is skipped without marking the issue as errored.

---

## Task 2: Process Lock (Prevent Concurrent Runs)

### Goal

Prevent the script from running concurrently on the server. Use a lock file approach
with PID tracking.

### Files Involved

| Path | Role |
|---|---|
| `src/scripts/process_evaluation_queue.py` | Main script to modify |

### Implementation Steps

#### Step 2.1: Define lock file path

Add a constant near the top of the file, after the existing constants (around line 50):

```python
LOCK_FILE = Path(os.path.dirname(__file__)) / ".process_evaluation_queue.lock"
```

#### Step 2.2: Implement `acquire_lock()` and `release_lock()`

Add these functions after `_token()`:

```python
def acquire_lock() -> int | None:
    """Acquire an exclusive lock file.

    Returns:
        The current PID if the lock was acquired, or the PID of the
        existing process if the lock is held.

    Raises:
        SystemExit: If the lock file exists and is older than 1 hour
            (stale lock), log a warning and proceed.
    """
    pid = os.getpid()

    if LOCK_FILE.exists():
        try:
            existing_pid = int(LOCK_FILE.read_text().strip())
        except ValueError:
            logger.warning("Lock file contains invalid PID, treating as stale.")
            existing_pid = None

        if existing_pid is not None:
            # Check if the existing process is still running
            try:
                os.kill(existing_pid, 0)
            except OSError:
                # Process is dead — stale lock, proceed
                logger.warning(
                    f"Lock file references dead process (PID {existing_pid}). "
                    "Treating as stale and proceeding."
                )
            else:
                logger.error(
                    f"Another process (PID {existing_pid}) is already running. "
                    f"Lock file: {LOCK_FILE}"
                )
                sys.exit(1)

    # Acquire the lock
    LOCK_FILE.write_text(str(pid))
    return pid


def release_lock() -> None:
    """Release the lock file.

    Removes the lock file if it matches the current PID.
    """
    try:
        if LOCK_FILE.exists() and LOCK_FILE.read_text().strip() == str(os.getpid()):
            LOCK_FILE.unlink()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to release lock file: {e}")
```

#### Step 2.3: Integrate lock acquisition into `main()`

At the start of `main()`, add:

```python
def main() -> None:
    """Process every unassigned model-evaluation-request issue once.

    Acquires an exclusive lock to prevent concurrent runs.
    """
    try:
        acquire_lock()
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to acquire lock: {e}")
        sys.exit(1)

    # ... existing code ...

    for is_errored, param_count, num_groups, issue, model_id, groups in candidates:
        # ... existing code ...
        time.sleep(1)
    finally:
        release_lock()
```

The `finally` block ensures the lock is released even if an exception occurs.

### Testing Considerations

- Unit test `acquire_lock()` by creating a lock file with a valid PID and verifying
  the function raises `SystemExit`.
- Unit test `release_lock()` by verifying the lock file is removed.
- Integration test: run the script twice simultaneously and verify the second instance
  exits with an error message including the PID of the first process.

---

## Task 3: Proper Evaluation Error Handling

### Goal

Distinguish between OOM errors (skip silently) and other errors (mark as errored).
Handle incomplete results by checking stderr for OOM patterns.

### Files Involved

| Path | Role |
|---|---|
| `src/scripts/process_evaluation_queue.py` | Main script to modify |

### Current State

- `run_euroeval()` (line 357) returns `(ok: bool, error_output: str)`.
- `process_issue()` (line 259) checks `if not ok:` and marks the issue as errored.
- There is no distinction between OOM and other errors.
- There is no check for incomplete results (missing language groups).

### Implementation Steps

#### Step 3.1: Add OOM detection function

Add after `run_euroeval()`:

```python
OOM_PATTERN = re.compile(r"CUDA out of memory|OutOfMemoryError|torch\.OutOfMemory")


def is_oom_error(error_output: str) -> bool:
    """Return True if ``error_output`` indicates an OOM failure.

    Checks stderr for CUDA OOM patterns.

    Args:
        error_output: The captured stderr/stdout from ``run_euroeval()``.

    Returns:
        True if the error was caused by OOM, False otherwise.
    """
    return bool(OOM_PATTERN.search(error_output))
```

#### Step 3.2: Implement incomplete results detection

Add after `is_oom_error()`:

```python
def get_completed_languages(results_path: Path, languages: list[str]) -> set[str]:
    """Return the set of languages that have results in the JSONL file.

    Checks ``euroeval_benchmark_results.jsonl`` for entries matching the
    given model ID. A language is considered "completed" if there is at
    least one result line for that language.

    Args:
        results_path: Path to the JSONL file.
        languages: The list of requested language codes.

    Returns:
        The set of languages that have results.
    """
    lines = read_jsonl_lines(path=results_path)
    completed: set[str] = set()
    for line in lines:
        try:
            entry = json.loads(line)
            # EEE format: eval_library.additional_details.languages contains
            # the language codes for this entry
            details = entry.get("eval_library", {}).get("additional_details", {})
            if isinstance(details, dict):
                langs_raw = details.get("languages", "")
                if isinstance(langs_raw, str):
                    # Parse the JSON-encoded list string
                    langs_list = json.loads(langs_raw)
                    if isinstance(langs_list, list):
                        completed.update(langs_list)
        except Exception:  # noqa: BLE001
            continue
    return completed
```

#### Step 3.3: Modify `process_issue()` to handle OOM vs. other errors

Replace the error handling block in `process_issue()` (around lines 284–295) with:

```python
    if not ok:
        # Check if this was an OOM error
        if is_oom_error(error_output):
            logger.info(
                f"#{number}: skipping -- OOM error for {model_id!r}. "
                "Returning to queue without marking errored."
            )
            unassign_issue(number=number)
            return

        # Non-OOM error: mark as errored
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

#### Step 3.4: Handle incomplete results

After the `ok` check but before the results posting (around line 297), add:

```python
    # Check for incomplete results
    completed_languages = get_completed_languages(RESULTS_PATH, languages)
    missing_languages = set(languages) - completed_languages

    if missing_languages:
        # Results are incomplete — check if it was OOM
        # We need to re-run with capture to check stderr
        # For now, treat incomplete results as non-OOM errors
        version = euroeval_version()
        logger.warning(
            f"#{number}: incomplete results for {model_id!r} — "
            f"missing languages: {missing_languages}. "
            "Returning to queue for retry."
        )
        # Don't mark errored; just return to queue
        unassign_issue(number=number)
        return
```

**Alternative approach** (preferred): Instead of re-running, we can check if the
`run_euroeval()` stderr already contains OOM patterns. Since `run_euroeval()` captures
stderr on failure, we can pass the `error_output` to `is_oom_error()` even when `ok=True`
but results are incomplete.

Refined implementation:

```python
    # Check for incomplete results
    completed_languages = get_completed_languages(RESULTS_PATH, languages)
    missing_languages = set(languages) - completed_languages

    if missing_languages:
        # Results are incomplete — check if it was OOM
        if is_oom_error(error_output):
            logger.info(
                f"#{number}: incomplete results for {model_id!r} — "
                f"missing languages: {missing_languages}. OOM detected, "
                "returning to queue."
            )
            unassign_issue(number=number)
            return
        else:
            # Non-OOM incomplete results — mark errored
            version = euroeval_version()
            comment = (
                f"Incomplete results for {model_id!r}:\n\n"
                f"Missing languages: {', '.join(sorted(missing_languages))}.\n\n"
                f"Error output:\n\n"
                f"```bash\n{error_output[-2000:].strip() or '(no output)'}\n```\n\n"
                f"EuroEval version: v{version}\n"
            )
            comment_on_issue(number=number, comment=comment)
            set_errored_marker(number=number, body=issue.get("body"), version=version)
            unassign_issue(number=number)
            logger.info(f"#{number}: marked incomplete on v{version}.")
            return
```

### Testing Considerations

- Unit test `is_oom_error()` with various OOM messages and non-OOM messages.
- Unit test `get_completed_languages()` with a mock JSONL file containing entries
  for some languages.
- Integration test: run the script against a model that causes OOM and verify the
  issue is returned to the queue without an errored marker.
- Integration test: run the script against a model with a non-OOM error and verify
  the issue is marked as errored.

---

## Task 4: HF API Response Caching

### Goal

Cache Hugging Face API responses to avoid rate limiting. Cache should persist across
script runs using a JSON file.

### Files Involved

| Path | Role |
|---|---|
| `src/scripts/process_evaluation_queue.py` | Python script cache |
| `src/frontend/services/huggingface.ts` | Frontend service cache |
| `api/submit-evaluation.ts` | API proxy cache |

### Implementation Steps

#### Step 4.1: Python cache implementation

Add to `process_evaluation_queue.py`:

```python
import json
import time
from pathlib import Path

CACHE_FILE = Path(__file__).parent / ".hf_cache.json"
CACHE_TTL_SECONDS = 3600  # 1 hour


def _load_cache() -> dict:
    """Load the cache from the JSON file."""
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    """Save the cache to the JSON file."""
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def cached_huggingface_model_info(model_id: str) -> object | None:
    """Return HF Hub metadata, using an in-memory + file cache.

    Args:
        model_id: The Hugging Face repo id to look up.

    Returns:
        The ``ModelInfo`` object, or None when unavailable.
    """
    cache = _load_cache()
    if model_id in cache:
        entry = cache[model_id]
        if time.time() - entry.get("_time", 0) < CACHE_TTL_SECONDS:
            # Cache hit — convert dict back to a simple object
            info = type("ModelInfo", (), {})()
            for key, value in entry.items():
                if key.startswith("_"):
                    continue
                setattr(info, key, value)
            return info

    # Cache miss — fetch from HF
    info = huggingface_model_info(model_id=model_id)
    if info is not None:
        # Serialize the ModelInfo to a dict
        cache_entry = {}
        for attr in dir(info):
            if not attr.startswith("_"):
                try:
                    cache_entry[attr] = getattr(info, attr)
                except Exception:
                    pass
        cache_entry["_time"] = time.time()
        cache[model_id] = cache_entry
        _save_cache(cache)

    return info
```

**Replace** `huggingface_model_info()` calls in `main()` with `cached_huggingface_model_info()`.

#### Step 4.2: Safetensors metadata cache

Add a similar cached version:

```python
def cached_get_safetensors_metadata(model_id: str) -> dict | None:
    """Return safetensors metadata, using a file cache.

    Args:
        model_id: The Hugging Face model id.

    Returns:
        The metadata dict with ``total_size``, or None if unavailable.
    """
    cache = _load_cache()
    cache_key = f"__safetensors__{model_id}"
    if cache_key in cache:
        entry = cache[cache_key]
        if time.time() - entry.get("_time", 0) < CACHE_TTL_SECONDS:
            return entry

    try:
        metadata = get_safetensors_metadata(model_id)
        cache[cache_key] = {"total_size": metadata["total_size"], "_time": time.time()}
        _save_cache(cache)
        return metadata
    except Exception:  # noqa: BLE001
        return None
```

#### Step 4.3: Frontend cache (`src/frontend/services/huggingface.ts`)

Add a simple in-memory cache with TTL:

```typescript
const hfCache = new Map<string, { data: HfModelSuggestion[]; timestamp: number }>();
const HF_CACHE_TTL = 60_000; // 1 minute

export async function searchHfModels(query: string, limit = 20): Promise<HfModelSuggestion[]> {
  const q = query.trim();
  if (!q) return [];

  // Check cache
  const cacheKey = `${q}-${limit}`;
  const cached = hfCache.get(cacheKey);
  if (cached && Date.now() - cached.timestamp < HF_CACHE_TTL) {
    return cached.data;
  }

  activeController?.abort();
  const controller = new AbortController();
  activeController = controller;

  const url =
    `https://huggingface.co/api/models?search=${encodeURIComponent(q)}` +
    `&limit=${limit}&sort=downloads&direction=-1`;

  try {
    const r = await fetch(url, { signal: controller.signal });
    if (!r.ok) return [];
    const data = (await r.json()) as Array<{ id?: string; modelId?: string; downloads?: number }>;
    const result = data
      .map((m) => ({ id: m.id ?? m.modelId ?? "", downloads: m.downloads }))
      .filter((m) => m.id);

    // Cache the result
    hfCache.set(cacheKey, { data: result, timestamp: Date.now() });
    return result;
  } catch (e) {
    if ((e as Error).name === "AbortError") return [];
    return [];
  }
}
```

#### Step 4.4: API proxy cache (`api/submit-evaluation.ts`)

The API proxy runs on Vercel edge functions, which don't persist state between invocations.
For this case, use a Redis-based cache (which is already available via `@upstash/redis`):

```typescript
// Add after the existing imports
const hfCache = new Redis.fromEnv();

// In huggingFaceModelExists, add caching:
async function huggingFaceModelExists(modelId: string): Promise<boolean> {
  // Check cache
  const cached = await hfCache.get(`hf:model:${modelId}`);
  if (cached !== null) {
    return cached === "true";
  }

  const r = await fetch(
    `https://huggingface.co/api/models/${encodeURIComponent(modelId)}`,
    { method: "GET" },
  );
  if (r.status === 200) {
    const data = (await r.json()) as { private?: boolean; gated?: boolean | string };
    const exists = !data.private && !(data.gated && data.gated !== false);
    // Cache for 5 minutes
    await hfCache.set(`hf:model:${modelId}`, String(exists), { expiry: "5min" });
    return exists;
  }
  // Cache negative result for 1 minute
  await hfCache.set(`hf:model:${modelId}`, "false", { expiry: "1min" });
  return false;
}
```

### Testing Considerations

- Unit test `_load_cache()` and `_save_cache()` with mock files.
- Unit test `cached_huggingface_model_info()` to verify cache hits and misses.
- Integration test: run the script against multiple models and verify the cache file
  is created and populated.
- Frontend test: verify the cache reduces API calls in the model search dropdown.

---

## Task 5: Improve Python Package Documentation

### Goal

Improve the documentation at `src/frontend/md/python-package.md` with better structure,
quick reference, and troubleshooting sections.

### Files Involved

| Path | Role |
|---|---|
| `src/frontend/md/python-package.md` | Documentation to improve |
| `src/frontend/config.yaml` | MkDocs navigation config |

### Implementation Steps

#### Step 5.1: Add a "Getting Started" section

Add a new section immediately after the package overview (after line 9):

```markdown
## Getting Started

The absolute minimum to run a benchmark:

1. Install the package:
   ```bash
   pip install euroeval[all]
   ```

1. Run a benchmark:

   ```bash
   euroeval --model <model-id>
   ```

That's it. The benchmark will run on all tasks and all European languages by default.

### Quick Reference

| Argument | Description | Example |
|---|---|---|
| `--model` | Hugging Face model ID | `--model meta-llama/Llama-3.1-8B-Instruct` |
| `--task` | Task to benchmark | `--task sentiment-classification` |
| `--language` | Language ISO code | `--language da --language en` |
| `--api-base` | Custom inference API URL | `--api-base http://localhost:8000` |
| `--api-key` | API key for authentication | `--api-key my-secret-key` |
| `--download-only` | Download data without running | `--download-only` |
| `--debug` | Output model responses to file | `--debug` |
| `--max-context-length` | Override max context length | `--max-context-length 4096` |
| `--vocabulary-size` | Override vocabulary size | `--vocabulary-size 32000` |
| `--dataset` | Custom dataset ID or path | `--dataset EuroEval/test_dataset` |
| `--trust-remote-code` | Allow remote dataset code | `--trust-remote-code` |

See `euroeval --help` for the full list.

```

#### Step 5.2: Add "Understanding Results" section

Add after the "Output format: Every Eval Ever (EEE)" section (around line 1068):

```markdown
## Understanding Results

### EEE Format Quick Guide

Each line in `euroeval_benchmark_results.jsonl` is a JSON object. The most important
fields for understanding results are:

| Field | What it tells you |
|---|---|
| `evaluation_results[N].score_details.score` | The final score for this metric (0-100 scale) |
| `evaluation_results[N].score_details.details.num_failed_instances` | How many samples failed silently |
| `eval_library.additional_details.raw_results` | Per-iteration scores and failure details |

### Interpreting Scores

- **Classification tasks** (MCC, accuracy, F1): Higher is better. Values range from 0 to 100.
- **Text generation tasks** (ROUGE, BLEU, BERTScore): Higher is better. Values range from 0 to 100.
- **Failed instances**: A non-zero `num_failed_instances` means the model's output
  couldn't be parsed or matched. This is common for generative models on structured tasks.

### Analyzing Failed Instances

When evaluating generative models, some samples may fail silently. To find out why:

```python
import json
from euroeval.data_models import BenchmarkResult

with open("euroeval_benchmark_results.jsonl") as f:
    for line in f:
        if line.strip():
            result = BenchmarkResult.from_dict(json.loads(line))
            raw = result.results.get("raw", [])
            for iteration in raw:
                for failure in iteration.get("failed_instances", []):
                    print(f"Sample {failure['sample_index']}: {failure['error']}")
```

Common failure reasons:

- `"Could not parse JSON from model output"` — The model didn't produce valid JSON.
  Increase `--max-generated-tokens` or use a better prompt.
- `"No candidate label found in model output"` — The model's output didn't match
  any expected label. Check the model's temperature settings.

```

#### Step 5.3: Add Troubleshooting section

Add at the end of the file (after line 1289):

```markdown
## Troubleshooting

### CUDA Out of Memory

**Symptom**: The benchmark crashes with `CUDA out of memory`.

**Solutions**:
1. Use a smaller model.
2. Reduce `--max-generated-tokens`.
3. Use `--device cpu` to run on CPU (slower but avoids OOM).
4. If using vLLM, reduce `--num-gpus` or use quantization.

### Model Not Found

**Symptom**: `huggingface_model_info()` returns None.

**Solutions**:
1. Verify the model ID on [Hugging Face Hub](https://huggingface.co/models).
2. Check for typos in the model ID.
3. The model must be public (not gated or private).

### Missing Language Results

**Symptom**: Some languages are missing from the results.

**Solutions**:
1. Check if the model supports all requested languages.
2. Check stderr for OOM errors — the script will skip OOM'd models silently.
3. Retry the evaluation; the issue will be re-queued.

### Slow Evaluation

**Symptom**: The benchmark takes a very long time.

**Solutions**:
1. Use `--task` to evaluate only specific tasks.
2. Use `--language` to evaluate only specific languages.
3. Use a smaller model or reduce few-shot examples.
4. Consider using `--download-only` to pre-download data.

### Dataset Loading Errors

**Symptom**: `DatasetNotFoundError` or `ValueError` during dataset loading.

**Solutions**:
1. Verify the dataset ID on the Hugging Face Hub.
2. For custom datasets, check the column names match EuroEval expectations.
3. Use `--trust-remote-code` for Hub-hosted datasets with Python configs.
```

#### Step 5.4: Update navigation in `config.yaml`

Add the new sections to the navigation structure. In `src/frontend/config.yaml`,
the `python-package` section already has `toc: true` (line 127), so the new headings
will automatically appear in the table of contents.

No changes needed to `config.yaml` since the new sections use standard Markdown
headings (`##`) which mkdocs-material will automatically add to the sidebar nav.

### Testing Considerations

- Verify the documentation renders correctly with `make docs`.
- Check that the quick reference table is readable and complete.
- Verify the troubleshooting section covers all common error scenarios.
- Ensure the sidebar navigation includes all new sections.

---

## Implementation Order

The tasks are independent and can be implemented in any order. Recommended order:

1. **Task 5** (Documentation) — Lowest risk, no code changes needed.
2. **Task 4** (Caching) — Low risk, defensive change.
3. **Task 2** (Process Lock) — Low risk, defensive change.
4. **Task 3** (Error Handling) — Medium risk, affects issue status logic.
5. **Task 1** (GPU Size Check) — Medium risk, adds new dependency (torch).

## Risk Assessment

| Task | Risk Level | Reason |
|---|---|---|
| Task 1 | Medium | Adds torch dependency, new model loading logic |
| Task 2 | Low | Purely defensive, no behavioral changes |
| Task 3 | Medium | Changes issue status logic, affects queue behavior |
| Task 4 | Low | Defensive caching, no behavioral changes |
| Task 5 | Low | Documentation only |

## Dependencies

| Task | New Dependencies |
|---|---|
| Task 1 | `torch` (already a dependency) |
| Task 2 | None |
| Task 3 | None |
| Task 4 | None (uses stdlib `json` + `time`) |
| Task 5 | None |

All tasks use existing dependencies. No new packages need to be installed.
