# EuroEval Evaluation Improvement Plan

A detailed, actionable implementation plan for five improvements to the EuroEval
server-side evaluation pipeline and package documentation.

---

## 1. GPU Fit Check via Safetensors Headers

### 1.1 Problem

`huggingface_param_count()` already reads `safetensors.total` from the HF API, but
only uses it for queue sorting. Models that exceed available VRAM are still queued
and evaluated, wasting server time and producing OOM errors.

### 1.2 Scope

- Add two new functions to `process_evaluation_queue.py`
- Modify the candidate-building loop in `main()` to filter out models that won't fit
- Add an environment-variable configuration gate

### 1.3 Implementation Steps

#### Step 1 — Detect available VRAM

Add `get_available_gpu_vram()` near the top of the file (after imports):

```python
def get_available_gpu_vram() -> int | None:
    """Return available GPU VRAM in bytes, or None if undetectable.

    Tries three strategies in order:
      1. ``nvidia-smi --query-gpu=memory.free --format=csv,noheader``
      2. ``torch.cuda.mem_get_info()``
      3. ``nvidia-ml-py`` (pynvml)

    Returns:
        Available VRAM in bytes, or ``None`` if no GPU or detection fails.
    """
    # Strategy 1: nvidia-smi
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            free_mb = float(result.stdout.strip().split("\n")[0])
            return int(free_mb * 1024 * 1024)
    except (subprocess.SubprocessError, ValueError, IndexError):
        pass

    # Strategy 2: torch.cuda
    try:
        import torch

        if torch.cuda.is_available():
            _, total = torch.cuda.mem_get_info()
            return total
    except (ImportError, RuntimeError):
        pass

    # Strategy 3: pynvml
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.free
    except Exception:
        return None
```

#### Step 2 — Estimate model VRAM requirement

Add `estimate_model_vram()`:

```python
def estimate_model_vram(param_count: int, dtype: str = "fp16") -> int:
    """Estimate VRAM needed to run a model in VRAM (bytes).

    The estimation covers:
      - Weights (dtype-dependent bytes per parameter)
      - Optimiser / KV-cache overhead (~30 %)
      - A fixed 2 GiB safety margin for activations and buffers

    Args:
        param_count: Total parameter count (from ``safetensors.total``).
        dtype: One of ``fp16``, ``bf16``, ``fp32``, ``int8``, ``int4``.

    Returns:
        Estimated VRAM requirement in bytes.
    """
    BYTES_PER_PARAM = {
        "fp16": 2,
        "bf16": 2,
        "fp32": 4,
        "int8": 1,
        "int4": 0.5,
    }
    bytes_per_param = BYTES_PER_PARAM.get(dtype, 2)
    weights_bytes = param_count * bytes_per_param
    overhead = weights_bytes * 0.30
    safety_margin = 2 * 1024**3  # 2 GiB
    return int(weights_bytes + overhead + safety_margin)
```

#### Step 3 — Gate models behind VRAM availability

Modify the candidate-building loop inside `main()`. After retrieving
`param_count`, insert a VRAM check:

```python
        param_count = huggingface_param_count(info=info)

        # Skip models that won't fit on any GPU
        vram_available = os.getenv("EUROEVAL_MIN_VRAM")
        if vram_available:
            min_vram = int(vram_available)
            estimated_vram = estimate_model_vram(param_count=param_count)
            if estimated_vram > min_vram:
                logger.info(
                    f"#{number}: skipping {model_id!r} "
                    f"(estimated {estimated_vram / 1024**3:.1f} GiB > "
                    f"{min_vram / 1024**3:.1f} GiB threshold)."
                )
                continue
```

#### Step 4 — Document the configuration

Add to the "Required env vars" section in the module docstring:

```text
EUROEVAL_MIN_VRAM    Minimum available VRAM in bytes. Models whose estimated
                      footprint exceeds this value are skipped without changing
                      issue status. Set to 0 to disable the check.
```

### 1.4 Checklist

- [ ] Add `get_available_gpu_vram()` function
- [ ] Add `estimate_model_vram()` function
- [ ] Insert VRAM gate in `main()` candidate loop
- [ ] Update module docstring with `EUROEVAL_MIN_VRAM` env var
- [ ] Add unit tests for `estimate_model_vram()` (pure function, easy to test)
- [ ] Test on a real GPU server to confirm `nvidia-smi` parsing

---

## 2. Concurrent Execution Prevention

### 2.1 Problem

Two instances of `process_evaluation_queue.py` can run simultaneously, racing on
GitHub issue assignment and the shared JSONL results file.

### 2.2 Scope

- Add file-locking with `fcntl.flock()` on `/tmp/euroeval_queue.lock`
- Write/read PID for stale-process detection
- Register signal handlers for clean lock release

### 2.3 Implementation Steps

#### Step 1 — Lock acquisition

Add `acquire_lock()` near the top of the file:

```python
import fcntl
import signal

LOCK_PATH = Path("/tmp/euroeval_queue.lock")


def acquire_lock() -> None:
    """Acquire an exclusive non-blocking flock.

    Exits with code 1 if the lock is held by another process.
    Writes the current PID to the lock file.

    A SIGTERM / SIGINT handler is registered to release the lock on exit.
    """
    global _lock_fd

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    _lock_fd = open(LOCK_PATH, "w")

    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        pid = _lock_fd.read().strip()
        exists_msg = ""
        if pid:
            try:
                os.kill(int(pid), 0)
                exists_msg = " (still running)"
            except OSError:
                exists_msg = " (stale)"
        logger.error(
            f"Another instance is already running (PID {pid}{exists_msg}). "
            f"Lock file: {LOCK_PATH}"
        )
        _lock_fd.close()
        sys.exit(1)

    _lock_fd.seek(0)
    _lock_fd.write(str(os.getpid()))
    _lock_fd.truncate()

    def _release_on_signal(signum: int, frame: object) -> None:
        release_lock()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _release_on_signal)
    signal.signal(signal.SIGINT, _release_on_signal)

    logger.info(f"Lock acquired (PID {os.getpid()}).")


def release_lock() -> None:
    """Release the flock and remove the lock file."""
    global _lock_fd
    if _lock_fd is not None:
        try:
            fcntl.flock(_lock_fd, fcntl.LOCK_UN)
            _lock_fd.close()
            LOCK_PATH.unlink(missing_ok=True)
        except Exception:
            pass
        _lock_fd = None
```

Declare `_lock_fd: int | None = None` at module level.

#### Step 2 — Wire into `main()`

At the top of `main()`, wrap execution in a try/finally:

```python
def main() -> None:
    acquire_lock()
    try:
        # ... existing logic ...
    finally:
        release_lock()
```

### 2.4 Checklist

- [ ] Add `fcntl` and `signal` imports
- [ ] Add `acquire_lock()` and `release_lock()` functions
- [ ] Declare `_lock_fd` module-level variable
- [ ] Wrap `main()` body in try/finally
- [ ] Test: run two instances concurrently and confirm second exits
- [ ] Test: kill first instance mid-run and confirm lock is released on SIGTERM

---

## 3. Evaluation Error Handling

### 3.1 Problem

All errors are treated identically. OOM errors are not differentiated. There is no
validation that the JSONL output contains all expected results.

### 3.2 Scope

- Add `is_oom_error()` classifier
- Add `validate_jsonl_completeness()` checker
- Refactor `process_issue()` to branch on error type

### 3.3 Implementation Steps

#### Step 1 — OOM classifier

```python
OOM_PATTERNS = [
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"torch\.OutOfMemoryError", re.IGNORECASE),
    re.compile(r"MemoryError", re.IGNORECASE),
    re.compile(r"cuMemAlloc failed", re.IGNORECASE),
    re.compile(r"hipErrorOutOfMemory", re.IGNORECASE),
]


def is_oom_error(error_output: str) -> bool:
    """Return True if ``error_output`` indicates an OOM failure.

    Args:
        error_output:
            The captured stderr/stdout from the failed euroeval run.

    Returns:
        True if any OOM pattern matches.
    """
    for pattern in OOM_PATTERNS:
        if pattern.search(error_output):
            return True
    return False
```

#### Step 2 — JSONL completeness validator

```python
def validate_jsonl_completeness(
    lines: list[str], expected_languages: list[str]
) -> tuple[bool, list[str]]:
    """Check whether every expected language has at least one result line.

    Args:
        lines:
            The new JSONL lines written since the last run.
        expected_languages:
            The ISO codes the evaluation was supposed to produce.

    Returns:
        A ``(complete, missing)`` pair. ``missing`` lists language codes
        with zero results.
    """
    present: set[str] = set()
    for line in lines:
        try:
            record = json.loads(line)
            lang = record.get("language")
            if lang:
                present.add(lang)
        except json.JSONDecodeError:
            # Invalid JSON is handled separately below
            pass
    missing = [lang for lang in expected_languages if lang not in present]
    return len(missing) == 0, missing
```

#### Step 3 — Refactor `process_issue()`

Replace the existing `process_issue()` with:

```python
def process_issue(issue: dict, model_id: str, groups: list[str]) -> None:
    """Claim, evaluate, and report back on a single queue issue.

    Args:
        issue:
            The GitHub issue object returned by the API.
        model_id:
            The Hugging Face model id to evaluate.
        groups:
            The selected language-group labels for this issue.
    """
    number = issue["number"]
    languages: list[str] = []
    for g in groups:
        languages.extend(LANGUAGE_GROUP_CODES[g])
    languages = sorted(set(languages))

    logger.info(f"#{number}: claiming issue for {model_id!r}, languages={languages}")
    assign_issue(number=number)

    before = set(read_jsonl_lines(path=RESULTS_PATH))
    ok, error_output = run_euroeval(model_id=model_id, languages=languages)
    after = read_jsonl_lines(path=RESULTS_PATH)
    new_lines = [line for line in after if line not in before]

    if not ok:
        # --- euroeval subprocess failed ---
        if is_oom_error(error_output):
            # OOM: skip silently without marking errored
            logger.warning(
                f"#{number}: OOM detected for {model_id!r}, skipping "
                f"(no error comment posted)."
            )
            unassign_issue(number=number)
            return

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

    # --- euroeval subprocess succeeded ---
    if not new_lines:
        logger.warning(
            f"#{number}: no new lines produced in {RESULTS_PATH} -- leaving "
            f"issue assigned for manual inspection."
        )
        return

    # Validate that every expected language has at least one result line
    complete, missing = validate_jsonl_completeness(
        lines=new_lines, expected_languages=languages
    )
    if not complete:
        version = euroeval_version()
        comment = (
            f"Incomplete results for `{model_id}`:\n\n"
            f"Expected results for {len(languages)} language(s) but got results "
            f"for {len(languages) - len(missing)}.\n\n"
            f"Missing languages: {', '.join(missing)}\n\n"
            f"EuroEval version: v{version}\n"
        )
        comment_on_issue(number=number, comment=comment)
        set_errored_marker(number=number, body=issue.get("body"), version=version)
        unassign_issue(number=number)
        logger.warning(
            f"#{number}: incomplete results — missing langs={missing}, "
            f"marked errored."
        )
        return

    # Validate each line is valid JSON before posting
    for line in new_lines:
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            logger.error(
                f"#{number}: invalid JSON in result line: {e}"
            )
            # Post partial results with warning
            break

    payload = "\n".join(new_lines)
    comment = f"Results for `{model_id}`:\n\n```jsonl\n{payload}\n```\n"
    comment_on_issue(number=number, comment=comment)
    logger.info(f"#{number}: posted {len(new_lines)} result line(s) as comment.")
```

### 3.4 Checklist

- [ ] Add `OOM_PATTERNS` and `is_oom_error()` function
- [ ] Add `validate_jsonl_completeness()` function
- [ ] Refactor `process_issue()` with OOM skip, completeness check, JSON validation
- [ ] Add unit tests for `is_oom_error()` and `validate_jsonl_completeness()`
- [ ] Test on a real server with an OOM-triggering model

---

## 4. HF API Caching

### 4.1 Problem

`HfApi().model_info()` is called fresh every time per model per script run.
Models don't change frequently, so this wastes bandwidth and triggers rate limits.

### 4.2 Scope

- Add a file-based cache at `~/.cache/euroeval/hf_models.json`
- Persist across script invocations
- Support `--clear-cache` CLI flag (passed via subprocess from the script, or
  use a sentinel env var)
- Log cache hits/misses

### 4.3 Implementation Steps

#### Step 1 — Cache infrastructure

Add near the top of the file (after imports):

```python
import functools
import hashlib

CACHE_DIR = Path.home() / ".cache" / "euroeval"
CACHE_FILE = CACHE_DIR / "hf_models.json"
CACHE_TTL_SECONDS = 24 * 3600  # 24 hours


def _load_cache() -> dict:
    """Load the HF model cache from disk, or return an empty dict."""
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    """Persist the cache dict to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, default=str), encoding="utf-8")
```

#### Step 2 — Cached `huggingface_model_info()`

Replace the existing function:

```python
def huggingface_model_info(model_id: str) -> object | None:
    """Return HF Hub metadata for ``model_id``, or None if unavailable.

    Results are cached in ``~/.cache/euroeval/hf_models.json`` with a 24-hour
    TTL. Set ``EUROEVAL_CLEAR_HF_CACHE=1`` to force a full refresh.

    Args:
        model_id:
            The Hugging Face repo id to look up.

    Returns:
        The ``ModelInfo`` object returned by ``HfApi``, or None when the
        lookup fails for any reason.
    """
    if os.environ.get("EUROEVAL_CLEAR_HF_CACHE") == "1":
        _save_cache({})

    cache = _load_cache()
    cache_key = hashlib.sha256(model_id.encode()).hexdigest()
    now = time.time()

    # Cache hit with valid TTL
    if cache_key in cache:
        entry = cache[cache_key]
        if now - entry["timestamp"] < CACHE_TTL_SECONDS:
            logger.debug(f"HF cache HIT for {model_id!r}")
            # Restore the serialised ModelInfo
            return _restore_model_info(entry)

    # Cache miss / expired
    logger.debug(f"HF cache MISS for {model_id!r}")
    info = _fetch_hf_model_info(model_id=model_id)
    if info is not None:
        cache[cache_key] = {
            "timestamp": now,
            "model_id": model_id,
            "data": _serialise_model_info(info),
        }
        _save_cache(cache)
    return info


def _fetch_hf_model_info(model_id: str) -> object | None:
    """Actual HF API call (called only on cache miss)."""
    try:
        return HfApi().model_info(repo_id=model_id)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"HF model lookup failed for {model_id}: {e}")
        return None


def _serialise_model_info(info: object) -> dict:
    """Serialise a ModelInfo object to a JSON-safe dict."""
    result = {"repo_id": info.repo_id}
    if hasattr(info, "safetensors") and info.safetensors is not None:
        st = info.safetensors
        if hasattr(st, "total"):
            result["safetensors_total"] = st.total
    return result


def _restore_model_info(entry: dict) -> object | None:
    """Reconstruct a minimal ModelInfo-like object from cache."""
    class _FakeModelInfo:
        pass

    obj = _FakeModelInfo()
    obj.repo_id = entry.get("model_id", "")
    safetensors = None
    if "safetensors_total" in entry:
        class _SafeTensors:
            pass
        st = _SafeTensors()
        st.total = entry["safetensors_total"]
        safetensors = st
    obj.safetensors = safetensors
    return obj
```

### 4.4 Checklist

- [ ] Add `_load_cache()`, `_save_cache()` helpers
- [ ] Replace `huggingface_model_info()` with cached version
- [ ] Add `_fetch_hf_model_info()`, `_serialise_model_info()`, `_restore_model_info()`
- [ ] Document `EUROEVAL_CLEAR_HF_CACHE` env var in module docstring
- [ ] Add unit tests for cache save/load/expiration
- [ ] Verify cache persists across separate script invocations

---

## 5. Package Documentation Improvements

### 5.1 Problem

- `src/euroeval/__init__.py` has only a one-line docstring
- `src/leaderboards/__init__.py` has only a one-line docstring
- No `docs/` directory exists (mkdocs-material is in dev deps but unused)
- README links to `euroeval.com` but the site may be thin

### 5.2 Scope

- Enhance module docstrings in `__init__.py` and leaderboards `__init__.py`
- Create a `docs/` directory with mkdocs.yml and core pages
- Auto-generate API reference from docstrings

### 5.3 Implementation Steps

#### Step 1 — Enhance `src/euroeval/__init__.py`

Replace the current one-liner with a comprehensive docstring:

```python
"""EuroEval - A robust benchmarking framework for European language models.

EuroEval evaluates large language models on a comprehensive suite of tasks
covering multiple European languages, including Scandinavian, Baltic, Slavic,
Romance, Germanic, and other language families.

Quick Start
-----------
Install the package::

    pip install euroeval

Evaluate a model on all supported languages::

    euroeval --model-id facebook/opt-1.3b

Evaluate a model on specific languages::

    euroeval --model-id meta-llama/Llama-2-7b-hf \\
        --language da --language sv --language fi

Evaluation results are written to ``euroeval_benchmark_results.jsonl`` in the
current working directory.

Configuration
-------------
Set the following environment variables as needed:

- ``HUGGINGFACE_API_KEY`` — Your Hugging Face API token (required for gated models).
- ``VLLM_WORKER_MULTIPROC_METHOD`` — Multiprocessing method for vLLM (default: ``spawn``).
- ``FULL_LOG`` — Set to ``1`` to show full log output including warnings.

Architecture
------------
EuroEval consists of:

- ``Benchmarker`` — The core engine that orchestrates dataset loading, model inference,
  and metric computation.
- ``DatasetConfig`` — Dataclass describing each dataset's tasks, languages, and prompts.
- Task types: ``MULTIPLE_CHOICE``, ``TEXT_CLASSIFICATION``, ``TOKEN_CLASSIFICATION``.

For advanced usage, see the full documentation at https://euroeval.com.
"""
```

#### Step 2 — Enhance `src/leaderboards/__init__.py`

```python
"""Leaderboards for the LLM evaluation framework EuroEval.

This package provides tools for generating and rendering leaderboard tables
from EuroEval benchmark results. It supports:

- CSV/JSON leaderboard exports
- Markdown table generation
- Automatic ranking computation across language groups

Usage
-----
Generate a leaderboard from benchmark results::

    from leaderboards import generate_leaderboard
    generate_leaderboard(
        results_path="euroeval_benchmark_results.jsonl",
        output_format="markdown",
    )
"""
```

#### Step 3 — Create `docs/` directory structure

Create the following files:

```text
docs/
├── index.md                          # Home page (redirects to quick start)
├── installation.md                   # Installation guide (pip, uv, from source)
├── quick-start.md                    # Step-by-step tutorial
├── configuration-reference.md        # Env vars, CLI args, model configs
├── adding-custom-datasets.md         # How to add new datasets/tasks
├── faq.md                            # Common issues and solutions
└── api/                              # Auto-generated API reference
```

#### Step 4 — Create `mkdocs.yml`

```yaml
site_name: EuroEval
site_description: The robust European language model benchmark
site_url: https://euroeval.com
repo_url: https://github.com/EuroEval/EuroEval
repo_name: EuroEval/EuroEval

theme:
  name: material
  icon:
    repo: fontawesome/brands/github
  features:
    - navigation.instant
    - navigation.tracking
    - search.highlight

plugins:
  - search
  - autoapi:
      type: python
      python_paths:
        - src
      members: true
      special-members: __init__, __version__
      merge_into_stub: true

nav:
  - Home: index.md
  - Installation: installation.md
  - Quick Start: quick-start.md
  - Configuration Reference: configuration-reference.md
  - Adding Custom Datasets: adding-custom-datasets.md
  - API Reference: api/index.md
  - FAQ: faq.md
```

#### Step 5 — Create key documentation pages

**`docs/installation.md`:**

````markdown
# Installation

## Via pip

```bash
pip install euroeval
```

## Via uv

```bash
uv add euroeval
```

## From source

```bash
git clone https://github.com/EuroEval/EuroEval.git
cd EuroEval
pip install -e .
```

## Requirements

- Python 3.12 or higher
- A GPU with at least 8 GiB VRAM for models under 7B parameters
- Hugging Face API key for gated models
````

**`docs/quick-start.md`:**

````markdown
# Quick Start

## Install EuroEval

```bash
pip install euroeval
```

## Export your Hugging Face API key

```bash
export HUGGINGFACE_API_KEY="hf_your_token_here"
```

## Run your first evaluation

```bash
euroeval --model-id facebook/opt-1.3b
```

This evaluates OPT-1.3B on all supported European languages. Results are saved
to `euroeval_benchmark_results.jsonl`.

## Evaluate specific languages

```bash
euroeval --model-id meta-llama/Llama-2-7b-hf \\
    --language da --language sv --language fi
```

## View results

The results file contains one JSON line per language, with metrics including
accuracy, F1, and other task-specific scores.
````

#### Step 6 — Cross-link from README

Ensure the README "Installation and usage" section links prominently to the docs,
and add a "Table of Contents" link to each major section pointing to the relevant
docs page.

### 5.4 Checklist

- [ ] Rewrite `src/euroeval/__init__.py` docstring with QuickStart, Config, Architecture
- [ ] Rewrite `src/leaderboards/__init__.py` docstring
- [ ] Add module-level docstrings to `cli.py`, `benchmarker.py`, `data_models.py`
- [ ] Create `docs/mkdocs.yml` with autoapi plugin
- [ ] Create `docs/installation.md`
- [ ] Create `docs/quick-start.md`
- [ ] Create `docs/configuration-reference.md`
- [ ] Create `docs/adding-custom-datasets.md`
- [ ] Create `docs/faq.md`
- [ ] Ensure all docstrings follow Google style (ruff already enforces this)
- [ ] Verify `mkdocs serve` builds without errors
- [ ] Cross-link README.md to documentation site

---

## Priority / Effort Matrix

| Task | Priority | Effort | Complexity | Risk |
| ---- | -------- | ------ | ---------- | ---- |
| **4. HF API Caching** | High | Small | Low | Low — pure read/write, no behavioral change |
| **3. Evaluation Error Handling** | High | Medium | Medium | Medium — changes error reporting semantics |
| **2. Concurrent Execution Prevention** | High | Small | Low | Low — standard flock pattern |
| **1. GPU Fit Check** | Medium | Medium | Medium | Medium — requires GPU hardware to test |
| **5. Package Documentation** | Low | Large | Low | Low — purely additive |

## Suggested Execution Order

1. **Task 4 — HF API Caching** (Small effort, low risk, immediate bandwidth savings)
2. **Task 2 — Concurrent Execution Prevention** (Small effort, prevents data corruption)
3. **Task 3 — Evaluation Error Handling** (Medium effort, improves server efficiency by
   distinguishing OOM from real bugs)
4. **Task 1 — GPU Fit Check** (Medium effort, complements Task 3 by preventing OOMs
   before they happen)
5. **Task 5 — Package Documentation** (Large effort, low urgency — can be done in
   parallel by a different contributor)

Tasks 1–4 can all be implemented in the same file (`process_evaluation_queue.py`)
with minimal cross-dependencies. Task 5 is independent and can be started at any time.
