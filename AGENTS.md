# AI Agent Guidelines for EuroEval

## Project Structure Quick Reference

Key directories and their purposes:

- `src/euroeval/` — The core Python benchmarking library (the main package)
- `src/scripts/` — Standalone scripts (queue processing, leaderboard generation)
- `src/frontend/` — Vue.js frontend source code
- `src/leaderboards/` — Leaderboard generation Python module
- `api/` — Vercel serverless API functions
- `tests/` — Test suite

## Key Files

### Server-Side Evaluation

- **`src/scripts/process_evaluation_queue.py`** — GPU compute server script
  that picks up GitHub issues, runs `euroeval`, and posts results
- **`src/scripts/collect_evaluation_results.py`** — Laptop script that
  collects results from GitHub comments and updates leaderboards

### Frontend

- **`src/frontend/services/github.ts`** — GitHub issue queue management
- **`src/frontend/services/huggingface.ts`** — HF model search
- **`src/frontend/components/EvaluationQueueView.vue`** — Main queue page
- **`src/frontend/components/ModelSubmitForm.vue`** — Model submission form

### Documentation

- **`src/frontend/md/python-package.md`** — Python package documentation (main docs)
- **`README.md`** — Main project overview

## Exploration Guidelines

When exploring this codebase:

1. **Be targeted.** EuroEval has a well-organized structure. If you need
  to understand the evaluation queue, start with
  `src/scripts/process_evaluation_queue.py`. Don't explore the entire
  frontend when you only need the server script.

2. **Know the entry points.** The server script is the compute server's
  entry point. The frontend is Vue.js with Vite. The API is Vercel
  serverless functions.

3. **Read specific functions.** When looking for how something works,
  read the specific function, not the entire file. For example, if you
  need `huggingface_model_info()`, read just that function.

## Planning

Before starting work on multi-step tasks, take time to plan. Break down the work into:

- Steps that require understanding existing code
- Steps that involve editing files
- Independent steps that can be done in parallel

## File Reading Best Practices

- **Don't read the entire `process_evaluation_queue.py`** — read only
  the functions you need (e.g., `huggingface_model_info()`,
  `run_euroeval()`, `main()`)
- **Don't explore the entire frontend** — if you need the GitHub
  service, read `src/frontend/services/github.ts` only
- **Read documentation files once** — `src/frontend/md/python-package.md`
  is the main docs file; read it once and note the sections you need

## EuroEval-Specific Patterns

### Queue Flow

1. Frontend → `api/submit-evaluation.ts` → GitHub issue created
2. Compute server → `process_evaluation_queue.py` → picks up unassigned issues
3. Server → runs `euroeval` → posts `jsonl` results as GitHub comment
4. Laptop → `collect_evaluation_results.py` → collects results → updates leaderboards

### Key Functions to Know

- `huggingface_model_info()` — HF model lookup (line ~220 in process_evaluation_queue.py)
- `run_euroeval()` — Runs the euroeval CLI (line ~357)
- `process_issue()` — Main issue processing logic (line ~259)
- `main()` — Entry point for queue processing (line ~80)

### Documentation Structure

The documentation uses MkDocs with markdown files in `src/frontend/md/`.
The main package docs are in `python-package.md`. New sections should use
`##` headings for automatic sidebar navigation.
