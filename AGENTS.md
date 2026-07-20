# EuroEval

This is a repository for the language model evaluation framework EuroEval, focused on
robust evaluation of language models on European tasks. Can evaluate encoders, base
decoders and instruction tuned decoders. The repo contains both the evaluation
framework, frontend, and leaderboard generation.

## Project Overview

### Evaluation Framework

Python package with config in `pyproject.toml` and source code in `src/euroeval`.

### Frontend

Vue.js + Typescript + Vite frontend with configs in the repo root (e.g., `package.json`,
`vite.config.ts`, `tsconfig.json`) and source code in `src/frontend`. Deployment is
handled with Vercel.

To build and deploy the frontend to production, run `make frontend`. This builds the
project with `vercel build --prod` and deploys it with
`vercel deploy --prebuilt --prod`.

### Leaderboard Generation

Python package with config in `pyproject.toml` (same as evaluation framework) and source
code in `src/leaderboards`. Leaderboards are CSVs and the evaluation queue is handled
via Github issues.

### Data Flow and Storage

Evaluation results flow through three storage layers:

1. **Hugging Face Bucket** (source of truth):
   - `EuroEval/results`: Tree layout with one JSON file per logical result:
     `results/<sanitise(model_id)>/<dataset>__<split_label>__<shot_label>.json`
   - Identity/filename keyed on `(model_id, dataset, validation_split, few_shot)` —
     distinct results never share a path, so sync is a union (no per-file clobbering)
   - All files contain metadata fields (`commercially_licensed`, `open`,
     `trained_from_scratch`)
   - Synced via `hf sync` command (requires `HF_TOKEN`)

2. **Local `results/` directory** (working copy):
   - Contains the same per-record JSON tree as the HF bucket
   - Git-ignored, synced bidirectionally with HF bucket

3. **`results.tar.gz`** (historical archive):
   - **Location:** `~/pCloud Drive/data/euroeval_backup/results.tar.gz`
   - **Override:** Set `EUROEVAL_RESULTS_BACKUP_DIR` environment variable
   - Contains the `results/` tree (all per-record JSON files)
   - Rebuilt from `results/**/*.json` files on each run
   - 43+ MB, not tracked in git

4. **Backup snapshots** (disaster recovery):
   - **Directory:** `~/pCloud Drive/data/euroeval_backup/`
   - Contains timestamped snapshots: `results_YYYYMMDD_HHMMSS.tar.gz`
   - Rotation: max 10 snapshots or 1 GB total
   - Always preserves at least one backup from a previous day (if any exist)

**Sync direction:** HF bucket → local `results/` → `results.tar.gz` → backup snapshots

The `src/scripts/collect_evaluation_results.py` script orchestrates this flow:

- Downloads new results from HF bucket
- Deduplicates against existing results in `results.tar.gz`
- Rebuilds `results.tar.gz` from the merged results
- Creates timestamped backup before overwrite

### Scripts

There are scripts in `scripts` of various kinds, including generation of leaderboards,
building SEO files, API reference, creating individual datasets, etc. All scripts in
this directory are _persistent_ scripts. One-off scripts don't belong in the repository

- if you need to run a one-off script, store it in /tmp or in-memory.

- **Dataset creation scripts** — Preserve upstream split boundaries when the source
  dataset already has splits. Do not concatenate train/dev/validation/test and
  regenerate new splits unless there is no source split information. Never let source
  training samples end up in validation or test splits. Cap every dataset by capping
  each source split independently at 1,024 train, 256 validation, and 2,048 test
  samples. Add a `-mini` suffix to the Hugging Face repo ID and `DatasetConfig.source`
  only when these caps truncate at least one split.
- **`create_language_spider_plot.py`** — Generate Plotly spider/radial plots comparing
  models across languages. Output is a PNG file (`language-spider-plot.png`). Use when
  asked to create a radial/spider plot for language performance comparison.
  `uv run src/scripts/create_language_spider_plot.py -m MODEL [-l LANGUAGE]`

### Tests

All evaluation framework tests are in `tests` and can be run with `make test`. This
takes a very long time though, so prefer to just running the tests in the modules you
have changed. There are no tests for the frontend or leaderboard generation.

## Formatting, linting, and type checking

All checks are run with `make check`, which runs all the pre-commit hooks. The following
tools are used:

- **Ruff** — Python formatter and linter (includes Jupyter notebook support)
- **ty** — Python type checker
- **markdownlint-cli2** — Markdown linting
- **vue-tsc** — TypeScript type checking for the frontend (`src/frontend/`)

The pre-commit hooks also include basic quality checks (end-of-file fixer, trailing
whitespace, debug statements, type annotation enforcement, and notebook stripping).

## Changelog

`CHANGELOG.md` is the release notes for the `euroeval` Python package published to PyPI.
Only add a changelog entry when the change touches the package itself (i.e. files under
`src/euroeval/`).

Do **not** add changelog entries for changes to:

- the frontend (`src/frontend/`)
- the Vercel API endpoints (`api/`)
- scripts (`src/scripts/`)
- tests, CI, build configuration, or documentation
