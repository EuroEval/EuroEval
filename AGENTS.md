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
project with `vercel build --prod` and deploys it with `vercel deploy --prebuilt --prod`.

### Leaderboard Generation

Python package with config in `pyproject.toml` (same as evaluation framework) and source
code in `src/leaderboards`. Leaderboards are CSVs and the evaluation queue is handled
via Github issues.

### Scripts

There are scripts in `scripts` of various kinds, including generation of leaderboards,
building SEO files, API reference, creating individual datasets, etc.

### Tests

All evaluation framework tests are in `tests` and can be run with `make test`. This take
s a very long time though, so prefer to just running the tests in the modules you have
changed. There are no tests for the frontend or leaderboard generation.

## Formatting, linting, and type checking

All checks are run with `make check`, which runs all the pre-commit hooks. The
following tools are used:

- **Ruff** — Python formatter and linter (includes Jupyter notebook support)
- **ty** — Python type checker
- **markdownlint-cli2** — Markdown linting
- **vue-tsc** — TypeScript type checking for the frontend (`src/frontend/`)

The pre-commit hooks also include basic quality checks (end-of-file fixer,
trailing whitespace, debug statements, type annotation enforcement, and
notebook stripping).

## Changelog

`CHANGELOG.md` is the release notes for the `euroeval` Python package published to
PyPI. Only add a changelog entry when the change touches the package itself (i.e.
files under `src/euroeval/`).

Do **not** add changelog entries for changes to:

- the frontend (`src/frontend/`)
- the Vercel API endpoints (`api/`)
- scripts (`src/scripts/`)
- tests, CI, build configuration, or documentation
