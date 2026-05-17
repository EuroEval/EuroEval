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

### Leaderboard Generation

Python package with config in `pyproject.toml` (same as evaluation framework) and source
code in `src/leaderboards`. Leaderboards are CSVs and the evaluation queue is handled
via Github issues.

### Scripts

There are scripts in `scripts` of various kinds, including generation of leaderboards,
building SEO files, API reference, creating individual datasets, etc.

### Tests

All evaluation framework tests are in `tests` and can be run with `make test`. There are
no tests for the frontend or leaderboard generation.

## Formatting, linting, type checking

All checks be run with `make check`, which runs all the pre-commit hooks. This includes
the Ruff formatter and linter, pyrefly type checker, and markdown and notebook linting.

## Changelog

If any changes are made to the evaluation framework, the changelog should be updated in
`CHANGELOG.md`.
