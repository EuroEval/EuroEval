# Leaderboards Generation

Generates the leaderboard CSVs consumed by the EuroEval frontend
(`src/frontend/csv/`). Ported from the standalone
[EuroEval/leaderboards](https://github.com/EuroEval/leaderboards) repo.

## Setup

```sh
cd scripts/leaderboards
make install        # installs uv + Python deps
```

## Generating leaderboards

1. Place benchmark results at `new_results.jsonl` (or extend the existing
   `results.tar.gz`). The original repo had a `make download` target that
   SCP'd these from internal benchmark servers; that step is intentionally
   left to operators.
2. Run:

   ```sh
   make update        # generate + copy to ../../src/frontend/csv/
   # or:
   make generate      # only regenerates leaderboards/*.csv
   make sync-frontend # only copies them into the frontend
   ```

## Layout

- `src/leaderboards/` — Python library (result loading, score computation,
  leaderboard generation).
- `src/scripts/generate_leaderboards.py` — CLI entry point.
- `leaderboard_configs/*.yaml` — per-language task → dataset mapping.
- `task_config.yaml` — task → primary/secondary metric mapping.
- `leaderboards/` — generated CSVs (per language + multi-language groupings).
