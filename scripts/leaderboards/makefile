# Leaderboard generation makefile.
#
# This is a slimmed-down port of the original EuroEval/leaderboards repo. It
# generates per-language leaderboard CSVs from a results.jsonl file and copies
# them into the frontend.
#
# The original makefile included `download` steps that SCP'd results from
# remote benchmark servers; those are intentionally omitted here. Drop your
# results in place at `new_results.jsonl` (or merge into `results.tar.gz`) and
# run `make generate`.

export PATH := ${HOME}/.local/bin:${HOME}/.cargo/bin:$(PATH)

CSV_OUT := ../../src/frontend/csv

generate:
	@uv run src/scripts/generate_leaderboards.py

force-generate:
	@uv run src/scripts/generate_leaderboards.py --force

# Copy generated CSVs into the frontend bundle.
sync-frontend:
	@mkdir -p $(CSV_OUT)
	@cp leaderboards/*.csv $(CSV_OUT)/
	@echo "Synced CSVs to $(CSV_OUT)/"

# End-to-end: generate + copy.
update: generate sync-frontend

install:
	@uv python install 3.12
	@uv sync --all-extras --all-groups --python 3.12
