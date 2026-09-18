# Private canary tools

These scripts generate and validate the private contamination-canary protocol. They
are research and protocol-generation tools, not the evidence collector, production
checker, benchmark, or leaderboard pipeline. They do not publish results. The separate
experimental evidence pipeline is documented in
[`docs/contamination-canary.md`](../../../docs/contamination-canary.md).

## Scripts

- `generate_private_canary.py` validates a source JSONL corpus and creates the
  augmented corpus plus plaintext private records.
- `run_private_canary_exposure.py` trains and scores the nested local exposure arms
  (0, 1, 2, 4, and 8 exposures), with resumable output and fingerprints.
- `analyse_private_canary.py` validates all five result arms and writes the grouped
  analysis report.

## Prerequisites and safety

Install the project dependencies with `uv sync`. The generator and exposure runner
may download the pinned tokenizer and model into the cache, and the exposure runner
also loads the pinned clean corpus. Network access is therefore required unless these
artefacts are already cached. A suitable local device and enough disk space for model
and training artefacts are also required.

The source corpus must be a local JSONL file with exactly 256 unique rows containing
string `row_id` and `text` fields. The key must be an external file containing the
protocol key, readable only by its owner (`0600`). Keep plaintext records, augmented
corpora, model caches, checkpoints, results, and reports outside the repository. Use
private directories with mode `0700`; the scripts reject repository paths for these
inputs and outputs. The examples below use directories under `~/.local` and contain
no secret values.

Set a restrictive umask before creating any new artefacts:

```sh
umask 077
export CANARY_HOME="$HOME/.local/state/euroeval/canary"
export CANARY_SOURCE="$HOME/.local/share/euroeval/canary/source.jsonl"
export CANARY_KEY="$HOME/.config/euroeval/canary.key"
mkdir -p "$CANARY_HOME" "$HOME/.local/share/euroeval/canary" \
  "$HOME/.config/euroeval"
chmod 700 "$CANARY_HOME" "$HOME/.local/share/euroeval/canary" \
  "$HOME/.config/euroeval"
chmod 600 "$CANARY_KEY"
```

Do not place the key or private records in the repository, shell history, logs, or
sample files. Do not print them or commit generated artefacts.

## Example workflow

Generate the protocol artefacts:

```sh
uv run src/scripts/canary/generate_private_canary.py \
  --corpus-jsonl "$CANARY_SOURCE" \
  --key "$CANARY_KEY" \
  --augmented-dir "$CANARY_HOME/augmented" \
  --private-dir "$CANARY_HOME/private"
```

Run the exposure study. The default cache is already external, but it can be set
explicitly when a separate private cache is preferred:

```sh
uv run src/scripts/canary/run_private_canary_exposure.py \
  --corpus-jsonl "$CANARY_SOURCE" \
  --augmented-jsonl "$CANARY_HOME/augmented/augmented.jsonl" \
  --private-dir "$CANARY_HOME/private" \
  --key "$CANARY_KEY" \
  --output-dir "$CANARY_HOME/results" \
  --cache-dir "$CANARY_HOME/cache"
```

Validate and analyse the completed arms:

```sh
uv run src/scripts/canary/analyse_private_canary.py \
  --results-dir "$CANARY_HOME/results" \
  --output "$CANARY_HOME/report.json"
```

## Artefacts

Generation writes `augmented/augmented.jsonl` and private manifest and record files.
The exposure runner writes a fingerprint, one JSON result per exposure arm, a study
summary, and optional model checkpoints when `--keep-checkpoints` is used. Analysis
writes the requested JSON report with the common fingerprint and grouped metrics.
All of these files are private local artefacts and must remain outside version
control.

For the protocol design and limitations, see
[`docs/contamination-canary.md`](../../../docs/contamination-canary.md). These tools
only support offline research and protocol validation; they must not be used to
alter production scores, rankings, or production checker behaviour. Production
collection is disabled by default and its offline checker is report-only.
