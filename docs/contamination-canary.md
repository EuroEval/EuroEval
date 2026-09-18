# Private contamination canary

The contamination canary is a private, local experiment for checking whether a model has
learned keyed associations from an evaluation-like corpus. It is not a benchmark task
and has no leaderboard or ranking impact. There is no production integration yet.

## Design

The canary contains 32 keyed association groups with eight rows per group (256 rows
total). Each group shares a trigger, exposed target and control target, while keyed
contexts keep prompts unique. Private generation creates the augmented rows and
plaintext records locally. The exposure study trains and scores nested local arms at 0,
1, 2, 4 and 8 exposures.

Analysis is performed at group level, not by treating the 256 rows as independent
observations. It reports exact-match, prefix-match and teacher-forced target
log-probability results, including the paired group-level comparisons and predeclared
gates.

The generator and study pin the SmolLM2 tokenizer/model revision used by the protocol.
The private Hugging Face dataset is `EuroEval/watermark-audit`, pinned at
`16d468bbacc284c912a8598a392239af2387ca53`. Only `row_id` and `text` are uploaded from
any source corpus. The canary secret, keyed associations, controls and all plaintext
private records remain external to the repository and are never uploaded by these
scripts.

## Running it

Use external, mode-restricted paths for every input and output. Do not put secrets or
canary records in the repository, logs or command history. The examples use placeholders
for paths and do not contain secret values.

```sh
uv run src/scripts/canary/generate_private_canary.py \
  --corpus-jsonl "$HOME/.local/share/euroeval/canary/source.jsonl" \
  --key "$HOME/.config/euroeval/canary.key" \
  --augmented-dir "$HOME/.local/state/euroeval/canary/augmented" \
  --private-dir "$HOME/.local/state/euroeval/canary/private"

uv run src/scripts/canary/run_private_canary_exposure.py \
  --corpus-jsonl "$HOME/.local/share/euroeval/canary/source.jsonl" \
  --augmented-jsonl "$HOME/.local/state/euroeval/canary/augmented/augmented.jsonl" \
  --private-dir "$HOME/.local/state/euroeval/canary/private" \
  --key "$HOME/.config/euroeval/canary.key" \
  --output-dir "$HOME/.local/state/euroeval/canary/results"

uv run src/scripts/canary/analyse_private_canary.py \
  --results-dir "$HOME/.local/state/euroeval/canary/results" \
  --output "$HOME/.local/state/euroeval/canary/report.json"
```

These commands perform no upload and do not call GitHub or the Hugging Face Hub for
publication. Keep the key at mode 0600 and private directories at mode 0700.

## Pilot status and limitations

The observed pilot was strong at learning rate `5e-4`. Strict pilot gates were not
formally passed: results showed saturation and the secondary control tolerance was not
met. This is evidence for continuing the investigation, not a validation claim. A
confirmatory protocol, including preregistered gates and independent replication, is
still required. The canary remains an offline research tool and must not be used to
alter production scores or rankings.
