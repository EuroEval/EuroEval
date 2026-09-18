# Private contamination canary

The contamination canary checks whether a decoder has learned keyed associations from
EuroEval's private 256-row audit corpus. It is not a benchmark task. Collection and
checking are experimental, disabled by default, report-only, and isolated from scores,
rankings, result promotion, finalisation, and deployment.

A positive result is evidence of exposure to this specific EuroEval release artefact. A
negative result means only that the checker found no evidence of that exposure. Neither
outcome proves whether the model saw an upstream benchmark dataset, and existing
score-only EEE records cannot be checked retroactively.

## Protocol

The canary contains 32 keyed association groups with eight rows per group. Each group
shares a trigger, exposed two-word target, and matched two-word control, while keyed
contexts keep all 256 prompts unique. Analysis uses the 32 groups as independent units;
it does not treat the eight rows within each group as independent observations.

The private Hugging Face corpus is `EuroEval/watermark-audit`, pinned at revision
`16d468bbacc284c912a8598a392239af2387ca53`. Its frozen augmented-corpus SHA-256 is
`37258fb324cf400bba4bd57cda430a73393928f5e09506594c3678adc38ff324`.
Only `row_id` and `text` are stored there. The key, targets, controls, private records,
model evidence, and reports remain outside the repository and the corpus repository.

The production evidence schema is `contamination-canary-evidence/v1`. Evidence contains
model and protocol provenance plus 256 ordered observations. Each observation stores
only `row_id`, a prompt SHA-256 digest, and the normalised first two generated words. It
must not contain prompts, corpus text, expected targets, controls, keys, credentials, or
raw exceptions.

## Evidence collection

When explicitly enabled, decoder evidence is collected while the model is already
loaded for its normal evaluation:

- local decoder evaluations reuse the loaded vLLM model;
- hosted decoder evaluations reuse the existing LiteLLM client;
- encoders produce the typed status `not_applicable` with reason `encoder` and make no
  canary generation calls;
- greedy continuations are bounded to six generated tokens and normalised to the
  first two Unicode-NFC words;
- immutable model revisions are cached by model identity and frozen protocol revision;
- mutable hosted aliases are recollected rather than represented as immutable evidence.

The ordinary CLI exposes `--contamination-canary`, but defaults to
`--no-contamination-canary`. Volunteer leases receive a canary reservation only when the
broker has `CONTAMINATION_CANARY_ENABLED=1`. A lease-bound authenticated endpoint sends
the worker the pinned `{row_id, text}` corpus. The worker never receives the Hugging Face
organisation token, scoring key, targets, controls, or private records.

Volunteer evidence has an independent durable outbox and acknowledgement. Ordinary
results can be finalised even if the evidence upload is temporarily unavailable. The
worker retries the byte-identical evidence later; broker reservations and receipts
provide replay and conflict protection. The broker writes one JSON object per immutable
model identity to the private bucket configured by `HF_CANARY_EVIDENCE_BUCKET` and
rejects a public bucket. The broker's `HF_TOKEN` is used server-side only.

## Offline report-only checking

Leaderboard collection invokes the checker before the no-new-results early return. The
checker consumes persisted evidence only: it never imports model-loading code, loads a
model, or queries a model provider. It remains separate from
`src/leaderboards/result_loading.py`, so canary output cannot enter public EEE records or
ranking data.

Checking is disabled unless `EUROEVAL_CANARY_CHECK_MODE=report-only`. There is no
enforcement mode. Configure maintainer-side checking with external, mode-restricted
paths:

```sh
umask 077
export EUROEVAL_CANARY_CHECK_MODE=report-only
export EUROEVAL_CANARY_KEY="$HOME/.config/euroeval/watermark-audit-v1.key"
export EUROEVAL_CANARY_PRIVATE_DIR="$HOME/.local/share/euroeval/private-canary-v5"
export HF_CANARY_EVIDENCE_BUCKET="EuroEval/private-canary-evidence"
# Optional overrides:
export EUROEVAL_CANARY_EVIDENCE_JSONL="$HOME/.local/state/euroeval/canary/evidence.jsonl"
export EUROEVAL_CANARY_REPORT_PATH="$HOME/.local/state/euroeval/canary/report.json"
```

`EUROEVAL_CANARY_PRIVATE_DIR` must contain `canary-manifest.json` and
`canary-records.jsonl`. The key and files must be owner-only; the checker also writes
synced evidence and reports with restrictive permissions. `HF_TOKEN` must be available
to read the private evidence bucket.

In report-only mode, malformed configuration, unavailable evidence, and scoring errors
are represented in the private report and do not interrupt leaderboard generation,
result upload, issue closing, or deployment. Historical score-only models receive no
inferred canary result.

## Local protocol research

The scripts under `src/scripts/canary/` generate the private protocol, run nested local
0/1/2/4/8 exposure studies, and analyse those studies. They do not operate the
production evidence pipeline or publish artefacts. Use external private paths:

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

## Validation status

The grouped pilots showed strong exact-completion discrimination at learning rate
`5e-4`, but the predeclared strict pilot gates did not formally pass: one replicate had
a small saturation reversal and another exceeded the secondary prefix-control limit.
Those gates have not been altered post hoc. Production collection therefore remains an
experimental report-only diagnostic. Any future policy or enforcement would require a
fresh preregistered confirmatory protocol and an explicit decision; no such enforcement
path exists in this implementation.
