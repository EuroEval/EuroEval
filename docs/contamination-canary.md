# Private contamination canary

EuroEval includes a prospective, non-scoring contamination canary. It tests for
exposure to one specific EuroEval release artefact; it does **not** prove that a model
saw an upstream benchmark dataset, identify its training source, or measure benchmark
score inflation.

## Protocol

The frozen private dataset is `EuroEval/watermark-audit` at revision
`16d468bbacc284c912a8598a392239af2387ca53`. Its 256 rows form 32 independent keyed
groups of eight. Evaluators receive only `{row_id, text}`. The scoring key, exposed
targets, matched controls, and private records remain maintainer-side.

The public evidence schema is `contamination-canary-evidence/v1`. Evidence contains
model and revision provenance, protocol and corpus revisions, row IDs, prompt SHA-256
digests, collection status, and at most two normalised completion words per row. It
never contains full prompts, corpus text, targets, controls, keys, credentials,
provider response objects, or raw exceptions.

## Collection and storage

Canary collection behaves as an auxiliary EuroEval dataset:

- it defaults on for suite, task, and language runs that do not specify `--dataset`;
- targeted `--dataset` runs omit it by default;
- `--contamination-canary` and `--no-contamination-canary` override the default;
- local decoders reuse the loaded vLLM model;
- hosted decoders use isolated requests through the existing LiteLLM wrapper;
- encoders emit `not_applicable/encoder` without a generation call;
- generation is greedy and bounded to six tokens, then reduced to the first two
  Unicode-NFC words.

The auxiliary `contamination-canary[-<language>]` result is appended to the ordinary
`euroeval_benchmark_results.jsonl`. EuroEval does not upload local results
implicitly. If a user keeps that file local, the evidence remains local. If the user
submits normal results, the canary record follows the same result endpoint and private
staging path as every other EEE record. Completion-bearing auxiliary records are scored
before approval and are not promoted to the public `EuroEval/results` bucket. They
remain in ordinary private staging (or the user's local result files) as the audit
source. Legacy queue and collection upload paths also retain auxiliary records locally
while excluding them from public bucket writes. There is no evidence sidecar, outbox,
reservation, receipt, or dedicated evidence bucket.

The private corpus uses the normal Hugging Face download cache and EuroEval's packaged,
obfuscated dataset credential, so no separate Hugging Face login is required.
`--download-only` caches the corpus alongside other requested artefacts. Offline
evaluation uses the cached revision; if it is missing or invalid, ordinary benchmarks
continue and the auxiliary result records `failed/corpus_unavailable`.

Volunteer workers receive the corpus through their authenticated lease because they do
not receive the broker's organisation token. Their auxiliary EEE result is still
uploaded through the ordinary result endpoint and staging bucket.

## Private leaderboard scoring

Only `src/leaderboards/` interprets submitted evidence. Evaluation code never receives
the private key or computes the decision. Leaderboard processing:

1. separates auxiliary canary records from ranking records;
2. validates their frozen corpus bindings and private-record provenance;
3. scores exposed and matched-control exact completions over 32 independent groups;
4. marks exposure when the exposed exact rate and exposed-minus-control difference are
   both at least 0.10 and the paired two-sided group sign-test p-value is at most 0.01;
5. warns the maintainer and asks whether to remove the whole canonical model from
   generated leaderboards using `[Y/n]` (default yes).

A non-interactive run defaults to removal when the detector is positive. Confirmed
removals are kept in owner-only audit state under `EUROEVAL_CANARY_PRIVATE_DIR`, so
repeated generation—including `--skip-results-processing`—continues to exclude the
model without publishing the evidence. Corrupt or unwritable durable exclusion state
blocks generation rather than silently re-ranking a removed model. Collected volunteer
evidence must also be scoreable before approval; a typed non-collected result still
allows unrelated ordinary results to proceed. Source records remain in private staging
or local storage for audit. Auxiliary records never become public metrics or rankings,
and all datasets/revisions/variants of a selected model are filtered from generated
leaderboards.

Configure private scoring with owner-only paths:

```sh
umask 077
export EUROEVAL_CANARY_KEY="$HOME/.config/euroeval/watermark-audit-v1.key"
export EUROEVAL_CANARY_PRIVATE_DIR="$HOME/.local/share/euroeval/private-canary-v5"
# Optional private report:
export EUROEVAL_CANARY_REPORT_PATH="$HOME/.local/state/euroeval/canary/report.json"
```

`EUROEVAL_CANARY_PRIVATE_DIR` must contain `canary-manifest.json` and
`canary-records.jsonl`. The directory must be mode `0700`; the key and private files
must be mode `0600`. Missing private configuration, unavailable corpus data, malformed
evidence, and scoring errors do not abort unrelated result processing. Historical
score-only records cannot be checked retroactively.

## Local protocol research

Research-only generation, exposure, and analysis tools live under
`src/scripts/canary/`; see its README. These scripts may use private targets and local
study artefacts. They are separate from production collection and leaderboard scoring.

The earlier green-list detector is retained only as a negative result: its five-arm
study produced about a 0.088 percentage-point lift and was rejected. Grouped keyed
completion pilots showed strong exact exposed/control separation, but the frozen pilot
reports remain failures under their predeclared stricter gates. The production policy
above is therefore versioned and should be changed only through a new protocol or
policy revision, never by post-hoc threshold adjustment.
