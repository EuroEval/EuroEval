# Mac volunteer workers through local inference servers

Investigation date: 2026-11-23. Repository revision:
`bd47b45bf114090d18e7ec3076038255a5919e84`.

## Recommendation

Add an external-server execution mode to the existing volunteer worker. Reuse
EuroEval's LiteLLM backend for inference and retain broker authentication, leases,
heartbeats, recovery, staging, canaries, and maintainer review. Start with a
manually specified, already loaded model. Automatic queue-to-MLX/GGUF model
selection remains a separate task.

The HTTP inference connection is already implemented. Public volunteer acceptance
needs changes because the current protocol assumes local NVIDIA inference and
verified Hugging Face weights. An arbitrary OpenAI-compatible endpoint alone does
not establish model identity or benchmark compatibility.

## What already exists

- The Docker worker landed in [PR #2174](https://github.com/EuroEval/EuroEval/pull/2174).
  [PR #2230](https://github.com/EuroEval/EuroEval/pull/2230) added the ARM64 candidate
  work, followed by managed updates in
  [PR #2302](https://github.com/EuroEval/EuroEval/pull/2302) and release verification
  in [PR #2303](https://github.com/EuroEval/EuroEval/pull/2303).
- At inspection, GitHub reports no open PRs, and `git ls-remote --heads origin`
  reports only `main`. Local HEAD matches remote `main`. This does not inspect forks.
- [The contributor guide](../VOLUNTEER_WORKER.md) documents the existing one-command
  launcher. ARM64 support means Linux/NVIDIA, including DGX Spark, rather than Mac
  GPU support.
- [Dockerfile.worker](../Dockerfile.worker) installs CUDA and the generative extra.
  [The launcher](../src/scripts/run_volunteer_worker.py) selects approved images and
  preserves cache/state across safe updates.
- [LiteLLMModel](../src/euroeval/benchmark_modules/litellm.py) already accepts
  `api_base`, normalizes custom server model IDs, probes request parameters, and
  handles some unsupported logprob and structured-output parameters. Native
  Ollama support also exists. Prefer the common OpenAI interface for this new mode.
- [pyproject.toml](../pyproject.toml) also includes a Darwin vLLM/Metal path. That is
  another existing local route, but does not remove the worker's NVIDIA assumptions.

## Where the current worker blocks this

1. [hardware.py](../src/euroeval_worker/hardware.py) requires `nvidia-smi` and chooses
   an NVIDIA GPU. [runtime.py](../src/euroeval_worker/runtime.py) pins CUDA and checks
   GPU availability before evaluation.
2. [claim.ts](../api/worker/claim.ts) requires a selected GPU UUID/index and memory
   report. Release approval assumes an approved container digest. Native workers
   need an explicit supported release identity, rather than a fabricated image ID.
3. [model.ts](../api/worker/_lib/model.ts) resolves public pinned HF models, rejects
   repository Python and unsupported weight files, and estimates disk/memory fit
   from the canonical weight files. Those estimates do not describe an independently
   loaded quantized server model.
4. [safety.py](../src/euroeval_worker/safety.py) independently repeats model checks
   and checks installed backend compatibility, including the vLLM registry for
   generative models. A server mode needs its own evidence checks.
5. [evaluator.py](../src/euroeval_worker/evaluator.py) evaluates
   `model_id@revision`, requires safetensors, and normalizes output identity to the
   lease. Sending that string to a server is insufficient. LiteLLM's
   `get_model_config` even interprets a non-main revision as a legacy parameter.
   Never normalize an unverified server response into a canonical revision.
6. [shot_modes.py](../src/euroeval/shot_modes.py) defaults LiteLLM to zero-shot while
   local instruction/reasoning models can run both zero-shot and few-shot. Follow
   the lease's concrete dataset/split/shot identities explicitly. Preserve canary
   collection independently of ordinary scope.
7. [bpc_scoring.py](../src/euroeval/bpc_scoring.py) requires prompt logprobs for BPC
   and explicitly rejects missing vLLM/Metal prompt logprobs. Ordinary chat support
   does not establish BPC, encoder, or every structured-output task's support.

## Proposed implementation sequence

### 1. Prove the benchmark client with LFM3

Use the existing oMLX installation and official
[LiquidAI/LFM3-1.7B-MLX-8bit](https://huggingface.co/LiquidAI/LFM3-1.7B-MLX-8bit).
The downloaded model card specifies MLX-VLM and says released mlx-lm support is
pending. Do not assume that `mlx_lm.server` can load this architecture.

Keep this experiment local. Test model discovery, a bounded chat completion, four
validation examples from SST-5, and four from SQuAD through `Benchmarker`. Preserve
upstream splits and label all reduced configurations unofficial. Use one iteration,
explicit instruction mode, zero-shot, and an 8,192-token context override. Save no
canonical results. Cap the experiment at 15 minutes, excluding download setup.

Expand implementation acceptance to few-shot, multiple choice, structured NER,
request rejection/fallback, and required canary collection. Measure request success
and parsed outputs separately from model quality. Eight examples are no quality
estimate.

### 2. Add a narrow worker execution choice

Proposed CLI inputs, not existing flags: external-server mode, API base URL, served
model ID, expected artifact manifest, and optional API-key environment variable.
Keep canonical model ID/revision separate from the request's served-model alias.
Reuse the existing evaluator protocol instead of adding a benchmark backend for
each application.

Perform preflight before claiming work. Check discovery where supported, a bounded
real completion, required request/response features, effective context limit,
loaded-model binding, and task support. Pin configuration for the lease; check it
again after reconnect or resume. Refuse incompatible work before taking the lease.
The endpoint and its credentials remain local configuration, not broker fetch URLs.

For the first development slice, use a local fixture broker and a fixed model
manifest. Do not consume the public queue while the protocol cannot express this
execution mode. Automatic model lookup and server lifecycle management are excluded.

### 3. Extend broker and review contracts

Add an explicit execution capability for external generative servers and keep the
existing NVIDIA path working. Update the claim, lease, persisted state, submission,
and review contracts together. Advertise verified task capabilities and let the
broker choose an allowed scope; workers must not silently drop required tasks.

Record the source checkpoint/revision, actual artifact repository/revision or GGUF
file hash, quantization, tokenizer/chat-template identity, server engine/version,
effective context, and inference settings. Distinguish locally verified evidence
from contributor declarations. `/v1/models` is an alias check, not weight attestation.
Record endpoint-mediated speed as such rather than comparing it to CUDA hardware
throughput without qualification.

Canonical coverage currently keys on model/revision and result identity. Decide
with the maintainer how quantized artifacts map to leaderboard entries before
enabling public claims or promotion. Otherwise an 8-bit result can incorrectly mark
the original checkpoint covered. Manual artifact binding is enough for the first
slice; automatic conversion matching can wait.

Preserve existing fencing, exact result digests, independent scope validation,
canary requirements, and manual review. Add regression checks for both execution
modes, identity mismatches, quantization collisions, missing task capabilities,
shot coverage, resume after server/model changes, and output upload retries.

### 4. Package the contributor experience

Start with a native macOS worker plus a native inference server. That gives direct
access to Apple hardware facts and avoids requiring Docker for Mac contributors.
Provide one pinned installation/launch command once native release verification is
defined. Keep the current NVIDIA launcher available.

A later CPU-only Linux ARM64 image could run the evaluation client while the Mac
hosts inference. This would reuse container release controls, but needs separate
image approval, explicit host-server connectivity, and a way to report actual
inference-host metadata. Replacing the CUDA base image alone cannot solve these.

Treat Ollama, llama.cpp, oMLX, and Unsloth Studio as candidate servers for the same
client. Document each as supported only after its endpoint passes the required
tasks. The common API removes adapter duplication, not differences in capabilities.
Primary API references include
[Ollama compatibility](https://docs.ollama.com/api/openai-compatibility),
[llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md),
and [MLX LM server](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md).

## Local experiment evidence

The bounded probe completed locally on 2026-11-23:

- oMLX 0.8.0 discovered and loaded the official 8-bit artifact. The downloaded
  revision is `4d6d94c9c2c2f249f1f740f4f08e071b0dbfe099`.
- A direct chat request returned `hello` after thinking was explicitly disabled in
  the isolated model settings. With the initial default, all 16 output tokens were
  reasoning and there was no visible answer. Preflight must check semantics and
  reasoning settings, not just HTTP success.
- Both reduced SST-5 and SQuAD runs completed through the unmodified `Benchmarker`
  and LiteLLM backend, with zero reported failed instances. Each source split was
  capped at four rows before normal bootstrapping and filtering. This is not eight
  distinct scored examples or a leaderboard-quality measurement.
- Initial runs with `bootstrap_samples=False` generated responses but failed during
  ground-truth extraction. `generation._extract_ground_truth` accesses the `label`
  column on a columnless empty cached dataset. The final probe uses normal
  bootstrapping. This separate defect was not changed in this investigation.
- The server alias has no HF metadata, so EuroEval reported unknown parameter and
  vocabulary counts. This is another reason to bind an explicit artifact manifest.
- The isolated client used Python 3.12, EuroEval `18.4.0.dev`, LiteLLM `1.110.2`,
  Transformers `5.24.0`, and PyTorch `2.15.0`. It installed the current repository
  without the generative extra and resolved current dependencies. This does not
  validate the repository's frozen lockfile or published Docker dependency set.

Reproduction inputs and evidence remain in `/tmp/euroeval-mac-probe`:
`benchmark_probe.py`, `report.json`, `benchmark-final.log`,
`benchmark-trace.log`, `chat-no-thinking.json`, and `server-no-thinking.log`.
The model files are in `models/LFM3-1.7B-MLX-8bit`. The temporary server was stopped
after the probe. No public lease was claimed and no result was submitted or published.

Remaining evidence gaps are other servers, few-shot scope, structured NER,
required canaries, BPC, public broker acceptance, and immutable artifact verification.

## Concrete PR plan: volunteer evaluation of a manually configured GGUF

Proposed title: `feat: support a fixed GGUF model in external-server workers`.

Sol plans and reviews. Mistral Large 4 implements small sequential slices. The PR
reuses the existing LiteLLM client and volunteer lifecycle. No second benchmark
CLI, llama.cpp Python bindings, server registry, automatic conversion discovery,
model downloader, or process supervisor.

### Prerequisite experiment, not a new feature

Sol installs llama.cpp v1.0.0 in an isolated location and downloads only
`LFM3-1.7B-Q8_0.gguf` from official repository
`LiquidAI/LFM3-1.7B-GGUF`, revision
`cd0c74c575b5c416f3011b5b4c298077f26ca8c2`. Verify the file's SHA-256 against HF
LFS metadata and record the binary build/backend. Both the release and artifact
exist at planning time. Validate the startup flags against that pinned binary:

```sh
llama-server -m /path/LFM3-1.7B-Q8_0.gguf \
  --alias lfm3-q8 --host 127.0.0.1 --port 8080 \
  --ctx-size 8192 --parallel 1 --reasoning off
```

Use the existing `euroeval --api-base` CLI and the previous temporary public
`Benchmarker` fixture. Cap validation rows independently within source splits,
label reduced datasets unofficial, isolate output caches, and save no canonical
results. Test SST-5, SQuAD, multiple choice, structured NER, then explicit few-shot.
Keep a manifest of the declared server settings, file hash, versions, exact
commands, sample IDs, parsed outputs, failures, and scoring under `/tmp`. Run both
MLX and GGUF using the same fixture and generation settings. Passing means
completed scoring and usable outputs, not identical scores or token sequences.
Cap each experiment at 15 minutes after setup.

Q8_0 is a useful first compatibility comparison with MLX 8-bit, not a numerical
reference. Their quantization differs. Use official BF16 GGUF only if needed to
investigate discrepancies; do not download every variant. There is no new
persistent script unless the experiment demonstrates a missing product capability.

Sources: [official GGUF model card][lfm3-gguf],
[llama.cpp v1.0.0][llama-release], and [server API][llama-server].

### One maintainer decision before public integration

Recommended pilot rule: queue the GGUF artifact repository as its own model entry,
not as fulfillment of the upstream BF16 checkpoint's issue. Bind exactly one GGUF
filename/hash to each approved artifact repository/revision. For the pilot, that
means Q8_0 only for the official LFM3 GGUF repository revision above. Other files
in the same repository are not interchangeable. A binding cannot later change
filename/hash while retaining the same result identity.

This preserves current model/revision coverage keys without adding a quantization
column to every canonical storage layer. Store the upstream checkpoint relationship
and selected artifact in submission metadata. A different quantization requires a
distinct approved model entry or a later identity-schema change; it cannot reuse
this pilot's identity. Do not relabel GGUF results as the upstream model.

The maintainer must accept this separate-entry, one-artifact-per-revision rule
before enabling the pilot. If they require multiple quantizations under one model
entry, stop this scope and plan the coverage/storage identity change first. Do not
smuggle that migration into this PR. Automatic matching to existing source-model
queue entries remains excluded.

### Concrete implementation slices for Mistral

1. **Fixed binding and claim contract.** Add one maintainer-owned configuration
   entry for the artifact repo/revision, filename/hash, and upstream config
   repo/revision. Read it on the broker; the worker must not supply authoritative
   mappings. An external-server claim advertises `external_generation_v1` and an
   artifact descriptor matching that entry. Only the explicitly queued artifact
   entry is eligible. Do not scan the ordinary checkpoint queue for conversions.
   In `api/worker/claim.ts` and `_lib/model.ts`, branch before the NVIDIA-only
   hardware/weight-fit checks. Resolve the approved artifact and its pinned upstream
   config for generative task metadata. Never bypass the CUDA checks for old clients.
2. **Explicit execution variant.** In Python `types.py` and TypeScript
   `_lib/protocol.ts`, add an optional execution discriminator: omitted means the
   existing CUDA mode; `external_server` carries the approved artifact descriptor.
   Keep GPU selection fields required for CUDA and absent for the external variant.
   External host facts are OS, architecture, RAM and disk; no invented GPU UUID or
   CUDA memory estimate. Broker version checks still apply. Use explicit native
   worker/evaluation version evidence for this mode; do not assign it a CUDA image
   digest or route it through the legacy image fallback. Update persisted state,
   manifest validation, and readers that currently assume image/GPU fields.
3. **Native worker command and safety.** Add `--api-base`, `--served-model`,
   `--gguf-file`, and explicit context/generative-type inputs to the existing worker
   CLI. Supplying this complete set selects external mode; partial input fails.
   The user starts the inference server. Hash the local file, verify the approved
   binding from the lease, and check a bounded visible completion and served alias.
   Treat alias/model binding as contributor evidence, not remote weight attestation.
   Reuse existing inference error handling. Keep endpoint credentials local and
   redacted. Store an endpoint/config/artifact fingerprint in the active state and
   require it to match on resume. External mode skips NVIDIA discovery, CUDA pinning,
   vLLM registry checks, and canonical weight-download fit checks only.
4. **Evaluator wiring and exact scope.** Extend the existing `EuroEvalEvaluator`
   configuration with endpoint and served alias rather than adding a backend class
   per server. Keep its `evaluate(lease, output_path)` public call. Every invocation,
   including narrowed work and required canaries, receives the endpoint configuration.
   Execute concrete broker-owned dataset/split/shot identities; do not depend on
   LiteLLM AUTO being equivalent to local-model AUTO. Bind record identity to the
   approved GGUF model entry/revision only after checking the binding. Preserve
   artifact metadata through conversion. BPC and encoder support are out of scope;
   unsupported required tasks fail before upload, never disappear silently.
5. **Submission and review.** Extend existing submission metadata with the artifact
   descriptor, declared server engine/version/settings, and native execution facts.
   Update result/finalisation/review validators to compare these against the lease
   and binding. Preserve exact bytes, canaries, scope completeness, heartbeats,
   fencing, upload retries, and manual promotion. Coverage and issue completion apply
   to the queued GGUF entry only, never its upstream checkpoint. Do not add a second
   staging pipeline or change canonical keys under the pilot rule.
6. **Tests and guide.** Add focused Python and broker tests through public interfaces.
   Extend current fixtures rather than starting another test framework. Update
   `VOLUNTEER_WORKER.md` with the verified llama.cpp recipe, native worker command,
   fixed-artifact restriction, settings, shutdown and recovery. Public activation is
   a maintainer release action, not a side effect of running tests.

Sol must turn the execution variant into exact Python/TypeScript declarations and
verify all affected readers before handing slice 1 to Mistral. File ownership is:
CLI/runtime/hardware/safety for execution selection, evaluator for benchmark calls,
broker/types/state for transport and recovery, and existing API/review modules for
lease and submission validation. Do not create a general plugin interface.

### Acceptance gates

- Existing NVIDIA worker fixtures still claim, evaluate, resume, and submit without
  new configuration. Legacy leases deserialize with their existing meaning.
- External claims never invoke `nvidia-smi`, fake an image digest, or lease an
  upstream checkpoint issue. Wrong file/hash/revision and unapproved artifacts fail.
- Two different GGUF files cannot establish coverage for one approved identity.
- The endpoint, artifact, or configuration changing across resume prevents further
  evaluation/upload. Fresh evaluations still check binding before each lease.
- The worker produces precisely the leased split/shot identities, including a
  narrowed scope and required canary. Missing task capability cannot finalize.
- A fixture broker exercises claim through staging/finalise, failed inference,
  interrupted execution, and upload-only recovery. No production credentials or
  public side effects are used by tests.
- A real Mac llama.cpp run completes bounded generation/scoring through the worker
  evaluator. Sol compares it with the MLX fixture without claiming score equivalence.
  An actual complete public pilot requires maintainer activation and full leased
  scope; the tiny experiment alone cannot establish public acceptance.
- Run changed-module tests, affected Node broker tests, generated scope checks if
  changed, and `make check`. If package code changes, update its changelog. Keep the
  unrelated bootstrap-disabled defect separate.

### Sol/Mistral workflow

Mistral Large 4 is available in the current catalogue through OpenCode as
`opencode/mistral-large-4` and Mistral Vibe as `mistral-large-4-0-3`. Resolve the
chosen configured provider again when implementation begins. This plan does not
start implementation or authorize paid overage.

Sol owns the contract, scope, official-source checks, and empirical acceptance.
Delegate one bounded implementation slice at a time to Mistral; require changed
files, focused check results, and unresolved issues. Stop after a slice that forces
an unplanned protocol or identity decision. No parallel implementations.

Apply `astrobyte-mode` and `astrobyte-architect` to scope and contracts,
`git-workflow-and-versioning` to isolation, applicable TDD/security/TypeScript
skills to changes, and `astrobyte-git-identity` before committing. An independent
read-only Sol review via `astrobyte-review` must return `PASS` on the exact final
diff before a push or PR. Small findings can be fixed by the lead; a material fix
needs a new independent review. Keep agent process details out of the actual PR
body. No implementation agents or publishing run are started by this planning task.

This is one cross-stack PR with sequential reviewable slices. If the native-release
or identity work grows beyond these boundaries, split it before implementation
rather than adding a generic release/attestation platform. Keep additional-server
certification, auto-download, queue conversion discovery and broad hardware
scheduling out of this PR.

[lfm3-gguf]: https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF
[llama-release]: https://github.com/ggml-org/llama.cpp/releases/tag/v1.0.0
[llama-server]: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

## Implementation evidence, 2026-11-23

Work is on `feat/gguf-volunteer-worker`. The external claim gate stays disabled
by default; no production configuration has changed. The implementer uses Mistral
Vibe. OpenCode must only be selected when the operator explicitly names it.

The official Q8_0 file downloaded at the pinned revision has SHA-256
`8f6b24182258205e9e01e21f1f24a1f208c1e8115578825d202ce39976af20b6`,
matching the anonymous HF metadata. The model API returns this as `lfs.sha256`,
not `lfs.oid`; a live response check caught and corrected the initial test fixture.

llama.cpp v1.0.0 built locally with Metal. The initial embedded UI asset provision
failed; disabling UI/prebuilt UI and moving the stale temporary build UI directory
allowed the server-only build to complete. The binary serves on loopback port
8766, with context 8192, one slot, and reasoning disabled.

Reduced zero-shot SST-5, SQuAD, Life in the UK, and CoNLL-en runs completed scoring
through the existing Benchmarker/LiteLLM client with zero reported failed instances.
Fixtures select at most 128 training rows and four validation/test rows per source
split, retain normal bootstrapping, and use isolated model-output caches. The checks
are unofficial, local, and too small to estimate quality. Evidence is in
`/tmp/euroeval-mac-probe/llama_probe.py`, `llama-report.json`,
`llama-benchmark.log`, and `llama-server.log`. Explicit few-shot runs also
completed all four tasks with zero reported failed instances and confirmed
`few_shot=True` in every record. Their evidence is `llama_fewshot_probe.py`,
`llama-fewshot-report.json`, and `llama-fewshot-benchmark.log` in the same temporary
directory. The integrated worker evaluator subsequently produced eight EEE records from
the
four tasks in both shot modes. Every record passed the leaderboard name reader and
EEE round-trip with the approved GGUF repository identity. Evidence is
`worker_probe.py`, `worker-probe.log`, and `worker-records.jsonl` in the same
temporary directory. No result was submitted to the public broker.

Final checks used the installed Node runtime for pre-commit's Markdown hook because
nodeenv could not parse an upstream prerelease version. All hooks then passed,
including Ruff, funcsort, ty, and markdownlint. The full API suite still native-aborts
in existing Redis-bridge tests under Node 26, Node 24, and tsx; focused external
claim/upload/finalise tests pass. No tests were skipped or weakened to avoid this.
