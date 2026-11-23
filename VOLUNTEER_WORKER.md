# EuroEval volunteer GPU worker

This guide is for volunteers who want to contribute GPU time. You do not need to set
project environment variables or provide GitHub, Hugging Face, or Upstash credentials.
EuroEval maintainers should use the [volunteer worker operations
guide](VOLUNTEER_WORKER_OPERATIONS.md) instead.

A worker claims one compatible model/language evaluation at a time, downloads the pinned
model and datasets, and uploads records to private staging. The broker skips results
already covered by the same model revision at the requested or a newer EuroEval version.
The worker evaluates only the missing official datasets. It does not publish results
directly to the public leaderboard.

## Contributor quick start

You need a Linux x86_64/amd64 or ARM64/aarch64 computer with an NVIDIA GPU, an NVIDIA
[driver](https://www.nvidia.com/en-us/drivers/) compatible with the image's CUDA 13
stack, [Docker Engine](https://docs.docker.com/engine/install/),
[uv](https://docs.astral.sh/uv/getting-started/installation/), `curl`, and the
[NVIDIA Container Toolkit][toolkit]. NVIDIA DGX Spark/GB10 is
supported. The published `latest` image selects the correct architecture automatically;
no separate ARM command is needed.

[toolkit]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

Use a GPU with enough free memory for the queued model and enough disk space for its
downloads. Avoid running the worker alongside another GPU workload unless sufficient
memory is available.

### 1. Verify that Docker can use the GPU

Copy and run both commands:

```sh
nvidia-smi
docker run --rm --gpus all ubuntu nvidia-smi
```

Both commands should list your GPU. If the second command fails, finish configuring the
NVIDIA Container Toolkit before continuing.

### 2. Start the worker

Install uv first using its [official installation instructions][uv-install] (for
example, the official standalone installer), then reopen your terminal if needed so
`uv` is on your PATH. uv supplies Python 3.12 and installs the launcher's only
third-party dependency, Click 8.1.8, in an isolated environment; no project installation
is needed.

Run the standalone launcher with one command. No repository clone or saved script is
needed:

```sh
curl -fsSL \
  https://github.com/EuroEval/EuroEval/raw/main/src/scripts/run_volunteer_worker.py |
  uv run --no-project --python 3.12 --with 'click==8.1.8' python -
```

Leave it running: it discovers the public approved release at startup and between jobs,
pulls that exact immutable image, and starts a fresh container for each safe handoff. No
Docker socket is mounted inside the worker. Docker access must already work as your
host user. The launcher uses Click on the host; evaluation and GitHub device sign-in
run inside the image, not in the host Python environment.

[uv-install]: https://docs.astral.sh/uv/getting-started/installation/

Append launcher options after the final `python -`. For example, `python - --once`
finishes at most one evaluation (or exits when idle). `--gpu-health-check` runs the
worker GPU diagnostic once without claiming work. Use
`--gpus 'device=0'` to select a GPU, or `--gpu-memory-utilisation 0.8` to limit offered
GPU memory, or `--context NAME` for another Docker context. Architecture is detected
from Docker; `--architecture arm64` additionally requires an ARM64 Docker host.
`--volume NAME` selects a named cache volume. Keep the default `--launcher-state-dir`
host journal alongside that cache; it contains only image/configuration metadata, not
credentials.

The launcher preserves the image's non-root user, read-only root, capability drop,
no-new-privileges, temporary filesystem, shared memory and persistent cache settings.
Discovery and pull failures retry with visible bounded backoff, then stop without
claiming new work. Repeated update-required responses also stop after bounded retries.
Ctrl-C forwards cancellation only to the owned container and waits; it never restarts
following cancellation. Re-run the same command later to contribute again.

### 3. Sign in and contribute

The launcher and the manual Docker alternative below run the same worker image. GitHub
sign-in, account permissions and evaluation behavior are the same for both.

On the first run, open the GitHub URL printed in the terminal, enter the short code, and
authorise the shared **EuroEval Volunteer Worker** GitHub App. Complete any GitHub
verification in the browser yourself. No repository or organisation membership is
needed, and the worker never asks you to paste a GitHub token.

The broker can post one `I'd like to evaluate this.` comment under your GitHub account
when needed for issue assignment. It retains your renewable GitHub grant encrypted on
the server; the worker receives only a revocable broker credential.

The worker then evaluates compatible queued work until you press Ctrl-C. If no work is
available, it waits and checks again. To claim at most one evaluation, append `--once`
to the launcher command (or after the image argument for manual Docker).

The named Docker volume preserves authentication, model downloads, and unfinished work
between runs. To contribute again later, run the same launcher command (or manual
`docker run` command); you normally will not need to authenticate again.

Do not add `--privileged`, host networking, the Docker socket, a home-directory mount,
or project credentials. The worker never needs your `HF_TOKEN`, GitHub token, or Upstash
credentials.

### Manual Docker alternative

Manual Docker remains available (update/restart responsibility stays with you):

```sh
docker volume create euroeval-worker-cache

docker run --rm -it \
  --pull=always \
  --name euroeval-worker \
  --gpus all \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=2g \
  --shm-size=2g \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --pids-limit=512 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v euroeval-worker-cache:/cache \
  ghcr.io/euroeval/euroeval-worker:latest
```

### Optional: pin the exact image version

For the launcher, use `--image ghcr.io/euroeval/euroeval-worker@sha256:...`. The pin
must match the broker-approved manifest-index digest for your architecture. If it no
longer matches, the launcher stops with guidance rather than advancing silently.
Original managed work still resumes before any upgrade. The following describes manual
Docker pinning:

Pull the current image and note the `sha256:...` value on its `Digest:` line:

```sh
docker pull ghcr.io/euroeval/euroeval-worker:latest
```

Replace `ghcr.io/euroeval/euroeval-worker:latest` in the manual Docker command with
`ghcr.io/euroeval/euroeval-worker@sha256:...`, using that exact digest. Record it for
later runs. Do not select the first entry from an existing image's `RepoDigests`: a
cached image can also have older digest references for the same platform image.

You can also use a specific immutable image reference supplied by a maintainer directly
in the manual Docker command, without pulling `latest` first. This permits validation
of a candidate image before managed release promotion; it does not mean that candidate
is already `latest` or broker-approved for the launcher. Use the same documented Docker
settings and sign-in flow above. After promotion, a fresh run of the recommended
launcher command validates the managed path separately.

A pin does not freeze the broker's supported worker version. If the broker requests a
worker update, pull the newly promoted image and replace the old pin.

## Native external-server mode (macOS, build-ready)

A native macOS worker can evaluate through a locally running OpenAI-compatible
inference server instead of an NVIDIA container. This mode is **build-ready but not
production-activated**: the broker feature gate `VOLUNTEER_EXTERNAL_WORKERS_ENABLED`
remains off by default, and turning it on is a separate maintainer release action, not
a side effect of installing or running this code. Until then, external claims receive
`404 External worker claims are not enabled`.

The pilot binds exactly one approved GGUF artifact. The artifact repository is queued
as **its own model entry**; the worker never claims the upstream BF16 checkpoint's
issue, and there is no automatic matching between queued checkpoints and GGUF
conversions. A different quantization of the same model is a different artifact and is
not covered by this pilot; it would need its own maintainer-approved binding and queue
entry. One approved filename/hash is bound per artifact repository revision, and a
binding can never change file or hash while keeping the same result identity.

### 1. Install the worker (base package only)

From a checkout containing this feature, install the base package **without** the
generative CUDA extra. External mode uses HTTP inference and never initializes
vLLM/CUDA:

```sh
uv venv .venv && . .venv/bin/activate
uv pip install -e .
```

### 2. Install llama.cpp v1 and the pinned GGUF file

Build or install [llama.cpp v1.0.0](https://github.com/ggml-org/llama.cpp/releases/tag/v1.0.0)
locally (Metal on Apple Silicon), then download only the pinned file from the official
artifact repository `LiquidAI/LFM3-1.7B-GGUF` at revision
`cd0c74c575b5c416f3011b5b4c298077f26ca8c2` and verify its SHA-256 against the approved
binding before starting anything:

```sh
curl -L -o LFM3-1.7B-Q8_0.gguf \
  "https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/cd0c74c575b5c416f3011b5b4c298077f26ca8c2/LFM3-1.7B-Q8_0.gguf"
echo "8f6b24182258205e9e01e21f1f24a1f208c1e8115578825d202ce39976af20b6  LFM3-1.7B-Q8_0.gguf" | shasum -a 256 -c -
```

### 3. Start the inference server yourself

The worker never starts, supervises, or stops the server. Use the observed llama.cpp
v1 server recipe:

```sh
llama-server -m /path/LFM3-1.7B-Q8_0.gguf \
  --alias lfm3-q8 --host 127.0.0.1 --port 8080 \
  --ctx-size 8192 --parallel 1 --reasoning off
```

Keep the server running with unchanged settings for the whole lease. Do not reload or
swap the model, change the context size, or restart the server between resume attempts:
the worker stores an endpoint/alias/artifact/context/type fingerprint in its state
directory and refuses to resume evaluation or upload when it changed. Shut the server
down only after the worker has finished (`--once` exits after at most one lease) or
after the lease expired and was released.

### 4. Run the worker once against the endpoint

```sh
euroeval-worker --once --state-dir ~/.euroeval-worker-state \
  --api-base http://127.0.0.1:8080/v1 \
  --served-model lfm3-q8 \
  --gguf-file /path/LFM3-1.7B-Q8_0.gguf \
  --gguf-binding /path/to/gguf-bindings.json \
  --max-context-length 8192 \
  --generative-type instruction_tuned
```

For an authenticated inference server, add `--api-key-env MY_SERVER_KEY` after setting
that environment variable locally. The key is never persisted or logged.

Supplying any of `--api-base`, `--served-model`, `--gguf-file`, `--gguf-binding`,
`--max-context-length`, or `--generative-type` requires all of them; partial input
fails. `--gguf-binding` is a JSON file with either one six-field artifact descriptor
or the `{"version": 1, "artifacts": [...]}` envelope — use the checked-in
`api/worker/gguf-bindings.json` from a repository clone or an exact copy of it. The
worker hashes the local file, requires it to match exactly one approved binding by
filename and SHA-256, runs one bounded preflight completion against the served alias,
and only then claims work. The endpoint, alias, context length, and any API key stay
local configuration: they are never sent to the broker, so the broker cannot and does
not attest the server engine, version, or effective settings — its evidence is the
approved artifact descriptor plus the client's inference-engine record. If the actual
server engine/version is not available in the submission metadata, it is reported as
declared/unknown, never invented. External claims skip NVIDIA discovery, CUDA pinning,
and the container image checks; they report only host facts (OS, architecture, RAM,
free disk) and never fabricate a GPU or image digest.

Use `--once` for the pilot and keep `--state-dir` on a private disk: it holds the
broker credential and the resume fingerprint, never the endpoint API key. Re-run the
same command to resume unfinished work before lease expiry; after expiry the broker
may lease the language again and the old local state is archived.

## How volunteer evaluation works

1. The worker completes the broker-mediated GitHub App device flow. Any GitHub account
   can participate without repository or organisation membership.
2. It reports NVIDIA hardware and asks for a compatible queued model/language.
3. The broker anonymously verifies that the queued Hugging Face model is public,
   ungated, pinned to an immutable revision, safetensors-only, free of repository Python
   and remote-code configuration, within the reported disk/GPU limits, and exposes one
   supported EuroEval capability (`encoder` or `generative`).
4. The broker checks canonical results for the exact model revision and EuroEval
   version, then leases only missing official result identities. Older results without a
   verifiable revision are not reused. The worker independently refetches the pinned
   model metadata and capability, then evaluates its assigned datasets with remote code
   disabled and safetensors required.
5. Each record is uploaded idempotently to private Hugging Face staging.
6. For a lease that requires contamination collection, the worker explicitly runs the
   `contamination-detection` task alongside the normal official tasks. Generative models
   reuse the already loaded model for bounded greedy completions; encoder models record
   that the task is not applicable. A lease without that requirement selects only the
   normal official datasets, despite the public default including the canary. The
   resulting canary record uses the ordinary result upload and review path, while its
   broker lease field remains control-plane metadata.
7. The broker validates the complete lease and marks it ready for review.

Public volunteer leases are owned by the authenticated account, the broker's short-lived
lease, and a signed issue marker. The broker reconciles issue assignments for active
workers and submissions awaiting review, preserves manual assignees, and removes its own
assignments when they are no longer needed. GitHub can refuse an assignment; the signed
lease remains authoritative and work can continue.

Accepted signed submissions provide volunteer credit; the repository maintainer is
excluded from volunteer credit but can run a worker. Manual evaluations and older
assignee-backed volunteer leases retain their existing ownership and credit rules.
Removing or replacing a signed marker fences the public lease. The queue shows **Pending
review** after submission and **Awaiting publish** after accepted completion.

A worker can be interrupted safely. The local state and exact result bytes remain in the
named volume. Restarting before lease expiry resumes the same lease; after expiry, the
broker may lease the language again and the old local state is archived. An interrupt
does not immediately release an active lease. Submitted records can be retried without
changing their digest. Completed saved outputs resume upload rather than being
re-evaluated; an incomplete dataset can be rerun.

A `No volunteer evaluation work` message means no compatible, uncovered work is
currently available. `No NVIDIA GPU` means that the host runtime did not expose a usable
GPU. Transient claim failures are retried with backoff. The launcher handles safe update
handoffs automatically. For an HTTP `426` worker update request in manual Docker mode,
pull `latest` again and restart with the same cache volume; do not delete your saved
state.

## Data and credential handling

The `/cache` volume is private worker state. When a lease requires canary collection it
may temporarily contain the private `{row_id, text}` corpus and plaintext-free evidence.
It never contains the scoring key, expected targets, matched controls, Hugging Face
organisation token, or private scoring records. Back it up only if the backup is
protected as a secret, and remove it when decommissioning a worker. To discard a
worker's local credential and cached data:

```sh
docker volume rm euroeval-worker-cache
```

Stop the worker and contact the EuroEval maintainer to revoke its broker credential and
retained GitHub App grant before deleting the volume. Deleting local data alone does not
revoke server-side access. You can also withdraw GitHub App authorisation in your GitHub
settings; ask the maintainer to revoke the broker credential as well.
