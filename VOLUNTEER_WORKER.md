# EuroEval volunteer GPU worker

This guide covers both running a worker and operating the broker-backed image. A
worker claims one compatible evaluation at a time, downloads the pinned model and
dataset, evaluates one language, and uploads records to the private staging bucket.
It does not publish results directly to the public leaderboard.

## Contributor quick start

### Requirements

- Linux x86_64/amd64
- An NVIDIA GPU and a current NVIDIA driver
- Docker Engine with the NVIDIA Container Toolkit
- Network access to `euroeval.com` and Hugging Face
- Enough local disk for model, dataset, and result caches

The published image is **amd64 only**. NVIDIA DGX Spark and other Linux arm64
machines are not supported by this image. The container deliberately does not
bundle a driver or replace `nvidia-smi`; the NVIDIA runtime passes through the
host's driver tools and libraries.

### Select an immutable image

The workflow prints the image digest after each successful publish. Use that full
reference, not `latest` or a branch tag:

```sh
IMAGE='ghcr.io/euroeval/euroeval-worker@sha256:<digest-from-the-workflow>'
docker pull "$IMAGE"
docker image inspect "$IMAGE" --format '{{index .RepoDigests 0}}'
```

The digest is the deployment identity. Keep it recorded with the worker's
configuration so an update can be audited and rolled back.

### Run securely

Create a dedicated Docker volume. It contains the opaque broker credential, model
weights, Hugging Face metadata, and retry state; it is not a general-purpose host
volume.

```sh
docker volume create euroeval-worker-cache

docker run --rm -it \
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
  "$IMAGE" --once
```

On the first run, the worker prints a GitHub device-flow URL and a short code.
Open the URL in a browser, enter the code, and approve the flow. The worker then
stores only an opaque broker credential and the verified GitHub login in
`/cache/state.json`; it never receives or stores a GitHub access token. Keep the
terminal attached for the first run so the prompt and approval are visible.

After the first successful run, omit `-it` and `--once` for continuous service
operation. Keep `--rm` only when a supervisor recreates the container; a named
volume preserves authentication and caches when the container is replaced. Stop
a running worker with `docker stop euroeval-worker`.

`--gpus all` is the supported passthrough mechanism. Before starting a worker,
verify the host and runtime independently:

```sh
nvidia-smi
docker run --rm --gpus all --cap-drop=ALL --entrypoint nvidia-smi "$IMAGE"
```

The second command is only a diagnostic override of the image entrypoint. Do not
use `--privileged`, host networking, the Docker socket, or a mount of the host's
home directory. Do not pass `HF_TOKEN`, GitHub tokens, or Upstash credentials to a
worker; the broker handles authentication and staging uploads.

## How volunteer evaluation works

1. The worker completes the broker-mediated GitHub device flow.
2. It reports NVIDIA hardware and asks for a compatible queued model/language.
3. The broker issues a time-limited lease for an immutable model revision.
4. The worker evaluates with remote code disabled and safetensors required.
5. Each record is uploaded idempotently to private Hugging Face staging.
6. The broker validates the complete lease and marks it ready for review.

A worker can be interrupted safely. The local state and exact result bytes remain in
the named volume. Restarting before lease expiry resumes the same lease; after expiry,
the broker may lease the language again and the old local state is archived. An
interrupt does not immediately release an active lease. Submitted records can be
retried without changing their digest. A `No volunteer evaluation work` message is
normal when the queue is empty. `No NVIDIA GPU` means that the host runtime did not
expose a usable GPU.

## Operator deployment

The worker broker runs as Vercel edge functions and uses Upstash Redis for short-
lived coordination. Configure these Vercel project environment variables:

| Variable | Purpose |
| --- | --- |
| `GITHUB_TOKEN` | GitHub token for issue reads, labels, comments, and assignment |
| `GITHUB_OAUTH_CLIENT_ID` | OAuth app client ID with device flow enabled |
| `GITHUB_OAUTH_CLIENT_SECRET` | OAuth secret used to revoke each device grant |
| `EUROEVAL_VERSION` | Exact release supported by the generated scope policy |
| `VOLUNTEER_WORKER_IMAGE_DIGEST` | Allowed worker image digest (`sha256:...`) |
| `WORKER_COORDINATOR_LOGIN` | Account used for temporary issue assignment |
| `HF_STAGING_BUCKET` | Private EU Hugging Face bucket (`namespace/bucket`) |
| `HF_TOKEN` | Token authorised to write that staging bucket |
| `UPSTASH_REDIS_REST_URL` | Upstash Redis REST URL |
| `UPSTASH_REDIS_REST_TOKEN` | Upstash Redis REST token |
| `WORKER_COORDINATOR_SECRET` | Secret for the local coordinator lock endpoints |
| `VOLUNTEER_PROMOTION_SECRET` | Secret for maintainer promotion requests |
| `VOLUNTEER_LEASE_SECONDS` | Optional bounded lease duration |
| `VOLUNTEER_SCOPE_POLICY_JSON` | Optional complete generated policy override |

Set secrets in Vercel's encrypted environment configuration, never in the
repository or workflow file. The OAuth client ID and secret are both required: the
broker uses the secret to revoke the short-lived GitHub grant before returning an
opaque broker credential. Neither the GitHub token nor any project secret is returned
to workers. Restrict the broker's Hugging Face token to the staging bucket.

`VOLUNTEER_WORKER_IMAGE_DIGEST` must be updated to the digest printed by the image
workflow. Deploy a new digest for upgrades; never configure a floating image name or
tag. Check that the digest is the expected `linux/amd64` manifest before updating the
environment. Generate the default scope with
`src/scripts/generate_volunteer_scope_policy.py`; use the JSON override only for an
intentional, reviewed deployment policy.

### GitHub queue and review

The coordinator login is assigned to an issue while a language lease is active.
The broker's marker and lease checks are the ownership fence; do not manually edit
those markers or assign a second coordinator to the same request. A worker must
only receive models that are public, ungated, immutable, safetensors-only, and free
of custom Python or remote-code configuration.

Community submissions go to a private Hugging Face bucket in the EU region and
receive the `community-review-ready` label after broker validation. Redis is only
coordination state and is never a review source. Configure the maintainer shell with
`HF_STAGING_BUCKET`, an `HF_TOKEN` that can read staging and write
`HF_RESULTS_BUCKET` (default `EuroEval/results`), and
`VOLUNTEER_PROMOTION_SECRET`. Then use the durable review commands:

```sh
uv run python src/scripts/review_volunteer_results.py list
uv run python src/scripts/review_volunteer_results.py show <submission-id>
uv run python src/scripts/review_volunteer_results.py \
  --reviewer <github-login> approve <submission-id> --reason "checks passed"
uv run python src/scripts/review_volunteer_results.py \
  --reviewer <github-login> reject <submission-id> --reason "reason"
```

Every decision re-downloads and validates the manifest and exact result bytes.
Approval checks every canonical destination, uploads only the submission's records,
verifies the resulting metadata and bytes, writes an immutable decision artifact,
and only then calls the broker. Rejection writes its decision before calling the
broker. Both operations are safe to rerun after a partial failure; the opposite
terminal outcome is refused.

A rejected language becomes leaseable again while its rejected submission remains in
the issue audit marker. Acceptance of one language does not award credit. Once every
selected exact language has an accepted submission, the broker adds `results-ready`
and writes the immutable Hall marker for the contributor with the largest sum of
server-derived accepted result counts. Lower-case GitHub login order breaks ties.

### Image publishing workflow

`.github/workflows/worker-image.yaml` builds only `linux/amd64`, enables BuildKit
layer caching, and emits provenance and an SBOM for published images. Pull requests
load a local image and smoke its normal entrypoint as UID 10001 without a GPU. They
cannot push. Only pushes to `main` and an explicit trusted workflow dispatch log in
and publish. The workflow prints the resulting immutable content digest for the
broker configuration.

If a build fails, do not switch the deployment to a mutable base image or install a
host driver in the image. Check the pinned CUDA base, the locked `uv.lock`
dependencies, and the NVIDIA runner/toolkit separately.

## Data and credential handling

The `/cache` volume is private worker state. Back it up only if the backup is
protected as a secret, and remove it when decommissioning a worker. To discard a
worker's local credential and cached data:

```sh
docker volume rm euroeval-worker-cache
```

Revoking or replacing the broker credential is an operator action; deleting the
volume does not revoke a credential already issued by the broker. Contact the
EuroEval maintainer before decommissioning a registered contributor worker.
