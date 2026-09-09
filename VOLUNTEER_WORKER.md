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

A worker can be interrupted safely. Heartbeats protect an active lease; failed or
interrupted work is released, and result submission can be retried without changing
a record digest. A `No volunteer evaluation work` message is normal when the queue
is empty. `No NVIDIA GPU` means that the host runtime did not expose a usable GPU.

## Operator deployment

The worker broker runs as Vercel edge functions and uses Upstash Redis for short-
lived coordination. Configure these Vercel project environment variables:

| Variable | Purpose |
| --- | --- |
| `GITHUB_OAUTH_CLIENT_ID` | GitHub OAuth app client ID with device flow enabled |
| `VOLUNTEER_WORKER_IMAGE` | Full GHCR image reference including `@sha256:` |
| `WORKER_COORDINATOR_LOGIN` | GitHub account used for temporary issue assignment |
| `HF_STAGING_BUCKET` | Private Hugging Face staging bucket name |
| `HF_TOKEN` | Token authorised to upload to that staging bucket |
| `HF_STAGING_UPLOAD_URL` | Absolute HTTPS JSON upload endpoint |
| `UPSTASH_REDIS_REST_URL` | Upstash Redis REST URL |
| `UPSTASH_REDIS_REST_TOKEN` | Upstash Redis REST token |
| `VOLUNTEER_LEASE_SECONDS` | Optional lease duration, bounded by the broker |

Set secrets in Vercel's encrypted environment configuration, never in the
repository or workflow file. The OAuth application needs only the client ID in the
broker deployment; GitHub access tokens are held briefly by the auth endpoint and
are not returned to workers. The staging upload service must accept the broker's
Bearer token and JSON payload, use the requested bucket/path, and avoid logging
credentials or result content. Restrict its token to the staging bucket.

`VOLUNTEER_WORKER_IMAGE` must be updated to the digest printed by the image
workflow. Deploy a new digest for upgrades; never make a floating tag the broker's
worker image. Check that the digest is the expected `linux/amd64` manifest before
updating the environment.

### GitHub queue and review

The coordinator login is assigned to an issue while a language lease is active.
The broker's marker and lease checks are the ownership fence; do not manually edit
those markers or assign a second coordinator to the same request. A worker must
only receive models that are public, ungated, immutable, safetensors-only, and free
of custom Python or remote-code configuration.

Community submissions go to private staging and receive the
`community-review-ready` label after broker validation. Review the records and
coverage before promoting anything to the public results bucket or leaderboard.
Promotion is intentionally manual. Credit the authenticated contributor with the
largest accepted share for each model in the volunteer Hall of Fame; do not award
credit from an unverified username in a request or result payload.

### Image publishing workflow

`.github/workflows/worker-image.yaml` builds only `linux/amd64`, enables BuildKit
layer caching, and emits provenance and an SBOM. Pull requests build without
logging in to GHCR and cannot push. Only pushes to `main` and an explicit trusted
workflow dispatch log in and publish. The workflow prints the resulting immutable
content digest for the broker configuration.

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
