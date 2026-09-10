# EuroEval volunteer GPU worker

This guide covers both running a worker and operating the broker-backed image. A
worker claims one compatible evaluation at a time, downloads the pinned model and
dataset, evaluates one language, and uploads records to the private staging bucket.
It does not publish results directly to the public leaderboard.

## Contributor quick start

You need a Linux x86_64/amd64 computer with an NVIDIA GPU, a current NVIDIA
[driver](https://www.nvidia.com/en-us/drivers/),
[Docker Engine](https://docs.docker.com/engine/install/), and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The current image does not support NVIDIA DGX Spark or other Linux arm64 machines.

### 1. Verify that Docker can use the GPU

Copy and run both commands:

```sh
nvidia-smi
docker run --rm --gpus all ubuntu nvidia-smi
```

Both commands should list your GPU. If the second command fails, finish configuring
the NVIDIA Container Toolkit before continuing.

### 2. Start the worker

Copy and run this block once:

```sh
IMAGE=ghcr.io/euroeval/euroeval-worker:latest
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
  "$IMAGE"
```

On the first run, open the GitHub URL printed in the terminal, enter the short code,
and approve access. The worker then starts evaluating compatible queued work and
continues until you press Ctrl-C. If no work is available, it waits and checks again.

The named Docker volume preserves authentication, model downloads, and unfinished
work between runs. To contribute again later, run the same `docker run` command; you
normally will not need to authenticate again.

Do not add `--privileged`, host networking, the Docker socket, a home-directory
mount, or project credentials. The worker never needs your `HF_TOKEN`, GitHub token,
or Upstash credentials.

### Optional: pin the exact image version

To pin a deployment, replace the `IMAGE=...:latest` line in step 2 with these lines,
then run the rest of that same block. The worker still starts only once, using the
immutable digest reference:

```sh
docker pull ghcr.io/euroeval/euroeval-worker:latest
IMAGE="$(docker image inspect ghcr.io/euroeval/euroeval-worker:latest \
  --format '{{index .RepoDigests 0}}')"
```

Record the resulting `ghcr.io/euroeval/euroeval-worker@sha256:...` value for later
runs if the deployment must remain pinned.

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
| `VOLUNTEER_WORKER_IMAGE_DIGEST` | Configured/required image digest (`sha256:...`); not runtime attestation |
| `VOLUNTEER_WORKER_VERSION` | Exact supported worker protocol/package version |
| `VOLUNTEER_MARKER_SECRET` | Required HMAC-SHA-256 secret for issue state and Hall credit markers |
| `WORKER_COORDINATOR_LOGIN` | Account used for temporary issue assignment |
| `HF_STAGING_BUCKET` | Private EU Hugging Face bucket (`namespace/bucket`) |
| `HF_TOKEN` | Token authorised to write that staging bucket |
| `UPSTASH_REDIS_REST_URL` | Upstash Redis REST URL |
| `UPSTASH_REDIS_REST_TOKEN` | Upstash Redis REST token |
| `WORKER_COORDINATOR_SECRET` | Secret for the local coordinator lock endpoints |
| `VOLUNTEER_COORDINATOR_STANDALONE` | Explicit isolated-migration lock bypass (`1` only) |
| `VOLUNTEER_PROMOTION_SECRET` | Secret for maintainer promotion requests |
| `VOLUNTEER_LEASE_SECONDS` | Optional bounded lease duration |
| `VOLUNTEER_SCOPE_POLICY_JSON` | Optional complete generated policy override |

Every internal queue host must set `VOLUNTEER_COORDINATOR_URL` and
`WORKER_COORDINATOR_SECRET`. Queue claims fail closed unless the shared broker lock
can be acquired; do not run a local queue against a shared issue set without both
values. For a deliberately isolated migration only, set
`VOLUNTEER_COORDINATOR_STANDALONE=1` to disable the shared lock. This override is
not safe for normal operation and must never be used on a shared queue host.

Set secrets in Vercel's encrypted environment configuration, never in the
repository or workflow file. The OAuth client ID and secret are both required: the
broker uses the secret to revoke the short-lived GitHub grant before returning an
opaque broker credential. Neither the GitHub token nor any project secret is returned
to workers. Restrict the broker's Hugging Face token to the staging bucket.

`VOLUNTEER_WORKER_IMAGE_DIGEST` must be updated to the digest printed by the image
workflow. It is configured/required provenance carried by the broker lease, not an
attestation of the runtime image. Deploy a new digest for upgrades; never configure a
floating image name or tag. Check that the digest is the expected `linux/amd64`
manifest before updating the environment. Human or independent verification of the
running image remains required. Generate the default scope with
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

Every decision first obtains a broker-side reservation keyed by issue and
submission. The reservation is authenticated with `VOLUNTEER_PROMOTION_SECRET`,
binds the outcome and exact identity/digest list, renews while work is retried, and
becomes a durable terminal record after promotion. A second reviewer may resume the
same outcome, but an opposite concurrent outcome is refused.

The review command then re-downloads and validates the manifest and exact result
bytes. Approval checks every canonical destination, uploads only the submission's
records, verifies the resulting metadata and bytes, writes an immutable decision
artifact, and only then calls the broker with the reservation token. Rejection writes
its decision before calling the broker and sends the same validated identity/digest
list so the broker can safely reclaim reservations. Both operations are safe to rerun
after a partial failure; the opposite terminal outcome is refused.

A rejected language becomes leaseable again while its rejected submission remains in
 the signed issue audit marker. Result identity reservations are reclaimed only after
 a token-checked broker transition; private audit manifests and result files remain
 available for review. A later local queue claim appends its VM marker and never
 deletes this signed history. Acceptance of one language does not award credit. Once every
selected exact language has an accepted submission, the broker adds `results-ready`
and writes the immutable Hall marker for the contributor with the largest sum of
server-derived accepted result counts. Lower-case GitHub login order breaks ties.

### Image publishing workflow

`.github/workflows/worker-image.yaml` builds only `linux/amd64` and enables BuildKit
layer caching. Pull requests disable attestations so Docker can load the local image,
then smoke its normal entrypoint as UID 10001 without a GPU; they cannot push. Trusted
pushes to `main` and workflow dispatches publish a commit-SHA tag with provenance and
an SBOM. Default-branch builds also update the contributor-facing `latest` tag.

After publishing, the workflow asks the GitHub Packages API to make the package
public, removes its Docker credentials, and anonymously pulls the exact published
digest. Changing package visibility requires package-admin permission. If the
workflow token cannot do this on first publication, an organization or package owner
must grant the repository's Actions workflow package-admin access or make the package
public once in GitHub's package settings. The workflow warns when API verification
cannot confirm public visibility and fails clearly if the anonymous pull does not
work; it never treats an unverified visibility change as success.

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
