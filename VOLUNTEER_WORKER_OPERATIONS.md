# EuroEval volunteer worker operations

This guide is for EuroEval maintainers and infrastructure operators who deploy the
broker, review staged results, and publish the worker image. Volunteers do not need to
set any environment variables described here; they should use the
[volunteer worker guide](VOLUNTEER_WORKER.md).

## Operator deployment

The worker broker runs as Vercel edge functions and uses Upstash Redis for short- lived
coordination. Configure these Vercel project environment variables:

| Variable                           | Purpose                                                                  |
| ---------------------------------- | ------------------------------------------------------------------------ |
| `GITHUB_TOKEN`                     | GitHub token for issue reads, labels, comments, and assignment           |
| `GITHUB_OAUTH_CLIENT_ID`           | OAuth app client ID with device flow enabled                             |
| `GITHUB_OAUTH_CLIENT_SECRET`       | OAuth secret used to revoke each device grant                            |
| `EUROEVAL_VERSION`                 | Exact release supported by the generated scope policy                    |
| `VOLUNTEER_WORKER_IMAGE_DIGEST`    | Configured/required image digest (`sha256:...`); not runtime attestation |
| `VOLUNTEER_WORKER_VERSION`         | Exact supported worker protocol/package version                          |
| `VOLUNTEER_MARKER_SECRET`          | Required HMAC-SHA-256 secret for issue state and Hall credit markers     |
| `WORKER_COORDINATOR_LOGIN`         | Account used for temporary issue assignment                              |
| `HF_STAGING_BUCKET`                | Private EU Hugging Face bucket (`namespace/bucket`)                      |
| `HF_TOKEN`                         | Token authorised to write that staging bucket                            |
| `UPSTASH_REDIS_REST_URL`           | Upstash Redis REST URL                                                   |
| `UPSTASH_REDIS_REST_TOKEN`         | Upstash Redis REST token                                                 |
| `WORKER_COORDINATOR_SECRET`        | Secret for the local coordinator lock endpoints                          |
| `VOLUNTEER_COORDINATOR_STANDALONE` | Explicit isolated-migration lock bypass (`1` only)                       |
| `VOLUNTEER_PROMOTION_SECRET`       | Secret for maintainer promotion requests                                 |
| `VOLUNTEER_LEASE_SECONDS`          | Optional bounded lease duration                                          |
| `VOLUNTEER_SCOPE_POLICY_JSON`      | Optional complete generated policy override                              |

Every internal queue host must set `VOLUNTEER_COORDINATOR_URL` and
`WORKER_COORDINATOR_SECRET`. Queue claims fail closed unless the shared broker lock can
be acquired; do not run a local queue against a shared issue set without both values.
For a deliberately isolated migration only, set `VOLUNTEER_COORDINATOR_STANDALONE=1` to
disable the shared lock. This override is not safe for normal operation and must never
be used on a shared queue host.

Set secrets in Vercel's encrypted environment configuration, never in the repository or
workflow file. The OAuth client ID and secret are both required: the broker uses the
secret to revoke the short-lived GitHub grant before returning an opaque broker
credential. Neither the GitHub token nor any project secret is returned to workers.
Restrict the broker's Hugging Face token to the staging bucket.

`VOLUNTEER_WORKER_IMAGE_DIGEST` must be updated to the digest printed by the image
workflow after the candidate has passed the physical canary and manual promotion. It is
configured/required provenance carried by the broker lease, not an attestation of the
runtime image. Deploy a new digest for upgrades; never configure a floating image name
or tag. Check that the digest is the expected `linux/amd64` manifest before updating the
environment. Human or independent verification of the running image remains required.
Generate the default scope with `src/scripts/generate_volunteer_scope_policy.py`; use
the JSON override only for an intentional, reviewed deployment policy.

### GitHub queue and review

The coordinator login is assigned to an issue while a language lease is active. The
broker's marker and lease checks are the ownership fence; do not manually edit those
markers or assign a second coordinator to the same request. A worker must only receive
models that are public, ungated, immutable, safetensors-only, free of custom Python or
remote-code configuration, compatible with the pinned EuroEval/Transformers/vLLM stack,
and unambiguously typed as `encoder` or `generative`. The broker and worker enforce
these checks independently; a missing or ambiguous capability is rejected rather than
inferred from a model-family allow-list.

Community submissions go to a private Hugging Face bucket in the EU region and receive
the `community-review-ready` label after broker validation. Redis is only coordination
state and is never a review source. Configure the maintainer shell with
`HF_STAGING_BUCKET`, an `HF_TOKEN` that can read staging and write `HF_RESULTS_BUCKET`
(default `EuroEval/results`), and `VOLUNTEER_PROMOTION_SECRET`. Then use the durable
review commands:

```sh
uv run python src/scripts/review_volunteer_results.py list
uv run python src/scripts/review_volunteer_results.py show <submission-id>
uv run python src/scripts/review_volunteer_results.py \
  --reviewer <github-login> approve <submission-id> --reason "checks passed"
uv run python src/scripts/review_volunteer_results.py \
  --reviewer <github-login> reject <submission-id> --reason "reason"
```

Every review command first re-downloads and validates the manifest and exact result
bytes. It also loads any existing durable decision before requesting a broker-side
reservation, so an expired Redis reservation cannot permit the opposite outcome.

The broker reservation is keyed by issue and submission, authenticated with
`VOLUNTEER_PROMOTION_SECRET`, and binds the outcome and exact identity/digest list. The
command then persists and verifies the immutable decision artifact before any canonical
result upload. Approval checks every canonical destination, uploads only the
submission's records, verifies the resulting metadata and bytes, and completes the
broker transition. Rejection calls the broker with the same validated identity/digest
list so it can safely reclaim reservations. Reservations renew while work is retried and
become durable terminal records after completion. A second reviewer may resume the same
outcome, but an opposite outcome is refused.

A rejected language becomes leaseable again while its rejected submission remains in the
signed issue audit marker. Result identity reservations are reclaimed only after a
token-checked broker transition; private audit manifests and result files remain
available for review. A later local queue claim appends its VM marker and never deletes
this signed history. Acceptance of one language does not award credit. Once every
selected exact language has an accepted submission, the broker adds `results-ready` and
writes the immutable Hall marker for the contributor with the largest sum of
server-derived accepted result counts. Lower-case GitHub login order breaks ties.

### Image publishing workflow

`.github/workflows/worker-image.yaml` builds only `linux/amd64` and enables BuildKit
layer caching. Pull requests do not publish or attest an image: Docker loads the local
image and smokes its normal entrypoint as UID 10001 without a GPU. Trusted runs publish
only the immutable commit-SHA candidate, with provenance and an SBOM. The workflow
checks that the GHCR package is public, pulls the exact candidate digest anonymously,
and prints the verified immutable reference. It never creates or updates `latest`.

Visibility and the first promotion are post-merge deployment steps. A new GHCR package
is private by default, so after the first post-merge candidate publication an
organization or package owner must make `euroeval-worker` public in GitHub Packages
settings and rerun the workflow. GitHub does not provide a supported REST operation for
this visibility change. Every candidate must pass the workflow's public-package and
anonymous exact-digest checks before an operator considers it for promotion.

`latest` appears only after an operator selects a verified candidate, runs the required
physical Linux `amd64` NVIDIA canary, and manually promotes that exact digest. Do not
promote a candidate before the canary succeeds. Run the following canary on the physical
Linux `amd64` NVIDIA host, using the immutable reference printed by the workflow:

```sh
set -eu
IMAGE=ghcr.io/euroeval/euroeval-worker
DIGEST=sha256:REPLACE_WITH_VERIFIED_DIGEST
CANDIDATE="$IMAGE@$DIGEST"
test "$(uname -s)" = Linux
test "$(uname -m)" = x86_64
test "$(docker info --format '{{.Architecture}}')" = x86_64
nvidia-smi
docker pull "$CANDIDATE"
docker run --rm --gpus all --pull=never "$CANDIDATE" --gpu-health-check
```

After the canary succeeds, promote the same digest with a GitHub PAT that has
`write:packages`. The block logs in only for the tag write, then logs out and uses a
fresh Docker config for the anonymous verification:

```sh
set -euo pipefail
IMAGE=ghcr.io/euroeval/euroeval-worker
DIGEST=sha256:REPLACE_WITH_VERIFIED_DIGEST
CANDIDATE="$IMAGE@$DIGEST"
DOCKER_CONFIG="$(mktemp -d)"
export DOCKER_CONFIG
trap 'docker logout ghcr.io >/dev/null 2>&1 || true; rm -rf "$DOCKER_CONFIG"' EXIT
read -r -p "GitHub username: " GHCR_USER
read -r -s -p "GHCR PAT (write:packages): " GHCR_PAT
printf '\n'
printf '%s' "$GHCR_PAT" | docker login ghcr.io \
  --username "$GHCR_USER" --password-stdin
unset GHCR_PAT
docker buildx imagetools create --tag "$IMAGE:latest" "$CANDIDATE"
docker logout ghcr.io >/dev/null
rm -rf "$DOCKER_CONFIG"
mkdir -p "$DOCKER_CONFIG"
docker image rm "$IMAGE:latest" >/dev/null 2>&1 || true
docker pull "$IMAGE:latest"
repo_digests="$(docker image inspect "$IMAGE:latest" \
  --format '{{range .RepoDigests}}{{println .}}{{end}}')"
printf '%s\n' "$repo_digests" | grep -Fxq "$CANDIDATE"
echo "Verified latest: $CANDIDATE"
```

Set `VOLUNTEER_WORKER_IMAGE_DIGEST` to the promoted digest only after this anonymous
verification. If a build fails, do not switch to a mutable base image or install a host
driver in the image. Check the pinned CUDA base, the locked `uv.lock` dependencies, and
the NVIDIA runner/toolkit separately.
