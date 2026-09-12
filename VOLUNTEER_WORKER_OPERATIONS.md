# EuroEval volunteer worker operations

This guide is for EuroEval maintainers and infrastructure operators. Volunteers
should use the [volunteer worker guide](VOLUNTEER_WORKER.md), which must not require
any of the operator credentials described here.

> **The broker API is already in this project.** The `api/worker/*.ts` files are
> Vercel Functions in the existing EuroEval Vercel project. No separately maintained
> API server is needed. When that project is deployed, Vercel automatically exposes
> those files as `/api/worker/...` routes. Vercel does **not** provision or configure
> GitHub OAuth, Upstash Redis, Hugging Face storage, GHCR, or their secrets.

Vercel Git integration can deploy commits automatically if it is enabled for the
project. Otherwise, deploy the existing project with `make frontend`. In either case,
external services and Vercel environment variables must be configured explicitly.

## Architecture and components

The volunteer path consists of these components:

- **EuroEval Vercel project:** The frontend and the edge broker functions under
  `api/worker/`. Vercel's `api/` convention maps each file to its matching HTTP route.
  The broker authenticates workers, finds GitHub issues, fences claims, validates
  results, and writes staged objects.
- **GitHub:** `EuroEval/EuroEval` is the work queue and audit log. Request issues are
  claimed by a coordinator account. Signed markers in issue bodies, labels, comments,
  and the broker lease protect ownership and review transitions.
- **GitHub OAuth device flow:** A volunteer authorises the OAuth app in a browser. The
  broker exchanges and immediately revokes the short-lived GitHub grant, then gives
  the worker an opaque, time-limited broker credential. GitHub and project secrets are
  never sent to the worker.
- **Upstash Redis:** The broker's short-lived state store for credentials, leases,
  rate limits, mutexes, result reservations, and promotion reservations. Redis is
  coordination state, not a review or result source.
- **Hugging Face Buckets:** The private EU staging bucket receives worker result files
  and manifests. After review, the maintainer tool copies verified records to the
  public canonical `EuroEval/results` bucket.
- **GHCR:** `ghcr.io/euroeval/euroeval-worker` is the public, immutable worker image.
  Workers use a digest selected by the maintainer; they do not receive a floating tag
  as their broker lease's image identity.
- **Maintainer tools:**
  `src/scripts/review_volunteer_results.py` independently validates staged evidence,
  records a durable decision, and calls the broker's promotion transition.
- **Legacy/local queue:** `src/scripts/process_evaluation_queue.py` may still process
  non-volunteer work. Its issue claims must use the same broker coordinator mutex as
  volunteer claims.

The normal data flow is worker authentication, issue claim, evaluation, one result
upload per identity, finalisation into a private manifest, maintainer review, and
promotion or rejection. Acceptance copies only verified records to the canonical
bucket. A complete set of accepted languages adds `results-ready` and contributor
credit; an individual accepted language does not.

## One-time prerequisites

Complete these steps before enabling the first volunteer worker. Values in examples
are placeholders; do not put real secrets in this file, Git, a workflow, or a worker
host.

### 1. Link the existing Vercel project

Confirm that the production Vercel project is the existing EuroEval project and that
its production domain is `https://euroeval.com`. Link the repository to that project
in Vercel. Do not create a second API project: the broker functions and frontend are
deployed together.

Choose one deployment method:

- Enable Vercel Git integration if automatic deployment of the intended branch is
  desired, and verify its production branch and checks.
- Otherwise install the Vercel CLI and use `make frontend` from a clean checkout.

Neither method creates external accounts or supplies environment variables.

### 2. Create the GitHub OAuth App

Create or select a GitHub OAuth App that is allowed to use the device flow. In the
app settings, enable **Device flow**. Record its client ID and client secret in a
password manager, then place them in the Vercel production environment as described
below. The broker requests the `read:user` scope; it does not need a volunteer's
repository token. The device flow has no callback URL to configure for this broker.

Keep the client secret available: the broker uses it to revoke every device grant
before issuing a broker credential. If the secret is missing or revocation fails,
no worker credential should be issued.

### 3. Prepare the maintainer GitHub access and queue labels

Create a maintainer-owned GitHub token with the least privilege that permits the
broker to read and update issues, assignees, labels, and comments in
`EuroEval/EuroEval`. Store it only in Vercel's encrypted environment variables. The
`WORKER_COORDINATOR_LOGIN` value is the GitHub account the broker temporarily assigns
to active request issues; that account must be a repository collaborator with issue
assignment permission. It may be a bot or a maintainer account, but it must be
consistent everywhere.

Create these labels in `EuroEval/EuroEval` with these exact names:

- `model evaluation request` — required for the broker to discover open requests.
- `community-review-ready` — added when a volunteer manifest is ready for review.
- `results-ready` — added when all selected languages have accepted submissions.

If the legacy/local queue will remain active, also retain its labels
`evaluation-failed` and `gated`. Do not manually remove signed volunteer markers,
change their contents, or assign a second coordinator to an active request.

### 4. Create Upstash Redis

Create the Redis database used only by this EuroEval broker, preferably in the region
closest to the Vercel deployment. Copy its REST URL and REST token into Vercel. Do
not use a Redis URL or token from a different application, and do not expose either
value to volunteers. Losing Redis loses active coordination state; it must not be
used as the durable review record.

### 5. Create private Hugging Face staging storage

Create a private Hugging Face Bucket in the EU region, for example
`<namespace>/<private-staging-bucket>`, and set that exact ID as `HF_STAGING_BUCKET`.
The broker refuses a staging bucket that is not private. Create a narrowly scoped
HF token that can write this staging bucket and use it as Vercel's `HF_TOKEN`.

For maintainer review, use a token that can read the staging bucket and write the
canonical results bucket (`EuroEval/results`). This can be a separately managed
local token; do not grant the Vercel function write access to the public results
bucket unless there is a separately reviewed reason to do so. Never paste a token in
this guide or commit it to a `.env` file.

### 6. Generate and align the scope policy

The checked-in `api/worker/scope-policy.json` is generated from the official dataset
and language contracts. Generate it for the exact EuroEval release that the broker
will advertise:

```sh
uv run python src/scripts/generate_volunteer_scope_policy.py \
  --version <euroeval-version>
git diff --check
```

Commit the generated JSON with the release change. Its policy version must match
`EUROEVAL_VERSION` in Vercel (including the repository's normal development-version
normalisation), and the worker package's `VOLUNTEER_WORKER_VERSION` must match the
worker image/package being published. The policy controls exact result identities;
do not hand-edit it. `VOLUNTEER_SCOPE_POLICY_JSON` is an optional complete override
for an intentional, reviewed deployment policy, not a way to patch one entry.

### 7. Publish a public GHCR package

The first trusted workflow run may create the
`ghcr.io/euroeval/euroeval-worker` package privately. A package or organisation owner
must make `euroeval-worker` public in GitHub Packages settings, then rerun the
workflow. GitHub does not provide a supported REST operation for this visibility
change. The workflow's anonymous pull check must pass before a volunteer is given an
image reference.

## Vercel environment variables

Set these in the existing Vercel project under **Settings → Environment Variables**.
The required list below means the **Production** environment used by
`https://euroeval.com`; add the same values to Preview only if Preview deployments
are intentionally used for broker testing. Do not mark any of these as public or
expose them as frontend variables.

### Required in Vercel Production

- `GITHUB_TOKEN` — maintainer GitHub token for issue reads, labels, comments, and
  assignment in `EuroEval/EuroEval`.
- `GITHUB_OAUTH_CLIENT_ID` — GitHub OAuth App client ID with device flow enabled.
- `GITHUB_OAUTH_CLIENT_SECRET` — OAuth App secret used to revoke device grants.
- `EUROEVAL_VERSION` — exact release represented by the generated scope policy.
- `VOLUNTEER_WORKER_VERSION` — exact supported worker protocol/package version.
- `VOLUNTEER_WORKER_IMAGE_DIGEST` — promoted `linux/amd64` image digest in the form
  `sha256:...`; this is configured lease provenance, not runtime attestation.
- `VOLUNTEER_MARKER_SECRET` — random HMAC-SHA-256 secret for signed issue state and
  Hall credit markers.
- `WORKER_COORDINATOR_LOGIN` — assignable GitHub login used for temporary issue
  assignment.
- `UPSTASH_REDIS_REST_URL` — REST URL for the broker's Upstash database.
- `UPSTASH_REDIS_REST_TOKEN` — REST token for that database.
- `HF_STAGING_BUCKET` — private EU staging bucket ID in `namespace/bucket` form.
- `HF_TOKEN` — scoped token that can write the private staging bucket.
- `WORKER_COORDINATOR_SECRET` — secret accepted by the coordinator lock, renew, and
  release endpoints. It must also be set on every local queue host.
- `VOLUNTEER_PROMOTION_SECRET` — secret accepted by promotion reservation and
  transition endpoints. It must also be set in the maintainer review shell.

Generate long random secrets outside the repository and rotate them as a coordinated
change: update Vercel and every dependent local host together. A rotation invalidates
old coordinator or promotion requests; do not print these values in logs.

### Optional in Vercel Production

These have safe code defaults and should be set only when the default is deliberately
changed:

- `VOLUNTEER_LEASE_SECONDS` — bounded lease duration; the default is 1,800 seconds.
- `VOLUNTEER_SCOPE_POLICY_JSON` — complete JSON policy override; normally leave unset
  so the checked-in generated policy is used.
- `COMMUNITY_REVIEW_LABEL` — review label override; default
  `community-review-ready`.
- `RESULTS_READY_LABEL` — completion label override; default `results-ready`.
- `COMMUNITY_MAINTAINER_LOGIN` — GitHub login mentioned in the finalisation comment;
  default `saattrupdan`.

`VOLUNTEER_COORDINATOR_URL`, `HF_RESULTS_BUCKET`,
`VOLUNTEER_BROKER_RESERVATION_URL`, and `VOLUNTEER_BROKER_PROMOTION_URL` are **not
Vercel variables**. They belong to the local queue or maintainer review shell and are
listed below. `VOLUNTEER_COORDINATOR_STANDALONE` is also local-only and unsafe for
normal shared operation.

## Recommended deployment order

1. Confirm the repository, production Vercel project, production domain, GitHub
   labels, coordinator login, OAuth device flow, Upstash database, private staging
   bucket, and scoped tokens.
2. Generate and commit `api/worker/scope-policy.json`. Confirm its version, the
   EuroEval package version, and the intended worker package/image version agree.
3. Build and publish the immutable GHCR commit-SHA candidate. Verify that it is
   public and that its exact digest can be pulled anonymously.
4. Run the physical GPU canary below, promote that same digest to `latest`, and
   anonymously verify that `latest` resolves to the same digest.
5. Add or update the Vercel Production variables, especially the promoted image
   digest. Never point `VOLUNTEER_WORKER_IMAGE_DIGEST` at a tag.
6. Deploy the existing Vercel project through enabled Git integration or with
   `make frontend`. Verify the deployment URL and production domain are the same
   application; deployment alone does not validate external services.
7. Run the safe route and configuration smoke tests below. Then configure the local
   queue host with the shared coordinator URL and secret before allowing it to claim
   issues.
8. Ask one volunteer to complete device authentication and run a controlled
   `--once` evaluation. Watch the issue marker, staging manifest, and logs before
   inviting more workers.

## Safe endpoint smoke tests

All broker functions accept `POST` only. `OPTIONS` is a harmless CORS/preflight-style
probe and returns `204` with `{}`; `GET` must return `405` with
`{"error":"Method not allowed"}`. Test the deployed domain without credentials:

```sh
set -eu
BASE=https://euroeval.com/api/worker
for route in \
  auth/start auth/poll claim heartbeat result finalise release \
  coordinator-lock coordinator-renew coordinator-release \
  promotion-lock promotion-reserve promote
  do
    code="$(curl -sS -o /tmp/euroeval-worker-smoke-body \
      -w '%{http_code}' "$BASE/$route")"
    test "$code" = 405 || {
      echo "$route returned $code, expected 405" >&2
      exit 1
    }
  done
echo "All worker routes reject GET as expected"
```

This checks route discovery and the method guard without creating a device flow,
lease, result, lock, or promotion. To check protected configuration without sending a
secret or mutating state, submit a valid protocol envelope but no authentication:

```sh
set -eu
BASE=https://euroeval.com/api/worker
BODY='{"protocol_version":"volunteer-worker/v1","issue_number":1}'
for route in coordinator-lock promotion-lock; do
  curl -sS -i -X POST "$BASE/$route" \
    -H 'content-type: application/json' \
    --data "$BODY"
done
```

For each protected route, the expected result is one of these:

- `503` with an error naming the missing `WORKER_COORDINATOR_SECRET` or
  `VOLUNTEER_PROMOTION_SECRET`: the route is deployed, but that Vercel variable is
  missing.
- `401` authentication failure: the variable exists and the deliberately absent or
  invalid secret was rejected. This is the expected safe result from the probe.
- `400` after valid authentication but malformed request data: the route and secret
  are working; do not use real credentials in a smoke-test script.

A `POST` to `claim` without `Authorization: Bearer <credential>` must similarly return
`401`, without touching GitHub or claiming work. Do **not** use `auth/start` as a
routine smoke test: a valid request starts a GitHub device grant and consumes Redis
state. Use a real worker for that one end-to-end test.

The live endpoint contract is:

- `POST /auth/start` and `POST /auth/poll` — start and poll the GitHub device flow;
  poll returns `202` while pending and `200` with a broker credential when authorised.
- `POST /claim` — bearer credential plus hardware report; returns `200` with a lease
  or `200` with `status: "no_work"`.
- `POST /heartbeat` — bearer credential and `lease_id`; returns `200` with a renewed
  expiry.
- `POST /result` — bearer credential and exact lease-bound record; returns `201` for
  an upload or `200` with `status: "duplicate"` for an idempotent repeat.
- `POST /finalise` — bearer credential and `lease_id`; returns `200` with
  `status: "ready"` and a submission ID, or `already_finalised` on a safe retry.
- `POST /release` — bearer credential and `lease_id`; returns `200` with
  `status: "released"`.
- `POST /coordinator-lock`, `/coordinator-renew`, and `/coordinator-release` — the
  `x-coordinator-secret` and protocol body; acquire returns an opaque token and its
  expiry, renew returns `status: "renewed"`, and release returns `status: "released"`.
- `POST /promotion-lock` (alias `/promotion-reserve`) —
  `x-promotion-secret`; creates or resumes a fenced review reservation.
- `POST /promote` — `x-promotion-secret` plus a matching reservation, decision digest,
  and evidence; returns the accepted or rejected terminal transition.

Errors use JSON with `protocol_version` where the endpoint has accepted the protocol,
and fail closed on missing configuration, invalid credentials, expired leases, or
fence conflicts. A `502`/`503` after valid authentication generally means to inspect
Vercel logs and the corresponding external dependency rather than retrying blindly.

## Legacy/local queue coordination

If `src/scripts/process_evaluation_queue.py` is running, it and the Vercel broker can
see the same GitHub issues. The queue must acquire the broker's same Redis-backed
issue mutex before claiming or changing an issue. On **every** shared queue host set:

```sh
export VOLUNTEER_COORDINATOR_URL=https://euroeval.com/api/worker
export WORKER_COORDINATOR_SECRET='<same value as Vercel, supplied out of band>'
```

The URL is the broker base URL, not a separately deployed server. The queue appends
`/coordinator-lock`, `/coordinator-renew`, and `/coordinator-release` itself. The
secret must match the Vercel `WORKER_COORDINATOR_SECRET` exactly. Keep it out of shell
history where practical and out of queue logs.

The queue fails closed if either value is absent. Do not set
`VOLUNTEER_COORDINATOR_STANDALONE=1` during normal operation. That local-only escape
hatch disables the shared lock and is permitted only for a deliberately isolated
migration after verifying that no broker, volunteer worker, or second queue can touch
the issue set. Never use it on a shared queue host.

The local queue also needs its existing `GITHUB_TOKEN` and Hugging Face access for its
non-volunteer work. Keep its results-writing credentials separate from the broker's
staging-only Vercel token where possible.

## Maintainer review and promotion

Run these commands from a checked-out EuroEval repository on a maintainer-controlled
host. The review shell needs a token that can read `HF_STAGING_BUCKET` and write
`HF_RESULTS_BUCKET` (default `EuroEval/results`), plus the exact promotion secret:

```sh
export HF_STAGING_BUCKET='<private namespace/staging-bucket>'
export HF_RESULTS_BUCKET=EuroEval/results
export HF_TOKEN='<scoped maintainer token>'
export VOLUNTEER_PROMOTION_SECRET='<same value as Vercel, supplied out of band>'
```

Listing and showing are non-mutating validation operations:

```sh
uv run python src/scripts/review_volunteer_results.py list
uv run python src/scripts/review_volunteer_results.py show <submission-id>
```

Approve or reject using the maintainer's lowercase GitHub login and a useful reason:

```sh
uv run python src/scripts/review_volunteer_results.py \
  --reviewer <github-login> approve <submission-id> \
  --reason "checks passed"
uv run python src/scripts/review_volunteer_results.py \
  --reviewer <github-login> reject <submission-id> \
  --reason "reason"
```

Each decision re-downloads and validates the manifest, exact result bytes, model and
language identity, and generated scope. Approval checks canonical destination
collisions, uploads only that submission's records, verifies every upload, and then
completes the broker transition. Rejection calls the same fenced transition and
releases reservations. The decision artifact is durable in staging, so an expired
Redis reservation cannot be reused for the opposite outcome. A retried reviewer may
resume the same outcome; an opposite outcome is refused.

Do not manually copy staged files, edit issue markers, delete audit manifests, or
award credit. A rejected language becomes leaseable again while its signed audit
history remains. When all selected languages are accepted, the broker adds
`results-ready` and records the contributor with the largest server-derived accepted
result count; lowercase GitHub login order breaks ties.

## Image publishing and canary

This is a per-release process, not a one-time prerequisite. The
`.github/workflows/worker-image.yaml` workflow builds only `linux/amd64` with BuildKit
layer caching. Pull requests do not publish or attest an image: they load the local
image and smoke its normal entrypoint as UID 10001 without a GPU. Trusted runs publish
only the immutable commit-SHA candidate with provenance and an SBOM, verify the public
GHCR package, pull the exact candidate anonymously, and print the digest. The workflow
never creates or updates `latest`.

After the package is public and the workflow has printed a verified digest, run this
on a physical Linux `amd64` NVIDIA host:

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

Only after that canary succeeds, promote the same digest with a GitHub PAT having
`write:packages`. The PAT is read interactively and the Docker configuration is
removed before anonymous verification:

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

Set `VOLUNTEER_WORKER_IMAGE_DIGEST` to this promoted digest only after anonymous
verification. Never configure a floating image name or tag. If a build fails, do not
switch to a mutable base image or install a host driver in the image. Check the
pinned CUDA base, locked `uv.lock` dependencies, and NVIDIA runner/toolkit separately.

## Ongoing maintenance and monitoring

### Per release

- Update the EuroEval package and worker protocol/image version together.
- Regenerate and review `api/worker/scope-policy.json`; ensure its version equals
  `EUROEVAL_VERSION` before deployment.
- Run the API tests (`npm run test:api`) and the relevant Python tests/checks.
- Publish, anonymously verify, canary, and manually promote the new immutable image.
- Update `VOLUNTEER_WORKER_IMAGE_DIGEST`, version, and policy variables in Vercel,
  then deploy and repeat the safe route probes.
- Keep the old image digest available for rollback, but do not leave the broker
  configured with an unpromoted or mutable reference.

### Recurring operations

- Watch Vercel function logs and error rates for `401`, `409`, `429`, `502`, and `503`.
  Investigate configuration and dependency failures; do not weaken fences or bypass
  authentication to clear them.
- Check Upstash availability, latency, rate-limit growth, and unexpected key growth.
  Redis outages should fail closed rather than permit duplicate claims.
- Review `community-review-ready` issues and staged manifests regularly. Use the
  maintainer review commands, not Redis contents, as the source of review truth.
- Check that the staging bucket remains private, tokens remain scoped, and promoted
  canonical records retain their expected metadata and bytes.
- Confirm the local queue still uses
  `VOLUNTEER_COORDINATOR_URL=https://euroeval.com/api/worker` and the matching secret.
  Remove retired queue hosts rather than running an uncoordinated copy.
- Rotate OAuth, GitHub, HF, Redis, coordinator, and promotion credentials on the
  organisation's normal schedule. Coordinate rotations, verify safe probes, and
  revoke old credentials.
- Re-run the physical GPU canary after changes to the CUDA base, locked dependencies,
  NVIDIA toolkit, or host driver. Human or independent verification of the running
  image remains required: the configured digest is provenance carried by a lease, not
  runtime attestation.
