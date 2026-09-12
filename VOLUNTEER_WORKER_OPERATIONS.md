# EuroEval volunteer worker operations

This guide is for EuroEval maintainers and infrastructure operators. Volunteers should
use the [volunteer worker guide](VOLUNTEER_WORKER.md), which must not require any of the
operator credentials described here.

> **The broker API is already in this project.** The `api/worker/*.ts` files are Vercel
> Functions in the existing EuroEval Vercel project. No separately maintained API server
> is needed: Vercel automatically routes those files to matching `/api/worker/...`
> endpoints when the project is deployed. Vercel does **not** provision or configure
> GitHub OAuth, Upstash Redis, Hugging Face storage, GHCR, or their secrets; those
> external services and all Vercel environment variables remain manual.

A clean Git checkout is not currently sufficient for a production deployment. The
leaderboard CSV assets under the ignored `src/frontend/csv/` directory must exist when
Vite builds. Use the leaderboard collection/generation deployment flow, which generates
and validates those assets before its prebuilt Vercel deploy. Alternatively, generate
the assets first in the checkout and then run `make frontend`. Do not rely on a clean
Git-based Vercel deployment to recreate these ignored assets.

## First-time procedure

Run this exact sequence from a clean checkout. The first command is a local,
non-network plan and the second is a read-only network check. Both are safe for
Pi/I to execute now; neither imports `.env` or mutates a file.

1. **AUTOMATED CHECK** - print the workflow, defaults, and required variable names:

   ```sh
   uv run python src/scripts/volunteer_worker_operations.py plan
   ```

2. **AUTOMATED CHECK** - run read-only local and configured-service diagnostics. Supply
   credentials through the maintainer's normal secret injection; never put values in
   the command or guide:

   ```sh
   make volunteer-worker-check
   ```

3. **MANUAL** - the maintainer chooses the existing `EuroEval/EuroEval` GitHub queue,
   `https://euroeval.com` Vercel project, recommended private EU staging bucket, the
   existing Vercel KV database used by the site submission limiter, OAuth app, and
   public GHCR package. A US staging bucket requires the explicit data-residency decision
   described below. The maintainer supplies credentials, selects an immutable image
   digest, and confirms production deployment. Pi/I can execute publication or
   deployment only after explicit approval, but cannot choose accounts or credentials,
   or perform a physical GPU canary without hardware.

4. **CONFIRMED AUTOMATION** - after reviewing step 2 and supplying the chosen values,
   explicitly apply only the selected scoped setup. There is intentionally no implicit
   all-components apply command:

   ```sh
   uv run python src/scripts/volunteer_worker_operations.py apply --github --yes
   uv run python src/scripts/volunteer_worker_operations.py apply --hf --yes \
     --hf-region eu
   # Use this only after an explicit US data-residency decision:
   uv run python src/scripts/volunteer_worker_operations.py apply --hf --yes \
     --hf-region us
   uv run python src/scripts/volunteer_worker_operations.py apply --reuse-vercel-kv --yes
   ```

   The GitHub and HF commands are optional when those components already pass. For the
   selected KV reuse setup, configure the remaining Vercel variables as described below;
   the separate command above supplies only the two broker aliases. The regular Vercel
   apply requires every value except versions, which are derived from source when
   absent.
   Marker, coordinator, and promotion secrets are durable: provide existing values; this
   automation never generates or rotates them. The KV reuse command copies the existing
   production KV values to the broker aliases without displaying them.

5. **MANUAL** - publish the candidate image using the workflow, carry its digest from
   the non-secret job summary/artifact, run the physical Linux `amd64` NVIDIA GPU
   canary, and manually promote that exact digest. Pi/I cannot perform a physical GPU
   canary without hardware. Publication and production deployment require explicit
   approval.

6. **AUTOMATED CHECK** - after deployment, run safe route probes. They issue only GET,
   OPTIONS, and unauthenticated POST requests; they never start authentication, claim
   work, acquire a lock, or touch staging:

   ```sh
   uv run python src/scripts/volunteer_worker_operations.py smoke
   ```

7. **MANUAL** - configure the local queue coordinator and ask one volunteer to run a
   controlled worker canary. Review the issue marker, staging manifest, and logs before
   allowing more workers. Use the later sections for detailed contracts and review.

## Architecture and components

The volunteer path consists of these components:

- **EuroEval Vercel project:** The frontend and the edge broker functions under
  `api/worker/`. Vercel's `api/` convention maps each file to its matching HTTP route.
  The broker authenticates workers, finds GitHub issues, fences claims, validates
  results, and writes staged objects.
- **GitHub:** `EuroEval/EuroEval` is the work queue and audit log. Issue assignees are
  the mutable, authoritative active evaluators and credit identities for both manual
  and volunteer paths. Signed markers in issue bodies are audit and integrity
  evidence; they do not override current assignees.
- **GitHub OAuth device flow:** A volunteer authorises the OAuth app in a browser. The
  broker exchanges and immediately revokes the short-lived GitHub grant, then gives the
  worker an opaque, time-limited broker credential. GitHub and project secrets are never
  sent to the worker.
- **Upstash Redis:** The broker's short-lived state store for credentials, leases, rate
  limits, mutexes, result reservations, and promotion reservations. Redis is
  coordination state, not a review or result source. The site's submission rate limiter
  and the volunteer broker use the same existing Vercel KV database under disjoint
  prefixes: `euroeval:submit` for submissions and `euroeval:worker` (and related worker
  prefixes) for broker state. This couples their outages, credential rotation, and
  flushes; this selected setup does not need a dedicated Upstash resource.
- **Hugging Face Buckets:** The private staging bucket (EU recommended) receives worker
  result files and manifests. After review, the maintainer tool copies verified records
  to the public canonical `EuroEval/results` bucket.
- **GHCR:** `ghcr.io/euroeval/euroeval-worker` is the public, immutable worker image.
  Workers use a digest selected by the maintainer; they do not receive a floating tag as
  their broker lease's image identity.
- **Maintainer tools:** `src/scripts/review_volunteer_results.py` independently
  validates staged evidence, records a durable decision, and calls the broker's
  promotion transition.
- **Legacy/local queue:** `src/scripts/process_evaluation_queue.py` may still process
  non-volunteer work. Its issue claims must use the same broker coordinator mutex as
  volunteer claims.

The normal data flow is worker authentication, issue claim, evaluation, one result
upload per identity, finalisation into a private manifest, maintainer review, and
promotion or rejection. Acceptance copies only verified records to the canonical bucket.
Issue assignees are the mutable, authoritative active evaluators and credit
identities for manual and volunteer evaluations. A request may have multiple assignees
when multiple volunteers contribute different languages; retain every assignee whose
submission is accepted. Maintainers may transfer ownership or credit by changing the
assignees, which fences old volunteer leases. Remove an assignee only when their work is
rejected or intentionally released. Signed marker contributors are retained as audit and
integrity evidence only. A complete set of accepted languages adds `results-ready`;
credit follows the accepted assignees, while an individual accepted language does not.

## One-time prerequisites

Complete these steps before enabling the first volunteer worker. Values in examples are
placeholders; do not put real secrets in this file, Git, a workflow, or a worker host.

### 1. Link the existing Vercel project

Confirm that the production Vercel project is the existing EuroEval project and that its
production domain is `https://euroeval.com`. Before any CLI deployment, authenticate
with Vercel and run `vercel link` from this repository, selecting that existing project
and the correct team/scope. Verify that the linked project is the one serving the
production domain. Do not create a second API project: the broker functions and frontend
are deployed together.

Choose one supported production flow:

- Prefer the leaderboard collection/generation flow (`make leaderboards`, or
  `make force-leaderboards` when regeneration is needed despite no new results). It
  regenerates the ignored CSV assets, validates them, and deploys a prebuilt frontend.
- If the assets already exist in the checkout, run `make frontend` to build and deploy
  the prebuilt frontend. This is not safe from a clean checkout until the leaderboard
  assets have been generated.
- Vercel Git integration may deploy the intended branch only when its build environment
  is also supplied with the generated assets. A clean Git-based deployment is not
  self-sufficient while `src/frontend/csv/` remains ignored; verify the production
  branch and checks rather than treating Git integration as the leaderboard flow.

Linking associates this repository with the existing Vercel project; it does not create
external accounts or supply environment variables. Configure those manually below.

### 2. Create the GitHub OAuth App

Create or select a GitHub OAuth App that is allowed to use the device flow. In the app
settings, enable **Device flow** and set the callback URL to the safe required value
`https://euroeval.com`. The device flow does not use that callback URL. Store the client
ID and client secret in a password manager, then inject them into the Vercel production
environment as described below. The broker requests the `read:user` scope; it does not
need a volunteer's repository token.

Keep the client secret available: the broker uses it to revoke every device grant before
issuing a broker credential. If the secret is missing or revocation fails, no worker
credential should be issued.

### 3. Prepare the maintainer GitHub access and queue labels

Create a maintainer-owned GitHub token with the least privilege that permits the broker
to read and update issues, assignees, labels, and comments in `EuroEval/EuroEval`. Store
it only in Vercel's encrypted environment variables. The broker uses this token to read
and update issues, assignees, labels, and comments. Every volunteer OAuth login must be
GitHub-assignable in this repository; otherwise claim stops with an actionable error and
the volunteer must use an eligible account.

Create these labels in `EuroEval/EuroEval` with these exact names:

- `model evaluation request` — required for the broker to discover open requests.
- `community-review-ready` — added when a volunteer manifest is ready for review.
- `results-ready` — added when all selected languages have accepted submissions.

If the legacy/local queue will remain active, also retain its labels `evaluation-failed`
and `gated`. Maintainers transfer ownership or credit by changing issue assignees.
Changing assignees fences old volunteer leases. Do not treat signed marker contributors
as current ownership; they are retained for audit and integrity checks.

### 4. Reuse the existing Vercel KV database

The site submission rate limiter and volunteer broker intentionally share the existing
Vercel KV database. Their keys remain disjoint: submissions use `euroeval:submit`, while
the broker uses `euroeval:worker` and related worker prefixes. This couples outages,
credential rotation, and flushes, so treat those operations as affecting both systems.
No dedicated Upstash resource is needed for this selected setup.

After the project link and the source Vercel KV variables are confirmed, run this exact
command. It verifies the linked project, reads `KV_REST_API_URL` and
`KV_REST_API_TOKEN` through Vercel's production environment, and adds only the two
broker aliases without exposing their values:

```sh
uv run python src/scripts/volunteer_worker_operations.py apply --reuse-vercel-kv --yes
```

If this operation must be rolled back, remove only the aliases. Never remove the source
KV variables, which the site uses:

```sh
vercel env rm UPSTASH_REDIS_REST_URL production --yes
vercel env rm UPSTASH_REDIS_REST_TOKEN production --yes
```

The operation is safe to repeat. It does not create a database, rotate credentials, or
modify `KV_REST_API_URL` or `KV_REST_API_TOKEN`.

### 5. Create private Hugging Face staging storage

The recommended staging location is a private Hugging Face Bucket in the EU region,
for example `<namespace>/<private-staging-bucket>`, with that exact ID set as
`HF_STAGING_BUCKET`. EU bucket creation requires an eligible Hugging Face organisation
plan (Team or Enterprise); EuroEval does not currently have that eligibility. If EU
creation is unavailable, the US region is an explicit data-residency decision, not an
automatic fallback. Use `--hf-region us` only after approving that decision.

The operations CLI defaults to EU and passes the selected region only when creating a
missing bucket. Existing bucket metadata cannot verify region, so confirm an existing
bucket's region manually in Hugging Face. The broker refuses a staging bucket that is
not private. Create a narrowly scoped HF token that can write this staging bucket and
use it as Vercel's `HF_TOKEN`.

For maintainer review, use a token that can read **and write** the staging bucket and,
for approval, also write the canonical results bucket (`EuroEval/results`). This can be
a separately managed local token; do not grant the Vercel function write access to the
public results bucket unless there is a separately reviewed reason to do so. Never paste
a token in this guide or commit it to a `.env` file.

### 6. Generate and align the scope policy

The checked-in `api/worker/scope-policy.json` is generated from the official dataset and
language contracts. Generate it for the exact EuroEval release that the broker will
advertise:

```sh
uv run python src/scripts/generate_volunteer_scope_policy.py \
  --version <euroeval-version>
git diff --check
```

Use `--check` to verify this file without writing, or `--dry-run` to preview whether it
would change. Commit the generated JSON with the release change. Keep the EuroEval
release and policy versions aligned separately: the policy's `euroeval_version` must
match the repository's EuroEval package version and
`EUROEVAL_VERSION` in Vercel (including the repository's
normal development-version normalisation). `VOLUNTEER_WORKER_VERSION` is independent
worker protocol/package versioning; it is currently `1.0.0` and must match the worker
image/package being published, but need not equal the EuroEval version. The policy
controls exact result identities; do not hand-edit it. `VOLUNTEER_SCOPE_POLICY_JSON` is
an optional complete override for an intentional, reviewed deployment policy, not a way
to patch one entry. If Vercel sets this override, the review shell must use the
identical policy JSON override.

### 7. Publish a public GHCR package

The first trusted workflow run may create the `ghcr.io/euroeval/euroeval-worker` package
privately. A package or organisation owner must make `euroeval-worker` public in GitHub
Packages settings, then rerun the workflow. GitHub does not provide a supported REST
operation for this visibility change. The workflow's anonymous manifest inspection must
pass before a volunteer is given an image reference.

## Vercel environment variables

Set these in the existing Vercel project under **Settings → Environment Variables**. The
required list below means the **Production** environment used by `https://euroeval.com`;
add the same values to Preview only if Preview deployments are intentionally used for
broker testing. Do not mark any of these as public or expose them as frontend variables.

### Required in Vercel Production

- `GITHUB_TOKEN` — maintainer GitHub token for issue reads, labels, comments, and
  assignment in `EuroEval/EuroEval`.
- `GITHUB_OAUTH_CLIENT_ID` — GitHub OAuth App client ID with device flow enabled.
- `GITHUB_OAUTH_CLIENT_SECRET` — OAuth App secret used to revoke device grants.
- `EUROEVAL_VERSION` — exact release represented by the generated scope policy.
- `VOLUNTEER_WORKER_VERSION` — exact supported worker protocol/package version.
- `VOLUNTEER_WORKER_IMAGE_DIGEST` — promoted `linux/amd64` image digest in the form
  `sha256:...`; this is configured lease provenance, not runtime attestation.
- `VOLUNTEER_MARKER_SECRET` — long-lived HMAC-SHA-256 secret for signed issue markers.
  Do **not** rotate this routinely: changing it invalidates existing signed markers. A
  change requires a planned migration that re-signs or otherwise migrates every live
  marker before the old secret is retired, with the transition tested and verified.
- `UPSTASH_REDIS_REST_URL` — Production alias for the existing Vercel KV REST URL.
- `UPSTASH_REDIS_REST_TOKEN` — Production alias for the existing Vercel KV REST token.
  Set both with the explicit reuse command in the first-time procedure; it does not
  display or rotate the `KV_REST_API_URL` and `KV_REST_API_TOKEN` source variables.
- `HF_STAGING_BUCKET` — private staging bucket ID in `namespace/bucket` form (EU
  recommended; US requires an explicit data-residency decision).
- `HF_TOKEN` — scoped token that can write the private staging bucket.
- `WORKER_COORDINATOR_SECRET` — secret accepted by the coordinator lock, renew, and
  release endpoints. It must also be set on every local queue host.
- `VOLUNTEER_PROMOTION_SECRET` — secret accepted by promotion reservation and transition
  endpoints. It must also be set in the maintainer review shell.

Generate secrets outside the repository and inject them from a password manager or a
hidden prompt; do not print them in logs. Rotate OAuth, GitHub, HF, Redis, coordinator,
and promotion credentials as coordinated changes: update Vercel and every dependent
local host together. A rotation invalidates old coordinator or promotion requests.
Treat `VOLUNTEER_MARKER_SECRET` as long-lived signed-marker key material, not a
routine credential; changing it requires the marker migration described above.

### Optional in Vercel Production

These have safe code defaults and should be set only when the default is deliberately
changed:

- `VOLUNTEER_LEASE_SECONDS` — bounded lease duration; the default is 1,800 seconds.
- `VOLUNTEER_SCOPE_POLICY_JSON` — complete JSON policy override; normally leave unset so
  the checked-in generated policy is used.
- `COMMUNITY_REVIEW_LABEL` — review label override; default `community-review-ready`.
- `RESULTS_READY_LABEL` — completion label override; default `results-ready`.
- `COMMUNITY_MAINTAINER_LOGIN` — GitHub login mentioned in the finalisation comment;
  default `saattrupdan`.

`VOLUNTEER_COORDINATOR_URL`, `HF_RESULTS_BUCKET`, `VOLUNTEER_BROKER_RESERVATION_URL`,
and `VOLUNTEER_BROKER_PROMOTION_URL` are **not Vercel variables**. They belong to the
local queue or maintainer review shell and are listed below.
`VOLUNTEER_COORDINATOR_STANDALONE` is also local-only and unsafe for normal shared
operation.

## Recommended deployment order

1. Confirm the repository, production Vercel project, production domain, GitHub labels,
   OAuth device flow, Upstash database, private staging bucket, and scoped tokens.
2. Generate and commit `api/worker/scope-policy.json`. Confirm that its policy and
   `EUROEVAL_VERSION` match the EuroEval package release. Separately confirm that the
   intended worker package/image uses `VOLUNTEER_WORKER_VERSION` (currently `1.0.0`).
3. Build and publish the immutable GHCR commit-SHA candidate. Verify that it is public
   and that its exact digest can be pulled anonymously.
4. Run the physical GPU canary below, promote that same digest to `latest`, and
   anonymously verify that `latest` resolves to the same digest.
5. Add or update the Vercel Production variables, especially the promoted image digest.
   Never point `VOLUNTEER_WORKER_IMAGE_DIGEST` at a tag.
6. Deploy the existing Vercel project with the leaderboard collection/generation
   deployment flow, or run `make frontend` only from a checkout where
   `src/frontend/csv/` has already been generated. Verify the deployment URL and
   production domain are the same application; deployment alone does not validate
   external services.
7. Run the safe route and configuration smoke tests below. Then configure the local
   queue host with the shared coordinator URL and secret before allowing it to claim
   issues.
8. Ask one volunteer to complete device authentication and run a controlled `--once`
   evaluation. Watch the issue marker, staging manifest, and logs before inviting more
   workers.

## Safe endpoint smoke tests

All broker functions accept `POST` only. `OPTIONS` is a harmless CORS/preflight-style
probe and returns `204` with an **empty body**, not `{}`; `GET` must return `405` with
`{"error":"Method not allowed"}`. Test the deployed domain without credentials:

```sh
set -eu
BASE=https://euroeval.com/api/worker
for route in \
  auth/start auth/poll auth/revoke claim heartbeat result finalise release \
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

This checks route discovery and the method guard without creating a device flow, lease,
result, lock, or promotion. To check protected configuration without sending a secret or
mutating state, submit a valid protocol envelope but no authentication:

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

A `POST` to `claim` without `Authorization: Bearer <credential>` must similarly return
`401`, without touching GitHub or claiming work. Do **not** use `auth/start` as a
routine smoke test: a valid request starts a GitHub device grant and consumes Redis
state. Use a real worker for that one end-to-end test.

The live endpoint contract is:

- `POST /auth/start` and `POST /auth/poll` — start and poll the GitHub device flow; poll
  returns `202` while pending and `200` with a broker credential when authorised.
- `POST /auth/revoke` — authenticated bearer credential; returns `200` with
  `status: "revoked"` and invalidates that credential.
- `POST /claim` — bearer credential plus hardware report; returns `200` with a lease or
  `200` with `status: "no_work"`.
- `POST /heartbeat` — bearer credential and `lease_id`; returns `200` with a renewed
  expiry.
- `POST /result` — bearer credential and exact lease-bound record; returns `201` for an
  upload or `200` with `status: "duplicate"` for an idempotent repeat.
- `POST /finalise` — bearer credential and `lease_id`; returns `200` with
  `status: "ready"` and a submission ID, or `already_finalised` on a safe retry.
- `POST /release` — bearer credential and `lease_id`; returns `200` with
  `status: "released"`.
- `POST /coordinator-lock`, `/coordinator-renew`, and `/coordinator-release` — the
  `x-coordinator-secret` and protocol body; acquire returns an opaque token and its
  expiry, renew returns `status: "renewed"`, and release returns `status: "released"`.
- `POST /promotion-lock` (alias `/promotion-reserve`) — `x-promotion-secret`; creates or
  resumes a fenced review reservation.
- `POST /promote` — `x-promotion-secret` plus a matching reservation, decision digest,
  and evidence; returns the accepted or rejected terminal transition.

Non-authentication endpoint errors normally include `protocol_version` after accepting
the protocol. Authentication handlers may omit it. All endpoints fail closed on missing
configuration, invalid credentials, expired leases, or fence conflicts. A `502`/`503`
after valid authentication generally means to inspect Vercel logs and the corresponding
external dependency rather than retrying blindly.

## Legacy/local queue coordination

If `src/scripts/process_evaluation_queue.py` is running, it and the Vercel broker can
see the same GitHub issues. The queue must acquire the broker's same Redis-backed issue
mutex before claiming or changing an issue. On **every** shared queue host set:

```sh
export VOLUNTEER_COORDINATOR_URL=https://euroeval.com/api/worker
read -r -s -p "Worker coordinator secret: " WORKER_COORDINATOR_SECRET
printf '\n'
export WORKER_COORDINATOR_SECRET
read -r -s -p "Volunteer marker secret: " VOLUNTEER_MARKER_SECRET
printf '\n'
export VOLUNTEER_MARKER_SECRET
```

Use a password-manager environment injection instead of the hidden prompt when one is
available. Never put the secret value directly in this command or in shell history.

The URL is the broker base URL, not a separately deployed server. The queue appends
`/coordinator-lock`, `/coordinator-renew`, and `/coordinator-release` itself. The
coordinator secret must match the Vercel `WORKER_COORDINATOR_SECRET` exactly. The
local queue also requires the same durable `VOLUNTEER_MARKER_SECRET` as Vercel to
verify signed broker markers. Keep both secrets out of shell history where practical
and out of queue logs. They are local maintainer configuration and are never passed to
volunteers or unrelated subprocesses.

The queue fails closed if any required coordinator value is absent. Do not set
`VOLUNTEER_COORDINATOR_STANDALONE=1` during normal operation. That local-only escape
hatch disables the shared lock and is permitted only for a deliberately isolated
migration after verifying that no broker, volunteer worker, or second queue can touch
the issue set. Never use it on a shared queue host.

Fresh local claims require an unassigned issue. A self-assigned issue is resumed only
when its sole assignee is the authenticated token login and its single VM marker matches
the current VM; manual self-assignment without that marker is not consumed.

The local queue also needs its existing `GITHUB_TOKEN` and Hugging Face access for its
non-volunteer work. Keep its results-writing credentials separate from the broker's
staging-only Vercel token where possible.

## Maintainer review and promotion

Run these commands from a checked-out EuroEval repository on a maintainer-controlled
host. The review shell needs a token that can read and write `HF_STAGING_BUCKET` and,
for approval, also write `HF_RESULTS_BUCKET` (default `EuroEval/results`), plus the
exact promotion secret. Inject secrets from a password manager or use hidden prompts; do
not put their values in shell history, this guide, or a `.env` file:

```sh
export HF_STAGING_BUCKET='<namespace>/<private-staging-bucket>'
export HF_RESULTS_BUCKET=EuroEval/results
read -r -s -p "Maintainer HF token: " HF_TOKEN
printf '\n'
read -r -s -p "Volunteer promotion secret: " VOLUNTEER_PROMOTION_SECRET
printf '\n'
export HF_TOKEN VOLUNTEER_PROMOTION_SECRET
```

If Vercel sets `VOLUNTEER_SCOPE_POLICY_JSON`, the review shell **must** use that exact
same complete JSON value before validating or approving a submission.

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
completes the broker transition. Rejection calls the same fenced transition and releases
reservations. The decision artifact is durable in staging, so an expired Redis
reservation cannot be reused for the opposite outcome. A retried reviewer may resume the
same outcome; an opposite outcome is refused.

Do not manually copy staged files, edit issue markers, delete audit manifests, or award
credit. A rejected language becomes leaseable again while its signed audit history
remains. When all selected languages are accepted, the broker adds `results-ready` and
retains the assignees whose submissions were accepted. If several volunteers contributed,
keep multiple assignees; remove an assignee only after rejection or intentional release.

## Image publishing and canary

This is a per-release process, not a one-time prerequisite. The
`.github/workflows/worker-image.yaml` workflow builds only `linux/amd64` with BuildKit
layer caching. Pull requests do not publish or attest an image: they load the local
image and smoke its normal entrypoint as UID 10001 without a GPU. Trusted runs publish
only the immutable commit-SHA candidate with provenance and an SBOM, verify the public
GHCR package, inspect the exact candidate manifest anonymously, and print the digest.
The workflow never creates or updates `latest`.

After the package is public and the workflow has printed a verified digest, run this on
a physical Linux `amd64` NVIDIA host:

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
`write:packages`. The PAT is read interactively and the Docker configuration is removed
before anonymous verification:

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
switch to a mutable base image or install a host driver in the image. Check the pinned
CUDA base, locked `uv.lock` dependencies, and NVIDIA runner/toolkit separately.

## Ongoing maintenance and monitoring

### Per release

- Update the EuroEval package and regenerate the scope policy so its version matches
  `EUROEVAL_VERSION`; this EuroEval/policy alignment is separate from worker releases.
- Update the worker package/image and `VOLUNTEER_WORKER_VERSION` independently when the
  worker protocol changes (the current worker version is `1.0.0`).
- Run the API tests (`npm run test:api`) and the relevant Python tests/checks.
- Publish, anonymously verify, canary, and manually promote the new immutable image.
- Update `VOLUNTEER_WORKER_IMAGE_DIGEST`, version, and policy variables in Vercel, then
  deploy and repeat the safe route probes.
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
  organisation's normal schedule. Coordinate rotations, verify safe probes, and revoke
  old credentials. Do not routinely rotate `VOLUNTEER_MARKER_SECRET`; changing it
  requires the signed-marker migration described above.
- Re-run the physical GPU canary after changes to the CUDA base, locked dependencies,
  NVIDIA toolkit, or host driver. Human or independent verification of the running image
  remains required: the configured digest is provenance carried by a lease, not runtime
  attestation.
