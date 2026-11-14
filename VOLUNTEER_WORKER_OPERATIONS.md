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

## Current promoted release

As of 13 November 2026, `latest` is a promoted multi-platform index containing the
physically tested `linux/amd64` and `linux/arm64` images, including DGX Spark/GB10:

```bash
IMAGE=ghcr.io/euroeval/euroeval-worker
PROMOTED_DIGEST=sha256:cc0656654f464d4fc89055307cb00f0b0a355eca3938d53575b580b15b61bf74
```

Both Production `VOLUNTEER_WORKER_IMAGE_DIGEST` and
`VOLUNTEER_WORKER_IMAGE_DIGEST_ARM64` use that index digest. Production advertises
EuroEval `18.4.0.dev0` and worker `1.0.0`. The candidate workflow's ARM metadata still
says `blocked-pending-physical-spark-canary`: that is a build-time gate, not the final
promotion record. Each future release must pass the hardware gates again.

Both architectures passed CUDA matmul, FlashInfer RMSNorm, actual Triton JIT execution,
and pinned public-model vLLM generation as UID 10001. ARM also passed the restrictive
Docker settings below. The amd64 RunPod validation established physical GPU execution
and UID, but did not establish those Docker isolation settings on RunPod.

## First-time procedure

Run this exact sequence from a clean checkout. The first command is a local, non-network
plan and the second is a read-only network check. Neither command imports `.env` or
mutates a file. Use Bash for the shell examples in this guide, especially blocks using
hidden prompts, `pipefail`, or `trap`.

1. **AUTOMATED CHECK** - print the workflow, defaults, and required variable names:

   ```bash
   uv run python src/scripts/volunteer_worker_operations.py plan
   ```

   When the operator prefers a local `.env` store, initialise only its missing broker
   secrets after explicit approval:

   ```bash
   uv run python src/scripts/volunteer_worker_operations.py apply \
     --local-secrets --yes
   ```

   This operation is local-only and does not import, source, or execute the file.

2. **AUTOMATED CHECK** - run read-only local and configured-service diagnostics. Supply
   credentials through the maintainer's normal secret injection; never put values in the
   command or guide:

   ```bash
   make volunteer-worker-check
   ```

   Unlike the review CLI, this check does not load `.env`. When that file is the chosen
   local store, the following explicitly parses it through `uv`, supplies KV aliases,
   and uses the maintainer's stored GitHub login for package metadata. It does not
   source the file or print credentials:

   ```bash
   GH_TOKEN="$(env -u GH_TOKEN -u GITHUB_TOKEN gh auth token)" \
     uv run --env-file .env bash -c '
       export UPSTASH_REDIS_REST_URL="${UPSTASH_REDIS_REST_URL:-$KV_REST_API_URL}"
       export UPSTASH_REDIS_REST_TOKEN="${UPSTASH_REDIS_REST_TOKEN:-$KV_REST_API_TOKEN}"
       exec uv run python src/scripts/volunteer_worker_operations.py check
     '
   ```

   Inject the intended public versions and image digest as well; stale local values are
   not the Production configuration. Keep this maintainer `GH_TOKEN` out of worker
   containers. The broker's narrower `GITHUB_TOKEN` may not read GHCR package metadata.
   The check verifies only its enumerated variables; also verify the seven GitHub App
   settings and the ARM image override below. A manual/unverified region or issue-write
   diagnostic does not prove that permission or region has been checked.

3. **MANUAL** - the maintainer chooses the existing `EuroEval/EuroEval` GitHub queue,
   `https://euroeval.com` Vercel project, recommended private EU staging bucket, the
   existing Vercel KV database used by the site submission limiter, shared GitHub App,
   and public GHCR package. A US staging bucket requires the explicit data-residency
   decision described below. The maintainer supplies credentials, selects an immutable
   image digest, and confirms production deployment. Pi/I can execute publication or
   deployment only after explicit approval, but cannot choose accounts or credentials,
   or perform a physical GPU canary without hardware.

4. **CONFIRMED AUTOMATION** - after reviewing step 2 and supplying the chosen values,
   explicitly apply only the selected scoped setup. There is intentionally no implicit
   all-components apply command:

   ```bash
   uv run python src/scripts/volunteer_worker_operations.py apply --github --yes
   uv run python src/scripts/volunteer_worker_operations.py apply --hf --yes \
     --hf-region eu
   # Use this only after an explicit US data-residency decision:
   uv run python src/scripts/volunteer_worker_operations.py apply --hf --yes \
     --hf-region us
   uv run python src/scripts/volunteer_worker_operations.py apply \
     --reuse-vercel-kv --yes
   ```

   The GitHub and HF commands are optional when those components already pass. For the
   selected KV reuse setup, configure the remaining Vercel variables as described below;
   the separate command above supplies only the two broker aliases. The regular Vercel
   apply requires every value except versions, which are derived from source when
   absent. Marker, coordinator, promotion, and canonical-publication secrets are
   durable. Cloud setup expects the operator to provide existing values; the KV reuse
   command copies the existing production KV values to the broker aliases without
   displaying them. If the operator has selected `.env` as the local store, the
   local-secrets operation generates only missing or empty values and never rotates
   existing values.

5. **MANUAL** - publish both candidate images using the workflow, carry their digests
   from the non-secret job summaries/artifacts, and run physical GPU canaries on the
   matching architectures. Combine only accepted candidates into the promoted
   multi-platform index, preserving their platform children and attestations. Configure
   the broker with the resulting promoted index digest, not an input candidate's index
   digest. Publication and production deployment require explicit approval.

6. **AUTOMATED CHECK** - after deployment, run safe route probes. They issue only GET,
   OPTIONS, and unauthenticated POST requests; they never start authentication, claim
   work, acquire a lock, or touch staging:

   ```bash
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
- **GitHub:** `EuroEval/EuroEval` is the work queue and audit log. Public volunteers
  need no repository access: their authenticated account, broker lease, and signed issue
  marker fence active work; accepted signed submissions provide credit. Manual work and
  historical assignee-backed leases retain their assignee rules.
- **GitHub App device flow:** A volunteer authorises the shared EuroEval Volunteer
  Worker App in a browser. The broker verifies the user and restricted installation,
  encrypts the renewable user grant in Redis, and issues an opaque broker credential. It
  uses the retained grant only for the verified volunteer's issue comment, not to grant
  repository membership. GitHub tokens and project secrets never reach the worker. A
  separately configured legacy OAuth fallback revokes its grant immediately; it is not
  the active Production flow.
- **Upstash Redis:** The broker's short-lived state store for credentials, leases, rate
  limits, mutexes, encrypted renewable GitHub grants, result reservations, and promotion
  reservations. The canonical-publication mutex has no expiry. Redis is coordination
  state, not the durable review or result source. The site's submission rate limiter and
  the volunteer broker use the same existing Vercel KV database under disjoint prefixes:
  `euroeval:submit` for submissions and `euroeval:worker` (and related worker prefixes)
  for broker state. This couples their outages, credential rotation, and flushes; this
  selected setup does not need a dedicated Upstash resource.
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
  non-volunteer work. Both claim paths take a shared cross-issue mutex before the
  per-issue mutex and recheck open requests for matching model/language ownership. This
  prevents a local claim on another issue from racing a public lease.

The normal data flow is worker authentication, issue claim, evaluation, one result
upload per identity, finalisation into a private manifest, maintainer review, and
promotion or rejection. Acceptance copies only verified records to the canonical bucket.
Public leases remain fenced by the signed, issue-bound marker and matching account and
lease identity. The broker also reconciles public assignees for active leases and
pending submissions. It preserves manual assignees, tracks only its own additions, and
cleans them up on release, expiry, or terminal review. Ineligible assignments do not
invalidate signed ownership. When needed, the App posts one `I'd like to evaluate this.`
comment as the verified volunteer, allowing a later assignment retry. The local queue
respects active and submitted languages under the shared coordinator lock. Accepted
signed submissions provide public volunteer credit only after review. Manual work and
historical assignee-backed leases continue to use issue assignees for ownership and
credit. Do not remove or edit signed markers to transfer public work; release or reject
it through the broker. A complete set of accepted languages adds `results-ready`.

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
  assets have been generated. `make frontend` verifies the exact function routes and
  runtimes emitted by the build before it can deploy them.
- Vercel Git integration may deploy the intended branch only when its build environment
  is also supplied with the generated assets. A clean Git-based deployment is not
  self-sufficient while `src/frontend/csv/` remains ignored; verify the production
  branch and checks rather than treating Git integration as the leaderboard flow.

Linking associates this repository with the existing Vercel project; it does not create
external accounts or supply environment variables. Configure those manually below.

### 2. Configure the shared GitHub App

Use the existing [EuroEval Volunteer Worker App][worker-app], not a separate app per
volunteer. Enable device flow and expiring user tokens. Its installation must select
only `EuroEval/EuroEval`, with exactly **Issues: read/write** and mandatory **Metadata:
read** permissions; do not add organisation, contents, or administration permissions.
The broker validates the installation and repository selection before onboarding.
Browser verification and consent are human-operated.

[worker-app]: https://github.com/apps/euroeval-volunteer-worker

Configure all seven server-side variables together:

- `VOLUNTEER_GITHUB_APP_CLIENT_ID` — App client ID.
- `VOLUNTEER_GITHUB_APP_CLIENT_SECRET` — App client secret.
- `VOLUNTEER_GITHUB_APP_ID` — App ID.
- `VOLUNTEER_GITHUB_APP_PRIVATE_KEY` — PKCS8 PEM private key; escaped newlines are
  accepted.
- `VOLUNTEER_GITHUB_APP_INSTALLATION_ID` — selected installation ID.
- `VOLUNTEER_GITHUB_APP_REPOSITORY_ID` — `386992479`, the EuroEval repository ID.
- `VOLUNTEER_GITHUB_APP_TOKEN_ENCRYPTION_KEY` — a base64-encoded 32-byte AES-GCM key.

Store secrets in the selected secret store and Vercel, never in Git or worker
containers. Partial configuration fails closed. The broker requests `offline_access`,
binds the exchange to the selected repository, verifies the GitHub user, and retains
encrypted access/refresh tokens tied to that broker credential. Refreshes revalidate
identity. Enabling App mode requires older OAuth-only workers to authorise again.

`POST /auth/revoke` removes local broker authority first, then attempts GitHub token
revocation. Preserve failed-revocation state for safe cleanup; do not restore an old
credential to retry it. An unavailable or corrupt grant must not keep a credential live.
Do not rotate the encryption key without a grant migration or coordinated revocation and
reauthorisation.

Keep `GITHUB_OAUTH_CLIENT_ID` and `GITHUB_OAUTH_CLIENT_SECRET` for the current
operations check and deliberate legacy rollback. With none of the seven App variables
set, the legacy OAuth device flow requests `read:user` and revokes the grant before
issuing a credential. Removing only some App variables is not a rollback.

### 3. Prepare the maintainer GitHub access and queue labels

Create a maintainer-owned GitHub token with the least privilege that permits the broker
to read and update issues, assignees, labels, and comments in `EuroEval/EuroEval`. Store
it only in the selected secret store and Vercel's sensitive environment variables. The
broker uses this token for issue markers, labels, operational comments, and assignment
reconciliation. The App user grant supplies the volunteer-authored comment. Public
volunteers only authenticate with GitHub; they do not need an invitation or repository
access.

Create these labels in `EuroEval/EuroEval` with these exact names:

- `model evaluation request` — required for the broker to discover open requests.
- `community-review-ready` — added when a volunteer manifest is ready for review.
- `results-ready` — added when all selected languages have accepted submissions.

If the legacy/local queue will remain active, also retain its labels `evaluation-failed`
and `gated`. Configure its shared coordinator lock and `VOLUNTEER_MARKER_SECRET` before
running it alongside public volunteers; otherwise it must not claim issues. Changing
assignees still fences historical assignee-backed leases, not new public leases.

### 4. Reuse the existing Vercel KV database

The site submission rate limiter and volunteer broker intentionally share the existing
Vercel KV database. Their keys remain disjoint: submissions use `euroeval:submit`, while
the broker uses `euroeval:worker` and related worker prefixes. This couples outages,
credential rotation, and flushes, so treat those operations as affecting both systems.
No dedicated Upstash resource is needed for this selected setup.

After the project link and the source Vercel KV variables are confirmed, run this exact
command. It verifies the linked project, reads `KV_REST_API_URL` and `KV_REST_API_TOKEN`
through Vercel's production environment, and adds only the two broker aliases without
exposing their values:

```bash
uv run python src/scripts/volunteer_worker_operations.py apply --reuse-vercel-kv --yes
```

If this operation must be rolled back, remove only the aliases. Never remove the source
KV variables, which the site uses:

```bash
vercel env rm UPSTASH_REDIS_REST_URL production --yes
vercel env rm UPSTASH_REDIS_REST_TOKEN production --yes
```

The operation is safe to repeat. It does not create a database, rotate credentials, or
modify `KV_REST_API_URL` or `KV_REST_API_TOKEN`.

### 5. Create private Hugging Face staging storage

The recommended staging location is a private Hugging Face Bucket in the EU region, for
example `<namespace>/<private-staging-bucket>`, with that exact ID set as
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
a token in this guide or commit it to Git.

### 6. Generate and align the scope policy

The checked-in `api/worker/scope-policy.json` and
`api/worker/_lib/scope-policy.generated.ts` are generated together from the official
dataset and language contracts. The JSON remains the cross-language policy artifact; the
TypeScript mirror avoids runtime JSON imports in Vercel handlers. Generate both for the
exact EuroEval release that the broker will advertise:

```bash
uv run python src/scripts/generate_volunteer_scope_policy.py \
  --version <euroeval-version>
git diff --check
```

Use `--check` to verify both synchronized files without writing, or `--dry-run` to
preview whether either would change. `--ts-output <path>` overrides the TypeScript
mirror location when using a custom JSON output. Commit both generated files with the
release change. Keep the EuroEval release and policy versions aligned exactly: bare
`.dev` is normalised to `.dev0`, not to a stable release. For example, `18.4.0.dev` and
`18.4.0.dev0` denote the same development version, while `18.4.0` is a distinct stable
version. Nonzero development suffixes and different base versions remain distinct.
`VOLUNTEER_WORKER_VERSION` is independent worker protocol/package versioning; it is
currently `1.0.0` and must match the worker image/package being published, but need not
equal the EuroEval version. The policy controls exact result identities; do not
hand-edit it. `VOLUNTEER_SCOPE_POLICY_JSON` is an optional complete override for an
intentional, reviewed deployment policy, not a way to patch one entry. If Vercel sets
this override, the review shell must use the identical policy JSON override.

### 7. Publish a public GHCR package

The first trusted workflow run may create the `ghcr.io/euroeval/euroeval-worker` package
privately. A package or organisation owner must make `euroeval-worker` public in GitHub
Packages settings, then rerun the workflow. GitHub does not provide a supported REST
operation for this visibility change. The workflow's anonymous manifest inspection must
pass before a volunteer is given an image reference.

## Vercel environment variables

Set these in the existing Vercel project under **Settings → Environment Variables**. The
required list below means the **Production** environment used by `https://euroeval.com`;
add values to Preview only for deliberately isolated broker testing. Keep secrets out of
frontend variables. Vercel's server-side `plain` storage type is not a `VITE_*` frontend
export: use it for nonsecret configuration and `sensitive` for credentials. Do not share
Production write credentials with an untrusted Preview deployment.

### Required in Vercel Production

- `GITHUB_TOKEN` — maintainer GitHub token for issue reads, labels, comments, and
  assignment in `EuroEval/EuroEval`.
- All seven `VOLUNTEER_GITHUB_APP_*` settings listed above — active App onboarding,
  renewable grants, and volunteer-authored comments.
- `GITHUB_OAUTH_CLIENT_ID` and `GITHUB_OAUTH_CLIENT_SECRET` — retained legacy fallback
  and current operations-check requirements, not the active App's credentials.
- `EUROEVAL_VERSION` — exact release represented by the generated scope policy.
- `VOLUNTEER_WORKER_VERSION` — exact supported worker protocol/package version.
- `VOLUNTEER_WORKER_IMAGE_DIGEST` — promoted index digest in the form `sha256:...`,
  containing an accepted `linux/amd64` image; this is configured lease provenance, not
  runtime attestation. The current promoted index also contains `linux/arm64`.
- `VOLUNTEER_WORKER_IMAGE_DIGEST_ARM64` — ARM64/aarch64 override. Production sets this
  to the same promoted multi-platform index. If unset, ARM falls back to the global
  digest, which must then contain an accepted ARM image.
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
- `VOLUNTEER_PROMOTION_SECRET` — secret accepted by promotion reservation, status, and
  transition endpoints. It must also be set in the maintainer review shell.
- `CANONICAL_PUBLICATION_SECRET` — a separate secret for the non-expiring canonical
  publication mutex. Set it in Production and every canonical-writing maintainer host,
  including collection, processing, sync, and review. There is no coordinator or
  promotion-secret fallback.

For local hosts, `.env` is the selected local secret store when that matches the
operator's preference. It remains Git-ignored; back it up according to the operator's
chosen backup practice. The guarded local-secrets operation parses assignments without
sourcing or executing the file, sets its mode to `0600`, generates only missing or empty
values, and never rotates existing values. It reports variable names and file
permissions, never secret values. Cloud credentials may still be injected by the
operator's normal preferred method. Rotate OAuth, GitHub, HF, Redis, coordinator, and
promotion, and canonical-publication credentials as coordinated changes: update Vercel
and every dependent local host together. App credential/encryption-key changes also
require coordinating retained grants. A rotation invalidates old coordinator or
promotion requests. Treat `VOLUNTEER_MARKER_SECRET` as long-lived signed-marker key
material, not a routine credential; changing it requires the marker migration described
above.

### Public configuration types and safe updates

Keep these five nonsecret Production settings as API type `plain`: `EUROEVAL_VERSION`,
`VOLUNTEER_WORKER_VERSION`, `VOLUNTEER_WORKER_IMAGE_DIGEST`,
`VOLUNTEER_WORKER_IMAGE_DIGEST_ARM64`, and `HF_STAGING_BUCKET`. If configured,
`CANONICAL_PUBLICATION_URL` is also plain. Credentials remain sensitive.

Vercel CLI 62's `--type config` stores API type `encrypted`, which does not satisfy the
operations check's plain-type contract. Its accepted CLI types also differ from the
operations helper's `apply --vercel` arguments. That helper does not configure the seven
App settings or ARM override. Do not use it as a complete CLI-62 setup procedure. For a
targeted existing nonsecret setting, use the project API and verify the exact
Production-only variable. This example requires `jq`, a linked project, and the reviewed
`PROMOTED_DIGEST` from the image promotion section:

```bash
(
set -euo pipefail
umask 077
PROJECT_ID="$(jq -r .projectId .vercel/project.json)"
TEAM_ID="$(jq -r .orgId .vercel/project.json)"
NAME=VOLUNTEER_WORKER_IMAGE_DIGEST
VALUE="$PROMOTED_DIGEST"
TEMP="$(mktemp -d)"
trap 'rm -rf "$TEMP"' EXIT
vercel api "/v10/projects/$PROJECT_ID/env?teamId=$TEAM_ID" \
  --method GET --raw > "$TEMP/envs.json"
VARIABLE_ID="$(jq -er --arg name "$NAME" '
  [.envs[] | select(.key == $name and .target == ["production"])] |
  if length == 1 then .[0].id else error("Need one Production-only entry") end
' "$TEMP/envs.json")"
jq -n --arg value "$VALUE" '{type: "plain", value: $value}' \
  > "$TEMP/update.json"
vercel api "/v9/projects/$PROJECT_ID/env/$VARIABLE_ID?teamId=$TEAM_ID" \
  --method PATCH --input "$TEMP/update.json" --raw > "$TEMP/updated.json"
jq -e --arg name "$NAME" --arg value "$VALUE" '
  .key == $name and .type == "plain" and .target == ["production"]
  and .value == $value
' "$TEMP/updated.json" > /dev/null
)
```

Repeat for `VOLUNTEER_WORKER_IMAGE_DIGEST_ARM64` with the same promoted index. For
versions or the staging ID, choose the corresponding nonsecret name and confirmed value.
This block is not a secret-update template. Do not change unrelated entries, remove
variables first, or print the full environment response. For first-time entries, create
an explicitly Production-targeted variable through the project API with the appropriate
type; keep secret values out of arguments and logs.

`vercel env pull` masks sensitive values as `[SENSITIVE]`; a pulled file is not a secret
backup and must never overwrite the selected local `.env`. Even pulled files can contain
credentials such as an OIDC token: use restrictive permissions and temporary storage.
Redeploy after changing configuration and verify the new Production alias; updating an
environment variable does not update an existing deployment.

### Optional in Vercel Production

These have safe code defaults and should be set only when the default is deliberately
changed:

- `VOLUNTEER_LEASE_SECONDS` — bounded lease duration; the default is 1,800 seconds.
- `VOLUNTEER_SCOPE_POLICY_JSON` — complete JSON policy override; normally leave unset so
  the checked-in generated policy is used.
- `COMMUNITY_REVIEW_LABEL` — review label override; default `community-review-ready`.
- `RESULTS_READY_LABEL` — completion label override; default `results-ready`.
- `COMMUNITY_MAINTAINER_LOGIN` — GitHub login mentioned in the finalisation comment and
  excluded from volunteer credit; default `saattrupdan`.
- `CANONICAL_PUBLICATION_URL` — optional canonical-lock broker base URL or full
  `/publication-lock` URL. Normally leave unset to use the public broker.
- `HF_RESULTS_BUCKET` — canonical bucket identity used by publication fencing; default
  `EuroEval/results`. Coverage lookup currently targets `EuroEval/results` in code. A
  different bucket requires a reviewed coverage migration as well as aligned writer
  configuration; changing an environment variable alone is insufficient.

`VOLUNTEER_COORDINATOR_URL`, `VOLUNTEER_BROKER_RESERVATION_URL`, and
`VOLUNTEER_BROKER_PROMOTION_URL` are local queue/review settings, not required Vercel
variables. `HF_RESULTS_BUCKET` and `CANONICAL_PUBLICATION_URL` also apply on local
canonical-writing hosts. `VOLUNTEER_BROKER_URL` overrides the local publication
adapter's default broker when no explicit publication URL is set.
`VOLUNTEER_COORDINATOR_STANDALONE` is also local-only and unsafe for normal shared
operation.

## Recommended deployment order

1. Confirm the repository, production Vercel project, production domain, GitHub labels,
   GitHub App device flow, Upstash database, private staging bucket, and scoped tokens.
   Check active leases and stop canonical writers before changing shared protocol,
   grant, or publication-lock configuration.
2. Generate and commit `api/worker/scope-policy.json` and
   `api/worker/_lib/scope-policy.generated.ts`. Confirm that their synchronized policy
   and `EUROEVAL_VERSION` match the EuroEval package release. Separately confirm that
   the intended worker package/image uses `VOLUNTEER_WORKER_VERSION` (currently
   `1.0.0`).
3. Build and publish the immutable GHCR commit-SHA candidate. Verify that it is public
   and that its exact digest can be pulled anonymously.
4. Run the physical GPU canaries below for both architectures. Promote an index with
   those unchanged platform images and attestations, then anonymously verify its digest
   and descriptors. An index combining two candidates has a new digest.
5. Add or update Production variables using the correct types, including the separate
   publication secret. Set both image variables to the verified promoted index so a
   fresh `latest` pull matches lease provenance. Never configure a floating tag or label
   an untested image with the accepted digest.
6. Deploy the existing Vercel project with the leaderboard collection/generation
   deployment flow, or run `make frontend` only from a checkout where
   `src/frontend/csv/` has already been generated. `make frontend` runs the read-only
   function verifier between `vercel build --prod` and deployment, rejecting missing,
   extra, malformed, or incorrectly-runtimed functions. To verify a build without
   deploying, run:

   ```bash
   uv run python src/scripts/build_vercel_frontend.py
   uv run python src/scripts/verify_vercel_functions.py
   ```

   The build helper refreshes actual Production configuration, generates matching scope
   policies for the build, and restores checkout policy bytes afterward. Verify READY
   status and the production alias; deployment alone does not validate external
   services. Stable/development version bumps regenerate source policies, and
   `make publish` synchronises and deploys the captured stable version before adding the
   next development suffix. For a separate version rollout, use
   `make sync-production-version version=<version>` after generating matching policies.
   That helper uses CLI `--type config`: recheck and correct the public API type before
   the final readiness check.

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

```bash
set -eu
BASE=https://euroeval.com/api/worker
for route in \
  auth/start auth/poll auth/revoke claim heartbeat result finalise release \
  coordinator-lock coordinator-renew coordinator-release \
  promotion-lock promotion-reserve promotion-status publication-lock promote
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

```bash
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
- `POST /promotion-status` — `x-promotion-secret` plus the exact issue, submission,
  outcome, decision digest, and record evidence; returns `absent`, `reserved`, or
  `terminal` without changing the reservation. Use the review tool's recovery path.
- `POST /publication-lock` — dedicated bearer publication secret and
  `{action, bucket, owner}`; actions are `acquire`, `assert`, `release`, and `status`.
  `status` is read-only and does not require an owner. This endpoint does not use the
  worker protocol envelope.
- `POST /promote` — `x-promotion-secret` plus a matching reservation, decision digest,
  and evidence; acceptance also requires the matching publication owner and bucket.
  Returns the accepted or rejected terminal transition.

Non-authentication endpoint errors normally include `protocol_version` after accepting
the protocol. Authentication handlers may omit it. All endpoints fail closed on missing
configuration, invalid credentials, expired leases, or fence conflicts. A `502`/`503`
after valid authentication generally means to inspect Vercel logs and the corresponding
external dependency. `503` with code `issue_busy` means transient mutex contention;
retry with backoff and honour `Retry-After`. Ownership, expiry, digest, and fence
conflicts remain `409` and must not be bypassed. Claim discovery/deadline failures are
transient and the worker retries them with a bounded backoff. HTTP `426` with
`worker_update_required` requires a new worker image before GPU work, not weaker
validation.

## Canonical coverage and duplicate prevention

Claim discovery checks canonical result identities for the exact immutable model
revision at the requested or a newer EuroEval version. The broker narrows each trusted
official scope alternative to missing identities, retaining the original scope and
coverage proof in the lease. Fully covered work is skipped. Unknown revisions, invalid
records, and unavailable coverage are not treated as proof that evaluation is complete.
The reviewer independently checks the proof against the official scope and canonical
records; workers must not choose arbitrary subsets.

Current workers advertise `canonical_coverage_v1`, evaluate the leased ordinary
datasets, and run only `contamination-detection` for a required canary. The same
benchmarker/model lifecycle is reused without letting a canary request pull in the
ordinary tasks again. Clients without this capability receive `426` before GPU work when
a narrowed scope is needed.

The broker persists the authoritative `record_revision_required` flag: modern leases
require the expected revision in every record, while legacy full-scope leases can omit
it with a warning. Explicitly wrong or null revisions still fail. Do not manufacture a
revision or edit raw records to get through validation. Missing-revision canonical
records cannot establish reusable coverage.

Local validated bucket refreshes also replace changed canonical objects atomically,
track content hashes/generations outside the result glob, and invalidate result caches.
Do not treat an existing local file or a previous unchanged download as proof that the
remote result is still current.

## Canonical publication and recovery

Every in-repository canonical writer uses `CanonicalPublisher` and the same durable
per-bucket publication lock: review, collection, ordinary processing, bucket sync, and
empty-object deletion. Configure `CANONICAL_PUBLICATION_SECRET` on every such host and
in Production. The default endpoint is
`https://euroeval.com/api/worker/publication-lock`; local `CANONICAL_PUBLICATION_URL`
takes precedence over `VOLUNTEER_BROKER_URL`.

The lock is non-expiring and ownership-fenced. A competing owner gets `409`; a repeat
acquire or release by the same owner is idempotent. Configure Redis persistence and
`noeviction`, and treat flushes, restores, credential rotation, and manual key changes
as publication incidents. Never use a TTL takeover or delete the lock key to clear a
busy writer.

A publication session checks exact predecessor bytes while owning the lock, archives and
verifies predecessors before overwriting, then verifies the new bytes. Ordinary
publisher history is content-addressed under `_history/<sha256>.bin` in the canonical
bucket. Volunteer-review predecessor archives are instead kept in private staging as
described below. Private canary records are never published canonically. Equal-ranked
formatting differences are no-ops; validated metadata enrichment requires an exact
source and unchanged evaluation data.

This protocol serialises participating writers; it is not storage-level compare-and-
swap or an atomic multi-file transaction. An external writer bypassing it or a Redis
reset can violate the guarantee. Keep all canonical write paths inside the publisher.

Before lock acquisition, the writer saves a private recovery receipt under
`euroeval-publication-locks/<owner>.json` in `XDG_CACHE_HOME`, or the system temporary
directory when unset. The directory is mode `0700`, the receipt `0600`; its owner token
is private authority. If a write or verification has an ambiguous outcome, the session
is poisoned and retains its lock and receipt rather than releasing unsafe state.

On an interrupted or failed publication:

1. Stop the original writer and all competing canonical writers. Keep its receipt and
   the durable staged decision, archive bytes, and any completion receipt unchanged.
2. Inspect canonical destinations and predecessor archives against the exact original
   plan. Use `list` and `show` for volunteer submissions and inspect the authenticated
   read-only promotion status through the review tool. A failed command can already have
   written some objects or bound a decision.
3. Reconcile the uncertain operation before releasing anything. Only after proving the
   original writer cannot resume and resolving every ambiguous write may the operator
   use the original receipt's owner with
   `PublicationLock.request(action="release", owner=...)`. Never print the owner or use
   a new owner to take over.
4. Obtain a new explicit confirmation before resuming an incomplete review decision.
   Recovery must retain the original outcome, evidence, fingerprint, predecessors,
   reviewer, and decision time; it must not adopt a newly changed canonical head.

A terminal broker status or valid durable completion receipt resolves a lost terminal
response without republishing historical data over a newer canonical head. There is no
approve-all, forced collision overwrite, or automatic failed-decision retry.

## Legacy/local queue coordination

If `src/scripts/process_evaluation_queue.py` is running, it and the Vercel broker can
see the same GitHub issues. The queue must acquire the broker's same Redis-backed issue
mutex before claiming or changing an issue. On **every** shared queue host set:

```bash
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
coordinator secret must match the Vercel `WORKER_COORDINATOR_SECRET` exactly. The local
queue also requires the same durable `VOLUNTEER_MARKER_SECRET` as Vercel to verify
signed broker markers. Keep both secrets out of shell history where practical and out of
queue logs. They are local maintainer configuration and are never passed to volunteers
or unrelated subprocesses.

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
exact promotion and canonical-publication secrets. When `.env` is the selected local
secret store, the review script loads it without printing values. Verify that the
Git-ignored file is mode `0600`; otherwise inject secrets from a password manager or use
hidden prompts. Do not put values in shell history, this guide, or command arguments:

```bash
export HF_STAGING_BUCKET='<namespace>/<private-staging-bucket>'
export HF_RESULTS_BUCKET=EuroEval/results
# The review script loads the selected local .env automatically, including
# VOLUNTEER_PROMOTION_SECRET and CANONICAL_PUBLICATION_SECRET, without printing values.
mode=$(stat -f '%Lp' .env 2>/dev/null || stat -c '%a' .env)
test "$mode" = 600
```

If `.env` is not the selected store, inject `HF_TOKEN` and `VOLUNTEER_PROMOTION_SECRET`,
and `CANONICAL_PUBLICATION_SECRET` from a password manager or hidden prompts instead;
never put their values in command arguments or shell history. If Vercel sets
`VOLUNTEER_SCOPE_POLICY_JSON`, the review shell **must** use that exact same complete
JSON value before validating or approving a submission. Do not print the contents of
`.env` or any loaded secret while checking the setup.

Listing and showing are non-mutating validation operations. `list` means pending
submissions; use `--all` when historical terminal decisions are needed:

```bash
uv run python src/scripts/review_volunteer_results.py list
uv run python src/scripts/review_volunteer_results.py list --all
uv run python src/scripts/review_volunteer_results.py show <submission-id>
```

For every pending submission, run `list` and then `show <submission-id>` before forming
an opinion. Compare like-for-like canonical/public results and similar models: check
model/revision/version/image provenance, hardware and speed, scope/counts, canary
evidence, score jumps, repeated or identical patterns, metadata mismatches, and all
warnings. Suspicious evidence requires manual investigation, not a recommendation to
approve. A configured image digest is not proof of the image that actually ran; never
rewrite a manifest to hide a discrepancy.

Failed-sample counts are informational, not an automatic rejection threshold, rate, or
denominator. The validator recomputes honest counts without double-counting nested,
aggregate, or repeated representations. Raw EEE records may lack enrichment fields such
as `commercially_licensed`, `open`, and `trained_from_scratch`; those omissions are
warnings, not permission to invent values. Malformed supplied metadata still fails.
Downstream leaderboard processing enriches metadata separately.

`show` displays canonical writes, replacements, conflicts, and a plan fingerprint. A
different destination can be replaced only by a strictly newer record, ordered by
EuroEval version and then retrieval timestamp. Every replacement archives the exact
predecessor in private staging at `volunteer/predecessors/<submission-id>/<sha256>.json`
and verifies it before any overwrite. Identical destinations need no archive. Ties and
conflicting identities block acceptance, not inspection or a deliberate rejection.

Get explicit permission immediately before **each** terminal command, after showing that
submission's evidence and replacement plan. Use the maintainer's verified lowercase
GitHub login, a useful reason, and the exact displayed fingerprint for approval:

```bash
uv run python src/scripts/review_volunteer_results.py \
  --reviewer <github-login> approve <submission-id> \
  --plan-fingerprint <displayed-plan-fingerprint> \
  --reason "checks passed"
uv run python src/scripts/review_volunteer_results.py \
  --reviewer <github-login> reject <submission-id> \
  --reason "reason"
```

Each decision re-downloads and validates immutable manifest/result bytes, model and
language identity, generated scope, and coverage proof. Acceptance acquires the shared
publication lock, binds the immutable plan and decision, archives predecessors, writes
only the approved ordinary records, verifies uploads, and completes the broker
transition. Rejection uses a fenced terminal transition without canonical publication.

The content-addressed decision is durable in private staging under
`volunteer/decisions/<submission-id>/<sha256>.json`; an expired Redis reservation does
not permit an opposite outcome. Active or expired reservation recovery preserves the
original decision and plan. After any failure, inspect durable decision/completion state
and `show` again before requesting permission to retry: a failed approval may already be
partly complete. Completed terminal retries do not republish old results. A changed
destination is not silently substituted into the confirmed plan.

Do not manually copy staged files, edit issue markers, delete audit manifests, or award
credit. A rejected language becomes leaseable again while its signed audit history
remains. When all selected languages are accepted, the broker adds `results-ready`;
accepted signed public submissions provide volunteer credit independently of whether
GitHub accepted their assignment. The maintainer is excluded from volunteer credit but
can authenticate and evaluate. Historical assignee-backed submissions retain their
assignee credit rules. The frontend shows `Pending review` before acceptance and
`Awaiting publish` when the request is ready for publication.

### Sequential review during leaderboard collection

`make leaderboards` and `make force-leaderboards` run the staged-review preflight before
harvesting results, rebuilding archives/CSVs, or deploying. In an interactive shell,
opting into review displays each report and asks `Approve submission <id>? [y/N]:`
separately. Only `y` or `yes` approves that displayed plan. Enter or `n` leaves it
pending; it does not reject it. The reviewer login comes from `GITHUB_ACTOR` or a
bounded `gh api user` lookup.

For agent-operated runs, also perform `list` and `show` and obtain explicit user
permission immediately before each affirmative response. There is no batch approval.
Report/approval errors halt collection after the review loop, and EOF or Ctrl-C stops it
immediately. Check decision state before rerunning. Noninteractive runs print manual
review commands and continue without automatic decisions. Missing/unavailable optional
staging can continue with canonical results; confirmed public staging fails closed.
Never turn a preflight failure into an automatic retry or rejection.

## Image publishing and canary

This is a per-release process. `.github/workflows/worker-image.yaml` builds separate
native `linux/amd64` and `linux/arm64` candidates with locked dependencies and BuildKit
caching. Pull requests smoke local images as UID 10001 without a GPU; they neither
publish nor provide physical GPU acceptance. Trusted runs publish commit-SHA candidates,
provenance, and SBOMs, verify public access anonymously, and upload metadata artifacts.
The workflow serialises builds for the same commit and never updates `latest`.

The ARM job/artifact's candidate-only label means each newly built candidate needs
physical acceptance. It does not revoke an older image's completed promotion. Accept
both architectures from the intended source/dependency set before combining them; never
configure an ARM-only index as the global amd64 image identity.

### Physical GPU acceptance

Use exact immutable candidate references from the trusted workflow, on their real
architectures: x86_64 for amd64 and DGX Spark/aarch64 for ARM. Record the source commit,
index/platform digests, image ID, UID, host architecture, GPU/driver, dependency
versions, available resources, commands, exit codes, and logs. CPU imports, QEMU, and a
Dockerfile rebuilt locally are not physical acceptance of a published candidate.

Before launching, check free disk/GPU/RAM, running jobs, and Docker/NVIDIA access. Do
not stop unrelated workloads. On paid hardware, obtain permission, set an hourly
rate/spend limit and independent cleanup deadline before provisioning, and verify
termination afterward. Do not authenticate a volunteer, claim broker work, submit
results, or approve anything for this hardware test.

Save the following two Python blocks as `/tmp/euroeval-jit-canary.py` and
`/tmp/euroeval-image-model-canary.py`, respectively. They exercise the published runtime
without project credentials or broker access. The model is public, ungated, and pinned
to an immutable revision; its tiny generation is separate from volunteer result
submission.

```python
"""Exercise writable inference caches and runtime CUDA compilation."""

import flashinfer
import torch
import triton
import triton.language as tl


@triton.jit
def add_one(input_ptr, output_ptr, block_size: tl.constexpr) -> None:
    """Add one to a single GPU block.

    Args:
        input_ptr: Source GPU values.
        output_ptr: Destination GPU values.
        block_size: Number of contiguous values.
    """
    offsets = tl.arange(0, block_size)
    values = tl.load(input_ptr + offsets)
    tl.store(output_ptr + offsets, values + 1)


values = torch.ones(size=(16, 256), device="cuda", dtype=torch.float16)
weight = torch.ones(size=(256,), device="cuda", dtype=torch.float16)
normalised = flashinfer.rmsnorm(values, weight)
torch.testing.assert_close(actual=normalised, expected=values, rtol=2e-3, atol=2e-3)
print("FlashInfer RMSNorm passed", flush=True)
source = torch.arange(256, device="cuda", dtype=torch.float32)
output = torch.empty_like(source)
add_one[(1,)](source, output, block_size=256)
torch.testing.assert_close(actual=output, expected=source + 1)
print("Triton runtime compilation and execution passed", flush=True)
```

The second script uses a 64 MiB KV cache, one sequence, and eager execution. Ensure its
resource budget fits the chosen GPU before running it:

```python
"""Run bounded real vLLM inference without contacting the volunteer broker."""

import json
import logging
import os
from pathlib import Path

from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

if __name__ == "__main__":
    logging.info("Starting pinned public-model GPU canary as UID %s", os.getuid())
    assert os.getuid() == 10001
    model = LLM(
        model="HuggingFaceTB/SmolLM2-135M-Instruct",
        revision="12fd25f77366fa6b3b4b768ec3050bf629380bac",
        tokenizer_revision="12fd25f77366fa6b3b4b768ec3050bf629380bac",
        trust_remote_code=False,
        dtype="float16",
        max_model_len=128,
        max_num_seqs=1,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.025,
        kv_cache_memory_bytes=64 * 1024 * 1024,
        enforce_eager=True,
        enable_prefix_caching=False,
        compilation_config={"mode": 0},
        seed=4242,
    )
    outputs = model.generate(
        prompts=["The capital of France is"],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=8),
        use_tqdm=False,
    )
    assert len(outputs) == 1
    completion = outputs[0].outputs[0]
    assert completion.token_ids and completion.text.strip()
    evidence = {
        "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
        "uid": os.getuid(),
        "tokens": len(completion.token_ids),
        "completion": completion.text,
    }
    Path("/cache/model-canary-evidence.json").write_text(json.dumps(evidence))
    logging.info("Real vLLM generation passed: %s", evidence)
```

Run this block in Bash on each physical host with that architecture's verified digest.
Choose a GPU and limits with spare capacity. The example requires at least 8 GiB of
available host memory; it never mounts an existing worker cache. Fresh `/cache` remains
executable for CUDA JIT, while `/tmp` is noexec and the image filesystem is read-only.

```bash
(
set -euo pipefail
IMAGE=ghcr.io/euroeval/euroeval-worker
DIGEST=sha256:REPLACE_WITH_VERIFIED_CANDIDATE_DIGEST
[[ "$DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]]
CANDIDATE="$IMAGE@$DIGEST"
GPU_DEVICE=0
test "$(uname -s)" = Linux
case "$(uname -m)" in
  x86_64) PLATFORM=amd64 ;;
  aarch64|arm64) PLATFORM=arm64 ;;
  *) echo "Unsupported canary architecture" >&2; exit 1 ;;
esac
nvidia-smi
docker pull "$CANDIDATE"
test "$(docker image inspect "$CANDIDATE" \
  --format '{{.Architecture}}')" = "$PLATFORM"
test "$(docker image inspect "$CANDIDATE" \
  --format '{{.Config.User}}')" = worker
CANARY="euroeval-image-canary-$(date +%s)-$$"
CACHE="$CANARY-cache"
docker volume create "$CACHE" > /dev/null
trap 'docker rm -f "$CANARY" >/dev/null 2>&1 || true;
      docker volume rm "$CACHE" >/dev/null 2>&1 || true' EXIT
GPU_RUN=(docker run --rm --name "$CANARY" --pull=never
  --gpus "device=$GPU_DEVICE" --memory=8g --cpus=4
  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=2g --shm-size=2g
  --cap-drop=ALL --security-opt=no-new-privileges --pids-limit=512
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e FULL_LOG=1
  -e OMP_NUM_THREADS=2 -e TOKENIZERS_PARALLELISM=false
  -v "$CACHE:/cache")
test "$("${GPU_RUN[@]}" --entrypoint id "$CANDIDATE" -u)" = 10001
"${GPU_RUN[@]}" "$CANDIDATE" --gpu-health-check
"${GPU_RUN[@]}" --entrypoint /opt/euroeval/.venv/bin/python \
  -v /tmp/euroeval-jit-canary.py:/canary.py:ro "$CANDIDATE" /canary.py
"${GPU_RUN[@]}" --entrypoint /opt/euroeval/.venv/bin/python \
  -v /tmp/euroeval-image-model-canary.py:/canary.py:ro "$CANDIDATE" /canary.py
docker run --rm --read-only -v "$CACHE:/cache:ro" --entrypoint cat \
  "$CANDIDATE" /cache/model-canary-evidence.json \
  > /tmp/euroeval-model-canary-evidence.json
)
```

Capture stdout/stderr and Docker configuration with the acceptance evidence. The
built-in `--gpu-health-check` imports PyTorch/vLLM and performs CUDA matmul; it does not
perform the separate FlashInfer, Triton JIT, or real-model checks above. Preserve exact
isolation evidence separately from physical runtime evidence when a hosted GPU provider
does not expose Docker controls. Never claim an untested isolation setting. Use
`FULL_LOG=1` to expose engine root causes. JIT/config caches must remain under `/cache`;
do not make the root writable or `/tmp` executable to hide a cache failure.

### Multi-platform promotion

After both physical canaries pass and promotion is explicitly authorised, combine the
original candidate indexes, preserving all platform and attestation descriptors. This
needs Bash, `jq`, Docker Buildx, and a maintainer PAT with `write:packages`. The
temporary Docker configs retain only plugin paths, not existing credentials; the empty
GHCR credential-helper entry keeps this login in the disposable config.

```bash
PROMOTION_EVIDENCE="$(mktemp -d /tmp/euroeval-image-promotion.XXXXXX)"
(
set -euo pipefail
umask 077
IMAGE=ghcr.io/euroeval/euroeval-worker
AMD64_DIGEST=sha256:REPLACE_WITH_ACCEPTED_AMD64_CANDIDATE
ARM64_DIGEST=sha256:REPLACE_WITH_ACCEPTED_ARM64_CANDIDATE
[[ "$AMD64_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]]
[[ "$ARM64_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]]
AMD64="$IMAGE@$AMD64_DIGEST"
ARM64="$IMAGE@$ARM64_DIGEST"
ORIGINAL_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}"
TEMP="$(mktemp -d)"
trap 'rm -rf "$TEMP"' EXIT
mkdir "$TEMP/auth" "$TEMP/anonymous"
if test -f "$ORIGINAL_CONFIG/config.json"; then
  jq --arg plugins "$ORIGINAL_CONFIG/cli-plugins" '
    {cliPluginsExtraDirs: ((.cliPluginsExtraDirs // []) + [$plugins]),
     credHelpers: {"ghcr.io": ""}}
  ' "$ORIGINAL_CONFIG/config.json" > "$TEMP/auth/config.json"
else
  printf '%s\n' '{"credHelpers":{"ghcr.io":""}}' > "$TEMP/auth/config.json"
fi
cp "$TEMP/auth/config.json" "$TEMP/anonymous/config.json"
unset BUILDX_BUILDER DOCKER_AUTH_CONFIG
export DOCKER_CONFIG="$TEMP/anonymous"
docker buildx imagetools inspect "$IMAGE:latest" \
  --format '{{json .Manifest}}' > "$TEMP/previous.json"
docker buildx imagetools inspect "$AMD64" --raw > "$TEMP/amd64.json"
docker buildx imagetools inspect "$ARM64" --raw > "$TEMP/arm64.json"
docker buildx imagetools create --dry-run --tag "$IMAGE:latest" \
  "$AMD64" "$ARM64" > "$TEMP/planned.json"
jq -e -s '
  ([.[0].manifests[], .[1].manifests[]] | sort_by(.digest)) ==
  (.[2].manifests | sort_by(.digest))
' "$TEMP/amd64.json" "$TEMP/arm64.json" "$TEMP/planned.json" > /dev/null
PREVIOUS_DIGEST="$(jq -r .digest "$TEMP/previous.json")"
printf 'Keep rollback digest: %s\n' "$PREVIOUS_DIGEST"
export DOCKER_CONFIG="$TEMP/auth"
read -r -p "GitHub username: " GHCR_USER
read -r -s -p "GHCR PAT (write:packages): " GHCR_PAT
printf '\n'
printf '%s' "$GHCR_PAT" | docker login ghcr.io \
  --username "$GHCR_USER" --password-stdin
unset GHCR_PAT
docker buildx imagetools create --tag "$IMAGE:latest" "$AMD64" "$ARM64"
rm -rf "$TEMP/auth"
export DOCKER_CONFIG="$TEMP/anonymous"
docker buildx imagetools inspect "$IMAGE:latest" \
  --format '{{json .Manifest}}' > "$TEMP/promoted.json"
jq -e -s '.[0].manifests == .[1].manifests' \
  "$TEMP/planned.json" "$TEMP/promoted.json" > /dev/null
PROMOTED_DIGEST="$(jq -er .digest "$TEMP/promoted.json")"
[[ "$PROMOTED_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]]
docker buildx imagetools inspect "$IMAGE@$PROMOTED_DIGEST" --raw \
  > "$TEMP/immutable.json"
jq -e -s '.[0].manifests == .[1].manifests' \
  "$TEMP/promoted.json" "$TEMP/immutable.json" > /dev/null
printf 'Verified promoted index: %s@%s\n' "$IMAGE" "$PROMOTED_DIGEST"
cp "$TEMP/previous.json" "$TEMP/amd64.json" "$TEMP/arm64.json" \
  "$TEMP/planned.json" "$TEMP/promoted.json" "$TEMP/immutable.json" \
  "$PROMOTION_EVIDENCE/"
)
PROMOTED_DIGEST="$(jq -er .digest "$PROMOTION_EVIDENCE/promoted.json")"
printf 'Promotion evidence: %s\n' "$PROMOTION_EVIDENCE"
```

The block retains public manifests under the printed evidence directory, removes its
private login configuration, and restores the caller's Docker environment on exit. Keep
those manifests and the previous digest with the release evidence. Verify the union
contains exactly the accepted Linux amd64 and ARM64 platform images, plus their
unchanged attestations. Anonymously pull the promoted index on each architecture and
compare its selected platform image/config identity with the physical acceptance record.

The combined index's digest is new even though the platform images are unchanged. Set
**both** Production image variables to `PROMOTED_DIGEST`, not to the individual
candidate-index digests. Redeploy and verify READY status, the production alias,
configuration, safe route probes, and anonymous `latest` resolution. Publishing or
moving a tag alone is not completion of a rollout.

For rollback, first inspect active leases and pause new workers. Restore the previous
accepted `latest` index, restore the corresponding architecture-specific Production
values and compatible versions/policies, redeploy, and repeat verification. Keep old
accepted indexes available; never rebuild or relabel a different image as an old digest.
Do not switch to a mutable CUDA base or install a host driver inside the image to fix a
failed build or canary.

## Ongoing maintenance and monitoring

### Per release

- Update the EuroEval package and regenerate the scope policy so its version matches
  `EUROEVAL_VERSION`; this EuroEval/policy alignment is separate from worker releases.
- Update the worker package/image and `VOLUNTEER_WORKER_VERSION` independently when the
  worker protocol changes (the current worker version is `1.0.0`).
- Run the API tests (`npm run test:api`) and the relevant Python tests/checks.
- Publish, anonymously verify, canary, and manually promote the new immutable image.
- Update both image variables to the accepted multi-platform index, align versions and
  policies, verify Vercel public/sensitive types, deploy, and repeat readiness and safe
  route probes. Confirm the production alias actually serves the new deployment.
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
- Rotate OAuth/App, GitHub, HF, Redis, coordinator, promotion, and publication secrets
  on the organisation's normal schedule. Coordinate rotations, verify safe probes, and
  revoke old credentials. Do not routinely rotate `VOLUNTEER_MARKER_SECRET`; changing it
  requires the signed-marker migration described above.
- Re-run the physical GPU canary after changes to the CUDA base, locked dependencies,
  NVIDIA toolkit, or host driver. Human or independent verification of the running image
  remains required: the configured digest is provenance carried by a lease, not runtime
  attestation.
