# EuroEval volunteer GPU worker

This guide is for volunteers who want to contribute GPU time. You do not need to set
project environment variables or provide GitHub, Hugging Face, or Upstash credentials.
EuroEval maintainers should use the
[volunteer worker operations guide](VOLUNTEER_WORKER_OPERATIONS.md) instead.

A worker claims one compatible model/language evaluation at a time, downloads the pinned
model and datasets, and uploads records to private staging. The broker skips results
already covered by the same model revision at the requested or a newer EuroEval version.
The worker evaluates only the missing official datasets. It does not publish results
directly to the public leaderboard.

## Contributor quick start

You need a Linux x86_64/amd64 or ARM64/aarch64 computer with an NVIDIA GPU, an NVIDIA
[driver](https://www.nvidia.com/en-us/drivers/) compatible with the image's CUDA 13
stack, [Docker Engine](https://docs.docker.com/engine/install/), and the [NVIDIA
Container Toolkit][toolkit]. NVIDIA DGX Spark/GB10 is supported. The published `latest`
image selects the correct architecture automatically; no separate ARM command is needed.

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

Copy and run this block once:

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

On the first run, open the GitHub URL printed in the terminal, enter the short code, and
authorise the shared **EuroEval Volunteer Worker** GitHub App. Complete any GitHub
verification in the browser yourself. No repository or organisation membership is
needed, and the worker never asks you to paste a GitHub token.

The broker can post one `I'd like to evaluate this.` comment under your GitHub account
when needed for issue assignment. It retains your renewable GitHub grant encrypted on
the server; the worker receives only a revocable broker credential.

The worker then evaluates compatible queued work until you press Ctrl-C. If no work is
available, it waits and checks again. To claim at most one evaluation, append `--once`
after the image argument.

The named Docker volume preserves authentication, model downloads, and unfinished work
between runs. To contribute again later, run the same `docker run` command; you normally
will not need to authenticate again.

Do not add `--privileged`, host networking, the Docker socket, a home-directory mount,
or project credentials. The worker never needs your `HF_TOKEN`, GitHub token, or Upstash
credentials.

### Optional: pin the exact image version

Pull the current image and note the `sha256:...` value on its `Digest:` line:

```sh
docker pull ghcr.io/euroeval/euroeval-worker:latest
```

Replace the final `ghcr.io/euroeval/euroeval-worker:latest` argument in step 2 with
`ghcr.io/euroeval/euroeval-worker@sha256:...`, using that exact digest. Record it for
later runs. Do not select the first entry from an existing image's `RepoDigests`: a
cached image can also have older digest references for the same platform image.

A pin does not freeze the broker's supported worker version. If the broker requests a
worker update, pull the newly promoted image and replace the old pin.

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
GPU. Transient claim failures are retried with backoff. For an HTTP `426` worker update
request, pull `latest` again and restart with the same cache volume; do not delete your
saved state.

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
