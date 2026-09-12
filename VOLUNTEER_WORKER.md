# EuroEval volunteer GPU worker

This guide is for volunteers who want to contribute GPU time. You do not need to set
project environment variables or provide GitHub, Hugging Face, or Upstash credentials.
EuroEval maintainers should use the
[volunteer worker operations guide](VOLUNTEER_WORKER_OPERATIONS.md) instead.

A worker claims one compatible evaluation at a time, downloads the pinned model and
dataset, evaluates one language, and uploads records to private staging. It does not
publish results directly to the public leaderboard.

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

Both commands should list your GPU. If the second command fails, finish configuring the
NVIDIA Container Toolkit before continuing.

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

On the first run, open the GitHub URL printed in the terminal, enter the short code, and
approve access. The worker then starts evaluating compatible queued work and continues
until you press Ctrl-C. If no work is available, it waits and checks again.

The named Docker volume preserves authentication, model downloads, and unfinished work
between runs. To contribute again later, run the same `docker run` command; you normally
will not need to authenticate again.

Do not add `--privileged`, host networking, the Docker socket, a home-directory mount,
or project credentials. The worker never needs your `HF_TOKEN`, GitHub token, or Upstash
credentials.

### Optional: pin the exact image version

To pin a deployment, replace the `IMAGE=...:latest` line in step 2 with these lines,
then run the rest of that same block. The worker still starts only once, using the
immutable digest reference:

```sh
docker pull ghcr.io/euroeval/euroeval-worker:latest
IMAGE="$(docker image inspect ghcr.io/euroeval/euroeval-worker:latest \
  --format '{{index .RepoDigests 0}}')"
```

Record the resulting `ghcr.io/euroeval/euroeval-worker@sha256:...` value for later runs
if the deployment must remain pinned.

## How volunteer evaluation works

1. The worker completes the broker-mediated GitHub device flow. The OAuth login must
   be assignable to issues in the EuroEval repository; otherwise the worker stops with
   instructions to use an eligible GitHub account.
2. It reports NVIDIA hardware and asks for a compatible queued model/language.
3. The broker anonymously verifies that the queued Hugging Face model is public,
   ungated, pinned to an immutable revision, safetensors-only, free of repository Python
   and remote-code configuration, within the reported disk/GPU limits, and exposes one
   supported EuroEval capability (`encoder` or `generative`).
4. The worker independently refetches the same immutable metadata and capability before
   evaluation, then evaluates with remote code disabled and safetensors required.
5. Each record is uploaded idempotently to private Hugging Face staging.
6. The broker validates the complete lease and marks it ready for review.

Issue assignees are the mutable, authoritative active evaluators and credit
identities for both manual and volunteer work. Maintainers can transfer ownership or
credit by changing assignees; that fences old volunteer leases. Multiple assignees are
valid when multiple volunteers contribute different languages, and accepted assignees
are retained. An assignee is removed only after rejection or intentional release.
Signed marker contributors are audit and integrity evidence only, not authoritative
identity.

A worker can be interrupted safely.
The local state and exact result bytes remain in the
named volume. Restarting before lease expiry resumes the same lease; after expiry, the
broker may lease the language again and the old local state is archived. An interrupt
does not immediately release an active lease. Submitted records can be retried without
changing their digest. A `No volunteer evaluation work` message is normal when the queue
is empty. `No NVIDIA GPU` means that the host runtime did not expose a usable GPU.

## Data and credential handling

The `/cache` volume is private worker state. Back it up only if the backup is protected
as a secret, and remove it when decommissioning a worker. To discard a worker's local
credential and cached data:

```sh
docker volume rm euroeval-worker-cache
```

Revoking or replacing the broker credential is an operator action; deleting the volume
does not revoke a credential already issued by the broker. Contact the EuroEval
maintainer before decommissioning a registered contributor worker.
