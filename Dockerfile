FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# If CUDA is not available, raise an error during build
RUN if ! command -v nvidia-smi >/dev/null 2>&1; then \
      echo "CUDA is not available. Please build this Docker image on a machine with CUDA."; \
      exit 1; \
    fi

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y update && \
    apt-get -y upgrade && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      gcc git-all python3.12 python3.12-venv python3.12-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /project

RUN python3.12 -m venv .venv && \
    .venv/bin/python3.12 -m pip install --upgrade pip wheel && \
    .venv/bin/python3.12 -m pip install --upgrade euroeval[all]

COPY euroeval_benchmark_results.jsonl* .

ENTRYPOINT [".venv/bin/euroeval"]
CMD ["--help"]
