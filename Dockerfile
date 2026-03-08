FROM nvidia/cuda:13.1.1-base-ubuntu24.04

RUN apt-get -y update && \
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
