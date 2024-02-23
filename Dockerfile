FROM python:3.11-slim-bookworm

# Install dependencies
RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get install -y gcc python3-dev && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install scandeval[all]

# Move the existing evaluation results into the container, to avoid re-running the
# evaluation
WORKDIR /project
COPY scandeval_benchmark_results* /project

# Set the environment variable with the evaluation arguments. These can be overridden
# when running the container
ENV args="missing-args"

# Run the script
CMD scandeval $args
