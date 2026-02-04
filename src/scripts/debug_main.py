"""Script for running main script in debug mode."""

from euroeval.benchmarker import Benchmarker

model = "Qwen/Qwen3-4B-Instruct-2507"

kwargs = {"task": "bfcl", "language": "en", "debug": True}

Benchmarker(**kwargs).benchmark(model=model)
