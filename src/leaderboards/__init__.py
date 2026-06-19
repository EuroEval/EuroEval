"""Leaderboards for the LLM evaluation framework EuroEval."""

import os

from dotenv import load_dotenv

# Loads environment variables
load_dotenv()


# Set the HF_TOKEN env var to copy the HUGGINGFACE_API_KEY env var, so the rest of the
# codebase only needs to read HF_TOKEN. We accept HUGGINGFACE_API_KEY as an alias but
# only document HF_TOKEN.
if os.getenv("HUGGINGFACE_API_KEY"):
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_API_KEY"]
