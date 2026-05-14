"""Generate all leaderboards."""

import logging
import re
import typing as t
import warnings
from pathlib import Path

import click
from dotenv import load_dotenv

from leaderboards.leaderboard_generation import generate_leaderboard
from leaderboards.result_processing import process_results

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s ⋅ %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

warnings.simplefilter(action="ignore", category=RuntimeWarning)

load_dotenv()


# Constants for leaderboard generation
MINIMUM_VERSION: str = "15.0.0"
MINIMUM_NUMBER_OF_MODEL_RECORDS: int = 7
BANNED_VERSIONS: list[str] = ["9.3.0", "10.0.0"]
BANNED_MODEL_PATTERNS: list[re.Pattern] = [
    re.compile("^meta-llama/Llama-3.1-405B-Instruct$"),  # Temporary ban
    re.compile("^utter-project/EuroVLM-9B-Preview$"),  # Temporary ban
]
API_MODEL_PATTERNS: list[re.Pattern] = [
    re.compile(r"gemini/.*"),
    re.compile(r"(openai/)?gpt-[456789].*"),
    re.compile(r"(anthropic/)?claude.*"),
    re.compile(r"(xai/)?grok.*"),
]
OPEN_SOURCE_MODEL_PATTERNS: list[re.Pattern] = []
TRAINED_FROM_SCRATCH_PATTERNS: list[re.Pattern] = [
    re.compile(r"Qwen/.*"),
    re.compile(r"google/.*"),
    re.compile(r"mistralai/.*"),
    re.compile(r"meta-llama/.*"),
    re.compile(r"facebook/.*"),
    re.compile(r"FacebookAI/.*"),
    re.compile(r"zai-org/.*"),
    re.compile(r"deepseek-ai/.*"),
    re.compile(r"PleIAs/.*"),
    re.compile(r"openai/.*"),
    re.compile(r"nvidia/.*"),
    re.compile(r"allenai/.*"),
    re.compile(r"utter-project/.*"),
    re.compile(r"CohereLabs/.*"),
    re.compile(r"speakleash/.*"),
    re.compile(r"yulan-team/.*"),
    re.compile(r"BSC-LT/.*"),
    re.compile(r"tencent/.*"),
    re.compile(r"LiquidAI/.*"),
    re.compile(r"HuggingFaceTB/.*"),
    re.compile(r"tiiuae/.*"),
    re.compile(r"AIDC-AI/.*"),
    re.compile(r"inclusionAI/.*"),
    re.compile(r"jhu-clsp/.*"),
    re.compile(r"vesteinn/(Dansk|Fo|Scandi)BERT.*"),
    re.compile(r"EuropeanParliament/EUBERT"),
    re.compile(r"microsoft/.*"),
    re.compile(r"EuroBERT/.*"),
    re.compile(r"fresh-.*"),
    re.compile(r"answerdotai/.*"),
    re.compile(r".*-scratch"),
]


@click.command()
@click.option(
    "--categories",
    "-c",
    default=["all", "nlu"],
    multiple=True,
    help="Categories to generate leaderboards for. Defaults to 'all' and 'nlu'.",
)
@click.option(
    "--force/--no-force",
    "-f",
    default=False,
    show_default=True,
    help="Force the generation of the leaderboard, even if no updates are found.",
)
def main(categories: tuple[t.Literal["all", "nlu"]], force: bool) -> None:
    """Generate all leaderboards.

    Args:
        categories:
            Categories to generate leaderboards for. Defaults to 'all' and 'nlu'.
        force:
            Whether to force the generation of the leaderboard, even if no updates are
            found.
    """
    process_results(
        min_version=MINIMUM_VERSION,
        min_number_of_model_records=MINIMUM_NUMBER_OF_MODEL_RECORDS,
        banned_versions=BANNED_VERSIONS,
        banned_model_patterns=BANNED_MODEL_PATTERNS,
        api_model_patterns=API_MODEL_PATTERNS,
        trained_from_scratch_patterns=TRAINED_FROM_SCRATCH_PATTERNS,
    )
    all_leaderboard_configs = Path("leaderboard_configs").glob("*.yaml")
    for config_path in all_leaderboard_configs:
        generate_leaderboard(
            leaderboard_config_path=config_path,
            categories=list(categories),
            force=force,
        )


if __name__ == "__main__":
    main()
