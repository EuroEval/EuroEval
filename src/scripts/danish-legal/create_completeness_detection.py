"""Create the legal completeness detection dataset."""

import logging
import random
import re
from pathlib import Path

import click
import pandas as pd
import tqdm
from datasets import Dataset, DatasetDict, Split
from docling.document_converter import DocumentConverter
from huggingface_hub import HfApi

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
COMPLETE_CONTRACT_PROBABILITY = 0.3
NON_REQUIRED_REMOVAL_PROB = 0.3
SYSTEM_PROMPT = (
    "Identificer om følgende kontrakt er komplet eller om der mangler nogle elementer. "
    "Beskriv hvilke elementer der mangler."
)
SECTION_HEADER_PATTERN = r"^## \d+\."

# Hardcoded mapping to the indices in `contract_sections`
# for each category.
REQUIRED_CATEGORIES_TO_INDICES = {
    "Partner/Identitet": [0],
    "Arbejdstid": [6],
    "Ansvarsområde": [2],
    "Ansættelse": [1],
    "Ferie/Fravær": [7, 8, 10, 11, 12, 13],
    "Ophør/Opsigelse": [17],
    "Løn/Vederlag": [3, 4, 5],
    "Klausuler": [18],
    "Personale vilkår": [9],
}


@click.command()
@click.argument("contract_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--num-samples",
    "-n",
    default=10,
    show_default=True,
    help="Number of samples to generate.",
)
def main(contract_path: Path, num_samples: int) -> None:
    """Create the legal completeness detection dataset and upload it to the HF Hub.

    This script modifies the input contract to create a dataset of samples
    with complete and incomplete contracts.
    """
    # Load and process contract
    logger.info(f"Processing contract from {contract_path}")
    if not contract_path.exists():
        click.echo(f"Error: Contract file {contract_path} does not exist", err=True)
        return

    converter = DocumentConverter()
    result = converter.convert(str(contract_path))
    markdown = result.document.export_to_markdown()

    # Parse contract into sections
    contract_sections = parse_contract_into_sections(markdown_text=markdown)

    not_required_indices = [
        i
        for i in range(len(contract_sections))
        if not any(i in indices for indices in REQUIRED_CATEGORIES_TO_INDICES.values())
    ]

    # Generate samples
    logger.info(f"Generating {num_samples} unique samples...")
    samples = generate_unique_samples(
        contract_sections=contract_sections,
        not_required_indices=not_required_indices,
        num_samples=num_samples,
    )

    df = pd.DataFrame(samples)
    dataset = DatasetDict(test=Dataset.from_pandas(df, split=Split.TEST))

    # Remove existing dataset if it exists
    dataset_id = "EuroEval/legal-completeness-detection"
    HfApi().delete_repo(dataset_id, repo_type="dataset", exists_ok=True)

    # Upload dataset
    dataset.push_to_hub(dataset_id, private=True)


def parse_contract_into_sections(markdown_text: str) -> list[str]:
    r"""Parse markdown into sections, excluding subsections with bullet points.

    Splits on all sections as
        `\n\n## 1. Tiltrædelse\n\n`

    And does not split on subsection as
        `\n\n## 1. Daglig administration\n\n-`

    Args:
        markdown_text: The markdown text of the contract

    Returns:
        A list of sections
    """
    sections = re.split(
        r"(?=^## \d+\.\s+(?!.*\n\n-))", markdown_text, flags=re.MULTILINE
    )
    return [section.strip() for section in sections if section.strip()]


def generate_unique_samples(
    contract_sections: list[str], not_required_indices: list[int], num_samples: int
) -> list[dict]:
    """Generate unique contract samples.

    Args:
        contract_sections: The contract as a list of sections
        not_required_indices: list of indices to exclude
        num_samples: Number of samples to generate

    Returns:
        A list of unique contract samples
    """
    samples: list[dict] = []
    seen_contracts = set()
    attempts = 0
    max_attempts = num_samples * 10

    # Use tqdm to show progress
    pbar = tqdm.tqdm(total=num_samples, desc="Generating unique samples")

    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1

        # Generate contract
        if random.random() < COMPLETE_CONTRACT_PROBABILITY:
            contract = _create_complete_contract(
                contract_sections, not_required_indices
            )
            target_text = (
                "Kontrakten er komplet og indeholder alle nødvendige elementer."
            )
        else:
            contract = _create_incomplete_contract(
                contract_sections=contract_sections,
                not_required_indices=not_required_indices,
            )
            target_text = (
                "Kontrakten mangler at beskrive følgende emner: "
                f"{', '.join(contract['missing_categories'])}"
            )

        # Check if this contract is unique
        contract_text = contract["contract_text"]
        if contract_text not in seen_contracts:
            seen_contracts.add(contract_text)

            sample = {
                "text": f"{SYSTEM_PROMPT}\n\n{contract_text}",
                "target_text": target_text,
                "is_complete": contract["is_complete"],
                "missing_categories": contract["missing_categories"],
            }

            samples.append(sample)
            pbar.update(1)

    pbar.close()
    return samples


def _renumber_section_title(section: str, new_number: int) -> str:
    """Renumber a section's title from `## X. Title` to `## new_number. Title`.

    Args:
        section: The section to renumber
        new_number: The new number to renumber the section to

    Returns:
        The renumbered section
    """
    return re.sub(SECTION_HEADER_PATTERN, f"## {new_number}.", section)


def _sample_missing_categories(min_missing: int = 1, max_missing: int = 3) -> list[str]:
    """Sample categories to exclude.

    Args:
        min_missing: The minimum number of categories to remove
        max_missing: The maximum number of categories to remove

    Returns:
        A list of categories to remove
    """
    num_to_remove = random.randint(min_missing, max_missing)
    available_categories = [
        cat for cat, indices in REQUIRED_CATEGORIES_TO_INDICES.items() if indices
    ]
    return random.sample(
        available_categories, min(num_to_remove, len(available_categories))
    )


def _create_complete_contract(
    contract_sections: list[str], not_required_indices: list[int]
) -> dict:
    """Create a complete contract by optionally removing only non-required sections.

    Args:
        contract_sections: The contract as a list of sections
        not_required_indices: list of indices to exclude

    Returns:
        A dictionary with the complete contract text, missing categories,
        excluded indices, and is_complete
    """
    # Optionally remove some non-required sections
    indices_to_exclude = {
        idx
        for idx in not_required_indices
        if random.random() < NON_REQUIRED_REMOVAL_PROB
    }

    # Filter out excluded sections
    remaining_sections = [
        section
        for i, section in enumerate(contract_sections)
        if i not in indices_to_exclude
    ]

    return {
        "contract_text": _renumber_and_join_sections(sections=remaining_sections),
        "missing_categories": [],
        "excluded_indices": list(indices_to_exclude),
        "is_complete": True,
    }


def _create_incomplete_contract(
    contract_sections: list[str], not_required_indices: list[int]
) -> dict:
    """Create an incomplete contract by removing required and optional sections.

    Args:
        contract_sections: The contract as a list of sections
        not_required_indices: list of indices to exclude

    Returns:
        A dictionary with the incomplete contract text, missing categories,
        excluded indices, and is_complete
    """
    # Sample required categories to remove (this makes it incomplete)
    missing_categories = _sample_missing_categories()

    # Get required indices to exclude
    required_indices_to_exclude = _get_indices_to_exclude(
        missing_categories=missing_categories
    )

    # Optionally exclude some non-required sections too
    optional_indices_to_exclude = {
        idx
        for idx in not_required_indices
        if random.random() < NON_REQUIRED_REMOVAL_PROB
    }

    # Combine all exclusions
    all_indices_to_exclude = required_indices_to_exclude.union(
        optional_indices_to_exclude
    )

    # Filter out excluded sections
    remaining_sections = [
        section
        for i, section in enumerate(contract_sections)
        if i not in all_indices_to_exclude
    ]

    return {
        "contract_text": _renumber_and_join_sections(sections=remaining_sections),
        "missing_categories": missing_categories,
        "excluded_indices": list(all_indices_to_exclude),
        "is_complete": False,
    }


def _get_indices_to_exclude(missing_categories: list[str]) -> set[int]:
    """Get all section indices that should be excluded.

    Args:
        missing_categories: list of categories to exclude

    Returns:
        A set of section indices to exclude
    """
    indices_to_exclude = set()
    for category in missing_categories:
        indices_to_exclude.update(REQUIRED_CATEGORIES_TO_INDICES[category])
    return indices_to_exclude


def _renumber_and_join_sections(sections: list[str]) -> str:
    """Renumber sections and join them into a complete contract.

    Args:
        sections: list of sections to renumber and join

    Returns:
        A string of the renumbered and joined sections
    """
    renumbered_sections = []
    section_counter = 1

    for section in sections:
        if re.match(SECTION_HEADER_PATTERN, section):
            renumbered_section = _renumber_section_title(
                section=section, new_number=section_counter
            )
            section_counter += 1
        else:
            renumbered_section = section
        renumbered_sections.append(renumbered_section)

    return "\n\n".join(renumbered_sections)


if __name__ == "__main__":
    main()
