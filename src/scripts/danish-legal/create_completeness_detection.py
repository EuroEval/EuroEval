# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.34.4",
#     "pandas==2.2.0",
#     "tqdm==4.66.3",
# ]
# ///

"""Create the legal completeness detection dataset."""

import hashlib
import random
import re
import typing as t

import pandas as pd
from contract_utils import COMPLETE_CONTRACT_EXAMPLE
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from tqdm.auto import tqdm

NUM_SAMPLES = 512
COMPLETE_CONTRACT_PROBABILITY = 0.3
MAX_REQUIRED_ELEMENTS_TO_REMOVE = 3
NICE_TO_HAVE_ELEMENT_REMOVAL_PROBABILITY = 0.3
SECTION_IDX_TAG = "[SECTION-IDX]"


REQUIRED_ELEMENTS = {
    "company-name": "Manglende angivelse af virksomhedens navn",
    "location": "Manglende angivelse af arbejdsstedets beliggenhed eller hvor "
    "arbejdet udføres",
    "title": "Manglende beskrivelse af medarbejderes stilling, titel, rang eller "
    "jobkategori",
    "start-date": "Manglende angivelse af ansættelsesforholdets begyndelsestidspunkt",
    "termination": "Manglende vilkår for opsigelse med varsler fra både ansatte og "
    "virksomhed",
    "salary": "Manglende angivelse af ansattes løn",
    "work-time": "Manglende angivelse af den ansattes arbejdstid",
    "social-security": "Manglende angivelse af hvilke bidrag til social sikring, som "
    "er knyttet til ansættelsesforholdet",
}

NICE_TO_HAVE_ELEMENTS = {
    "trial-period": "Manglende angivelse af vilkår og varihed for en prøvetid for "
    "ansættelse",
    "holiday": "Manglende definition af retten til ferie",
    "extra-holiday": "Manglende definition af retten til feriefridage",
    "illness": "Manglende beskrivelse af rettigheder i forbindelse med sygdom",
    "parental-leave": "Manglende beskrivelse af rettigheder i forbindelse med "
    "graviditet og barsel",
    "collective-agreement": "Manglende angivelse af hvilken overneskomst og aftale "
    "som ansættelsesforholdet er omfattet af",
    "pension": "Manglende definition af pensionsvilkår",
    "training": "Manglende angivelse af retten til efteruddannelse",
}


def main() -> None:
    """Create the legal completeness detection dataset and upload it to the HF Hub."""
    samples: list[dict[str, t.Any]] = list()
    hashes: set[str] = set()
    with tqdm(total=NUM_SAMPLES, desc="Generating samples") as pbar:
        while len(samples) < NUM_SAMPLES:
            contract, missing_required_elements, missing_nice_to_have_elements = (
                generate_contract()
            )
            contract_hash = hashlib.md5(contract.encode("utf-8")).hexdigest()
            if contract_hash in hashes:
                continue

            hashes.add(contract_hash)

            if len(missing_required_elements) == 0:
                label = "Kontrakten er fuldstændig."
            else:
                label = "Manglende lovpligtige elementer:\n" + "\n".join(
                    f"{idx}. {element}"
                    for idx, element in enumerate(missing_required_elements, start=1)
                )
            if len(missing_nice_to_have_elements) > 0:
                label += "\n\nManglende ikke-lovpligtige elementer:\n" + "\n".join(
                    f"{idx}. {element}"
                    for idx, element in enumerate(
                        missing_nice_to_have_elements, start=10
                    )
                )

            sample = dict(
                text=contract,
                label=label,
                missing_required_elements=missing_required_elements,
                missing_nice_to_have_elements=missing_nice_to_have_elements,
                num_missing_required_elements=len(missing_required_elements),
                num_missing_nice_to_have_elements=len(missing_nice_to_have_elements),
                is_complete=len(missing_required_elements) == 0,
            )
            samples.append(sample)
            pbar.update(1)

    # Create the dataset from the records
    df = pd.DataFrame.from_records(samples)
    split = Dataset.from_pandas(df, split=Split.TEST)
    dataset = DatasetDict(test=split)

    # Upload dataset
    dataset_id = "EuroEval/legal-completeness-detection"
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)


def generate_contract() -> tuple[str, list[str], list[str]]:
    """Generate a contract.

    Returns:
        A tuple (contract, missing_required_elements, missing_nice_to_have_elements),
        where `contract` is the generated contract, and the other two lists contain the
        missing required and nice-to-have elements, respectively.
    """
    contract = COMPLETE_CONTRACT_EXAMPLE
    missing_required_elements = []
    missing_nice_to_have_elements = []

    # Decide if the contract should be complete or not
    is_complete = random.random() < COMPLETE_CONTRACT_PROBABILITY

    # If not complete, remove some required elements
    if not is_complete:
        num_elements_to_remove = random.randint(1, MAX_REQUIRED_ELEMENTS_TO_REMOVE)
        elements_to_remove = random.sample(
            list(REQUIRED_ELEMENTS.keys()), num_elements_to_remove
        )
        for element in elements_to_remove:
            contract = re.sub(
                rf"<{element}>.*?</{element}>", "", contract, flags=re.DOTALL
            )
            missing_required_elements.append(REQUIRED_ELEMENTS[element])

    # Randomly remove some nice-to-have elements
    for element, description in NICE_TO_HAVE_ELEMENTS.items():
        if random.random() < NICE_TO_HAVE_ELEMENT_REMOVAL_PROBABILITY:
            contract = re.sub(
                rf"<{element}>.*?</{element}>", "", contract, flags=re.DOTALL
            )
            missing_nice_to_have_elements.append(description)

    # Clean up the contract, which removes the remaining XML tags, sets up the section
    # numbering, and general whitespace cleanup
    contract = clean_up_contract(contract=contract)

    return contract, missing_required_elements, missing_nice_to_have_elements


def clean_up_contract(contract: str) -> str:
    """Clean up the contract by removing extra newlines and spaces.

    Args:
        contract:
            The contract to clean up.

    Returns:
        The cleaned up contract.
    """
    # Remove all XML tags that might be left
    contract = re.sub(pattern=r"</?[^>]+>", repl="", string=contract, flags=re.DOTALL)

    # Remove multiple newlines
    contract = re.sub(pattern=r"(\s*\n\s*){2,}", repl="\n\n", string=contract)

    # Remove multiple spaces
    contract = re.sub(pattern=r" {2,}", repl=" ", string=contract)

    # Strip leading and trailing whitespace
    contract = contract.strip()

    # Set up section numbering
    section_idx = 1
    while SECTION_IDX_TAG in contract:
        contract = contract.replace(SECTION_IDX_TAG, str(section_idx), 1)
        section_idx += 1

    return contract


if __name__ == "__main__":
    main()
