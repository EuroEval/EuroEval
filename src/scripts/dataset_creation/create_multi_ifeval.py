# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==4.0.0",
#     "huggingface-hub==0.34.4",
# ]
# ///
"""Create the IFEval instruction-following datasets and upload to HF Hub."""

from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm.auto import tqdm

SUBSET_LANGUAGES: list[str] = [
    "sq",
    "be",
    "bs",
    "bg",
    "ca",
    "hr",
    "cs",
    "da",
    "nl",
    "en",
    "et",
    "fo",
    "fi",
    "fr",
    "de",
    "el",
    "hu",
    "is",
    "it",
    "lv",
    "lt",
    "lb",
    "no",
    "nn",
    "pl",
    "pt-pt",
    "ro",
    "sr",
    "sk",
    "sl",
    "es",
    "sv",
    "uk",
]
SOURCE_REPO = "alexandrainst/multi-ifeval"
TARGET_REPO = "EuroEval/multi-ifeval-{language}"


def main() -> None:
    """Create the MultiIFEval datasets and upload to HF Hub."""
    for subset_name in tqdm(SUBSET_LANGUAGES, desc="Creating datasets"):
        dataset = load_dataset(SOURCE_REPO, name=subset_name)

        def transform(row: dict) -> dict:
            """Transform the dataset to match the expected format.

            Args:
                row:
                    The row to transform.

            Returns:
                The transformed row.
            """
            return dict(
                text=row["prompt"],
                target_text=dict(
                    instruction_id_list=row["instruction_id_list"], kwargs=row["kwargs"]
                ),
            )

        dataset = dataset.map(transform).select_columns(["text", "target_text"])

        target_repo = TARGET_REPO.format(language=subset_name.split("-")[0])
        HfApi().delete_repo(repo_id=target_repo, repo_type="dataset", missing_ok=True)
        dataset.push_to_hub(repo_id=target_repo, private=True)


if __name__ == "__main__":
    main()
