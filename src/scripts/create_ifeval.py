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

LANGUAGES = {"en": "tartuNLP/ifeval_en", "et": "tartuNLP/ifeval_et"}
TARGET_REPO = "EuroEval/ifeval-{language}"


def main() -> None:
    """Create the IFEval datasets and upload to HF Hub."""
    for language in LANGUAGES:
        source_repo_id = LANGUAGES[language]
        dataset = load_dataset(source_repo_id)

        def transform(row: dict) -> dict:
            """Transform the dataset to match the expected format.

            Args:
                row: The row to transform.

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

        target_repo = TARGET_REPO.format(language=language)
        HfApi().delete_repo(repo_id=target_repo, repo_type="dataset", missing_ok=True)
        dataset.push_to_hub(repo_id=target_repo, private=True)


if __name__ == "__main__":
    main()
