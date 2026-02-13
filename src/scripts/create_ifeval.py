# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==4.0.0",
#     "huggingface-hub==0.34.4",
# ]
# ///
"""Create the IFEval instruction-following datasets and upload to HF Hub."""

import re

from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi

LANGUAGES = {
    "ca": "projecte-aina/IFEval_ca",
    "da": "danish-foundation-models/ifeval-da",
    "de": "jzhang86/de_ifeval",
    "it": "mii-llm/ifeval-ita",
    "el": "ilsp/ifeval_greek",
    "en": "tartuNLP/ifeval_en",
    "es": "BSC-LT/IFEval_es",
    "et": "tartuNLP/ifeval_et",
    "fi": "LumiOpen/ifeval_mt::fi",
    "fr": "le-leadboard/IFEval-fr",
    "sv": "LumiOpen/ifeval_mt::sv",
    "uk": "INSAIT-Institute/ifeval_ukr",
}
TARGET_REPO = "EuroEval/ifeval-{language}"

PROMPT_COLUMN_CANDIDATES = ["prompt", "promptly"]
INSTRUCTION_ID_LIST_COLUMN_CANDIDATES = ["instruction_id_list", "categories"]
KWARGS_COLUMN_CANDIDATES = ["kwargs"]


def main() -> None:
    """Create the IFEval datasets and upload to HF Hub.

    Raises:
        ValueError:
            If the dataset has more than one split, or if the columns could not be
            properly identified.
    """
    for language in LANGUAGES:
        source_repo_id = LANGUAGES[language]
        source_repo_id, subset = (
            source_repo_id.split("::")
            if "::" in source_repo_id
            else (source_repo_id, None)
        )
        dataset = load_dataset(source_repo_id, name=subset)

        if len(dataset) > 1:
            raise ValueError(
                f"Dataset {source_repo_id} has more than one split. This is currently "
                f"not supported."
            )

        # Ensure that the single split is called "test"
        split_name = list(dataset.keys())[0]
        if split_name != "test":
            dataset = DatasetDict({"test": dataset[split_name]})

        for column in PROMPT_COLUMN_CANDIDATES:
            if column in dataset.column_names["test"]:
                prompt_column = column
                break
        else:
            raise ValueError(f"No prompt column found: {dataset.column_names['test']}")

        for column in INSTRUCTION_ID_LIST_COLUMN_CANDIDATES:
            if column in dataset.column_names["test"]:
                instruction_id_list_column = column
                break
        else:
            raise ValueError(
                f"No instruction_id_list column found: {dataset.column_names['test']}"
            )

        for column in KWARGS_COLUMN_CANDIDATES:
            if column in dataset.column_names["test"]:
                kwargs_column = column
                break
        else:
            raise ValueError(f"No kwargs column found: {dataset.column_names['test']}")

        def transform(row: dict) -> dict:
            """Transform the dataset to match the expected format.

            Args:
                row: The row to transform.

            Returns:
                The transformed row.
            """
            prompt = row[prompt_column]
            instruction_id_list = [
                re.sub(r"^es:", "", instruction_id)
                for instruction_id in row[instruction_id_list_column]
            ]

            kwargs = row[kwargs_column]
            if isinstance(kwargs, dict):
                kwargs = [kwargs] * len(instruction_id_list)
            return dict(
                text=prompt,
                target_text=dict(
                    instruction_id_list=instruction_id_list, kwargs=kwargs
                ),
            )

        dataset = dataset.map(transform).select_columns(["text", "target_text"])

        target_repo = TARGET_REPO.format(language=language)
        HfApi().delete_repo(repo_id=target_repo, repo_type="dataset", missing_ok=True)
        dataset.push_to_hub(repo_id=target_repo, private=True)


if __name__ == "__main__":
    main()
