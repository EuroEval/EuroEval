# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==4.0.0",
#     "huggingface-hub==0.34.4",
# ]
# ///
"""Create the IFEval instruction-following datasets and upload to HF Hub."""

import json

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
    "fr": "json:https://raw.githubusercontent.com/lightblue-tech/M-IFEval/refs/heads/main/data/fr_input_data.jsonl",
    "pt": "facebook/Multi-IF?language=Portuguese",
    "sv": "LumiOpen/ifeval_mt::sv",
    "uk": "INSAIT-Institute/ifeval_ukr",
}
TARGET_REPO = "EuroEval/ifeval-{language}"

PROMPT_COLUMN_CANDIDATES = ["prompt", "promptly", "turn_1_prompt"]
INSTRUCTION_ID_LIST_COLUMN_CANDIDATES = [
    "instruction_id_list",
    "categories",
    "turn_1_instruction_id_list",
]
KWARGS_COLUMN_CANDIDATES = ["kwargs", "turn_1_kwargs"]


def main() -> None:
    """Create the IFEval datasets and upload to HF Hub.

    Raises:
        ValueError:
            If the dataset has more than one split, or if the columns could not be
            properly identified.
    """
    for language in LANGUAGES:
        source_repo_id = LANGUAGES[language]

        # Extract the source repo ID, subset and arguments
        if "::" in source_repo_id:
            if "?" in source_repo_id:
                source_repo_id, args = source_repo_id.split("?")
                args, subset = args.split("::")
            else:
                source_repo_id, subset = source_repo_id.split("::")
                args = None
        elif "?" in source_repo_id:
            source_repo_id, args = source_repo_id.split("?")
            subset = None
        else:
            args = None
            subset = None

        if source_repo_id.startswith("json:"):
            source_repo_id = source_repo_id[len("json:") :]
            dataset = load_dataset("json", data_files={"test": source_repo_id})
        else:
            dataset = load_dataset(source_repo_id, name=subset)

        if len(dataset) > 1:
            raise ValueError(
                f"Dataset {source_repo_id} has more than one split. This is currently "
                f"not supported."
            )

        if args is not None:
            filter_dict = {
                arg.split("=")[0]: arg.split("=")[1] for arg in args.split("&")
            }
            dataset = dataset.filter(
                lambda x: all(x[arg] == value for arg, value in filter_dict.items())
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
            instruction_id_list = row[instruction_id_list_column]

            # Ensure that the prompt is a non-JSON string
            try:
                prompt = json.loads(prompt)
            except json.JSONDecodeError:
                # If the prompt is not valid JSON, keep the original string value.
                pass
            if isinstance(prompt, dict):
                prompt = prompt["content"]

            # Ensure that the instruction_id_list is a list of strings
            if isinstance(instruction_id_list, str):
                instruction_id_list = json.loads(instruction_id_list)

            # Ensure that kwargs is a list of dicts
            kwargs = row[kwargs_column]
            if isinstance(kwargs, str):
                kwargs = json.loads(kwargs)
            if any(isinstance(kwarg, str) for kwarg in kwargs):
                kwargs = [json.loads(kwarg) for kwarg in kwargs]
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
