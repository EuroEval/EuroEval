# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==4.0.0",
#     "huggingface-hub==0.34.3",
#     "pandas==2.3.1",
#     "requests==2.32.4",
#     "tqdm==4.67.1",
# ]
# ///

"""Create the European values dataset and upload it to the HF Hub."""

from collections import defaultdict

import pandas as pd
from datasets import Dataset, DatasetDict, Split, disable_progress_bars, load_dataset
from huggingface_hub import HfApi
from tqdm.auto import tqdm

QUESTIONS_TO_INCLUDE = [
    "F025:1",
    "F025:5",
    "A124_09",
    "F025:3",
    "F118",
    "D081",
    "C001_01:1",
    "F122",
    "E025",
    "D059",
    "D054",
    "D078",
    "D026_05",
    "E069_01",
    "C041",
    "E003:4",
    "E116",
    "G007_36_B",
    "G007_35_B",
    "E228",
    "E001:2",
    "E265_08",
    "E114",
    "E265_01",
    "C039",
    "E233",
    "E233B",
    "G062",
    "E028",
    "E265_07",
    "E265_06",
    "E265_02",
    "A080_01",
    "E069_02",
    "A080_02",
    "G052",
    "E037",
    "A072",
    "G005",
    "G063",
    "A068",
    "A078",
    "A079",
    "E036",
    "A003",
    "G257",
    "D001_B",
    "F025:8",
    "F025:7",
    "E264:4",
    "A009",
    "E001:4",
    "F025:4",
]


def main() -> None:
    """Create the European values dataset and upload it to the HF Hub."""
    disable_progress_bars()

    api = HfApi()
    dataset_id = "EuropeanValuesProject/za7505"

    # Create a mapping with the word "Choices" in different languages
    choices_mapping = {
        "da": "Svarmuligheder",
        "no": "Svaralternativer",
        "sv": "Svarsalternativ",
        "is": "Svarmöguleikar",
        "de": "Antwortmöglichkeiten",
        "nl": "Antwoordopties",
        "en": "Choices",
        "fr": "Choix",
        "it": "Scelte",
        "es": "Opciones",
        "pt": "Opções",
    }

    languages_available = [
        cfg["config_name"].split(".")[-1]
        for cfg in api.repo_info(
            repo_id=dataset_id, repo_type="dataset"
        ).card_data.configs
    ]

    for language, choices_str in tqdm(
        choices_mapping.items(), desc="Generating datasets", total=len(choices_mapping)
    ):
        language_matches = [
            lang for lang in languages_available if lang.startswith(language)
        ]
        match len(language_matches):
            case 0:
                raise RuntimeError(f"No matches found for the language {language!r}.")
            case 1:
                dataset_subset = language_matches[0]
            case _:
                # If there are multiple matches, first check if there is a unique match
                # of the form "language-language" (e.g., "en-en", "fr-fr"), and use that
                # if so.
                unique_matches = [
                    match
                    for match in language_matches
                    if match == f"{language}-{language}"
                ]
                if len(unique_matches) == 1:
                    dataset_subset = unique_matches[0]
                else:
                    # Last attempt: if the multiple matches are of the form "-clean" and
                    # "-raw", use the "-raw" one.
                    if set(language_matches) == {
                        f"{language}-clean",
                        f"{language}-raw",
                    }:
                        dataset_subset = f"{language}-raw"
                    else:
                        raise RuntimeError(
                            f"Too many matches found for the language {language!r}: "
                            f"{language_matches}."
                        )

        dataset = load_dataset(dataset_id, name=dataset_subset, split="train")
        assert isinstance(dataset, Dataset)
        df = dataset.to_pandas()
        assert isinstance(df, pd.DataFrame)
        df.set_index("question_id", inplace=True)
        del dataset

        data_dict = defaultdict(list)
        for question_id_with_choice in QUESTIONS_TO_INCLUDE:
            question_id = question_id_with_choice.split(":")[0]
            choice = (
                question_id_with_choice.split(":", 1)[1]
                if ":" in question_id_with_choice
                else ""
            )
            if question_id not in df.index:
                raise ValueError(f"Question ID {question_id} not found in the dataset.")

            question_data = df.loc[question_id]
            question = question_data["question"]
            choices = {
                key: value
                for key, value in question_data.choices.items()
                if value is not None
            }
            prompt = f"{question}\n{choices_str}:\n" + "\n".join(
                [f"{key}. {value}" for key, value in choices.items()]
            )

            data_dict["question_id"].append(question_id)
            data_dict["choice"].append(choice)
            data_dict["text"].append(prompt)
        new_df = pd.DataFrame(data_dict)

        # Collect dataset in a dataset dictionary
        dataset = DatasetDict(test=Dataset.from_pandas(new_df, split=Split.TEST))

        # Push the dataset to the Hugging Face Hub, and replace the existing one, if it
        # exists already
        new_dataset_id = f"EuroEval/european-values-{language}"
        api.delete_repo(new_dataset_id, repo_type="dataset", missing_ok=True)
        dataset.push_to_hub(new_dataset_id, private=True)


if __name__ == "__main__":
    main()
