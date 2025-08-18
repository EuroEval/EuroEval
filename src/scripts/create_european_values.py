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
        "fi": "Vastausvaihtoehdot",
    }
    subset_mapping = {
        "da": "da-dk",
        "no": "no-no",
        "sv": "sv-se",
        "is": "is-is",
        "de": "de-de",
        "nl": "nl-nl",
        "en": "en-raw",
        "fr": "fr-fr",
        "it": "it-it",
        "es": "es-es",
        "pt": "pt-pt",
        "fi": "fi-fi",
    }
    no_yes_mapping = {
        "da": {"0": "Nej", "1": "Ja"},
        "no": {"0": "Nei", "1": "Ja"},
        "sv": {"0": "Nej", "1": "Ja"},
        "is": {"0": "Nei", "1": "Já"},
        "de": {"0": "Nein", "1": "Ja"},
        "nl": {"0": "Nee", "1": "Ja"},
        "en": {"0": "No", "1": "Yes"},
        "fr": {"0": "Non", "1": "Oui"},
        "it": {"0": "No", "1": "Sì"},
        "es": {"0": "No", "1": "Sí"},
        "pt": {"0": "Não", "1": "Sim"},
        "fi": {"0": "Ei", "1": "Kyllä"},
    }

    for language, choices_str in tqdm(
        choices_mapping.items(), desc="Generating datasets", total=len(choices_mapping)
    ):
        dataset = load_dataset(
            path=dataset_id, name=subset_mapping[language], split="train"
        )
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

            # Extract the question and choices
            question_data = df.loc[question_id]
            question = question_data["question"]
            choices = {
                key: value[0].upper() + value[1:]
                for key, value in question_data.choices.items()
                if value is not None
            }

            # Binary choices are stated as "selected" and "not selected", which only
            # makes sense when you're ticking off boxes, so we map them to (the language
            # equivalent of) "yes" and "no"
            if sorted(choices.keys()) == ["0", "1"]:
                choices = no_yes_mapping[language]

            # Create the prompt string, joining the question and choices
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
