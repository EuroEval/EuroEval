# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
#     "scikit-learn<1.6.0",
# ]
# ///

"""Create the Belebele-mini datasets and upload them to the HF Hub."""

import re
from collections import Counter

import pandas as pd
from constants import (
    MAX_NUM_CHARS_IN_INSTRUCTION,
    MAX_NUM_CHARS_IN_OPTION,
    MAX_REPETITIONS,
    MIN_NUM_CHARS_IN_INSTRUCTION,
    MIN_NUM_CHARS_IN_OPTION,
)
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from requests import HTTPError
from sklearn.model_selection import train_test_split


def main() -> None:
    """Create the Belebele-mini datasets and upload them to the HF Hub."""
    repo_id = "facebook/belebele"

    language_mapping = {
        "da": "dan_Latn",
        "no": "nob_Latn",
        "sv": "swe_Latn",
        "is": "isl_Latn",
        "de": "deu_Latn",
        "nl": "nld_Latn",
        "en": "eng_Latn",
        "fr": "fra_Latn",
        "fi": "fin_Latn",
        "it": "ita_Latn",
        "es": "spa_Latn",
    }
    text_mapping = {
        "da": "Tekst",
        "no": "Tekst",
        "sv": "Text",
        "is": "Texti",
        "de": "Text",
        "nl": "Tekst",
        "en": "Text",
        "fr": "Texte",
        "fi": "Teksti",
        "it": "Testo",
        "es": "Texto",
    }
    question_mapping = {
        "da": "Spørgsmål",
        "no": "Spørsmål",
        "sv": "Fråga",
        "is": "Spurning",
        "de": "Fragen",
        "nl": "Vraag",
        "en": "Question",
        "fr": "Question",
        "fi": "Kysymys",
        "it": "Domanda",
        "es": "Pregunta",
    }
    choices_mapping = {
        "da": "Svarmuligheder",
        "no": "Svaralternativer",
        "sv": "Svarsalternativ",
        "is": "Svarmöguleikar",
        "de": "Antwortmöglichkeiten",
        "nl": "Antwoordopties",
        "en": "Choices",
        "fr": "Choix",
        "fi": "Vaihtoehdot",
        "it": "Opzioni",
        "es": "Opciones",
    }

    for language in choices_mapping.keys():
        # Download the dataset
        dataset = load_dataset(
            path=repo_id, name=language_mapping[language], split="test", token=True
        )
        assert isinstance(dataset, Dataset)

        # Convert the dataset to a dataframe
        df = dataset.to_pandas()
        assert isinstance(df, pd.DataFrame)

        # Rename the columns
        df.rename(
            columns=dict(
                correct_answer_num="label",
                mc_answer1="option_a",
                mc_answer2="option_b",
                mc_answer3="option_c",
                mc_answer4="option_d",
            ),
            inplace=True,
        )

        # Convert the label to letters
        label_mapping = {"1": "a", "2": "b", "3": "c", "4": "d"}
        df.label = df.label.map(label_mapping)

        # Create the instruction from the passage and question
        df["instruction"] = (
            text_mapping[language]
            + ": "
            + df.flores_passage
            + "\n"
            + question_mapping[language]
            + ": "
            + df.question
        )

        # Remove the samples with overly short or long texts
        df = df[
            (df.instruction.str.len() >= MIN_NUM_CHARS_IN_INSTRUCTION)
            & (df.instruction.str.len() <= MAX_NUM_CHARS_IN_INSTRUCTION)
            & (df.option_a.str.len() >= MIN_NUM_CHARS_IN_OPTION)
            & (df.option_a.str.len() <= MAX_NUM_CHARS_IN_OPTION)
            & (df.option_b.str.len() >= MIN_NUM_CHARS_IN_OPTION)
            & (df.option_b.str.len() <= MAX_NUM_CHARS_IN_OPTION)
            & (df.option_c.str.len() >= MIN_NUM_CHARS_IN_OPTION)
            & (df.option_c.str.len() <= MAX_NUM_CHARS_IN_OPTION)
            & (df.option_d.str.len() >= MIN_NUM_CHARS_IN_OPTION)
            & (df.option_d.str.len() <= MAX_NUM_CHARS_IN_OPTION)
        ]

        def is_repetitive(text: str) -> bool:
            """Return True if the text is repetitive."""
            max_repetitions = max(Counter(text.split()).values())
            return max_repetitions > MAX_REPETITIONS

        # Remove overly repetitive samples
        df = df[
            ~df.instruction.apply(is_repetitive)
            & ~df.option_a.apply(is_repetitive)
            & ~df.option_b.apply(is_repetitive)
            & ~df.option_c.apply(is_repetitive)
            & ~df.option_d.apply(is_repetitive)
        ]

        # Make a `text` column with all the options in it
        df["text"] = [
            re.sub(r"\n+", "\n", row.instruction).strip() + "\n"
            f"{choices_mapping[language]}:\n"
            "a. " + re.sub(r"\n+", "\n", row.option_a).strip() + "\n"
            "b. " + re.sub(r"\n+", "\n", row.option_b).strip() + "\n"
            "c. " + re.sub(r"\n+", "\n", row.option_c).strip() + "\n"
            "d. " + re.sub(r"\n+", "\n", row.option_d).strip()
            for _, row in df.iterrows()
        ]

        # Only keep the `text` and `label` columns
        df = df[["text", "label"]]

        # Remove duplicates
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Create validation split
        val_size = 64
        traintest_arr, val_arr = train_test_split(
            df, test_size=val_size, random_state=4242
        )
        traintest_df = pd.DataFrame(traintest_arr, columns=df.columns)
        val_df = pd.DataFrame(val_arr, columns=df.columns)

        # Create train and test split
        train_size = 256
        train_arr, test_arr = train_test_split(
            traintest_df, train_size=train_size, random_state=4242
        )
        train_df = pd.DataFrame(train_arr, columns=df.columns)
        test_df = pd.DataFrame(test_arr, columns=df.columns)

        # Reset the index
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        # Collect datasets in a dataset dictionary
        dataset = DatasetDict(
            train=Dataset.from_pandas(train_df, split=Split.TRAIN),
            val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
            test=Dataset.from_pandas(test_df, split=Split.TEST),
        )

        # Create dataset ID
        if language == "en":
            dataset_id = "EuroEval/belebele-mini"
        else:
            dataset_id = f"EuroEval/belebele-{language}-mini"

        # Remove the dataset from Hugging Face Hub if it already exists
        try:
            api = HfApi()
            api.delete_repo(dataset_id, repo_type="dataset")
        except HTTPError:
            pass

        # Push the dataset to the Hugging Face Hub
        dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
