# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==4.0.0",
#     "huggingface-hub==0.34.4",
#     "pandas==2.3.1",
# ]
# ///

"""Create the EU-MMLU knowledge datasets and upload them to the HF Hub.

EU-MMLU is a human-translated subset of the English MMLU dataset, produced by
professional translators at the European Commission's Directorate-General for
Translation together with master's students from the European Master's in Translation
network. It is described in:

  Sánchez-Gijón, P., Valdez, S., Kokkinidou, A., Bellemont, F., Calvo Del Barrio, S., &
  Brasoveanu, M. C. (2026). Building a European Multilingual Evaluation Dataset: The
  MMLU Localisation Project within the EMT Network. Proceedings of the 3rd
  International Conference on New Trends in Translation and Interpreting Technology,
  46-53.
  https://doi.org/10.26615/issn.2815-4711.2026_007
  https://arxiv.org/abs/2607.18432
"""

import pandas as pd
from constants import CHOICES_MAPPING
from datasets import Dataset, DatasetDict, NamedSplit, Split
from huggingface_hub import HfApi, hf_hub_download

# The source is not a proper Hugging Face dataset, but a collection of loose CSV files
# at the root of the repository, so we download and parse them by hand.
REPO_ID = "EC-DGT-AI/EU-MMLU"

# We pin the revision rather than tracking `main`, since the upstream dataset is still
# being extended with more languages and content, and we want this script to keep
# producing exactly the same datasets.
REVISION = "23d6315128e97df241035c419e1dedeac27a612a"

# Mapping from EuroEval language code to the locale used in the source file names. The
# `EN_GB` subset is deliberately absent, as it is the unmodified original English MMLU
# rather than a translation - it is only used to sanity check the answer keys. The
# `GA_IE` subset is absent as Irish is not yet a supported EuroEval language.
LOCALES = {
    "cs": "CS_CZ",
    "de": "DE_DE",
    "el": "EL_GR",
    "fr": "FR_FR",
    "hr": "HR_HR",
    "hu": "HU_HU",
    "it": "IT_IT",
    "lt": "LT_LT",
    "nl": "NL_NL",
    "pl": "PL_PL",
    "pt": "PT_PT",
    "ro": "RO_RO",
    "sk": "SK_SK",
    "sl": "SL_SI",
}

# The columns we use from the source CSV files. We list them explicitly since some of
# the files carry extra unnamed trailing columns.
COLUMNS = [
    "Subject",
    "Split",
    "Index",
    "Question",
    "Choice_0",
    "Choice_1",
    "Choice_2",
    "Choice_3",
    "Answer",
]

# The source stores the answer as the index of the correct choice.
ANSWER_MAPPING = {0: "a", 1: "b", 2: "c", 3: "d"}


def main() -> None:
    """Create the EU-MMLU datasets and upload them to the HF Hub.

    Raises:
        ValueError:
            If a subset is no longer aligned with the English original, or if a training
            split does not contain all of the labels.
    """
    # The English subset is the untranslated original, so we can use its answer keys to
    # check that the translated subsets are still aligned with the source items
    english_df = load_locale_dataframe(locale="EN_GB")
    english_answers = english_df.set_index(["Subject", "Split", "Index"])["Answer"]

    for language, locale in LOCALES.items():
        df = load_locale_dataframe(locale=locale)

        # Check that the answer keys still line up with the English originals, which
        # catches both upstream changes and mistakes in the processing above
        answers = df.set_index(["Subject", "Split", "Index"])["Answer"]
        shared_ids = answers.index.intersection(english_answers.index)
        if len(shared_ids) != len(answers):
            raise ValueError(
                f"The {locale!r} subset has {len(answers) - len(shared_ids):,} items "
                "which do not appear in the English subset."
            )
        if not answers.loc[shared_ids].equals(english_answers.loc[shared_ids]):
            num_disagreements = int(
                (answers.loc[shared_ids] != english_answers.loc[shared_ids]).sum()
            )
            raise ValueError(
                f"The {locale!r} subset has {num_disagreements:,} items whose answer "
                "does not match the English original."
            )

        df = process_dataframe(df=df, language=language)

        # The source subsets are small, so we keep the original MMLU splits rather than
        # subsampling them
        dataset = DatasetDict(
            {
                "train": build_split(df=df, source_split="dev", split=Split.TRAIN),
                "val": build_split(
                    df=df, source_split="validation", split=Split.VALIDATION
                ),
                "test": build_split(df=df, source_split="test", split=Split.TEST),
            }
        )

        # Few-shot examples are picked by cycling through the labels, so the training
        # split has to contain at least one example of each of them
        train_labels = set(dataset["train"]["label"])
        missing_labels = set(ANSWER_MAPPING.values()) - train_labels
        if missing_labels:
            raise ValueError(
                f"The {language!r} training split has no examples with the label(s) "
                f"{', '.join(sorted(missing_labels))}."
            )

        split_sizes = ", ".join(
            f"{len(split):,} {name}" for name, split in dataset.items()
        )
        num_subjects = df["category"].nunique()
        print(f"{language}: {split_sizes} ({num_subjects} subjects)")

        dataset_id = f"EuroEval/eu-mmlu-{language}"

        # Remove the dataset from Hugging Face Hub if it already exists
        HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

        # Push the dataset to the Hugging Face Hub
        dataset.push_to_hub(dataset_id, private=True)


def load_locale_dataframe(locale: str) -> pd.DataFrame:
    """Load and clean the source CSV file for a single locale.

    Args:
        locale:
            The locale to load, such as "NL_NL".

    Returns:
        The cleaned dataframe, with one row per item.

    Raises:
        ValueError:
            If the subset contains missing values.
    """
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"eu_mmlu_{locale}.csv",
        repo_type="dataset",
        revision=REVISION,
    )

    # The files are encoded with a byte order mark, which we have to strip to avoid it
    # ending up in the name of the first column
    df = pd.read_csv(path, encoding="utf-8-sig")[COLUMNS]

    # Some files store the index as an integer and others as a string, and one of them
    # has an index which has been overwritten with a stray fragment of the translation
    # ("12 másodperc"), so we normalise the index to its leading integer to allow the
    # subsets to be compared across locales
    df["Index"] = df["Index"].astype(str).str.extract(r"^(\d+)", expand=False)
    if df["Index"].isna().any():
        raise ValueError(f"The {locale!r} subset has items without a numeric index.")

    # A handful of items appear twice, with the duplicate being a leftover from another
    # language, so we only keep the first occurrence of each item
    df = df.drop_duplicates(subset=["Subject", "Split", "Index"], keep="first")

    if df.isna().any(axis=None):
        raise ValueError(f"The {locale!r} subset contains missing values.")

    return df.reset_index(drop=True)


def process_dataframe(df: pd.DataFrame, language: str) -> pd.DataFrame:
    """Convert a cleaned source dataframe to the EuroEval format.

    Note that, unlike the other MMLU creation scripts, we do not filter out items based
    on their length or repetitiveness. Those filters exist to catch machine translation
    artifacts, whereas this dataset was translated by professional translators, so here
    they would only remove valid items - in particular short questions and the history
    items which quote a longer source passage.

    Args:
        df:
            The cleaned source dataframe.
        language:
            The EuroEval language code of the subset.

    Returns:
        The dataframe, with `text`, `label`, `category` and `Split` columns.
    """
    df = df.copy()

    df["label"] = df["Answer"].map(ANSWER_MAPPING)

    # Make a `text` column with all the options in it
    df["text"] = [
        row["Question"].replace("\n", " ").strip() + "\n"
        f"{CHOICES_MAPPING[language]}:\n"
        "a. " + row["Choice_0"].replace("\n", " ").strip() + "\n"
        "b. " + row["Choice_1"].replace("\n", " ").strip() + "\n"
        "c. " + row["Choice_2"].replace("\n", " ").strip() + "\n"
        "d. " + row["Choice_3"].replace("\n", " ").strip()
        for _, row in df.iterrows()
    ]

    # Keep the MMLU subject as the category, for parity with the other MMLU datasets
    df["category"] = df["Subject"]

    return df[["text", "label", "category", "Split"]]


def build_split(df: pd.DataFrame, source_split: str, split: NamedSplit) -> Dataset:
    """Build a single split of the dataset.

    Args:
        df:
            The processed dataframe, containing all splits.
        source_split:
            The name of the split in the source dataset.
        split:
            The EuroEval split to build.

    Returns:
        The split, as a Hugging Face dataset.

    Raises:
        ValueError:
            If the split ends up being empty.
    """
    split_df = df[df["Split"] == source_split].drop(columns="Split")
    split_df = split_df.drop_duplicates().reset_index(drop=True)

    if split_df.empty:
        raise ValueError(f"The {source_split!r} split is empty.")

    return Dataset.from_pandas(split_df, split=split)


if __name__ == "__main__":
    main()
