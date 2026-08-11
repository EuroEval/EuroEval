# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==5.0.0",
#     "huggingface-hub==1.20.1",
#     "pandas==3.0.3",
#     "requests==2.34.2",
#     "scikit-learn==1.6.1",
# ]
# ///

"""Create the zebra puzzle datasets and upload them to the HF Hub."""

import json
import typing as t

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

# All themes except Danish smørrebrød variants
# Format: (theme, language_code, difficulty)
THEMES = [
    # Arabic
    ("ar_manazil_2x3_5rh", "ar", "easy"),
    ("ar_manazil_4x5_5rh", "ar", "hard"),
    # Belarusian
    ("be_damy_2x3_5rh", "be", "easy"),
    ("be_damy_4x5_5rh", "be", "hard"),
    # Bulgarian
    ("bg_kashti_2x3_5rh", "bg", "easy"),
    ("bg_kashti_4x5_5rh", "bg", "hard"),
    # Bosnian
    ("bs_kuce_2x3_5rh", "bs", "easy"),
    ("bs_kuce_4x5_5rh", "bs", "hard"),
    # Catalan
    ("ca_cases_2x3_5rh", "ca", "easy"),
    ("ca_cases_4x5_5rh", "ca", "hard"),
    # Czech
    ("cs_domy_2x3_5rh", "cs", "easy"),
    ("cs_domy_4x5_5rh", "cs", "hard"),
    # Danish (house theme)
    ("da_huse_2x3_5rh", "da", "easy"),
    ("da_huse_4x5_5rh", "da", "hard"),
    # German
    ("de_hauser_2x3_5rh", "de", "easy"),
    ("de_hauser_4x5_5rh", "de", "hard"),
    # Greek
    ("el_spitia_2x3_5rh", "el", "easy"),
    ("el_spitia_4x5_5rh", "el", "hard"),
    # English
    ("en_houses_2x3_5rh", "en", "easy"),
    ("en_houses_4x5_5rh", "en", "hard"),
    # Spanish
    ("es_casas_2x3_5rh", "es", "easy"),
    ("es_casas_4x5_5rh", "es", "hard"),
    # Estonian
    ("et_majad_2x3_5rh", "et", "easy"),
    ("et_majad_4x5_5rh", "et", "hard"),
    # Basque
    ("eu_etxeak_2x3_5rh", "eu", "easy"),
    ("eu_etxeak_4x5_5rh", "eu", "hard"),
    # Finnish
    ("fi_talot_2x3_5rh", "fi", "easy"),
    ("fi_talot_4x5_5rh", "fi", "hard"),
    # Faroese
    ("fo_hus_2x3_5rh", "fo", "easy"),
    ("fo_hus_4x5_5rh", "fo", "hard"),
    # French
    ("fr_maisons_2x3_5rh", "fr", "easy"),
    ("fr_maisons_4x5_5rh", "fr", "hard"),
    # Frisian
    ("fy_huzen_2x3_5rh", "fy", "easy"),
    ("fy_huzen_4x5_5rh", "fy", "hard"),
    # Irish
    ("ga_tithe_2x3_5rh", "ga", "easy"),
    ("ga_tithe_4x5_5rh", "ga", "hard"),
    # Hindi
    ("hi_ghar_2x3_5rh", "hi", "easy"),
    ("hi_ghar_4x5_5rh", "hi", "hard"),
    # Croatian
    ("hr_kuce_2x3_5rh", "hr", "easy"),
    ("hr_kuce_4x5_5rh", "hr", "hard"),
    # Hungarian
    ("hu_hazak_2x3_5rh", "hu", "easy"),
    ("hu_hazak_4x5_5rh", "hu", "hard"),
    # Icelandic
    ("is_husum_2x3_5rh", "is", "easy"),
    ("is_husum_4x5_5rh", "is", "hard"),
    # Italian
    ("it_case_2x3_5rh", "it", "easy"),
    ("it_case_4x5_5rh", "it", "hard"),
    # Japanese
    ("ja_ie_2x3_5rh", "ja", "easy"),
    ("ja_ie_4x5_5rh", "ja", "hard"),
    # Luxembourgish
    ("lb_haiser_2x3_5rh", "lb", "easy"),
    ("lb_haiser_4x5_5rh", "lb", "hard"),
    # Lithuanian
    ("lt_namai_2x3_5rh", "lt", "easy"),
    ("lt_namai_4x5_5rh", "lt", "hard"),
    # Latvian
    ("lv_majas_2x3_5rh", "lv", "easy"),
    ("lv_majas_4x5_5rh", "lv", "hard"),
    # Macedonian
    ("mk_kukji_2x3_5rh", "mk", "easy"),
    ("mk_kukji_4x5_5rh", "mk", "hard"),
    # Marathi
    ("mr_ghare_2x3_5rh", "mr", "easy"),
    ("mr_ghare_4x5_5rh", "mr", "hard"),
    # Norwegian Bokmål
    ("nb_hus_2x3_5rh", "nb", "easy"),
    ("nb_hus_4x5_5rh", "nb", "hard"),
    # Dutch
    ("nl_huizen_2x3_5rh", "nl", "easy"),
    ("nl_huizen_4x5_5rh", "nl", "hard"),
    # Norwegian Nynorsk
    ("nn_hus_2x3_5rh", "nn", "easy"),
    ("nn_hus_4x5_5rh", "nn", "hard"),
    # Polish
    ("pl_domy_2x3_5rh", "pl", "easy"),
    ("pl_domy_4x5_5rh", "pl", "hard"),
    # Portuguese
    ("pt_casas_2x3_5rh", "pt", "easy"),
    ("pt_casas_4x5_5rh", "pt", "hard"),
    # Romanian
    ("ro_case_2x3_5rh", "ro", "easy"),
    ("ro_case_4x5_5rh", "ro", "hard"),
    # Russian
    ("ru_doma_2x3_5rh", "ru", "easy"),
    ("ru_doma_4x5_5rh", "ru", "hard"),
    # Scots
    ("sco_hooses_2x3_5rh", "sco", "easy"),
    ("sco_hooses_4x5_5rh", "sco", "hard"),
    # Slovak
    ("sk_domy_2x3_5rh", "sk", "easy"),
    ("sk_domy_4x5_5rh", "sk", "hard"),
    # Slovenian
    ("sl_hise_2x3_5rh", "sl", "easy"),
    ("sl_hise_4x5_5rh", "sl", "hard"),
    # Albanian
    ("sq_shtepi_2x3_5rh", "sq", "easy"),
    ("sq_shtepi_4x5_5rh", "sq", "hard"),
    # Serbian
    ("sr_kuce_2x3_5rh", "sr", "easy"),
    ("sr_kuce_4x5_5rh", "sr", "hard"),
    # Swedish
    ("sv_hus_2x3_5rh", "sv", "easy"),
    ("sv_hus_4x5_5rh", "sv", "hard"),
    # Ukrainian
    ("uk_budynky_2x3_5rh", "uk", "easy"),
    ("uk_budynky_4x5_5rh", "uk", "hard"),
    # Chinese (Simplified — no bare 'zh' Language in languages.py; using zh-cn)
    ("zh_fangzi_2x3_5rh", "zh-cn", "easy"),
    ("zh_fangzi_4x5_5rh", "zh-cn", "hard"),
]
# Split sizes from original dataset (arXiv:2511.03553)
n_train = 128
n_val = 128
n_test = 1024


def main() -> None:
    """Create the zebra puzzle datasets and upload them to the HF Hub."""
    # Define the base download URL
    repo_id = "alexandrainst/multi-zebra-logic"

    for theme, lang_code, difficulty in THEMES:
        # Load dataset using load_dataset for robustness and proper dataset handling
        train_data: Dataset = load_dataset(repo_id, f"dataset_{theme}", split="train")
        val_data: Dataset = load_dataset(repo_id, f"dataset_{theme}", split="val")
        test_data: Dataset = load_dataset(repo_id, f"dataset_{theme}", split="test")

        # Check length
        assert len(train_data) == n_train
        assert len(val_data) == n_val
        assert len(test_data) == n_test

        # Convert the dataset to a dataframe
        train_df: pd.DataFrame = t.cast(pd.DataFrame, train_data.to_pandas())
        val_df: pd.DataFrame = t.cast(pd.DataFrame, val_data.to_pandas())
        test_df: pd.DataFrame = t.cast(pd.DataFrame, test_data.to_pandas())

        # Remove unused columns
        train_df = train_df[["introduction", "clues", "solution"]]
        val_df = val_df[["introduction", "clues", "solution"]]
        test_df = test_df[["introduction", "clues", "solution"]]

        # Combine introduction and clues into a single text column
        train_df["text"] = train_df["introduction"] + train_df["clues"].apply(
            lambda clues: "\n".join(clues)
        )
        val_df["text"] = val_df["introduction"] + val_df["clues"].apply(
            lambda clues: "\n".join(clues)
        )
        test_df["text"] = test_df["introduction"] + test_df["clues"].apply(
            lambda clues: "\n".join(clues)
        )

        # Rename the solution column as label
        train_df.rename(columns={"solution": "target_text"}, inplace=True)
        val_df.rename(columns={"solution": "target_text"}, inplace=True)
        test_df.rename(columns={"solution": "target_text"}, inplace=True)

        # Convert numpy arrays in target_text (the values of each dict) to lists
        train_df["target_text"] = train_df["target_text"].apply(
            lambda sol: {k: v.tolist() for k, v in sol.items()}
        )
        val_df["target_text"] = val_df["target_text"].apply(
            lambda sol: {k: v.tolist() for k, v in sol.items()}
        )
        test_df["target_text"] = test_df["target_text"].apply(
            lambda sol: {k: v.tolist() for k, v in sol.items()}
        )

        # Convert target_text from dict to string
        train_df["target_text"] = train_df["target_text"].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )
        val_df["target_text"] = val_df["target_text"].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )
        test_df["target_text"] = test_df["target_text"].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )

        # Collect datasets in a dataset dictionary
        dataset = DatasetDict(
            {
                "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
                "val": Dataset.from_pandas(val_df, split=Split.VALIDATION),
                "test": Dataset.from_pandas(test_df, split=Split.TEST),
            }
        )

        # Create dataset ID
        dataset_id = f"EuroEval/zebra-puzzles-{difficulty}-{lang_code}"

        # Remove the dataset from Hugging Face Hub if it already exists
        try:
            HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
        except HfHubHTTPError as e:
            print(f"Could not delete existing dataset {dataset_id}: {e}")

        # Push the dataset to the Hugging Face Hub
        dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
