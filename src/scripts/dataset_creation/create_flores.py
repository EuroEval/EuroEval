# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==4.0.0",
#     "huggingface-hub==0.34.4",
#     "python-dotenv==1.0.1",
#     "tqdm==4.67.1",
# ]
# ///

"""Create the FLORES+ translation datasets for all official EuroEval languages.

[FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus) is a multi-way
parallel machine-translation benchmark maintained by the Open Language Data Initiative.
Every sentence is professionally translated from an English original into each language,
so both sides of every pair are human gold and share a stable `id` across languages.

For each official language we build a bidirectional pair of datasets (`en-{code}` and
`{code}-en`), taking the training and validation splits from FLORES+ `dev` and the test
split from FLORES+ `devtest`.
"""

import os
from random import Random

from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from tqdm.auto import tqdm

SOURCE_REPO_ID = "openlanguagedata/flores_plus"
TARGET_REPO_ID = "EuroEval/flores-{source}-{target}"
ENGLISH_CODE = "eng_Latn"

# Mapping from EuroEval language code to the FLORES+ (ISO-639-3 + ISO-15924) config.
LANGUAGES = {
    "sq": "als_Latn",
    "be": "bel_Cyrl",
    "bs": "bos_Latn",
    "bg": "bul_Cyrl",
    "ca": "cat_Latn",
    "hr": "hrv_Latn",
    "cs": "ces_Latn",
    "da": "dan_Latn",
    "nl": "nld_Latn",
    "et": "ekk_Latn",
    "fo": "fao_Latn",
    "fi": "fin_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "hu": "hun_Latn",
    "is": "isl_Latn",
    "it": "ita_Latn",
    "lb": "ltz_Latn",
    "lv": "lvs_Latn",
    "lt": "lit_Latn",
    "no": "nob_Latn",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ro": "ron_Latn",
    "sr": "srp_Cyrl",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "es": "spa_Latn",
    "sv": "swe_Latn",
    "uk": "ukr_Cyrl",
}

TRAIN_SIZE = 128
VAL_SIZE = 256
TEST_SIZE = 1024
SEED = 42


def main() -> None:
    """Create the FLORES+ translation datasets for all official languages."""
    load_dotenv()

    english = _load_language(ENGLISH_CODE)

    for language, flores_code in tqdm(
        iterable=LANGUAGES.items(), desc="Creating FLORES+ datasets", unit="language"
    ):
        target = _load_language(flores_code)

        # `dev` provides the train and validation splits, `devtest` the test split. The
        # ids are shared across languages, so aligning on them keeps the pairs parallel.
        dev_ids = sorted(set(english["dev"]) & set(target["dev"]))
        devtest_ids = sorted(set(english["devtest"]) & set(target["devtest"]))
        Random(SEED).shuffle(dev_ids)

        train_ids = dev_ids[:TRAIN_SIZE]
        val_ids = dev_ids[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE]
        test_ids = devtest_ids[:TEST_SIZE]

        # English -> language: English is the source text, the language the target.
        en_xx = _build_dataset(
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            source=english,
            target=target,
        )
        en_xx.push_to_hub(
            TARGET_REPO_ID.format(source="en", target=language), private=True
        )

        # Language -> English: the language is the source text, English the target.
        xx_en = _build_dataset(
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            source=target,
            target=english,
        )
        xx_en.push_to_hub(
            TARGET_REPO_ID.format(source=language, target="en"), private=True
        )


def _build_dataset(
    train_ids: list[int],
    val_ids: list[int],
    test_ids: list[int],
    source: dict[str, dict[int, str]],
    target: dict[str, dict[int, str]],
) -> DatasetDict:
    """Build a directional translation dataset from aligned FLORES+ sentences.

    Args:
        train_ids:
            The `dev` ids to use for the training split.
        val_ids:
            The `dev` ids to use for the validation split.
        test_ids:
            The `devtest` ids to use for the test split.
        source:
            The split-to-(id-to-text) mapping for the source language.
        target:
            The split-to-(id-to-text) mapping for the target language.

    Returns:
        The dataset with `text` (source) and `target_text` (target) columns.
    """

    def rows(split: str, ids: list[int]) -> Dataset:
        return Dataset.from_list(
            [{"text": source[split][i], "target_text": target[split][i]} for i in ids]
        )

    return DatasetDict(
        {
            "train": rows("dev", train_ids),
            "val": rows("dev", val_ids),
            "test": rows("devtest", test_ids),
        }
    )


def _load_language(flores_code: str) -> dict[str, dict[int, str]]:
    """Load a FLORES+ language and index its sentences by split and id.

    Args:
        flores_code:
            The FLORES+ config name, e.g. `eng_Latn`.

    Returns:
        A mapping from split name (`dev`, `devtest`) to a mapping from sentence id to
        the sentence text.
    """
    dataset = load_dataset(SOURCE_REPO_ID, flores_code)
    return {
        split: {row["id"]: row["text"] for row in dataset[split]}
        for split in ("dev", "devtest")
    }


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    main()
