# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==4.0.0",
#     "huggingface-hub==0.34.4",
#     "tqdm==4.67.1",
# ]
# ///

"""Create the WMT24++ datasets."""

from datasets import Dataset, DatasetDict, load_dataset
from tqdm.auto import tqdm

LANGUAGES = {
    "bg": "en-bg_BG",
    "ca": "en-ca_ES",
    "cs": "en-cs_CZ",
    "da": "en-da_DK",
    "de": "en-de_DE",
    "el": "en-el_GR",
    "et": "en-et_EE",
    "fi": "en-fi_FI",
    "fr": "en-fr_FR",
    "hr": "en-hr_HR",
    "hu": "en-hu_HU",
    "is": "en-is_IS",
    "it": "en-it_IT",
    "lt": "en-lt_LT",
    "lv": "en-lv_LV",
    "nl": "en-nl_NL",
    "no": "en-no_NO",
    "pl": "en-pl_PL",
    "pt": "en-pt_PT",
    "ro": "en-ro_RO",
    "sk": "en-sk_SK",
    "sl": "en-sl_SI",
    "sr": "en-sr_RS",
    "sv": "en-sv_SE",
    "uk": "en-uk_UA",
}


def main() -> None:
    """Create the WMT24++ datasets."""
    target_repo_id = "EuroEval/wmt24pp-en-{target}"

    for language, subset in tqdm(
        iterable=LANGUAGES.items(), desc="Creating WMT24++ datasets", unit="language"
    ):
        ds = load_dataset("google/wmt24pp", name=subset, split="train").shuffle(seed=42)
        assert isinstance(ds, Dataset)

        ds = ds.filter(lambda x: not x["is_bad_source"])
        ds = ds.rename_columns(dict(source="text", target="target_text"))

        train_size = 64
        val_size = 128
        test_size = len(ds) - train_size - val_size

        train = ds.select(range(train_size))
        val = ds.select(range(train_size, train_size + val_size))
        test = ds.select(
            range(train_size + val_size, train_size + val_size + test_size)
        )

        assert isinstance(train, Dataset)
        assert isinstance(val, Dataset)
        assert isinstance(test, Dataset)

        new_ds = DatasetDict({"train": train, "val": val, "test": test})
        new_ds.push_to_hub(target_repo_id.format(target=language), private=True)


if __name__ == "__main__":
    main()
