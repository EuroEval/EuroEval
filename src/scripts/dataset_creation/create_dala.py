# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
# ]
# ///

"""Create the DaLA dataset and upload it to the HF Hub."""

import os
import tempfile

# Set cache before importing datasets to avoid Python 3.14 pickle issues
_cache_dir = tempfile.mkdtemp()
os.environ["HF_DATASETS_CACHE"] = _cache_dir

from datasets.dataset_dict import DatasetDict  # noqa: E402
from datasets.load import load_dataset  # noqa: E402
from huggingface_hub.hf_api import HfApi  # noqa: E402


def main() -> None:
    """Create the DaLA dataset and upload it to the HF Hub.

    Fetches the DaLA dataset from the original source and mirrors it to the
    EuroEval organisation as a private dataset. No filtering, capping, or
    re-splitting is performed — the dataset is uploaded as-is.
    """
    source = "giannor/dala"
    target = "EuroEval/dala"

    dataset = load_dataset(path=source, token=True)
    assert isinstance(dataset, DatasetDict)

    HfApi().delete_repo(repo_id=target, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(repo_id=target, private=True)


if __name__ == "__main__":
    main()
