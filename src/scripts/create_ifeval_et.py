# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==4.0.0",
#     "huggingface-hub==0.34.4",
# ]
# ///
"""Create the Estonian IFEval instruction-following dataset and upload to HF Hub."""

from datasets import load_dataset
from huggingface_hub import HfApi


def main() -> None:
    """Create the Estonian IFEval dataset and upload to HF Hub."""
    source_repo_id = "tartuNLP/ifeval_et"
    target_repo_id = "EuroEval/ifeval-et"

    ds = load_dataset(source_repo_id, split="test")
    ds = ds.rename_column("prompt", "text")
    ds = ds.map(
        lambda row: {
            "target_text": {
                "instruction_id_list": row["instruction_id_list"],
                "kwargs": row["kwargs"],
            }
        }
    )
    ds = ds.select_columns(["key", "text", "target_text"])

    HfApi().delete_repo(target_repo_id, repo_type="dataset", missing_ok=True)

    ds.push_to_hub(target_repo_id, private=True)


if __name__ == "__main__":
    main()
