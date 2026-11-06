# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "numpy==1.26.0",
#     "pandas==2.2.0",
#     "scikit-learn<1.6.0",
# ]
# ///

"""Create the Faroese Semantic Relations dataset and upload it to the HF Hub."""

import sys

if __package__:
    from .faroese_processing import create_faroese_dataset
else:
    from faroese_processing import create_faroese_dataset


def main(source: str) -> None:
    """Create Faroese semantic relations from JSONL data.

    Args:
        source:
            Path or URL to the source JSONL file.
    """
    create_faroese_dataset(source=source, task="semantic-relations")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <path-or-url-to-jsonl>")
    main(source=sys.argv[1])
