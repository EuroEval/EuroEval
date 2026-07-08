# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
# ]
# ///

"""Create the LuxGen-Summ summarisation dataset.

Based on the LuxGen benchmark from arxiv.org/abs/2412.09415.
This script processes Luxembourgish text-generation data into summarisation format.
"""


def main() -> None:
    """Create LuxGen-Summ dataset.

    TODO: This script is a placeholder. The LuxGen benchmark paper needs to be
    reviewed to extract the summarisation component, or a new dataset needs to
    be created from Luxembourgish news/articles with human-written summaries.
    """
    print("LuxGen Summarisation Dataset Creation")
    print("=" * 50)
    print()
    print("This dataset needs to be created from:")
    print("  1. The LuxGen benchmark (arxiv.org/abs/2412.09415)")
    print("  OR")
    print("  2. Luxembourgish news articles with summaries")
    print()
    print("Required format:")
    print("  - text: Full article/document")
    print("  - summary: Human-written summary")
    print()
    print("Target splits:")
    print("  - train: 1,024 samples")
    print("  - val: 256 samples")
    print("  - test: 2,048 samples")
    print()
    print("Upload to: EuroEval/luxgen-summ")
    print()
    print("NOTE: Marked as official in config, but dataset must be created first.")


if __name__ == '__main__':
    main()
