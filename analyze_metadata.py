import json
from collections import Counter


def analyze_metadata(filename):
    print(f"--- Analyzing Metadata of {filename} ---")

    test_ids = []
    sections = []
    subsections = []

    with open(filename, "r") as f:
        for line in f:
            entry = json.loads(line)
            test_ids.append(entry.get("test_id"))
            sections.append(entry.get("section"))
            subsections.append(entry.get("subsection"))

    print(f"\nTotal Entries: {len(test_ids)}")

    print("\n--- Test IDs ---")
    for k, v in Counter(test_ids).most_common(20):
        print(f"{k}: {v}")

    print("\n--- Sections ---")
    for k, v in Counter(sections).most_common(20):
        print(f"{k}: {v}")

    print("\n--- Subsections ---")
    for k, v in Counter(subsections).most_common(20):
        print(f"{k}: {v}")


if __name__ == "__main__":
    analyze_metadata("data/aiskolprov/test.jsonl")
