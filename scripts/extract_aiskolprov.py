import json
from pathlib import Path

# Path to the source file in the cloned repo
source_path = Path(
    "tmp_aiskolprov/results/ekgren-swedish_skolprov_all_anthropic-claude-3.5-haiku.jsonl"
)
# Path to the output file
output_path = Path("data/aiskolprov/test.jsonl")


def extract_data():
    if not source_path.exists():
        print(f"Error: Source file {source_path} not found.")
        return

    extracted_count = 0
    with (
        open(source_path, "r", encoding="utf-8") as f_in,
        open(output_path, "w", encoding="utf-8") as f_out,
    ):
        for line in f_in:
            try:
                data = json.loads(line)
                item = data.get("item")
                if item:
                    # EuroEval expects 'text' and 'label' or similar fields depending on the task.
                    # For KNOW/MULTIPLE_CHOICE, we need 'text' (question + options usually) and 'label' (answer).
                    # But EuroEval's multiple choice templates might expect specific fields.
                    # Let's look at the multiple_choice.py template again.
                    # It uses {text} and {label}.
                    # We should probably format the text to include options A-E if the template doesn't handle it automatically.
                    # However, looking at EuroEval's MMLU implementation or similar might reveal if it expects options in 'text' or as separate fields.
                    # The prompt template in multiple_choice.py: "Fråga: {text}\nSvar: {label}"
                    # This implies {text} should contain the question AND the options.

                    question = item.get("question", "")
                    if resource := item.get("question_resource"):
                        question = f"{resource}\n\n{question}"

                    options = []
                    for key in [
                        "option_a",
                        "option_b",
                        "option_c",
                        "option_d",
                        "option_e",
                    ]:
                        val = item.get(key)
                        letter = key.split("_")[1]
                        if val:
                            options.append(f"{letter}. {val}")
                        else:
                            # Must include placeholder to keep alignment with a,b,c,d,e
                            options.append(f"{letter}. -")

                    # Ensure double newline before choices so prepare_examples can strip the separator
                    formatted_text = f"{question}\n\n" + "\n".join(options)

                    # The label in the source is 'A', 'B', 'C', 'D', 'E'.
                    # EuroEval usually expects the label text or the label ID depending on config.
                    # Let's start with the label character as the label.
                    # Label should be lowercase "a", "b", etc.
                    label = item.get("answer", "").lower()

                    new_entry = {
                        "text": formatted_text,
                        "label": label,
                        # valuable metadata
                        "test_id": item.get("test_id"),
                        "section": item.get("section"),
                        "subsection": item.get("subsection"),
                        "question_id": item.get("question_id"),
                    }

                    f_out.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
                    extracted_count += 1
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line[:50]}...")

    print(f"Extracted {extracted_count} items to {output_path}")


if __name__ == "__main__":
    extract_data()
