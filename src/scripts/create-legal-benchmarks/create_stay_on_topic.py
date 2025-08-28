# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "openai==1.66.5",
#     "pandas==2.2.0",
#     "python-dotenv==1.0.1",
#     "requests==2.32.3",
# ]
# ///

"""Create the Stay on Topic dataset and upload it to the HF Hub."""

import logging
import os
import random

import pandas as pd
from datasets import Dataset, DatasetDict, Split
from dotenv import load_dotenv
from huggingface_hub import HfApi
from openai import OpenAI
from pydantic import BaseModel
from requests import HTTPError

logging.basicConfig(format="%(asctime)s ⋅ %(message)s", level=logging.INFO)
logger = logging.getLogger("create_stay_on_topic")

load_dotenv()


NUM_SAMPLES = 20
TEMPERATURE = 0.3

CATEGORIES = [
    "Selskab",
    "Strafferet",
    "Udenlandsk lovgivning",
    "Familie og Arveret",
    "Skatteretsligt",
    "Ansættelseskonflikt",
    "Miljøretsligt",
]


class SystemPrompt(BaseModel):
    """Represents a system prompt."""

    system_prompt: str


class Question(BaseModel):
    """Represents a single question-answer pair."""

    question: str


def main() -> None:
    """Create the Stay on Topic dataset and upload it to the HF Hub."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    all_data = []
    for _ in range(NUM_SAMPLES):
        num_categories = random.randint(1, len(CATEGORIES))
        categories = random.sample(CATEGORIES, num_categories)
        system_prompt = generate_system_prompt(client=client, categories=categories)
        allowed = random.random() < 0.5
        question = generate_question(
            client=client, system_prompt=system_prompt, allowed=allowed
        )
        data = {
            "system_prompt": system_prompt,
            "text": question,
            "target_text": "",
            "allowed": allowed,
        }
        all_data.append(data)

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Shuffle the data
    df = df.sample(frac=1, random_state=4242).reset_index(drop=True)

    # Create train/val/test splits (optional; disabled for now)
    total_size = len(df)
    test_size = int(0.50 * total_size)
    train_size = int(0.25 * total_size)

    test_df = df[:test_size]
    train_df = df[test_size : test_size + train_size]
    val_df = df[test_size + train_size :]

    # Keep only the required columns for the final dataset
    columns_to_keep = ["system_prompt", "text", "target_text", "allowed"]
    train_df = train_df[columns_to_keep]
    val_df = val_df[columns_to_keep]
    test_df = test_df[columns_to_keep]

    # Create dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    dataset_id = "EuroEval/legal-stay-on-topic"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    logger.info(f"Uploading dataset to {dataset_id}...")
    dataset.push_to_hub(dataset_id, private=True)
    logger.info("Dataset uploaded successfully!")


def generate_system_prompt(client: OpenAI, categories: list[str]) -> str:
    """Generate a system prompt."""
    prompt = (
        "Generer en sætning, hvor du beskriver, at du enten kun "
        "må snakke om emnerne i listen, "
        "eller at du ikke må snakke om emnerne i listen. "
        "Emnerne er: " + ", ".join(categories) + "."
        "System promptet skal altid være på dansk."
    )
    response = client.responses.parse(
        model="gpt-4o",
        input=[{"role": "user", "content": prompt}],
        text_format=SystemPrompt,
        temperature=TEMPERATURE,
    )
    return response.output_parsed.system_prompt


def generate_question(client: OpenAI, system_prompt: str, allowed: bool) -> str:
    """Generate a question."""
    prompt = (
        "Du skal generere et spørgsmål. Du får givet en system prompt. "
        "Hvis `allowed` er `True`, så skal du generere et spørgsmål som lægger op "
        "et svar som overholder system promptet. Hvis `allowed` er `False`, så skal du "
        "generere et spørgsmål som ikke overholder system promptet. System promptet er:"
        f"{system_prompt}"
        f"`allowed` er: {allowed}"
        "Spørgsmålet skal være på dansk."
    )

    response = client.responses.parse(
        model="gpt-4o",
        input=[{"role": "user", "content": prompt}],
        text_format=Question,
        temperature=TEMPERATURE,
    )
    return response.output_parsed.question


if __name__ == "__main__":
    main()
