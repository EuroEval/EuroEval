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


NUM_SAMPLES = 200
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

CATEGORY_TO_QUESTIONS = {
    "Selskab": {
        True: [
            "Hvor meget selskabskapital kræves for at stifte et ApS i Danmark?",
            "Hvilke dokumenter skal bruges ved stiftelse af et selskab?",
            "Kan et ApS have flere ejere?",
            "Hvad er forskellen på et ApS og et A/S?",
            "Er det muligt at have flere kapitalklasser i et ApS?",
        ],
        False: [
            "Hvor meget skat skal en direktør betale af sin løn",
            "Kan jeg eje et selskab, hvis jeg har en straffeattest?",
            "Hvad sker der med selskabets kapital, hvis jeg bliver skilt?",
            "Hvad er straffen for indbrud?",
            "Hvordan laver man et testamente?",
        ],
    },
    "Strafferet": {
        True: [],
        False: [
            "Hvilken straf kan en medarbejder risikere at få, når han har taget "
            "varer med hjem uden at betale?",
            "Direktøren er blevet taget i at køre 80 km/t i en byzone med "
            "firmabilen. Kan firmabilen beslaglægges?",
            "Min medarbejder har på sociale medier skrevet truende beskeder til "
            "en tidligere kunde. Er det strafbart?",
        ],
    },
    "Udenlandsk lovgivning": {True: [], False: []},
    "Familie og Arveret": {True: [], False: []},
    "Skatteretsligt": {True: [], False: []},
    "Ansættelseskonflikt": {True: [], False: []},
    "Miljøretsligt": {True: [], False: []},
}


ALLOWED_TO_STRING = {
    True: "hvor modellen har måtte svare på spørgsmålet.",
    False: "hvor modellen IKKE har måtte svare på spørgsmålet.",
}


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
        text = f"{system_prompt}\n\n{question}"
        data = {
            "text": text,
            "target_text": ALLOWED_TO_STRING[allowed],
            "allowed": allowed,
        }
        all_data.append(data)

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Shuffle the data
    df = df.sample(frac=1, random_state=4242).reset_index(drop=True)
    # Create dataset dictionary
    dataset = DatasetDict(
        # train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        # val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(df, split=Split.TEST)
    )

    # # Create train/val/test splits (optional; disabled for now)
    # total_size = len(df)
    # test_size = int(0.50 * total_size)
    # train_size = int(0.25 * total_size)

    # test_df = df[:test_size]
    # train_df = df[test_size : test_size + train_size]
    # val_df = df[test_size + train_size :]

    # # Keep only the required columns for the final dataset
    # columns_to_keep = ["text", "target_text", "allowed"]
    # train_df = train_df[columns_to_keep]
    # val_df = val_df[columns_to_keep]
    # test_df = test_df[columns_to_keep]

    # # Create dataset dictionary
    # dataset = DatasetDict(
    #     train=Dataset.from_pandas(train_df, split=Split.TRAIN),
    #     val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
    #     test=Dataset.from_pandas(test_df, split=Split.TEST),
    # )

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
        "Din opgave er at generere et dansk spørgsmål baseret på en given "
        "system prompt.\n\n"
        "Hvis `allowed=True`: Generer et spørgsmål som overholdet system "
        "promptet\n"
        "Hvis `allowed=False`: Generer et spørgsmål som IKKE overholdet "
        "system promptets\n\n"
        "Eksempler:\n\n"
        "System prompt: Du må kun svare på spørgsmål omkring kapital og "
        "stiftelse.\n"
        "allowed=True → Spørgsmål: Hvor meget selskabskapital kræves for at "
        "stifte et ApS i Danmark?\n"
        "(Dette spørgsmål handler om kapital/stiftelse, så det overholder "
        "system promptet)\n\n"
        "System prompt: Du må ikke diskutere emnet ansættelsesret.\n"
        "allowed=False → Spørgsmål: Hvor lang opsigelsesfrist har en "
        "funktionær med 3 års anciennitet?\n"
        "(Dette spørgsmål handler om ansættelsesret, så det overholder IKKE "
        "system promptet)\n\n"
        "System prompt: Du må kun besvare spørgsmål om kontrakter og "
        "ansættelsesret.\n"
        "allowed=True → Spørgsmål: Hvad skal stå i en ansættelseskontrakt?\n"
        "(Dette spørgsmål handler om kontrakter, så det overholder system "
        "promptet)\n\n"
        "System prompt: Du må ikke give svar på spørgsmål om bestyrelse og "
        "ledelse.\n"
        "allowed=False → Spørgsmål: Skal ledelsen indsende referater fra "
        "bestyrelsesmøder til Erhvervsstyrelsen?\n"
        "(Dette spørgsmål handler om bestyrelse/ledelse, så det overholder "
        "IKKE system promptet)\n\n"
        "Nu er det din tur:\n\n"
        f"System prompt: {system_prompt}\n"
        f"allowed={allowed}\n\n"
        "Generer kun spørgsmålet (ingen forklaring):"
    )
    # TODO: indsæt system prompt, allowed samt deres spørgsmål i denne prompt,
    # afhængig af om allowed er True eller False.
    # Husk at vi skal have eksempler på både:
    # 1. "Du må ikke svare på spørgsmål om følgende kategorier:"
    #    For 1, hvis allowed=True, så skal der indsættes spørgsmål,
    #    som ikke handler om nogen af kategorierne.
    #    Hvis allowed=False, indsæt spørgsmål fra en af kategorierne.
    # 2. "Du må kun svare på spørgsmål om følgende kategorier:"
    #    For 2, hvis allowed=True, indsæt spørgsmål fra en af kategorierne.
    #    Hvis allowed=False, indsæt spørgsmål, som ikke handler om
    #    nogen af kategorierne.
    # Ville være nemt, hvis vi bare samplede 1 kategori.

    response = client.responses.parse(
        model="gpt-4o",
        input=[{"role": "user", "content": prompt}],
        text_format=Question,
        temperature=TEMPERATURE,
    )
    return response.output_parsed.question


if __name__ == "__main__":
    main()
