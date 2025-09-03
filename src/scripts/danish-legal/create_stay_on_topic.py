# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "openai==1.66.5",
#     "pandas==2.2.0",
#     "python-dotenv==1.0.1",
#     "requests==2.32.3",
#     "click==8.2.1",
#     "openpyxl==3.1.5",
# ]
# ///

"""Create the legal Stay on Topic dataset and upload it to the HF Hub."""

import logging

import click
import pandas as pd
from datasets import Dataset, DatasetDict, Split
from dotenv import load_dotenv
from huggingface_hub import HfApi
from openai import OpenAI
from pydantic import BaseModel
from requests import HTTPError

logging.basicConfig(format="%(asctime)s ⋅ %(message)s", level=logging.INFO)
logger = logging.getLogger("create_legal_stay_on_topic")

load_dotenv()


NUM_SAMPLES = 200
TEMPERATURE = 0.3
MODEL = "gpt-4o"

GENERATION_PROMPT = """Din opgave er at generere et dansk spørgsmål baseret på en given
system prompt.

Hvis `allowed=True`: Generer et spørgsmål som overholder system promptet
Hvis `allowed=False`: Generer et spørgsmål som IKKE overholder system promptet

Eksempler:

System prompt: Du må kun svare på spørgsmål omkring kapital og stiftelse.
allowed=True → Spørgsmål: Hvor meget selskabskapital kræves for at stifte et ApS i
Danmark?
(Dette spørgsmål handler om kapital/stiftelse, så det overholder system promptet)

System prompt: Du må ikke diskutere emnet ansættelsesret.
allowed=False → Spørgsmål: Hvor lang opsigelsesfrist har en funktionær med 3 års
anciennitet?
(Dette spørgsmål handler om ansættelsesret, så det overholder IKKE system promptet)

System prompt: Du må kun besvare spørgsmål om kontrakter og ansættelsesret.
allowed=True → Spørgsmål: Hvad skal stå i en ansættelseskontrakt?
(Dette spørgsmål handler om kontrakter, så det overholder system promptet)

System prompt: Du må ikke give svar på spørgsmål om bestyrelse og ledelse.
allowed=False → Spørgsmål: Skal ledelsen indsende referater fra bestyrelsesmøder til
Erhvervsstyrelsen?
(Dette spørgsmål handler om bestyrelse/ledelse, så det overholder IKKE system promptet)

Nu er det din tur:

System prompt: {system_prompt}
allowed={allowed}

Generer kun spørgsmålet (ingen forklaring):
"""


class SystemPrompt(BaseModel):
    """Represents a system prompt."""

    system_prompt: str


class Question(BaseModel):
    """Represents a single question."""

    question: str


@click.command()
@click.option(
    "--path",
    type=click.Path(exists=True),
    required=True,
    help="The path to the Excel file containing the data.",
)
def main(path: str) -> None:
    """Create the Stay on Topic dataset and upload it to the HF Hub."""
    df = read_data_from_sheet(path=path)

    # TODO: Generate data
    # generate_data()

    # Shuffle the data
    df = df.sample(frac=1, random_state=4242).reset_index(drop=True)

    # Create dataset dictionary
    dataset = DatasetDict(test=Dataset.from_pandas(df, split=Split.TEST))

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


def read_data_from_sheet(path: str) -> pd.DataFrame:
    """Read data from a Google Sheet.

    Args:
        path:
            The path to the Excel file.

    Returns:
        The data.
    """
    df = pd.read_excel(path, sheet_name="Evals - Stay on Topic")
    df["allowed"] = df["allowed"].map(bool)

    dfs = split_df_by_empty_rows(df)
    for df in dfs:
        category = df.iloc[0]["category"]
        if df.iloc[0]["allowed"]:
            system_prompt = (
                f"Du må KUN svare på spørgsmål om kategorien {category.lower()!r}."
            )
        else:
            system_prompt = (
                f"Du må IKKE svare på spørgsmål om kategorien {category.lower()!r}."
            )
        df["text"] = system_prompt + "\n\n" + df["question"]
    df = pd.concat(dfs)

    keep_columns = ["text", "allowed"]
    df = df[keep_columns]
    assert isinstance(df, pd.DataFrame)

    df = df.dropna().reset_index(drop=True)

    return df


def split_df_by_empty_rows(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split DataFrame by empty rows into multiple DataFrames.

    Args:
        df:
            The DataFrame to split.

    Returns:
        The split DataFrames.
    """
    # Find rows where all values are NaN
    empty_rows = df.isnull().all(axis=1)

    # Get indices of empty rows
    empty_indices = df[empty_rows].index.tolist()

    # Add start and end indices for slicing
    split_indices = [df.index[0] - 1] + empty_indices + [df.index[-1] + 1]

    # Create list of DataFrames
    dfs = []
    for i in range(len(split_indices) - 1):
        start = split_indices[i] + 1
        end = split_indices[i + 1]

        # Get slice of DataFrame (excluding empty rows)
        chunk = df.loc[start : end - 1]

        # Only add non-empty chunks
        if not chunk.empty:
            dfs.append(chunk.reset_index(drop=True))

    return dfs


# def generate_data() -> list[dict]:
#     """Generate data.

#     Returns:
#         The generated data.
#     """
#     # TODO: Read CATEGORIES from excel sheet
#     client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
#     all_data = []
#     for _ in range(NUM_SAMPLES):
#         num_categories = random.randint(1, len(CATEGORIES))
#         categories = random.sample(CATEGORIES, num_categories)
#         system_prompt = generate_system_prompt(client=client, categories=categories)
#         allowed = random.random() < 0.5
#         question = generate_question(
#             client=client, system_prompt=system_prompt, allowed=allowed
#         )
#         text = f"{system_prompt}\n\n{question}"
#         data = {
#             "text": text,
#             "target_text": "",
#             "allowed": allowed,
#         }
#         all_data.append(data)
#     return all_data


def generate_system_prompt(client: OpenAI, categories: list[str]) -> str:
    """Generate a system prompt.

    Args:
        client:
            The OpenAI client.
        categories:
            The categories to include in the system prompt.

    Returns:
        The system prompt.

    Raises:
        ValueError:
            If no response is received from OpenAI.
    """
    categories_str = (
        ", ".join(categories[:-1]) + " og " + categories[-1]
        if len(categories) > 1
        else categories[0]
    )
    prompt = f"""
        Generér en sætning, hvor du beskriver, at du enten kun må snakke om emnerne i
        listen, eller at du ikke må snakke om emnerne i listen. Emnerne er
        {categories_str}. System prompten skal altid være på dansk.
    """.replace("\n", " ").strip()
    response = client.responses.parse(
        model=MODEL,
        input=[{"role": "user", "content": prompt}],
        text_format=SystemPrompt,
        temperature=TEMPERATURE,
    ).output_parsed
    if response is None:
        raise ValueError("No response from OpenAI")
    return response.system_prompt


def generate_question(client: OpenAI, system_prompt: str, allowed: bool) -> str:
    """Generate a question.

    Args:
        client:
            The OpenAI client.
        system_prompt:
            The system prompt.
        allowed:
            Whether the question is allowed.

    Returns:
        The question.

    Raises:
        ValueError:
            If no response is received from OpenAI.
    """
    prompt = GENERATION_PROMPT.format(system_prompt=system_prompt, allowed=allowed)
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
        model=MODEL,
        input=[{"role": "user", "content": prompt}],
        text_format=Question,
        temperature=TEMPERATURE,
    ).output_parsed
    if response is None:
        raise ValueError("No response from OpenAI")
    return response.question


if __name__ == "__main__":
    main()
