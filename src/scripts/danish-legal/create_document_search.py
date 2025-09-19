"""Create the Danish legal benchmark dataset Document Search."""

import os

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from openai import OpenAI
from pydantic import BaseModel

# Constants
TEMPERATURE = 0.3
MODEL = "gpt-4o"


class QuestionAnswerResponse(BaseModel):
    """Represents the response structure for question and answer generation."""

    question: str
    answer: str


def main() -> None:
    """Create the Danish legal benchmark dataset Document Search."""
    # Load the contracts dataset
    contracts_dataset = load_dataset("alexandrainst/contracts")

    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    samples: list[dict[str, str]] = []

    for contract in contracts_dataset["train"]:
        contract_text = contract["content"]

        # Generate a question and answer using OpenAI
        question, answer = generate_question_answer(
            client=client, contract=contract_text
        )

        text = f"{contract_text}\n\n{question}"
        target_text = answer
        samples.append({"text": text, "target_text": target_text})

    # Convert samples to a DataFrame
    df = pd.DataFrame(samples)
    split = Dataset.from_pandas(df, split=Split.TEST)
    dataset = DatasetDict(test=split)

    # Upload dataset
    dataset_id = "EuroEval/legal-document-search"
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)


def generate_question_answer(client: OpenAI, contract: str) -> tuple[str, str]:
    """Generate a question and answer for a contract."""
    # Define the prompt for generating a question and answer in Danish
    # TODO: Generate multiple question answer pairs for each contract?
    # E.g.: "Generer n spørgsmål og svar til den følgende kontrakt."
    prompt = (
        "Læs den følgende kontrakt og generér et spørgsmål, der kan besvares "
        "ved hjælp af oplysningerne i kontrakten. Svaret skal findes eksplicit "
        "i kontraktteksten. Både spørgsmålet og svaret skal være på dansk.\n\n"
        f"{contract}\n\nSpørgsmål:"
    )

    # Send the prompt to OpenAI
    response = client.responses.parse(
        model=MODEL,
        input=[{"role": "user", "content": prompt}],
        text_format=QuestionAnswerResponse,
        temperature=TEMPERATURE,
    ).output_parsed

    if response is None:
        raise ValueError("No response from OpenAI")

    # Access the structured response
    question = response.question
    answer = response.answer

    return question, answer


if __name__ == "__main__":
    main()
