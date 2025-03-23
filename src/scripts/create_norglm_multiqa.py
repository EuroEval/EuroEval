# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==2.15.0",
#     "huggingface-hub==0.24.0",
#     "openai==1.66.5",
#     "pandas==2.2.0",
#     "python-dotenv==1.0.1",
#     "requests==2.32.3",
# ]
# ///

"""Create the NorGLM NO-multi question answering dataset."""

import ast
import os

import pandas as pd
from constants import (
    MAX_NUM_CHARS_IN_CONTEXT,
    MAX_NUM_CHARS_IN_QUESTION,
    MIN_NUM_CHARS_IN_CONTEXT,
    MIN_NUM_CHARS_IN_QUESTION,
)
from datasets import Dataset, DatasetDict, Split, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from openai import OpenAI
from requests import HTTPError

load_dotenv()


def main() -> None:
    """Create the NorGLM NO-multi question answering dataset and upload to HF Hub."""
    dataset_id = "NorGLM/NO-Multi-QA-Sum"
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    dataset = load_dataset(dataset_id, split="train", token=True)
    assert isinstance(dataset, Dataset)

    dataset = dataset.rename_columns(column_mapping=dict(article="context"))

    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)

    # Drop unneeded index column
    df.drop("Unnamed: 0", inplace=True, axis=1)

    # Drop non-article by index
    df.drop(index=359, inplace=True)

    # Shuffle and drop first duplicate
    df = df.sample(frac=1, random_state=4242).drop_duplicates(subset="context")

    # Reset the index
    df = df.reset_index(drop=True)

    # Convert the question_answer column to a list of tuples
    def qa_to_list(row: str) -> list:
        """Converts a string representation of a list of tuples to an actual list.

        Args:
            row:
                The row to convert

        Returns:
            The row converted to an actual list of tuples
        """
        qa_list = ast.literal_eval(row)
        quest_ans = [(q.strip(), a.strip()) for q, a in qa_list]
        return quest_ans

    df["question_answer"] = df["question_answer"].apply(qa_to_list)

    # split question_answer list [(question, answer) ...] column to question and answer
    # with same context
    df = df.explode("question_answer")
    df[["question", "answer"]] = pd.DataFrame(
        df.question_answer.tolist(), index=df.index
    )
    df.drop(["question_answer", "summary"], inplace=True, axis=1)
    df.reset_index(drop=True, inplace=True)

    # Only work with samples where the context is not very large or small
    lengths = df.context.str.len()
    lower_bound = MIN_NUM_CHARS_IN_CONTEXT
    upper_bound = MAX_NUM_CHARS_IN_CONTEXT
    df = df[lengths.between(lower_bound, upper_bound)]

    # Only work with samples where the question is not very large or small
    lengths = df.question.str.len()
    lower_bound = MIN_NUM_CHARS_IN_QUESTION
    upper_bound = MAX_NUM_CHARS_IN_QUESTION
    df = df[lengths.between(lower_bound, upper_bound)]

    def rephrase_answer(question: str, answer: str, context: str) -> str:
        """Rephrase the answer such that it is in the context.

        Args:
            question:
                The question.
            answer:
                The answer.
            context:
                The context.

        Returns:
            The rephrased answer (if the answer is already in the context, it is
            returned as is).
        """
        answer = answer[:-1] if answer.endswith(".") else answer

        if answer.lower() in context.lower():
            return answer

        # Use OpenAI to rephrase the answer
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Given the following context: '{context}', and the "
                        f"'{question}' please rephrase the answer, so that it matches "
                        "exactly with a phrase from the context in a case-insensitive "
                        "way. E.g. it must be possible to find the rephrased answer "
                        "in the context without any modifications. The rephrased "
                        "answer should be as concise as possible, preferable not "
                        "more than 7 words, and ideally 3 or less words. Now "
                        f"rephrase this answer: '{answer}'"
                    ),
                }
            ],
            model="gpt-4o",
            seed=4242,
            temperature=0,
        )

        rephrased_answer = chat_completion.choices[0].message.content
        return rephrased_answer or answer

    # Use the original answer if it could be found in the context
    df_orig_bool = df[["answer", "context"]].apply(
        lambda row: row["answer"].lower() in row["context"].lower(), axis=1
    )
    df_orig = df[df_orig_bool]

    # For the rest of the answers, we try to rephrase them
    df_no_context = df[~df_orig_bool]

    # Rephrase the answers
    df_no_context.loc[:, "answer"] = df_no_context.apply(
        lambda row: rephrase_answer(row["question"], row["answer"], row["context"]),
        axis=1,
    )

    # Remove non-word start and end characters from answers
    df_no_context.loc[:, "answer"] = df_no_context["answer"].str.replace(
        r"(^\W|\W$)", "", regex=True
    )

    # Only keep the rephrased answers where the answer could be found in the context
    df_in_context = df_no_context[["answer", "context"]].apply(
        lambda row: row["answer"].lower() in row["context"].lower(), axis=1
    )
    df_with_answer = df_no_context[df_in_context]

    # Combine original with the rephrased answers
    cleaned_df = pd.concat([df_orig, df_with_answer])

    cleaned_df = cleaned_df.reset_index(drop=True)

    # Convert to 'answers' style column
    cleaned_df.loc[:, "answers"] = cleaned_df.apply(
        lambda row: {
            "text": [row["answer"]],
            "answer_start": row["context"].lower().index(row["answer"].lower()),
        },
        axis=1,
    )

    # Overwrite the original dataframe with the cleaned version
    df = cleaned_df

    # should not change since we use seed and temp 0 for rephrasing
    assert len(df) == 2406

    # Create validation split
    val_size = 256
    val_df = df.sample(n=val_size, random_state=4242)

    # Create train split
    train_size = 1024
    filtered_df = df[~df.index.isin(val_df.index)]
    train_df = filtered_df.sample(n=train_size, random_state=4242)

    # Create test split, using the remaining samples
    test_df = filtered_df[~filtered_df.index.isin(train_df.index)]

    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    dataset_id = "EuroEval/norglm-multi-qa"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api: HfApi = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
