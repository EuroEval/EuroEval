# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "beautifulsoup4",
#     "datasets==2.15.0",
#     "huggingface-hub==0.24.0",
#     "joblib==1.4.2",
#     "pandas==2.2.0",
#     "requests==2.32.3",
#     "tqdm==4.67.1",
# ]
# ///

"""Create the HotterAndColderSentiment dataset and upload it to the HF Hub."""

import datetime as dt
import io
import logging
import re
import warnings
from time import sleep
from zipfile import ZipFile

import joblib
import pandas as pd
import requests as rq
from bs4 import BeautifulSoup, NavigableString, Tag
from constants import MAX_NUM_CHARS_IN_DOCUMENT, MIN_NUM_CHARS_IN_DOCUMENT  # noqa
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from tqdm.auto import tqdm

logger = logging.getLogger("create_hotter_and_colder")


def main() -> None:
    """Create the HotterAndColderSentiment dataset and upload it to the HF Hub.

    Raises:
        HTTPError:
            If the dataset could not be downloaded.
    """
    # Download the ZIP file
    url = (
        "https://repository.clarin.is/repository/xmlui/bitstream/handle/"
        "20.500.12537/352/Icelandic_Sentiment_Corpus.zip"
    )
    response = rq.get(url=url)
    response.raise_for_status()

    # Get the unhydrated data from the ZIP file
    with ZipFile(file=io.BytesIO(initial_bytes=response.content)) as zip_file:
        csv_files = [
            zip_file.read(name=file_name)
            for file_name in zip_file.namelist()
            if file_name.endswith(".csv")
        ]
        assert len(csv_files) == 1, (
            f"Expected one CSV file in the ZIP file, but found {len(csv_files)}."
        )
        df = pd.read_csv(filepath_or_buffer=io.BytesIO(initial_bytes=csv_files[0]))

    # Set up the dataframe for the dataset, resulting in 'text' and 'label' columns,
    # with 'text' being the hydrated comment content and 'label' being the sentiment
    # label ('negative', 'neutral', 'positive')
    df = hydrate_data(df=df.query("annotation_task_name == 'sentiment'"))
    df = df.rename(columns=dict(comment_content="text", label_given_by_user="label"))
    df.label = df.label.map({"0": "negative", "1": "neutral", "2": "positive"})
    df = df.dropna(subset="label")
    df = df.drop(columns=[col for col in df.columns if col not in ["text", "label"]])

    # Create validation split
    val_size = 256
    val_df = df.sample(n=val_size, random_state=4242)
    df = df.drop(index=val_df.index.tolist())

    # Create train split
    train_size = 1024
    train_df = df.sample(n=train_size, random_state=4242)

    # Create test split, just being the rest of the samples
    test_df = df.drop(index=train_df.index.tolist())

    # Sampling maintains the original index, so we need to reset it, as otherwise the
    # conversion to Dataset objects will create extra index columns
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Only work with samples where the document is not very large or small
    # We do it after we have made the splits to ensure that the dataset is minimally
    # affected.
    new_train_df = train_df.copy()
    new_train_df["text_len"] = new_train_df.text.str.len()
    new_train_df = new_train_df.query("text_len >= @MIN_NUM_CHARS_IN_DOCUMENT").query(
        "text_len <= @MAX_NUM_CHARS_IN_DOCUMENT"
    )
    new_val_df = val_df.copy()
    new_val_df["text_len"] = new_val_df.text.str.len()
    new_val_df = new_val_df.query("text_len >= @MIN_NUM_CHARS_IN_DOCUMENT").query(
        "text_len <= @MAX_NUM_CHARS_IN_DOCUMENT"
    )
    new_test_df = test_df.copy()
    new_test_df["text_len"] = new_test_df.text.str.len()
    new_test_df = new_test_df.query("text_len >= @MIN_NUM_CHARS_IN_DOCUMENT").query(
        "text_len <= @MAX_NUM_CHARS_IN_DOCUMENT"
    )

    dataset = DatasetDict(
        train=Dataset.from_pandas(new_train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(new_val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(new_test_df, split=Split.TEST),
    )

    dataset_id = "EuroEval/hotter-and-colder-sentiment"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except rq.HTTPError:
        pass

    dataset.push_to_hub(dataset_id, private=True)


def hydrate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Hydrate the data in the DataFrame.

    Args:
        df:
            The DataFrame containing the data to hydrate.

    Returns:
        The DataFrame with the hydrated data.
    """
    with warnings.catch_warnings():
        warnings.simplefilter(
            action="ignore", category=pd.errors.SettingWithCopyWarning
        )
        df.comment_datetime = pd.to_datetime(arg=df.comment_datetime)

    with joblib.Parallel(n_jobs=-1, backend="threading") as parallel:
        comment_texts = parallel(
            joblib.delayed(scrape_comment_content)(
                url=row.full_link, target_datetime=row.comment_datetime
            )
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Hydrating comments")
        )

    with warnings.catch_warnings():
        warnings.simplefilter(
            action="ignore", category=pd.errors.SettingWithCopyWarning
        )
        df["comment_content"] = comment_texts

    return df.dropna(subset=["comment_content"])


def scrape_comment_content(url: str, target_datetime: dt.datetime) -> str | None:
    """Scrape the content of a comment with a given datetime from a URL.

    Args:
        url:
            The URL to scrape the content from.
        target_datetime:
            The datetime of the comment to find.

    Returns:
        The content of the comment with the given datetime, or None if the comment could
        not be found.
    """
    num_attempts = 10
    for _ in range(num_attempts):
        try:
            response = rq.get(url=url)
            response.raise_for_status()
            break
        except (rq.HTTPError, rq.ConnectionError):
            sleep(1)
            continue
    else:
        logger.error(f"Could not fetch {url!r} after {num_attempts} attempts.")
        return None

    soup = BeautifulSoup(markup=response.content, features="html.parser")

    comment_divs: list[Tag] = soup.find_all(
        name="div",
        attrs={
            "class": [
                "comment even registered",
                "comment odd registered",
                "comment even registered own",
                "comment odd registered own",
                "comment odd unregistered",
                "comment even unregistered",
            ]
        },
    )
    for comment in comment_divs:
        comment_body = comment.find(name="div", class_="comment-body")
        signature_body = comment.find(name="p", class_="comment-signature")

        # Sanity check to ensure that we're not leaving any comments behind; this
        # ensures that the two bodies are either bs4 Tag objects or None.
        assert not isinstance(comment_body, NavigableString)
        assert not isinstance(signature_body, NavigableString)

        if isinstance(comment_body, Tag) and isinstance(signature_body, Tag):
            comment_datetime = extract_datetime_from_signature(
                signature_body=signature_body
            )
            if comment_datetime == target_datetime:
                return comment_body.text.strip()
    else:
        logger.error(f"Error scraping comment content for {url!r}")
        return None


def extract_datetime_from_signature(signature_body: Tag) -> dt.datetime | None:
    """Parse the signature to extract the datetime information.

    Args:
        signature_body:
            The signature body to extract the datetime information from.

    Returns:
        The extracted datetime information, or None if no datetime information was
        found.
    """
    signature_str = "".join([str(child) for child in signature_body.children]).strip()
    signature_str = re.sub("\n", " ", signature_str)
    signature_str = re.sub(r"\s+", " ", signature_str)

    datetime_match = re.search(
        pattern=r"(\d{1,2}\.\d{1,2}\.\d{4})\s+kl\.\s+(\d{1,2}:\d{2})",
        string=signature_str,
    )
    if not datetime_match:
        return None

    date_str, time_str = datetime_match.groups()
    return dt.datetime.strptime(f"{date_str} {time_str}", "%d.%m.%Y %H:%M")


if __name__ == "__main__":
    main()
