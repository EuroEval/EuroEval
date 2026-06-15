"""Translate hallucination datasets between languages while preserving span labels.
Copied from https://github.com/KRLabsOrg/LettuceDetect/blob/main/scripts/translate/translate.py
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import httpx
import tqdm
from datasets import Dataset
from dotenv import load_dotenv
from lettucedetect.datasets.hallucination_dataset import (
    HallucinationData,
    HallucinationSample,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

TRANSLATION_ANSWER = """
Translate the following text from {source_lang} to {target_lang}.
- If the original text contains <HAL> tags, translate the content inside <HAL> tags and ensure the number of the <HAL> tags remain exactly the same in the output.
- If the original text do not contain <HAL> tags, just translate the text.
- Do NOT add any <HAL> tags if they were not in the original text.
- Do NOT remove any <HAL> tags that were in the original text.
- Do not include any additional sentences summarizing or explaining the translation.
- Your output should be just the translated text, nothing else.

{source_lang}:
======START======
{text}
======END======

Output in {target_lang}:
"""

TRANSLATION_PROMPT = """
Translate the following prompt from {source_lang} to {target_lang}.
- Translate only the given prompt.
- Do not include any additional sentences summarizing or explaining the translation.
- Your output should be just the translated prompt, nothing else.
- Structured JSON objects should be translated as well by translating both the keys and values.

{source_lang}:
======START-PROMPT======
{text}
======END-PROMPT======

Output in {target_lang}:
"""


TRANSLATION_PROMPT_DATA2TXT = """
Translate the following prompt from {source_lang} to {target_lang}.
- Translate only the given prompt.
- Do not include any additional sentences summarizing or explaining the translation.
- Your output should be just the translated prompt, nothing else.
- Always translate JSON object as well by translating both the keys and values, e.g. "review_text": "..." -> should be translated to the language of the target language (e.g. "Bewertungstext": "...")

{source_lang}:
======START-PROMPT======
{text}
======END-PROMPT======

Output in {target_lang}:
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("translator")

# Match repo convention: load .env into process env for API credentials.
load_dotenv()


class TranslationError(Exception):
    """Exception raised for errors during translation."""

    pass


class RetryableTranslationError(Exception):
    """Exception raised for transient translation/API errors that should be retried."""

    pass


def setup_logging(output_dir: Path) -> None:
    """Set up logging to file in the output directory.

    :param output_dir: Directory to save log file
    """
    log_file = output_dir / "translation.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)


def get_openai_client() -> dict[str, Any]:
    """Get HTTP client configuration from environment variables.

    :return: Dict with `url` and `headers` for chat completion requests
    :raises ValueError: If API key is not set
    """
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Export it in your shell or load it from .env "
            "before running translation."
        )
    return {
        "url": f"{base_url.rstrip('/')}/chat/completions",
        "headers": {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    }


@retry(
    retry=retry_if_exception_type((RetryableTranslationError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        f"API call failed. Retrying in {retry_state.next_action.sleep if retry_state.next_action else 0} seconds... "
        f"(Attempt {retry_state.attempt_number}/5)"
    ),
)
async def translate_text(
    text: str,
    http_client: httpx.AsyncClient,
    url: str,
    semaphore: asyncio.Semaphore,
    model: str,
    task_type: str,
    source_lang: str = "EN",
    target_lang: str = "DE",
    prompt: bool = False,
) -> str:
    """Translate text using OpenAI-compatible HTTP API with automatic retries.

    :param text: Text to translate
    :param http_client: Shared async HTTP client
    :param url: API endpoint URL
    :param semaphore: Concurrency limiter
    :param model: Model to use for translation
    :param source_lang: Source language code
    :param target_lang: Target language code
    :param prompt: Whether the text is a prompt
    :return: Translated text
    :raises TranslationError: If translation fails after retries
    """
    if not text.strip():
        return ""

    try:
        translation_prompt = (
            TRANSLATION_ANSWER
            if prompt
            else TRANSLATION_PROMPT_DATA2TXT
            if task_type == "Data2txt"
            else TRANSLATION_PROMPT
        )
        translation_prompt = translation_prompt.format(
            source_lang=source_lang, target_lang=target_lang, text=text
        )

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": translation_prompt}],
            "temperature": 0.4,
        }

        async with semaphore:
            response = await http_client.post(url, json=payload)

        if response.status_code >= 400:
            try:
                error_obj = response.json().get("error", {})
                error_message = error_obj.get("message") or response.text
                error_type = error_obj.get("type")
                error_code = error_obj.get("code")
            except Exception:
                error_message = response.text
                error_type = None
                error_code = None

            message = (
                f"API error {response.status_code}"
                f" (type={error_type}, code={error_code}): {error_message}"
            )

            # Retry transient errors only.
            if response.status_code == 429 or response.status_code >= 500:
                logger.warning(message)
                if response.status_code == 429:
                    retry_after = response.headers.get("retry-after")
                    if retry_after:
                        try:
                            await asyncio.sleep(float(retry_after))
                        except ValueError:
                            pass
                raise RetryableTranslationError(message)

            # Do not retry non-transient client errors (e.g., bad request / context too long).
            raise TranslationError(message)

        response_json = response.json()

        # Strip lines starting with the character '='
        content = "\n".join(
            [
                line
                for line in response_json["choices"][0]["message"]["content"].split(
                    "\n"
                )
                if not line.strip().startswith("=")
            ]
        )

        return content.strip()
    except RetryableTranslationError:
        raise
    except TranslationError:
        raise
    except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as e:
        message = f"Transient network error: {e!s}"
        logger.warning(message)
        raise RetryableTranslationError(message) from e
    except Exception as e:
        logger.error(f"Translation error: {e!s}")
        raise TranslationError(f"Failed to translate text: {e!s}") from e


def merge_overlapping_spans(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge overlapping hallucination spans into a single span.

    :param labels: List of label spans to merge
    :return: List of merged spans
    """
    if not labels:
        return []

    labels_copy = sorted(labels, key=lambda x: x["start"])
    new_labels = []
    current_span = labels_copy[0].copy()

    for span in labels_copy[1:]:
        if span["start"] <= current_span["end"]:
            current_span["end"] = max(current_span["end"], span["end"])
        else:
            new_labels.append(current_span)
            current_span = span.copy()

    new_labels.append(current_span)
    return new_labels


def put_hallucination_tags(
    sample: HallucinationSample, answer: str
) -> tuple[str, list[dict[str, Any]]]:
    """Add hallucination tags to the text.

    :param sample: Sample containing labels
    :param answer: Text to add tags to
    :return: Tuple of (tagged text, merged labels)
    """
    # Skip the process if there are no labels
    if not sample.labels:
        return answer, []

    labels = merge_overlapping_spans(sample.labels)
    labels = sorted(labels, key=lambda x: (x["end"], x["start"]), reverse=True)
    tagged_answer = answer

    for label in labels:
        start, end = label["start"], label["end"]
        if start < 0 or end > len(tagged_answer) or start >= end:
            logger.warning(
                f"Invalid span: {start}-{end} for text of length {len(tagged_answer)}. Skipping."
            )
            continue

        tagged_answer = tagged_answer[:end] + "</HAL>" + tagged_answer[end:]
        tagged_answer = tagged_answer[:start] + "<HAL>" + tagged_answer[start:]

    return tagged_answer, labels


def find_hallucination_tags(
    text: str, labels: list[dict[str, Any]], sample_index: int
) -> tuple[list[tuple[int, int, str]], str]:
    """Find hallucination tags in the translated text and remove them.

    :param text: Text to search for tags
    :param labels: Original labels
    :param sample_index: Index of the sample
    :return: Tuple of (list of (start, end, label) tuples, cleaned text without HAL tags)
    """
    if not labels:
        return [], text

    # Find all <HAL> and </HAL> tags
    pattern = r"<(/?HAL)>"

    cleaned_text = ""
    open_tags = {}  # Maps an index to the starting position in cleaned text
    hal_spans = []  # List to store (start, end, label) tuples

    last_index = 0
    tag_count = 0

    for match in re.finditer(pattern, text):
        start, end = match.span()
        tag_type = match.group(1)

        cleaned_text += text[last_index:start]

        if tag_type == "HAL":  # Opening tag
            open_tags[tag_count] = len(cleaned_text)
        elif tag_type == "/HAL":  # Closing tag
            if tag_count in open_tags:
                label_text = "Unknown"
                if tag_count < len(labels):
                    label_text = labels[tag_count].get("label", "Unknown")
                else:
                    message = f"IndexError: No label for hallucinated text at sample ({sample_index}), span {tag_count}"
                    logger.warning(message)

                hal_spans.append((open_tags[tag_count], len(cleaned_text), label_text))
                tag_count += 1
            else:
                message = f"Warning: Found closing HAL tag without matching opening tag in sample {sample_index}"
                logger.warning(message)

        last_index = end

    # Add remaining text
    cleaned_text += text[last_index:]

    if tag_count < len(labels):
        message = f"Warning: Not all hallucination spans were found in sample {sample_index}. Found {tag_count}, expected {len(labels)}"
        logger.warning(message)

    return hal_spans, cleaned_text


async def translate_sample(
    sample: HallucinationSample,
    http_client: httpx.AsyncClient,
    url: str,
    semaphore: asyncio.Semaphore,
    model: str,
    sample_index: int,
    log_file: Path,
    source_lang: str,
    target_lang: str,
    dataset: str,
) -> HallucinationSample | None:
    """Translate a single sample.

    :param sample: Sample to translate
    :param http_client: Shared async HTTP client
    :param url: API endpoint URL
    :param semaphore: Concurrency limiter
    :param model: Model to use
    :param sample_index: Sample index
    :param log_file: Path to log file
    :param source_lang: Source language code
    :param target_lang: Target language code
    :param dataset: Dataset name
    :return: Translated sample or None if translation failed
    """
    try:
        # Skip processing if we have an empty sample
        if not sample.prompt.strip() or not sample.answer.strip():
            logger.warning(
                f"Sample {sample_index} has empty prompt or answer. Skipping."
            )
            return None

        tagged_answer, labels = put_hallucination_tags(sample, sample.answer)

        prompt_task = asyncio.create_task(
            translate_text(
                sample.prompt,
                http_client,
                url,
                semaphore,
                model,
                sample.task_type,
                source_lang,
                target_lang,
            )
        )
        answer_task = asyncio.create_task(
            translate_text(
                tagged_answer,
                http_client,
                url,
                semaphore,
                model,
                sample.task_type,
                source_lang,
                target_lang,
                prompt=True,
            )
        )
        translated_prompt, translated_answer = await asyncio.gather(
            prompt_task, answer_task
        )

        # Default to the translated answer (will be replaced if there are hallucination spans)
        cleaned_answer = translated_answer

        labels = []
        if sample.labels:
            # Get hallucination spans and cleaned text (without HAL tags)
            hal_spans, cleaned_answer = find_hallucination_tags(
                translated_answer, sample.labels, sample_index
            )

            for span in hal_spans:
                start, end, label = span
                labels.append({"start": start, "end": end, "label": label})

        return HallucinationSample(
            prompt=translated_prompt,
            answer=cleaned_answer,
            labels=labels,
            split=sample.split,
            task_type=sample.task_type,
            dataset=dataset,
            language=target_lang.lower(),
        )
    except Exception as e:
        logger.error(f"Error translating sample {sample_index}: {e!s}")
        with open(log_file, "a") as log:
            log.write(f"Error translating sample {sample_index}: {e!s}\n")
        return None


def load_check_existing_data(output_file: Path) -> HallucinationData:
    """Load existing data or create new data.

    :param output_file: Path to the output file
    :return: Existing HallucinationData or new empty HallucinationData
    """
    if output_file.exists():
        try:
            return HallucinationData.from_json(json.loads(output_file.read_text()))
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(
                f"Error loading existing data: {e!s}. Starting with empty dataset."
            )
            return HallucinationData(samples=[])
    else:
        return HallucinationData(samples=[])


async def process_batch(
    samples: list[HallucinationSample],
    http_client: httpx.AsyncClient,
    url: str,
    semaphore: asyncio.Semaphore,
    model: str,
    start_idx: int,
    log_file: Path,
    source_lang: str,
    target_lang: str,
    dataset: str,
) -> list[HallucinationSample]:
    """Process a batch of samples concurrently using asyncio.

    :param samples: List of samples to process
    :param http_client: Shared async HTTP client
    :param url: API endpoint URL
    :param semaphore: Concurrency limiter
    :param model: Model to use
    :param start_idx: Starting index for the batch
    :param log_file: Path to log file
    :param source_lang: Source language code
    :param target_lang: Target language code
    :param dataset: Dataset name
    :return: List of translated samples (excluding failed translations)
    """
    tasks = []
    for i, sample in enumerate(samples, start=start_idx):
        task = asyncio.create_task(
            translate_sample(
                sample,
                http_client,
                url,
                semaphore,
                model,
                i,
                log_file,
                source_lang,
                target_lang,
                dataset,
            )
        )
        tasks.append(task)

    results = []
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in batch_results:
        if isinstance(result, Exception):
            logger.error(f"Error in sample processing: {result!s}")
        elif result:
            results.append(result)

    return results


async def run_translation(
    remaining_samples: list[HallucinationSample],
    translated_data: HallucinationData,
    output_file: Path,
    dataset: str,
    target_lang: str,
    output_dir: Path,
    model: str,
    source_lang: str,
    total_samples: int,
    num_processed: int,
    batch_size: int,
    max_workers: int,
    test: bool,
    log_file: Path,
    client_config: dict[str, Any],
) -> None:
    """Run batched async translation with connection pooling.

    :param remaining_samples: Samples that still need translation
    :param translated_data: Existing translated data to append to
    :param output_file: Output JSON file path
    :param dataset: Dataset name
    :param target_lang: Target language code
    :param output_dir: Output directory
    :param model: Model to use
    :param source_lang: Source language code
    :param total_samples: Number of remaining samples
    :param num_processed: Number of already translated samples from resume
    :param batch_size: Number of samples per batch
    :param max_workers: Maximum concurrent API requests
    :param test: Whether to run in test mode
    :param log_file: Path to error log
    :param client_config: API URL and headers
    """
    progress_bar = tqdm.tqdm(total=total_samples, desc="Translating")
    start_time = time.time()
    save_interval = 60
    last_save_time = start_time

    limits = httpx.Limits(
        max_connections=max_workers * 2, max_keepalive_connections=max_workers
    )
    timeout = httpx.Timeout(60.0)
    semaphore = asyncio.Semaphore(max_workers)

    try:
        async with httpx.AsyncClient(
            headers=client_config["headers"], timeout=timeout, limits=limits
        ) as http_client:
            for i in range(0, total_samples, batch_size):
                if test and i > 0:
                    break

                batch = remaining_samples[i : i + batch_size]
                batch_results = await process_batch(
                    batch,
                    http_client,
                    client_config["url"],
                    semaphore,
                    model,
                    num_processed + i,
                    log_file,
                    source_lang,
                    target_lang,
                    dataset,
                )

                translated_data.samples.extend(batch_results)
                progress_bar.update(len(batch_results))

                current_time = time.time()
                if (
                    current_time - last_save_time > save_interval
                    or i + batch_size >= total_samples
                ):
                    save_progress(
                        translated_data, output_file, dataset, target_lang, output_dir
                    )
                    last_save_time = current_time

                current_count = len(translated_data.samples)
                elapsed_time = current_time - start_time
                samples_per_sec = (
                    current_count / elapsed_time if elapsed_time > 0 else 0
                )

                logger.info(
                    f"Processed {current_count}/{num_processed + total_samples} samples "
                    f"({samples_per_sec:.2f} samples/sec)"
                )
    finally:
        progress_bar.close()


def save_progress(
    translated_data: HallucinationData,
    output_file: Path,
    dataset: str,
    target_lang: str,
    output_dir: Path,
) -> None:
    """Save progress to file with backup handling.

    :param translated_data: Data to save
    :param output_file: Primary output file
    :param dataset: Dataset name for backup file
    :param target_lang: Target language for backup file
    :param output_dir: Output directory for backup file
    """
    try:
        output_file.write_text(json.dumps(translated_data.to_json(), indent=2))
    except Exception as e:
        logger.error(f"Error saving progress: {e!s}")
        # Try to save to a backup file
        backup_file = (
            output_dir
            / f"{dataset}_data_{target_lang.lower()}_backup_{int(time.time())}.json"
        )
        try:
            backup_file.write_text(json.dumps(translated_data.to_json(), indent=2))
            logger.info(f"Saved backup to {backup_file}")
        except Exception as e2:
            logger.error(f"Error saving backup: {e2!s}")


def push_translated_data_to_hub(
    translated_data: HallucinationData, repo_id: str, config_name: str, private: bool
) -> None:
    """Push translated hallucination data to Hugging Face Hub.

    :param translated_data: Translated samples to push
    :param repo_id: Target Hugging Face dataset repo id
    :param config_name: Dataset config/subset name (typically language code)
    :param private: Whether to keep dataset private on Hub
    """
    if not translated_data.samples:
        logger.warning("No translated samples available; skipping Hub upload.")
        return

    rows = [
        {
            "prompt": sample.prompt,
            "answer": sample.answer,
            "labels": sample.labels,
            "split": sample.split,
            "task_type": sample.task_type,
            "dataset": sample.dataset,
            "language": sample.language,
        }
        for sample in translated_data.samples
    ]

    dataset = Dataset.from_list(rows)
    dataset.push_to_hub(repo_id=repo_id, config_name=config_name, private=private)
    logger.info(
        "Pushed translated dataset to hub: %s (config=%s)", repo_id, config_name
    )


def push_test_subset_to_hub(
    translated_data: HallucinationData,
    repo_id: str,
    config_name: str,
    private: bool,
    n: int = 2048,
    validation_n: int = 256,
) -> None:
    """Push test and validation subsets of translated data to the Hub.

    :param translated_data: Translated samples to filter and push
    :param repo_id: Target Hugging Face dataset repo id
    :param config_name: Dataset config/subset name (typically language code)
    :param private: Whether to keep dataset private on Hub
    :param n: Maximum number of test samples to upload
    :param validation_n: Maximum number of validation samples to upload
    """
    samples = translated_data.samples[:]
    random.shuffle(samples)

    def _to_rows(items: list[HallucinationSample]) -> list[dict[str, Any]]:
        return [
            {
                "prompt": sample.prompt,
                "answer": sample.answer,
                "labels": sample.labels,
                "split": sample.split,
                "task_type": sample.task_type,
                "dataset": sample.dataset,
                "language": sample.language,
            }
            for sample in items
        ]

    test_samples = [s for s in samples if s.split == "test"][:n]
    if test_samples:
        dataset = Dataset.from_list(_to_rows(test_samples))
        dataset.push_to_hub(
            repo_id=repo_id, config_name=config_name, split="test", private=private
        )
        logger.info(
            "Pushed %d test samples to hub: %s (config=%s)",
            len(test_samples),
            repo_id,
            config_name,
        )
    else:
        logger.warning("No test samples available; skipping test split upload.")

    # RAGTruth only ships train/test splits, so carve the validation split from train.
    validation_samples = [s for s in samples if s.split == "train"][:validation_n]
    if validation_samples:
        dataset = Dataset.from_list(_to_rows(validation_samples))
        dataset.push_to_hub(
            repo_id=repo_id, config_name=config_name, split="val", private=private
        )
        logger.info(
            "Pushed %d validation samples to hub: %s (config=%s)",
            len(validation_samples),
            repo_id,
            config_name,
        )
    else:
        logger.warning(
            "No validation samples available; skipping val split upload."
        )


def main(
    input_dir: Path,
    output_dir: Path,
    model: str,
    source_lang: str,
    target_lang: str,
    dataset: str = "ragtruth",
    batch_size: int = 5,
    max_workers: int = 5,
    resume: bool = True,
    test: bool = False,
    push_to_hub: bool = False,
    hub_repo_id: str | None = None,
    private: bool = True,
    push_test_subset: bool = False,
    test_subset_repo_id: str | None = None,
    test_subset_size: int = 2048,
    validation_subset_size: int = 256,
) -> None:
    """Translates the preprocessed data using parallel processing.

    :param input_dir: Path to the input directory
    :param output_dir: Path to the output directory
    :param model: OpenAI model to use
    :param source_lang: Source language code
    :param target_lang: Target language code
    :param dataset: Dataset name (ragtruth, ragbench, etc.)
    :param batch_size: Number of samples to process in each batch
    :param max_workers: Maximum number of concurrent API requests
    :param resume: Whether to resume from previous run
    :param test: Test mode, only translate 1 sample
    :param push_to_hub: Whether to push translated output to Hugging Face Hub
    :param hub_repo_id: Optional explicit Hugging Face dataset repo id
    :param private: Whether pushed dataset should be private
    :param push_test_subset: Whether to also push a test-only subset to the Hub
    :param test_subset_repo_id: Optional explicit repo id for the test subset
    :param test_subset_size: Maximum number of test samples in the subset
    :param validation_subset_size: Maximum number of validation samples in the subset
    """
    # Set up directories
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    setup_logging(output_dir)

    # Set up files
    input_file = input_dir / f"{dataset}_data.json"
    output_file = output_dir / f"{dataset}_data_{target_lang.lower()}.json"
    log_file = output_dir / "error_log.txt"

    # Load existing translated data if resume is enabled
    if resume and output_file.exists():
        translated_data = load_check_existing_data(output_file=output_file)
        num_processed = len(translated_data.samples)
        logger.info(f"Resuming from {num_processed} previously translated samples")
    else:
        translated_data = HallucinationData(samples=[])
        num_processed = 0

    # Load source data. If we already have a complete translated output and the
    # source file is missing, allow push-only mode without requiring the input.
    if input_file.exists():
        try:
            data = HallucinationData.from_json(json.loads(input_file.read_text()))
        except Exception as e:
            logger.error(f"Error loading input data: {e!s}")
            raise
        remaining_samples = data.samples[num_processed:]
    elif num_processed > 0:
        logger.warning(
            f"Input file not found: {input_file}. "
            f"Proceeding in push-only mode with {num_processed} existing translated samples."
        )
        remaining_samples = []
    else:
        logger.error(f"Input file not found: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")

    total_samples = len(remaining_samples)

    if total_samples == 0:
        logger.info("No samples to translate. Exiting.")
        if push_to_hub:
            resolved_repo_id = (
                hub_repo_id
                if hub_repo_id
                else f"EuroEval/{dataset}-translated-hallucinations"
            )
            push_translated_data_to_hub(
                translated_data=translated_data,
                repo_id=resolved_repo_id,
                config_name=target_lang.lower(),
                private=private,
            )
        if push_test_subset:
            resolved_test_repo_id = (
                test_subset_repo_id
                if test_subset_repo_id
                else f"EuroEval/{dataset}-translated-hallucinations-{target_lang.lower()}-mini"
            )
            push_test_subset_to_hub(
                translated_data=translated_data,
                repo_id=resolved_test_repo_id,
                config_name=target_lang.lower(),
                private=private,
                n=test_subset_size,
                validation_n=validation_subset_size,
            )
        return

    # Get OpenAI client
    client = get_openai_client()

    logger.info(f"Translating {dataset} from {source_lang} to {target_lang}")
    logger.info(f"Using model: {model}")
    logger.info(f"Total samples to process: {total_samples}")
    logger.info(f"Batch size: {batch_size}, Max workers: {max_workers}")

    try:
        asyncio.run(
            run_translation(
                remaining_samples=remaining_samples,
                translated_data=translated_data,
                output_file=output_file,
                dataset=dataset,
                target_lang=target_lang,
                output_dir=output_dir,
                model=model,
                source_lang=source_lang,
                total_samples=total_samples,
                num_processed=num_processed,
                batch_size=batch_size,
                max_workers=max_workers,
                test=test,
                log_file=log_file,
                client_config=client,
            )
        )

    except KeyboardInterrupt:
        logger.info("Translation interrupted by user. Saving progress...")
        save_progress(translated_data, output_file, dataset, target_lang, output_dir)
        logger.info(
            f"Saved {len(translated_data.samples)} translated samples to {output_file}"
        )
        return

    except Exception as e:
        logger.error(f"Unexpected error: {e!s}")
        save_progress(translated_data, output_file, dataset, target_lang, output_dir)
        raise

    logger.info(
        f"Translation complete. Translated {len(translated_data.samples)} samples."
    )
    logger.info(f"Output saved to {output_file}")

    if push_to_hub:
        resolved_repo_id = (
            hub_repo_id
            if hub_repo_id
            else f"EuroEval/{dataset}-translated-hallucinations"
        )
        push_translated_data_to_hub(
            translated_data=translated_data,
            repo_id=resolved_repo_id,
            config_name=target_lang.lower(),
            private=private,
        )

    if push_test_subset:
        resolved_test_repo_id = (
            test_subset_repo_id
            if test_subset_repo_id
            else f"EuroEval/{dataset}-translated-hallucinations-{target_lang.lower()}-mini"
        )
        push_test_subset_to_hub(
            translated_data=translated_data,
            repo_id=resolved_test_repo_id,
            config_name=target_lang.lower(),
            private=private,
            n=test_subset_size,
            validation_n=validation_subset_size,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate hallucination dataset to another language"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input data files",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save output files"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use for translation",
    )
    parser.add_argument(
        "--source-lang",
        type=str,
        default="EN",
        help="Source language code (e.g., EN, DE, FR, etc.)",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="DA",
        help="Target language code (e.g., EN, DE, FR, etc.)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ragtruth",
        help="Dataset name (ragtruth, ragbench, etc.)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=30,
        help="Number of samples to process in each batch",
    )
    parser.add_argument(
        "--max-workers", type=int, default=30, help="Maximum number of worker threads"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from previous run, start fresh",
    )
    parser.add_argument(
        "--test", action="store_true", help="Test mode, only translate 1 sample"
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push translated dataset to Hugging Face Hub after translation",
    )
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        default=None,
        help=(
            "Target Hugging Face dataset repo id. "
            "Default: EuroEval/<dataset>-translated-hallucinations"
        ),
    )
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether uploaded Hub dataset should be private",
    )
    parser.add_argument(
        "--push-test-subset",
        action="store_true",
        help="Also push a test-only subset (up to --test-subset-size samples) to the Hub",
    )
    parser.add_argument(
        "--test-subset-repo-id",
        type=str,
        default=None,
        help=(
            "Target Hugging Face dataset repo id for the test subset. "
            "Default: EuroEval/<dataset>-translated-hallucinations-<lang>-mini"
        ),
    )
    parser.add_argument(
        "--test-subset-size",
        type=int,
        default=1000,
        help="Maximum number of test samples to include in the test subset (default: 1000)",
    )
    parser.add_argument(
        "--validation-subset-size",
        type=int,
        default=256,
        help=(
            "Maximum number of validation samples to include in the test subset "
            "(default: 256)"
        ),
    )
    args = parser.parse_args()
    main(
        Path(args.input_dir),
        Path(args.output_dir),
        args.model,
        args.source_lang,
        args.target_lang,
        args.dataset,
        args.batch_size,
        args.max_workers,
        not args.no_resume,
        args.test,
        args.push_to_hub,
        args.hub_repo_id,
        args.private,
        args.push_test_subset,
        args.test_subset_repo_id,
        args.test_subset_size,
        args.validation_subset_size,
    )
