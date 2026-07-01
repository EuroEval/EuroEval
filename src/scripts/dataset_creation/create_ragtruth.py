# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==5.0.0",
#     "huggingface-hub==1.20.1",
#     "httpx==0.28.1",
#     "python-dotenv==1.0.1",
#     "lettucedetect==0.1.8",
# ]
# ///

"""Download and translate RAGTruth hallucination dataset.

Downloads source_info.jsonl and response.jsonl from RAGTruth GitHub repository,
joins them, and translates to all 30 European target languages while preserving
hallucination spans.
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
import time
import typing as t
from pathlib import Path

import httpx
import tqdm
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from lettucedetect.datasets.hallucination_dataset import (
    HallucinationData,
    HallucinationSample,
)

from euroeval.languages import (
    ALBANIAN,
    BELARUSIAN,
    BOSNIAN,
    BULGARIAN,
    CATALAN,
    CROATIAN,
    CZECH,
    DANISH,
    DUTCH,
    ENGLISH,
    ESTONIAN,
    FAROESE,
    FINNISH,
    FRENCH,
    GERMAN,
    GREEK,
    HUNGARIAN,
    ICELANDIC,
    ITALIAN,
    LATVIAN,
    LITHUANIAN,
    NORWEGIAN,
    POLISH,
    PORTUGUESE,
    ROMANIAN,
    SERBIAN,
    SLOVAK,
    SLOVENE,
    SPANISH,
    SWEDISH,
    UKRAINIAN,
    Language,
)

# =============================================================================
# Configuration constants
# =============================================================================

# RAGTruth data source
SOURCE_INFO_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/refs/heads/main/dataset/source_info.jsonl"
RESPONSE_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/refs/heads/main/dataset/response.jsonl"

# Translation settings
SOURCE_LANG: Language = ENGLISH

TARGET_LANGS: list[Language] = [
    ALBANIAN,
    BELARUSIAN,
    BOSNIAN,
    BULGARIAN,
    CATALAN,
    CROATIAN,
    CZECH,
    DANISH,
    DUTCH,
    ESTONIAN,
    FAROESE,
    FINNISH,
    FRENCH,
    GERMAN,
    GREEK,
    HUNGARIAN,
    ICELANDIC,
    ITALIAN,
    LATVIAN,
    LITHUANIAN,
    NORWEGIAN,
    POLISH,
    PORTUGUESE,
    ROMANIAN,
    SERBIAN,
    SLOVAK,
    SLOVENE,
    SPANISH,
    SWEDISH,
    UKRAINIAN,
]
# MAX_WORKERS bounds the number of in-flight API requests (each sample issues two:
# prompt + answer). BATCH_SIZE is kept well above MAX_WORKERS so the worker pool
# stays saturated between the per-batch save points instead of draining to zero at
# every batch boundary. The practical ceiling here is network egress, not the API
# rate limit: on the inference-server host ~400 workers caused escalating
# connection retries, while 100 ran cleanly. Tune via --max-workers per environment.
BATCH_SIZE = 400
MAX_WORKERS = 100

# Retry settings for transient API errors (HTTP 429/5xx and network blips).
MAX_TRANSLATION_RETRIES = 5
RETRY_BASE_DELAY_SECONDS = 2.0
RETRY_MAX_DELAY_SECONDS = 60.0

# API pricing in USD per 1M tokens as (input_price, output_price), keyed by model
# ID, used only for cost estimation. Values as listed on the OpenAI pricing page
# (https://developers.openai.com/api/docs/pricing); update as prices change.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    # gpt-4.1 series (legacy; standard published rates, no longer on the page).
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    # gpt-5.4 / gpt-5.5 series (current page).
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-pro": (30.00, 180.00),
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.5-pro": (30.00, 180.00),
}

# Running tally of measured token usage, populated by translate_text /
# translate_sample. Used in test mode to extrapolate full-dataset cost.
TOKEN_USAGE: dict[str, int] = {
    "input_tokens": 0,
    "output_tokens": 0,
    "api_calls": 0,
    "samples": 0,
}

# Output settings
OUTPUT_DIR = Path("data/ragtruth")
DATASET_NAME = "ragtruth"

# HF Hub settings
PUSH_TO_HUB = True
HUB_REPO_ID = "alexandrainst/ragtruth-translated-hallucinations"
PRIVATE_UPLOAD = True
PUSH_TEST_SUBSET = True
TEST_SUBSET_REPO_ID = "EuroEval/ragtruth-translated-hallucinations-{lang}-mini"
TEST_SUBSET_SIZE = 1000
VALIDATION_SUBSET_SIZE = 256


# =============================================================================
# Translation prompts
# =============================================================================

TRANSLATION_ANSWER = (
    "\n"
    "Translate the following text from {source_lang} to {target_lang}.\n"
    "- The text may contain <HAL> ... </HAL> tags. Translate the text (including "
    "the words inside the tags) and keep each pair of tags around the translation "
    "of the same content it surrounded in the original.\n"
    "- Copy the tags EXACTLY as the two tokens `<HAL>` and `</HAL>`. NEVER write "
    "variants such as `<HAL<HAL>>`, `<<HAL>>`, `< HAL >`, `<HAL></HAL><HAL>` or "
    "`</HAL</HAL>>`.\n"
    "- NEVER nest or duplicate tags: every `<HAL>` must be closed by exactly one "
    "matching `</HAL>`, tags may not appear inside other tags, and the output must "
    "contain the same number of `<HAL>...</HAL>` pairs as the input.\n"
    "- If the original text does not contain any <HAL> tags, just translate the "
    "text and do NOT add any tags.\n"
    "- Do not include any additional sentences summarising or explaining the "
    "translation.\n"
    "- Your output should be just the translated text, nothing else.\n"
    "\n"
    "Example (English to German):\n"
    "Input: The capital <HAL>and largest city</HAL> of France is Paris.\n"
    "Output: Die Hauptstadt <HAL>und größte Stadt</HAL> von Frankreich ist Paris.\n"
    "\n"
    "{source_lang}:\n"
    "======START======\n"
    "{text}\n"
    "======END======\n"
    "\n"
    "Output in {target_lang}:\n"
)

TRANSLATION_PROMPT = (
    "\n"
    "Translate the following prompt from {source_lang} to {target_lang}.\n"
    "- Translate only the given prompt.\n"
    "- Do not include any additional sentences summarising or explaining the "
    "translation.\n"
    "- Your output should be just the translated prompt, nothing else.\n"
    "- Structured JSON objects should be translated as well by translating both "
    "the keys and values.\n"
    "\n"
    "{source_lang}:\n"
    "======START-PROMPT======\n"
    "{text}\n"
    "======END-PROMPT======\n"
    "\n"
    "Output in {target_lang}:\n"
)


TRANSLATION_PROMPT_DATA2TXT = (
    "\n"
    "Translate the following prompt from {source_lang} to {target_lang}.\n"
    "- Translate only the given prompt.\n"
    "- Do not include any additional sentences summarising or explaining the "
    "translation.\n"
    "- Your output should be just the translated prompt, nothing else.\n"
    "- Always translate JSON object as well by translating both the keys and "
    'values, e.g. "review_text": "..." -> should be translated to the language '
    'of the target language (e.g. "Bewertungstext": "...")\n'
    "\n"
    "{source_lang}:\n"
    "======START-PROMPT======\n"
    "{text}\n"
    "======END-PROMPT======\n"
    "\n"
    "Output in {target_lang}:\n"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("translator")

load_dotenv()


class TranslationError(Exception):
    """Exception raised for errors during translation."""

    pass


class RetryableTranslationError(Exception):
    """Exception raised for transient translation/API errors that should be retried."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        """Initialise the error.

        Args:
            message:
                The error message.
            retry_after:
                Server-provided number of seconds to wait before retrying, if any.
        """
        super().__init__(message)
        self.retry_after = retry_after


class ClientConfig(t.TypedDict):
    """Configuration for OpenAI client."""

    url: str
    headers: dict[str, str]


def main() -> None:
    """Download RAGTruth data, translate to all target languages, and upload to Hub."""
    parser = argparse.ArgumentParser(
        description="Download and translate the RAGTruth hallucination dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model ID to use for translation (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help=(
            "Max concurrent in-flight API requests. Raise it if your rate-limit "
            f"tier allows (default: {MAX_WORKERS})."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=(
            "Number of samples processed between save points; keep it well above "
            f"--max-workers to keep the worker pool saturated (default: {BATCH_SIZE})."
        ),
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run a limited smoke test (few samples, no Hub uploads).",
    )
    parser.add_argument(
        "--test-num-samples",
        type=int,
        default=10,
        help="Number of samples to translate per language in test mode.",
    )
    parser.add_argument(
        "--test-all-languages",
        action="store_true",
        help="In test mode, translate all target languages instead of only Danish.",
    )
    args = parser.parse_args()
    model = args.model

    # Set up directories
    input_dir = OUTPUT_DIR
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    setup_logging(output_dir)

    # Load source data if available
    input_file = input_dir / f"{DATASET_NAME}_data.json"
    data: HallucinationData | None = None
    if input_file.exists():
        try:
            data = HallucinationData.from_json(json.loads(input_file.read_text()))
        except Exception as e:
            logger.error(f"Error loading input data: {e!s}")
            raise
    if data is None:
        # Download and join RAGTruth dataset from GitHub
        logger.info(
            f"Source data not found locally. Downloading from GitHub: {input_file}"
        )
        data = load_ragtruth_data()

        # Cache the downloaded data for future runs
        input_file.write_text(json.dumps(data.to_json(), separators=(",", ":")))
        logger.info(f"Cached source data to {input_file}")

    # Get OpenAI client once (shared across all languages)
    client = get_openai_client()

    # In test mode we smoke-test a single language (Danish) unless the caller asks
    # for all languages.
    if args.test_mode and not args.test_all_languages:
        target_langs = [DANISH]
    else:
        target_langs = TARGET_LANGS

    # Translate to each target language
    for target_lang in target_langs:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Translating to {target_lang.name} ({target_lang.code})")
        logger.info(f"{'=' * 60}")

        _translate_to_language(
            source_data=data,
            target_lang=target_lang,
            client_config=client,
            model=model,
            max_workers=args.max_workers,
            batch_size=args.batch_size,
            test_mode=args.test_mode,
            test_num_samples=args.test_num_samples,
        )

    # In test mode, report measured token usage and extrapolate to the full run.
    if args.test_mode:
        _report_token_estimate(total_source_samples=len(data.samples), model=model)


def download_jsonl(url: str) -> list[dict[str, t.Any]]:
    """Download a JSONL file from a URL with streaming to reduce memory usage.

    Args:
        url: URL to the JSONL file.

    Returns:
        List of parsed JSON objects from each line.
    """
    logger.info(f"Downloading {url}...")
    with httpx.Client(timeout=300.0) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            lines = []
            for line in response.iter_lines():
                line = line.strip()
                if line:
                    lines.append(json.loads(line))
            return lines


def load_ragtruth_data() -> HallucinationData:
    """Download and join RAGTruth dataset from GitHub.

    Returns:
        HallucinationData with joined samples.
    """
    source_info_list = download_jsonl(SOURCE_INFO_URL)
    response_list = download_jsonl(RESPONSE_URL)

    logger.info(f"Downloaded {len(source_info_list)} source records")
    logger.info(f"Downloaded {len(response_list)} response records")

    source_lookup: dict[str, dict[str, t.Any]] = {
        src["source_id"]: src for src in source_info_list
    }

    samples: list[HallucinationSample] = []
    for resp in response_list:
        source_id = resp.get("source_id")
        if source_id not in source_lookup:
            logger.warning(f"No source info for response {resp.get('id')}")
            continue

        src = source_lookup[source_id]
        labels = [
            {"start": lbl["start"], "end": lbl["end"], "label": lbl["label_type"]}
            for lbl in resp.get("labels", [])
        ]

        samples.append(
            HallucinationSample(
                prompt=src.get("prompt", ""),
                answer=resp.get("response", ""),
                labels=labels,
                split=resp.get("split", "train"),
                task_type=src.get("task_type", "unknown"),
                dataset="ragtruth",
                language="en",
            )
        )

    logger.info(f"Created {len(samples)} samples from RAGTruth dataset")
    return HallucinationData(samples=samples)


def load_check_existing_data(output_file: Path) -> tuple[HallucinationData, int | None]:
    """Load existing data or create new data.

    Args:
        output_file: Path to the output file.

    Returns:
        Tuple of (HallucinationData, last_processed_index). last_processed_index
        is None if the file doesn't exist or has no metadata.
    """
    if output_file.exists():
        try:
            saved = json.loads(output_file.read_text())
            # save_progress writes {"samples": [...], "_metadata": {...}}, but
            # HallucinationData.from_json expects the bare list of samples. Also
            # tolerate an older bare-list format.
            last_processed_index = None
            if isinstance(saved, dict):
                metadata = saved.get("_metadata") or {}
                last_processed_index = metadata.get("last_processed_index")
                samples_json = saved.get("samples", [])
            else:
                samples_json = saved
            return HallucinationData.from_json(samples_json), last_processed_index
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(
                f"Error loading existing data: {e!s}. Starting with empty dataset."
            )
            return HallucinationData(samples=[]), None
    else:
        return HallucinationData(samples=[]), None


def _setup_http_client(
    max_workers: int,
) -> tuple[httpx.Limits, httpx.Timeout, asyncio.Semaphore]:
    """Set up HTTP client configuration.

    Args:
        max_workers:
            Maximum number of concurrent workers.

    Returns:
        Tuple of (limits, timeout, semaphore).
    """
    limits = httpx.Limits(
        max_connections=max_workers * 2, max_keepalive_connections=max_workers
    )
    timeout = httpx.Timeout(60.0)
    semaphore = asyncio.Semaphore(max_workers)
    return limits, timeout, semaphore


def _log_translation_progress(current_count: int, total: int, elapsed: float) -> None:
    """Log translation progress.

    Args:
        current_count:
            Number of samples processed so far.
        total:
            Total number of samples.
        elapsed:
            Elapsed time in seconds.
    """
    samples_per_sec = current_count / elapsed if elapsed > 0 else 0
    logger.info(
        f"Processed {current_count}/{total} samples ({samples_per_sec:.2f} samples/sec)"
    )


async def run_translation(
    remaining_samples: list[HallucinationSample],
    translated_data: HallucinationData,
    output_file: Path,
    target_lang: Language,
    num_processed: int,
    client_config: ClientConfig,
    model: str,
    max_workers: int,
    batch_size: int,
) -> None:
    """Run batched async translation with connection pooling.

    Args:
        remaining_samples:
            Samples that still need translation.
        translated_data:
            Existing translated data to append to.
        model:
            Model ID to use for translation.
        output_file:
            Output JSON file path.
        target_lang:
            Target language code.
        num_processed:
            Number of already translated samples from cache.
        client_config:
            API URL and headers.
        max_workers:
            Maximum number of concurrent in-flight API requests.
        batch_size:
            Number of samples processed between save points.
    """
    total_samples = len(remaining_samples)
    log_file = OUTPUT_DIR / "error_log.txt"

    limits, timeout, semaphore = _setup_http_client(max_workers)

    progress_bar = tqdm.tqdm(total=total_samples, desc="Translating")
    start_time = time.time()
    save_interval = 60
    last_save_time = start_time

    try:
        async with httpx.AsyncClient(
            headers=client_config["headers"], timeout=timeout, limits=limits
        ) as http_client:
            for i in range(0, total_samples, batch_size):
                batch = remaining_samples[i : i + batch_size]
                batch_results = await process_batch(
                    samples=batch,
                    http_client=http_client,
                    url=client_config["url"],
                    semaphore=semaphore,
                    model=model,
                    start_idx=num_processed + i,
                    log_file=log_file,
                    source_lang=SOURCE_LANG,
                    target_lang=target_lang,
                    dataset=DATASET_NAME,
                )

                translated_data.samples.extend(batch_results)
                progress_bar.update(len(batch_results))

                # Update last_processed_index by batch size (not results count)
                last_processed_index = num_processed + i + len(batch)

                current_time = time.time()
                if (
                    current_time - last_save_time > save_interval
                    or i + batch_size >= total_samples
                ):
                    save_progress(
                        translated_data=translated_data,
                        output_file=output_file,
                        target_lang=target_lang,
                        last_processed_index=last_processed_index,
                    )
                    last_save_time = current_time

                current_count = len(translated_data.samples)
                elapsed_time = current_time - start_time
                _log_translation_progress(
                    current_count=current_count,
                    total=num_processed + total_samples,
                    elapsed=elapsed_time,
                )
    finally:
        progress_bar.close()


async def process_batch(
    samples: list[HallucinationSample],
    http_client: httpx.AsyncClient,
    url: str,
    semaphore: asyncio.Semaphore,
    model: str,
    start_idx: int,
    log_file: Path,
    source_lang: Language,
    target_lang: Language,
    dataset: str,
) -> list[HallucinationSample]:
    """Process a batch of samples concurrently using asyncio.

    Args:
        samples: List of samples to process.
        http_client: Shared async HTTP client.
        url: API endpoint URL.
        semaphore: Concurrency limiter.
        model: Model to use.
        start_idx: Starting index for the batch.
        log_file: Path to log file.
        source_lang: Source language code.
        target_lang: Target language code.
        dataset: Dataset name.

    Returns:
        List of translated samples (excluding failed translations).
    """
    tasks = []
    for i, sample in enumerate(samples, start=start_idx):
        task = asyncio.create_task(
            translate_sample(
                sample=sample,
                http_client=http_client,
                url=url,
                semaphore=semaphore,
                model=model,
                sample_index=i,
                log_file=log_file,
                source_lang=source_lang,
                target_lang=target_lang,
                dataset=dataset,
            )
        )
        tasks.append(task)

    results = []
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in batch_results:
        if isinstance(result, BaseException):
            logger.error(f"Error in sample processing: {result!s}")
        elif result:
            results.append(result)

    return results


async def translate_sample(
    sample: HallucinationSample,
    http_client: httpx.AsyncClient,
    url: str,
    semaphore: asyncio.Semaphore,
    model: str,
    sample_index: int,
    log_file: Path,
    source_lang: Language,
    target_lang: Language,
    dataset: str,
) -> HallucinationSample | None:
    """Translate a single sample.

    Args:
        sample: Sample to translate.
        http_client: Shared async HTTP client.
        url: API endpoint URL.
        semaphore: Concurrency limiter.
        model: Model to use.
        sample_index: Sample index.
        log_file: Path to log file.
        source_lang: Source language code.
        target_lang: Target language code.
        dataset: Dataset name.

    Returns:
        Translated sample or None if translation failed.
    """
    try:
        if not sample.prompt.strip() or not sample.answer.strip():
            logger.warning(
                f"Sample {sample_index} has empty prompt or answer. Skipping."
            )
            return None

        tagged_answer, merged_labels = put_hallucination_tags(
            sample=sample, answer=sample.answer
        )

        prompt_task = asyncio.create_task(
            translate_text(
                text=sample.prompt,
                http_client=http_client,
                url=url,
                semaphore=semaphore,
                model=model,
                task_type=sample.task_type,
                source_lang=source_lang,
                target_lang=target_lang,
            )
        )
        answer_task = asyncio.create_task(
            translate_text(
                text=tagged_answer,
                http_client=http_client,
                url=url,
                semaphore=semaphore,
                model=model,
                task_type=sample.task_type,
                source_lang=source_lang,
                target_lang=target_lang,
                prompt=True,
            )
        )
        translated_prompt, translated_answer = await asyncio.gather(
            prompt_task, answer_task
        )

        cleaned_answer = translated_answer

        labels = []
        if merged_labels:
            # Use merged labels from put_hallucination_tags to ensure alignment
            hal_spans, cleaned_answer = find_hallucination_tags(
                text=translated_answer, labels=merged_labels, sample_index=sample_index
            )

            for span in hal_spans:
                start, end, label = span
                labels.append({"start": start, "end": end, "label": label})
        else:
            # This sample has no hallucination spans, so it should carry no tags.
            # Strip any (possibly malformed) tags the model added regardless.
            cleaned_answer = re.sub(r"(?:</?HAL)+>+", "", translated_answer)

        TOKEN_USAGE["samples"] += 1

        return HallucinationSample(
            prompt=translated_prompt,
            answer=cleaned_answer,
            labels=labels,
            split=sample.split,
            task_type=sample.task_type,
            dataset=t.cast(t.Literal["ragtruth", "ragbench"], dataset),
            language=target_lang.code,  # ty:ignore[invalid-argument-type]
        )
    except Exception as e:
        logger.error(f"Error translating sample {sample_index}: {e!s}")
        with open(log_file, "a") as log:
            log.write(f"Error translating sample {sample_index}: {e!s}\n")
        return None


def merge_overlapping_spans(labels: list[dict[str, t.Any]]) -> list[dict[str, t.Any]]:
    """Merge overlapping hallucination spans into a single span.

    Args:
        labels: List of label spans to merge.

    Returns:
        List of merged spans.
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
) -> tuple[str, list[dict[str, t.Any]]]:
    """Add hallucination tags to the text.

    Args:
        sample: Sample containing labels.
        answer: Text to add tags to.

    Returns:
        Tuple of (tagged text, merged labels).
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
                f"Invalid span: {start}-{end} for text of length "
                f"{len(tagged_answer)}. Skipping."
            )
            continue

        tagged_answer = tagged_answer[:end] + "</HAL>" + tagged_answer[end:]
        tagged_answer = tagged_answer[:start] + "<HAL>" + tagged_answer[start:]

    return tagged_answer, labels


def normalize_hallucination_tags(text: str) -> str:
    """Repair malformed <HAL>/</HAL> tags emitted by the translation model.

    The model occasionally duplicates, nests or glues tags together (e.g.
    ``<HAL<HAL>>``, ``<HAL<HAL<HAL>>>``, ``<HAL><HAL>`` or ``</HAL></HAL>``).
    This flattens any such structure to a single, well-formed, non-nested level
    of tags: glued/nested tags are first un-glued into standalone tokens, then a
    left-to-right scan keeps only the outermost ``<HAL>`` and its matching
    ``</HAL>``, drops stray unbalanced tags, and closes any tag left open at the
    end so the spans are always balanced.

    Args:
        text:
            Translated text that may contain malformed <HAL>/</HAL> tags.

    Returns:
        Text with well-formed, non-nested, balanced <HAL>...</HAL> tags.
    """
    # Collapse glued/nested tag markers into a single canonical token. The model
    # writes nested openings as "<HAL" repeated k times followed by k ">", e.g.
    # "<HAL<HAL>>" or "<HAL<HAL<HAL>>>"; likewise for closings. A single regex
    # pass handles arbitrary depth because the whole run is matched at once.
    text = re.sub(r"(?:<HAL)+>+", "<HAL>", text)
    text = re.sub(r"(?:</HAL)+>+", "</HAL>", text)

    # Scan tokens, flattening nesting to a single level and dropping stray tags.
    result: list[str] = []
    depth = 0
    pos = 0
    for match in re.finditer(r"<HAL>|</HAL>", text):
        result.append(text[pos : match.start()])
        pos = match.end()
        if match.group() == "<HAL>":
            if depth == 0:
                result.append("<HAL>")
            depth += 1
        else:  # "</HAL>"
            if depth > 0:
                depth -= 1
                if depth == 0:
                    result.append("</HAL>")
            # A stray closing tag with no matching opening is dropped.
    result.append(text[pos:])
    normalized = "".join(result)

    # Close any tag left open at the end so spans stay balanced.
    if depth > 0:
        normalized = normalized.rstrip() + "</HAL>"

    return normalized


def find_hallucination_tags(
    text: str, labels: list[dict[str, t.Any]], sample_index: int
) -> tuple[list[tuple[int, int, str]], str]:
    """Find hallucination tags and extract span positions inside tags.

    Preserves <HAL> and </HAL> tags in output. Label spans point to the
    text INSIDE the tags (not including the tags themselves).

    Args:
        text:
            Text containing <HAL>...</HAL> tags.
        labels:
            Original labels for each span.
        sample_index:
            Index of the sample for logging.

    Returns:
        Tuple of (list of (start, end, label) tuples for text inside tags,
        text with HAL tags preserved).
    """
    if not labels:
        return [], text

    hal_spans = []

    # Repair any malformed tags the model produced (nested/duplicated/unbalanced)
    # so the spans below are well-formed, non-nested and balanced.
    text = normalize_hallucination_tags(text)

    open_positions = [m.end() for m in re.finditer(r"<HAL>", text)]
    close_positions = [m.start() for m in re.finditer(r"</HAL>", text)]

    for i, (open_pos, close_pos) in enumerate(zip(open_positions, close_positions)):
        label_text = "Unknown"
        if i < len(labels):
            label_text = labels[i].get("label", "Unknown")
        else:
            logger.warning(
                f"IndexError: No label for hallucinated text at sample "
                f"({sample_index}), span {i}"
            )

        # Span points to text INSIDE the tags (after <HAL>, before </HAL>)
        hal_spans.append((open_pos, close_pos, label_text))

    if len(hal_spans) < len(labels):
        logger.warning(
            f"Warning: Not all hallucination spans were found in sample "
            f"{sample_index}. Found {len(hal_spans)}, expected {len(labels)}"
        )
    if len(open_positions) != len(close_positions):
        logger.warning(
            f"Mismatched HAL tags in sample {sample_index}: "
            f"{len(open_positions)} opening, {len(close_positions)} closing"
        )

    # Return text with tags preserved
    return hal_spans, text


async def translate_text(
    text: str,
    http_client: httpx.AsyncClient,
    url: str,
    semaphore: asyncio.Semaphore,
    model: str,
    task_type: str,
    source_lang: Language = ENGLISH,
    target_lang: Language = DANISH,
    prompt: bool = False,
) -> str:
    """Translate text using OpenAI-compatible HTTP API with automatic retries.

    Args:
        text: Text to translate.
        http_client: Shared async HTTP client.
        url: API endpoint URL.
        semaphore: Concurrency limiter.
        model: Model to use for translation.
        task_type: Type of the translation task.
        source_lang: Source language code.
        target_lang: Target language code.
        prompt: Whether the text is a prompt.

    Returns:
        Translated text.

    Raises:
        RetryableTranslationError: If a transient error occurs.
        TranslationError: If translation fails after retries.
    """
    if not text.strip():
        return ""

    translation_prompt = (
        TRANSLATION_ANSWER
        if prompt
        else TRANSLATION_PROMPT_DATA2TXT
        if task_type == "Data2txt"
        else TRANSLATION_PROMPT
    )
    translation_prompt = translation_prompt.format(
        source_lang=source_lang.name, target_lang=target_lang.name, text=text
    )

    # Cap output tokens to prevent runaway repetition loops.
    # len(text)/3 overestimates token count (~3 chars/token); 1.5x margin for
    # expansion in translation; min 256 buffer for safety. The 8192 ceiling is a
    # backstop that still leaves headroom for long RAG contexts / summaries so
    # their trailing </HAL> tags are not truncated away.
    max_tokens = min(8192, int(len(text) / 3 * 1.5) + 256)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": translation_prompt}],
        "temperature": 0.4,
        "max_tokens": max_tokens,
    }

    last_error: Exception | None = None
    for attempt in range(MAX_TRANSLATION_RETRIES):
        try:
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
                    retry_after: float | None = None
                    if response.status_code == 429:
                        retry_after_header = response.headers.get("retry-after")
                        if retry_after_header:
                            try:
                                retry_after = float(retry_after_header)
                            except ValueError:
                                retry_after = None
                    raise RetryableTranslationError(message, retry_after=retry_after)

                # Do not retry non-transient client errors (e.g., bad request /
                # context too long).
                raise TranslationError(message)

            response_json = response.json()
            choice = response_json["choices"][0]

            # Tally token usage for cost estimation.
            usage = response_json.get("usage") or {}
            TOKEN_USAGE["input_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            TOKEN_USAGE["output_tokens"] += int(usage.get("completion_tokens", 0) or 0)
            TOKEN_USAGE["api_calls"] += 1

            # Log finish_reason to detect truncation or other issues.
            finish_reason = choice.get("finish_reason")
            if finish_reason == "length":
                logger.warning(
                    f"Translation truncated (finish_reason='length') for sample "
                    f"with input length {len(text)} chars (max_tokens={max_tokens})"
                )

            # Strip lines starting with the character '='
            content = "\n".join(
                [
                    line
                    for line in choice["message"]["content"].split("\n")
                    if not line.strip().startswith("=")
                ]
            )

            return content.strip()

        except RetryableTranslationError as e:
            last_error = e
            logger.warning(f"{e!s} (attempt {attempt + 1}/{MAX_TRANSLATION_RETRIES})")
            if attempt < MAX_TRANSLATION_RETRIES - 1:
                # Honour a server-provided Retry-After, else exponential backoff
                # with full jitter to avoid synchronised retry stampedes.
                if e.retry_after is not None:
                    delay = e.retry_after
                else:
                    ceiling = min(
                        RETRY_MAX_DELAY_SECONDS, RETRY_BASE_DELAY_SECONDS * (2**attempt)
                    )
                    delay = random.uniform(0, ceiling)
                await asyncio.sleep(delay)
        except TranslationError:
            raise
        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as e:
            last_error = e
            logger.warning(
                f"Transient network error: {e!s} "
                f"(attempt {attempt + 1}/{MAX_TRANSLATION_RETRIES})"
            )
            if attempt < MAX_TRANSLATION_RETRIES - 1:
                ceiling = min(
                    RETRY_MAX_DELAY_SECONDS, RETRY_BASE_DELAY_SECONDS * (2**attempt)
                )
                await asyncio.sleep(random.uniform(0, ceiling))
        except Exception as e:
            logger.error(f"Translation error: {e!s}")
            raise TranslationError(f"Failed to translate text: {e!s}") from e

    raise TranslationError(
        f"Failed to translate text after {MAX_TRANSLATION_RETRIES} attempts: "
        f"{last_error!s}"
    )


def save_progress(
    translated_data: HallucinationData,
    output_file: Path,
    target_lang: Language,
    last_processed_index: int | None = None,
) -> None:
    """Save progress to file with backup handling.

    Uses global configuration constants for dataset name and output directory.

    Args:
        translated_data:
            Data to save.
        output_file:
            Primary output file.
        target_lang:
            Target language for backup file.
        last_processed_index:
            Index of last processed source sample (for caching).
    """
    # Wrap samples in dict structure expected by from_json
    data_dict: dict[str, t.Any] = {"samples": translated_data.to_json()}
    if last_processed_index is not None:
        data_dict["_metadata"] = {"last_processed_index": last_processed_index}
    try:
        output_file.write_text(json.dumps(data_dict))
    except Exception as e:
        logger.error(f"Error saving progress: {e!s}")

        # Try to save to a backup file
        backup_file = (
            OUTPUT_DIR
            / f"{DATASET_NAME}_data_{target_lang.code}_backup_{int(time.time())}.json"
        )
        try:
            backup_file.write_text(json.dumps(data_dict))
            logger.info(f"Saved backup to {backup_file}")
        except Exception as e2:
            logger.error(f"Error saving backup: {e2!s}")


def push_translated_data_to_hub(
    translated_data: HallucinationData, repo_id: str, config_name: str, private: bool
) -> None:
    """Push translated hallucination data to Hugging Face Hub.

    Args:
        translated_data: Translated samples to push.
        repo_id: Target Hugging Face dataset repo id.
        config_name: Dataset config/subset name (typically language code).
        private: Whether to keep dataset private on Hub.
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

    Args:
        translated_data: Translated samples to filter and push.
        repo_id: Target Hugging Face dataset repo id.
        config_name: Dataset config/subset name (typically language code).
        private: Whether to keep dataset private on Hub.
        n: Maximum number of test samples to upload.
        validation_n: Maximum number of validation samples to upload.
    """
    samples = translated_data.samples[:]
    random.shuffle(samples)

    def _to_rows(items: list[HallucinationSample]) -> list[dict[str, t.Any]]:
        """Preprocess samples for the hallucination (question-answering) task.

        The task group expects ``id``, ``context``, ``question`` and ``answers``
        columns, derived here from the RAG ``prompt`` and reference ``answer``
        while keeping the original columns intact.

        Returns:
            List of dictionaries with the required columns for the QA task.
        """
        return [
            {
                "id": str(index),
                "prompt": sample.prompt,
                "answer": sample.answer,
                "labels": sample.labels,
                "split": sample.split,
                "task_type": sample.task_type,
                "dataset": sample.dataset,
                "language": sample.language,
                "context": sample.prompt,
                "question": "",
                "answers": {"text": [sample.answer], "answer_start": [0]},
            }
            for index, sample in enumerate(items)
        ]

    test_samples = [s for s in samples if s.split == "test"][:n]

    # RAGTruth only ships train/test splits, so carve the validation split from train.
    validation_samples = [s for s in samples if s.split == "train"][:validation_n]

    if not test_samples and not validation_samples:
        logger.warning("No test or validation samples available; skipping upload.")
        return

    # Push all splits in a single commit so the entire config is overwritten
    # atomically. Pushing splits one at a time fails when the existing splits on the
    # hub use a different (e.g. older) schema, since the new split's features are
    # validated against the stale ones.
    splits: dict[str, Dataset] = {}
    if test_samples:
        splits["test"] = Dataset.from_list(_to_rows(test_samples))
    else:
        logger.warning("No test samples available; skipping test split upload.")
    if validation_samples:
        splits["val"] = Dataset.from_list(_to_rows(validation_samples))
    else:
        logger.warning("No validation samples available; skipping val split upload.")

    DatasetDict(splits).push_to_hub(
        repo_id=repo_id, config_name=config_name, private=private
    )
    for split_name, split_dataset in splits.items():
        logger.info(
            "Pushed %d %s samples to hub: %s (config=%s)",
            len(split_dataset),
            split_name,
            repo_id,
            config_name,
        )


def setup_logging(output_dir: Path) -> None:
    """Set up logging to file in the output directory.

    Args:
        output_dir: Directory to save log file.
    """
    log_file = output_dir / "translation.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)


def _push_if_no_samples(
    translated_data: HallucinationData, target_lang: Language
) -> None:
    """Push data to hub when there are no samples to translate.

    Args:
        translated_data:
            Translated data to push.
        target_lang:
            Target language code.
    """
    logger.info("No samples to translate. Exiting.")
    if PUSH_TO_HUB:
        push_translated_data_to_hub(
            translated_data=translated_data,
            repo_id=HUB_REPO_ID,
            config_name=target_lang.code,
            private=PRIVATE_UPLOAD,
        )
    if PUSH_TEST_SUBSET:
        resolved_test_repo_id = (
            f"EuroEval/{DATASET_NAME}-translated-hallucinations-{target_lang.code}-mini"
        )
        push_test_subset_to_hub(
            translated_data=translated_data,
            repo_id=resolved_test_repo_id,
            config_name=target_lang.code,
            private=PRIVATE_UPLOAD,
            n=TEST_SUBSET_SIZE,
            validation_n=VALIDATION_SUBSET_SIZE,
        )


def _translate_to_language(
    source_data: HallucinationData | None,
    target_lang: Language,
    client_config: ClientConfig,
    model: str,
    max_workers: int,
    batch_size: int,
    test_mode: bool = False,
    test_num_samples: int = 10,
) -> None:
    """Translate RAGTruth data to a single target language.

    Uses global configuration constants for Hub settings.

    Args:
        source_data:
            Source hallucination data (None if resuming without source).
        target_lang:
            Target language to translate to.
        client_config:
            OpenAI client configuration.
        model:
            Model ID to use for translation.
        max_workers:
            Maximum number of concurrent in-flight API requests.
        batch_size:
            Number of samples processed between save points.
        test_mode:
            If True, run a limited smoke test and skip Hub uploads.
        test_num_samples:
            Number of samples to translate per language in test mode.

    Raises:
        FileNotFoundError:
            If source data is not available and no cached translation exists.
    """
    # Set up files for this language
    output_file = OUTPUT_DIR / f"{DATASET_NAME}_data_{target_lang.code}.json"

    # Load existing translated data if output file exists (cache)
    last_processed_index: int | None = None
    if output_file.exists():
        translated_data, last_processed_index = load_check_existing_data(
            output_file=output_file
        )

        # Use metadata index if available, fall back to sample count
        num_processed = (
            last_processed_index
            if last_processed_index is not None
            else len(translated_data.samples)
        )
        logger.info(f"Resuming from index {num_processed}")
    else:
        translated_data = HallucinationData(samples=[])
        num_processed = 0

    # Get remaining samples to translate
    remaining_samples: list[HallucinationSample] = []
    if source_data is not None:
        remaining_samples = source_data.samples[num_processed:]
    elif num_processed > 0:
        logger.warning(
            f"Proceeding in push-only mode with {num_processed} existing "
            "translated samples."
        )
    else:
        logger.error("No source data available and no cached translation exists")
        raise FileNotFoundError(
            "Source data not found and no cached translation to resume"
        )

    # In test mode, translate a small number of samples per language. Spread the
    # picks evenly across the whole source rather than taking the first N, so the
    # measured token usage covers the different task types (QA / Summary /
    # Data2txt) and yields a representative full-dataset estimate.
    if test_mode and len(remaining_samples) > test_num_samples > 0:
        step = len(remaining_samples) / test_num_samples
        remaining_samples = [
            remaining_samples[int(i * step)] for i in range(test_num_samples)
        ]

    total_samples = len(remaining_samples)

    if total_samples == 0:
        if test_mode:
            logger.info("No samples to translate (test mode); skipping Hub uploads.")
        else:
            _push_if_no_samples(translated_data, target_lang)
        return

    logger.info(f"Using model: {model}")
    logger.info(f"Total samples to process: {total_samples}")
    logger.info(f"Batch size: {batch_size}, Max workers: {max_workers}")

    try:
        asyncio.run(
            run_translation(
                remaining_samples=remaining_samples,
                translated_data=translated_data,
                output_file=output_file,
                target_lang=target_lang,
                num_processed=num_processed,
                client_config=client_config,
                model=model,
                max_workers=max_workers,
                batch_size=batch_size,
            )
        )

    except KeyboardInterrupt:
        logger.info("Translation interrupted by user. Saving progress...")
        save_progress(
            translated_data=translated_data,
            output_file=output_file,
            target_lang=target_lang,
            last_processed_index=last_processed_index,
        )
        logger.info(
            f"Saved {len(translated_data.samples)} translated samples to {output_file}"
        )
        return

    except Exception as e:
        logger.error(f"Unexpected error: {e!s}")
        save_progress(
            translated_data=translated_data,
            output_file=output_file,
            target_lang=target_lang,
            last_processed_index=last_processed_index,
        )
        raise

    logger.info(
        f"Translation complete. Translated {len(translated_data.samples)} samples."
    )
    logger.info(f"Output saved to {output_file}")

    if test_mode:
        logger.info("Test mode: skipping Hub uploads.")
        return

    if PUSH_TO_HUB:
        push_translated_data_to_hub(
            translated_data=translated_data,
            repo_id=HUB_REPO_ID,
            config_name=target_lang.code,
            private=PRIVATE_UPLOAD,
        )

    if PUSH_TEST_SUBSET:
        resolved_test_repo_id = (
            f"EuroEval/{DATASET_NAME}-translated-hallucinations-{target_lang.code}-mini"
        )
        push_test_subset_to_hub(
            translated_data=translated_data,
            repo_id=resolved_test_repo_id,
            config_name=target_lang.code,
            private=PRIVATE_UPLOAD,
            n=TEST_SUBSET_SIZE,
            validation_n=VALIDATION_SUBSET_SIZE,
        )


def _report_token_estimate(total_source_samples: int, model: str) -> None:
    """Log measured token usage and extrapolate it to the full dataset.

    The estimate multiplies the average per-sample token usage measured during
    this (test) run by the full dataset size (all source samples x all target
    languages). It is only as representative as the sampled subset, so scale up
    ``--test-num-samples`` for a tighter estimate.

    Args:
        total_source_samples:
            Number of source samples in the full (untranslated) dataset.
        model:
            Model ID used for translation, for the pricing lookup.
    """
    measured = TOKEN_USAGE["samples"]
    if measured == 0 or total_source_samples == 0:
        logger.warning("No samples measured; skipping token-usage estimate.")
        return

    in_tok = TOKEN_USAGE["input_tokens"]
    out_tok = TOKEN_USAGE["output_tokens"]
    avg_in = in_tok / measured
    avg_out = out_tok / measured

    full_samples = total_source_samples * len(TARGET_LANGS)
    est_in = avg_in * full_samples
    est_out = avg_out * full_samples

    logger.info("=" * 60)
    logger.info("TOKEN USAGE")
    logger.info(
        f"Measured over {measured} translated samples "
        f"({TOKEN_USAGE['api_calls']} API calls):"
    )
    logger.info(f"  input : {in_tok:,} tokens (avg {avg_in:,.1f}/sample)")
    logger.info(f"  output: {out_tok:,} tokens (avg {avg_out:,.1f}/sample)")
    logger.info(
        f"Full-dataset estimate ({total_source_samples:,} source samples x "
        f"{len(TARGET_LANGS)} languages = {full_samples:,} samples):"
    )
    logger.info(f"  est. input : {est_in / 1e6:,.1f}M tokens")
    logger.info(f"  est. output: {est_out / 1e6:,.1f}M tokens")

    pricing = MODEL_PRICING.get(model)
    if pricing is not None:
        input_price, output_price = pricing
        measured_cost = in_tok / 1e6 * input_price + out_tok / 1e6 * output_price
        est_cost = est_in / 1e6 * input_price + est_out / 1e6 * output_price
        logger.info(f"  measured cost so far: ${measured_cost:,.4f}")
        logger.info(f"  est. full-dataset cost ({model}): ${est_cost:,.2f}")
    else:
        logger.info(f"  (no pricing configured for model '{model}'; cost omitted)")
    logger.info("=" * 60)


def get_openai_client() -> ClientConfig:
    """Get HTTP client configuration from environment variables.

    Returns:
        Dict with `url` and `headers` for chat completion requests.

    Raises:
        ValueError: If API key is not set.
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


if __name__ == "__main__":
    main()
