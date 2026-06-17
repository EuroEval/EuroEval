"""Module-level constants shared across the leaderboards package.

Centralises every fixed value used by the leaderboard pipeline so the
constants live in one place rather than being scattered across the modules
that happen to use them. This module imports only from the standard library
and ``euroeval``; the rest of the package imports from here.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from euroeval.enums import TaskGroup
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
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _env_path(name: str, default: Path) -> Path:
    """Return the path from environment variable ``name``, or ``default``.

    Args:
        name:
            The environment variable to read.
        default:
            The path to use when the variable is unset or empty.

    Returns:
        The user-expanded path from the environment, or the default.
    """
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


# This package's directory (`src/leaderboards/`).
PACKAGE_DIR: Path = Path(__file__).resolve().parent

# Per-leaderboard language list. The datasets and task metadata are derived
# from `euroeval` at runtime (see `task_metadata.py`).
LEADERBOARD_CONFIGS_DIR: Path = PACKAGE_DIR / "leaderboard_configs"

# Config for the core-model list that drives which models we re-evaluate
# when datasets change (see `core_models.py`).
CORE_MODELS_CONFIG: Path = PACKAGE_DIR / "core_models.yaml"

# Persistent cache of model IDs the user has explicitly opted to keep in the
# results despite no URL being found, so we don't re-prompt every run.
MODELS_WITHOUT_URLS_CACHE: Path = PACKAGE_DIR / "models_without_urls_cache.yaml"

# Repository root (assumes the package lives at <repo>/src/leaderboards/).
REPO_ROOT: Path = PACKAGE_DIR.parent.parent

# Generated CSVs are written directly into the frontend bundle.
OUTPUT_DIR: Path = REPO_ROOT / "src" / "frontend" / "csv"

# Off-repo backup location for snapshots of results.tar.gz.
# The working copy lives at BACKUPS_DIR/results.tar.gz.
# Snapshots are timestamped and pruned when exceeding limits.
BACKUPS_DIR: Path = _env_path(
    "EUROEVAL_RESULTS_BACKUP_DIR",
    Path.home() / "pCloud Drive" / "data" / "euroeval_backup",
)
BACKUPS_MAX_BYTES: int = 1_000_000_000  # ~1 GB total size cap

# Historical archive of all benchmark records. Stored in BACKUPS_DIR,
# not tracked in git (43+ MB compressed).
RESULTS_PATH: Path = BACKUPS_DIR / "results.tar.gz"

# Incremental jsonl of new benchmark records to fold into RESULTS_PATH.
NEW_RESULTS_PATH: Path = _env_path(
    "EUROEVAL_NEW_RESULTS_PATH", REPO_ROOT / "new_results.jsonl"
)

# Single directory for all per-model JSONL files with metadata attached.
# Git-ignored, synced with hf://buckets/EuroEval/results/.
RESULTS_DIR: Path = REPO_ROOT / "results"

# ---------------------------------------------------------------------------
# GitHub API
# ---------------------------------------------------------------------------

REPO = "EuroEval/EuroEval"
MODEL_REQUEST_LABEL = "model evaluation request"
FAILED_LABEL = "evaluation-failed"
GATED_LABEL = "gated"
RESULTS_READY_LABEL = "results-ready"
ISSUE_TITLE_PREFIX = "[MODEL EVALUATION REQUEST]"
USER_AGENT = "euroeval-leaderboards"

# ---------------------------------------------------------------------------
# Buckets
# ---------------------------------------------------------------------------

HF_RESULTS_BUCKET = "EuroEval/results"

# ---------------------------------------------------------------------------
# Tasks & categories
# ---------------------------------------------------------------------------

# Tasks displayed on every EuroEval leaderboard, in column order.
LEADERBOARD_TASKS: list[str] = [
    "sentiment-classification",
    "named-entity-recognition",
    "linguistic-acceptability",
    "reading-comprehension",
    "summarization",
    "knowledge",
    "common-sense-reasoning",
    "simplification",
    "european-values",
]

# The two leaderboard categories that every model is ranked within. The
# "generative" variant scores all tasks; "all_models" only scores NLU tasks so
# non-generative models can compete.
LEADERBOARD_CATEGORIES: tuple[str, str] = ("generative", "all_models")

# TaskGroup -> "nlu"/"nlg". The "all_models" leaderboard variant only
# scores NLU tasks so non-generative models can compete.
NLU_TASK_GROUPS: frozenset[TaskGroup] = frozenset(
    {
        TaskGroup.SEQUENCE_CLASSIFICATION,
        TaskGroup.TOKEN_CLASSIFICATION,
        TaskGroup.QUESTION_ANSWERING,
    }
)

# ---------------------------------------------------------------------------
# Model classification
# ---------------------------------------------------------------------------

# Matches the labels used by `GenerativeType` in the euroeval lib (snake_case
# auto-enum). The lib enum has BASE / INSTRUCTION_TUNED / REASONING. Mapped to
# the leaderboards' own `ModelType` (see `core_models.py`).
GENERATIVE_TYPE_TO_MODEL_TYPE: dict[str, str] = {
    "base": "base_decoder",
    "instruction_tuned": "instruction_tuned_decoder",
    "reasoning": "reasoning_decoder",
}

# Single source of truth for API model identification -- also imported by
# `scripts/generate_leaderboards.py` so the leaderboard pipeline and the
# core-model updater agree on which ids are API models.
API_MODEL_PATTERNS: list[re.Pattern] = [
    re.compile(r"gemini/.*"),
    re.compile(r"(openai/)?gpt-[456789].*"),
    re.compile(r"(anthropic/)?claude.*"),
    re.compile(r"(xai/)?grok.*"),
]

# Models matching these patterns are excluded from the core-model list
# entirely. We currently drop `ollama_chat/*` -- those are hard to
# re-evaluate in CI (Ollama-based local serving), so including them in
# the "must re-run when datasets change" list just creates churn.
EXCLUDED_MODEL_PATTERNS: list[re.Pattern] = [
    re.compile(r"^ollama_chat/.*"),
    re.compile(r"^bigscience/bloom.*"),
]

# Match the first ``<N>B`` / ``<N>M`` token in a model id (e.g. ``22B`` in
# ``EuroLLM-22B-Instruct-2512``, ``270m`` in ``gemma-3-270m``). The lookbehind
# stops us from picking up tokens like ``A3B`` in ``80B-A3B``; the lookahead
# excludes things like ``multilingual``.
PARAMS_FROM_ID_RE = re.compile(r"(?<![A-Za-z\d])(\d+(?:\.\d+)?)([BbMm])(?![A-Za-z])")

# Keyword patterns mapping a model id to a generative type, checked in order.
# A None type marks a non-generative model (e.g. an encoder).
GENERATIVE_TYPE_KEYWORDS: list[tuple[list[str], str | None]] = [
    (["bert", "xlm-r", "encoder"], None),
    (["-base", "-pt"], "base"),
    (["-instruct", "-it$", "-chat"], "instruction_tuned"),
    (["^o[1-9]$", "^o[1-9]-", "deepseek-r1"], "reasoning"),
]

# Order in which size buckets are sorted in the GitHub issue (see
# `core_models.py::SizeBucket`).
PARAM_SIZE_BUCKET_ORDER: dict[str, int] = {
    "encoder": 0,
    "tiny": 1,
    "small": 2,
    "medium": 3,
    "large": 4,
    "xlarge": 5,
    "api": 6,
}

# ---------------------------------------------------------------------------
# OSAI scraping
# ---------------------------------------------------------------------------

OSAI_BASE = "https://osai-index.eu"
OSAI_DB_URL = (
    f"{OSAI_BASE}/database/"
    "?type=text&weights_basemodel=1&trainingcode=1&datasources_basemodel=1"
)

# Required openness fields (all must be "open"). For weights, see
# OSAI_WEIGHT_CRITERIA below -- at least one of base/end must be open.
OSAI_REQUIRED_OPEN = ("trainingcode", "datasources_basemodel")
OSAI_WEIGHT_CRITERIA = ("weights_basemodel", "weights_endmodel")

# All openness fields we count to rank models. Drawn from the rendered
# OSAI methodology page. Order doesn't matter; only the count of "open"s.
OSAI_RANKING_FIELDS = (
    "weights_basemodel",
    "weights_endmodel",
    "trainingcode",
    "code",
    "datasources_basemodel",
    "datasources_endmodel",
    "datasheet",
    "modelcard",
    "package",
    "license_basemodel",
    "license_endmodel",
    "hardware_architecture",
    "preprint",
    "paper",
)
OSAI_BUNDLE_MARKER = "system:{name:"

# Matches the start of each model entry in the bundle. Each model is a JS
# object literal opening with `system:{name:"...",link:"...",type:"...",...,
# endmodelname:"...",...}`. The openness-criteria fields follow that block.
OSAI_SYSTEM_RE = re.compile(
    r"system:\{"
    r"name:\"(?P<name>[^\"]+)\","
    r"link:\"(?P<link>[^\"]*)\","
    r"type:\"(?P<type>[^\"]+)\","
    r"performanceclass:\"[^\"]+\","
    r"basemodelname:\"[^\"]+\","
    r"endmodelname:\"(?P<endmodelname>[^\"]+)\""
)

# Matches one openness criterion declaration: <field>:{class:"<cls>"...
OSAI_FIELD_RE = re.compile(r"(\w+):\{class:\"(open|closed|partial)\"")

# Within an openness block, the link can be a single string or a JSON array
# of strings. We extract the first HF URL we find (string-form).
OSAI_BLOCK_LINK_RE = re.compile(
    r"link:(?:\"(?P<single>[^\"]*)\"|\[\"(?P<first>[^\"]*)\")"
)

# Matches the whole weights_endmodel/basemodel block so we can scan inside
# it for an HF link.
OSAI_WEIGHTS_BLOCK_RE = re.compile(
    r"(weights_endmodel|weights_basemodel):\{(?P<body>[^}]*)\}"
)

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

# Number of bootstrap replicates for both confidence interval estimation of rank
# scores and tie-breaking in leaderboard generation. 50 is sufficient because
# tie-breaking only needs to distinguish models that are statistically tied --
# the bootstrap test is a one-sided test at alpha=0.05, and 50 replicates give a
# reasonable resolution for the rank-difference distribution without unnecessary
# computation.
NUM_BOOTSTRAPS = 50

# z-score for a two-sided 95% confidence interval under a normal approximation.
Z_SCORE_95 = 1.96

# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------

HF_CACHE_PATH = Path(
    os.environ.get(
        "EUROEVAL_QUEUE_HF_CACHE",
        str(Path.home() / ".cache" / "euroeval" / "queue_hf_cache.json"),
    )
)
HF_CACHE_TTL_SECONDS = 6 * 60 * 60

VM_MARKER_RE = re.compile(r"<!--\s*vm-id:\s*([^\s>]+)\s*-->")

# Matches the "Model ID" section in an issue body (the form template renders
# the model id as the line immediately following a "### Model ID" heading).
MODEL_ID_BODY_RE = re.compile(r"(?:^|\n)#{1,6}\s*Model ID\s*\n+([^\n]+)")

# euroeval emits a summary line like "errored 3 benchmarks" when individual
# (dataset, language) combinations fail without crashing the whole run.
ERRORED_BENCHMARKS_RE = re.compile(r"errored\s+(\d+)\s+benchmarks?", re.IGNORECASE)

# euroeval emits a summary line like "skipped 2 benchmarks" when individual
# (dataset, language) combinations are deliberately skipped. These are not
# failures.
SKIPPED_BENCHMARKS_RE = re.compile(r"skipped\s+(\d+)\s+benchmarks?", re.IGNORECASE)

# Phrase euroeval prints when it cannot load a model because the repo is gated
# and the subprocess lacks the necessary HF token / accepted access terms.
GATED_OUTPUT_RE = re.compile(r"is a gated repository", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

# Regex for stripping HTML anchor tags from model IDs like
# `<a href='...'>org/model</a>`
ANCHOR_RE = re.compile(r"<a [^>]*>(?P<inner>[^<]+)</a>")

# Strips trailing ``(zero-shot)``, ``(val)``, ``(zero-shot, val)`` etc.
# annotations that `extract_model_ids_from_record` appends to variants.
VARIANT_SUFFIX_RE = re.compile(r"\s*\((?:zero-shot|val)(?:,\s*(?:zero-shot|val))*\)$")

REQUIRED_METADATA_FIELDS = ["commercially_licensed", "open", "trained_from_scratch"]

# ---------------------------------------------------------------------------
# Language groups
# ---------------------------------------------------------------------------

_BALTIC = "Baltic languages (Latvian, Lithuanian)"
_FINNIC = "Finnic languages (Estonian, Finnish)"
_ROMANCE = "Romance languages (Catalan, French, Italian, Portuguese, Romanian, Spanish)"
_SCANDI = "Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)"
_SLAVIC = (
    "Slavic languages (Belarusian, Bulgarian, Bosnian, Croatian, Czech, Polish,"
    " Serbian, Slovak, Slovenian, Ukrainian)"
)
_WGERMANIC = "West Germanic languages (Dutch, English, German)"

LANGUAGE_GROUP_CODES: dict[str, list[str]] = {
    _BALTIC: [LATVIAN.code, LITHUANIAN.code],
    _FINNIC: [ESTONIAN.code, FINNISH.code],
    _ROMANCE: [
        CATALAN.code,
        FRENCH.code,
        ITALIAN.code,
        PORTUGUESE.code,
        ROMANIAN.code,
        SPANISH.code,
    ],
    _SCANDI: [DANISH.code, FAROESE.code, ICELANDIC.code, NORWEGIAN.code, SWEDISH.code],
    _SLAVIC: [
        BELARUSIAN.code,
        BULGARIAN.code,
        BOSNIAN.code,
        CROATIAN.code,
        CZECH.code,
        POLISH.code,
        SERBIAN.code,
        SLOVAK.code,
        SLOVENE.code,
        UKRAINIAN.code,
    ],
    _WGERMANIC: [DUTCH.code, ENGLISH.code, GERMAN.code],
    "Albanian": [ALBANIAN.code],
    "Greek": [GREEK.code],
    "Hungarian": [HUNGARIAN.code],
}

DTYPE_BYTES: dict[str, int] = {
    "F64": 8,
    "I64": 8,
    "U64": 8,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "U16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}

# Multiplier applied to weight bytes to leave room for activations, KV cache,
# and framework overhead when judging whether a model fits in GPU memory.
GPU_FIT_OVERHEAD = 1.2

# ---------------------------------------------------------------------------
# Leaderboard generation
# ---------------------------------------------------------------------------

MINIMUM_VERSION: str = "15.0.0"
MINIMUM_NUMBER_OF_MODEL_RECORDS: int = 7
CORE_MODELS_STALE_DAYS: int = 30
BANNED_VERSIONS: list[str] = ["9.3.0", "10.0.0"]
BANNED_MODEL_PATTERNS: list[re.Pattern[str]] = [
    re.compile("^meta-llama/Llama-3.1-405B-Instruct$"),  # Temporary ban
    re.compile("^utter-project/EuroVLM-9B-Preview$"),  # Temporary ban
]
TRAINED_FROM_SCRATCH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"Qwen/.*"),
    re.compile(r"google/.*"),
    re.compile(r"mistralai/.*"),
    re.compile(r"meta-llama/.*"),
    re.compile(r"facebook/.*"),
    re.compile(r"FacebookAI/.*"),
    re.compile(r"zai-org/.*"),
    re.compile(r"deepseek-ai/.*"),
    re.compile(r"PleIAs/.*"),
    re.compile(r"openai/.*"),
    re.compile(r"nvidia/.*"),
    re.compile(r"allenai/.*"),
    re.compile(r"utter-project/.*"),
    re.compile(r"CohereLabs/.*"),
    re.compile(r"speakleash/.*"),
    re.compile(r"yulan-team/.*"),
    re.compile(r"BSC-LT/.*"),
    re.compile(r"tencent/.*"),
    re.compile(r"LiquidAI/.*"),
    re.compile(r"HuggingFaceTB/.*"),
    re.compile(r"tiiuae/.*"),
    re.compile(r"AIDC-AI/.*"),
    re.compile(r"inclusionAI/.*"),
    re.compile(r"jhu-clsp/.*"),
    re.compile(r"vesteinn/(Dansk|Fo|Scandi)BERT.*"),
    re.compile(r"EuropeanParliament/EUBERT"),
    re.compile(r"microsoft/.*"),
    re.compile(r"EuroBERT/.*"),
    re.compile(r"fresh-.*"),
    re.compile(r"answerdotai/.*"),
    re.compile(r".*-scratch"),
]
