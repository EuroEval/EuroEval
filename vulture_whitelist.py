# Vulture whitelist — KNOWN FALSE POSITIVES ONLY.
#
# Vulture reports each name below as unused, but every one is genuinely needed.
# This file is listed in `[tool.vulture] paths` (pyproject.toml); referencing a
# name here marks it "used" so vulture stops flagging it. It is excluded from
# ruff and ty (the bare names are deliberately "undefined" here).
#
# ┌───────────────────────────────────────────────────────────────────────────┐
# │ BEFORE ADDING A NAME HERE, STOP.                                          │
# │                                                                           │
# │ This is NOT a dumping ground for silencing vulture. Only add a name if    │
# │ you have confirmed it is a genuine false positive AND recorded WHY, in    │
# │ the right section below. If the code is actually unused, DELETE IT —      │
# │ do not whitelist it. Each entry must be justified; unexplained entries    │
# │ should be treated as bugs.                                                │
# └───────────────────────────────────────────────────────────────────────────┘


# ---------------------------------------------------------------------------
# 1. Type-only imports.
#
# These are imported purely for use in type annotations. Because the codebase
# uses `from __future__ import annotations`, annotations are lazy strings that
# vulture cannot see, so it thinks the imports are unused. They ARE used (by
# ty and at documentation-time) and removing them breaks type checking.
# ---------------------------------------------------------------------------
PreTrainedTokenizer     # base.py, hf.py, generation_utils.py, *task_group_utils*
ComputeMetricsFunction  # benchmark_modules/base.py
ModelResponse           # benchmark_modules/litellm.py
RequestOutput           # bpc_scoring.py
EvaluationModule        # metrics/huggingface.py
LinguaLanguageDetector  # metrics/language_detection.py
Pipeline                # metrics/pipeline.py
TrainerCallback         # task_group_utils/question_answering.py
Predictions             # task_group_utils/*, utils.py
Labels                  # task_group_utils/*
NDArray                 # numpy ndarray annotations (euroeval/types.py)


# ---------------------------------------------------------------------------
# 2. Parameters required by an interface / protocol.
#
# These are positional parameters that a Python protocol dictates by position,
# so they must stay in the signature even though this codebase never reads them.
# Deleting or renaming them would break the protocol contract.
# ---------------------------------------------------------------------------
last_values  # enums.py: Enum._generate_next_value_(name, start, count, last_values)
exc_type     # logging_utils.py: context-manager __exit__(self, exc_type, exc_val, exc_tb)
exc_val      # logging_utils.py: __exit__ signature
exc_tb       # logging_utils.py: __exit__ signature
__context    # create_icelandic_standardized_tests.py: pydantic model_post_init(self, __context)
logs         # finetuning.py: no_logging(logs, start_time) — matches the logging callback signature


# ---------------------------------------------------------------------------
# 3. Test mock / stub / lambda parameters.
#
# These parameters exist so a mock, stub, or lambda mirrors the real API it
# stands in for (or is injected positionally by @patch). The test needs the
# parameter present even when the body ignores it.
# ---------------------------------------------------------------------------
ignore_times          # test_bucket_sync.py, test_collect_evaluation_results.py, test_process_evaluation_queue.py — stub signatures mirroring the real sync API
add_special_tokens    # test_vllm.py — lambda mirroring tokeniser.encode(text, add_special_tokens=...)
mock_llm_model_class  # test_llm_as_a_judge.py — injected positionally by @patch, unused in body
bucket_id             # test_collect_evaluation_results.py — stub method mirroring the real bucket API
