# Missing Tests

This document lists test cases that should be added to improve test coverage for the
EuroEval codebase.

## 5. `test_string_utils.py` - Missing

**Module:** `euroeval.string_utils`

### Tests to add

- `test_scramble_and_unscramble_roundtrip`
  - Test that scrambling and unscrambling restores original string

- `test_scramble_deterministic_with_seed`
  - Test that scramble produces same result with same seed

- `test_extract_json_dict_from_string_with_prefix_suffix`
  - Test JSON extraction from string with surrounding text

- `test_extract_json_dict_from_string_no_json_returns_none`
  - Test None returned when no JSON found

- `test_extract_json_dict_from_string_invalid_json_returns_none`
  - Test None returned when JSON is invalid

- `test_extract_json_dict_from_string_non_dict_returns_none`
  - Test None returned when JSON is not a dictionary

- `test_extract_json_dict_from_string_non_string_keys_returns_none`
  - Test None returned when JSON has non-string keys

- `test_extract_multiple_choice_labels_with_labels`
  - Test label extraction with provided candidate labels

- `test_extract_multiple_choice_labels_without_labels`
  - Test label extraction falls back to alphabet

- `test_extract_multiple_choice_labels_case_insensitive`
  - Test case-insensitive label extraction

- `test_extract_multiple_choice_labels_no_match_raises_error`
  - Test InvalidBenchmark raised when no labels found

- `test_split_model_id_simple_id`
  - Test splitting a simple model ID

- `test_split_model_id_with_revision`
  - Test splitting model ID with @revision

- `test_split_model_id_with_param`
  - Test splitting model ID with #param

- `test_split_model_id_with_both_revision_and_param`
  - Test splitting model ID with both @revision and #param

- `test_split_model_id_invalid_raises_error`
  - Test InvalidModel raised for invalid model ID

## 6. `test_speed_benchmark.py` - Missing

**Module:** `euroeval.speed_benchmark`

### Tests to add

- `test_benchmark_speed_single_iteration_vllm_model`
  - Test speed benchmark with VLLM model type

- `test_benchmark_speed_single_iteration_litellm_model`
  - Test speed benchmark with LiteLLM model type

- `test_benchmark_speed_single_iteration_huggingface_encoder_model`
  - Test speed benchmark with HuggingFace encoder model type

- `test_benchmark_speed_single_iteration_invalid_model_raises_error`
  - Test ValueError raised for unsupported model types

- `test_benchmark_speed_single_iteration_cuda_oom_error`
  - Test InvalidBenchmark raised when CUDA OOM occurs

## 7. `test_llm_as_a_judge.py` - Missing

**Module:** `euroeval.metrics.llm_as_a_judge`

### Tests to add

- `test_llm_as_a_judge_call_mismatched_lengths_raises_error`
  - Test InvalidBenchmark when predictions and references have different lengths

- `test_llm_as_a_judge_call_empty_inputs_returns_none`
  - Test that empty predictions/references return None

- `test_llm_as_a_judge_call_parse_failure_logs_warning`
  - Test warning logged when judge output parsing fails

- `test_llm_as_a_judge_call_no_valid_outputs_returns_none`
  - Test None returned when no valid judge outputs

- `test_llm_as_a_judge_apply_user_prompt_with_condition`
  - Test prompt application with condition

- `test_llm_as_a_judge_apply_user_prompt_without_condition`
  - Test prompt application without condition

- `test_llm_as_a_judge_apply_user_prompt_condition_required_raises_error`
  - Test InvalidBenchmark when condition is required but not provided

- `test_llm_as_a_judge_batch_scoring_fn_from_scoring_fn`
  - Test batch scoring function derived from single scoring function

- `test_llm_as_a_judge_batch_scoring_fn_both_raises_error`
  - Test InvalidBenchmark when both scoring_fn and batch_scoring_fn provided

- `test_llm_as_a_judge_batch_scoring_fn_neither_raises_error`
  - Test InvalidBenchmark when neither scoring function provided

- `test_create_model_graded_fact_metric_default_judge`
  - Test metric creation with default judge model

- `test_create_model_graded_fact_metric_custom_judge`
  - Test metric creation with custom judge model

- `test_create_model_graded_fact_metric_custom_scoring_fn`
  - Test metric creation with custom scoring function

- `test_create_model_graded_fact_metric_custom_response_format`
  - Test metric creation with custom response format

## 8. `test_exceptions.py` - Additional Tests

**Module:** `euroeval.exceptions`

### Additional tests to add

- `test_needs_extra_installed_message_format`
  - Test that message contains extra name

- `test_needs_manual_dependency_message_format`
  - Test that message contains package name

- `test_needs_system_dependency_message_format`
  - Test that message contains dependency and instructions

- `test_needs_additional_argument_cli_mode_message`
  - Test message format for CLI mode

- `test_needs_additional_argument_script_mode_message`
  - Test message format for script mode

- `test_needs_environment_variable_message_format`
  - Test that message contains environment variable name

## 9. `test_task_group_utils.py` - Missing

**Module:** `euroeval.task_group_utils.*`

### Tests to add

For each task group utility module:

- `test_multiple_choice_classification_process_batch`
- `test_question_answering_process_batch`
- `test_sequence_classification_process_batch`
- `test_text_to_text_process_batch`
- `test_token_classification_process_batch`

## 10. `test_benchmark_modules.py` - Missing

**Module:** `euroeval.benchmark_modules.*`

### Tests to add for `litellm.py`

- `test_lite_llm_model_init`
- `test_lite_llm_model_generate_with_messages`
- `test_lite_llm_model_get_model_config`
- `test_lite_llm_model_get_pytorch_module_not_supported`
- `test_lite_llm_model_get_tokeniser_not_supported`

### Tests to add for `vllm.py`

- `test_vllm_model_init`
- `test_vllm_model_generate_with_text`
- `test_vllm_model_get_model_config`
- `test_vllm_model_check_vllm_installation`
- `test_vllm_model_get_pytorch_module_not_supported`
- `test_vllm_model_get_tokeniser_not_supported`

### Tests to add for `fresh.py`

- `test_fresh_model_init`
- `test_fresh_model_generate`
- `test_fresh_model_get_model_config`

## 11. `test_metrics.py` - Missing

**Module:** `euroeval.metrics.*`

### Tests to add for `llm_as_a_judge.py`

(See section 7 above - these should be in a dedicated test_llm_as_a_judge.py)

### Tests to add for `pipeline.py`

- `test_pipeline_metric_call`
- `test_pipeline_metric_process_batch_predictions`
- `test_pipeline_metric_process_batch_references`

### Tests to add for `tool_calling.py`

- `test_extract_tool_calls_from_output`
- `test_match_predicted_tools_to_reference_tools`
- `test_tool_calling_metric_call`

### Tests to add for `bias.py`

- `test_bias_metric_call`
- `test_bias_metric_process_batch`

## 12. `test_model_loading.py` - Missing

**Module:** `euroeval.model_loading`

### Tests to add

- `test_load_model_huggingface_encoder`
- `test_load_model_vllm`
- `test_load_model_litellm`
- `test_load_model_fails_without_required_dependency`

## 13. `test_data_models.py` - Missing

**Module:** `euroeval.data_models`

### Tests to add

- `test_dataset_config_with_labels`
- `test_dataset_config_with_prompt_label_mapping`
- `test_dataset_config_with_preprocessing_func`
- `test_model_config_with_cache_dir`
- `test_benchmark_config_with_all_options`

## 14. `test_logging_utils.py` - Missing

**Module:** `euroeval.logging_utils`

### Tests to add

- `test_log_once_prevents_duplicate_logs`
- `test_block_terminal_output`
- `test_get_pbar_disabled_when_false`
- `test_get_pbar_enabled_when_true`
- `test_get_pbar_with_custom_desc`

## 15. `test_callbacks.py` - Missing

**Module:** `euroeval.callbacks`

### Tests to add

- `test_never_leave_progress_callback_on_epoch_end`
- `test_never_leave_progress_callback_on_epoch_start`

## 16. `test_utils.py` - Missing

**Module:** `euroeval.utils`

### Tests to add

- `test_enforce_reproducibility_sets_seeds`
- `test_clear_memory_frees_cuda_cache`
- `test_clear_memory_frees_mps_cache`
