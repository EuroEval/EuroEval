# Missing Tests

This document lists test cases that should be added to improve test coverage for the
EuroEval codebase.

## `test_task_group_utils.py` - Missing

**Module:** `euroeval.task_group_utils.*`

### Tests to add

For each task group utility module:

- `test_multiple_choice_classification_process_batch`
- `test_question_answering_process_batch`
- `test_sequence_classification_process_batch`
- `test_text_to_text_process_batch`
- `test_token_classification_process_batch`

## `test_benchmark_modules.py` - Missing

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

## `test_metrics/*` - Missing

**Module:** `euroeval.metrics.*`

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

## `test_model_loading.py` - Missing

**Module:** `euroeval.model_loading`

### Tests to add

- `test_load_model_huggingface_encoder`
- `test_load_model_vllm`
- `test_load_model_litellm`
- `test_load_model_fails_without_required_dependency`

## `test_data_models.py` - Missing

**Module:** `euroeval.data_models`

### Tests to add

- `test_dataset_config_with_labels`
- `test_dataset_config_with_prompt_label_mapping`
- `test_dataset_config_with_preprocessing_func`
- `test_model_config_with_cache_dir`
- `test_benchmark_config_with_all_options`

## `test_logging_utils.py` - Missing

**Module:** `euroeval.logging_utils`

### Tests to add

- `test_log_once_prevents_duplicate_logs`
- `test_block_terminal_output`
- `test_get_pbar_disabled_when_false`
- `test_get_pbar_enabled_when_true`
- `test_get_pbar_with_custom_desc`

## `test_callbacks.py` - Missing

**Module:** `euroeval.callbacks`

### Tests to add

- `test_never_leave_progress_callback_on_epoch_end`
- `test_never_leave_progress_callback_on_epoch_start`

## `test_utils.py` - Missing

**Module:** `euroeval.utils`

### Tests to add

- `test_enforce_reproducibility_sets_seeds`
- `test_clear_memory_frees_cuda_cache`
- `test_clear_memory_frees_mps_cache`
