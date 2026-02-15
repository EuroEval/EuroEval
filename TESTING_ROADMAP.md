# Testing Roadmap for >90% Coverage

## Current Status

### Modules with >90% Coverage (Estimated)
- ✅ async_utils (11 tests)
- ✅ caching_utils (6 tests)  
- ✅ closest_match (11 tests)
- ✅ tasks (11 tests)
- ✅ constants (existing tests)
- ✅ enums (existing tests)
- ✅ languages (existing tests)

### Modules with Good Coverage (60-90%)
- 🟨 logging_utils (17 tests, needs edge cases)
- 🟨 model_cache (15 tests, needs error paths)
- 🟨 data_models (existing tests, needs validation)
- 🟨 exceptions (existing tests, needs all error types)

### Modules Needing Significant Work (<60%)
- 🟥 custom_dataset_configs (8 tests, needs 90%+)
- 🟥 prompt_templates (13 tests, mostly config validation)
- 🟥 metrics/speed (7 tests, basic only)

### Modules Without Tests (Priority Order)

#### High Priority - Utility Modules
1. **generation_utils.py** (182 lines) - Critical for model evaluation
2. **string_utils.py** (59 lines) - Text processing utilities
3. **tokenisation_utils.py** (204 lines) - Token handling
4. **utils.py** (112 lines) - General utilities
5. **scores.py** (36 lines) - Score calculation

#### Medium Priority - Backend Modules  
6. **benchmark_modules/base.py** (113 lines) - Base class
7. **benchmark_modules/hf.py** (462 lines) - HuggingFace integration
8. **benchmark_modules/litellm.py** (580 lines) - LiteLLM integration
9. **benchmark_modules/vllm.py** (493 lines) - vLLM integration
10. **benchmark_modules/fresh.py** (96 lines) - Fresh backend

#### Medium Priority - Metrics
11. **metrics/base.py** (17 lines) - Base metric class
12. **metrics/huggingface.py** (71 lines) - HF metrics
13. **metrics/llm_as_a_judge.py** (84 lines) - LLM evaluation
14. **metrics/pipeline.py** (74 lines) - Pipeline metrics
15. **metrics/ifeval/metric.py** (37 lines) - Instruction following

#### Lower Priority - Dataset Configs
16-45. **dataset_configs/*.py** (30 language files) - Mostly configuration

#### Lower Priority - Task Group Utils
46. **task_group_utils/sequence_classification.py** (104 lines)
47. **task_group_utils/question_answering.py** (218 lines)
48. **task_group_utils/text_to_text.py** (50 lines)
49. **task_group_utils/token_classification.py** (152 lines)
50. **task_group_utils/multiple_choice_classification.py** (74 lines)

## Estimated Effort

### Tests Needed for >90% Coverage
- **Utility modules** (5 modules): ~100-150 tests
- **Backend modules** (5 modules): ~150-200 tests
- **Metrics modules** (5 modules): ~80-100 tests
- **Dataset configs** (30 modules): ~150-200 tests
- **Task group utils** (5 modules): ~120-150 tests
- **Other modules** (15 modules): ~150-200 tests

**Total: 750-1000+ tests**
**Estimated time: 40-60 hours**

## Testing Strategy

### Phase 1: Complete Existing Test Coverage (Week 1)
- Enhance all 9 newly added test files to >90% coverage
- Fix any gaps in existing test files
- Target: ~50-70 additional tests

### Phase 2: High-Priority Utilities (Week 2)
- Add comprehensive tests for generation_utils
- Add comprehensive tests for string_utils
- Add comprehensive tests for tokenisation_utils
- Add comprehensive tests for utils
- Target: ~100-120 new tests

### Phase 3: Backend Modules (Week 3)
- Test benchmark_modules/base with mocks
- Test benchmark_modules/hf with mocks
- Test benchmark_modules/litellm with mocks
- Target: ~80-100 new tests

### Phase 4: Metrics & Remaining (Week 4)
- Test all metrics modules
- Test task_group_utils modules
- Test dataset configs (focus on unique logic)
- Target: ~150-200 new tests

## Testing Best Practices

### For Utility Modules
- Test all public functions
- Test edge cases (empty inputs, None values, etc.)
- Test error conditions
- Use parametrized tests for similar cases

### For Backend Modules  
- Mock external dependencies (HuggingFace, APIs)
- Test initialization and configuration
- Test error handling
- Focus on business logic, not integration

### For Config Modules
- Validate all configuration values
- Test serialization/deserialization
- Test validation logic
- Use fixtures for common configs

## Progress Tracking

- [ ] Phase 1: Complete (0/9 modules at >90%)
- [ ] Phase 2: High-Priority Utilities (0/5 modules)
- [ ] Phase 3: Backend Modules (0/5 modules)
- [ ] Phase 4: Metrics & Remaining (0/35 modules)

**Current Coverage: ~41%**
**Target Coverage: >90%**

## Notes

This is a substantial undertaking requiring dedicated effort over multiple weeks. The roadmap prioritizes the most critical and frequently-used modules first. Dataset config files contain mostly configuration data and may not require 90% line coverage if they follow a consistent pattern.
