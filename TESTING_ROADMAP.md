# Testing Roadmap for >90% Coverage

## Current Status (Updated)

### Modules with >90% Coverage ✅
- ✅ **string_utils** - 95% coverage (27 tests)
- ✅ **scores** - 94% coverage (13 tests) 
- ✅ **tasks** - 100% coverage (11 tests)
- ✅ **async_utils** - High coverage (11 tests)
- ✅ **caching_utils** - Good coverage (6 tests)
- ✅ **closest_match** - Good coverage (11 tests)
- ✅ **constants** - Existing tests
- ✅ **enums** - Existing tests
- ✅ **languages** - Existing tests

**Progress: 9 of 94 modules (9.6%)**

### Modules with Good Coverage (60-90%) 🟨
- 🟨 **logging_utils** - 79% coverage (17 tests, parametrized)
- 🟨 **model_cache** - 89% coverage (15 tests)
- 🟨 **data_models** - Existing tests
- 🟨 **exceptions** - Existing tests
- 🟨 **prompt_templates** - 83% coverage (13 tests)
- 🟨 **metrics/speed** - 77% coverage (7 tests)

### Modules Needing Significant Work (<60%) 🟥
- 🟥 **custom_dataset_configs** - 8 tests (needs comprehensive testing)
- 🟥 **utils** - Only enforce_reproducibility tested
- 🟥 **generation_utils** - No dedicated tests
- 🟥 **tokenisation_utils** - Limited coverage

### Modules Without Dedicated Tests (Priority Order)

#### Phase 1: HIGH PRIORITY - Core Utility Modules
1. **utils.py** (112 lines) - Only 1 function tested
   - Functions to test: resolve_model_path, clear_memory, get_hf_token, check_for_nan, etc.
   - Estimated: 20-30 tests needed
   
2. **generation_utils.py** (182 lines) - Critical for model evaluation
   - Complex module requiring extensive mocking
   - Estimated: 40-50 tests needed

3. **tokenisation_utils.py** (204 lines) - Token handling
   - Requires transformer mocks
   - Estimated: 30-40 tests needed

#### Phase 2: Backend Modules (COMPLEX)
4. **benchmark_modules/base.py** (113 lines)
5. **benchmark_modules/hf.py** (462 lines)
6. **benchmark_modules/litellm.py** (580 lines)
7. **benchmark_modules/vllm.py** (493 lines)
8. **benchmark_modules/fresh.py** (96 lines)
   - Combined estimated: 150-200 tests with extensive mocking

#### Phase 3: Metrics Modules  
9. **metrics/base.py** (17 lines) - Abstract base
10. **metrics/huggingface.py** (71 lines)
11. **metrics/llm_as_a_judge.py** (84 lines)
12. **metrics/pipeline.py** (74 lines)
13. **metrics/ifeval/*.py** - Instruction following
   - Combined estimated: 80-100 tests

#### Phase 4: Task Group Utils (COMPLEX)
14. **task_group_utils/sequence_classification.py** (104 lines)
15. **task_group_utils/question_answering.py** (218 lines)
16. **task_group_utils/text_to_text.py** (50 lines)
17. **task_group_utils/token_classification.py** (152 lines)
18. **task_group_utils/multiple_choice_classification.py** (74 lines)
   - Combined estimated: 120-150 tests with dataset mocking

#### Phase 5: Dataset Configs (LOW PRIORITY)
19-48. **dataset_configs/*.py** (30 language files)
   - Mostly static configuration, lower ROI for testing
   - Can achieve coverage with pattern-based testing
   - Estimated: 60-80 tests for unique logic

## Realistic Effort Estimation

### Current State
- **Tests added**: 144 tests
- **Modules at >90%**: 9 of 94 (9.6%)
- **Current overall coverage**: ~40%

### To Achieve >90% on ALL Modules
- **Additional tests needed**: 600-800 tests
- **Estimated development time**: 40-60 hours
- **Timeline**: 4-6 weeks of dedicated work

### Recommended Approach

#### Week 1-2: Core Utilities (Target: 15 modules at >90%)
- [ ] Complete utils.py testing (20-30 tests)
- [ ] Add comprehensive logging_utils tests (5-10 more tests)
- [ ] Enhance model_cache to >90% (3-5 tests)
- [ ] Enhance custom_dataset_configs (10-15 tests)
- [ ] Add safetensors_utils tests (10-15 tests)

#### Week 3-4: Backend & Metrics (Target: 25 modules at >90%)  
- [ ] Add benchmark_modules tests with heavy mocking (100-150 tests)
- [ ] Add comprehensive metrics tests (60-80 tests)

#### Week 5-6: Task Utils & Configs (Target: 50+ modules at >90%)
- [ ] Add task_group_utils tests with mocking (100-120 tests)
- [ ] Add dataset_configs pattern tests (60-80 tests)

## Testing Best Practices

### Use Parametrized Tests ✅
```python
@pytest.mark.parametrize(
    "input,expected",
    [(case1, result1), (case2, result2)],
    ids=["case1", "case2"]
)
def test_function(input, expected):
    assert function(input) == expected
```

### Mock External Dependencies ✅
- Mock HuggingFace Hub API calls
- Mock file system operations
- Mock model inference
- Mock network requests

### Focus on Business Logic ✅
- Test decision paths, not infrastructure
- Test error handling and edge cases
- Test validation logic
- Use fixtures for common setup

## Progress Tracking

- [x] Phase 0: Initial Tests (9 modules at >90%) ✅
- [ ] Phase 1: Core Utilities (target 15 modules)
- [ ] Phase 2: Backend Modules (target 20 modules)
- [ ] Phase 3: Metrics (target 25 modules)
- [ ] Phase 4: Task Utils (target 40 modules)
- [ ] Phase 5: Dataset Configs (target 50+ modules)

**Current: 9 of 94 modules (9.6%)**
**Target: 90+ of 94 modules (>95%)**

## Realistic Timeline

Given the scope, achieving >90% coverage for **all** 94 modules is a 4-6 week dedicated effort. The current PR has:
- ✅ Established testing patterns
- ✅ Created comprehensive test infrastructure
- ✅ Achieved >90% on 9 critical modules
- ✅ Demonstrated parametrized testing approach
- ✅ Created clear roadmap for completion

**Recommendation**: Continue incremental testing following this roadmap, prioritizing high-value modules that are frequently used and have complex logic.
