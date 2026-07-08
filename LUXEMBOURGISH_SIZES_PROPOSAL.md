# Luxembourgish Dataset Size Proposal

**Standard EuroEval cap:** 1,024 train / 256 val / 2,048 test

## Dataset Overview

| Dataset | Task | Original Size | Proposed Size | Notes |
|---------|------|---------------|---------------|-------|
| `ltzglue-hc` | Headline Classification | 20,716 / 2,960 / 5,919 | 1,024 / 256 / 2,048 | ✓ Full cap applied |
| `ltzglue-la` | Linguistic Acceptability (Binary) | 14,678 / 2,094 / 4,045 | 1,024 / 256 / 2,048 | ✓ Full cap applied |
| `ltzglue-la-multi` | Linguistic Acceptability (Multi-class) | 14,678 / 2,094 / 4,045 | 1,024 / 256 / 2,048 | ✓ Full cap applied |
| `ltzglue-ner` | Named Entity Recognition | 27,245 / 3,327 / 3,821 | 1,024 / 256 / 2,048 | ✓ Full cap applied |
| `ltzglue-tc` | Topic Classification | 9,932 / 1,240 / 1,245 | 1,024 / 256 / 1,245 | Test limited to 1,245 |
| `ltzglue-sa` | Sentiment Analysis | 3,022 / 597 / 926 | 1,024 / 256 / 926 | Test limited to 926 |
| `ltzglue-rte` | Recognizing Textual Entailment | 1,877 / 197 / 626 | 1,024 / 197 / 626 | Val limited to 197, test to 626 |
| `ltzglue-id` | Intent Detection | 0 / 53 / 66 | 0 / 53 / 66 | No training data available |
| `multi-wiki-qa-lb` | Reading Comprehension | 5,003 (single split) | TBD | Extract Luxembourgish subset |
| `scala-lb` | Linguistic Acceptability | ~100 (UD treebank) | TBD | Too small — needs expansion |
| `luxgen-summ` | Summarisation | — | TBD | Needs extraction from benchmark |

## Recommendations

### 1. Datasets Ready for Full Cap (5 datasets)
These have sufficient data for the standard 1,024/256/2,048 split:
- `ltzglue-hc`, `ltzglue-la`, `ltzglue-la-multi`, `ltzglue-ner`, `ltzglue-tc`

**Action:** Update creation scripts to cap at standard sizes with stratified sampling.

### 2. Datasets with Limited Test/Val Splits (2 datasets)
- **`ltzglue-sa`**: Use all 597 val + 926 test (cannot reach 2,048 test target)
- **`ltzglue-rte`**: Use all 197 val + 626 test (both below targets)

**Action:** Use all available data for val/test; cap train at 1,024.

### 3. Datasets Requiring Special Handling (1 dataset)
- **`ltzglue-id`**: No training data exists. Only 53 val + 66 test samples across 10 intent classes (some with only 1-2 examples).

**Action:** Mark as `unofficial` and exclude from official leaderboard. Consider combining val+test for single evaluation split.

### 4. Datasets Not Yet Created (3 datasets)
- **`multi-wiki-qa-lb`**: Extract Luxembourgish subset (~5,003 samples) from `alexandrainst/multi-wiki-qa`, then cap to 1,024/256/2,048
- **`scala-lb`**: Current UD treebank has only ~100 sentences — insufficient. Needs expansion before creation
- **`luxgen-summ`**: Requires extraction from LuxGen benchmark paper

## Implementation Plan

1. **Update existing scripts** for the 7 uploaded ltzGLUE datasets:
   - Apply 1,024/256/2,048 cap where possible
   - Use stratified sampling to maintain label balance
   - Document actual sizes in markdown docs

2. **Re-upload datasets** with capped sizes (keep private until review)

3. **Update documentation** to reflect final split sizes

4. **Create remaining datasets** once sources are available

## Label Distribution Considerations

For stratified sampling to work properly, ensure:
- Minimum 2 samples per class in each split (for stratification)
- `ltzglue-id` has classes with only 1 sample — stratification not possible
- `ltzglue-la-multi` error type distribution should be checked
