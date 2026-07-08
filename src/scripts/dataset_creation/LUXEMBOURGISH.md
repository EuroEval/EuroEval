# Luxembourgish Dataset Creation Scripts

Scripts to create Luxembourgish datasets for EuroEval from the ltzGLUE benchmark
and other sources.

## Datasets

### MultiWikiQA-lb (Reading Comprehension)
- **Source:** alexandrainst/multi-wiki-qa (subset "lb")
- **Status:** ✅ Already on HuggingFace
- **Script:** `create_multi_wiki_qa_lb.py` (verification only)

### ltzGLUE Benchmark Datasets
From the ltzGLUE paper (arxiv.org/abs/2604.17976):

1. **ltzGLUE-HC** - Headline Classification
2. **ltzGLUE-ID** - Intent Detection
3. **ltzGLUE-LA** - Linguistic Acceptability (binary & multi-class)
4. **ltzGLUE-NER** - Named Entity Recognition
5. **ltzGLUE-RTE** - Recognising Textual Entailment
6. **ltzGLUE-SA** - Sentiment Analysis
7. **ltzGLUE-TC** - Topic Classification

## Prerequisites

```bash
# Clone the ltzGLUE repository with LFS data
cd /path/to/euroeval
git clone https://github.com/plumaj/ltzGLUE.git
cd ltzGLUE
git lfs pull
cd ..
```

## Running the Scripts

All scripts use uv script syntax - run them with:

```bash
# Verify MultiWikiQA-lb
uv run src/scripts/dataset_creation/create_multi_wiki_qa_lb.py

# Create ltzGLUE datasets (requires ltzGLUE repo to be cloned)
uv run src/scripts/dataset_creation/create_ltzglue_hc.py
uv run src/scripts/dataset_creation/create_ltzglue_sa.py
uv run src/scripts/dataset_creation/create_ltzglue_la.py
uv run src/scripts/dataset_creation/create_ltzglue_ner.py
uv run src/scripts/dataset_creation/create_ltzglue_rte.py
uv run src/scripts/dataset_creation/create_ltzglue_tc.py
uv run src/scripts/dataset_creation/create_ltzglue_id.py
```

## Output

Each script uploads the processed dataset to the EuroEval organization on
HuggingFace:

- `EuroEval/ltzglue-hc`
- `EuroEval/ltzglue-sa`
- `EuroEval/ltzglue-lab` (binary LA)
- `EuroEval/ltzglue-lam` (multi-class LA)
- `EuroEval/ltzglue-ner`
- `EuroEval/ltzglue-rte`
- `EuroEval/ltzglue-tc`
- `EuroEval/ltzglue-id`

All datasets are marked as `private=True` initially and can be made public
after review.

## Split Sizes

Standard EuroEval splits:
- **Train:** 1,024 samples (or 50% if dataset is smaller)
- **Validation:** 256 samples (or 15% if dataset is smaller)
- **Test:** Remaining samples (up to 2,048)

Stratified sampling is used when labels are available.
