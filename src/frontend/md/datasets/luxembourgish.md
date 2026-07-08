# 🇱🇺 Luxembourgish

## Reading Comprehension

### MultiWikiQA-lb

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages, including Luxembourgish.

The original dataset consists of 5,003 samples in a single train split. We use this
entire split as our test set for evaluation.

Here are a few representative examples:

```json
{
    "context": "Lëtzebuerg, offiziell d'Groussherzogtum Lëtzebuerg, ass e klengen am Floss vun Europa...",
    "question": "Wou läit Lëtzebuerg?",
    "answers": {
        "answer_start": [45],
        "text": ["an Zentral-Europa"]
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information):

- Number of few-shot examples: 4
- Prefix prompt:
  ```text
  Fir dësen Text ass eng Fro an d'Äntwert uginn.
  ```
- Base prompt template:
  ```text
  Text: {text}
  Fro: {question}
  Äntwert: {label}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-lb
```

## Sentiment Classification

### Unofficial: ltzGLUE-SA

This dataset is part of the ltzGLUE benchmark for Luxembourgish, introduced in
[this paper](https://arxiv.org/abs/2604.17976). It contains Luxembourgish texts
annotated with sentiment labels (negative, neutral, positive).

The dataset is uploaded to `EuroEval/ltzglue-sa` on the Hugging Face Hub.

```bash
euroeval --model <model-id> --dataset ltzglue-sa
```

## Linguistic Acceptability

### Unofficial: ltzGLUE-LA (Binary)

Part of the ltzGLUE benchmark. Binary classification task where Luxembourgish
sentences must be classified as grammatically correct or incorrect.

Uploaded to `EuroEval/ltzglue-lab` on Hugging Face.

```bash
euroeval --model <model-id> --dataset ltzglue-la-binary
```

### Unofficial: ltzGLUE-LA (Multi-class)

Part of the ltzGLUE benchmark. Multi-class version where errors are classified
by error type.

Uploaded to `EuroEval/ltzglue-lam` on Hugging Face.

```bash
euroeval --model <model-id> --dataset ltzglue-la-multi
```

### Unofficial: ScaLA-lb

Linguistic acceptability dataset created from the
[UD Luxembourgish-LuxBank treebank](https://github.com/UniversalDependencies/UD_Luxembourgish-LuxBank)
using the ScaLA algorithm. Sentences are corrupted by word swaps or deletions
to create grammatically incorrect examples.

**Note:** The current UD Luxembourgish treebank only contains ~100 sentences
(translated Cairo Cycling examples). A proper ScaLA-lb dataset requires
additional Luxembourgish text sources.

Uploaded to `EuroEval/scala-lb` on Hugging Face (pending).

```bash
euroeval --model <model-id> --dataset scala-lb
```

## Named Entity Recognition

### Unofficial: ltzGLUE-NER

Part of the ltzGLUE benchmark. Luxembourgish NER dataset with standard entity
types (PER, ORG, LOC, MISC).

Uploaded to `EuroEval/ltzglue-ner` on Hugging Face.

```bash
euroeval --model <model-id> --dataset ltzglue-ner
```

## Natural Language Inference

### Unofficial: ltzGLUE-RTE

Part of the ltzGLUE benchmark. Recognising Textual Entailment task with
three-way classification (entailment, contradiction, neutral).

Uploaded to `EuroEval/ltzglue-rte` on Hugging Face.

```bash
euroeval --model <model-id> --dataset ltzglue-rte
```

## Text Classification

### Unofficial: ltzGLUE-HC

Headline Classification from ltzGLUE. Binary task determining if a headline
is appropriate for a news article.

Uploaded to `EuroEval/ltzglue-hc` on Hugging Face.

```bash
euroeval --model <model-id> --dataset ltzglue-hc
```

### Unofficial: ltzGLUE-ID

Intent Detection from ltzGLUE. Multi-class classification of sentence intent.

Uploaded to `EuroEval/ltzglue-id` on Hugging Face.

```bash
euroeval --model <model-id> --dataset ltzglue-id
```

### Unofficial: ltzGLUE-TC

Topic Classification from ltzGLUE. Multi-class classification of news article
topics.

Uploaded to `EuroEval/ltzglue-tc` on Hugging Face.

```bash
euroeval --model <model-id> --dataset ltzglue-tc
```

## Summarisation

### LuxGen-Summ

Luxembourgish summarisation dataset from the
[LuxGen benchmark](https://arxiv.org/abs/2412.09415).

**Note:** Dataset creation script is a placeholder - actual dataset needs to be
extracted from LuxGen or created from Luxembourgish articles with summaries.

To be uploaded to `EuroEval/luxgen-summ` on Hugging Face.

```bash
euroeval --model <model-id> --dataset luxgen-summ
```

## Knowledge

### Unofficial: Luxembourgish Citizenship Tests

⏳ **To be created** — Knowledge dataset based on citizenship test questions
from [vivre-ensemble.lu](https://old.vivre-ensemble.lu/).

---

**Note:** All ltzGLUE datasets are marked as unofficial. Dataset creation
scripts are available in `src/scripts/dataset_creation/`. To upload the datasets,
clone the ltzGLUE repository with LFS support and run the scripts with uv.
