---
hide:
    - navigation
---
# Evaluation Methodology

The evaluation methodology is different depending on the architecture of the model. For
encoder models, we use a finetuning approach, where we finetune the model on the
training data of the task, and evaluate it on the test data. For decoder models, we use
either a few-shot or zero-shot approach, where we evaluate the model on the test data
without any finetuning, but where the few-shot examples come from the training data of
the task. It [has been shown](https://doi.org/10.48550/arXiv.2309.05858) that the few-shot
approach corresponds to finetuning in the sense of being equivalent to gradient updates
on the training data, making the two evaluation methodologies comparable.

This page describes the evaluation procedure shared across all tasks. The exact metric
(or metrics) used for each task, along with task-specific details, is described on the
individual [task pages](/tasks).

## Dataset Preparation

EuroEval does not evaluate on the full source datasets. Each dataset is re-sampled into
fixed splits — typically 1,024 training, 256 validation and 1,024 or 2,048 test samples —
so that the same evaluation can be run affordably across many model-and-dataset
combinations. The exact original and re-sampled split sizes for each dataset are listed on
its entry in the [dataset overview pages](/datasets). Because of this re-sampling, a
EuroEval score is not directly comparable to a score reported on the full original
dataset.

Before re-sampling, samples are filtered by length to remove degenerate or impractically
large examples and to help the assembled prompt fit within a model's context window. The
exact thresholds depend on the field being measured, but as a guide documents are kept to
roughly 2–5,000 characters, question-answering contexts to 30–5,000 characters,
questions to 10–150 characters, summarization articles to 30–6,000 characters, and
multiple-choice instructions to 30–2,000 characters with options up to 1,000 characters.
Highly repetitive samples (a single token repeated more than 50 times) are also dropped.
These filters are applied before subsampling, so they shape which samples can end up in
the EuroEval splits.

## Model Types

EuroEval evaluates several families of models, and the evaluation strategy depends on
the family:

- **Encoder models** (e.g. BERT-style models) are evaluated by finetuning on the task's
  training data and scoring the test split (see the encoder section below).
- **Base decoder models** — completion-only models that are not instruction tuned — are
  evaluated few-shot with a plain completion prompt. They expose token-level
  log-probabilities, which EuroEval uses to score the classification tasks.
- **Instruction-tuned decoder models** are evaluated few-shot using the model's own chat
  template.
- **Reasoning decoder models** are treated like instruction-tuned models but are given a
  large thinking budget; their `<think> … </think>` spans are removed before the answer
  is scored.
- **Hosted API models** (OpenAI, Anthropic, Google, xAI, and others) are evaluated
  through their APIs, where the available features (log-probabilities, structured output)
  depend on the provider.

The model type is detected automatically from the model's configuration and name, and
can be overridden with the `--generative-type` flag.

## Robust Evaluation

For each model and dataset, we evaluate the model as described above 10 times, each time
on a bootstrapped (i.e., sampling with replacement) version of the training and test
set. The evaluation score is then the mean of these scores, along with a 95% confidence
interval, computed as the mean ± 1.96 x [standard
error](https://en.wikipedia.org/wiki/Standard_error) of the mean, where the standard
error of the mean is the [sample standard
deviation](https://en.wikipedia.org/wiki/Standard_deviation#Corrected_sample_standard_deviation)
divided by the square root of the number of samples.

The bootstrap theorem means that this mean and associated confidence interval will be
asymptotically correct, giving us a more reliable estimate of the true performance of
the model, rather than just the performance on a single test set, which can be noisy.

## Encoder Finetuning

Encoder models are finetuned anew for every bootstrap iteration: EuroEval runs ten
independent finetunes, one per resampled training set, rather than finetuning once and
re-scoring it ten times. This keeps the encoder scores on the same footing as the
bootstrapped few-shot scores of the generative models.

Every finetune uses the same model-agnostic recipe:

- **Optimiser:** AdamW with a learning rate of 2e-5.
- **Schedule:** up to 10,000 steps with a short warmup and early stopping (a patience of
  two evaluations) on the validation split, restoring the best checkpoint at the end. In
  practice early stopping almost always halts training long before the step limit.
- **Batch size:** an effective batch size of roughly 32, reached through gradient
  accumulation. The per-device batch size is reduced automatically if the GPU runs out of
  memory, and mixed precision falls back from bf16/fp16 to fp32 if numerical instability
  (NaNs) is detected.
- **Reproducibility:** a fixed seed per iteration (4242 plus the iteration index) with
  deterministic backend flags set.

The validation split is used only for early stopping; the reported score always comes
from the test split. Inputs longer than the 8,192-token context limit are truncated.

## Prompt Structure

We set up the prompts differently depending on whether the model is instruction tuned or
not, as the instruction tuned models require a different prompt structure to ensure that
they generate the correct output.

For the base (i.e., non-instruction tuned) models, we use the following prompt
structure:

```text
[prefix prompt]

{% for each few-shot example %}
  [document prefix]: [few-shot example document]
  [label prefix]: [few-shot example label]
{% end for %}

[document prefix]: [new document]
[label prefix]:
```

For the instruction tuned models, we use the following prompt structure:

```text
{% for each few-shot example %}
  USER: [instruction with few-shot example]
  ASSISTANT: [label]
{% end for %}
USER: [instruction with new example]
ASSISTANT:
```

Here we would use the model's chat template to set up the `USER` and `ASSISTANT` parts
of the prompt. See all the specific prompts used for each dataset in the [dataset
configs module](/src/euroeval/dataset_configs/#euroeval.dataset_configs).

A few tasks need task-specific handling on top of this template:

- **Sentiment classification:** the models generate translations of the three labels
  (positive, negative and neutral).
- **Linguistic acceptability:** also a text classification task, where we use the
  translations of "yes" and "no" as the two labels, corresponding to whether the document
  is grammatically correct or not.
- **Extractive question answering:** the model outputs the answer directly. We found that
  changing the label prefix from "Answer" to "Answer in max 3 words" gave a drastic
  improvement, as many instruction-tuned answers otherwise start with unnecessary text
  like "The answer is".
- **Named entity recognition:** the output must be a JSON dictionary whose keys are the
  translated named entity tags and whose values are lists of the named entities in each
  category. To avoid biasing the evaluation toward models that happen to know the JSON
  format, we use structured generation via the
  [XGrammar](https://github.com/mlc-ai/xgrammar) package, which constrains the model's
  logits so that the output is always a valid JSON dictionary in the aforementioned
  format.

## Few-shot Evaluation

Generative models are not finetuned. Instead, each test example is preceded by a small
number of labelled examples drawn from the (bootstrapped) training split — the few-shot
examples — laid out using the prompt structure above. The number of few-shot examples is
task-dependent: for instance, twelve for sentiment and linguistic acceptability, five for
most multiple-choice tasks, and fewer for long-input tasks such as summarisation.

How the few-shot examples are chosen matters:

- For classification tasks they are **label-stratified** — drawn evenly across the labels
  so that no single class dominates the context.
- They are **re-sampled for every bootstrap iteration** (with the seed 4242 plus the
  iteration index), so the particular choice of few-shot examples contributes to the
  measured variance rather than being a single lucky or unlucky draw.
- Over-long examples are filtered out so that the assembled prompt fits the context
  window.

Generation itself uses a few fixed settings:

- **Greedy decoding** (temperature 0) by default, so the output is deterministic given
  the prompt.
- A **per-task cap on the number of generated tokens** — a handful for classification,
  more for question answering and summarisation. Reasoning models are given a much larger
  budget (8,192 tokens) for their thinking before the answer.
- A **hard prompt limit of 8,192 tokens.** When a prompt would exceed it, EuroEval first
  drops few-shot examples one at a time (for instruction-tuned and reasoning models) and
  only truncates the text itself as a last resort, so the actual question is preserved.

## Score Aggregation

Each model is evaluated on many datasets spanning several tasks and languages, and we
need to combine these into a single comparable score. A good aggregation method should
satisfy the following criteria:

- **Task Fairness:** every task is weighted equally, regardless of metric range or
  variance.
- **Comparison:** scores stay meaningful when comparing models across languages.
- **Robustness:** models that do not differ significantly end up with similar scores.
- **Magnitude Preservation:** larger gaps between models survive aggregation.
- **Minimal Change:** adding a new model barely moves the existing models' scores.

The two obvious methods each satisfy only part of this. The **mean score** (averaging the
raw scores) preserves magnitude and robustness, but fails Task Fairness and Comparison,
since metrics have different ranges and variances and datasets differ in difficulty from
language to language. The **mean rank** (averaging each model's per-dataset ranks) fixes
Task Fairness and Comparison by recasting every dataset onto a common scale, but discards
magnitude and robustness by construction. They satisfy disjoint halves of the criteria,
so we combine them into the **mean rank score**.

### Mean rank score

For each dataset _d_ and model _m_ we assign a _dataset rank score_:

```text
rank_score(m, d) = 1 + (best_mean_score(d) - mean_score(m, d)) / pooled_std(d)
```

Here `mean_score(m, d)` is model _m_'s mean bootstrap score on dataset _d_,
`best_mean_score(d)` is the highest such mean across all models on _d_, and
`pooled_std(d)` is the standard deviation of all bootstrap scores from all models on _d_.
The best model on a dataset therefore scores exactly **1**, and every other model scores
**1 plus the number of pooled standard deviations it sits behind the leader**. Dividing
by `pooled_std(d)` places all datasets on a common scale (Task Fairness, Comparison)
while preserving the size of the gaps between models (Magnitude Preservation).

We then aggregate with **unweighted means up the hierarchy** — dataset rank scores into
a task rank score, task rank scores into a language rank score, and language rank scores
into the overall mean rank score (shown in the **Rank score** column). Equal weights at
every level keep each dataset, task, and language equally important, regardless of how
many of each there happen to be.

### Confidence intervals

The leaderboard reports each overall mean rank score as **score ± margin**, where the
margin is a 95% confidence interval computed by bootstrap resampling rather than by
propagating analytical error through the nested means:

1. **Resample datasets with replacement**, stratified by task (each task's datasets are
   resampled independently, preserving the task structure).
2. **Recompute the full hierarchy** — dataset scores, task means, language means, overall
   mean — for every model.
3. **Repeat** 100 times and take the 2.5th and 97.5th percentiles as the interval, with
   the median as the point estimate.

Because every model is scored on the same resampled datasets, the bootstrap correctly
accounts for the correlation between models that share datasets, which makes near-ties
visible (Robustness). Adding a model can only shift `pooled_std(d)` and the per-dataset
leader locally, so its effect on other models is small and shrinks as more models are
added (Minimal Change).

### Rank

Alongside the mean rank score the leaderboard shows an integer **Rank** column — a
_dense_ ordinal ranking. After sorting the models by overall mean rank score (lower is
better), we walk down the list and compare each model to its current tie-group anchor
using a one-sided bootstrap test (α = 0.05) on the **difference** of their overall scores:

- If the anchor is not significantly better (p ≥ 0.05), the candidate joins the anchor's
  tie group and shares its rank.
- Otherwise, it starts a new tie group with the next rank.

The result is a contiguous **1, 2, 3, …** sequence in which models can share a rank, with
no gaps after a tie. Because the test reuses the same resampled datasets for both models,
it properly accounts for their shared-dataset correlation.

## Bits-per-character Scoring

For base decoder models, EuroEval supports bits-per-character (BPC) scoring via the
`--use-bits-per-character`/`-bpc` flag. This computes the information content of the
ground-truth answer conditioned on the question. It is useful for evaluating training
checkpoints, as it gives a more granular signal than the task metrics: small models and
early checkpoints typically struggle with complex formats like multiple-choice
classification, where the standard metrics saturate near chance.

For multiple-choice tasks, BPC treats the benchmark like text-to-text: the model sees a
bare-question prompt (no choice options listed) and is scored on the full answer text. We
compute `sum(log P(answer_tokens | prompt))` and normalise by character length, matching
the approach used by Llama and the EleutherAI LM Evaluation Harness.

For other task types, BPC scores the ground-truth answer directly. The one exception is
named entity recognition (and token classification more broadly), where the answer is a
serialised JSON dictionary (as described above): since most of that string is fixed
scaffolding (the bracket, brace and key characters) that a model predicts near-perfectly
after the few-shot examples, we normalise only by the number of characters in the tagged
entities themselves rather than the full serialised string, so the score reflects how
well the model predicts the entities rather than the JSON format.

BPC is only supported by the vLLM backend with base decoder models; instruction-tuned
models raise an error. BPC runs are excluded from official leaderboards, which only
display standard accuracy scores for consistency.

## Leaderboards

The public leaderboards add a few rules on top of the per-model evaluation described
above:

- **Model categories.** Each language has a _generative_ leaderboard, covering every
  task, and an _all models_ leaderboard restricted to the NLU tasks, so that encoder
  models can be compared on an equal footing. Models also carry metadata — generative
  type, open vs. closed weights, commercial-use permission, whether the model is a merge,
  parameter count and context length — which can be filtered on the site.
- **One result per model.** When several result records exist for the same model and
  dataset, EuroEval keeps the one produced by the newest framework version and drops
  validation-split results whenever a test-split result is available, so no model is
  counted twice.
- **Orthogonal tasks.** A few evaluations — currently the European values survey — probe
  a model's values rather than its capabilities. These are reported in their own columns
  and are deliberately left out of the rank score.
- **Inference speed.** EuroEval can also measure inference speed (in GPT-2 tokens per
  second), but speed is reported separately and kept out of the rank score, which
  reflects quality only.

## Papers

Check out more in-depth descriptions of the methodology in the associated research
papers:

- [Encoder vs Decoder: Comparative Analysis of Encoder and Decoder Language Models on
  Multilingual NLU Tasks](https://doi.org/10.48550/arXiv.2406.13469)
- [ScandEval: A Benchmark for Scandinavian Natural Language
  Processing](https://aclanthology.org/2023.nodalida-1.20/)
