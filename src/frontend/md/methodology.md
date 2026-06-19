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

## Formulating NLU Tasks as Generative Tasks

In this section we describe how we rephrase the NLU tasks as text-to-text tasks, which
makes it possible to evaluate generative models on the tasks. We set up the prompts
differently depending on whether the model is instruction tuned or not, as the
instruction tuned models require a different prompt structure to ensure that they
generate the correct output.

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

For the sentiment classification task, we simply have the models generate translations
of the three labels (positive, negative and neutral). For the linguistic acceptability
task, also a text classification task, we use the translations of "yes" and "no" as the
two labels, corresponding to whether the document is grammatically correct or not. For
the extractive question answering task, we have the model output the answer directly.
For this task we found that changing the label prefix from "Answer" to "Answer in max 3
words" resulted in a drastic improvement, due to many of the answers of instruction
tuned models starting with unnecessary text akin to "The answer is". Lastly, for the
named entity recognition task, we require the output to be a JSON dictionary, with keys
being the translated named entity tags, and values being lists of named entities of that
category. To ensure that we are not biasing the evaluation toward models knowing the
JSON format, we employ structured generation using the
[XGrammar](https://github.com/mlc-ai/xgrammar) package, which modifies the logits
outputted by the model to ensure that the output is always a valid JSON dictionary in
the aforementioned format.

## Score Aggregation

From the raw scores of the 10 evaluations per dataset, we need to aggregate
the model scores into a single score. We want an aggregation method that satisfies the
following criteria:

- **Task Fairness:** Each task should be weighted equally.
- **Comparison:** If we evaluate models in multiple languages, then it should be
  possible to meaningfully compare the language scores of these models with each other.
- **Robustness:** If two models do not have a significantly different score on a
  dataset, then the aggregated score should reflect this.
- **Magnitude Preservation:** The magnitude of the difference between the dataset score
  of two models should be reflected in the aggregated score.
- **Minimal Change:** Adding a new model should minimally affect the aggregated scores
  of the other models.

Before we introduce our chosen aggregation method, we will briefly discuss some common
aggregation methods and how they do not satisfy the criteria.

The **mean score** is the most common aggregation method, which would simply be the mean
of the 10 scores for each dataset, and then the mean of the dataset scores for each
task. This method does not satisfy the Task Fairness criterion, as it does not take into
account that metrics have different ranges and variances. The Comparison criterion is
also not satisfied, as datasets vary from language to language, with some datasets being
more difficult than others. It _does_, however, satisfy the Robustness, Magnitude
Preservation and Minimal Change criteria.

The **mean rank** is another common aggregation method, where we compute the rank of
each model on each dataset, and then take the mean of the ranks. This method satisfies
the Task Fairness criterion, as it re-casts the scores into a common comparable
framework, which therefore weights each task equally. For the same reason, it also
satisfies the Comparison criterion (it is important here that we evaluate all the models
on all the languages for this to be satisfied). It does not satisfy the Robustness and
Magnitude Preservation criteria, by definition of rank. It partially satisfies the
Minimal Change criterion, since it only affects the scores of the models which are worse
than the new model.

We thus see that the mean score and mean rank methods satisfy a disjoint set of the
criteria, but that they together satisfy all the criteria. Based on this observation, we
introduce the **mean rank score** method, defined as follows.

### Mean rank score

For each dataset _d_ and model _m_, we assign a _dataset rank score_:

```text
rank_score(m, d) = 1 + (best_mean_score(d) - mean_score(m, d)) / pooled_std(d)
```

Here `mean_score(m, d)` is model _m_'s mean bootstrap score on dataset _d_,
`best_mean_score(d)` is the highest such mean across all models on _d_, and
`pooled_std(d)` is the standard deviation of all bootstrap scores from all models on
_d_. The best model on each dataset therefore gets a rank score of exactly **1**,
and every other model gets **1 plus the number of pooled standard deviations it
sits behind the leader**.

We then aggregate by taking **unweighted means at each level of the hierarchy**:

- **Task rank score:** mean of the dataset rank scores for that task.
- **Language rank score:** mean of the task rank scores for that language.
- **Overall mean rank score:** mean of the language rank scores.

Each level weights its children equally — every dataset within a task, every task
within a language, every language overall. This preserves Task Fairness regardless
of how many datasets a task happens to have or how many tasks a language happens to
have.

#### Confidence intervals

The dataset rank score inherits a 95% confidence interval directly from the
bootstrap standard error of the underlying mean score, divided by `pooled_std(d)`.
Because every aggregation step above is an unweighted mean, variances propagate by
the standard rule:

```text
Var(mean(x_1, ..., x_n)) = (Var(x_1) + ... + Var(x_n)) / n^2
```

The leaderboard reports the overall mean rank score (shown in the **Rank
score** column) as **score ± margin**, where
the margin is a 95% confidence interval computed via bootstrap resampling.

#### Bootstrap confidence intervals

Rather than propagating analytical error bounds through the nested mean
structure, we compute the CIs empirically:

1. **Resample datasets with replacement**, stratified by task (each task's
datasets are resampled independently, preserving the task structure).
2. **Recompute the full hierarchy** on the resampled datasets — dataset scores,
task means, language means, overall mean — for every model.
3. **Repeat** 100 times and collect the distribution of overall scores.
4. **Take the 2.5th and 97.5th percentiles** as the 95% CI bounds, with the
median as the point estimate.

This approach respects the hierarchical structure (dataset → task → language →
overall) and the correlation between models that share datasets. Because both
models are evaluated on the same resampled datasets, their bootstrap scores are
correlated, and the difference distribution correctly accounts for this.

#### Why this works

This metric satisfies **Task Fairness** because we normalise every score by the
dataset's pooled standard deviation and aggregate with equal weights at every level.
**Magnitude Preservation** holds because the magnitude of the difference between two
models' dataset scores survives the linear normalisation and the mean aggregation.
**Comparison** holds because all models are placed on a common scale (same argument as
the mean rank method). **Robustness** is satisfied by the bootstrap confidence
intervals on the overall mean rank score: overlapping intervals make near-ties
immediately visible, and the dense Rank column described below shares a rank between
any two models whose intervals overlap. **Minimal Change** is partially satisfied —
adding a new model can shift `pooled_std(d)` and, if it becomes the new leader on
some dataset, shift `best_mean_score(d)`. Both effects are local to the affected
dataset(s) and tend to zero as the number of models grows.

### Rank

Alongside the mean rank score the leaderboard shows an integer **Rank** column,
which is a _dense_ ordinal ranking computed via bootstrap hypothesis testing.
After sorting the models by overall mean rank score (lower is better), we walk
down the list and compare each model to the current tie-group anchor using a
one-sided bootstrap test (α = 0.05):

- We compute the bootstrap distribution of the **difference** between the anchor's
and candidate's overall scores (lower = better).
- If the p-value ≥ 0.05 (the anchor is not significantly better), the candidate
joins the anchor's tie group and shares its rank.
- Otherwise, it starts a new tie group with rank one larger than the previous
group's.

The result is a contiguous **1, 2, 3, …** sequence in which multiple models can
share 1st place, 2nd place, and so on — there are never gaps after a tie.

The bootstrap test is statistically proper: it uses the same resampled datasets
for both models, so the difference distribution correctly accounts for the
correlation induced by shared datasets. This addresses the limitations of the
previous approaches — the analytical paired t-test (which ignored the hierarchical
structure) and the CI-overlap heuristic (which was not a formal test).

## Papers

Check out more in-depth descriptions of the methodology in the associated research
papers:

- [Encoder vs Decoder: Comparative Analysis of Encoder and Decoder Language Models on
  Multilingual NLU Tasks](https://doi.org/10.48550/arXiv.2406.13469)
- [ScandEval: A Benchmark for Scandinavian Natural Language
  Processing](https://aclanthology.org/2023.nodalida-1.20/)
