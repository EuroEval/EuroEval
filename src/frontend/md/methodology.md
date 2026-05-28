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

For each dataset, we assign every model a _dataset rank score_:

$$
r_{m,d} \;=\; 1 \;+\; \frac{\bar{s}^{\,*}_d - \bar{s}_{m,d}}{\sigma_d}
$$

where $\bar{s}_{m,d}$ is model $m$'s mean score on dataset $d$, $\bar{s}^{\,*}_d$ is
the highest mean score on dataset $d$ across all models, and $\sigma_d$ is the pooled
standard deviation of the bootstrap scores across all models on the dataset. The best
model on each dataset therefore gets a rank score of exactly 1, and every other model
gets 1 plus the (positive) number of pooled standard deviations it sits behind the
leader. There is no Welch's t-test gating in this step — every dataset score is
normalised the same way, so a small bootstrap-noise difference contributes a small
rank-score difference rather than being collapsed to zero.

We then aggregate by taking unweighted means at each level of the hierarchy. The
_task rank score_ is the mean of the dataset rank scores for that task, the
_language rank score_ is the mean of the task rank scores for that language, and the
_overall mean rank score_ is the mean of the language rank scores:

$$
r_{m,\text{task}} \;=\; \frac{1}{|D_{\text{task}}|} \sum_{d \in D_{\text{task}}} r_{m,d},
\quad
r_{m,\text{lang}} \;=\; \frac{1}{|T_{\text{lang}}|} \sum_{t \in T_{\text{lang}}} r_{m,t},
\quad
r_{m} \;=\; \frac{1}{|L|} \sum_{\ell \in L} r_{m,\ell}.
$$

Each level is weighted equally (per dataset within a task, per task within a
language, per language overall), which preserves Task Fairness regardless of how many
datasets a task happens to have or how many tasks a language happens to have.

#### Confidence intervals

The dataset rank score has a 95% confidence interval derived directly from the
bootstrap standard error of $\bar{s}_{m,d}$, scaled by $1/\sigma_d$. Because every
aggregation step above is an unweighted mean, variances propagate by the standard
rule $\mathrm{Var}(\bar{x}) = \sum_i \mathrm{Var}(x_i)/n^2$, and the leaderboard
reports the overall mean rank score as $\text{score} \pm \text{margin}$ at the 95%
level using the propagated variance.

#### Why this works

This metric satisfies Task Fairness since we normalise all the scores by dividing by
the standard deviation of the dataset scores, and aggregate with equal weights at
every level. The Magnitude Preservation criterion is satisfied because the magnitude
of the difference between two models' dataset scores is preserved in the rank score
(and survives the mean aggregation). Comparison is satisfied because models are
compared on a common scale (same argument as the mean rank method). Robustness is
satisfied at display time rather than per-dataset: instead of gating each pairwise
comparison with a t-test, we expose the propagated 95% confidence interval on the
overall mean rank score, so overlapping intervals make near-ties immediately
visible. Minimal Change is partially satisfied — adding new models can shift
$\sigma_d$ and, if a new model becomes the leader on some dataset, shift the anchor
$\bar{s}^{\,*}_d$. Both effects are local to the dataset(s) involved and tend to
zero as the number of models grows.

### Rank

Alongside the mean rank score the leaderboard shows an integer **Rank** column,
which is a _dense_ ordinal ranking with Welch's-t-test-based tie detection. After
sorting the models by overall mean rank score, we walk down the list and compare
each model's raw bootstrap scores to those of the current tie-group anchor using a
one-tailed Welch's t-test. If the anchor is _not_ significantly better (p ≥ 0.05),
the model joins the anchor's tie group and shares its rank. Otherwise it starts a
new tie group with rank one larger than the previous group's. The result is a
contiguous 1, 2, 3, … sequence in which multiple models can share 1st place, 2nd
place, and so on — there are never gaps after a tie.

## Papers

Check out more in-depth descriptions of the methodology in the associated research
papers:

- [Encoder vs Decoder: Comparative Analysis of Encoder and Decoder Language Models on
  Multilingual NLU Tasks](https://doi.org/10.48550/arXiv.2406.13469)
- [ScandEval: A Benchmark for Scandinavian Natural Language
  Processing](https://aclanthology.org/2023.nodalida-1.20/)
