# EuroEval

Terms used to describe EuroEval's models, benchmark evaluations, results, and public
leaderboards.

## Models and benchmark design

**Benchmark**: A defined evaluation of one or more models on a collection of tasks and
datasets. _Avoid_: benchmark run (when referring to the evaluation itself)

**Model**: A language model being evaluated, identified by its model identity and
associated metadata.

**Model type**: A broad capability family: encoder, generative model, or zero-shot
classifier. This is distinct from generative type, which distinguishes base,
instruction-tuned, and reasoning generative models.

**Task**: An evaluation objective, such as sequence classification, question answering,
or translation, with its applicable metrics and evaluation conventions.

**Task group**: A family of tasks sharing a broad output or evaluation form, such as
token classification, multiple-choice classification, or text-to-text generation. A task
is a specific objective within a task group.

**Dataset**: A collection of examples in a language or language pair, used to evaluate a
task.

**Dataset configuration**: The definition of how a dataset is presented for evaluation,
including its task, languages, splits, and any dataset-specific settings. _Avoid_:
dataset (when referring specifically to its evaluation definition)

**Benchmark configuration**: The selection and settings for a benchmark, including the
datasets and evaluation options.

**Split**: A partition of a dataset, commonly train, validation, or test. Results
identify whether they were evaluated on the validation or test split. _Avoid_: subset
(when referring to a named dataset partition)

**Few-shot**: Evaluation in which a prompt includes examples from the task to guide the
model.

**Zero-shot**: Evaluation without few-shot examples in the prompt.

**Auto shot mode**: A request to select applicable few-shot or zero-shot evaluation
modes automatically according to the model and evaluation context; it is a policy, not a
result mode. _Avoid_: automatic-shot result

**Metric**: A measure used to assess model performance on a task, such as accuracy or
F1.

**Score**: A metric's measured value for a model on a dataset or task; scores from
different metrics or datasets are not necessarily directly comparable.

**Language pair**: An ordered source and target language used by a translation dataset
or task. _Avoid_: languages (when the direction of translation matters)

**Multilingual / monolingual**: A multilingual dataset, evaluation, or leaderboard
covers multiple languages; a monolingual one covers one language. These describe
language coverage, not a leaderboard category.

## Results and leaderboards

**Result record**: The result of evaluating a particular model on a dataset, including
its scores and relevant evaluation details. _Avoid_: benchmark (when referring to one
recorded model-dataset outcome)

**Result identity**: The combination of model, dataset, evaluation split, and shot mode
that distinguishes one result record from another.

**Leaderboard**: A public comparison of eligible models using results within a defined
category and task scope.

**Leaderboard category**: A named comparison grouping—Chat, Generative, Understanding,
or All Models—whose eligible models and task set determine its leaderboard. Category is
distinct from language coverage: a category may be presented for a particular language.
_Avoid_: language category

**Leaderboard scope**: The set of tasks and eligible results included in a particular
leaderboard; rankings apply only within that scope. _Avoid_: leaderboard category (when
referring to the included tasks)

**Mean rank score**: A leaderboard score aggregating per-dataset performance relative to
other models, with equal weighting through dataset, task, and language levels; lower is
better.

**Rank**: A dense ordinal leaderboard position derived from mean rank scores and their
uncertainty; models judged statistically tied can share a position. _Avoid_: rank score
(when referring to the ordinal position)

## Evaluation queue and review

**Evaluation request**: A request to evaluate a model, typically specifying the model
and requested language coverage; it enters the volunteer evaluation queue. _Avoid_:
result submission

**Volunteer result submission**: Evaluation records uploaded by a volunteer worker for
an evaluation request and provided for review. It is distinct from the request itself;
review determines whether submitted results are accepted for publication and volunteer
credit. _Avoid_: evaluation request

**Volunteer review**: The assessment of a volunteer result submission before its results
are accepted or rejected. It is distinct from running the evaluation.
