# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [Unreleased]
### Added
- Now automatically uses multiple GPUs when evaluating generative models with vLLM.
- Now allows "unofficial" datasets, which are datasets which are not included on the
  official leaderboards and models will only be benchmarked on them if they have been
  explicitly set using the `--dataset` argument (or `dataset` argument if using the
  `Benchmarker` API). This allows the inclusion of more datasets, without bloating the
  evaluation time of "official" evaluations, as well as removing the need to remove old
  datasets when they are replaced by newer ones.
- The following datasets have been added as unofficial, all datasets that used to be
  part of ScandEval but has since been replaced:
    1. ARC-da
    2. ARC-no
    3. ARC-sv
    4. ARC-is
    5. ARC-de
    6. ARC-nl
    7. ARC
    8. DaNE
    9. WikiANN-fo

### Changed
- Computation of the BERTScore metric for summarisation tasks are now using the device
  stated in the benchmark config, making the metric computation significantly faster if
  a GPU is being used. This defaults to processing 32 samples at a time, which is
  reduced if OOM errors occur. If OOM errors occur with a batch size of 1 then the
  scores are computed on CPU, as before.
- Updated `transformers` dependency to `>=4.38.1,<4.39.0`, and `vllm` dependency to
  `>=0.3.2,<0.4.0`. This allows the benchmarking of the new Gemma and OLMO models.


## [v11.0.0] - 2024-02-16
### Added
- Added arguments to `Benchmarker.benchmark` (or simply `Benchmarker.__call_`),
  corresponding to the same arguments during initialisation. The idea here is that the
  default parameters are set during initialisation, and then any of these can be
  changed if needed when performing a concrete evaluation, without having to
  re-initialise the `Benchmarker`.
- Added the Danish knowledge datasets `danske-talemaader` and `danish-citizen-tests`.
  Both are multiple choice datasets, where the first one tests knowledge about Danish
  idioms, and the second one tests knowledge about the Danish society. These replace
  the machine translated MMLU-da dataset.
- Added a `--num-iterations` flag (`num_iterations` in the Python CLI), which controls
  the number of times each model should be evaluated, defaulting to the usual 10
  iterations. This is only meant to be changed for power users, and if it is changed
  then the resulting scores will not be included in the leaderboards.

### Changed
- The default value of the languages are now all languages, rather than only Danish,
  Swedish and Norwegian.
- Changed all summarisation datasets to use one few-shot example (some were set to 2),
  and increased the maximum amount of generated tokens to 256 rather than the previous
  128, since many of the gold standard summaries are around 200 tokens.

### Fixed
- There was an error caused if an old version of the `openai` package was installed and
  if the `scandeval` package was checking if a model exists as an OpenAI model. Now an
  informative error is thrown if the model is not found on any available platforms, as
  well as noting the extras that are missing, which prevents the package from checking
  existence on those platforms.
- Changed the prompt for the English sentiment classification dataset SST5, where it
  previously stated that the documents were tweets - these have now been renamed to
  "texts".
- Correctly assess whether the `openai` extra should be used, which made it impossible
  to benchmark OpenAI models.
- Disabled `lmformatenforcer` logging, which happens in the rare case when we're
  few-shot evaluating a model on NER and there are no JSON-valid tokens to generate.

### Removed
- Removed all machine translated ARC datasets, as they had a near 100% correlation with
  the machine translated version of the MMLU datasets.


## [v10.0.1] - 2024-02-12
### Fixed
- A prefix space was added to labels in sequence classification tasks that
  automatically adds a prefix space (such as Mistral). We now check for this and ensure
  to only manually add prefix space to models that don't automatically do this (such as
  the Yi models).


## [v10.0.0] - 2024-02-12
### Added
- Now throws a more informative error when attempting to benchmark a non-generative
  model on a generative task.

### Changed
- Many dependencies are now optional, to make the package less bloated. These extras
  are `jax`, for models based on the JAX framework, `generative` for evaluating
  generative models, `olmo` for models based on the OLMO architecture, `openai` for
  evaluating OpenAI models, and `all` to install all of them.
- Updated many dependencies. In particular now uses `openai` version 1.x.x, which
  required some changes to the code base as they changed their API.
- Changed the `--dataset-task` CLI argument (`dataset_task` in the Python API) to
  `--task` (`task`). This is now the preferred way to choose what to benchmark a model
  on, rather than remembering all the names of the datasets. E.g., to benchmark a model
  on all Danish question-answering datasets, we call `scandeval -m <model_id> -l da -t
  question-answering`. All the names of the tasks is shown in `scandeval --help`.
- Renamed the `--no-ignore-duplicates` to `--force` (shorthand: `-f`), which _forces_
  the evaluation, meaning that it evaluates the model even if it has previously been
  evaluated.
- Renamed the `--model-id` to `--model`.

### Fixed
- Error when encoding a batch of size 1 with OpenAI models.
- Error when benchmarking OpenAI models on MacOS due to the `tiktoken.Encoding` object
  not being picklable.
- Fixed an issue with OOM errors when changing from benchmarking one generative model
  to another.
- Now allows loading tokenisers that require remote code, if `--trust-remote-code` has
  been set.
- Fixed an issue where the `max_sequence_length` parameter in the Hugging Face model
  configuration wasn't used to determine the `max_model_len` parameter in the
  `vllm.LLM` initialisation, causing some models not being loaded in vLLM.
- An error occured if a tokenizer had no defined BOS token, which happens for some
  generative models. It is now set to be equal to the EOS token in that case.
- Fixed error related to the extraction of predicted labels in sequence classification
  tasks for generative models, which unfairly evaluated generative models that require
  a prefix space on the labels (which are most of them currently).

### Removed
- Removed the `-d` shorthand for `--dataset` in the CLI, to encourage the use of `-t`
  (`--task`) and `-l` (`--language`) instead.


## [v9.3.2] - 2024-02-05
### Fixed
- Using model revisions did not work with vLLM models - this has now been fixed. These
  revisions are specified using the '@' operator in the model ID, e.g., `scandeval -m
  gpt2@main`.


## [v9.3.1] - 2024-01-31
### Fixed
- The prompts were not stripped correctly, causing bad evaluations for sequence
  classification tasks.


## [v9.3.0] - 2024-01-29
### Changed
- Now requires `transformers` versions `4.37.x`. As they often introduce breaking
  changes in minor versions, we now only allow a patch version difference and manually
  update to `4.38.x` when it comes out.
- Swapped primary/secondary metrics for the multiple choice tasks, where we now set MCC
  as the primary metric and accuracy and secondary. This is due to the fact that MCC
  handles class imbalance better.
- Removed speculative ngram sampling again, as `transformers` now requires the batch
  size to be 1, which doesn't make it any faster than normal.
- Swapped primary/secondary metrics for the multiple choice tasks, where we now set MCC
  as the primary metric and accuracy and secondary. This is due to the fact that MCC
  handles class imbalance better.
- Number of generated tokens for sequence classification tasks has been changed back to
  3 (from 1). This makes no difference to open source models, as we only use the
  logprobs from the first token anyway, but it *does* make a difference to closed
  source models where the logprobs are not available (like OpenAI's chat models), as
  we're instead calculating word edit distance to the labels.

### Fixed
- Prevents FP16 overflow by using -1e3 instead of -1e9 for ~0% probability logprobs
  during generation with vLLM.
- Avoids excessive disk usage by not caching processed datasets to disk, as we are
  never using the cached versions anyway.
- We now only strip the prompts if the model's tokenizer includes a prefix space when
  tokenizing the labels.
- When testing a model's maximum sequence length, we put dummy inputs into them. This
  causes errors if the dummy inputs are one of the special tokens. Since the special
  tokens have not always been set up in the tokenizer, we instead rely on a heuristic
  that the 100th token ID is not a special token.
- An import depended on `vllm`, which is not installed on non-Linux devices, causing an
  `ImportError`. This has now been removed.
- Fixed an issue where structured generation wasn't triggered when vLLM wasn't
  available.


## [v9.2.0] - 2024-01-24
### Added
- Added (the English) datasets MMLU, ARC and HellaSwag, as well as Norwegian and
  Icelandic translations of it. Now the `knowledge` and `common-sense-reasoning` tasks
  are covered in all supported languages except Faroese (i.e., da, sv, no, is, de, nl &
  en).
- Now uses speculative ngram sampling for text generation when vLLM is not available.
  This has no effect on performance and increases evaluation speed by 3x on generation
  heavy tasks like NER and summarization.
- Added structured generation for the NER task, which enables the models to (almost)
  always output correct JSON, separating the NER capabilities from the JSON
  capabilities. JSON can be tested separately in a (future) coding benchmark.
- Now adds `scandeval_version` to the output JSONL results, to make it easier to
  determine when outdated results need re-benchmarking.

### Changed
- Swapped primary/secondary metrics for the NER task, as the `MISC` tag varies too much
  from dataset to dataset to be meaningful as a primary metric. Now uses micro-average
  F1-score across all tags except the `MISC` tag as a primary metric.

### Fixed
- There was a bug where all models were removed from disk prior to benchmarking. This
  will now only happen if the `--clear-model-cache` flag is set.
- The `vllm` package cannot be installed when CUDA is not available - this is now
  neither installed nor used when this is the case, and generative few-shot evaluation
  is done using the `transformers` package rather than `vllm`.
- Previously `temperature` was wrongly not set for vLLM and OpenAI models, instead
  defaulting to their 1.0 values. This was due to the fact that this is set in
  `transformers` using the `do_sample=False` argument, which doesn't transfer to the
  other libraries. This has now been set to 0.0.
- Now catches OpenAI `InvalidRequestError`s.
- Removed overly long or repetitive samples in the multiple choice datasets, which
  caused errors when evaluating OpenAI models on them.
- Now sets the `top_k` parameter in the vLLM `SamplingParams` based on the value it has
  in the `GenerationConfig`. This caused a discrepancy, as vLLM defaulted to -1 and
  `transformers` to 50.
- When loading a model using `transformers` then the quantized compute dtype is now
  correctly set to either `bfloat16` or `float16`, depending on the GPU available,
  rather than the previous `float32`. This does not affect generation performance.
- Fixed formatting of summarization metrics.
- Removed print output from `bert_score` during summarization metric computation.
- Now clears GPU memory properly after finishing the benchmark of a generative model
  with vLLM.


## [v9.1.2] - 2024-01-16
### Fixed
- When checking if a model has already been benchmarked, we only care about the
  `few_shot` parameter if the model is generative.


## [v9.1.1] - 2024-01-15
### Fixed
- Now adds a `generative` key to the logged results, to enable parsing few-shot
  evaluated models correctly when building leaderboards.


## [v9.1.0] - 2024-01-14
### Changed
- Now only stores the top-10 log probabilities of generated tokens when the generation
  length is less than 8 tokens. Also now keeps separate caches for each (model,
  dataset) combination, where it previously had a single cache for each model. Both of
  these help reduce the memory usage of the model output cache.
- Optimised cache saving/loading a bit, making the waiting time in between iterations
  slightly shorter.
- Removes the model output cache for a (model, dataset) combination when the
  benchmarking of the model on the dataset finishes successfully. Also removed indents
  in model output cache JSON files. Both of these help reducing the disk space used on
  caching.

### Fixed
- Only require generative models to output logprobs if the dataset is of a task that
  requires it. This caused the benchmarking to use excessive memory when benchmarking
  datasets that require long generative outputs, such as NER.

### Removed
- Removed some vLLM logging.


## [v9.0.0] - 2024-01-12
### Added
- Now caches the completions of open source generative models, which effectively makes
  benchmarking of these ~33% faster. We cannot store all logits for storage reasons (it
  quickly gets >100GB in that case), so we instead store the top-100 logits for each
  generated token, but only if the generated sequence is shorter than 50 tokens. We
  thus assume that (a) these are the only logits needed, and (b) that the generations
  don't change. We argue that (a) is the case since we only use the logits in
  classification tasks, in which case we only use the first token anyway. Further,
  since we're using a temperature of 0 anyway, the generations will be as close to
  deterministic as possible (up to small rounding fluctuations of logits, which is
  negligible). This is a breaking change, since it is not compatible with the previous
  way we cached OpenAI model outputs.
- Added a new `--clear-model-cache` flag, which removes the cached models after
  finishing the benchmarking of each model, to save disk space. This doesn't remove the
  cached model outputs or datasets.
- Added the following new datasets:
    - `fone`, a Faroese NER dataset, which replaces the previous `wikiann-fo` dataset.
    - `dansk`, a Danish NER dataset, which replaces the previous `dane` dataset.
    - `norquad`, a Norwegian question answering dataset, which replaces the previous
      `scandiqa-no` dataset.
    - Danish, Swedish, German and Dutch versions of the MMLU, ARC and HellaSwag
      datasets, testing knowledge and common sense reasoning of generative models.
      These have been machine translated by the University of Oregon using
      GPT-3.5-turbo. Machine translation is not adequate, of course, so see this as a
      first version of these kinds of evaluations, to get some benchmarks going asap.
    - `squad-nl`, a Dutch extract question answering dataset, which is a machine
      translated version of SQuAD-v2. As with the datasets mentioned above, this is
      meant as a first version of a Dutch QA dataset, until we have a better one
      available.
- Added `--only-validation-split` flag, which only benchmarks the model on the
  validation split, which is 5-10x smaller than the test split (depending on the
  dataset). This is especially useful with paid models like OpenAI models. The value of
  this flag is stored in the benchmark results, so this will be visible on
  leaderboards.
- Now uses vLLM as the underlying engine for few-shot evaluating generative models,
  which drastically improves the evaluation speed, as well as requiring less GPU
  memory.

### Changed
- Now compatible with`transformers >= 4.36.2`, and this is required now as they have
  changed their generation API in a breaking manner.
- Now removes all newlines from texts in the summarization task, where previously these
  were merely "squashed" to single newlines. This makes the separation of few-shot
  examples for generative models easier.
- Also removes newlines from the NER task, where these were not removed at all
  previously.
- Now doesn't force ASCII characters in the NER task for generative models, making the
  target JSON dictionary more consistent with the input text.
- If a model is stored in the Safetensors format on Hugging Face Hub, then we read out
  the number of parameters directly from those files. This results in more accurate
  parameter counts as opposed to loading in the model in 4-bit and counting manually.
- Samples with excessively short or long texts have been removed.
- Adjusted number of few-shot examples in datasets to ensure that the resulting prompt
  is at most ~3000 tokens long.
- When timeout errors occur when loading a model then we will try again at most 5 times
  now, where previously we would attempt to re-load it indefinitely.

### Fixed
- Removed `text2text-generation` temporarily from the tags defining generative models,
  since we do not support the benchmarking of these yet. This will be added back in as
  soon as we support them.
- Now catches `OSError`s when loading Hugging Face model configurations, which happen
  when there is no `config.json` file in the model repo.
- When sampling few-shot examples for question answering tasks we previously sampled
  among examples with context length less than 1024 characters, to keep the prompt
  short. This is too small for some datasets, so now we dynamically set this threshold
  based on the dataset itself, starting from 512 and doubling until we have at least
  the number of desired few-shot examples to choose from.
- Now only sets `torch_dtype` is CUDA is available, as otherwise errors are caused.
- Previously text generation in a batch would be stopped if any of the samples in the
  batch reached the stopping criteria, causing a lot of incomplete completions. Now
  the model continues to generate text until the entire batch is complete, and the
  excess generation is removed afterwards.
- When benchmarking encoder models on QA tasks the contexts are split up if they exceed
  the model's context length. The stride value used caused errors in rare cases where
  the model's maximum context length was really small (128). This has been fixed now.
- Now sets `ignore_mismatched_sizes` when loading models if the model cannot be loaded
  otherwise. This previously caused some issues when loading certain models.
- Fixed bug where some encoder models did not work properly when loaded in with FP16
  mixed precision due to overflow. We now load in models with BF16 as these have a
  larger range, but fall back to FP16 if BF16 is not available. If both lead to
  overflow then we attempt again with full FP32, and lastly throw an informative error
  and block evaluation if the overflow persists.
- When few-shot evaluating models on NER tasks, we are now more lenient towards the
  generated model output. Instead of taking the output as-is, we are now extracting the
  first dictionary (enclosed in curly brackets), as well as replacing all single
  apostrophes (') with double ones (").
- If a model is already pre-quantized then we will not attempt to quantize it as well.


## [v8.2.1] - 2023-12-20
### Fixed
- Removed the non-existent IsReC, FoReC and FoQA datasets.


## [v8.2.0] - 2023-12-20
### Added
- Added the following new datasets:
    - `sb10k`, a German sentiment classification dataset.
    - `dutch-social`, a Dutch sentiment classification dataset.
    - `sst5`, an English sentiment classification dataset.
    - `germeval`, a German NER dataset.
    - `conll-nl`, a Dutch NER dataset.
    - `conll-en`, an English NER dataset.
    - `scala-de`, a German linguistic acceptability dataset.
    - `scala-nl`, a Dutch linguistic acceptability dataset.
    - `scala-en`, an English linguistic acceptability dataset.
    - `nqii`, an Icelandic extractive question answering dataset.
    - `germanquad`, a German extractive question answering dataset.
    - `squad`, an English extractive question answering dataset.
    - `cnn-dailymail`, an English summarization dataset.

### Fixed
- Fixed bug with question answering benchmarking when the answer was a proper subset of
  the first token in the context, causing errors when benchmarking some models.
- Some models have been stored in mixed precision as well as containing an
  implementation of layer normalisation which is incompatible with such mixed
  precision. When loading models we now only load in mixed precision if `torch_dtype`
  has been specified in the Hugging Face model configuration (as with the Mistral
  model, for instance).
- When sampling examples to use in few-shot prompts in a sequence classification, we
  previously required that the samples are stratified with respect to the labels. This
  caused an issue if the dataset did not contain all labels, so now we only stratify
  with respect to the labels present in the dataset.
- When few-shot benchmarking on question answering datasets we previously only used the
  samples whose contexts were at most 512 characters long. This turns out to be too few
  for `germeval`, so this has been upped to 1024.


## [v8.1.0] - 2023-12-04
### Added
- Now added support for text-to-text tasks, which include tasks such as abstractive
  summarization, abstractive question-answering and translation. These can only be
  benchmarked with generative models. In this release, this includes the following
  datasets:
    - `nordjylland-news`, a Danish summarization dataset based on news articles.
    - `swedn`, a Swedish summarization dataset based on news articles.
    - `no-sammendrag`, a Norwegian summarization dataset based on news articles.
    - `rrn`, an Icelandic summarization dataset based on news articles.
    - `mlsum`, a German summarization dataset based on news articles.
    - `wiki-lingua-nl`, a Dutch summarization dataset based on WikiHow articles.
  These are all of the task `summarization`, meaning that they can also all be run
  using `scandeval --dataset-task summarization --model-id <model_id>`.
- A `--use-flash-attention` flag has been added, which enables Flash Attention 2.0,
  which is required by some models, such as Mistral-based ones. If `flash-attn` has not
  been installed then an informative error message will be raised. Thanks to @peter-sk
  for this contribution!

### Changed
- Now uses 8-bit AdamW whenever CUDA is available, as opposed to regular AdamW.
  Experiments shows that this does not affect benchmarking performance, but reduces
  memory usage and thus allows benchmarking of larger models

### Fixed
- A bug was removed which caused some overlap between the dataset splits of the
  ScandiQA datasets.
- Now allows loading in models in the data type that they were trained in, which
  previously caused errors if they weren't trained in float32.


## [v8.0.0] - 2023-11-29
### Added
- Support for few-shot evaluation of decoder models, both from the Hugging Face Hub and
  OpenAI models. This currently happens automatically when specifying a generative
  model from the Hugging Face Hub, and with all OpenAI models.
- Now stores model caches in separate directories, enabling parallel evaluations.
  Thanks to @KennethEnevoldsen for this contribution! :tada:
- Added `--device` argument to the CLI, which can be used to overwrite the automatic
  detection of device (CPU, CUDA GPU, MPS GPU, TPU) to use.
- Added `--trust-remote-code/--no-trust-remote-code` argument to the CLI, as some
  models require this flag to be loaded. It defaults to `False` for security reasons,
  however.
- Added `--load-in-4bit/--no-load-in-4bit` argument to the CLI, which can be used to
  overwrite the automatic 4bit loading of models. By default only generative models
  will be loaded in 4bit, and only if a CUDA GPU is available, as this is required by
  the underlying `bitsandbytes` package.
- Now manually adjusts the maximum sequence length of a model to ensure that the
  reported maximum length is correct.

### Changed
- Now only supports Python 3.10 and above.
- Changed the variation in the speed benchmark. Rather than using a fixed length
  document and computing iterations per second, it now uses varied length documents and
  computes tokens per second. This also has the added benefit of being able to better
  compare models with varying level of maximum sequence lengths. Further, it now uses
  GPU rather than CPU to accomodate 4-bit models, as these cannot be run on CPU.
- Changed the `--model-framework` argument to `--framework`.
- Changed the `--use-auth-token` and `--auth-token` arguments to `--use-token` and
  `--token`, reflecting the same change in the `transformers` package.
- Now reports all model parameters, rather than just the trainable ones.
- Now uses 8-bit AdamW optimizer when CUDA is available rather than the default AdamW,
  to save memory when working with larger models.

### Removed
- Previously generative models had their maximum sequence length altered by subtracting
  their padding token ID. This is not needed anymore and have been removed.

### Fixed
- Handles timeouts better now, when fetching models from the Hugging Face Hub. Instead
  of simply throwing the error, cancelling the benchmarking process, it simply tries
  again until the connection is up again.
- Some models output both logits and hidden states, which caused unnecessary
  out-of-memory issues. This is now handled using the `preprocess_logits_for_metrics`
  argument in `Trainer`.
- Now catches errors while loading model configurations.


## [v7.1.1] - 2023-07-01
### Fixed
- The feature names of the NER datasets have been changed, so the code have been
  updated to reflect this.


## [v7.1.0] - 2023-05-15
### Added
- Added support for the NorBERT3 models.


## [v7.0.0] - 2023-05-13
### Changed
- Now uses PyTorch 2.0, which (among other things) includes more control over the MPS.
  This means that MPS out of memory errors will now be caught and dealt with like CUDA
  out of memory errors, and we clear the MPS cache in between runs.

### Fixed
- Ensure that `type_vocab_size` is not changed if it was previously set to 0. This
  caused issues for some models when benchmarking question answering tasks.


## [v6.3.0] - 2023-04-12
### Added
- Now added support for benchmarking local models in the Hugging Face format (i.e.,
  saved with the `save_pretrained` method). This automatically detects the framework
  based on the file extension, but can also be set using the new `--model-framework`
  argument. Thanks to @peter-sk for implementing this! :tada:

### Fixed
- Now handles word-token alignment properly with SentencePiece tokenisers, which caused
  some models not being able to be benchmarked on token classification tasks.
- Now handles UNK tokens during word-token alignment, where it locates the word that is
  being tokenised into the UNK token, extracting the original value of the UNK token
  and replacing the token by that value.


## [v6.2.4] - 2023-03-10
### Fixed
- If the Hugging Face Hub is down, throwing a `HfHubHTTPError`, then catch it, wait 30
  seconds, and try again.
- Now always fixes the `model_max_length` attribute of the tokenizer, to prevent index
  errors during finetuning.

### Changed
- Changed `raise-error-on-invalid-model` to `raise-errors`. The flag now raises all
  errors instead of skipping the model evaluations, which can be used for debugging.


## [v6.2.3] - 2023-02-27
### Fixed
- Ensure that the `max_position_embeddings` fix from v6.2.2 only occurs if the
  tokenizer has a padding token, as this is used to set the `model_max_length`.
- If a model only has a JAX model but also has tags on the Hugging Face Hub from
  another framework, then re-try the evaluation with `from_flax` set to `True`.


## [v6.2.2] - 2023-02-25
### Fixed
- If `max_position_embeddings` is smaller than any of the context lengths specified in
  `model_max_length` and `max_model_input_sizes` then we use that as the the
  tokenization max length. This avoids dimension errors related to truncation.


## [v6.2.1] - 2023-02-22
### Fixed
- Now does not include models with the word "finetuned" in their name when benchmarking
  all models. These can still be benchmarked if specified directly.


## [v6.2.0] - 2023-01-09
### Changed
- Does not include by default models which indicate in their name that they're using
  more than a billion parameters, such as `EleutherAI/gpt-j-6B`.

### Fixed
- Now sets the default language for the (upcoming) XMOD models.
- If a model's `token_type_embeddings` layer has size (1, ...) when benchmarking the
  model for question answering, it is expanded to size (2, ...) with the second row
  being randomly initialised. This is required as question answering tasks need a least
  two token type embeddings.
- Now catches `OSError` when loading tokenizers.


## [v6.1.1] - 2023-01-02
### Fixed
- Fixed error where some tokenizers did not have special token IDs registered.
- Now catches `JSONDecodeError` when loading tokenizers.
- Now catches `KeyError` when loading model configurations.


## [v6.1.0] - 2022-12-29
### Added
- Added model inference speed estimation benchmark. This can now be run by setting
  either `task` or `dataset` to "speed". E.g., `scandeval -m <model_id> -d speed` or
  `scandeval -m <model_id> -dt speed`. This runs 10 iterations of 100 model inferences
  on a document of length 2,600 (the document "This is a dummy document. " repeated 100
  times). The inference speed includes tokenization, and is powered by the `pyinfer`
  package.


## [v6.0.1] - 2022-12-28
### Fixed
- Added prefix space to DeBERTa models.
- Now automatically changes a model's `type_vocab_size` to at least 2 when benchmarking
  the model on question-answering tasks. This previously caused an error when a model
  config had it set to 1.


## [v6.0.0] - 2022-12-24
### Added
- Added support for decoder models such as the GPT-series.
- Added new Swedish sentiment classification dataset, SweReC, which is not
  aspect-based, contrary to the previous ABSAbank-Imm dataset. This dataset is a
  three-way classification task into the classical `positive`, `neutral` and `negative`
  classes, thereby establishing uniformity between the sentiment classification
  datasets in the different languages. The dataset comes from reviews from both
  se.trustpilot.com and reco.se, and has been created by Kristoffer Svensson as part of
  his Bachelor thesis "Sentiment Analysis With Convolutional Neural Networks:
  Classifying sentiment in Swedish reviews".
- Added historic BERT models from `dbmdz` as part of the default multilingual list.
- Added the `--batch-size` argument, which can be used to manually select a batch size.
  Must be among 1, 2, 4, 8, 16 and 32.

### Removed
- As SweReC is a drop-in replacement for ABSAbank-Imm, the latter has been removed from
  the ScandEval benchmark.

### Fixed
- Now deals with an issue with DeBERTaV2 models where `pooler_hidden_size` has been set
  to a value different to `hidden_size` in its configuration, which made it impossible
  to do sequence classification with the model. The former is now forced to be the same
  as the latter, fixing the issue.
- Now ensures that tokenizers, model configurations and metrics are cached to the
  ScandEval cache, rather than the default Hugging Face cache.
- Previously, if a model's context length was greater than 1,000 it would be reduced to
  512, since an unset context length results in a very large `model_max_length` value
  of the tokenizer. This conflicted with longformer-style models whose context length
  _actually_ was greater than 1,000, so now this upper bound has been increased to
  100,000.
- Now includes `sacremoses` as a dependency, as this is required by some tokenizers.
- Converted the `id` column in ScandiQA to a string, to avoid integer overflow errors
  during preprocessing.
- If there is a `torch` operation which does not have a deterministic component, then a
  warning will be issued instead of raising an error.


## [v5.0.0] - 2022-11-03
### Added
- A new argument, `ignore_duplicates` (or `--ignore-duplicates/--no-ignore-duplicates`
  in the CLI) further ignores an evaluation if it has previously been evaluated. This
  argument defaults to `True`.
- Now stores the task and the dataset languages to the evaluation file with each
  evaluation.
- Now stores model metadata to the `scandeval_benchmark_results` file. Currently, this
  includes the number of trainable model parameters, the size of the model's vocabulary
  and the model's maximum sequence length.

### Changed
- Evaluation results are now saved in a JSONL file instead of a JSON file, and results
  are appended onto the file after every evaluation.
- You can now specify your Hugging Face authentication token in the `use_auth_token`
  argument of `Benchmarker` rather than manually logging in with `huggingface-cli
  login`. In the CLI an authentication token can also be applied directly using the new
  `--auth-token` argument. If an authentication is provided in this way in the CLI,
  then there is no need to add the `--use-auth-token` flag.
- The "random" models have now been renamed to "fresh", to emphasise that they are not
  random, but instead randomly initialized.
- The fresh models are now task independent, meaning that `fresh-xlmr-base` will now
  adapt to the task at hand, rather than having to benchmark, e.g.,
  `fresh-xlmr-base-sequence-clf` and `fresh-xlmr-base-token-clf` separately.

### Fixed
- ScandEval now works on TPUs.
- Removed `bf16` precision, as it only works for some GPUs.
- Should output less `transformers` logging now.
- Models were previously loaded in twice in the beginning of a benchmark. They are now
  only loaded in once (but re-loaded during each of the 10 iterations to ensure that we
  are starting from the same point).
- Changed the model architecture of the `fresh-xlmr-base` from `Roberta` to
  `XLMRoberta`.
- The `--dataset-task` is now correctly filtering the datasets benchmarked.
- Some tokenizers are not adding special tokens, despite them having registered them.
  These are now manually added, to ensure a proper evaluation of the models.

### Removed
- Removed support for evaluating finetuned models, as the package was primarily used to
  benchmark pretrained models anyway, and the change in datasets means that many
  finetuned models would have been trained on (part of) the test sets, resulting in
  artificially large scores. For evaluation of finetuned models, please check out the
  `aiai_eval` Python package instead.


## [v4.0.2] - 2022-07-22
### Fixed
- Now garbage collects properly, where previously (from v4 onwards) the `model` and
  `model_dict` were not removed from memory after each run, potentially causing a
  memory leak.

### Added
- Added the `HuggingFaceHubDown` and `NoInternetConnection` exceptions, to give more
  information to the user when benchmarking fails.
- Added unit tests.


## [v4.0.1] - 2022-07-14
### Fixed
- Removed temporary printing of scores for each iteration.


## [v4.0.0] - 2022-07-14
### Added
- Compatibility with Apple Silicon. If no CUDA GPU is available then MPS GPUs will
  automatically be used, if available.
- Added the datasets `scala-da`, `scala-sv`, `scala-nb`, `scala-nn`, `scala-is` and
  `scala-fo`. These are all linguistic acceptability datasets, being a binary text
  classification where a sentence has to be marked as grammatically correct or not.
- New randomly initialised ELECTRA-small model available for benchmarking, simply set
  `model-id` to either 'random-electra-small-sequence-clf or
  'random-electra-small-token-clf'. The randomly initialised XLM-RoBERTa-base model is
  still available by replacing 'electra-small' with 'xlmr-base'.
- Added `--raise-error-on-invalid-model` (`-r`) flag which raises an exception if an
  invalid model is specified. By default this is off, meaning that it simply skips the
  model if it is invalid.
- Added `--model-language` (`-ml`) and `--dataset-language` (`-dl`), which can be used
  to specify the model/dataset languages to benchmark. The `--language` (`-l`) argument
  will now be used for both models and datasets, where the `--model-language` and
  `--dataset-language` will override `--language` for models/datasets if specified.
- Added `--use-auth-token`, which is a flag that can be used when evaluating private
  models on Hugging Face Hub. This requires that the user has logged in via the
  `huggingface-cli login` command.
- Added scripts used to create all the datasets used in ScandEval, to ensure full
  transparency.

### Changed
- Models are now evaluated every 30 training steps (corresponding to having processed
  960 training samples) rather than every epoch. This decreases benchmarking time
  significantly, as early stopping kicks in earlier if the model is not learning
  anything.
- All training splits of datasets have been truncated to 1,024 samples. This has
  multiple benefits:
    - Faster benchmarking
    - More reliance on pretraining data
    - Enables consistent comparisons between different languages on the same task.
- Now uses `warmup_ratio` rather than `warmup_steps`, to ensure that 10% of the dataset
  is used to warm up the learning rate.
- All CLI arguments now use hyphens (`-`) rather than underscores (`_`). For instance,
  the `--model_id` argument has now been changed to `--model-id`.
- Text classification datasets are now using Matthew's correlation coefficient as
  metric, following the GLUE custom.
- Now requires PyTorch 1.12.0 or newer, to ensure compatibility with Apple Silicon.
- Renamed the `Benchmark` class to `Benchmarker`.

### Deprecated
- Deprecated support for evaluating finetuned models, as the package was primarily used to
  benchmark pretrained models anyway, and the change in datasets means that many
  finetuned models would have been trained on (part of) the test sets, resulting in
  artificially large scores. For evaluation of finetuned models, please check out the
  `aiai_eval` Python package instead (under development).

### Removed
- Removed support for Python 3.7, as this was incompatible with support for Apple
  Silicon.
- Removed the Danish sentiment analysis datasets `twitter-sent`, `europarl` and `lcc`,
  and instead using only the `angry-tweets` dataset for this task.
- Removed datasets `dkhate`, `nordial` and `dalaj`, to ensure a larger amount of
  benchmark uniformity across languages.
- Removed all part-of-speech datasets from the benchmark, as there was too little
  variance among the scores to differentiate models properly.
- Removed all dependency parsing datasets from the benchmark, both to focus more on the
  semantic tasks as that's closer to what is being used in practice, as well as to
  reduce the benchmarking time, as these datasets took way longer to benchmark than the
  others, due to the high number of labels.
- Removed the `load_dataset` function, as all datasets can now be found on the Hugging
  Face Hub and can thus be loaded using the `datasets` package. All the datasets can be
  found at `https://huggingface.com/ScandEval`.

### Fixed
- Now disables tokenizer progress bars properly, using the
  `datasets.utils.disable_progress_bar` function.
- Many of the datasets contained duplicate entries. These have now all been fixed.
- The `--model-id` now works as intended, where previously one was forced to use the
  shortcut `-m`.
- Now correctly determines whether a NER dataset contains `MISC` tags. Previously this
  required that both `B-MISC` and `I-MISC` tags were present in the dataset, where it
  has now been changed to at least one of them.


## [v3.0.0] - 2022-04-19
### Changed
- During finetuning, the i'th model will only be evaluated on the i'th
  bootstrapped dataset. This ensures that there will always be 10 scores, no
  matter if we're finetuning or purely evaluating, which means that the
  confidence intervals will be more comparable.

### Fixed
- Now sets `seed` in `TrainingArguments` rather than setting it explicitly in
  PyTorch. This has the added bonus of ensuring that the `DataLoader`s used
  during training also uses this seed, ensuring better reproducibility.
- Initialises model parameters with (fixed) different seeds during every
  iteration, to ensure variability and reproducibility.
- Explicitly uses the PyTorch implementation of `AdamW` now, rather than the
  (deprecated) `transformers` implementation.
- Fixed an error when a tokenizer has `max_model_input_sizes` set, but it being
  empty. In this case, the default truncation length is set to 512.


## [v2.3.2] - 2022-02-11
### Fixed
- Fixed a bug where a model's framework and pipeline tag were
  indistinguishable, as they are both using the same `tag-white` tag now.


## [v2.3.1] - 2022-02-11
### Fixed
- Changed the `tag-red`, which referred to the HTML class containing the model
  framework, to `tag-white`. This caused models to not be benchmarkable, as
  their framework could not be determined.


## [v2.3.0] - 2022-01-20
### Added
- Specific branches/commits/tags can now be benchmarked, using the `@`
  delimiter. For instance, `scandeval -m model_id@commit_hash` will benchmark
  the model with model ID `model_id`, stored at commit with hash `commit_hash`.
  Thanks to [@versae](https://github.com/versae) for contributing!


## [v2.2.0] - 2022-01-18
### Added
- Added more label synonyms for the DKHate dataset.


## [v2.1.0] - 2022-01-17
### Added
- Added support for `flax` models. Thanks to
  [@versae](https://github.com/versae) for contributing!


## [v2.0.0] - 2022-01-07
### Fixed
- Changed the anonymisation procedure for the tweet datasets `angry-tweets` and
  `twitter-sent`, now replacing user names by @USER and links by [LINK].


## [v1.5.9] - 2021-12-14
### Fixed
- Now removing all empty documents from datasets, as well as catching
  `KeyError` when trying to remove empty documents from dataset.


## [v1.5.8] - 2021-12-13
### Fixed
- Now explicitly removing empty tokenisations from the dataset.


## [v1.5.7] - 2021-12-10
### Fixed
- Now catching _all_ `CUDA error` exceptions and treating them as running out
  of memory. No harm done if this is not the case, however, as the script will
  simply decrease the batch size until it reaches 1, and if CUDA errors persist
  then it will skip that benchmark.


## [v1.5.6] - 2021-12-10
### Fixed
- When benchmarking a token classification dataset with a model whose tokenizer
  does not have a fast variant yet, this raised an error as the `word_ids`
  method of `BatchEncoding` objects only works when the tokenizer is fast. In
  that case these word IDs are now computed manually. This can currently handle
  WordPiece and SentencePiece prefixes (i.e., `##` and `▁`), and will raise an
  error if the manual alignment of words and tokens fail.
- Catch the CUDA error `CUDA error: CUBLAS_STATUS_ALLOC_FAILED`, which in this
  case is due to OOM.


## [v1.5.5] - 2021-12-08
### Fixed
- Deal with CUDA OOM errors when they occur on a replica, when multiple cores
  are used.


## [v1.5.4] - 2021-12-08
### Fixed
- Remove reference to `trainer` when CUDA OOM error is dealt with.


## [v1.5.3] - 2021-12-08
### Fixed
- Only try to to merge the `id2label` and `label2id` conversions if the model
  is finetuned. This caused some errors when a model was not finetuned but
  somehow still had conversion dictionaries.


## [v1.5.2] - 2021-12-08
### Fixed
- Deal with models with tasks `feature-extraction` or `sentence-similarity` as
  if they were `fill-mask`, meaning assume that they are merely pretrained
  models, rather than finetuned.


## [v1.5.1] - 2021-11-27
### Fixed
- Fixed bug when evaluating a finetuned model.


## [v1.5.0] - 2021-11-26
### Changed
- Added progress bar description when evaluating models without finetuning them
  first.
- Lowered the package requirements to the earliest possible versions.

### Removed
- Removed support for TensorFlow and Jax models, due to them not working
  properly anyway. They might be included at a later point, properly.


## [v1.4.0] - 2021-11-25
### Changed
- Now also outputting aggregated metrics in the resulting
  `scandeval_benchmark_results.json` file. This `json` file now has keys
  `raw_metrics` and `total`, with `raw_metrics` containing the previous (raw)
  scores, and the value of the new `total` key has aggregated scores (means and
  standard errors).


## [v1.3.8] - 2021-11-25
### Changed
- All training/evaluation progress bars are now removed when they are finished,
  and the training progress bar has no total anymore, as it was misleading.


## [v1.3.7] - 2021-11-25
### Fixed
- Removed `transformers` logging during evaluation as well.


## [v1.3.6] - 2021-11-25
### Changed
- Now only updating the list of benchmarks in the `Benchmark` during
  initialisation, and also logs it. This should make subsequent calls to the
  `benchmark` method faster.

### Fixed
- Removed `transformers` logging properly.


## [v1.3.5] - 2021-11-23
### Fixed
- Set the number of warmup steps to be the intended one training set pass,
  where previously it was effectively 8x that amount, due to gradient
  accumulation.
- Added the NER label synonyms `OBJORG=ORG`, `LOCPRS=LOC`, `LOCORG=LOC` and
  `ORGPRS=ORG`.
- Explicitly added `numpy` to the `install_requires` list. This is normally not
  a problem, as it's a requirement for other required packages, but this
  depends on the order in which the requirements are installed. This avoids
  such errors caused by misordering the requirements.


## [v1.3.4] - 2021-11-11
### Fixed
- Indexing error during synonym setup of finetuned models.


## [v1.3.3] - 2021-11-11
### Fixed
- When a finetuned model has labels which are synonyms of each other, they are
  now properly treated as synonyms, where previously this caused the model to
  have misaligned `id2label` and `label2id` conversion dictionaries.


## [v1.3.2] - 2021-11-11
### Fixed
- Added the NER label synonyms `GPE_LOC=LOC`, `GPE_ORG=ORG`, `LOC/ORG=LOC`,
  `ORG/PRS=ORG`, `OBJ/ORG=ORG`, as Norwegian and Swedish models tend to use
  these.


## [v1.3.1] - 2021-11-11
### Fixed
- Fixed a bug in label synonyms when benchmarking a finetuned spaCy for NER.


## [v1.3.0] - 2021-11-11
### Added
- Added label synonyms for NER benchmarking, which will enforce a more fair
  comparison of finetuned NER models, if the models have been trained on
  datasets with different labelling (e.g., `Person` instead of `PER`).


## [v1.2.1] - 2021-11-11
### Removed
- Properly removed the Icelandic WikiANN-IS data files. It was removed from the
  package, but the underlying files were still lying in the repository.


## [v1.2.0] - 2021-10-15
### Added
- Added the Icelandic NER dataset MIM-GOLD-NER. This can now be loaded as
  `mim-gold-ner` in the `Benchmark` class and through the CLI.

### Removed
- Removed the Icelandic WikiANN-IS dataset, as this has now been replaced by
  the MIM-GOLD-NER dataset.


## [v1.1.3] - 2021-10-04
### Fixed
- Added truncation and padding when tokenising token classification datasets.


## [v1.1.2] - 2021-09-27
### Fixed
- Missing dependency parsing tags.


## [v1.1.1] - 2021-09-27
### Fixed
- Reduce validation batch size if CUDA runs out of memory, rather than only
  reducing training batch size.


## [v1.1.0] - 2021-09-13
### Added
- Added Icelandic and Faroese translations of the Norwegian `NoReC` sentiment
  analysis dataset. These can be loaded as `norec-is` and `norec-fo`,
  respectively.

### Changed
- When loading datasets with `load_dataset`, the result is now four dataframes,
  rather than dictionaries. As the data can be accessed in the same way as with
  dictionaries, this maintains backwards compatibility.
- If a finetuned NER model has been trained on NER tags not present amongst the
  ones in the dataset, then these are either converted to `MISC` tags (if these
  are present in the dataset) and otherwise `O` tags. This will make the
  benchmarking of finetuned diverse NER models more fair.

### Fixed
- There was an error when a SpaCy model was benchmarked on a dataset that it
  was not trained on. It now raises an appropriate `InvalidBenchmark`
  exception, and will be skipped in the CLI and with the `Benchmark` class.


## [v1.0.2] - 2021-09-09
### Fixed
- Replaced abbreviations with spaces, such as "o s v" in the SDT corpus, with
  their proper version "o.s.v.".


## [v1.0.1] - 2021-09-09
### Fixed
- The URLs for the `wikiann-is` and `wikiann-fo` were wrong and have been
  corrected.


## [v1.0.0] - 2021-09-09
### Added
- Added the Icelandic and Faroese WikiANN datasets, for NER evaluation. They
  can be loaded as `wikiann-is` and `wikiann-fo` in the CLI and via the
  `Benchmark` class.
- Added the Icelandic and Faroese parts of the Universal Dependencies datasets,
  containing POS and dependency parsing tags. They can be loaded as `idt-pos`,
  `idt-dep`, `fdt-pos` and `fdt-dep`, respectively.


## [v0.17.0] - 2021-09-09
### Added
- Added the Dataset for Linguistic Acceptability Judgments (DaLaJ) dataset,
  which is here used as a binary classification dataset, in which sentences
  have to be classified as correct Swedish or not. It can be loaded as `dalaj`
  in the CLI and via the `Benchmark` class.
- Added the ABSAbank-Imm dataset, which is an aspect-based sentiment analysis
  dataset in Swedish, namely, the sentiment towards immigration. The original
  dataset featured a floating point score between 0 and 5, which has been
  reduced to a classifical three-way classification (`negative`, `neutral` and
  `positive`). It can be loaded as `absabank-imm` in the CLI and via the
  `Benchmark` class.
- Added the POS and dependency parsing parts of the Swedish Dependency Treebank
  (SDT). They can be loaded as `sdt-pos` and `sdt-dep` in the CLI and via the
  `Benchmark` class.
- Added the Stockholm-Umeå corpus 3.0 (SUC 3.0), a Swedish NER dataset. It can
  be loaded as `suc3` in the CLI and via the `Benchmark` class.
- Added abstract `NerBenchmark`, `PosBenchmark` and `DepBenchmark` classes, to
  ensure uniformity.

### Changed
- Uniformised all the NER datasets. They now all only have the NER tags `PER`,
  `LOC`, `ORG` and `MISC`.
- Uniformised all the dependency parsing datasets. They now all only have the
  main dependency parsing tags, without the subtags (so `acl:cleft` has been
  changed to `acl`, for instance).
- Changed the columns in all text classification datasets to `text` and
  `label`, to make it more uniform.


## [v0.16.0] - 2021-09-07
### Fixed
- Upped the number index tokens for dependency parsing from 100 to 512. This
  will need to be done better in the future, but is a fix for now.

### Added
- Added the random models `random-roberta-sequence-clf` and
  `random-roberta-token-clf` to the default list of model IDs when benchmarking
  all models.


## [v0.15.1] - 2021-09-03
### Fixed
- The list of dependency tags in the `ndt-nb-dep` and `ndt-nn-dep` were wrong.
  They have now been changed to all the tags occurring in the training sets.
- The `europarl_sent` data folder has now been renamed to `europarl`, so that
  it can be loaded correctly with `load_dataset`.


## [v0.15.0] - 2021-09-02
### Added
- Added the Bokmål and Nynorsk POS and DEP parts of the Norwegian Dependency
  Treebank dataset (NDT). They can be loaded as `ndt-nb-pos`, `ndt-nn-pos`,
  `ndt-nb-dep` and `ndt-nn-dep`, respectively, from the CLI and the `Benchmark`
  class.

### Removed
- Removed the `EuroparlSubj` and `TwitterSubj` datasets, as they were too easy
  and did not really differentiate models.
- Removed the abstract `SentimentClassificationBenchmark` and
  `BinaryClassificationBenchmark`, to simplify the classes. There is now only
  one `TextClassificationBenchmark`, which always evaluates with macro-F1.

### Changed
- Changed the name of `europarl-sent` to `europarl`, as `europarl-subj` now
  does not exist anymore.
- Changed the `nordial` dataset to the original 4-way classification dataset.


## [v0.14.1] - 2021-09-02
### Fixed
- Remove duplicate model IDs when calling the CLI or `Benchmark` class without
  any specified model IDs.


## [v0.14.0] - 2021-08-31
### Added
- Added the Bokmål and Nynorsk parts of the NorNE dataset, for named entity
  recognition. They can be loaded with the `norne-nb` and `norne-nn` names.
- There is now a `load_dataset` function, which can load any dataset, using the
  dataset's name (same name as in the CLI). For instance,
  `load_dataset('angry-tweets')` loads the `AngryTweets` dataset. This can be
  imported directly from the package: `from scandeval import load_dataset`. The
  individual dataset loading functions can still be imported as before; e.g.,
  `from scandeval.datasets import load_angry_tweets`.

### Changed
- Refactored folder structure with benchmarks and datasets.
- Separated `dane` and `dane-no-misc` into two distinct benchmark classes. The
  `dane-no-misc` can now also be loaded with the `load_dataset` function.


## [v0.13.0] - 2021-08-30
### Added
- Added the Norwegian Review Corpus (NoReC), a sentiment classification dataset
  in Norwegian.
- Added the Bokmål/Nynorsk part of the Norwegian Dialect dataset (NorDial), a
  binary classification dataset in Norwegian.

### Changed
- Changed the early stopping patience to `2 + 1000 // len(train)` from `2 + 250
  // len(train)`, to allow more patience (and thus, more stability), for
  smaller datasets.


## [v0.12.0] - 2021-08-26
### Changed
- Merged the `lcc1` and `lcc2` datasets into one `lcc` dataset, which is
  reasonable as they have been annotated by the same person. The `lcc2` dataset
  was too small to give reasonable benchmarking results.
- Renamed the `europarl2` dataset to `europarl_sent`

### Removed
- Removed the `europarl1` dataset, as it was too small to give reliable
  benchmarking results. This dataset could not simply be added to the
  `europarl2` dataset, as with the new `lcc` dataset, as the annotaters are not
  the same.

### Fixed
- If errors occur during benchmarking, then garbage collect before skipping to
  the next benchmark, to avoid memory issues.


## [v0.11.2] - 2021-08-25
### Fixed
- Issue with `model_max_length` in tokenizer meant that models with an ill-set
  value of `max_position_embeddings` could not be benchmarked. Now, if
  `model_max_length` is not set then the minimal value of the sizes in
  `max_model_input_sizes` will be used (which is usually 512).

### Changed
- Disabling CUDNN benchmark when using the `pytorch` framework, to enforce
  better reproducibility.


## [v0.11.1] - 2021-08-24
### Changed
- Rather than bootstrapping the training dataset and using the results to
  compute an estimator of the standard deviation, the same training dataset is
  trained on all ten times, and the mean of these along with a confidence
  interval is outputted.

### Fixed
- Updated the model metadata fetching to the new HTML structure of the
  HuggingFace Hub.
- A random seed is now set for all libraries, via the `transformers.set_seed`
  function.
- Always update the list of all the benchmarks when calling the
  `Benchmark.benchmark` method, to allow for possibility of setting new
  benchmark parameters after initialisation.


## [v0.11.0] - 2021-08-23
### Added
- The subjective/objective part of the `TwitterSent` and `Europarl2` datasets
  have now been added as binary classification tasks, called `TwitterSubj` and
  `EuroparlSubj`, respectively. These can now be benchmarked with the
  `Benchmark` class and the CLI using the `twitter-subj` and `europarl-subj`
  names, respectively.
- Added an abstract `BinaryClassificationBenchmark`, to streamline the binary
  classification benchmark datasets, which now includes the `DKHate`,
  `TwitterSubj` and `EuroparlSubj` datasets.


## [v0.10.1] - 2021-08-20
### Fixed
- Now catches `IndexError` during training.


## [v0.10.0] - 2021-08-20
### Fixed
- Properly filters by languages now via the `language` argument in the CLI and
  the `Benchmark` class. As HuggingFace Hub does not have a keyword for
  language, a search for language also means that any other non-language tag
  with that name also shows up in the results. These are now manually removed.
  This means it takes a few more seconds to compile the model list, but it will
  at least be accurate.
- In case `model_max_length` has not been set in a model configuration, it
  defaults to the value of `max_position_embeddings`. This fixes a problem with
  some models not being able to be trained on datasets whose texts were too
  long.
- Now handles the case where a non-classification model, such as a seq-to-seq
  model, are being benchmarked on a classification dataset.

### Added
- All the benchmark classes and `Benchmark` now has a `benchmark` method, which
  does the same as the `__call__` method. This is primarily so that it shows up
  in the Sphinx documentation.
- Added the default `LABEL_0` and `LABEL_1` label synonyms for `NOT` and `OFF`
  in the `DKHate` benchmark.
- Added the possibility of benchmarking randomly initialised RoBERTa models,
  using the model IDs `random-roberta-sequence-clf` and
  `random-roberta-token-clf`.


## [v0.9.0] - 2021-08-19
### Added
- Added the separate `nb` (Norwegian Bokmål) and `nn` (Norwegian Nynorsk)
  language tags, on top of the general `no` (Norwegian).
- Added more multilingual models.

### Fixed
- SpaCy models was evaluated wrongly on the `dane-no-misc` dataset, as their
  `MISC` predictions was not replaced with `O` tags.
- When evaluating models finetuned for token classification on a text
  classification task, a `ValueError` was raised, rather than an
  `InvalidBenchmark` exception.
- If none of the model's labels are among the dataset's labels, and are not
  even synonyms of them, then raise an `InvalidBenchmark`. This prevents things
  like evaluating a finetuned sentiment model on a NER task.
- When `evaluate_train` was `True`, this previously evaluated the test set
  instead.

### Changed
- Changed `Benchmark` API. Now the constructor and the `__call__` method have
  the same arguments, except the `model_id` and `dataset` in `__call__`, where
  the constructor sets the default values and the `__call__` method can change
  these to specific cases.
- Changed the benchmarking order. Now benchmarks all datasets for a model,
  before moving on to the next model
- Renamed the `multilabel` argument to the more descriptive `two_labels`.
- Updated docstrings to be more accurate.
- Early stopping patience is now set to `2 + 250 // len(train)`, so that
  smaller datasets can enjoy a bit more patience, but if the dataset contains
  at least 250 samples then it will remain at the current 2 patience.

### Removed
- Removed `learning_rate`, `batch_size`, `warmup_steps` and `num_finetunings`
  arguments from the benchmarks. These are now fixed to 2e-5, 32, 25% of the
  training dataset and 10, respectively. Note that the batch size will still
  automatically decrease if the GPU runs out of memory.


## [v0.8.0] - 2021-08-18
### Changed
- Models are now being trained for much longer, but with an early stopping
  callback with patience 2. This will enable a more uniform comparison between
  models that require a different number of finetuning epochs.

### Fixed
- There was a bug when evaluating a finetuned PyTorch model on a sequence
  classification task, if the model had only been trained on a proper subset of
  the labels present in the dataset.

### Removed
- All individual benchmarks have been removed from `__init__.py`. They can
  still be imported using their individual modules, for instance
  `from scandeval.dane import DaneBenchmark`, but the idea is to use the
  general `Benchmark` class instead.


## [v0.7.0] - 2021-08-17
### Changed
- Always ensure that a model can deal with the labels in the dataset when
  finetuning. If the model has not been trained on the label, then this will
  result in the model always getting that label wrong. For instance, this is
  the case for finetuned NER models not having been trained on MISC tags, if
  they are being evaluated on the DaNE dataset.

### Fixed
- Fixed bug when evaluating SpaCy models.
- Only removing objects at memory cleanup if they exist at all.


## [v0.6.0] - 2021-08-15
### Added
- When finetuning models, 10% of the training data is used to evaluate the
  models, which is used to choose the best performing model across all the
  epochs trained. This will allow for a more fair comparison, as some models
  degrade over time, while other models need a longer time to train.

### Changed
- Uniformised the `_log_metrics` method for all benchmarks, now only defined in
  `BaseBenchmark`.

### Fixed
- Garbage collects when downsizing batch size, to not keep all the previous
  models in memory.
- Typos in logging.


## [v0.5.2] - 2021-08-13
### Fixed
- Fixed bug when `evaluate_train` was set to False.


## [v0.5.1] - 2021-08-13
### Fixed
- The bootstrapping of the datasets is now done properly. Previously the
  bootstrapped datasets were not converted to HuggingFace Dataset objects.


## [v0.5.0] - 2021-08-12
### Added
- It is possible to only evaluate on the test sets, to save some time. This can
  be done in the `Benchmark` class using the `evaluate_train` argument, and in
  the CLI with the `--evaluate_train` flag.
- Added `progress_bar` argument to `Benchmark` to control whether progress bars
  should be shown, and added the `no_progress_bar` flag to the CLI for the same
  reason.

### Changed
- Updated `epochs` and `warmup_steps` of all the datasets to something more
  reasonable, enabling better comparisons of the finetuned models.
- Changed calculation of confidence intervals, which is now based on
  bootstrapping rather than the analytic approach. It will now evaluate ten
  times on the test set and compute a bootstrap estimate of the standard error,
  which is uses to compute an interval around the score on the entire test set.


## [v0.4.3] - 2021-08-12
### Fixed
- RuntimeErrors occuring during training will now raise an `InvalidBenchmark`
  exception, which means that the CLI and the `Benchmark` class will skip it.
  This is for instance caused when `max_length` has not been specified in the
  model config, meaning that the tokeniser does not know how much to truncate.


## [v0.4.2] - 2021-08-12
### Fixed
- Now catching the error where tokenisation is not possible, due to the model
  having been trained on a different task than what is present in the dataset.
  E.g., if a generator model is trained on a classification task.


## [v0.4.1] - 2021-08-12
### Fixed
- Now catching the error when the model's config does not align with the model
  class. When using the CLI or `Benchmark`, these will be skipped.


## [v0.4.0] - 2021-08-11
### Added
- Added confidence intervals for finetuned models, where there is a 95%
  likelihood that the true score would belong to the interval, given infinite
  data from the same distribution. In the case of "raw" pretrained models, this
  radius is added onto the existing interval, so that both the uncertainty in
  model initialisation as well as sample size of the validation dataset affects
  the size of the interval.
- Added garbage collection after each benchmark, which will (hopefully) prevent
  memory leaking when benchmarking several models.

### Changed
- New logo, including the Faroe Islands!
- Allow the possibility to include all languages and/or tasks in the CLI and
  the `Benchmark` class.
- Added Icelandic and Faroese to default list of languages in CLI and the
  `Benchmark` class.
- The default value for `task` is now all tasks, which also includes models
  that haven't been assigned any task on the HuggingFace Hub;
- If a model cannot be trained without running out of CUDA memory, even with a
  batch size of 1, then the model will be skipped in `Benchmark` and the CLI.

### Fixed
- New model is initialised if CUDA runs out of memory, to ensure that we are
  now continuing to train the previous model.
- Dependency parsing now implemented properly as two-label classification, with
  associated UAS and LAS metric computations. Works for pretrained SpaCy models
  as well as finetuning general language models.


## [v0.3.1] - 2021-08-10
### Fixed
- Reduces batch size if CUDA runs out of memory during evaluation.
- Loading of text classification datasets now working properly.


## [v0.3.0] - 2021-08-10
### Changed
- The `W036` warning message from SpaCy is no longer shown.

### Fixed
- Raise `InvalidBenchmark` if model cannot be loaded from the HuggingFace Hub.


## [v0.2.0] - 2021-08-09
### Added
- Added the part-of-speech tagging task from the Danish Dependency Treebank.
  Can be loaded with `load_ddt_pos` and used in `Benchmark` as `ddt-pos`.
- Added the dependency parsing task from the Danish Dependency Treebank.
  Can be loaded with `load_ddt_ddt` and used in `Benchmark` as `ddt-dep`.
- Documentation section and link to `README`
- The `Benchmark` class and the CLI now accepts a `batch_size` argument

### Changed
- `Benchmark` arguments `languages`, `tasks`, `model_ids` and `datasets` have
  been renamed to `language`, `task`, `model_id` and `dataset`, to keep it
  consistent with the CLI.
- When loading datasets, these will now be four dictionaries instead of lists,
  to allow for distinguishing features and labels.
- `batch_size` arguments can now only be among 1, 2, 4, 8, 16 and 32, and the
  corresponding gradient accumulation will be set to 32, 16, 8, 4, 2 and 1,
  respectively. This is to ensure that all finetuning is done using the same
  effective batch size, to ensure fair comparisons.
- Batch sizes are automatically halved if the GPU runs out of memory, with
  gradient accumulation correspondingly doubles.
- Evaluation of `SpaCy` models on token classification tasks are more accurate.

### Fixed
- `README` typos fixed, and image renders correctly


## [v0.1.0] - 2021-08-05
### Added
- First beta release
- Features Danish sentiment, hate speech detection and named entity
  recognition datasets for benchmarking
