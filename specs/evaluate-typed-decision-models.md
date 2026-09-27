# Evaluate zero-shot typed-decision models (e.g. Laya) as full models

Resolves EuroEval/EuroEval#2195. PR opened from the `mathiasesn/EuroEval` fork.

## Problem / Why

Models like [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)
are non-generative "decision" models: an encoder with trained heads that answers
typed questions (`choice`, `score`, `noul`) with calibrated probabilities. EuroEval
can't evaluate them as they ship:

- `HuggingFaceEncoderModel` needs a root `config.json` (Laya has none) and would
  fine-tune the model, which throws away the heads.
- The generative backends (vLLM, LiteLLM) need a model that generates text.

## Goals

- Evaluate the full model (encoder plus heads) zero-shot, loaded through the
  model's own package.
- Support sequence classification and multiple-choice classification tasks by
  turning each sample into a `choice` question over the candidate labels.
- Return the per-label probabilities as `GenerativeModelOutput.scores`, the same way
  `DummyModel` does, so the existing label extraction and metrics work unchanged.
- Skip unsupported task groups (token classification, QA, summarisation, …) cleanly.

## Non-goals

- Fine-tuning these models.
- Changing Laya's own package.
- Adding calibration metrics (could be a follow-up).
- Redesigning the frontend; only the minimum needed to show the new type (label/filter).

## Constraints

- The `laya` package (PyPI 0.3.6, Apache-2.0, needs torch, transformers ≥ 4.48,
  safetensors, huggingface_hub, numpy) goes in as an optional extra; importing
  EuroEval without it must still work (`NeedsExtraInstalled`).
- Follow existing module patterns: `BenchmarkModule` subclass in
  `src/euroeval/benchmark_modules/`, dispatch in `model_loading.load_model`, a new
  `InferenceBackend` value, with `DummyModel` as the template.
- `make check` passes (ruff, ty, funcsort, slopo); add tests in `tests/`, plus a
  CHANGELOG entry since `src/euroeval/` changes.

## Proposed approach

1. Generic interface: a `ZeroShotClassifierModel` benchmark module that handles
   everything task-related (building questions, mapping scores, skipping unsupported
   tasks, zero-shot validation) and hands off to a small adapter protocol,
   e.g. `ZeroShotClassifierAdapter` with `matches(model_id) -> bool`,
   `load(model_config)`, and
   `classify(texts, candidate_labels, instructions) -> list[dict[str, float]]`.
   Adapters live in `src/euroeval/zero_shot_adapters/` and are kept in a registry;
   `LayaAdapter` is the only one in this PR. Adding NLI or GLiClass adapters later
   needs no changes to the module.
2. New `InferenceBackend.ZERO_SHOT_CLASSIFIER` value; `model_exists` asks each
   registered adapter; `LayaAdapter.matches` detects Laya repos (e.g. the
   `rl_agent_config.json` file, or a `laya` tag/library on the Hub) so that
   `HuggingFaceEncoderModel` doesn't claim them first. Set `high_priority` if
   needed.
3. `generate()`: build a state from the sample text and a `choice` question whose
   instructions come from the dataset's prompt/instruction template and whose
   criteria are `prompt_label_mapping` values (with MC options for multiple-choice
   tasks). Map the returned probabilities to logprobs in `scores`.
4. Checkpoint choice: an explicit variant through the existing `#param` model ID
   syntax. `LayaAdapter.allowed_params` = `multilingual`, `typed-decisions`; no param
   means the root (English) checkpoint. Laya's `Router` is not used, so each result
   is exactly one checkpoint and `num_params` is well defined.
5. New `ModelType.ZERO_SHOT_CLASSIFIER` in `src/euroeval/enums.py`, with
   `generative_type = None`. Go through the ~34 `model_type` uses in `src/euroeval`
   so that encoder-only paths (fine-tuning, iterations) and decoder-only paths
   (few-shot, BPC, generation config) don't run for it. Store `model_type` in the
   result record so the leaderboard can tell it apart; currently
   `src/leaderboards/core_models.py::_classify_model` treats
   `generative_type is None` as ENCODER, which would mislabel Laya. Add
   `ZERO_SHOT_CLASSIFIER` to the leaderboard's `ModelType`/`SizeBucket`
   and to the Pareto categories, plus a minimal frontend label or filter.
6. Few-shot: zero-shot only; force/validate zero-shot like the existing zero-shot
   handling does.

## Acceptance criteria

- `euroeval --model convaiinnovations/laya --dataset <a Danish sentiment dataset>`
  runs end to end, with no fine-tuning, and records a result; the same holds for
  `convaiinnovations/laya#multilingual`, and an unknown variant raises a clear error.
- Multiple-choice datasets run; unsupported task groups are skipped with a clear
  message.
- Without the extra installed you get `NeedsExtraInstalled` with install instructions.
- A fake in-test adapter proves the interface works independently of Laya.
- Unit tests for detection, question building and score mapping, with Laya mocked
  so the tests don't need downloads.
- Leaderboard generation classifies these results as `zero_shot_classifier`, not
  `encoder`.
- `make check` is green; the PR from `mathiasesn/EuroEval` links #2195.

## Open questions

None.

## Risks

- The `laya` API is young (0.x) and may change; pin a minimum version.
- Prompt templates are written for decoders; the instruction text given to Laya
  may need its own wording per task.
- Detection could misfire on other repos with unusual layouts.
- Long MC contexts may exceed the 512-token context of the English checkpoint.
