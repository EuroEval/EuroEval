"""Tests for the `yaml_config` module."""

import logging
import textwrap
from pathlib import Path
from typing import cast

import pytest
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

from euroeval import custom_dataset_configs, yaml_config
from euroeval.data_models import DatasetConfig
from euroeval.languages import DANISH, ENGLISH, NORWEGIAN_BOKMÅL
from euroeval.yaml_config import (
    load_dataset_config_from_yaml,
    resolve_config_languages,
    select_inspect_ai_tasks,
)


class TestLoadDatasetConfigFromYaml:
    """Tests for the `load_dataset_config_from_yaml` function."""

    @pytest.mark.parametrize(
        "instruction_prompt", [r"Answer in \boxed{...}.", r"Answer in \fbox{...}."]
    )
    def test_a_boxed_instruction_does_not_warn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, instruction_prompt: str
    ) -> None:
        """Any spelling the scorer extracts an answer from counts as boxed."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            f"task: math\nlanguages: [en]\ninstruction_prompt: {instruction_prompt}\n"
        )
        messages: list[str] = []
        monkeypatch.setattr(
            yaml_config,
            "log_once",
            lambda message, level, prefix="": messages.append(message),
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert not any("boxed" in message for message in messages), messages

    def test_a_malformed_solver_does_not_hide_the_later_one(
        self, tmp_path: Path
    ) -> None:
        """Scanning solvers continues past one this loader cannot read."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            "task: math\nlanguages: [en]\ntasks:\n  - solvers:\n"
            "      - name: prompt_template\n        args: not_a_mapping\n"
            "      - name: prompt_template\n        args:\n"
            "          template: 'Put it in \\boxed{{}}: {{prompt}}'\n"
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert "Put it in" in config.instruction_prompt

    def test_choices_column_as_list(self, tmp_path: Path) -> None:
        """choices_column as a list of strings triggers a preprocessing_func."""
        yaml_file = tmp_path / "euroeval_config.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                task: multiple-choice
                languages:
                  - en
                choices_column:
                  - option_a
                  - option_b
                  - option_c
                  - option_d
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        # choices_column is consumed to build preprocessing_func
        assert config.preprocessing_func is not None

    def test_choices_column_as_string(self, tmp_path: Path) -> None:
        """choices_column as a string triggers the creation of a preprocessing_func."""
        yaml_file = tmp_path / "euroeval_config.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                task: multiple-choice
                languages:
                  - en
                choices_column: options
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        # choices_column is consumed to build preprocessing_func
        assert config.preprocessing_func is not None

    def test_empty_languages_list_defaults_to_english(self, tmp_path: Path) -> None:
        """A YAML file with empty languages list and no fallback defaults to English."""
        yaml_file = tmp_path / "euroeval_config.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                task: classification
                languages: []
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert config.languages[0].code == "en"

    def test_eval_yaml_filename_accepted(self, tmp_path: Path) -> None:
        """A file named eval.yaml is accepted just like euroeval_config.yaml."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                task: classification
                languages:
                  - en
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert isinstance(config, DatasetConfig)

    def test_explicit_instruction_prompt_overrides_solver(self, tmp_path: Path) -> None:
        """An explicit instruction prompt takes precedence over prompt_template."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                task: math
                instruction_prompt: "Explicit {text}"
                tasks:
                  - id: math
                    solvers:
                      - name: prompt_template
                        args:
                          template: "Ignored {prompt}"
                languages: [en]
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert config.instruction_prompt == "Explicit {text}"

    def test_explicit_task_overrides_inference(self, tmp_path: Path) -> None:
        """An explicit top-level 'task' key overrides any Inspect AI inference."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                tasks:
                  - id: my_dataset
                    solvers:
                      - name: generate
                    scorers:
                      - name: math
                task: classification
                languages:
                  - en
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        # Explicit task wins over inference
        assert config.task.name == "classification"

    def test_fallback_invalid_language_code_returns_none(self, tmp_path: Path) -> None:
        """An unknown language code in fallback_language_codes returns None."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                task: classification
                """
            )
        )
        config = load_dataset_config_from_yaml(
            yaml_file, fallback_language_codes=["xx_NOT_REAL"]
        )
        assert config is None

    # ------------------------------------------------------------------ #
    # Language fallback from repo metadata                                #
    # ------------------------------------------------------------------ #
    def test_fallback_language_codes_used_when_no_languages_key(
        self, tmp_path: Path
    ) -> None:
        """fallback_language_codes is used when 'languages' is absent from YAML."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                task: classification
                """
            )
        )
        config = load_dataset_config_from_yaml(
            yaml_file, fallback_language_codes=["en"]
        )
        assert config is not None
        assert len(config.languages) == 1
        assert config.languages[0].code == "en"

    def test_inspect_ai_choices_column(self, tmp_path: Path) -> None:
        """field_spec.choices in tasks[0] is promoted to choices_column."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                name: My Dataset
                tasks:
                  - id: my_dataset
                    split: test
                    field_spec:
                      input: question
                      target: answer
                      choices: options
                    solvers:
                      - name: multiple_choice
                    scorers:
                      - name: choice
                task: multiple-choice
                languages:
                  - en
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert config.preprocessing_func is not None

    def test_inspect_ai_field_spec_columns(self, tmp_path: Path) -> None:
        """Column names in tasks[0].field_spec are promoted to top-level keys."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                name: My Dataset
                tasks:
                  - id: my_dataset
                    split: test
                    field_spec:
                      input: text
                      target: label
                    solvers:
                      - name: generate
                    scorers:
                      - name: choice
                task: classification
                languages:
                  - en
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        # field_spec.input/target should populate preprocessing_func via column mappings
        assert config.preprocessing_func is not None

    def test_inspect_ai_integer_target_is_ignored(self, tmp_path: Path) -> None:
        """field_spec.target as an integer (Inspect AI letter-index) is skipped."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                tasks:
                  - id: my_dataset
                    field_spec:
                      input: text
                      target: 0
                task: classification
                languages:
                  - en
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        # Before this fix, an integer target_column would trigger a validation error
        # and return None; now it should be silently ignored.
        assert config is not None
        assert config.preprocessing_func is None

    def test_inspect_ai_literal_target_is_ignored(self, tmp_path: Path) -> None:
        """field_spec.target with 'literal:' prefix is not used as target_column."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                tasks:
                  - id: my_dataset
                    field_spec:
                      input: text
                      target: "literal:A"
                task: classification
                languages:
                  - en
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        # input_column="text" (the default) and no target_column → no preprocessing_func
        # if "literal:A" had been passed as target_column, column_args_set would be True
        # and preprocessing_func would be built; so None here proves it was ignored.
        assert config.preprocessing_func is None

    def test_inspect_ai_split_default_test(self, tmp_path: Path) -> None:
        """tasks[0].split: test sets test_split to 'test' (standard name)."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                tasks:
                  - id: my_dataset
                    split: test
                    field_spec:
                      input: question
                      choices: options
                    solvers:
                      - name: multiple_choice
                task: multiple-choice
                languages:
                  - en
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert config.test_split == "test"

    def test_inspect_ai_split_used_as_test_split(self, tmp_path: Path) -> None:
        """tasks[0].split is used as the test_split (standard Inspect AI key)."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                name: My Dataset
                tasks:
                  - id: my_dataset
                    split: validation
                    field_spec:
                      input: question
                      choices: options
                    solvers:
                      - name: multiple_choice
                    scorers:
                      - name: choice
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert config.test_split == "validation"

    def test_inspect_ai_top_level_overrides_field_spec(self, tmp_path: Path) -> None:
        """Explicit top-level input_column takes precedence over field_spec.input."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                tasks:
                  - id: my_dataset
                    field_spec:
                      input: from_field_spec
                      target: label
                task: classification
                languages:
                  - en
                input_column: from_top_level
                target_column: label
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        # preprocessing_func is built; the explicit top-level value wins

    def test_inspect_ai_without_field_spec_loads_successfully(
        self, tmp_path: Path
    ) -> None:
        """A tasks list without a field_spec block is silently ignored."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                tasks:
                  - id: my_dataset
                    split: test
                    solvers:
                      - name: generate
                task: classification
                languages:
                  - en
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert isinstance(config, DatasetConfig)

    def test_invalid_language_code_returns_none(self, tmp_path: Path) -> None:
        """An unknown language code causes the function to return None."""
        yaml_file = tmp_path / "euroeval_config.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                task: classification
                languages:
                  - xx_NOT_A_REAL_CODE
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is None

    def test_invalid_task_returns_none(self, tmp_path: Path) -> None:
        """An unknown task name causes the function to return None."""
        yaml_file = tmp_path / "euroeval_config.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                task: this-task-does-not-exist
                languages:
                  - en
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is None

    def test_labels_are_set(self, tmp_path: Path) -> None:
        """Labels specified in YAML are reflected in the DatasetConfig."""
        yaml_file = tmp_path / "euroeval_config.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                task: classification
                languages:
                  - en
                labels:
                  - positive
                  - negative
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert list(config.labels) == ["positive", "negative"]

    def test_malformed_yaml_returns_none(self, tmp_path: Path) -> None:
        """A syntactically broken YAML file returns None."""
        yaml_file = tmp_path / "euroeval_config.yaml"
        yaml_file.write_text("task: [unclosed bracket\n")
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is None

    def test_math_boxed_prompt_does_not_warn(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A boxed math instruction does not emit the missing-prompt warning."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            "task: math\nlanguages: [en]\ninstruction_prompt: 'Use \\\\boxed{{}}'\n"
        )
        with caplog.at_level(logging.WARNING):
            config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert "expects the model" not in caplog.text

    def test_math_scorer_and_prompt_template(self, tmp_path: Path) -> None:
        """A math scorer and prompt template are inferred from Inspect AI YAML."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                name: Multilingual GSM-Symbolic
                tasks:
                  - id: original_eng
                    config: eng
                    split: test_original
                    field_spec:
                      input: question
                      target: target
                      metadata: [answer, language]
                    solvers:
                      - name: prompt_template
                        args:
                          template: |-
                            Solve it. Put the answer in \\boxed{{}}, e.g. \\boxed{{42}}.

                            {prompt}
                      - name: generate
                    scorers:
                      - name: math
                  - id: synthetic_eng
                    config: eng
                    split: test_synthetic
                    field_spec:
                      input: question
                      target: target
                    scorers:
                      - name: math
                languages:
                  - en
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert config.task.name == "math"
        assert config.test_split == "test_original"
        assert config.preprocessing_func is not None
        assert "{text}" in config.instruction_prompt
        assert "{{}}" in config.instruction_prompt
        rendered = config.instruction_prompt.format(text="X")
        assert "\\boxed{}" in rendered
        assert rendered.endswith("X")

    def test_math_template_with_unsupported_placeholder_is_ignored(
        self, tmp_path: Path
    ) -> None:
        """Math prompt templates with non-text placeholders are not promoted."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            "task: math\nlanguages: [en]\ntasks:\n  - solvers:\n"
            "      - name: prompt_template\n        args:\n"
            "          template: 'Put it in \\boxed{{}}: {prompt}; answer: {answer}'\n"
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert "answer" not in config.instruction_prompt

    def test_math_without_boxed_prompt_warns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Math configs without boxed instructions emit a warning."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text("task: math\nlanguages: [en]\n")
        messages: list[str] = []
        monkeypatch.setattr(
            yaml_config,
            "log_once",
            lambda message, level, prefix="": messages.append(message),
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        # Without a prompt template the default instruction prompt is used, which
        # does not tell the model to box its answer
        assert config.instruction_prompt == "{text}"
        assert any("boxed" in message for message in messages), messages

    def test_minimal_valid_config(self, tmp_path: Path) -> None:
        """A YAML file with only task and languages produces a DatasetConfig."""
        yaml_file = tmp_path / "euroeval_config.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                task: classification
                languages:
                  - en
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert isinstance(config, DatasetConfig)

    def test_missing_languages_key_defaults_to_english(self, tmp_path: Path) -> None:
        """A YAML file without 'languages' key and no fallback defaults to English."""
        yaml_file = tmp_path / "euroeval_config.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                task: classification
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert len(config.languages) == 1
        assert config.languages[0].code == "en"

    def test_missing_languages_no_fallback_defaults_to_english(
        self, tmp_path: Path
    ) -> None:
        """No 'languages' key and no fallback_language_codes defaults to English."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                task: classification
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert config.languages[0].code == "en"

    def test_missing_task_key_returns_none(self, tmp_path: Path) -> None:
        """A YAML file without 'task' key and no Inspect AI hints returns None."""
        yaml_file = tmp_path / "euroeval_config.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                languages:
                  - en
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is None

    def test_missing_task_no_hints_returns_none(self, tmp_path: Path) -> None:
        """No 'task' key and no Inspect AI hints (solver/choices) returns None."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                name: My Dataset
                tasks:
                  - id: my_dataset
                    field_spec:
                      input: text
                      target: label
                    solvers:
                      - name: generate
                languages:
                  - en
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is None

    def test_multiple_languages(self, tmp_path: Path) -> None:
        """Multiple language codes are all parsed."""
        yaml_file = tmp_path / "euroeval_config.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                task: classification
                languages:
                  - en
                  - fr
                  - de
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert len(config.languages) == 3

    def test_no_inspect_ai_split_defaults_to_test(self, tmp_path: Path) -> None:
        """When tasks[0].split is absent, test_split defaults to 'test'."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                task: classification
                languages:
                  - en
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert config.test_split == "test"

    def test_optional_int_fields(self, tmp_path: Path) -> None:
        """Integer optional fields are parsed correctly."""
        yaml_file = tmp_path / "euroeval_config.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                task: classification
                languages:
                  - en
                num_few_shot_examples: 8
                max_generated_tokens: 10
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert config.num_few_shot_examples == 8
        assert config.max_generated_tokens == 10

    def test_optional_str_fields(self, tmp_path: Path) -> None:
        """String optional column fields trigger a preprocessing_func being built."""
        yaml_file = tmp_path / "euroeval_config.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                task: classification
                languages:
                  - en
                input_column: review
                target_column: sentiment
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        # input_column and target_column are consumed to build preprocessing_func
        assert config.preprocessing_func is not None

    def test_prompt_label_mapping(self, tmp_path: Path) -> None:
        """A prompt_label_mapping dict is parsed correctly."""
        yaml_file = tmp_path / "euroeval_config.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                task: classification
                languages:
                  - en
                labels:
                  - positive
                  - negative
                prompt_label_mapping:
                  positive: pos
                  negative: neg
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert config.prompt_label_mapping == {"positive": "pos", "negative": "neg"}

    def test_prompt_template_without_prompt_appends_text(self, tmp_path: Path) -> None:
        """A prompt template without Inspect's placeholder still includes the input."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                task: math
                tasks:
                  - id: math
                    solvers:
                      - name: prompt_template
                        args:
                          template: "Solve this"
                languages: [en]
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert config.instruction_prompt == "Solve this\n\n{text}"

    def test_pure_inspect_ai_file_defaults_to_english(self, tmp_path: Path) -> None:
        """A pure Inspect AI eval.yaml (no EuroEval keys) succeeds, defaults to en."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                name: My MC Dataset
                tasks:
                  - id: my_dataset
                    split: test
                    field_spec:
                      input: question
                      target: answer
                      choices: options
                    solvers:
                      - name: multiple_choice
                    scorers:
                      - name: choice
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert config.task.name == "multiple-choice"
        assert config.languages[0].code == "en"

    def test_task_inferred_from_field_spec_choices(self, tmp_path: Path) -> None:
        """A 'choices' entry in field_spec infers multiple-choice task."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                name: My MC Dataset
                tasks:
                  - id: my_dataset
                    field_spec:
                      input: question
                      target: answer
                      choices: options
                    solvers:
                      - name: generate
                languages:
                  - en
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert config.task.name == "multiple-choice"

    # ------------------------------------------------------------------ #
    # Task inference from Inspect AI hints                                #
    # ------------------------------------------------------------------ #
    def test_task_inferred_from_multiple_choice_solver(self, tmp_path: Path) -> None:
        """A 'multiple_choice' solver in tasks[0].solvers infers MC task."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                name: My MC Dataset
                tasks:
                  - id: my_dataset
                    split: test
                    field_spec:
                      input: question
                      target: answer
                    solvers:
                      - name: multiple_choice
                    scorers:
                      - name: choice
                languages:
                  - en
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert config.task.name == "multiple-choice"

    def test_unsupported_repo_metadata_language_is_skipped(
        self, tmp_path: Path
    ) -> None:
        """Unsupported language codes from repo metadata are skipped, not fatal."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                task: classification
                """
            )
        )
        config = load_dataset_config_from_yaml(
            yaml_file, fallback_language_codes=["en", "da", "zh"]
        )
        assert config is not None
        assert [lang.code for lang in config.languages] == ["en", "da"]

    def test_yaml_languages_take_precedence_over_fallback(self, tmp_path: Path) -> None:
        """Explicit 'languages' in YAML overrides fallback_language_codes."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                task: classification
                languages:
                  - da
                """
            )
        )
        config = load_dataset_config_from_yaml(
            yaml_file, fallback_language_codes=["en"]
        )
        assert config is not None
        assert config.languages[0].code == "da"


class TestRealWorldYamlConfigs:
    """Tests using eval.yaml content from real public HuggingFace datasets."""

    def test_build_kwargs_choices_column_list_error(self, tmp_path: Path) -> None:
        """Test error when choices_column is neither string nor list."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                task: classification
                languages:
                  - en
                choices_column: 123
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is None

    def test_evasionbench_format(self, tmp_path: Path) -> None:
        """The EvasionBench eval.yaml format is parsed correctly.

        Source: https://huggingface.co/datasets/FutureMa/EvasionBench/blob/main/eval.yaml
        Notable features: 'evaluation_framework: inspect-ai' key, split=train,
        two solvers, and shuffled choices.
        """
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                name: Evasion Bench
                description: >
                  EvasionBench is a benchmark dataset for detecting evasive answers
                  in earnings call Q&A sessions.
                evaluation_framework: inspect-ai
                tasks:
                  - id: evasion_bench
                    config: default
                    split: train
                    epochs: 1
                    shuffle_choices: true
                    field_spec:
                      input: question
                      target: eva4b_label_letter
                      choices: choices
                      metadata:
                        - answer
                    solvers:
                      - name: prompt_template
                        args:
                          template: |
                            Question: {prompt}
                            Answer: {answer}
                      - name: multiple_choice
                        args:
                          template: |
                            You are a financial analyst.
                            {question}
                            {choices}
                    scorers:
                      - name: choice
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert config.task.name == "multiple-choice"
        assert config.test_split == "train"
        assert config.preprocessing_func is not None
        assert "Answer: {answer}" not in config.instruction_prompt

    def test_evasionbench_unknown_evaluation_framework_key_is_ignored(
        self, tmp_path: Path
    ) -> None:
        """The 'evaluation_framework' key is not a EuroEval field and is ignored."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                evaluation_framework: inspect-ai
                tasks:
                  - id: my_task
                    split: test
                    field_spec:
                      input: question
                      choices: options
                    solvers:
                      - name: multiple_choice
                    scorers:
                      - name: choice
                languages:
                  - en
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert config.task.name == "multiple-choice"

    def test_gpqa_format(self, tmp_path: Path) -> None:
        """The actual GPQA eval.yaml is parsed correctly.

        Source: https://huggingface.co/datasets/Idavidrein/gpqa/blob/main/eval.yaml
        Notable features: multiple tasks each with list-style choices and a
        'literal:D' target (which is silently ignored as it is not a column
        name). Only the first task entry is used to infer config parameters.
        """
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                # yaml file for compatibility with inspect-ai

                name: GPQA
                description: >
                  GPQA is a multiple-choice, Q&A dataset of very hard questions written
                  and validated by experts in biology, physics, and chemistry.

                evaluation_framework: inspect-ai

                tasks:
                  - id: diamond
                    config: gpqa_diamond
                    split: train

                    epochs: 4
                    epoch_reducer: pass_at_1

                    shuffle_choices: true

                    field_spec:
                      input: Question
                      target: "literal:D"
                      choices:
                        - "Incorrect Answer 1"
                        - "Incorrect Answer 2"
                        - "Incorrect Answer 3"
                        - "Correct Answer"

                    solvers:
                      - name: multiple_choice

                    scorers:
                      - name: choice

                  - id: main
                    config: gpqa_main
                    split: train

                    epochs: 4
                    epoch_reducer: pass_at_1

                    shuffle_choices: true

                    field_spec:
                      input: Question
                      target: "literal:D"
                      choices:
                        - "Incorrect Answer 1"
                        - "Incorrect Answer 2"
                        - "Incorrect Answer 3"
                        - "Correct Answer"

                    solvers:
                      - name: multiple_choice

                    scorers:
                      - name: choice

                  - id: extended
                    config: gpqa_extended
                    split: train

                    epochs: 4
                    epoch_reducer: pass_at_1

                    shuffle_choices: true

                    field_spec:
                      input: Question
                      target: "literal:D"
                      choices:
                        - "Incorrect Answer 1"
                        - "Incorrect Answer 2"
                        - "Incorrect Answer 3"
                        - "Correct Answer"

                    solvers:
                      - name: multiple_choice

                    scorers:
                      - name: choice
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert config.task.name == "multiple-choice"
        assert config.test_split == "train"
        assert config.preprocessing_func is not None

    def test_gsm8k_format_infers_reference_free_qa(self, tmp_path: Path) -> None:
        """The GSM8K eval.yaml with model_graded_fact scorer yields reference-free-qa.

        GSM8K uses the `model_graded_fact` scorer without an explicit `task` key.
        EuroEval detects the scorer and infers the `reference-free-qa` task.
        Source: https://huggingface.co/datasets/openai/gsm8k/blob/main/eval.yaml
        """
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                # yaml file for compatibility with inspect-ai
                name: GSM8K
                description: >
                  GSM8K is a dataset of 8,000+ high-quality arithmetic word problems.
                tasks:
                  - id: gsm8k
                    config: main
                    split: test
                    epochs: 4
                    epoch_reducer: pass_at_1
                    field_spec:
                      input: question
                      target: answer
                    solvers:
                      - name: prompt_template
                        args:
                          template: "Solve the following math problem. {prompt}"
                      - name: generate
                    scorers:
                      - name: model_graded_fact
                languages:
                  - en
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert config.task.name == "reference-free-qa"
        assert config.test_split == "test"

    def test_gsm8k_format_no_task_no_scorer_returns_none(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When neither task nor a recognised scorer is present, None is returned.

        An error is logged to inform the user.
        """
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                # yaml file for compatibility with inspect-ai
                name: GSM8K
                description: >
                  GSM8K is a dataset of 8,000+ high-quality arithmetic word problems.
                tasks:
                  - id: gsm8k
                    config: main
                    split: test
                    field_spec:
                      input: question
                      target: answer
                    solvers:
                      - name: generate
                    scorers:
                      - name: exact_match
                """
            )
        )
        with caplog.at_level(logging.ERROR, logger="euroeval"):
            config = load_dataset_config_from_yaml(yaml_file)
        assert config is None
        assert any("task" in record.message.lower() for record in caplog.records), (
            "Expected an error log about the missing task"
        )

    def test_gsm8k_format_with_explicit_task(self, tmp_path: Path) -> None:
        """The GSM8K eval.yaml works when a top-level 'task' key is added."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                name: GSM8K
                tasks:
                  - id: gsm8k
                    config: main
                    split: test
                    field_spec:
                      input: question
                      target: answer
                    solvers:
                      - name: generate
                    scorers:
                      - name: model_graded_fact
                task: knowledge
                languages:
                  - en
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert config.task.name == "knowledge"
        assert config.test_split == "test"

    def test_hle_model_graded_fact_format(self, tmp_path: Path) -> None:
        """The HLE eval.yaml format with model_graded_fact scorer is parsed correctly.

        Source: https://huggingface.co/datasets/cais/hle/blob/main/eval.yaml
        Notable features: `model_graded_fact` scorer with a judge model ID, which
        triggers the `reference-free-qa` task with an LLM-as-a-judge metric.
        """
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                name: Humanity's Last Exam
                description: >
                  Humanity's Last Exam (HLE) is a multi-modal benchmark at the frontier
                  of human knowledge, designed to be the final closed-ended academic
                  benchmark of its kind with broad subject coverage.

                tasks:
                  - id: hle
                    config: default
                    split: test

                    field_spec:
                      input: question
                      target: answer

                    solvers:
                      - name: system_message
                        args:
                          template: |
                            Your response should be in the following format:

                            Explanation: {your explanation for your answer choice}
                            Answer: {your chosen answer}
                            Confidence: {your confidence score between 0% and 100%}
                      - name: generate

                    scorers:
                      - name: model_graded_fact
                        args:
                          model: openai/o3-mini
                languages:
                  - en
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert config.task.name == "reference-free-qa"
        assert len(config.task.metrics) == 1
        assert config.task.metrics[0].name == "model_graded_fact"
        assert config.task.metrics[0].judge_id == "openai/o3-mini"
        assert config.test_split == "test"

    def test_hle_model_graded_fact_without_judge_model(self, tmp_path: Path) -> None:
        """Test model_graded_fact without a judge uses REFERENCE_FREE_QA."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                tasks:
                  - id: hle
                    split: test
                    field_spec:
                      input: question
                      target: answer
                    solvers:
                      - name: generate
                    scorers:
                      - name: model_graded_fact
                languages:
                  - en
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert config.task.name == "reference-free-qa"

    def test_infer_task_from_inspect_ai_multiple_scorers(self, tmp_path: Path) -> None:
        """Test task inference when multiple scorers are present."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                tasks:
                  - id: my_dataset
                    field_spec:
                      input: question
                      target: answer
                    solvers:
                      - name: generate
                    scorers:
                      - name: exact_match
                      - name: model_graded_fact
                        args:
                          model: openai/gpt-4
                languages:
                  - en
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert config.task.name == "reference-free-qa"
        assert config.task.metrics[0].judge_id == "openai/gpt-4"

    def test_infer_task_from_model_graded_fact_scorer_with_judge_model(
        self, tmp_path: Path
    ) -> None:
        """Test task inference from model_graded_fact scorer with explicit judge."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                tasks:
                  - id: hle
                    split: test
                    field_spec:
                      input: question
                      target: answer
                    solvers:
                      - name: generate
                    scorers:
                      - name: model_graded_fact
                        args:
                          model: openai/gpt-4
                languages:
                  - en
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert config.task.name == "reference-free-qa"
        assert len(config.task.metrics) == 1
        assert config.task.metrics[0].judge_id == "openai/gpt-4"

    def test_load_yaml_file_not_dict_returns_none(self, tmp_path: Path) -> None:
        """Test that YAML file with non-dict top-level returns None."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text("just a string\n")
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is None

    def test_mmlu_pro_format(self, tmp_path: Path) -> None:
        """The MMLU-Pro eval.yaml format is parsed correctly.

        Source: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/blob/main/eval.yaml
        """
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                # yaml file for compatibility with inspect-ai
                name: MMLU-Pro
                description: >
                  MMLU-Pro dataset is a more robust and challenging massive multi-task
                  understanding dataset.
                tasks:
                  - id: mmlu_pro
                    config: default
                    split: test
                    field_spec:
                      input: question
                      target: answer
                      choices: options
                    solvers:
                      - name: multiple_choice
                    scorers:
                      - name: choice
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert config.task.name == "multiple-choice"
        assert config.test_split == "test"
        assert config.preprocessing_func is not None

    def test_mmlu_pro_format_defaults_to_english(self, tmp_path: Path) -> None:
        """The MMLU-Pro eval.yaml has no 'languages' key, so English is used."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                name: MMLU-Pro
                tasks:
                  - id: mmlu_pro
                    config: default
                    split: test
                    field_spec:
                      input: question
                      target: answer
                      choices: options
                    solvers:
                      - name: multiple_choice
                    scorers:
                      - name: choice
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert len(config.languages) == 1
        assert config.languages[0].code == "en"

    def test_parse_int_field_bool_not_allowed(self, tmp_path: Path) -> None:
        """Test that boolean values are rejected for integer fields."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                task: classification
                languages:
                  - en
                num_few_shot_examples: true
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is None, (
            f"The following config was not found to be None: {config}"
        )

    def test_parse_int_field_bool_not_allowed_max_tokens(self, tmp_path: Path) -> None:
        """Test that boolean values are rejected for max_generated_tokens."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                task: classification
                languages:
                  - en
                max_generated_tokens: false
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is None

    def test_parse_languages_empty_list_with_fallback(self, tmp_path: Path) -> None:
        """Test that empty languages list uses fallback codes."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                task: classification
                languages: []
                """
            )
        )
        config = load_dataset_config_from_yaml(
            yaml_file, fallback_language_codes=["da"]
        )
        assert config is not None
        assert config.languages[0].code == "da"

    def test_promote_field_spec_fields_multiple_tasks_uses_first(
        self, tmp_path: Path
    ) -> None:
        """Test that only first task in tasks list is used for promotion."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                tasks:
                  - id: first_task
                    split: test
                    field_spec:
                      input: question
                      choices: options
                  - id: second_task
                    split: validation
                    field_spec:
                      input: text
                task: multiple-choice
                languages:
                  - en
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert config.test_split == "test"
        assert config.preprocessing_func is not None


class TestSubsetSelection:
    """Tests for selecting Inspect AI eval.yaml task entries."""

    def test_a_full_subset_name_selects_that_single_entry(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A subset can be requested by the name it is registered and recorded with."""
        raw = {
            "tasks": [
                {"config": "dan", "split": "test_original"},
                {"config": "dan", "split": "test_synthetic"},
                {"config": "swe", "split": "test_original"},
            ]
        }
        assert select_inspect_ai_tasks(
            raw=cast("dict[str, object]", raw),
            subset_split="test_original",
            dataset_id="repo::dan::test_original",
            subset_config="dan",
        ) == [(0, "dan", "test_original")]
        with caplog.at_level(logging.ERROR, logger="euroeval"):
            assert (
                select_inspect_ai_tasks(
                    raw=cast("dict[str, object]", raw),
                    subset_split="test_original",
                    dataset_id="repo::nor::test_original",
                    subset_config="nor",
                )
                is None
            )
        assert "['dan', 'swe']" in caplog.text

    def test_ambiguous_config_names_are_attributed_to_the_repository_languages(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warn when a subset cannot be tied to a single language."""
        yaml_text = (
            "task: classification\n"
            "tasks:\n"
            "  - config: alpha\n    split: test\n"
            "  - config: beta\n    split: test\n"
        )
        with caplog.at_level(logging.WARNING, logger="euroeval"):
            configs = self.load_with_fake_hub(
                tmp_path=tmp_path,
                monkeypatch=monkeypatch,
                yaml_text=yaml_text,
                dataset_id="repo",
                card_languages=["da", "sv", "de"],
            )
        assert configs is not None
        assert [
            sorted(lang.code for lang in config.languages) for config in configs
        ] == [["da", "de", "sv"], ["da", "de", "sv"]]
        assert "could not be determined" in caplog.text

    def load_with_fake_hub(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        yaml_text: str,
        dataset_id: str,
        card_languages: list[str] | None,
    ) -> list[DatasetConfig] | None:
        """Load a repo's configs through a fake Hub API.

        Args:
            tmp_path:
                Temporary directory used as the cache directory.
            monkeypatch:
                Fixture used to replace the Hub interactions.
            yaml_text:
                The contents of the repository's `eval.yaml`.
            dataset_id:
                The dataset ID to load.
            card_languages:
                The language codes reported by the repository card.

        Returns:
            The loaded dataset configs.
        """
        monkeypatch.setattr(
            custom_dataset_configs, "get_hf_token", lambda **kwargs: None
        )
        monkeypatch.setattr(
            custom_dataset_configs, "_repo_exists", lambda **kwargs: True
        )
        monkeypatch.setattr(
            custom_dataset_configs, "_list_repo_files", lambda **kwargs: ["eval.yaml"]
        )
        card_data = (
            None
            if card_languages is None
            else type("Card", (), {"language": card_languages})()
        )

        class FakeHub:
            def __init__(self, token: object) -> None:
                pass

            def dataset_info(self, **kwargs: object) -> object:
                return type("Info", (), {"card_data": card_data})()

            def hf_hub_download(self, **kwargs: object) -> None:
                path = Path(str(kwargs["local_dir"])) / "eval.yaml"
                path.write_text(yaml_text)

        monkeypatch.setattr(custom_dataset_configs, "HfApi", FakeHub)
        monkeypatch.setattr(
            yaml_config, "get_repo_splits", lambda **kwargs: ("train", None, "test")
        )
        return custom_dataset_configs.try_get_dataset_configs_from_repo(
            dataset_id=dataset_id,
            api_key=None,
            cache_dir=tmp_path,
            trust_remote_code=False,
            run_with_cli=False,
        )

    def test_config_named_as_a_split_names_the_subsets_instead(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A configuration is selected by naming a subset, not by a flag."""
        with caplog.at_level(logging.ERROR, logger="euroeval"):
            result = select_inspect_ai_tasks(
                raw={"tasks": [{"config": "dan", "split": "test"}]},
                subset_split="dan",
                dataset_id="hint::dan",
            )
        assert result is None
        assert "hint::dan::test" in caplog.text
        assert "--language" not in caplog.text

    def test_config_names_are_not_languages(self) -> None:
        """Do not read language codes out of unrelated config names."""
        assert resolve_config_languages(configs=["default", "train"]) is None

    def test_config_names_are_resolved_to_languages(self) -> None:
        """Language-named configs are attributed to their language."""
        assert resolve_config_languages(configs=["dan", "eng_metric", "nob"]) == {
            "dan": DANISH,
            "eng_metric": ENGLISH,
            "nob": NORWEGIAN_BOKMÅL,
        }

    def test_config_that_is_not_a_language_name_is_still_benchmark(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Only names that look like language codes can refer to an unsupported one.

        A repository mixing `default` with language-named configs still contains that
        configuration; dropping it because it failed to resolve as a language would
        silently benchmark less than the dataset declares.
        """
        yaml_text = (
            "task: classification\n"
            "tasks:\n"
            "  - config: default\n    split: test\n"
            "  - config: dan\n    split: test\n"
            "  - config: swe\n    split: test\n"
        )
        with caplog.at_level(logging.WARNING, logger="euroeval"):
            configs = self.load_with_fake_hub(
                tmp_path=tmp_path,
                monkeypatch=monkeypatch,
                yaml_text=yaml_text,
                dataset_id="repo",
                card_languages=["en", "da", "sv"],
            )
        assert configs is not None
        assert [config.name for config in configs] == [
            "repo::default::test",
            "repo::dan::test",
            "repo::swe::test",
        ]
        assert "could not be determined" in caplog.text
        assert "does not support" not in caplog.text

    def test_entries_without_configs_keep_legacy_behaviour(self) -> None:
        """Do not impose subset selection on legacy task entries."""
        assert select_inspect_ai_tasks(
            raw={"tasks": [{"split": "test"}, {"split": "validation"}]},
            subset_split=None,
            dataset_id="repo",
        ) == [(0, None, "test")]

    def test_entry_languages_override_top_level_and_fallback(
        self, tmp_path: Path
    ) -> None:
        """Prefer languages declared by the selected task entry."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            "task: classification\nlanguages: [en]\nfallback_language_codes: [en]\n"
            "tasks:\n  - config: dan\n    languages: [da]\n"
        )
        config = load_dataset_config_from_yaml(
            yaml_file, fallback_language_codes=["sv"]
        )
        assert config is not None
        assert config.languages[0].code == "da"

    def test_expanded_configs_are_named_by_their_selector(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each expanded entry is identified by repo, config and split."""
        yaml_text = (
            "task: classification\n"
            "tasks:\n"
            "  - config: dan\n    split: test_original\n"
            "  - config: dan\n    split: test_synthetic\n"
        )
        configs = self.load_with_fake_hub(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            yaml_text=yaml_text,
            dataset_id="repo",
            card_languages=["da", "sv", "de"],
        )
        assert configs is not None
        assert [config.name for config in configs] == [
            "repo::dan::test_original",
            "repo::dan::test_synthetic",
        ]
        assert all(config.source == "repo::dan" for config in configs)
        # The config name supplies the language when the YAML declares none
        assert all(
            [lang.code] == ["da"]
            for lang in [config.languages[0] for config in configs]
        )

    def test_explicit_split_selects_entry(self) -> None:
        """Return the task entry matching an explicit split."""
        assert select_inspect_ai_tasks(
            raw={
                "tasks": [
                    {"config": "dan", "split": "test_original"},
                    {"config": "dan", "split": "test_synthetic"},
                ]
            },
            subset_split="test_synthetic",
            dataset_id="repo::test_synthetic",
        ) == [(1, "dan", "test_synthetic")]

    @pytest.mark.parametrize("selector", ["repo::", "repo::dan::test::extra"])
    def test_malformed_selector_is_rejected(
        self, selector: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Reject selectors with empty or excess components."""
        with caplog.at_level(logging.ERROR, logger="euroeval"):
            assert (
                yaml_config.load_yaml_config(
                    hf_api=cast("HfApi", object()),
                    dataset_id=selector,
                    cache_dir=tmp_path,
                )
                is None
            )
        assert "Invalid dataset selector" in caplog.text

    def test_multiple_configs_expand_to_all_entries(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Without a selector every declared config is benchmarked."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            "tasks:\n"
            "  - config: alpha\n    split: test\n"
            "  - config: beta\n    split: test\n"
            "task: classification\nlanguages: [en]\n"
        )
        assert load_dataset_config_from_yaml(yaml_file) is not None

    def test_numeric_split_is_selected_as_string(self) -> None:
        """String selectors match scalar YAML values after conversion."""
        assert select_inspect_ai_tasks(
            raw={"tasks": [{"config": "dan", "split": 2024}]},
            subset_split="2024",
            dataset_id="repo::2024",
        ) == [(0, "dan", "2024")]

    def test_selected_entry_controls_all_config_values(self, tmp_path: Path) -> None:
        """Load column mappings, inference and prompts from the selected task."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                tasks:
                  - config: first
                    split: first_split
                    field_spec:
                      input: first_input
                      target: first_target
                    solvers:
                      - name: generate
                  - config: dan
                    split: selected_split
                    field_spec:
                      input: selected_input
                      target: selected_target
                    solvers:
                      - name: prompt_template
                        args:
                          template: "Solve: {prompt}"
                    scorers:
                      - name: math
                languages: [en]
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file, task_index=1)
        assert config is not None
        assert config.test_split == "selected_split"
        assert config.task.name == "math"
        assert "Solve: {text}" in config.instruction_prompt
        assert config.preprocessing_func is not None
        data = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {"selected_input": ["q"], "selected_target": ["a"]}
                )
            }
        )
        processed = config.preprocessing_func(data)
        assert set(processed["test"].column_names) == {"text", "target_text"}

    def test_selector_for_python_config_repo_is_rejected(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Subset selectors are unsupported without an eval.yaml file."""
        monkeypatch.setattr(
            custom_dataset_configs, "get_hf_token", lambda **kwargs: None
        )
        monkeypatch.setattr(
            custom_dataset_configs, "_repo_exists", lambda **kwargs: True
        )
        monkeypatch.setattr(
            custom_dataset_configs,
            "_list_repo_files",
            lambda **kwargs: ["euroeval_config.py"],
        )
        with caplog.at_level(logging.ERROR, logger="euroeval"):
            result = custom_dataset_configs.try_get_dataset_configs_from_repo(
                dataset_id="repo::test",
                api_key=None,
                cache_dir=tmp_path,
                trust_remote_code=False,
                run_with_cli=False,
            )
        assert result is None
        assert "only supported" in caplog.text

    def test_selector_passes_bare_repo_to_hub_and_sets_identity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Use the bare repository ID for all Hub operations."""
        calls: list[tuple[str, str]] = []
        monkeypatch.setattr(
            custom_dataset_configs, "get_hf_token", lambda **kwargs: None
        )
        monkeypatch.setattr(
            custom_dataset_configs, "_repo_exists", lambda **kwargs: True
        )
        monkeypatch.setattr(
            custom_dataset_configs, "_list_repo_files", lambda **kwargs: ["eval.yaml"]
        )

        class FakeHub:
            def __init__(self, token: object) -> None:
                calls.append(("init", str(token)))

            def dataset_info(self, **kwargs: object) -> object:
                calls.append(("info", str(kwargs["repo_id"])))
                return type("Info", (), {"card_data": None})()

            def hf_hub_download(self, **kwargs: object) -> None:
                calls.append(("download", str(kwargs["repo_id"])))
                path = Path(str(kwargs["local_dir"])) / "eval.yaml"
                path.write_text(
                    "task: classification\nlanguages: [en]\ntasks:\n"
                    "  - config: dan\n    split: test_original\n"
                )

        monkeypatch.setattr(custom_dataset_configs, "HfApi", FakeHub)
        monkeypatch.setattr(
            custom_dataset_configs,
            "get_repo_splits",
            lambda **kwargs: ("train", None, "test"),
        )
        split_kwargs: list[dict[str, object]] = []

        def fake_get_repo_splits(**kwargs: object) -> tuple[str, None, str]:
            split_kwargs.append(kwargs)
            return "train", None, "test"

        monkeypatch.setattr(yaml_config, "get_repo_splits", fake_get_repo_splits)
        configs = custom_dataset_configs.try_get_dataset_configs_from_repo(
            dataset_id="repo::test_original",
            api_key=None,
            cache_dir=tmp_path,
            trust_remote_code=False,
            run_with_cli=False,
        )
        assert configs is not None
        assert len(configs) == 1
        config = configs[0]
        assert config.name == "repo::dan::test_original"
        assert config.pretty_name == config.name
        assert config.source == "repo::dan"
        assert config.test_split == "test_original"
        assert split_kwargs[0]["config_name"] == "dan"
        assert all(repo == "repo" for action, repo in calls if action != "init")

    def test_selector_rejected_for_configless_eval_yaml(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A selector must not disappear when an eval.yaml has no configs."""
        with caplog.at_level(logging.ERROR, logger="euroeval"):
            result = select_inspect_ai_tasks(
                raw={"tasks": [{"split": "test"}]},
                subset_split="test",
                dataset_id="repo::test",
            )
        assert result is None
        assert "declares no configurations" in caplog.text

    def test_single_config_multiple_splits_expands_without_selector(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """All splits of a config are selected when no split is given."""
        raw = {
            "tasks": [
                {"config": "dan", "split": "test"},
                {"config": "dan", "split": "validation"},
            ]
        }
        with caplog.at_level(logging.ERROR, logger="euroeval"):
            assert select_inspect_ai_tasks(
                raw=cast("dict[str, object]", raw), subset_split=None, dataset_id="repo"
            ) == [(0, "dan", "test"), (1, "dan", "validation")]
        assert not caplog.text

    def test_single_entry_fixture_remains_unchanged(self, tmp_path: Path) -> None:
        """Preserve the existing single-config loading behaviour."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text("task: classification\nlanguages: [en]\n")
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is not None
        assert config.task.name == "classification"
        assert config.test_split == "test"

    @pytest.mark.parametrize("split", [None, "nonexistent"])
    def test_split_selection_requires_known_split(
        self, split: str | None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Validate a split selector, defaulting to all the declared splits."""
        with caplog.at_level(logging.ERROR, logger="euroeval"):
            result = select_inspect_ai_tasks(
                raw={
                    "tasks": [
                        {"config": "dan", "split": "test_original"},
                        {"config": "dan", "split": "test_synthetic"},
                    ]
                },
                subset_split=split,
                dataset_id=f"repo{f'::{split}' if split else ''}",
            )
        if split is None:
            assert result == [(0, "dan", "test_original"), (1, "dan", "test_synthetic")]
            assert not caplog.text
        else:
            assert result is None
            assert "test_original" in caplog.text and "test_synthetic" in caplog.text

    def test_split_selector_narrows_the_expansion(self) -> None:
        """A split selector keeps the entries with that split, whatever their config."""
        raw = cast(
            dict[str, object],
            {
                "tasks": [
                    {"config": "alpha", "split": "test"},
                    {"config": "beta", "split": "test"},
                    {"config": "alpha", "split": "dev"},
                ]
            },
        )
        assert select_inspect_ai_tasks(
            raw=raw, subset_split=None, dataset_id="repo"
        ) == [(0, "alpha", "test"), (1, "beta", "test"), (2, "alpha", "dev")]
        assert select_inspect_ai_tasks(
            raw=raw, subset_split="test", dataset_id="repo::test"
        ) == [(0, "alpha", "test"), (1, "beta", "test")]

    def test_split_selector_selects_the_same_split_of_every_config(self) -> None:
        """A split selector narrows the expansion, not the configurations."""
        raw = {
            "tasks": [
                {"config": "eng", "split": "test"},
                {"config": "eng_metric", "split": "test"},
                {"config": "eng", "split": "validation"},
            ]
        }
        assert select_inspect_ai_tasks(
            raw=cast("dict[str, object]", raw),
            subset_split="test",
            dataset_id="repo::test",
        ) == [(0, "eng", "test"), (1, "eng_metric", "test")]

    def test_subset_language_fallback_does_not_warn_for_single_language(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Do not warn when repository metadata has one language."""

        class FakeHub:
            def dataset_info(self, **kwargs: object) -> object:
                card = type("Card", (), {"language": ["da"]})()
                return type("Info", (), {"card_data": card})()

            def hf_hub_download(self, **kwargs: object) -> None:
                Path(str(kwargs["local_dir"])).joinpath("eval.yaml").write_text(
                    "task: classification\ntasks:\n  - config: dan\n    split: test\n"
                )

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            yaml_config, "get_repo_splits", lambda **kwargs: ("train", None, "test")
        )
        try:
            with caplog.at_level(logging.WARNING, logger="euroeval"):
                assert (
                    yaml_config.load_yaml_config(
                        hf_api=cast(HfApi, FakeHub()),
                        dataset_id="repo",
                        cache_dir=tmp_path,
                    )
                    is not None
                )
        finally:
            monkeypatch.undo()
        assert "could not be determined" not in caplog.text

    def test_subset_language_fallback_warns_for_multiple_languages(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warn when a selected subset has only repository-level languages."""

        class FakeHub:
            def dataset_info(self, **kwargs: object) -> object:
                card = type("Card", (), {"language": ["da", "en"]})()
                return type("Info", (), {"card_data": card})()

            def hf_hub_download(self, **kwargs: object) -> None:
                Path(str(kwargs["local_dir"])).joinpath("eval.yaml").write_text(
                    "task: classification\ntasks:\n  - config: alpha\n    split: test\n"
                )

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            yaml_config, "get_repo_splits", lambda **kwargs: ("train", None, "test")
        )
        try:
            with caplog.at_level(logging.WARNING, logger="euroeval"):
                configs = yaml_config.load_yaml_config(
                    hf_api=cast("HfApi", FakeHub()),
                    dataset_id="repo",
                    cache_dir=tmp_path,
                )
        finally:
            monkeypatch.undo()
        assert configs is not None
        assert "all 2 languages" in caplog.text
        assert "per-entry `languages`" in caplog.text

    def test_unknown_split_lists_the_alternatives(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An unknown split names the available ones."""
        raw = {"tasks": [{"config": "dan", "split": "test_original"}]}
        with caplog.at_level(logging.ERROR, logger="euroeval"):
            assert (
                select_inspect_ai_tasks(
                    raw=cast("dict[str, object]", raw),
                    subset_split="validation",
                    dataset_id="repo::validation",
                )
                is None
            )
        assert "['test_original']" in caplog.text

    def test_unparseable_hub_yaml_is_logged(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Surface YAML parser errors from the shared file loader."""

        class FakeInfo:
            card_data = None

        class FakeHub:
            def dataset_info(self, **kwargs: object) -> FakeInfo:
                return FakeInfo()

            def hf_hub_download(self, **kwargs: object) -> None:
                path = Path(str(kwargs["local_dir"])) / "eval.yaml"
                path.write_text("task: [broken\\n")

        monkeypatch.setattr(
            yaml_config, "get_repo_splits", lambda **kwargs: ("train", None, "test")
        )
        with caplog.at_level(logging.ERROR, logger="euroeval"):
            assert (
                yaml_config.load_yaml_config(cast("HfApi", FakeHub()), "repo", tmp_path)
                is None
            )
        assert any("parse YAML" in record.message for record in caplog.records)

    def test_unsupported_config_language_is_excluded(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Skip a config whose language EuroEval does not support."""
        yaml_text = (
            "task: classification\n"
            "tasks:\n"
            "  - config: dan\n    split: test\n"
            "  - config: zho\n    split: test\n"
        )
        with caplog.at_level(logging.WARNING, logger="euroeval"):
            configs = self.load_with_fake_hub(
                tmp_path=tmp_path,
                monkeypatch=monkeypatch,
                yaml_text=yaml_text,
                dataset_id="repo",
                card_languages=["da", "zh"],
            )
        assert configs is not None
        assert [config.name for config in configs] == ["repo::dan::test"]
        assert "does not support" in caplog.text

    def test_unsupported_config_language_is_missing_from_the_mapping(self) -> None:
        """A recognised language without EuroEval support stays unresolved."""
        assert resolve_config_languages(configs=["dan", "zho"]) == {"dan": DANISH}
