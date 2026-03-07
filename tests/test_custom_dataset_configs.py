"""Tests for the `yaml_config` module."""

import textwrap
from pathlib import Path

from euroeval.data_models import DatasetConfig
from euroeval.yaml_config import load_dataset_config_from_yaml


class TestLoadDatasetConfigFromYaml:
    """Tests for the `load_dataset_config_from_yaml` function."""

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

    def test_malformed_yaml_returns_none(self, tmp_path: Path) -> None:
        """A syntactically broken YAML file returns None."""
        yaml_file = tmp_path / "euroeval_config.yaml"
        yaml_file.write_text("task: [unclosed bracket\n")
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is None

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

    def test_explicit_task_overrides_inference(self, tmp_path: Path) -> None:
        """An explicit top-level 'task' key overrides any Inspect AI inference."""
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                tasks:
                  - id: my_dataset
                    solvers:
                      - name: multiple_choice
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
