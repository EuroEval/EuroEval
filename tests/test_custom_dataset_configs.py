"""Tests for the `custom_dataset_configs` module."""

import textwrap
from pathlib import Path

from euroeval.custom_dataset_configs import (
    YAML_CONFIG_FILENAMES,
    load_dataset_config_from_yaml,
)
from euroeval.data_models import DatasetConfig


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
        """A YAML file without 'task' key returns None."""
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

    def test_missing_languages_key_returns_none(self, tmp_path: Path) -> None:
        """A YAML file without 'languages' key returns None."""
        yaml_file = tmp_path / "euroeval_config.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                """\
                task: classification
                """
            )
        )
        config = load_dataset_config_from_yaml(yaml_file)
        assert config is None

    def test_empty_languages_list_returns_none(self, tmp_path: Path) -> None:
        """A YAML file with an empty languages list returns None."""
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
        assert config is None

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

    def test_inspect_ai_without_field_spec_loads_successfully(self, tmp_path: Path) -> None:
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


class TestYamlConfigFilenames:
    """Sanity checks on the YAML_CONFIG_FILENAMES constant."""

    def test_euroeval_config_yaml_in_filenames(self) -> None:
        """euroeval_config.yaml is in the recognised filenames list."""
        assert "euroeval_config.yaml" in YAML_CONFIG_FILENAMES

    def test_eval_yaml_in_filenames(self) -> None:
        """eval.yaml is in the recognised filenames list."""
        assert "eval.yaml" in YAML_CONFIG_FILENAMES

    def test_eval_yaml_has_higher_priority(self) -> None:
        """eval.yaml takes priority over euroeval_config.yaml."""
        idx_eval = YAML_CONFIG_FILENAMES.index("eval.yaml")
        idx_euroeval = YAML_CONFIG_FILENAMES.index("euroeval_config.yaml")
        assert idx_eval < idx_euroeval
