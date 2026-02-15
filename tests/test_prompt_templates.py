"""Tests for the `prompt_templates` package."""


from euroeval.data_models import PromptConfig
from euroeval.languages import DANISH, ENGLISH, SWEDISH
from euroeval.prompt_templates import (
    CLASSIFICATION_TEMPLATES,
    EMPTY_TEMPLATES,
    LA_TEMPLATES,
    MULTIPLE_CHOICE_TEMPLATES,
    NER_TEMPLATES,
    RC_TEMPLATES,
    SENT_TEMPLATES,
    SIMPL_TEMPLATES,
    SUMM_TEMPLATES,
    TOKEN_CLASSIFICATION_TEMPLATES,
)


def test_classification_templates_structure() -> None:
    """Test that classification templates have expected structure."""
    assert isinstance(CLASSIFICATION_TEMPLATES, dict)
    assert len(CLASSIFICATION_TEMPLATES) > 0

    # Check a few languages are present
    assert ENGLISH in CLASSIFICATION_TEMPLATES
    assert DANISH in CLASSIFICATION_TEMPLATES

    # Check that values are PromptConfig objects
    for lang, config in CLASSIFICATION_TEMPLATES.items():
        assert isinstance(config, PromptConfig)
        assert hasattr(config, 'default_prompt_prefix')
        assert hasattr(config, 'default_prompt_template')


def test_sentiment_templates_structure() -> None:
    """Test that sentiment classification templates have expected structure."""
    assert isinstance(SENT_TEMPLATES, dict)
    assert len(SENT_TEMPLATES) > 0
    assert ENGLISH in SENT_TEMPLATES

    for config in SENT_TEMPLATES.values():
        assert isinstance(config, PromptConfig)


def test_linguistic_acceptability_templates_structure() -> None:
    """Test that linguistic acceptability templates have expected structure."""
    assert isinstance(LA_TEMPLATES, dict)
    assert len(LA_TEMPLATES) > 0
    assert ENGLISH in LA_TEMPLATES


def test_ner_templates_structure() -> None:
    """Test that NER templates have expected structure."""
    assert isinstance(NER_TEMPLATES, dict)
    assert len(NER_TEMPLATES) > 0
    assert ENGLISH in NER_TEMPLATES


def test_reading_comprehension_templates_structure() -> None:
    """Test that reading comprehension templates have expected structure."""
    assert isinstance(RC_TEMPLATES, dict)
    assert len(RC_TEMPLATES) > 0
    assert ENGLISH in RC_TEMPLATES


def test_multiple_choice_templates_structure() -> None:
    """Test that multiple choice templates have expected structure."""
    assert isinstance(MULTIPLE_CHOICE_TEMPLATES, dict)
    assert len(MULTIPLE_CHOICE_TEMPLATES) > 0
    assert ENGLISH in MULTIPLE_CHOICE_TEMPLATES


def test_simplification_templates_structure() -> None:
    """Test that simplification templates have expected structure."""
    assert isinstance(SIMPL_TEMPLATES, dict)
    assert len(SIMPL_TEMPLATES) > 0


def test_summarization_templates_structure() -> None:
    """Test that summarization templates have expected structure."""
    assert isinstance(SUMM_TEMPLATES, dict)
    assert len(SUMM_TEMPLATES) > 0


def test_token_classification_templates_structure() -> None:
    """Test that token classification templates have expected structure."""
    assert isinstance(TOKEN_CLASSIFICATION_TEMPLATES, dict)
    assert len(TOKEN_CLASSIFICATION_TEMPLATES) > 0


def test_empty_templates_structure() -> None:
    """Test that empty templates have expected structure."""
    assert isinstance(EMPTY_TEMPLATES, dict)


def test_prompt_config_has_required_fields() -> None:
    """Test that prompt configs have all required fields."""
    config = CLASSIFICATION_TEMPLATES[ENGLISH]

    assert isinstance(config.default_prompt_prefix, str)
    assert isinstance(config.default_prompt_template, str)
    assert isinstance(config.default_instruction_prompt, str)
    assert config.default_prompt_label_mapping is not None


def test_prompt_templates_contain_placeholders() -> None:
    """Test that prompt templates contain expected placeholders."""
    config = CLASSIFICATION_TEMPLATES[ENGLISH]

    # Check that templates have placeholders
    assert '{text}' in config.default_prompt_template
    assert '{label}' in config.default_prompt_template
    assert '{text}' in config.default_instruction_prompt


def test_multiple_languages_have_templates() -> None:
    """Test that multiple languages have classification templates."""
    languages_to_check = [ENGLISH, DANISH, SWEDISH]

    for lang in languages_to_check:
        assert lang in CLASSIFICATION_TEMPLATES
        config = CLASSIFICATION_TEMPLATES[lang]
        assert isinstance(config, PromptConfig)
        # Each should have non-empty templates
        assert len(config.default_prompt_prefix) > 0
        assert len(config.default_prompt_template) > 0
