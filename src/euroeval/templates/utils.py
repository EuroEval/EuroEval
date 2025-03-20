"""Utility functions for the templates used in EuroEval."""

from ..data_models import Language, PromptConfig, Task
from ..tasks import COMMON_SENSE, KNOW, LA, MCRC, NER, RC, SENT, SPEED, SUMM
from .common_sense import COMMON_SENSE_TEMPLATES
from .knowledge import KNOW_TEMPLATES
from .linguistic_acceptability import LA_TEMPLATES
from .multiple_choice import MCRC_TEMPLATES
from .named_entity import NER_TAG_MAPPING, NER_TEMPLATES
from .reading_comprehension import RC_TEMPLATES
from .sentiment import SENT_TEMPLATES
from .summarization import SUMM_TEMPLATES


def get_prompt_config(task: Task, language: Language) -> PromptConfig:
    """Gets the prompt config for a specific task and language.

    Args:
        task (Task): The `Task` to get templates for.
        language (Language): The `Language` for the task.

    Raises:
        NotImplementedError: If the task is not supported/implemented.
        KeyError: If the language doesn't have a template for the task.

    Returns:
        PromptConfig: The `PromptConfig` dataclass for the given task and language.
    """
    try:
        if task == COMMON_SENSE:
            return COMMON_SENSE_TEMPLATES[language]
        elif task == KNOW:
            return KNOW_TEMPLATES[language]
        elif task == LA:
            return LA_TEMPLATES[language]
        elif task == MCRC:
            return MCRC_TEMPLATES[language]
        elif task == NER:
            return NER_TEMPLATES[language]
        elif task == RC:
            return RC_TEMPLATES[language]
        elif task == SENT:
            return SENT_TEMPLATES[language]
        elif task == SPEED:
            return PromptConfig(
                prompt_prefix="", prompt_template="", instruction_prompt=""
            )
        elif task == SUMM:
            return SUMM_TEMPLATES[language]
        else:
            raise NotImplementedError(f"Unsupported task: {task}.")
    except KeyError:
        raise KeyError(f"No template found for language '{language}' in task '{task}'")


def get_ner_tag(ner_tag: str, language: Language) -> str:
    """Get the translation for a NER tag in the specified language.

    Args:
        ner_tag: The NER tag (e.g., 'b-per', 'i-loc')
        language: The target language

    Returns:
        The translated tag, if one is found otherwise the original given tag
    """
    # Strip 'b-' or 'i-' prefix to get base tag
    base_tag = ner_tag.split("-")[1]
    try:
        return NER_TAG_MAPPING[base_tag][language]
    except KeyError:
        return ner_tag
