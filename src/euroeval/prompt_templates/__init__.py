"""The different prompt templates used in EuroEval."""

import typing as t

from ..data_models import PromptConfig
from ..languages import get_all_languages
from .classification import CLASSIFICATION_TEMPLATES
from .linguistic_acceptability import LA_TEMPLATES
from .multiple_choice import MULTIPLE_CHOICE_TEMPLATES
from .named_entity_recognition import NER_TEMPLATES
from .reading_comprehension import RC_TEMPLATES
from .sentiment_classification import SENT_TEMPLATES
from .simplification import SIMPL_TEMPLATES
from .summarization import SUMM_TEMPLATES
from .token_classification import TOKEN_CLASSIFICATION_TEMPLATES
from .translation import TRANSLATION_TEMPLATES

if t.TYPE_CHECKING:
    from ..languages import Language


EMPTY_TEMPLATES: dict["Language", PromptConfig] = {
    lang: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="{text}",
        default_prompt_label_mapping="auto",
    )
    for lang in get_all_languages().values()
}
