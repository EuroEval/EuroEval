"""The different templates used in EuroEval."""

from .base import PromptConfig
from .common_sense import COMMON_SENSE_TEMPLATES
from .knowledge import KNOW_TEMPLATES
from .linguistic_acceptability import LA_TEMPLATES
from .multiple_choice import MCRC_TEMPLATES
from .named_entity import NER_TEMPLATES
from .reading_comprehension import RC_TEMPLATES
from .sentiment import SENT_TEMPLATES
from .summarization import SUMM_TEMPLATES

TEMPLATES_DICT = {
    "common-sense-reasoning": COMMON_SENSE_TEMPLATES,
    "knowledge": KNOW_TEMPLATES,
    "linguistic-acceptability": LA_TEMPLATES,
    "multiple-choice-reading-comprehension": MCRC_TEMPLATES,
    "named-entity-recognition": NER_TEMPLATES,
    "reading-comprehension": RC_TEMPLATES,
    "sentiment-classification": SENT_TEMPLATES,
    "summarization": SUMM_TEMPLATES,
}
