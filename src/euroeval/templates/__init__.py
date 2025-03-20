"""The different templates used in EuroEval."""

from .common_sense import COMMON_SENSE_TEMPLATES
from .knowledge import KNOW_TEMPLATES
from .linguistic_acceptability import LA_TEMPLATES
from .multiple_choice import MCRC_TEMPLATES
from .named_entity import NER_TAG_MAPPING, NER_TEMPLATES
from .reading_comprehension import RC_TEMPLATES
from .sentiment import SENT_TEMPLATES
from .summarization import SUMM_TEMPLATES
from .utils import get_ner_tag, get_prompt_config
