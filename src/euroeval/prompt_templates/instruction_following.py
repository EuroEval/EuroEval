"""Templates for the Instruction Following task."""

import typing as t

from ..data_models import PromptConfig
from ..languages import ENGLISH, ESTONIAN

if t.TYPE_CHECKING:
    from ..languages import Language

INSTRUCTION_FOLLOWING_TEMPLATES: dict["Language", PromptConfig] = {
    ENGLISH: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="{text}",
        default_instruction_prompt="{text}",
        default_prompt_label_mapping="auto",
    ),
    ESTONIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="{text}",
        default_instruction_prompt="{text}",
        default_prompt_label_mapping="auto",
    ),
}
