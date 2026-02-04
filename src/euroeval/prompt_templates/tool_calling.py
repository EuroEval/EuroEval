"""Templates for the tool calling task."""

import typing as t

from euroeval.data_models import PromptConfig
from euroeval.languages import ENGLISH

if t.TYPE_CHECKING:
    from ..languages import Language

TOOL_CALLING_TEMPLATES: dict["Language", PromptConfig] = {
    ENGLISH: PromptConfig(  # TODO:
        default_prompt_prefix="The following are questions.",
        default_prompt_template="Question: {text}",
        default_instruction_prompt="Question: {text}\n\nAnswer the above question",
        default_prompt_label_mapping="auto",
    )
}
