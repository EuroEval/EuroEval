"""Templates for the tool calling task."""

import typing as t

from euroeval.data_models import PromptConfig

from ..languages import ENGLISH

if t.TYPE_CHECKING:
    from ..languages import Language

TOOL_CALLING_TEMPLATES: dict["Language", PromptConfig] = {
    ENGLISH: PromptConfig(
        default_prompt_prefix="",
        default_instruction_prompt="",
        default_prompt_template=(
            "These are names and descriptions of functions available:\n"
            "{function_info}\n"
            "Given the following user request: '{question}', "
            "answer with a list of function(s) to use in the exact format:\n"
            "{output_format_example}\n"
            "The answer MUST be a valid JSON, using double quotes for all keys"
            "and strings, and must not include any additional explanatory text. "
            'Each item should have "function" (the function name) '
            'and "arguments" (an object with the parameters).'
        ),
        default_prompt_label_mapping="auto",
    )
}
