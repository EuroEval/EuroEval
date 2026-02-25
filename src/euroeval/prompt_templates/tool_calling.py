"""Templates for the tool calling task."""

import typing as t

from euroeval.constants import TOOL_CALLING_ARGUMENTS_KEY, TOOL_CALLING_FUNCTION_KEY

from ..data_models import PromptConfig
from ..languages import ENGLISH

if t.TYPE_CHECKING:
    from ..languages import Language

TOOL_CALLING_TEMPLATES: dict["Language", PromptConfig] = {
    ENGLISH: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt=(
            "A list of names and descriptions of functions available, "
            "and a user question is given below: \n"
            "{text}\n"
            "Answer the user question with a list of function call(s) to execute "
            "to fulfill the users request. "
            "The answer MUST be a valid JSON deserializable to a list of dicts, "
            "using double quotes for all keys "
            "and strings, and must not include any additional explanatory text. "
            "Each dict in the list should have exactly the keys "
            f'"{TOOL_CALLING_FUNCTION_KEY}" (the function name) '
            f'and "{TOOL_CALLING_ARGUMENTS_KEY}" '
            "(a dict with the keywords and values describing the parameters)."
        ),
        default_prompt_label_mapping=dict(),
    )
}
