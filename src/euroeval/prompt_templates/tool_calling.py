"""Templates for the tool calling task."""

import typing as t

import pydantic

from ..data_models import PromptConfig
from ..languages import ENGLISH

if t.TYPE_CHECKING:
    from ..languages import Language


class ToolCall(pydantic.BaseModel):
    """For structured generation purposes in tool-calling."""

    function: str
    arguments: dict[str, str]


class ToolCallingResponse(pydantic.RootModel[list[ToolCall]]):
    """For structured generation purposes in tool-calling."""

    root: list[ToolCall]


TOOL_CALLING_FUNCTION_KEY = "function"
TOOL_CALLING_ARGUMENTS_KEY = "arguments"
TOOL_CALLING_KEYS = [TOOL_CALLING_FUNCTION_KEY, TOOL_CALLING_ARGUMENTS_KEY]

_DEFAULT_TOOL_CALLING_PROMPT_TEMPLATE_EN = (
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
)

TOOL_CALLING_TEMPLATES: dict["Language", PromptConfig] = {
    ENGLISH: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template=_DEFAULT_TOOL_CALLING_PROMPT_TEMPLATE_EN,
        default_instruction_prompt=_DEFAULT_TOOL_CALLING_PROMPT_TEMPLATE_EN,
        default_prompt_label_mapping=dict(),
    )
}
