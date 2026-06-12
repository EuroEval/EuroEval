"""Templates for the tool calling task."""

import typing as t

import pydantic

from ..constants import (
    TOOL_CALLING_ARGUMENTS_KEY,
    TOOL_CALLING_CALLS_KEY,
    TOOL_CALLING_FUNCTION_KEY,
)
from ..data_models import PromptConfig
from ..languages import ENGLISH

if t.TYPE_CHECKING:
    from ..languages import Language

ToolCall = pydantic.create_model(
    "ToolCall",
    __base__=pydantic.BaseModel,
    **{TOOL_CALLING_FUNCTION_KEY: str, TOOL_CALLING_ARGUMENTS_KEY: dict[str, str]},
)


ToolCallingResponse = pydantic.create_model(
    "ToolCallingResponse",
    __base__=pydantic.BaseModel,
    **{TOOL_CALLING_CALLS_KEY: list[ToolCall]},
)


def _reformat(s: str) -> str:
    return s.replace("{", "{{").replace("}", "}}").replace("$", "")


TOOL_CALLING_TEMPLATES: dict["Language", PromptConfig] = {
    ENGLISH: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt=(
            "A list of names and descriptions of functions available, "
            "and a user question is given below: \n"
            "{text}\n"
            "Answer with a JSON, strictly following this "
            f"(model) schema: "
            f"{_reformat(ToolCallingResponse.schema_json())}. "
            f"The value of {TOOL_CALLING_CALLS_KEY} must list the "
            "function call(s) to execute "
            "to fulfill the users request, in the right order and number, "
            "using double quotes for all keys and strings, and nothing else "
            "(no additional explanatory text)."
        ),
        default_prompt_label_mapping=dict(),
    )
}

pass
