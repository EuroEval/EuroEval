"""Public-interface tests for tool-calling response schemas."""

from euroeval.prompt_templates.tool_calling import ToolCallingResponse


def test_tool_calling_schema_accepts_json_argument_values() -> None:
    """Tool arguments can contain scalar and nested JSON values."""
    response = ToolCallingResponse.model_validate(
        {
            "tool_calls": [
                {
                    "function": "update_record",
                    "arguments": {
                        "count": 3,
                        "enabled": False,
                        "metadata": {"tags": ["danish", 2]},
                    },
                }
            ]
        }
    )

    assert response.tool_calls[0].arguments == {
        "count": 3,
        "enabled": False,
        "metadata": {"tags": ["danish", 2]},
    }
