"""Templates for the Logical Reasoning task."""

import typing as t

from ..data_models import PromptConfig
from ..languages import DANISH

if t.TYPE_CHECKING:
    from ..data_models import Language

# TODO: There should not be a question in default_prompt_template. Low priority.
# Remove the explanation of JSON too?
LOGIC_TEMPLATES: dict["Language", PromptConfig] = {
    DANISH: PromptConfig(
        default_prompt_prefix="Følgende er en række gåder med svar.",
        default_prompt_template="Gåde:"
        "\n\n{text}"
        "\n\nHvem har hvilke egenskaber og bor i hvilket hus?"
        "\n\nAngiv venligst dit svar som et JSON dictionary. Hver key skal være "
        "object_X hvor X er husnummeret. Hver value skal være en liste med de "
        "egenskaber fra kategorierne ovenfor som tilhører personen i hus nr. X."
        "\n\nSvar:"
        "\n\n{target_text}",
        default_instruction_prompt="Gåde:"
        "\n\n{text}"
        "\n\nHvem har hvilke egenskaber og bor i hvilket hus?"
        "\n\nAngiv venligst dit svar som et JSON dictionary. Hver key skal være "
        "object_X hvor X er husnummeret. Hver value skal være en liste med de "
        "egenskaber fra kategorierne ovenfor som tilhører personen i hus nr. X.",
        default_prompt_label_mapping=dict(),
    )
}
