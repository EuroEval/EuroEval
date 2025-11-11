"""Templates for the Logical Reasoning task."""

import typing as t

from ..data_models import PromptConfig
from ..languages import DANISH

if t.TYPE_CHECKING:
    from ..data_models import Language


LOGIC_TEMPLATES: dict["Language", PromptConfig] = {
    DANISH: PromptConfig(
        default_prompt_prefix="Følgende er en række gåder med svar.",
        default_prompt_template="Gåde:"
        "\n\n{introduction}"
        "\n{'\n'.join(clues)}"
        "\n\nHvem har hvilke egenskaber og bor i hvilket hus?"
        "\n\nAngiv venligst dit svar som et JSON dictionary. Hver key skal være "
        "object_X hvor X er husnummeret. Hver value skal være en liste med de "
        "egenskaber fra kategorierne ovenfor som tilhører personen i hus nr. X."
        "\n\nSvar:"
        "\n\n{label}",
        default_instruction_prompt="Gåde:"
        "\n\n{introduction}"
        "\n{'\n'.join(clues)}"
        "\n\nHvem har hvilke egenskaber og bor i hvilket hus?"
        "\n\nAngiv venligst dit svar som et JSON dictionary. Hver key skal være "
        "object_X hvor X er husnummeret. Hver value skal være en liste med de "
        "egenskaber fra kategorierne ovenfor som tilhører personen i hus nr. X."
        "\n\nFølgende er et eksempel på svarformatet:"
        "\n\n{format_example}"
        "\n\nSvar:"
        "\n\n{label}",
        default_prompt_label_mapping=dict(),
    )
}
