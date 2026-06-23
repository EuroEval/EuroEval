"""Templates for the Logical Reasoning task."""

import typing as t

from ..data_models import PromptConfig
from ..languages import DANISH

if t.TYPE_CHECKING:
    from ..data_models import Language

LOGIC_TEMPLATES: dict["Language", PromptConfig] = {
    DANISH: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Her er en gåde:\n"
        "<riddle>\n{text}\n</riddle>\n"
        "Hvem har hvilke egenskaber og bor i hvilket hus? Angiv venligst dit svar "
        "som en JSON dictionary. Hver key skal være object_X hvor X er husnummeret. "
        "Hver value skal være en liste med de egenskaber fra kategorierne ovenfor "
        "som tilhører personen i hus nr. X.",
        default_prompt_label_mapping=dict(),
    )
}
