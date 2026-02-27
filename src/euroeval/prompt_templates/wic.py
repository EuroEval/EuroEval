"""Templates for the Word in Context task."""

import typing as t

from ..data_models import PromptConfig
from ..languages import DANISH

if t.TYPE_CHECKING:
    from ..languages import Language

WIC_TEMPLATES: dict["Language", PromptConfig] = {
    DANISH: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="ja", different_sense="nej"),
        default_prompt_prefix="Følgende er eksempler på ord brugt i to kontekster og "
        "om de har samme betydning.",
        default_prompt_template="{text}\nSamme betydning: {label}",
        default_instruction_prompt="{text}\n\nHar ordet den samme betydning i de to "
        "kontekster? Svar kun med {labels_str}, og intet andet.",
    ),
}
