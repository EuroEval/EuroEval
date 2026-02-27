"""Templates for the Natural Language Inference task."""

import typing as t

from ..data_models import PromptConfig
from ..languages import DANISH

if t.TYPE_CHECKING:
    from ..languages import Language

NLI_TEMPLATES: dict["Language", PromptConfig] = {
    DANISH: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="sand", neutral="neutral", contradiction="falsk"
        ),
        default_prompt_prefix="Følgende er par af udsagn og om det andet udsagn følger "
        "af det første, hvilket kan være {labels_str}.",
        default_prompt_template="Udsagn: {text}\nEntailment: {label}",
        default_instruction_prompt="Udsagn: {text}\n\nBestem om det andet udsagn følger "
        "af det første udsagn. Svar kun med {labels_str}, og intet andet.",
    ),
}
