"""Templates for the Summarization task."""

from ..data_models import PromptConfig
from ..languages import DANISH, UKRAINIAN, Language

MT_TEMPLATES: dict[tuple[Language, Language], PromptConfig] = {
    (DANISH, UKRAINIAN): PromptConfig(
        default_prompt_prefix=(
            "Nedenfor er danske tekster med tilhørende ukrainske oversættelser."
        ),
        default_prompt_template=(
            "Dansk tekst: {text}\nUkrainsk oversættelse: {target_text}"
        ),
        default_instruction_prompt=(
            "Dansk tekst: {text}\n\nOversæt den ovenstående tekst til ukrainsk."
        ),
        default_prompt_label_mapping=dict(),
    )
}
