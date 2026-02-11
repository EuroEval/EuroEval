"""Templates for the Machine Translation task."""

from ..data_models import PromptConfig
from ..languages import DANISH, ENGLISH, UKRAINIAN, Language

MT_TEMPLATES: dict[tuple[Language, Language], PromptConfig] = {
    (ENGLISH, DANISH): PromptConfig(
        default_prompt_prefix=(
            "The following are English sentences with Danish translations."
        ),
        default_prompt_template="English: {text}\nDanish: {target_text}",
        default_instruction_prompt=(
            "English: {text}\n\nTranslate the above into Danish. Respond with only "
            "the Danish translation."
        ),
        default_prompt_label_mapping=dict(),
    ),
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
