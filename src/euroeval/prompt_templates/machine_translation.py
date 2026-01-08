"""Templates for the Summarization task."""

from ..data_models import PromptConfig
from ..language_pairs_mt import DANISH_TO_UKRAINIAN, LanguagePair

MT_TEMPLATES: dict[LanguagePair, PromptConfig] = {
    DANISH_TO_UKRAINIAN: PromptConfig(
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
