"""Templates for tasks using LLM-as-a-judge."""

from ..data_models import PromptConfig
from ..languages import DA

LLM_AS_A_JUDGE_TEMPLATES = {
    DA: PromptConfig(
        default_prompt_prefix="Følgende er spørgsmål med tilhørende svar.",
        default_prompt_template="Spørgsmål: {text}\nSvar: {target_text}",
        default_instruction_prompt="{text}",
        default_prompt_label_mapping=dict(),
    )
}
