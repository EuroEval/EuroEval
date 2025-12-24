"""Templates for the Simplification task."""

from ..data_models import PromptConfig
from ..languages import DUTCH, ENGLISH

SIMPL_TEMPLATES = {
    ENGLISH: PromptConfig(
        default_prompt_prefix="The following are documents with accompanying "
        "simplifications.",
        default_prompt_template="Document: {text}\nSimplification: {target_text}",
        default_instruction_prompt="Document: {text}\n\nWrite a simplification "
        "of the above document.",
        default_prompt_label_mapping=dict(),
    ),
    DUTCH: PromptConfig(
        default_prompt_prefix="Hieronder volgen documenten met bijbehorende "
        "versimpelingen.",
        default_prompt_template="Document: {text}\nVersimpeling: {target_text}",
        default_instruction_prompt="Document: {text}\n\nVersimpel het "
        "bovenstaande document.",
        default_prompt_label_mapping=dict(),
    ),
}
