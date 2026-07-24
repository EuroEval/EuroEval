"""Templates for the Grammatical Error Correction task."""

from ..data_models import PromptConfig
from ..languages import DANISH, ENGLISH, FAROESE

GEC_TEMPLATES = {
    DANISH: PromptConfig(
        default_prompt_prefix="Følgende er sætninger med grammatiske fejl og deres "
        "korrigerede versioner.",
        default_prompt_template="Sætning: {text}\nKorrigeret sætning: {target_text}",
        default_instruction_prompt="Sætning: {text}\n\nKorriger de grammatiske fejl "
        "i sætningen og skriv den korrigerede sætning, og intet andet.",
        default_prompt_label_mapping=dict(),
    ),
    ENGLISH: PromptConfig(
        default_prompt_prefix="The following are sentences with grammatical errors "
        "and their corrected versions.",
        default_prompt_template="Sentence: {text}\nCorrected sentence: {target_text}",
        default_instruction_prompt="Sentence: {text}\n\nCorrect the grammatical "
        "errors in the sentence and write the corrected sentence, and nothing else.",
        default_prompt_label_mapping=dict(),
    ),
    FAROESE: PromptConfig(
        default_prompt_prefix="Hetta eru setningar við málvillum og teirra "
        "rættaðu útgávur.",
        default_prompt_template="Setningur: {text}\nRættaður setningur: {target_text}",
        default_instruction_prompt="Setningur: {text}\n\nRætta málvillurnar í "
        "setninginum og skriva rættaða setningin, og einki annað.",
        default_prompt_label_mapping=dict(),
    ),
}
