"""Templates for the Sentiment Analysis task."""

from ..data_models import PromptConfig
from ..languages import DA, DE, EN, FO, FR, IS, IT, NB, NL, NN, NO, SV
from ..types import TemplateDict

SENT_TEMPLATES: TemplateDict = {
    DA: PromptConfig(
        prompt_prefix="Følgende er tekster og deres sentiment, som kan være 'positiv', "
        "'neutral' eller 'negativ'.",
        prompt_template="Tekst: {text}\nSentiment: {label}",
        instruction_prompt="Tekst: {text}\n\nKlassificer sentimentet i teksten. Svar "
        "kun med 'positiv', 'neutral' eller 'negativ', og intet andet.",
    ),
    DE: PromptConfig(prompt_prefix="", prompt_template="", instruction_prompt=""),
    EN: PromptConfig(prompt_prefix="", prompt_template="", instruction_prompt=""),
    FO: PromptConfig(prompt_prefix="", prompt_template="", instruction_prompt=""),
    FR: PromptConfig(prompt_prefix="", prompt_template="", instruction_prompt=""),
    IS: PromptConfig(prompt_prefix="", prompt_template="", instruction_prompt=""),
    IT: PromptConfig(prompt_prefix="", prompt_template="", instruction_prompt=""),
    NB: PromptConfig(prompt_prefix="", prompt_template="", instruction_prompt=""),
    NL: PromptConfig(prompt_prefix="", prompt_template="", instruction_prompt=""),
    NN: PromptConfig(prompt_prefix="", prompt_template="", instruction_prompt=""),
    NO: PromptConfig(prompt_prefix="", prompt_template="", instruction_prompt=""),
    SV: PromptConfig(prompt_prefix="", prompt_template="", instruction_prompt=""),
}
