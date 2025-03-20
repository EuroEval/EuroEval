"""Templates for the Reading Comprehension task."""

from ..data_models import PromptConfig
from ..languages import DA, DE, EN, FO, FR, IS, IT, NB, NL, NN, NO, SV
from ..types import TemplateDict

RC_TEMPLATES: TemplateDict = {
    DA: PromptConfig(
        prompt_prefix="Følgende er tekster med tilhørende spørgsmål og svar.",
        prompt_template="Tekst: {text}\nSpørgsmål: {question}\nSvar med maks. 3 ord: "
        "{label}",
        instruction_prompt="Tekst: {text}\n\nBesvar følgende spørgsmål om teksten "
        "ovenfor med maks. 3 ord.\n\nSpørgsmål: {question}",
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
