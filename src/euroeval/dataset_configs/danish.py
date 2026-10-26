"""All Danish dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import DANISH, ENGLISH
from ..tasks import (
    COMMON_SENSE,
    EUROPEAN_VALUES,
    GED,
    HALLU,
    INSTRUCTION_FOLLOWING,
    KNOW,
    LA,
    LOGIC,
    MCRC,
    NER,
    NLI,
    RC,
    SENT,
    SUMM,
    TOOL_CALLING,
    TRANSLATION,
    WIC,
)

# Official datasets ###

WMT24PP_EN_DA_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-da",
    pretty_name="WMT24++-en-da",
    source="EuroEval/wmt24pp-en-da",
    task=TRANSLATION,
    languages=[DANISH],
    source_language=ENGLISH,
    target_language=DANISH,
    revision="8822374a57b68e8505bfc436da43f4524d9ed361",
)

FLORES_EN_DA_CONFIG = TranslationDatasetConfig(
    name="flores-en-da",
    pretty_name="FLORES-en-da",
    source="EuroEval/flores-en-da",
    task=TRANSLATION,
    languages=[DANISH],
    source_language=ENGLISH,
    target_language=DANISH,
    revision="7d6c44d9d867e712862ea2bd5cd9b40204d35dee",
)

ANGRY_TWEETS_CONFIG = DatasetConfig(
    name="angry-tweets",
    pretty_name="AngryTweets",
    source="EuroEval/angry-tweets-mini",
    task=SENT,
    languages=[DANISH],
    revision="457785e82655186a01a89026cb35611fb2d71663",
)

DALA_CONFIG = DatasetConfig(
    name="dala",
    pretty_name="DaLA",
    source="EuroEval/dala",
    task=LA,
    languages=[DANISH],
    revision="39ce7c29a4a4f1623250e7096d11d0e2aa6aca04",
)

DANSK_CONFIG = DatasetConfig(
    name="dansk",
    pretty_name="DANSK",
    source="EuroEval/dansk-mini",
    task=NER,
    languages=[DANISH],
    revision="d45aeb92cbb77e336a4607b31c0608e977b63336",
)

MULTI_WIKI_QA_DA_CONFIG = DatasetConfig(
    name="multi-wiki-qa-da",
    pretty_name="MultiWikiQA-da",
    source="EuroEval/multi-wiki-qa-da-mini",
    task=RC,
    languages=[DANISH],
    revision="c3e8cd6dc3f9a880cc890d350b2529275737ddd1",
)

NORDJYLLAND_NEWS_CONFIG = DatasetConfig(
    name="nordjylland-news",
    pretty_name="Nordjylland News",
    source="EuroEval/nordjylland-news-mini",
    task=SUMM,
    languages=[DANISH],
    revision="ab9db69e5d387fd6a68073405cfaedb46baa0cbc",
)

DANSKE_TALEMAADER_CONFIG = DatasetConfig(
    name="danske-talemaader",
    pretty_name="Danske Talemåder",
    source="EuroEval/danske-talemaader",
    task=KNOW,
    languages=[DANISH],
    revision="ac92b93946f718e57a441bbb78ae2ec67d790768",
)

DANISH_CITIZEN_TESTS_CONFIG = DatasetConfig(
    name="danish-citizen-tests",
    pretty_name="Danish Citizen Tests",
    source="EuroEval/danish-citizen-tests-updated",
    task=KNOW,
    languages=[DANISH],
    labels=["a", "b", "c"],
    revision="39c3c6926968be9fc2414b01e8d911a75ca7f699",
)

IFEVAL_DA_CONFIG = DatasetConfig(
    name="ifeval-da",
    pretty_name="IFEval-da",
    source="EuroEval/ifeval-da",
    task=INSTRUCTION_FOLLOWING,
    languages=[DANISH],
    train_split=None,
    val_split=None,
    revision="70e8a7e9484f25c35170a42ac63ff63efdbbb870",
)

VALEU_DA_CONFIG = DatasetConfig(
    name="valeu-da",
    pretty_name="ValEU-da",
    source="EuroEval/european-values-da",
    task=EUROPEAN_VALUES,
    languages=[DANISH],
    train_split=None,
    val_split=None,
    bootstrap_samples=False,
    revision="74561f39812572ec8e1b51b109be452eb116d9fe",
)

RAGTRUTH_DA_CONFIG = DatasetConfig(
    name="ragtruth-da",
    pretty_name="RAGTruth-da",
    source="EuroEval/ragtruth-translated-hallucinations-da-mini",
    task=HALLU,
    languages=[DANISH],
    train_split=None,
    revision="ac137eab1b8f87fdd6023470a649e3fe73e75d75",
)


WINOGRANDE_DA_CONFIG = DatasetConfig(
    name="winogrande-da",
    pretty_name="Winogrande-da",
    source="EuroEval/winogrande-da",
    task=COMMON_SENSE,
    languages=[DANISH],
    labels=["a", "b"],
    revision="9c41302e1e2058fbc499e8e8098f1e74fe0a76a9",
)

ZEBRA_PUZZLE_EASY_DA_CONFIG = DatasetConfig(
    name="zebra-puzzles-easy-da",
    pretty_name="ZebraPuzzlesEasy-da",
    source="EuroEval/zebra-puzzles-easy-da",
    task=LOGIC,
    languages=[DANISH],
    revision="dd4788ef2d00462e90039a452712716dcbb1bc68",
)

DANWIC_CONFIG = DatasetConfig(
    name="danwic",
    pretty_name="DanWiC",
    source="EuroEval/danwic",
    task=WIC,
    languages=[DANISH],
    revision="8bee48497c806d9dca8143d08b491f4805446844",
)


# Unofficial datasets ###

MULTI_BFCL_DA_CONFIG = DatasetConfig(
    name="multi-bfcl-da",
    pretty_name="MultiBFCL-da",
    source="EuroEval/multi-bfcl-da-mini",
    task=TOOL_CALLING,
    languages=[DANISH],
    train_split=None,
    revision="b8b5b7e25f040172e6ead9f207fcc9bdbc662d38",
)


WMT24PP_DA_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-da-en",
    pretty_name="WMT24++-da-en",
    source="EuroEval/wmt24pp-da-en",
    task=TRANSLATION,
    languages=[DANISH],
    source_language=DANISH,
    target_language=ENGLISH,
    unofficial=True,
    revision="0c9c241e28d66db9ee513b7db8f55fc4766b5b7f",
)

FLORES_DA_EN_CONFIG = TranslationDatasetConfig(
    name="flores-da-en",
    pretty_name="FLORES-da-en",
    source="EuroEval/flores-da-en",
    task=TRANSLATION,
    languages=[DANISH],
    source_language=DANISH,
    target_language=ENGLISH,
    unofficial=True,
    revision="097041a88b040e13246e28a820a4a6090ee91c4c",
)

HELLASWAG_DA_CONFIG = DatasetConfig(
    name="hellaswag-da",
    pretty_name="HellaSwag-da",
    source="EuroEval/hellaswag-da-mini",
    task=COMMON_SENSE,
    languages=[DANISH],
    unofficial=True,
    revision="682a7826f244c110fdb6a9a8e588241dffec4395",
)

SCALA_DA_CONFIG = DatasetConfig(
    name="scala-da",
    pretty_name="ScaLA-da",
    source="EuroEval/scala-da",
    task=LA,
    languages=[DANISH],
    unofficial=True,
    revision="4c455e8a3a4c1dce093eb333edcfadd1b9dea121",
)

DANE_CONFIG = DatasetConfig(
    name="dane",
    pretty_name="DaNE",
    source="EuroEval/dane-mini",
    task=NER,
    languages=[DANISH],
    unofficial=True,
    revision="f379fd9af10f2178159a6c7fdf6d3a8f10dccc74",
)

MMLU_DA_CONFIG = DatasetConfig(
    name="mmlu-da",
    pretty_name="MMLU-da",
    source="EuroEval/mmlu-da-mini",
    task=KNOW,
    languages=[DANISH],
    unofficial=True,
    revision="a9962ab03f4b4b8aef38c3118301672a7cce6314",
)

ARC_DA_CONFIG = DatasetConfig(
    name="arc-da",
    pretty_name="ARC-da",
    source="EuroEval/arc-da-mini",
    task=KNOW,
    languages=[DANISH],
    unofficial=True,
    revision="49c5d0340ce21b75ae921937361d51908659c223",
)

BELEBELE_DA_CONFIG = DatasetConfig(
    name="belebele-da",
    pretty_name="Belebele-da",
    source="EuroEval/belebele-da-mini",
    task=MCRC,
    languages=[DANISH],
    unofficial=True,
    revision="4be2b4ba2f8bf07a8a835e478daf69c510a31938",
)

SCANDIQA_DA_CONFIG = DatasetConfig(
    name="scandiqa-da",
    pretty_name="ScandiQA-da",
    source="EuroEval/scandiqa-da-mini",
    task=RC,
    languages=[DANISH],
    unofficial=True,
    revision="67ba2addef566b023ff683f77543559c57ba5710",
)

GOLDENSWAG_DA_CONFIG = DatasetConfig(
    name="goldenswag-da",
    pretty_name="GoldenSwag-da",
    source="EuroEval/goldenswag-da-mini",
    task=COMMON_SENSE,
    languages=[DANISH],
    unofficial=True,
    revision="990a271e39c2410422a0ab6e1173cfafc5858eb1",
)

DANISH_SENTIMENT_IN_CONTEXT_CONFIG = DatasetConfig(
    name="danish-sentiment-in-context",
    pretty_name="Danish Sentiment in Context",
    source="EuroEval/danish-sentiment-in-context",
    task=SENT,
    languages=[DANISH],
    prompt_prefix="Følgende er ord med kontekst og ordets sentiment, som kan være "
    "{labels_str}.",
    prompt_template="{text}\nSentiment: {label}",
    instruction_prompt="{text}\n\nKlassificer sentimentet for det angivne ord i "
    "konteksten. Svar kun med {labels_str}, og intet andet.",
    unofficial=True,
    revision="10849d12e025875f34b0001a780e3553a4545a6e",
)

DANISH_ENTAILMENT_CONFIG = DatasetConfig(
    name="danish-entailment",
    pretty_name="The Danish Entailment Dataset",
    source="EuroEval/danish-entailment",
    task=NLI,
    languages=[DANISH],
    val_split=None,
    unofficial=True,
    revision="1a3b0b2a976094bdf6deb15ec090301c89a96e46",
)

DANISH_LEXICAL_INFERENCE_CONFIG = DatasetConfig(
    name="danish-lexical-inference",
    pretty_name="Danish Lexical Inference",
    source="EuroEval/danish-lexical-inference",
    task=NLI,
    languages=[DANISH],
    labels=["entailment", "contradiction"],
    unofficial=True,
    revision="4651741a107706082a0d86180946b33212d0831d",
)

DAMETA_CONFIG = DatasetConfig(
    name="dameta",
    pretty_name="DAMETA",
    source="EuroEval/dameta",
    task=KNOW,
    languages=[DANISH],
    unofficial=True,
    revision="312554be07085d6f6f7efbf21f142dbc1cf03806",
)

DANISH_SIMILARITY_OUTLIER_CONFIG = DatasetConfig(
    name="danish-similarity-outlier",
    pretty_name="Danish Similarity Outlier Detection",
    source="EuroEval/danish-similarity-outlier-mini",
    task=KNOW,
    languages=[DANISH],
    labels=["a", "b", "c", "d", "e", "f"],
    unofficial=True,
    revision="aa20fd18571e124a852fae1c38d36c4f540e6d35",
)

GERLANGMOD_DA_CONFIG = DatasetConfig(
    name="gerlangmod-da",
    pretty_name="GerLangMod-da",
    source="EuroEval/gerlangmod-da",
    task=GED,
    languages=[DANISH],
    unofficial=True,
    revision="704c78dc841601bbc9f24645c568e6321fdd6a59",
)

ZEBRA_PUZZLE_HARD_DA_CONFIG = DatasetConfig(
    name="zebra-puzzles-hard-da",
    pretty_name="ZebraPuzzlesHard-da",
    source="EuroEval/zebra-puzzles-hard-da",
    task=LOGIC,
    languages=[DANISH],
    unofficial=True,
    revision="e742116242dce6493c15c7ed090f6aeb05fdbe7f",
)

MULTI_IFEVAL_DA_CONFIG = DatasetConfig(
    name="multi-ifeval-da",
    pretty_name="MultiIFEval-da",
    source="EuroEval/multi-ifeval-da",
    task=INSTRUCTION_FOLLOWING,
    languages=[DANISH],
    train_split=None,
    val_split=None,
    unofficial=True,
    revision="5d9ffa719bb2249d3e3ad50c8e9562d487d59534",
)

MKQA_DA_CONFIG = DatasetConfig(
    name="mkqa-da",
    pretty_name="MKQA-da",
    source="EuroEval/mkqa-da-mini",
    task=KNOW,
    languages=[DANISH],
    unofficial=True,
    revision="413986b2ff562c68a05a2ae9e87fcc147eec966e",
)
