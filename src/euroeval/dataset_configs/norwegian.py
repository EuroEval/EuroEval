"""All Norwegian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import ENGLISH, NORWEGIAN, NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK
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
    RC,
    SENT,
    SUMM,
    TEXT_CLASSIFICATION,
    TOOL_CALLING,
    TRANSLATION,
)

# Official datasets ###

WMT24PP_EN_NO_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-no",
    pretty_name="WMT24++-en-no",
    source="EuroEval/wmt24pp-en-no",
    task=TRANSLATION,
    languages=[NORWEGIAN],
    source_language=ENGLISH,
    target_language=NORWEGIAN,
    revision="12f864763019da8cb1ac20f3ce73334657d7d860",
)

FLORES_EN_NO_CONFIG = TranslationDatasetConfig(
    name="flores-en-no",
    pretty_name="FLORES-en-no",
    source="EuroEval/flores-en-no",
    task=TRANSLATION,
    languages=[NORWEGIAN],
    source_language=ENGLISH,
    target_language=NORWEGIAN,
    revision="f2e6e0076ba2e7c3cbf5e5f26f95bb014a0c1b8b",
)

NOREC_CONFIG = DatasetConfig(
    name="norec",
    pretty_name="NoReC",
    source="EuroEval/norec-mini",
    task=SENT,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    revision="07b99ab3363c2e7f8f87015b01c21f4d9b917ce3",
)

SCALA_NB_CONFIG = DatasetConfig(
    name="scala-nb",
    pretty_name="ScaLA-nb",
    source="EuroEval/scala-nb",
    task=LA,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN],
    revision="544d09f5b6e784ca615330d9219b74ebfc173f32",
)

SCALA_NN_CONFIG = DatasetConfig(
    name="scala-nn",
    pretty_name="ScaLA-nn",
    source="EuroEval/scala-nn",
    task=LA,
    languages=[NORWEGIAN_NYNORSK, NORWEGIAN],
    revision="442bc8cc1b97ac280d6f8b3631bada3eb7ffc53a",
)

NORNE_NB_CONFIG = DatasetConfig(
    name="norne-nb",
    pretty_name="NorNE-nb",
    source="EuroEval/norne-nb-mini",
    task=NER,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN],
    revision="67710dfa772469bf5d3653bf0ca9f431e782d85c",
)

NORNE_NN_CONFIG = DatasetConfig(
    name="norne-nn",
    pretty_name="NorNE-nn",
    source="EuroEval/norne-nn-mini",
    task=NER,
    languages=[NORWEGIAN_NYNORSK, NORWEGIAN],
    revision="b34982a3ffc391bf3c5d6c999a2c1804ae170780",
)

NORQUAD_CONFIG = DatasetConfig(
    name="norquad",
    pretty_name="NorQuAD",
    source="EuroEval/norquad-mini",
    task=RC,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    num_few_shot_examples=2,
    revision="a47881440ce4b18ef61a99be66dc4badbf5aac6e",
)

NO_SAMMENDRAG_CONFIG = DatasetConfig(
    name="no-sammendrag",
    pretty_name="NoSammendrag",
    source="EuroEval/no-sammendrag-mini",
    task=SUMM,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    revision="e75ef1a845b0c516ca71f0b21fc2457f67ea963c",
)

NRK_QUIZ_QA_CONFIG = DatasetConfig(
    name="nrk-quiz-qa",
    pretty_name="NRK Quiz QA",
    source="EuroEval/nrk-quiz-qa-mini",
    task=KNOW,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    revision="452459a682790222fbe385f62464b8d04ae06bc5",
)

IDIOMS_NO_CONFIG = DatasetConfig(
    name="idioms-no",
    pretty_name="Idioms-no",
    source="EuroEval/idioms-no",
    task=KNOW,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    revision="c6f26c882b02cffd99a5920a677e4e0857cc3602",
)

NOR_COMMON_SENSE_QA_CONFIG = DatasetConfig(
    name="nor-common-sense-qa",
    pretty_name="NorCommonSenseQA",
    source="EuroEval/nor-common-sense-qa",
    task=COMMON_SENSE,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    labels=["a", "b", "c", "d", "e"],
    revision="f67b02d29612ff84b5a999df15a085a479fb53ed",
)

VALEU_NO_CONFIG = DatasetConfig(
    name="valeu-no",
    pretty_name="VaLEU-no",
    source="EuroEval/european-values-no",
    task=EUROPEAN_VALUES,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    train_split=None,
    val_split=None,
    bootstrap_samples=False,
    instruction_prompt="{text}",
    revision="591a5845d5f6338986dacf139b1118c88e4510a7",
)

MULTI_IFEVAL_NB_CONFIG = DatasetConfig(
    name="multi-ifeval-nb",
    pretty_name="MultiIFEval-nb",
    source="EuroEval/multi-ifeval-no",
    task=INSTRUCTION_FOLLOWING,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN],
    train_split=None,
    val_split=None,
    revision="1ebec885d5be1db88669b51529947635a1e39eb0",
)

MULTI_IFEVAL_NN_CONFIG = DatasetConfig(
    name="multi-ifeval-nn",
    pretty_name="MultiIFEval-nn",
    source="EuroEval/multi-ifeval-nn",
    task=INSTRUCTION_FOLLOWING,
    languages=[NORWEGIAN_NYNORSK, NORWEGIAN],
    train_split=None,
    val_split=None,
    revision="11604aa8dc4de6f67e1970c2642c47b081667495",
)

ZEBRA_PUZZLE_EASY_NB_CONFIG = DatasetConfig(
    name="zebra-puzzles-easy-nb",
    pretty_name="ZebraPuzzlesEasy-nb",
    source="EuroEval/zebra-puzzles-easy-nb",
    task=LOGIC,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN],
    revision="a45a2d7c76b5cfdf8d214b3f75086a7806bbdd38",
)

ZEBRA_PUZZLE_EASY_NN_CONFIG = DatasetConfig(
    name="zebra-puzzles-easy-nn",
    pretty_name="ZebraPuzzlesEasy-nn",
    source="EuroEval/zebra-puzzles-easy-nn",
    task=LOGIC,
    languages=[NORWEGIAN_NYNORSK, NORWEGIAN],
    revision="e6c15c0172a31294492be3260d6ce94406d2103b",
)

RAGTRUTH_NO_CONFIG = DatasetConfig(
    name="ragtruth-no",
    pretty_name="RAGTruth-no",
    source="EuroEval/ragtruth-translated-hallucinations-no-mini",
    task=HALLU,
    languages=[NORWEGIAN],
    train_split=None,
    revision="b0d661a44d4ddf5718d942b1f44c18ae6e29a9ef",
)


# Unofficial datasets ###

MULTI_BFCL_NO_CONFIG = DatasetConfig(
    name="multi-bfcl-no",
    pretty_name="MultiBFCL-no",
    source="EuroEval/multi-bfcl-no-mini",
    task=TOOL_CALLING,
    languages=[NORWEGIAN],
    train_split=None,
    revision="05e01c1e43d5242742cf81b611e1b8e7fa610520",
)


WMT24PP_NO_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-no-en",
    pretty_name="WMT24++-no-en",
    source="EuroEval/wmt24pp-no-en",
    task=TRANSLATION,
    languages=[NORWEGIAN],
    source_language=NORWEGIAN,
    target_language=ENGLISH,
    unofficial=True,
    revision="e41f27974afc9c5d149aa58c68133473a1275ca5",
)

FLORES_NO_EN_CONFIG = TranslationDatasetConfig(
    name="flores-no-en",
    pretty_name="FLORES-no-en",
    source="EuroEval/flores-no-en",
    task=TRANSLATION,
    languages=[NORWEGIAN],
    source_language=NORWEGIAN,
    target_language=ENGLISH,
    unofficial=True,
    revision="7e1f2fe00b6fb86513f298ab76d24490f4c234db",
)

IFEVAL_NB_CONFIG = DatasetConfig(
    name="ifeval-nb",
    pretty_name="IFEval-nb",
    source="EuroEval/ifeval-nb",
    task=INSTRUCTION_FOLLOWING,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN],
    train_split=None,
    val_split=None,
    unofficial=True,
    revision="da1beb9c6452d2a22295adcc172dbf210ce5d5e7",
)

IFEVAL_NN_CONFIG = DatasetConfig(
    name="ifeval-nn",
    pretty_name="IFEval-nn",
    source="EuroEval/ifeval-nn",
    task=INSTRUCTION_FOLLOWING,
    languages=[NORWEGIAN_NYNORSK, NORWEGIAN],
    train_split=None,
    val_split=None,
    unofficial=True,
    revision="c82317c81c037cee2f19ce2096dd1ba529989b40",
)

NO_COLA_CONFIG = DatasetConfig(
    name="no-cola",
    pretty_name="NoCoLA",
    source="EuroEval/no-cola-mini",
    task=LA,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN],
    unofficial=True,
    revision="bab066efed8efea49ee45af37a36ff32ff99bd33",
)

NORGLM_MULTI_QA = DatasetConfig(
    name="norglm-multi-qa",
    pretty_name="NorGLM-Multi-QA",
    source="EuroEval/norglm-multi-qa",
    task=RC,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    unofficial=True,
    revision="63fe792355afabdd20c63e44361f876fbeef5761",
)

NORGLM_MULTI_SUM = DatasetConfig(
    name="norglm-multi-sum",
    pretty_name="NorGLM-Multi-Sum",
    source="EuroEval/norglm-multi-sum",
    task=SUMM,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    unofficial=True,
    revision="92908fb650a9bba30db9678b20c86e725ce9ad3b",
)

SCHIBSTED_NO_CONFIG = DatasetConfig(
    name="schibsted-no",
    pretty_name="Schibsted-no",
    source="EuroEval/schibsted-article-summaries-no",
    task=SUMM,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    unofficial=True,
    revision="29988b762ec8150e58cd92c8f4745d3df7afa385",
)

SCHIBSTED_FRONT_TITLE_NO_CONFIG = DatasetConfig(
    name="vg-front-title",
    pretty_name="VG Front Title",
    source="EuroEval/vg-front-title",
    task=SUMM,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    unofficial=True,
    max_generated_tokens=64,
    prompt_prefix="Her følger nyhetsartikler med tilhørende titler.",
    prompt_template="Nyhetsartikkel: {text}\nTittel: {target_text}",
    instruction_prompt=(
        "Nyhetsartikkel: {text}\n\nSkriv en tittel for den ovennevnte artikkelen."
    ),
    revision="b83571c11144521ff04b531fb276277624c6a5fd",
)

PERSONAL_SUM_CONFIG = DatasetConfig(
    name="personal-sum",
    pretty_name="Personal Sum",
    source="EuroEval/personal-sum",
    task=SUMM,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    unofficial=True,
    revision="15b42bb4b52f98071d0aa8bb75d6bd5fbcfa4de9",
)

MMLU_NO_CONFIG = DatasetConfig(
    name="mmlu-no",
    pretty_name="MMLU-no",
    source="EuroEval/mmlu-no-mini",
    task=KNOW,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    unofficial=True,
    revision="a3c4f7df841c4a37ef0f8a2521167b4c268bb1ce",
)

ARC_NO_CONFIG = DatasetConfig(
    name="arc-no",
    pretty_name="ARC-no",
    source="EuroEval/arc-no-mini",
    task=KNOW,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    unofficial=True,
    revision="14e4184dce15d3cfbd84dccd5f7b37ccad6d83be",
)

HELLASWAG_NO_CONFIG = DatasetConfig(
    name="hellaswag-no",
    pretty_name="HellaSwag-no",
    source="EuroEval/hellaswag-no-mini",
    task=COMMON_SENSE,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    unofficial=True,
    revision="9de6012252553eeec5efa207e58fbc1f74af22aa",
)

BELEBELE_NO_CONFIG = DatasetConfig(
    name="belebele-no",
    pretty_name="Belebele-no",
    source="EuroEval/belebele-no-mini",
    task=MCRC,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    unofficial=True,
    revision="25ded4467d392d22fdcd966b4ba794b96a72c31a",
)

MULTI_WIKI_QA_NB_CONFIG = DatasetConfig(
    name="multi-wiki-qa-nb",
    pretty_name="MultiWikiQA-nb",
    source="EuroEval/multi-wiki-qa-no-mini",
    task=RC,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN],
    unofficial=True,
    revision="730b3f845cf419d3aadaa13a19d5396898516648",
)

MULTI_WIKI_QA_NN_CONFIG = DatasetConfig(
    name="multi-wiki-qa-nn",
    pretty_name="MultiWikiQA-nn",
    source="EuroEval/multi-wiki-qa-nn-mini",
    task=RC,
    languages=[NORWEGIAN_NYNORSK, NORWEGIAN],
    unofficial=True,
    revision="a5079bcb0a81d342c871fe0fd0b7209fac233912",
)

WINOGRANDE_NO_CONFIG = DatasetConfig(
    name="winogrande-no",
    pretty_name="Winogrande-no",
    source="EuroEval/winogrande-no",
    task=COMMON_SENSE,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    labels=["a", "b"],
    unofficial=True,
    revision="c8cb6c334b6333cce61ae28a6998547d53c1cea5",
)

NORDIAL_CONFIG = DatasetConfig(
    name="nordial",
    pretty_name="NorDial",
    source="EuroEval/nordial",
    task=TEXT_CLASSIFICATION,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK, NORWEGIAN],
    labels=["bokmål", "nynorsk", "dialectal", "mixed"],
    prompt_prefix="Følgende er norske tweets og hvilken skriftform de er skrevet på, "
    "som kan være {labels_str}.",
    prompt_template="Tweet: {text}\nSkriftform: {label}",
    instruction_prompt="Tweet: {text}\n\nKlassifiser skriftformen av tweeten. Svar med "
    "{labels_str}, og ikke noe annet.",
    prompt_label_mapping=dict(
        bokmål="bokmål", nynorsk="nynorsk", dialectal="dialekt", mixed="blandet"
    ),
    unofficial=True,
    revision="3d2f485f3078496e5137be1e6bbcea31d1ed8b0b",
)

NORSUMM_NB_CONFIG = DatasetConfig(
    name="norsumm-nb",
    pretty_name="NorSumm-nb",
    source="EuroEval/norsumm-nb",
    task=SUMM,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN],
    unofficial=True,
    val_split=None,
    revision="7863c9e3e794d360e55fe650ec20328af64d2228",
)

NORSUMM_NN_CONFIG = DatasetConfig(
    name="norsumm-nn",
    pretty_name="NorSumm-nn",
    source="EuroEval/norsumm-nn",
    task=SUMM,
    languages=[NORWEGIAN_NYNORSK, NORWEGIAN],
    unofficial=True,
    val_split=None,
    revision="ec04568fcb9459510ce350feaebca1a43ebcbbad",
)

GERLANGMOD_NB_CONFIG = DatasetConfig(
    name="gerlangmod-nb",
    pretty_name="GerLangMod-nb",
    source="EuroEval/gerlangmod-nb",
    task=GED,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN],
    unofficial=True,
    revision="3c0e7fb276cedac05dfe461f30392e9d5491c970",
)

GERLANGMOD_NN_CONFIG = DatasetConfig(
    name="gerlangmod-nn",
    pretty_name="GerLangMod-nn",
    source="EuroEval/gerlangmod-nn",
    task=GED,
    languages=[NORWEGIAN_NYNORSK, NORWEGIAN],
    unofficial=True,
    revision="6e419359599dde827c700f3666cc013407b8afc9",
)

ZEBRA_PUZZLE_HARD_NB_CONFIG = DatasetConfig(
    name="zebra-puzzles-hard-nb",
    pretty_name="ZebraPuzzlesHard-nb",
    source="EuroEval/zebra-puzzles-hard-nb",
    task=LOGIC,
    languages=[NORWEGIAN_BOKMÅL, NORWEGIAN],
    unofficial=True,
    revision="3b3975d340bea19a3970458c27d2ca94a9549ffc",
)

ZEBRA_PUZZLE_HARD_NN_CONFIG = DatasetConfig(
    name="zebra-puzzles-hard-nn",
    pretty_name="ZebraPuzzlesHard-nn",
    source="EuroEval/zebra-puzzles-hard-nn",
    task=LOGIC,
    languages=[NORWEGIAN_NYNORSK, NORWEGIAN],
    unofficial=True,
    revision="5fd33b16465ed3df4d15d0c70268c2069b0800a7",
)
