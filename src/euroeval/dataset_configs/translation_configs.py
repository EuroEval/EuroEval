"""Dataset configurations for the translation task."""

from ..data_models import DatasetConfig
from ..languages import (
    BULGARIAN,
    CATALAN,
    CROATIAN,
    CZECH,
    DANISH,
    DUTCH,
    ENGLISH,
    ESTONIAN,
    FINNISH,
    FRENCH,
    GERMAN,
    GREEK,
    HUNGARIAN,
    ICELANDIC,
    ITALIAN,
    LATVIAN,
    LITHUANIAN,
    NORWEGIAN,
    POLISH,
    PORTUGUESE,
    ROMANIAN,
    SERBIAN,
    SLOVAK,
    SLOVENE,
    SWEDISH,
    UKRAINIAN,
)
from ..tasks import TRANSLATION

WMT24PP_EN_BG_CONFIG = DatasetConfig(
    name="wmt24pp-en-bg",
    pretty_name="WMT24++-bg",
    source="EuroEval/wmt24pp-en-bg",
    task=TRANSLATION,
    languages=[ENGLISH, BULGARIAN],
    unofficial=True,
)

WMT24PP_EN_CA_CONFIG = DatasetConfig(
    name="wmt24pp-en-ca",
    pretty_name="WMT24++-ca",
    source="EuroEval/wmt24pp-en-ca",
    task=TRANSLATION,
    languages=[ENGLISH, CATALAN],
    unofficial=True,
)

WMT24PP_EN_CS_CONFIG = DatasetConfig(
    name="wmt24pp-en-cs",
    pretty_name="WMT24++-cs",
    source="EuroEval/wmt24pp-en-cs",
    task=TRANSLATION,
    languages=[ENGLISH, CZECH],
    unofficial=True,
)

WMT24PP_EN_DA_CONFIG = DatasetConfig(
    name="wmt24pp-en-da",
    pretty_name="WMT24++-da",
    source="EuroEval/wmt24pp-en-da",
    task=TRANSLATION,
    languages=[ENGLISH, DANISH],
    unofficial=True,
)

WMT24PP_EN_DE_CONFIG = DatasetConfig(
    name="wmt24pp-en-de",
    pretty_name="WMT24++-de",
    source="EuroEval/wmt24pp-en-de",
    task=TRANSLATION,
    languages=[ENGLISH, GERMAN],
    unofficial=True,
)

WMT24PP_EN_EL_CONFIG = DatasetConfig(
    name="wmt24pp-en-el",
    pretty_name="WMT24++-el",
    source="EuroEval/wmt24pp-en-el",
    task=TRANSLATION,
    languages=[ENGLISH, GREEK],
    unofficial=True,
)

WMT24PP_EN_ET_CONFIG = DatasetConfig(
    name="wmt24pp-en-et",
    pretty_name="WMT24++-et",
    source="EuroEval/wmt24pp-en-et",
    task=TRANSLATION,
    languages=[ENGLISH, ESTONIAN],
    unofficial=True,
)

WMT24PP_EN_FI_CONFIG = DatasetConfig(
    name="wmt24pp-en-fi",
    pretty_name="WMT24++-fi",
    source="EuroEval/wmt24pp-en-fi",
    task=TRANSLATION,
    languages=[ENGLISH, FINNISH],
    unofficial=True,
)

WMT24PP_EN_FR_CONFIG = DatasetConfig(
    name="wmt24pp-en-fr",
    pretty_name="WMT24++-fr",
    source="EuroEval/wmt24pp-en-fr",
    task=TRANSLATION,
    languages=[ENGLISH, FRENCH],
    unofficial=True,
)

WMT24PP_EN_HR_CONFIG = DatasetConfig(
    name="wmt24pp-en-hr",
    pretty_name="WMT24++-hr",
    source="EuroEval/wmt24pp-en-hr",
    task=TRANSLATION,
    languages=[ENGLISH, CROATIAN],
    unofficial=True,
)

WMT24PP_EN_HU_CONFIG = DatasetConfig(
    name="wmt24pp-en-hu",
    pretty_name="WMT24++-hu",
    source="EuroEval/wmt24pp-en-hu",
    task=TRANSLATION,
    languages=[ENGLISH, HUNGARIAN],
    unofficial=True,
)

WMT24PP_EN_IS_CONFIG = DatasetConfig(
    name="wmt24pp-en-is",
    pretty_name="WMT24++-is",
    source="EuroEval/wmt24pp-en-is",
    task=TRANSLATION,
    languages=[ENGLISH, ICELANDIC],
    unofficial=True,
)

WMT24PP_EN_IT_CONFIG = DatasetConfig(
    name="wmt24pp-en-it",
    pretty_name="WMT24++-it",
    source="EuroEval/wmt24pp-en-it",
    task=TRANSLATION,
    languages=[ENGLISH, ITALIAN],
    unofficial=True,
)

WMT24PP_EN_LT_CONFIG = DatasetConfig(
    name="wmt24pp-en-lt",
    pretty_name="WMT24++-lt",
    source="EuroEval/wmt24pp-en-lt",
    task=TRANSLATION,
    languages=[ENGLISH, LITHUANIAN],
    unofficial=True,
)

WMT24PP_EN_LV_CONFIG = DatasetConfig(
    name="wmt24pp-en-lv",
    pretty_name="WMT24++-lv",
    source="EuroEval/wmt24pp-en-lv",
    task=TRANSLATION,
    languages=[ENGLISH, LATVIAN],
    unofficial=True,
)

WMT24PP_EN_NL_CONFIG = DatasetConfig(
    name="wmt24pp-en-nl",
    pretty_name="WMT24++-nl",
    source="EuroEval/wmt24pp-en-nl",
    task=TRANSLATION,
    languages=[ENGLISH, DUTCH],
    unofficial=True,
)

WMT24PP_EN_NO_CONFIG = DatasetConfig(
    name="wmt24pp-en-no",
    pretty_name="WMT24++-no",
    source="EuroEval/wmt24pp-en-no",
    task=TRANSLATION,
    languages=[ENGLISH, NORWEGIAN],
    unofficial=True,
)

WMT24PP_EN_PL_CONFIG = DatasetConfig(
    name="wmt24pp-en-pl",
    pretty_name="WMT24++-pl",
    source="EuroEval/wmt24pp-en-pl",
    task=TRANSLATION,
    languages=[ENGLISH, POLISH],
    unofficial=True,
)

WMT24PP_EN_PT_CONFIG = DatasetConfig(
    name="wmt24pp-en-pt",
    pretty_name="WMT24++-pt",
    source="EuroEval/wmt24pp-en-pt",
    task=TRANSLATION,
    languages=[ENGLISH, PORTUGUESE],
    unofficial=True,
)

WMT24PP_EN_RO_CONFIG = DatasetConfig(
    name="wmt24pp-en-ro",
    pretty_name="WMT24++-ro",
    source="EuroEval/wmt24pp-en-ro",
    task=TRANSLATION,
    languages=[ENGLISH, ROMANIAN],
    unofficial=True,
)

WMT24PP_EN_SK_CONFIG = DatasetConfig(
    name="wmt24pp-en-sk",
    pretty_name="WMT24++-sk",
    source="EuroEval/wmt24pp-en-sk",
    task=TRANSLATION,
    languages=[ENGLISH, SLOVAK],
    unofficial=True,
)

WMT24PP_EN_SL_CONFIG = DatasetConfig(
    name="wmt24pp-en-sl",
    pretty_name="WMT24++-sl",
    source="EuroEval/wmt24pp-en-sl",
    task=TRANSLATION,
    languages=[ENGLISH, SLOVENE],
    unofficial=True,
)

WMT24PP_EN_SR_CONFIG = DatasetConfig(
    name="wmt24pp-en-sr",
    pretty_name="WMT24++-sr",
    source="EuroEval/wmt24pp-en-sr",
    task=TRANSLATION,
    languages=[ENGLISH, SERBIAN],
    unofficial=True,
)

WMT24PP_EN_SV_CONFIG = DatasetConfig(
    name="wmt24pp-en-sv",
    pretty_name="WMT24++-sv",
    source="EuroEval/wmt24pp-en-sv",
    task=TRANSLATION,
    languages=[ENGLISH, SWEDISH],
    unofficial=True,
)

WMT24PP_EN_UK_CONFIG = DatasetConfig(
    name="wmt24pp-en-uk",
    pretty_name="WMT24++-uk",
    source="EuroEval/wmt24pp-en-uk",
    task=TRANSLATION,
    languages=[ENGLISH, UKRAINIAN],
    unofficial=True,
)
