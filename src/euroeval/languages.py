"""List of languages and their ISO 639-1 codes.

Taken from https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes.

Last updated 19 June 2022.
"""

from .data_models import Language


def get_all_languages() -> dict[str, Language]:
    """Get a list of all the languages.

    Returns:
        A mapping between language codes and their configurations.
    """
    return {cfg.code: cfg for cfg in globals().values() if isinstance(cfg, Language)}


### Currently Supported Lanuages ###
DA = Language(code="da", name="Danish", _and_separator="og", _or_separator="eller")
NL = Language(code="nl", name="Dutch", _and_separator="en", _or_separator="of")
EN = Language(code="en", name="English", _and_separator="and", _or_separator="or")
FO = Language(code="fo", name="Faroese", _and_separator="og", _or_separator="ella")
FR = Language(code="fr", name="French", _and_separator="et", _or_separator="ou")
DE = Language(code="de", name="German", _and_separator="und", _or_separator="oder")
IS = Language(code="is", name="Icelandic", _and_separator="og", _or_separator="eða")
IT = Language(code="it", name="Italian", _and_separator="e", _or_separator="o")
NO = Language(code="no", name="Norwegian", _and_separator="og", _or_separator="eller")
NB = Language(
    code="nb", name="Norwegian Bokmål", _and_separator="og", _or_separator="eller"
)
NN = Language(
    code="nn", name="Norwegian Nynorsk", _and_separator="og", _or_separator="eller"
)
ES = Language(code="es", name="Spanish", _and_separator="y", _or_separator="o")
SV = Language(code="sv", name="Swedish", _and_separator="och", _or_separator="eller")
### Currently Supported Languages ###

AB = Language(code="ab", name="Abkhazian")
AA = Language(code="aa", name="Afar")
AF = Language(code="af", name="Afrikaans")
SQ = Language(code="sq", name="Albanian")
AM = Language(code="am", name="Amharic")
AR = Language(code="ar", name="Arabic")
AN = Language(code="an", name="Aragonese")
HY = Language(code="hy", name="Armenian")
AS = Language(code="as", name="Assamese")
AV = Language(code="av", name="Avaric")
AE = Language(code="ae", name="Avestan")
AY = Language(code="ay", name="Aymara")
AZ = Language(code="az", name="Azerbaijani")
BM = Language(code="bm", name="Bambara")
BA = Language(code="ba", name="Bashkir")
EU = Language(code="eu", name="Basque")
BE = Language(code="be", name="Belarusian")
BN = Language(code="bn", name="Bengali")
BI = Language(code="bi", name="Bislama")
BS = Language(code="bs", name="Bosnian")
BR = Language(code="br", name="Breton")
BG = Language(code="bg", name="Bulgarian")
MY = Language(code="my", name="Burmese")
CA = Language(code="ca", name="Catalan")
CH = Language(code="ch", name="Chamorro")
CE = Language(code="ce", name="Chechen")
NY = Language(code="ny", name="Chichewa")
ZH = Language(code="zh", name="Chinese")
CU = Language(code="cu", name="Church Slavic")
CV = Language(code="cv", name="Chuvash")
KW = Language(code="kw", name="Cornish")
CO = Language(code="co", name="Corsican")
CR = Language(code="cr", name="Cree")
HR = Language(code="hr", name="Croatian")
CS = Language(code="cs", name="Czech")
DV = Language(code="dv", name="Divehi")
DZ = Language(code="dz", name="Dzongkha")
EO = Language(code="eo", name="Esperanto")
ET = Language(code="et", name="Estonian")
EE = Language(code="ee", name="Ewe")
FJ = Language(code="fj", name="Fijian")
FI = Language(code="fi", name="Finnish")
FY = Language(code="fy", name="Western Frisian")
FF = Language(code="ff", name="Fulah")
GD = Language(code="gd", name="Gaelic")
GL = Language(code="gl", name="Galician")
LG = Language(code="lg", name="Ganda")
KA = Language(code="ka", name="Georgian")
EL = Language(code="el", name="Greek")
KL = Language(code="kl", name="Greenlandic")
GN = Language(code="gn", name="Guarani")
GU = Language(code="gu", name="Gujarati")
HT = Language(code="ht", name="Haitian")
HA = Language(code="ha", name="Hausa")
HE = Language(code="he", name="Hebrew")
HZ = Language(code="hz", name="Herero")
HI = Language(code="hi", name="Hindi")
HO = Language(code="ho", name="Hiri Motu")
HU = Language(code="hu", name="Hungarian")
IO = Language(code="io", name="Ido")
IG = Language(code="ig", name="Igbo")
ID = Language(code="id", name="Indonesian")
IA = Language(code="ia", name="Interlingua")
IE = Language(code="ie", name="Interlingue")
IU = Language(code="iu", name="Inuktitut")
IK = Language(code="ik", name="Inupiaq")
GA = Language(code="ga", name="Irish")
JA = Language(code="ja", name="Japanese")
KN = Language(code="kn", name="Kannada")
KR = Language(code="kr", name="Kanuri")
KS = Language(code="ks", name="Kashmiri")
KK = Language(code="kk", name="Kazakh")
KM = Language(code="km", name="Central Khmer")
KI = Language(code="ki", name="Kikuyu")
RW = Language(code="rw", name="Kinyarwanda")
KY = Language(code="ky", name="Kirghiz")
KV = Language(code="kv", name="Komi")
KG = Language(code="kg", name="Kongo")
KO = Language(code="ko", name="Korean")
KJ = Language(code="kj", name="Kuanyama")
KU = Language(code="ku", name="Kurdish")
LO = Language(code="lo", name="Lao")
LA = Language(code="la", name="Latin")
LV = Language(code="lv", name="Latvian")
LI = Language(code="li", name="Limburgan")
LN = Language(code="ln", name="Lingala")
LT = Language(code="lt", name="Lithuanian")
LU = Language(code="lu", name="Luba-Katanga")
LB = Language(code="lb", name="Luxembourgish")
MK = Language(code="mk", name="Macedonian")
MG = Language(code="mg", name="Malagasy")
MS = Language(code="ms", name="Malay")
ML = Language(code="ml", name="Malayalam")
MT = Language(code="mt", name="Maltese")
GV = Language(code="gv", name="Manx")
MI = Language(code="mi", name="Maori")
MR = Language(code="mr", name="Marathi")
MH = Language(code="mh", name="Marshallese")
MN = Language(code="mn", name="Mongolian")
NA = Language(code="na", name="Nauru")
NV = Language(code="nv", name="Navajo")
ND = Language(code="nd", name="Northern Ndebele")
NR = Language(code="nr", name="South Ndebele")
NG = Language(code="ng", name="Ndonga")
NE = Language(code="ne", name="Nepali")
II = Language(code="ii", name="Sichuan Yi")
OC = Language(code="oc", name="Occitan")
OJ = Language(code="oj", name="Ojibwa")
OR = Language(code="or", name="Oriya")
OM = Language(code="om", name="Oromo")
OS = Language(code="os", name="Ossetian")
PI = Language(code="pi", name="Pali")
PS = Language(code="ps", name="Pashto")
FA = Language(code="fa", name="Persian")
PL = Language(code="pl", name="Polish")
PT = Language(code="pt", name="Portuguese")
PA = Language(code="pa", name="Punjabi")
QU = Language(code="qu", name="Quechua")
RO = Language(code="ro", name="Romanian")
RM = Language(code="rm", name="Romansh")
RN = Language(code="rn", name="Rundi")
RU = Language(code="ru", name="Russian")
SE = Language(code="se", name="Northern Sami")
SM = Language(code="sm", name="Samoan")
SG = Language(code="sg", name="Sango")
SA = Language(code="sa", name="Sanskrit")
SC = Language(code="sc", name="Sardinian")
SR = Language(code="sr", name="Serbian")
SN = Language(code="sn", name="Shona")
SD = Language(code="sd", name="Sindhi")
SI = Language(code="si", name="Sinhala")
SK = Language(code="sk", name="Slovak")
SL = Language(code="sl", name="Slovenian")
SO = Language(code="so", name="Somali")
ST = Language(code="st", name="Sotho")
SU = Language(code="su", name="Sundanese")
SW = Language(code="sw", name="Swahili")
SS = Language(code="ss", name="Swati")
TL = Language(code="tl", name="Tagalog")
TY = Language(code="ty", name="Tahitian")
TG = Language(code="tg", name="Tajik")
TA = Language(code="ta", name="Tamil")
TT = Language(code="tt", name="Tatar")
TE = Language(code="te", name="Telugu")
TH = Language(code="th", name="Thai")
BO = Language(code="bo", name="Tibetan")
TI = Language(code="ti", name="Tigrinya")
TO = Language(code="to", name="Tonga")
TS = Language(code="ts", name="Tsonga")
TN = Language(code="tn", name="Tswana")
TR = Language(code="tr", name="Turkish")
TK = Language(code="tk", name="Turkmen")
TW = Language(code="tw", name="Twi")
UG = Language(code="ug", name="Uighur")
UK = Language(code="uk", name="Ukrainian")
UR = Language(code="ur", name="Urdu")
UZ = Language(code="uz", name="Uzbek")
VE = Language(code="ve", name="Venda")
VI = Language(code="vi", name="Vietnamese")
VO = Language(code="vo", name="Volapük")
WA = Language(code="wa", name="Walloon")
CY = Language(code="cy", name="Welsh")
WO = Language(code="wo", name="Wolof")
XH = Language(code="xh", name="Xhosa")
YI = Language(code="yi", name="Yiddish")
YO = Language(code="yo", name="Yoruba")
ZA = Language(code="za", name="Zhuang")
ZU = Language(code="zu", name="Zulu")
