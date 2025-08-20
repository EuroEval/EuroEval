"""List of languages and their language codes.

The language codes contain both all the ISO 639-1 codes, as well as the ISO 639-3 codes
for languages that do not have an ISO 639-1 code.
"""

from .data_models import Language


def get_all_languages() -> dict[str, Language]:
    """Get a list of all the languages.

    Returns:
        A mapping between language codes and their configurations.
    """
    return {cfg.code: cfg for cfg in globals().values() if isinstance(cfg, Language)}


AB = Language(code="ab", name="Abkhazian", _and_separator="и", _or_separator="ма")
AA = Language(code="aa", name="Afar", _and_separator="kee", _or_separator="maleey")
AF = Language(code="af", name="Afrikaans", _and_separator="en", _or_separator="of")
SQ = Language(code="sq", name="Albanian", _and_separator="dhe", _or_separator="ose")
AM = Language(code="am", name="Amharic", _and_separator="እና", _or_separator="ወይም")
AR = Language(code="ar", name="Arabic", _and_separator="و", _or_separator="أو")
AN = Language(code="an", name="Aragonese", _and_separator="y", _or_separator="u")
HY = Language(code="hy", name="Armenian", _and_separator="և", _or_separator="կամ")
AS = Language(code="as", name="Assamese", _and_separator="আৰু", _or_separator="বা")
AV = Language(code="av", name="Avaric", _and_separator="ги", _or_separator="яги")
AE = Language(code="ae", name="Avestan", _and_separator="utā", _or_separator="vā")
AY = Language(code="ay", name="Aymara", _and_separator="-mpi", _or_separator="jan ukax")
AZ = Language(code="az", name="Azerbaijani", _and_separator="və", _or_separator="və ya")
BM = Language(code="bm", name="Bambara", _and_separator="ani", _or_separator="walima")
BA = Language(code="ba", name="Bashkir", _and_separator="һәм", _or_separator="йәки")
EU = Language(code="eu", name="Basque", _and_separator="eta", _or_separator="edo")
BE = Language(code="be", name="Belarusian", _and_separator="і", _or_separator="або")
BN = Language(code="bn", name="Bengali", _and_separator="এবং", _or_separator="অথবা")
BI = Language(code="bi", name="Bislama", _and_separator="mo", _or_separator="o")
BS = Language(code="bs", name="Bosnian", _and_separator="i", _or_separator="ili")
BR = Language(code="br", name="Breton", _and_separator="ha", _or_separator="pe")
BG = Language(code="bg", name="Bulgarian", _and_separator="и", _or_separator="или")
MY = Language(code="my", name="Burmese", _and_separator="နှင့်", _or_separator="သို့မဟုတ်")
CA = Language(code="ca", name="Catalan", _and_separator="i", _or_separator="o")
CH = Language(code="ch", name="Chamorro", _and_separator="yan", _or_separator="pat")
CE = Language(code="ce", name="Chechen", _and_separator="а", _or_separator="я")
NY = Language(code="ny", name="Chichewa", _and_separator="ndi", _or_separator="kapena")
ZH = Language(
    code="zh", name="Simplified Chinese", _and_separator="和", _or_separator="或"
)
ZH_CN = Language(
    code="zh-cn", name="Simplified Chinese", _and_separator="和", _or_separator="或"
)
ZH_TW = Language(
    code="zh-tw", name="Traditional Chinese", _and_separator="與", _or_separator="或"
)
CU = Language(code="cu", name="Church Slavic", _and_separator="и", _or_separator="или")
CV = Language(code="cv", name="Chuvash", _and_separator="тата", _or_separator="е")
KW = Language(code="kw", name="Cornish", _and_separator="ha", _or_separator="po")
CO = Language(code="co", name="Corsican", _and_separator="e", _or_separator="o")
CR = Language(code="cr", name="Cree", _and_separator="ēkwa", _or_separator="kamāc")
HR = Language(code="hr", name="Croatian", _and_separator="i", _or_separator="ili")
CS = Language(code="cs", name="Czech", _and_separator="a", _or_separator="nebo")
DA = Language(code="da", name="Danish", _and_separator="og", _or_separator="eller")
NL = Language(code="nl", name="Dutch", _and_separator="en", _or_separator="of")
DV = Language(code="dv", name="Divehi", _and_separator="އަދި", _or_separator="ނުވަތަ")
DZ = Language(code="dz", name="Dzongkha", _and_separator="དང་", _or_separator="ཡང་མེན")
EN = Language(code="en", name="English", _and_separator="and", _or_separator="or")
EO = Language(code="eo", name="Esperanto", _and_separator="kaj", _or_separator="aŭ")
ET = Language(code="et", name="Estonian", _and_separator="ja", _or_separator="või")
EE = Language(code="ee", name="Ewe", _and_separator="kple", _or_separator="alo")
FO = Language(code="fo", name="Faroese", _and_separator="og", _or_separator="ella")
FJ = Language(code="fj", name="Fijian", _and_separator="kei", _or_separator="se")
FI = Language(code="fi", name="Finnish", _and_separator="ja", _or_separator="tai")
FR = Language(code="fr", name="French", _and_separator="et", _or_separator="ou")
FY = Language(
    code="fy", name="Western Frisian", _and_separator="en", _or_separator="of"
)
FF = Language(code="ff", name="Fulah", _and_separator="e", _or_separator="ma")
GD = Language(code="gd", name="Gaelic", _and_separator="agus", _or_separator="no")
GL = Language(code="gl", name="Galician", _and_separator="e", _or_separator="ou")
LG = Language(code="lg", name="Ganda", _and_separator="ne", _or_separator="oba")
KA = Language(code="ka", name="Georgian", _and_separator="და", _or_separator="ან")
DE = Language(code="de", name="German", _and_separator="und", _or_separator="oder")
EL = Language(code="el", name="Greek", _and_separator="και", _or_separator="ή")
KL = Language(
    code="kl", name="Greenlandic", _and_separator="aamma", _or_separator="imaluunniit"
)
GN = Language(code="gn", name="Guarani", _and_separator="ha", _or_separator="térã")
GU = Language(code="gu", name="Gujarati", _and_separator="અને", _or_separator="અથવા")
HT = Language(code="ht", name="Haitian", _and_separator="ak", _or_separator="oswa")
HA = Language(code="ha", name="Hausa", _and_separator="da", _or_separator="ko")
HE = Language(code="he", name="Hebrew", _and_separator="ו", _or_separator="או")
HZ = Language(code="hz", name="Herero", _and_separator="na", _or_separator="po")
HI = Language(code="hi", name="Hindi", _and_separator="और", _or_separator="या")
HU = Language(code="hu", name="Hungarian", _and_separator="és", _or_separator="vagy")
IS = Language(code="is", name="Icelandic", _and_separator="og", _or_separator="eða")
IO = Language(code="io", name="Ido", _and_separator="e", _or_separator="o")
IG = Language(code="ig", name="Igbo", _and_separator="na", _or_separator="ma ọ bụ")
ID = Language(code="id", name="Indonesian", _and_separator="dan", _or_separator="atau")
IA = Language(code="ia", name="Interlingua", _and_separator="e", _or_separator="o")
IE = Language(code="ie", name="Interlingue", _and_separator="e", _or_separator="o")
IU = Language(
    code="iu", name="Inuktitut", _and_separator="alu", _or_separator="immaqaa"
)
IK = Language(code="ik", name="Inupiaq", _and_separator="ġu", _or_separator="luunniit")
GA = Language(code="ga", name="Irish", _and_separator="agus", _or_separator="nó")
IT = Language(code="it", name="Italian", _and_separator="e", _or_separator="o")
JA = Language(code="ja", name="Japanese", _and_separator="と", _or_separator="または")
KN = Language(code="kn", name="Kannada", _and_separator="ಮತ್ತು", _or_separator="ಅಥವಾ")
KR = Language(code="kr", name="Kanuri", _and_separator="-a", _or_separator="yáá")
KS = Language(code="ks", name="Kashmiri", _and_separator="تہٕ", _or_separator="یا")
KK = Language(code="kk", name="Kazakh", _and_separator="және", _or_separator="немесе")
KM = Language(code="km", name="Central Khmer", _and_separator="និង", _or_separator="ឬ")
KI = Language(code="ki", name="Kikuyu", _and_separator="na", _or_separator="kana")
RW = Language(
    code="rw", name="Kinyarwanda", _and_separator="na", _or_separator="cyangwa"
)
KY = Language(code="ky", name="Kirghiz", _and_separator="жана", _or_separator="же")
KV = Language(code="kv", name="Komi", _and_separator="да", _or_separator="либӧ")
KG = Language(code="kg", name="Kongo", _and_separator="ye", _or_separator="kana")
KO = Language(code="ko", name="Korean", _and_separator="그리고", _or_separator="또는")
KJ = Language(code="kj", name="Kuanyama", _and_separator="na", _or_separator="nenge")
KU = Language(code="ku", name="Kurdish", _and_separator="û", _or_separator="an")
LO = Language(code="lo", name="Lao", _and_separator="และ", _or_separator="ຫຼື")
LA = Language(code="la", name="Latin", _and_separator="et", _or_separator="aut")
LV = Language(code="lv", name="Latvian", _and_separator="un", _or_separator="vai")
LI = Language(code="li", name="Limburgan", _and_separator="en", _or_separator="of")
LN = Language(code="ln", name="Lingala", _and_separator="na", _or_separator="to")
LT = Language(code="lt", name="Lithuanian", _and_separator="ir", _or_separator="arba")
LU = Language(
    code="lu", name="Luba-Katanga", _and_separator="ne", _or_separator="nansha"
)
LB = Language(
    code="lb", name="Luxembourgish", _and_separator="an", _or_separator="oder"
)
MK = Language(code="mk", name="Macedonian", _and_separator="и", _or_separator="или")
MG = Language(code="mg", name="Malagasy", _and_separator="sy", _or_separator="na")
MS = Language(code="ms", name="Malay", _and_separator="dan", _or_separator="atau")
ML = Language(
    code="ml", name="Malayalam", _and_separator="ഉം", _or_separator="അല്ലെങ്കിൽ"
)
MT = Language(code="mt", name="Maltese", _and_separator="u", _or_separator="jew")
GV = Language(code="gv", name="Manx", _and_separator="as", _or_separator="ny")
MI = Language(code="mi", name="Maori", _and_separator="me", _or_separator="rānei")
MR = Language(code="mr", name="Marathi", _and_separator="आणि", _or_separator="किंवा")
MH = Language(code="mh", name="Marshallese", _and_separator="im", _or_separator="ak")
MN = Language(code="mn", name="Mongolian", _and_separator="ба", _or_separator="эсвэл")
NA = Language(code="na", name="Nauru", _and_separator="ma", _or_separator="me")
NV = Language(
    code="nv", name="Navajo", _and_separator="áádóó", _or_separator="doodaiiʼ"
)
ND = Language(
    code="nd", name="Northern Ndebele", _and_separator="lo", _or_separator="kumbe"
)
NR = Language(
    code="nr", name="South Ndebele", _and_separator="na", _or_separator="namkha"
)
NG = Language(code="ng", name="Ndonga", _and_separator="na", _or_separator="nenge")
NE = Language(code="ne", name="Nepali", _and_separator="र", _or_separator="वा")
NO = Language(code="no", name="Norwegian", _and_separator="og", _or_separator="eller")
NB = Language(
    code="nb", name="Norwegian Bokmål", _and_separator="og", _or_separator="eller"
)
NN = Language(
    code="nn", name="Norwegian Nynorsk", _and_separator="og", _or_separator="eller"
)
OC = Language(code="oc", name="Occitan", _and_separator="e", _or_separator="o")
OJ = Language(
    code="oj", name="Ojibwa", _and_separator="miinawaa", _or_separator="jiishin"
)
OR = Language(code="or", name="Oriya", _and_separator="ଏବଂ", _or_separator="କିମ୍ବା")
OM = Language(code="om", name="Oromo", _and_separator="fi", _or_separator="yookan")
OS = Language(code="os", name="Ossetian", _and_separator="æмæ", _or_separator="кæнæ")
PI = Language(code="pi", name="Pali", _and_separator="ca", _or_separator="vā")
PS = Language(code="ps", name="Pashto", _and_separator="او", _or_separator="يا")
FA = Language(code="fa", name="Persian", _and_separator="و", _or_separator="یا")
PL = Language(code="pl", name="Polish", _and_separator="i", _or_separator="lub")
PT = Language(
    code="pt", name="European Portuguese", _and_separator="e", _or_separator="ou"
)
PT_PT = Language(
    code="pt-pt", name="European Portuguese", _and_separator="e", _or_separator="ou"
)
PT_BR = Language(
    code="pt-br", name="Brazilian Portuguese", _and_separator="e", _or_separator="ou"
)
PA = Language(code="pa", name="Punjabi", _and_separator="ਅਤੇ", _or_separator="ਜਾਂ")
QU = Language(code="qu", name="Quechua", _and_separator="-pas", _or_separator="ichataq")
RO = Language(code="ro", name="Romanian", _and_separator="și", _or_separator="sau")
RM = Language(code="rm", name="Romansh", _and_separator="e", _or_separator="u")
RN = Language(code="rn", name="Rundi", _and_separator="na", _or_separator="canke")
RU = Language(code="ru", name="Russian", _and_separator="и", _or_separator="или")
SE = Language(
    code="se", name="Northern Sami", _and_separator="ja", _or_separator="dahje"
)
SM = Language(code="sm", name="Samoan", _and_separator="ma", _or_separator="poʻo")
SG = Language(code="sg", name="Sango", _and_separator="na", _or_separator="wala")
SA = Language(code="sa", name="Sanskrit", _and_separator="च", _or_separator="वा")
SC = Language(code="sc", name="Sardinian", _and_separator="e", _or_separator="o")
SR = Language(code="sr", name="Serbian", _and_separator="и", _or_separator="или")
SN = Language(code="sn", name="Shona", _and_separator="uye", _or_separator="kana")
SD = Language(code="sd", name="Sindhi", _and_separator="۽", _or_separator="يا")
SI = Language(code="si", name="Sinhala", _and_separator="සහ", _or_separator="හෝ")
SK = Language(code="sk", name="Slovak", _and_separator="a", _or_separator="alebo")
SL = Language(code="sl", name="Slovenian", _and_separator="in", _or_separator="ali")
SO = Language(code="so", name="Somali", _and_separator="iyo", _or_separator="ama")
ST = Language(code="st", name="Sotho", _and_separator="le", _or_separator="kapa")
ES = Language(code="es", name="Spanish", _and_separator="y", _or_separator="o")
SU = Language(
    code="su", name="Sundanese", _and_separator="jeung", _or_separator="atawa"
)
SW = Language(code="sw", name="Swahili", _and_separator="na", _or_separator="au")
SS = Language(code="ss", name="Swati", _and_separator="na", _or_separator="noma")
SV = Language(code="sv", name="Swedish", _and_separator="och", _or_separator="eller")
TL = Language(code="tl", name="Tagalog", _and_separator="at", _or_separator="o")
TY = Language(code="ty", name="Tahitian", _and_separator="e", _or_separator="aore ra")
TG = Language(code="tg", name="Tajik", _and_separator="ва", _or_separator="ё")
TA = Language(code="ta", name="Tamil", _and_separator="மற்றும்", _or_separator="அல்லது")
TT = Language(code="tt", name="Tatar", _and_separator="һәм", _or_separator="яки")
TE = Language(code="te", name="Telugu", _and_separator="మరియు", _or_separator="లేదా")
TH = Language(code="th", name="Thai", _and_separator="และ", _or_separator="หรือ")
BO = Language(code="bo", name="Tibetan", _and_separator="དང་", _or_separator="ཡང་ན")
TI = Language(code="ti", name="Tigrinya", _and_separator="ን", _or_separator="ወይ")
TO = Language(code="to", name="Tonga", _and_separator="mo", _or_separator="pe")
TS = Language(code="ts", name="Tsonga", _and_separator="na", _or_separator="kumbe")
TN = Language(code="tn", name="Tswana", _and_separator="le", _or_separator="kgotsa")
TR = Language(code="tr", name="Turkish", _and_separator="ve", _or_separator="veya")
TK = Language(code="tk", name="Turkmen", _and_separator="we", _or_separator="ýa-da")
TW = Language(code="tw", name="Twi", _and_separator="ne", _or_separator="anaa")
UG = Language(code="ug", name="Uighur", _and_separator="ۋە", _or_separator="ياكى")
UK = Language(code="uk", name="Ukrainian", _and_separator="і", _or_separator="або")
UR = Language(code="ur", name="Urdu", _and_separator="اور", _or_separator="یا")
UZ = Language(code="uz", name="Uzbek", _and_separator="va", _or_separator="yoki")
VE = Language(code="ve", name="Venda", _and_separator="na", _or_separator="kana")
VI = Language(code="vi", name="Vietnamese", _and_separator="và", _or_separator="hoặc")
VO = Language(code="vo", name="Volapük", _and_separator="e", _or_separator="u")
WA = Language(code="wa", name="Walloon", _and_separator="et", _or_separator="ou")
CY = Language(code="cy", name="Welsh", _and_separator="a", _or_separator="neu")
WO = Language(code="wo", name="Wolof", _and_separator="ak", _or_separator="walla")
XH = Language(code="xh", name="Xhosa", _and_separator="kwaye", _or_separator="okanye")
YI = Language(code="yi", name="Yiddish", _and_separator="און", _or_separator="אָדער")
YO = Language(code="yo", name="Yoruba", _and_separator="àti", _or_separator="tàbí")
ZA = Language(code="za", name="Zhuang", _and_separator="kae", _or_separator="aevih")
ZU = Language(code="zu", name="Zulu", _and_separator="futhi", _or_separator="noma")
ACE = Language(code="ace", name="Acehnese", _and_separator="ngon", _or_separator="atɔ")
ADY = Language(code="ady", name="Adyghe", _and_separator="рэ", _or_separator="е")
ALT = Language(
    code="alt", name="Southern Altai", _and_separator="ла", _or_separator="эмезе"
)
AMI = Language(code="ami", name="Amis", _and_separator="ato", _or_separator="o")
ANG = Language(
    code="ang", name="Old English", _and_separator="and", _or_separator="oþþe"
)
ANP = Language(code="anp", name="Angika", _and_separator="आर", _or_separator="या")
ARC = Language(code="arc", name="Aramaic", _and_separator="ܘ", _or_separator="ܐܘ")
ARY = Language(
    code="ary", name="Moroccan Arabic", _and_separator="w", _or_separator="wella"
)
ARZ = Language(
    code="arz", name="Egyptian Arabic", _and_separator="و", _or_separator="أو"
)
AST = Language(code="ast", name="Asturian", _and_separator="y", _or_separator="o")
ATJ = Language(code="atj", name="Atikamekw", _and_separator="et", _or_separator="ou")
AVK = Language(code="avk", name="Kotava", _and_separator="is", _or_separator="en")
AWA = Language(code="awa", name="Awadhi", _and_separator="अउ", _or_separator="या")
AZB = Language(
    code="azb", name="South Azerbaijani", _and_separator="و", _or_separator="یوخسا"
)
BAN = Language(code="ban", name="Balinese", _and_separator="lan", _or_separator="utawi")
BAR = Language(code="bar", name="Bavarian", _and_separator="und", _or_separator="oda")
BCL = Language(
    code="bcl", name="Central Bikol", _and_separator="asin", _or_separator="o"
)
BJN = Language(code="bjn", name="Banjar", _and_separator="wan", _or_separator="atawa")
BLK = Language(code="blk", name="Pa'O", _and_separator="နန်", _or_separator="မု")
BPY = Language(code="bpy", name="Bishnupriya", _and_separator="आ", _or_separator="বা")
BUG = Language(
    code="bug", name="Buginese", _and_separator="na", _or_separator="iyarega"
)
BXR = Language(code="bxr", name="Buriat", _and_separator="ба", _or_separator="али")
CDO = Language(
    code="cdo", name="Mindong Chinese", _and_separator="共", _or_separator="或者"
)
CEB = Language(code="ceb", name="Cebuano", _and_separator="ug", _or_separator="o")
CHR = Language(code="chr", name="Cherokee", _and_separator="ᎠᎴ", _or_separator="ᎠᎴ")
CHY = Language(code="chy", name="Cheyenne", _and_separator="na", _or_separator="hēme")
CKB = Language(
    code="ckb", name="Central Kurdish", _and_separator="و", _or_separator="یان"
)
CRH = Language(
    code="crh", name="Crimean Tatar", _and_separator="ve", _or_separator="ya da"
)
CSB = Language(code="csb", name="Kashubian", _and_separator="ë", _or_separator="abò")
DAG = Language(code="dag", name="Dagbani", _and_separator="n-ti", _or_separator="bee")
DIN = Language(code="din", name="Dinka", _and_separator="ka", _or_separator="ke")
DIQ = Language(code="diq", name="Dimli", _and_separator="û", _or_separator="ya")
DSB = Language(
    code="dsb", name="Lower Sorbian", _and_separator="a", _or_separator="abo"
)
DTY = Language(code="dty", name="Doteli", _and_separator="र", _or_separator="या")
EXT = Language(code="ext", name="Extremaduran", _and_separator="y", _or_separator="u")
FAT = Language(code="fat", name="Fanti", _and_separator="na", _or_separator="anaa")
FON = Language(code="fon", name="Fon", _and_separator="kpóɖó", _or_separator="kabi")
FRP = Language(code="frp", name="Arpitan", _and_separator="et", _or_separator="ou")
FRR = Language(
    code="frr", name="Northern Frisian", _and_separator="an", _or_separator="of"
)
FUR = Language(code="fur", name="Friulian", _and_separator="e", _or_separator="o")
GAG = Language(code="gag", name="Gagauz", _and_separator="hem", _or_separator="ya")
GAN = Language(code="gan", name="Gan Chinese", _and_separator="同", _or_separator="或")
GCR = Language(
    code="gcr", name="Guianan Creole", _and_separator="ké", _or_separator="ou"
)
GLK = Language(code="glk", name="Gilaki", _and_separator="و", _or_separator="یا")
GOM = Language(
    code="gom", name="Goan Konkani", _and_separator="आनी", _or_separator="वा"
)
GOR = Language(
    code="gor", name="Gorontalo", _and_separator="wawu", _or_separator="meyalo"
)
GOT = Language(code="got", name="Gothic", _and_separator="jah", _or_separator="aiþþau")
GPE = Language(
    code="gpe", name="Ghanaian Pidgin", _and_separator="and", _or_separator="anaa"
)
GUC = Language(code="guc", name="Wayuu", _and_separator="je", _or_separator="yaa")
GUR = Language(code="gur", name="Frafra", _and_separator="la", _or_separator="bee")
GUW = Language(code="guw", name="Gun", _and_separator="pódó", _or_separator="yèkì")
HAK = Language(
    code="hak", name="Hakka Chinese", _and_separator="同", _or_separator="或者"
)
HAW = Language(
    code="haw", name="Hawaiian", _and_separator="a", _or_separator="a i ʻole"
)
HIF = Language(code="hif", name="Fiji Hindi", _and_separator="aur", _or_separator="ki")
HSB = Language(
    code="hsb", name="Upper Sorbian", _and_separator="a", _or_separator="abo"
)
HYW = Language(
    code="hyw", name="Western Armenian", _and_separator="եւ", _or_separator="կամ"
)
ILO = Language(code="ilo", name="Iloko", _and_separator="ken", _or_separator="wenno")
INH = Language(code="inh", name="Ingush", _and_separator="и", _or_separator="е")
JAM = Language(
    code="jam", name="Jamaican Creole", _and_separator="an", _or_separator="ar"
)
JBO = Language(code="jbo", name="Lojban", _and_separator="e", _or_separator="a")
KAA = Language(
    code="kaa", name="Kara-Kalpak", _and_separator="ha'm", _or_separator="yamasa"
)
KAB = Language(code="kab", name="Kabyle", _and_separator="d", _or_separator="neɣ")
KBD = Language(code="kbd", name="Kabardian", _and_separator="рэ", _or_separator="хэтӀэ")
KBP = Language(code="kbp", name="Kabiyè", _and_separator="nɛ", _or_separator="yaa")
KCG = Language(code="kcg", name="Tyap", _and_separator="ma", _or_separator="a̠ni")
KOI = Language(
    code="koi", name="Komi-Permyak", _and_separator="да", _or_separator="либӧ"
)
KRC = Language(
    code="krc", name="Karachay-Balkar", _and_separator="бла", _or_separator="не да"
)
LAD = Language(code="lad", name="Ladino", _and_separator="i", _or_separator="o")
LBE = Language(code="lbe", name="Lak", _and_separator="ва", _or_separator="ягу")
LEZ = Language(code="lez", name="Lezghian", _and_separator="ва", _or_separator="я")
LFN = Language(
    code="lfn", name="Lingua Franca Nova", _and_separator="e", _or_separator="o"
)
LIJ = Language(code="lij", name="Ligurian", _and_separator="e", _or_separator="ò")
LLD = Language(code="lld", name="Ladin", _and_separator="y", _or_separator="o")
LMO = Language(code="lmo", name="Lombard", _and_separator="e", _or_separator="o")
LTG = Language(code="ltg", name="Latgalian", _and_separator="i", _or_separator="voi")
MAD = Language(code="mad", name="Madurese", _and_separator="ban", _or_separator="o")
MAI = Language(code="mai", name="Maithili", _and_separator="आ", _or_separator="वा")
MDF = Language(code="mdf", name="Moksha", _and_separator="ди", _or_separator="или")
MHR = Language(code="mhr", name="Eastern Mari", _and_separator="да", _or_separator="o")
MIN = Language(
    code="min", name="Minangkabau", _and_separator="jo", _or_separator="atau"
)
MNI = Language(
    code="mni", name="Manipuri", _and_separator="ꯑꯃꯁꯨꯡ", _or_separator="ꯅꯠꯇ꯭ꯔꯒꯥ"
)
MNW = Language(code="mnw", name="Mon", _and_separator="ကဵု", _or_separator="ဟွံသေင်မ္ဂး")
MRJ = Language(code="mrj", name="Western Mari", _and_separator="да", _or_separator="o")
MWL = Language(code="mwl", name="Mirandese", _and_separator="i", _or_separator="ó")
MYV = Language(code="myv", name="Erzya", _and_separator="ды", _or_separator="эли")
MZN = Language(code="mzn", name="Mazanderani", _and_separator="و", _or_separator="یا")
NAP = Language(code="nap", name="Neapolitan", _and_separator="e", _or_separator="o")
NDS = Language(code="nds", name="Low German", _and_separator="un", _or_separator="oder")
NEW = Language(code="new", name="Newari", _and_separator="व", _or_separator="या")
NIA = Language(code="nia", name="Nias", _and_separator="ba", _or_separator="mazi")
NOV = Language(code="nov", name="Novial", _and_separator="e", _or_separator="o")
NQO = Language(code="nqo", name="N'Ko", _and_separator="ߣߌ߫", _or_separator="ߥߟߊ߫")
NSO = Language(
    code="nso", name="Northern Sotho", _and_separator="le", _or_separator="goba"
)
OLO = Language(
    code="olo", name="Livvi-Karelian", _and_separator="da", _or_separator="libo"
)
PAG = Language(
    code="pag", name="Pangasinan", _and_separator="tan", _or_separator="odino"
)
PAM = Language(code="pam", name="Pampanga", _and_separator="at", _or_separator="o")
PAP = Language(code="pap", name="Papiamento", _and_separator="y", _or_separator="o")
PCD = Language(code="pcd", name="Picard", _and_separator="pi", _or_separator="ou")
PCM = Language(
    code="pcm", name="Nigerian Pidgin", _and_separator="and", _or_separator="abi"
)
PDC = Language(
    code="pdc", name="Pennsylvania German", _and_separator="un", _or_separator="odder"
)
PFL = Language(
    code="pfl", name="Palatine German", _and_separator="un", _or_separator="odda"
)
PMS = Language(code="pms", name="Piemontese", _and_separator="e", _or_separator="o")
PNB = Language(
    code="pnb", name="Western Punjabi", _and_separator="تے", _or_separator="یا"
)
PNT = Language(code="pnt", name="Pontic", _and_separator="και", _or_separator="ή")
PWN = Language(code="pwn", name="Paiwan", _and_separator="dja", _or_separator="uri")
RMY = Language(
    code="rmy", name="Vlax Romani", _and_separator="thaj", _or_separator="vaj"
)
RUE = Language(code="rue", name="Rusyn", _and_separator="і", _or_separator="або")
SAH = Language(code="sah", name="Yakut", _and_separator="уонна", _or_separator="эбэтэр")
SAT = Language(code="sat", name="Santali", _and_separator="ᱟᱨ", _or_separator="ᱥᱮ")
SCN = Language(code="scn", name="Sicilian", _and_separator="e", _or_separator="o")
SCO = Language(code="sco", name="Scots", _and_separator="an", _or_separator="or")
SHI = Language(code="shi", name="Tachelhit", _and_separator="d", _or_separator="neɣ")
SHN = Language(code="shn", name="Shan", _and_separator="လႄႈ", _or_separator="หรือ")
SKR = Language(code="skr", name="Saraiki", _and_separator="تے", _or_separator="یا")
SMN = Language(
    code="smn", name="Inari Sami", _and_separator="ja", _or_separator="teikkâ"
)
SRN = Language(code="srn", name="Sranan", _and_separator="nanga", _or_separator="efu")
STQ = Language(
    code="stq", name="Saterland Frisian", _and_separator="un", _or_separator="of"
)
SZL = Language(code="szl", name="Silesian", _and_separator="a", _or_separator="abo")
SZY = Language(code="szy", name="Sakizaya", _and_separator="ata", _or_separator="uduli")
TAY = Language(code="tay", name="Atayal", _and_separator="daha", _or_separator="ima")
TCY = Language(code="tcy", name="Tulu", _and_separator="ಬೊಕ್ಕ", _or_separator="ಅತ್ತಂಡ")
TET = Language(code="tet", name="Tetum", _and_separator="no", _or_separator="ka")
TLY = Language(code="tly", name="Talysh", _and_separator="u", _or_separator="jo")
TPI = Language(code="tpi", name="Tok Pisin", _and_separator="na", _or_separator="o")
TRV = Language(code="trv", name="Taroko", _and_separator="daha", _or_separator="ima")
TUM = Language(code="tum", name="Tumbuka", _and_separator="na", _or_separator="panji")
TYV = Language(
    code="tyv", name="Tuvinian", _and_separator=" болгаш", _or_separator="азы"
)
UDM = Language(code="udm", name="Udmurt", _and_separator="но", _or_separator="яке")
VEC = Language(code="vec", name="Venetian", _and_separator="e", _or_separator="o")
VEP = Language(code="vep", name="Veps", _and_separator="da", _or_separator="vai")
VLS = Language(code="vls", name="West Flemish", _and_separator="en", _or_separator="of")
WAR = Language(code="war", name="Waray", _and_separator="ug", _or_separator="o")
WUU = Language(code="wuu", name="Wu Chinese", _and_separator="搭", _or_separator="或者")
XAL = Language(code="xal", name="Kalmyk", _and_separator="болн", _or_separator="эсвл")
XMF = Language(
    code="xmf", name="Mingrelian", _and_separator="დო", _or_separator="ვარდა"
)
ZEA = Language(code="zea", name="Zeelandic", _and_separator="en", _or_separator="of")
YUE = Language(code="yue", name="Cantonese", _and_separator="同", _or_separator="或者")
