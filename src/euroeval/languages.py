"""List of languages and their language codes.

The language codes contain both all the ISO 639-1 codes, as well as the ISO 639-3 codes
for languages that do not have an ISO 639-1 code.
"""

from dataclasses import dataclass, field


@dataclass
class Language:
    """A benchmarkable language.

    Attributes:
        code:
            The ISO 639-1 language code of the language.
        name:
            The name of the language.
        and_separator (optional):
            The word 'and' in the language.
        or_separator (optional):
            The word 'or' in the language.
    """

    code: str
    name: str
    _and_separator: str | None = field(repr=False, default=None)
    _or_separator: str | None = field(repr=False, default=None)

    def __hash__(self) -> int:
        """Return a hash of the language."""
        return hash(self.code)

    @property
    def and_separator(self) -> str:
        """Get the word 'and' in the language.

        Returns:
            The word 'and' in the language.

        Raises:
            NotImplementedError:
                If `and_separator` is `None`.
        """
        if not self._and_separator:
            raise NotImplementedError(
                f"Separator for the word 'and' has not been defined for {self.name}."
            )
        return self._and_separator

    @and_separator.setter
    def and_separator(self, value: str | None) -> None:
        self._and_separator = value

    @property
    def or_separator(self) -> str:
        """Get the word 'or' in the language.

        Returns:
            The word 'or' in the language.

        Raises:
            NotImplementedError:
                If `or_separator` is `None`.
        """
        if not self._or_separator:
            raise NotImplementedError(
                f"Separator for the word 'or' has not been defined for {self.name}."
            )
        return self._or_separator

    @or_separator.setter
    def or_separator(self, value: str | None) -> None:
        self._or_separator = value


def get_all_languages() -> dict[str, Language]:
    """Get a list of all the languages.

    Returns:
        A mapping between language codes and their configurations.
    """
    return {cfg.code: cfg for cfg in globals().values() if isinstance(cfg, Language)}


ABKHAZIAN: Language = Language(
    code="ab", name="Abkhazian", _and_separator="и", _or_separator="ма"
)
AFAR: Language = Language(
    code="aa", name="Afar", _and_separator="kee", _or_separator="maleey"
)
AFRIKAANS: Language = Language(
    code="af", name="Afrikaans", _and_separator="en", _or_separator="of"
)
ALBANIAN: Language = Language(
    code="sq", name="Albanian", _and_separator="dhe", _or_separator="ose"
)
AMHARIC: Language = Language(
    code="am", name="Amharic", _and_separator="እና", _or_separator="ወይም"
)
ARABIC: Language = Language(
    code="ar", name="Arabic", _and_separator="و", _or_separator="أو"
)
ARAGONESE: Language = Language(
    code="an", name="Aragonese", _and_separator="y", _or_separator="u"
)
ARMENIAN: Language = Language(
    code="hy", name="Armenian", _and_separator="և", _or_separator="կամ"
)
ASSAMESE: Language = Language(
    code="as", name="Assamese", _and_separator="আৰু", _or_separator="বা"
)
AVARIC: Language = Language(
    code="av", name="Avaric", _and_separator="ги", _or_separator="яги"
)
AVESTAN: Language = Language(
    code="ae", name="Avestan", _and_separator="utā", _or_separator="vā"
)
AYMARA: Language = Language(
    code="ay", name="Aymara", _and_separator="-mpi", _or_separator="jan ukax"
)
AZERBAIJANI: Language = Language(
    code="az", name="Azerbaijani", _and_separator="və", _or_separator="və ya"
)
BAMBARA: Language = Language(
    code="bm", name="Bambara", _and_separator="ani", _or_separator="walima"
)
BASHKIR: Language = Language(
    code="ba", name="Bashkir", _and_separator="һәм", _or_separator="йәки"
)
BASQUE: Language = Language(
    code="eu", name="Basque", _and_separator="eta", _or_separator="edo"
)
BELARUSIAN: Language = Language(
    code="be", name="Belarusian", _and_separator="і", _or_separator="або"
)
BENGALI: Language = Language(
    code="bn", name="Bengali", _and_separator="এবং", _or_separator="অথবা"
)
BISLAMA: Language = Language(
    code="bi", name="Bislama", _and_separator="mo", _or_separator="o"
)
BOSNIAN: Language = Language(
    code="bs", name="Bosnian", _and_separator="i", _or_separator="ili"
)
BRETON: Language = Language(
    code="br", name="Breton", _and_separator="ha", _or_separator="pe"
)
BULGARIAN: Language = Language(
    code="bg", name="Bulgarian", _and_separator="и", _or_separator="или"
)
BURMESE: Language = Language(
    code="my", name="Burmese", _and_separator="နှင့်", _or_separator="သို့မဟုတ်"
)
CATALAN: Language = Language(
    code="ca", name="Catalan", _and_separator="i", _or_separator="o"
)
CHAMORRO: Language = Language(
    code="ch", name="Chamorro", _and_separator="yan", _or_separator="pat"
)
CHECHEN: Language = Language(
    code="ce", name="Chechen", _and_separator="а", _or_separator="я"
)
CHICHEWA: Language = Language(
    code="ny", name="Chichewa", _and_separator="ndi", _or_separator="kapena"
)
SIMPLIFIED_CHINESE: Language = Language(
    code="zh-cn", name="Simplified Chinese", _and_separator="和", _or_separator="或"
)
TRADITIONAL_CHINESE: Language = Language(
    code="zh-tw", name="Traditional Chinese", _and_separator="與", _or_separator="或"
)
CHURCH_SLAVIC: Language = Language(
    code="cu", name="Church Slavic", _and_separator="и", _or_separator="или"
)
CHUVASH: Language = Language(
    code="cv", name="Chuvash", _and_separator="тата", _or_separator="е"
)
CORNISH: Language = Language(
    code="kw", name="Cornish", _and_separator="ha", _or_separator="po"
)
CORSICAN: Language = Language(
    code="co", name="Corsican", _and_separator="e", _or_separator="o"
)
CREE: Language = Language(
    code="cr", name="Cree", _and_separator="ēkwa", _or_separator="kamāc"
)
CROATIAN: Language = Language(
    code="hr", name="Croatian", _and_separator="i", _or_separator="ili"
)
CZECH: Language = Language(
    code="cs", name="Czech", _and_separator="a", _or_separator="nebo"
)
DANISH: Language = Language(
    code="da", name="Danish", _and_separator="og", _or_separator="eller"
)
DUTCH: Language = Language(
    code="nl", name="Dutch", _and_separator="en", _or_separator="of"
)
DIVEHI: Language = Language(
    code="dv", name="Divehi", _and_separator="އަދި", _or_separator="ނުވަތަ"
)
DZONGKHA: Language = Language(
    code="dz", name="Dzongkha", _and_separator="དང་", _or_separator="ཡང་མེན"
)
ENGLISH: Language = Language(
    code="en", name="English", _and_separator="and", _or_separator="or"
)
ESPERANTO: Language = Language(
    code="eo", name="Esperanto", _and_separator="kaj", _or_separator="aŭ"
)
ESTONIAN: Language = Language(
    code="et", name="Estonian", _and_separator="ja", _or_separator="või"
)
EWE: Language = Language(
    code="ee", name="Ewe", _and_separator="kple", _or_separator="alo"
)
FAROESE: Language = Language(
    code="fo", name="Faroese", _and_separator="og", _or_separator="ella"
)
FIJIAN: Language = Language(
    code="fj", name="Fijian", _and_separator="kei", _or_separator="se"
)
FINNISH: Language = Language(
    code="fi", name="Finnish", _and_separator="ja", _or_separator="tai"
)
FRENCH: Language = Language(
    code="fr", name="French", _and_separator="et", _or_separator="ou"
)
WESTERN_FRISIAN: Language = Language(
    code="fy", name="Western Frisian", _and_separator="en", _or_separator="of"
)
FULAH: Language = Language(
    code="ff", name="Fulah", _and_separator="e", _or_separator="ma"
)
GAELIC: Language = Language(
    code="gd", name="Gaelic", _and_separator="agus", _or_separator="no"
)
GALICIAN: Language = Language(
    code="gl", name="Galician", _and_separator="e", _or_separator="ou"
)
GANDA: Language = Language(
    code="lg", name="Ganda", _and_separator="ne", _or_separator="oba"
)
GEORGIAN: Language = Language(
    code="ka", name="Georgian", _and_separator="და", _or_separator="ან"
)
GERMAN: Language = Language(
    code="de", name="German", _and_separator="und", _or_separator="oder"
)
GREEK: Language = Language(
    code="el", name="Greek", _and_separator="και", _or_separator="ή"
)
GREENLANDIC: Language = Language(
    code="kl", name="Greenlandic", _and_separator="aamma", _or_separator="imaluunniit"
)
GUARANI: Language = Language(
    code="gn", name="Guarani", _and_separator="ha", _or_separator="térã"
)
GUJARATI: Language = Language(
    code="gu", name="Gujarati", _and_separator="અને", _or_separator="અથવા"
)
HAITIAN: Language = Language(
    code="ht", name="Haitian", _and_separator="ak", _or_separator="oswa"
)
HAUSA: Language = Language(
    code="ha", name="Hausa", _and_separator="da", _or_separator="ko"
)
HEBREW: Language = Language(
    code="he", name="Hebrew", _and_separator="ו", _or_separator="או"
)
HERERO: Language = Language(
    code="hz", name="Herero", _and_separator="na", _or_separator="po"
)
HINDI: Language = Language(
    code="hi", name="Hindi", _and_separator="और", _or_separator="या"
)
HUNGARIAN: Language = Language(
    code="hu", name="Hungarian", _and_separator="és", _or_separator="vagy"
)
ICELANDIC: Language = Language(
    code="is", name="Icelandic", _and_separator="og", _or_separator="eða"
)
IDO: Language = Language(code="io", name="Ido", _and_separator="e", _or_separator="o")
IGBO: Language = Language(
    code="ig", name="Igbo", _and_separator="na", _or_separator="ma ọ bụ"
)
INDONESIAN: Language = Language(
    code="id", name="Indonesian", _and_separator="dan", _or_separator="atau"
)
INTERLINGUA: Language = Language(
    code="ia", name="Interlingua", _and_separator="e", _or_separator="o"
)
INTERLINGUE: Language = Language(
    code="ie", name="Interlingue", _and_separator="e", _or_separator="o"
)
INUKTITUT: Language = Language(
    code="iu", name="Inuktitut", _and_separator="alu", _or_separator="immaqaa"
)
INUPIAQ: Language = Language(
    code="ik", name="Inupiaq", _and_separator="ġu", _or_separator="luunniit"
)
IRISH: Language = Language(
    code="ga", name="Irish", _and_separator="agus", _or_separator="nó"
)
ITALIAN: Language = Language(
    code="it", name="Italian", _and_separator="e", _or_separator="o"
)
JAPANESE: Language = Language(
    code="ja", name="Japanese", _and_separator="と", _or_separator="または"
)
KANNADA: Language = Language(
    code="kn", name="Kannada", _and_separator="ಮತ್ತು", _or_separator="ಅಥವಾ"
)
KANURI: Language = Language(
    code="kr", name="Kanuri", _and_separator="-a", _or_separator="yáá"
)
KASHMIRI: Language = Language(
    code="ks", name="Kashmiri", _and_separator="تہٕ", _or_separator="یا"
)
KAZAKH: Language = Language(
    code="kk", name="Kazakh", _and_separator="және", _or_separator="немесе"
)
CENTRAL_KHMER: Language = Language(
    code="km", name="Central Khmer", _and_separator="និង", _or_separator="ឬ"
)
KIKUYU: Language = Language(
    code="ki", name="Kikuyu", _and_separator="na", _or_separator="kana"
)
KINYARWANDA: Language = Language(
    code="rw", name="Kinyarwanda", _and_separator="na", _or_separator="cyangwa"
)
KIRGHIZ: Language = Language(
    code="ky", name="Kirghiz", _and_separator="жана", _or_separator="же"
)
KOMI: Language = Language(
    code="kv", name="Komi", _and_separator="да", _or_separator="либӧ"
)
KONGO: Language = Language(
    code="kg", name="Kongo", _and_separator="ye", _or_separator="kana"
)
KOREAN: Language = Language(
    code="ko", name="Korean", _and_separator="그리고", _or_separator="또는"
)
KUANYAMA: Language = Language(
    code="kj", name="Kuanyama", _and_separator="na", _or_separator="nenge"
)
KURDISH: Language = Language(
    code="ku", name="Kurdish", _and_separator="û", _or_separator="an"
)
LAO: Language = Language(code="lo", name="Lao", _and_separator="และ", _or_separator="ຫຼື")
LATIN: Language = Language(
    code="la", name="Latin", _and_separator="et", _or_separator="aut"
)
LATVIAN: Language = Language(
    code="lv", name="Latvian", _and_separator="un", _or_separator="vai"
)
LIMBURGAN: Language = Language(
    code="li", name="Limburgan", _and_separator="en", _or_separator="of"
)
LINGALA: Language = Language(
    code="ln", name="Lingala", _and_separator="na", _or_separator="to"
)
LITHUANIAN: Language = Language(
    code="lt", name="Lithuanian", _and_separator="ir", _or_separator="arba"
)
LUBA_KATANGA: Language = Language(
    code="lu", name="Luba-Katanga", _and_separator="ne", _or_separator="nansha"
)
LUXEMBOURGISH: Language = Language(
    code="lb", name="Luxembourgish", _and_separator="an", _or_separator="oder"
)
MACEDONIAN: Language = Language(
    code="mk", name="Macedonian", _and_separator="и", _or_separator="или"
)
MALAGASY: Language = Language(
    code="mg", name="Malagasy", _and_separator="sy", _or_separator="na"
)
MALAY: Language = Language(
    code="ms", name="Malay", _and_separator="dan", _or_separator="atau"
)
MALAYALAM: Language = Language(
    code="ml", name="Malayalam", _and_separator="ഉം", _or_separator="അല്ലെങ്കിൽ"
)
MALTESE: Language = Language(
    code="mt", name="Maltese", _and_separator="u", _or_separator="jew"
)
MANX: Language = Language(
    code="gv", name="Manx", _and_separator="as", _or_separator="ny"
)
MAORI: Language = Language(
    code="mi", name="Maori", _and_separator="me", _or_separator="rānei"
)
MARATHI: Language = Language(
    code="mr", name="Marathi", _and_separator="आणि", _or_separator="किंवा"
)
MARSHALLESE: Language = Language(
    code="mh", name="Marshallese", _and_separator="im", _or_separator="ak"
)
MONGOLIAN: Language = Language(
    code="mn", name="Mongolian", _and_separator="ба", _or_separator="эсвэл"
)
NAURU: Language = Language(
    code="na", name="Nauru", _and_separator="ma", _or_separator="me"
)
NAVAJO: Language = Language(
    code="nv", name="Navajo", _and_separator="áádóó", _or_separator="doodaiiʼ"
)
NORTHERN_NDEBELE: Language = Language(
    code="nd", name="Northern Ndebele", _and_separator="lo", _or_separator="kumbe"
)
SOUTH_NDEBELE: Language = Language(
    code="nr", name="South Ndebele", _and_separator="na", _or_separator="namkha"
)
NDONGA: Language = Language(
    code="ng", name="Ndonga", _and_separator="na", _or_separator="nenge"
)
NEPALI: Language = Language(
    code="ne", name="Nepali", _and_separator="र", _or_separator="वा"
)
NORWEGIAN: Language = Language(
    code="no", name="Norwegian", _and_separator="og", _or_separator="eller"
)
NORWEGIAN_BOKMÅL: Language = Language(
    code="nb", name="Norwegian Bokmål", _and_separator="og", _or_separator="eller"
)
NORWEGIAN_NYNORSK: Language = Language(
    code="nn", name="Norwegian Nynorsk", _and_separator="og", _or_separator="eller"
)
OCCITAN: Language = Language(
    code="oc", name="Occitan", _and_separator="e", _or_separator="o"
)
OJIBWA: Language = Language(
    code="oj", name="Ojibwa", _and_separator="miinawaa", _or_separator="jiishin"
)
ORIYA: Language = Language(
    code="or", name="Oriya", _and_separator="ଏବଂ", _or_separator="କିମ୍ବା"
)
OROMO: Language = Language(
    code="om", name="Oromo", _and_separator="fi", _or_separator="yookan"
)
OSSETIAN: Language = Language(
    code="os", name="Ossetian", _and_separator="æмæ", _or_separator="кæнæ"
)
PALI: Language = Language(
    code="pi", name="Pali", _and_separator="ca", _or_separator="vā"
)
PASHTO: Language = Language(
    code="ps", name="Pashto", _and_separator="او", _or_separator="يا"
)
PERSIAN: Language = Language(
    code="fa", name="Persian", _and_separator="و", _or_separator="یا"
)
POLISH: Language = Language(
    code="pl", name="Polish", _and_separator="i", _or_separator="lub"
)
PORTUGUESE: Language = Language(
    code="pt", name="Portuguese", _and_separator="e", _or_separator="ou"
)
EUROPEAN_PORTUGUESE: Language = Language(
    code="pt-pt", name="European Portuguese", _and_separator="e", _or_separator="ou"
)
BRAZILIAN_PORTUGUESE: Language = Language(
    code="pt-br", name="Brazilian Portuguese", _and_separator="e", _or_separator="ou"
)
PUNJABI: Language = Language(
    code="pa", name="Punjabi", _and_separator="ਅਤੇ", _or_separator="ਜਾਂ"
)
QUECHUA: Language = Language(
    code="qu", name="Quechua", _and_separator="-pas", _or_separator="ichataq"
)
ROMANIAN: Language = Language(
    code="ro", name="Romanian", _and_separator="și", _or_separator="sau"
)
ROMANSH: Language = Language(
    code="rm", name="Romansh", _and_separator="e", _or_separator="u"
)
RUNDI: Language = Language(
    code="rn", name="Rundi", _and_separator="na", _or_separator="canke"
)
RUSSIAN: Language = Language(
    code="ru", name="Russian", _and_separator="и", _or_separator="или"
)
NORTHERN_SAMI: Language = Language(
    code="se", name="Northern Sami", _and_separator="ja", _or_separator="dahje"
)
SAMOAN: Language = Language(
    code="sm", name="Samoan", _and_separator="ma", _or_separator="poʻo"
)
SANGO: Language = Language(
    code="sg", name="Sango", _and_separator="na", _or_separator="wala"
)
SANSKRIT: Language = Language(
    code="sa", name="Sanskrit", _and_separator="च", _or_separator="वा"
)
SARDINIAN: Language = Language(
    code="sc", name="Sardinian", _and_separator="e", _or_separator="o"
)
SERBIAN: Language = Language(
    code="sr", name="Serbian", _and_separator="и", _or_separator="или"
)
SHONA: Language = Language(
    code="sn", name="Shona", _and_separator="uye", _or_separator="kana"
)
SINDHI: Language = Language(
    code="sd", name="Sindhi", _and_separator="۽", _or_separator="يا"
)
SINHALA: Language = Language(
    code="si", name="Sinhala", _and_separator="සහ", _or_separator="හෝ"
)
SLOVAK: Language = Language(
    code="sk", name="Slovak", _and_separator="a", _or_separator="alebo"
)
SLOVENIAN: Language = Language(
    code="sl", name="Slovenian", _and_separator="in", _or_separator="ali"
)
SOMALI: Language = Language(
    code="so", name="Somali", _and_separator="iyo", _or_separator="ama"
)
SOTHO: Language = Language(
    code="st", name="Sotho", _and_separator="le", _or_separator="kapa"
)
SPANISH: Language = Language(
    code="es", name="Spanish", _and_separator="y", _or_separator="o"
)
SUNDANESE: Language = Language(
    code="su", name="Sundanese", _and_separator="jeung", _or_separator="atawa"
)
SWAHILI: Language = Language(
    code="sw", name="Swahili", _and_separator="na", _or_separator="au"
)
SWATI: Language = Language(
    code="ss", name="Swati", _and_separator="na", _or_separator="noma"
)
SWEDISH: Language = Language(
    code="sv", name="Swedish", _and_separator="och", _or_separator="eller"
)
TAGALOG: Language = Language(
    code="tl", name="Tagalog", _and_separator="at", _or_separator="o"
)
TAHITIAN: Language = Language(
    code="ty", name="Tahitian", _and_separator="e", _or_separator="aore ra"
)
TAJIK: Language = Language(
    code="tg", name="Tajik", _and_separator="ва", _or_separator="ё"
)
TAMIL: Language = Language(
    code="ta", name="Tamil", _and_separator="மற்றும்", _or_separator="அல்லது"
)
TATAR: Language = Language(
    code="tt", name="Tatar", _and_separator="һәм", _or_separator="яки"
)
TELUGU: Language = Language(
    code="te", name="Telugu", _and_separator="మరియు", _or_separator="లేదా"
)
THAI: Language = Language(
    code="th", name="Thai", _and_separator="และ", _or_separator="หรือ"
)
TIBETAN: Language = Language(
    code="bo", name="Tibetan", _and_separator="དང་", _or_separator="ཡང་ན"
)
TIGRINYA: Language = Language(
    code="ti", name="Tigrinya", _and_separator="ን", _or_separator="ወይ"
)
TONGA: Language = Language(
    code="to", name="Tonga", _and_separator="mo", _or_separator="pe"
)
TSONGA: Language = Language(
    code="ts", name="Tsonga", _and_separator="na", _or_separator="kumbe"
)
TSWANA: Language = Language(
    code="tn", name="Tswana", _and_separator="le", _or_separator="kgotsa"
)
TURKISH: Language = Language(
    code="tr", name="Turkish", _and_separator="ve", _or_separator="veya"
)
TURKMEN: Language = Language(
    code="tk", name="Turkmen", _and_separator="we", _or_separator="ýa-da"
)
TWI: Language = Language(
    code="tw", name="Twi", _and_separator="ne", _or_separator="anaa"
)
UIGHUR: Language = Language(
    code="ug", name="Uighur", _and_separator="ۋە", _or_separator="ياكى"
)
UKRAINIAN: Language = Language(
    code="uk", name="Ukrainian", _and_separator="і", _or_separator="або"
)
URDU: Language = Language(
    code="ur", name="Urdu", _and_separator="اور", _or_separator="یا"
)
UZBEK: Language = Language(
    code="uz", name="Uzbek", _and_separator="va", _or_separator="yoki"
)
VENDA: Language = Language(
    code="ve", name="Venda", _and_separator="na", _or_separator="kana"
)
VIETNAMESE: Language = Language(
    code="vi", name="Vietnamese", _and_separator="và", _or_separator="hoặc"
)
VOLAPÜK: Language = Language(
    code="vo", name="Volapük", _and_separator="e", _or_separator="u"
)
WALLOON: Language = Language(
    code="wa", name="Walloon", _and_separator="et", _or_separator="ou"
)
WELSH: Language = Language(
    code="cy", name="Welsh", _and_separator="a", _or_separator="neu"
)
WOLOF: Language = Language(
    code="wo", name="Wolof", _and_separator="ak", _or_separator="walla"
)
XHOSA: Language = Language(
    code="xh", name="Xhosa", _and_separator="kwaye", _or_separator="okanye"
)
YIDDISH: Language = Language(
    code="yi", name="Yiddish", _and_separator="און", _or_separator="אָדער"
)
YORUBA: Language = Language(
    code="yo", name="Yoruba", _and_separator="àti", _or_separator="tàbí"
)
ZHUANG: Language = Language(
    code="za", name="Zhuang", _and_separator="kae", _or_separator="aevih"
)
ZULU: Language = Language(
    code="zu", name="Zulu", _and_separator="futhi", _or_separator="noma"
)
ACEHNESE: Language = Language(
    code="ace", name="Acehnese", _and_separator="ngon", _or_separator="atɔ"
)
ADYGHE: Language = Language(
    code="ady", name="Adyghe", _and_separator="рэ", _or_separator="е"
)
SOUTHERN_ALTAI: Language = Language(
    code="alt", name="Southern Altai", _and_separator="ла", _or_separator="эмезе"
)
AMIS: Language = Language(
    code="ami", name="Amis", _and_separator="ato", _or_separator="o"
)
OLD_ENGLISH: Language = Language(
    code="ang", name="Old English", _and_separator="and", _or_separator="oþþe"
)
ANGIKA: Language = Language(
    code="anp", name="Angika", _and_separator="आर", _or_separator="या"
)
ARAMAIC: Language = Language(
    code="arc", name="Aramaic", _and_separator="ܘ", _or_separator="ܐܘ"
)
MOROCCAN_ARABIC: Language = Language(
    code="ary", name="Moroccan Arabic", _and_separator="w", _or_separator="wella"
)
EGYPTIAN_ARABIC: Language = Language(
    code="arz", name="Egyptian Arabic", _and_separator="و", _or_separator="أو"
)
ASTURIAN: Language = Language(
    code="ast", name="Asturian", _and_separator="y", _or_separator="o"
)
ATIKAMEKW: Language = Language(
    code="atj", name="Atikamekw", _and_separator="et", _or_separator="ou"
)
KOTAVA: Language = Language(
    code="avk", name="Kotava", _and_separator="is", _or_separator="en"
)
AWADHI: Language = Language(
    code="awa", name="Awadhi", _and_separator="अउ", _or_separator="या"
)
SOUTH_AZERBAIJANI: Language = Language(
    code="azb", name="South Azerbaijani", _and_separator="و", _or_separator="یوخسا"
)
BALINESE: Language = Language(
    code="ban", name="Balinese", _and_separator="lan", _or_separator="utawi"
)
BAVARIAN: Language = Language(
    code="bar", name="Bavarian", _and_separator="und", _or_separator="oda"
)
CENTRAL_BIKOL: Language = Language(
    code="bcl", name="Central Bikol", _and_separator="asin", _or_separator="o"
)
BANJAR: Language = Language(
    code="bjn", name="Banjar", _and_separator="wan", _or_separator="atawa"
)
PAO: Language = Language(
    code="blk", name="Pa'O", _and_separator="နန်", _or_separator="မု"
)
BISHNUPRIYA: Language = Language(
    code="bpy", name="Bishnupriya", _and_separator="आ", _or_separator="বা"
)
BUGINESE: Language = Language(
    code="bug", name="Buginese", _and_separator="na", _or_separator="iyarega"
)
BURIAT: Language = Language(
    code="bxr", name="Buriat", _and_separator="ба", _or_separator="али"
)
MINDONG_CHINESE: Language = Language(
    code="cdo", name="Mindong Chinese", _and_separator="共", _or_separator="或者"
)
CEBUANO: Language = Language(
    code="ceb", name="Cebuano", _and_separator="ug", _or_separator="o"
)
CHEROKEE: Language = Language(
    code="chr", name="Cherokee", _and_separator="ᎠᎴ", _or_separator="ᎠᎴ"
)
CHEYENNE: Language = Language(
    code="chy", name="Cheyenne", _and_separator="na", _or_separator="hēme"
)
CENTRAL_KURDISH: Language = Language(
    code="ckb", name="Central Kurdish", _and_separator="و", _or_separator="یان"
)
CRIMEAN_TATAR: Language = Language(
    code="crh", name="Crimean Tatar", _and_separator="ve", _or_separator="ya da"
)
KASHUBIAN: Language = Language(
    code="csb", name="Kashubian", _and_separator="ë", _or_separator="abò"
)
DAGBANI: Language = Language(
    code="dag", name="Dagbani", _and_separator="n-ti", _or_separator="bee"
)
DINKA: Language = Language(
    code="din", name="Dinka", _and_separator="ka", _or_separator="ke"
)
DIMLI: Language = Language(
    code="diq", name="Dimli", _and_separator="û", _or_separator="ya"
)
LOWER_SORBIAN: Language = Language(
    code="dsb", name="Lower Sorbian", _and_separator="a", _or_separator="abo"
)
DOTELI: Language = Language(
    code="dty", name="Doteli", _and_separator="र", _or_separator="या"
)
EXTREMADURAN: Language = Language(
    code="ext", name="Extremaduran", _and_separator="y", _or_separator="u"
)
FANTI: Language = Language(
    code="fat", name="Fanti", _and_separator="na", _or_separator="anaa"
)
FON: Language = Language(
    code="fon", name="Fon", _and_separator="kpóɖó", _or_separator="kabi"
)
ARPITAN: Language = Language(
    code="frp", name="Arpitan", _and_separator="et", _or_separator="ou"
)
NORTHERN_FRISIAN: Language = Language(
    code="frr", name="Northern Frisian", _and_separator="an", _or_separator="of"
)
FRIULIAN: Language = Language(
    code="fur", name="Friulian", _and_separator="e", _or_separator="o"
)
GAGAUZ: Language = Language(
    code="gag", name="Gagauz", _and_separator="hem", _or_separator="ya"
)
GAN_CHINESE: Language = Language(
    code="gan", name="Gan Chinese", _and_separator="同", _or_separator="或"
)
GUIANAN_CREOLE: Language = Language(
    code="gcr", name="Guianan Creole", _and_separator="ké", _or_separator="ou"
)
GILAKI: Language = Language(
    code="glk", name="Gilaki", _and_separator="و", _or_separator="یا"
)
GOAN_KONKANI: Language = Language(
    code="gom", name="Goan Konkani", _and_separator="आनी", _or_separator="वा"
)
GORONTALO: Language = Language(
    code="gor", name="Gorontalo", _and_separator="wawu", _or_separator="meyalo"
)
GOTHIC: Language = Language(
    code="got", name="Gothic", _and_separator="jah", _or_separator="aiþþau"
)
GHANAIAN_PIDGIN: Language = Language(
    code="gpe", name="Ghanaian Pidgin", _and_separator="and", _or_separator="anaa"
)
WAYUU: Language = Language(
    code="guc", name="Wayuu", _and_separator="je", _or_separator="yaa"
)
FRAFRA: Language = Language(
    code="gur", name="Frafra", _and_separator="la", _or_separator="bee"
)
GUN: Language = Language(
    code="guw", name="Gun", _and_separator="pódó", _or_separator="yèkì"
)
HAKKA_CHINESE: Language = Language(
    code="hak", name="Hakka Chinese", _and_separator="同", _or_separator="或者"
)
HAWAIIAN: Language = Language(
    code="haw", name="Hawaiian", _and_separator="a", _or_separator="a i ʻole"
)
FIJI_HINDI: Language = Language(
    code="hif", name="Fiji Hindi", _and_separator="aur", _or_separator="ki"
)
UPPER_SORBIAN: Language = Language(
    code="hsb", name="Upper Sorbian", _and_separator="a", _or_separator="abo"
)
WESTERN_ARMENIAN: Language = Language(
    code="hyw", name="Western Armenian", _and_separator="եւ", _or_separator="կամ"
)
ILOKO: Language = Language(
    code="ilo", name="Iloko", _and_separator="ken", _or_separator="wenno"
)
INGUSH: Language = Language(
    code="inh", name="Ingush", _and_separator="и", _or_separator="е"
)
JAMAICAN_CREOLE: Language = Language(
    code="jam", name="Jamaican Creole", _and_separator="an", _or_separator="ar"
)
LOJBAN: Language = Language(
    code="jbo", name="Lojban", _and_separator="e", _or_separator="a"
)
KARA_KALPAK: Language = Language(
    code="kaa", name="Kara-Kalpak", _and_separator="ha'm", _or_separator="yamasa"
)
KABYLE: Language = Language(
    code="kab", name="Kabyle", _and_separator="d", _or_separator="neɣ"
)
KABARDIAN: Language = Language(
    code="kbd", name="Kabardian", _and_separator="рэ", _or_separator="хэтӀэ"
)
KABIYÈ: Language = Language(
    code="kbp", name="Kabiyè", _and_separator="nɛ", _or_separator="yaa"
)
TYAP: Language = Language(
    code="kcg", name="Tyap", _and_separator="ma", _or_separator="a̠ni"
)
KOMI_PERMYAK: Language = Language(
    code="koi", name="Komi-Permyak", _and_separator="да", _or_separator="либӧ"
)
KARACHAY_BALKAR: Language = Language(
    code="krc", name="Karachay-Balkar", _and_separator="бла", _or_separator="не да"
)
LADINO: Language = Language(
    code="lad", name="Ladino", _and_separator="i", _or_separator="o"
)
LAK: Language = Language(
    code="lbe", name="Lak", _and_separator="ва", _or_separator="ягу"
)
LEZGHIAN: Language = Language(
    code="lez", name="Lezghian", _and_separator="ва", _or_separator="я"
)
LINGUA_FRANCA_NOVA: Language = Language(
    code="lfn", name="Lingua Franca Nova", _and_separator="e", _or_separator="o"
)
LIGURIAN: Language = Language(
    code="lij", name="Ligurian", _and_separator="e", _or_separator="ò"
)
LADIN: Language = Language(
    code="lld", name="Ladin", _and_separator="y", _or_separator="o"
)
LOMBARD: Language = Language(
    code="lmo", name="Lombard", _and_separator="e", _or_separator="o"
)
LATGALIAN: Language = Language(
    code="ltg", name="Latgalian", _and_separator="i", _or_separator="voi"
)
MADURESE: Language = Language(
    code="mad", name="Madurese", _and_separator="ban", _or_separator="o"
)
MAITHILI: Language = Language(
    code="mai", name="Maithili", _and_separator="आ", _or_separator="वा"
)
MOKSHA: Language = Language(
    code="mdf", name="Moksha", _and_separator="ди", _or_separator="или"
)
EASTERN_MARI: Language = Language(
    code="mhr", name="Eastern Mari", _and_separator="да", _or_separator="o"
)
MINANGKABAU: Language = Language(
    code="min", name="Minangkabau", _and_separator="jo", _or_separator="atau"
)
MANIPURI: Language = Language(
    code="mni", name="Manipuri", _and_separator="ꯑꯃꯁꯨꯡ", _or_separator="ꯅꯠꯇ꯭ꯔꯒꯥ"
)
MON: Language = Language(
    code="mnw", name="Mon", _and_separator="ကဵု", _or_separator="ဟွံသေင်မ္ဂး"
)
WESTERN_MARI: Language = Language(
    code="mrj", name="Western Mari", _and_separator="да", _or_separator="o"
)
MIRANDESE: Language = Language(
    code="mwl", name="Mirandese", _and_separator="i", _or_separator="ó"
)
ERZYA: Language = Language(
    code="myv", name="Erzya", _and_separator="ды", _or_separator="эли"
)
MAZANDERANI: Language = Language(
    code="mzn", name="Mazanderani", _and_separator="و", _or_separator="یا"
)
NEAPOLITAN: Language = Language(
    code="nap", name="Neapolitan", _and_separator="e", _or_separator="o"
)
LOW_GERMAN: Language = Language(
    code="nds", name="Low German", _and_separator="un", _or_separator="oder"
)
NEWARI: Language = Language(
    code="new", name="Newari", _and_separator="व", _or_separator="या"
)
NIAS: Language = Language(
    code="nia", name="Nias", _and_separator="ba", _or_separator="mazi"
)
NOVIAL: Language = Language(
    code="nov", name="Novial", _and_separator="e", _or_separator="o"
)
NKO: Language = Language(
    code="nqo", name="N'Ko", _and_separator="ߣߌ߫", _or_separator="ߥߟߊ߫"
)
NORTHERN_SOTHO: Language = Language(
    code="nso", name="Northern Sotho", _and_separator="le", _or_separator="goba"
)
LIVVI_KARELIAN: Language = Language(
    code="olo", name="Livvi-Karelian", _and_separator="da", _or_separator="libo"
)
PANGASINAN: Language = Language(
    code="pag", name="Pangasinan", _and_separator="tan", _or_separator="odino"
)
PAMPANGA: Language = Language(
    code="pam", name="Pampanga", _and_separator="at", _or_separator="o"
)
PAPIAMENTO: Language = Language(
    code="pap", name="Papiamento", _and_separator="y", _or_separator="o"
)
PICARD: Language = Language(
    code="pcd", name="Picard", _and_separator="pi", _or_separator="ou"
)
NIGERIAN_PIDGIN: Language = Language(
    code="pcm", name="Nigerian Pidgin", _and_separator="and", _or_separator="abi"
)
PENNSYLVANIA_GERMAN: Language = Language(
    code="pdc", name="Pennsylvania German", _and_separator="un", _or_separator="odder"
)
PALATINE_GERMAN: Language = Language(
    code="pfl", name="Palatine German", _and_separator="un", _or_separator="odda"
)
PIEMONTESE: Language = Language(
    code="pms", name="Piemontese", _and_separator="e", _or_separator="o"
)
WESTERN_PUNJABI: Language = Language(
    code="pnb", name="Western Punjabi", _and_separator="تے", _or_separator="یا"
)
PONTIC: Language = Language(
    code="pnt", name="Pontic", _and_separator="και", _or_separator="ή"
)
PAIWAN: Language = Language(
    code="pwn", name="Paiwan", _and_separator="dja", _or_separator="uri"
)
VLAX_ROMANI: Language = Language(
    code="rmy", name="Vlax Romani", _and_separator="thaj", _or_separator="vaj"
)
RUSYN: Language = Language(
    code="rue", name="Rusyn", _and_separator="і", _or_separator="або"
)
YAKUT: Language = Language(
    code="sah", name="Yakut", _and_separator="уонна", _or_separator="эбэтэр"
)
SANTALI: Language = Language(
    code="sat", name="Santali", _and_separator="ᱟᱨ", _or_separator="ᱥᱮ"
)
SICILIAN: Language = Language(
    code="scn", name="Sicilian", _and_separator="e", _or_separator="o"
)
SCOTS: Language = Language(
    code="sco", name="Scots", _and_separator="an", _or_separator="or"
)
TACHELHIT: Language = Language(
    code="shi", name="Tachelhit", _and_separator="d", _or_separator="neɣ"
)
SHAN: Language = Language(
    code="shn", name="Shan", _and_separator="လႄႈ", _or_separator="หรือ"
)
SARAIKI: Language = Language(
    code="skr", name="Saraiki", _and_separator="تے", _or_separator="یا"
)
INARI_SAMI: Language = Language(
    code="smn", name="Inari Sami", _and_separator="ja", _or_separator="teikkâ"
)
SRANAN: Language = Language(
    code="srn", name="Sranan", _and_separator="nanga", _or_separator="efu"
)
SATERLAND_FRISIAN: Language = Language(
    code="stq", name="Saterland Frisian", _and_separator="un", _or_separator="of"
)
SILESIAN: Language = Language(
    code="szl", name="Silesian", _and_separator="a", _or_separator="abo"
)
SAKIZAYA: Language = Language(
    code="szy", name="Sakizaya", _and_separator="ata", _or_separator="uduli"
)
ATAYAL: Language = Language(
    code="tay", name="Atayal", _and_separator="daha", _or_separator="ima"
)
TULU: Language = Language(
    code="tcy", name="Tulu", _and_separator="ಬೊಕ್ಕ", _or_separator="ಅತ್ತಂಡ"
)
TETUM: Language = Language(
    code="tet", name="Tetum", _and_separator="no", _or_separator="ka"
)
TALYSH: Language = Language(
    code="tly", name="Talysh", _and_separator="u", _or_separator="jo"
)
TOK_PISIN: Language = Language(
    code="tpi", name="Tok Pisin", _and_separator="na", _or_separator="o"
)
TAROKO: Language = Language(
    code="trv", name="Taroko", _and_separator="daha", _or_separator="ima"
)
TUMBUKA: Language = Language(
    code="tum", name="Tumbuka", _and_separator="na", _or_separator="panji"
)
TUVINIAN: Language = Language(
    code="tyv", name="Tuvinian", _and_separator=" болгаш", _or_separator="азы"
)
UDMURT: Language = Language(
    code="udm", name="Udmurt", _and_separator="но", _or_separator="яке"
)
VENETIAN: Language = Language(
    code="vec", name="Venetian", _and_separator="e", _or_separator="o"
)
VEPS: Language = Language(
    code="vep", name="Veps", _and_separator="da", _or_separator="vai"
)
WEST_FLEMISH: Language = Language(
    code="vls", name="West Flemish", _and_separator="en", _or_separator="of"
)
WARAY: Language = Language(
    code="war", name="Waray", _and_separator="ug", _or_separator="o"
)
WU_CHINESE: Language = Language(
    code="wuu", name="Wu Chinese", _and_separator="搭", _or_separator="或者"
)
KALMYK: Language = Language(
    code="xal", name="Kalmyk", _and_separator="болн", _or_separator="эсвл"
)
MINGRELIAN: Language = Language(
    code="xmf", name="Mingrelian", _and_separator="დო", _or_separator="ვარდა"
)
ZEELANDIC: Language = Language(
    code="zea", name="Zeelandic", _and_separator="en", _or_separator="of"
)
CANTONESE: Language = Language(
    code="yue", name="Cantonese", _and_separator="同", _or_separator="或者"
)
