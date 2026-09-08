"""List of languages and their language codes.

The language codes contain both all the ISO 639-1 codes, as well as the ISO 639-3 codes
for languages that do not have an ISO 639-1 code.
"""

import collections.abc as c
from dataclasses import dataclass, field


@dataclass(init=False)
class Language:
    """A benchmarkable language.

    Attributes:
        name:
            The name of the language.
        code_3:
            The ISO 639-3 language code of the language. Dataset configurations are
            commonly named after this code, such as `dan` for Danish, so it is what
            makes such a configuration resolvable.
        code_1 (optional):
            The ISO 639-1 language code of the language, carrying a region subtag
            where EuroEval distinguishes regional variants, such as `pt-br`. Missing
            for the languages that have no ISO 639-1 code at all.
        and_separator (optional):
            The word 'and' in the language.
        or_separator (optional):
            The word 'or' in the language.

    The constructor retains the historical ``Language(code, name)`` contract. The
    ISO 639-3 and ISO 639-1 codes can be supplied as keyword arguments for language
    definitions that carry both codes.
    """

    name: str
    code_3: str
    code_1: str | None = field(default=None)
    _and_separator: str | None = field(repr=False, default=None)
    _or_separator: str | None = field(repr=False, default=None)

    def __init__(
        self,
        code: str | None = None,
        name: str | None = None,
        _and_separator: str | None = None,
        _or_separator: str | None = None,
        *,
        code_3: str | None = None,
        code_1: str | None = None,
    ) -> None:
        """Initialise a language using its legacy or ISO code arguments.

        Args:
            code (optional):
                The historical preferred language code. This remains the first
                positional argument and the accepted ``code=`` keyword.
            name (optional):
                The language name.
            _and_separator (optional):
                The word 'and' in the language.
            _or_separator (optional):
                The word 'or' in the language.
            code_3 (optional):
                The ISO 639-3 language code.
            code_1 (optional):
                The ISO 639-1 language code, including an optional region subtag.

        Raises:
            TypeError:
                If neither a code nor a name is provided.
        """
        if code is None:
            code = code_1 if code_1 is not None else code_3
        if code is None:
            raise TypeError("Language requires a code or code_3 argument")
        if name is None:
            raise TypeError("Language requires a name argument")

        if code_3 is None:
            code_3 = code
        if code_1 is None and len(code.partition("-")[0]) == 2:
            code_1 = code

        self.name = name
        self.code_3 = code_3
        self.code_1 = code_1
        self._and_separator = _and_separator
        self._or_separator = _or_separator

    def __hash__(self) -> int:
        """Return a hash of the language."""
        return hash(self.code)

    @property
    def and_separator(self) -> str:
        """The word 'and' in the language.

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
    def code(self) -> str:
        """The code identifying the language.

        Returns:
            The ISO 639-1 code when the language has one and the ISO 639-3 code
            otherwise, which is how the registry of languages is keyed.
        """
        return self.code_1 if self.code_1 is not None else self.code_3

    @property
    def or_separator(self) -> str:
        """The word 'or' in the language.

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


def get_correct_language_codes(
    language_codes: str | c.Sequence[str],
) -> c.Sequence[str]:
    """Get correct language code(s).

    Args:
        language_codes:
            The language codes of the languages to include, both for models and
            datasets. Here 'no' means both Bokmål (nb) and Nynorsk (nn). Set this
            to 'all' if all languages should be considered.

    Returns:
        The correct language codes.
    """
    # Create a dictionary that maps languages to their associated language objects
    language_mapping = get_all_languages()

    # Create the list `languages`
    if "all" in language_codes:
        languages = list(language_mapping.keys())
    elif isinstance(language_codes, str):
        languages = [language_codes]
    else:
        languages = list(language_codes)

    # If `languages` contains 'no' then also include 'nb' and 'nn'. Conversely, if
    # either 'nb' or 'nn' are specified then also include 'no'.
    if "no" in languages:
        languages = list(set(languages) | {"nb", "nn"})
    elif "nb" in languages or "nn" in languages:
        languages = list(set(languages) | {"no"})

    return languages


def get_all_languages() -> dict[str, Language]:
    """Get a list of all the languages.

    Returns:
        A mapping between language codes and their configurations.
    """
    return {cfg.code: cfg for cfg in globals().values() if isinstance(cfg, Language)}


ABKHAZIAN: Language = Language(
    code_1="ab", code_3="abk", name="Abkhazian", _and_separator="и", _or_separator="ма"
)
AFAR: Language = Language(
    code_1="aa", code_3="aar", name="Afar", _and_separator="kee", _or_separator="maleey"
)
AFRIKAANS: Language = Language(
    code_1="af", code_3="afr", name="Afrikaans", _and_separator="en", _or_separator="of"
)
ALBANIAN: Language = Language(
    code_1="sq",
    code_3="sqi",
    name="Albanian",
    _and_separator="dhe",
    _or_separator="ose",
)
AMHARIC: Language = Language(
    code_1="am", code_3="amh", name="Amharic", _and_separator="እና", _or_separator="ወይም"
)
ARABIC: Language = Language(
    code_1="ar", code_3="ara", name="Arabic", _and_separator="و", _or_separator="أو"
)
ARAGONESE: Language = Language(
    code_1="an", code_3="arg", name="Aragonese", _and_separator="y", _or_separator="u"
)
ARMENIAN: Language = Language(
    code_1="hy", code_3="hye", name="Armenian", _and_separator="և", _or_separator="կամ"
)
ASSAMESE: Language = Language(
    code_1="as", code_3="asm", name="Assamese", _and_separator="আৰু", _or_separator="বা"
)
AVARIC: Language = Language(
    code_1="av", code_3="ava", name="Avaric", _and_separator="ги", _or_separator="яги"
)
AVESTAN: Language = Language(
    code_1="ae", code_3="ave", name="Avestan", _and_separator="utā", _or_separator="vā"
)
AYMARA: Language = Language(
    code_1="ay",
    code_3="aym",
    name="Aymara",
    _and_separator="-mpi",
    _or_separator="jan ukax",
)
AZERBAIJANI: Language = Language(
    code_1="az",
    code_3="aze",
    name="Azerbaijani",
    _and_separator="və",
    _or_separator="və ya",
)
BAMBARA: Language = Language(
    code_1="bm",
    code_3="bam",
    name="Bambara",
    _and_separator="ani",
    _or_separator="walima",
)
BASHKIR: Language = Language(
    code_1="ba",
    code_3="bak",
    name="Bashkir",
    _and_separator="һәм",
    _or_separator="йәки",
)
BASQUE: Language = Language(
    code_1="eu", code_3="eus", name="Basque", _and_separator="eta", _or_separator="edo"
)
BELARUSIAN: Language = Language(
    code_1="be",
    code_3="bel",
    name="Belarusian",
    _and_separator="і",
    _or_separator="або",
)
BENGALI: Language = Language(
    code_1="bn", code_3="ben", name="Bengali", _and_separator="এবং", _or_separator="অথবা"
)
BISLAMA: Language = Language(
    code_1="bi", code_3="bis", name="Bislama", _and_separator="mo", _or_separator="o"
)
BOSNIAN: Language = Language(
    code_1="bs", code_3="bos", name="Bosnian", _and_separator="i", _or_separator="ili"
)
BRETON: Language = Language(
    code_1="br", code_3="bre", name="Breton", _and_separator="ha", _or_separator="pe"
)
BULGARIAN: Language = Language(
    code_1="bg", code_3="bul", name="Bulgarian", _and_separator="и", _or_separator="или"
)
BURMESE: Language = Language(
    code_1="my", code_3="mya", name="Burmese", _and_separator="နှင့်", _or_separator="သို့မဟုတ်"
)
CATALAN: Language = Language(
    code_1="ca", code_3="cat", name="Catalan", _and_separator="i", _or_separator="o"
)
CHAMORRO: Language = Language(
    code_1="ch",
    code_3="cha",
    name="Chamorro",
    _and_separator="yan",
    _or_separator="pat",
)
CHECHEN: Language = Language(
    code_1="ce", code_3="che", name="Chechen", _and_separator="а", _or_separator="я"
)
CHICHEWA: Language = Language(
    code_1="ny",
    code_3="nya",
    name="Chichewa",
    _and_separator="ndi",
    _or_separator="kapena",
)
SIMPLIFIED_CHINESE: Language = Language(
    code_1="zh-cn",
    code_3="zho",
    name="Simplified Chinese",
    _and_separator="和",
    _or_separator="或",
)
TRADITIONAL_CHINESE: Language = Language(
    code_1="zh-tw",
    code_3="zho",
    name="Traditional Chinese",
    _and_separator="與",
    _or_separator="或",
)
CHURCH_SLAVIC: Language = Language(
    code_1="cu",
    code_3="chu",
    name="Church Slavic",
    _and_separator="и",
    _or_separator="или",
)
CHUVASH: Language = Language(
    code_1="cv", code_3="chv", name="Chuvash", _and_separator="тата", _or_separator="е"
)
CORNISH: Language = Language(
    code_1="kw", code_3="cor", name="Cornish", _and_separator="ha", _or_separator="po"
)
CORSICAN: Language = Language(
    code_1="co", code_3="cos", name="Corsican", _and_separator="e", _or_separator="o"
)
CREE: Language = Language(
    code_1="cr", code_3="cre", name="Cree", _and_separator="ēkwa", _or_separator="kamāc"
)
CROATIAN: Language = Language(
    code_1="hr", code_3="hrv", name="Croatian", _and_separator="i", _or_separator="ili"
)
CZECH: Language = Language(
    code_1="cs", code_3="ces", name="Czech", _and_separator="a", _or_separator="nebo"
)
DANISH: Language = Language(
    code_1="da", code_3="dan", name="Danish", _and_separator="og", _or_separator="eller"
)
DUTCH: Language = Language(
    code_1="nl", code_3="nld", name="Dutch", _and_separator="en", _or_separator="of"
)
DIVEHI: Language = Language(
    code_1="dv", code_3="div", name="Divehi", _and_separator="އަދި", _or_separator="ނުވަތަ"
)
DZONGKHA: Language = Language(
    code_1="dz",
    code_3="dzo",
    name="Dzongkha",
    _and_separator="དང་",
    _or_separator="ཡང་མེན",
)
ENGLISH: Language = Language(
    code_1="en", code_3="eng", name="English", _and_separator="and", _or_separator="or"
)
ESPERANTO: Language = Language(
    code_1="eo",
    code_3="epo",
    name="Esperanto",
    _and_separator="kaj",
    _or_separator="aŭ",
)
ESTONIAN: Language = Language(
    code_1="et", code_3="est", name="Estonian", _and_separator="ja", _or_separator="või"
)
EWE: Language = Language(
    code_1="ee", code_3="ewe", name="Ewe", _and_separator="kple", _or_separator="alo"
)
FAROESE: Language = Language(
    code_1="fo", code_3="fao", name="Faroese", _and_separator="og", _or_separator="ella"
)
FIJIAN: Language = Language(
    code_1="fj", code_3="fij", name="Fijian", _and_separator="kei", _or_separator="se"
)
FINNISH: Language = Language(
    code_1="fi", code_3="fin", name="Finnish", _and_separator="ja", _or_separator="tai"
)
FRENCH: Language = Language(
    code_1="fr", code_3="fra", name="French", _and_separator="et", _or_separator="ou"
)
WESTERN_FRISIAN: Language = Language(
    code_1="fy",
    code_3="fry",
    name="Western Frisian",
    _and_separator="en",
    _or_separator="of",
)
FULAH: Language = Language(
    code_1="ff", code_3="ful", name="Fulah", _and_separator="e", _or_separator="ma"
)
GAELIC: Language = Language(
    code_1="gd", code_3="gla", name="Gaelic", _and_separator="agus", _or_separator="no"
)
GALICIAN: Language = Language(
    code_1="gl", code_3="glg", name="Galician", _and_separator="e", _or_separator="ou"
)
GANDA: Language = Language(
    code_1="lg", code_3="lug", name="Ganda", _and_separator="ne", _or_separator="oba"
)
GEORGIAN: Language = Language(
    code_1="ka", code_3="kat", name="Georgian", _and_separator="და", _or_separator="ან"
)
GERMAN: Language = Language(
    code_1="de", code_3="deu", name="German", _and_separator="und", _or_separator="oder"
)
GREEK: Language = Language(
    code_1="el", code_3="ell", name="Greek", _and_separator="και", _or_separator="ή"
)
GREENLANDIC: Language = Language(
    code_1="kl",
    code_3="kal",
    name="Greenlandic",
    _and_separator="aamma",
    _or_separator="imaluunniit",
)
GUARANI: Language = Language(
    code_1="gn", code_3="grn", name="Guarani", _and_separator="ha", _or_separator="térã"
)
GUJARATI: Language = Language(
    code_1="gu",
    code_3="guj",
    name="Gujarati",
    _and_separator="અને",
    _or_separator="અથવા",
)
HAITIAN: Language = Language(
    code_1="ht", code_3="hat", name="Haitian", _and_separator="ak", _or_separator="oswa"
)
HAUSA: Language = Language(
    code_1="ha", code_3="hau", name="Hausa", _and_separator="da", _or_separator="ko"
)
HEBREW: Language = Language(
    code_1="he", code_3="heb", name="Hebrew", _and_separator="ו", _or_separator="או"
)
HERERO: Language = Language(
    code_1="hz", code_3="her", name="Herero", _and_separator="na", _or_separator="po"
)
HINDI: Language = Language(
    code_1="hi", code_3="hin", name="Hindi", _and_separator="और", _or_separator="या"
)
HUNGARIAN: Language = Language(
    code_1="hu",
    code_3="hun",
    name="Hungarian",
    _and_separator="és",
    _or_separator="vagy",
)
ICELANDIC: Language = Language(
    code_1="is",
    code_3="isl",
    name="Icelandic",
    _and_separator="og",
    _or_separator="eða",
)
IDO: Language = Language(
    code_1="io", code_3="ido", name="Ido", _and_separator="e", _or_separator="o"
)
IGBO: Language = Language(
    code_1="ig", code_3="ibo", name="Igbo", _and_separator="na", _or_separator="ma ọ bụ"
)
INDONESIAN: Language = Language(
    code_1="id",
    code_3="ind",
    name="Indonesian",
    _and_separator="dan",
    _or_separator="atau",
)
INTERLINGUA: Language = Language(
    code_1="ia", code_3="ina", name="Interlingua", _and_separator="e", _or_separator="o"
)
INTERLINGUE: Language = Language(
    code_1="ie", code_3="ile", name="Interlingue", _and_separator="e", _or_separator="o"
)
INUKTITUT: Language = Language(
    code_1="iu",
    code_3="iku",
    name="Inuktitut",
    _and_separator="alu",
    _or_separator="immaqaa",
)
INUPIAQ: Language = Language(
    code_1="ik",
    code_3="ipk",
    name="Inupiaq",
    _and_separator="ġu",
    _or_separator="luunniit",
)
IRISH: Language = Language(
    code_1="ga", code_3="gle", name="Irish", _and_separator="agus", _or_separator="nó"
)
ITALIAN: Language = Language(
    code_1="it", code_3="ita", name="Italian", _and_separator="e", _or_separator="o"
)
JAPANESE: Language = Language(
    code_1="ja",
    code_3="jpn",
    name="Japanese",
    _and_separator="と",
    _or_separator="または",
)
KANNADA: Language = Language(
    code_1="kn",
    code_3="kan",
    name="Kannada",
    _and_separator="ಮತ್ತು",
    _or_separator="ಅಥವಾ",
)
KANURI: Language = Language(
    code_1="kr", code_3="kau", name="Kanuri", _and_separator="-a", _or_separator="yáá"
)
KASHMIRI: Language = Language(
    code_1="ks", code_3="kas", name="Kashmiri", _and_separator="تہٕ", _or_separator="یا"
)
KAZAKH: Language = Language(
    code_1="kk",
    code_3="kaz",
    name="Kazakh",
    _and_separator="және",
    _or_separator="nemесе",
)
CENTRAL_KHMER: Language = Language(
    code_1="km",
    code_3="khm",
    name="Central Khmer",
    _and_separator="និង",
    _or_separator="ឬ",
)
KIKUYU: Language = Language(
    code_1="ki", code_3="kik", name="Kikuyu", _and_separator="na", _or_separator="kana"
)
KINYARWANDA: Language = Language(
    code_1="rw",
    code_3="kin",
    name="Kinyarwanda",
    _and_separator="na",
    _or_separator="cyangwa",
)
KIRGHIZ: Language = Language(
    code_1="ky", code_3="kir", name="Kirghiz", _and_separator="жана", _or_separator="же"
)
KOMI: Language = Language(
    code_1="kv", code_3="kom", name="Komi", _and_separator="да", _or_separator="либӧ"
)
KONGO: Language = Language(
    code_1="kg", code_3="kon", name="Kongo", _and_separator="ye", _or_separator="kana"
)
KOREAN: Language = Language(
    code_1="ko",
    code_3="kor",
    name="Korean",
    _and_separator="그리고",
    _or_separator="또는",
)
KUANYAMA: Language = Language(
    code_1="kj",
    code_3="kua",
    name="Kuanyama",
    _and_separator="na",
    _or_separator="nenge",
)
KURDISH: Language = Language(
    code_1="ku", code_3="kur", name="Kurdish", _and_separator="û", _or_separator="an"
)
LAO: Language = Language(
    code_1="lo", code_3="lao", name="Lao", _and_separator="และ", _or_separator="ຫຼື"
)
LATIN: Language = Language(
    code_1="la", code_3="lat", name="Latin", _and_separator="et", _or_separator="aut"
)
LATVIAN: Language = Language(
    code_1="lv", code_3="lav", name="Latvian", _and_separator="un", _or_separator="vai"
)
LIMBURGAN: Language = Language(
    code_1="li", code_3="lim", name="Limburgan", _and_separator="en", _or_separator="of"
)
LINGALA: Language = Language(
    code_1="ln", code_3="lin", name="Lingala", _and_separator="na", _or_separator="to"
)
LITHUANIAN: Language = Language(
    code_1="lt",
    code_3="lit",
    name="Lithuanian",
    _and_separator="ir",
    _or_separator="arba",
)
LUBA_KATANGA: Language = Language(
    code_1="lu",
    code_3="lub",
    name="Luba-Katanga",
    _and_separator="ne",
    _or_separator="nansha",
)
LUXEMBOURGISH: Language = Language(
    code_1="lb",
    code_3="ltz",
    name="Luxembourgish",
    _and_separator="an",
    _or_separator="oder",
)
MACEDONIAN: Language = Language(
    code_1="mk",
    code_3="mkd",
    name="Macedonian",
    _and_separator="и",
    _or_separator="или",
)
MALAGASY: Language = Language(
    code_1="mg", code_3="mlg", name="Malagasy", _and_separator="sy", _or_separator="na"
)
MALAY: Language = Language(
    code_1="ms", code_3="msa", name="Malay", _and_separator="dan", _or_separator="atau"
)
MALAYALAM: Language = Language(
    code_1="ml",
    code_3="mal",
    name="Malayalam",
    _and_separator="ഉം",
    _or_separator="അല്ലെങ്കിൽ",
)
MALTESE: Language = Language(
    code_1="mt", code_3="mlt", name="Maltese", _and_separator="u", _or_separator="jew"
)
MANX: Language = Language(
    code_1="gv", code_3="glv", name="Manx", _and_separator="as", _or_separator="ny"
)
MAORI: Language = Language(
    code_1="mi", code_3="mri", name="Maori", _and_separator="me", _or_separator="rānei"
)
MARATHI: Language = Language(
    code_1="mr",
    code_3="mar",
    name="Marathi",
    _and_separator="आणि",
    _or_separator="किंवा",
)
MARSHALLESE: Language = Language(
    code_1="mh",
    code_3="mah",
    name="Marshallese",
    _and_separator="im",
    _or_separator="ak",
)
MONGOLIAN: Language = Language(
    code_1="mn",
    code_3="mon",
    name="Mongolian",
    _and_separator="ба",
    _or_separator="эсвэл",
)
NAURU: Language = Language(
    code_1="na", code_3="nau", name="Nauru", _and_separator="ma", _or_separator="me"
)
NAVAJO: Language = Language(
    code_1="nv",
    code_3="nav",
    name="Navajo",
    _and_separator="áádóó",
    _or_separator="doodaiiʼ",
)
NORTHERN_NDEBELE: Language = Language(
    code_1="nd",
    code_3="nde",
    name="Northern Ndebele",
    _and_separator="lo",
    _or_separator="kumbe",
)
SOUTH_NDEBELE: Language = Language(
    code_1="nr",
    code_3="nbl",
    name="South Ndebele",
    _and_separator="na",
    _or_separator="namkha",
)
NDONGA: Language = Language(
    code_1="ng", code_3="ndo", name="Ndonga", _and_separator="na", _or_separator="nenge"
)
NEPALI: Language = Language(
    code_1="ne", code_3="nep", name="Nepali", _and_separator="र", _or_separator="वा"
)
NORWEGIAN: Language = Language(
    code_1="no",
    code_3="nor",
    name="Norwegian",
    _and_separator="og",
    _or_separator="eller",
)
NORWEGIAN_BOKMÅL: Language = Language(
    code_1="nb",
    code_3="nob",
    name="Norwegian Bokmål",
    _and_separator="og",
    _or_separator="eller",
)
NORWEGIAN_NYNORSK: Language = Language(
    code_1="nn",
    code_3="nno",
    name="Norwegian Nynorsk",
    _and_separator="og",
    _or_separator="eller",
)
OCCITAN: Language = Language(
    code_1="oc", code_3="oci", name="Occitan", _and_separator="e", _or_separator="o"
)
OJIBWA: Language = Language(
    code_1="oj",
    code_3="oji",
    name="Ojibwa",
    _and_separator="miinawaa",
    _or_separator="jiishin",
)
ORIYA: Language = Language(
    code_1="or", code_3="ori", name="Oriya", _and_separator="ଏବଂ", _or_separator="କିମ୍ବା"
)
OROMO: Language = Language(
    code_1="om", code_3="orm", name="Oromo", _and_separator="fi", _or_separator="yookan"
)
OSSETIAN: Language = Language(
    code_1="os",
    code_3="oss",
    name="Ossetian",
    _and_separator="æмæ",
    _or_separator="кæнæ",
)
PALI: Language = Language(
    code_1="pi", code_3="pli", name="Pali", _and_separator="ca", _or_separator="vā"
)
PASHTO: Language = Language(
    code_1="ps", code_3="pus", name="Pashto", _and_separator="او", _or_separator="يا"
)
PERSIAN: Language = Language(
    code_1="fa", code_3="fas", name="Persian", _and_separator="و", _or_separator="یا"
)
POLISH: Language = Language(
    code_1="pl", code_3="pol", name="Polish", _and_separator="i", _or_separator="lub"
)
PORTUGUESE: Language = Language(
    code_1="pt", code_3="por", name="Portuguese", _and_separator="e", _or_separator="ou"
)
EUROPEAN_PORTUGUESE: Language = Language(
    code_1="pt-pt",
    code_3="por",
    name="European Portuguese",
    _and_separator="e",
    _or_separator="ou",
)
BRAZILIAN_PORTUGUESE: Language = Language(
    code_1="pt-br",
    code_3="por",
    name="Brazilian Portuguese",
    _and_separator="e",
    _or_separator="ou",
)
PUNJABI: Language = Language(
    code_1="pa", code_3="pan", name="Punjabi", _and_separator="ਅਤੇ", _or_separator="ਜਾਂ"
)
QUECHUA: Language = Language(
    code_1="qu",
    code_3="que",
    name="Quechua",
    _and_separator="-pas",
    _or_separator="ichataq",
)
ROMANIAN: Language = Language(
    code_1="ro", code_3="ron", name="Romanian", _and_separator="și", _or_separator="sau"
)
ROMANSH: Language = Language(
    code_1="rm", code_3="roh", name="Romansh", _and_separator="e", _or_separator="u"
)
RUNDI: Language = Language(
    code_1="rn", code_3="run", name="Rundi", _and_separator="na", _or_separator="canke"
)
RUSSIAN: Language = Language(
    code_1="ru", code_3="rus", name="Russian", _and_separator="и", _or_separator="или"
)
NORTHERN_SAMI: Language = Language(
    code_1="se",
    code_3="sme",
    name="Northern Sami",
    _and_separator="ja",
    _or_separator="dahje",
)
SAMOAN: Language = Language(
    code_1="sm", code_3="smo", name="Samoan", _and_separator="ma", _or_separator="poʻo"
)
SANGO: Language = Language(
    code_1="sg", code_3="sag", name="Sango", _and_separator="na", _or_separator="wala"
)
SANSKRIT: Language = Language(
    code_1="sa", code_3="san", name="Sanskrit", _and_separator="च", _or_separator="वा"
)
SARDINIAN: Language = Language(
    code_1="sc", code_3="srd", name="Sardinian", _and_separator="e", _or_separator="o"
)
SERBIAN: Language = Language(
    code_1="sr", code_3="srp", name="Serbian", _and_separator="и", _or_separator="или"
)
SHONA: Language = Language(
    code_1="sn", code_3="sna", name="Shona", _and_separator="uye", _or_separator="kana"
)
SINDHI: Language = Language(
    code_1="sd", code_3="snd", name="Sindhi", _and_separator="۽", _or_separator="يا"
)
SINHALA: Language = Language(
    code_1="si", code_3="sin", name="Sinhala", _and_separator="සහ", _or_separator="හෝ"
)
SLOVAK: Language = Language(
    code_1="sk", code_3="slk", name="Slovak", _and_separator="a", _or_separator="alebo"
)
SLOVENE: Language = Language(
    code_1="sl", code_3="slv", name="Slovene", _and_separator="in", _or_separator="ali"
)
SOMALI: Language = Language(
    code_1="so", code_3="som", name="Somali", _and_separator="iyo", _or_separator="ama"
)
SOTHO: Language = Language(
    code_1="st", code_3="sot", name="Sotho", _and_separator="le", _or_separator="kapa"
)
SPANISH: Language = Language(
    code_1="es", code_3="spa", name="Spanish", _and_separator="y", _or_separator="o"
)
SUNDANESE: Language = Language(
    code_1="su",
    code_3="sun",
    name="Sundanese",
    _and_separator="jeung",
    _or_separator="atawa",
)
SWAHILI: Language = Language(
    code_1="sw", code_3="swa", name="Swahili", _and_separator="na", _or_separator="au"
)
SWATI: Language = Language(
    code_1="ss", code_3="ssw", name="Swati", _and_separator="na", _or_separator="noma"
)
SWEDISH: Language = Language(
    code_1="sv",
    code_3="swe",
    name="Swedish",
    _and_separator="och",
    _or_separator="eller",
)
TAGALOG: Language = Language(
    code_1="tl", code_3="tgl", name="Tagalog", _and_separator="at", _or_separator="o"
)
TAHITIAN: Language = Language(
    code_1="ty",
    code_3="tah",
    name="Tahitian",
    _and_separator="e",
    _or_separator="aore ra",
)
TAJIK: Language = Language(
    code_1="tg", code_3="tgk", name="Tajik", _and_separator="ва", _or_separator="ё"
)
TAMIL: Language = Language(
    code_1="ta",
    code_3="tam",
    name="Tamil",
    _and_separator="மற்றும்",
    _or_separator="அல்லது",
)
TATAR: Language = Language(
    code_1="tt", code_3="tat", name="Tatar", _and_separator="һәм", _or_separator="яки"
)
TELUGU: Language = Language(
    code_1="te", code_3="tel", name="Telugu", _and_separator="మరియు", _or_separator="లేదా"
)
THAI: Language = Language(
    code_1="th", code_3="tha", name="Thai", _and_separator="และ", _or_separator="หรือ"
)
TIBETAN: Language = Language(
    code_1="bo",
    code_3="bod",
    name="Tibetan",
    _and_separator="དང་",
    _or_separator="ཡང་ན",
)
TIGRINYA: Language = Language(
    code_1="ti", code_3="tir", name="Tigrinya", _and_separator="ን", _or_separator="ወይ"
)
TONGA: Language = Language(
    code_1="to", code_3="ton", name="Tonga", _and_separator="mo", _or_separator="pe"
)
TSONGA: Language = Language(
    code_1="ts", code_3="tso", name="Tsonga", _and_separator="na", _or_separator="kumbe"
)
TSWANA: Language = Language(
    code_1="tn",
    code_3="tsn",
    name="Tswana",
    _and_separator="le",
    _or_separator="kgotsa",
)
TURKISH: Language = Language(
    code_1="tr", code_3="tur", name="Turkish", _and_separator="ve", _or_separator="veya"
)
TURKMEN: Language = Language(
    code_1="tk",
    code_3="tuk",
    name="Turkmen",
    _and_separator="we",
    _or_separator="ýa-da",
)
TWI: Language = Language(
    code_1="tw", code_3="twi", name="Twi", _and_separator="ne", _or_separator="anaa"
)
UIGHUR: Language = Language(
    code_1="ug", code_3="uig", name="Uighur", _and_separator="ۋە", _or_separator="ياكى"
)
UKRAINIAN: Language = Language(
    code_1="uk", code_3="ukr", name="Ukrainian", _and_separator="і", _or_separator="або"
)
URDU: Language = Language(
    code_1="ur", code_3="urd", name="Urdu", _and_separator="اور", _or_separator="یا"
)
UZBEK: Language = Language(
    code_1="uz", code_3="uzb", name="Uzbek", _and_separator="va", _or_separator="yoki"
)
VENDA: Language = Language(
    code_1="ve", code_3="ven", name="Venda", _and_separator="na", _or_separator="kana"
)
VIETNAMESE: Language = Language(
    code_1="vi",
    code_3="vie",
    name="Vietnamese",
    _and_separator="và",
    _or_separator="hoặc",
)
VOLAPÜK: Language = Language(
    code_1="vo", code_3="vol", name="Volapük", _and_separator="e", _or_separator="u"
)
WALLOON: Language = Language(
    code_1="wa", code_3="wln", name="Walloon", _and_separator="et", _or_separator="ou"
)
WELSH: Language = Language(
    code_1="cy", code_3="cym", name="Welsh", _and_separator="a", _or_separator="neu"
)
WOLOF: Language = Language(
    code_1="wo", code_3="wol", name="Wolof", _and_separator="ak", _or_separator="walla"
)
XHOSA: Language = Language(
    code_1="xh",
    code_3="xho",
    name="Xhosa",
    _and_separator="kwaye",
    _or_separator="okanye",
)
YIDDISH: Language = Language(
    code_1="yi",
    code_3="yid",
    name="Yiddish",
    _and_separator="און",
    _or_separator="אָדער",
)
YORUBA: Language = Language(
    code_1="yo", code_3="yor", name="Yoruba", _and_separator="àti", _or_separator="tàbí"
)
ZHUANG: Language = Language(
    code_1="za",
    code_3="zha",
    name="Zhuang",
    _and_separator="kae",
    _or_separator="aevih",
)
ZULU: Language = Language(
    code_1="zu", code_3="zul", name="Zulu", _and_separator="futhi", _or_separator="noma"
)
ACEHNESE: Language = Language(
    code_3="ace", name="Acehnese", _and_separator="ngon", _or_separator="atɔ"
)
ADYGHE: Language = Language(
    code_3="ady", name="Adyghe", _and_separator="рэ", _or_separator="е"
)
SOUTHERN_ALTAI: Language = Language(
    code_3="alt", name="Southern Altai", _and_separator="ла", _or_separator="эмезе"
)
AMIS: Language = Language(
    code_3="ami", name="Amis", _and_separator="ato", _or_separator="o"
)
OLD_ENGLISH: Language = Language(
    code_3="ang", name="Old English", _and_separator="and", _or_separator="oþþe"
)
ANGIKA: Language = Language(
    code_3="anp", name="Angika", _and_separator="आर", _or_separator="या"
)
ARAMAIC: Language = Language(
    code_3="arc", name="Aramaic", _and_separator="ܘ", _or_separator="ܐܘ"
)
MOROCCAN_ARABIC: Language = Language(
    code_3="ary", name="Moroccan Arabic", _and_separator="w", _or_separator="wella"
)
EGYPTIAN_ARABIC: Language = Language(
    code_3="arz", name="Egyptian Arabic", _and_separator="و", _or_separator="أو"
)
ASTURIAN: Language = Language(
    code_3="ast", name="Asturian", _and_separator="y", _or_separator="o"
)
ATIKAMEKW: Language = Language(
    code_3="atj", name="Atikamekw", _and_separator="et", _or_separator="ou"
)
KOTAVA: Language = Language(
    code_3="avk", name="Kotava", _and_separator="is", _or_separator="en"
)
AWADHI: Language = Language(
    code_3="awa", name="Awadhi", _and_separator="अउ", _or_separator="या"
)
SOUTH_AZERBAIJANI: Language = Language(
    code_3="azb", name="South Azerbaijani", _and_separator="و", _or_separator="یوخسا"
)
BALINESE: Language = Language(
    code_3="ban", name="Balinese", _and_separator="lan", _or_separator="utawi"
)
BAVARIAN: Language = Language(
    code_3="bar", name="Bavarian", _and_separator="und", _or_separator="oda"
)
CENTRAL_BIKOL: Language = Language(
    code_3="bcl", name="Central Bikol", _and_separator="asin", _or_separator="o"
)
BANJAR: Language = Language(
    code_3="bjn", name="Banjar", _and_separator="wan", _or_separator="atawa"
)
PAO: Language = Language(
    code_3="blk", name="Pa'O", _and_separator="နန်", _or_separator="မု"
)
BISHNUPRIYA: Language = Language(
    code_3="bpy", name="Bishnupriya", _and_separator="आ", _or_separator="বা"
)
BUGINESE: Language = Language(
    code_3="bug", name="Buginese", _and_separator="na", _or_separator="iyarega"
)
BURIAT: Language = Language(
    code_3="bxr", name="Buriat", _and_separator="ба", _or_separator="али"
)
MINDONG_CHINESE: Language = Language(
    code_3="cdo", name="Mindong Chinese", _and_separator="共", _or_separator="或者"
)
CEBUANO: Language = Language(
    code_3="ceb", name="Cebuano", _and_separator="ug", _or_separator="o"
)
CHEROKEE: Language = Language(
    code_3="chr", name="Cherokee", _and_separator="ᎠᎴ", _or_separator="ᎠᎴ"
)
CHEYENNE: Language = Language(
    code_3="chy", name="Cheyenne", _and_separator="na", _or_separator="hēme"
)
CENTRAL_KURDISH: Language = Language(
    code_3="ckb", name="Central Kurdish", _and_separator="و", _or_separator="یان"
)
CRIMEAN_TATAR: Language = Language(
    code_3="crh", name="Crimean Tatar", _and_separator="ve", _or_separator="ya da"
)
KASHUBIAN: Language = Language(
    code_3="csb", name="Kashubian", _and_separator="ë", _or_separator="abò"
)
DAGBANI: Language = Language(
    code_3="dag", name="Dagbani", _and_separator="n-ti", _or_separator="bee"
)
DINKA: Language = Language(
    code_3="din", name="Dinka", _and_separator="ka", _or_separator="ke"
)
DIMLI: Language = Language(
    code_3="diq", name="Dimli", _and_separator="û", _or_separator="ya"
)
LOWER_SORBIAN: Language = Language(
    code_3="dsb", name="Lower Sorbian", _and_separator="a", _or_separator="abo"
)
DOTELI: Language = Language(
    code_3="dty", name="Doteli", _and_separator="र", _or_separator="या"
)
EXTREMADURAN: Language = Language(
    code_3="ext", name="Extremaduran", _and_separator="y", _or_separator="u"
)
FANTI: Language = Language(
    code_3="fat", name="Fanti", _and_separator="na", _or_separator="anaa"
)
FON: Language = Language(
    code_3="fon", name="Fon", _and_separator="kpóɖó", _or_separator="kabi"
)
ARPITAN: Language = Language(
    code_3="frp", name="Arpitan", _and_separator="et", _or_separator="ou"
)
NORTHERN_FRISIAN: Language = Language(
    code_3="frr", name="Northern Frisian", _and_separator="an", _or_separator="of"
)
FRIULIAN: Language = Language(
    code_3="fur", name="Friulian", _and_separator="e", _or_separator="o"
)
GAGAUZ: Language = Language(
    code_3="gag", name="Gagauz", _and_separator="hem", _or_separator="ya"
)
GAN_CHINESE: Language = Language(
    code_3="gan", name="Gan Chinese", _and_separator="同", _or_separator="或"
)
GUIANAN_CREOLE: Language = Language(
    code_3="gcr", name="Guianan Creole", _and_separator="ké", _or_separator="ou"
)
GILAKI: Language = Language(
    code_3="glk", name="Gilaki", _and_separator="و", _or_separator="یا"
)
GOAN_KONKANI: Language = Language(
    code_3="gom", name="Goan Konkani", _and_separator="आनी", _or_separator="वा"
)
GORONTALO: Language = Language(
    code_3="gor", name="Gorontalo", _and_separator="wawu", _or_separator="meyalo"
)
GOTHIC: Language = Language(
    code_3="got", name="Gothic", _and_separator="jah", _or_separator="aiþþau"
)
GHANAIAN_PIDGIN: Language = Language(
    code_3="gpe", name="Ghanaian Pidgin", _and_separator="and", _or_separator="anaa"
)
WAYUU: Language = Language(
    code_3="guc", name="Wayuu", _and_separator="je", _or_separator="yaa"
)
FRAFRA: Language = Language(
    code_3="gur", name="Frafra", _and_separator="la", _or_separator="bee"
)
GUN: Language = Language(
    code_3="guw", name="Gun", _and_separator="pódó", _or_separator="yèkì"
)
HAKKA_CHINESE: Language = Language(
    code_3="hak", name="Hakka Chinese", _and_separator="同", _or_separator="或者"
)
HAWAIIAN: Language = Language(
    code_3="haw", name="Hawaiian", _and_separator="a", _or_separator="a i ʻole"
)
FIJI_HINDI: Language = Language(
    code_3="hif", name="Fiji Hindi", _and_separator="aur", _or_separator="ki"
)
UPPER_SORBIAN: Language = Language(
    code_3="hsb", name="Upper Sorbian", _and_separator="a", _or_separator="abo"
)
WESTERN_ARMENIAN: Language = Language(
    code_3="hyw", name="Western Armenian", _and_separator="եւ", _or_separator="կամ"
)
ILOKO: Language = Language(
    code_3="ilo", name="Iloko", _and_separator="ken", _or_separator="wenno"
)
INGUSH: Language = Language(
    code_3="inh", name="Ingush", _and_separator="и", _or_separator="е"
)
JAMAICAN_CREOLE: Language = Language(
    code_3="jam", name="Jamaican Creole", _and_separator="an", _or_separator="ar"
)
LOJBAN: Language = Language(
    code_3="jbo", name="Lojban", _and_separator="e", _or_separator="a"
)
KARA_KALPAK: Language = Language(
    code_3="kaa", name="Kara-Kalpak", _and_separator="ha'm", _or_separator="yamasa"
)
KABYLE: Language = Language(
    code_3="kab", name="Kabyle", _and_separator="d", _or_separator="neɣ"
)
KABARDIAN: Language = Language(
    code_3="kbd", name="Kabardian", _and_separator="рэ", _or_separator="хэтӀэ"
)
KABIYÈ: Language = Language(
    code_3="kbp", name="Kabiyè", _and_separator="nɛ", _or_separator="yaa"
)
TYAP: Language = Language(
    code_3="kcg", name="Tyap", _and_separator="ma", _or_separator="a̠ni"
)
KOMI_PERMYAK: Language = Language(
    code_3="koi", name="Komi-Permyak", _and_separator="да", _or_separator="либӧ"
)
KARACHAY_BALKAR: Language = Language(
    code_3="krc", name="Karachay-Balkar", _and_separator="бла", _or_separator="не да"
)
LADINO: Language = Language(
    code_3="lad", name="Ladino", _and_separator="i", _or_separator="o"
)
LAK: Language = Language(
    code_3="lbe", name="Lak", _and_separator="ва", _or_separator="ягу"
)
LEZGHIAN: Language = Language(
    code_3="lez", name="Lezghian", _and_separator="ва", _or_separator="я"
)
LINGUA_FRANCA_NOVA: Language = Language(
    code_3="lfn", name="Lingua Franca Nova", _and_separator="e", _or_separator="o"
)
LIGURIAN: Language = Language(
    code_3="lij", name="Ligurian", _and_separator="e", _or_separator="ò"
)
LADIN: Language = Language(
    code_3="lld", name="Ladin", _and_separator="y", _or_separator="o"
)
LOMBARD: Language = Language(
    code_3="lmo", name="Lombard", _and_separator="e", _or_separator="o"
)
LATGALIAN: Language = Language(
    code_3="ltg", name="Latgalian", _and_separator="i", _or_separator="voi"
)
MADURESE: Language = Language(
    code_3="mad", name="Madurese", _and_separator="ban", _or_separator="o"
)
MAITHILI: Language = Language(
    code_3="mai", name="Maithili", _and_separator="आ", _or_separator="वा"
)
MOKSHA: Language = Language(
    code_3="mdf", name="Moksha", _and_separator="ди", _or_separator="или"
)
EASTERN_MARI: Language = Language(
    code_3="mhr", name="Eastern Mari", _and_separator="да", _or_separator="o"
)
MINANGKABAU: Language = Language(
    code_3="min", name="Minangkabau", _and_separator="jo", _or_separator="atau"
)
MANIPURI: Language = Language(
    code_3="mni", name="Manipuri", _and_separator="ꯑꯃꯁꯨꯡ", _or_separator="ꯅꯠꯇ꯭ꯔꯒꯥ"
)
MON: Language = Language(
    code_3="mnw", name="Mon", _and_separator="ကဵု", _or_separator="ဟွံသေင်မ္ဂး"
)
WESTERN_MARI: Language = Language(
    code_3="mrj", name="Western Mari", _and_separator="да", _or_separator="o"
)
MIRANDESE: Language = Language(
    code_3="mwl", name="Mirandese", _and_separator="i", _or_separator="ó"
)
ERZYA: Language = Language(
    code_3="myv", name="Erzya", _and_separator="ды", _or_separator="эли"
)
MAZANDERANI: Language = Language(
    code_3="mzn", name="Mazanderani", _and_separator="و", _or_separator="یا"
)
NEAPOLITAN: Language = Language(
    code_3="nap", name="Neapolitan", _and_separator="e", _or_separator="o"
)
LOW_GERMAN: Language = Language(
    code_3="nds", name="Low German", _and_separator="un", _or_separator="oder"
)
NEWARI: Language = Language(
    code_3="new", name="Newari", _and_separator="व", _or_separator="या"
)
NIAS: Language = Language(
    code_3="nia", name="Nias", _and_separator="ba", _or_separator="mazi"
)
NOVIAL: Language = Language(
    code_3="nov", name="Novial", _and_separator="e", _or_separator="o"
)
NKO: Language = Language(
    code_3="nqo", name="N'Ko", _and_separator="ߣߌ߫", _or_separator="ߥߟߊ߫"
)
NORTHERN_SOTHO: Language = Language(
    code_3="nso", name="Northern Sotho", _and_separator="le", _or_separator="goba"
)
LIVVI_KARELIAN: Language = Language(
    code_3="olo", name="Livvi-Karelian", _and_separator="da", _or_separator="libo"
)
PANGASINAN: Language = Language(
    code_3="pag", name="Pangasinan", _and_separator="tan", _or_separator="odino"
)
PAMPANGA: Language = Language(
    code_3="pam", name="Pampanga", _and_separator="at", _or_separator="o"
)
PAPIAMENTO: Language = Language(
    code_3="pap", name="Papiamento", _and_separator="y", _or_separator="o"
)
PICARD: Language = Language(
    code_3="pcd", name="Picard", _and_separator="pi", _or_separator="ou"
)
NIGERIAN_PIDGIN: Language = Language(
    code_3="pcm", name="Nigerian Pidgin", _and_separator="and", _or_separator="abi"
)
PENNSYLVANIA_GERMAN: Language = Language(
    code_3="pdc", name="Pennsylvania German", _and_separator="un", _or_separator="odder"
)
PALATINE_GERMAN: Language = Language(
    code_3="pfl", name="Palatine German", _and_separator="un", _or_separator="odda"
)
PIEMONTESE: Language = Language(
    code_3="pms", name="Piemontese", _and_separator="e", _or_separator="o"
)
WESTERN_PUNJABI: Language = Language(
    code_3="pnb", name="Western Punjabi", _and_separator="تے", _or_separator="یا"
)
PONTIC: Language = Language(
    code_3="pnt", name="Pontic", _and_separator="και", _or_separator="ή"
)
PAIWAN: Language = Language(
    code_3="pwn", name="Paiwan", _and_separator="dja", _or_separator="uri"
)
VLAX_ROMANI: Language = Language(
    code_3="rmy", name="Vlax Romani", _and_separator="thaj", _or_separator="vaj"
)
RUSYN: Language = Language(
    code_3="rue", name="Rusyn", _and_separator="і", _or_separator="або"
)
YAKUT: Language = Language(
    code_3="sah", name="Yakut", _and_separator="уонна", _or_separator="эбэтэр"
)
SANTALI: Language = Language(
    code_3="sat", name="Santali", _and_separator="ᱟᱨ", _or_separator="ᱥᱮ"
)
SICILIAN: Language = Language(
    code_3="scn", name="Sicilian", _and_separator="e", _or_separator="o"
)
SCOTS: Language = Language(
    code_3="sco", name="Scots", _and_separator="an", _or_separator="or"
)
TACHELHIT: Language = Language(
    code_3="shi", name="Tachelhit", _and_separator="d", _or_separator="neɣ"
)
SHAN: Language = Language(
    code_3="shn", name="Shan", _and_separator="လႄႈ", _or_separator="หรือ"
)
SARAIKI: Language = Language(
    code_3="skr", name="Saraiki", _and_separator="تے", _or_separator="یا"
)
INARI_SAMI: Language = Language(
    code_3="smn", name="Inari Sami", _and_separator="ja", _or_separator="teikkâ"
)
SRANAN: Language = Language(
    code_3="srn", name="Sranan", _and_separator="nanga", _or_separator="efu"
)
SATERLAND_FRISIAN: Language = Language(
    code_3="stq", name="Saterland Frisian", _and_separator="un", _or_separator="of"
)
SILESIAN: Language = Language(
    code_3="szl", name="Silesian", _and_separator="a", _or_separator="abo"
)
SAKIZAYA: Language = Language(
    code_3="szy", name="Sakizaya", _and_separator="ata", _or_separator="uduli"
)
ATAYAL: Language = Language(
    code_3="tay", name="Atayal", _and_separator="daha", _or_separator="ima"
)
TULU: Language = Language(
    code_3="tcy", name="Tulu", _and_separator="ಬೊಕ್ಕ", _or_separator="ಅತ್ತಂಡ"
)
TETUM: Language = Language(
    code_3="tet", name="Tetum", _and_separator="no", _or_separator="ka"
)
TALYSH: Language = Language(
    code_3="tly", name="Talysh", _and_separator="u", _or_separator="jo"
)
TOK_PISIN: Language = Language(
    code_3="tpi", name="Tok Pisin", _and_separator="na", _or_separator="o"
)
TAROKO: Language = Language(
    code_3="trv", name="Taroko", _and_separator="daha", _or_separator="ima"
)
TUMBUKA: Language = Language(
    code_3="tum", name="Tumbuka", _and_separator="na", _or_separator="panji"
)
TUVINIAN: Language = Language(
    code_3="tyv", name="Tuvinian", _and_separator=" болгаш", _or_separator="азы"
)
UDMURT: Language = Language(
    code_3="udm", name="Udmurt", _and_separator="но", _or_separator="яке"
)
VENETIAN: Language = Language(
    code_3="vec", name="Venetian", _and_separator="e", _or_separator="o"
)
VEPS: Language = Language(
    code_3="vep", name="Veps", _and_separator="da", _or_separator="vai"
)
WEST_FLEMISH: Language = Language(
    code_3="vls", name="West Flemish", _and_separator="en", _or_separator="of"
)
WARAY: Language = Language(
    code_3="war", name="Waray", _and_separator="ug", _or_separator="o"
)
WU_CHINESE: Language = Language(
    code_3="wuu", name="Wu Chinese", _and_separator="搭", _or_separator="或者"
)
KALMYK: Language = Language(
    code_3="xal", name="Kalmyk", _and_separator="болн", _or_separator="эсвл"
)
MINGRELIAN: Language = Language(
    code_3="xmf", name="Mingrelian", _and_separator="დო", _or_separator="ვარდა"
)
ZEELANDIC: Language = Language(
    code_3="zea", name="Zeelandic", _and_separator="en", _or_separator="of"
)
CANTONESE: Language = Language(
    code_3="yue", name="Cantonese", _and_separator="同", _or_separator="或者"
)


def is_language_code(name: str) -> bool:
    """Return whether a name is a language code rather than some other label.

    Dataset configurations are often named after the language they hold, and telling
    `zho`, a language EuroEval lacks, apart from `default`, which was never a language,
    decides whether a configuration is skipped or attributed from the repository card.
    EuroEval only knows the codes it supports, so an unsupported one cannot be looked
    up; what it can be is recognised as a code by shape, which is how `zho` and
    `default` are told apart.

    Args:
        name: A configuration name or bare language code, in any case.

    Returns:
        Whether the name is an ISO 639-1 or ISO 639-3 code, or is shaped like one.
    """
    code = name.lower()
    if get_language(code) is not None:
        return True
    return len(code) in (2, 3) and code.isascii() and code.isalpha()


def get_language(language_code: str) -> Language | None:
    """Look up a language by its ISO 639-1 or ISO 639-3 code.

    Args:
        language_code:
            The ISO 639-1 or ISO 639-3 code of the language, in any case.

    Returns:
        The language, if it is supported by EuroEval, and None otherwise. A 639-3 code
        shared by regional variants which have no untagged entry of their own is
        ambiguous, and resolves to no language at all.
    """
    code = language_code.lower()
    language = get_all_languages().get(code)
    if language is not None:
        return language
    candidates = [
        candidate
        for candidate in get_all_languages().values()
        if candidate.code_3 == code
    ]
    if not candidates:
        return None
    untagged = [candidate for candidate in candidates if "-" not in candidate.code]
    if len(untagged) == 1:
        return untagged[0]
    if len(candidates) == 1:
        return candidates[0]
    return None
