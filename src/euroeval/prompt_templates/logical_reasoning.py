"""Templates for the Logical Reasoning task."""

import typing as t

from ..data_models import PromptConfig
from ..languages import (
    ALBANIAN,
    BELARUSIAN,
    BOSNIAN,
    BULGARIAN,
    CATALAN,
    CROATIAN,
    CZECH,
    DANISH,
    DUTCH,
    ENGLISH,
    ESTONIAN,
    FAROESE,
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
    NORWEGIAN_BOKMÅL,
    NORWEGIAN_NYNORSK,
    POLISH,
    PORTUGUESE,
    ROMANIAN,
    SERBIAN,
    SLOVAK,
    SLOVENE,
    SPANISH,
    SWEDISH,
    UKRAINIAN,
)

if t.TYPE_CHECKING:
    from ..languages import Language

LOGIC_TEMPLATES: dict["Language", PromptConfig] = {
    ALBANIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Ja një enigmë:\n<riddle>\n{text}\n</riddle>\n\n"
        "Kush ka cilat karakteristika dhe banon në cilin shtëpi?\n\n"
        "Ju lutemi jepni përgjigjen tuaj si një JSON dictionary. Çdo key duhet të "
        "jetë object_X ku X është numri i shtëpisë. Çdo value duhet të jetë një listë "
        "e karakteristikave nga kategoritë e mësipërme që i përkasin personit në "
        "shtëpinë nr. X.",
        default_prompt_label_mapping=dict(),
    ),
    BELARUSIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Вось загадка:\n<riddle>\n{text}\n</riddle>\n\n"
        "Хто мае якія ўласцівасці і жыве ў якім доме?\n\n"
        "Калі ласка, дайце свой адказ у выглядзе JSON dictionary. Кожны key павінен "
        "быць object_X, дзе X — нумар дома. Кожнае value павінна быць спісам "
        "уласцівасцей з вышэйпералічаных катэгорый, якія належаць чалавеку ў доме "
        "нумар X.",
        default_prompt_label_mapping=dict(),
    ),
    BOSNIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Evo zagonetke:\n<riddle>\n{text}\n</riddle>\n\n"
        "Tko ima koje osobine i živi u kojoj kući?\n\n"
        "Molimo da svoj odgovor date kao JSON dictionary. Svaki key treba biti "
        "object_X gdje je X broj kuće. Svaka value treba biti popis osobina iz "
        "gornjih kategorija koje pripadaju osobi u kući broj X.",
        default_prompt_label_mapping=dict(),
    ),
    BULGARIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Ето една гатанка:\n<riddle>\n{text}\n</riddle>\n\n"
        "Кой има какви характеристики и живее в коя къща?\n\n"
        "Моля, предоставете отговора си като JSON dictionary. Всеки key трябва да "
        "бъде object_X, където X е номерът на къщата. Всяка value трябва да бъде "
        "списък с характеристиките от категориите по-горе, които принадлежат на "
        "лицето в къща номер X.",
        default_prompt_label_mapping=dict(),
    ),
    CATALAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Aquí teniu un enigma:\n"
        "<riddle>\n{text}\n</riddle>\n\n"
        "Qui té quines característiques i viu en quina casa?\n\n"
        "Si us plau, proporcioneu la vostra resposta com un JSON dictionary. Cada "
        "key ha de ser object_X on X és el número de la casa. Cada value ha de ser "
        "una llista de les característiques de les categories anteriors que pertanyen "
        "a la persona de la casa número X.",
        default_prompt_label_mapping=dict(),
    ),
    CROATIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Evo zagonetke:\n<riddle>\n{text}\n</riddle>\n\n"
        "Tko ima koje osobine i živi u kojoj kući?\n\n"
        "Molimo da svoj odgovor date kao JSON dictionary. Svaki key treba biti "
        "object_X gdje je X broj kuće. Svaka value treba biti popis osobina iz "
        "gornjih kategorija koje pripadaju osobi u kući broj X.",
        default_prompt_label_mapping=dict(),
    ),
    CZECH: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Zde je hádanka:\n<riddle>\n{text}\n</riddle>\n\n"
        "Kdo má jaké vlastnosti a bydlí v kterém domě?\n\n"
        "Uveďte prosím svou odpověď jako JSON dictionary. Každý key by měl být "
        "object_X, kde X je číslo domu. Každá value by měla být seznamem vlastností "
        "z výše uvedených kategorií, které patří osobě v domě číslo X.",
        default_prompt_label_mapping=dict(),
    ),
    DANISH: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Her er en gåde:\n<riddle>\n{text}\n</riddle>\n\n"
        "Hvem har hvilke egenskaber og bor i hvilket hus?\n\n"
        "Angiv venligst dit svar som et JSON dictionary. Hver key skal være object_X "
        "hvor X er husnummeret. Hver value skal være en liste med de egenskaber fra "
        "kategorierne ovenfor som tilhører personen i hus nr. X.",
        default_prompt_label_mapping=dict(),
    ),
    DUTCH: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Hier is een raadsel:\n"
        "<riddle>\n{text}\n</riddle>\n\n"
        "Wie heeft welke eigenschappen en woont in welk huis?\n\n"
        "Geef je antwoord als een JSON dictionary. Elke key moet object_X zijn, waarbij"
        " X het huisnummer is. Elke value moet een lijst zijn van de eigenschappen uit "
        "de bovengenoemde categorieën die horen bij de persoon in huis nr. X.",
        default_prompt_label_mapping=dict(),
    ),
    ENGLISH: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Here is a riddle:\n<riddle>\n{text}\n</riddle>\n\n"
        "Who has which attributes and lives in which house?\n\n"
        "Please submit your answer as a JSON dictionary. Each key must be object_X "
        "where X is the house number. Each value must be a list of the attributes from "
        "the aforementioned categories that belong to the person in house no. X.",
        default_prompt_label_mapping=dict(),
    ),
    ESTONIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Siin on mõistatus:\n<riddle>\n{text}\n</riddle>\n\n"
        "Kellel on millised omadused ja kes elab millises majas?\n\n"
        "Palun esitage oma vastus JSON dictionary. Iga key peaks olema object_X, "
        "kus X on maja number. Iga value peaks olema loend omadustest ülaltoodud "
        "kategooriatest, mis kuuluvad majas nummer X elavale isikule.",
        default_prompt_label_mapping=dict(),
    ),
    FAROESE: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Her er ein gáta:\n<riddle>\n{text}\n</riddle>\n\n"
        "Hvør hevur hvørjar eginleikar og býr í hvørjum húsum?\n\n"
        "Vinarliga gev títt svar sum JSON dictionary. Hvør key skal vera object_X har X"
        " er húsanummarið. Hvør value skal vera ein listi við eginleikum úr áðurnevndu "
        "flokkunum, sum tilhoyra persóninum í húsi nr. X.",
        default_prompt_label_mapping=dict(),
    ),
    FINNISH: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Tässä on arvoitus:\n<riddle>\n{text}\n</riddle>\n\n"
        "Kenellä on mitkä ominaisuudet ja kuka asuu missäkin talossa?\n\n"
        "Anna vastauksesi JSON dictionary. Jokaisen key tulee olla object_X, "
        "jossa X on talon numero. Jokaisen value tulee olla luettelo yllä olevien "
        "kategorioiden ominaisuuksista, jotka kuuluvat talossa numero X asuvalle "
        "henkilölle.",
        default_prompt_label_mapping=dict(),
    ),
    FRENCH: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Voici une énigme :\n<riddle>\n{text}\n</riddle>\n\n"
        "Qui a quelles caractéristiques et habite dans quelle maison ?\n\n"
        "Veuillez fournir votre réponse sous forme de JSON dictionary. Chaque key "
        "doit être object_X où X est le numéro de la maison. Chaque value doit être "
        "une liste des caractéristiques des catégories ci-dessus qui appartiennent à "
        "la personne dans la maison numéro X.",
        default_prompt_label_mapping=dict(),
    ),
    GERMAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Hier ist ein Rätsel:\n"
        "<riddle>\n{text}\n</riddle>\n\n"
        "Wer hat welche Eigenschaften und wohnt in welchem Haus?\n\n"
        "Bitte geben Sie Ihre Antwort als JSON-Dictionary an. Jeder Key sollte object_X"
        " sein, wobei X die Hausnummer ist. Jeder Value sollte eine Liste der "
        "Eigenschaften aus den aufgelisteten Kategorien sein, die zur Person in Haus "
        "Nummer X gehören.",
        default_prompt_label_mapping=dict(),
    ),
    GREEK: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Ορίστε ένα αίνιγμα:\n"
        "<riddle>\n{text}\n</riddle>\n\n"
        "Ποιος έχει ποια χαρακτηριστικά και μένει σε ποιο σπίτι;\n\n"
        "Παρακαλώ δώστε την απάντησή σας ως JSON dictionary. Κάθε key πρέπει να "
        "είναι object_X όπου X είναι ο αριθμός του σπιτιού. Κάθε value πρέπει να "
        "είναι λίστα με τα χαρακτηριστικά από τις παραπάνω κατηγορίες που ανήκουν "
        "στο άτομο στο σπίτι αριθμός X.",
        default_prompt_label_mapping=dict(),
    ),
    HUNGARIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Íme egy találós kérdés:\n"
        "<riddle>\n{text}\n</riddle>\n\n"
        "Kinek milyen tulajdonságai vannak és melyik házban lakik?\n\n"
        "Kérjük, adja meg válaszát JSON dictionary. Minden key object_X legyen, ahol "
        "X a ház száma. Minden value legyen lista a fenti kategóriákból származó "
        "tulajdonságokkal, amelyek az X. számú házban lakó személyhez tartoznak.",
        default_prompt_label_mapping=dict(),
    ),
    ICELANDIC: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Hér er ráða:\n<riddle>\n{text}\n</riddle>\n\n"
        "Hver hefur hvaða einkenni og býr í hvaða húsi?\n\n"
        "Vinsamlegast gefðu svarið þitt sem JSON dictionary. Hvert key ætti að vera "
        "object_X þar sem X er húsnúmerið. Hvert value ætti að vera listi með einkennum"
        " úr áðurnefndum flokkum sem tilheyra einstaklingnum í húsi nr. X.",
        default_prompt_label_mapping=dict(),
    ),
    ITALIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Ecco un indovinello:\n"
        "<riddle>\n{text}\n</riddle>\n\n"
        "Chi ha quali attributi e vive in quale casa?\n\n"
        "Fornisci la tua risposta come JSON dictionary. Ogni key dovrebbe essere "
        "object_X dove X è il numero della casa. Ogni value dovrebbe essere un elenco "
        "degli attributi dalle categorie sopra che appartengono alla persona nella "
        "casa numero X.",
        default_prompt_label_mapping=dict(),
    ),
    LATVIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Lūk mīkla:\n<riddle>\n{text}\n</riddle>\n\n"
        "Kam ir kādas īpašības un kurš dzīvo kurā mājā?\n\n"
        "Lūdzu, sniedziet savu atbildi kā JSON dictionary. Katrai key jābūt "
        "object_X, kur X ir mājas numurs. Katrai value jābūt sarakstam ar īpašībām "
        "no augstāk minētajām kategorijām, kas pieder personai mājas numurā X.",
        default_prompt_label_mapping=dict(),
    ),
    LITHUANIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Štai mįslė:\n<riddle>\n{text}\n</riddle>\n\n"
        "Kas turi kokias savybes ir gyvena kuriame name?\n\n"
        "Prašom pateikti savo atsakymą kaip JSON dictionary. Kiekvienas key turi būti "
        "object_X, kur X yra namo numeris. Kiekviena value turi būti savybių iš "
        "aukščiau nurodytų kategorijų, priklausančių asmeniui name numeriu X, sąrašas.",
        default_prompt_label_mapping=dict(),
    ),
    NORWEGIAN_BOKMÅL: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Her er en gåte:\n<riddle>\n{text}\n</riddle>\n\n"
        "Hvem har hvilke egenskaper og bor i hvilket hus?\n\n"
        "Vennligst oppgi svaret ditt som en JSON-dictionary. Hver key skal være "
        "object_X der X er husnummeret. Hver value skal være en liste over egenskapene "
        "fra nevnte kategorier som tilhører personen i hus nr. X.",
        default_prompt_label_mapping=dict(),
    ),
    NORWEGIAN_NYNORSK: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Her er ei gåte:\n<riddle>\n{text}\n</riddle>\n\n"
        "Kven har kva eigenskapar og bur i kva for eit hus?\n\n"
        "Gjer vel å oppgi svaret ditt som ein JSON-dictionary. Kvar key skal vera "
        "object_X der X er husnummeret. Kvar value skal vera ei liste over eigenskapane"
        " frå nemnde kategoriar som høyrer til personen i hus nr. X.",
        default_prompt_label_mapping=dict(),
    ),
    NORWEGIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Her er en gåte:\n<riddle>\n{text}\n</riddle>\n\n"
        "Hvem har hvilke egenskaper og bor i hvilket hus?\n\n"
        "Vennligst oppgi svaret ditt som en JSON-dictionary. Hver key skal være "
        "object_X der X er husnummeret. Hver value skal være en liste over egenskapene "
        "fra nevnte kategorier som tilhører personen i hus nr. X.",
        default_prompt_label_mapping=dict(),
    ),
    POLISH: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Oto zagadka:\n<riddle>\n{text}\n</riddle>\n\n"
        "Kto ma jakie cechy i mieszka w którym domu?\n\n"
        "Proszę podać odpowiedź jako JSON dictionary. Każdy key powinien być object_X, "
        "gdzie X to numer domu. Każda value powinna być listą cech z powyższych "
        "kategorii, które należą do osoby w domu numer X.",
        default_prompt_label_mapping=dict(),
    ),
    PORTUGUESE: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Aqui está um enigma:\n"
        "<riddle>\n{text}\n</riddle>\n\n"
        "Quem tem quais atributos e vive em qual casa?\n\n"
        "Por favor, forneça sua resposta como um JSON dictionary. Cada key deve ser "
        "object_X onde X é o número da casa. Cada value deve ser uma lista dos "
        "atributos das categorias acima que pertencem à pessoa na casa número X.",
        default_prompt_label_mapping=dict(),
    ),
    ROMANIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Iată o ghicitoare:\n<riddle>\n{text}\n</riddle>\n\n"
        "Cine are ce atribute și locuiește în care casă?\n\n"
        "Vă rugăm să furnizați răspunsul dvs. ca un JSON dictionary. Fiecare key "
        "trebuie să fie object_X unde X este numărul casei. Fiecare value trebuie să "
        "fie o listă a atributelor din categoriile de mai sus care aparțin persoanei "
        "din casa numărul X.",
        default_prompt_label_mapping=dict(),
    ),
    SERBIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Evo zagonetke:\n<riddle>\n{text}\n</riddle>\n\n"
        "Ko ima koje osobine i živi u kojoj kući?\n\n"
        "Molimo da svoj odgovor date kao JSON dictionary. Svaki key treba da bude "
        "object_X gde je X broj kuće. Svaka value treba da bude lista osobina iz "
        "gornjih kategorija koje pripadaju osobi u kući broj X.",
        default_prompt_label_mapping=dict(),
    ),
    SLOVAK: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Tu je hádanka:\n<riddle>\n{text}\n</riddle>\n\n"
        "Kto má aké vlastnosti a býva v ktorom dome?\n\n"
        "Prosím, uveďte svoju odpoveď ako JSON dictionary. Každý key by mal byť "
        "object_X, kde X je číslo domu. Každá value by mala byť zoznamom vlastností "
        "z vyššie uvedených kategórií, ktoré patria osobe v dome číslo X.",
        default_prompt_label_mapping=dict(),
    ),
    SLOVENE: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Tukaj je uganka:\n<riddle>\n{text}\n</riddle>\n\n"
        "Kdo ima katere lastnosti in živi v kateri hiši?\n\n"
        "Prosimo, podajte svoj odgovor kot JSON dictionary. Vsak key naj bo object_X, "
        "kjer je X številka hiše. Vsaka value naj bo seznam lastnosti iz zgornjih "
        "kategorij, ki pripadajo osebi v hiši številka X.",
        default_prompt_label_mapping=dict(),
    ),
    SPANISH: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Aquí hay un acertijo:\n"
        "<riddle>\n{text}\n</riddle>\n\n"
        "¿Quién tiene qué atributos y vive en qué casa?\n\n"
        "Por favor, proporciona tu respuesta como un JSON dictionary. Cada key debe "
        "ser object_X donde X es el número de la casa. Cada value debe ser una lista "
        "de los atributos de las categorías anteriores que pertenecen a la persona en "
        "la casa número X.",
        default_prompt_label_mapping=dict(),
    ),
    SWEDISH: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Här är en gåta:\n<riddle>\n{text}\n</riddle>\n\n"
        "Vem har vilka egenskaper och bor i vilket hus?\n\n"
        "Vänligen ange ditt svar som en JSON dictionary. Varje key ska vara object_X "
        "där X är husnumret. Varje value ska vara en lista över egenskaperna från "
        "ovannämnda kategorier som tillhör personen i hus nr. X.",
        default_prompt_label_mapping=dict(),
    ),
    UKRAINIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Ось загадка:\n<riddle>\n{text}\n</riddle>\n\n"
        "Хто має які характеристики і живе в якому будинку?\n\n"
        "Будь ласка, надайте свою відповідь як JSON dictionary. Кожен key має бути "
        "object_X, де X — номер будинку. Кожне value має бути списком "
        "характеристик з вищевказаних категорій, які належать особі в будинку "
        "номер X.",
        default_prompt_label_mapping=dict(),
    ),
}
