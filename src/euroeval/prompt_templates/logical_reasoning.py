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
        default_instruction_prompt="Ja një enigmë:\n<riddle>{text}</riddle>\n"
        "Kush ka cilat karakteristika dhe banon në cilin shtëpi? Ju lutemi jepni "
        "përgjigjen tuaj si një fjalor JSON. Çdo kyç duhet të jetë object_X ku X është "
        "numri i shtëpisë. Çdo vlerë duhet të jetë një listë e karakteristikave nga "
        "kategoritë e mësipërme që i përkasin personit në shtëpinë nr. X.",
        default_prompt_label_mapping=dict(),
    ),
    BELARUSIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Вось загадка:\n<riddle>{text}</riddle>\n"
        "Хто мае якія ўласцівасці і жыве ў якім доме? Калі ласка, дайце свой адказ у "
        "выглядзе JSON-слоўніка. Кожны ключ павінен быць object_X, дзе X — нумар дома. "
        "Кожнае значэнне павінна быць спісам уласцівасцей з вышэйпералічаных "
        "катэгорый, "
        "якія належаць чалавеку ў доме нумар X.",
        default_prompt_label_mapping=dict(),
    ),
    BOSNIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Evo zagonetke:\n<riddle>{text}</riddle>\n"
        "Tko ima koje osobine i živi u kojoj kući? Molimo da svoj odgovor date kao "
        "JSON rječnik. Svaki ključ treba biti object_X gdje je X broj kuće. Svaka "
        "vrijednost treba biti popis osobina iz gornjih kategorija koje pripadaju "
        "osobi u kući broj X.",
        default_prompt_label_mapping=dict(),
    ),
    BULGARIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Ето една гатанка:\n<riddle>{text}</riddle>\n"
        "Кой има какви характеристики и живее в коя къща? Моля, предоставете отговора "
        "си като JSON речник. Всеки ключ трябва да бъде object_X, където X е номерът "
        "на къщата. Всяка стойност трябва да бъде списък с характеристиките от "
        "категориите по-горе, които принадлежат на лицето в къща номер X.",
        default_prompt_label_mapping=dict(),
    ),
    CATALAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Aquí teniu un enigma:\n<riddle>{text}</riddle>\n"
        "Qui té quines característiques i viu en quina casa? Si us plau, proporcioneu "
        "la vostra resposta com un diccionari JSON. Cada clau ha de ser object_X on X "
        "és el número de la casa. Cada valor ha de ser una llista de les "
        "característiques de les categories anteriors que pertanyen a la persona de "
        "la casa número X.",
        default_prompt_label_mapping=dict(),
    ),
    CROATIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Evo zagonetke:\n<riddle>{text}</riddle>\n"
        "Tko ima koje osobine i živi u kojoj kući? Molimo da svoj odgovor date kao "
        "JSON rječnik. Svaki ključ treba biti object_X gdje je X broj kuće. Svaka "
        "vrijednost treba biti popis osobina iz gornjih kategorija koje pripadaju "
        "osobi u kući broj X.",
        default_prompt_label_mapping=dict(),
    ),
    CZECH: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Zde je hádanka:\n<riddle>{text}</riddle>\n"
        "Kdo má jaké vlastnosti a bydlí v kterém domě? Uveďte prosím svou odpověď "
        "jako JSON slovník. Každý klíč by měl být object_X, kde X je číslo domu. "
        "Každá hodnota by měla být seznamem vlastností z výše uvedených kategorií, "
        "které patří osobě v domě číslo X.",
        default_prompt_label_mapping=dict(),
    ),
    DANISH: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Her er en gåde:\n<riddle>{text}</riddle>\n"
        "Hvem har hvilke egenskaber og bor i hvilket hus? Angiv venligst dit svar som "
        "en JSON dictionary. Hver key skal være object_X hvor X er husnummeret. Hver "
        "value skal være en liste med de egenskaber fra kategorierne ovenfor som "
        "tilhører personen i hus nr. X.",
        default_prompt_label_mapping=dict(),
    ),
    DUTCH: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Hier is een raadsel:\n<riddle>{text}</riddle>\n"
        "Wie heeft welke eigenschappen en woont in welk huis? Geef je antwoord als "
        "een JSON-woordenboek. Elke sleutel moet object_X zijn, waarbij X het "
        "huisnummer is. Elke waarde moet een lijst zijn van de eigenschappen uit de "
        "bovenstaande categorieën die behoren tot de persoon in huis nummer X.",
        default_prompt_label_mapping=dict(),
    ),
    ENGLISH: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Here is a riddle:\n<riddle>{text}</riddle>\n"
        "Who has which attributes and lives in which house? Please provide your "
        "answer as a JSON dictionary. Each key should be object_X where X is the "
        "house number. Each value should be a list of the attributes from the "
        "categories above that belong to the person in house number X.",
        default_prompt_label_mapping=dict(),
    ),
    ESTONIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Siin on mõistatus:\n<riddle>{text}</riddle>\n"
        "Kellel on millised omadused ja kes elab millises majas? Palun esitage oma "
        "vastus JSON-sõnastikuna. Iga võti peaks olema object_X, kus X on maja number. "
        "Iga väärtus peaks olema loend omadustest ülaltoodud kategooriatest, mis "
        "kuuluvad majas nummer X elavale isikule.",
        default_prompt_label_mapping=dict(),
    ),
    FAROESE: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Her er ein gáta:\n<riddle>{text}</riddle>\n"
        "Hvør hevur hvørjar eginleikar og býr í hvørjum húsi? Vinarliga gev títt svar "
        "sum eina JSON orðabók. Hvør lykil skal vera object_X har X er húsavalið. "
        "Hvørt virði skal vera ein listi yvir eginleikarnar frá bólkunum omanfyri, "
        "sum hoyra til persóninum í húsi nr. X.",
        default_prompt_label_mapping=dict(),
    ),
    FINNISH: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Tässä on arvoitus:\n<riddle>{text}</riddle>\n"
        "Kenellä on mitkä ominaisuudet ja kuka asuu missäkin talossa? Anna "
        "vastauksesi JSON-sanakirjana. Jokaisen avaimen tulee olla object_X, jossa X "
        "on talon numero. Jokaisen arvon tulee olla luettelo yllä olevien "
        "kategorioiden "
        "ominaisuuksista, jotka kuuluvat talossa numero X asuvalle henkilölle.",
        default_prompt_label_mapping=dict(),
    ),
    FRENCH: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Voici une énigme :\n<riddle>{text}</riddle>\n"
        "Qui a quelles caractéristiques et habite dans quelle maison ? Veuillez "
        "fournir votre réponse sous forme de dictionnaire JSON. Chaque clé doit être "
        "object_X où X est le numéro de la maison. Chaque valeur doit être une liste "
        "des caractéristiques des catégories ci-dessus qui appartiennent à la "
        "personne dans la maison numéro X.",
        default_prompt_label_mapping=dict(),
    ),
    GERMAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Hier ist ein Rätsel:\n<riddle>{text}</riddle>\n"
        "Wer hat welche Eigenschaften und wohnt in welchem Haus? Bitte geben Sie "
        "Ihre Antwort als JSON-Wörterbuch an. Jeder Schlüssel sollte object_X sein, "
        "wobei X die Hausnummer ist. Jeder Wert sollte eine Liste der Attribute aus "
        "den obigen Kategorien sein, die zur Person in Haus Nummer X gehören.",
        default_prompt_label_mapping=dict(),
    ),
    GREEK: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Ορίστε ένα αίνιγμα:\n<riddle>{text}</riddle>\n"
        "Ποιος έχει ποια χαρακτηριστικά και μένει σε ποιο σπίτι; Παρακαλώ δώστε την "
        "απάντησή σας ως λεξικό JSON. Κάθε κλειδί πρέπει να είναι object_X όπου X "
        "είναι ο αριθμός του σπιτιού. Κάθε τιμή πρέπει να είναι λίστα με τα "
        "χαρακτηριστικά από τις παραπάνω κατηγορίες που ανήκουν στο άτομο στο σπίτι "
        "αριθμός X.",
        default_prompt_label_mapping=dict(),
    ),
    HUNGARIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Íme egy találós kérdés:\n<riddle>{text}</riddle>\n"
        "Kinek milyen tulajdonságai vannak és melyik házban lakik? Kérjük, adja meg "
        "válaszát JSON szótárként. Minden kulcs object_X legyen, ahol X a ház száma. "
        "Minden érték legyen lista a fenti kategóriákból származó "
        "tulajdonságokkal, amelyek az X. számú házban lakó személyhez tartoznak.",
        default_prompt_label_mapping=dict(),
    ),
    ICELANDIC: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Hér er ráða:\n<riddle>{text}</riddle>\n"
        "Hver hefur hvilkeiginleika og býr í hvílíku húsi? Vinsamlega gefðu svar þitt "
        "sem JSON orðabók. Hver lykill á að vera object_X þar sem X er númer hússins. "
        "Hvert gildi á að vera listi yfir eiginleikana frá flokkunum hér að ofan sem "
        "tilheyra manneskjunni í húsi númer X.",
        default_prompt_label_mapping=dict(),
    ),
    ITALIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Ecco un indovinello:\n<riddle>{text}</riddle>\n"
        "Chi ha quali attributi e vive in quale casa? Fornisci la tua risposta come "
        "dizionario JSON. Ogni chiave dovrebbe essere object_X dove X è il numero "
        "della casa. Ogni valore dovrebbe essere un elenco degli attributi dalle "
        "categorie sopra che appartengono alla persona nella casa numero X.",
        default_prompt_label_mapping=dict(),
    ),
    LATVIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Lūk mīkla:\n<riddle>{text}</riddle>\n"
        "Kam ir kādas īpašības un kurš dzīvo kurā mājā? Lūdzu, sniedziet savu atbildi "
        "kā JSON vārdnīcu. Katrai atslēgai jābūt object_X, kur X ir mājas numurs. "
        "Katrai vērtībai jābūt sarakstam ar īpašībām no augstāk minētajām kategorijām, "
        "kas pieder personai mājas numurā X.",
        default_prompt_label_mapping=dict(),
    ),
    LITHUANIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Štai mįslė:\n<riddle>{text}</riddle>\n"
        "Kas turi kokias savybes ir gyvena kuriame name? Prašom pateikti savo "
        "atsakymą kaip JSON žodyną. Kiekvienas raktas turi būti object_X, kur X yra "
        "namo numeris. Kiekviena reikšmė turi būti savybių iš aukščiau nurodytų "
        "kategorijų, priklausančių asmeniui name numeriu X, sąrašas.",
        default_prompt_label_mapping=dict(),
    ),
    NORWEGIAN_BOKMÅL: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Her er en gåte:\n<riddle>{text}</riddle>\n"
        "Hvem har hvilke egenskaper og bor i hvilket hus? Vennligst oppgi svaret ditt "
        "som en JSON-ordbok. Hver nøkkel skal være object_X hvor X er husnummeret. "
        "Hver verdi skal være en liste med egenskapene fra kategoriene ovenfor som "
        "tilhører personen i hus nummer X.",
        default_prompt_label_mapping=dict(),
    ),
    NORWEGIAN_NYNORSK: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Her er ei gåte:\n<riddle>{text}</riddle>\n"
        "Kven har kva eigenskapar og bur i kva hus? Ver vennleg og oppgi svaret ditt "
        "som ei JSON-ordbok. Kvar nøkkel skal vere object_X der X er husnummeret. "
        "Kvar verdi skal vere ein liste med eigenskapane frå kategoriane ovanfor som "
        "høyrer til personen i hus nummer X.",
        default_prompt_label_mapping=dict(),
    ),
    NORWEGIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Her er en gåte:\n<riddle>{text}</riddle>\n"
        "Hvem har hvilke egenskaper og bor i hvilket hus? Vennligst oppgi svaret ditt "
        "som en JSON-ordbok. Hver nøkkel skal være object_X hvor X er husnummeret. "
        "Hver verdi skal være en liste med egenskapene fra kategoriene ovenfor som "
        "tilhører personen i hus nummer X.",
        default_prompt_label_mapping=dict(),
    ),
    POLISH: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Oto zagadka:\n<riddle>{text}</riddle>\n"
        "Kto ma jakie cechy i mieszka w którym domu? Proszę podać odpowiedź jako "
        "słownik JSON. Każdy klucz powinien być object_X, gdzie X to numer domu. "
        "Każda wartość powinna być listą cech z powyższych kategorii, które należą "
        "do osoby w domu numer X.",
        default_prompt_label_mapping=dict(),
    ),
    PORTUGUESE: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Aqui está um enigma:\n<riddle>{text}</riddle>\n"
        "Quem tem quais atributos e vive em qual casa? Por favor, forneça sua "
        "resposta como um dicionário JSON. Cada chave deve ser object_X onde X é o "
        "número da casa. Cada valor deve ser uma lista dos atributos das categorias "
        "acima que pertencem à pessoa na casa número X.",
        default_prompt_label_mapping=dict(),
    ),
    ROMANIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Iată o ghicitoare:\n<riddle>{text}</riddle>\n"
        "Cine are ce atribute și locuiește în care casă? Vă rugăm să furnizați "
        "răspunsul dvs. ca un dicționar JSON. Fiecare cheie trebuie să fie object_X "
        "unde X este numărul casei. Fiecare valoare trebuie să fie o listă a "
        "atributelor din categoriile de mai sus care aparțin persoanei din casa "
        "numărul X.",
        default_prompt_label_mapping=dict(),
    ),
    SERBIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Evo zagonetke:\n<riddle>{text}</riddle>\n"
        "Ko ima koje osobine i živi u kojoj kući? Molimo da svoj odgovor date kao "
        "JSON rečnik. Svaki ključ treba da bude object_X gde je X broj kuće. Svaka "
        "vrednost treba da bude lista osobina iz gornjih kategorija koje pripadaju "
        "osobi u kući broj X.",
        default_prompt_label_mapping=dict(),
    ),
    SLOVAK: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Tu je hádanka:\n<riddle>{text}</riddle>\n"
        "Kto má aké vlastnosti a býva v ktorom dome? Prosím, uveďte svoju odpoveď "
        "ako JSON slovník. Každý kľúč by mal byť object_X, kde X je číslo domu. "
        "Každá hodnota by mala byť zoznamom vlastností z vyššie uvedených kategórií, "
        "ktoré patria osobe v dome číslo X.",
        default_prompt_label_mapping=dict(),
    ),
    SLOVENE: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Tukaj je uganka:\n<riddle>{text}</riddle>\n"
        "Kdo ima katere lastnosti in živi v kateri hiši? Prosimo, podajte svoj odgovor "
        "kot JSON slovar. Vsak ključ naj bo object_X, kjer je X številka hiše. Vsaka "
        "vrednost naj bo seznam lastnosti iz zgornjih kategorij, ki pripadajo osebi v "
        "hiši številka X.",
        default_prompt_label_mapping=dict(),
    ),
    SPANISH: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Aquí hay un acertijo:\n<riddle>{text}</riddle>\n"
        "¿Quién tiene qué atributos y vive en qué casa? Por favor, proporciona tu "
        "respuesta como un diccionario JSON. Cada clave debe ser object_X donde X es "
        "el número de la casa. Cada valor debe ser una lista de los atributos de las "
        "categorías anteriores que pertenecen a la persona en la casa número X.",
        default_prompt_label_mapping=dict(),
    ),
    SWEDISH: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Här är ett kapitel:\n<riddle>{text}</riddle>\n"
        "Vem har vilka egenskaper och bor i vilket hus? Vänligen ange ditt svar som "
        "en JSON-ordbok. Varje nyckel ska vara object_X där X är husnummer. Varje "
        "värde ska vara en lista över egenskaperna från kategorierna ovan som "
        "tillhör personen i hus nummer X.",
        default_prompt_label_mapping=dict(),
    ),
    UKRAINIAN: PromptConfig(
        default_prompt_prefix="",
        default_prompt_template="",
        default_instruction_prompt="Ось загадка:\n<riddle>{text}</riddle>\n"
        "Хто має які характеристики і живе в якому будинку? Будь ласка, надайте свою "
        "відповідь як JSON-словник. Кожен ключ має бути object_X, де X — номер "
        "будинку. Кожне значення має бути списком характеристик з вищевказаних "
        "категорій, які належать особі в будинку номер X.",
        default_prompt_label_mapping=dict(),
    ),
}
