# 🇷🇴 Romanian

This is an overview of all the datasets used in the Romanian part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### HuSST

This dataset was first published in [this repository](https://github.com/katakonst/sentiment-analysis-tensorflow)
and consists of reviews from yelp and imdb.

The original dataset contains 17,941 / 11,005 samples for the training and test splits,
respectively. We use 1,024 / 256 / 2,048 samples for our training,
validation and test splits, respectively. The train and test splits are subsets of
the original splits, while the validation split is created from the training split.

Here are a few examples from the training split:

```json
{
    "text": "acest film are mari staruri in anii lor mai devreme: ingor stevens nu a fost niciodata mai frumos; yul brynner a fost un jean lafitte foarte convingator, in conflict cu pirateria sa si dorind sa pastreze neutralitatea cu statele unite. charlton heston a facut o treaba destul de buna ca andrew jackson, dar cateva momente au fost un pic stomac. este un film bun pentru elevii sa invete acea parte a istoriei noastre si arata ca toate incheierile fericite nu includ iubitorii care se intalnesc unul cu celalalt - uneori, sfarsitul fericit este acela ca ei navigheaza departe si gasesc parteneri de acelasi gen care le va intelege mai bine pe termen lung. am vazut-o in fiecare an de cel putin doua ori, timp de 16 ani; si desi nu este cel mai bun film pe care l-am vazut vreodata, il iubesc de fiecare data!",
    "label": "positive"
}
```

```json
{
    "text": "un film foarte interesant, inteligent si bine facut. liam neeson si tim roth joaca foarte bine rolurile lor. cinematografia este remarcabila. scenele de lupta sunt uimitoare. acesta este un film pe care il voi bucura de vizionarea din nou si din nou. unul dintre preferatele mele.",
    "label": "positive"
}
```

```json
{
    "text": "prea tare filmul!!!!de-abia ieri l-am vazut si mi-e placut foarte mult! merita vazut!:x",
    "label": "positive"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Urmează documentele și sentimentul acestora, care poate fi pozitiv, neutru sau negativ.
  ```

- Base prompt template:

  ```text
  Document: {text}
  Sentiment: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Document: {text}

  Clasificați sentimentul documentului. Răspundeți cu pozitiv, neutru sau negativ, și nimic altceva.
  ```

- Label mapping:
  - `positive` ➡️ `pozitiv`
  - `neutral` ➡️ `neutru`
  - `negative` ➡️ `negativ`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ro-sent
```

## Named Entity Recognition

### RoNEC

This dataset was published in [this paper](https://aclanthology.org/2020.lrec-1.546/).
The sentences have been extracted from a copy-right free newspaper,
covering several styles.

The original dataset consists of 9,000 / 1,330 / 2,000 samples for the
training, validation, and test splits, respectively. We use 1,024 / 256 / 2,048
samples for our training, validation and test splits, respectively. The training
and validation splits are subsets of the original splits, while the test split is
created using additional samples from the validation split.

Here are a few examples from the training split:

```json
{
    "tokens": ["În", "secolele", "al", "XVII", "-lea", "și", "al", "XVIII", "-lea", ",", "acestea", "erau", ":", "Conseil", "d'en", "haut", "(", "„", "Înaltul", "Consiliu", "”", ")", "-", "format", "din", "rege", ",", "prințul", "moștenitor", "(", "„", "le", "dauphin", "”", ")", ",", "cancelarul", ",", "controlorul", "general", "de", "finanțe", "și", "din", "secretarul", "de", "stat", "responsabil", "cu", "afacerile", "externe", "."],
    "labels": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-PER", "O", "B-PER", "O", "O", "O", "O", "O", "O", "O", "O", "B-PER", "O", "B-PER", "O", "O", "O", "O", "O", "B-PER", "O", "O", "O", "O", "O", "O", "O"]
}
```

```json
{
    "tokens": ["După", "ce", "am", "trecut", "de", "Obârșia-Cloșani", "(", "localitate", "renumită", "datorită", "Peșterii", "Cloșani", ",", "în", "interiorul", "căreia", ",", "în", "1961", ",", "s-", "a", "înființat", "prima", "Stațiune", "de", "cercetări", "speologice", "din", "România", ")", ",", "urcăm", "la", "Cumpăna", "Apelor", ",", "de", "unde", "coborâm", "brâul", "drumului", "în", "serpentine", "strâmte", ",", "până", "în", "Valea", "Cernei", "."],
    "labels": ["O", "O", "O", "O", "O", "B-LOC", "O", "O", "O", "O", "B-MISC", "I-MISC", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-LOC", "O", "O", "O", "O", "B-MISC", "I-MISC", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-MISC", "I-MISC", "O"]
}
```

```json
{
    "tokens": ["La", "data", "de", "26", "octombrie", "1994", ",", "și-", "a", "susținut", "teza", "de", "doctorat", "în", "limba", "franceză", ",", "cu", "denumirea", "de", "La", "de", "l'homme", "la", "du", "Dumitru", "Stăniloae", "(", ")", ".", "Cartea", "a", "fost", "publicată", "la", "Editura", "Trinitas", "din", "Iași", ",", "în", "2003", ",", "cu", "prilejul", "„", "Anului", "Stăniloae", "”", "(", "100", "ani", "de", "la", "naștere", "și", "10", "de", "la", "trecerea", "sa", "la", "cele", "veșnice", ")", "."],
    "labels": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-ORG", "I-ORG", "O", "B-LOC", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Mai jos sunt propoziții și dicționare JSON cu entitățile numite
  care apar în propoziția dată.
  ```

- Base prompt template:

  ```text
  Propoziție: {text}
  Entități numite: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Propoziție: {text}

  Identifică entitățile numite din propoziție. Ar trebui să le enumeri
  ca un dicționar JSON cu cheile {labels_str}. Valorile cheilor ar
  trebui să fie liste de entități numite de tipul respectiv, exact
  cum apar în propoziție.
  ```

- Label mapping:
  - `B-PER` ➡️ `persoană`
  - `I-PER` ➡️ `persoană`
  - `B-LOC` ➡️ `locație`
  - `I-LOC` ➡️ `locație`
  - `B-ORG` ➡️ `organizație`
  - `I-ORG` ➡️ `organizație`
  - `B-MISC` ➡️ `diverse`
  - `I-MISC` ➡️ `diverse`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ronec
```

## Linguistic Acceptability

### ScaLA-ro

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Romanian Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Romanian-RRT) by assuming that
the documents in the treebank are correct, and corrupting the samples to create
grammatically incorrect samples. The corruptions were done by either removing a word
from a sentence, or by swapping two neighbouring words in a sentence. To ensure that
this does indeed break the grammaticality of the sentence, a set of rules were used on
the part-of-speech tags of the words in the sentence.

The original full dataset consists of 1,024 / 256 / 2,048 samples for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
used as-is in the framework.

Here are a few examples from the training split:

```json
{
    "text": "Era o fantomă singuratică, rostind un adevăr pe care nimeni nu avea să -l audă vreodată.",
    "label": "correct"
}
```

```json
{
    "text": "Pe multe locuri avem apoi dovezi de o solicitudine deosebită, nu numai pentru paza pădurilor, dar și pentru nevoile locuitorilor săteni.",
    "label": "correct"
}
```

```json
{
    "text": "Dacă experiența nu ne- a reușit însă, este numai numai și din pricina timpului urât de afară.",
    "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Următoarele sunt fraze și dacă sunt gramatical corecte.
  ```

- Base prompt template:

  ```text
  Fraza: {text}
  Gramatical corect: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Fraza: {text}

  Stabiliți dacă fraza este gramatical corectă sau nu. Răspundeți cu 'da' dacă este corectă, și cu 'nu' dacă nu este corectă. Răspundeți doar cu acest cuvânt, și nimic altceva.
  ```

- Label mapping:
  - `correct` ➡️ `da`
  - `incorrect` ➡️ `nu`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-ro
```

## Reading Comprehension

### MultiWikiQA-ro

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
    "context": "Cornel Brahaș, pe numele real Ionel Vițu (n. 23 mai 1950, Poiana, Galați – d. 23 noiembrie 2005, Brăhășești, Galați), a fost scriitor și deputat român în legislatura 1992-1996, ales în municipiul București pe listele PUNR.\n\nBiografie\nA fost membru al Uniunii Scriitorilor din România. \n\nA scris nouă volume de poezie, trei romane-document, o carte de reportaj, două romane și un volum de reportaj-document. \n\nActivitate politică/Funcții: \n membru PUNR (1990-1994, exclus), apoi Partidul Dreapta Românească (din 1995) si PPR;\n deputat PUNR de București (27.09.1992-3.11.1996);\n vicepreședinte al PUNR (3.10.1992) și președinte al filialei București (1992-9.11.1994);\n purtător de cuvânt al PUNR (eliberat la 7.09.1994);\n secretar executiv al Partidului Dreapta Românească (1995-2000);\n vicepreședinte al PPR (3.02.2000)\n\nOpera\nPână la capăt și mai departe (roman-document)\n53 de poeme de dragoste și speranță\nPoezii foarte frumoase\nSfârșit de vânătoare\nPenultimele poeme de dragoste\nPoezii din capul meu\nÎntors\nDespre morți numai de bine (reportaj-document)\nClasa muncitoare - clasa deschisă (roman în probe)\nMocănești. Oamenii dracului\nMorții nu mai știu drumul către casă (roman, Ed. Militară 1990)\nAnno Domini - 2004 \nLaptus Vulgata\nJurnal dactilografiat (1985-1989)\nPoezii fără mijloace\n\nNașteri în 1950\nDecese în 2005\nDeputați români 1992-1996\nScriitori români din secolul al XX-lea\nPoliticieni români din secolul al XX-lea\nScriitori români din secolul al XXI-lea\nPoliticieni români din secolul al XXI-lea\nMembri ai Uniunii Scriitorilor din România\nRomâni cunoscuți sub pseudonimele folosite\nPoeți români din secolul al XX-lea\nPoeți români din secolul al XXI-lea\nMembri ai PUNR\nScriitori cunoscuți sub pseudonimele folosite",
    "question": "Cum se numește romanul pe care Cornel Brahaș l-a publicat în anul 1990 la Editura Militară?",
    "answers": {
        "answer_start": [1136],
        "text": ["Morții nu mai știu drumul către casă"]
    }
}
```

```json
{
    "context": "Gerardus Mercator () a fost un cartograf, geograf și matematician flamand de renume din Renaștere. Acest nume este latinizat, un obicei pe atunci foarte răspândit; numele său real în germană a fost Gerhard Kremer („Kremer” înseamnă „negustor”). S-a născut la 5 martie 1512 la Rupelmonde, Flandra, și a murit la 2 decembrie 1594 în Duisburg, Germania. A fost considerat un  \"Ptolemeu contemporan\".\n\nMercator se considera cercetător cosmograf care nu e nevoit să vândă hărți. De la el au rămas doar 5 hărți, păstrate în Muzeul de istorie din Duisburg. În anul 1562 realizează prima hartă a Europei, care este una din hărțile atlasului său. Numele și l-a schimbat în perioada când era la Universitatea Essen-Duisburg.\n\nRealizări \n\n 1530 devine \"Magister\" la \"Universitatea catolică\" din Leuven\n 1537 însărcinează pe meșteșugarul Gaspard van der Heyden să-i confecționeze globul terestru, și bolta cerului\n 1537 Harta \"Pământului sfânt\"\n 1538 o hartă mică de proiecție în formă de inimă a lumii, și o hartă de perete a Flandrei\n 1540 publică cartea  Literarum latinarum, quas italicas, cursoriasque vocant, scribendarum ratio, (pe lemn)\n 1541 își continuă cercetările de proiecție a globului pe o hartă (plan), are probleme  cu biserica catolică (acuzat de erezie)\n 1551 realizează un nou glob pământesc și unul al boltei cerești\n 1552 urmărit de inchiziție se refugiază cu toată familia la Duisburg, principatul Jülich-Kleve-Berg, prințul  Wilhelm der Reiche fiind sub influența humanistului Erasmus von Rotterdam\n 1554 Realizarea lui cea mai valoroasă este \"Proiecția Mercator\", o proiecție a globului terestru pe un plan (hartă). Această proiecție redă fidel unghiurile, fiind prin aceasta de importanță majoră pentru navigația pe Pământ.\n 1559 - 1562 predă matematică și cosmologie la Gimnaziul din Duisburg\n 1563 este numit de  Wilhelm der Reiche cartograf princiar\n 1562 Sub îndrumările lui Johannes Corputius, întocmește o hartă exactă a Duisburgului\n 1594 moare ca un om respectat și bogat, fiind îngropat în cimitirul bisericii \"Salvator\" din Duisburg.\n\nNote\n\nBibliografie\n\nLegături\xa0externe\n\n Cartographic images of maps and globes \nMercator\'s maps at the Eran Laor Cartographic Collection, the National Library of Israel\n\nNașteri în 1512\nDecese în 1594\nExploratori belgieni\nCartografi flamanzi\nPerioada Marilor descoperiri\nIstoria navigației\nEponime ale craterelor de pe Lună\nEponime ale asteroizilor",
    "question": "În ce an s-a născut Gerardus Mercator?",
    "answers": {
        "answer_start": [259],
        "text": ["5 martie 1512"]
    }
}
```

```json
{
    "context": "Un cod de aeroport ICAO sau un identificator de locație ICAO este un cod alfanumeric, format din patru litere, care desemnează fiecare din aeroporturile din lume.  Aceste coduri au fost definite de International Civil Aviation Organization și au fost publicate în documentul ICAO 7910: Location Indicators ().\n\nCodurile ICAO sunt folosite în controlul traficului aerian și în operările liniilor aeriene cum ar fi planificarea zborurilor.  Ele nu sunt același lucru cu codurile IATA, întâlnite de publicul obișnuit și folosite de către companiile aeriene în orarele zborurilor, rezervări și operațiile legate de bagaje.  Codurile ICAO sunt folosite de asemenea pentru identificarea altor locații precum stații meteo, stații internaționale de servicii ale zborurilor sau centre de control al zonelor, fie că acestea sunt amplasate sau nu în aeroporturi.\n\nSpre deosebire de codurile IATA, codurile ICAO au o structură regională la bază, astfel încât ele nu sunt duplicate ci identifica un singur aeroport.  În general, prima literă alocată după continent și reprezintă o țară sau un grup de țări de pe acel continent.  A doua literă în general reprezintă o țară din acea regiune, iar celelate două litere rămase sunt folosite la identificarea fiecărui aeroport.  Excepțiile de la această regulă sunt țările foarte întinse care au coduri de țară formate dintr-o singură literă, iar celelalte trei litere care rămân desemnează aeroportul.\n\nÎn zona întinsă formată de Statele Unite și Canada, celor mai multor aeroporturi li se asociază codurile de trei litere IATA, care sunt aceleași cu codurile lor ICAO, însă fără litera K sau C de la început, d.e., YYC și CYYC (Calgary International Airport, Calgary, Alberta), IAD și KIAD (Dulles International Airport, Chantilly, Virginia).  Aceste coduri nu trebuie confundate cu semnalele de apel pentru radio sau pentru televiziune, chiar dacă ambele țări folosesc semnale de apel de formate din patru litere care încep cu aceste litere.\n\nTotuși, fiindcă Alaska, Hawaii și alte teritorii din Statele Unite au propriile prefixe ICAO formate din două litere, situația pentru ele este similară altor țări mici, iar codurile ICAO ale aeroporturilor lor sunt în general diferite de identificatoarele FAA/IATA formate din trei litere.  De exemplu, Hilo International Airport (PHTO comparativ cu ITO) și Juneau International Airport (PAJN comparativ cu JNU).\n\nZZZZ este un cod special care se folosește atunci când nu există nici un cod ICAO pentru aeroport, și este folosit de obicei în planurile de zbor.\n\nAeroportul Internațional Henri Coandă din Otopeni are codul LROP, iar Aeroportul Internațional Aurel Vlaicu de la Băneasa are codul LRBS  .\n\nPrefixuri\n\nVezi și\nListă de aeroporturi după codul ICAO\nListă de aeroporturi după codul IATA\nAeroport\n\nLegături externe\nInternational Civil Aviation Organization (official site)\nICAO On-line Publications Purchasing  (official site)\nICAO 7910 - Location Indicators (online version provided by EUROCONTROL)\nCatalogue of ICAO Airfields \nICAO airport code prefixes \n\nCoduri\nAeroporturi",
    "question": "Care este codul ICAO pentru un aeroport?",
    "answers": {
        "answer_start": [66],
        "text": ["un cod alfanumeric, format din patru litere, care desemnează fiecare din aeroporturile din lume"]
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Iată texte cu întrebări și răspunsuri însoțite.
  ```

- Base prompt template:

  ```text
  Text: {text}
  Întrebare: {question}
  Răspuns de maxim 3 cuvinte:
  ```

- Instruction-tuned prompt template:

  ```text
  Text: {text}

  Răspunde la următoarea întrebare referitoare la textul de mai sus folosind maxim 3 cuvinte.

  Întrebare: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-ro
```

## Knowledge

### MMLU-hu

This dataset is a machine translated version of the English [MMLU
dataset](https://openreview.net/forum?id=d7KBjmI3GmQ) and features questions within 57
different topics, such as elementary mathematics, US history and law. The translation to
Hungarian was done by the University of Oregon as part of [this
paper](https://aclanthology.org/2023.emnlp-demo.28/), using GPT-3.5-turbo.

The original full dataset consists of 278 / 1,408 / 13,024 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 2,048 split for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
new and there can thus be some overlap between the original validation and test sets and
our validation and test sets.

Here are a few examples from the training split:

```json
{
    "text": "Ha a College Board az egyik évben elhanyagolta volna az agykutatással kapcsolatos kérdések feltételét az AP pszichológiai vizsgán, a teszt hiányozni foghat.\nVálaszlehetőségek:\na. konstruktum validitást.\nb. prediktív validitást.\nc. egyidejű validitást.\nd. tartalmi validitást.",
    "label": "d"
}
```

```json
{
    "text": "Ha $\\log_{b}343=-\\frac{3}{2}$, mennyi az $b$ értéke?\nVálaszlehetőségek:\na. 3\nb. \\frac{1}{49}\nc. \\frac{1}{7}\nd. 7",
    "label": "b"
}
```

```json
{
    "text": "Egy gyalog, akinek lakhelye az A államban van, az B államban keresztezte az utat, amikor egy külföldi állampolgár által vezetett autó elgázolta. Mindkét fél sérüléseket szenvedett. A gyalog $100,000 kártérítési összeget kérő kártérítési pert indított a vezetővel szemben az B állam szövetségi kerületi bíróságában. A vezető úgy véli, hogy a gyalog illegálisan keresztezte az utat, és ezért ő a felelős az ütközésért. Az ügyvéd tanácsadást kér a vezetőtől arra vonatkozóan, hogy hogyan kell a legjobban reagálni a keresetre. Feltételezzük, hogy B állam egy olyan hozzájáruló hanyagság állam, amely szerint mindkét fél részben felelős az esetért. Hogyan tanácsolja az ügyvéd a vezetőnek, hogy reagáljon erre?\nVálaszlehetőségek:\na. Válaszként adjon be egy beadványt, amelyben az hozzájáruló hanyagság pozitív védelmét és a gondatlanság elleni ellenkérelmet emeli, a vezető sérüléseinek kártérítési összegét kérve.\nb. Válaszként adjon be egy beadványt, amelyben az hozzájáruló hanyagság pozitív védelmét és az anyagi bizonyíték alapján történő ítélet kérelmével védekezik.\nc. Kérje az ügy elutasítását a személyi hatáskör hiánya miatt, mert az autó vezetője nem B állam állampolgára.\nd. Kérje az ügy elutasítását az ügy tárgyi hatáskörének hiánya miatt, mert az autó vezetője nem amerikai állampolgár.",
    "label": "a"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Az alábbiakban több választási lehetőséget tartalmazó kérdések találhatók (válaszokkal együtt).
  ```

- Base prompt template:

  ```text
  Kérdés: {text}
  Válaszlehetőségek:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Válasz: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Kérdés: {text}

  Válaszoljon a fenti kérdésre az elérhető lehetőségek közül 'a', 'b', 'c' vagy 'd' használatával, és semmi mással.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset mmlu-hu
```

## Common-sense Reasoning

### Winogrande-hu

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2506.19468)
and is a translated and filtered version of the English [Winogrande
dataset](https://doi.org/10.1145/3474381).

The original full dataset consists of 47 / 1,210 samples for training and testing, and
we use 128 of the test samples for validation, resulting in a 47 / 128 / 1,085 split for
training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
    "text": "Nem tudtam irányítani a nedvességet úgy, mint az esőt, mert a _ mindenhol bejött. Mire utal a hiányzó _?\nVálaszlehetőségek:\na. nedvesség\nb. eső",
    "label": "a"
}
```

```json
{
    "text": "Jessica úgy gondolta, hogy a Sandstorm a valaha írt legjobb dal, de Patricia utálta. _ jegyet vett a jazz koncertre. Mire utal a hiányzó _?\nVálaszlehetőségek:\na. Jessica\nb. Patricia",
    "label": "b"
}
```

```json
{
    "text": "A termosztát azt mutatta, hogy lent húsz fokkal hűvösebb volt, mint fent, így Byron a _ maradt, mert fázott. Mire utal a hiányzó _?\nVálaszlehetőségek:\na. lent\nb. fent",
    "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Az alábbiakban több választási lehetőséget tartalmazó kérdések találhatók (válaszokkal együtt).
  ```

- Base prompt template:

  ```text
  Kérdés: {text}
  Lehetőségek:
  a. {option_a}
  b. {option_b}
  Válasz: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Kérdés: {text}
  Lehetőségek:
  a. {option_a}
  b. {option_b}

  Válaszoljon a fenti kérdésre az elérhető lehetőségek közül 'a' vagy 'b' használatával, és semmi mással.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset winogrande-hu
```

## Summarisation

### HunSum

[The dataset](https://huggingface.co/datasets/ariel-ml/hun-sum-chatml-5k) consists of samples
from Hungarian news articles, with the summaries given by the lead paragraphs.

The original full dataset consists of 5,000 / 200 / 200 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 2,048 split for training,
validation and testing, respectively.

Here are a few examples from the training split:

```json
{
    "text": "Másfél éven belül rend lehet Szíriában\n\nA szíriai kormány és az ellenzéki csoportok képviselői még idén tárgyalásokat kezdenének, fél éven belül átmeneti kormány alakulna, másfél éven belül pedig választásokat tartanának a tervek szerint – közölte Frank-Walter Steinmeier német külügyminiszter.\n\nJohn Kerry amerikai külügyminiszter szerint ahhoz, hogy mindezt elérjék, tűzszünetet kell hirdetni a kormány és a lázadó csoportok között. Az ENSZ Biztonsági Tanácsának öt állandó tagja megegyezett, hogy határozatot fogad el erről. Kerry szerint a legfontosabb, hogy ne a mérsékelt ellenzékkel szembeni harc, hanem az Iszlám Állam (IÁ) és az an-Nuszra Front ellen küzdelem folytatódjon.\n\nAz amerikai külügyminiszter elmondta: az Egyesült Államok és Oroszország között véleménykülönbségek vannak, azonban folytatni kell a közös munkát, ahogy ezt az Iránnal folytatott tárgyalások kapcsán tették korábban, és hozzátette: a tárgyalópartnerek mindannyian Szíria stabilitását tartják szem előtt.\n\nSzergej Lavrov orosz külügyminiszter a sajtótájékoztatón kijelentette: csak a szíriai emberek dönthetnek országuk és elnökük sorsáról. Lavrov szerint a valódi ellenség azonban nem Aszad, hanem az IÁ. Elmondta azt is, hogy a tárgyaláson részt vevő országok számba vették a terrorcsoportokat, ezen listák összehangolását Jordánia végzi majd, és az ENSZ Biztonsági Tanácsa szavazni fog róla.\n\nA békefolyamatot Staffan de Mistura, az ENSZ szíriai különmegbízottja vezeti és szervezi majd – mondta Frank-Walter Steinmeier a 17 ország magas rangú képviselőinek részvételével zajló tanácskozás után.",
    "target_text": "Swaney Elizabeth trükkök nélkül mutatta be a gyakorlatait, pedig ennek a sportágnak pont az lenne a lényege."
}
```

```json
{
    "text": "Hoffmann Rózsa a CEU-ról: eddig is jártak magyar fiatalok bécsi egyetemekre\n\nAz ATV Egyenes beszéd című műsorának vendége volt hétfő este Hoffmann Rózsa. A volt köznevelésért felelős államtitkárt a CEU-ról is kérdezték, ezzel kapcsolatban a politikus azt mondta, szerinte nem a kormány űzte el az egyetemet, hanem az intézmény döntött úgy, hogy az amerikai diplomát adó képzéseiket kiviszik az országból.\n\nAmikor a műsorvezető megkérdezte Hoffmanntól, hogy jól van-e ez így, Hoffmann azt válaszolta:\n\n  Nem tudom, hogy jól van, vagy nincs jól, de Bécs nincs a világ végén.\n\nA politikus hozzátette, nincs ebben semmi különös, hiszen eddig is jártak magyar fiatalok bécsi egyetemekre, ingázni is sokan ingáztak eddig. Hoffmann azt mondta, "emberileg" megérti a CEU vezetőségének elkeseredését, de szerinte ez egy túlpolitizált ügy.\n\nHétfőn eldőlt, hogy a CEU Bécsbe költözteti el amerikai diplomát adó képzéseit, miután az elmúlt 20 hónapban mindent megtettek azért, hogy megfeleljenek a törvényeknek, a magyar hatóságok viszont annak ellenére sem írták alá a működéshez szükséges államközi megállapodást, hogy a CEU az amerikai hatóságok által jóváhagyott felsőoktatási képzést indított az Egyesült Államokban.\n\nAz egyetem ugyanakkor közleménye szerint megőrzi magyar egyetemi akkreditációját, és arra törekszik, hogy a jövőben is folytasson tanítási és kutatási tevékenységet Budapesten.",
    "target_text": "A volt köznevelési államtitkár \"emberileg\" megérti az egyetem vezetőinek elkeseredettségét."
}
```

```json
{
    "text": "Pörög a turizmus Budapesten: elképesztően erős volt az október\n\nUgyanakkor kérdésesnek nevezik, hogy a kiugró növekedés tartósnak bizonyul-e november-decemberben is, és ami talán még ennél is fontosabb: a küszöbön álló - immár 2020. január 31-i határidővel élesített - Brexit, és annak gazdasági következményei milyen hatást idéznek elő a következő hónapok, évek budapesti vendégforgalmában és a kiutazási trendekben.\nA fővárosi kereskedelmi szálláshelyek árbevétele megközelítette a 25 milliárd forintot. Hosszú idő óta először nem csupán a szállásdíj-bevételek emelkedtek számottevően, hanem a vendégforgalom is - jegyezték meg.\nBudapesten a vendégérkezések 5,5 százalékkal nőttek a vendégéjszakák pedig 6,3 százalékkal.\nAz elemzés szerint ezen belül a húzóerő a külföldi vendégforgalom volt: októberben 372 068 vendég érkezett és 862 427 vendégéjszakát töltött el, előbbi 8,3 százalékos, utóbbi 9,6 százalékos növekedést mutat az előző év tizedik hónapjával összehasonlítva. Mindeközben a belföldről érkező vendégforgalom tovább csökkent.\nA Széchenyi Pihenő Kártya költési értéke októberben 69,4 millió forintot ért el Budapest kereskedelmi szálláshelyein, ez az első 10 havi - budapesti - SZÉP Kártya-bevétel 10 százaléka. A január óta Budapesten keletkezett, nagyságrendileg 700 millió forintos SZÉP Kártya-árbevétel 55 százalékos növekedés a tavalyi év azonos időszakában elért 450 millió forint közeli árbevételhez képest.\nA küldőországok között például kiemelték, hogy impozáns növekedési ütemet mutat a francia, az izraeli, az orosz és a brit küldőpiac, az utóbbi hónapokban pedig felzárkózott a TOP10-be Lengyelország is.", "target_text": "Az idei október volt a 2019-es év legdinamikusabban növekvő hónapja a vendégérkezéseket és a vendégéjszakákat tekintve Budapesten - hívta fel a figyelmet a Budapesti Fesztivál- és Turisztikai Központ (BFTK) elemzésében."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:

  ```text
  Az alábbi szövegek tartalmazzák az eredeti cikkeket és azok összefoglalóit.
  ```

- Base prompt template:

  ```text
  Szöveg: {text}
  Összefoglaló: {target_text}
  ```

- Instruction-tuned prompt template:

  ```text
  Szöveg: {text}

  Adjon egy rövid összefoglalót a fenti szövegről.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset hunsum
```
