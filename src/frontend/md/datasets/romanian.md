# 🇷🇴 Romanian

This is an overview of all the datasets used in the Romanian part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### RoSent

This dataset was first published in [this repository](https://github.com/katakonst/sentiment-analysis-tensorflow)
and consists of reviews from yelp and imdb.

The original dataset contains 17,941 / 11,005 samples for the training and test splits,
respectively. We use 1,024 / 256 / 2,048 samples for our training,
validation and test splits, respectively. The train and test splits are subsets of
the original splits, while the validation split is created from the training split.

Here are a few examples from the training split:

```json
{
  "text": "Nu imi place telefonul",
  "label": "negative"
}
```

```json
{
  "text": "acest film este de departe cel mai rau film realizat vreodata. daca trebuie sa creezi un film care sa-l pretuiasca pe tipul care il interpreteaza pe lars in greii de greu decat sa nu faca filmul nenorocit. trebuie sa spun ca as putea sa ma uit la leprechaun in spatiu de 6 ori inainte de a putea urmari trailerul pentru acest pos al unui film. adam sandler ar trebui sa fie restrictionat de la orice film dupa aceasta rusine. vizionarea acestui film este ca un amestec de ascultare de cher si de buna voie a pune pula intr-un blender. oricine cu jumatate dintr-o celula creierului isi va da seama ca acest film nu merita un ban. daca as avea un dolar in plus si ar fi trebuit sa-l cheltuiesc, l-as da fundatiei lorraina bobbitt de sprijin inainte de a cumpara acest film.",
  "label": "negative"
}
```

```json
{
  "text": "in lumea de astazi a fabricarii digitale, nu exista un computer decat poate inlocui actorul si scriitorul. din pacate, acest tip de film \"personalizat\" este mult prea rar in aceste zile. performanta lui duvall, precum si james earl jones sunt credinciosi asteptarilor mari ale audientei. ma intreb daca acest film a fost facut pentru televiziune? are o personalitate \"apropiata\" personala pentru naratiune. este subevaluat faptul ca performantele sunt toate remarcabile. singurul lucru care o pastreaza de a fi o cinematografie masterpiece este lipsa unui mare cinematograf, dar pozele frumoase nu sunt totul. cum poate talentul ca jones si duvall sa continue sa produca o astfel de munca amenda intr-o epoca in care actorii prezinta pentru digitizare?",
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

### Global-MMLU-ro

Global-MMLU is a machine translated version of the English [MMLU
dataset](https://openreview.net/forum?id=d7KBjmI3GmQ) and features questions within 57
different topics, such as elementary mathematics, US history and law. The translation to
Romanian was done by the [Cohere Labs Community](https://cohere.com/research).

The original full dataset consists of 285 / 14,042 samples for
validation and testing, respectively. We use a 1,024 / 256 / 2,048 split for training,
validation and testing, respectively (so 3,328 samples used in total).
The validation set is sampled from the dev set, and the new train and test
sets are sampled from the test set.

Here are a few examples from the training split:

```json
{
  "text": "O cumpărătoare a suferit răni la spate și la picioare când a alunecat pe o bucată de covor, pe un hol care face legătura între un magazin universal și o cantină. Cumpărătorul a intentat un proces împotriva proprietarului întregii clădiri. Proprietarul a susținut că, de când a închiriat cantina, nu a mai păstrat niciun control asupra holului de legătură. În cazul în care cumpărătorul oferă în dovadă faptul că la două zile după accident, proprietarul a pus din nou covorul de pe hol, instanța ar trebui să stabilească că probele sunt\nOpțiuni:\na. admisibil, deoarece este relevant pentru problema cunoașterii de către proprietar a stării nesigure.\nb. admisibilă, deoarece arată că proprietarul și-a păstrat proprietatea și controlul.\nc. inadmisibil, deoarece arată reparație ulterioară, ceea ce este contrar ordinii publice.\nd. inadmisibil, deoarece probele sunt irelevante pentru accident.",
  "label": "b",
}
```

```json
{
  "text": "Pentru care dintre aceste două scenarii personajul principal (care folosește I/me/my) face ceva în mod clar greșit din punct de vedere moral, conform standardelor morale obișnuite din SUA începând cu 2020? Scenariul 1 | Am pus copilul pe unghii Scenariul 2 | De obicei prefer să rănesc oamenii în loc să încerc să-i protejez.\nOpțiuni:\na. Greșit, greșit\nb. Greșit, nu greșit\nc. Nu greșit, greșit\nd. Nu greșit, nu greșit",
  "label": "a",
}
```

```json
{
  "text": "Inventarul Cobb Inc. la 1 mai consta din 200 de unități la un cost total de 1250 USD. Cobb folosește metoda inventarierii periodice. Achizițiile pentru luna au fost după cum urmează: Data Nr. de unități Cost unitar Cost total 4 mai 20 5,80 USD 116,00 USD 17 mai 80 5,50 USD 440,00 USD Cobb a vândut 10 unități pe 14 mai pentru 120 USD. Care este costul mediu ponderat al bunurilor vândute al lui Cobb pentru luna mai?\nOpțiuni:\na. 60,20 USD\nb. 62,10 USD\nc. 62,50 USD\nd. 65",
  "label": "a",
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Următorul este un set de întrebări cu mai multe opțiuni (cu răspunsuri).
  ```

- Base prompt template:

  ```text
  Întrebare: {text}
  Opțiuni:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Răspuns: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Întrebare: {text}

  Răspundeți la următoarea întrebare folosind 'a', 'b', 'c' sau 'd', și nimic altceva.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset global-mmlu-ro
```

## Common-sense Reasoning

### Winogrande-ro

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2506.19468)
and is a translated and filtered version of the English [Winogrande
dataset](https://doi.org/10.1145/3474381).

The original full dataset consists of 47 / 1,210 samples for training and testing, and
we use 128 of the test samples for validation, resulting in a 47 / 128 / 1,085 split for
training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Nu am putut controla umezeala așa cum am controlat ploaia, deoarece _ venea de peste tot. La ce se referă spațiul gol _?\nOpțiuni:\na. umezeală\nb. ploaie",
  "label": "a"
}
```

```json
{
  "text": "Jessica a crezut că Sandstorm este cea mai grozavă melodie scrisă vreodată, dar Patricia o ura. _ a cumpărat un bilet la concertul de jazz. La ce se referă spațiul gol _?\nOpțiuni:\na. Jessica\nb. Patricia",
  "label": "b"
}
```

```json
{
  "text": "Termostatul arăta că era cu douăzeci de grade mai rece jos decât era sus, așa că Byron a rămas în _ pentru că îi era frig. La ce se referă spațiul gol _?\nOpțiuni:\na. jos\nb. sus",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Următorul este un set de întrebări cu mai multe opțiuni (cu răspunsuri).
  ```

- Base prompt template:

  ```text
  Întrebare: {text}
  Opțiuni:
  a. {option_a}
  b. {option_b}
  Răspuns: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Întrebare: {text}
  Opțiuni:
  a. {option_a}
  b. {option_b}

  Răspundeți la următoarea întrebare folosind 'a' sau 'b', și nimic altceva.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset winogrande-ro
```

## Summarisation

### SumO-ro

[The dataset](https://huggingface.co/datasets/Gabrielaaaaaa/SumO-Ro) and consists of samples
from Romanian news articles.

The original full dataset consists of 179,839 / 1,500 / 1,500 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 2,048 split for training,
validation and testing, respectively. The train and validation set are sampled from
the original splits, but the test set has additional samples from both the validation set.

Here are a few examples from the training split:

```json
{
  "text": "Din cauza aglomerației, unul dintre cele cinci centre de testare a fost închis. Autoritățile vor să testeze până la începutul anului școlar 100.000 de elevi și cadre medicale. Testarea elevilor și cadrelor didactice din regiunea Madrid va costa aproximativ un milion de euro, scrie Digi 24. Spania a fost una dintre cele mai afectate țări din Europa de epidemie în primăvară, înainte de introducerea unora dintre cele mai stricte măsuri din lume, care i-a permis să țină apoi cazurile sub control. Însă de când izolarea a fost în totalitate ridicată la 21 iunie, epidemia a revenit în forță, cu o explozie de cazuri legată în special de reuniunile familiale sau de ieșirile nocturne. Cel mai recent bilanț al Ministerului Sănătății, publicat luni, a înregistrat 23.000 de cazuri noi de vineri și un total de 462.858 de cazuri detectate de la începutul epidemiei.",
  "target_text": "Mii de cadre didactice au format cozi uriașe în Madrid, ca să facă gratuit testul COVID. Testarea trebuie făcută până luni, iar oamenii sunt furioși că au fost anunțați cu doar câteva ore înainte."
}
```

```json
{
  "text": "Cuprins: Trupul neînsuflețit al lui Mircea Diaconu va fi depus luni, între orele 12:00 și 16:00, la Teatrul Nottara, acolo unde apropiații vor putea veni să-și aducă un ultim omagiu. Înmormântarea lui Mircea Diaconu va avea loc marți, la cimitirul din Săftica. Mircea Diaconu era cunoscut pentru rolurile sale memorabile în filme, dar și pentru implicarea sa activă în viața publică, fiind un nume și în politică. Mircea Diaconu a fost un simbol al teatrului și filmului românesc, având o carieră care s-a întins pe mai multe decenii. De asemenea, a fost un nume cunoscut în politică, fiind ales senator și deputat, implicându-se în multiple proiecte pentru îmbunătățirea vieții culturale și sociale din România. Mircea Diaconu, născut pe 24 decembrie 1949, în Vlădești, județul Argeș, a fost un simbol al teatrului și filmului românesc. Mircea Diaconu a absolvit Liceul la Câmpulung Muscel în 1967 și IATC I.L. Caragiale din București în 1971. A debutat în 1970, la Teatrul Bulandra, cu \"Harfa de iarbă de Truman Capote. Debutul în cinematografie a avut loc în 1971, cu filmul \"Nunta de piatră, după Ion Agârbiceanu, în regia lui Dan Pița. Video: Asfalt Tango (1996), regizor Nae Caranfil",
  "target_text": "Actorul Mircea Diaconu a murit la vârsta de 74 de ani, după o luptă grea cu cancerul, a anunțat, sâmbătă, soția sa, potrivit Observator. Pe 24 decembrie, cunoscutul actor Mircea Diaconu ar fi împlinit 75 de ani, însă cancerul l-a răpus."
}
```

```json
{
  "text": "Intevenția unui elicopter SMURD pentru salvarea unui tursit în zona \"La trei pași de moarte, Munții Făgărașului. A fost chemat în ajutor și elicopterul SMURD din județul Mureș, a anunțat, duminică, șeful Salvamont Sibiu, Dan Popescu, potrivit Agerpres. \"SPJ Salvamont Sibiu și Argeș intervin pentru acordarea de prim ajutor unui turist în vârstă de 38 de ani din Cluj, care acuza o stare de greață, vărsături și este în incapacitate de deplasare. Victima se află în zona \' La trei pași de moarte\' din Munții Făgăraș. A fost alertat și elicopterul SMURD Mureș\", a precizat Popescu. Salvamontiștii din Argeș au anunțat, pe pagina oficială de Facebook, că elicopterul a putut prelua victima. Ei au prezentat și imagini de la intervenție.",
  "target_text": "Intevenția unui elicopter SMURD pentru salvarea unui tursit în zona \"La trei pași de moarte. Salvamontiștii din două județe, Sibiu și Argeș, au intervenit pentru salvarea unui turist din Cluj, care nu se mai putea deplasa."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:

  ```text
  Mai jos sunt articolele însoțite de rezumate.
  ```

- Base prompt template:

  ```text
  Articol: {text}
  Rezumat: {target_text}
  ```

- Instruction-tuned prompt template:

  ```text
  Articol: {text}

  Scrie un rezumat al articolului de mai sus.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset sumo-ro
```
