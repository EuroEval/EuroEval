# 🇭🇺 Hungarian

This is an overview of all the datasets used in the Hungarian part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### HuSST

This dataset was published in [this paper](https://acta.bibl.u-szeged.hu/75891/1/msznykonf_018_431-446.pdf)
and is the Hungarian version of the Stanford Sentiment Treebank.

The original dataset contains 9,328 / 1,165 / 1,165 samples for the training, validation,
and test splits, respectively. We use 1,024 / 256 / 2,048 samples for our training,
validation and test splits, respectively. The train and validation splits are subsets of
the original splits. The orignial test split does not contain any labels, so our test split
is created from the training split.

Here are a few examples from the training split:

```json
{
    "text": "Egy varázslatos film, amely egy merész utazást kínál a múltba, és forró ölelésébe zárja a szentpétervári Ermitázs Múzeumban található kulturális ereklyéket.",
    "label": "positive"
}
```

```json
{
    "text": "Az elmúlt időszakban jellemző volt a többszereplős romantikus filmek lánca... de Petter Mattei Szerelem a pénz idején című műve különválik azáltal, hogy olyan kapcsolati láncolatot hoz létre, ami teljes körré áll össze, hogy pozitív “még ha tragikus is” véget kanyarítson a történetnek.",
    "label": "positive"
}
```

```json
{
    "text": "A \"Fehér Olajfű\" film olyan, mintha a forrásanyag a Reader's Digest tömörített változata lenne.",
    "label": "negative"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Az alábbiak dokumentumok és érzelmük, ami lehet pozitív, semleges vagy negatív.
  ```

- Base prompt template:

  ```text
  Dokumentum: {text}
  Érzelem: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Dokumentum: {text}

  Osztályozza az érzelmet a dokumentumban. Válaszoljon pozitív, semleges, vagy negatív kifejezéssel, és semmi mással.
  ```

- Label mapping:
  - `positive` ➡️ `pozitív`
  - `neutral` ➡️ `semleges`
  - `negative` ➡️ `negatív`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset husst
```

## Named Entity Recognition

### SzegedNER

This dataset was published in [this paper](https://aclanthology.org/L06-1215/).
The data is a segment of the Szeged Corpus, consisting of short business news
articles collected from MTI (Hungarian News Agency, <www.mti.hu>).

The original dataset consists of 8,220 / 874 / 1,656 samples for the
training, validation, and test splits, respectively. We use 1,024 / 256 / 2,048
samples for our training, validation and test splits, respectively. All the new
splits are subsets of the original splits.

Here are a few examples from the training split:

```json
{
    "tokens": ["Ráadásul", "kirúgták", "a", "brüsszeli", "bizottságtól", "azt", "az", "alkalmazottat", ",", "aki", "egy", "csokor", "gyanús", "tényrõl", "szóló", "információkat", "juttatott", "el", "az", "Európai", "Parlament", "(", "EP", ")", "néhány", "képviselõjének", "."],
    "labels": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-ORG", "I-ORG", "O", "B-ORG", "O", "O", "O", "O"]
}
```

```json
{
    "tokens": ["A", "londoni", "Európai", "Újjáépítési", "és", "Fejlesztési", "Bank", "(", "EBRD", ")", "10,1", "millió", "euróért", "részvényeket", "vesz", "a", "szlovák", "Polnobankából", "az", "olasz", "UniCredito", "pénzintézettől", "."],
    "labels": ["O", "O", "B-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "O", "B-ORG", "O", "O", "O", "O", "O", "O", "O", "O", "B-ORG", "O", "O", "B-ORG", "O", "O"]
}
```

```json
{
    "tokens": ["Clinton", "a", "Netanjahuval", "tartott", "vasárnapi", "találkozó", "utáni", "sajtókonferencián", "sürgette", "a", "palesztinokat", "kötelezettségeik", "betartására", ",", "de", "egyúttal", "felszólította", "Izraelt", ",", "hogy", "ne", "függessze", "fel", "az", "októberi", "megállapodás", "végrehajtását", "."],
    "labels": ["B-PER", "O", "B-PER", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-LOC", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Az alábbiakban mondatok és JSON szótárak találhatók
  az adott mondatokban előforduló névjegyzékkel.
  ```

- Base prompt template:

  ```text
  Mondat: {text}
  Névjegyzék: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Mondat: {text}

  Nevezze meg a mondatban szereplő neveket. JSON szótárként adja meg a 'személy', 'helyszín', 'szervezet' és 'egyéb' kulcsszavakat. Az értékek a mondatban szereplő névjegyzékek listái legyenek, pontosan úgy, ahogyan megjelennek.
  ```

- Label mapping:
  - `B-PER` ➡️ `személy`
  - `I-PER` ➡️ `személy`
  - `B-LOC` ➡️ `helyszín`
  - `I-LOC` ➡️ `helyszín`
  - `B-ORG` ➡️ `szervezet`
  - `I-ORG` ➡️ `szervezet`
  - `B-MISC` ➡️ `egyéb`
  - `I-MISC` ➡️ `egyéb`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset szeged-ner
```

## Linguistic Acceptability

### ScaLA-hu

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Hungarian Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Hungarian-Szeged) by assuming that
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
    "text": "A kiskereskedelemben teljesen más okra vezethető vissza a mamutvállalkozások létrejötte, mint az élelmiszeriparban.",
    "label": "correct"
}
```

```json
{
    "text": "Még egy jövő évi költségvetési mérleggel sem tisztelte meg a kormány a képviselőházat, az államháztartási mérlegből kellene azt a képviselőknek kibogarászniuk.",
    "label": "correct"
}
```

```json
{
    "text": "A Nawa Bányászati Kft. ahhoz Nawa a cégcsoporthoz tartozott, amely a taxisblokád idején jelentette be, hogy az akkor hordónként 29 dolláros világpiaci árnál olcsóbban, 22-23 dollárért tud olajat szerezni.",
    "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  A következő mondatok, és hogy helyesek-e nyelvtanilag.
  ```

- Base prompt template:

  ```text
  Mondat: {text}
  Nyelvtanilag helyes: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Mondat: {text}

  Határozza meg, hogy a mondat nyelvtanilag helyes-e vagy sem. Csak 'igen'-nel válaszoljon, ha helyes, és 'nem'-mel, ha nem helyes. Csak ezzel a szóval válaszoljon, és semmi mással.
  ```

- Label mapping:
  - `correct` ➡️ `да`
  - `incorrect` ➡️ `не`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-hu
```

## Reading Comprehension

### MultiWikiQA-hu

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
    "context": "Az utolsó mester (The Last of the Masters) Philip K. Dick egyik novellája, amelyet 1953-ban írt, majd 1954-ben az Orbit Science Fiction magazin november-decemberi számában jelent meg. Magyarul a Lenn a sivár Földön című novelláskötetben olvasható.\n\nTörténet \n\nA világon kétszáz éve az anarchia uralkodik. Akkor történt, hogy először Európában, majd szerte a világban fellázadtak a polgárok, és megdöntötték a kormányokat. Megölték a vezetőket, elpusztították a robotokat és megsemmisítettek minden addig a kormány kezében lévő kutatási anyagot, elpusztították az atombombákat. A világon most egyetlenegy szervezet van, az Anarchista Szövetség, aki csak arra ügyel, hogy nehogy valaki újra felépítsen magának egy rendszert. A robotok közül viszont az egyik – Bors – túlélte a pusztítást, és bujdosva a kétszáz év alatt felépített magának egy kis eldugott birodalmat. Ennek a birodalomnak vannak a legmodernebb eszközei (hiszen a két évszázaddal ezelőtti kutatási eredmények már csak Bors agyában maradtak meg), földcsuszamlásoknak álcázva elzárták a telephez vezető földutakat, és a szomszédos falukban elhelyezett kémeknek köszönhetően mindig időben tudták, ha a Szövetség ügynökei közelednek, így mindig időben félresöpörték őket. Nem sikerül azonban ezt megtenni Edward Tolbyval és lányával, Silviával. Így (bár Silviát sikerül elkapni) Tolby egyedül próbálja meg felvenni a harcot az erőddel. Az őrségen könnyen átjut, hiszen azok soha nem harcoltak, de végül mégis elkezdik őt üldözni. Bemenekül Fowler, Bors egyik helyettesének szobájába. Szerencséjére Fowlernek az az ötlete támad, hogy Tolbyval öleti meg Borst (mivel ő maga erre nem lenne képes, viszont az anarchia szimpatikus neki). Tolbynak végül is sikerül szétvernie Bors robotfejét, akinek halála miatt szétesik az általa felépített rendszer. Fowler a biztonság kedvéért elteszi Bors adatbázisát, hátha még szüksége lesz rá…\n\nForrások \n Philip K. Dick: Lenn a sivár földön (Agave Kiadó, 2005)\n\nPhilip K. Dick-novellák",
    "question": "Mely kutató munkáját pusztították el a felkelők?",
    "answers": {
        "answer_start": [407],
        "text": ["a kormány"]
    }
}
```

```json
{
    "context": 'Az U–1230 tengeralattjárót a német haditengerészet rendelte a hamburgi Deutsche Werft AG-től 1941. október 14-én. A hajót 1944. január 26-án vették hadrendbe. Egy járőrutat tett, amelyen egy hajót süllyesztett el.\n\nPályafutása \nAz U–1230 első és egyetlen harci küldetésére Hans Hilbig kapitány irányításával 1944. október 8-án futott ki Hortenből. Az Atlanti-óceán északi részén kelt át, majd november 29-én – az Elster hadművelet (németül Unternehmen Elster, magyarul Szarka hadművelet) – két német ügynököt rakott partra az amerikai Hancock Pointnál. Ezután az Amerikai Egyesült Államok partjainál, Connecticuttól északra vadászott. \n\nDecember 3-án Maine állam partjainak közelében megtorpedózta a kanadai Cornwallis nevű gőzöst, amely Barbadosról tartott St. Johnba, fedélzetén cukorral és melasszal. A Cornwallis 1942. szeptember 11-én kapott már torpedótalálatot Bridgetownban az U–514-től, de akkor még ki lehetett emelni a sekély vízből. Az U–1230 torpedója azonban végzetes volt, a fedélzeten tartózkodó 48 emberből 43 meghalt.\n\nŐrjárata befejeztével a tengeralattjáró visszatért Norvégiába, majd onnan 1945. február 20-án Flensburgba hajózott. 1945. május 5-én a németországi Heligolandnál adta meg magát. 1945. július 24-én Wilhelmshavenből indult a skóciai Loch Ryanbe, ahol a szövetségesek a megsemmisítésre kijelölt búvárhajókat gyűjtötték. Az U– össze 1230-at a HMS Cubitt brit fregatt süllyesztette el a Deadlight hadműveletben.\n\nKapitány\n\nŐrjárat\n\nElsüllyesztett hajó\n\nJegyzetek\n\nForrások \n  \n  \n  \n  \n\nIXC/40 típusú német tengeralattjárók',
    "question": "Ki rendelte meg az U-1230-as tengeralattjárót?",
    "answers": {
        "answer_start": [62],
        "text": ["hamburgi Deutsche Werft AG-től"]
    }
}
```

```json
{
    "context": "A budapesti 56B jelzésű villamos Hűvösvölgy és a Csóka utca között közlekedett a 2022-es budafoki vágányzár idején. A viszonylatot a Budapesti Közlekedési Zrt. üzemeltette.\n\nTörténete \n\n1981. október 22-étől a Széll Kálmán (akkor Moszkva) tér és Hűvösvölgy közötti pályafelújítási munkálatok miatt az 56-os villamos megosztott útvonalon, 56A jelzéssel a Széll Kálmán tér felől, 56B jelzéssel pedig Hűvösvölgy felől Budagyöngyéig közlekedett. 1982. május 24-étől az 56B rövidített útvonalon, minden nap 6 és 12 óra között Budagyöngyétől a Vadaskerti utcáig, majd 12 óra után a Nagyhíd megállóhelyig járt. 1982. szeptember 18-án a felújítás befejeztével megszűnt. 1983. június 13. és 19. között ismét közlekedett, ekkor a Budagyöngye és a Nyéki út közötti szakaszon. November 8-án újraindult a Heinrich István útig, majd november 24-én végleg megszűnt.\n\n2022. október 3. és november 18. között a Hűvösvölgy és a Csóka utca között közlekedett a budafoki vágányzár idején.\n\nÚtvonala\n\nMegállóhelyei \nAz átszállási kapcsolatok között a Hűvösvölgy és a Móricz Zsigmond körtér között azonos útvonalon közlekedő 56-os és 56A villamos nincs feltüntetve.\n\n!Perc\xa0(↓)\n!Megállóhely\n!Perc\xa0(↑)\n!Átszállási kapcsolatok a járat közlekedése idején\n|-\n|0||Hűvösvölgyvégállomás||41\n|align=left|\n|-\n|2||Heinrich István utca||38\n|align=left|\n|-\n|3||Völgy utca||37\n|align=left|\n|-\n|4||Vadaskerti utca||36\n|align=left|\n|-\n|5||Nagyhíd||35\n|align=left|\n|-\n|6||Zuhatag sor||34\n|align=left|\n|-\n|8||Kelemen László utca||33\n|align=left|\n|-\n|9||Akadémia||32\n|align=left|\n|-\n|10||Budagyöngye||31\n|align=left|\n|-\n|11||Nagyajtai utca||29\n|align=left|\n|-\n|14||Szent\xa0János\xa0Kórház||27\n|align=left|\n|-\n|15||Városmajor||26\n|align=left|\n|-\n|16||Nyúl utca||25\n|align=left|\n|-\n|18||Széll\xa0Kálmán\xa0tér\xa0M||24\n|align=left|\n|-\n|20||Déli pályaudvar M||22\n|align=left|\n|-\n|21||Mikó utca||20\n|\n|-\n|22||Krisztina tér||18\n|align=left|\n|-\n|24||Dózsa György tér||16\n|align=left|\n|-\n|26||Döbrentei tér||14\n|align=left|\n|-\n|27||Rudas Gyógyfürdő||13\n|align=left|\n|-\n|30||Szent Gellért tér – Műegyetem M||11\n|align=left|\n|-\n|32||Gárdonyi tér||9\n|align=left|\n|-\n|35||Móricz Zsigmond körtér\xa0M||6\n|align=left|\n|-\n|37||Kosztolányi Dezső tér||4\n|align=left|\n|-\n|38||Karolina út||2\n|align=left|\n|-\n|39||Csóka utcavégállomás||0\n|align=left|\n|}\n\nJegyzetek\n\nForrások \n\nBudapest megszűnt villamosvonalai",
    "question": "A 2022-es budafoki vágányzár alatt mikor járt az 56B jelzésű villamos a Hűvösvölgy és a Csóka utca között?",
    "answers": {
        "answer_start": [852],
        "text": ["2022. október 3. és november 18. között"]
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Az alábbiakban szövegek szerepelnek a hozzájuk tartozó kérdésekkel és válaszokkal.
  ```

- Base prompt template:

  ```text
  Szöveg: {text}
  Kérdés: {question}
  Válasz legfeljebb 3 szóban:
  ```

- Instruction-tuned prompt template:

  ```text
  Szöveg: {text}

  Válaszoljon az alábbi kérdésre a fenti szöveg alapján legfeljebb 3 szóban.

  Kérdés: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-hu
```

## Knowledge

### Exams-bg

This dataset was published in [this paper](https://aclanthology.org/2023.acl-long.487/)
and contains questions collected from high school (HS) examinations in Bulgaria.

The original full dataset consists of 1,329 / 365 / 1,472 samples for
training, validation and testing, respectively. We only keep samples that have 4 choices,
and we thus use a 1,024 / 94 / 2,048 split for training, validation and testing,
respectively. The train and validation set are sampled from the original splits, but
the test set has additional samples from both the original train and validation sets.

Here are a few examples from the training split:

```json
{
    "text": "При свързването на три аминокиселини се образува:\nВъзможности:\na. тризахарид\nb. трипептид\nc. тринуклеотид\nd. триглицерид",
    "label": "b"
}
```

```json
{
    "text": "През 1911 г. Българското книжовно дружество се преименува на:\nВъзможности:\na. Народна библиотека „Кирил и Методий”\nb. Софийски държавен университет\nc. Българска академия на науките\nd. Висше педагогическо училище",
    "label": "c"
}
```

```json
{
    "text": "Коя земеделска култура се отглежда само в Южна България?\nВъзможности:\na. тютюн\nb. слънчоглед\nc. ориз\nd. царевица",
    "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Следват въпроси с множествен избор (с отговори).
  ```

- Base prompt template:

  ```text
  Въпрос: {text}
  Възможности:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Отговор: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Въпрос: {text}

  Отговорете на горния въпрос като отговорите с 'a', 'b', 'c' или 'd', и нищо друго.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset exams-bg
```

## Common-sense Reasoning

### Winogrande-bg

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2506.19468)
and is a translated and filtered version of the English [Winogrande
dataset](https://doi.org/10.1145/3474381).

The original full dataset consists of 47 / 1,210 samples for training and testing, and
we use 128 of the test samples for validation, resulting in a 47 / 128 / 1,085 split for
training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
    "text": "Не можех да контролирам влагата както контролирах дъжда, защото _ идваше отвсякъде. На какво се отнася празното място _?\nВъзможности:\na. влага\nb. дъжд",
    "label": "a"
}
```

```json
{
    "text": "Джесика смяташе, че "Sandstorm" е най-великата песен, писана някога, но Патриция я мразеше. _ купи билет за джаз концерта. На какво се отнася празното място _?\nВъзможности:\na. Джесика\nb. Патриция",
    "label": "b"
}
```

```json
{
    "text": "Термостатът показа, че долу е двадесет градуса по-хладно, отколкото горе, затова Байрон остана в _ защото му беше студено. На какво се отнася празното място _?\nВъзможности:\na. долу\nb. горе",
    "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Следват въпроси с множествен избор (с отговори).
  ```

- Base prompt template:

  ```text
  Въпрос: {text}
  Възможности:
  a. {option_a}
  b. {option_b}
  Отговор: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Въпрос: {text}
  Възможности:
  a. {option_a}
  b. {option_b}

  Отговорете на горния въпрос като отговорите с 'a' или 'b', и нищо друго.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset winogrande-bg
```
