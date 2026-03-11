# 🇭🇺 Hungarian

This is an overview of all the datasets used in the Hungarian part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### HuSST

This dataset was published in [this
paper](https://acta.bibl.u-szeged.hu/75891/1/msznykonf_018_431-446.pdf) and is the
Hungarian version of the Stanford Sentiment Treebank.

The original dataset contains 9,328 / 1,165 / 1,165 samples for the training,
validation, and test splits, respectively. We use 1,024 / 256 / 2,048 samples for our
training, validation and test splits, respectively. The train and validation splits are
subsets of the original splits. The original test split does not contain any labels, so
our test split is created from the training split.

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
  - `correct` ➡️ `igen`
  - `incorrect` ➡️ `nem`

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

### Unofficial: INCLUDE-hu

This dataset is part of [INCLUDE](https://doi.org/10.48550/arXiv.2411.19799), a
comprehensive knowledge- and reasoning-centric benchmark that evaluates multilingual
LLMs across 44 languages. It contains 4-option multiple-choice questions extracted from
academic and professional exams, covering 57 topics including regional knowledge.

The original dataset consists of a 'validation' split used as training data and a 'test'
split. We use the 'validation' split as the training split, which has 25 samples. We
sample 64 samples from the 'test' split for the validation split, and use the remaining
512 samples for the test split. The sampling is done stratified by the subject column.

Here are a few examples from the dataset:

```json
{
    "text": "Melyik hormon szabályozza a tejleadást?\nVálaszlehetőségek:\na. az oxitocin\nb. a progeszteron\nc. az ösztrogén\nd. az adrenalin",
    "label": "a",
    "subject": "Agriculture"
}
```

```json
{
    "text": "Melyik hazánk legszelesebb tája?\nVálaszlehetőségek:\na. Tiszántúl\nb. Kisalföld\nc. Északi-középhegység\nd. Duna−Tisza köze",
    "label": "b",
    "subject": "Environmental studies and forestry"
}
```

```json
{
    "text": "Melyik igaz az alábbi állítások közül?\nVálaszlehetőségek:\na. Az RNS hidrolízise aldohexózt is eredményez.\nb. Az amidok vizes oldatban erős bázisként viselkednek.\nc. Az adenin, citozin és a guanin a DNS- és RNS-molekulák hidrolízisének termékei.\nd. A fehérjék savas hidrolízisében foszforsav is keletkezik.",
    "label": "c",
    "subject": "Chemistry"
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
  Válasz: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Kérdés: {text}

  Válaszoljon a fenti kérdésre az elérhető lehetőségek közül {labels_str} használatával, és semmi mással.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset include-hu
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

## Hallucination Detection

### MultiWikiHalluQA-hu

This dataset uses the same data as [MultiWikiQA-hu](#multiwikiqa-hu), published in
[this paper](https://doi.org/10.48550/arXiv.2509.04111), containing Wikipedia articles
with LLM-generated questions and answers in 300+ languages. Rather than evaluating the
correctness of the generated answer, this task evaluates the degree to which the model
hallucinates, i.e., generates tokens that are not grounded in the provided context.

The hallucination detection is performed using the
[LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect) library, which uses a
transformer-based classifier to predict hallucination at the token level. The metric
reported is the hallucination rate, computed as the ratio of hallucinated tokens to
total tokens in the generated answers.

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
euroeval --model <model-id> --dataset multi-wiki-hallucination-qa-hu
```
