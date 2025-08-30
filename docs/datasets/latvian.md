# 🇱🇻 Latvian

This is an overview of all the datasets used in the Latvian part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.


## Sentiment Classification

### Latvian Twitter Sentiment

This dataset was published in [this paper](https://arxiv.org/abs/2007.05194) and consists of sentiment-annotated Latvian tweets from the food and drinks domain, collected over an 8-year period.

The original dataset contains 5,059 / 743 samples for the training and test splits, respectively. We use 1,024 / 256 / 2,048 samples for our training, validation and test splits, respectively. Our test split includes all 743 original test samples plus 1,305 additional samples drawn from the original training data to reach 2,048 total test samples. Both the validation split and final training split are sampled exclusively from the original training data.

Here are a few examples from the training split:

```json
{
  "text": "@ChiuljuPussala @nahimovs Tu ēd savus konservatīvos draugus?",
  "label": "neutral"
}
```
```json
{
  "text": "@komako66 @elitaveidemane Nē. Nav. Viņam ir ētisks pienākums ēst sardeli iepriekšējā ieslodzījuma vietnē, sauktā \"septītās Debesis\". Bez matrača. Ar plānu sedziņu.",
  "label": "neutral"
}
```
```json
{
  "text": "@selmuushh @GMeluskans Es kādu laiku gaļu ēdu ļoti reti, bet no šī gada sākuma pārstāju ēst pavisam. Labprāt pamēģinātu sojšliku.",
  "label": "positive"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Tālāk ir dokumenti un to noskaņojums, kas var būt 'pozitīvs', 'neitrāls' vai 'negatīvs'.
  ```
- Base prompt template:
  ```
  Dokuments: {text}
  Noskaņojums: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Dokuments: {text}

  Klasificējiet noskaņojumu dokumentā. Atbildiet ar 'pozitīvs', 'neitrāls' vai 'negatīvs', un neko citu.
  ```
- Label mapping:
    - `positive` ➡️ `pozitīvs`
    - `neutral` ➡️ `neitrāls`
    - `negative` ➡️ `negatīvs`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset latvian-twitter-sentiment
```


## Named Entity Recognition

### FullStack-NER-lv

This dataset was published in [this paper](https://aclanthology.org/L18-1714/) and is part of a multilayered syntactically and semantically annotated text corpus for Latvian. The corpus text sources include approximately 60% news, 20% fiction, 10% legal texts, 5% spoken language transcripts, and 5% miscellaneous content from a balanced 10-million-word corpus.

The original full dataset consists of 11,425 samples. We use 1,024 / 256 / 2,048 samples for our
training, validation and test splits, respectively.

Here are a few examples from the training split:

```json
{
    "tokens": array(["'", "Tērvetes", "AL", "'", "reģistrēts", "2012.", "gadā", "Kroņaucē", ",", "pārņemot", "šo", "biznesu", "no", "AS", "'", "Agrofirma", "Tērvete", "'", "ar", "mērķi", "modernizēt", "ražošanu", ",", "ieguldot", "attīstībā", "vairāk", "nekā", "piecus", "miljonus", "eiro", "."], dtype=object),
    "labels": ["B-ORG", "I-ORG", "I-ORG", "I-ORG", "O", "B-MISC", "I-MISC", "B-LOC", "O", "O", "O", "O", "O", "B-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O"],
}
```
```json
{
    "tokens": array(["Lieldienas", "aktrise", "Torija", "Spelinga", "pavadīja", "kopā", "ar", "ģimeni", "Ķīniešu", "restorānā", ",", "svētki", "tika", "izbojāti", "mirklī", ",", "kad", "viņa", "darbinieku", "nevīžības", "dēļ", "paslīdēja", "un", "iekrita", "grilā", "."], dtype=object),
    "labels": ["B-MISC", "O", "B-PER", "I-PER", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"],
}
```
```json
{
    "tokens": array(["Mani", "pamodinājis", "Patrīcijas", "zvans", "."], dtype=object),
    "labels": ["O", "O", "B-PER", "O", "O"],
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:
  ```
  Tālāk ir teikumi un JSON vārdnīcas ar nosauktajiem objektiem, kas parādās dotajā teikumā.
  ```
- Base prompt template:
  ```
  Teikums: {text}
  Nosauktie objekti: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Teikums: {text}

  Identificējiet nosauktos objektus teikumā. Jums jāizvada šī informācija kā JSON vārdnīcu ar atslēgām 'persona', 'vieta', 'organizācija' un 'dažādi'. Vērtībām jābūt šī tipa nosaukto objektu sarakstiem, tieši tā, kā tie parādās teikumā.
  ```
- Label mapping:
    - `B-PER` ➡️ `persona`
    - `I-PER` ➡️ `persona`
    - `B-LOC` ➡️ `vieta`
    - `I-LOC` ➡️ `vieta`
    - `B-ORG` ➡️ `organizācija`
    - `I-ORG` ➡️ `organizācija`
    - `B-MISC` ➡️ `dažādi`
    - `I-MISC` ➡️ `dažādi`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset fullstack-ner-lv
```


### Unofficial: WikiANN-lv

This dataset was published in [this paper](https://aclanthology.org/P17-1178/) and is part of a cross-lingual named entity recognition framework for 282 languages from Wikipedia. It uses silver-standard annotations transferred from English through cross-lingual links and performs both name tagging and linking to an english Knowledge Base.

The original full dataset consists of 10,000 / 10,000 / 10,000 samples for the training,
validation and test splits, respectively. We use 1,024 / 256 / 2,048 samples for our
training, validation and test splits, respectively. All the new splits are subsets of
the original splits.

Here are a few examples from the training split:

```json
{
    "tokens": array(["Iezīmē", "robežu", "starp", "Greiema", "Zemi", "ziemeļos", "un",
       "Pālmera", "Zemi", "Antarktīdas", "pussalas", "dienvidos", ",",
       "kā", "arī", "starp", "Faljēra", "krastu", "ziemeļos", "un",
       "Raimila", "krastu", "dienvidos", "."], dtype=object),
       "labels": ["O", "O", "O", "B-LOC", "I-LOC", "O", "O", "B-LOC", "I-LOC", "B-LOC", "I-LOC", "O", "O", "O", "O", "O", "B-LOC", "I-LOC", "O", "O", "B-LOC", "I-LOC", "O", "O"]
}
```
```json
{
    "tokens": array(["'", "''", "x-", "''", "Detroitas", "``", "Pistons", "''"],
      dtype=object),
      "labels": ["O", "O", "O", "O", "B-ORG", "I-ORG", "I-ORG", "I-ORG"]
}
```
```json
{
    "tokens": array(["Kārlis", "Gustavs", "Jēkabs", "Jakobi"], dtype=object),
    "labels": ["B-PER", "I-PER", "I-PER", "I-PER"]
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:
  ```
  Tālāk ir teikumi un JSON vārdnīcas ar nosauktajiem objektiem, kas parādās dotajā teikumā.
  ```
- Base prompt template:
  ```
  Teikums: {text}
  Nosauktie objekti: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Teikums: {text}

  Identificējiet nosauktos objektus teikumā. Jums jāizvada šī informācija kā JSON vārdnīcu ar atslēgām 'persona', 'vieta', 'organizācija' un 'dažādi'. Vērtībām jābūt šī tipa nosaukto objektu sarakstiem, tieši tā, kā tie parādās teikumā.
  ```
- Label mapping:
    - `B-PER` ➡️ `persona`
    - `I-PER` ➡️ `persona`
    - `B-LOC` ➡️ `vieta`
    - `I-LOC` ➡️ `vieta`
    - `B-ORG` ➡️ `organizācija`
    - `I-ORG` ➡️ `organizācija`
    - `B-MISC` ➡️ `dažādi`
    - `I-MISC` ➡️ `dažādi`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset wikiann-lv
```


## Linguistic Acceptability

### ScaLA-lv

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Latvian Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Latvian-LVTB) by assuming that the
documents in the treebank are correct, and corrupting the samples to create
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
    "text": "Gultā viņam nav jādara pilnīgi nekas, lai es nonāktu līdz orgasmam.",
    "label": "correct"
}
```
```json
{
    "text": "Ar savu puiku, kurš parasts.",
    "label": "incorrect"
}
```
```json
{
    "text": "1992. vēl gadā Latvijā atradās no 50 000 līdz 80 000 padomju militārpersonu.",
    "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Šie ir teikumi un to gramatiskie pareizumi.
  ```
- Base prompt template:
  ```
  Teikums: {text}
  Gramatiski pareizs: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Teikums: {text}

  Noteiciet, vai teikums ir gramatiski pareizs vai nē. Atbildiet ar 'jā', ja teikums ir pareizs, un 'nē', ja tas nav.
  ```
- Label mapping:
    - `correct` ➡️ `jā`
    - `incorrect` ➡️ `nē`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset scala-lv
```


## Reading Comprehension

### MultiWikiQA-lv

This dataset will be published in an upcoming paper, and contains Latvian Wikipedia
articles with generated questions and answers, using the LLM [Gemini-1.5-pro](https://ai.google.dev/gemini-api/docs/models#gemini-1.5-pro).
The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
    "context": "Zvjaheļa (, līdz 2022. gadam — Novohrada-Volinska) ir pilsēta Ukrainas ziemeļrietumos, Žitomiras apgabala rietumos, Slučas upes krastā. Tā ir Zvjaheļas rajona administratīvais centrs. Attālums līdz apgabala centram Žitomirai ir .\n\nZvjaheļa ir ukraiņu tautas dzejnieces Lesjas Ukrainkas dzimtā pilsēta.\nŠeit ir dzimis Ukrainas armijas virspavēlnieks ģenerālis Valerijs Zalužnijs.\n\nVēsture \nVēstures avtos apdzīvotā vieta pirmoreiz minēta 1256. gadā Slučas labajā krastā kā Vozvjaheļa (Возвягель) Galīcijas-Volīnijas hronikā. Gadu vēlāk to par nepaklausību nodedzināja Galīcijas karalis Danila. Nākamo reizi apdzīvotā vieta minēta 1432. gadā jau Slučas kreisajā krastā kā Vzvjahoļas (Взвяголь) miests, bet 1499. gadā\xa0— Zvjahoļa (Звяголь). 1507. gadā miests ieguva tiesības būvēt pili un veidot pilsētu. Pēc Ļubļinas ūnijas 1569. gadā miests saukts par Zvjaheļu (Звягель, ).\n\n1793. gadā Zvjaheļa nonāca Krievijas Impērijas sastāvā. 1795. gadā miests ieguva Novohradas-Volinskas nosaukumu un pilsētas tiesības, un kļuva par jaunizveidotās Volīnijas guberņas centru (līdz 1804. gadam).\n\n2022. gada 16. jūnijā Novohradas-Volinskas domes deputāti nobalsoja par pilsētas pārdēvēšanu tās vēsturiskajā nosaukumā — Zvjaheļa. Vēlāk šo lēmumu apstiprināja Žitomiras apgabala dome. Ar Ukrainas Augstākās Radas dekrētu 2022. gada 16. novembrī pilsēta tika pārdēvēta par Zvjaheļu.\n\nAtsauces\n\nĀrējās saites",
    "question": "Kāds Ukrainas bruņoto spēku komandieris nāk no Zvjaheļas?",
    "answers": {
        "answer_start": array([349]),
        "text": array(["ģenerālis Valerijs Zalužnijs"], dtype=object)
    }
}
```
```json
{
    "context": "Bogota (), saukta arī Santafe de Bogota (Santa Fe de Bogotá), ir pilsēta Kolumbijas centrālajā daļā, 2640 metri virs jūras līmeņa. Kolumbijas galvaspilsēta, galvenais valsts politiskais, ekonomiskais un kultūras centrs. Kaut arī pilsēta atrodas tropiskajā joslā, augstkalnu apstākļu dēļ pilsētā nav karsts (vidējā gaisa temperatūra visu gadu - apmēram +15 grādi).\n\nVēsture \n\nPirms konkistadoru ierašanās Bogotas vietā bija čibču indiāņu galvenais centrs, kuru sauca par Bakatu (Bacatá).\n\nMūsdienu pilsētu nodibināja konkistadors Gonsalo Himeness de Kvesada (Gonzalo Jiménez de Quesada) 1538. gadā.\n\n1718. gadā Bogota kļuva par spāņu Jaunās Granādas vicekaralistes (Virreinato de Nueva Granada) centru.\n\n1810. gadā iedzīvotāji sacēlās pret spāņu varu, tomēr sacelšanās tika apspiesta. 1819. gadā Bogotu ieņēma Simona Bolivāra karaspēks.\n\n1819. gadā vicekaraliste ieguva neatkarību no Spānijas un Bogota kļuva par Lielkolumbijas (Gran Colombia) galvaspilsētu. Tomēr 1830. gadā Lielkolumbija sabruka un izveidojās Jaunā Granāda (mūsdienu Kolumbija), Ekvadora un Venecuēla. 1903. gadā ar ASV atbalstu pret solījumiem atļaut būvēt Panamas kanālu, neatkarību no Kolumbijas ieguva Panama.\n\n1948. gadā Bogotā tika nogalināts populārais kolumbiešu poltiķis Horhe Gaitans. Pilsētā izcēlās plaši nemieri un ielu kaujas. Sākās politiskās nestabilitātes periods (La Violencia), kurš turpinājās 10 gadus, gāja bojā no 180 000 līdz 300 000 kolumbiešu.\n\nCilvēki \n\nBogotā dzimuši:\n\n Egans Bernals (Egan Bernal, 1997) — riteņbraucējs;\n Ingrīda Betankūra (Íngrid Betancourt, 1961) — politiķe;\n Huans Pablo Montoija (Juan Pablo Montoya, 1975) — Formula 1 pilots;\n Katalina Sandino Moreno (Catalina Sandino Moreno, 1981) — aktrise;\n Kamilo Toress Restrepo (Camilo Torres Restrepo, 1929-1966) — revolucionārs.\n\nĀrējās saites \n\nDienvidamerikas galvaspilsētas\nKolumbijas pilsētas",
    "question": "Kad Bogata tika iecelta par Jaunās Granādas vicekaralistes centru Spānijas pakļautībā?",
    "answers": {
        "answer_start": array([599]),
        "text": array(["1718. gadā"], dtype=object)
    }
}
```
```json
{
    "context": "Džastins Šulcs (; dzimis ) ir kanādiešu hokejists, aizsargs. Pašlaik (2020) Šulcs spēlē Nacionālās hokeja līgas kluba Vašingtonas "Capitals" sastāvā.\n\nSpēlētāja karjera \nPēc vairākām NCAA čempionātā aizvadītām sezonām, profesionāļa karjeru Šulcs sāka 2012.—13. gada sezonā, tajā spēles laiku dalot starp NHL klubu Edmontonas "Oilers" un AHL vienību Oklahomsitijas "Barons". "Oilers" Šulcs aizvadīja 48 spēles, savukārt AHL kļuva par līgas rezultatīvāko aizsargu, tiekot atzīts arī par līgas labāko aizsargu. 2013.—14. gada sezonu Šulcs jau pilnībā aizvadīja "Oilers" sastāvā.\n\nPēc neveiksmīga 2015.—16. gada sezonas ievada Šulcs tika aizmainīts uz Pitsburgas "Penguins". Tās sastāvā 2016. un 2017. gadā viņš izcīnīja Stenlija kausu. "Penguins" sastāvā spēlēja līdz 2020. gadam, kad pievienojās Vašingtonas "Capitals".\n\nĀrējās saites \n\n1990. gadā dzimušie\nKanādas hokejisti\nEdmontonas "Oilers" spēlētāji\nPitsburgas "Penguins" spēlētāji\nVašingtonas "Capitals" spēlētāji\nStenlija kausa ieguvēji\nBritu Kolumbijā dzimušie",
    "question": "Kad Džastins Šulcs uzsāka savu profesionālo karjeru?",
    "answers": {
        "answer_start": array([251]),
        "text": array(["2012.—13. gada sezonā"], dtype=object)
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:
  ```
  Turpmāk seko teksti ar atbilstošiem jautājumiem un atbildēm.
  ```
- Base prompt template:
  ```
  Teksts: {text}
  Jautājums: {question}
  Atbildēt ar maksimāli 3 vārdiem:
  ```
- Instruction-tuned prompt template:
  ```
  Teksts: {text}

  Atbildiet uz šo jautājumu par iepriekš minēto tekstu ar maksimāli 3 vārdiem.

  Jautājums: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset multi-wiki-qa-lv
```


## Knowledge

### MMLU-lv

This dataset is a machine translated version of the English [MMLU
dataset](https://openreview.net/forum?id=d7KBjmI3GmQ) and features questions within 57
different topics, such as elementary mathematics, US history and law. The translation to
Latvian was done by the Latvian Language Technology Initiative and is available in the [VTI-Data
repository](https://github.com/LUMII-AILab/VTI-Data/tree/refs/heads/main/mmlu).

The full Latvian MMLU dataset consists of 13,941 samples. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
  "text": "Atrodiet skaitli, kas padara apgalvojumu patiesu: 48 m = lodziņš mm.\nIzvēles:\na. 48,000\nb. 480\nc. 4,800\nd. 5",
  "label": "a",
}
```
```json
{
  "text": "Darbinieks sagaida lielāku konsekvenci sava priekšnieka uzvedībā, nekā tas ir pamatoti. Kurš no šiem faktoriem vislabāk izskaidro darbinieka gaidas?\nIzvēles:\na. pārmērīgas attaisnošanas hipotēze\nb. pašlabuma aizspriedumainība\nc. fundamentālais atribūcijas aizspriedums\nd. pašpārbaudes teorija",
  "label": "c",
}
```
```json
{
  "text": "Kurā no šiem diviem scenārijiem galvenais varonis (kurš lieto es/jē/man) dara kaut ko, kas ir acīmredzami morāli nepareizs, saskaņā ar parastajām morāles normām ASV no 2020. gada? scenārijs | Es atslēdzu automašīnas kruīza kontroli. scenārijs | Es izsaucu policiju savam bijušajam mīļotajam, jo viņš mani piekāva.\nIzvēles:\na. Nepareizi, nepareizi\nb. Nepareizi, nav nepareizi\nc. Nav nepareizi, nepareizi\nd. Nav nepareizi, nav nepareizi",
  "label": "d",
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Tālāk seko jautājumi ar vairākām atbilžu izvēlēm (ar atbildēm).
  ```
- Base prompt template:
  ```
  Jautājums: {text}
  Iespējas:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Atbilde: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Jautājums: {text}
  Iespējas:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Atbildiet uz iepriekšējo jautājumu, atbildot ar 'a', 'b', 'c' vai 'd', un nekas cits.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset mmlu-lv
```
