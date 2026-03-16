# 🇱🇻 Latvian

This is an overview of all the datasets used in the Latvian part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### Latvian Twitter Sentiment

This dataset was published in [this paper](https://arxiv.org/abs/2007.05194) and
consists of sentiment-annotated Latvian tweets from the food and drinks domain,
collected over an 8-year period.

The original dataset contains 5,059 / 743 samples for the training and test splits,
respectively. We use 1,024 / 256 / 2,048 samples for our training, validation and test
splits, respectively. Our test split includes all 743 original test samples plus 1,305
additional samples drawn from the original training data to reach 2,048 total test
samples. Both the validation split and final training split are sampled exclusively from
the original training data.

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

  ```text
  Tālāk ir dokumenti un to noskaņojums, kas var būt 'pozitīvs', 'neitrāls' vai 'negatīvs'.
  ```

- Base prompt template:

  ```text
  Dokuments: {text}
  Noskaņojums: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Dokuments: {text}

  Klasificējiet noskaņojumu dokumentā. Atbildiet ar 'pozitīvs', 'neitrāls' vai 'negatīvs', un neko citu.
  ```

- Label mapping:
  - `positive` ➡️ `pozitīvs`
  - `neutral` ➡️ `neitrāls`
  - `negative` ➡️ `negatīvs`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset latvian-twitter-sentiment
```

## Named Entity Recognition

### FullStack-NER-lv

This dataset was published in [this paper](https://aclanthology.org/L18-1714/) and is
part of a multilayered syntactically and semantically annotated text corpus for Latvian.
The corpus text sources include approximately 60% news, 20% fiction, 10% legal texts, 5%
spoken language transcripts, and 5% miscellaneous content from a balanced
10-million-word corpus.

The original full dataset consists of 11,425 samples. We use 1,024 / 256 / 2,048 samples
for our
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

  ```text
  Tālāk ir teikumi un JSON vārdnīcas ar nosauktajiem objektiem, kas parādās dotajā teikumā.
  ```

- Base prompt template:

  ```text
  Teikums: {text}
  Nosauktie objekti: {label}
  ```

- Instruction-tuned prompt template:

  ```text
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
euroeval --model <model-id> --dataset fullstack-ner-lv
```

### Unofficial: WikiANN-lv

This dataset was published in [this paper](https://aclanthology.org/P17-1178/) and is
part of a cross-lingual named entity recognition framework for 282 languages from
Wikipedia. It uses silver-standard annotations transferred from English through
cross-lingual links and performs both name tagging and linking to an english Knowledge
Base.

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

  ```text
  Tālāk ir teikumi un JSON vārdnīcas ar nosauktajiem objektiem, kas parādās dotajā teikumā.
  ```

- Base prompt template:

  ```text
  Teikums: {text}
  Nosauktie objekti: {label}
  ```

- Instruction-tuned prompt template:

  ```text
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
euroeval --model <model-id> --dataset wikiann-lv
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

  ```text
  Šie ir teikumi un to gramatiskie pareizumi.
  ```

- Base prompt template:

  ```text
  Teikums: {text}
  Gramatiski pareizs: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Teikums: {text}

  Noteiciet, vai teikums ir gramatiski pareizs vai nē. Atbildiet ar 'jā', ja teikums ir pareizs, un 'nē', ja tas nav.
  ```

- Label mapping:
  - `correct` ➡️ `jā`
  - `incorrect` ➡️ `nē`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-lv
```

## Reading Comprehension

### MultiWikiQA-lv

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

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

  ```text
  Turpmāk seko teksti ar atbilstošiem jautājumiem un atbildēm.
  ```

- Base prompt template:

  ```text
  Teksts: {text}
  Jautājums: {question}
  Atbildēt ar maksimāli 3 vārdiem:
  ```

- Instruction-tuned prompt template:

  ```text
  Teksts: {text}

  Atbildiet uz šo jautājumu par iepriekš minēto tekstu ar maksimāli 3 vārdiem.

  Jautājums: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-lv
```

## Knowledge

### MMLU-lv

This dataset is a machine translated version of the English [MMLU
dataset](https://openreview.net/forum?id=d7KBjmI3GmQ) and features questions within 57
different topics, such as elementary mathematics, US history and law. The translation to
Latvian was done by the Latvian Language Technology Initiative and is available in the
[VTI-Data
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

  ```text
  Tālāk seko jautājumi ar vairākām atbilžu izvēlēm (ar atbildēm).
  ```

- Base prompt template:

  ```text
  Jautājums: {text}
  Izvēles:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Atbilde: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Jautājums: {text}
  Izvēles:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Atbildiet uz iepriekšējo jautājumu, atbildot ar 'a', 'b', 'c' vai 'd', un nekas cits.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset mmlu-lv
```

## Common-sense Reasoning

### COPA-lv

This dataset was published in [this
paper](https://aclanthology.org/2025.resourceful-1.22/) and is a translated version of
the English [COPA dataset](https://aclanthology.org/S12-1052/), which was created from
scratch by the authors. The dataset was machine translated using the [Tilde Translation
service](https://tilde.ai/machine-translation/), and the test samples were manually
post-edited.

The original full dataset consists of 214 / 57 / 132 samples, and we keep the splits
as-is.

Here are a few examples from the training split (which have _not_ been post-edited):

```json
{
  "text": "Īrnieki tika izlikti no dzīvokļa.\nIzvēles:\na. Viņi savu īri nemaksāja.\nb. Viņi sapratās ar savu saimnieku.",
  "label": "a"
}
```

```json
{
  "text": "Svešinieks man svešvalodā kliedza.\nIzvēles:\na. ES truli blenzu uz viņu.\nb. ES apstājos, lai papļāpātu ar viņu.",
  "label": "a"
}
```

```json
{
  "text": "Pagriezu gaismas slēdzi uz augšu un uz leju.\nIzvēles:\na. Gaisma izdzisa.\nb. Gaisma mirgoja.",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Tālāk seko jautājumi ar vairākām atbilžu izvēlēm (ar atbildēm).
  ```

- Base prompt template:

  ```text
  Jautājums: {text}
  Izvēles:
  a. {option_a}
  b. {option_b}
  Atbilde: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Jautājums: {text}
  Izvēles:
  a. {option_a}
  b. {option_b}

  Atbildiet uz iepriekšējo jautājumu, atbildot ar 'a' vai 'b', un nekas cits.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset copa-lv
```

### Unofficial: Winogrande-lv

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2506.19468)
and is a translated and filtered version of the English [Winogrande
dataset](https://doi.org/10.1145/3474381).

The original full dataset consists of 47 / 1,210 samples for training and testing, and
we use 128 of the test samples for validation, resulting in a 47 / 128 / 1,085 split for
training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Pērkot māju, Patrīcijai nav tik daudz naudas, ko tērēt kā Tanjai, tāpēc _ nopērk vienas guļamistabas māju. Ko norāda tukšums _?\nIzvēles:\na. Patrīcija\nb. Tanja",
  "label": "a"
}
```

```json
{
  "text": "Es nevarēju kontrolēt mitrumu, kā es kontrolēju lietu, jo _ nāca no visām pusēm. Ko norāda tukšums _?\nIzvēles:\na. mitrums\nb. lietus",
  "label": "a"
}
```

```json
{
  "text": "Derriks nespēja koncentrēties darbā, atšķirībā no Džastina, jo _ bija jautrs darbs. Ko norāda tukšums _?\nIzvēles:\na. Derriks\nb. Džastins",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Tālāk seko jautājumi ar vairākām atbilžu izvēlēm (ar atbildēm).
  ```

- Base prompt template:

  ```text
  Jautājums: {text}
  Izvēles:
  a. {option_a}
  b. {option_b}
  Atbilde: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Jautājums: {text}
  Izvēles:
  a. {option_a}
  b. {option_b}

  Atbildiet uz iepriekšējo jautājumu, atbildot ar 'a' vai 'b', un nekas cits.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset winogrande-lv
```

## Summarisation

### LSM

This dataset contains news articles and their corresponding summaries from the Latvian
public media news portal [LSM.lv](https://www.lsm.lv/).

Samples were collected using the
[lsm_scraper](https://github.com/alexandrainst/lsm_scraper). We use 1,024 / 256 / 2,048
samples for training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "FOTO: Raimonda Paula un Elīnas Garančas satikšanās koncertā «Ja tevis nebūtu...»\n\nIdeja svinēt apaļo jubileju uz vienas skatuves ar izcilo operdziedātāju Elīnu Garanču Maestro radusies, kopā uzstājoties jau pirms pieciem gadiem. Maestro neslēpj gandarījumu, ka pandēmijas dēļ pārceltais koncerts beidzot notiks. Raimonds Pauls koncertprogrammā “Ja tevis nebūtu...” dziedātājai veltījis divus jaunus dziesmu ciklus ar kopīgi atlasītu Vizmas Belševicas un Ojāra Vācieša dzeju. Savukārt koncerta otrajā daļā iekļautas Paula dziesmas no kinofilmām un teātra izrādēm. Kamerorķestra “Simfonietta Rīga” pavadījumā populāras melodijas atšķirīgās noskaņās izskanēs jaunos aranžējumos, ko veidojuši tādi izcili komponisti kā Lolita Ritmane, Rihards Dubra, Jēkabs Jančevskis un Raimonds Macats. “Man šī otrā daļa ar kino un teātra mūziku ir tāds sapnis, kas ir piepildījies. Jo šis žanrs mani vienmēr ir ļoti interesējis. Varētu teikt, ka es operas žanrā esmu nokļuvusi faktiski nejauši, jo sirds aicinājums no paša sākuma bija tieši teātris,” atklāj Elīna Garanča. Ojārs Rubenis atzīst: “Es varu tikai apbrīnot gan Maestro 85 gados – izturību un to darbu, ko viņš var izdarīt. Un, protams, arī Elīnu Garanču, kura vienkārši ir apbrīnojama savā neambiciozitātē pret visu pārējo un ambiciozitātē pret mākslu. Tas ir tas lielmākslinieku kods!” Maestro un Elīnas Garančas atkalsatikšanās Nacionālajā teātrī būs skatāma piektdien un sestdien, savukārt Latvijas Televīzijā šo koncertu varēs vērot šī gada rudenī.",
  "target_text": "Viņiem bija iecerēts tikties jau šī gada sākumā, bet pandēmijas dēļ Raimonda Paula 85. jubilejai veltītais koncerts ar pasaulslavenās operdziedātājas Elīnas Garančas piedalīšanos tika pārcelts. Šajā nedēļas nogalē Nacionālo teātri beidzot pieskandinās abu izcilo mūzikas personību atkalsatikšanās ar skatītājiem koncertā “Ja tevis nebūtu...”."
}
```

```json

{
"text": "Ukrainā tūkstošiem cilvēku protestē pret korupcijas apkarotāju vājināšanu; Zelenskis sola jaunu likumu\n\nCilvēki pauž neapmierinātību par\xa0korupcijas apkarotāju vājināšanu Trešdienas vakarā Kijivā\xa0bija pilns\xa0Ivana Franka laukums, kas ir tuvākā vieta pie prezidenta Volodimira Zelenska darba vietas, kur var brīvi piekļūt cilvēki. Pārsvarā gados jauni cilvēki bija sanākuši, lai paustu protestu, nožēlu un neapmierinātību ar Augstākās Radas pieņemto likumprojektu, kas paredz atcelt Ukrainas Korupcijas apkarošanas biroja un specializētās pretkorupcijas prokuratūras neatkarību, iestāžu pārraudzību nododot ģenerālprokuroram, kas ir politiski izraudzīts. Cilvēki skandēja visdažādākos saukļus – arī \"Rokas nost no NABU!\", \"Neklusē!\", \"Kauns!\", \"Slava Ukrainai!\", \"Varoņiem slava!\" un daudzus citus. Tā kā pamatā tie bija jaunieši, viņi bija ļoti skaļi un aktīvi. Rokās daudziem bija pašdarināti plakāti. Piemēram, \"Augstākā nodevība\" – spēlējoties ar Augstākās Radas jeb parlamenta nosaukumu. Kāds jaunietis arī bija izveidojis plakātu, kur puse sejas bija no prezidenta Zelenska, otra puse – no bēdīgi slavenā prokrieviskā eksprezidenta Viktora Janukoviča, kurš 2014.\xa0gadā pēc Eiromaidana jeb Pašcieņas revolūcijas asiņainajiem notikumiem aizbēga no Ukrainas un šobrīd slēpjas Krievijā. Aktīvisti Ukrainas protestā pret korupcijas apkarotāju vājināšanu 00:00 / 01:09 Lejuplādēt Indra Sprance Latvijas Radio parunājās ar dažiem no aktīvistiem. Marina: Esmu šeit, jo esmu ļoti sašutusi par pašreizējo situāciju ar likumprojektu. Ir pieņemts likums, kas pilnībā neatbilst Eiropas Savienības un tautas prasībām. Mēs atgriežamies pie tā stāvokļa, kāds bija 2013. gadā, kad mūsu tauta cīnījās par savu ceļu uz Eiropas Savienību. Mans brālis pašlaik karo Pokrovskas tuvumā. Visa šī situācija man šķiet kā spļāviens sejā visiem tiem karavīriem, kas mūs sargā, riskējot ar dzīvībām,\xa0– vara viņiem demonstrē, ka esam tuvāk nevis Eiropas Savienībai un mūsu Rietumu partneriem, bet Krievijai. Ihors: Man gandrīz visi vīriešu kārtas radinieki šobrīd karo, un man nav tiesību šobrīd stāvēt malā. Aleksa: Ukrainā šobrīd notiek ļoti briesmīgas lietas – kamēr daži cilvēki atdod savas dzīvības, lai mēs varētu šeit normāli dzīvot, kāds sagrauj valsti. Un tas nav labi. Mums šeit ir jābūt.\xa0 Tas ir svarīgi. Trešdienas vakarā protesta akcija notika arī Ukrainas otrā lielākajā pilsētā Harkivā, tur pēc \"Radio Brīvība\" aplēsēm bijis līdz pustūkstotim cilvēku. Protesti notikuši arī Černihivā, Zaporižjā, Ļvivā, Dņipro, Krivijrihā, Ivanofrankivskā, Ternopiļā, Odesā un citur. Šī ir jau otrā diena, kad cilvēki iziet ielās. Iepriekš tie bija spontāni protesti, reaģējot uz Augstākās Radas lēmumu, bet trešdien jau daudzviet cilvēkus ielās aicinājušas dažādas sabiedriskās organizācijas. Zelenskis sola jaunu likumu Prezidents Volodimirs Zelenskis trešdien bija noorganizējis tikšanos ar visu Ukrainas tiesību aizsardzības iestāžu vadītājiem, tajā skaitā abu pretkorupcijas iestāžu – NABU un specializētās prokuratūras vadītājiem. Saruna bijusi atklāta un vērtīga. Nākamnedēļ notikšot dziļāka darba tikšanās saistībā ar kopīgajiem darbiem. Pēcāk videouzrunā Zelenskis sacīja, ka ir sadzirdējis cilvēku bažas. Zelenskis piedāvās Augstākajai Radai savu – prezidenta likumprojektu, kas nodrošinās tiesību aizsardzības sistēmas spēku un to, ka nebūs nekāda Krievijas iejaukšanās iestāžu darbā. Jau vēlāk Zelenskis likumprojektu iesniedzis. Vēl gan nav skaidrs, kas tieši šajā likumprojektā ir un kad tieši par to balsos parlaments. Kā likumprojektu komentējis Zelenskis, tas paredz pilnīgas korupcijas apkarošanas iestāžu neatkarības garantijas. Tas arī paredzot reālas iespējas pārliecināties, ka iestāžu darbībā neiejaucas Krievija. Ikvienam, kam ir pieeja valsts noslēpumiem - ne tikai Nacionālajam pretkorupcijas birojam un Specializētajai pretkorupcijas prokuratūrai, bet arī Valsts izmeklēšanas birojam un Valsts policijai - ir jāveic melu detektora pārbaudes un tām jābūt regulārām, likumprojekta saturu komentēja Zelenskis. Likumprojektā ir iekļauti arī noteikumi, kas aizsargā pret dažādiem pārkāpumiem, piebilda prezidents. Pēc jaunā likumprojekta pārskatīšanas Nacionālais pretkorupcijas birojs paziņojumā norādīja, ka ierosinātais likumprojekts patiesi atjaunos visas procesuālās pilnvaras un neatkarības garantijas gan birojā, gan Specializētajā pretkorupcijas prokuratūrā. Arī Ukrainas Korupcijas apkarošanas rīcības centrs, kas ir uzraudzības iestāde, atbalstīja iniciatīvu, sakot, ka tā atjaunos principus, ko iepriekš bija nojaukusi Augstākā Rada. Centrs gan brīdināja, ka pat vienas nedēļas kavēšanās var būt pietiekama, lai iznīcinātu virkni abās pretkorupcijas iestādēs esošās tiesvedības pret augstākajām korumpētajām amatpersonām. KONTEKSTS: Ukrainas parlaments 22. jūlijā apstiprināja likuma grozījumus, kas mazina Ukrainas korupcijas apkarošanas iestāžu neatkarību. Ukrainas Nacionālais pretkorupcijas birojs (NABU) un specializētā prokuratūra turpmāk būs pakļauti Ukrainas ģenerālprokuroram, kas ir Ukrainas prezidenta Volodimira Zelenska izvirzīta amatpersona. Tas izraisījis bažas par korupcijas apkarošanas dienestu pakļaušanu Zelenska komandas interesēm. Ukrainas Drošības dienests iepriekš veicis plaša mēroga kratīšanas pie NABU un specializētās prokuratūras darbiniekiem. Šie soļi izraisījuši protestus Ukrainas iekšienē, kā arī kritiku no Ukrainas partneriem, kas raizējas par demokrātijas standartu vājināšanu un nepietiekamo aktivitāti korupcijas apkarošanā. Tas varētu apgrūtināt Ukrainas izredzes kļūt par Eiropas Savienības dalībvalsti.",
"target_text": "Ukrainā trešdienas vakarā, reaģējot uz šonedēļ lielā steigā pieņemto likumu, kas atceļ pretkorupcijas iestāžu neatkarību, tūkstošiem cilvēku izgāja ielās. Latvijas Radio bija klāt Kijivā, kur pulcējās liels skaits cilvēku."
}
```

```json
{
"text": "Norvēģijas dziesma Eirovīzijā – folkmūzikas, elektronikas un viduslaiku estētikas sintēze\n\nAlessandro ir spāņu izcelsmes, viņš runā četrās valodās, iedvesmojas no dažādu pasaules tautu mūzikas, kā arī ir labs dejotājs. Alessandro dziesma \"Lighter\" ieturēta popmūzikas stilistikā, kurā ievīti daudz dažādi elementi. Te var sadzirdēt gan norvēģu folkmūzikas, gan elektroniskās deju mūzikas notis, gan Balkānu popmūzikai raksturīgos ritmus un pat viduslaiku estētiku. Dziesma aicina noticēt sev un būt pašam savai dzirkstij. Dziesmas \"Lighter \" vārdi Golden girl dressed in ice A heart as dark as night You got me to dim my light No more, (no more) I really think I bought your lies Did anything to keep you mine You kept me hooked on your line No more, (no more) Somewhere along the way I lost my mind I had to walk a hundred thousand miles I’m not afraid to set it all on fire I won’t fall again, I’ll be my own lighter (Eh-Eh-Eh-Eh) Nothing can burn me now (Eh-Eh-Eh-Eh) I’ll be my own lighter I feel a spark inside me I don’t need saving (No way, no way) ‘Cause I’m my own, I’m my own lighter I’m tired of a million tries To fight, the signs And when everybody tried to tell me I should’ve known that it was time to break free Your reigns that kept me at your mercy I’ll burn them to the ground No more, no more Ignite the fire Somewhere along the way I lost my mind I had to walk a hundred thousand miles I’m not afraid to set it all on fire I won’t fall again, I’ll be my own lighter (Eh-Eh-Eh-Eh) Nothing can burn me now (Eh-Eh-Eh-Eh) I’ll be my own lighter I feel a spark inside me I don’t need saving (No way, no way) ‘Cause I’m my own, I’m my own lighter Silence fills the room And I’ve taken off my jewels I wish none of this was true But there’s a fire growing too Yeah! (Eh-Eh-Eh-Eh) Nothing can burn me now (Eh-Eh-Eh-Eh) I’ll be my own lighter I feel a spark inside me I don’t need saving (No way, no way) ‘Cause I’m my own, I’m my own lighter (Eh-Eh-Eh-Eh) Nothing can burn me down (Eh-Eh-Eh-Eh) I’m my own, I’m my own lighter Eirovīzija\xa02025 – dalībnieki Vairāk KONTEKSTS: 2025. gada Eirovīzijas dziesmu konkurss notiks Šveicē, Bāzelē, un savu dalību tajā apstiprinājušas 37 valstis. 31 no visām dalībvalstīm sacentīsies pusfinālos\xa013. maijā un 15. maijā. Desmit\xa0labākie no katra pusfināla kvalificēsies lielajam finālam 17. maijā, pievienojoties pērnā gada uzvarētājai Šveicei un \"lielajam piecniekam\" – Spānijai, Apvienotajai\xa0Karalistei, Vācijai, Itālijai un Francijai. Eirovīzijas konkursa pusfināli un fināli šogad sāksies pulksten 22.00 pēc Latvijas laika. Tiešraides būs skatāmas Latvijas Sabiedriskā medija portālā LSM.lv un satura atskaņotājā REplay.lv, kā arī LTV1. Šī gada Latvijas nacionālajā atlasē \"Supernova\" uzvarēja un uz Eirovīziju dosies grupa \"Tautumeitas\" . \"Tautumeitas\" kāps uz skatuves Eirovīzijas konkursa otrajā pusfinālā. Tajā kopā ar Latviju piedalīsies arī Armēnija, Austrālija, Austrija, Grieķija, Īrija, Lietuva, Melnkalne, Čehija, Dānija, Somija, Gruzija, Izraēla, Luksemburga, Malta un Serbija.",
"target_text": "Norvēģiju Eirovīzijas dziesmu konkursā pārstāv jaunais dziedātājs Kails Alessandro ( Kyle Alessandro ). Plašāka auditorija dziedātāju iepazina jau 10 gadu vecumā, kad viņš veiksmīgi piedalījās\xa0TV šovā \"Norway’s Got Talent\"."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:

  ```text
  Tālāk ir dokumenti ar pievienotām kopsavilkumiem.
  ```

- Base prompt template:

  ```text
  Dokuments: {text}
  Kopsavilkums: {target_text}
  ```

- Instruction-tuned prompt template:

  ```text
  Dokuments: {text}

  Uzrakstiet kopsavilkumu par iepriekš minēto dokumentu.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset lsm
```

## Hallucination Detection

### MultiWikiHalluQA-lv

This dataset uses the same data as [MultiWikiQA-lv](#multiwikiqa-lv), published in
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

  ```text
  Turpmāk seko teksti ar atbilstošiem jautājumiem un atbildēm.
  ```

- Base prompt template:

  ```text
  Teksts: {text}
  Jautājums: {question}
  Atbildēt ar maksimāli 3 vārdiem:
  ```

- Instruction-tuned prompt template:

  ```text
  Teksts: {text}

  Atbildiet uz šo jautājumu par iepriekš minēto tekstu ar maksimāli 3 vārdiem.

  Jautājums: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-hallucination-qa-lv
```
