# 🇸🇮 Slovene

This is an overview of all the datasets used in the Slovene part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### Sentinews

This dataset was published in
[this paper](https://link.springer.com/article/10.1007/s10579-018-9413-3) and consists
of five Slovene media resources on the web.

The original dataset contains 168,899 samples. We use 1,024 / 256 / 2,048 samples for
our training, validation and test splits, respectively.

Here are a few examples from the training split:

```json
{
  "text": "V državo pa je vpeljal stabilnost, katero je Rusija potrebovala.",
  "label": "positive"
}
```

```json
{
  "text": "Po najbolj črnogledih napovedih bo konec leta že sto tisoč brezposelnih.",
  "label": "negative"
}
```

```json
{
  "text": "Zdenko Pavček bo vložil zasebno tožbo zoper Walterja Wolfa zaradi kaznivega dejanja razžalitve.",
  "label": "negative"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Spodaj so dokumenti in njihov sentiment, ki je lahko 'pozitivno', 'nevtralno' ali 'negativno'.
  ```

- Base prompt template:

  ```text
  Dokument: {text}
  Sentiment: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Dokument: {text}

  Klasificirajte sentiment v dokumentu. Odgovorite z 'pozitivno', 'nevtralno' ali 'negativno', in nič drugega.
  ```

- Label mapping:
  - `positive` ➡️ `pozitivno`
  - `neutral` ➡️ `nevtralno`
  - `negative` ➡️ `negativno`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset sentinews
```

## Named Entity Recognition

### ssj500k-NER

This dataset was published in
[this paper](https://nl.ijs.si/jtdh20/pdf/JT-DH_2020_Krek-et-al_The-ssj500k-Training-Corpus-for-Slovene-Language-Processing.pdf),
and consists of a collection of text samples from the
[FidaPLUS](https://www.sketchengine.eu/fida-plus-corpus/) corpus of written modern
Slovene.

The original dataset consists of 9,489 samples. We use 1,024 / 256 / 2,048 samples for
our training, validation and test splits, respectively.

Here are a few examples from the training split:

```json
{
  "tokens": [
    "Prireditev",
    "Po",
    "domače",
    "pri",
    "Repanšku",
    "bo",
    "povezoval",
    "igralec",
    "in",
    "humorist",
    "Kondi",
    "Pižorn",
    ",",
    "za",
    "zabavo",
    "in",
    "ples",
    "pa",
    "bo",
    "letos",
    "igral",
    "ansambel",
    "Razpotniki",
    "."
  ],
  "labels": [
    "O",
    "B-MISC",
    "I-MISC",
    "I-MISC",
    "I-MISC",
    "O",
    "O",
    "O",
    "O",
    "O",
    "B-PER",
    "I-PER",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "B-ORG",
    "I-ORG",
    "O"
  ]
}
```

```json
{
  "tokens": [
    "Upoštevano",
    "je",
    ",",
    "da",
    "nekaj",
    "ljudi",
    "iz",
    "te",
    "bolnišnice",
    "odide",
    "drugam",
    ",",
    "nekaj",
    "pa",
    "jih",
    "pride",
    "iz",
    "drugih",
    "."
  ],
  "labels": [
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O"
  ]
}
```

```json
{
  "tokens": [
    "Ta",
    "ukazna",
    "vrstica",
    "obdela",
    "ali",
    "pošlje",
    "dokument",
    "v",
    "datoteko",
    ",",
    "ki",
    "se",
    "nahaja",
    "v",
    "imeniku",
    "/",
    "var",
    "/",
    "spool",
    "."
  ],
  "labels": [
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O"
  ]
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Naslednje so povedi in JSON slovarji z poimenovanimi entitetami, ki se pojavijo v dani povedi.
  ```

- Base prompt template:

  ```text
  Poved: {text}
  Poimenovane entitete: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Poved: {text}

  Identificirajte poimenovane entitete v povedi. To morate izpisati kot JSON slovar s ključi 'oseba', 'kraj', 'organizacija' in 'razno'. Vrednosti morajo biti seznami poimenovanih entitet te kategorije, tako kot se pojavijo v povedi.
  ```

- Label mapping:
  - `B-PER` ➡️ `oseba`
  - `I-PER` ➡️ `oseba`
  - `B-LOC` ➡️ `kraj`
  - `I-LOC` ➡️ `kraj`
  - `B-ORG` ➡️ `organizacija`
  - `I-ORG` ➡️ `organizacija`
  - `B-MISC` ➡️ `razno`
  - `I-MISC` ➡️ `razno`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ssj500k-ner
```

## Linguistic Acceptability

### ScaLA-sl

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the
[Slovene Universal Dependencies treebank](https://github.com/UniversalDependencies/UD_Slovene-SSJ)
by assuming that the documents in the treebank are correct, and corrupting the samples
to create grammatically incorrect samples. The corruptions were done by either removing
a word from a sentence, or by swapping two neighbouring words in a sentence. To ensure
that this does indeed break the grammaticality of the sentence, a set of rules were used
on the part-of-speech tags of the words in the sentence.

The original full dataset consists of 1,024 / 256 / 2,048 samples for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
used as-is in the framework.

Here are a few examples from the training split:

```json
{
  "text": "Potem je nekdo planil na sejo in povedal, da je v Trade Centru prišlo do eksplozije.",
  "label": "correct"
}
```

```json
{
  "text": "Miroslav Klun: S primerno zakonodajo lahko slovenska obrt ponudi 60.000 novih delovnih mest.",
  "label": "correct"
}
```

```json
{
  "text": "Priročno za vse, ki veliko kupujete drugih v državah.",
  "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Sledeče so stavki in ali so slovnično pravilni.
  ```

- Base prompt template:

  ```text
  Stavek: {text}
  Slovnično pravilno: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Stavek: {text}

  Ugotovite, ali je stavek slovnično pravilen ali ne. Odgovorite z 'da', če je stavek pravilen, in 'ne', če ni. Odgovorite le s to besedo in nič drugega.
  ```

- Label mapping:
  - `correct` ➡️ `da`
  - `incorrect` ➡️ `ne`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-sl
```

## Reading Comprehension

### MultiWikiQA-sl

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
  "context": "1891 (MDCCCXCI) je bilo navadno leto, ki se je po gregorijanskem koledarju začelo na četrtek, po 12 dni počasnejšem julijanskem koledarju pa na torek.\n\nDogodki \n 15. maj – papež Leon XIII. izda encikliko Rerum Novarum\n\nRojstva \n 22. januar - Antonio Gramsci, italijanski filozof, politik in politični teoretik († 1937)\n 24. januar - Abraham Samojlovič Bezikovič, ruski matematik († 1970)\n 11. marec - Michael Polanyi, madžarsko-britanski kemik in filozof († 1976)\n 24. marec - John Knittel, švicarski pisatelj († 1970)\n 14. april - Bhimrao Ramdži Ambedkar, indijski budistični socialni reformator, pravnik in filozof († 1956)\n 22. april - sir Harold Jeffreys, angleški geofizik, astronom, matematik († 1989)\n 23. april - Sergej Sergejevič Prokofjev, ruski skladatelj, pianist († 1953)\n 15. maj - Mihail Afanasjevič Bulgakov, ruski pisatelj († 1940)\n 18. maj - Rudolf Carnap, nemški filozof († 1970)\n 16. junij - Vladimir Aleksandrovič Albicki, ruski astronom († 1952)\n 19. avgust - Milton Lasell Humason, ameriški astronom († 1972)\n 26. september - Hans Reichenbach, nemški filozof († 1953)\n 20. oktober - sir James Chadwick, angleški fizik, nobelovec 1935 († 1974)\n 12. november - Seth Barnes Nicholson, ameriški astronom († 1963)\n 15. november - Erwin Rommel, nemški feldmaršal in vojaški strateg († 1944)\n 26. december - Henry Miller, ameriški pisatelj († 1980)\n\nSmrti \n 23. junij - Norman Robert Pogson, angleški astronom (* 1829)\n 23. junij - Wilhelm Eduard Weber nemški fizik (* 1804)\n 3. oktober - Édouard Lucas, francoski matematik (* 1842)\n 10. november - Štefan Žemlič, madžarsko-slovenski pisatelj (* 1840)\n 20. december - George Bassett Clark, ameriški astronom, optik (* 1827)\n 29. december - Leopold Kronecker, nemški matematik, logik (* 1823)",
  "question": "Kateri je bil prvi dan leta 1891 po gregorijanskem koledarju?",
  "answers": {
    "answer_start": [82],
    "text": ["na četrtek"]
  }
}
```

```json
{
  "context": "The Night the Light Went On in Long Beach je prvi album v živo skupine Electric Light Orchestra, ki je izšel leta 1974, posnet pa je bil 12. maja 1974 v Long Beach Auditoriumu na Long Beachu. Naslov albuma je sposojen od pesmi »The Night the Lights Went Out in Georgia«, ki jo je posnela Vicki Lawrence.\n\nOzadje in omejena izdaja \nAlbum je bil predviden kot naslednik albuma On the Third Day, a so bili posnetki poškodovani zaradi tehničnih težav tako na odru kot zunaj njega. Težave so se začele ko se je tovornjak z opremo skupine na poti pokvaril, pred koncertom pa skupina ni imela dovolj časa za preizkus zvoka.\n\nŠtevilna prešanja albuma so bila tako slabe kvalitete, da je vodstvo skupine vložilo tožbo proti proizvodnem podjetju. Naslovnico albuma je oblikoval Mick Haggerty. \n\nNa koncu sta se tako ameriška kot britanska založba odločili da ne izdata albuma. Album je tako izšel le v Nemčiji in nekaterih drugih državah, leta 1985 pa je vseeno izšel v Združenem kraljestvu. Album ni nikoli izšel v ZDA, čeprav je bil tja uvožen in se je dobro prodajal, je pa živa verzija »10538 Overture« izšla kot b-stran singla »Evil Woman« z albuma Face the Music. Živa verzija »Roll Over Beethoven« je v ZDA izšla kot b-stran alternativne verzije »Telephone Line« v seriji reizdaj.\n\nObnova \nRemastering v 90. letih je popravil slabo kvaliteto albuma. Odkrito je bilo, da je bila originalna LP plošča masterizirana z uporabo slabše kopije koncerta, zaradi katere je bila kvaliteta zvoka slaba. Originalni trak je bil odkrit v trezorju proizvodnje plošč in album je bil obnovljen v boljši kvaliteti zvoka.\n\nTo je edini živi album ELO iz časa začetkov skupine.\n\nSeznam skladb\n\nZasedba \nJeff Lynne\t– solo vokal, električna kitara\nBev Bevan – bobni\nRichard Tandy – Wurlitzer, Minimoog\nMike de Albuquerque – solo vokal, spremljevalni vokal, bas\nMik Kaminski – violina\nHugh McDowell – čelo\nMike Edwards – čelo\n\nSklici \n\nAlbumi leta 1974\nAlbumi Electric Light Orchestra\nAlbumi v živo\nAlbumi, ki jih je produciral Jeff Lynne",
  "question": "Zaradi česa je bila slaba kakovost albuma The Night the Light Went On in Long Beach odpravljena?",
  "answers": {
    "answer_start": [1287],
    "text": ["Remastering v 90. letih"]
  }
}
```

```json
{
  "context": "Surangel S. Whipps ml., palavski poslovnež in politik; * 9. avgust 1968, Baltimore, Maryland, Združene države Amerike.\n\nOd 21. januarja 2021 je predsednik Palava. Senator je bil od leta 2008 do 2016. Prihaja iz dežele Ngatpang.\n\nZgodnje življenje in izobraževanje \nWhipps se je rodil v Baltimorju v Marylandu materi samohranilki Surangel Whipps Sr., rojeni v Marylandu. Diplomiral je iz poslovne administracije in ekonomije na Univerzi Andrews in magistriral iz poslovne znanosti na kalifornijski univerzi v Los Angelesu. Poleg tega vodi verigo supermarketov Palauan. Na splošnih volitvah leta 2016 v Palavu se je potegoval proti svojemu zetu, predsedniku Thomasu Remengesauju mlajšemu.\n\nMandat \nWhipps je na predsedniških volitvah 2020 kandidiral za predsednika in premagal podpredsednika Raynolda Oiloucha. V intervjuju za Guardian je takratni izvoljeni predsednik Whipps ml. podal izjavo, da se bo Palav odločneje upiral ukrepom kitajske vlade, vključno z nezakonitim ribolovom in vstopom v palavske vode ter obljubil, da bo ohranil priznanje Tajvana. Poleg tega je predlagal distribucijo cepiva proti COVID-19 med Palavčani, s poudarkom na zdravstvenih delavcih.\n\nSklici \nWhipps, Surangel\nWhipps, Surangel",
  "question": "Proti komu je Whipps zmagal na predsedniških volitvah leta 2020?",
  "answers": {
    "answer_start": [790],
    "text": ["Raynolda Oiloucha"]
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Spodaj so besedila z ustreznimi vprašanji in odgovori.
  ```

- Base prompt template:

  ```text
  Besedilo: {text}
  Vprašanje: {question}
  Odgovor v največ 3 besedah:
  ```

- Instruction-tuned prompt template:

  ```text
  Besedilo: {text}

  Odgovorite na naslednje vprašanje o zgornjem besedilu v največ 3 besedah.

  Vprašanje: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-sl
```

## Knowledge

### MMLU-sl

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2410.08928)
and is a machine translated version of the English
[MMLU dataset](https://openreview.net/forum?id=d7KBjmI3GmQ) and features questions
within 57 different topics, such as elementary mathematics, US history and law. The
translation to Slovene was done using DeepL.

The original full dataset consists of 285 / 1,531 / 14,042 samples for training,
validation, and testing, respectively. These splits were merged, duplicates removed, and
new splits were created with 1,024 / 256 / 2048 samples for training, validation, and
testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Kaj je deklarativna teorija priznanja?\nMožnosti:\na. Priznanje je odločilno za obstoj državnosti\nb. Priznanje je zgolj deklarativno za državnost, ni pa odločilno\nc. Priznanje je zgolj izjava o interesu\nd. Priznanje zahteva izjavo novoustanovljene državnosti",
  "label": "b"
}
```

```json
{
  "text": "Katera od naslednjih možnosti bi bila verjeten odziv na ugotovljeno nenormalnost ostanka?\nMožnosti:\na. Uporabite logaritemsko funkcionalno obliko namesto linearne\nb. Dodajte zamike spremenljivk na desni strani regresijskega modela\nc. Ocenite model v prvi diferencirani obliki\nd. Iz podatkov odstranite vsa velika odstopanja.",
  "label": "d"
}
```

```json
{
  "text": "To vprašanje se nanaša na naslednje informacije. Stopnje pismenosti med rusko govorečim prebivalstvom pozne carske Rusije in Sovjetske zveze, 1897-1955 Stopnja pismenosti 1897 24% 1917 45% 1926 56% 1937 75% 1939 81.10% 1955 99.90% Vir: Podatki iz popisa prebivalstva in sovjetskega ministrstva za šolstvo Informacije, predstavljene v zgornji tabeli, je najbolje razumeti v katerem od naslednjih zgodovinskih kontekstov?\nMožnosti:\na. Izobraževalna reforma v moderni dobi\nb. Centralizirane in od države usmerjene kampanje modernizacije\nc. Eksperimentiranje s sindikalističnimi oblikami družbenoekonomske organizacije\nd. Druga faza industrializacije v nezahodnem svetu",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Naslednja so vprašanja z več možnostmi (z odgovori).
  ```

- Base prompt template:

  ```text
  Vprašanje: {text}
  Možnosti:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Odgovor: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Vprašanje: {text}

  Odgovorite na navedeno vprašanje z uporabo 'a', 'b', 'c' ali 'd', in nič drugega.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset mmlu-sl
```

### Unofficial: EU-MMLU-sl

This dataset is a human-translated subset of the English
[MMLU dataset](https://openreview.net/forum?id=d7KBjmI3GmQ), covering 7 of the original
57 subjects: college biology, college chemistry, college physics, global facts,
international law, management and sociology. Unlike the other MMLU variants in EuroEval
it was not machine translated - the translation was carried out by professional
translators at the European Commission's Directorate-General for Translation, together
with master's students from the European Master's in Translation network, as described
in [this paper](https://arxiv.org/abs/2607.18432).

The original English subset consists of 1,185 samples in total. We keep the original
MMLU splits rather than creating new ones, giving 34 / 91 / 859 samples for training,
validation and testing, respectively (so 984 samples in total). The translation is a
work in progress and not every subject has been translated into every language yet, so
the splits are smaller for some languages than for others.

Here are a few examples from the training split:

```json
{
  "text": "Napovejte število spektralnih črt v spektru EPR raztopine izotopno (13C) označenega metilnega radikala (13CH3•), ob predpostavki, da se črte ne prekrivajo.\nMožnosti:\na. 4\nb. 3\nc. 6\nd. 24",
  "label": "a"
}
```

```json
{
  "text": "Kolikšen delež Američanov se leta 2019 strinja, da se pri vodenju države gleda na korist vseh ljudi?\nMožnosti:\na. 31 %\nb. 46 %\nc. 61 %\nd. 76 %",
  "label": "b"
}
```

```json
{
  "text": "V določeni populaciji ima eden od 400 ljudi raka, ki ga povzroča popolnoma recesivni alel b. Ob predpostavki, da je populacija v Hardy-Weinbergovem ravnovesju, kateri od naslednjih je pričakovani delež posameznikov, ki so nosilci alela b, vendar se pri njih ne pričakuje razvoj raka?\nMožnosti:\na. 1/400\nb. 19/400\nc. 20/400\nd. 38/400",
  "label": "d"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5

- Prefix prompt:

  ```text
  Naslednja so vprašanja z več možnostmi (z odgovori).
  ```

- Base prompt template:

  ```text
  Vprašanje: {text}
  Odgovor: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Vprašanje: {text}

  Odgovorite na navedeno vprašanje z uporabo {labels_str}, in nič drugega.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset eu-mmlu-sl
```

## Common-sense Reasoning

### Winogrande-sl

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2506.19468)
and is a translated and filtered version of the English
[Winogrande dataset](https://doi.org/10.1145/3474381).

The original full dataset consists of 47 / 1,210 samples for training and testing, and
we use 128 of the test samples for validation, resulting in a 47 / 128 / 1,085 split for
training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Nisem mogel nadzorovati vlage, kot sem nadzoroval dež, ker je _ prihajal od vsepovsod. Na kaj se nanaša prazno mesto _?\nMožnosti:\na. vlaga\nb. dež",
  "label": "a"
}
```

```json
{
  "text": "Jessica je mislila, da je Sandstorm najboljša pesem, kar jih je bilo kdaj napisanih, vendar jo je Patricia sovražila. _ je kupila vstopnico za jazz koncert. Na kaj se nanaša prazno mesto _?\nMožnosti:\na. Jessica\nb. Patricia",
  "label": "b"
}
```

```json
{
  "text": "Termostat je pokazal, da je bilo spodaj dvajset stopinj hladneje kot zgoraj, zato je Byron ostal v _, ker mu je bilo hladno. Na kaj se nanaša prazno mesto _?\nMožnosti:\na. spodaj\nb. zgoraj",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Naslednja so vprašanja z več možnostmi (z odgovori).
  ```

- Base prompt template:

  ```text
  Vprašanje: {text}
  Možnosti:
  a. {option_a}
  b. {option_b}
  Odgovor: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Vprašanje: {text}
  Možnosti:
  a. {option_a}
  b. {option_b}

  Odgovorite na navedeno vprašanje z uporabo 'a' ali 'b', in nič drugega.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset winogrande-sl
```

## Instruction-following

### MultiIFEval-sl

MultiIFEval-sl is part of the MultiIFEval benchmark spanning 305 languages. It is
generated by translating and localising the English IFEval dataset using a structured
LLM generation pipeline. For each target language, a randomly selected Wikipedia article
in that language provides contextual grounding to reduce hallucination and improve
cultural localisation. The pipeline preserves instruction_id_list values for
traceability to the original English samples, and retains kwargs keys with values
localised where appropriate, enabling programmatic constraint verification. The dataset
was published [here](https://huggingface.co/datasets/EuroEval/multi-ifeval-sl).

This dataset is part of the MultiIFEval benchmark introduced in
[this draft paper](https://raw.githubusercontent.com/alexandrainst/multi_ifeval/refs/heads/feat/add-paper/paper/acl_latex.tex).

We use the dataset as the test split, and do not include other splits, as we only
evaluate models zero-shot and the size is too small to warrant a validation set.

Here are a few examples from the test split:

```json
{
  "text": "Napišite povzetek članka iz Wikipedije \"https://sl.wikipedia.org/wiki/France_Prešeren\" z vsaj 300 besedami. Ne uporabljajte vejic in poudarite vsaj 3 razdelke z naslovi v formatu markdown, na primer *poudarjeni razdelek 1. del*, *poudarjeni razdelek 2. del*, *poudarjeni razdelek 3. del*.",
  "target_text": {
    "instruction_id_list": [
      "punctuation:no_comma",
      "detectable_format:number_highlighted_sections",
      "length_constraints:number_words"
    ],
    "kwargs": [
      {},
      { "num_highlights": 3 },
      { "num_words": 300, "relation": "at least" }
    ]
  }
}
```

```json
{
  "text": "Načrtujem potovanje na Japonsko in želim, da napišete načrt potovanja v Shakespearovem slogu. V svojem odgovoru ne smete uporabljati vejic.",
  "target_text": {
    "instruction_id_list": ["punctuation:no_comma"],
    "kwargs": [{}]
  }
}
```

```json
{
  "text": "Napišite življenjepis za osebo, ki je pravkar končala srednjo šolo in si išče prvo zaposlitev. Prepričajte se, da vključuje vsaj 12 nadomestnih besed, predstavljenih z oglatimi oklepaji, kot so [naslov], [ime].",
  "target_text": {
    "instruction_id_list": ["detectable_content:number_placeholders"],
    "kwargs": [{ "num_placeholders": 12 }]
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 0
- No prefix prompt, as only instruction-tuned models are evaluated on this task.
- No base prompt template, as only instruction-tuned models are evaluated on this task.
- Instruction-tuned prompt template:

  ```text
  {text}
  ```

  I.e., we just use the instruction directly as the prompt.

You can evaluate a model on this dataset as follows:

```bash
euroeval --model <model-id> --dataset multi-ifeval-sl
```

## Hallucination Detection

### RAGTruth-sl

This dataset is a Slovene translation of the
[RAGTruth](https://aclanthology.org/2024.acl-long.585/) hallucination benchmark, which
contains retrieval-augmented generation (RAG) prompts together with model-generated
answers annotated for hallucinations. Rather than evaluating the correctness of the
generated answer, this task evaluates the degree to which the model hallucinates, i.e.,
generates tokens that are not grounded in the provided context.

The hallucination detection is performed using the
[LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect) library, which uses a
[transformer-based classifier](https://arxiv.org/abs/2605.02504) to predict
hallucination at the token level. The metric reported is the hallucination rate,
computed as the ratio of hallucinated tokens to total tokens in the generated answers.

Here are a few examples from the test split:

```json
{
  "prompt": "Povzemite naslednje novice v 116 besedah:\nPred sedemdesetimi leti je Anne Frank umrla za tifusom v nacističnem koncentracijskem taborišču pri 15 letih. Le dva tedna po njeni domnevni smrti 31. marca 1945 je bilo osvobojeno koncentracijsko taborišče Bergen-Belsen, kjer je bila zaprta – časovni okvir, ki je pokazal, kako blizu je bila judovska diaristka preživetju holokavsta. Vendar nova raziskava, ki jo je objavil Muzej Anne Frank, kaže, da sta Anne in njena starejša sestra Margot Frank umrli vsaj mesec dni prej, kot se je doslej mislilo. Raziskovalci so ponovno pregledali arhive Rdečega križa, Mednarodne usposabljalne službe in spominskega centra Bergen-Belsen ter pričevanja preživelih. Ugotovili so, da Anne in Margot verjetno nista preživeli do marca 1945 – kar nasprotuje datumu smrti, ki so ga prej določile nizozemske oblasti. Leta 1944 so bili Anne in še sedem drugih, ki so se skrivali v amsterdamskem tajnem aneksu, aretirani in poslani v koncentracijsko taborišče Auschwitz-Birkenau. Zadnji vnos Anne Frank. Istega leta sta bili Anne in Margot ločeni od matere in poslani na prisilno delo v taborišče Bergen-Belsen v Nemčiji. Dnevi v taborišču so bili polni terorja in strahu, so povedali priče. Sestre so bile v delu prenatrpanega taborišča brez osvetlitve, z malo vode in brez latrina. Spali so na slami, okuženi z uši, in nasilne nevihte so trgale šotore, so ugotovili raziskovalci. Tako kot drugi zaporniki so sestre prenašale dolge ure na zboru. Njena sošolka, Nannette Blitz, se je spomnila, da je Anne tam videla decembra 1944: \"Do takrat ni bila več kot okostje. Bila je zavita v odejo; oblačil ni mogla nositi, ker so bila prepredena z ušmi.\" Prisluhnite prijateljem Anne Frank, ki opisujejo njeno izkušnjo v koncentracijskem taborišču. Ko so Rusi napredovali, je postalo taborišče Bergen-Belsen še bolj prenatrpano, kar je prineslo več bolezni. Smrtonosna epidemija tifusa je povzročila, da je vsak dan umrlo na tisoče ljudi. Tifus je nalezljiva bolezen, ki jo povzročajo uši in se pojavi na mestih s slabo higieno. Bolezen povzroča visoko vročino, mrzlico in kožne izpuščaje. \"Zaradi uši, ki so napadle njeno posteljnino in oblačila, je bila Anne dalj časa izpostavljena glavnemu prenašalcu epidemskega tifusa,\" so zapisali raziskovalci muzeja. Ugotovili so, da je malo verjetno, da bi sestre preživele do marca, saj so priče v taborišču povedale, da sta imeli obe simptome pred 7. februarjem. \"Večina smrti, ki jih povzroči tifus, se zgodi približno dvanajst dni po pojavu prvih simptomov,\" sta zapisala avtorja Erika Prins in Gertjan Broek. Natančni datumi smrti Anne in Margot ostajajo nejasni. Margot je umrla pred Anne. \"Anne nikoli ni obupala,\" je dejala Blitz, njena prijateljica. \"Bila je popolnoma prepričana, da bo preživela.\" Njen dnevnik ostaja ena najbolj priljubljenih knjig na svetu. Preberite več o Anne Frankinih bratrancih, skrbnikih njenega zapuščine."
}
```

```json
{
  "prompt": "Povzemite naslednje novice v 116 besedah:\nPred sedemdesetimi leti je Anne Frank umrla za tifusom v nacističnem koncentracijskem taborišču pri petnajstih letih. Le dva tedna po njeni domnevni smrti 31. marca 1945 je bilo taborišče Bergen-Belsen, kjer je bila zaprta, osvobojeno -- časovno usklajeno, kar je pokazalo, kako blizu je bila judovska diaristka preživetju holokavsta. Toda nova raziskava, ki jo je objavil Muzej Anne Frank, kaže, da sta Anne in njena starejša sestra Margot Frank umrli vsaj mesec dni prej, kot se je doslej menilo. Raziskovalci so ponovno pregledali arhive Rdečega križa, Mednarodne usposabljalne službe in spominskega centra Bergen-Belsen, skupaj s pričevanji preživelih. Sklenili so, da Anne in Margot verjetno nista preživeli do marca 1945 -- kar nasprotuje datumu smrti, ki so ga prej določile nizozemske oblasti. Leta 1944 so bili Anne in še sedem drugih, ki so se skrivali v amsterdamskem tajnem annexu, aretirani in poslani v koncentracijsko taborišče Auschwitz-Birkenau. Zadnji vnos Anne Frank. Istega leta sta bili Anne in Margot ločeni od matere in poslani na prisilno delo v taborišče Bergen-Belsen v Nemčiji. Dnevi v taborišču so bili polni terorja in groze, so povedali priče. Sestre so ostale v delu prenatrpanega taborišča brez osvetlitve, z malo vode in brez latrina. Spali so na slami, okuženi z uši, in nasilne nevihte so raztrgale šotore, so ugotovili raziskovalci. Tako kot drugi zaporniki so sestre prenašale dolge ure pri pregledu. Njena sošolka, Nannette Blitz, se je spomnila, da je Anne tam videla decembra 1944: \"Do takrat je bila le še skelet. Bila je ovita v odejo; oblačil ni mogla nositi več, ker so bila polna uši.\" Poslušajte prijatelje Anne Frank, ki opisujejo njeno izkušnjo v koncentracijskem taborišču. Ko so Rusi napredovali, je postalo taborišče Bergen-Belsen še bolj prenatrpano, kar je prineslo več bolezni. Smrtonosna epidemija tifusa je povzročila, da je vsak dan umrlo na tisoče ljudi. Tifus je nalezljiva bolezen, ki jo prenašajo uši in izbruhne na mestih s slabo higieno. Bolezen povzroča visoko vročino, mrzlico in kožne izpuščaje. \"Zaradi uši, ki so napadle slamo in njena oblačila, je bila Anne dalj časa izpostavljena glavnemu prenašalcu epidemskega tifusa,\" so zapisali raziskovalci muzeja. Sklenili so, da je malo verjetno, da sta sestro preživeli do marca, saj so priče v taborišču povedale, da sta obe sestro imeli simptome pred 7. februarjem. \"Večina smrti zaradi tifusa se zgodi približno dvanajst dni po prvih simptomih,\" sta napisala avtorja Erika Prins in Gertjan Broek. Natančni datumi smrti za Anne in Margot ostajajo nejasni. Margot je umrla pred Anne. \"Anne nikoli ni obupala,\" je dejala Blitz, njena prijateljica. \"Bila je prepričana, da bo preživela.\" Njen dnevnik ostaja ena najbolj priljubljenih knjig na svetu. Preberite več o Anninem bratrancu, skrbniku njenega zapuščine.\n\noutput:"
}
```

```json
{
  "prompt": "Povzemite naslednje novice v 116 besedah:\nPred sedemdesetimi leti je Anne Frank umrla za tifusom v nacističnem koncentracijskem taborišču pri 15 letih. Le dva tedna po njeni domnevni smrti 31. marca 1945 je bilo taborišče Bergen-Belsen, kjer je bila zaprta, osvobojeno -- časovni okvir, ki je pokazal, kako blizu je bila judovska dnevniška avtorica preživetju holokavsta. Vendar nove raziskave, ki jih je objavil Muzej Anne Frank, kažejo, da sta Anne in njena starejša sestra Margot Frank umrli vsaj mesec dni prej, kot se je prej mislilo. Raziskovalci so ponovno pregledali arhive Rdečega križa, Mednarodne usposabljalne službe in spomenika Bergen-Belsen ter priče preživelih. Sklenili so, da Anne in Margot verjetno nista preživeli do marca 1945 -- kar nasprotuje datumu smrti, ki so ga prej določile nizozemske oblasti. Leta 1944 so aretirali Anne in še sedem drugih, ki so se skrivali v amsterdamskem tajnem podstrešju, in jih poslali v koncentracijsko taborišče Auschwitz-Birkenau. Zadnji vnos Anne Frank. Istega leta sta bili Anne in Margot ločeni od matere in odpeljani na prisilno delo v taborišče Bergen-Belsen v Nemčiji. Dnevi v taborišču so bili polni terorja in strahu, so povedali priče. Sestre so ostale v delu prenatrpanega taborišča brez osvetlitve, z malo vode in brez latrina. Spali so na slami, prežeti z uši, in nasilne nevihte so raztrgale šotore, pravijo raziskovalci. Tako kot drugi zaporniki so sestre prenašale dolge ure na zboru. Njena sošolka, Nannette Blitz, se je spomnila, da je Anne tam videla decembra 1944: \"Takrat ni bila več kot okostnjak. Bila je zavita v odejo; oblačil ni mogla nositi več, ker so bila prežeta z ušmi.\" Poslušajte prijatelje Anne Frank, ki opisujejo njeno izkušnjo v koncentracijskem taborišču. Ko so Rusi napredovali, je postalo taborišče Bergen-Belsen še bolj prenatrpano, kar je prineslo več bolezni. Smrtonosna epidemija tifusa je povzročila smrt tisočev vsak dan. Tifus je nalezljiva bolezen, ki jo povzročajo uši in izbruhne na mestih s slabo higieno. Bolezen povzroča visoko vročino, mrzlico in kožne izpuščaje. \"Zaradi uši, ki so napadle posteljnino in njena oblačila, je bila Anne dalj časa izpostavljena glavnemu prenašalcu epidemskega tifusa,\" so zapisali raziskovalci muzeja. Sklenili so, da je malo verjetno, da bi sestre preživele do marca, ker so priče v taborišču povedale, da sta imeli sestre simptome pred 7. februarjem. \"Večina smrti zaradi tifusa se zgodi približno dvanajst dni po pojavu prvih simptomov,\" sta zapisala avtorja Erika Prins in Gertjan Broek. Natančni datumi smrti za Anne in Margot ostajajo nejasni. Margot je umrla pred Anne. \"Anne nikoli ni obupala,\" je rekla Blitz, njena prijateljica. \"Bila je prepričana, da bo preživela.\" Njen dnevnik ostaja ena najbolj priljubljenih knjig na svetu. Preberite več o Anne Frankinih bratrancih, ki ohranja njen dediščino.\n\noutput:"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information):

- Number of few-shot examples: 0 (zero-shot only)
- Instruction prompt:

  ```text
  {prompt}
  ```

  I.e., we just use the instruction directly as the prompt.

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ragtruth-sl
```

## Translation

### WMT24++ English to Slovene

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2502.12404)
and is an extension of the original WMT24 dataset. It contains manually translated
examples for the English-Slovene translation pair.

The original full dataset consists of 998 samples for this language pair. A small
portion of the samples were marked as bad, however, and we exclude those. We use 64
samples for the training split, 128 samples for the validation split, and the rest for
the test split.

We use the Slovene translation pair from the dataset, where the source text is in
English and the target text is in Slovene.

Here are a few examples from the training split:

```json
{
  "text": "Hello, how are you?",
  "target_text": "Živjo, kako si?"
}
```

```json
{
  "text": "The brown fox jumps over the lazy dog.",
  "target_text": "Rjava lisica skače čez lenega psa."
}
```

```json
{
  "text": "Ljubljana is the capital of Slovenia.",
  "target_text": "Ljubljana je glavno mesto Slovenije."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  The following are English texts with corresponding Slovene translations.
  ```

- Base prompt template:

  ```text
  English text: {text}
  Slovene translation: {target_text}
  ```

- Instruction-tuned prompt template:

  ```text
  English text: {text}

  Translate the above text into Slovene.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset wmt24pp-en-sl
```

### WMT24++ Slovene to English

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2502.12404)
and is an extension of the original WMT24 dataset. It contains manually translated
examples for the English-Slovene translation pair.

The original full dataset consists of 998 samples for this language pair. A small
portion of the samples were marked as bad, however, and we exclude those. We use 64
samples for the training split, 128 samples for the validation split, and the rest for
the test split.

We use the Slovene translation pair from the dataset, where the source text is in
Slovene and the target text is in English.

Here are a few examples from the training split:

```json
{
  "text": "Živjo, kako si?",
  "target_text": "Hello, how are you?"
}
```

```json
{
  "text": "Rjava lisica skače čez lenega psa.",
  "target_text": "The brown fox jumps over the lazy dog."
}
```

```json
{
  "text": "Ljubljana je glavno mesto Slovenije.",
  "target_text": "Ljubljana is the capital of Slovenia."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Spodaj so besedila v slovenščini z ustreznimi prevodi v English.
  ```

- Base prompt template:

  ```text
  Besedilo v slovenščini: {text}
  Prevod v English: {target_text}
  ```

- Instruction-tuned prompt template:

  ```text
  Besedilo v slovenščini: {text}

  Prevedite zgornje besedilo v English.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset wmt24pp-sl-en
```

### Unofficial: FLORES+ English to Slovene

This dataset is part of the FLORES+ benchmark, maintained by the Open Language Data
Initiative (OLDI), which extends the original FLORES-200 dataset. It consists of
sentences sampled from English Wikimedia articles (Wikinews, Wikijunior and Wikivoyage)
and professionally translated into each language, with every sentence translated in a
multi-way parallel fashion.

We use 128 / 256 / 1,012 samples for the training, validation and test splits,
respectively. The training and validation splits are sampled from the FLORES+ `dev`
split, and the test split is the entire FLORES+ `devtest` split.

We use the English-to-Slovene direction, where the source text is in English and the
target text is in Slovene.

Here are a few examples from the training split:

```json
{
  "text": "Given how remote many of the pueblos are, you won't be able to find a significant amount of nightlife without traveling to Albuquerque or Santa Fe.",
  "target_text": "Ker so nekateri pueblosi tako odročni, omembe vrednega nočnega življenja ne boste našli drugje kot v Albuquerqueju ali Santa Feju."
}
```

```json
{
  "text": "For instance, children who identify with a racial minority that is stereotyped as not doing well in school tend to not do well in school once they learn about the stereotype associated with their race.",
  "target_text": "Denimo, otroci, ki se poistovetijo z rasno manjšino, ki se stereotipno v šoli ne odreže najbolje, so v šoli pogosto manj uspešni, potem ko slišijo za stereotip, povezan z njihovo raso."
}
```

```json
{
  "text": "It's compacted snow with crevasses filled in and marked by flags. It can only be traveled by specialized tractors, hauling sleds with fuel and supplies.",
  "target_text": "Gre za utrjen sneg z razpokami, ki so zapolnjene in označene z zastavami. Cesta je prevozna samo s posebnimi traktorji, ki za sabo vlečejo gorivo in zaloge."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  The following are English texts with corresponding Slovene translations.
  ```

- Base prompt template:

  ```text
  English text: {text}
  Slovene translation: {target_text}
  ```

- Instruction-tuned prompt template:

  ```text
  English text: {text}

  Translate the above text into Slovene.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset flores-en-sl
```

### Unofficial: FLORES+ Slovene to English

This dataset is part of the FLORES+ benchmark, maintained by the Open Language Data
Initiative (OLDI), which extends the original FLORES-200 dataset. It consists of
sentences sampled from English Wikimedia articles (Wikinews, Wikijunior and Wikivoyage)
and professionally translated into each language, with every sentence translated in a
multi-way parallel fashion.

We use 128 / 256 / 1,012 samples for the training, validation and test splits,
respectively. The training and validation splits are sampled from the FLORES+ `dev`
split, and the test split is the entire FLORES+ `devtest` split.

We use the Slovene-to-English direction, where the source text is in Slovene and the
target text is in English.

Here are a few examples from the training split:

```json
{
  "text": "Ker so nekateri pueblosi tako odročni, omembe vrednega nočnega življenja ne boste našli drugje kot v Albuquerqueju ali Santa Feju.",
  "target_text": "Given how remote many of the pueblos are, you won't be able to find a significant amount of nightlife without traveling to Albuquerque or Santa Fe."
}
```

```json
{
  "text": "Denimo, otroci, ki se poistovetijo z rasno manjšino, ki se stereotipno v šoli ne odreže najbolje, so v šoli pogosto manj uspešni, potem ko slišijo za stereotip, povezan z njihovo raso.",
  "target_text": "For instance, children who identify with a racial minority that is stereotyped as not doing well in school tend to not do well in school once they learn about the stereotype associated with their race."
}
```

```json
{
  "text": "Gre za utrjen sneg z razpokami, ki so zapolnjene in označene z zastavami. Cesta je prevozna samo s posebnimi traktorji, ki za sabo vlečejo gorivo in zaloge.",
  "target_text": "It's compacted snow with crevasses filled in and marked by flags. It can only be traveled by specialized tractors, hauling sleds with fuel and supplies."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Spodaj so besedila v slovenščini z ustreznimi prevodi v English.
  ```

- Base prompt template:

  ```text
  Besedilo v slovenščini: {text}
  Prevod v English: {target_text}
  ```

- Instruction-tuned prompt template:

  ```text
  Besedilo v slovenščini: {text}

  Prevedite zgornje besedilo v English.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset flores-sl-en
```
