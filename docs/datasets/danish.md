# 🇩🇰 Danish

This is an overview of all the datasets used in the Danish part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.


## Sentiment Classification

### Angry Tweeets

This dataset was published in [this
paper](https://aclanthology.org/2021.nodalida-main.53/) and was a crowd-sourcing effort
to annotate sentiment of Danish tweets.

The original full dataset consists of 3,458 samples, and we are using a split of 1,024 /
256 / 2,048 samples for training, validation and testing, respectively (so 3,328 samples
used in total). All the samples in the original test set are included in our test set,
but our test set is furthermore using a subset of the original training set as test
samples as well. The original dataset did not have a validation split, so we have
created one by sampling from the training set.

Here are a few examples from the training split:

```json
{
  "text": "Jeg tror, det der var kampen. Goff virker lost",
  "label": "negative"
}
```
```json
{
  "text": "@USER @USER Vi bruger også snildt 1-2 timer (nogle gange flere timer end det) på at putte den yngste. Det er oftest Tommi, som gør det, for jeg går helt amok i processen. Så smører jeg madpakker og rydder op i stedet.",
  "label": "neutral"
}
```
```json
{
  "text": "Er du nysgerrig på, hvordan du diskvalificerer dig selv fra at blive taget seriøst i den offentlige debat? Naser har svaret. #dkpol #dkmedier [LINK]",
  "label": "negative"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Følgende er tweets og deres sentiment, som kan være 'positiv', 'neutral' eller 'negativ'.
  ```
- Base prompt template:
  ```
  Tweet: {text}
  Sentiment: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Tweet: {text}

  Klassificer sentimentet i tweetet. Svar kun med 'positiv', 'neutral' eller 'negativ'.
  ```
- Label mapping:
    - `positive` ➡️ `positiv`
    - `neutral` ➡️ `neutral`
    - `negative` ➡️ `negativ`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset angry-tweeets
```


## Named Entity Recognition

### DANSK

This dataset was published in [this
paper](https://doi.org/10.3384/nejlt.2000-1533.2024.5249) and is a manually annotated
subset of [Danish Gigaword](https://aclanthology.org/2021.nodalida-main.46/) with the 18
different named entities, following the OntoNotes 5.0 scheme. It was annotated by 10
different annotators.

The original full dataset consists of 15,062 samples, and we are using a split of 1,024
/ 256 / 1,024 samples for training, validation and testing, respectively (so 2,304
samples used in total). All samples in the validation and test sets of our version also
belong to the original validation and test set, respectively.

We have furthermore converted the OntoNotes 5.0 labelling scheme to the CoNLL-2003
labelling scheme, which is more common in the NER literature. The mapping is as follows:

- `PERSON` ➡️ `PER`
- `LOCATION` ➡️ `LOC`
- `FACILITY` ➡️ `LOC`
- `GPE` ➡️ `LOC`
- `ORGANIZATION` ➡️ `PER`
- `EVENT` ➡️ `MISC`
- `LANGUAGE` ➡️ `MISC`
- `PRODUCT` ➡️ `MISC`
- `WORK OF ART` ➡️ `MISC`
- `NORP` ➡️ `MISC`
- `CARDINAL` ➡️ `O`
- `DATE` ➡️ `O`
- `LAW` ➡️ `O`
- `MONEY` ➡️ `O`
- `ORDINAL` ➡️ `O`
- `PERCENT` ➡️ `O`
- `QUANTITY` ➡️ `O`
- `TIME` ➡️ `O`

Here are a few examples from the training split:

```json
{
  "tokens": array(['I', 'dette', 'efterår', 'har', 'Grønland', 'taget', 'en', 'stor', 'beslutning', 'ved', 'folkeafstemningen', 'den', '25.', 'november', '.'], dtype=object),
  "labels": array(['O', 'O', 'O', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'O', 'O', 'O', 'O'], dtype=object)
}
```
```json
{
  "tokens": array(['Åh', ',', 'Petra', ',', 'vis', 'mig', 'din', 'krop', '.'], dtype=object),
  "labels": array(['O', 'O', 'B-PER', 'O', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
```
```json
{
  "tokens": array(['Fravalget', 'af', 'revision', 'registreres', 'automatisk', 'ved', 'anmeldelse', 'af', 'stiftelse', 'af', 'selskabet', 'hos', 'Erhvervs-styrelsen', '.'], dtype=object),
  "labels": array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'O'], dtype=object)
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:
  ```
  Følgende er sætninger og JSON-ordbøger med de navngivne enheder, som forekommer i den givne sætning.
  ```
- Base prompt template:
  ```
  Sætning: {text}
  Navngivne enheder: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Sætning: {text}

  Identificér de navngivne enheder i sætningen. Du skal outputte dette som en JSON-ordbog med nøglerne 'person', 'sted', 'organisation' og 'diverse'. Værdierne skal være lister over de navngivne enheder af den type, præcis som de forekommer i sætningen.
  ```
- Label mapping:
    - `B-PER` ➡️ `person`
    - `I-PER` ➡️ `person`
    - `B-LOC` ➡️ `sted`
    - `I-LOC` ➡️ `sted`
    - `B-ORG` ➡️ `organisation`
    - `I-ORG` ➡️ `organisation`
    - `B-MISC` ➡️ `diverse`
    - `I-MISC` ➡️ `diverse`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset dansk
```


### Unofficial: DaNE

This dataset was published in [this paper](https://aclanthology.org/2020.lrec-1.565/)
and is a manually NER annotated version of the [Danish Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Danish-DDT/tree/master). The NER
labels follow the CoNLL-2003 labelling scheme.

The original full dataset consists of 4,383 / 564 / 565 samples for training, validation
and testing, respectively. We use a 1,024 / 256 / 2,048 split for training, validation
and testing, respectively (so 3,328 samples used in total). These splits are new and
there can thus be some overlap between the original validation and test sets and our
validation and test sets.

Here are a few examples from the training split:

```json
{
  "tokens": array(['Det', 'var', 'det', 'år', ',', 'hans', 'første', 'LP', ',', '"', 'With', 'A', 'Little', 'Help', 'From', 'My', 'Friends', '"', ',', 'udkom', '.'], dtype=object),
  "labels": array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O', 'O', 'O', 'O'], dtype=object)
}
```
```json
{
  "tokens": array(['Eddie', 'Carbone', ',', 'italiensk-amerikansk', 'havnearbejder', 'i', 'New', 'York', '.'], dtype=object),
  "labels": array(['B-PER', 'I-PER', 'O', 'B-MISC', 'O', 'O', 'B-LOC', 'I-LOC', 'O'], dtype=object)
}
```
```json
{
  "tokens": array(['"', 'Jeg', 'er', 'mig', '!', '"', 'insisterer', 'han', 'under', 'det', 'flere', 'hundrede', 'år', 'gamle', 'egetræ', ',', 'liggende', ',', 'som', 'den', 'popflab', 'han', 'er', ',', 'på', 'ryggen', 'i', 'sine', 'orange', 'jeans', ',', 't-shirt', '-', 'som', 'naturligvis', 'stiller', 'et', 'solbrunt', 'behåret', 'bryst', 'til', 'skue', '-', 'et', 'par', '68er', '"', 'make', 'love', 'not', 'war', '"', 'solbriller', 'han', 'netop', 'har', 'købt', 'i', 'Paris', ',', 'og', 'en', 'Kings', 'i', 'kæften', '.'], dtype=object),
  "labels": array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'O', 'O', 'B-MISC', 'O', 'O', 'O'], dtype=object)
}
```


## Linguistic Acceptability

### ScaLA-da

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Danish Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Danish-DDT/tree/master) by
assuming that the documents in the treebank are correct, and corrupting the samples to
create grammatically incorrect samples. The corruptions were done by either removing a
word from a sentence, or by swapping two neighbouring words in a sentence. To ensure
that this does indeed break the grammaticality of the sentence, a set of rules were used
on the part-of-speech tags of the words in the sentence.

The original dataset consists of 5,512 samples, from which we use 1,024 / 256 / 2,048 samples for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
used as-is in the framework.

Here are a few examples from the training split:

```json
{
  "text": "Samme dame dukkede netop nu op sammen med Odd-Catla's erklærede yndling, væbneren Aikin af Cantir.",
  "label": "correct"
}
```
```json
{
  "text": "Gebyrets størrelse afhænger nemlig af helt, i hvilken kategori den pågældende \"levnedsmiddelvirksomhed\" placeres.",
  "label": "incorrect"
}
```
```json
{
  "text": "Den statsansatte dyrlæge Kronfågels på slagteri i Kristiansstad, Karl Erik Bjørkman, understreger, belægningen hos producenten betyder meget for dyrenes trivsel:",
  "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Følgende er sætninger og om de er grammatisk korrekte.
  ```
- Base prompt template:
  ```
  Sætning: {text}
  Grammatisk korrekt: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Sætning: {text}

  Bestem om sætningen er grammatisk korrekt eller ej. Svar med 'ja', hvis sætningen er korrekt, og 'nej', hvis den ikke er.
  ```
- Label mapping:
    - `correct` ➡️ `ja`
    - `incorrect` ➡️ `nej`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset scala-da
```


## Reading Comprehension

### ScandiQA-da

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the Danish part of the [MKQA
dataset](https://aclanthology.org/2021.tacl-1.82/). The MKQA dataset is based on the
English [Natural Questions dataset](https://aclanthology.org/Q19-1026/), based on search
queries from the Google search engine. The questions and answers were manually
translated to Danish (and other languages) as part of MKQA, and the contexts were in
ScandiQA-da machine translated using the [DeepL translation
API](https://www.deepl.com/en/pro-api/). A rule-based approach was used to ensure that
the translated contexts still contained the answer to the question, potentially by
changing the answers slightly.

The original full dataset consists of 6,810 / 500 / 500 samples for training,
validation and testing, respectively (so 3,328 samples used in total).
We use a 1,024 / 256 / 2,048 split for training, validation and testing, respectively,
where the splits are made by randomly sampling from the full dataset without considering
the original train/validation/test splits.

Here are a few examples from the training split:

```json
{
  "context": '"(Sittin\' On) The Dock of the Bay" er en sang, der er skrevet af soul-sangeren Otis Redding og guitaristen Steve Cropper sammen. Den blev indspillet af Redding to gange i 1967, herunder en gang få dage før hans død i et flystyrt. Sangen blev udgivet på Stax Records\' Volt-label i 1968 og blev den første posthume single, der lå øverst på hitlisterne i USA. Den nåede op som nummer 3 på den britiske single-liste.',
  "question": 'Hvem sang sitting on the dock of the bay?',
  "answers": {
    "answer_start": array([79]),
    "text": array(['Otis Redding'], dtype=object)
  }
}
```
```json
{
  "context": "The Cat in the Hat Knows a Lot About That!\nKatten i hatten ved meget om det!\n\n\n\nKatten i hatten pilot\n\n\n\nGenre\nBørne-tv/undervisning/komedie\n\n\nInstrueret af\nTony Collingwood\n\n\nStemmer fra\nMartin Short\nJacob Ewaniuk\nAlexa Torrington\nRob Tinkler\n\n\nKomponist af temamusik\nDavid Schweitzer\n\n\nKomponist(er)\nDavid Schweitzer\n\n\nOprindelsesland\nCanada\nDet Forenede Kongerige\nUSA\n\n\nOprindelige sprog\nEngelsk\n\n\nAntal sæsoner\n2\n\n\nAntal episoder\n60 (liste over episoder)\n\n\nProduktion\n\n\nLøbetid\n30 minutter\n\n\nProduktionsselskab(er)\nCollingwood O'Hare Productions\nPortfolio Entertainment\nRandom House Children's Entertainment\nTreehouse TV\n\n\nDistributør\nTreehouse TV\n\n\nUdgivelse\n\n\nOprindelige netværk\nTreehouse TV (Canada)\nPBS Kids (USA)\nCITV og Tiny Pop (UK)\n\n\nBilledformat\n480i (SDTV)\n1080i (HDTV)\n\n\nOriginaludgivelse\n7. august 2010 (2010-08-07) - nu\n\n\nEksterne links\n\n\nWebsted\npbskids.org/catinthehat/",
  "question": 'Hvem synger titelmelodien til the cat in the hat?',
  "answers": {
    "answer_start": array([269]),
    "text": array(['David Schweitzer'], dtype=object)
  }
}
```
```json
{
  "context": 'Modern Slavery Act 2015\nLoven om moderne slaveri fra 2015 er en lov fra Det Forenede Kongeriges parlament. Den har til formål at bekæmpe slaveri i Det Forenede Kongerige og konsoliderer tidligere lovovertrædelser vedrørende menneskehandel og slaveri. Loven gælder for England og Wales. Lovforslaget blev forelagt underhuset i udkast i oktober 2013 af James Brokenshire, parlamentarisk undersekretær for kriminalitet og sikkerhed, i oktober 2013. Lovforslagets sponsorer i indenrigsministeriet var Theresa May og Lord Bates. Det fik kongelig samstemmende udtalelse og blev lov den 26. marts 2015.',
  "question": 'Hvornår trådte den moderne slaveri i kraft?',
  "answers": {
    "answer_start": array([580]),
    "text": array(['26. marts 2015'], dtype=object)
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:
  ```
  Følgende er tekster med tilhørende spørgsmål og svar.
  ```
- Base prompt template:
  ```
  Tekst: {text}
  Spørgsmål: {question}
  Svar med maks. 3 ord: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Tekst: {text}

  Besvar følgende spørgsmål om teksten ovenfor med maks. 3 ord.

  Spørgsmål: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset scandiqa-da
```


## Knowledge

### Danske Talemåder

This dataset was created by The Danish Language and Literature Society, published
[here](https://sprogteknologi.dk/dataset/1000-talemader-evalueringsdatasaet). The
dataset features Danish idioms along with their official meaning. For each idiom, three
negative samples were created: (a) a random idiom, (b) a concrete made-up idiom, and (c)
an abstract made-up idiom. The dataset was created to evaluate the ability of language
models to understand Danish idioms.

The original full dataset consists of 1,000 samples. We use a 128 / 64 / 808 split for
training, validation and testing, respectively (so 1,000 samples used in total).

Here are a few examples from the training split:

```json
{
  "text": "Hvad betyder udtrykket 'tale nogen efter munden'?\nSvarmuligheder:\na. være føjelig og give nogen ret selvom man ikke nødvendigvis er enig\nb. erklære sig helt enig med en anden person\nc. sige det præcis samme som en anden; efterabe\nd. være egoistisk og snæversynet; kun tænke på sig selv",
  "label": "a"
}
```
```json
{
  "text": "Hvad betyder udtrykket 'der falder en sten fra éns hjerte'?\nSvarmuligheder:\na. en bestemt (kriminel, eftersøgt) person er forsvundet\nb. man bliver fri for en sorg eller bekymring; man bliver lettet\nc. man mister én man har kær\nd. en sten forlader et hjerte man er i besiddelse af",
  "label": "b"
}
```
```json
{
  "text": "Hvad betyder udtrykket 'have spidse albuer'?\nSvarmuligheder:\na. person der har det meget dårligt fysisk og psykisk\nb. have ophobet vrede over længere tid\nc. hævde sig på andres bekostning\nd. have knogler der træder tydeligt frem på ens albuer",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Følgende er multiple choice spørgsmål (med svar).
  ```
- Base prompt template:
  ```
  Hvad er betydningen af følgende talemåde: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svar: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Hvad er betydningen af følgende talemåde: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Besvar ovenstående spørgsmål ved at svare med 'a', 'b', 'c' eller 'd', og intet andet.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset danske-talemaader
```


### Danish Citizen Tests

This dataset was created by scraping the Danish citizenship tests (indfødsretsprøven)
and permanent residency tests (medborgerskabsprøven) from 2016 to 2024. These are
available on the [official website of the Danish Ministry of International Recruitment
and Integration](https://danskogproever.dk/).

The original full dataset consists of 870 samples. We use an 345 / 90 / 525 split for
training, validation and testing, respectively. Here all the citizenship tests belong to
the test split, as well as the newest permanent residency tests. The validation split
contains the newer permanent residency tests after the ones in the test split, and the
training split contains the oldest permanent residency tests.

Here are a few examples from the training split:

```json
{
  "text": "Hvilket parti tilhørte Lars Løkke Rasmussen, da han var statsminister i perioderne 2009-11 og 2015-19?\nSvarmuligheder:\na. Venstre\nb. Socialdemokratiet\nc. Det Konservative Folkeparti",
  "label": "a"
}
```
```json
{
  "text": "Hvilket af følgende områder har kommunerne ansvaret for driften af?\nSvarmuligheder:\na. Domstole\nb. Vuggestuer\nc. Sygehuse",
  "label": "b"
}
```
```json
{
  "text": "Hvilken organisation blev Danmark medlem af i 1945?\nSvarmuligheder:\na. Verdenshandelsorganisationen (WTO)\nb. Den Europæiske Union (EU)\nc. De Forenede Nationer (FN)",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Følgende er multiple choice spørgsmål (med svar).
  ```
- Base prompt template:
  ```
  Spørgsmål: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  Svar: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Spørgsmål: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}
  c. {option_c}

  Besvar ovenstående spørgsmål ved at svare med 'a', 'b', 'c' eller 'd', og intet andet.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset danish-citizen-tests
```


### Unofficial: MMLU-da

This dataset is a machine translated version of the English [MMLU
dataset](https://openreview.net/forum?id=d7KBjmI3GmQ) and features questions within 57
different topics, such as elementary mathematics, US history and law. The translation to
Danish was done by the University of Oregon as part of [this
paper](https://aclanthology.org/2023.emnlp-demo.28/), using GPT-3.5-turbo.

The original full dataset consists of 269 / 1,410 / 13,200 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 2,048 split for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
new and there can thus be some overlap between the original validation and test sets and
our validation and test sets.

Here are a few examples from the training split:

```json
{
  "text": "Hvilket af følgende coronavirusser har forårsaget tusindvis af dødsfald over hele verden som en 'opstået' virus?\nSvarmuligheder:\na. MERS\nb. SARS\nc. OC43\nd. HKU1",
  "label": "a"
}
```
```json
{
  "text": "Hvilken orbitale væg er mest sandsynligt at kollapse i en 'blow out' fraktur?\nSvarmuligheder:\na. Taget\nb. Gulvet\nc. Den laterale væg\nd. Den mediale væg",
  "label": "b"
}
```
```json
{
  "text": "Hvad er navnet på den største struktur i Teotihuacán, og hvor mange platforme og pyramider blev bygget der?\nSvarmuligheder:\na. Månepyramiden; 250\nb. Templet for den fjerkræklædte slange; 400\nc. Solpyramiden; 600\nd. Inskriptionstemplen; 700",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Følgende er multiple choice spørgsmål (med svar).
  ```
- Base prompt template:
  ```
  Spørgsmål: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svar: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Spørgsmål: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Besvar ovenstående spørgsmål ved at svare med 'a', 'b', 'c' eller 'd', og intet andet.
  ```


### Unofficial: ARC-da

This dataset is a machine translated version of the English [ARC
dataset](https://doi.org/10.48550/arXiv.1803.05457) and features US grade-school science
questions. The translation to Danish was done by the University of Oregon as part of
[this paper](https://aclanthology.org/2023.emnlp-demo.28/), using GPT-3.5-turbo.

The original full dataset consists of 1,110 / 297 / 1,170 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 1,024 split for training,
validation and testing, respectively (so 2,304 samples used in total). All new splits
are subsets of the original splits.

Here are a few examples from the training split:

```json
{
  "text": "Et farmaceutisk firma har offentliggjort resultaterne af et begrænset eksperiment, der undersøger den beskyttende virkning af en kemisk forbindelse mod høje doser af UV-stråler på hudceller. Senere blev det opdaget, at resultaterne ikke var reproducerbare. Hvilken handling kunne forskere fra firmaet have foretaget for at undgå at offentliggøre fejlagtige resultater?\nSvarmuligheder:\na. Udfør flere forsøg.\nb. Brug kun lave niveauer af stråling.\nc. Brug forskellige bølgelængder af stråling.\nd. Undersøg resultaterne af lignende eksperimenter, før man dannede en hypotese.",
  "label": "a"
}
```
```json
{
  "text": "En ingeniør skal beregne den potentielle energi af en rutschebanekabine øverst på en skråning. Hvilken information ville bedst hjælpe ingeniøren med at bestemme den potentielle energi af kabine?\nSvarmuligheder:\na. den afstand, som rutschebanekabinen skal rejse\nb. massen af rutschebanekabinen ved fuld kapacitet\nc. den gennemsnitlige vægt af en tom rutschebanekabine\nd. retningen, som rutschebanekabinen bevæger sig i",
  "label": "b"
}
```
```json
{
  "text": "En studerende hældte vand i en plastbakke. Studerende satte derefter bakken i fryseren. Hvilken egenskab ved vand ændrede sig, da vandet fryser?\nSvarmuligheder:\na. Vandet blev til en gas.\nb. Massen af vandet steg.\nc. Vandet tog en bestemt form.\nd. Smagen af vandet ændrede sig ikke.",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Følgende er multiple choice spørgsmål (med svar).
  ```
- Base prompt template:
  ```
  Spørgsmål: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_c}
  Svar: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Spørgsmål: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Besvar ovenstående spørgsmål ved at svare med 'a', 'b', 'c' eller 'd', og intet andet.
  ```


## Common-sense Reasoning

### HellaSwag-da

This dataset is a machine translated version of the English [HellaSwag
dataset](https://aclanthology.org/P19-1472/). The original dataset was based on both
video descriptions from ActivityNet as well as how-to articles from WikiHow. The dataset
was translated by the University of Oregon as part of [this
paper](https://aclanthology.org/2023.emnlp-demo.28/), using GPT-3.5-turbo.

The original full dataset consists of 9,310 samples. We use a 1,024 / 256 / 2,048 split
for training, validation and testing, respectively (so 3,328 samples used in total).

Here are a few examples from the training split:

```json
{
  "text": "Disse mennesker træder pedalerne med kun det ene ben og står midt på cyklen med det andet ben, der holder deres hænder oppe. næste gør de\nSvarmuligheder:\na. en anden øvelse, hvor de sætter det ene ben på pedalen, mens de har det andet ben ude og hopper op og ned.\nb. tager hinandens hænder og udfører en eller anden dansebevægelse på børsterne, som de bruger til at snurre rundt med deres kroppe og hoppe med hænderne oppe.\nc. drejer med deres forstenede hænder, laver en U-vending og starter derefter deres handlinger igen og igen.\nd. skifter til at stå ved hjælp af to arme for at balancere sig selv.",
  "label": "a"
}
```
```json
{
  "text": "[header] Sådan dræber du frugtfluer [title] Brug rådden frugt. [step] Dit problem med frugtfluer begyndte sandsynligvis først, da du opdagede, at du havde efterladt nogle frugter, der til sidst blev rådne. Brug den metode, der samlede fluene første gang til at fange dem igen, men denne gang før dem til en mere morbide slutning.\nSvarmuligheder:\na. Dræb fluene ved at trække dem fra deres rede eller ved at bruge tunge kæder med tænger til at fange dem og placere dem i en spand eller stuen. Du kan også bruge dyreafføring såsom fiske- og ande-urin.\nb. Placer et stykke rådden frugt i en skål og stræk klart plastik over toppen. Skær flere små huller i plastikken med en tandstik og lad det stå tæt på stedet med fluene.\nc. Efter at have forsøgt at fange dobbelt så mange fluer, som du kan, skal du fjerne de ubehagelige frugtstykker fra pakken og bage dem i 2-3 minutter. Fluene vil flyde øverst på den søde marmelade, når du fjerner frugten fra marmeladen.\nd. [substeps] Tjek dåser for knotten, melbiller og fluer. Køb blomster fra havecentret, hvis du ikke har al produktion i nærheden.",
  "label": "b"
}
```
```json
{
  "text": "En mand står indendørs på en platform foran tre tilskuere og løfter en tung vægtstang. En mand nærmer sig en vægtstang på gulvet og står foran den og forbereder sig på at løfte den. manden\nSvarmuligheder:\na. løfter vægtstangen, der hænger i luften på platformen, og vender sig mod tilskuerne.\nb. løfter vægtstangen og viser, hvordan han udfører det, idet han pauser på hver stang for at måle vægten.\nc. bøjer sig derefter i knæene og lægger hænderne på vægtens stangdel.\nd. løfter derefter klokken på sine skuldre, læner sig tilbage, sætter armene bag hovedet og løfter den let.",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Følgende er multiple choice spørgsmål (med svar).
  ```
- Base prompt template:
  ```
  Spørgsmål: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_c}
  Svar: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Spørgsmål: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Besvar ovenstående spørgsmål ved at svare med 'a', 'b', 'c' eller 'd', og intet andet.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset hellaswag-da
```


## Summarization

### Nordjylland News

This dataset is based on news articles from the Danish news site [TV2
Nord](https://www.tv2nord.dk/), where the summaries are taken as the introductory
paragraphs of the articles.

The original full dataset consists of 75,200 samples. We use an 1,024 / 256 / 2,048
split for training, validation and testing, respectively (so 3,328 samples used in
total).

Here are a few examples from the training split:

```json
{
  "text": "Jacob Emil Andersen viste søndag rundt på Halvorsminde Efterskole ved Hjørring. Skolen har ligget på samme sted siden 1903. Han er selv elev, da en IT-linje på skolen fangede hans interesse. - Det betyder meget for mig, jeg ville ikke have været lige så interesseret i den her skole, hvis der ikke havde været IT, fortæller Jacob Emil Andersen, der oprindeligt stammer fra Aalborg, til TV2 Nord. En af dem, han viser rundt til Efterskolernes dag, er Isabella Kristensen, der går i skole i Hune. Hun er på jagt efter noget helt specielt. - Helt sikkert dans, springgymnastik og fitness med noget puls, forklarer Isabella Kristensen til TV2 Nord. Netop efterskolernes specialisering er en af grundene til, at rekordmange vælger at bruge et år væk fra familien i 8.-, 9.- eller 10.-klasse. De særlige linjefag har man flere af på Halvorsminde Efterskole. Jern og metal, arbejde med træ og vinterbadning er blot nogle af de aktiviteter, eleverne kan støde ind i på de forskellige linjefag, som skolen tilbyder. Men efterskolerne skal også huske at have fokus på den faglighe kvalitet, lyder det fra forstanderen. - Vi skal være skarpe på nogle nicheprodukter og nogle linjer med noget god kvalitet. Så skal vi også lave god skole, fortæller forstander på Halvorsminde Efterskole, Jens Beermann, til TV2 Nord. Han bliver bakket op af sin kollega fra Hørby Efterskole ved Sæby omkring 30 kilometer fra Halvorsminde. - Når man laver sit valgfagsudbud, skal det ikke være tilfældigt. Man skal ikke tænke, at ’det er smart! Det må trække elever, det her!’ Der skal være en velovervejet refleksion i forhold til, om det passer ind i det, vi gerne vil som skole,, siger forstander på Hørby Efterskole, Mogens Vestergård, til TV2 Nord. Alene i Nordjylland gik mere end 2.000 elever på efterskole i skoleåret 2018-2019. Både Halvorsminde Efterskole og Hørby Skole har plads til 130 elever. Og noget tyder på, at der i hvert fald er sikret en ny elev til næste skoleår efter dagens åbent hus. - Jeg synes at det ser spændende ud, og jeg har endnu mere lyst til at gå her nu, siger Isabella Kristensen.",
  "target_text": "Søndag inviterede efterskoler landet over potentielle nye elever inden for. Efterskolerne specialiserer sig for at tiltrække elever, men den gode faglighed må ikke blive glemt, lyder det fra nordjyske forstandere."
}
```
```json
{
  "text": "Efter en nat med spejl glatte veje i Nordjylland melder Nordjyllands Politi om en helt problemfri morgen. Selvom politikredse i TV2 Nords sendeområde melder om en rolig nat uden større uheld, så kan de bilister, der skal af sted lørdag morgen godt forvente lidt længere rejsetid. Der er nemlig stadig glatte veje, og der er faldet en del sne i Nordjylland. Saltvogne og sneplove har allerede været på vejene, og Politiet opfordre forsat bilisterne til at køre forsigtigt ude på de snefyldte veje.",
  "target_text": "Nordjyllands Politi melder om en stille morgen trods glatte veje og stort snefald i nat."
}
```
```json
{
  "text": "Det var meget tæt på at gå galt for en 10-årig tysk dreng onsdag eftermiddag. Klokken 15:55 modtog alarmcentralen et opkald om en drengen, der var begravet i sand ved Vorupør Strand. - Nogle børn legede på stranden, og her har de så gravet et hul ind i klitten. Det er så det, der er kollapset omkring drengen, fortæller vagtchef Carsten Henriksen ved Midt- og Vestjyllands Politi. Det vides ikke præcist, hvor meget sand der væltede ned over barnet, men det var nok til, at drengen ikke selv kunne komme fri. De tilstedeværende på stranden måtte grave ham fri. Han var helt begravet i sand i omkring fem minutter. - Der var en tysk læge på stranden, der kunne give førstehjælp, indtil ambulancen kunne komme frem, fortæller vagtchefen. Drengen kom sig hurtigt og har det godt, men blev alligevel kørt til tjek på Aalborg Sygehus.",
  "target_text": "Børn på Vorupør Strand havde gravet et hul ind i klitterne, som kollapsede omkring en 10-årig dreng."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:
  ```
  Følgende er nyhedsartikler med tilhørende resuméer.
  ```
- Base prompt template:
  ```
  Nyhedsartikel: {text}
  Resumé: {target_text}
  ```
- Instruction-tuned prompt template:
  ```
  Nyhedsartikel: {text}

  Skriv et resumé af ovenstående artikel.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset nordjylland-news
```
