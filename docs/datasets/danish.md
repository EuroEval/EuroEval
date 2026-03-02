# 🇩🇰 Danish

This is an overview of all the datasets used in the Danish part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### Angry Tweets

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

  ```text
  Følgende er tweets og deres sentiment, som kan være 'positiv', 'neutral' eller 'negativ'.
  ```

- Base prompt template:

  ```text
  Tweet: {text}
  Sentiment: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Tweet: {text}

  Klassificer sentimentet i tweetet. Svar kun med 'positiv', 'neutral' eller 'negativ'.
  ```

- Label mapping:
  - `positive` ➡️ `positiv`
  - `neutral` ➡️ `neutral`
  - `negative` ➡️ `negativ`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset angry-tweets
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

  ```text
  Følgende er sætninger og JSON-ordbøger med de navngivne enheder, som forekommer i den givne sætning.
  ```

- Base prompt template:

  ```text
  Sætning: {text}
  Navngivne enheder: {label}
  ```

- Instruction-tuned prompt template:

  ```text
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
euroeval --model <model-id> --dataset dansk
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

The original dataset consists of 5,512 samples, from which we use 1,024 / 256 / 2,048
samples for training, validation and testing, respectively (so 3,328 samples used in
total). These splits are used as-is in the framework.

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

  ```text
  Følgende er sætninger og om de er grammatisk korrekte.
  ```

- Base prompt template:

  ```text
  Sætning: {text}
  Grammatisk korrekt: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Sætning: {text}

  Bestem om sætningen er grammatisk korrekt eller ej. Svar med 'ja', hvis sætningen er korrekt, og 'nej', hvis den ikke er.
  ```

- Label mapping:
  - `correct` ➡️ `ja`
  - `incorrect` ➡️ `nej`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-da
```

### Unofficial: DaLA

This dataset was published in [this paper](https://arxiv.org/abs/2512.04799) and,
similarly to ScaLA, was automatically created from the
[Danish Universal Dependencies treebank](https://github.com/UniversalDependencies/UD_Danish-DDT/tree/master)
by assuming that the documents in the treebank are correct, and corrupting the
samples to create grammatically incorrect samples.

This is an extension of ScaLA-da based on an analysis of most common errors made
by Danish speakers. It adds 12 new linguistically grounded error types on top of
the existing 2 from ScaLA-da, the corruption type applied to each sentence is also
indicated in the dataset (`corruption_type` column). The corruptions have been based
on linguistic features (e.g. POS tags, morphology features etc.) so to both ground
them on linguistic rules and ensure the unacceptability of the corrupted sentences.
The corruption quality has been both automatically and manually validated as
detailed in the paper.

Like ScaLA-da, the original dataset consists of 5,512 samples, from which we use
1,024 / 256 / 2,048 samples for training, validation and testing, respectively
(so 3,328 samples used in total).

Here are a few examples from the training split:

```json
{
  "text": "Det gøres bedst i en enhedsskole med differentieret undervisning.",
  "label": "correct",
  "corruption_type": null
}
```

```json
{
  "text": "Endelig er også inspektionsskiben Fylla til salg.",
  "label": "incorrect",
  "corruption_type": "flip_en_et_suffix"
}
```

```json
{
  "text": "Men trods betalingen har kreditorene jo stadig haft milliontab på sagen?",
  "label": "incorrect",
  "corruption_type": "corrupt_ende_ene"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Følgende er sætninger og om de er grammatisk korrekte.
  ```

- Base prompt template:

  ```text
  Sætning: {text}
  Grammatisk korrekt: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Sætning: {text}

  Bestem om sætningen er grammatisk korrekt eller ej. Svar med 'ja', hvis sætningen er korrekt, og 'nej', hvis den ikke er.
  ```

- Label mapping:
  - `correct` ➡️ `ja`
  - `incorrect` ➡️ `nej`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset dala
```

## Natural Language Inference

### The Danish Entailment Dataset

This dataset was published in [this
paper](https://aclanthology.org/2024.lrec-main.1421/) as part of the Danish Semantic
Reasoning Benchmark and was developed at the Centre for Language Technology at the
University of Copenhagen. Each sample pairs two Danish statements and asks whether the
second statement follows from the first.

The original dataset contains 319 usable samples after filtering. We use a split of
64 / 64 / 190 samples for training, validation and testing, respectively (so 318 samples
used in total after deduplication). The splits were created by randomly sampling from
the full dataset.

Here are a few examples from the training split:

```json
{
  "text": "Udsagn 1: Peter efterlod sig kone og tre små børn.\nUdsagn 2: Peter havde en kone og mindre børn.",
  "label": "entailment"
}
```

```json
{
  "text": "Udsagn 1: Peter efterlod sig kone og tre små børn.\nUdsagn 2: Peter er stadig gift med sin kone.",
  "label": "contradiction"
}
```

```json
{
  "text": "Udsagn 1: Regeringen afskaffede karaktergivning på de første tre klassetrin i grundskolen.\nUdsagn 2: Der gives ikke længere karakterer til eleverne på grundskolens mellemste klassetrin.",
  "label": "neutral"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Følgende er par af udsagn og om det andet udsagn følger af det første, hvilket kan være 'sand', 'neutral' eller 'falsk'.
  ```

- Base prompt template:

  ```text
  Udsagn: {text}
  Entailment: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Udsagn: {text}

  Bestem om det andet udsagn følger af det første udsagn. Svar kun med 'sand', 'neutral' eller 'falsk', og intet andet.
  ```

- Label mapping:
  - `entailment` ➡️ `sand`
  - `neutral` ➡️ `neutral`
  - `contradiction` ➡️ `falsk`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset danish-entailment
```

## Reading Comprehension

### MultiWikiQA-da

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
    "context": 'Rødspætten (Pleuronectes platessa) er en fladfisk, der findes overalt i de danske farvande. Den er i øvrigt udbredt fra Middelhavet til Island og Hvidehavet. Den foretrækker steder, hvor bunden består af sten, sand og grus. De unge rødspætter findes på lavt vand, mens de voksne foretrækker 10-50 meters dybde. Rødspætten er en højrevendt fladfisk, idet det normalt er højre side, der under larvens forvandling bliver til overside.\n\nUdseende \nRødspætten kan blive op til 100 centimeter, men bliver i Danmark sjældent over 50 centimeter. Den kendes bedst på, at der bag øjnene løber en buet køl med 4-7 benknuder. Skællene er små og glatte og ikke taglagte. Munden er lille med ret tykke læber. Begge øjne findes normalt på fiskens højre side. På oversiden er rødspætten oftest brunlig med et grønligt skær og med spredte rødlige pletter, der ofte er omgivet af lyse eller mørke ringe. Undersiden er hvid.\n\nLevevis \nRødspætten lever især af børsteorme og tyndskallede muslinger. Den er mest aktiv i døgnets mørke timer, mens den skjuler sig på bunden om dagen. Den skifter farve efter bundens farve og struktur. Rødspættens naturlige fjender er ud over mennesket f.eks. krabber og torsk.\n\nForplantning \nHannerne bliver i Nordsøen kønsmodne 3-4 år gamle og en længde på 20 centimeter, mens hunnerne kønsmodner et par år senere. I Østersøen bliver begge køn tidligere kønsmodne. Gydningen foregår normalt i 20-50 meters dybde i perioden januar til juni. Rødspætten foretrækker en temperatur på 6\xa0°C til gydningen. Æggene er glasklare med en diameter på cirka 2 millimeter og flyder op til overfladen. Efter 2-3 uger klækkes de 6 millimeter store larver. Larverne lever af planktonorganismer og begynder efter cirka 5 uger med en længde på 1 centimeter en forvandling, hvor venstre øje vandrer op over hovedet, der vrides, og kroppen bliver bredere. Til at begynde med svømmer de små rødspætter skråt og siden med højre side opad. Med en længde på 1,2-1,4 centimeter skifter de fra et pelagisk liv til at leve på lavt vand langs kysterne. I det første efterår måler rødspætten 7-12 centimeter og trækker ud, for at overvintre på dybere vand.\n\nKilder/Henvisninger \n\n C. V. Otterstrøm (1881-1962).\xa0Danmarks Fauna. Fisk II. Blødfinnefisk. G.E.C. Gads Forlag. København 1914.\n\nFladfisk',
    "question": 'Hvilken side af rødspætten vender typisk opad?',
    "answers": {
        "answer_start": array([369]),
        "text": array(['højre side'], dtype=object)
    }
}
```

```json
{
    "context": 'Mzilikazi ("blodvejen" eller "den store vej" ca. 1790–9. september 1868) var en sydafrikansk konge som grundlagde matabelekongedømmet i det område, som nu er Zimbabwe. Han var søn af Matshobana og blev født nær Mkuze i Zululand (nu del af Sydafrika) og døde ved Ingama i Matabeleland (nær Bulawayo, Zimbabwe). Mange regner ham som den største sydafrikanske militærleder efter zulukongen Shaka.\n\nHan førte sin stamme, khumalo, på en 800 km lang rejse fra Zululand til det, som nu er Zimbabwe. På vejen viste han betydelige statsmandsevner, da han samlede sit eget folk og de mange stammer han erobrede, til et stort,  etnisk rigt og centraliseret kongedømme.\n\nHan var oprindelig en af Shakas løjtnanter, men i 1823 gjorde han oprør. Frem for at møde rituel henrettelse, flygtede han sammen med sin stamme. Han rejste først til Mozambique og i 1826 ind i Transvaal på grund af fortsatte angreb fra sine fjender.\n\nFortsatte angreb fik ham først til at flytte til dagens Botswana og i 1837 til det, som nu er Zambia Han klarede ikke at erobre den indfødte kololo–nation der og rejste til det, som blev kendt som Matabeleland (i dagens Zimbabwe) og slog sig ned der i 1840.\n\nEfter hans ankomst organiserede han sine tilhængere i et militærsystem med regiment–kraaler som kong Shakas, som blev stærke nok til at afvise boernes angreb i 1847–1851 og tvinge den Sydfrikanske Republiks regering til at underskrive en fredsaftale med ham i 1852.\n\nMzilikazi var generelt venlig over for europæisk rejsende, førte opdagelsen af guld i Matabeleland i 1867 til en flom af bosættere, som han ikke kunne kontrollere, og som førte til kongedømmets endelige nederlag under hans efterfølger Lobengula.\n\nKongelige fra historiske riger',
    "question": 'Med hvilket øgenavn var Mzilikazi kendt?',
    "answers": {
        "answer_start": array([11]),
        "text": array(['"blodvejen" eller "den store vej"'], dtype=object)
    }
}
```

```json
{
    "context": 'Jean-Nicolas Bouilly (24. januar 1763 i La Coudraye ved Tours – 14. april 1842 i Paris) var en fransk forfatter. \n\nEfter at have studeret jura sluttede Bouilly sig ved revolutionens udbrud til Mirabeau og Barnave og beklædte forskellige embeder, i hvilke han navnlig virkede for indførelsen af primærskoler og for folkeoplysning i det hele taget. Senere trak han sig tilbage og vedblev at leve uafhængig til sin død. 1790 opførtes hans opéra comique Pierre le Grand, med musik af Grétry. Af hans senere dramatiske arbejder kan nævnes L\'abbé de l\'Épée(1795), Les deux journées (1800), komponeret af Cherubini, Fanchon (1802), komponeret af Himmel, L\'intrigue aux fenêtres, Une folie (1803, med musik af Méhul; på dansk ved N.T. Bruun: "Ungdom og Galskab" [1806], med musik af Du Puy), Mme. de Sévigné (1805) og så videre. Desuden oversatte han flere stykker af Kotzebue. Hans skrifter for ungdommen stod i sin tid i høj kurs; hans stil er vidtsvævende og retorisk, hans billeder skruede, hele tonen så sentimental, at han fik navnet le poète lacrymal. Af disse skrifter kan nævnes: Contes offerts aux enfants de France, Contes à ma fille (1809), Conseils à ma fille (1811) og Les jeunes femmes (1819).\n\nKilder \n\n \n\nDramatikere fra Frankrig\nFranskmænd i 1700-tallet\nFranskmænd i 1800-tallet\nSalmonsens',
    "question": 'Med hvilke politiske personer allierede Bouilly sig ved revolutionens begyndelse?',
    "answers": {
        "answer_start": array([193]),
        "text": array(['Mirabeau og Barnave'], dtype=object)
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Følgende er tekster med tilhørende spørgsmål og svar.
  ```

- Base prompt template:

  ```text
  Tekst: {text}
  Spørgsmål: {question}
  Svar med maks. 3 ord: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Tekst: {text}

  Besvar følgende spørgsmål om teksten ovenfor med maks. 3 ord.

  Spørgsmål: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-da
```

### Unofficial: ScandiQA-da

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
  "context": "\"(Sittin\' On) The Dock of the Bay\" er en sang, der er skrevet af soul-sangeren Otis Redding og guitaristen Steve Cropper sammen. Den blev indspillet af Redding to gange i 1967, herunder en gang få dage før hans død i et flystyrt. Sangen blev udgivet på Stax Records\' Volt-label i 1968 og blev den første posthume single, der lå øverst på hitlisterne i USA. Den nåede op som nummer 3 på den britiske single-liste.",
  "question": "Hvem sang sitting on the dock of the bay?",
  "answers": {
    "answer_start": array([79]),
    "text": array(["Otis Redding"], dtype=object)
  }
}
```

```json
{
  "context": "The Cat in the Hat Knows a Lot About That!\nKatten i hatten ved meget om det!\n\n\n\nKatten i hatten pilot\n\n\n\nGenre\nBørne-tv/undervisning/komedie\n\n\nInstrueret af\nTony Collingwood\n\n\nStemmer fra\nMartin Short\nJacob Ewaniuk\nAlexa Torrington\nRob Tinkler\n\n\nKomponist af temamusik\nDavid Schweitzer\n\n\nKomponist(er)\nDavid Schweitzer\n\n\nOprindelsesland\nCanada\nDet Forenede Kongerige\nUSA\n\n\nOprindelige sprog\nEngelsk\n\n\nAntal sæsoner\n2\n\n\nAntal episoder\n60 (liste over episoder)\n\n\nProduktion\n\n\nLøbetid\n30 minutter\n\n\nProduktionsselskab(er)\nCollingwood O'Hare Productions\nPortfolio Entertainment\nRandom House Children's Entertainment\nTreehouse TV\n\n\nDistributør\nTreehouse TV\n\n\nUdgivelse\n\n\nOprindelige netværk\nTreehouse TV (Canada)\nPBS Kids (USA)\nCITV og Tiny Pop (UK)\n\n\nBilledformat\n480i (SDTV)\n1080i (HDTV)\n\n\nOriginaludgivelse\n7. august 2010 (2010-08-07) - nu\n\n\nEksterne links\n\n\nWebsted\npbskids.org/catinthehat/",
  "question": "Hvem synger titelmelodien til the cat in the hat?",
  "answers": {
    "answer_start": array([269]),
    "text": array(["David Schweitzer"], dtype=object)
  }
}
```

```json
{
  "context": "Modern Slavery Act 2015\nLoven om moderne slaveri fra 2015 er en lov fra Det Forenede Kongeriges parlament. Den har til formål at bekæmpe slaveri i Det Forenede Kongerige og konsoliderer tidligere lovovertrædelser vedrørende menneskehandel og slaveri. Loven gælder for England og Wales. Lovforslaget blev forelagt underhuset i udkast i oktober 2013 af James Brokenshire, parlamentarisk undersekretær for kriminalitet og sikkerhed, i oktober 2013. Lovforslagets sponsorer i indenrigsministeriet var Theresa May og Lord Bates. Det fik kongelig samstemmende udtalelse og blev lov den 26. marts 2015.",
  "question": "Hvornår trådte den moderne slaveri i kraft?",
  "answers": {
    "answer_start": array([580]),
    "text": array(["26. marts 2015"], dtype=object)
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Følgende er tekster med tilhørende spørgsmål og svar.
  ```

- Base prompt template:

  ```text
  Tekst: {text}
  Spørgsmål: {question}
  Svar med maks. 3 ord: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Tekst: {text}

  Besvar følgende spørgsmål om teksten ovenfor med maks. 3 ord.

  Spørgsmål: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scandiqa-da
```

### Unofficial: BeleBele-da

This dataset was published in [this paper](https://aclanthology.org/2024.acl-long.44/)
and features multiple-choice reading comprehension questions across 122 languages.

The original dataset contains 900 unique multiple-choice reading comprehension passages
and questions. From these, we use a 256 / 64 / 580 split for training, validation and
testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Tekst: Prognoserne siger, at stormen, der er omkring 645 mil (1040 km) vest for Kap Verde-øerne, sandsynligvis vil forsvinde, før den truer nogen landområder. Fred har i øjeblikket vinde på 165 km/t og bevæger sig mod nordvest. Fred er den heftigste tropiske cyklon, der nogensinde er blevet registreret så sydligt og østligt i Atlanterhavet, siden man begyndte at bruge satellitbilleder, og kun den tredje store orkan, der er registreret øst for 35°V.\nSpørgsmål: Da Fred befandt sig nær Kap Verde-øerne, hvilken retning bevægede den sig så mod?\nSvarmuligheder:\na. Vest\nb. Syd\nc. Øst\nd. Nordvest",
  "label": "d"
}
```

```json
{
  "text": "Tekst: "Siden Pakistan i 1947 blev uafhængigt af det britiske styre, har den pakistanske præsident udpeget ""politiske agenter"", som styrer FATA, og som har næsten fuldstændig kontrol over områderne. Disse agenter er ansvarlige for at levere regerings- og retstjenester i henhold til artikel 247 i den pakistanske forfatning."\nSpørgsmål: Hvem leverer retslige tjenester til FATA?\nSvarmuligheder:\na. Den pakistanske regering\nb. Politiske agenter\nc. Pakistans præsident\nd. Den britiske regering",
  "label": "b"
}
```

```json
{
  "text": "Tekst: Alle er en del af samfundet og benytter transportsystemerne. Næsten alle klager over transportsystemerne. I udviklede lande hører du sjældent ligeså mange klager over vandkvalitet eller broer, der styrter sammen. Hvorfor giver transportsystemerne anledning til sådanne klager, hvorfor svigter de på daglig basis? Er transportingeniører blot inkompetente? Eller foregår der noget mere fundamentalt?\nSpørgsmål: Hvilken offentlig service siges at skabe størst utilfredshed i udviklede lande?\nSvarmuligheder:\na. Vandkvalitet\nb. Brobyggelse\nc. Offentlig transport\nd. Uddannelse",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Følgende er multiple choice spørgsmål (med svar).
  ```

- Base prompt template:

  ```text
  Spørgsmål: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svar: {label}
  ```

- Instruction-tuned prompt template:

  ```text
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
euroeval --model <model-id> --dataset belebele-da
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

  ```text
  Følgende er multiple choice spørgsmål (med svar).
  ```

- Base prompt template:

  ```text
  Hvad er betydningen af følgende talemåde: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svar: {label}
  ```

- Instruction-tuned prompt template:

  ```text
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
euroeval --model <model-id> --dataset danske-talemaader
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

  ```text
  Følgende er multiple choice spørgsmål (med svar).
  ```

- Base prompt template:

  ```text
  Spørgsmål: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  Svar: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Spørgsmål: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}
  c. {option_c}

  Besvar ovenstående spørgsmål ved at svare med 'a', 'b', 'c' eller 'd', og intet andet.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset danish-citizen-tests
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

  ```text
  Følgende er multiple choice spørgsmål (med svar).
  ```

- Base prompt template:

  ```text
  Spørgsmål: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svar: {label}
  ```

- Instruction-tuned prompt template:

  ```text
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

  ```text
  Følgende er multiple choice spørgsmål (med svar).
  ```

- Base prompt template:

  ```text
  Spørgsmål: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svar: {label}
  ```

- Instruction-tuned prompt template:

  ```text
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

  ```text
  Følgende er multiple choice spørgsmål (med svar).
  ```

- Base prompt template:

  ```text
  Spørgsmål: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svar: {label}
  ```

- Instruction-tuned prompt template:

  ```text
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
euroeval --model <model-id> --dataset hellaswag-da
```

### Unofficial: GoldenSwag-da

This dataset is a filtered and machine translated version of the English [HellaSwag
dataset](https://aclanthology.org/P19-1472/), featuring both video descriptions from
ActivityNet as well as how-to articles from WikiHow. The machine translated version was
published in [this paper](https://doi.org/10.48550/arXiv.2410.08928) and was done using
DeepL, and the filtering was published in [this
paper](https://doi.org/10.48550/arXiv.2504.07825), which resulted in higher quality
samples.

The original full dataset consists of 1530 / 1530 samples for training and validation,
respectively. However, they are exactly equal. We use a split of 660 / 256 / 2,048
samples for training, validation, and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Sådan giver du dig selv en fransk manicure ved hjælp af tape. Gnid en vatpind med neglelakfjerner på alle dine negle. Det vil ikke kun fjerne afskallet lak eller rester af lak, men det vil også fjerne fugtighedscreme fra neglen. Hvis du har et fugtighedsbevarende middel, såsom lotion eller olie, på neglen, vil lakken ikke sidde ordentligt fast.\nSvarmuligheder:\na. Kom lakfjerneren i en lille skål. Du skal bruge den om et par minutter til at få denne opløsning på tæerne.\nb. Fordel et fugtgivende pulver over alle dine negle med cirkulære bevægelser, indtil du kommer i kontakt med huden. Polér altid neglene, inden du går i gang.\nc. Skum vattet i lakfjerneren. Brug en blød vaskeklud til at samle lakken op.\nd. Sørg for, at du har skabt et perfekt lærred til din franske manicure. Påfør din basisfarve på hele neglen.",
  "label": "d"
}
```

```json
{
  "text": "Sådan forbedrer du et lille barns tale. Kom ned på deres niveau. Sæt dig på hug eller på gulvet. Det vil få deres opmærksomhed.\nSvarmuligheder:\na. Du vil tale med dit barn i stedet for til det. Hun vil også kunne se din mund og få visuelle tegn på, hvordan man siger bestemte lyde.\nb. Løft om nødvendigt hænderne sammen til knytnæver. Hvis du strækker dine hænder til knytnæver og gør det, mens du taler, vil dit barn sandsynligvis gøre det samme.\nc. Prøv at være så stille som muligt, og tal kun til dem, når de er rolige. Hvis du taler længe nok, vil de til sidst høre din stemme.\nd. Lad dem bede dig om at rykke tættere på dem. Hvis det er muligt, så brug en siddepind i hovedhøjde.",
  "label": "a"
}
```

```json
{
  "text": "Sådan bruger du en bodysuit. Vælg en bodysuit, der smigrer dine yndlingstræk. Med så mange muligheder og stilarter kan bodysuiten virkelig være universelt flatterende. For at finde en body, der ser godt ud på dig, skal du overveje, hvilken del af din krop du vil fremhæve.\nSvarmuligheder:\na. Det kan være underarmene, benene eller andre steder, der stikker ud. Måske har du for eksempel en flot læbespalte, som du gerne vil fremhæve.\nb. Find ud af, hvilken del af din krop du vil fremhæve, og skær så ned på det, der fremhæver denne del. Hvis du for eksempel ønsker, at overdelene skal fremhæve dine bryster mest muligt, kan bikinitrusserne også bæres omkring det område.\nc. Hvis du for eksempel er stolt af dine tonede arme, skal du vælge en body uden ærmer eller med halterneck. Start med en bodysuit i t-shirt-stil, hvis du er ved at varme op til trenden.\nd. Beslut dig for, hvor mange forskellige dele af dig, din body skal fremhæve. Hvis du for eksempel vil have et sporty look, skal din body også fremhæve en del af din krop i stedet for en særlig iøjnefaldende del.",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Følgende er multiple choice spørgsmål (med svar).
  ```

- Base prompt template:

  ```text
  Spørgsmål: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svar: {label}
  ```

- Instruction-tuned prompt template:

  ```text
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
euroeval --model <model-id> --dataset goldenswag-da
```

### Unofficial: Winogrande-da

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2506.19468)
and is a translated and filtered version of the English [Winogrande
dataset](https://doi.org/10.1145/3474381).

The original full dataset consists of 47 / 1,210 samples for training and testing, and
we use 128 of the test samples for validation, resulting in a 47 / 128 / 1,085 split for
training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Natalie synes, at smaragder er smukke ædelstene, men Betty gør ikke. _ købte en halskæde med en stor smaragd. Hvad refererer det tomme _ til?\nSvarmuligheder:\na. Natalie\nb. Betty",
  "label": "a"
}
```

```json
{
  "text": "Jeg kunne ikke kontrollere fugten, som jeg kontrollerede regnen, fordi _ kom ind overalt. Hvad refererer det tomme _ til?\nSvarmuligheder:\na. fugt\nb. regn",
  "label": "a"
}
```

```json
{
  "text": "At håndtere nødsituationer var aldrig særlig svært for Kevin, men det var det for Nelson, fordi _ ikke var i stand til at forblive rolig under pres. Hvad refererer det tomme _ til?\nSvarmuligheder:\na. Kevin\nb. Nelson",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Følgende er multiple choice spørgsmål (med svar).
  ```

- Base prompt template:

  ```text
  Spørgsmål: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}
  Svar: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Spørgsmål: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}

  Besvar ovenstående spørgsmål ved at svare med 'a' eller 'b', og intet andet.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset winogrande-da
```

## Summarisation

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

  ```text
  Følgende er nyhedsartikler med tilhørende resuméer.
  ```

- Base prompt template:

  ```text
  Nyhedsartikel: {text}
  Resumé: {target_text}
  ```

- Instruction-tuned prompt template:

  ```text
  Nyhedsartikel: {text}

  Skriv et resumé af ovenstående artikel.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset nordjylland-news
```

## Instruction-following

### IFEval-da

This dataset was published
[here](https://huggingface.co/datasets/danish-foundation-models/ifeval-da) and is a
translation of the English IFEval dataset, which was published in [this
paper](https://doi.org/10.48550/arXiv.2311.07911) and contains 541 prompts, each with a
combination of one or more of 25 different constraints. The dataset was professionally
translated and localised by expert native speakers.

We use the original dataset as the test split, and do not include the other splits, as
we only evaluate models zero-shot and the size is too small to warrant an even smaller
validation set.

Here are a few examples from the test split:

```json
{
    "text": "Gentag først nedenstående forespørgsel ord for ord uden ændringer, og giv derefter dit svar. Tilføj ikke noget før du gentager nedenstående forespørgsel.\n\nSkriv en historie om en mand, der forsøger at få styr på sit liv. Skriv historiens navn i dobbelte vinkelparenteser, dvs. <<historien om xyz>>.",
    "target_text": {
        "instruction_id_list": [
            "detectable_format:title",
            "combination:repeat_prompt"
        ],
        "kwargs": [
            {},
            {
                "prompt_to_repeat": "Skriv en historie om en mand, der forsøger at få styr på sit liv. Skriv historiens navn i dobbelte vinkelparenteser, dvs. <<historien om xyz>>."
            }
        ]
    }
}
```

```json
{
    "text": "Skriv en mærkelig annonce for en advokat, der repræsenterer hekse og som er specialist i ophavsretskrænkelser. Brug kun små bogstaver. Dit svar skal indeholde en titel, omsluttet af dobbelte vinkelparenteser, dvs. <<titel>>.",
    "target_text": {
        "instruction_id_list": [
            "detectable_format:title"
        ],
        "kwargs": [
            {}
        ]
    }
}
```

```json
{
    "text": "Skriv følgende sætning kun på svensk, intet andet sprog er tilladt, og undgå brug af kommaer: \"Vi kan muligvis forbedre vores model til næste år. Vi vil kunne sammenligne vores data med data fra det foregående år og se hvordan vores model har præsteret. Vi kan også sammenligne vores model med en model, der blev trænet på data fra det foregående år, og se, hvordan vores model klarer sig.\" Intet andet sprog end svensk må bruges i dit svar.",
    "target_text": {
        "instruction_id_list": [
            "punctuation:no_comma",
            "language:response_language"
        ],
        "kwargs": [
            {},
            {
                "language": "sv"
            }
        ]
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

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ifeval-da
```

## European Values

### ValEU-da

This dataset is the official Danish version of questions from the [European values
study](https://europeanvaluesstudy.eu/). The dataset contains multiple-choice
questions regarding people's values and beliefs across a variety of topics, such as
politics, religion and society.

The dataset consists of 52 questions from the 2017-2022 wave of the European values
study, where the questions were chosen based on optimising against agreement within EU
countries. We use only zero-shot evaluation on this dataset, and thus require no splits.

Here are a few examples from the training split:

```json
{
    "question_id": "C039",
    "text": "Hvor enig eller uenig er du i følgende udsagn?\nDet er ens pligt over for samfundet at arbejde.\nSvarmuligheder:\na. Helt enig\nb. Enig\nc. Hverken enig eller uenig\nd. Uenig\ne. Helt uenig",
}
```

```json
{
    "question_id": "F122",
    "text": "Fortæl for hver af handlingerne på dette kort, i hvilken grad du billiger handlingen. 1 betyder, at du slet ikke billiger dem, 10 betyder, at du i høj grad billiger dem\nAktiv dødshjælp\nSvarmuligheder:\na. Aldrig\nb. 2\nc. 3\nd. 4\ne. 5\nf. 6\ng. 7\nh. 8\ni. 9\nj. Altid",
}
```

```json
{
    "question_id": "C041",
    "text": "Hvor enig eller uenig er du i følgende udsagn?\nArbejde kommer først, også selv om det betyder mindre fritid.\nSvarmuligheder:\na. Helt enig\nb. Enig\nc. Hverken enig eller uenig\nd. Uenig\ne. Helt uenig"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 0
- Prefix prompt:

  ```text
  Følgende er multiple choice spørgsmål (med svar).
  ```

- Base prompt template:

  ```text
  Spørgsmål: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}
  (...)
  k. {option_k}
  Svar: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Spørgsmål: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}
  (...)
  k. {option_k}

  Besvar ovenstående spørgsmål ved at svare med 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
  'i', 'j' eller 'k', og intet andet.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset valeu-da
```
