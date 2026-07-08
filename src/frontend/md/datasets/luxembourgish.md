# 🇱🇺 Luxembourgish

## Linguistic Acceptability

### ltzGLUE-LA

Binary linguistic acceptability task from the ltzGLUE benchmark, introduced in
[this paper](https://arxiv.org/abs/2604.17976). Luxembourgish sentences must be
classified as grammatically correct or incorrect. The dataset includes naturally
occurring text as well as systematically corrupted examples.

The original ltzGLUE-LA dataset contains 14,678 / 2,094 / 4,045 samples
for training, validation and testing, with balanced classes.

Here are a few examples from the training split:

```json
{
  "text": "Den Apel fält net wäit vum Bam.",
  "label": "correct"
}
```

```json
{
  "text": "Ech ginn hautoff an d'Schoul.",
  "label": "incorrect"
}
```

```json
{
  "text": "Mir wänken eise léiwe Gäscht.",
  "label": "correct"
}
```

When evaluating generative models:

- Number of few-shot examples: 12
- Prefix prompt:
  ```text
  Fir dëse Satz ass uginn ob e grammatesch korrekt oder net ass.
  ```
- Base prompt template:
  ```text
  Saz: {text}
  Grammatesch korrekt: {label}
  ```
- Instruction-tuned prompt template:
  ```text
  Saz: {text}
  
  Bestëmmt ob de Saz grammatesch korrekt oder net ass. Äntwert mat 'jo' wann de Saz korrekt ass, an 'nee' wann en net korrekt ass.
  ```
- Label mapping:
  - `correct` ➡️ `jo`
  - `incorrect` ➡️ `nee`

```bash
euroeval --model <model-id> --dataset ltzglue-la-binary
```

### Unofficial: ltzGLUE-LA (Multi-class)

Multi-class version of the ltzGLUE linguistic acceptability task, where
grammatical errors are classified by error type. The dataset contains
14,678 / 2,094 / 4,045 samples for training, validation and testing.
This provides more fine-grained evaluation of Luxembourgish language understanding.

The error types include various categories such as word order violations, agreement
errors, and morphological errors.

```bash
euroeval --model <model-id> --dataset ltzglue-la-multi
```

### Unofficial: ScaLA-lb

Linguistic acceptability dataset created from the
[UD Luxembourgish-LuxBank treebank](https://github.com/UniversalDependencies/UD_Luxembourgish-LuxBank)
using the [ScaLA algorithm](https://aclanthology.org/2023.nodalida-1.20/). Correct
sentences are systematically corrupted by word swaps or deletions to create
grammatically incorrect examples.

**Note:** The current UD Luxembourgish treebank only contains approximately 100
sentences (translated Cairo Cycling examples). A comprehensive ScaLA-lb dataset
requires additional Luxembourgish text sources or expansion of the treebank.

```bash
euroeval --model <model-id> --dataset scala-lb
```

## Named Entity Recognition

### ltzGLUE-NER

Luxembourgish Named Entity Recognition dataset from the ltzGLUE benchmark.
Annotated with standard entity types following CoNLL-2003 format:

- `PER` (Person)
- `LOC` (Location)
- `ORG` (Organization)
- `MISC` (Miscellaneous)

The dataset contains 27,245 / 3,327 / 3,821 samples for training, validation
and testing.

Here are a few examples from the training split:

```json
{
  "tokens": ["Hien", "dozéiert", "op", "der", "Universitéit", "vun", "Innsbruck", "an", "Éisträich", ",", "a", "praktizéiert", "do", "am", "Fraendepartement", "vun", "den", "Universitéitskliniken", "."],
  "labels": ["O", "O", "O", "O", "B-ORG", "O", "B-LOC", "O", "B-LOC", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]
}
```

```json
{
  "tokens": ["De", "Roland", "Meyer", "ass", "bestuet", "a", "Papp", "vun", "zwéi", "Jongen", "."],
  "labels": ["O", "B-PER", "I-PER", "O", "O", "O", "O", "O", "O", "O", "O"]
}
```

When evaluating generative models:

- Number of few-shot examples: 8
- Prefix prompt:
  ```text
  Fir de Saz ass eng JSON-Tabell mat den enthalte Entitéiten.
  ```
- Base prompt template:
  ```text
  Saz: {text}
  Entitéiten: {label}
  ```
- Instruction-tuned prompt template:
  ```text
  Saz: {text}
  
  Identifizéiert déi genannt Entitéiten am Saz. Output als JSON-Tabell mat de Schlësselen 'persoun', 'plaz', 'organisatioun' an 'divers'.
  ```
- Label mapping:
  - `B-PER` / `I-PER` ➡️ `persoun`
  - `B-LOC` / `I-LOC` ➡️ `plaz`
  - `B-ORG` / `I-ORG` ➡️ `organisatioun`
  - `B-MISC` / `I-MISC` ➡️ `divers`

```bash
euroeval --model <model-id> --dataset ltzglue-ner
```

## Natural Language Inference

### ltzGLUE-RTE

Recognising Textual Entailment from ltzGLUE. Binary classification task
determining if a hypothesis follows from a premise:

- `entailment` (0): The hypothesis follows from the premise
- `contradiction` (1): The hypothesis does not follow / contradicts the premise

The dataset contains 1,877 / 197 / 626 samples for training, validation
and testing with stratified sampling.

Here are a few examples from the training split:

```json
{
  "premise": "Keng Massenzerstéierungswaffen am Irak fonnt.",
  "hypothesis": "Massenzerstéierungswaffen am Irak fonnt.",
  "label": "entailment"
}
```

```json
{
  "premise": "Eng Plaz vu Trauer, nodeems de Poopst Johannes Paul II gestuerwen ass, ass eng Plaz fir ze feieren ginn, wéi réimesch-kathoulesch Gleeweger sech am Zentrum vu Chicago versammelt hunn, fir d'Installatioun vum neie Poopst Benedikt XVI ze markéieren.",
  "hypothesis": "De Poopst Benedikt XVI. ass den neie Leader vun der réimesch-kathoulescher Kierch.",
  "label": "contradiction"
}
```

When evaluating generative models:

- Number of few-shot examples: 8
- Prefix prompt:
  ```text
  Fir dëse Sazpaar ass uginn ob d'Hypothees aus der Premiss follegt.
  ```
- Base prompt template:
  ```text
  Premiss: {premise}
  Hypothees: {hypothesis}
  Logesch Relatioun: {label}
  ```
- Instruction-tuned prompt template:
  ```text
  Premiss: {premise}
  
  Hypothees: {hypothesis}
  
  Bestëmmt d'logesch Relatioun ant der Premiss an der Hypothees. Äntwert mat 'folgerung', 'widdersträit' oder 'neutral'.
  ```
- Label mapping:
  - `entailment` ➡️ `folgerung`
  - `contradiction` ➡️ `widdersträit`
  - `neutral` ➡️ `neutral`

```bash
euroeval --model <model-id> --dataset ltzglue-rte
```

## Sentiment Classification

### ltzGLUE-SA

Sentiment analysis dataset from the ltzGLUE benchmark, introduced in
[this paper](https://arxiv.org/abs/2604.17976). Contains Luxembourgish texts
annotated with sentiment labels (negative, neutral, positive), covering various
domains including social media posts and news comments.

The original ltzGLUE-SA dataset contains 3,022 / 597 / 926 samples
for training, validation and testing, respectively, with stratified sampling to
maintain label balance.

Here are a few examples from the training split:

```json
{
  "text": "En klengen Cadeau fir den Patron: Wann no der Proufzäit en CDI ënnerschriwwen gëtt kritt en Soziallaaschten vun der Proufzäit erëm.",
  "label": "positive"
}
```

```json
{
  "text": "A bis Oktober kënnt do och net méi vill no. Jidder Politiker weess, datt een an engem Land vun iwwerduerchschnëttlech ville Proprietären mat enger Hausse vun der Grondsteier kee Walkampf gewanne kann.",
  "label": "negative"
}
```

```json
{
  "text": "E Politmonitor, an deem bal d'Hallschent vun de Leit, 43% fir genee ze sinn, awer zum Ausdrock bruecht hunn, datt si den Zoustand vun der Lëtzebuerger Gesellschaft als ongerecht empfannen.",
  "label": "neutral"
}
```

When evaluating generative models:

- Number of few-shot examples: 12
- Prefix prompt:
  ```text
  Fir dësen Text ass d'Sentiment uginn, a kann 'negativ', 'neutral' oder 'positiv' sinn.
  ```
- Base prompt template:
  ```text
  Text: {text}
  Sentiment: {label}
  ```
- Instruction-tuned prompt template:
  ```text
  Text: {text}
  
  Klassifizéiert d'Sentiment vum Text. Äntwert nëmme mat 'negativ', 'neutral' oder 'positiv'.
  ```
- Label mapping:
  - `negative` ➡️ `negativ`
  - `neutral` ➡️ `neutral`
  - `positive` ➡️ `positiv`

```bash
euroeval --model <model-id> --dataset ltzglue-sa
```

## Reading Comprehension

### MultiWikiQA-lb

This dataset was published in
[this paper](https://doi.org/10.48550/arXiv.2509.04111) and contains Wikipedia
articles with LLM-generated questions and answers in 300+ languages, including
Luxembourgish.

The original dataset consists of 5,003 samples in a single train split. We use the
entire split for test evaluation, as no validation split was provided in the original
release.

Here are a few examples from the dataset:

```json
{
    "context": "Lëtzebuerg, offiziell d'Groussherzogtum Lëtzebuerg, ass e klengen am Floss vun Europa. Et ass ee vun de siechzéng Länner déi dono der Opléisung vun der Sowjetunioun an Däitschland an Éisträich bäigetruede sinn. Lëtzebuerg läit tëscht Däitschland, Frankräich an der Belsch, an ass ee vun de klengste souverän Staaten an Europa.",
    "question": "Wou läit Lëtzebuerg?",
    "answers": {
        "answer_start": [97],
        "text": ["tëscht Däitschland, Frankräich an der Belsch"]
    }
}
```

```json
{
    "context": "De Jean-Claude Juncker, gebuer den 9. Dezember 1954 zu Réiden op der Atert, ass e lëtzebuergesche Politiker (CSV). Hie war vum 1. Januar 2014 bis den 30. November 2019 President vun der Europäescher Kommissioun. Virdru war hie vun 1995 bis 2013 Premierminister vu Lëtzebuerg an huet och d'Amt vum Finanzminister ausgeëbt.",
    "question": "Wéini ass de Jean-Claude Juncker gebuer?",
    "answers": {
        "answer_start": [41],
        "text": ["den 9. Dezember 1954"]
    }
}
```

```json
{
    "context": "D'Escher Alzette ass e Floss zu Lëtzebuerg. Si huet hir Quell zu Bascharage an der Gemeng Käerch op enger Héicht vu 320 Meter. D'Uertschafte laanscht d'Alzette sinn: Esch-Uelzecht, Fréiseng, Rëmeleng, Péiteng, Mënsbech, Lëtzebuerg-Kierchbierg, an d'Mënsbecht. D'Escher Alzette fléisst bei der Stad Lëtzebuerg an d'Uelzecht.",
    "question": "Wou huet d'Escher Alzette hir Quell?",
    "answers": {
        "answer_start": [37],
        "text": ["zu Bascharage an der Gemeng Käerch"]
    }
}
```

When evaluating generative models:

- Number of few-shot examples: 4
- Prefix prompt:
  ```text
  Fir dësen Text ass eng Fro an d'Äntwert uginn.
  ```
- Base prompt template:
  ```text
  Text: {text}
  Fro: {question}
  Äntwert: {label}
  ```
- Instruction-tuned prompt template:
  ```text
  Text: {text}
  
  Beäntwert déi folleg Fro iwwer den Text hei driwwer a maximal 3 Wierder.
  
  Fro: {question}
  ```

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-lb
```

## Summarisation

### LuxGen-Summ

Luxembourgish summarisation dataset from the
[LuxGen benchmark](https://arxiv.org/abs/2412.09415). Contains Luxembourgish
documents with human-written summaries, testing the model's ability to condense
information while preserving key content.

**TODO:** Dataset creation pending - requires extraction from the LuxGen benchmark
or creation from Luxembourgish articles with summaries.

Target format:
- `text`: Full Luxembourgish article or document
- `summary`: Human-written Luxembourgish summary

This is a placeholder dataset pending creation from the LuxGen benchmark.

```bash
euroeval --model <model-id> --dataset luxgen-summ
```

## Text Classification

### ltzGLUE-HC

Headline Classification from ltzGLUE. Binary task determining if a headline
is appropriate for a given news article. Tests the model's ability to
understand document-level semantics and headline-article coherence.

The dataset consists of article-headline pairs with binary labels (yes/no for
appropriateness). The dataset contains 20,716 / 2,960 / 5,919 samples
for training, validation and testing.

Here are a few examples:

```json
{
  "text": "D'Regierung huet e neie Budget fir 2024 presentéiert. D'Steieren bleiwen onverännert. | Regierung presentéiert Budget 2024: Steieren onverännert",
  "label": "yes"
}
```

```json
{
  "text": "De FCSV huet säi nächsten Match gewonnen. D'Ekipp steet elo un der Spëtzt vun der Tabelle. | Lëtzebuerg gewënnt Olympesch Goldmedail",
  "label": "no"
}
```

```bash
euroeval --model <model-id> --dataset ltzglue-hc
```

### Unofficial: ltzGLUE-ID

Intent Detection from ltzGLUE. Multi-class classification of sentence intent,
testing pragmatic understanding. The dataset contains 53 / 66 samples for
validation and testing (no training data provided). Intent categories include:

- `question`: Asking for information
- `statement`: Making a declarative statement
- `command`: Giving an instruction or request
- `exclamation`: Expressing strong emotion

```bash
euroeval --model <model-id> --dataset ltzglue-id
```

### Unofficial: ltzGLUE-TC

Topic Classification from ltzGLUE. Multi-class classification of news articles
into topic categories such as politics, sports, culture, economy, and technology.
The dataset contains 9,932 / 1,240 / 1,245 samples for training, validation and
testing. Tests the model's ability to identify document-level topics.

```bash
euroeval --model <model-id> --dataset ltzglue-tc
```

## Knowledge

### Unofficial: Luxembourgish Citizenship Tests

Luxembourgish knowledge dataset based on citizenship test questions from
[vivre-ensemble.lu](https://old.vivre-ensemble.lu/). Multiple-choice questions
test knowledge about Luxembourgish society, history, geography, and civic
institutions.

**TODO:** Dataset creation pending - requires extraction and curation from
the citizenship test materials.

```bash
euroeval --model <model-id> --dataset lux-citizenship
```

---

**Note:** All ltzGLUE datasets are sourced from the ltzGLUE benchmark
(arxiv.org/abs/2604.17976). Primary datasets (ltzGLUE-LA, ltzGLUE-NER, ltzGLUE-RTE,
ltzGLUE-SA, ltzGLUE-HC) are marked as official. Dataset creation scripts are
available in `src/scripts/dataset_creation/`. To upload the datasets, clone the
ltzGLUE repository with LFS support and run the scripts with uv.
