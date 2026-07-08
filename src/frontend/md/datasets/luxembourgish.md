# 🇱🇺 Luxembourgish

## Sentiment Classification

### Unofficial: ltzGLUE-SA

This dataset is part of the ltzGLUE benchmark for Luxembourgish, introduced in
[this paper](https://arxiv.org/abs/2604.17976). It contains Luxembourgish texts
annotated with sentiment labels (negative, neutral, positive), covering various
domains including social media posts and news comments.

The original ltzGLUE-SA dataset is processed into 1,024 / 256 / ~2,000 samples
for training, validation and testing, respectively, with stratified sampling to
maintain label balance.

Here are a few examples from the training split:

```json
{
  "text": "Dat ass wierklech eng fantastesch Noriicht!",
  "label": "positive"
}
```

```json
{
  "text": "Ech sinn net d'accord mat dëser Meenung.",
  "label": "negative"
}
```

```json
{
  "text": "De rechten ass fir muer um 14 Auer festgeluecht.",
  "label": "neutral"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information):

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

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ltzglue-sa
```

## Named Entity Recognition

### Unofficial: ltzGLUE-NER

Luxembourgish Named Entity Recognition dataset from the ltzGLUE benchmark.
Annotated with standard entity types following CoNLL-2003 format:

- `PER` (Person)
- `LOC` (Location)
- `ORG` (Organization)
- `MISC` (Miscellaneous)

The dataset is sampled into 1,024 / 256 / ~2,000 samples for training, validation
and testing.

Here are a few examples from the training split:

```json
{
  "tokens": ["De", "Max", "ass", "zu", "Lëtzebuerg", "gebuer", "."],
  "labels": ["O", "B-PER", "O", "O", "B-LOC", "O", "O"]
}
```

```json
{
  "tokens": ["D'", "CSV", "huet", "d'", "Walen", "gewonnen", "."],
  "labels": ["O", "B-ORG", "O", "O", "O", "O", "O"]
}
```

```json
{
  "tokens": ["D'", "Sauer", "fléisst", "duerch", "Lëtzebuerg", "an", "Däitschland", "."],
  "labels": ["O", "B-LOC", "O", "O", "B-LOC", "O", "B-LOC", "O"]
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

## Linguistic Acceptability

### Unofficial: ltzGLUE-LA (Binary)

Part of the ltzGLUE benchmark. Binary classification task where Luxembourgish
sentences must be classified as grammatically correct or incorrect. The dataset
includes naturally occurring text as well as artificially corrupted examples.

The dataset is sampled into 1,024 / 256 / ~2,000 samples for training, validation
and testing, with balanced classes.

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
grammatical errors are classified by error type. This provides more fine-grained
evaluation of Luxembourgish language understanding.

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
  
  Bestëmmt ob de Saz grammatesch korrekt ass. Äntwert mat 'jo' fir korrekt, 'nee' fir net korrekt.
  ```

```bash
euroeval --model <model-id> --dataset scala-lb
```

## Natural Language Inference

### Unofficial: ltzGLUE-RTE

Recognising Textual Entailment from ltzGLUE. Three-way classification task
determining the logical relationship between a premise and hypothesis:

- `entailment`: The hypothesis follows from the premise
- `contradiction`: The hypothesis contradicts the premise
- `neutral`: Neither entailment nor contradiction

The dataset is sampled into 1,024 / 256 / ~2,000 samples for training, validation
and testing with stratified sampling.

Here are a few examples from the training split:

```json
{
  "premise": "De Premierminister huet eng Ried an der Chamber gehalen.",
  "hypothesis": "Eng Ried gouf an der Chamber gehalen.",
  "label": "entailment"
}
```

```json
{
  "premise": "D'Chamber ass an de grousse Vakanz gefuer.",
  "hypothesis": "D'Chamber schafft weiderhin.",
  "label": "contradiction"
}
```

```json
{
  "premise": "Lëtzebuerg ass e Member vun der EU.",
  "hypothesis": "Lëtzebuerg huet den Euro als Wärung.",
  "label": "neutral"
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

## Reading Comprehension

### MultiWikiQA-lb

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages, including Luxembourgish.

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

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-lb
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

## Summarisation

### LuxGen-Summ

Luxembourgish summarisation dataset from the
[LuxGen benchmark](https://arxiv.org/abs/2412.09415). The dataset contains
Luxembourgish documents with human-written summaries, testing the model's
ability to condense information while preserving key content.

**TODO:** This dataset documentation needs to be completed once the LuxGen
benchmark dataset is extracted and processed. Final split sizes and examples
will be added when the dataset is created.

Target format:
- `text`: Full Luxembourgish article or document
- `summary`: Human-written Luxembourgish summary

Target splits: 1,024 / 256 / 2,048 samples for train, validation and test.

```bash
euroeval --model <model-id> --dataset luxgen-summ
```

## Text Classification

### Unofficial: ltzGLUE-HC

Headline Classification from ltzGLUE. Binary task determining if a headline
is appropriate for a given news article. This tests the model's ability to
understand document-level semantics and headline-article coherence.

The dataset consists of article-headline pairs with binary labels (yes/no for
appropriateness).

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
testing the model's pragmatic understanding. Intent categories include:

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
The dataset tests the model's ability to identify document-level topics.

```bash
euroeval --model <model-id> --dataset ltzglue-tc
```

---

**Note:** All ltzGLUE datasets are sourced from the ltzGLUE benchmark
(arxiv.org/abs/2604.17976). Dataset creation scripts are available in
`src/scripts/dataset_creation/`. To upload the datasets, clone the ltzGLUE
repository with LFS support and run the scripts with uv.
