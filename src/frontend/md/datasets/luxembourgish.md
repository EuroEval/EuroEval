# 🇱🇺 Luxembourgish

Luxembourgish (Lëtzebuergesch) is a West Germanic language spoken primarily in
Luxembourg. It is a Moselle Franconian dialect with significant French and German
influences. Despite being the national language of Luxembourg, it has limited digital
resources compared to other European languages.

## Sentiment Classification

### ltzGLUE-SA

This dataset was published in [this paper](https://arxiv.org/abs/2604.17976) and
contains Luxembourgish texts annotated with sentiment labels (negative, neutral,
positive). The data is collected from Luxembourgish social media posts and news website
comments, covering various domains including politics, culture, and daily life.

The original ltzGLUE-SA dataset contains 3,022 / 597 / 926 samples for training,
validation and testing, respectively, with stratified sampling to maintain label
balance. Due to limited test data, we use all 926 test samples and cap training at 1,024
samples.

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

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

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

### ltzGLUE-NER

This dataset was published in [this paper](https://arxiv.org/abs/2604.17976) and
contains Luxembourgish news articles annotated with BIO-formatted entity tags for
person, location, organization, and miscellaneous entities. The articles are sourced
from major Luxembourgish news outlets including Luxemburger Wort, RTL Lëtzebuerg,
paperjam.lu, today.lu, and wort.lu.

The original ltzGLUE-NER dataset contains 27,245 / 3,327 / 3,821 samples for training,
validation and testing, respectively. We apply the standard EuroEval cap of 1,024 / 256
/ 2,048 samples using random sampling (stratification not applicable for sequence
tagging with multiple token-level labels per sample).

Here are a few examples from the training split:

```json
{
  "tokens": [
    "D'",
    "Regierung",
    "huet",
    "e",
    "neie",
    "Plang",
    "fir",
    "d'",
    "Stad",
    "Lëtzebuerg",
    "."
  ],
  "labels": ["O", "B-ORG", "O", "O", "O", "O", "O", "O", "O", "B-LOC", "O"]
}
```

```json
{
  "tokens": [
    "Den",
    "Premierminister",
    "Xavier",
    "Bettel",
    "huet",
    "d'",
    "Rede",
    "gehalen",
    "."
  ],
  "labels": ["O", "O", "B-PER", "I-PER", "O", "O", "O", "O", "O"]
}
```

```json
{
  "tokens": ["D'", "EU", "huet", "neie", "Regelen", "fir", "d'", "Land", "agesat", "."],
  "labels": ["O", "B-ORG", "O", "O", "O", "O", "O", "O", "O", "O"]
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Folgend si Sätz a JSON-Wierderbicher mat den benannten Entitéiten, déi am Satz virkommen.
  ```

- Base prompt template:

  ```text
  Saz: {text}
  Benannt Entitéiten: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Saz: {text}

  Identifizéiert déi benannt Entitéiten am Saz. Dir sollt dës als JSON-Wierderbuch mat de Schlësselen 'persoun', 'plaz', 'organisatioun' an 'divers' ausginn. D'Wäerter solle Lëschte vun de benannten Entitéite vun deem Typ sinn, genee sou wéi se am Saz virkommen.
  ```

- Label mapping:
  - `B-PER` ➡️ `persoun`
  - `I-PER` ➡️ `persoun`
  - `B-LOC` ➡️ `plaz`
  - `I-LOC` ➡️ `plaz`
  - `B-ORG` ➡️ `organisatioun`
  - `I-ORG` ➡️ `organisatioun`
  - `B-MISC` ➡️ `divers`
  - `I-MISC` ➡️ `divers`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ltzglue-ner
```

## Linguistic Acceptability

### ltzGLUE-LA

This dataset was published in [this paper](https://arxiv.org/abs/2604.17976).
Luxembourgish sentences must be classified as grammatically correct or incorrect. The
dataset includes naturally occurring text from Luxembourgish news outlets (Luxemburger
Wort, RTL Lëtzebuerg, paperjam.lu, today.lu, and wort.lu) as well as systematically
corrupted examples with various error types (word order, agreement, case, etc.).

The original ltzGLUE-LA dataset contains 14,678 / 2,094 / 4,045 samples for training,
validation and testing, with balanced classes. We apply the standard EuroEval cap of
1,024 / 256 / 2,048 samples using stratified sampling on the correct/incorrect labels.

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

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

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

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ltzglue-la-binary
```

### Unofficial: ltzGLUE-LA-multi

Multi-class variant of the ltzGLUE-LA linguistic acceptability task where models must
identify the specific error type in addition to detecting incorrectness. This dataset is
marked as **unofficial** due to poor zero-shot performance (0.00% MCC with GPT-4o-mini),
suggesting the task may be too difficult or the prompt template needs refinement. It is
intended for diagnostic purposes only. Error categories include word order, subject-verb
agreement, case marking, determiner-noun agreement, and other grammatical violations.
The data includes naturally occurring text from Luxembourgish news outlets (Luxemburger
Wort, RTL Lëtzebuerg, paperjam.lu, today.lu, and wort.lu) as well as systematically
corrupted examples.

The original dataset contains 14,678 / 2,094 / 4,045 samples for training, validation
and testing. We apply the standard EuroEval cap of 1,024 / 256 / 2,048 samples using
stratified sampling on the error type labels.

Here are a few examples from the training split:

```json
{
  "text": "Den Apel fält net wäit vum Bam.",
  "label": "correct"
}
```

```json
{
  "text": "D'Kand spillt am Gaart.",
  "label": "agreement"
}
```

```json
{
  "text": "Hien gëtt an d'Schoul.",
  "label": "correct"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Folgend sinn Sätz. Bestëmmt ob se grammatesch korrekt sinn, oder identifizéiert de Feeler Typ.
  ```

- Base prompt template:

  ```text
  Saz: {text}
  Äntwert: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Saz: {text}

  Bestëmmt ob de Saz grammatesch korrekt ass (äntwert 'correct'), oder identifizéiert de Feeler Typ: 'word_order' (falsch Wuert-Reiefolleg), 'agreement' (Subject-Verb oder Determiner-Noun Stëmmung net korrekt), 'morphology' (falsch Wortform), oder 'other'. Äntwert nëmme mat engem vun dësen Etiketten: correct, word_order, agreement, morphology, other.
  ```

- Label mapping: (direct mapping to Luxembourgish error type names)

This dataset is marked as **unofficial** and is used for diagnostic purposes only.

```bash
euroeval --model <model-id> --dataset ltzglue-la-multi
```

## Natural Language Inference

### ltzGLUE-RTE

This dataset was published in [this paper](https://arxiv.org/abs/2604.17976). Given a
premise and hypothesis pair from Luxembourgish text sources, models must determine if
the hypothesis is entailed by or contradicts the premise (binary classification).

The original ltzGLUE-RTE dataset contains 1,877 / 197 / 626 samples for training,
validation and testing, respectively. Due to limited validation and test data, we use
all available samples (197 validation, 626 test) and cap training at 1,024 samples using
stratified sampling on the entailment/contradiction labels.

Here are a few examples from the training split:

```json
{
  "text": "Premise: D'Regierung huet e neie Plang presentéiert.\nHypothesis: Et gëtt eng nei Presentatioun vun der Regierung.",
  "label": "entailment"
}
```

```json
{
  "text": "Premise: Et regnet zu Lëtzebuerg.\nHypothesis: D'Sonn scheint zu Lëtzebuerg.",
  "label": "contradiction"
}
```

```json
{
  "text": "Premise: De Premierminister huet eng Rede gehalen.\nHypothesis: De Xavier Bettel huet e Virdrag gehalen.",
  "label": "entailment"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Fir dës Puer ass uginn ob d'Hypothees d'Premisse follegt oder widderleet.
  ```

- Base prompt template:

  ```text
  Text: {text}
  Relatioun: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Text: {text}

  Bestëmmt ob déi zweet Aussag aus der éischter follegt oder hir widdersträit.
  Äntwert nëmme mat 'folgerung' oder 'widdersträit'.
  ```

- Label mapping:
  - `entailment` ➡️ `folgerung`
  - `contradiction` ➡️ `widdersträit`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ltzglue-rte
```

## Reading Comprehension

### MultiWikiQA-lb

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains question-answer pairs derived from
[Luxembourgish Wikipedia](https://lb.wikipedia.org/) articles about Luxembourg and
Luxembourgish culture, history, and geography. The questions and answers are generated
using large language models based on the Wikipedia content.

The dataset is extracted from the
[alexandrainst/multi-wiki-qa](https://huggingface.co/datasets/alexandrainst/multi-wiki-qa)
dataset (subset "lb"), containing 5,003 samples. We apply the standard EuroEval cap of
1,024 / 256 / 2,048 samples for evaluation splits.

Here are a few examples from the training split:

```json
{
  "question": "Wéi eng Sprooch gëtt zu Lëtzebuerg geschwat?",
  "answer": "Lëtzebuergesch ass d'Nationalsprooch vun Lëtzebuerg.",
  "context": "Lëtzebuergesch ass eng westgermanesch Sprooch déi haaptsächlech zu Lëtzebuerg geschwat gëtt."
}
```

```json
{
  "question": "Wer ass de aktuelle Premierminister vu Lëtzebuerg?",
  "answer": "Den Xavier Bettel ass de Premierminister.",
  "context": "Den Xavier Bettel war de Premierminister vu Lëtzebuerg zënter 2013."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Folgend sinn Texter mat dozou gehéiereg Froen an Äntwerten.
  ```

- Base prompt template:

  ```text
  Text: {text}
  Fro: {question}
  Äntwert mat maximal 3 Wierder: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Text: {text}

  Äntwert d'Fro iwwer den Text hei uewen mat maximal 3 Wierder.

  Fro: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-lb
```

## Text Classification

### ltzGLUE-HC

This dataset was published in [this paper](https://arxiv.org/abs/2604.17976). Given a
news headline and its article text from Luxembourgish news outlets (Luxemburger Wort,
RTL Lëtzebuerg, paperjam.lu, today.lu, and wort.lu), models must predict whether the
headline correctly represents the article content.

The original ltzGLUE-HC dataset contains 20,716 / 2,960 / 5,919 samples for training,
validation and testing, respectively. We apply the standard EuroEval cap of 1,024 / 256
/ 2,048 samples using stratified sampling on the yes/no labels.

Here are a few examples from the training split:

```json
{
  "text": "Neie Plang fir d'Stad Lëtzebuerg presentéiert | D'Regierung huet e neie Plang fir d'Entwécklung vun der Stad Lëtzebuerg presentéiert, deen ënner anerem méi Wunnengen a gréi Zonen virgesäit.",
  "label": "yes"
}
```

```json
{
  "text": "Et regnet de ganzen Dag | D'Sonn scheint zu Lëtzebuerg bei héijen Temperaturen.",
  "label": "no"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Folgend sinn Iwwerschrëften an ob den Artikel dës Behaaptung bestätegt.
  ```

- Base prompt template:

  ```text
  Iwwerschrëft: {text}
  Bestätegt: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Iwwerschrëft: {text}

  Bestëmmt ob den Artikel d'Behaaptung an der Iwwerschrëft bestätegt. Äntwert nëmme mat 'yes' oder 'no'.
  ```

- Label mapping: (direct mapping, no translation)

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ltzglue-hc
```

### ltzGLUE-TC

This dataset was published in [this paper](https://arxiv.org/abs/2604.17976) and
contains Luxembourgish news articles categorized by topic. The articles are sourced from
major Luxembourgish news outlets including Luxemburger Wort, RTL Lëtzebuerg,
paperjam.lu, today.lu, and wort.lu, covering politics, economy, sports, and culture.

The original dataset contains 9,932 / 1,240 / 1,245 samples for training, validation and
testing. We cap training at 1,024 samples using stratified sampling on the topic labels,
and use all available test data (1,245 samples, below the 2,048 target).

Here are a few examples from the training split:

```json
{
  "text": "Technologie: D'Firma huet e neie Smartphone presentéiert.",
  "label": "technology"
}
```

```json
{
  "text": "Business: D'Bourse hannertléisst Rekuerdresultater.",
  "label": "business"
}
```

```json
{
  "text": "Kultur: Den Jazzfestival lockt Dausende vu Visiteur un.",
  "label": "culture"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Folgend sinn Noriichten-Artikelen an hir Themen.
  ```

- Base prompt template:

  ```text
  Artikel: {text}
  Thema: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Artikel: {text}

  Klassifizéiert d'Thema vum Artikel. Äntwert nëmme mat engem vun dësen Etiketten: technology, business, culture, animals, sports.
  ```

- Label mapping: (direct mapping)

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ltzglue-tc
```

### ltzGLUE-ID

Intent detection dataset from the ltzGLUE benchmark, containing Luxembourgish queries
annotated with intent categories such as weather lookup, restaurant booking, music
playback, alarm setting, and creative work search. The queries are synthetically
generated to cover common voice assistant and search engine intents.

The original dataset contains 53 validation and 66 test samples with no training data.
We combine these and create train/val/test splits (59/17/41 samples) without
stratification due to some classes having very few samples.

Here are a few examples from the training split:

```json
{
  "text": "Wéi ass d'Wieder zu Lëtzebuerg?",
  "label": "weather find"
}
```

```json
{
  "text": "Spill mir e Lidd vun den Beatles.",
  "label": "playmusic"
}
```

```json
{
  "text": "Reservéier en Dësch an engem Restaurant.",
  "label": "bookrestaurant"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Folgend si Benotzerufruffen an hir Intentiounen.
  ```

- Base prompt template:

  ```text
  Ufruff: {text}
  Intentioun: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Ufruff: {text}

  Identifizéiert d'Intentioun vum Benotzer. Äntwert nëmme mat engem vun dësen Etiketten: addtoplaylist, alarm cancel alarm, alarm set alarm, alarm show alarms, bookrestaurant, playmusic, ratebook, reminder set reminder, searchcreativework, searchscreeningevent, weather find.
  ```

- Label mapping: (direct mapping to intent names)

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ltzglue-id
```
