# 🇱🇺 Luxembourgish

Luxembourgish (Lëtzebuergesch) is a West Germanic language spoken primarily in
Luxembourg. It is a Moselle Franconian dialect with significant French and German
influences. Despite being the national language of Luxembourg, it has limited digital
resources compared to other European languages.

## Sentiment Classification

### ltzGLUE-SA

This dataset was published in
[this paper](https://arxiv.org/abs/2604.17976) and contains Luxembourgish texts
annotated with sentiment labels (negative, neutral, positive). The data is collected
from Luxembourgish social media posts and news website comments, covering various
domains including politics, culture, and daily life.

The original ltzGLUE-SA dataset contains 3,022 / 597 / 926 samples
for training, validation and testing, respectively, with stratified sampling to
maintain label balance. Due to limited test data, we use all 926 test samples and
cap training at 1,024 samples.

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

This dataset was published in
[this paper](https://arxiv.org/abs/2604.17976) and contains Luxembourgish news articles
annotated with BIO-formatted entity tags for person, location, organization, and
miscellaneous entities. The articles are sourced from major Luxembourgish news outlets
including Luxemburger Wort, RTL Lëtzebuerg, paperjam.lu, today.lu, and wort.lu.

The original ltzGLUE-NER dataset contains 27,245 / 3,327 / 3,821 samples
for training, validation and testing, respectively. We apply the standard EuroEval
cap of 1,024 / 256 / 2,048 samples using random sampling (stratification not applicable
for sequence tagging with multiple token-level labels per sample).

Here are a few examples from the training split:

```json
{
  "tokens": ["D'", "Regierung", "huet", "e", "neie", "Plang", "fir", "d'", "Stad", "Lëtzebuerg", "."],
  "labels": ["O", "B-ORG", "O", "O", "O", "O", "O", "O", "O", "B-LOC", "O"]
}
```

```json
{
  "tokens": ["Den", "Premierminister", "Xavier", "Bettel", "huet", "d'", "Rede", "gehalen", "."],
  "labels": ["O", "O", "B-PER", "I-PER", "O", "O", "O", "O", "O"]
}
```

```json
{
  "tokens": ["D'", "EU", "huet", "neie", "Regelen", "fir", "d'", "Land", " agesat", "."],
  "labels": ["O", "B-ORG", "O", "O", "O", "O", "O", "O", "O", "O"]
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Fir dëse Satz gëtt et eng JSON-Liste mat den Entitéiten.
  ```

- Base prompt template:

  ```text
  Saz: {text}
  Entitéiten: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Saz: {text}

  Identifizéiert déi benannt Entitéiten am Satz. Output als JSON-Liste mat den Schlësselen 'persoun', 'plaz', 'organesatioun' an 'divers'.
  ```

- Label mapping:
  - `B-PER` ➡️ `persoun`
  - `I-PER` ➡️ `persoun`
  - `B-LOC` ➡️ `plaz`
  - `I-LOC` ➡️ `plaz`
  - `B-ORG` ➡️ `organesatioun`
  - `I-ORG` ➡️ `organesatioun`
  - `B-MISC` ➡️ `divers`
  - `I-MISC` ➡️ `divers`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ltzglue-ner
```

## Linguistic Acceptability

### ltzGLUE-LA

This dataset was published in
[this paper](https://arxiv.org/abs/2604.17976). Luxembourgish sentences must be
classified as grammatically correct or incorrect. The dataset includes naturally
occurring text from Luxembourgish news outlets (Luxemburger Wort, RTL Lëtzebuerg,
paperjam.lu, today.lu, and wort.lu) as well as systematically corrupted examples
with various error types (word order, agreement, case, etc.).

The original ltzGLUE-LA dataset contains 14,678 / 2,094 / 4,045 samples
for training, validation and testing, with balanced classes. We apply the standard
EuroEval cap of 1,024 / 256 / 2,048 samples using stratified sampling on the
correct/incorrect labels.

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
euroeval --model <model-id> --dataset ltzglue-la
```

### ltzGLUE-LA-multi

This dataset was published in
[this paper](https://arxiv.org/abs/2604.17976). Multi-class variant of the linguistic
acceptability task where models must identify the specific error type in addition to
detecting incorrectness. Error categories include word order, subject-verb agreement,
case marking, determiner-noun agreement, and other grammatical violations. The data
includes naturally occurring text from Luxembourgish news outlets (Luxemburger Wort,
RTL Lëtzebuerg, paperjam.lu, today.lu, and wort.lu) as well as systematically corrupted
examples.

The original dataset contains 14,678 / 2,094 / 4,045 samples
for training, validation and testing. We apply the standard EuroEval cap of
1,024 / 256 / 2,048 samples using stratified sampling on the error type labels.

Here are a few examples from the training split:

```json
{
  "text": "Den Apel fält net wäit vum Bam.",
  "label": "correct"
}
```

```json
{
  "text": "Ech hunn dem Mann d'Buch ginn.",
  "label": "case_error"
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
  Fir dëse Satz ass de FeelerTyp uginn, oder 'correct' wann keng Feeler.
  ```

- Base prompt template:

  ```text
  Saz: {text}
  FeelerTyp: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Saz: {text}

  Identifizéiert de FeelerTyp am Saz. Äntwert mat 'correct', 'word_order', 'agreement', 'case', 'determiner', oder 'other'.
  ```

- Label mapping: (direct mapping to Luxembourgish error type names)

This dataset is marked as **unofficial** and is used for diagnostic purposes only.

```bash
euroeval --model <model-id> --dataset ltzglue-la-multi
```

## Natural Language Inference

### ltzGLUE-RTE

This dataset was published in
[this paper](https://arxiv.org/abs/2604.17976). Given a premise and hypothesis pair
from Luxembourgish text sources, models must determine if the hypothesis is entailed
by or contradicts the premise.

The original ltzGLUE-RTE dataset contains 1,877 / 197 / 626 samples
for training, validation and testing, respectively. Due to limited validation and
test data, we use all available samples (197 validation, 626 test) and cap training
at 1,024 samples using stratified sampling on the entailment/contradiction labels.

Here are a few examples from the training split:

```json
{
  "premise": "D'Regierung huet e neie Plang presentéiert.",
  "hypothesis": "Et gëtt eng nei Presentatioun vun der Regierung.",
  "label": "entailment"
}
```

```json
{
  "premise": "Et regnet zu Lëtzebuerg.",
  "hypothesis": "D'Sonn scheint zu Lëtzebuerg.",
  "label": "contradiction"
}
```

```json
{
  "premise": "De Premierminister huet eng Rede gehalen.",
  "hypothesis": "De Xavier Bettel huet e Virdrag gehalen.",
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
  Premisse: {premise}
  Hypothees: {hypothesis}
  Relatioun: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Premisse: {premise}
  Hypothees: {hypothesis}

  Bestëmmt ob d'Hypothees d'Premisse follegt oder widderleet. Äntwert mat 'entailment' oder 'contradiction'.
  ```

- Label mapping:
  - `entailment` ➡️ `entailment`
  - `contradiction` ➡️ `contradiction`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ltzglue-rte
```

## Reading Comprehension

### MultiWikiQA-lb

This dataset was published in
[this paper](https://doi.org/10.48550/arXiv.2509.04111) and contains question-answer
pairs derived from [Luxembourgish Wikipedia](https://lb.wikipedia.org/) articles about
Luxembourg and Luxembourgish culture, history, and geography. The questions and answers
are generated using large language models based on the Wikipedia content.

The dataset is extracted from the
[alexandrainst/multi-wiki-qa](https://huggingface.co/datasets/alexandrainst/multi-wiki-qa)
dataset (subset "lb"), containing 5,003 samples. We apply the standard EuroEval
cap of 1,024 / 256 / 2,048 samples for evaluation splits.

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

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Fir dës Fro gëtt et eng Äntwert.
  ```

- Base prompt template:

  ```text
  Kontext: {context}
  Fro: {question}
  Äntwert: {answer}
  ```

- Instruction-tuned prompt template:

  ```text
  Kontext: {context}
  Fro: {question}

  Beäntwert d'Fro baséiert op dem Kontext.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multiwikiqa-lb
```

## Summarisation

### LuxGen-Summ

This dataset was published in
["Text Generation Models for Luxembourgish with Limited Data: A Balanced Multilingual Strategy"](https://arxiv.org/abs/2412.09415)
and is based on Luxembourgish news articles from the LuxGen benchmark. The articles
are sourced from major Luxembourgish news outlets including Luxemburger Wort,
RTL Lëtzebuerg, paperjam.lu, today.lu, and wort.lu, with human-written summaries
covering politics, economy, culture, and sports.

The original dataset consists of 5,003 samples. We use a
1,024 / 256 / 2,048 split for training, validation and testing, respectively
(so 3,328 samples used in total). All splits are created using random sampling
(stratification not applicable for summarisation tasks).

Here are a few examples from the training split:

```json
{
  "text": "D'Regierung huet e neie Plang fir d'Wirtschaft presentéiert. D'Mesure sollen den imprese hëllefen an Akommes kreéieren. D'Oppositioun kritiséiert de Plang awer als net genuch.",
  "target_text": "Neie Wirtschaftsplang presentéiert."
}
```

```json
{
  "text": "D'Walen am Land stinn virun der Dier. Verschidde Parteie hunn hir Programmer presentéiert. D'Themen sinn Educatioun, Santé an Ekologie. D'Parteie verspriechen Reformen.",
  "target_text": "Wahlprogrammer presentéiert."
}
```

```json
{
  "text": "De Lëtzebuerger Sportler huet d'Goldmedail bei den Olympesche Spiller gewonnen. Et ass déi éischt Medail fir d'Land an dëser Disziplin. De Gewinner ass frou iwwer de Success.",
  "target_text": "Goldmedail fir Lëtzebuerg."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Evaluation metric: ROUGE-L
- Prefix prompt:

  ```text
  Hei sinn Artikelen an hir Zesummefaassungen.
  ```

- Base prompt template:

  ```text
  Artikel: {text}
  Zesummefaassung: {target_text}
  ```

- Instruction-tuned prompt template:

  ```text
  Artikel: {text}

  Fass den Artikel an engem Satz zesummen.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset luxgen-summ
```

## Text Classification

### ltzGLUE-HC

This dataset was published in
[this paper](https://arxiv.org/abs/2604.17976). Given a news headline and its
article text from Luxembourgish news outlets (Luxemburger Wort, RTL Lëtzebuerg,
paperjam.lu, today.lu, and wort.lu), models must predict whether the headline
correctly represents the article content.

The original ltzGLUE-HC dataset contains 20,716 / 2,960 / 5,919 samples
for training, validation and testing, respectively. We apply the standard EuroEval
cap of 1,024 / 256 / 2,048 samples using stratified sampling on the yes/no labels.

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
  Fir dësen Titel an Text ass uginn ob den Titel korrekt ass.
  ```

- Base prompt template:

  ```text
  Titel: {title}
  Text: {text}
  Korrekt: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Titel: {title}
  Text: {text}

  Ass den Titel eng korrekt Zesummefaassung vum Text? Äntwert mat 'jo' oder 'nee'.
  ```

- Label mapping:
  - `yes` ➡️ `jo`
  - `no` ➡️ `nee`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ltzglue-hc
```

### ltzGLUE-TC

This dataset was published in
[this paper](https://arxiv.org/abs/2604.17976) and contains Luxembourgish news
articles categorized by topic. The articles are sourced from major Luxembourgish
news outlets including Luxemburger Wort, RTL Lëtzebuerg, paperjam.lu, today.lu, and
wort.lu, covering politics, economy, sports, and culture.

The original dataset contains 9,932 / 1,240 / 1,245 samples
for training, validation and testing. We cap training at 1,024 samples using
stratified sampling on the topic labels, and use all available test data
(1,245 samples, below the 2,048 target).

Here are a few examples from the training split:

```json
{
  "text": "Politik: D'Regierung huet e neie Plang presentéiert.",
  "label": "politics"
}
```

```json
{
  "text": "Sport: D'Lëtzebuerger Ekipp huet de Match gewonnen.",
  "label": "sports"
}
```

```json
{
  "text": "Kultur: Den neie Festival ass e groussen Erfolleg.",
  "label": "culture"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Fir dësen Artikel ass d'Thema uginn.
  ```

- Base prompt template:

  ```text
  Artikel: {text}
  Thema: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Artikel: {text}

  Klassifizéiert d'Thema vum Artikel. Äntwert nëmme mat 'politics', 'sports', 'culture', 'economy' oder 'technology'.
  ```

- Label mapping: (direct mapping to Luxembourgish category names)

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ltzglue-tc
```

### Unofficial: ltzGLUE-ID

Intent detection dataset from the ltzGLUE benchmark, containing Luxembourgish
queries annotated with intent categories such as weather lookup, restaurant booking,
music playback, alarm setting, and creative work search. The queries are synthetically
generated to cover common voice assistant and search engine intents.

The original dataset only provides validation and test splits (53 and 66 samples
respectively) with no training data. The dataset contains 10 intent classes, some
with only 1-2 samples, making stratified splitting impossible.

Due to the extremely limited size and lack of training data, this dataset is marked
as **unofficial** and is intended for diagnostic purposes only. It is not suitable
for leaderboard inclusion.

Example categories include: `weather/find`, `BookRestaurant`, `SearchCreativeWork`,
`PlayMusic`, `alarm/set_alarm`, `SearchScreeningEvent`, `reminder/set_reminder`,
`RateBook`, `AddToPlaylist`, `alarm/cancel_alarm`.
