# 🇱🇺 Luxembourgish

Luxembourgish (Lëtzebuergesch) is a West Germanic language spoken primarily in
Luxembourg. It is a Moselle Franconian dialect with significant French and German
influences. Despite being the national language of Luxembourg, it has limited digital
resources compared to other European languages.

## Sentiment Classification

### ltzGLUE-SA

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2604.17976)
and contains Luxembourgish texts annotated with sentiment labels (negative, neutral,
positive). The data is collected from Luxembourgish social media posts and news website
comments, covering various domains including politics, culture, and daily life.

The original ltzGLUE-SA dataset contains 3,022 / 597 / 926 samples for training,
validation and testing, respectively. We cap each split independently to at most 1,024
train, 256 val, and 2,048 test samples, preserving source split boundaries (no samples
are moved between splits). The published mini dataset contains up to 1,024 / 256 / 926
samples (test split is below the cap).

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

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2604.17976)
and contains Luxembourgish news articles annotated with BIO-formatted entity tags for
person, location, organization, and miscellaneous entities. The articles are sourced
from major Luxembourgish news outlets including Luxemburger Wort, RTL Lëtzebuerg,
paperjam.lu, today.lu, and wort.lu.

The original ltzGLUE-NER dataset contains 27,245 / 3,327 / 3,821 samples for training,
validation and testing, respectively. We cap each split independently to at most 1,024
train, 256 val, and 2,048 test samples, preserving source split boundaries (no samples
are moved between splits). The published mini dataset contains up to 1,024 / 256 / 2,048
samples.

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

### ltzGLUE-LA-binary

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2604.17976).
Luxembourgish sentences must be classified as grammatically correct or incorrect. The
dataset includes naturally occurring text from Luxembourgish news outlets (Luxemburger
Wort, RTL Lëtzebuerg, paperjam.lu, today.lu, and wort.lu) as well as systematically
corrupted examples with various error types (word order, agreement, case, etc.).

The original ltzGLUE-LA dataset contains 14,678 / 2,094 / 4,045 samples for training,
validation and testing, with balanced classes. We cap each split independently to at
most 1,024 train, 256 val, and 2,048 test samples, preserving source split boundaries
(no samples are moved between splits). The published mini dataset contains up to 1,024 /
256 / 2,048 samples.

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

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2604.17976).

Multi-class variant of the ltzGLUE-LA linguistic acceptability task where models must
identify the specific error type in addition to detecting incorrectness. The data
includes naturally occurring text from Luxembourgish news outlets (Luxemburger Wort, RTL
Lëtzebuerg, paperjam.lu, today.lu, and wort.lu) as well as systematically corrupted
examples with various error types (word order, agreement, case, etc.).

The original dataset contains 14,678 / 2,094 / 4,045 samples for training, validation
and testing. We cap each split independently to at most 1,024 train, 256 val, and 2,048
test samples, preserving source split boundaries (no samples are moved between splits).
The published mini dataset contains up to 1,024 / 256 / 2,048 samples.

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

- Label mapping:
  - `correct` ➡️ `correct`
  - `word_order` ➡️ `word_order`
  - `agreement` ➡️ `agreement`
  - `morphology` ➡️ `morphology`
  - `other` ➡️ `other`

```bash
euroeval --model <model-id> --dataset ltzglue-la-multi
```

## Natural Language Inference

### ltzGLUE-RTE

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2604.17976).

Given a premise and hypothesis pair from Luxembourgish text sources, models must
determine if the hypothesis is entailed by or contradicts the premise (binary
classification).

The original ltzGLUE-RTE dataset contains 1,877 / 197 / 626 samples for training,
validation and testing, respectively. We cap each split independently to at most 1,024
train, 256 val, and 2,048 test samples, preserving source split boundaries (no samples
are moved between splits). The published mini dataset contains up to 1,024 / 197 / 626
samples (train is capped, val and test are below the caps).

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
  - `contradiction` ➡️ `widdersträit`
  - `entailment` ➡️ `folgerung`

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
1,024 / 256 / 2,048 samples for evaluation splits. The new splits are subsets of the
original splits.

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

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2604.17976).

Given a news headline and its article text from Luxembourgish news outlets (Luxemburger
Wort, RTL Lëtzebuerg, paperjam.lu, today.lu, and wort.lu), models must predict whether
the headline correctly represents the article content.

The original ltzGLUE-HC dataset contains 20,716 / 2,960 / 5,919 samples for training,
validation and testing, respectively. We cap each split independently to at most 1,024
train, 256 val, and 2,048 test samples, preserving source split boundaries (no samples
are moved between splits). The published mini dataset contains up to 1,024 / 256 / 2,048
samples.

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

- Label mapping:
  - `no` ➡️ `no`
  - `yes` ➡️ `yes`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ltzglue-hc
```

### ltzGLUE-TC

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2604.17976)
and contains Luxembourgish news articles categorized by topic. The articles are sourced
from major Luxembourgish news outlets including Luxemburger Wort, RTL Lëtzebuerg,
paperjam.lu, today.lu, and wort.lu, covering politics, economy, sports, and culture.

The original dataset contains 9,932 / 1,240 / 1,245 samples for training, validation and
testing. We cap each split independently to at most 1,024 train, 256 val, and 2,048 test
samples, preserving source split boundaries (no samples are moved between splits). The
published mini dataset contains up to 1,024 / 256 / 1,245 samples (train and val are
capped, test is below the cap).

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

- Label mapping:
  - `technology` ➡️ `technology`
  - `business` ➡️ `business`
  - `culture` ➡️ `culture`
  - `animals` ➡️ `animals`
  - `sports` ➡️ `sports`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ltzglue-tc
```

### ltzGLUE-ID

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2604.17976).

Intent detection dataset from the ltzGLUE benchmark, containing Luxembourgish queries
annotated with intent categories such as weather lookup, restaurant booking, music
playback, alarm setting, and creative work search. The queries are synthetically
generated to cover common voice assistant and search engine intents.

The original dataset contains 53 validation and 66 test samples with **no training
data**. We cap each split independently to at most 256 val and 2,048 test samples,
preserving source split boundaries. The published mini dataset contains up to 53 / 66
samples (both splits are below the caps). There is no `train` split in the published
dataset.

Here are a few examples from the validation split:

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

- **Few-shot evaluation is not available** due to the absence of training data.
- Both base and instruction-tuned models can be evaluated in zero-shot mode.
- Instruction-tuned prompt template:

  ```text
  Ufruff: {text}

  Identifizéiert d'Intentioun vum Benotzer. Äntwert nëmme mat engem vun dësen Etiketten: addtoplaylist, alarm cancel alarm, alarm set alarm, alarm show alarms, bookrestaurant, playmusic, ratebook, reminder set reminder, searchcreativework, searchscreeningevent, weather find.
  ```

- Label mapping:
  - `addtoplaylist` ➡️ `addtoplaylist`
  - `alarm cancel alarm` ➡️ `alarm cancel alarm`
  - `alarm set alarm` ➡️ `alarm set alarm`
  - `alarm show alarms` ➡️ `alarm show alarms`
  - `bookrestaurant` ➡️ `bookrestaurant`
  - `playmusic` ➡️ `playmusic`
  - `ratebook` ➡️ `ratebook`
  - `reminder set reminder` ➡️ `reminder set reminder`
  - `searchcreativework` ➡️ `searchcreativework`
  - `searchscreeningevent` ➡️ `searchscreeningevent`
  - `weather find` ➡️ `weather find`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ltzglue-id
```

## Instruction-following

### MultiIFEval-lb

MultiIFEval-lb is part of the MultiIFEval benchmark spanning 305 languages. It is
generated by translating and localising the English IFEval dataset using a structured
LLM generation pipeline. For each target language, a randomly selected Wikipedia article
in that language provides contextual grounding to reduce hallucination and improve
cultural localisation. The pipeline preserves instruction_id_list values for
traceability to the original English samples, and retains kwargs keys with values
localised where appropriate, enabling programmatic constraint verification. The dataset
was published [here](https://huggingface.co/datasets/EuroEval/multi-ifeval-lb).

This dataset is part of the MultiIFEval benchmark introduced in
[this draft paper](https://raw.githubusercontent.com/alexandrainst/multi_ifeval/refs/heads/feat/add-paper/paper/acl_latex.tex).

We use the dataset as the test split, and do not include other splits, as we only
evaluate models zero-shot and the size is too small to warrant a validation set.

Here are a few examples from the test split:

```json
{
  "text": "Schreift eng Zesummefaassung vun der Wikipedia Säit \"https://lb.wikipedia.org/wiki/Lëtzebuergesch\" mat op d'mannst 200 Wierder. Benotzt keng Kommaen a markéiert op d'mannst 3 Sektiounen déi Titelen hunn am Markdown Format, zum Beispill *markéiert Sektioun Deel 1*, *markéiert Sektioun Deel 2*, *markéiert Sektioun Deel 3*.",
  "target_text": {
    "instruction_id_list": [
      "punctuation:no_comma",
      "detectable_format:number_highlighted_sections",
      "length_constraints:number_words"
    ],
    "kwargs": [
      {},
      { "num_highlights": 3 },
      { "num_words": 200, "relation": "at least" }
    ]
  }
}
```

```json
{
  "text": "Ech plangen eng Rees op Lëtzebuerg a wëll dDir mir e Reesplan am Shakespeare-Stil schreift. Dir dirft keng Kommaen an Ärer Äntwert benotzen.",
  "target_text": {
    "instruction_id_list": ["punctuation:no_comma"],
    "kwargs": [{}]
  }
}
```

```json
{
  "text": "Maacht e CV fir e frësche Student deen sech fir säin éischte Job bewierft. Stellt sécher datt Dir op d'mannst 12 Plazhalter a quadratesche Klameren abaut, wéi zum Beispill [Numm] oder [Adress].",
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
euroeval --model <model-id> --dataset multi-ifeval-lb
```

## Hallucination Detection

### RAGTruth-lb

This dataset is a Luxembourgish version of the
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
  "prompt": "\u00c4ntwert kuerz d'folgend Fro:\nwat ass den \u00cbnnerscheed t\u00ebscht raffin\u00e9ierte an unraffin\u00e9ierte Kokoos\u00f6l\nHuel am Kapp datt deng \u00c4ntwert strikt op d\u00e9i folgend dr\u00e4i Passagen bas\u00e9iert:\npassage 1:\u2022 Raffin\u00e9ierte Kokoos\u00f6l ass aus der gedr\u00e9chent Kokoosfrucht gemaach, w\u00e4hrend unraffin\u00e9ierte Kokoos\u00f6l, och genannt pur oder jung Kokoos\u00f6l, aus dem Fleesch vun de fr\u00ebsche Kokoosen gemaach g\u00ebtt. \u2022 Unraffin\u00e9ierte Kokoos\u00f6l huet e charakteristesche Aroma an Geschmaach, deen w\u00e4hrend dem Raffin\u00e9ieren verluer geet.\n\npassage 2:Allerd\u00e9ngs huet raffin\u00e9ierte Kokoos\u00f6l e Rauchpunkt vun 450 Grad F, w\u00e4hrend de Rauchpunkt vum unraffin\u00e9ierte Kokoos\u00f6l 350 Grad F ass. W\u00e9inst senger m\u00e9i h\u00e9ijer H\u00ebtztbest\u00e4ndegkeet kann raffin\u00e9ierte Kokoos\u00f6l eng besser Optioun fir h\u00e9ich Temperatur Kachmethoden w\u00e9i F\u00ebschfritt\u00e9ieren sinn.\n\npassage 3:Eent vun de charakteristesche \u00cbnnerscheeder t\u00ebscht raffin\u00e9ierte an unraffin\u00e9ierte Kokoos\u00f6l ass den Geschmaach. W\u00e9inst der Deodor\u00e9ierung verl\u00e9iert raffin\u00e9ierte Kokoos\u00f6l s\u00e4in typesche Kokoosgeschmaach an huet keen erkennbaren Geschmaach oder Aroma. W\u00e9i och \u00ebmmer, unraffin\u00e9iert \u00d6l huet e m\u00ebllen Kokoosgeschmaach an Aroma.\n\nWann d'Passagen net d\u00e9i n\u00e9ideg Informatiounen enthalen fir d'Fro ze be\u00e4ntweren, \u00e4ntwert w.e.g. mat: \"Net f\u00e4eg ze \u00e4ntweren bas\u00e9iert op de gegebene Passagen.\""
}
```

```json
{
  "prompt": "Kuerz \u00e4ntwerten d'Fro: W\u00e9i laang dauert et fir Poulet z'artikelen am Crockpot? \nHuel am Kapp, datt d'\u00c4ntwert strikt op de folgende dr\u00e4i Passagen bas\u00e9iert:\npassage 1: Plaz all Wurzelgem\u00e9is w\u00e9i Zwiebel, Kartoffelen an M\u00ebllchen am Crockpot an setz d'Poulet uewen op d'Gem\u00e9is. F\u00fc\u00fcgt eng hallef T\u00e9i vum Bouillon, W\u00e4in oder Waasser an de Pot. Kacht op niddreg fir 6 - 8 Stonnen oder bis et f\u00e4erdeg ass. (Dir k\u00ebnnt och op h\u00e9ich fir 1 Stonn kachen, dann op niddreg fir ongef\u00e9ier 5 Stonnen oder bis et f\u00e4erdeg ass.).\npassage 2: F\u00fc\u00fcgt eng hallef T\u00e9i vum Bouillon, W\u00e4in oder Waasser an de Pot. Kacht op niddreg fir 6 - 8 Stonnen oder bis et f\u00e4erdeg ass. (Dir k\u00ebnnt och op h\u00e9ich fir 1 Stonn kachen, dann op niddreg fir ongef\u00e9ier 5 Stonnen oder bis et f\u00e4erdeg ass.).\npassage 3: Bew\u00e4ertung Neisten Altesten. Bescht \u00c4ntwert: Dir g\u00e9ift 8 Pouletbr\u00e9ifst\u00e9cker op niddreg fir 8 Stonnen an engem Slow Cooker kachen. Ech maachen oft Poulet Tacos, an esou maachen ech et. Dir k\u00ebnnt et fir 4 Stonnen op h\u00e9ich kachen wann Dir Z\u00e4it dr\u00e9ngt. Et w\u00e4ert nach \u00ebmmer gutt ausgehen. Allgemeng w\u00e4ert Pouletbr\u00e9if \u00ebmmer zart sinn, solang et genuch Feuchtigkeit g\u00ebtt. Et funktion\u00e9iert och fir R\u00ebndfleesch, fir zerrasselt R\u00ebndfleesch Tacos an Taquitos. Ech maachen oft Poulet Tacos, an esou maachen ech et. Dir k\u00ebnnt et fir 4 Stonnen op h\u00e9ich kachen wann Dir Z\u00e4it dr\u00e9ngt. Et w\u00e4ert nach \u00ebmmer gutt ausgehen. Allgemeng w\u00e4ert Pouletbr\u00e9if \u00ebmmer zart sinn, solang et genuch Feuchtigkeit g\u00ebtt. Et funktion\u00e9iert och fir R\u00ebndfleesch, fir zerrasselt R\u00ebndfleesch Tacos an Taquitos.\n\nWann d'Passagen net d'n\u00e9ideg Informatioun enthalen fir d'Fro ze be\u00e4ntweren, \u00e4ntwert w.e.g. mat: \"Net f\u00e4eg ze be\u00e4ntweren bas\u00e9iert op de gegebene Passagen.\"\noutput:"
}
```

```json
{
  "prompt": "Instruktioun:\nSchrei eng objektiv Iwwersiicht iwwer d'folgend lokal Gesch\u00e4fter bas\u00e9iert n\u00ebmme op den zur Verf\u00fcgung gestellten struktur\u00e9ierte Daten am JSON-Format. Dir sollt Detailer abegraff an d'Informatiounen abdecken, d\u00e9i an den Rezensiounen vun de Clienten ernimmt ginn. D'Iwwersiicht sollt 100 - 200 Wierder hunn. Maacht keng Informatiounen aus. Struktur\u00e9iert Daten:\n{'Numm': 'La Casa De Maria', 'Adress': '800 El Bosque Rd', 'Stad': 'Santa Barbara', 'Staat': 'CA', 'Kategorien': 'Gesondheetsretreats, Plazen & Evenementer, Hoteller & Rees, Restauranten, Evenementplanung & Servicer', 'Stonnen': {'M\u00e9indeg': '8:0-23:0', 'D\u00ebnschdeg': '8:0-23:0', 'M\u00ebttwoch': '8:0-23:0', 'Donneschdeg': '8:0-23:0', 'Freideg': '8:0-23:0', 'Samschdeg': '8:0-23:0', 'Sonndeg': '8:0-23:0'}, 'Attributer': {'Gesch\u00e4ftsparken': None, 'Restaurantenreservatiounen': None, 'Draussen S\u00ebtzplazen': None, 'WiFi': None, 'Restauranten Nimm zu': None, 'Restauranten Gutt fir Gruppen': None, 'Musik': None, 'Ambiance': None}, 'Gesch\u00e4ftsst\u00e4ren': 4.5, 'Rezensiounsinformatioun': [{'Rezensiounsst\u00e4ren': 5.0, 'Rezensiounsdatum': '2018-01-10 17:43:18', 'Rezensiounstext': \"Ech hunn d\u00ebse Plaz g\u00e4r. Ech g\u00e9if hei fir \u00ebmmer bleiwen, wann ech k\u00e9int. D'\u00cbnnerkonft ass e b\u00ebsse rustikal, awer dat ass Deel vum Charme. Sch\u00e9in Plaz fir Selbstentdeckung an Kontemplatioun.\"}, {'Rezensiounsst\u00e4ren': 5.0, 'Rezensiounsdatum': '2017-08-05 01:28:13', 'Rezensiounstext': \"Wat eng sch\u00e9i Plaz an entspaanten Atmosph\u00e4r. D'\u00cbnnerkonften sinn relativ bequem, awer et sinn d'Grondst\u00e9cker, d\u00e9i d\u00ebs Juwel vun enger Plaz besonnesch maachen. (D'Iessen schadet och net.) Kritt eng Massage vun Vesalina wann et m\u00e9iglech ass; et war d\u00e9i bescht, d\u00e9i ech jee hat. Och, vergiess net d'G\u00e4rten an d'Kapellen ze besichen.\"}, {'Rezensiounsst\u00e4ren': 1.0, 'Rezensiounsdatum': '2017-04-15 17:16:29', 'Rezensiounstext': 'Ech schreiwen d\u00ebs Rezensioun fir zwou Gr\u00ebnn. 1) D'Wuel vum zuk\u00fcnftege G\u00e4scht; 2) an well d'Probleem ni vun irgendjemandem an der Facilit\u00e9it ugeschwat gouf z\u00ebnter ech meng Saachen gepackt hunn, eng Dag virum Enn vun mengem $piritual Retreat, an hunn de ganz deier Dag verbruecht fir alles wat ech hat ze desinfiz\u00e9ieren -- inklusiv mengem Auto ze detaill\u00e9ieren, alles meng Kleeder ze fumig\u00e9ieren/w\u00e4schend/drockeren -- ier ech heem gaang sinn zu menger Fr\u00ebndin an dem sch\u00e9inen Bett, dat si e puer M\u00e9int virdrun kaaft huet..\\n\\nBettbugs. Widderhuelen. Bettbugs, an mengem Fall, Z\u00ebmmer 12. Befall. Ech hunn an der Duschstall geschlof d\u00e9i lescht Nacht, well leider war d\u00ebst net meng \u00e9ischt Ritt mat Bettbugs. Notiz: Ech sinn net d\u00ebnn-h\u00e4erzlech wann et \u00ebm d'Natur geet. Ech hunn fir verschidde Summer an Alaska aus engem Zelt geleeft. Ech hunn d'Hiking g\u00e4r, Campen, an sinn an engem l\u00e4ndleche Gebitt am M\u00ebnsch gewuess. Kuerz gesot, ech sinn net angscht virun engem Raccoon (oder engem Bear fir dat Thema) oder lafen fir Schutz wann ech vun engem Hornet gebuzz ginn. \\n\\nAwer Bettbugs?   (yeeeych) \\n\\nJo, d\u00ebs wonnerbar kleng reesend Insekten -- vill w\u00e9i Fl\u00e9ien -- d\u00e9i heem mat Iech kommen w\u00ebllen. D'Zort Bugs d\u00e9i bissen an bissen -- an mengem Fall, mengem Hals an Been -- an anscheinend k\u00ebnnen iwwer all Pestizid oder W\u00e4sch liewen. \\n\\nEch g\u00e9if l\u00e9iwer M\u00ebss an mengem Gesiicht hunn, w\u00e9i w\u00ebsst ech an engem Bett(Z\u00ebmmer) geschlof mat Bettbugs. \\n\\nFir gerecht ze sinn, de jonke Mann, deen d'Desk de Nacht wou ech eng vun den gut-gr\u00e9isst Kreaturen an engem Plastikbecher \"gefaangen\" hunn, war sympathesch an huet gesicht fir mir eng aner Z\u00ebmmer de n\u00e4chste Moie ze kr\u00e9ien. Ech war esou juckend an entt\u00e4uscht, datt ech d'Noeffekter vum Bettbugged ze d'konfront\u00e9ieren, ech hunn all meng Kontaktinformatioun geliwwert an hunn direkt verlooss. Dat ass d'lescht wat ech vun Casa de la Maria h\u00e9ieren hunn trotz 3 E-Maile an 2 Uruff. (Ech hunn no Reinigungskompensatioun gefrot an d'zwei N\u00e4chte, d\u00e9i ech an Z\u00ebmmer 12 war, fir mech kompens\u00e9iert an dat war alles.)\\n\\nD'Grondst\u00e9cker? Hey, et ass Montecito/San Ysidro Ranch Land. Et ass sch\u00e9in!  A m\u00e9cht Santa Barbara w\u00e9i eng Virstad vu Bakersfield ausgesin...An d'Mitarbeiter waren fr\u00ebndlech an haapts\u00e4chlech h\u00ebllefr\u00e4ich. D'Fakt, datt si eng relativ laut an fuerderend Grupp vun G\u00e4scht an der M\u00ebddeg duerch eng sch\u00eblleg Buddhist Retreat gebucht hunn? Eh, ech verstinn et.  Koexist\u00e9ieren, Leit.  Just net mat Bettbugs.'}]}\nIwwersiicht:"
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
euroeval --model <model-id> --dataset ragtruth-lb
```
