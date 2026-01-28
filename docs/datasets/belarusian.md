# 🇧🇾 Belarusian

This is an overview of all the datasets used in the Belarusian part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### BeSLS

This dataset was introduced in [this paper](https://aclanthology.org/2025.acl-long.25/).
It comprises 2,000 sentences that have been manually annotated for sentiment polarity:
positive (1) or negative (0).

The original split of the dataset consists of 1,500 samples for training, 250 for
validation, and 250 for testing. In EuroEval, we use 256 samples for training, 128 for
validation, and 1,616 for testing. The train and validation splits are subsets of the
original train/validation splits, while the test split includes the remaining samples
from the original training and validation sets.

Here are a few examples from the training split:

```json
{
  "text": "Пры вельмі сціплым бюджэце ў 20 млн даляраў Стахельскі зняў эталонны экшэн.",
  "label": "positive",
}
```

```json
{
    "text": "Гэта лічба толькі пацвярджае, што фестываль з кожным годам набірае моцы, пашыраючы сваю геаграфію.",
    "label": "positive",
}
```

```json
{
    "text": "Яна цудоўна абудзіла апетыт, апетыт да падрабязнасцяў, да разгадвання, да спазнання.",
    "label": "positive",
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Ніжэй прыведзены дакументы і іх сентымент, які можа быць 'станоўчы', 'нейтральны' або 'адмоўны'.
  ```

- Base prompt template:

  ```text
  Дакумент: {text}
  Сентымент: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Дакумент: {text}

  Класіфікуйце сентымент у дакуменце. Адкажыце толькі 'станоўчы', 'нейтральны' або 'адмоўны', і нічога іншага.
  ```

- Label mapping:
  - `positive` ➡️ `станоўчы`
  - `neutral` ➡️ `нейтральны`
  - `negative` ➡️ `адмоўны`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset besls
```

## Named Entity Recognition

### WikiANN-be

This dataset was published in [this paper](https://aclanthology.org/P17-1178/) and is
part of a cross-lingual named entity recognition framework for 282 languages from
Wikipedia. It uses silver-standard annotations transferred from English through
cross-lingual links and performs both name tagging and linking to an english Knowledge
Base.

The original full dataset consists of 15,000 / 1,000 / 1,000 samples for the training,
validation and test splits, respectively. We use 1,024 / 256 / 1,000 samples for our
training, validation and test splits, respectively. All the new splits are subsets of
the original splits.

Here are a few examples from the training split:

```json
{
  "tokens": ["Сцюарт", "Бінэм", "(", "4", ")"],
  "labels": ["B-PER", "I-PER", "O", "O", "O"],
}
```

```json
{
  "tokens": ["Пасля", "гуляў", "таксама", "за", "моладзевую", "зборную", "Беларусі", "."],
  "labels": ["O", "O", "O", "O", "B-ORG", "I-ORG", "I-ORG", "O"],
}
```

```json
{
  "tokens": ["Горад", "Кампен", ",", "Нідэрланды"],
  "labels": ["B-LOC", "I-LOC", "I-LOC", "I-LOC"],
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Ніжэй прыведзены сказы і JSON-слоўнікі з іменаванымі сутнасцямі, якія прысутнічаюць у дадзеным сказе.
  ```

- Base prompt template:

  ```text
  Сказ: {text}
  Іменаваныя сутнасці: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Сказ: {text}

  Ідэнтыфікуйце іменаваныя сутнасці ў сказе. Вы павінны вывесці гэта як JSON-слоўнік з ключамі 'асоба', 'месца', 'арганізацыя' і 'рознае'. Значэнні павінны быць спісамі іменаваных сутнасцей гэтага тыпу, дакладна такімі, як яны з'яўляюцца ў сказе.
  ```

- Label mapping:
  - `B-PER` ➡️ `асоба`
  - `I-PER` ➡️ `асоба`
  - `B-LOC` ➡️ `месца`
  - `I-LOC` ➡️ `месца`
  - `B-ORG` ➡️ `арганізацыя`
  - `I-ORG` ➡️ `арганізацыя`
  - `B-MISC` ➡️ `рознае`
  - `I-MISC` ➡️ `рознае`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset wikiann-be
```

## Linguistic Acceptability

### ScaLA-be

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Belarusian Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Belarusian-HSE) by assuming that
the documents in the treebank are correct, and corrupting the samples to create
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
    "text": "Скончыла Беларускую акадэмію мастацтваў (курс Міхаіла Жданоўскага) і курс дакументальнага кіно Doc Pro у Школе Вайды (Варшава).",
    "label": "correct"
}
```

```json
{
    "text": "Дзяржаўныя СМІ не расказалі пра тыя рэкамэндацыі WHO, якіх Беларусь не выконвае",
    "label": "correct"
}
```

```json
{
    "text": "Але праз 19 гадоў Статут новы ВКЛ скасаваў большасьць палажэньняў Люблінскай уніі.",
    "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Ніжэй прыведзены сказы і ці з'яўляюцца яны граматычна правільнымі.
  ```

- Base prompt template:

  ```text
  Сказ: {text}
  Граматычна правільны: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Сказ: {text}

  Вызначце, ці сказ граматычна правільны ці не. Адкажыце толькі {labels_str}, і нічога іншага.
  ```

- Label mapping:
  - `correct` ➡️ `так`
  - `incorrect` ➡️ `не`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-be
```
