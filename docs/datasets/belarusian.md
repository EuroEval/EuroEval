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
