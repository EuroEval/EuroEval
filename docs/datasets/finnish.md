# 🇫🇮 Finnish

This is an overview of all the datasets used in the Finnish part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### ScandiSent-fi

This dataset consists of reviews from Trustpilot and was published [here](https://github.com/timpal0l/ScandiSent). It is a binary sentiment classification dataset, with labels "positive" and "negative".

For the Finnish part of the dataset, there are 10,000 training samples. From these samples, we have created a 1,024 / 256 / 2,048 split for the train, validation and test splits, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Kaikki meni niinkuin piti. Nopea toimitus.",
  "label": "positive"
}
```
```json
{
  "text": "En pidä tästä, kun ei löydy linkkiä mistä pääsis heti maksamaan. En todellakaan pidä siitä, että joka tieto pitää kopioida erikseen. Haluaisin päästä suoraan oston jälkeen maksamaa mobiilipankkiin. Pari laskua on jäänyt tän takia kokonaan huomioimatta. Ja ihan turhaa.... ärsyttää sitten se kotiin tuleva muistutuslasku.",
  "label": "negative"
}
```
```json
{
  "text": "Todella hidas toimitus, ja virheellistä tietoa tuotteiden saatavuudesta, paketti ja tuotteet perillä vasta kuukauden päästä tilauksesta....",
  "label": "negative"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Seuraavassa on arvosteluja ja niiden tunnesävy, joka voi olla 'positiivinen' tai 'negatiivinen'.
  ```
- Base prompt template:
  ```
  Teksti: {text}
  Tunnesävy: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Teksti: {text}

  Luokittele arvostelun tunnesävy. Vastaa vain 'positiivinen' tai 'negatiivinen', ei muuta.
  ```
- Label mapping:
    - `positive` ➡️ `positiivinen`
    - `negative` ➡️ `negatiivinen`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset scandisent-fi
```

## Named Entity Recognition

## Linguistic Acceptability

## Reading Comprehension

## Knowledge

## Common-sense Reasoning

## Summarization
