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

### Turku-NER-fi

This dataset was published in [this paper](https://aclanthology.org/2020.lrec-1.567/).

The original dataset contains 12,217 / 1,364 / 1,555 samples for the training, validation and test splits, respectively. We use 1,024 / 256 / 2,048 samples for our training, validation and test splits, respectively. All the new splits are subsets of the original splits.

Here are a few examples from the training split:

```json
{
  "tokens": ["Suomalaiset", "vaihtoivat", "Tukholman", "Tallinnaan"],
  "labels": ["O", "O", "B-LOC", "B-LOC"]
}
```
```json
{
  "tokens": ["Liuhto", "nosti", "Kreikan", "tapauksen", "yhteydessä", "esille", "kysymyksen", "siitä", ",", "miten", "Euroopan", "unionissa", "yleisesti", "sanktioidaan", "pelisääntöjen", "rikkomisesta", "."],
  "labels": ["B-PER", "O", "B-LOC", "O", "O", "O", "O", "O", "O", "O", "B-ORG", "I-ORG", "O", "O", "O", "O", "O"],
}
```
```json
{
  "tokens": ["Mithridates", "oli", "Pontoksen", "merkittävin", "kuningas", "ja", "Rooman", "valtakunnan", "vaarallisin", "vihollinen", "ensimmäisellä", "vuosisadalla", "eaa.", "."],
  "labels": ["B-PER", "O", "B-LOC", "O", "O", "O", "B-LOC", "I-LOC", "O", "O", "O", "O", "O", "O"],
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:
  ```
  Seuraavassa on lauseita ja JSON-sanakirjoja, jotka sisältävät annetussa lauseessa esiintyvät nimetyt entiteetit.
  ```
- Base prompt template:
  ```
  Lause: {text}
  Nimetyt entiteetit: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Lause: {text}

  Tunnista lauseessa esiintyvät nimetyt entiteetit. Tulosta ne JSON-sanakirjana, jonka avaimet ovat 'henkilö', 'paikka', 'organisaatio' ja 'muut'. Arvojen tulee olla listoja kyseisen tyypin nimetyistä entiteeteistä täsmälleen siinä muodossa kuin ne esiintyvät lauseessa.
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
$ euroeval --model <model-id> --dataset turku-ner-fi
```



## Linguistic Acceptability

## Reading Comprehension

### TydiQA-fi

This question-answering dataset was published in [this paper](https://arxiv.org/abs/2003.05002).

The original Finnish TydiQA dataset contains 6,855 training and 782 validation samples (we use the [secondary task subset](https://huggingface.co/datasets/google-research-datasets/tydiqa/viewer/secondary_task?views%5B%5D=secondary_task_train)).  We created a 1,024 / 256 / 2,024 split, where the samples from the train and validation split are sampled from the original train and validation splits, respectively. The test set consists of the remaining samples from the original validation split + additional samples from the original train split.

Here are a few examples from the training split:

```json
{
  "question": "Kuka näytteli Dumbledorea Harry Potter elokuvissa?",
  "context": "Dumbledorea esittää kirjasarjasta tehdyssä elokuvasarjassa Richard Harris kahdessa ensimmäisessä elokuvassa. Harrisin kuoltua Michael Gambon esitti hahmoa sarjan lopuissa elokuvissa.",
  "answers": {
    "text": ["Richard Harris kahdessa ensimmäisessä elokuvassa. Harrisin kuoltua Michael Gambon"],
    "answer_start": [59]
  }
}

```json
{
  "question": "Milloin Cristiano Ronaldo liittyi Juventukseen?",
  "context": "Ronaldo siirtyi heinäkuussa 2018 Juventukseen 105 miljoonalla eurolla. Sopimus on nelivuotinen, ja sen aikana hän tienaa verojen jälkeen noin 120 miljoonaa euroa.[133]",
  "answers": {
    "text": ["heinäkuussa 2018"],
    "answer_start": [16]
  }
}
```json
{
  "question": "Kuka hallitsi Mithridates VI jälkeen?",
  "context": "Mithridates laajensi valtakuntaansa ympäri Mustanmeren rantoja, ja hän ajautui kolmesti sotaan Rooman valtakuntaa vastaan. Ensimmäisessä sodassa (89 eaa.–85 eaa.) hän valtasi suuren osan Vähää-Aasiaa ja Rooman valtakunnalle kuuluneet osat, jolloin hänen sanotaan teloittaneen 80000 roomalaista. Mithridates valtasi myös Kreikan, mutta konsuli Sulla kukisti hänen joukkonsa vuonna 85 eaa., ja Mithridateen oli luovuttava valloituksistaan. Toinen sota (83 eaa.–81 eaa.) oli suppeampi laajuudeltaan. Kolmannessa sodassa (73 eaa.–63 eaa.) roomalaiset sotapäälliköt Lucullus ja Pompeius kukistivat Mithridateen perusteellisesti. Mithridates surmasi tai surmautti itsensä jouduttuaan poikansa Farnakes II:n syrjäyttämäksi.",
  "answers": {
    "text": ["Farnakes II"],
    "answer_start": [687]
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:
  ```
  Seuraavassa on tekstejä ja niihin liittyviä kysymyksiä ja vastauksia.
  ```
- Base prompt template:
  ```
  Teksti: {text}
  Kysymys: {question}
  Vastaa enintään 3 sanalla: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Teksti: {text}

  Vastaa seuraavaan kysymykseen yllä olevasta tekstistä enintään 3 sanalla.

  Kysymys: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset tydiqa-fi
```

## Knowledge

## Common-sense Reasoning

## Summarization
