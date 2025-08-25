# 🇪🇪 Estonian

This is an overview of all the datasets used in the Estonian part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.


## Sentiment Classification

### Estonian Valence Corpus

This dataset was published in [this paper](http://dx.doi.org/10.7592/FEJF2016.64.polarity). The dataset was compiled of articles of different rubrics of online
dailies, weeklies, and reader comments, while the polarity of each paragraph was determined by native Estonian readers.

There are 4 labels in the original dataset instead of the usual 3.
Examples with the labels representing 'mixed' emotion (vastuoluline) were filtered out
mainly to be consistent with rest of the languages in EuroEval.

The original full dataset consists of 3,277 / 818 samples for the training and test splits,
respectively. Having filtered out 'mixed' examples, we truncate the train split to 1,024
examples, and redistribute the rest to validation and test resulting in the final size of
1,024 / 256 / 2,048 for the training, validation and test splits, respectively.


Here are a few examples from the training split:

```json
{
  "text": "Sügisest algav pikk koolitee Oskari perekonda ei hirmuta.",
  "label": "positiivne"
}
```
```json
{
  "text": "Sellises eas, nagu teie olete, tundub muidugi ka 20-aastane üsna laps ...",
  "label": "neutraalne"
}
```
```json
{
  "text": "ka ainus märkimisväärne saavutus temalt selle loo esituse juures.",
  "label": "negatiivne"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Järgmised on dokumendid ja nende meelestatus, mis võib olla 'positiivne', 'neutraalne' või 'negatiivne'.
  ```
- Base prompt template:
  ```
  Dokument: {text}
  Meelestatus: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Dokument: {text}

  Klassifitseeri dokument meelestatuse järgi. Võimalikud vastused: 'positiivne', 'neutraalne' või 'negatiivne'. Muud vastused ei ole lubatud.
  ```
- Label mapping:
    - `positive` ➡️ `positiivne`
    - `neutral` ➡️ `neutraalne`
    - `negative` ➡️ `negatiivne`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset estonian-valence
```
