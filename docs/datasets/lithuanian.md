# 🇱🇹 Lithuanian

This is an overview of all the datasets used in the Lithuanian part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.


## Sentiment Classification

### Lithuanian Emotions

This dataset is a combination of machine translated versions of the [GoEmotions dataset](https://arxiv.org/abs/2005.00547) and the [Kaggle emotions dataset](https://www.kaggle.com/datasets/nelgiriyewithana/emotions). GoEmotions consists of English Reddit comments and the Kaggle dataset contains English Twitter messages. Both datasets have been machine translated to Lithuanian.

The original dataset contains 377k / 47.1k / 5.43k / 41.7k samples for the combined training, combined validation, Lithuanian GoEmotions test, and Lithuanian Twitter emotions test splits, respectively. We use 1,024 / 256 / 2,048 samples for our training, validation and test splits, respectively. Our splits are based on the original splits.

Here are a few examples from the training split:

```json
Dataset uploaded to EuroEval/lithuanian-emotions-mini
```json
{
  "text": "Aš jaučiuosi taip nekantrus, kad turiu laukti daugiau nei mėnesį ir tuo pačiu labai stengiuosi nelinkėti to laiko",
  "label": "positive"
}
```
```json
{
  "text": "Jaučiuosi gana bendras šeimininkas Toros",
  "label": "negative"
}
```
```json
{
  "text": "Florida, jis gavo du",
  "label": "neutral"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Toliau pateikti dokumentai ir jų nuotaika,
  kuri gali būti 'teigiamas', 'neutralus' arba 'neigiamas'.
  ```
- Base prompt template:
  ```
  Dokumentas: {text}
  Nuotaika: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Dokumentas: {text}

  Klasifikuokite nuotaiką dokumente. Atsakykite su 'teigiamas', 'neutralus' arba 'neigiamas', ir nieko kito.
  ```
- Label mapping:
    - `positive` ➡️ `teigiamas`
    - `neutral` ➡️ `neutralus`
    - `negative` ➡️ `neigiamas`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset lithuanian-emotions
```


## Named Entity Recognition

### WikiANN-lt

This dataset was published in [this paper](https://aclanthology.org/P17-1178/) and is
part of a cross-lingual named entity recognition framework for 282 languages from
Wikipedia. It uses silver-standard annotations transferred from English through
cross-lingual links and performs both name tagging and linking to an english Knowledge
Base.

The original full dataset consists of 10,000 / 10,000 / 10,000 samples for the training,
validation and test splits, respectively. We use 1,024 / 256 / 2,048 samples for our
training, validation and test splits, respectively. All the new splits are subsets of
the original splits.

Here are a few examples from the training split:

```json
{
  "tokens": array(["'", "''", 'Michael', 'Schumacher', "''", "'"], dtype=object),
  "labels": ["O", "O", "B-PER", "I-PER", "O", "O"]
}
```
```json
{
  "tokens": array(['Keliu', 'sujungtas', 'su', 'Alta', '.'], dtype=object),
  "labels": ["O", "O", "O", "B-LOC", "O"]
}
```
```json
{
  "tokens": array(['Amazonės', 'lamantinas', '(', "''Trichechus", 'inunguis', "''",
       ')'], dtype=object),
  "labels": ["B-LOC", "I-LOC", "O", "O", "O", "O", "O"]
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:
  ```
  Toliau pateikti sakiniai ir JSON žodynai su vardiniais vienetais, kurie pateikiame sakinyje.
  ```
- Base prompt template:
  ```
  Sakinys: {text}
  Vardiniai vienetai: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Sakinys: {text}

  Identifikuokite vardinius vienetus sakinyje. Turėtumėte pateikti tai kaip JSON žodyną su raktais 'asmuo', 'vieta', 'organizacija' ir 'kita'. Reikšmės turi būti to tipo vardinių vienetų sąrašai, tiksliai taip, kaip jie rodomi sakinyje.
  ```
- Label mapping:
    - `B-PER` ➡️ `asmuo`
    - `I-PER` ➡️ `asmuo`
    - `B-LOC` ➡️ `vieta`
    - `I-LOC` ➡️ `vieta`
    - `B-ORG` ➡️ `organizacija`
    - `I-ORG` ➡️ `organizacija`
    - `B-MISC` ➡️ `kita`
    - `I-MISC` ➡️ `kita`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset wikiann-lt
```


## Linguistic Acceptability

### ScaLA-lt

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Lithuanian Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Lithuanian-ALKSNIS) by assuming that the
documents in the treebank are correct, and corrupting the samples to create
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
  "text": "Be to, tai, kad turi man neįprastų drabužių, primena, jog ir daugiau man nežinomo gyvenimo.",
  "label": "incorrect"
}
```
```json
{
  "text": "Juos sukelia kokia nors konkreti organinė ir šiuo atveju galvos skausmas yra tik tam tikros ligos simptomas.",
  "label": "incorrect"
}
```
```json
{
  "text": "Juos sukelia kokia nors konkreti organinė ir šiuo atveju galvos skausmas yra tik tam tikros ligos simptomas.",
  "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Toliau pateikti sakiniai ir ar jie yra gramatiškai teisingi.
  ```
- Base prompt template:
  ```
  Sakinys: {text}
  Gramatiškai teisingas: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Sakinys: {text}

  Nustatykite, ar sakinys yra gramatiškai teisingas, ar ne. Atsakykite su 'taip' arba 'ne', ir nieko kito.
  ```
- Label mapping:
    - `correct` ➡️ `taip`
    - `incorrect` ➡️ `ne`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset scala-lt
```


## Reading Comprehension


## Knowledge

### LT-History

This dataset was published in [this paper](https://aclanthology.org/2025.nbreal-1.1/), and consists of Lithuanian national and general history
questions and answers.

The dataset originally contains 593 samples, which are reduced to 559 after filtering. Due to the limited number of examples, there is no training split; instead, the data is divided into 47 samples for validation and 512 for testing.

Here are a few examples from the validation split:

```json
{
  "text": "Abiejų Tautų Respublikos Ketverių metų seimo nutarimu:\nPasirinkimai:\na. valstiečiams suteikta asmens laisvė.\nb. bajorai atleisti nuo valstybinių mokesčių;\nc. miestiečiams leista užimti valstybines tarnybas;\nd. įteisinta absoliuti monarcho valdžia;",
  "label": "c"
}
```
```json
{
  "text": "Kurioje eilutėje visos išvardytos asmenybės gyveno Renesanso epochoje?\nPasirinkimai:\na. Vaskas da Gama, Maksimiljenas Robespjeras, Johanas Gutenbergas.\nb. Nikola Makiavelis, Šarlis Monteskjė, Kristupas Kolumbas.\nc. Mikalojus Kopernikas, Ferdinandas Magelanas, Leonardas da Vinčis.\nd. Johanas Gutenbergas, Žanas Žakas Ruso, Leonardas da Vinčis.",
  "label": "c"
}
```
```json
{
  "text": "Lietuvos teritorija suskirstyta į 10 apskričių: Vilniaus, Kauno, Klaipėdos, Šiaulių, Panevėžio, Alytaus ir...\nPasirinkimai:\na. Tauragės, Utenos, Marijampolės ir Telšių;\nb. Tauragės, Trakų, Kėdainių ir Plungės;\nc. Utenos, Marijampolės, Šalčininkų ir Telšių.\nd. Marijampolės, Telšių, Ukmergės ir Neringos;",
  "label": "a"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Toliau pateikti daugiavariančiai klausimai (su atsakymais).
  ```
- Base prompt template:
  ```
  Klausimas: {text}
  Pasirinkimai:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Atsakymas: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Klausimas: {text}
  Pasirinkimai:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Atsakykite į aukščiau pateiktą klausimą atsakydami 'a', 'b', 'c' arba 'd', ir nieko daugiau.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset lt-history
```


## Common-sense Reasoning

### Winogrande-lt

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2506.19468)
and is a translated and filtered version of the English [Winogrande
dataset](https://doi.org/10.1145/3474381).

The original full dataset consists of 47 / 1,210 samples for training and testing, and
we use the same splits.

Here are a few examples from the training split:

```json
{
  "text": "Derrick negalėjo susikoncentruoti darbe, skirtingai nei Justin, nes _ turėjo smagų darbą. Ką reiškia tuščia vieta _?\nPasirinkimai:\na. Pasirinkimas A: Derrick\nb. Pasirinkimas B: Justin",
  "label": "b"
}
```

```json
{
  "text": "Vieną kartą Lenkijoje Dennis mėgavosi kelione labiau nei Jason, nes _ turėjo paviršutinišką lenkų kalbos supratimą. Ką reiškia tuščia vieta _?\nPasirinkimai:\na. Pasirinkimas A: Dennis\nb. Pasirinkimas B: Jason",
  "label": "b"
}
```

```json
{
  "text": "Natalie mano, kad smaragdai yra gražūs brangakmeniai, bet Betty taip nemano. _ nusipirko vėrinį su dideliu smaragdu. Ką reiškia tuščia vieta _?\nPasirinkimai:\na. Pasirinkimas A: Natalie\nb. Pasirinkimas B: Betty",
  "label": "a"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Toliau pateikti daugiavariančiai klausimai (su atsakymais).
  ```
- Base prompt template:
  ```
  Klausimas: {text}
  Pasirinkimai:
  a. {option_a}
  b. {option_b}
  Atsakymas: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Klausimas: {text}
  Pasirinkimai:
  a. {option_a}
  b. {option_b}

  Atsakykite į aukščiau pateiktą klausimą atsakydami 'a' arba 'b', ir nieko daugiau.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset winogrande-lt
```


## Summarisation
