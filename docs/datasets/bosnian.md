# 🇧🇦 Bosnian

This is an overview of all the datasets used in the Bosnian part of EuroEval. The
datasets are grouped by their task – see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### MMS-bs

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2306.07902).
The corpus consists of 79 manually selected datasets from over 350 datasets reported in the
scientific literature based on strict quality criteria.

The original dataset contains a single split with 77,594 Croatian samples.
We use 1,024 / 256 / 2,048 samples for our training, validation, and test splits,
respectively.
We have employed stratified sampling based on the label column from the original
dataset to ensure balanced splits.

Here are a few examples from the training split:

```json
{
    "text": "Jaoo kako cjadko, izasla si s momkom  ju ar filing loved, o maj gash! Awwww. POV RA CA CU",
    "label": "negative"
}
```

```json
{
    "text": "@aneldzoko sta se to desava u Neumu?",
    "label": "neutral"
}
```

```json
{
    "text": "Zasto se inspirator zove inspirator kad se s njim usisava?",
    "label": "neutral"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Slijede dokumenti i njihova osjetila, koja mogu biti pozitivno, neutralno ili negativno.
  ```

- Base prompt template:

  ```text
  Dokument: {text}
  Osjetilo: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Dokument: {text}

  Klasificirajte osjećaj u dokumentu. Odgovorite samo s pozitivno, neutralno, ili negativno, i ništa drugo.
  ```

- Label mapping:
  - `positive` ➡️ `pozitivno`
  - `neutral` ➡️ `neutralno`
  - `negative` ➡️ `negativno`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset mms-bs
```

## Named Entity Recognition

### WikiANN-bs

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
    "tokens": ["Čehoslovačka", ",", "Francuska", ",", "Mađarska", ",", "Meksiko", ",", "Švicarska", ",", "Urugvaj"],
    "labels": ["B-LOC", "O", "B-LOC", "O", "B-LOC", "O", "B-LOC", "O", "B-LOC", "O", "B-LOC"],
}
```

```json
{
    "tokens": ["godine", ",", "naselje", "je", "ukinuto", "i", "pripojeno", "naselju", "Bribir", "."],
    "labels": ["O", "O", "O", "O", "O", "O", "O", "O", "B-LOC", "O"],
}
```

```json
{
    "tokens": ["Administrativno", "središte", "oblasti", "je", "Tjumenj", "."],
    "labels": ["O", "O", "O", "O", "B-LOC", "O"],
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Slijede rečenice i JSON riječnici s imenovanim entitetima koji se pojavljuju u rečenicama.
  ```

- Base prompt template:

  ```text
  Rečenica: {text}
  Imenovani entiteti: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Rečenica: {text}

  Identificirajte imenovane entitete u rečenici. Prikažite ih kao JSON riječnik s ključevima 'osoba', 'mjesto', 'organizacija' i 'razno'. Vrijednosti trebaju biti popisi imenovanih entiteta navedenog tipa, točno kako se pojavljuju u rečenici.
  ```

- Label mapping:
  - `B-PER` ➡️ `osoba`
  - `I-PER` ➡️ `osoba`
  - `B-LOC` ➡️ `mjesto`
  - `I-LOC` ➡️ `mjesto`
  - `B-ORG` ➡️ `organizacija`
  - `I-ORG` ➡️ `organizacija`
  - `B-MISC` ➡️ `razno`
  - `I-MISC` ➡️ `razno`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset wikiann-bs
```

## Linguistic Acceptability

### ScaLA-hr

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Croatian Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Croatian-SET) by assuming that the
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
    "text": "Nakon kratke intervencije, tijekom koje sam saznala kada se taj osjećaj prvog puta pojavio i zbog čega, sve je nestalo i već mjesecima živim bez opterećenja koji me pratilo cijelog života.",
    "label": "correct"
}
```

```json
{
    "text": "Svaki od tih sklopova, i dijelova mora biti homologiran i sukladan s ostalima.",
    "label": "incorrect"
}
```

```json
{
    "text": "Prvi među njima je Laurent Blanc, koji drži Romu na čekanju, a s Parkom prinčeva povezivan je i Fabio Capello.",
    "label": "correct"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Sljedeće su rečenice i jesu li gramatički ispravne.
  ```

- Base prompt template:

  ```text
  Rečenica: {text}
  Gramatički ispravna: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Rečenica: {text}

  Odredite je li rečenica gramatički ispravna ili ne. Odgovorite s 'da' ako je ispravna, i s 'ne' ako nije. Odgovorite samo tom riječju i ničim drugim.
  ```

- Label mapping:
  - `correct` ➡️ `da`
  - `incorrect` ➡️ `ne`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-hr
```

## Reading Comprehension

### MultiWikiQA-bs

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
    "context": "NGC 3803 (također poznat kao PGC 36204) je eliptična galaksija koja je udaljena oko 164 miliona sg od Zemlje i nalazi se u sazviježđu Lav. Najveći prečnik je 0,40 (19 hiljada sg) a najmanji 0,4 uglovnih minuta (19 hiljada sg). Prvo otkriće je napravio R. J. Mitchell 27. marta 1856. godine.\n\nNajbliži NGC/IC objekti \nSljedeći spisak sadrži deset najbližih NGC/IC objekata.\n\nTakođer pogledajte \n Novi opći katalog\n Spisak NGC objekata\n Spisak galaksija\n\nBilješke \n  Prividna magnituda od 15,5 – Apsolutna magnituda: M = m - 5 ((log10 DL) - 1), gdje je m=15,5 i DL=50,4 * 106.\n  0,40 uglovnih minuta – S = A * D * 0,000291 * P, gdje je A=0,40, D=50,4 i P = 3,2616.\n  Bazirano na euklidsku udaljenost.\n\nReference\n\nLiteratura\n\nVanjski linkovi\n\nNGC 3803 \n\n  NGC 3803 na Aladin pregledaču\n\nNGC katalog \n  Interaktivni NGC Online Katalog\n  Astronomska baza podataka SIMBAD\n  NGC katalog na Messier45.com \n  NGC/IC projekt\n  NGC2000 na NASA sajtu\n  NGC na The Night Sky Atlas sajtu\n\nEliptične galaksije\nLav (sazviježđe)\nNGC objekti\nPGC objekti",
    "question": "Koliki je najmanji kutni promjer NGC 3803 izražen u kutnim minutama?",
    "answers": {
        "answer_start": [158],
        "text": ["0,4"]
    }
}
```

```json
{
    "context": "Po popisu stanovništva, domaćinstava i stanova 2011. u  Srbiji, koji je proveden od 1. do 15. oktobra 2011, u općini Crna Trava živjelo je ukupno 1663 stanovnika, što predstavlja 0,02% od ukupnog broja stanovnika Srbije, odnosno 0,77% od od ukupnog broja stanovnika Jablaničkog okruga.  Popis stanovništva provoden je na temelju Zakona o popisu stanovništva, domaćinstava i stanova u 2011. Godini ("Službeni glasnik RS", br. 104/09 i 24/11).\n\nRezultati popisa\n\nNacionalna pripadnost\n\nMaternji jezik\n\nVjeroispovijest\n\nStarosna piramida \nOd ukupnog broja stanovnika u općini Crna Trava bilo je 838 (50,39%) muškaraca i 825 (49,61%) žena, što predstavlja omjer muškaraca i žena 1.016:1000. Prosječna starost stanovništva bila je 53,7 godina, muškaraca 51,4 godina, a žena 56,1 godina. Udio osoba starijih od 18 godina je 91,5% (1.521), kod muškaraca 92,0% (771), a kod žena 90,9% (750).\n\nTakođer pogledajte\n\nNapomene\n\nReference\n\nVanjski linkovi \n Republički zavod za statistiku Srbije \n\nCrna Trava\nCrna Trava",
    "question": "Koliko godina u prosjeku imaju stanovnici općine Crna Trava?",
    "answers": {
        "answer_start": [726],
        "text": ["53,7 godina"]
    }
}
```

```json
{
    "context": "IC 910 (također poznat kao IRAS 13387+2331, MCG 4-32-25 i PGC 48424) je spiralna galaksija koja je udaljena oko 374 miliona sg od Zemlje i nalazi se u sazviježđu Volar. Najveći prečnik je 0,50 (54 hiljade sg) a najmanji 0,4 uglovnih minuta (44 hiljade sg). Prvo otkriće je napravio Stephane Javelle 16. juna 1892. godine.\n\nNajbliži NGC/IC objekti \nSljedeći spisak sadrži deset najbližih NGC/IC objekata.\n\nTakođer pogledajte \n Novi opći katalog\n Spisak IC objekata\n Spisak galaksija\n\nBilješke \n  Prividna magnituda od 14,4 – Apsolutna magnituda: M = m - 5 ((log10 DL) - 1), gdje je m=14,4 i DL=114,6 * 106.\n  0,50 uglovnih minuta – S = A * D * 0,000291 * P, gdje je A=0,50, D=114,6 i P = 3,2616.\n  Bazirano na euklidsku udaljenost.\n\nReference\n\nLiteratura\n\nVanjski linkovi\n\nIC 910 \n\n  IC 910 na Aladin pregledaču\n\nIC katalog \n  Interaktivni NGC Online Katalog\n  Astronomska baza podataka SIMBAD\n  IC katalog na Messier45.com \n  NGC/IC projekt\n  NGC2000 na NASA sajtu\n  IC na The Night Sky Atlas sajtu\n\nIC objekti\nIRAS objekti\nMCG objekti\nPGC objekti\nSpiralne galaksije\nVolar (sazviježđe)",
    "question": "Kolika je distanca između Zemlje i galaksije IC 910?",
    "answers": {
        "answer_start": [108],
        "text": ["oko 374 miliona sg"]
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

```text
Slijede tekstovi s pitanjima i odgovorima.
```

- Base prompt template:

```text
Tekst: {text}
Pitanje: {question}
Odgovor s najviše 3 riječi:
```

- Instruction-tuned prompt template:

```text
Tekst: {text}

Odgovorite na sljedeće pitanje o gornjem tekstu s najviše 3 riječi.

Pitanje: {question}
```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-bs
```

## Knowledge

### MMLU-hr

This dataset was published in
[this paper](https://doi.org/10.48550/arXiv.2410.08928) and is a machine
translated version of the English [MMLU dataset](https://openreview.net/forum?id=d7KBjmI3GmQ).
It features questions within 57 different topics, such as elementary mathematics, US
history, and law. DeepL was used to translate the dataset to Croatian.

The original full dataset consists of 254 / 12,338 samples for
validation and testing. These splits were merged, duplicates removed, and
new splits were created with 1,024 / 256 / 2048 samples for training, validation, and
testing, respectively.

Here are a few examples from the training split:

```json
{
    "text": "Kako se odvija lateralna komunikacija u organizaciji?\nIzbori:\na. Informacije se prenose prema gore.\nb. Informacije se prenose prema dolje.\nc. Informacije su dvosmjerni proces.\nd. Informacije se prenose između različitih odjela i funkcija.",
    "label": "d"
}
```

```json
{
    "text": "Kako astronomi misle da Jupiter generira svoju unutarnju toplinu?\nIzbori:\na. kroz egzotermne kemijske reakcije koje pretvaraju kemijsku potencijalnu energiju u toplinsku energiju\nb. nuklearna fuzija\nc. kontrakcijom koja mijenja gravitacijsku potencijalnu energiju u toplinsku energiju\nd. unutarnje trenje zbog njegove brze rotacije i diferencijalne rotacije",
    "label": "c"
}
```

```json
{
    "text": "Ako se parabola $y_1 = x^2 + 2x + 7$ i pravac $y_2 = 6x + b$ sijeku u samo jednoj točki, koja je vrijednost $b$?\nIzbori:\na. 7\nb. 3\nc. 12\nd. 4",
    "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Sljedeća su pitanja s višestrukim izborom (s odgovorima).
  ```

- Base prompt template:

  ```text
  Pitanje: {text}
  Izbori:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Odgovor: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Pitanje: {text}

  Odgovorite na gornje pitanje koristeći 'a', 'b', 'c' ili 'd', i ništa drugo.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset mmlu-hr
```

## Common-sense Reasoning

### Winogrande-hr

This dataset was published in
[this paper](https://doi.org/10.48550/arXiv.2506.19468) and is a translated
and filtered version of the English
[Winogrande dataset](https://doi.org/10.1145/3474381). DeepL was used to
translate the dataset to Croatian.

The original full dataset consists of 47 / 1,210 samples for training and testing, and
we use 128 of the test samples for validation, resulting in a 47 / 128 / 1,085 split for
training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
    "text": "Nisam mogao kontrolirati vlagu kao što sam kontrolirao kišu, jer je _ dolazila odasvud. Na što se odnosi praznina _?\nIzbori:\na. vlaga\nb. kiša",
    "label": "a"
}
```

```json
{
    "text": "Jessica je mislila da je Sandstorm najbolja pjesma ikad napisana, ali Patricia ju je mrzila. _ je kupila kartu za jazz koncert. Na što se odnosi praznina _?\nIzbori:\na. Jessica\nb. Patricia",
    "label": "b"
}
```

```json
{
    "text": "Termostat je pokazivao da je dolje dvadeset stupnjeva hladnije nego gore, pa je Byron ostao u _ jer mu je bilo hladno. Na što se odnosi praznina _?\nIzbori:\na. dolje\nb. gore",
    "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Sljedeća su pitanja s višestrukim izborom (s odgovorima).
  ```

- Base prompt template:

  ```text
  Pitanje: {text}
  Mogućnosti:
  a. {option_a}
  b. {option_b}
  Odgovor: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Pitanje: {text}
  Mogućnosti:
  a. {option_a}
  b. {option_b}

  Odgovorite na gornje pitanje koristeći 'a' ili 'b', i ništa drugo.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset winogrande-hr
```
