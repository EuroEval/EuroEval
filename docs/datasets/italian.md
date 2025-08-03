# 🇮🇹 Italian

This is an overview of all the datasets used in the Italian part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.


## Sentiment Classification

### Sentipolc-16

This dataset was published in [this paper](https://ceur-ws.org/Vol-1749/paper_026.pdf)
and slightly modified in [this paper](https://aclanthology.org/2022.lrec-1.27).
It is based on Italian tweets, which were manually annotated by three annotators.

The original full dataset consists of 1,839 / 324 / 870 samples, and we use a 1,024 /
256 / 1,024 split for training, validation and testing, respectively. The splits are new
and there can thus be some overlap between the original validation and test sets and our
validation and test sets.

Here are a few examples from the training split:

```json
{
  "text": "RT @user: Siamo dei falsi. I ragazzi vogliono le ragazze timide e poi stanno con le troie. Le ragazze vogliono i dolci e poi amano con…",
  "label": "negative"
}
```
```json
{
  "text": "Ho aggiunto un video a una playlist di @user: http ROMA PRESENTAZIONE LIBRO SVIMEZ SULL’ECONOMIA DEL",
  "label": "neutral"
}
```
```json
{
  "text": "RT @user: @user te lo auguro di cuore e farò il possibile affinché sia così. Un abbraccio",
  "label": "positive"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:
  ```
  Di seguito sono riportati i testi e il loro sentimento, che può essere 'positivo', 'neutro' o 'negativo'.
  ```
- Base prompt template:
  ```
  Tweet: {text}
  Sentimento: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Tweet: {text}

  Classificare il sentimento nel Tweet. Rispondete con 'positivo', 'neutro' o 'negativo', e nient'altro.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset sentipolc16
```


## Named Entity Recognition

### MultiNERD IT

This dataset was published in [this
paper](https://aclanthology.org/2022.findings-naacl.60/) and consists of sentences from
Wikipedia and Wikinews in 10 different languages. It is an extension of the combination
of [WikiNEuRal](https://www.github.com/Babelscape/wikineural) and
[NER4EL](https://www.github.com/Babelscape/ner4el). The original test set was created
from manual annotations, while the training set is based on an automatic annotation
pipeline.

The Italian part of the original dataset consists of 181,927 sentences, split into
145,520 / 18,190 / 18,217 for training, validation, and testing respectively. We use
given splits, and use 1,024 / 256 / 2,048 samples for training, validation, and testing,
respectively.

We have furthermore converted their fine-grained labelling scheme to the CoNLL-2003
labelling scheme, which is more common in the NER literature. The mapping is as follows:

- `PERS` ➡️ `PER`
- `LOC` ➡️ `LOC`
- `ORG` ➡️ `ORG`
- `MISC` ➡️ `MISC`
- `TIME` ➡️ `O`
- `ANIM` ➡️ `MISC`
- `BIO` ➡️ `MISC`
- `CEL` ➡️ `MISC`
- `DIS` ➡️ `MISC`
- `EVE` ➡️ `MISC`
- `FOOD` ➡️ `MISC`
- `INST` ➡️ `MISC`
- `MEDIA` ➡️ `MISC`
- `MYTH` ➡️ `MISC`
- `PLANT` ➡️ `MISC`
- `VEHI` ➡️ `MISC`

Here are a few examples from the training split:

```json
{
  "tokens": array(['Alcune' 'statue' 'che' 'la' 'rappresentano' 'vennero' 'ritrovate' 'non' 'lontano' 'da' 'Tani' ',' 'anche' 'se' 'in' 'nessuna' 'di' 'queste' 'si' 'è' 'conservato' 'il' 'volto' ',' 'mentre' 'nella' 'seconda' 'cateratta' 'è' 'registrata' 'una' 'piena' 'del' 'Nilo' 'datata' 'al' 'suo' '3º' 'anno' 'di' 'regno' '.'], dtype=object),
  "labels": array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
```
```json
{
  "tokens": array(['Nella' 'seconda' 'metà' 'del' 'XX' 'secolo' 'gli' 'infinitesimi' 'sono' 'stati' 'recuperati' ',' 'in' 'una' 'prospettiva' 'rigorosa' ',' 'da' 'Abraham' 'Robinson' ',' 'nella' 'formulazione' 'di' 'quella' 'che' 'lui' 'chiamò' 'analisi' 'non' 'standard' '.'], dtype=object),
  "labels": array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
```
```json
{
  "tokens": array(['Il' 'monumento' 'a' 'Carlo' 'Emanuele' 'III' 'di' 'Savoia' 'è' 'ubicato' 'nella' 'piazza' 'omonima' 'sul' 'lungomare' '.'], dtype=object),
  "labels": array(['O', 'O', 'O', 'B-PER', 'I-PER', 'I-PER', 'I-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:
  ```
  Di seguito sono riportate le frasi e i dizionari JSON con le entità denominate presenti nella frase data.
  ```
- Base prompt template:
  ```
  Frase: {text}
  Entità denominate: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Frase: {text}

  Identificare le entità nominate nella frase. Il risultato dovrebbe essere un dizionario JSON con le chiavi 'persona', 'posizione', 'organizzazione' e 'varie'. I valori devono essere elenchi di entità nominate di quel tipo, esattamente come appaiono nella frase.
  ```
- Label mapping:
    - `B-PER` ➡️ `persona`
    - `I-PER` ➡️ `persona`
    - `B-LOC` ➡️ `posizione`
    - `I-LOC` ➡️ `posizione`
    - `B-ORG` ➡️ `organizzazione`
    - `I-ORG` ➡️ `organizzazione`
    - `B-MISC` ➡️ `varie`
    - `I-MISC` ➡️ `varie`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset multinerd-it
```

### Unofficial: WikiNEuRal IT

This dataset was published in [this
paper](https://aclanthology.org/2021.findings-emnlp.215) and
consists of sentences from Wikipedia in 9 different languages. The annotations are
automatic but at the time novel and state-of-the-art methodologies.

The Italian part of the original dataset consists of 110,519 sentences, split into
88,400 / 11,050 / 11,069 for training, validation, and testing respectively. We use
given splits, and use 1,024 / 256 / 2,048 samples for training, validation, and testing,
respectively.

Here are a few examples from the training split:

```json
{
  "tokens": array(['Comunque' ',' 'il' 'poema' 'sarebbe' 'stato' 'influenzato' 'da' 'una' '"' 'tematica' 'di' 'regime' '"' 'voluta' 'dalla' 'politica' 'culturale' 'di' 'Domiziano' 'nella' 'quale' 'rientrano' 'anche' 'i' '"' 'Punica' '"' 'di' 'Silio' 'Italico' '.']),
  "labels": array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'O', 'O', 'B-PER', 'I-PER', 'O'])
}
```
```json
{
  "tokens": array(['È' 'stato' 'uno' 'degli' 'artisti' 'più' 'importanti' "dell'" 'etichetta' 'discografica' 'di' 'musica' 'soul' 'Stax' 'Records' 'che' 'negli' 'anni' 'sessanta' 'e' 'settanta' 'era' 'la' 'principale' 'antagonista' 'della' 'Motown' 'nel' 'campo' 'della' 'black' 'music' '.']),
  "labels": array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'O', 'O', 'O', 'O', 'O', 'O'])
}
```
```json
{
  "tokens": array(['Decise' 'di' 'scrivere' 'una' 'serie' 'di' 'saggi' 'e' 'presentarli' 'in' 'un' 'periodico' 'intitolato' '"' 'The' 'Rambler' '"' 'che' 'sarebbe' 'stato' 'messo' 'in' 'vendita' 'per' 'pochi' 'centesimi' 'ogni' 'martedì' 'e' 'sabato' '.']),
  "labels": array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'I-MISC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'])
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:
  ```
  Di seguito sono riportate le frasi e i dizionari JSON con le entità denominate presenti nella frase data.
  ```
- Base prompt template:
  ```
  Frase: {text}
  Entità denominate: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Frase: {text}

  Identificare le entità nominate nella frase. Il risultato dovrebbe essere un dizionario JSON con le chiavi 'persona', 'posizione', 'organizzazione' e 'varie'. I valori devono essere elenchi di entità nominate di quel tipo, esattamente come appaiono nella frase.
  ```
- Label mapping:
    - `B-PER` ➡️ `persona`
    - `I-PER` ➡️ `persona`
    - `B-LOC` ➡️ `posizione`
    - `I-LOC` ➡️ `posizione`
    - `B-ORG` ➡️ `organizzazione`
    - `I-ORG` ➡️ `organizzazione`
    - `B-MISC` ➡️ `varie`
    - `I-MISC` ➡️ `varie`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset wikineural-it
```

## Linguistic Acceptability

### ScaLA-it

This dataset was published in [this paper](https://aclanthology.org/W13-2308/)
is automatically created from the [Italian Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Italian-ISDT) by assuming that
the documents in the treebank are correct, and corrupting the samples to create
grammatically incorrect samples. The corruptions were done by either removing a
word from a sentence, or by swapping two neighbouring words in a sentence. To ensure
that this does indeed break the grammaticality of the sentence, a set of rules were
used on the part-of-speech tags of the words in the sentence.

The original full dataset consists of 13,121 / 564 / 482 samples for training,
validation and testing, respectively. We use 512 / 128 / 1,024, sampled from a
combination of all the splits.

Here are a few examples from the training split:

```json
{
  "text": "Il Presidente della di la Repubblica non è responsabile degli di gli atti compiuti nell' in l' esercizio delle di le sue funzioni, tranne che per alto tradimento o per attentato alla a la Costituzione.",
  "label": "correct"
}
```
```json
{
  "text": "Ottimamente ha retto invece il cuore nuovo di Saverio Pallucca - alle a le spalle tre infarti, quattro by-pass, un trapianto cardiaco meno di due anni fa - nell' in l' ultima edizione della di la famosa maratona di New York.",
  "label": "correct"
}
```
```json
{
  "text": "Un secondo gruppo di problemi riguarda la necessità di garantire che il sistema economico venga percepito come fondamentalmente equo, che rappresenta la chiave della la di sua sostenibilità politica.",
  "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Di seguito sono riportate le frasi e la loro correttezza grammaticale.
  ```
- Base prompt template:
  ```
  Frase: {text}
  Grammaticalmente corretto: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Frase: {text}

  Stabilite se la frase è grammaticalmente corretta o meno. Rispondete con 'si' se la frase è corretta e con 'no' se non lo è, e nient'altro.
  ```
- Label mapping:
    - `correct` ➡️ `si`
    - `incorrect` ➡️ `no`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset scala-it
```


## Reading Comprehension

### SQuAD-it

This dataset is derived from the SQuAD 1.1 dataset and was published in
[this paper](https://doi.org/10.1007/978-3-030-03840-3_29).
The questions and answers were obtained through "semi-automatic" translation, using
DeepL, of the SQuAD dataset to Italian. The dataset consists of 54,159 / 7,609
question/answer pairs for training and test respectively. We use 1,024 / 256 / 2,048
samples for training, validation, and testing, respectively. Our training split is a
subset of the original training split, and our validation and testing splits are subsets
of the original test split.

Here are a few examples from the training split:

```json
{
  "context": "Lo studio del Corano e dell' Hadith prosperò in un' atmosfera così studiosa. Filosofia, Fiqh e teologia (kalaam) sono stati ulteriormente sviluppati, in particolare da Avicenna e dai suoi avversari. Al-Razi e Al-Farabi avevano fornito metodologie e conoscenze in medicina e filosofia. Avicenna ha avuto accesso alle grandi biblioteche di Balkh, Khwarezm, Gorgan, Rey, Isfahan e Hamadan. Vari testi (come il' Ahd con Bahmanyar') mostrano che egli ha dibattuto punti filosofici con i più grandi studiosi del tempo. Aruzi Samarqandi descrive come prima che Avicenna lasciasse Khwarezm aveva conosciuto Al-Biruni (un famoso scienziato e astronomo), Abu Nasr Iraqi (un famoso matematico), Abu Sahl Masihi (un illustre filosofo) e Abu al-Khayr Khammar (un grande medico).",
  "question": "Che cosa è stato un tema che Avicenna ha ulteriormente sviluppato?",
  "answers": {
    "answer_start":  array([95]),
    "text": array(['teologia'], dtype=object)
  }
}
```
```json
{
  "context": "Florida Alta Velocità ferroviaria è stata proposta ferroviaria ad alta velocità sostenuta dal governo che avrebbe collegato Miami, Orlando e Tampa. La prima fase è stata pianificata per collegare Orlando e Tampa ed è stato offerto un finanziamento federale, ma è stato respinto dal governatore Rick Scott nel 2011. La seconda fase della linea è stata prevista per collegare Miami. Entro il 2014, un progetto privato conosciuto come All Aboard Florida da parte di una società della storica Florida East Coast Railway ha iniziato la costruzione di una linea ferroviaria ad alta velocità nel sud della Florida che dovrebbe terminare all' aeroporto internazionale di Orlando.",
  "question": "In quale anno ha iniziato All Aboard Florida?",
  "answers": {
    "answer_start": array([390]),
    "text": array(['2014'], dtype=object)
  }
}
```
```json
{
  "context": "Gli insetti sociali, come le termiti, le formiche e molte api e vespe, sono la specie più familiare di animali eusociali. Vivono insieme in grandi colonie ben organizzate che possono essere così strettamente integrate e geneticamente simili che le colonie di alcune specie sono talvolta considerate superorganismi. Talvolta si sostiene che le varie specie di api da miele siano gli unici invertebrati (e addirittura uno dei pochi gruppi non umani) ad aver evoluto un sistema di comunicazione simbolica astratta in cui un comportamento viene utilizzato per rappresentare e trasmettere informazioni specifiche su qualcosa nell' ambiente. In questo sistema di comunicazione, chiamato linguaggio dance, l' angolo in cui una danza d' ape rappresenta una direzione relativa al sole, e la lunghezza della danza rappresenta la distanza da volare. 309-311 Anche se forse non così avanzato come le api mellifere, anche i bombi hanno potenzialmente alcuni comportamenti di comunicazione sociale.",
  "question": "Termiti, api, vespe e quali altri insetti sono insetti sociali?",
  "answers": {
    "answer_start": array([41]),
    "text": array(['formiche'], dtype=object)
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:
  ```
  I testi che seguono sono accompagnati da domande e risposte.
  ```
- Base prompt template:
  ```
  Testo: {text}
  Domanda: {question}
  Rispondere in massimo 3 parole: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Testo: {text}

  Rispondi alla seguente domanda sul in un massimo di 3 parole.

  Domanda: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset squad-it
```


### Unofficial: BeleBele-it

This dataset was published in [this paper](https://aclanthology.org/2024.acl-long.44/)
and features multiple-choice reading comprehension questions across 122 languages.

The original dataset contains 900 unique multiple-choice reading comprehension passages
and questions. From these, we use a 256 / 64 / 580 split for training, validation and
testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Testo: Con la decisione del signor Rudd di firmare l’accordo sul clima di Kyoto, gli Stati Uniti, che ora saranno l’unica nazione sviluppata a non averlo ratificato, rimangono isolati. Il precedente governo conservatore australiano aveva rifiutato di ratificare gli accordi di Kyoto asserendo che avrebbero danneggiato l'economia, data la pesante dipendenza dalle esportazioni di carbone, mentre gli obiettivi sulle emissioni non sarebbero stati vincolanti per Paesi come l'India e la Cina.\nDomanda: Il precedente governo australiano pensava che la ratifica di Kyoto avrebbe causato danni a cosa?\nOpzioni:\na. Stati Uniti\nb. Economia del Paese\nc. Esportazioni di carbone\nd. Gli obiettivi di emissione del Paese",
  "label": "b"
}
```
```json
{
  "text": "Testo: "I commenti, in diretta televisiva, hanno rappresentato la prima occasione per autorevoli fonti iraniane per ammettere che le sanzioni sono efficaci. Esse comprendono limitazioni finanziarie e il divieto dell\'Unione europea all\'esportazione di petrolio greggio, che rappresenta l\'80% del reddito estero nell\'economia dell\'Iran. Secondo l\'ultimo rapporto mensile dell’OPEC, il volume delle esportazioni di greggio è sceso al livello più basso degli ultimi vent\'anni, con 2,8 milioni di barili al giorno. Il leader supremo del Paese, l’Ayatollah Ali Khamenei, ha parlato della dipendenza dal petrolio paragonandola ad ""una trappola"" che risale al periodo precedente la rivoluzione islamica iraniana del 1979 e dalla quale il Paese si dovrebbe liberare."\nDomanda: Secondo il passaggio, chi ha ammesso gli effetti delle sanzioni sull\'economia iraniana?\nOpzioni:\na. Autorevoli fonti\nb. OPEC\nc. Ayatollah Ali Khamenei\nd. L\'Unione Europea",
  "label": "a"
}
```
```json
{
  "text": "Testo: Il dottor Lee si è detto preoccupato anche in merito ai rapporti che rivelano che i bambini in Turchia ora sono stati contagiati dal virus dell'influenza aviaria A(H5N1) senza ammalarsi. Ha sottolineato che secondo alcuni studi la malattia diventerà meno mortale prima che possa causare un'epidemia globale. Si teme che se permangono sintomi influenzali di lieve entità, i pazienti possano continuare a contagiare più persone durante la loro routine quotidiana.\nDomanda: Secondo il brano, cosa dovrebbe accadere alla malattia prima di causare un'epidemia globale?\nOpzioni:\na. Deve diventare meno letale\nb. I sintomi devono rimanere lievi\nc. Occorre che più pazienti vengano infettati\nd. I bambini devono manifestare i sintomi",
  "label": "a"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Le seguenti sono domande a scelta multipla (con relative risposte).
  ```
- Base prompt template:
  ```
  Domanda: {text}
  Opzioni:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Risposta: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Domanda: {text}
  Opzioni:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Rispondete alla domanda precedente con 'a', 'b', 'c' o 'd', e nient'altro.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset belebele-it
```


### Unofficial: MultiWikiQA-it

This dataset will be published in an upcoming paper, and contains Italian Wikipedia
articles with generated questions and answers, using the LLM Gemini-1.5-pro.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
    "context": "I Campionati canadesi di sci alpino 2015 si sono svolti a Mont-Sainte-Anne e Nakiska dal 24 febbraio al 29 marzo. Il programma ha incluso gare di supergigante, slalom gigante, slalom speciale e combinata, tutte sia maschili sia femminili; tuttavia le gare di combinata sono state annullate.\n\nTrattandosi di competizioni valide anche ai fini del punteggio FIS, vi hanno partecipato anche sciatori di altre federazioni, senza che questo consentisse loro di concorrere al titolo nazionale canadese.\n\nRisultati\n\nUomini\n\nSupergigante \n\nData: 24 febbraio\nLocalità: Nakiska\nOre: 11.00 (UTC-5)\nPista: \nPartenza: 2\xa0255\xa0m\xa0s.l.m.\nArrivo: 1\xa0790\xa0m\xa0s.l.m.\nDislivello: 465\xa0m\nTracciatore: Richard Jagger\n\nSlalom gigante \n\nData: 26 marzo\nLocalità: Mont-Sainte-Anne\n1ª manche:\nOre: \nPista: \nPartenza: 615\xa0m\xa0s.l.m.\nArrivo: 265\xa0m\xa0s.l.m.\nDislivello: 350\xa0m\nTracciatore: John Kucera\n\n2ª manche:\nOre: \nPista: \nPartenza: 615\xa0m\xa0s.l.m.\nArrivo: 265\xa0m\xa0s.l.m.\nDislivello: 350\xa0m\nTracciatore: Mathieu Roy\n\nSlalom speciale \n\nData: 28 marzo\nLocalità: Mont-Sainte-Anne\n1ª manche:\nOre: \nPista: \nPartenza: 515\xa0m\xa0s.l.m.\nArrivo: 315\xa0m\xa0s.l.m.\nDislivello: 200\xa0m\nTracciatore: Johnny Crichton\n\n2ª manche:\nOre: \nPista: \nPartenza: 515\xa0m\xa0s.l.m.\nArrivo: 315\xa0m\xa0s.l.m.\nDislivello: 200\xa0m\nTracciatore: Duane Baird\n\nCombinata \nLa gara, originariamente in programma il 26 marzo a Mont-Sainte-Anne, è stata annullata.\n\nDonne\n\nSupergigante \n\nData: 24 febbraio\nLocalità: Nakiska\nOre: 9.30 (UTC-5)\nPista: \nPartenza: 2\xa0255\xa0m\xa0s.l.m.\nArrivo: 1\xa0790\xa0m\xa0s.l.m.\nDislivello: 465\xa0m\nTracciatore: Richard Jagger\n\nSlalom gigante \n\nData: 27 marzo\nLocalità: Mont-Sainte-Anne\n1ª manche:\nOre: \nPista: \nPartenza: 615\xa0m\xa0s.l.m.\nArrivo: 265\xa0m\xa0s.l.m.\nDislivello: 350\xa0m\nTracciatore: Peter Rybárik\n\n2ª manche:\nOre: \nPista: \nPartenza: 615\xa0m\xa0s.l.m.\nArrivo: 265\xa0m\xa0s.l.m.\nDislivello: 350\xa0m\nTracciatore: Martin Durocher\n\nSlalom speciale \n\nData: 28 marzo\nLocalità: Mont-Sainte-Anne\n1ª manche:\nOre: \nPista: \nPartenza: 515\xa0m\xa0s.l.m.\nArrivo: 315\xa0m\xa0s.l.m.\nDislivello: 200\xa0m\nTracciatore: Pierre-Luc Dumoulin\n\n2ª manche:\nOre: \nPista: \nPartenza: 515\xa0m\xa0s.l.m.\nArrivo: 315\xa0m\xa0s.l.m.\nDislivello: 200\xa0m\nTracciatore: Brett Zagazowski\n\nCombinata \nLa gara, originariamente in programma il 27 marzo a Mont-Sainte-Anne, è stata annullata.\n\nNote\n\nCollegamenti esterni \n \n \n\nCanadesi\n2015\nSport a Beaupré",
    "question": "Qual è stato l'autore del tracciato della prima manche dello slalom speciale maschile a Mont-Sainte-Anne?",
    "answers": {
        "answer_start": array([1134]),
        "text": array(["Johnny Crichton"], dtype=object)
    }
}
```
```json
{
    "context": "\n\nCarriera\nTra il 1991 ed il 1995 è tesserato del , club della prima divisione inglese: nelle prime 2 stagioni gioca nelle giovanili, mentre dal 1993 al 1995 è aggregato alla prima squadra, in cui comunque gioca solamente una partita ufficiale, il 14 agosto 1994, quando subentra dalla panchina al 64' nel Charity Shield perso per 2-0 contro il  a Wembley. Nell'arco di queste stagioni trascorre anche un breve periodo in prestito al , club di quarta divisione, con cui nella parte finale della stagione 1993-1994 gioca 11 partite di campionato. Nella seconda parte della stagione 1994-1995 viene ceduto a titolo definitivo allo , con cui realizza 9 reti in 20 partite di campionato, non riuscendo comunque ad evitare la retrocessione in terza divisione del club, con cui in compenso raggiunge le semifinali di Coppa di Lega, risultato a cui contribuisce realizzando 2 reti in altrettante presenze nella competizione. L'anno seguente con 10 reti in 26 presenze contribuisce all'immediato ritorno del club in seconda divisione, categoria nella quale nella stagione 1996-1997 mette a segno 8 reti in 31 presenze.\n\nNell'estate del 1997 passa allo , altro club di seconda divisione, con cui mette a segno 12 reti in 36 partite nel campionato 1997-1998, che si conclude con la retrocessione in terza divisione delle Potteries; l'anno seguente realizza 9 reti in 34 presenze in questa categoria, mentre nella stagione 1999-2000 oltre a vincere un Football League Trophy realizza 24 reti in 45 partite di campionato, a cui aggiunge 16 reti in 38 partite nel campionato successivo. Nella stagione 2000-2001 realizza invece 4 reti in 5 presenze per poi essere ceduto al , altro club di terza divisione, con cui nella rimanente parte della stagione mette a segno 8 reti in 26 presenze. Nella stagione 2002-2003 vince invece i play-off di terza divisione, dopo aver segnato 13 reti in 46 partite di campionato; nella stagione 2003-2004 torna quindi nuovamente a giocare in seconda divisione, categoria nella quale va a segno per 13 volte in 23 presenze. L'anno seguente, che è anche il suo ultimo nel Cardiff City, gioca con maggior regolarità e va nuovamente in doppia cifra di reti segnate: chiude infatti il campionato con 31 presenze e 12 reti. Tra il 2005 ed il 2007 gioca ancora in seconda divisione, con la maglia del , ma con un ruolo da comprimario: nell'arco di 2 stagioni segna infatti solamente una rete in complessive 36 partite di campionato. Al termine della stagione 2006-2007 scende di categoria e si accasa al , in quarta divisione: qui, nelle stagioni 2007-2008 e 2008-2009 gioca stabilmente da titolare e torna a segnare con regolarità (31 reti in 70 partite di campionato nell'arco del biennio), mentre nella stagione 2009-2010, la sua ultima in carriera, perde il posto in squadra e gioca in totale solamente 9 partite fra tutte le competizioni (7 in campionato e 2 nel Football League Trophy) senza mai segnare.\n\nIn carriera ha totalizzato complessivamente 495 presenze e 174 reti nei campionati professionistici inglesi (play-off inclusi), più 25 presenze e 2 reti in FA Cup, 27 presenze e 14 reti in Coppa di Lega, una presenza nel Community Shield e 13 presenze e 7 reti nel Football League Trophy, per un totale complessivo di 561 presenze e 197 reti in carriera in partite ufficiali.\n\nPalmarès\n\nClub\n\nCompetizioni nazionali\n\nStoke: 1999-2000\n\nNote\n\nCollegamenti esterni",
    "question": "In quale torneo ha disputato l'unico incontro ufficiale il calciatore con il Manchester City?",
    "answers": {
        "answer_start": array([306]),
        "text": array(["Charity Shield"], dtype=object)
    }
}
```
```json
{
    "context": "HD 56779 è una stella bianco-azzurra nella sequenza principale di magnitudine 5,01 situata nella costellazione della Poppa. Dista 959 anni luce dal sistema solare.\n\nOsservazione\nSi tratta di una stella situata nell'emisfero celeste australe. La sua posizione moderatamente australe fa sì che questa stella sia osservabile specialmente dall'emisfero sud, in cui si mostra alta nel cielo nella fascia temperata; dall'emisfero boreale la sua osservazione risulta invece più penalizzata, specialmente al di fuori della sua fascia tropicale. La sua magnitudine pari a 5 fa sì che possa essere scorta solo con un cielo sufficientemente libero dagli effetti dell'inquinamento luminoso.\n\nIl periodo migliore per la sua osservazione nel cielo serale ricade nei mesi compresi fra dicembre e maggio; nell'emisfero sud è visibile anche all'inizio dell'inverno, grazie alla declinazione australe della stella, mentre nell'emisfero nord può essere osservata limitatamente durante i mesi della tarda estate boreale.\n\nCaratteristiche fisiche\nLa stella è una bianco-azzurra nella sequenza principale; possiede una magnitudine assoluta di -2,33 e la sua velocità radiale positiva indica che la stella si sta allontanando dal sistema solare.\n\nVoci correlate\nStelle principali della costellazione della Poppa\n\nCollegamenti esterni\n\nStelle di classe spettrale B\nStelle bianco-azzurre di sequenza principale",
    "question": "Quanto è distante HD 56779 dal nostro sistema solare?",
    "answers": {
        "answer_start": array([130]),
        "text": array(["959 anni luce"], dtype=object)
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:
  ```
  I testi che seguono sono accompagnati da domande e risposte.
  ```
- Base prompt template:
  ```
  Testo: {text}
  Domanda: {question}
  Rispondere in massimo 3 parole: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Testo: {text}

  Rispondi alla seguente domanda sul in un massimo di 3 parole.

  Domanda: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset multi-wiki-qa-it
```


## Knowledge

### MMLU-it

This dataset is a machine translated version of the English [MMLU
dataset](https://openreview.net/forum?id=d7KBjmI3GmQ) and features questions within 57
different topics, such as elementary mathematics, US history and law. The translation to
Italian was done by the University of Oregon as part of [this
paper](https://aclanthology.org/2023.emnlp-demo.28/), using GPT-3.5-turbo.

The original full dataset consists of 269 / 1,410 / 13,200 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 2,048 split for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
new and there can thus be some overlap between the original validation and test sets and
our validation and test sets.

Here are a few examples from the training split:

```json
{
  "text": "Quale delle seguenti situazioni è meglio modellata dalla distribuzione binomiale?\nScelte:\na. Il numero di minuti in un'ora in cui la media Dow-Jones è superiore alla sua media iniziale del giorno.\nb. Il numero di città tra le 10 più grandi dello Stato di New York in cui il tempo è nuvoloso per la maggior parte di un determinato giorno.\nc. Il numero di conducenti che indossano le cinture di sicurezza se 10 conducenti consecutivi vengono fermati in un posto di blocco della polizia.\nd. Nessuna delle precedenti.",
  "label": "d"
}
```
```json
{
  "text": "Il 'nuovo razzismo' si riferisce a:\nScelte:\na. una forma più sottile di pregiudizio, mascherata dall'orgoglio nazionale\nb. una decostruzione post-moderna delle idee razziste per rivelarne la mancanza di profondità\nc. pratiche razziste riscontrabili in aree sociali di recente emergenza, come il cyberspazio\nd. un movimento antifascista che sfida le politiche nazionaliste",
  "label": "a"
}
```
```json
{
  "text": "Tutti i seguenti possono agire come messaggeri intracellulari, TRANNE\nScelte:\na. ioni di calcio\nb. cAMP\nc. acetilcolina\nd. inositolo 1,4,5-trifosfato",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Le seguenti sono domande a scelta multipla (con relative risposte).
  ```
- Base prompt template:
  ```
  Domanda: {text}
  Scelte:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Risposta: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Domanda: {text}
  Scelte:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Rispondete alla domanda precedente con 'a', 'b', 'c' o 'd' e nient'altro.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset mmlu-it
```


## Common-sense Reasoning

### HellaSwag-it

This dataset is a machine translated version of the English [HellaSwag
dataset](https://aclanthology.org/P19-1472/). The original dataset was based on both
video descriptions from ActivityNet as well as how-to articles from WikiHow. The dataset
was translated by the University of Oregon as part of [this
paper](https://aclanthology.org/2023.emnlp-demo.28/), using GPT-3.5-turbo.

The original full dataset consists of 9,193 samples. We use a 1,024 / 256 / 2,048 split
for training, validation and testing, respectively (so 3,328 samples used in total).

Here are a few examples from the training split:

```json
{
    "text": "[header] Come pulire il tappeto stainmaster [title] Aspirare lo sporco. [step] In primo luogo, si può rimuovere molta della sporcizia con un aspirapolvere. Poi, si possono affrontare le macchie rimanenti con sapone e acqua.\nScelte:\na. Quindi, iniziare a spolverare e aspirare. Non usare prodotti chimici aggressivi, come detergenti abrasivi, poiché questi possono causare la formazione di muffe sul tappeto.\nb. [substeps] Fai spolverare la superficie prima di aspirare. Puoi farlo con un panno in microfibra o una spazzola.\nc. [title] Usare sapone e acqua sulla macchia. [step] Mescolare acqua e ¼ di tazza (21 grammi) di sapone liquido in una bottiglia spray e poi spruzzare direttamente questa miscela sulla macchia.\nd. Cerca fango o macchie nere che puoi pulire localmente. [substeps] Se il tuo tappeto stainmaster non è pulito, potrebbe essere necessario pulirlo da un professionista.",
    "label": "c"
}
```
```json
{
    "text": "[header] Come sapere perché un bambino (sotto i 2 anni) sta piangendo [title] Ascolta il pianto forte, quasi un lamento. [step] Questo di solito significa \"ho dolore\" o \"sono malato\". Il bambino farà una pausa, poi urlerà di nuovo e ripeterà il processo.\nScelte:\na. Questo tipo di pianto è di solito solo un segnale di avvertimento della fame. Un bambino piangerà anche leggermente di più se ha fame.\nb. Questo può essere molto sconvolgente da guardare, quindi fai venire un genitore ad aiutare il bambino. [substeps] Solo un genitore può giudicare l'età del loro bambino.\nc. Questo di solito finirà dopo circa tre minuti. [title] Fai attenzione agli occhi chiusi del bambino.\nd. È persistente, penetrante e inequivocabile. Se senti questo pianto, vai immediatamente dal bambino.",
    "label": "d"
}
```
```json
{
    "text": "Una donna mostra come asciugare la superficie del bancone e il lavandino dall'acqua schizzata dal rubinetto con un asciugamano di carta. una donna\nScelte:\na. mostra il suo metodo preparatorio meticoloso per il bancone e il pavimento sui quali applicherà un asciugamano.\nb. sta in cucina accanto al lavandino e parla alla telecamera.\nc. impugna un asciugamano di carta e inizia a pulire una bevanda appoggiata sulla superficie del bancone e del lavandino.\nd. sta di fronte ad un set di utensili sul bancone, prende un asciugacapelli con le sue parti accessorie fissate e sicure con una barra sul lavandino asciutto.",
    "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Le seguenti sono domande a scelta multipla (con relative risposte).
  ```
- Base prompt template:
  ```
  Domanda: {text}
  Scelte:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Risposta: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Domanda: {text}
  Scelte:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Rispondete alla domanda precedente con 'a', 'b', 'c' o 'd' e nient'altro.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset hellaswag-it
```


### Unofficial: GoldenSwag-it

This dataset is a filtered and machine translated version of the English [HellaSwag dataset](https://aclanthology.org/P19-1472/), featuring both video descriptions from ActivityNet as well as how-to articles from WikiHow. The machine translated version was published in [this paper](https://doi.org/10.48550/arXiv.2410.08928) and was done using DeepL, and the filtering was published in [this paper](https://doi.org/10.48550/arXiv.2504.07825), which resulted in higher quality samples.

The original full dataset consists of 1530 / 1530 samples for training and validation, respectively. However, they are exactly equal. We use a split of 660 / 256 / 2,048 samples for training, validation, and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Come sapere cosa indossare. Identificate la vostra tonalità di pelle. Ci sono molti termini usati per descrivere la tonalità della pelle, da quella chiara o scura, a quella pallida o olivastra. Il modo più accurato per capire quali colori vi stanno bene è capire il sottotono della vostra pelle.\nScelte:\na. Questa è la chiave numero uno per identificare il colore della vostra pelle. Se avete un misto di pelle olivastra e sottotono caldo (come una pelle avorio), il vostro tono di pelle è probabilmente a metà tra il caldo e il freddo.\nb. Se avete una corporatura media o calda, in genere avete sottotoni evidenti. Ecco alcuni sottotoni comuni: la pelle calda e i sottotoni caldi comprendono tutti e tre i toni medi, tutti e tre i toni freddi, tutti e quattro i toni caldi e tutti e quattro i toni caldi.\nc. La vostra pelle sarà del colore delle vostre spalle, dal collo alle dita, alle unghie dei piedi. Il sottotono è un colore di base per il vostro aspetto generale, come espressione primaria della vostra carnagione.\nd. Ne esistono tre tipi: caldo, freddo e neutro. Poiché si cercano i sottotoni della pelle, non basta guardarsi allo specchio per averne conferma.",
  "label": "d"
}
```

```json
{
  "text": "Come fare la treccia. Spazzolare i capelli. Spazzolate i capelli in modo che siano leggeri e soffici. Dovete eliminare tutti i nodi in modo che la treccia sia liscia come la seta! Questa operazione facilita anche il processo di intreccio, quindi assicuratevi di farlo.\nScelte:\na. Prendete tre o quattro pollici (da 5 a 10 cm) di capelli dalla nuca, pettinateli e metteteli in un porta-treccia. Legateli e rimetteteli nel supporto.\nb. Se i capelli sono molto aggrovigliati, potrebbero gocciolare e potreste non riuscire a intrecciarli in modo così ordinato! Avvolgere i capelli. Con i capelli raccolti in rulli, arricciateli intorno al dito in modo che tutti i rulli siano infilati.\nc. Decidete dove fare la treccia. Sarà dietro la testa in una coda di cavallo? Sarà laterale o più bassa, vicino al collo? Decidete questo per determinare dove e come sarà più bella.\nd. Inumidite i capelli e scompigliateli delicatamente con le dita, in modo da ottenere un risultato bello e soffice. Probabilmente sarà facile separarli tirandoli un po', ma fate attenzione a non farlo.",
  "label": "c"
}
```

```json
{
  "text": "Come mettere la carta velina in un sacchetto regalo. Raccogliete i materiali. Avrete bisogno di carta velina, del regalo, di nastri o abbellimenti, di un sacchetto regalo e di un biglietto. Avrete bisogno di diversi colori di carta velina che si abbinino al colore del sacchetto regalo.\nScelte:\na. Acquistate o realizzate un sacchetto di carta velina bianco o crema in un negozio di artigianato. La carta velina vi darà un colore rosa pastello e si completerà con il colore del sacchetto regalo.\nb. La carta velina colorata rende il regalo più festoso! Assicuratevi che il vostro sacchetto regalo sia adatto all'occasione. Se avete intenzione di arricciare il nastro per aggiungerlo come decorazione, avrete bisogno di forbici per arricciare il nastro o di un nastro già arricciato.\nc. Potreste aver bisogno di andare in un negozio di antiquariato o in un negozio dell'usato per trovare tutti i colori che vi servono. Considerate la possibilità di utilizzare diversi colori per il biglietto, tra cui carta commestibile, carta da regalo o carta da costruzione.\nd. Potete utilizzare carta di scarto, carta in rotoli, carta riciclata o carta da costruzione. Prendete un pezzo di carta velina, di carta igienica o di qualsiasi altro foglio di carta colorata.",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Le seguenti sono domande a scelta multipla (con relative risposte).
  ```
- Base prompt template:
  ```
  Domanda: {text}
  Scelte:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Risposta: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Domanda: {text}
  Scelte:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Rispondete alla domanda precedente con 'a', 'b', 'c' o 'd' e nient'altro.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset goldenswag-it
```


## Summarization

### IlPost-Sum

This dataset was published in [this paper](https://www.mdpi.com/2078-2489/13/5/228) and
consists of news articles from [Il Post](https://www.ilpost.it/). The summaries were
written by the journalists themselves (the "target" field in the original dataset).

The original dataset consists of 35,201 / 4,400 / 4,400 samples for training,
validation and testing, respectively. We use 1,024 / 256 / 2,048 samples for training,
validation, and testing, respectively. All our splits are subsets of the original ones.

Here are a few examples from the training split:

```json
{
  "text": "Mai come nel 2013 abbiamo riflettuto sulla quantità di dati e informazioni su ciascuno di noi che nel corso degli anni hanno immagazzinato le grandi società di Internet. Ne eravamo consapevoli anche prima, ma soprattutto in seguito alle rivelazioni sui sistemi usati dalla National Security Agency statunitense per spiare le attività di centinaia di milioni di persone in giro per il mondo abbiamo iniziato a farci qualche domanda in più su che fine facciano le email, le foto e gli aggiornamenti sui social network quando li carichiamo online. Sappiamo meglio di prima che tutte queste cose vengono consegnate alla rete “per sempre” e che continueranno a esistere su qualche server, anche se faremo clic sull’icona di un cestino o su un tasto rosso con scritto sopra “Cancella”. E forse proprio per questo motivo, in molti iniziano a provare sollievo nell’avere a disposizione servizi e applicazioni che fanno l’esatto contrario: che rendono effimera e del tutto temporanea l’esistenza di qualcosa di nostro online. Come spiega Farhad Manjoo sul Wall Street Journal, la cosa più rilevante in campo tecnologico nel 2013 è stata probabilmente Snapchat, un’applicazione basata su comunicazioni temporanee. In pochi anni ha ottenuto un successo considerevole, soprattutto negli Stati Uniti, attirando l’attenzione di alcune grandi società come Facebook e Google che si dice abbiano offerto diversi miliardi di dollari per acquisirla. Le offerte sono state fin qui rifiutate da quelli di Snapchat, che per ora sembrano essere solo interessati a migliorare e rendere ancora più diffusa la loro applicazione.",
  "target_text": "Snapchat e l’Internet “temporanea”. Come funziona – e cosa implica, per gli utenti – la popolare applicazione per mandarsi messaggi e foto che spariscono dopo pochi secondi, contesa a colpi di offerte miliardarie."
}
```
```json
{
  "text": "Con trovata da entertainer, nel suo discorso da sconfitto al ballottaggio delle primarie del centrosinistra, Matteo Renzi ha citato Bersani, “ma non Pierluigi, Samuele”. è sempre bellissima la cicatrice che mi ricorderà di esser stato felice",
  "target_text": "Pesce d’aprile, Samuele Bersani. La canzone citata da Matteo Renzi nel suo \"concession speech\"."
}
```
```json
{
  "text": "Questa mattina i carabinieri hanno arrestato più di 50 persone accusate di essere a capo o affiliate al clan mafioso D’Abramo-Sforza. Gli arresti sono avvenuti a Bari, Altamura (Bari), Foggia, Cerignola (Foggia), Matera, Lecce e Roma. Le accuse contro gli arrestati sono di associazione armata di tipo mafioso, detenzione e porto d’armi anche da guerra, traffico di sostanze stupefacenti, omicidio, tentato omicidio, estorsione, turbativa d’asta. L’operazione è stata disposta dal gip di Bari su richiesta della Direzione distrettuale antimafia; le indagini sono state condotte dal nucleo investigativo del Comando provinciale Carabinieri di Bari.",
  "target_text": "Sono state arrestate più di 50 persone accusate di far parte del clan mafioso D’Abramo-Sforza."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:
  ```
  Di seguito sono riportati gli articoli con i relativi riassunti.
  ```
- Base prompt template:
  ```
  Articolo di cronaca: {text}
  Sintesi: {target_text}
  ```
- Instruction-tuned prompt template:
  ```
  Articolo di cronaca: {text}

  Scrivete un riassunto dell'articolo sopra citato.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset ilpost-sum
```
