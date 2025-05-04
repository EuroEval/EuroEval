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
  'context': "Lo studio del Corano e dell' Hadith prosperò in un' atmosfera così studiosa. Filosofia, Fiqh e teologia (kalaam) sono stati ulteriormente sviluppati, in particolare da Avicenna e dai suoi avversari. Al-Razi e Al-Farabi avevano fornito metodologie e conoscenze in medicina e filosofia. Avicenna ha avuto accesso alle grandi biblioteche di Balkh, Khwarezm, Gorgan, Rey, Isfahan e Hamadan. Vari testi (come il' Ahd con Bahmanyar') mostrano che egli ha dibattuto punti filosofici con i più grandi studiosi del tempo. Aruzi Samarqandi descrive come prima che Avicenna lasciasse Khwarezm aveva conosciuto Al-Biruni (un famoso scienziato e astronomo), Abu Nasr Iraqi (un famoso matematico), Abu Sahl Masihi (un illustre filosofo) e Abu al-Khayr Khammar (un grande medico).",
  'question': "Che cosa è stato un tema che Avicenna ha ulteriormente sviluppato?",
  'answers': {
    'answer_start':  array([95]),
    'text': array(['teologia'], dtype=object)
  }
}
```
```json
{
  'context': "Florida Alta Velocità ferroviaria è stata proposta ferroviaria ad alta velocità sostenuta dal governo che avrebbe collegato Miami, Orlando e Tampa. La prima fase è stata pianificata per collegare Orlando e Tampa ed è stato offerto un finanziamento federale, ma è stato respinto dal governatore Rick Scott nel 2011. La seconda fase della linea è stata prevista per collegare Miami. Entro il 2014, un progetto privato conosciuto come All Aboard Florida da parte di una società della storica Florida East Coast Railway ha iniziato la costruzione di una linea ferroviaria ad alta velocità nel sud della Florida che dovrebbe terminare all' aeroporto internazionale di Orlando.",
  'question': "In quale anno ha iniziato All Aboard Florida?",
  'answers': {
    'answer_start': array([390]),
    'text': array(['2014'], dtype=object)
  }
}
```
```json
{
  'context': "Gli insetti sociali, come le termiti, le formiche e molte api e vespe, sono la specie più familiare di animali eusociali. Vivono insieme in grandi colonie ben organizzate che possono essere così strettamente integrate e geneticamente simili che le colonie di alcune specie sono talvolta considerate superorganismi. Talvolta si sostiene che le varie specie di api da miele siano gli unici invertebrati (e addirittura uno dei pochi gruppi non umani) ad aver evoluto un sistema di comunicazione simbolica astratta in cui un comportamento viene utilizzato per rappresentare e trasmettere informazioni specifiche su qualcosa nell' ambiente. In questo sistema di comunicazione, chiamato linguaggio dance, l' angolo in cui una danza d' ape rappresenta una direzione relativa al sole, e la lunghezza della danza rappresenta la distanza da volare. 309-311 Anche se forse non così avanzato come le api mellifere, anche i bombi hanno potenzialmente alcuni comportamenti di comunicazione sociale.",
  'question': "Termiti, api, vespe e quali altri insetti sono insetti sociali?",
  'answers': {
    'answer_start': array([41]),
    'text': array(['formiche'], dtype=object)
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

This dataset was published in [this paper](https://aclanthology.org/2024.acl-long.44/) and features multiple-choice reading comprehension questions across 122 languages.

The original dataset contains 900 unique multiple-choice reading comprehension passages and questions. From these, we use a 256 / 64 / 580 split for training, validation and testing, respectively.

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
  Réponse: {label}
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
  Réponse: {label}
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
