# Catalan

This is an overview of all the datasets used in the Croatian part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### GuiaCat

This dataset was published in [here](https://huggingface.co/datasets/projecte-aina/GuiaCat).
The dataset consists of 5,750 restaurant reviews in Catalan, with five associated scores
and a sentiment label. The data was provided by [GuiaCat](https://guiacat.cat/).

The original full dataset consists of 4,750 / 500 / 500 samples for the training,
validation and test splits, respectively. We use 1,024 / 256 / 2,048 samples for our
training, validation and test splits, respectively. The train and validation splits
are subsets of the original dataset, while the test split has additional samples from
the train split to reach the desired size.

Here are a few examples from the training split:

```json
{
    "text": "Ens han servit un menú d'allò més bo, amb la característica de la cuina catalana, canelons, i conill, el preu correcte, hem sortit satisfets.",
    "label": "positive"
}
```

```json
{
    "text": "Cuina catalana típica de la regió amb un menú molt variat. El menú és econòmic i de qualitat",
    "label": "positive"
}
```

```json
{
    "text": "Lloc petit,molt tranquil i acollidor. Molt bons els embotits i les carns, que les agafen de la  propia xarcuteria que tenen abaix. Sempre que hi he anat sempre acabo comprant alguna que altre cosa de lo que m'ha agradat. Gran lloc per anar acompanyat. Molt recomanable als amants dels embotits que disfrutaran com nens petits . ",
    "label": "positive"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Els documents següents i el seu sentiment, que pot ser positiu, neutral o negatiu.
  ```

- Base prompt template:

  ```text
  Document: {text}
  Sentiment: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Document: {text}

  Classifiqueu el sentiment en el document. Contesteu només amb positiu, neutral, o negatiu, i res més.
  ```

- Label mapping:
  - `positive` ➡️ `positiu`
  - `neutral` ➡️ `neutral`
  - `negative` ➡️ `negatiu`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset guia-cat
```

## Named Entity Recognition

### WikiANN-ca

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
    "tokens": ["Actualment", "es", "conserva", "al", "Museu", "Nacional", "d'Art", "de", "Catalunya", "."],
    "labels": ["O", "O", "O", "O", "B-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "O"],
}
```

```json
{
    "tokens": ["Carlos", "Henrique", "Casimiro"],
    "labels": ["B-PER", "I-PER", "I-PER"],
}
```

```json
{
    "tokens": ["''", "Megalancistrus", "parananus", "''"],
    "labels": ["O", "B-LOC", "I-LOC", "O"],
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Aquestes són frases i diccionaris JSON amb els noms que apareixen en les frases.
  ```

- Base prompt template:

  ```text
  Frase: {text}
  Entitats anomenades: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Frase: {text}

  Identifiqueu les entitats anomenades en la frase. Mostreu-les com a diccionari JSON
  amb les claus 'persona', 'lloc', 'organització' i 'miscel·lània'. Els valors han de ser
  els llistats de les entitats anomenades del tipus, tal com apareixen en la frase.
  ```

- Label mapping:
  - `B-PER` ➡️ `persona`
  - `I-PER` ➡️ `persona`
  - `B-LOC` ➡️ `lloc`
  - `I-LOC` ➡️ `lloc`
  - `B-ORG` ➡️ `organització`
  - `I-ORG` ➡️ `organització`
  - `B-MISC` ➡️ `miscel·lània`
  - `I-MISC` ➡️ `miscel·lània`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset wikiann-ca
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
    "text": "El portaveu del govern català ha demanat a Maragall que \"ens deixi fer, perquè al final les coses les hem de fer nosaltres dins de la nostra coalició i que cadascú s' ocupi de casa seva\".",
    "label": "correct"
}
```

```json
{
    "text": "La petrolera francesa Total llançarà una oferta de compra (OPA) sobre el 41 % de la Petrofina belga per 1,8 bilions de pessetes.",
    "label": "incorrect"
}
```

```json
{
    "text": "Els fluids de l'organisme poden trencar les boles, i això provoca la intoxicació mortal.",
    "label": "correct"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Les següents oracions indiquen si són gramaticalment correctes.
  ```

- Base prompt template:

  ```text
  Oració: {text}
  Gramaticalment correcta: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Oració: {text}

  Determina si l'oració és gramaticalment correcta o no. Respon amb 'sí' si és correcta, i amb 'no' si no ho és. Respon només amb aquesta paraula i res més.
  ```

- Label mapping:
  - `correct` ➡️ `sí`
  - `incorrect` ➡️ `no`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-ca
```

## Reading Comprehension

### MultiWikiQA-ca

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
    "context": "Ca l'Alzamora és un casa d'Anglesola (Urgell) inclosa a l'Inventari del Patrimoni Arquitectònic de Catalunya.\n\nDescripció \nEdifici típicament renaixentista amb una tipologia quadrangular i dividida en tres pisos. Consta d'una planta baixa que es caracteritza per una entrada principal situada al mig i dos més als laterals. La principal té una llinda amb relleu esglaonat i adopta uns volums ondulants combinant-ho amb perfils rectes. Les portes laterals són en forma d'arc rebaixat. Entre la planta baixa i la segona hi ha una cornisa sobresortint que fa de separació entre l'una i l'altra. Aquesta primera planta es caracteritza per la senzillesa de tres obertures rectangulars amb perfils totalment rectes, sense cap mena de decoració aparent. Finalment, la tercera planta o golfes és de gran alçada i feta amb maons, construcció totalment contemporània.\n\nL'interior de l'habitatge conserva quasi intactes l'estructura del Casal del  i XIX en el que cal destacar una interessant balustrada de fusta a l'escala principal.\n\nHistòria \nLa façana tenia segons documents gràfics del 1921, dos porxos, però sembla que en tenia més. Els dos últims foren enderrocats el 1936-1937. La casa fou propietat de la família Alzamora, notaris fins que passà a la dels Mestres Apotecaris d'Anglesola. La primitiva façana tenia un interessant ràfec de diferents esglaons fets amb rajola.\n\nReferències \n\nPatrimoni monumental d'Anglesola\nEdificis d'Anglesola",
    "question": "Com són les portes dels costats de la planta baixa?",
    "answers": {
        "answer_start": [470],
        "text": ["arc rebaixat"]
    }
}
```

```json
{
    "context": "The Circus (El circ) és una pel·lícula muda de 1928 dirigida per Charles Chaplin.\n\nArgument \nCharlot es troba vagant en una fira, on és confós per un lladre i és perseguit per la policia. Fugint de l'oficial entra en un circ i sense adonar-se'n es converteix en l'estrella de la funció. Aconsegueix escapar. El propietari del circ, veient que es troba a la ruïna i que Charlot feia riure, el crida i li fa una prova que esdevé un fracàs. Després, uns treballadors del circ, que no havien cobrat la seva paga, se'n van enmig de la funció, per la qual cosa el director demana a l'encarregat que contracti el primer home que vegi, que resulta ser Charlot, que des d'un forat de la carpa observa la filla de l'amo. Charlot comença a treballar portant el material d'un mag a l'escenari i acaba per arruïnar-ho tot. Però a la gent li fa gràcia i el propietari del circ s'adona que és una estrella. Charlot no ho sap, i és contractat i explotat com un simple treballador. La filla de l'amo diu a Charlot que ell és l'estrella del circ i tot i així li donen un tracte miserable. El propietari vol colpejar la seva filla, però Charlot l'amenaça dient que si la segueix tractant igual i no li paga més se n'anirà. L'amo accepta les condicions i segueixen treballant fins que un dia una vident llegeix el futur a la noia i li diu que el seu gran amor es troba a prop. Charlot, creient que és ell, es disposa a proposar-li el matrimoni, però s'adona que és un altre: l'equilibrista. En una de les funcions l'equilibrista no compareix, i Charlot el substitueix. La seva actuació és pèssima: va a parar en una fruiteria. Ràpidament torna a entrar al circ justament quan l'amo pega la seva filla. Charlot dona un cop de puny a l'amo i automàticament és acomiadat. La noia li demana que se l'emporti amb ell. Charlot li diu que amb ell no tindrà futur, però que té una solució. Torna al circ, crida l'equilibrista i li diu que ha de proposar el matrimoni a la filla de l'amo, cosa que fa de seguida. Ja casats, el propietari del circ intenta tornar a tractar malament la seva filla, i l'equilibrista li para els peus. L'amo els pregunta si volen conservar la seva feina. Ells diuen que sí, però amb la condició que també contracti Charlot. Així ho fa. El circ marxa, però Charlot decideix no anar-se'n amb ells.\n\nRepartiment \n Charlie Chaplin: un vagabund\n A l'Ernest Garcia: propietari del circ\n Merna Kennedy: fillastra del propietari del circ\n Harry Crocker: Rex\n George Davis: mag\n Henry Bergman: pallasso\n Steve Murphy: lladre\n Tiny Sandford\n John Rand\n\nAl voltant de la pel·lícula \nThe Circus es va convertir en la setena pel·lícula més taquillera de la història del cinema mut: va recaptar quasi quatre milions de dòlars.\n\nLa producció de la pel·lícula va ser l'experiència més difícil en la carrera de Chaplin. Va tenir nombrosos problemes, incloent-hi un incendi en l'estudi de foc, i la filmació va ser interrompuda durant gairebé un any per l'amarg divorci de Chaplin, de la seva segona dona, Lita Grey, i les reclamacions d'impostos per part de l'Internal Revenue Service.\n\nVa ser nominat com a l'Oscar al millor actor i al millor director d'una comèdia. Chaplin va ser guardonat amb un Oscar honorífic per la versatilitat i el talent per a actuar, escriure, dirigir i produir la pel·lícula.\n\nEl 1970, Chaplin va tornar a editar la pel·lícula acompanyada de música.\n\nEnllaços externs \n \n\nPel·lícules mudes dirigides per Charles Chaplin\nPel·lícules dels Estats Units del 1928\nCirc\nPel·lícules dels Estats Units en blanc i negre",
    "question": "En quin any va Chaplin reeditar el film amb una banda sonora?",
    "answers": {
        "answer_start": [3292],
        "text": ["1970"]
    }
}
```

```json
{
    "context": "La paràbola de la figuera estèril és una narració de Jesús que recull l'evangeli segons Lluc (Lc 13, 1).\n\nArgument \nUn home tenia una figuera que feia tres anys que no donava fruit. Va plantejar-se de tallar-la, ja que els treballs que li suposava tenir-ne cura no compensaven si era un arbre estèril. L'amo dels terrenys el va instar a conservar-la un any més, posant-hi adob i especial esforç abans de tallar-la.\n\nAnàlisi \nDéu és el propietari que espera amb paciència que l'arbre (el creient) doni fruit (es converteixi o es comporti segons els preceptes de la fe). Envia els mitjans perquè això passi però adverteix que si continua sent un arbre estèril, es tallarà, és a dir, es condemnarà l'ànima d'aquella persona. La paràbola pertany al grup de narracions sobre la necessitat de seguir Jesús si es vol gaudir del cel, en una línia similar al que es relata a la paràbola de les verges nècies i prudents. La figuera ha gaudit de tres anys, per això ara té un ultimàtum. Aquests tres anys són paral·lels al que dura el Ministeri de Jesús.\n\nTal com passa a la paràbola del gra de mostassa, Jesús usa un símil vegetal parlant de germinació per fer referència al creixement espiritual. L'elecció de la figuera no és casual, era un conreu freqüent a la zona de l'audiència dels evangelis i sovint s'ha identificat amb Israel.\n\nReferències \n\nFiguera Esteril",
    "question": "A què fa referència el fruit en aquesta paràbola?",
    "answers": {
        "answer_start": [508],
        "text": ["es converteixi o es comporti segons els preceptes de la fe"]
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

```text
Els textos següents contenen preguntes i respostes.
```

- Base prompt template:

```text
Text: {text}
Pregunta: {question}
Resposta amb un màxim de 3 paraules:
```

- Instruction-tuned prompt template:

```text
Text: {text}

Respon a la següent pregunta sobre el text anterior amb un màxim de 3 paraules.

Pregunta: {question}
```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-ca
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
