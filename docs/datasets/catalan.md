# Catalan

This is an overview of all the datasets used in the Catalan part of EuroEval. The
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

### ScaLA-ca

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Catalan Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Catalan-AnCora) by assuming that the
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

### MMLU-ca

This dataset is a machine translated version of the English [MMLU
dataset](https://openreview.net/forum?id=d7KBjmI3GmQ) and features questions within 57
different topics, such as elementary mathematics, US history and law. The translation to
Catalan was done by the University of Oregon as part of [this
paper](https://aclanthology.org/2023.emnlp-demo.28/), using GPT-3.5-turbo.

The original full dataset consists of 269 / 1,410 / 13,200 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 2,048 split for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
new and there can thus be some overlap between the original validation and test sets and
our validation and test sets.

Here are a few examples from the training split:

```json
{
    "text": "Quin és el valor de y en l'equació y/4 = 8?\nOpcions:\na. 2\nb. 4\nc. 12\nd. 32",
    "label": "d",
}
```

```json
{
    "text": "Mill afirma que una de les objeccions més fortes al utilitarisme es basa en la idea de:\nOpcions:\na. deure.\nb. justícia.\nc. virtut.\nd. supererogació.",
    "label": "b",
}
```

```json
{
    "text": "Tots els següents són exemples de condicions motivadores secundàries MENYS\nOpcions:\na. oci\nb. sexe\nc. aprovació\nd. amistat",
    "label": "b",
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Les següents són preguntes de selecció múltiple (amb respostes).
  ```

- Base prompt template:

  ```text
  Pregunta: {text}
  Resposta: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Pregunta: {text}
  Opcions de resposta:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Respon a la pregunta anterior utilitzant només 'a', 'b', 'c' o 'd', i res més.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset mmlu-ca
```

## Common-sense Reasoning

### Winogrande-ca

This dataset was published in
[this paper](https://doi.org/10.48550/arXiv.2506.19468) and is a translated
and filtered version of the English
[Winogrande dataset](https://doi.org/10.1145/3474381). DeepL was used to
translate the dataset to Catalan.

The original full dataset consists of 47 / 1,210 samples for training and testing, and
we use 128 of the test samples for validation, resulting in a 47 / 128 / 1,085 split for
training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
    "text": "No podia controlar la humitat com controlava la pluja, perquè el _ entrava per tot arreu. A què es refereix el guió _?\nOpcions:\na. humitat\nb. pluja",
    "label": "a"
}
```

```json
{
    "text": "La Jessica pensava que \"Sandstorm\" era la millor cançó mai escrita, però la Patricia la odiava. _ va comprar una entrada per al concert de jazz. A què es refereix el guió _?\nOpcions:\na. Jessica\nb. Patricia",
    "label": "b"
}
```

```json
{
    "text": "El termòstat mostrava que feia vint graus menys a la planta baixa que a la planta superior, així que Byron es va quedar a la _ perquè tenia fred. A què es refereix el guió _?\nOpcions:\na. planta baixa\nb. planta superior",
    "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Les següents són preguntes de selecció múltiple (amb respostes).
  ```

- Base prompt template:

  ```text
  Pregunta: {text}
  Opcions:
  a. {option_a}
  b. {option_b}
  Resposta: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Pregunta: {text}
  Opcions:
  a. {option_a}
  b. {option_b}

  Respon a la pregunta anterior utilitzant només 'a' o 'b', i res més.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset winogrande-ca
```

## Summarisation

### DACSA-ca

This dataset was published in [this paper](https://aclanthology.org/2022.naacl-main.434/).
The original DACSA dataset consists of Catalan and Spanish news articles, but this
configuration (DACSA-ca) contains only Catalan articles.

The original full dataset consists of 636,596 / 35,376 samples for the training and
validation splits, respectively. The dataset has two test splits with 35,376 and
17,836 samples. The first test split contains samples present in the train and
validation splits, while the second split only has samples that are not present in
the training and validation splits.

We use a 1,024 / 256 / 2,048 split for training, validation, and testing, respectively
(totaling 3,328 samples). All new splits are subsets of the original splits.
Importantly, our new test split is exclusively derived from the original test split
that does not overlap with the training and validation splits.

Here are a few examples from the training split:

```json
{
    "text": "Entre 0,50 i 2,50 euros per persona en funció de la ciutat i l'allotjament. Això és el que paguen els turistes a Catalunya per pernoctar al país des que el 2012 es va aprovar la taxa turística. Els diners recaptats, que no han deixat d’augmentar en paral·lel a l’increment dels turistes, s’utilitzen precisament per promocionar aquesta activitat, que és el compromís que es va assolir en el moment en què es va aprovar. ¿Té sentit que segueixi sent així? Segons qui contesti la pregunta. De fet, la taxa turística genera adhesions, crítiques i peticions de canvi a parts iguals. Ciutats com Barcelona i Tarragona, per exemple, demanen incrementar el retorn als municipis, mentre que a la Costa Brava els empresaris del sector continuen oposant-se a cobrar-la, i a la Costa Daurada, en general, es mostren satisfets. En qualsevol cas, el Parlament ja va deixar clar fa més d’un any amb els vots de CiU i ERC que la capital catalana no havia de rebre un tracte singular sobre aquesta taxa, que recapta i gestiona la Generalitat. 540x30. Marques turístique Marques turístique. De mica en mica, la taxa s’ha convertit en una important font de finançament. El 2015 l’impost sobre estades en establiments turístics va recaptar 43,5 milions d’euros a Catalunya -el màxim registrat fins llavors-, d’entre els quals 22,2 els va aportar directament la ciutat de Barcelona, que, tot i que inicialment havia de recuperar-ne el 34% (uns 7,5 milions d’euros), després d’un conveni segellat al desembre amb el Govern va rebre dos milions d’euros addicionals. L’Ajuntament s’ha fixat l’objectiu a curt termini de consolidar aquest retorn extra i, a llarg termini, assolir el 100% de la taxa. També vol acabar amb la imposició que els diners es destinin només a promoció turística. Malestar a la Costa Brav. L’equip de Colau vol poder compensar, per mitjà d’aquesta taxa, les zones que, com Ciutat Vella i la Sagrada Família, suporten més els problemes derivats de la pressió turística. Turisme de Barcelona, que rep la meitat del que torna a la ciutat -l’altra meitat va a parar a mans de l’Ajuntament-, diu que veu amb bons ulls la batalla del consistori. Però això no és el que diuen les comarques gironines, que després de les de Barcelona són les que reben més diners de la taxa: 7,64 milions d’euros l’any passat, dels quals veuran com en tornen 2,26 a través de la partida que la Generalitat destina als ens locals. A les comarques gironines els ingressos també han crescut any rere any, però el sector no està satisfet amb l’aplicació de la taxa. “Ens agradaria rebre’n una petita part. Que un 2% o un 3% es destinés a l’empresariat”, diu Martí Sabrià, gerent de l’associació Grup Costa Brava Centre Hotels. Antoni Escudero, president de la Federació d’Hostaleria de les Comarques de Girona, es mostra encara més contundent: “Mai no hem estat d’acord amb la taxa turística. L’experiència no és gens bona i la majoria dels empresaris turístics voldrien que s’anul·lés”, assegura. Promoció del sol i platj. El sector turístic de la Costa Daurada, per la seva banda, creu que en els dos últims anys ha millorat el tracte que rep la seva oferta de sol i platja en les campanyes de promoció turística del país. Diverses veus de l’empresariat havien criticat en el passat l’escassa presència del sol i platja en general i de la Costa Daurada en particular en l’estratègia de promoció de la Generalitat, en benefici del turisme rural o el cultural. L’argument era que el sol i platja era, darrere de Barcelona, la tipologia de turisme que més diners recaptava, però que, en canvi, aquests recursos no es retornaven en forma de promoció. Ara, però, els empresaris tarragonins accepten que la situació ha millorat. “Hi ha més sensibilitat”, diu Berta Cabré, presidenta de l’Associació de Càmpings de Tarragona, que no qüestiona que es promocioni el turisme d’interior encara que porti menys turistes, sinó “el pes” que té cada modalitat en la promoció. “Cada vegada se’ns té més en compte, i això és fruit d’haver defensat els nostres interessos”, exposa Joan Antón, gerent de la Federació de Turisme de Tarragona. Marta Farrero, directora tècnica del Patronat de Turisme de la Costa Daurada, creu que “s’ha fet un pas endavant en el reconeixement del sol i platja”. En la mateixa línia està el president de l’Associació Hotelera Salou - Cambrils - La Pineda, Xavier Roig, que, tot i això, alerta contra la proposta d’utilitzar els fons recaptats per reduir els impactes del turisme en lloc de fer que serveixin per promocionar-lo. Si això es consumés, avisa: “Seria legítim que els hotelers deixéssim de recaptar la taxa, perquè no s’estaria complint la normativa”. Per tot això, només hi ha unanimitat en una qüestió relacionada amb aquest impost: que aquest 2016 es tornarà a batre el rècord de recaptació. Una bona notícia, com a mínim, per als comptes del Govern.",
    "target_text": "Barcelona vol el 100% del que recapta i localitats com Tarragona també demanen un retorn més elevat."
}
```

```json
{
    "text": "Una dona de 35 anys va morir aquest dimecres al barri de Los Gladiolos de Santa Cruz de Tenerife en un presumpte cas de violència de gènere. L'agressor, de 42 anys, havia estat la seva parella i pare dels seus fills, segons han informat fonts de la Policia Nacional.\xa0Després de 20 anys de relació, estaven separats des d'en feia dos. L'home va abordar\xa0la dona al\xa0replà de l'escala de l'edifici on vivia\xa0ella amb els nens. Van discutir i\xa0ell\xa0la va apunyalar amb un ganivet de caça en presència dels seus dos fills menors d'edat, han assegurat fonts de la investigació. La parella feia tres anys que estava\xa0separada, encara que ell vivia en un bloc pròxim al de la seva exparella al barri de Los Gladiolos a Santa Cruz de Tenerife. Després de cometre presumptament els fets, el detingut es va presentar a la comissaria del districte nord, va confessar els fets i va ser detingut. La\xa0dona va rebre almenys cinc punyalades a l'esquena, una al pit i una altra a l'abdomen. És la primera víctima de l'any per violència de gènere a les Canàries. Tant la víctima com la seva exparella, de 42 anys, són naturals de Tenerife i, segons ha informat la Delegació del Govern a les Canàries, no hi havia denúncies\xa0prèvies per maltractaments en la parella. Més informació de Societat.",
    "target_text": "L'agressor la va abordar davant l'ascensor quan portava els nens a l'escola."
}
```

```json
{
    "text": "Aquesta informació es va publicar originalment el 12 de març de 2016 i, per tant, la informació que hi apareix fa referència a la data especificada. Puig Antich continua sent incòmode. Fins i tot quan se li ha dedicat un monument recordatori, sembla que s’hagi fet amb recança. L’homenatge a l’últim català executat per Franco s’ha traslladat a la part més alta de Nou Barris, una plaça de poc pas i que costa de trobar si no ets del barri. Una plaça presidida per un pàrquing, al damunt del qual hi ha un mirador on s’ha instal·lat el monument al jove anarquista... De fet, Puig Antich i el seu barri pòstum, Roquetes –ell no n’era, d’allà-, comparteixen aquesta incomoditat; tots dos se surten del discurs oficial. L’un era un activista que preconitzava l’acció directa i que defensava una ideologia que se sortia dels canals coneguts de navegació, també de la navegació de l’oposició clandestina al franquisme. I Roquetes és una zona que no ha comptat mai gaire per a la Barcelona benpensant. Li va esquitxar poc l’onada olímpica. El que ha aconseguit Roquetes s’ho han hagut de guanyar els veïns, els darrers dels darrers de la immigració del segle passat; encara ara podem detectar en els seus carrers els edificis d’autoconstrucció dignificats, resseguint carrers on fins i tot un cotxe en primera té problemes per pujar. Fins al 2001 no hi va haver un ascensor per salvar els desnivells més importants.. Esperem que ara, quan el monument ja és a lloc, Puig Antich i Roquetes puguin beneficiar-se mútuament. El diumenge posterior a la inauguració d’Ada Colau, el carrer de la Cantera era un seguit de persones pujant a veure l’obra. Un seguit discret, però continuat. Molts d’ells visitaven Roquetes per primer cop, i anaven preguntant a l’un i l’altre la direcció: no hi ha gaire cartells indicatius. Esperem que això també canviï.",
    "target_text": "«L’homenatge a l'últim català executat per Franco s'ha traslladat a la part més alta de Nou Barris, una plaça de poc pas i que costa de trobar si no ets del barri»."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:

  ```text
  A continuació es mostren els documents amb els resums adjunts.
  ```

- Base prompt template:

  ```text
  Document: {text}
  Resum: {target_text}
  ```

- Instruction-tuned prompt template:

  ```text
  Document: {text}

  Escriu un resum del document anterior.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset dacsa-ca
```
