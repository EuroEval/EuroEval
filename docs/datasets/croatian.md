# 🇭🇷 Croatian

This is an overview of all the datasets used in the Croatian part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### MMS-hr

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2306.07902).
The corpus consists of 79 manually selected datasets from over 350 datasets reported in the
scientific literature based on strict quality criteria.

The original dataset contains a single split with 77,594 Croatian samples. We use
1,024 / 256 / 2,048 samples for our training, validation and test splits, respectively.

Here are a few examples from the training split:

```json
{
    "text": "ali kako mozete biti ovako trijezni u ovo doba ajde molim vas",
    "label": "negative"
}
```

```json
{
    "text": "RT @bsimun: Thompson okupio 100 000 ljudi u Čavoglavama. Sad će valjda platiti porez. #domoljublje #DanPobjede",
    "label": "neutral"
}
```

```json
{
    "text": "\n Šesti \"El Clásico\" za\n \n  Luku Modrića\n \n bio je i najdraži. Real je dobio Barçu 3-1, a hrvatski veznjak bio je jedan od najboljih igrača \"kraljeva\".\n\n\n\n - Otkako sam u Madridu, meni je to djelovalo kao\n \n  najuvjerljivija demonstracija moći\n \n . Barça je izgledala manje moćno jer je Real odigrao impresivno. Meni ta pobjeda više govori o snazi naše momčadi, o potvrdi kako forma koju iskazujemo već osam-devet utakmica nije slučajna - rekao je Luka za\n \n  SN\n \n .\n\n\n - Imali su psihološku prednost zbog stanja na ljestvici i manjeg imperativa. Zato je\n \n  Realov uspjeh impresivan\n \n , tim prije što smo gubili 0-1 - dodao je.\n\n\n\n  Izvorni članak pročitajte u\n  \n   Sportskim novostima\n  \n  .\n \n\n\n Pohvalio suigrače\n  \n\n\n -\n \n  Čudesna utakmica\n \n cijele momčadi i pobjeda protiv Barcelone. Ajmo, halá Madrid! - napisao je Modrić na društvenim mrežama.\n  \n",
    "label": "positive"
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
euroeval --model <model-id> --dataset mms-hr
```

## Named Entity Recognition

### WikiANN-hr

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
    "tokens": array(["Ubrzo", "su", "uslijedile", "narudžbe", "iz", "cijele", "Britanske", "zajednice", "naroda", "."], dtype=object),
    "labels": ["O", "O", "O", "O", "O", "O", "B-ORG", "I-ORG", "I-ORG", "O"]
}
```

```json
{
    "tokens": array(["``", "(", "Cole", "Porter", ")"], dtype=object),
    "labels": ["O", "O", "B-PER", "I-PER", "O"]
}
```

```json
{
    "tokens": array(["'", "''", "La", "Liga", "2009.", "/", "10", "."], dtype=object),
    "labels": ["O", "O", "B-ORG", "I-ORG", "O", "O", "O", "O"]
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Sljedeće su rečenice i JSON rječnici s imenicama koje se pojavljuju u rečenicama.
  ```

- Base prompt template:

  ```text
  Rečenica: {text}
  Imenovane entiteti: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Rečenica: {text}

  Identificirajte imenovane entitete u rečenici. Prikažite ih kao JSON rječnik s ključevima 'osoba', 'mjesto', 'organizacija' i 'razno'. Vrijednosti trebaju biti popisi imenovanih entiteta navedenog tipa, točno kako se pojavljuju u rečenici.
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
euroeval --model <model-id> --dataset wikiann-hr
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

### MultiWikiQA-sl

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
    "context": "1891 (MDCCCXCI) je bilo navadno leto, ki se je po gregorijanskem koledarju začelo na četrtek, po 12 dni počasnejšem julijanskem koledarju pa na torek.\n\nDogodki \n 15. maj – papež Leon XIII. izda encikliko Rerum Novarum\n\nRojstva \n 22. januar - Antonio Gramsci, italijanski filozof, politik in politični teoretik († 1937)\n 24. januar - Abraham Samojlovič Bezikovič, ruski matematik († 1970)\n 11. marec - Michael Polanyi, madžarsko-britanski kemik in filozof († 1976)\n 24. marec - John Knittel, švicarski pisatelj († 1970)\n 14. april - Bhimrao Ramdži Ambedkar, indijski budistični socialni reformator, pravnik in filozof († 1956)\n 22. april - sir Harold Jeffreys, angleški geofizik, astronom, matematik († 1989)\n 23. april - Sergej Sergejevič Prokofjev, ruski skladatelj, pianist († 1953)\n 15. maj - Mihail Afanasjevič Bulgakov, ruski pisatelj († 1940)\n 18. maj - Rudolf Carnap, nemški filozof († 1970)\n 16. junij - Vladimir Aleksandrovič Albicki, ruski astronom († 1952)\n 19. avgust - Milton Lasell Humason, ameriški astronom († 1972)\n 26. september - Hans Reichenbach, nemški filozof († 1953)\n 20. oktober - sir James Chadwick, angleški fizik, nobelovec 1935 († 1974)\n 12. november - Seth Barnes Nicholson, ameriški astronom († 1963)\n 15. november - Erwin Rommel, nemški feldmaršal in vojaški strateg († 1944)\n 26. december - Henry Miller, ameriški pisatelj († 1980)\n\nSmrti \n 23. junij - Norman Robert Pogson, angleški astronom (* 1829)\n 23. junij - Wilhelm Eduard Weber nemški fizik (* 1804)\n 3. oktober - Édouard Lucas, francoski matematik (* 1842)\n 10. november - Štefan Žemlič, madžarsko-slovenski pisatelj (* 1840)\n 20. december - George Bassett Clark, ameriški astronom, optik (* 1827)\n 29. december - Leopold Kronecker, nemški matematik, logik (* 1823)",
    "question": "Kateri je bil prvi dan leta 1891 po gregorijanskem koledarju?",
    "answers": {
        "answer_start": [82],
        "text": ["na četrtek"]
    }
}
```

```json
{
    "context": "The Night the Light Went On in Long Beach je prvi album v živo skupine Electric Light Orchestra, ki je izšel leta 1974, posnet pa je bil 12. maja 1974 v Long Beach Auditoriumu na Long Beachu. Naslov albuma je sposojen od pesmi »The Night the Lights Went Out in Georgia«, ki jo je posnela Vicki Lawrence.\n\nOzadje in omejena izdaja \nAlbum je bil predviden kot naslednik albuma On the Third Day, a so bili posnetki poškodovani zaradi tehničnih težav tako na odru kot zunaj njega. Težave so se začele ko se je tovornjak z opremo skupine na poti pokvaril, pred koncertom pa skupina ni imela dovolj časa za preizkus zvoka.\n\nŠtevilna prešanja albuma so bila tako slabe kvalitete, da je vodstvo skupine vložilo tožbo proti proizvodnem podjetju. Naslovnico albuma je oblikoval Mick Haggerty. \n\nNa koncu sta se tako ameriška kot britanska založba odločili da ne izdata albuma. Album je tako izšel le v Nemčiji in nekaterih drugih državah, leta 1985 pa je vseeno izšel v Združenem kraljestvu. Album ni nikoli izšel v ZDA, čeprav je bil tja uvožen in se je dobro prodajal, je pa živa verzija »10538 Overture« izšla kot b-stran singla »Evil Woman« z albuma Face the Music. Živa verzija »Roll Over Beethoven« je v ZDA izšla kot b-stran alternativne verzije »Telephone Line« v seriji reizdaj.\n\nObnova \nRemastering v 90. letih je popravil slabo kvaliteto albuma. Odkrito je bilo, da je bila originalna LP plošča masterizirana z uporabo slabše kopije koncerta, zaradi katere je bila kvaliteta zvoka slaba. Originalni trak je bil odkrit v trezorju proizvodnje plošč in album je bil obnovljen v boljši kvaliteti zvoka.\n\nTo je edini živi album ELO iz časa začetkov skupine.\n\nSeznam skladb\n\nZasedba \nJeff Lynne\t– solo vokal, električna kitara\nBev Bevan – bobni\nRichard Tandy – Wurlitzer, Minimoog\nMike de Albuquerque – solo vokal, spremljevalni vokal, bas\nMik Kaminski – violina\nHugh McDowell – čelo\nMike Edwards – čelo\n\nSklici \n\nAlbumi leta 1974\nAlbumi Electric Light Orchestra\nAlbumi v živo\nAlbumi, ki jih je produciral Jeff Lynne",
    "question": "Zaradi česa je bila slaba kakovost albuma The Night the Light Went On in Long Beach odpravljena?",
    "answers": {
        "answer_start": [1287],
        "text": ["Remastering v 90. letih"]
    }
}
```

```json
{
    "context": "Surangel S. Whipps ml., palavski poslovnež in politik; * 9. avgust 1968, Baltimore, Maryland, Združene države Amerike.\n\nOd 21. januarja 2021 je predsednik Palava. Senator je bil od leta 2008 do 2016. Prihaja iz dežele Ngatpang.\n\nZgodnje življenje in izobraževanje \nWhipps se je rodil v Baltimorju v Marylandu materi samohranilki Surangel Whipps Sr., rojeni v Marylandu. Diplomiral je iz poslovne administracije in ekonomije na Univerzi Andrews in magistriral iz poslovne znanosti na kalifornijski univerzi v Los Angelesu. Poleg tega vodi verigo supermarketov Palauan. Na splošnih volitvah leta 2016 v Palavu se je potegoval proti svojemu zetu, predsedniku Thomasu Remengesauju mlajšemu.\n\nMandat \nWhipps je na predsedniških volitvah 2020 kandidiral za predsednika in premagal podpredsednika Raynolda Oiloucha. V intervjuju za Guardian je takratni izvoljeni predsednik Whipps ml. podal izjavo, da se bo Palav odločneje upiral ukrepom kitajske vlade, vključno z nezakonitim ribolovom in vstopom v palavske vode ter obljubil, da bo ohranil priznanje Tajvana. Poleg tega je predlagal distribucijo cepiva proti COVID-19 med Palavčani, s poudarkom na zdravstvenih delavcih.\n\nSklici \nWhipps, Surangel\nWhipps, Surangel",
    "question": "Proti komu je Whipps zmagal na predsedniških volitvah leta 2020?",
    "answers": {
        "answer_start": [790],
        "text": ["Raynolda Oiloucha"]
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Spodaj so besedila z ustreznimi vprašanji in odgovori.
  ```

- Base prompt template:

  ```text
  Besedilo: {text}
  Vprašanje: {question}
  Odgovor v največ 3 besedah:
  ```

- Instruction-tuned prompt template:

  ```text
  Besedilo: {text}

  Odgovorite na naslednje vprašanje o zgornjem besedilu v največ 3 besedah.

  Vprašanje: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-sl
```

## Knowledge

### MMLU-sl

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2410.08928)
and is a machine translated version of the English [MMLU
dataset](https://openreview.net/forum?id=d7KBjmI3GmQ) and features questions within 57
different topics, such as elementary mathematics, US history and law. The translation to
Slovenian was done using DeepL.

The original full dataset consists of 285 / 1,531 / 14,042 samples for training,
validation, and testing, respectively. These splits were merged, duplicates removed, and
new splits were created with 1,024 / 256 / 2048 samples for training, validation, and
testing, respectively.

Here are a few examples from the training split:

```json
{
    "text": "Kaj je deklarativna teorija priznanja?\nMožnosti:\na. Priznanje je odločilno za obstoj državnosti\nb. Priznanje je zgolj deklarativno za državnost, ni pa odločilno\nc. Priznanje je zgolj izjava o interesu\nd. Priznanje zahteva izjavo novoustanovljene državnosti",
    "label": "b",
}
```

```json
{
    "text": "Katera od naslednjih možnosti bi bila verjeten odziv na ugotovljeno nenormalnost ostanka?\nMožnosti:\na. Uporabite logaritemsko funkcionalno obliko namesto linearne\nb. Dodajte zamike spremenljivk na desni strani regresijskega modela\nc. Ocenite model v prvi diferencirani obliki\nd. Iz podatkov odstranite vsa velika odstopanja.",
    "label": "d",
}
```

```json
{
    "text": "To vprašanje se nanaša na naslednje informacije. Stopnje pismenosti med rusko govorečim prebivalstvom pozne carske Rusije in Sovjetske zveze, 1897-1955 Stopnja pismenosti 1897 24% 1917 45% 1926 56% 1937 75% 1939 81.10% 1955 99.90% Vir: Podatki iz popisa prebivalstva in sovjetskega ministrstva za šolstvo Informacije, predstavljene v zgornji tabeli, je najbolje razumeti v katerem od naslednjih zgodovinskih kontekstov?\nMožnosti:\na. Izobraževalna reforma v moderni dobi\nb. Centralizirane in od države usmerjene kampanje modernizacije\nc. Eksperimentiranje s sindikalističnimi oblikami družbenoekonomske organizacije\nd. Druga faza industrializacije v nezahodnem svetu",
    "label": "b",
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Naslednja so vprašanja z več možnostmi (z odgovori).
  ```

- Base prompt template:

  ```text
  Vprašanje: {text}
  Možnosti:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Odgovor: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Vprašanje: {text}

  Odgovorite na navedeno vprašanje z uporabo 'a', 'b', 'c' ali 'd', in nič drugega.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset mmlu-sl
```

## Common-sense Reasoning

### Winogrande-sl

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2506.19468)
and is a translated and filtered version of the English [Winogrande
dataset](https://doi.org/10.1145/3474381).

The original full dataset consists of 47 / 1,210 samples for training and testing, and
we use 128 of the test samples for validation, resulting in a 47 / 128 / 1,085 split for
training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
    "text": "Nisem mogel nadzorovati vlage, kot sem nadzoroval dež, ker je _ prihajal od vsepovsod. Na kaj se nanaša prazno mesto _?\nMožnosti:\na. vlaga\nb. dež",
    "label": "a"
}
```

```json
{
    "text": "Jessica je mislila, da je Sandstorm najboljša pesem, kar jih je bilo kdaj napisanih, vendar jo je Patricia sovražila. _ je kupila vstopnico za jazz koncert. Na kaj se nanaša prazno mesto _?\nMožnosti:\na. Jessica\nb. Patricia",
    "label": "b"
}
```

```json
{
    "text": "Termostat je pokazal, da je bilo spodaj dvajset stopinj hladneje kot zgoraj, zato je Byron ostal v _, ker mu je bilo hladno. Na kaj se nanaša prazno mesto _?\nMožnosti:\na. spodaj\nb. zgoraj",
    "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Naslednja so vprašanja z več možnostmi (z odgovori).
  ```

- Base prompt template:

  ```text
  Vprašanje: {text}
  Možnosti:
  a. {option_a}
  b. {option_b}
  Odgovor: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Vprašanje: {text}
  Možnosti:
  a. {option_a}
  b. {option_b}

  Odgovorite na navedeno vprašanje z uporabo 'a' ali 'b', in nič drugega.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset winogrande-sl
```
