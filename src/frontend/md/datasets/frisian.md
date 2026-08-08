# 🇳🇱 Frisian

This is an overview of all the datasets used in the Frisian part of EuroEval. The
datasets are grouped by their task – see the [task overview](/tasks) for more
information about what these constitute.

## Logical Reasoning

### ZebraPuzzleEasy-fy

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2511.03553)
and consists of logic grid puzzles (also known as Einstein's riddles or Zebra puzzles),
where the task is to determine which attributes belong to which house based on a set of
clues. This is the easy variant with 2 houses and 3 attribute categories.

The original full dataset consists of 128 / 128 / 1,024 samples for training, validation
and testing, respectively (so 1,280 samples used in total). We use the same splits.

Here are a few examples from the training split:

```json
{
  "text": "In rige hûzen is fan links nei rjochts nûmere fan 1 oant 2.\n\nYn elk hûs wennet in persoan mei in unike eigenskip yn elk fan de folgjende kategoryen:\n\nBanen: bakker en minister.\nDranken: kofje en sap.\nLeafste boekgenres: detektive en fantasy.\n\nFierders witte we it folgjende:\n\n1. De persoan mei in tattoo wennet net op hûsnûmer 1.\n2. De fantasylêzer wennet op hûsnûmer 2.\n3. De persoan mei in kavia wennet net op hûsnûmer 1.\n4. De fantasylêzer spilet kompjûterspultsjes.\n5. De kofjedrinker hâldt fan natuerkunde.\n6. De persoan dy't gitaar spilet wennet op hûsnûmer 2.\n7. De minister wennet op hûsnûmer 2.\n8. De sapdrinker wennet net op hûsnûmer 2.",
  "target_text": {
    "object_1": [
      "bakker",
      "sap",
      "detektive"
    ],
    "object_2": [
      "minister",
      "kofje",
      "fantasy"
    ]
  }
}
```

```json
{
  "text": "In rige hûzen is fan links nei rjochts nûmere fan 1 oant 2.\n\nYn elk hûs wennet in persoan mei in unike eigenskip yn elk fan de folgjende kategoryen:\n\nHúsdieren: hûn en sebra.\nLeafste boekgenres: detektive en horror.\nHobbys: fuotbal en heakje.\n\nFierders witte we it folgjende:\n\n1. De detektivelêzer wennet neist de persoan dy't yn Kanada west hat.\n2. De hûnbesitter wennet op hûsnûmer 2.\n3. De horrorlêzer wennet op hûsnûmer 2.\n4. De persoan mei in masterstitel yn wiskunde wennet net op hûsnûmer 2.\n5. De persoan dy't kompjûterspultsjes spilet hat read hier.\n6. De persoan dy't heaket wennet net op hûsnûmer 2.\n7. De persoan mei in bril hâldt fan natuerkunde.\n8. De sebrabesitter wennet neist de persoan dy't faak silet.",
  "target_text": {
    "object_1": [
      "sebra",
      "detektive",
      "heakje"
    ],
    "object_2": [
      "hûn",
      "horror",
      "fuotbal"
    ]
  }
}
```

```json
{
  "text": "In rige hûzen is fan links nei rjochts nûmere fan 1 oant 2.\n\nYn elk hûs wennet in persoan mei in unike eigenskip yn elk fan de folgjende kategoryen:\n\nDranken: frisdrank en molke.\nLeafste boekgenres: detektive en non-fiksje.\nLeafste fruchtsoarten: ierdbei en sinesapel.\n\nFierders witte we it folgjende:\n\n1. De non-fiksjelêzer wennet op hûsnûmer 2.\n2. De frisdrankdrinker is goede freonen mei de persoan mei in sus.\n3. De persoan dy't fan ierdbeien hâldt wennet op hûsnûmer 1.\n4. De non-fiksjelêzer wennet neist de persoan mei in fyts.\n5. De molkedrinker is goede freonen mei de persoan mei in bril.\n6. De persoan dy't yn Kanada west hat spilet gitaar.\n7. De persoan dy't fan ierdbeien hâldt wit dat kofje kafeïne befettet.\n8. De molkedrinker hâldt fan sinesapels.",
  "target_text": {
    "object_1": [
      "frisdrank",
      "detektive",
      "ierdbei"
    ],
    "object_2": [
      "molke",
      "non-fiksje",
      "sinesapel"
    ]
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt: (empty)
- Instruction prompt:

  ```text
  Hjir is in riedling:
  <riddle>
  {text}
  </riddle>

  Wa hat hokker attributen en wennet yn hokker hûs?

  Leverje jo antwurd as in JSON dictionary. Elk key moat object_X wêze, wêr X it hûsnûmer is. Elk value moat in list wêze fan de attributen út de neamde kategoryen dy't heart by de persoan yn hûs nr. X.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset zebra-puzzles-easy-fy
```

### Unofficial: ZebraPuzzleHard-fy

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2511.03553)
and consists of logic grid puzzles (also known as Einstein's riddles or Zebra puzzles),
where the task is to determine which attributes belong to which house based on a set of
clues. This is the hard variant with 4 houses and 5 attribute categories.

The original full dataset consists of 128 / 128 / 1,024 samples for training, validation
and testing, respectively (so 1,280 samples used in total). We use the same splits.

Here are a few examples from the training split:

```json
{
  "text": "In rige hûzen is fan links nei rjochts nûmere fan 1 oant 4.\n\nYn elk hûs wennet in persoan mei in unike eigenskip yn elk fan de folgjende kategoryen:\n\nNasjonaliteiten: Denemarken, Noarwegen, Spanje en Sweden.\nBanen: bakker, ferpleechkundige, learaar en minister.\nDranken: frisdrank, molke, sap en sûkolademolke.\nLeafste boekgenres: detektive, horror, non-fiksje en wittenskipsfiksje.\nHobbys: boerdspullen, fuotbal, hânbal en tennis.\n\nFierders witte we it folgjende:\n\n1. It sinnestelsel beweecht mei in faasje fan sawat 200 km/s om it sintrum fan de Molkwei.\n2. De Deen is gjin learaar.\n3. Der stiet ien hûs tusken de persoan dy't hânballet en de persoan dy't fuotballet.\n4. De minister wennet net neist de persoan dy't hânballet, en se binne net deselde persoan.\n5. De Deen wennet tusken de Noar en de sûkolademolkedrinker.\n6. De persoan dy't boerdspullen spilet wennet neist de persoan dy't kompjûterspultsjes spilet.\n7. De molkedrinker wennet neist de persoan mei in tattoo.\n8. De learaar tinkt dat mango de op ien nei bêste fruit is.\n9. It is aardich om riedsels op te lossen.\n10. De Sweed wennet direkt rjochts fan de persoan dy't tennist.\n11. De molkedrinker wennet net tusken de wittenskipsfiksjelêzer en de persoan dy't fuotballet, en it binne trije ferskillende persoanen.\n12. Der stiet ien hûs tusken de Deen en de Spanjert.\n13. De horrorlêzer wennet op hûsnûmer 4.\n14. De sapdrinker wennet rjochts fan de non-fiksjelêzer.\n15. Tusken de bakker en de detektivelêzer steane 2 hûzen.\n16. De Noar wennet net tusken de sapdrinker en de wittenskipsfiksjelêzer, en it binne trije ferskillende persoanen.",
  "target_text": {
    "object_1": [
      "Noarwegen",
      "learaar",
      "molke",
      "detektive",
      "hânbal"
    ],
    "object_2": [
      "Denemarken",
      "ferpleechkundige",
      "frisdrank",
      "wittenskipsfiksje",
      "tennis"
    ],
    "object_3": [
      "Sweden",
      "minister",
      "sûkolademolke",
      "non-fiksje",
      "fuotbal"
    ],
    "object_4": [
      "Spanje",
      "bakker",
      "sap",
      "horror",
      "boerdspullen"
    ]
  }
}
```

```json
{
  "text": "In rige hûzen is fan links nei rjochts nûmere fan 1 oant 4.\n\nYn elk hûs wennet in persoan mei in unike eigenskip yn elk fan de folgjende kategoryen:\n\nNasjonaliteiten: Denemarken, Faröer, Nederlân en Noarwegen.\nBanen: bakker, ferpleechkundige, learaar en winkelbetsjinner.\nHúsdieren: hûn, knyn, slak en wanneltwiich.\nLeafste boekgenres: horror, non-fiksje, poëzy en wittenskipsfiksje.\nHobbys: boerdspullen, boulderjen, fuotbal en hânbal.\n\nFierders witte we it folgjende:\n\n1. Tusken de Noar en de knynbesitter steane 2 hûzen.\n2. De winkelbetsjinner hat in kavia.\n3. De learaar wennet op hûsnûmer 3.\n4. De winkelbetsjinner wennet direkt lofts fan de persoan dy't boerdspullen spilet.\n5. De persoan dy't hânballet hat read hier.\n6. De horrorlêzer wennet net neist de wittenskipsfiksjelêzer, en se binne net deselde persoan.\n7. Der stiet ien hûs tusken de Nederlanner en de horrorlêzer.\n8. De wanneltwiichbesitter wit dat der folle auto's op 'e dyk binne.\n9. De Nederlanner wennet direkt lofts fan de persoan dy't fuotballet.\n10. De persoan dy't kompjûterspultsjes spilet hat in fyts.\n11. De Deen wennet op hûsnûmer 3.\n12. De hûnbesitter wennet direkt lofts fan de wanneltwiichbesitter.\n13. De Deen wennet neist de persoan dy't nei skânsspringen sjocht.\n14. De wanneltwiichbesitter wennet net neist de persoan dy't bouldert, en se binne net deselde persoan.\n15. De bakker wennet direkt lofts fan de hûnbesitter.\n16. De Deen wennet direkt lofts fan de poëzylêzer.",
  "target_text": {
    "object_1": [
      "Nederlân",
      "bakker",
      "knyn",
      "wittenskipsfiksje",
      "boulderjen"
    ],
    "object_2": [
      "Faröer",
      "winkelbetsjinner",
      "hûn",
      "non-fiksje",
      "fuotbal"
    ],
    "object_3": [
      "Denemarken",
      "learaar",
      "wanneltwiich",
      "horror",
      "boerdspullen"
    ],
    "object_4": [
      "Noarwegen",
      "ferpleechkundige",
      "slak",
      "poëzy",
      "hânbal"
    ]
  }
}
```

```json
{
  "text": "In rige hûzen is fan links nei rjochts nûmere fan 1 oant 4.\n\nYn elk hûs wennet in persoan mei in unike eigenskip yn elk fan de folgjende kategoryen:\n\nNasjonaliteiten: Denemarken, Grut-Brittanje, Letlân en Nederlân.\nBanen: ferpleechkundige, learaar, plysjeman en softwareûntwikkelder.\nLeafste boekgenres: fantasy, horror, non-fiksje en poëzy.\nHobbys: boulderjen, fuotbal, heakje en tennis.\nLeafste fruchtsoarten: apel, banaan, sinesapel en swarte bes.\n\nFierders witte we it folgjende:\n\n1. De plysjeman lêst gjin poëzy.\n2. De Deen wennet direkt rjochts fan de fantasylêzer.\n3. De Brit hâldt fan banannen.\n4. De poëzylêzer wennet lofts fan de persoan dy't fan swarte bessen hâldt.\n5. De persoan dy't fan sinesapels hâldt wit dat kofje kafeïne befettet.\n6. De persoan mei in fyts wennet op hûsnûmer 1.\n7. Tusken de ferpleechkundige en de persoan dy't heaket steane 2 hûzen.\n8. De Nederlanner wennet lofts fan de non-fiksjelêzer.\n9. De Brit lêst gjin non-fiksje.\n10. Komkommer is in beze.\n11. De non-fiksjelêzer wennet net neist de persoan dy't tennist, en se binne net deselde persoan.\n12. De horrorlêzer wennet net tusken de persoan dy't bouldert en de persoan dy't fan swarte bessen hâldt, en it binne trije ferskillende persoanen.\n13. De horrorlêzer is goede freonen mei de persoan dy't tinkt dat mango de op ien nei bêste fruit is.\n14. De persoan dy't fan swarte bessen hâldt wennet tusken de horrorlêzer en de persoan dy't fan sinesapels hâldt.\n15. De ferpleechkundige wennet net tusken de poëzylêzer en de persoan dy't bouldert, en it binne trije ferskillende persoanen.\n16. De softwareûntwikkelder wit dat hjerring in fisk is.\n17. Der stiet ien hûs tusken de softwareûntwikkelder en de persoan dy't fan apels hâldt.",
  "target_text": {
    "object_1": [
      "Nederlân",
      "learaar",
      "poëzy",
      "heakje",
      "sinesapel"
    ],
    "object_2": [
      "Letlân",
      "softwareûntwikkelder",
      "non-fiksje",
      "fuotbal",
      "swarte bes"
    ],
    "object_3": [
      "Grut-Brittanje",
      "plysjeman",
      "fantasy",
      "boulderjen",
      "banaan"
    ],
    "object_4": [
      "Denemarken",
      "ferpleechkundige",
      "horror",
      "tennis",
      "apel"
    ]
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt: (empty)
- Instruction prompt:

  ```text
  Hjir is in riedling:
  <riddle>
  {text}
  </riddle>

  Wa hat hokker attributen en wennet yn hokker hûs?

  Leverje jo antwurd as in JSON dictionary. Elk key moat object_X wêze, wêr X it hûsnûmer is. Elk value moat in list wêze fan de attributen út de neamde kategoryen dy't heart by de persoan yn hûs nr. X.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset zebra-puzzles-hard-fy
```
