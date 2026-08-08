<!-- markdownlint-disable MD013 -->

# <img src='data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 50 28" width="1000" height="560"><path d="M0,0 v28 h50 v-28 z" fill="#D52B1E"/><path d="M0,0 L50,28 M50,0 L0,28" stroke="#009B48" stroke-width="4.3"/><path d="M25,0 v28 M0,14 h50" stroke="#fff" stroke-width="4.3"/></svg>' alt='' style='height:0.9em;vertical-align:-0.05em;display:inline-block'> Basque

This is an overview of all the datasets used in the Basque part of EuroEval. The
datasets are grouped by their task – see the [task overview](/tasks) for more
information about what these constitute.

## Logical Reasoning

### ZebraPuzzleEasy-eu

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2511.03553)
and consists of logic grid puzzles (also known as Einstein's riddles or Zebra puzzles),
where the task is to determine which attributes belong to which house based on a set of
clues. This is the easy variant with 2 houses and 3 attribute categories.

The original full dataset consists of 128 / 128 / 1,024 samples for training, validation
and testing, respectively (so 1,280 samples used in total). We use the same splits.

Here are a few examples from the training split:

```json
{
  "text": "Etxe-ilara bat 1etik 2era dago zenbakitua, ezkerretik eskuinera.\n\nEtxe bakoitzean pertsona bat bizi da, eta honako kategoria bakoitzeko ezaugarri bakarra du:\n\nNazionalitateak: Espainia eta Suedia.\nLanbideak: erizain eta poliziakide.\nZaletasunak: futbol eta kakorratz.\n\nHonako hauek ere badakizkigu:\n\n1. Eski-jauzi-ikuslea ez da 2. etxean bizi.\n2. Espainiarra gitarra jotzen duen pertsonaren ondoan bizi da.\n3. Erizaina bizikletadun pertsonaren ondoan bizi da.\n4. Poliziakidea kakorratz-egilearen eskuinean bizi da.\n5. Kalean auto ugari daude.\n6. Kakorratz-egileak badaki kafeak kafeina duela.\n7. Espainiarra erizaina da.",
  "target_text": {
    "object_1": [
      "Espainia",
      "erizain",
      "kakorratz"
    ],
    "object_2": [
      "Suedia",
      "poliziakide",
      "futbol"
    ]
  }
}
```

```json
{
  "text": "Etxe-ilara bat 1etik 2era dago zenbakitua, ezkerretik eskuinera.\n\nEtxe bakoitzean pertsona bat bizi da, eta honako kategoria bakoitzeko ezaugarri bakarra du:\n\nLanbideak: irakasle eta ministro.\nEdariak: esne eta zuku.\nLiburu-generoak: maitasun-nobela eta nobela beltza.\n\nHonako hauek ere badakizkigu:\n\n1. Zuko-edalea Kanadan egon da.\n2. Irakasleak ez du zukurik edaten.\n3. Esne-edalea bere lajiarentzat zaharra den animalia duen pertsonaren ondoan bizi da.\n4. Mango bigarren fruitu onena dela uste duena ez du kaktusik.\n5. Bideo-joko-zalea 2. etxean bizi da.\n6. Ministroa maitasun-nobela-irakurlearen eskuinean bizi da.\n7. Ministroa nabigaziozalearen ondoan bizi da.",
  "target_text": {
    "object_1": [
      "irakasle",
      "esne",
      "maitasun-nobela"
    ],
    "object_2": [
      "ministro",
      "zuku",
      "nobela beltza"
    ]
  }
}
```

```json
{
  "text": "Etxe-ilara bat 1etik 2era dago zenbakitua, ezkerretik eskuinera.\n\nEtxe bakoitzean pertsona bat bizi da, eta honako kategoria bakoitzeko ezaugarri bakarra du:\n\nLanbideak: erizain eta okin.\nMaskotak: barraskilo eta untxi.\nLiburu-generoak: poesia eta saiakera.\n\nHonako hauek ere badakizkigu:\n\n1. Okinak ez du poesiarik irakurtzen.\n2. Barraskiloaren jabea 1. etxean bizi da.\n3. Poesia-irakurlea ile gorriko pertsonaren ondoan bizi da.\n4. Barraskiloaren jabeak badaki kalean auto ugari daudela.\n5. Erizaina 2. etxean bizi da.\n6. Poesia-irakurleak badaki kafeak kafeina duela.\n7. Okina akuria du.\n8. Okinak badaki eguzki-sistemak galaxiaren zentroan 200 km/s inguruko abiaduran ibiltzen duela.",
  "target_text": {
    "object_1": [
      "okin",
      "barraskilo",
      "saiakera"
    ],
    "object_2": [
      "erizain",
      "untxi",
      "poesia"
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
  Hona hemen asmakizun bat:
  <riddle>
  {text}
  </riddle>

  Nork zein atributu ditu eta zein etxetan bizi da?

  Mesedez, eman zure erantzuna JSON dictionary gisa. Key bakoitza object_X izan behar da, non X etxeko zenbakia den. Value bakoitza goiko kategorietako atributuen zerrenda bat izan behar da, X. zenbakiko etxeko pertsonari dagozkionak.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset zebra-puzzles-easy-eu
```

### Unofficial: ZebraPuzzleHard-eu

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2511.03553)
and consists of logic grid puzzles (also known as Einstein's riddles or Zebra puzzles),
where the task is to determine which attributes belong to which house based on a set of
clues. This is the hard variant with 4 houses and 5 attribute categories.

The original full dataset consists of 128 / 128 / 1,024 samples for training, validation
and testing, respectively (so 1,280 samples used in total). We use the same splits.

Here are a few examples from the training split:

```json
{
  "text": "Etxe-ilara bat 1etik 4era dago zenbakitua, ezkerretik eskuinera.\n\nEtxe bakoitzean pertsona bat bizi da, eta honako kategoria bakoitzeko ezaugarri bakarra du:\n\nNazionalitateak: Herbehereak, Islandia, Letonia eta Norvegia.\nLanbideak: erizain, irakasle, saltzaile eta software garatzaile.\nEdariak: kafe, kakao, te eta zuku.\nLiburu-generoak: beldurrezko nobela, fantasia, nobela beltza eta zientzia-fikzio.\nFruta gogokoenak: andere-mahats beltz, banana, marrubi eta udare.\n\nHonako hauek ere badakizkigu:\n\n1. Kafe-edalea 4. etxean bizi da.\n2. Kafe-edalea zientzia-fikzio-irakurlearen eskuineko etxean bertan bizi da.\n3. Sardinzarrak arrainak dira.\n4. Norvegiarraren eta software garatzailearen artean 2 etxe daude.\n5. Bideo-joko-zalea ez du kaktusik.\n6. Letoniarra ez da herbeheretarraren eta zuko-edalearen artean bizi, eta hiru pertsona desberdinak dira.\n7. Te-edaleak ez du zientzia-fikziorik irakurtzen.\n8. Islandiarraren eta udare-zalearen artean etxe bat dago.\n9. Letoniarra beldurrezko nobelen irakurlearen ondoan bizi da.\n10. Erizaina nobela-beltzen irakurlearen ondoan bizi da.\n11. Saltzailea ez da fantasia-irakurlearen ondoan bizi, eta pertsona desberdinak dira.\n12. Norvegiarra ez da herbeheretarraren eta irakaslearen artean bizi, eta hiru pertsona desberdinak dira.\n13. Eski-jauzi-ikuslea 2. etxean bizi da.\n14. Letoniarra marrubi-zalearen ezkerreko etxean bertan bizi da.\n15. Barraskiloak molusku dira.\n16. Kafe-edaleak marrubia maite du.\n17. Irakaslea bizikleta du.\n18. Banana-zalea norvegiarraren eta marrubi-zalearen artean bizi da.",
  "target_text": {
    "object_1": [
      "Norvegia",
      "saltzaile",
      "zuku",
      "nobela beltza",
      "andere-mahats beltz"
    ],
    "object_2": [
      "Herbehereak",
      "erizain",
      "te",
      "beldurrezko nobela",
      "udare"
    ],
    "object_3": [
      "Letonia",
      "irakasle",
      "kakao",
      "zientzia-fikzio",
      "banana"
    ],
    "object_4": [
      "Islandia",
      "software garatzaile",
      "kafe",
      "fantasia",
      "marrubi"
    ]
  }
}
```

```json
{
  "text": "Etxe-ilara bat 1etik 4era dago zenbakitua, ezkerretik eskuinera.\n\nEtxe bakoitzean pertsona bat bizi da, eta honako kategoria bakoitzeko ezaugarri bakarra du:\n\nLanbideak: irakasle, okin, poliziakide eta saltzaile.\nMaskotak: barraskilo, makil-intsektu, untxi eta zebra.\nLiburu-generoak: beldurrezko nobela, fantasia, nobela beltza eta poesia.\nZaletasunak: bouldering, futbol, kakorratz eta tenis.\nFruta gogokoenak: andere-mahats beltz, banana, laranja eta sagar.\n\nHonako hauek ere badakizkigu:\n\n1. Saltzaileak ez du sagarra maite.\n2. Banana-zalea makil-intsektuaren jabearen eta laranja-zalearen artean bizi da.\n3. Untxiaren jabea ilea gorria du.\n4. Barraskiloaren jabea kakorratz-egilearen eskuineko etxean bertan bizi da.\n5. Bouldering-egilea laranja-zalearen ezkerreko etxean bertan bizi da.\n6. Beldurrezko nobelen irakurlea ez da andere-mahats-beltz-zalearen eta banana-zalearen artean bizi, eta hiru pertsona desberdinak dira.\n7. Barraskiloaren jabea beldurrezko nobelen irakurlearen eskuinean bizi da.\n8. Fantasia-irakurleak bouldering egiten du.\n9. Okina barraskiloaren jabearen eskuinean bizi da.\n10. Laranja-zalea eta Kanadan egon den pertsona lagunak dira.\n11. Kakorratz-egilea betaurrekoak daramatza.\n12. Untxiaren jabea ez da tenis-jokalariaren eta kakorratz-egilearen artean bizi, eta hiru pertsona desberdinak dira.\n13. Sardinzarrak arrainak dira.\n14. Laranja-zaleak badaki pepinoa baia bat dela.\n15. Poesia-irakurlea bouldering-egilearen ezkerrean bizi da.\n16. Saltzailea irakaslearen ezkerrean bizi da.\n17. Poesia-irakurlea andere-mahats-beltz-zalearen ondoan bizi da.\n18. Kakorratz-egilea 2. etxean bizi da.",
  "target_text": {
    "object_1": [
      "poliziakide",
      "makil-intsektu",
      "beldurrezko nobela",
      "tenis",
      "sagar"
    ],
    "object_2": [
      "saltzaile",
      "zebra",
      "poesia",
      "kakorratz",
      "banana"
    ],
    "object_3": [
      "irakasle",
      "barraskilo",
      "fantasia",
      "bouldering",
      "andere-mahats beltz"
    ],
    "object_4": [
      "okin",
      "untxi",
      "nobela beltza",
      "futbol",
      "laranja"
    ]
  }
}
```

```json
{
  "text": "Etxe-ilara bat 1etik 4era dago zenbakitua, ezkerretik eskuinera.\n\nEtxe bakoitzean pertsona bat bizi da, eta honako kategoria bakoitzeko ezaugarri bakarra du:\n\nNazionalitateak: Frantzia, Islandia, Norvegia eta Suedia.\nLanbideak: erizain, irakasle, ministro eta poliziakide.\nLiburu-generoak: fantasia, maitasun-nobela, poesia eta saiakera.\nZaletasunak: eskubaloi, mahai-joko, margotze eta tenis.\nFruta gogokoenak: andere-mahats beltz, banana, laranja eta marrubi.\n\nHonako hauek ere badakizkigu:\n\n1. Etxe guztiek leiho handiak dituzte.\n2. Irakaslea fisika maite du.\n3. Ministroa 3. etxean bizi da.\n4. Eskubaloi-jokalariaren eta banana-zalearen artean etxe bat dago.\n5. Tatuajedun pertsona 4. etxean bizi da.\n6. Poliziakidea 1. etxean bizi da.\n7. Suediarra fantasia-irakurlearen eskuineko etxean bertan bizi da.\n8. Mango bigarren fruitu onena dela uste duena bideo-jokoak jokatzen ditu.\n9. Frantsesa ez da norvegiarraren ondoan bizi, eta pertsona desberdinak dira.\n10. Norvegiarraren eta laranja-zalearen artean 2 etxe daude.\n11. Erizaina eskubaloi-jokalariaren eskuineko etxean bertan bizi da.\n12. Suediarra margolariaren eskuinean bizi da.\n13. Poesia-irakurlea maitasun-nobela-irakurlearen eskuinean bizi da.\n14. Norvegiarrak ez du fantasiarik irakurtzen.\n15. Bizikletadun pertsona 3. etxean bizi da.\n16. Tenis-jokalaria maitasun-nobela-irakurlearen eta banana-zalearen artean bizi da.\n17. Norvegiarra ez da andere-mahats-beltz-zalearen ondoan bizi, eta pertsona desberdinak dira.",
  "target_text": {
    "object_1": [
      "Norvegia",
      "poliziakide",
      "saiakera",
      "margotze",
      "banana"
    ],
    "object_2": [
      "Islandia",
      "irakasle",
      "fantasia",
      "tenis",
      "marrubi"
    ],
    "object_3": [
      "Suedia",
      "ministro",
      "maitasun-nobela",
      "eskubaloi",
      "andere-mahats beltz"
    ],
    "object_4": [
      "Frantzia",
      "erizain",
      "poesia",
      "mahai-joko",
      "laranja"
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
  Hona hemen asmakizun bat:
  <riddle>
  {text}
  </riddle>

  Nork zein atributu ditu eta zein etxetan bizi da?

  Mesedez, eman zure erantzuna JSON dictionary gisa. Key bakoitza object_X izan behar da, non X etxeko zenbakia den. Value bakoitza goiko kategorietako atributuen zerrenda bat izan behar da, X. zenbakiko etxeko pertsonari dagozkionak.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset zebra-puzzles-hard-eu
```
