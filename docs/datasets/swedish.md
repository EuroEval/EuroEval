# 🇸🇪 Swedish

This is an overview of all the datasets used in the Swedish part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.


## Sentiment Classification

### SweReC

This dataset was published [in this B.Sc.
thesis](https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1105494&dswid=3392) and
is a manually annotated dataset of Swedish reviews from both Trustpilot and Reco.se.

The original dataset contains 10,757 reviews. We use a split of 1,024 / 256 / 2,048
samples for training, validation, and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Jättebra och rekommenderas till alla",
  "label": "positive"
}
```
```json
{
  "text": "Lugnt och trevlig stämning, inte för bullrigt. god mat, lite mer variation hade önskats på de varma rätterna. trevlig personal, dock missade de att ta dryckesbeställningar från oss vilket var ett litet minus. överlag trevlig ställe.",
  "label": "neutral"
}
```
```json
{
  "text": "Extremt dålig mottagning - både gsm och 3g? samtalen bryts hela tiden och så tar dom betalt för en ny uppkopplingsavgift varje gång.",
  "label": "negative"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Följande är recensioner och deras sentiment, som kan vara 'positiv', 'neutral' eller 'negativ'.
  ```
- Base prompt template:
  ```
  Recension: {text}
  Sentiment: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Recension: {text}

  Klassificera sentimentet i recensionen. Svara med 'positiv', 'neutral' eller 'negativ'.
  ```
- Label mapping:
    - `positive` ➡️ `positiv`
    - `neutral` ➡️ `neutral`
    - `negative` ➡️ `negativ`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset swerec
```


## Named Entity Recognition

### SUC 3.0

This dataset, also known as the Stockholm-Umeå Corpus 3.0, was published
[here](https://doi.org/10.23695%2Fwy84-ar30) and is a manually NER-annotated dataset,
based on Swedish texts from the 1990s. The dataset does not follow the CONLL format, so
we convert it into that format using the following mapping:

- `animal` ➡️ `MISC`
- `event` ➡️ `MISC`
- `inst` ➡️ `ORG`
- `myth` ➡️ `MISC`
- `other` ➡️ `MISC`
- `person` ➡️ `PER`
- `place` ➡️ `LOC`
- `product` ➡️ `MISC`
- `work` ➡️ `MISC`

The dataset consists of 74,245 samples, which we split into 1,024 / 256 / 2,048 samples
for training, validation, and testing, respectively.

Here are a few examples from the training split:

```json
{
  "tokens": array(['Det', 'låter', 'som', 'en', 'västanfläkt', 'jämfört', 'med', 'den', 'i', 'filmen', 'förkättrade', 'biljätten', 'General', 'Motors', ',', 'som', 'friställt', '35000', 'jobbare', 'i', 'staden', 'Flint', ',', 'Michigan', '.'], dtype=object),
  "labels": array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'B-LOC', 'O'], dtype=object)
}
```
```json
{
  "tokens": array(['En', 'liknande', 'kunskapsteoretisk', 'grundfråga', ',', 'fast', 'i', 'mer', 'modernt', 'sofistikerad', 'form', ',', 'når', 'oss', 'nu', 'från', 'Paris', ':'], dtype=object),
  "labels": array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O'], dtype=object)
}
```
```json
{
  "tokens": array(['-', 'Dessvärre', ',', 'sa', 'man', ',', 'vi', 'har', 'ingen', 'Björn', 'Eriksson', 'på', 'passagerarlistan', '.'], dtype=object),
  "labels": array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PER', 'I-PER', 'O', 'O', 'O'], dtype=object)
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:
  ```
  Följande är meningar och JSON-ordböcker med de namngivna enheter som förekommer i den givna meningen.
  ```
- Base prompt template:
  ```
  Mening: {text}
  Namngivna entiteter: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Mening: {text}

  Identifiera de namngivna enheterna i meningen. Du ska outputta detta som en JSON-ordbok med nycklarna 'person', 'plats', 'organisation' och 'diverse'. Värdena ska vara listor över de namngivna enheter av den typen, precis som de förekommer i meningen.
  ```
- Label mapping:
    - `B-PER` ➡️ `person`
    - `I-PER` ➡️ `person`
    - `B-LOC` ➡️ `plats`
    - `I-LOC` ➡️ `plats`
    - `B-ORG` ➡️ `organisation`
    - `I-ORG` ➡️ `organisation`
    - `B-MISC` ➡️ `diverse`
    - `I-MISC` ➡️ `diverse`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset suc3
```


## Linguistic Acceptability

### ScaLA-sv

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Swedish Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Swedish-Talbanken) by assuming
that the documents in the treebank are correct, and corrupting the samples to create
grammatically incorrect samples. The corruptions were done by either removing a word
from a sentence, or by swapping two neighbouring words in a sentence. To ensure that
this does indeed break the grammaticality of the sentence, a set of rules were used on
the part-of-speech tags of the words in the sentence.

The original dataset consists of 6,026 samples, from which we use 1,024 / 256 / 2,048 samples for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
used as-is in the framework.

Here are a few examples from the training split:

```json
{
  "text": "U-länderna måste ta en genväg för att komma i fatt.",
  "label": "correct"
}
```
```json
{
  "text": "Undra att vi blev lite undandragna.",
  "label": "incorrect"
}
```
```json
{
  "text": "Det är också att viktigt ha tillräckligt korta dubbar.",
  "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Följande är meningar och huruvida de är grammatiskt korrekta.
  ```
- Base prompt template:
  ```
  Mening: {text}
  Grammatisk korrekt: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Mening: {text}

  Bestäm om meningen är grammatiskt korrekt eller inte. Svara med 'ja' om meningen är korrekt och 'nej' om den inte är.
  ```
- Label mapping:
    - `correct` ➡️ `ja`
    - `incorrect` ➡️ `nej`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset scala-sv
```


## Reading Comprehension

### ScandiQA-sv

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the Swedish part of the [MKQA
dataset](https://aclanthology.org/2021.tacl-1.82/). The MKQA dataset is based on the
English [Natural Questions dataset](https://aclanthology.org/Q19-1026/), based on search
queries from the Google search engine. The questions and answers were manually
translated to Swedish (and other languages) as part of MKQA, and the contexts were in
ScandiQA-sv machine translated using the [DeepL translation
API](https://www.deepl.com/en/pro-api/). A rule-based approach was used to ensure that
the translated contexts still contained the answer to the question, potentially by
changing the answers slightly.

The original full dataset consists of 6,810 / 500 / 500 samples for training,
validation and testing, respectively (so 3,328 samples used in total).
We use a 1,024 / 256 / 2,048 split for training, validation and testing, respectively,
where the splits are made by randomly sampling from the full dataset without considering
the original train/validation/test splits.

Here are a few examples from the training split:

```json
{
  "context": "I Freedom Cry får spelaren ta rollen som Adéwalé, en frigiven slav från Trinidad som blev Edward Kenways kvartermästare och senare medlem i Assassin Order. Berättelseläget utspelar sig 15 år efter händelserna i Assassin's Creed IV: Black Flag där Adéwalé har blivit en tränad lönnmördare och finner sig själv skeppsbruten i Saint-Domingue, där han ställs öga mot öga med något av det mest brutala slaveriet i Västindien. DLC:n är skriven av Jill Murray, som skrev Liberation och Aveline-innehållet för Black Flag. I februari 2014 meddelades att Freedom Cry skulle släppas som en fristående titel till PlayStation 4 och PlayStation 3 den 18 februari 2014 för Nordamerika och den 19 februari 2014 för Europa. Det släpptes för PC den 25 februari 2014.",
  "question": "När släpptes assassin's creed freedom cry?",
  "answers": {
    "answer_start": array([637]),
    "text": array(['18 februari 2014'], dtype=object)
  }
}
```
```json
{
  "context": 'Political history of the United Kingdom (1945–present)\nÅr 1950 orsakade Koreakriget ett nytt tungt tryck på statskassan för militära utgifter. Detta orsakade en bitter splittring inom Labourpartiet.  De konservativa gjorde åtstramningspolitiken till en viktig fråga i parlamentsvalet 1950. Labour förlorade det mesta av sin stora majoritet. Svängningen var 3,6 % mot dem och de förlorade 78 platser, vilket gav Attlee en knapp majoritet i parlamentet. Ett år senare förlorade Labour dock parlamentsvalet 1951 trots att det fick fler röster än i valet 1945, och faktiskt fler röster än det konservativa partiet.',
  "question": 'Hur många år har det varit sen 1940?',
  "answers": {
    "answer_start": array([388]),
    "text": array(['78'], dtype=object)
  }
}
```
```json
{
  "context": 'Data link layer\nOSI-modellen\nper skikt\n\n\n\n\n7.  Applikationslager[visa]\n\n\nNNTP\nSIP\nSSI\nDNS\nFTP\nGopher\nHTTP\nNFS\nNTP\nSMPP\nSMTP\nSNMP\nTelnet\nDHCP\nNetconf\nmer....\n\n\n\n\n\n\n\n\n6.  Presentationslager[visa]\n\n\nMIME\nXDR\n\n\n\n\n\n\n\n\n5.  Sessionsskikt[visa]\n\n\nNamngiven pipe\nNetBIOS\nSAP\nPPTP\nRTP\nSOCKS\nSPDY\n\n\n\n\n\n\n\n\n4.  Transportlager[visa]\n\n\nTCP\nUDP\nSCTP\nDCCP\nSPX\n\n\n\n\n\n\n\n\n3.  Nätverksskikt[visa]\n\n\nIP\n\nIPv4\nIPv6\n\n\nICMP\nIPsec\nIGMP\nIPX\nAppleTalk\nX.25 PLP\n\n\n\n\n\n\n\n\n2.  Datalänkskiktet[visa]\n\n\nATM\nARP\nIS-IS\nSDLC\nHDLC\nCSLIP\nSLIP\nGFP\nPLIP\nIEEE 802.2\nLLC\nMAC\nL2TP\nIEEE 802.3\nFrame Relay\nITU-T G.hn DLL\nPPP\nX.25 LAPB\nQ.921 LAPD\nQ.922 LAPF\n\n\n\n\n\n\n\n\n1.  Fysiskt lager[visa]\n\n\nEIA/TIA-232\nEIA/TIA-449\nITU-T V-serien\nI.430\nI.431\nPDH\nSONET/SDH\nPON\nOTN\nDSL\nIEEE 802.3\nIEEE 802.11\nIEEE 802.15\nIEEE 802.16\nIEEE 1394\nITU-T G.hn PHY\nUSB\nBluetooth\nRS-232\nRS-449\n\n\n\n\n\n\n\n\n\nv\nt\ne',
  "question": 'Vilket lager av osi-modellen är uppdelad i två delskikt?',
  "answers": {
    "answer_start": array([0]),
    "text": array(['Data link layer'], dtype=object)
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:
  ```
  Nedan följer texter med tillhörande frågor och svar.
  ```
- Base prompt template:
  ```
  Text: {text}
  Fråga: {question}
  Svar på max 3 ord: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Text: {text}

  Besvara följande fråga om texten ovan med högst 3 ord.

  Fråga: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset scandiqa-sv
```


### Unofficial: BeleBele-sv

This dataset was published in [this paper](https://aclanthology.org/2024.acl-long.44/) and features multiple-choice reading comprehension questions across 122 languages.

The original dataset contains 900 unique multiple-choice reading comprehension passages and questions. From these, we use a 256 / 64 / 580 split for training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Sundarbans är det största kustmangrovebältet i världen och sträcker sig 80 km in i Bangladesh och det indiska inlandet från kusten. Sundarbans har antagits på Unescos världsarvslista. Den del av skogen som ligger på indiskt territorium kallas Sundarbans National Park. Skogarna är dock inte bara mangroveträsk — de innehåller några av de sista kvarvarande ståndorterna av de stora djunglerna som en gång täckte Gangesslätten. Sundarban täcker ett område på 3 850 km², varav ungefär en tredjedel utgörs av våtmarker. Sedan 1966 har Sundarbans varit ett reservat för vilda djur, och det uppskattas att det nu finns 400 bengaliska tigrar och omkring 30 000 axishjortar i området.\nFråga: Vilken del av skogen ligger på indiskt territorium?\nSvarsalternativ:\na. Sundarbans National Park\nb. Reservatet för vilda djur\nc. Världsarvet\nd. Gangesslätten",
  "label": "a"
}
```
```json
{
  "text": "Italiens nationella fotboll, tillsammans med det tyska fotbollslaget, är världens näst mest framgångsrika lag och var mästare i FIFA-världscupen år 2006. Populära sporter inkluderar fotboll, basket, volleyboll, vattenpolo, fäktning, rugby, cykel, ishockey, rullskridskohockey och Formel 1. Vintersporter är mest populära i de norra regionerna, där italienare tävlar i internationella tävlingar och olympiska evenemang.\nFråga: I vilken av följande sporter vann Italien en världscup enligt avsnittet?\nSvarsalternativ:\na. Fotboll\nb. Vattenpolo\nc. Basket\nd. Cykel",
  "label": "a"
}
```
```json
{
  "text": "Bokning i förväg ger resenären trygghet och en försäkran om att de kommer att ha någonstans att sova när de anländer till sin destination. Resebyråer har ofta avtal med särskilda hotell, men det kan vara möjligt att boka andra typer av boenden, såsom campingplatser, genom en resebyrå. Resebyråer erbjuder ofta paket som inkluderar frukost, transfer till och från flygplatsen, och till och med paketresor som kombinerar flyg och hotell. De kan också reservera din bokning åt dig om du behöver tid att tänka över erbjudandet eller skaffa fram ytterligare dokument som krävs för din destination (t.ex. visering). Alla ändringar och förfrågningar ska dock gå genom resebyrån först, och inte direkt till hotellet.\nFråga: Vilken typ av resenär kommer sannolikt inte att dra nytta av att använda sig av tjänster från en resebyrå, enligt det som står i texten?\nSvarsalternativ:\na. En obeslutsam resenär\nb. En resenär som är spontan\nc. En resenär som inte har skaffat visum än\nd. En resenär som föredra att boka paketerbjudanden",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Följande är flervalsfrågor (med svar).
  ```
- Base prompt template:
  ```
  Fråga: {text}
  Svarsalternativ:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svar: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Fråga: {text}
  Svarsalternativ:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Besvara följande fråga med 'a', 'b', 'c' eller 'd', och inget annat.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset belebele-sv
```


## Knowledge

### MMLU-sv

This dataset is a machine translated version of the English [MMLU
dataset](https://openreview.net/forum?id=d7KBjmI3GmQ) and features questions within 57
different topics, such as elementary mathematics, US history and law. The translation to
Swedish was done by the University of Oregon as part of [this
paper](https://aclanthology.org/2023.emnlp-demo.28/), using GPT-3.5-turbo.

The original full dataset consists of 269 / 1,410 / 13,200 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 2,048 split for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
new and there can thus be some overlap between the original validation and test sets and
our validation and test sets.

Here are a few examples from the training split:

```json
{
  "text": "Varför är tidpunkten för monumental byggnation vid Ceibal signifikant?\nSvarsalternativ:\na. Det motsäger hypotesen att den monumental byggnationen av Maya i huvudsak inspirerades av Olmekerna.\nb. Det bekräftar att invånarna i Ceibal inspirerades av Olmekerna för att bygga stora plattformar.\nc. Det motsäger hypotesen att utvecklingen av monumental byggnation bland Maya var en intern process.\nd. Det bekräftar att Olmekerna, som byggde de flesta Maya-monumenten, inspirerades av egyptierna.",
  "label": "a"
}
```
```json
{
  "text": "Vilken populationsstatistik visar födelsetalet vid vilket en befolkning precis får tillräckligt med födslar för att ersätta föräldrarna och kompensera för tidiga dödsfall?\nSvarsalternativ:\na. Rå födelsetal\nb. Ersättningstal\nc. Dödlighetstal\nd. Total fertilitetstal",
  "label": "b"
}
```
```json
{
  "text": "En subenhet av DNA och protein som består av 134-baspar långa sträckor av DNA som omger en proteinoktomer kallas (a)\nSvarsalternativ:\na. histon\nb. kromatin\nc. nukleosom\nd. solenoid",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Följande är flervalsfrågor (med svar).
  ```
- Base prompt template:
  ```
  Fråga: {text}
  Svar: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Fråga: {text}

  Besvara följande fråga med 'a', 'b', 'c' eller 'd', och inget annat.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset mmlu-sv
```


### Unofficial: ARC-sv

This dataset is a machine translated version of the English [ARC
dataset](https://doi.org/10.48550/arXiv.1803.05457) and features US grade-school science
questions. The translation to Swedish was done by the University of Oregon as part of
[this paper](https://aclanthology.org/2023.emnlp-demo.28/), using GPT-3.5-turbo.

The original full dataset consists of 1,110 / 297 / 1,170 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 1,024 split for training,
validation and testing, respectively (so 2,304 samples used in total). All new splits
are subsets of the original splits.

Here are a few examples from the training split:

```json
{
  "text": "En typ av fågel i Afrika äter blodsugande insekter från stora däggdjur. Vilket ord beskriver bäst relationen mellan fågeln och däggdjuren?\nSvarsalternativ:\na. mutualism\nb. parasitism\nc. neutralism\nd. kommensalism",
  "label": "a"
}
```
```json
{
  "text": "Mr. Pratt gör en vetenskaplig demonstration. Han blåser upp en ballong, placerar den i en frys och tar sedan ut den efter 10 minuter. Vilket alternativ beskriver bäst ballongens volym när den är i frysen och efter att den har tagits ut och åter tillåtits att värmas upp?\nSvarsalternativ:\na. expanderar i frysen och kontraherar sedan när den blir varmare igen\nb. kontraherar i frysen och expanderar sedan när den blir varmare igen\nc. expanderar i frysen och håller sedan den volymen när den värms upp\nd. kontraherar i frysen och håller sedan den volymen när den värms upp",
  "label": "b"
}
```
```json
{
  "text": "En elev tillsätter vatten och rengöringsmedel till en kopp med jord. Blandningen skakas och tillåts sätta sig. Eleven observerar att silt-partiklar förblir uppsuspenderade långt efter att de andra partiklarna bildar lager på botten av behållaren. Den mest troliga förklaringen är att silt-partiklarna är\nSvarsalternativ:\na. organiska.\nb. upplösta.\nc. mindre tätt packade.\nd. rör sig snabbare.",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Följande är flervalsfrågor (med svar).
  ```
- Base prompt template:
  ```
  Fråga: {text}
  Svarsalternativ:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svar: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Fråga: {text}
  Svarsalternativ:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Besvara följande fråga med 'a', 'b', 'c' eller 'd', och inget annat.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset arc-sv
```


## Common-sense Reasoning

### HellaSwag-sv

This dataset is a machine translated version of the English [HellaSwag
dataset](https://aclanthology.org/P19-1472/). The original dataset was based on both
video descriptions from ActivityNet as well as how-to articles from WikiHow. The dataset
was translated by the University of Oregon as part of [this
paper](https://aclanthology.org/2023.emnlp-demo.28/), using GPT-3.5-turbo.

The original full dataset consists of 9,310 samples. We use a 1,024 / 256 / 2,048 split
for training, validation and testing, respectively (so 3,328 samples used in total).

Here are a few examples from the training split:

```json
{
  "text": "[header] Hur man hittar de perfekta brudtärneklänningarna [title] Internet är en underbar resurs för att hitta brudtärneklänningar. [step] Vi rekommenderar också att bläddra genom populära bröllopstidningar, såsom brudens och moderna brudtärnets tidningar. Rekommenderat är att bruden går och handlar med en eller två av sina brudtärnor och ser vilka stilar de gillar.\nSvarsalternativ:\na. När du har begränsat urvalet kan du sedan få input från dina andra brudtärnor om du önskar det. [title] Vilka är de senaste trenderna i brudtärneklänningar? [title] A-linje klänningar som ser bra ut på alla olika kroppsformer och storlekar är mycket populära.\nb. Tyvärr kan du inte handla lika ofta som om du letade efter matchade brudtärnor. [title] När du väljer din brud, välj tre olika stilar: [step] Klipp längd, klipp tjocklek och från de flesta \"för-skjutna\" stilarna till de grundläggande.\nc. Medan varje brud är annorlunda, alla är både olika och har olika smaker. [title] Se om bruden har en favoritlook för sin bröllopsklänning.\nd. [title] Börja söka efter idéer eller allmänna åsikter om särskilda bröllopsklänningar. [step] Försök att inte bli för stel och sök bara efter några klänningar som du tror kan fungera bra tillsammans.",
  "label": "a"
}
```
```json
{
  "text": "[header] Hur man gör en pedikyr [title] Ta bort all befintlig färg med nagellacksborttagare. [step] Täck toppen på din nagellacksborttagare med en bomullstuss, vänd snabbt upp och ner den och omedelbart upp och ner igen för att applicera lite av produkten. Gnugga sedan nagellacksborttagaren över dina tånaglar för att ta bort färgen.\nSvarsalternativ:\na. [title] Låt dina tånaglar blötläggas i vatten i 10 till 20 minuter. [step] Vatten kan göra dina naglar vitare genom att lösa upp andra föreningar, särskilt syror.\nb. [substeps] Flytta bomullstussen i små, cirkulära rörelser om du har svårt att ta bort färgen. [title] Fyll en fotspa eller en balja med varmt vatten.\nc. [substeps] Om du inte har nagellacksborttagare kan du överväga att använda den vita nagellacksborttagaren från föregående steg för en enklare applikation. [title] Täck dina händer med bandage eller tejp med canvas-lining.\nd. [title] Använd aceton på dina tånaglar. [step] Aceton kan verkligen hjälpa till att ta bort gammalt nagellack från dina naglar.",
  "label": "b"
}
```
```json
{
  "text": "Han fortsätter att klippa gräset. Kameran fokuserar på det rinnande vattnet igen. Den går tillbaka till mannen som klipper gräset. sedan\nSvarsalternativ:\na. den går tillbaka till filmen av mannen som klipper jord.\nb. återvänder till honom och dem som pratar igen.\nc. växlar tillbaka till det rinnande vattnet.\nd. mörk himmel igen.",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Följande är flervalsfrågor (med svar).
  ```
- Base prompt template:
  ```
  Fråga: {text}
  Svarsalternativ:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svar: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Fråga: {text}
  Svarsalternativ:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Besvara följande fråga med 'a', 'b', 'c' eller 'd', och inget annat.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset hellaswag-sv
```


## Summarization

### SweDN

This dataset was published in [this
paper](https://aclanthology.org/2023.emnlp-main.506/) and are based on news articles
from the Swedish newspaper Dagens Nyheter, with the summaries being the first paragraph
of the article (and that paragraph being removed from the article).

The original dataset consists of 29,800 / 4,530 / 3,750 samples for training, validation
and testing, respectively. We use a 1,024 / 256 / 2,048 split for training, validation
and testing, respectively (so 3,328 samples used in total). All the new splits are
subsets of the original splits.

Here are a few examples from the training split:

```json
{
  "text": "Ett överraskande ras på den ryska lastbilsmarknaden har gjort att Scania blivit frånsprunget av konkurrenten Volvo som ökat sina leveranser, skriver Dagens Industri. Bakom Scanias tapp på 24 procent ligger bland annat problem med tillstånden för att producera Euro-3 lastbilar i fabriken i S:t Petersburg. Men det räknar Scanias Rysslandschef Hans Tardell med att ta tillbaka under året. Konkurrenten Volvo, som ökat leveranserna med 40 procent och orderingången med 68 procent jämfört mot första kvartalet 2011, hoppas kunna växa ytterligare.  ",
  "target_text": "Ett överraskande ras på den ryska lastbilsmarknaden har gjort att Scania blivit frånsprunget av konkurrenten Volvo som ökat sina leveranser, skriver Dagens Industri."
}
```
```json
{
  "text": "Scenen som beskrivs i åtalet kunde vara hämtad ur en skräckfilm. Den då tolvåriga flickan har berättat hur hon försågs med handbojor och kedjades vid en krok i taket. Enligt åtalet ska hon även ha fått ett koppel kring halsen och piskats. Åklagaren menar att det handlar om ett utdraget förlopp. – En tolvårig flicka ska inte sitta fastsatt i en krok i taket, säger åklagare Daniel Veivo Pettersson, som nu har åtalat en 25-årig man för grov våldtäkt mot barn. I veckan berättade TT att sju män dömts för att vid olika tillfällen ha utsatt samma flicka för sexuella övergrepp. Männen fick kontakt med flickan via forum på nätet och tjatade sig till träffar med henne. En av männen band och våldtog henne i en skog. 25-åringen blir nu den åttonde mannen som åtalas för övergrepp. – Man häpnar när man hör hennes berättelse. Hon är mycket trovärdig och vi har även kunnat styrka åtalen mot männen genom teknisk bevisning som chattkonversationer och i något fall fanns dna på en kondom och på en bh, säger Daniel Veivo Pettersson. Vid en husrannsakan i 25-åringens hem i Stockholm, där våldtäkten ska ha begåtts under hösten 2013, hittades kedjor, handbojor, koppel och en piska. Enligt flickan hade delar av övergreppen filmats. Polisen misstänkte att filmerna kunde ha sparats i en så kallad molntjänst, och åklagaren fick ta hjälp av Microsoft i USA. – Det drog ut på tiden, men tyvärr hittade vi inte det vi letade efter. Han har raderat en hel del information i sin dator, säger Daniel Veivo Pettersson. 25-åringen åtalas dessutom för ytterligare en våldtäkt på flickan, eftersom han misstänks ha våldtagit henne på en toalett. Mannen är tidigare dömd för övergrepp på en annan minderårig flicka, och åklagaren har nu begärt honom häktad i sin frånvaro. – Han kan vara hemma, men han kan även vara utomlands. Om han häktas i sin utevaro kommer han att efterlysas, säger Daniel Veivo Pettersson. 25-åringen försvaras av advokat Thomas Bodström. Han vill inte berätta om 25-åringen kommer närvara vid häktningsförhandlingen, men han säger: – Han nekar till samtliga brott, är helt oskyldig och det finns ingen grund för häktning. Enligt åklagaren misstänks flickan ha utsatts av ytterligare minst en man som polisen inte har lyckats identifiera. Männen i härvan 37-åring, Östergötland: Våldtäkt mot barn och barnpornografibrott – fem års fängelse. 26-åring, Dalarna: Sexuellt ofredande – skyddstillsyn. 29-åring, Stockholmstrakten: Våldtäkt mot barn (två tillfällen) – tre års fängelse. 26-åring, Stockholmstrakten: Våldtäkt mot barn – två och ett halvt års fängelse. 27-åring, Stockholmstrakten: Grov våldtäkt mot barn och våldtäkt mot barn (fyra tillfällen) – sju års fängelse. 55-åring, Östergötland: Utnyttjande av barn för sexuell posering (elva tillfällen) och sexuellt ofredande (två tillfällen) – åtta månaders fängelse. 19-åring, Västra Götaland: Våldtäkt mot barn – åtta månaders fängelse (domen är överklagad). 25-åring, Stockholmstrakten: Åtalad för grov våldtäkt mot barn och våldtäkt mot barn. ",
  "target_text": "Den tolvåriga flickan kedjades vid en krok i taket och våldtogs. En 25-årig man har nu åtalats för grov våldtäkt mot barn, men det är oklart var han är. Sju män dömdes nyss för övergrepp på samma flicka."
}
```
```json
{
  "text": "Det är Gröna partiets ledare Jill Stein som har uppmanat valkommissionen i delstaten Wisconsin att räkna om rösterna, det skriver Reuters och Wisconsins valkommission. Valkommissionen skriver att man ”räknar med att omräkningen börjar inom en vecka efter det att Steins kampanj har betalat avgiften omräkningen, som vi fortfarande håller på att beräkna”. En omräkning ska vara genomförd före den 13 december. Delstaten vanns av Donald Trump med 47,9 procent av rösterna mot Hillary Clintons 46,9 procent och gav honom 10 elektorsröster. Skillnaden mellan de två kandidaterna var 23.000 röster. Jill Stein har tidigare sagt att hon är beredd att även försöka få rösterna i Michigan och Pennsylvania omräknade. Om hon ska begära en omräkning också i dessa två delstater måste den begäran inkomma under nästa vecka, skriver NBC News. Jill Stein. Foto: AP För att få till stånd en omräkning måste Gröna partiet ha pengar nog att driva en sådan. Enligt Washington Post har partiet lyckats samla in 4,5 miljoner dollar som ska täcka juridiska omkostnader och annat som har med en eventuell omräkning att göra i de tre delstaterna. Enligt tidningen kommer det sannolikt att behövas sammanlagt mellan 6 och 7 miljoner för att genomföra en omräkning. Om Clinton skulle gå segrande ur en omräkning i Wisconsin skulle detta ändå inte innebära någon skillnad när det gäller utgången av presidentvalet. Skulle Clinton vinna även i Michigan och Pennsylvania skulle det däremot betyda en annan utgång av valet. Även om få tror att en omräkning skulle betyda något i praktiken, Hillary Clinton har redan erkänt sig besegrad, så skulle en omräkning i hennes favör i Wisconsin och Pennsylvania ge henne 30 elektorsröster medan Trump förlorar lika många. Om så, rent hypotetiskt, skulle bli fallet, skiljer bara 10 elektorsröster till Trumps fördel – och då återstår ännu Michigans röster att sluträknas. Skulle Clinton vinna även dem så har hon flest antal elektorsröster. Jill Stein har i en intervju själv sagt att hon inte begär en omräkning för att gynna någon av kandidaterna utan för att ”amerikanerna inte blev särskilt glada över utgången av valet”. Sett till enbart rösterna, och inte till elektorerna, leder just nu Hillary Clinton med 48,1 procent av rösterna mot Donald Trumps 46,6 procent. I antal röster leder Clinton med 2.012.331 röster. ",
  "target_text": "Valkommissionen i Wisconsin i har fått en uppmaning om att rösterna i presidentvalet ska räknas om. Wisconsin har nu börjat förbereda en omräkning. Och det kan bli fler."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:
  ```
  Nedan följer artiklar med tillhörande sammanfattningar.
  ```
- Base prompt template:
  ```
  Artikel: {text}
  Sammanfattning: {target_text}
  ```
- Instruction-tuned prompt template:
  ```
  Artikel: {text}

  Skriv en sammanfattning av artikeln ovan.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset swedn
```


### Unofficial: Schibsted-sv

This dataset was published
[here](https://huggingface.co/datasets/Schibsted/schibsted-article-summaries) and
features summaries of news articles from Schibsted Medias Swedish newsroom, from
Aftonbladet.

The original dataset has 528 / 96 / 89 samples for training, validation and testing,
respectively. We use these splits as-is.

Here are a few examples from the training split:

```json
{
  "text": "Richard Jomshof blir upprörd och vägrar svara på frågor: SD-toppen Richard Jomshof vägrar kommentera kritiken efter påhoppet på Daniel Riazat (V).  När Aftonbladet möter honom i riksdagen blir han upprörd och går iväg. – Jag uppskattar inte skjutjärnsjournalistik, det är ett oseriöst sätt att jobba, säger han.  Justitieutskottets ordförande Richard Jomshof (SD) får hård kritik för sitt uttalande att V-ledamoten Daniel Riazat borde flytta från Sverige.  Flera i den politiska oppositionen dömer ut det som rasistiskt. Även i Tidöpartierna hörs protester.  ”Är man svensk medborgare så är man. Skamligt var ordet!” skriver L-politikern Jan Jönsson i ett uttalande på X.  ”Ta det med pressavdelningen” Aftonbladet var på plats utanför justitieutskottets möte i riksdagen vid lunchtid på tisdagen. Jomshof anlände först av alla ledamöter, tio minuter innan mötet inleddes, men ville inte svara på frågor.  – Du får ta det med pressavdelningen. Varför vill du inte svara, det är ju du som har skrivit de här tweetsen? – Du får ta det med pressavdelningen. Du kan läsa min senaste tweet förresten, så kan vi utgå från den. Varför tycker du att han borde lämna Sverige? – Börja med att läsa min tweet, det framgår väldigt tydligt där. ”Uppskattar inte skjutjärnsjournalistik” Inlägget som Jomshof syftar på lades upp kort innan justitieutskottets möte. Jomshof går där till nytt angrepp mot Riazat. Han anklagar honom för att ha ett ”sunkigt” beteende, att vara oförskämd och komma med aggressiva påhopp på politiska motståndare.  Mötet med justitieutskottet varade en timme, när Richard Jomshof kom ut från salen var upprörd över Aftonbladets närvaro. Detta trots att media brukar bevaka mötena och att ledamöterna i utskottet ofta tar tillfälle att ge intervjuer efteråt.  – För det första, vill ni prata med mig så går ni till pressavdelningen. Jag uppskattar inte skjutjärnsjournalistik, det är ett oseriöst sätt att jobba. Två, jag har inget mer att tillägga än det jag lagt ut på plattformen X. Där framgår det tydligt vad det här handlar om. Tre, ett tips i all vänlighet, ni kan ju prata med Riazat själv, om hans oförskämdheter och aggressiva beteende, om varför han inte vill ta politiska motståndare och kvinnor i hand. Nu tänker jag gå och äta lunch, säger Jomshof.  Busch: Jag är ganska osugen Daniel Riazat kallade igår Richard Jomshofs uttalande för rasistiskt och uppmanar statsminister Ulf Kristersson (M) att ta avstånd. Aftonbladet har sökt Kristersson, hans pressekreterare ber att få återkomma om statsministern har möjlighet att uttala sig. Vice statsminister Ebba Busch (KD) var fåordig när hon fick frågor om det på tisdagen.  – Jag är ganska osugen på att bidra till det rubrikspelet, sa hon i samband med en utfrågning i riksdagen.  Vice ordförande i justitieutskottet, Ardalan Shekarabi (S), har tidigare krävt Jomshofs avgång. Han uppmanar företrädare för regeringen att sluta ge Jomshof stöd.  – Tyvärr är det ett konsekvent beteende han har. Han verkar för splittring, motsättningar och i vissa fall hat mot folkgrupper. Han använder den plattform som ordförande i justitieutskottet medför till att bedriva den typen av agitation, säger han.  Aftonbladet har sökt Sverigedemokraternas pressavdelning. De ber om att få frågorna till Richard Jomshof på mejl och att få återkomma senare. Aftonbladet har sökt Daniel Riazat. Vänsterpartiets pressavdelning ber att få återkomma. ",
  "target_text": "SD-toppen Richard Jomshof vägrar kommentera kritiken för sitt påstående att Vänsterpartiets riksdagsledamot Daniel Riazat borde lämna Sverige. Många inom den politiska oppositionen kallar uttalandet rasistiskt När Jomshof konfronteras med frågor från Aftonbladet vid ett utskottsmöte i riksdagen, blir han upprörd och går iväg utan att svara på frågorna. Han hänvisar till SD:s pressavdelning."
}
```
```json
{
  "text": "Fredrik Bolanders uttalande i ”Robinson” får kritik: ”Skriver att jag är en mansgris”: Kvinnor är bra på att städa, laga mat och hålla ordning.  Killar vill äta mat, är starkare och bättre. Fredrik Bolanders uttalande i ”Robinson” har fått många att reagera. – Jag vet att folk stör sig på sådana uttalanden, det är ju ett sådan samhälle vi lever vi, säger han. – Om jag hade fått bestämma hade det varit en kvinna i laget för de är ju bra på att laga mat, de är bra på att hålla ordning och städa. Där har vi det negativa med att inte ha en kvinna i laget. Vi män vill ju äta såklart. Uttalandet från ”Robinson”-deltagaren Fredrik Bolander, 40, har fått många att reagera, bland annat på ”Robinsons” sociala medier.  Ändringen i ”Robinson” 2024 I årets säsong delas kvinnor och män upp i olika lag.  När programledaren Anders Lundin, 65, frågar Bolander om han tror att det ger kvinnorna en större chans att vinna i år får han ett snabbt svar.  – Nej, det blir en kille som vinner i år. Killar är ofta lite starkare och bättre än tjejer. Flera deltagare reagerar på uttalandet i programmet. Tjejerna protesterar högljutt och Gustav Jacobson, 27, gör en förskräckt min.  Bolander säger även i programmet att han inte går så bra ihop med kvinnor och feminister. – Jag är väldigt manlig i mig själv, och jag har en väldigt manlig jargong, och tycker att det ska vara jämlikt men man ska också förstå vem som är mannen i huset. ”Skriver att jag är en mansgris” När Aftonbladet pratar med Bolander samma dag som ”Robinson” har premiär berättar han att han redan fått reaktioner och meddelanden från tittare.  – De skriver att jag är en mansgris och att jag har fel kvinnosyn. Samtidigt är han medveten om att det han säger om kvinnor triggar folk.  – Jag älskar att provocera. Det är klart att jag gillar att se reaktioner, det vill jag ju, säger Bolander.  Han fortsätter:  – Jag vet att folk stör sig på sådana uttalanden, det är ju ett sådan samhälle vi lever vi. Så det var roligt att köra lite tvärtom tänkte jag. Fredrik Bolander om reaktionerna Just uttalandet om att det behövs en kvinna för att städa och laga mat i killarnas lag är det han fått mest reaktioner på.  – Många som skrivit är ju inte jätteglada. Vad skriver folk? – Att vi lever i 2024 och man ska inte vara så och alla ska vara lika och allt det där. Men samtidigt så, man gör ju det man är bra på? Men män kan väl också vara bra på att laga mat och städa? – Jo men vi har ju mycket annat att göra? Som att träna med stenar? – Exakt. Pumpa muskler och träna, vi måste tänka på hur vi ser ut, vi måste se solbrända ut och det tar tid. Det här är ju ett uttalande som upprör många. Känner du att du kan stå för det uttalandet? – Det där är en svår fråga. Jag säger så här; man får se lite under programmets gång om det är något jag står för eller inte. Så kan jag säga. Många undrar också om du är seriös eller skojar? – Det är det som är frågan, skojar jag eller är jag seriös? Det svarar jag inte på. Varför inte? – Antingen kanske jag står för det senare eller så gör jag inte det. Det får ni se. ”Robinson” sänds söndagar klockan 21.00 samt måndag till torsdag klockan 19.30 på TV4 och på TV4 play. ",
  "target_text": "\"Robinson\"-deltagaren Fredrik Bolander har hamnat i blåsväder efter sina uttalanden om kvinnor och män, och får kritik på sociala medier. Han påstår att kvinnor är bra på att laga mat och städning medan män är starkare och bättre, och detta upprörde andra deltagare och tittare. Bolander säger att han älskar att provocera, men vägrar svara på frågan om han skämtar eller är seriös."
}
```
```json
{
  "text": "Polisen om den övergivna diplomatbilen: ”Vi undersöker immunitetsfrågan”: En diplomatbil lämnades övergiven på ett tågspår i centrala Stockholm i helgen. Fordonet tillhör Etiopiens ambassad som har bett om ursäkt för vansinnesfärden. Men när Aftonbladet knackar på är de fåordiga.  – Vi återkommer så fort det går, säger en anställd på ambassaden. Det var natten till söndag som minibussen krockade på tvärbanans spår vid Alviks strand i Stockholm. ”Vår ambassad ber om ursäkt för olyckan och besvären den orsakat. Vi har startat en internutredning för att ta reda på hur olyckan ska ha skett”, skriver Etiopiens ambassad i Stockholm i ett mail till Aftonbladet. I övrigt har de inte kommenterat händelsen och när Aftonbladet knackar på hos ambassaden är svaret kort. – Vi håller på att jobba med det. Vi återkommer så fort det går, säger en anställd på ambassaden. Men när vill de inte svara på. 17 300 kronor i obetalda böter Tågtrafiken var tillfälligt avstängd under söndagsmorgonen och bilen fick bärgas med hjälp av en spårtraktor. Den har troligtvis kört upp på spåret vid Gröndal, enligt SL. Där kör bilar och spårvagnar på gatan innan rälsen viker av på en egen banvall. – Därefter ska den i så fall ha kört två kilometer på kross och makadam innan den krockat med en stolpe, säger Claes Keisu, pressansvarig på SL. Minibussen har också obetalda böter på 17 300 kronor, enligt Transportstyrelsen.  ”Har skett en gång tidigare” Den här typen av felkörning sker cirka tio gånger om året. Under februari skedde det två gånger, just vid Gröndal. Vanligtvis upptäcks misstaget tidigt och då brukar föraren kunna backa tillbaka på vägen. – Det här fordonet har lite högre markfrigång så det kan förklara att den kunnat ta sig längre, säger Claes Keisu. Men att bilen lyckats ta sig så långt är väldigt ovanligt. – Vad vi vet har det bara skett en gång tidigare. 2012 var det en Ålänning med sin familj som kom upp på banan i Hammarby sjöstad och körde hela vägen till Gullmarsplan, säger Keisu. Föraren ska då ha kört uppemot en kilometer på spåret. ”Vi undersöker immunitetsfrågan” Polisen har inlett en förundersökning om vårdslöshet i trafik. Det är fortfarande oklart om någon kan åtalas.  – Vi undersöker immunitetsfrågan, säger Nadya Norton, presstalesperson vid Stockholmspolisen. ”Utredningen får visa om personen som körde bilen hade immunitet eller inte. Om en person har immunitet kan denne inte lagföras i Sverige”, skriver förundersökningsledaren, Timmy Malmgren, i ett mail till Aftonbladet. Diplomater får inte straffas i landet de arbetar i, enligt internationella överrenskommelser. – Jag har inga uppgifter om någon är misstänkt i ärendet, säger Nadya Norton. Hade fest under kvällen Kvällen innan bilen hittades på tågspåret ska Ambassaden anordnat en fest i sina lokaler. ”Vi på Ambassaden för Demokratiska förbundsrepubliken Etiopien på våning 3 kommer att ha ett event på lördag den 2. Observera att vi kommer ha gäster. Vi hoppas att vi inte stör er, kära grannar. Tack för er förståelse”, skriver de på en lapp som sitter i fastighetens hiss.",
  "target_text": "En bil från Etiopiens ambassad lämnades övergiven på ett tågspår i centrala Stockholm under helgen, vilket ledde till tillfälligt avstängd tågtrafik. Ambassaden har bett om ursäkt och påbörjat en intern utredning för att ta reda på händelseförloppet. En polisutredning är igång för vårdslöshet i trafik, men det är oklart om någon kan åtalas på grund av diplomatisk immunitet."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:
  ```
  Nedan följer artiklar med tillhörande sammanfattningar.
  ```
- Base prompt template:
  ```
  Artikel: {text}
  Sammanfattning: {target_text}
  ```
- Instruction-tuned prompt template:
  ```
  Artikel: {text}

  Skriv en sammanfattning av artikeln ovan.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset schibsted-sv
```
