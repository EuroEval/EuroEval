# 🇪🇪 Estonian

This is an overview of all the datasets used in the Estonian part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.


## Sentiment Classification

### Estonian Valence Corpus

This dataset was published in [this paper](http://dx.doi.org/10.7592/FEJF2016.64.polarity). The dataset was compiled of articles of different rubrics of online
dailies, weeklies, and reader comments, while the polarity of each paragraph was determined by native Estonian readers.

There are 4 labels in the original dataset instead of the usual 3.
Examples with the labels representing 'mixed' emotion (vastuoluline) were filtered out
mainly to be consistent with rest of the languages in EuroEval.

The original full dataset consists of 3,277 / 818 samples for the training and test splits,
respectively. Having filtered out 'mixed' examples, we truncate the train split to 1,024
examples, and redistribute the rest to validation and test resulting in the final size of
1,024 / 256 / 2,048 for the training, validation and test splits, respectively.


Here are a few examples from the training split:

```json
{
  "text": "Sügisest algav pikk koolitee Oskari perekonda ei hirmuta.",
  "label": "positiivne"
}
```
```json
{
  "text": "Sellises eas, nagu teie olete, tundub muidugi ka 20-aastane üsna laps ...",
  "label": "neutraalne"
}
```
```json
{
  "text": "ka ainus märkimisväärne saavutus temalt selle loo esituse juures.",
  "label": "negatiivne"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Järgmised on dokumendid ja nende meelestatus, mis võib olla 'positiivne', 'neutraalne' või 'negatiivne'.
  ```
- Base prompt template:
  ```
  Dokument: {text}
  Meelestatus: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Dokument: {text}

  Klassifitseeri dokument meelestatuse järgi. Võimalikud vastused: 'positiivne', 'neutraalne' või 'negatiivne'. Muud vastused ei ole lubatud.
  ```
- Label mapping:
    - `positive` ➡️ `positiivne`
    - `neutral` ➡️ `neutraalne`
    - `negative` ➡️ `negatiivne`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset estonian-valence
```

## Common-sense Reasoning

### WinoGrande-ET

The dataset includes the [WinoGrande](https://doi.org/10.48550/arXiv.1907.10641) test set translated and culturally adapted by hand by a professional translator (citation TBA).
The structure of the dataset is identical to the original. Since train and dev splits were not translated manually, we employ
the GPT-4o model to translate the expected number of examples starting from the beginning of the respective splits.
The final dataset size is 1,024 / 256 / 1,767 for the training, validation and test splits, respectively.

Here are a few examples from the training split (note that unlike the test split these are machine translated):

```json
{
  "text": "Ian vabatahtlikult sõi Dennise menudo pärast seda, kui oli juba kausi söönud, sest _ põlgas soolte söömist.\nVastusevariandid:\na. Ian\nb. Dennis",
  "label": "b"
}
```
```json
{
  "text": "Ian vabatahtlikult sõi Dennise menudo pärast seda, kui oli juba kausitäie söönud, sest _ nautis soolte söömist.\nVastusevariandid:\na. Ian\nb. Dennis", "label": "a"
}
```
```json
{
  "text": "Ta ei tule kunagi minu koju, aga mina lähen alati tema majja, sest _ on väiksem.\nVastusevariandid:\na. kodu\nb. maja",
  "label": "a"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Sulle esitatakse lüngaga (_) tekstülesanne ja kaks vastusevarianti (a ja b).
  ```
- Base prompt template:
  ```
  Tekstülesanne: {text}
  Vastusevariandid:
  a. {option_a}
  b. {option_b}
  Vastus: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Tekstülesanne: {text}
  Vastusevariandid:
  a. {option_a}
  b. {option_b}

  Sinu ülesanne on valida lünka sobiv vastusevariant. Vasta ainult 'a' või 'b'. Muud vastused ei ole lubatud.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset winogrande-et
```

## Named Entity Recognition

### EstNER

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.76/).
The dataset is a manually annotated collection of Estonian news and
social media texts.

The original dataset contains 16,966 / 3,297 / 2,797 samples for the training,
validation and test splits, respectively. We use 1,024 / 256 / 2,048 samples for our
training, validation and test splits, respectively. All the new splits are subsets of
the original splits.

Here are a few examples from the training split:

```json
{
  "tokens": ["Katse", "lõpuni", "jätkas", "34aastane", "tiitlijahtija", "kolmel", "rattal", "."],
  "labels": ["O", "O", "O", "O", "O", "O", "O", "O"]
}
```
```json
{
  "tokens": ["“", "Kui", "tegemist", "oleks", "olnud", "piletiga", "peoga", ",", "peaksime", "inimestele", "raha", "tagasi", "maksma", ",", "nüüd", "saame", "üksnes", "külalistelt", "vabandust", "paluda", ",”ütles", "Järvamaa", "omavalitsuste", "liidu", "tegevdirektor", "Krista", "Nurm", "ajalehele", "Järvamaa", "Teataja", "."],
  "labels": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-ORG", "I-ORG", "I-ORG", "B-MISC", "B-PER", "I-PER", "O", "B-MISC", "I-MISC", "O"]
}
```
```json
{
  "tokens": ["Makke", "lõpetas", "Quincy", "keskkooli", "Illinoisi", "osariigis", "ja", "plaanib", "sportlasstipendiumi", "abil", "edasi", "õppida", "Lääne-Illinoisi", "ülikoolis", ",", "mille", "korvpallimeeskond", "kuulub", "USA", "üliõpilasliiga", "NCAA", "esimesse", "divisjoni", "."],
  "labels": ["B-PER", "O", "B-ORG", "I-ORG", "B-MISC", "I-MISC", "O", "O", "O", "O", "O", "O", "B-ORG", "I-ORG", "O", "O", "O", "O", "B-ORG", "I-ORG", "B-ORG", "O", "O", "O"]
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:
  ```
  Allpool on laused ja JSON-sõnastikud, mis sisaldavad antud lauses esinevaid nimetatud üksuseid.
  ```
- Base prompt template:
  ```
  Lause: {text}
  Nimetatud üksused: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Lause: {text}

  Tuvasta lauses nimetatud üksused. Väljund peaks olema JSON-sõnastik, mille võtmed on 'inimene', 'asukoht', 'organisatsioon' ja 'muu'.
  Väärtused peaksid olema kindlat tüüpi nimetatud üksuste loendid, täpselt nii nagu need lauses esinevad.
  ```
- Label mapping:
    - `B-PER` ➡️ `inimene`
    - `I-PER` ➡️ `inimene`
    - `B-LOC` ➡️ `asukoht`
    - `I-LOC` ➡️ `asukoht`
    - `B-ORG` ➡️ `organisatsioon`
    - `I-ORG` ➡️ `organisatsioon`
    - `B-MISC` ➡️ `muu`
    - `I-MISC` ➡️ `muu`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset estner
```

## Summarization

### ERRNews

The dataset was released in [this paper](https://doi.org/10.22364/bjmc.2022.10.3.23).

The dataset consists of news story transcripts of ERR News broadcasts scraped from from the
[ERR Archive](https://arhiiv.err.ee/err-audioarhiiv) News generated by an ASR pipeline paired with the human written
summary from the archive.

The original full dataset consists of 10,420 / 523 / 523 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 2,048 split for training,
validation and testing, respectively. The test split is extended with additional examples
from the test split.

```json
{
  "text": "Ta mainis seda, et et ta on mures, et Kreml on oma propagandakampaaniat ka Eesti suhtes tugevdanud. Aga ma ise ütlesin talle, et kui ta peab silmas seda postitust, siis tegemist on Jaak Madissoni kolme aasta taguse taguse mõtteavaldusega, nagu ta oli 20 aastane ja ei olnud veel meie erakonna liige ja, ja Jaak Madison ei ole tegelikult selle teemaga hiljem minu teada tegelenud. Jaak Madisson teemal olen rääkinud, palusin tal selle postituse kõrvaldada ja ütles ka, et nojah, et aga seal ei ole ju midagi, ma lihtsalt arutlesin sel teemal ja seal ei ole midagi, seal ei ole midagi püüda siia külge panna mingisugust silti, nagu me oleksime mingisugune Hitler, juugend. No andke andeks ja, ja mina ütlen teile, kui te seda teemat tahate üleval hoida, laske käia, me ei saa teid takistada, aga te teete Kremli tööd, tehke oma järeldused, te teete Kremli tööd, see on just see, mida Krem tahab. Et kuskilt otsitakse üles mingi vana postitus, kuskil puhutakse sellele tuul, purjedesse süüdistatakse värskelt riigikogu liikmeks saanud noort meest kolme aasta taguses naiivses poisikese sõnavõtus. Ma ei saa teid takistada selles, aga ma ütlen teile, et see ei ole õige, mida te, Jaak Madisson, sel ajal ei olnud erakonna liige, kolm aastat tagasi, kui te otsite välja kolme aasta taguse postituse, millest mul ei olnud kuni tänase päevani õrna aimugi siis ei saa see olla erakonna seisukoht. Tegemist on Jaak Madissoni kolme aasta taguse taguse mõtteavaldusega kui ta oli 20 aastane. Ja ei olnud veel meie erakonna liige.",
  "target_text": "Eesti Konservatiivse Rahvaerakonna esimehe Mart Helme sõnul on riigikokku valitud erakonnakaaslase Jaak Madisoni sotsiaalmeedias levinud natsi-Saksamaad kiitev blogitekst üksikisiku arvamus ning lihtsalt hinnang Saksamaale ja tolle majandusele."
}
```
```json
{
  "text": "Eelkõige uute toodete ja teenuste disainimisel me püüame lähtuda või soovitav oleks disaineritel lähtuda eelkõige vajadusest ja mitte ainult lõbu pärast maailma kuhja ta võib-olla mittevajalike asjadega. Te annate välja ka need tootedisaini auhinnad nimega Bruno, kas ütleme siis auhinnatakse ka rohkem neid inimesi, kes siin rohkem sellist vajaduspõhist disaini hindavad? Disaini auhind, tänase pruun antakse välja iga kahe aasta tagant ja, ja tegelikult tõenäoliselt on enamus tooteid hakatud juba disainima kaks aastat tagasi tootearendusprotsess on päris pikk, aga kuna meil on rahvusvahelisel žüriil ette kirjutatud kriteeriumid, mille järgi nad hindavad, siis ma võin öelda, et seal edetabeli lõpus alles on esteetika ga esimesed mitu mitu kategooriat on pühendatud eelkõige just ka kasutaja mugavusele kasutaja vajadusele keskkonnasõbralikkusele materjalide taaskasutusele, nii et, et hästi palju kategooriaid on tulnud nagu tavapärase disaini hindamisele juurde. Kui vanasti hinnati võibolla ainult vormi ja emotsiooni ja ilu, siis nüüd on disainerite ees palju suuremad nõudmised. See festival kestab päris mitu päeva, et oskate te kohe anda mõne soovituse ka, et mida inimesed saaksid vaatama-kuulama tulla. Jah, no arvestades praeguseid niisuguseid olusid, et väga palju inimesi ei kuhjuks ühel ajal, siis me oleme praegu õnnega koos, et meie üritus kõigepealt toimub Põjala tehases, kus mahub isegi siis, kui inimesed peaksid seisma kaks pluss kaks, mida nad ilmselt peaksidki tegema, mahub 500 inimest ja kuna näitused on lahti kella 12. kaheksani iga päev, siis võib hajutatult käia neid vaatamas terve nädala jooksul. Aga kes tahab nagu pidulikumalt osaleda näituste üldisel avamisel, kus saab näha ka, et ausi, kas teksadega pääseb paradiisi, on näituse nimi, aga ma etenduse nimi on Totali outo fassion. Et siis võib tulla esmaspäeval juba kella viieks kohale ja kuni peaaegu kella kaheksani toimuvad siis kõikide näituste avamised ja, ja siis ka see maaetendus, mida ma nimetasin",
  "target_text": "Täna algab Eesti Disainiöö festival, mis sel aastal toimub juba 15. korda."
}
```
```json
{
  "text": "Mihhail Korbilt saadikupuutumatuse äravõtmine oli vajalik selleks, et tema suhtes saaks jätkata kriminaalmenetlust. Riigiprokuratuur esitas õiguskantslerile sellekohase taotluse kuu aega tagasi. Peaprokuröri taotlusest nähtub, et riigikogu liikmele Mihhail Korbile on esitatud kahtlustus karistusseadustiku paragrahv 298 prim lõikes üks sätestatud kuriteo toimepanemises. See on mõjuvõimuga kauplemine. Lühidalt väidab peaprokurör, et Mihhail Korb lubas Hillar Tederile üheksandal veebruaril 2020 kasutada oma mõjuvõimu Tallinna linnapea Mihhail Kõlvarti üle. Selleks, et kolmas isik, porto Franco osaühing, saaks ametiisikult Mihhail Kõlvartilt avaliku huvi seisukohalt põhjendamatu eelise. See siis puudutas servituudi hinda. Ja küsimus sellest, et kas porto Franco peaks väljasõidud rajama enda maale või peaks saama rajada need linnamaale ja madalama hinna eest, kui algselt välja pakuti. Ning Madiselisas. Olles kriminaaltoimiku materjalidega ja ka jälituslubadega tutvunud kinnitan teile, et märki sellest, et tegemist oleks ilmselgelt põhjendamatu menetlusega või poliitiliselt kallutatud menetlusega. Meie ei leidnud. Mihhail Korb tuletas oma sõnavõtus meelde, et kahtlustuse esitamisest on möödas 15 kuud ja tegelikult oleks asi juba ammu võinud kohtus olla, sest tema on huvitatud maksimaalselt kiirest tõe väljaselgitamisest. Täna ma pöördun kogu saali poole, samuti toetada minul saadikute puutumatuse äravõtmise. Seda kõik selleks, et maksimaalselt kiiresti teieni jõuda. Saadikupuutumatuse äravõtmise poolt hääletas 82 saadikut, vastu ei olnud keegi, mõned jätsid hääletamata.",
  "target_text": "Õiguskantsler Ülle Madise tegi Riigikogule ettepaneku anda nõusolek Riigikogu liikmelt Mihhail Korbilt saadikupuutumatuse äravõtmiseks."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:
  ```
  Allpool on dokumendid koos kokkuvõtetega.
  ```
- Base prompt template:
  ```
  Dokument: {text}
  Kokkuvõte: {target_text}
  ```
- Instruction-tuned prompt template:
  ```
  Document: {text}

  Koosta ülaltoodud dokumendi kokkuvõte.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset err-news
```
