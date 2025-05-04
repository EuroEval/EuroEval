# 🇫🇮 Finnish

This is an overview of all the datasets used in the Finnish part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### ScandiSent-fi

This dataset consists of reviews from Trustpilot and was published [here](https://aclanthology.org/2021.nodalida-main.42/). It is a binary sentiment classification dataset, with labels "positive" and "negative".

For the Finnish part of the dataset, there are 10,000 training samples. From these samples, we have created a 1,024 / 256 / 2,048 split for the train, validation and test splits, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Kaikki meni niinkuin piti. Nopea toimitus.",
  "label": "positive"
}
```
```json
{
  "text": "En pidä tästä, kun ei löydy linkkiä mistä pääsis heti maksamaan. En todellakaan pidä siitä, että joka tieto pitää kopioida erikseen. Haluaisin päästä suoraan oston jälkeen maksamaa mobiilipankkiin. Pari laskua on jäänyt tän takia kokonaan huomioimatta. Ja ihan turhaa.... ärsyttää sitten se kotiin tuleva muistutuslasku.",
  "label": "negative"
}
```
```json
{
  "text": "Todella hidas toimitus, ja virheellistä tietoa tuotteiden saatavuudesta, paketti ja tuotteet perillä vasta kuukauden päästä tilauksesta....",
  "label": "negative"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Seuraavassa on arvosteluja ja niiden tunnesävy, joka voi olla 'positiivinen' tai 'negatiivinen'.
  ```
- Base prompt template:
  ```
  Teksti: {text}
  Tunnesävy: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Teksti: {text}

  Luokittele arvostelun tunnesävy. Vastaa vain 'positiivinen' tai 'negatiivinen', ei muuta.
  ```
- Label mapping:
    - `positive` ➡️ `positiivinen`
    - `negative` ➡️ `negatiivinen`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset scandisent-fi
```


## Named Entity Recognition

### Turku-NER-fi

This dataset was published in [this paper](https://aclanthology.org/2020.lrec-1.567/). The dataset is a manually annotated corpus built on the Universal Dependencies Finnish corpus. The corpus was created by the Turku NLP group.

The original dataset contains 12,217 / 1,364 / 1,555 samples for the training, validation and test splits, respectively. We use 1,024 / 256 / 2,048 samples for our training, validation and test splits, respectively. All the new splits are subsets of the original splits.

Here are a few examples from the training split:

```json
{
  "tokens": ["Suomalaiset", "vaihtoivat", "Tukholman", "Tallinnaan"],
  "labels": ["O", "O", "B-LOC", "B-LOC"]
}
```
```json
{
  "tokens": array(['Liuhto', 'nosti', 'Kreikan', 'tapauksen', 'yhteydessä', 'esille', 'kysymyksen', 'siitä', ',', 'miten', 'Euroopan', 'unionissa', 'yleisesti', 'sanktioidaan', 'pelisääntöjen', 'rikkomisesta', '.'], dtype=object),
  "labels": array(['B-PER', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
```
```json
{
  "tokens": array(['Mithridates', 'oli', 'Pontoksen', 'merkittävin', 'kuningas', 'ja', 'Rooman', 'valtakunnan', 'vaarallisin', 'vihollinen', 'ensimmäisellä', 'vuosisadalla', 'eaa.', '.'], dtype=object),
  "labels": array(['B-PER', 'O', 'B-LOC', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:
  ```
  Seuraavassa on lauseita ja JSON-sanakirjoja, jotka sisältävät annetussa lauseessa esiintyvät nimetyt entiteetit.
  ```
- Base prompt template:
  ```
  Lause: {text}
  Nimetyt entiteetit: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Lause: {text}

  Tunnista lauseessa esiintyvät nimetyt entiteetit. Tulosta ne JSON-sanakirjana, jonka avaimet ovat 'henkilö', 'paikka', 'organisaatio' ja 'muut'. Arvojen tulee olla listoja kyseisen tyypin nimetyistä entiteeteistä täsmälleen siinä muodossa kuin ne esiintyvät lauseessa.
  ```
- Label mapping:
    - `B-PER` ➡️ `person`
    - `I-PER` ➡️ `person`
    - `B-LOC` ➡️ `sted`
    - `I-LOC` ➡️ `sted`
    - `B-ORG` ➡️ `organisation`
    - `I-ORG` ➡️ `organisation`
    - `B-MISC` ➡️ `diverse`
    - `I-MISC` ➡️ `diverse`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset turku-ner-fi
```


## Linguistic Acceptability

### ScaLA-fi

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Finnish Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Finnish-TDT/tree/master) by
assuming that the documents in the treebank are correct, and corrupting the samples to
create grammatically incorrect samples. The corruptions were done by either removing a
word from a sentence, or by swapping two neighbouring words in a sentence. To ensure
that this does indeed break the grammaticality of the sentence, a set of rules were used
on the part-of-speech tags of the words in the sentence.

The original dataset consists of 15,136 samples, from which we use 1,024 / 256 / 2,048 samples for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
used as-is in the framework.

Here are a few examples from the training split:

```json
{
  "text": "Vuotta aiempaan verrattuna uusia ajoneuvoja rekisteröitiin 17,6 prosenttia enemmän.",
  "label": "correct"
}
```
```json
{
  "text": "20-vuotias sai aiemmin marraskuussa 2006 Helsingin käräjäoikeudelta 30 päiväsakkoa Ta... varastettujen vaatteiden hallussapidosta.",
  "label": "correct"
}
```
```json
{
  "text": "Kun käyttäjä kirjoittaa viestin, se näkyy käyttäjän käyttäjälistassa.",
  "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Seuraavat ovat lauseita ja ovatko ne kieliopillisesti oikein.
  ```
- Base prompt template:
  ```
  Lause: {text}
  Kieliopillisesti oikein: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Lause: {text}

  Määritä onko lause kieliopillisesti oikein vai ei. Vastaa 'kyllä', jos lause on oikein, ja 'ei', jos se ei ole.
  ```
- Label mapping:
    - `correct` ➡️ `kyllä`
    - `incorrect` ➡️ `ei`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset scala-fi
```


## Reading Comprehension

### TydiQA-fi
This question-answering dataset was published in [this paper](https://aclanthology.org/2020.tacl-1.30/). TydiQA is a multilingual dataset covering 11 typologically diverse languages with 204K question-answer pairs collected from native speakers genuinely seeking information. It was designed to evaluate models across languages with varied linguistic features and contains questions written directly in each language without translation.

The original Finnish TydiQA dataset contains 6,855 training and 782 validation samples (we use the [secondary task subset](https://huggingface.co/datasets/google-research-datasets/tydiqa/viewer/secondary_task?views%5B%5D=secondary_task_train)).  We created a 1,024 / 256 / 2,024 split, where the samples from the train and validation split are sampled from the original train and validation splits, respectively. The test set consists of the remaining samples from the original validation split + additional samples from the original train split.

Here are a few examples from the training split:

```json
{
  "question": "Kuka näytteli Dumbledorea Harry Potter elokuvissa?",
  "context": "Dumbledorea esittää kirjasarjasta tehdyssä elokuvasarjassa Richard Harris kahdessa ensimmäisessä elokuvassa. Harrisin kuoltua Michael Gambon esitti hahmoa sarjan lopuissa elokuvissa.",
  "answers": {
    "text": ["Richard Harris kahdessa ensimmäisessä elokuvassa. Harrisin kuoltua Michael Gambon"],
    "answer_start": [59]
  }
}

```json
{
  "question": "Milloin Cristiano Ronaldo liittyi Juventukseen?",
  "context": "Ronaldo siirtyi heinäkuussa 2018 Juventukseen 105 miljoonalla eurolla. Sopimus on nelivuotinen, ja sen aikana hän tienaa verojen jälkeen noin 120 miljoonaa euroa.[133]",
  "answers": {
    "text": ["heinäkuussa 2018"],
    "answer_start": [16]
  }
}
```
```json
{
  "question": "Kuka hallitsi Mithridates VI jälkeen?",
  "context": "Mithridates laajensi valtakuntaansa ympäri Mustanmeren rantoja, ja hän ajautui kolmesti sotaan Rooman valtakuntaa vastaan. Ensimmäisessä sodassa (89 eaa.–85 eaa.) hän valtasi suuren osan Vähää-Aasiaa ja Rooman valtakunnalle kuuluneet osat, jolloin hänen sanotaan teloittaneen 80000 roomalaista. Mithridates valtasi myös Kreikan, mutta konsuli Sulla kukisti hänen joukkonsa vuonna 85 eaa., ja Mithridateen oli luovuttava valloituksistaan. Toinen sota (83 eaa.–81 eaa.) oli suppeampi laajuudeltaan. Kolmannessa sodassa (73 eaa.–63 eaa.) roomalaiset sotapäälliköt Lucullus ja Pompeius kukistivat Mithridateen perusteellisesti. Mithridates surmasi tai surmautti itsensä jouduttuaan poikansa Farnakes II:n syrjäyttämäksi.",
  "answers": {
    "text": ["Farnakes II"],
    "answer_start": [687]
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:
  ```
  Seuraavassa on tekstejä ja niihin liittyviä kysymyksiä ja vastauksia.
  ```
- Base prompt template:
  ```
  Teksti: {text}
  Kysymys: {question}
  Vastaa enintään 3 sanalla: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Teksti: {text}

  Vastaa seuraavaan kysymykseen yllä olevasta tekstistä enintään 3 sanalla.

  Kysymys: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset tydiqa-fi
```


### Unofficial: BeleBele-fi

This dataset was published in [this paper](https://aclanthology.org/2024.acl-long.44/) and features multiple-choice reading comprehension questions across 122 languages.

The original dataset contains 900 unique multiple-choice reading comprehension passages and questions. From these, we use a 256 / 64 / 580 split for training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Toisin kuin muut kädelliset, isot ihmisapinat eivät enää käytä käsiään liikkumiseen, painon kannattelemiseen tai liikkumiseen puissa itseään heilautellen. Simpanssin käsi ja jalka ovat samankokoisia ja -pituisia, mikä viittaa siihen, että kädelle varataan painoa rystykävelyssä. Ihmisen käsi on lyhyempi kuin jalka, ja sen sormiluut ovat suoremmat. Kahden-kolmen miljoonan vuoden ikäiset käsiluiden fossiilit paljastavat käden erikoistumisessa tämän muutoksen liikkumisesta käyttelyyn.\nKysymys: Mikä seuraavista kuvaa tarkasti simpanssin sormiluita?\nVaihtoehdot:\na. Ne ovat suoremmat kuin ihmisillä\nb. Niiden kädet ja jalat ovat erikokoisia\nc. Niitä käytetään painon kannattelemiseen\nd. Niitä käytetään pääasiassa käyttelyyn",
  "label": "c"
}
```
```json
{
  "text": "Panaman paperit on yläkäsite panamalaisen lakiyrityksen Mossack Fonsecan noin kymmenelle miljoonalle asiakirjalle, jotka vuodettiin lehdistölle keväällä 2016. Asiakirjoista selvisi, että neljätoista pankkia auttoi varakkaita asiakkaita piilottamaan miljardeja USA:n dollareita verojen ja muiden sääntelyjen välttämiseksi. Brittiläisen sanomalehden The Guardianin mukaan Deutsche Bank hallitsi tämän toteuttamiseen käytetyistä 1 200 postilaatikkoyrityksestä suunnilleen kolmasosaa. Seurasi maailmanlaajuisia protesteja ja useita rikossyytteitä, ja Islannin ja Pakistanin hallitusten johtajat kumpikin erosivat.\nKysymys: Kuka brittiläisen lehdistön väitteen mukaan hallinnoi monia varojen piilottamisessa käytettyjä yrityksiä tekstikatkelman mukaan?\nVaihtoehdot:\na. Eri pankkien varakkaat asiakkaat\nb. Panamalainen lakiyritys\nc. Deutsche Bank\nd. Pakistanin hallitus",
  "label": "c"
}
```
```json
{
  "text": "Teksti: Sundarban on maailman suurin mangrovemetsäalue. Se ulottuu 80 kilometriä (50 mailia) rannikolta Bangladeshin ja Intian takamaille. Sundarban on julistettu Unescon maailmanperintökohteeksi. Metsän Intian puolella sijaitsevaa osaa kutsutaan Sundarbanin kansallispuistoksi. Metsät eivät kuitenkaan ole pelkkiä mangrovesoita, vaan niihin kuuluu joitakin viimeisiä jäänteitä niistä mahtavista viidakoista, jotka aikoinaan peittivät koko Gangesin tasangon. Sundarban kattaa 3 850 neliökilometrin alueen, josta noin kolmasosa on vesi- tai suoalueiden peitossa. Vuodesta 1966 asti Sundarbans on ollut villieläinten suojelualue. Arvioidaan, että siellä on nykyään 400 intiantiikeriä ja suunnilleen 30 000 aksishirveä.\nKysymys: Mikä metsän osa on Intian puolella?\nVaihtoehdot:\na. Sundarbanin kansallispuisto\nb. Villieläinten suojelualue\nc. Maailmanperintökohde\nd. Gangesin tasanko",
  "label": "a"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Seuraavat ovat monivalintakysymyksiä (vastauksineen).
  ```
- Base prompt template:
  ```
  Kysymys: {text}
  Vaihtoehdot:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Vastaus: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Kysymys: {text}
  Vaihtoehdot:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Vastaa yllä olevaan kysymykseen käyttämällä 'a', 'b', 'c' tai 'd', äläkä mitään muuta.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset belebele-fi
```


## Common-sense Reasoning

### HellaSwag-fi

This dataset is a machine translated version of the English [HellaSwag
dataset](https://aclanthology.org/P19-1472/). The [dataset](https://huggingface.co/datasets/Finnish-NLP/hellaswag-fi-google-translate) was created by Finnish-NLP using Google Translate. The dataset is designed to
be used in EuroEval and it therefore already has a 1,024 / 256 / 2,048 split for the train, validation and test splits, respectively.

Here are a few examples from the training split:

```json
{
  "text": "[Otsikko ] Tiikkihuonekalujen tahraus [vaihe] Pyyhi lika, pöly ja roskat pois. [vaihe] Voit harjata lian pois kuivalla paperipyyhkeellä tai liinalla. Jos puhdistettavia kohtia on sitkeämpiä, voit hieroa ne puhtaaksi kostealla rievulla.\nVastausvaihtoehdot:\na. [vaihe] Poista tahrat tiikistä pyyhkimällä ne kuivalla talouspaperilla. [vaihe] Noudata samoja puhdistustoimenpiteitä, joita käytit tahran kanssa.\nb. Aja niiden yli puhdistusaineella, kunnes tahra on poissa. [vaihe] Kokeile puupetsin ja öljyn yhdistelmää.\nc. [välivaiheet] Älä käytä puhdistusaineita. Saatat vahingoittaa puuta, mutta vaikeutat varmasti värjäysprosessia.\nd. Poista mahdollisimman paljon likaa levittämällä tahra kevyelle, pörröiselle liinalle tai kädelle ja pyyhkimällä se pois. [vaihe] Käytä hankaamiseen valkaisuainetta ja vettä.",
  "label": "c",
}
```
```json
{
  "text": "Pieni ryhmä ihmisiä nähdään uimassa altaan ympärillä ja johtaa useisiin laukauksiin, joissa uimari heittää pallon verkkoon. Maalivahti torjuu muutaman laukauksen ja vaihtaa sitten toisen joukkuetoverinsa kanssa yleisön hurraten. ihmisiä\nVastausvaihtoehdot:\na. cheer vielä kerran ja palaa uimaan uima-altaan ympärille.\nb. vaihda jatkuvasti pois ja johtaa siihen, että yksi joukkue voittaa ja juhlii kaikki yhdessä vedessä.\nc. Curra ja hyppää vuorotellen ylös ja eteenpäin pelaamalla biljardia.\nd. ensimmäinen video, jossa muut joukkuetoverit sukeltavat altaaseen ja hyppäävät ylös ja alas ponnahduslaudalla.",
  "label": "b",
}
```
```json
{
  "text": "Kahden ihmisen nähdään kävelevän pöytäjalkapallopöydän ympärillä pelaamassa. ihmisiä\nVastausvaihtoehdot:\na. pitäkää kupit ylös ja alakaa pelata peliä ja lyödä toisianne.\nb. Tartu sauvoista ja lyö palloa pöydän ympärillä.\nc. Jatka kävelemistä ja yksi henkilö lyö pallon verkon yli.\nd. siirrä ympäri pöytää heittäen palloa ympäriinsä, kun ihmiset katselevat sivuilla.",
  "label": "b",
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Seuraavat ovat monivalintakysymyksiä (vastauksineen).
  ```
- Base prompt template:
  ```
  Kysymys: {text}
  Vastausvaihtoehdot:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Vastaus: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Kysymys: {text}
  Vastausvaihtoehdot:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Vastaa yllä olevaan kysymykseen käyttämällä 'a', 'b', 'c' tai 'd', äläkä mitään muuta.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset hellaswag-fi
```


## Summarization

### XLSum-fi

This dataset is a machine translation of the XL-Sum dataset, which was published in [this paper](https://aclanthology.org/2021.findings-acl.413/). [TurkuNLP](https://huggingface.co/datasets/TurkuNLP) has translated the dataset to Finnish using DeepL.

The original Finnish XL-Sum dataset contains 54,966 / 1,803 / 1,791 training, validation and test samples, respectively. We use 1,024 / 256 / 2,048 samples for our training, validation and test splits, respectively. The new training and validation splits are subsets of the original splits. The test split is the same as the original test split + additional samples from the original validation split.

Here are a few examples from the training split:

```json
{
  "text": "Poliisi kutsuttiin Century Wharfiin keskiviikkona noin kello 14:15 GMT. 66-vuotias mies on pidätetty murhasta epäiltynä, ja häntä pidetään vangittuna. Etelä-Walesin poliisi ilmoitti, että se siirtää asian vapaaehtoisesti riippumattoman poliisin valituslautakunnan käsiteltäväksi.",
  "target_text": "Murhatutkinta on aloitettu sen jälkeen, kun 65-vuotiaan naisen ruumis löytyi Cardiff Bayn asunnosta."
}
```
```json
{
  "text": "Yritys on nimittänyt KPMG:n tarkastelemaan uudelleenjärjestelyvaihtoehtoja sen jälkeen, kun paikallisviranomaisten menojen leikkaukset heikensivät sen liiketoimintanäkymiä. Southern tarjoaa hoitoa yli 31 000 ihmiselle, ja suurin osa rahoituksesta tulee NHS:ltä ja kunnilta. Yrityksen mukaan budjettileikkaukset merkitsivät sitä, että sen vuokrataakka oli 'kestämätön'. Southern kertoi keskustelevansa vuokranantajien kanssa uudelleenjärjestelystä ja varoitti myös, että se oli vaarassa jättää velkansa maksamatta. 'Yhtiön lainanantajat ovat tietoisia uhkaavasta pankkikovenanttirikkomuksesta, mutta ne tukevat edelleen täysin toimia, joihin yhtiö ryhtyy ongelmiensa ratkaisemiseksi', Southern sanoi lausunnossaan. Yhtiö vahvisti myös, ettei se enää keskustele mahdollisten ostajien kanssa. 'Hallitus katsoo, että yksikään näistä ehdotuksista ei todennäköisesti johda siihen, että lähitulevaisuudessa tehtäisiin mielekäs tarjous, ja se on päättänyt olla jatkamatta niiden käsittelyä', Southern totesi. Southernin osakkeet, joiden arvo oli 606 penceä vuonna 2007, olivat keskipäivällä 6,3 penniä.",
  "target_text": "Yhdistyneen kuningaskunnan suurimman hoivakotien ylläpitäjän Southern Cross Healthcaren osakkeet ovat romahtaneet 60 prosenttia, kun on uutisoitu, että taloudelliset ongelmat ovat lisääntymässä."
}
```
```json
{
  "text": "Pohjois-Walesin palo- ja pelastusviranomainen vahvisti maanantaina talousarvionsa vuosiksi 2015-16. Viranomainen on suostunut leikkaamaan neljä johtotehtävää, leikkaamaan joitakin palveluja ja käyttämään vararahastoa, jotta se voi hyväksyä 32,1 miljoonan punnan talousarvionsa. On pelätty, että sadat palomiehet voivat lähteä seuraavien viiden vuoden aikana tehtävien budjettileikkausten seurauksena.",
  "target_text": "Pohjois-Walesin palomiehet lopettavat suurten eläinten pelastamisen ja vähentävät väärien hälytysten määrää, jotta talous saataisiin tasapainoon."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:
  ```
  Seuraavassa on artikkeleita ja niihin liittyviä tiivistelmiä.
  ```
- Base prompt template:
  ```
  Uutisartikkeli: {text}
  Tiivistelmä: {target_text}
  ```
- Instruction-tuned prompt template:
  ```
  Uutisartikkeli: {text}

  Kirjoita tiivistelmä yllä olevasta artikkelista.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset xlsum-fi
```
