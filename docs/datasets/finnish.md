# 🇫🇮 Finnish

This is an overview of all the datasets used in the Finnish part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### ScandiSent-fi

This dataset consists of reviews from Trustpilot and was published
[here](https://aclanthology.org/2021.nodalida-main.42/). It is a binary sentiment
classification dataset, with labels "positive" and "negative".

For the Finnish part of the dataset, there are 10,000 training samples. From these
samples, we have created a 1,024 / 256 / 2,048 split for the train, validation and test
splits, respectively.

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

  ```text
  Seuraavassa on arvosteluja ja niiden tunnesävy, joka voi olla 'positiivinen' tai 'negatiivinen'.
  ```

- Base prompt template:

  ```text
  Teksti: {text}
  Tunnesävy: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Teksti: {text}

  Luokittele arvostelun tunnesävy. Vastaa vain 'positiivinen' tai 'negatiivinen', ei muuta.
  ```

- Label mapping:
  - `positive` ➡️ `positiivinen`
  - `negative` ➡️ `negatiivinen`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scandisent-fi
```

## Named Entity Recognition

### Turku-NER-fi

This dataset was published in [this paper](https://aclanthology.org/2020.lrec-1.567/).
The dataset is a manually annotated corpus built on the Universal Dependencies Finnish
corpus. The corpus was created by the Turku NLP group.

The original dataset contains 12,217 / 1,364 / 1,555 samples for the training,
validation and test splits, respectively. We use 1,024 / 256 / 2,048 samples for our
training, validation and test splits, respectively. All the new splits are subsets of
the original splits.

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

  ```text
  Seuraavassa on lauseita ja JSON-sanakirjoja, jotka sisältävät annetussa lauseessa esiintyvät nimetyt entiteetit.
  ```

- Base prompt template:

  ```text
  Lause: {text}
  Nimetyt entiteetit: {label}
  ```

- Instruction-tuned prompt template:

  ```text
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
euroeval --model <model-id> --dataset turku-ner-fi
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

The original dataset consists of 15,136 samples, from which we use 1,024 / 256 / 2,048
samples for training, validation and testing, respectively (so 3,328 samples used in
total). These splits are used as-is in the framework.

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

  ```text
  Seuraavat ovat lauseita ja ovatko ne kieliopillisesti oikein.
  ```

- Base prompt template:

  ```text
  Lause: {text}
  Kieliopillisesti oikein: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Lause: {text}

  Määritä onko lause kieliopillisesti oikein vai ei. Vastaa 'kyllä', jos lause on oikein, ja 'ei', jos se ei ole.
  ```

- Label mapping:
  - `correct` ➡️ `kyllä`
  - `incorrect` ➡️ `ei`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-fi
```

## Reading Comprehension

### TydiQA-fi

This question-answering dataset was published in [this
paper](https://aclanthology.org/2020.tacl-1.30/). TydiQA is a multilingual dataset
covering 11 typologically diverse languages with 204K question-answer pairs collected
from native speakers genuinely seeking information. It was designed to evaluate models
across languages with varied linguistic features and contains questions written directly
in each language without translation.

The original Finnish TydiQA dataset contains 6,855 training and 782 validation samples
(we use the [secondary task
subset](https://huggingface.co/datasets/google-research-datasets/tydiqa/viewer/secondary_task?views%5B%5D=secondary_task_train)).
We created a 1,024 / 256 / 2,024 split, where the samples from the train and validation
split are sampled from the original train and validation splits, respectively. The test
set consists of the remaining samples from the original validation split + additional
samples from the original train split.

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
```

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

  ```text
  Seuraavassa on tekstejä ja niihin liittyviä kysymyksiä ja vastauksia.
  ```

- Base prompt template:

  ```text
  Teksti: {text}
  Kysymys: {question}
  Vastaa enintään 3 sanalla: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Teksti: {text}

  Vastaa seuraavaan kysymykseen yllä olevasta tekstistä enintään 3 sanalla.

  Kysymys: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset tydiqa-fi
```

### Unofficial: BeleBele-fi

This dataset was published in [this paper](https://aclanthology.org/2024.acl-long.44/)
and features multiple-choice reading comprehension questions across 122 languages.

The original dataset contains 900 unique multiple-choice reading comprehension passages
and questions. From these, we use a 256 / 64 / 580 split for training, validation and
testing, respectively.

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

  ```text
  Seuraavat ovat monivalintakysymyksiä (vastauksineen).
  ```

- Base prompt template:

  ```text
  Kysymys: {text}
  Vaihtoehdot:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Vastaus: {label}
  ```

- Instruction-tuned prompt template:

  ```text
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
euroeval --model <model-id> --dataset belebele-fi
```

### Unofficial: MultiWikiQA-fi

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
    "context": "Aarne Silvio Heikinheimo (20. maaliskuuta 1894 Tornio – 24. tammikuuta 1938) oli suomalainen jääkärikenraalimajuri. Hänen vanhempansa olivat ylimetsänhoitaja Johan Henrik Heikel ja Sally Armida Thauvón. Hänet vihittiin avioliittoon vuonna 1919 Sylvi Amalia Jurveliuksen kanssa.\n\nOpinnot\nHeikinheimo kirjoitti ylioppilaaksi Oulun suomalaisesta yhteiskoulusta vuonna 1913 ja liittyi Pohjois-Pohjalaiseen Osakuntaan. Opintojaan hän jatkoi Teknillisen korkeakoulun koneinsinööriosastolla vuosina 1913–1914. Hän seurasi opetusta Sotakorkeakoulun komentajakurssilla vuonna 1925 ja kävi Sotakorkeakoulun yleisen osaston vuosina 1926–1927.\n\nJääkäriaika\nHän liittyi yhtenä ensimmäisten vapaaehtoisten joukkoon, jonka päämääränä oli Saksassa sotilaskoulutusta antava Pfadfinder-kurssi, joka järjestettiin Pohjois-Saksassa sijaitsevalla Lockstedter Lagerin harjoitusalueella. Leirille hän ilmoittautui 25. helmikuuta 1915. Hänet sijoitettiin joukon 1. komppaniaan. Myöhemmin hänet sijoitettiin Kuninkaallisen, Preussin Jääkäripataljoona 27:n 1. komppaniaan. Hän otti osaa taisteluihin ensimmäisessä maailmansodassa Saksan itärintamalla Misse-joella, Riianlahdella ja Aa-joella. Hän osallistui kesällä vuonna 1917 Libaussa järjestetyille moottoriveneenkuljettaja- ja konekivääriasemestarikursseille ja  elokuussa vuonna 1917 Schaulenissa järjestetylle autokurssille  sekä syksyllä Libaussa vuonna 1917 järjestetylle räjäytyskurssille.\n\nSuomen sisällissota\n\nKatso myös: Suomen sisällissota\nHän saapui Suomeen oberzugführer Friedel Jacobssonin komennuskunnan mukana 30. tammikuuta 1918 ja liittyi Perä-Pohjolan suojeluskuntajoukkoihin Tervolassa. Hänet komennettiin joukkueenjohtajaksi Tervolaa ja Torniota vastaan taisteleviin joukkoihin. Tervolan ja Tornion valtausten jälkeen hänet nimitettiin Kemin kaupungin komendantiksi 7. helmikuuta, kunnes 5. maaliskuuta hänet nimitettiin Perä-Pohjolan pataljoonan komentajaksi. Hän johdatti pataljoonansa taisteluihin Vilkkilässä, Haavistolla (Oriveden), Tervaniemessä, Lempäälässä, Vesilahdella, Karkussa ja Tyrväällä. Sisällissodan loppuvaiheissa hän sai tehtäväkseen muodostaa Lahdessa Itä-Uudenmaan rykmentin.\n\nSisällissodan jälkeinen aika\n\nSisällissodan jälkeen Heikinheimo määrättiin 1. heinäkuuta 1918 alkaen 1. Divisioonan adjutantiksi ja myöhemmin väliaikaiseksi esikuntapäälliköksi, josta hänet siirrettiin 15. elokuuta 1918 Suomen valkoisen kaartin I pataljoonan komentajaksi ja edelleen komentajaksi 11. syyskuuta 1918 Viipurin rykmentin II pataljoonaan. II Polkupyöräpataljoonan komentajaksi hänet siirrettiin 27. huhtikuuta 1921 ja Viipurin rykmentin komentajaksi 15. elokuuta 1924. Hän toimi 12. elokuuta 1926 alkaen komentajana Jääkäriprikaatissa, josta hänet siirrettiin komentajaksi 3. Divisioonaan 9. kesäkuuta 1928. Esikuntatehtäviin hänet siirrettiin 25. elokuuta 1934 ja sijoitettiin Yleisesikuntaan ja määrättiin jalkaväen tarkastajaksi. Hän menehtyi tapaturmaisesti koeammunnoissa Harakan saarella kranaatinheittimen putken räjähdettyä 24. tammikuuta 1938. Hänet on haudattu Ouluun Intiön hautausmaalle, aivan sankarihautojen viereen.\n\nLuottamustoimet\nHeikinheimo toimi 2. Divisioonan kunniatuomioistuimen puheenjohtajana vuonna 1920 ja  3. Divisioonan kunniatuomioistuimen puheenjohtajana vuosina 1921 ja 1925. Polkupyöräjoukkojen erikoiskysymyksiä käsitelleen komitean jäsenenä hän toimi vuonna 1922 ja polkupyöräjoukkojen ohjesääntökomitean puheenjohtajana vuonna 1924 sekä pikakiväärinkokeilukomitean jäsenenä vuosina 1924–1925. Talvivarustuskomitean jäsenenä hän toimi vuonna 1924 ja kenttävarustustoimikunnan puheenjohtajana vuosina 1931–1934 sekä ohjesääntökomitean puheenjohtajana vuonna 1934. Mikkelin kaupunkiseurakunnan lisätyn kirkkovaltuuston jäsenenä hän toimi vuosina 1933–1934.\n\nLähteet \n Puolustusministeriön Sotahistoriallisen toimiston julkaisuja IV, Suomen jääkärien elämäkerrasto, WSOY Porvoo 1938.\n Sotatieteen Laitoksen Julkaisuja XIV, Suomen jääkärien elämäkerrasto 1975, Vaasa 1975 ISBN 951-99046-8-9.\n\nViitteet \n\nJääkärikenraalit\nVuonna 1894 syntyneet\nVuonna 1938 kuolleet",
    "question": "Milloin Aarne Heikinheimo sai ylioppilastutkinnon suoritettua?",
    "answers": {
        "answer_start": array([365]),
        "text": array(["1913"], dtype=object)
    }
}
```

```json
{
    "context": "Peter Costa (s. 17. tammikuuta Kíti, Kypros) on englantilainen Las Vegasissa asuva pokeriammattilainen. Hänen vanhempansa ovat kyproksenkreikkalaisia. Perhe muutti Liverpooliin Peterin ollessa nuori. Perheen yritys myi \"fish and chipsejä\" ja yritys laajentui myöhemmin ketjuksi.\n\nBritteinsaarilla Costa tuli tunnetuksi voitettuaan Late Night Pokerin kuudennen tuotantokauden finaalin. Lopun kaksinpelissä Costa kukisti itävaltalaisen Jin Cai Linin ja ansaitsi 60\xa0000 puntaa.\n\nTammikuussa 2003 Costa voitti Aussie Millions -tapahtuman pääturnauksen ja ansaitsi ykköstilastaan 394\xa0870 Australian dollaria. Costalla on myös useita turnausvoittoja Yhdysvalloista: esimerkiksi kesäkuussa 2002 hän voitti kolme turnausta kolmessa viikossa – kaikissa näissä ykköspalkinto oli yli 110\xa0000 dollaria. \n\nWorld Series of Pokerissa Costa on parhaimmillaan ollut seitsemäs (kaksi kertaa). World Poker Tourilta hänellä on rahasijoja, mutta ei toistaiseksi finaalipöytäsijoituksia.\n\nVuosina 2002 ja 2003 Costa oli ehdolla Europaan parhaan pelaajan palkinnon saajaksi. Hän teki maailmanennätyksen voitettuaan kaikkien aikojen suurimman (1\xa0166 pelaajaa) limiitti-hold\'em -turnauksen Orleansin kasinolla heinäkuussa 2003.\n\nKesäkuussa 2007 Costan pokeriuran turnausansiot ylittivät 1,7 miljoonaa dollaria.\n\nLähteet\n\nAiheesta muualla \n\n \n WPT:n profiili\n PokerListings.com:n profiili \n\nBrittiläiset pokerinpelaajat",
    "question": "Mikä on Peter Costan asuinpaikka?",
    "answers": {
        "answer_start": array([63]),
        "text": array(["Las Vegasissa"], dtype=object)
    }
}
```

```json
{
    "context": "Sigrid Vaasa (1566–1633) oli Ruotsin kuninkaan Eerik XIV:n ja hänen puolisonsa Kaarina Maununtyttären tytär.\n\nSigrid Vaasa asui lapsuudessaan äitinsä Kaarina Maununtyttären kanssa Liuksialan kartanossa ja jäätyään kahdesti leskeksi palasi asumaan sinne kuolemaansa asti. Vuonna 1597 hän avioitui Henrik Klaunpoika Tottin kanssa. Sen jälkeen oli Kirkniemen ja Sjundbyn kartanoiden emäntä. Heidän lapsistaan merkittävin oli Åke Tott, joka sai mainetta kuningas Kustaa II Aadolfin johtamissa sodissa. Kaarle-herttuan ja Sigismundin valtataistelun aikana Henrik Tott asettui suomalaisten aatelismiesten ja sitä kautta myös Sigismundin puolelle, minkä vuoksi hän joutui pakenemaan maasta ja kuoli ilmeisesti noin vuonna 1603 maanpaossa. Sigrid solmi uuden avioliiton vuonna 1609 Natt och Dag -sukuun kuuluvan Nils Nilsinpojan kanssa, muutti Ruotsiin mutta jäi neljän vuoden kuluttua leskeksi. Leskeksi jäätyään hän palasi Suomeen ja kuoli Liuksialassa.\n\nLähteet\n\nRuotsin prinsessat\nVuonna 1566 syntyneet\nVuonna 1633 kuolleet",
    "question": "Millä kartanolla Sigrid Vaasa vietti lapsuusvuotensa?",
    "answers": {
        "answer_start": array([180]),
        "text": array(["Liuksialan kartanossa"], dtype=object)
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Seuraavassa on tekstejä ja niihin liittyviä kysymyksiä ja vastauksia.
  ```

- Base prompt template:

  ```text
  Teksti: {text}
  Kysymys: {question}
  Vastaa enintään 3 sanalla: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Teksti: {text}

  Vastaa seuraavaan kysymykseen yllä olevasta tekstistä enintään 3 sanalla.

  Kysymys: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-fi
```

## Knowledge

### Unofficial: INCLUDE-fi

This dataset is part of [INCLUDE](https://doi.org/10.48550/arXiv.2411.19799), a
comprehensive knowledge- and reasoning-centric benchmark that evaluates multilingual
LLMs across 44 languages. It contains 4-option multiple-choice questions extracted from
academic and professional exams, covering 57 topics including regional knowledge.

The original dataset consists of a 'validation' split used as training data and a 'test'
split. We use the 'validation' split as the training split, which has 25 samples. We
sample 64 samples from the 'test' split for the validation split, and use the remaining
512 samples for the test split. The sampling is done stratified by the subject column.

Here are a few examples from the dataset:

```json
{
  "text": "Mikä on Suomen pääkaupunki?\nVastausvaihtoehdot:\na. Turku\nb. Tampere\nc. Helsinki\nd. Oulu",
  "label": "c"
}
```

```json
{
  "text": "Kuka kirjoitti romaanin 'Seitsemän veljestä'?\nVastausvaihtoehdot:\na. Zacharias Topelius\nb. Aleksis Kivi\nc. Väinö Linna\nd. Mika Waltari",
  "label": "b"
}
```

```json
{
  "text": "Mikä soluorganelli on vastuussa energian tuotannosta?\nVastausvaihtoehdot:\na. Ribosomi\nb. Kloroplasti\nc. Mitokondriot\nd. Golgin laite",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Seuraavat ovat monivalintakysymyksiä (vastauksineen).
  ```

- Base prompt template:

  ```text
  Kysymys: {text}
  Vastaus: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Kysymys: {text}

  Vastaa yllä olevaan kysymykseen käyttämällä {labels_str}, äläkä mitään muuta.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset include-fi
```

## Common-sense Reasoning

### HellaSwag-fi

This dataset is a machine translated version of the English [HellaSwag
dataset](https://aclanthology.org/P19-1472/). The
[dataset](https://huggingface.co/datasets/Finnish-NLP/hellaswag-fi-google-translate) was
created by Finnish-NLP using Google Translate. The dataset is designed to be used in
EuroEval and it therefore already has a 1,024 / 256 / 2,048 split for the train,
validation and test splits, respectively.

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

  ```text
  Seuraavat ovat monivalintakysymyksiä (vastauksineen).
  ```

- Base prompt template:

  ```text
  Kysymys: {text}
  Vastausvaihtoehdot:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Vastaus: {label}
  ```

- Instruction-tuned prompt template:

  ```text
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
euroeval --model <model-id> --dataset hellaswag-fi
```

### Unofficial: GoldenSwag-fi

This dataset is a filtered and machine translated version of the English [HellaSwag
dataset](https://aclanthology.org/P19-1472/), featuring both video descriptions from
ActivityNet as well as how-to articles from WikiHow. The machine translated version was
published in [this paper](https://doi.org/10.48550/arXiv.2410.08928) and was done using
DeepL, and the filtering was published in [this
paper](https://doi.org/10.48550/arXiv.2504.07825), which resulted in higher quality
samples.

The original full dataset consists of 1530 / 1530 samples for training and validation,
respectively. However, they are exactly equal. We use a split of 660 / 256 / 2,048
samples for training, validation, and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Miten auton ulkoinen pesu tehdään oikein. Ensimmäinen asia, joka sinun on tehtävä kunnolla, on pestä autosi tehokkaasti. Ei ole mitään järkeä yrittää tehdä auton ulkoista detaljointia, jos päädyt vain naarmuttamaan ducosi entistä enemmän, koska jätit autoosi likaa. Sinun on ensin huuhdeltava autosi letkulla kovalla paineella.\nVastausvaihtoehdot:\na. Tämä poistaa suurimman osan liasta moottoristasi ja pitää moottorin moitteettomana. Käytä autosi pesemiseen korkeapainepesukoneita.\nb. Sitten sinun on alettava imuroida likaa pois. Kun olet poistanut mahdollisimman paljon likaa, voit palata ajoneuvon luokse keräämään roskia.\nc. Vie letku kaasuttimesta moottorilohkon yläosaan, odota viisi minuuttia, sulje sitten vesi ja päästä ilma ulos jäähdyttimestä. Irrota vanhat tiivisteet ja aloita vedellä pesu moottorin kannesta alas.\nd. Älä käytä letkusta lasertyyppistä pesua, vaan mieluummin pientä suppiloa. Aloita aina ylhäältä ja etene alaspäin.",
  "label": "d"
}
```

```json
{
  "text": "Miten kylpeä merisuolalla. Varaa itsellesi riittävästi aikaa 15-20 minuutin kylpyyn. Kylpy ei ole kuin suihku, jossa usein kiirehditään. Sen sijaan niiden on tarkoitus kestää pidempään, jotta keho ja mieli voivat rentoutua.\nVastausvaihtoehdot:\na. Ennen kylpyä haluat, että kehosi rentoutuu, ota päivittäin noin minuutti rentoutumista. Kylvystä voi saada samoja hyötyjä: suolahoito on helpompaa, mikä voi vähentää stressiä.\nb. Jotta saisit kylvystäsi suurimman hyödyn, suunnittele, että vietät vedessä 15-20 minuuttia. Ota suolakylpy illalla, jos haluat hoitaa unettomuutta.\nc. Jos haluat nopean kylpyläkokemuksen, 15-20 minuutin kylpy voi olla hyvä valinta. Anna itsellesi muutama tunti aikaa tottua lämpimään, rentouttavaan veteen.\nd. Jos sinulla on kiire, saatat jännittyä niin paljon, että menetät ajantajusi. Jos väsyt, ota myös nopea 15-20 minuutin kylpy.",
  "label": "b"
}
```

```json
{
  "text": "Kuinka tehdä ylösnousemussämpylöitä. Kaada maito kulhoon. Jotta hiiva aktivoituu, sinun on sekoitettava se lämpimään nesteeseen. Lisää ½ kupillista (118 ml) lämmintä maitoa tehosekoittimen kulhoon.\nVastausvaihtoehdot:\na. Jos haluat pidemmän prosessin, voit juoksuttaa hiivan lavuaarissa ennen kuin jatkat.... Sekoita maito ja seos vähitellen vispilällä.\nb. Sekoita, kunnes maito on hyvin vaaleaa (noin 110 ml). Jos maito on liian pehmeää tähän reseptiin, lisää 1/2 kupillista (120 ml) smetanaa.\nc. Maidon lämpötilan tulisi olla 105 °f (41 °c). Voit käyttää 1- tai 2-prosenttista maitoa, mutta täysmaidosta saadaan yleensä parhaat sämpylät.\nd. Jos sinulla on sauvasekoitin, voit tehdä sämpylöiden taikinan itse. Tarvitset vain 2 kuppia (500 ml) maitoa.",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Seuraavat ovat monivalintakysymyksiä (vastauksineen).
  ```

- Base prompt template:

  ```text
  Kysymys: {text}
  Vastausvaihtoehdot:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Vastaus: {label}
  ```

- Instruction-tuned prompt template:

  ```text
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
euroeval --model <model-id> --dataset goldenswag-fi
```

### Unofficial: Winogrande-fi

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2506.19468)
and is a translated and filtered version of the English [Winogrande
dataset](https://doi.org/10.1145/3474381).

The original full dataset consists of 47 / 1,210 samples for training and testing, and
we use 128 of the test samples for validation, resulting in a 47 / 128 / 1,085 split for
training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Kun Dennis oli Puolassa, hän nautti matkasta enemmän kuin Jason, koska _ ymmärsi puolaa syvällisemmin. Mihin tyhjä _ viittaa?\nVastausvaihtoehdot:\na. Dennis\nb. Jason",
  "label": "a"
}
```

```json
{
  "text": "Michael halusi viedä Craigin vesille uudella veneellään. _ sanoi, että olisi hauskaa näyttää hänelle köysiä. Mihin tyhjä _ viittaa?\nVastausvaihtoehdot:\na. Michael\nb. Craig",
  "label": "a"
}
```

```json
{
  "text": "Voittaaksemme käyttäytymisharhan, meidän täytyy keskittyä enemmän tietoisten toimien muuttamiseen kuin tiedostamattomien toimien muuttamiseen, koska _ toimet ovat vapaaehtoisia. Mihin tyhjä _ viittaa?\nVastausvaihtoehdot:\na. tiedostamattomat\nb. tietoiset",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Seuraavat ovat monivalintakysymyksiä (vastauksineen).
  ```

- Base prompt template:

  ```text
  Kysymys: {text}
  Vastausvaihtoehdot:
  a. {option_a}
  b. {option_b}
  Vastaus: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Kysymys: {text}
  Vastausvaihtoehdot:
  a. {option_a}
  b. {option_b}

  Vastaa yllä olevaan kysymykseen käyttämällä 'a' tai 'b', äläkä mitään muuta.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset winogrande-fi
```

## Summarisation

### XLSum-fi

This dataset is a machine translation of the XL-Sum dataset, which was published in
[this paper](https://aclanthology.org/2021.findings-acl.413/).
[TurkuNLP](https://huggingface.co/datasets/TurkuNLP) has translated the dataset to
Finnish using DeepL.

The original Finnish XL-Sum dataset contains 54,966 / 1,803 / 1,791 training, validation
and test samples, respectively. We use 1,024 / 256 / 2,048 samples for our training,
validation and test splits, respectively. The new training and validation splits are
subsets of the original splits. The test split is the same as the original test split +
additional samples from the original validation split.

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

  ```text
  Seuraavassa on artikkeleita ja niihin liittyviä tiivistelmiä.
  ```

- Base prompt template:

  ```text
  Uutisartikkeli: {text}
  Tiivistelmä: {target_text}
  ```

- Instruction-tuned prompt template:

  ```text
  Uutisartikkeli: {text}

  Kirjoita tiivistelmä yllä olevasta artikkelista.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset xlsum-fi
```

## Instruction-following

### IFEval-fi

This dataset was published [here](https://huggingface.co/datasets/LumiOpen/ifeval_mt)
and is a translation of the English IFEval dataset, which was published in [this
paper](https://doi.org/10.48550/arXiv.2311.07911) and contains 541 prompts, each with a
combination of one or more of 25 different constraints. The dataset was machine
translated with DeepL and manually reviewed and corrected by native speakers.

We use the original dataset as the test split, and do not include the other splits, as
we only evaluate models zero-shot and the size is too small to warrant an even smaller
validation set.

Here are a few examples from the test split:

```json
{
    "text": "Työskentelen markkinointiosastolla ja tarvitsen apuasi. Tarvitsen mallin uuden tuotteen, kannettavan kameran, mainosta varten. Kirjoita mallissa muutama sana isolla alkukirjaimella pääkohtien korostamiseksi. Rajoita isoilla kirjaimilla kirjoitettujen sanojen määrä alle neljään. Vastauksesi tulisi sisältää vähintään kymmenen lausetta.",
    "target_text": {
        "instruction_id_list": [
            "change_case:capital_word_frequency",
            "length_constraints:number_sentences",
            "language:response_language"
        ],
        "kwargs": [
            {
                "capital_frequency": 4,
                "capital_relation": "less than"
            },
            {
                "num_sentences": 10,
                "relation": "at least"
            },
            {
                "language": "fi"
            }
        ]
    }
}
```

```json
{
    "text": "Luo mainoskopio laajentamalla \"Kulje 100 kilometriä litralla moottoritiellä\" QA:n muotoon oudolla tyylillä. Vastauksesi tulisi sisältää alle 8 lausetta. Älä sisällä vastauksessasi avainsanoja \"kilometrimäärä\" tai \"polttoaine\".",
    "target_text": {
        "instruction_id_list": [
            "length_constraints:number_sentences",
            "keywords:forbidden_words",
            "language:response_language"
        ],
        "kwargs": [
            {
                "num_sentences": 8,
                "relation": "less than"
            },
            {
                "forbidden_words": [
                    "kilometrimäärä",
                    "polttoaine"
                ],
            },
            {
                "language": "fi"
            }
        ]
    }
}
```

```json
{
    "text": "Mitkä ovat hyviä ideoita aloittaville yrityksille? Kirjoita suomenkielinen runo tästä. Käytä vastauksessasi vain suomea, muita kieliä ei sallita.",
    "target_text": {
        "instruction_id_list": [
            "language:response_language"
        ],
        "kwargs": [
            {
                "language": "fi"
            }
        ]
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 0
- No prefix prompt, as only instruction-tuned models are evaluated on this task.
- No base prompt template, as only instruction-tuned models are evaluated on this task.
- Instruction-tuned prompt template:

  ```text
  {text}
  ```

  I.e., we just use the instruction directly as the prompt.

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ifeval-fi
```

## European Values

### ValEU-fi

This dataset is the official Finnish version of questions from the [European values
study](https://europeanvaluesstudy.eu/). The dataset contains multiple-choice
questions regarding people's values and beliefs across a variety of topics, such as
politics, religion and society.

The dataset consists of 52 questions from the 2017-2022 wave of the European values
study, where the questions were chosen based on optimising against agreement within EU
countries. We use only zero-shot evaluation on this dataset, and thus require no splits.

Here are a few examples from the training split:

```json
{
  "question_id": "D001_B",
  "text": "Kysyisin seuraavaksi, miten paljon luotat eri ryhmiin kuuluviin ihmisiin. Sano kunkin kohdalla, luotatko tähän ryhmään täysin, jonkin verran, et kovinkaan paljoa vai et lainkaan.\nOma perheesi\nVastausvaihtoehdot:\na. Luotan täysin\nb. Luotan jonkin verran\nc. En luota kovinkaan paljon\nd. En luota lainkaan"
}
```

```json
{
  "question_id": "E028",
  "text": "Seuraavassa on lueteltu joitakin poliittisen osallistumisen muotoja. Arvioi jokaisen kohdalla oletko tehnyt tällaista, saattaisitko tehdä tällaista vai etkö missään olosuhteissa tekisi tällaista.\nOsallistunut laittomiin lakkoihin, korpilakkoihin\nVastausvaihtoehdot:\na. Olen tehnyt\nb. Saattaisin tehdä\nc. En voisi koskaan tehdä"
}
```

```json
{
  "question_id": "F025",
  "text": "Kuulutko kirkkoon tai johonkin uskonnolliseen yhteisöön?\nVastausvaihtoehdot:\na. Ei kuulu kirkkokuntaan.\nb. Roomalaiskatolinen\nc. Protestantti\nd. Ortodoksinen (venäläinen/kreikkalainen/jne.)\ne. Juutalainen\nf. Muslimit\ng. Hindu\nh. Buddhalainen\ni. Muu kristillinen (evankelinen/helluntailainen/vapaakirkollinen/jne.)\nj. Muu"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 0
- Prefix prompt:

  ```text
  Seuraavat ovat monivalintakysymyksiä (vastauksineen).
  ```

- Base prompt template:

  ```text
  Kysymys: {text}
  Vastausvaihtoehdot:
  a. {option_a}
  b. {option_b}
  (...)
  k. {option_k}
  Vastaus: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Kysymys: {text}
  Vastausvaihtoehdot:
  a. {option_a}
  b. {option_b}
  (...)
  k. {option_k}

  Vastaa yllä olevaan kysymykseen käyttämällä 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
  'i', 'j' tai 'k', äläkä mitään muuta.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset valeu-fi
```
