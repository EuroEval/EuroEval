# 🇪🇪 Estonian

This is an overview of all the datasets used in the Estonian part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.


## Sentiment Classification

### Estonian Valence Corpus

This dataset was published in [this
paper](http://dx.doi.org/10.7592/FEJF2016.64.polarity). The dataset was compiled of
articles of different rubrics of online dailies, weeklies, and reader comments, while
the polarity of each paragraph was determined by native Estonian readers.

There are 4 labels in the original dataset instead of the usual 3. Examples with the
labels representing 'mixed' emotion (vastuoluline) were filtered out mainly to be
consistent with rest of the languages in EuroEval.

The original full dataset consists of 3,277 / 818 samples for the training and test
splits, respectively. Having filtered out 'mixed' examples, we truncate the train split
to 1,024 examples, and redistribute the rest to validation and test resulting in the
final size of 1,024 / 256 / 2,048 for the training, validation and test splits,
respectively.


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


## Named Entity Recognition

### EstNER

This dataset was published in [this
paper](https://aclanthology.org/2023.nodalida-1.76/). The dataset is a manually
annotated collection of Estonian news and social media texts.

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


## Linguistic Acceptability

### Grammar-et

The dataset is a reorganized and simplified version of the [TartuNLP EstGEC
dataset](https://github.com/TartuNLP/estgec) dataset. The dataset includes the original
sentences and their corrected versions.

The original full dataset consists of 7,937 / 1,000 samples for training and testing,
respectively. The original dataset consists of 8,937 samples, from which we use 1,024 /
256 / 2,048 samples for training, validation and testing, respectively. The test split
is extended with additional examples from the train split. The validation split is also
created using examples from the train split.

Here are a few examples from the training split:

```json
{
  "text": "Meie kahe rahva peaks lõpetama tobedused teineteise vastu ja mõista et tuleme siin naabrideks olema igavesti.",
  "label": "incorrect"
}
```
```json
{
  "text": "Esiteks valid sa raamatu ise, kui sul on seda vaja, näiteks õppekirjanduse puhul, või tuleb see mingil teisel põhjusel läbi lugeda, näiteks sõbra nõuandel.",
  "label": "correct"
}
```
```json
{
  "text": "Ma olen kindel et mitte amet rikkub inimest, aga raha.",
  "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:
  ```
  Järgnevad on laused ja kas need on grammatiliselt õiged.
  ```
- Base prompt template:
  ```
  Lause: {text}
  Grammatikaliselt õige:: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Lause: {text}

  Otsusta, kas lause on grammatiliselt õige või mitte. Vasta {labels_str}, ja mitte midagi muud.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset grammar-et
```


### Unofficial: ScaLA-da

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Estonian Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Estonian-EDT) by assuming that the
documents in the treebank are correct, and corrupting the samples to create
grammatically incorrect samples. The corruptions were done by either removing a word
from a sentence, or by swapping two neighbouring words in a sentence. To ensure that
this does indeed break the grammaticality of the sentence, a set of rules were used on
the part-of-speech tags of the words in the sentence.

The original dataset consists of 19,298 samples, from which we use 1,024 / 256 / 2,048
samples for training, validation and testing, respectively (so 3,328 samples used in
total). These splits are used as-is in the framework.

Here are a few examples from the training split:

```json
{
  "text": "Selleks kirjeldatakse programmi käitumust spetsiaalse juhtvoograafi ehk CFG abil (CFG - Control Flow Graph).",
  "label": "correct"
}
```
```json
{
  "text": "Karuks ütleb, et oma natuuri tõttu huvitub ta ka väga paljudest muudest asjadest: sise- ja välispoliitikast, ajaloost, erinevatest ühiskonnaprobleemidest.",
  "label": "correct"
}
```
```json
{
  "text": "Juta teab oma kogemusest, et selline söök tahab pikka ja tunneb kohe ära, et varem valmis tehtud asi on mikrolaineahjus üles soojendatud.",
  "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Järgnevad on laused ja kas need on grammatiliselt õiged.
  ```
- Base prompt template:
  ```
  Lause: {text}
  Grammatikaliselt õige: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Sætning: {text}

  Bestem om sætningen er grammatisk korrekt eller ej. Svar med 'ja', hvis sætningen er korrekt, og 'nej', hvis den ikke er.
  ```
- Label mapping:
    - `correct` ➡️ `jah`
    - `incorrect` ➡️ `ei`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset scala-et
```


## Reading Comprehension

### MultiWikiQA-et

This dataset will be published in an upcoming paper, and contains Estonian Wikipedia
articles with generated questions and answers, using the LLM Gemini-1.5-pro.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
    "context": "Kõrgõzstani 2010. aasta ülestõus, mäss,  revolutsioon või riigipööre oli sündmusteahel, mis kukutas president Kurmanbek Bakijevi ja tõi võimule ajutise valitsuse Roza Otunbajevaga eesotsas.\n\nRahutused algasid 6. aprillil Talasis ja levisid järgmisel päeval teistesse Kõrgõzstani põhjaosa linnadesse. Rahutustes hukkus vähemalt 85 inimest, täiendavaid ohvreid tõid kaasa järgnenud korratused.\n\nRahutuste põhjused \n\nVõimuvahetuse järel kokku tulnud "sõltumatu ühiskondlik komisjon" 6. kuni 8. aprilli sündmuste uurimiseks, leidis, et rahutused põhjustas Bakijevi korruptsioonis, onupojapoliitikas ja riigi vara kõrvaletoimetamises süüdistatud valitsuse "laostav poliitika", sotsiaalse kindlustunde vähenemine, laiade ühiskonnakihtide vaesumine ([48;56;245;2016;3430tostujõu vähenemine ja kiire inflatsioon), opositsiooni ja ajakirjanike laiaulatuslik tagakiusamine, poliitiliste mõrvade seeria ning sõltumatute trükiväljaannete, raadio- ja telekanalite sulgemine.\n\nRahutuste vahetuks käivitajaks on peetud hinnatõusu. Elektri- ja küttehinnad olid kahekordistunud, tõusnud olid näiteks mobiilteenuste hinnad.\n\nTalasi sündmused \n\n6. aprilli hommikul vahistati Talasis partei Ata-Meken aseesimees Bolotbek Šernijazov. Lõunaks kogunes sadu vahistatu poolehoidjaid miilitsajaoskonna juurde ja Šernijazov vabastati. Meeleavaldajad liikusid oblastivalitsuse hoone juurde ja võtsid pantvangi oblastijuhi. Hangiti kive ja bensiini, valmistati ette pudeleid süüteseguga.\n\nÕhtul saabusid Biškekist lennukid ja helikopterid 200 korrakaitsejõudude teenistujaga, kes pidid asepeaminister Akõlbek Džaparovi juhtimisel läbi viima erioperatsiooni oblastijuhi vabastamiseks. Meeleavaldajate vastu kasutati pisargaasi, kummikuule ja valgusmüra granaate(?). Oblastijuht vabastati ja Bolotbek Šernijazov võeti jõudu kasutades uuesti vahi alla.\n\nRahvahulk piiras sisse grupi miilitsatöötajaid, asus neid peksma ja süütas oblastivalitsuse hoone, kuhu umbes 150 miilitsat varjunud olid. Miilitsail lõppes erivarustus, täiendusi toonud sõiduki hõivasid mässajad ja kasutasid valdusse võetud vahendeid miilitsa vastu.\n\nÖöl vastu 7. aprilli vahistati Biškekis ja Ošis mitu opositsioonijuhti, sealhulgas Ömürbek Tekebajev, Almazbek Atambajev ja Temir Sarijev.\n\nTuhanded inimesed kogunesid piirama presidendi- ja valitsushoonet Biškekis. Julgeolekujõud kasutasid rahva laialiajamiseks tulirelvi. Bakijev lahkus pealinnast Džalal-Abadi. Roza Otunbajeva, Tekebajev, Atambajev ja teised tuntud opositsionäärid kuulutasid endid ajutiseks valitsuseks. Bakijev varjas end kuni 15. aprillini ja lahkus seejärel Valgevenesse.\n\nViited\n\nKõrgõzstani poliitika\n2010\nRevolutsioonid",
    "question": "Kes eemaldati võimult Kõrgõzstani 2010. aasta rahutuste ajal?",
    "answers": {
        "answer_start": array([100]),
        "text": array(["president Kurmanbek Bakijev"], dtype=object)
    }
}
```
```json
{
    "context": "Lepna haruraamatukogu on raamatukogu, mis tegutseb Lääne-Viru maakonnas Rakvere vallas Lepna külas. See on Sõmeru raamatukogu struktuuriüksus.\n\nAjalugu \n\nRaamatukogu täpne ja ametlik asutamise aeg pole kindlalt teada. Raamatukogus leiduvas kroonika kirjutaja arvates ulatub see 1892. aastasse, kuid dokumentaalset kinnitust sellele ei ole leitud. Kirjas on, et raamatukogu asutati Tõrma valla kooli juurde. Küll aga saab ajalehest Virulane (10. mai 1928) lugeda, et Rakvere valla avalik raamatukogu asutati 1925. aasta lõpul "Avalikkude raamatukogu seaduse" alusel ja alustas tegevust 1. jaanuaril 1926.\n\nRaamatukogu aluspõhi töötati välja Rakvere vallas töötavate seltside poolt, kes oma raamatukogud täies ulatuses avalikule raamatukogule üle andsid. Raamatukogu tegevust toetasid rahaliselt nii Rakvere vald kui Haridusministeerium.\n\nRakvere valla avaliku raamatukogu all töötasid erinevad osakonnad (7. osakond),\xa0mille juhatajateks olid nii koolide juhatajad ja teised haridustegelased. Edukamaks osakonnaks aga peeti Mädapea osakonda, mida juhatas Mädapea Algkooli juhataja Jakob Awik. Raamatukogu asus Mädapea mõisas, Mädapea Algkooliga ühe katuse all. 1934. aastal laenutas raamatuid välja toonane kooliõpetaja Helene Tammik.\n\n1965–1991 oli Mädapea mõis Energia kolhoosi hallata.\n\n1991 avati Lepna alevikus paiknev Lepna teeninduspunkt kortermaja neljatoalises korteris kolmandal korrusel. Raamatukogu kogu oli jaotunud kahte hoonesse: Mädapea mõisa ja Lepna kortermajja.\n\n1995 kolis raamatukogu Mädapealt Lepna alevikku kortermaja kahte korterisse, mis asusid maja kolmandal korrusel.\n\n26. juunist 1996 nimetatakse Mädapea raamatukogu Lepna raamatukoguks.\n\nAlates 12. septembrist 2018 kuulub raamatukogu haruraamatukoguna Sõmeru raamatukogustruktuuri.\n\nRaamatukogul on kaks teeninduspunkti: Veltsi Lasteaed-Algkoolis ja Lasila Põhikoolis.\n\nTöötajad \nMädapea raamatukogu töötajad alates 1927. aastast vastavalt Lepna raamatukogu üleandmise ja vastuvõtmise aktidele:\n Jakob Awik 1927\n Helene Tammik 1934\n Ellen Kuusik 01.11.1945–20.02.1954\n Salme Partei 20.02.1949–25.11.1954\n Vilma Niitla (Laar) 25.11.1954–25.05.1955\n Evi Suurväli 25.05.1955–06.08.1956\n Elvi Sats 06.08.1956–30.05.1957\n Maie Tiigi 30.05.1957–05.04.1959\n Vaike Salamets 05.04.1959–15.09.1961\n Elfride Salumäe 15.09.1961–28.03.1963\n Linda Rünk 28.03.1963–07.03.1964\n Evi Lahi 07.03.1964–01.08.1964\n Liia Pall 1964–1947\n Ille Laarman 1975–1977\n Anne Pilipenko1978–20.09.2009 (Mädapea Raamatukogu / Lepna Raamatukogu)\n Jaana Ant 01.09.2009–01.07.2016\n Olga Samra (raamatukogu direktori kohusetäitja) 01.09.2016-märts 2017\n Silja Raudsepp märts 2017– august 2018\n Jaana Ant 22.10.2018 –\n\nKirjandus \n Virulane, 10. mai 1928. lk 4\n\nViited\n\nVälislingid \n Rakvere valla koduleht, rakvere valla raamatukogud\n Lepna raamatukogu koduleht\n Lepna raamatukogu Facebookis\n\nLääne-Viru maakonna raamatukogud\nRakvere vald",
    "question": "Kes valitses Mädapea mõisat ajavahemikul 1965–1991?",
    "answers": {
        "answer_start": array([1261]),
        "text": array(["Energia kolhoosi"], dtype=object)
    }
}
```
```json
{
    "context": "Kannuka pank ehk Kannuka klindineemik on Põhja-Eesti panga osa Ida-Viru maakonnas Narva-Jõesuu linnas ja Sillamäe linnas. Kannuka pank algab peale Sõtke klindilahest ja kulgeb kuni Pimestiku panga moodustavate klindisaarteni.\n\nKannuka klindineemik on läänepoolsem 2,5 km pikkune ja 1,5 km laiune osa Vaivara Sinimägedega sarnase, Sillamäe rikkevööndi, tektooniliselt rikutud keerulise ehitusega klindivööndist. Panga paeplatoo kõrgub 28...30 meetrit üle merepinna. Paeplatool avanevad õhukese moreenikihi all Kesk-Ordoviitsiumi Volhovi lademe Toila kihistu glaukoniiti sisaldavad lubjakivid.\n\nKlindiserv on kaetud rusukalletega ning paljandid peaaegu puuduvad välja arvatud Sillamäe linnas mõned üksikud lõigud. Kannuka pank on nüüdisajal kasvanud suures osas Sillamäe linna sisse, kuid kohati on säilinud väikeste lõikudena ka pangale omast klindimetsa.\n\nVaata ka\nEesti pankade loend\nMeriküla pank\nPargimäe pank\nPimestiku pank\nPõrguaugumäe pank\nTornimäe pank\nUtria kõrgekallas\nUdria maastikukaitseala\nUtria savikallas\nVaivara klindilõik\nPuhkovo-Olgina klindiplatoo (Puhkovo-Vodava-Olgina klindiplatoo)\nLaagna klindisaarestik\n\nKirjandus\nKalle Suuroja. Põhja-Eesti pangad, Tallinn 2004.\n\nIda-Viru maakonna paljandid\nNarva-Jõesuu linn",
    "question": "Mis klindivööndis asub Kannuka pank?",
    "answers": {
        "answer_start": array([330]),
        "text": array(["Sillamäe rikkevööndi"], dtype=object)
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:
  ```
  Järgnevad on tekstid koos küsimuste ja vastustega.
  ```
- Base prompt template:
  ```
  Tekst: {text}
  Küsimus: {question}
  Vasta maksimaalselt 3 sõnaga: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Tekst: {text}

  Vasta järgmisele küsimusele ülevaltoodud teksti kohta maksimaalselt 3 sõnaga.

  Küsimus: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset multi-wiki-qa-et
```


## Knowledge

### Exam-et

This dataset was released in [this
repository](https://huggingface.co/datasets/TalTechNLP/exam_et) and contains questions
with multiple-choice answers from Estonian high-school tests.

The original full dataset contains 1,614 samples in a single split, across eight
different subjects. We use a 512 / 64 / 896 split for training, validation and testing,
respectively, with stratification based on the subject.

Here are a few examples from the training split:

```json
{
  "text": "Kas väide iseloomustab Eestit perioodil 1920-1934 või 1934-1940: riigikogu valiti iga kolme aasta järel?\nVastusevariandid:\na. Eesti 1920-1934\nb. Eesti 1934-1940",
  "label": "a"
}
```
```json
{
  "text": "Kas väide on tõene või väär? Veendumuste pärast võib isikult Eesti kodakondsuse ära võtta.\nVastusevariandid:\na. tõene\nb. väär",
  "label": "b"
}
```
```json
{
  "text": "Kellel on Eesti vabariigis õigus kehtestada eriolukord?\nVastusevariandid:\na. politseil\nb. õiguskantsleril\nc. vabariigi valitsusel\nd. riigikohtul",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Järgnevad on vastusevariantidega küsimused (koos vastustega).
  ```
- Base prompt template:
  ```
  Küsimus: {text}
  Vastusevariandid:
  a. {option_a}
  b. {option_b}
  Vastus: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Küsimus: {text}
  Vastusevariandid:
  a. {option_a}
  b. {option_b}
  (...)
  o. {option_o}

  Vasta ülaltoodud küsimusele ainult 'a', 'b', (...), 'n' või 'o', ja mitte millegi
  muuga.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset exam-et
```


## Common-sense Reasoning

### WinoGrande-ET

The dataset includes the [WinoGrande](https://doi.org/10.48550/arXiv.1907.10641) test
set translated and culturally adapted by hand by a professional translator (citation
TBA). The structure of the dataset is identical to the original. Since train and dev
splits were not translated manually, we employ the GPT-4o model to translate the
expected number of examples starting from the beginning of the respective splits. The
final dataset size is 1,024 / 256 / 1,767 for the training, validation and test splits,
respectively.

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


## Summarization

### ERRNews

The dataset was released in [this paper](https://doi.org/10.22364/bjmc.2022.10.3.23).

The dataset consists of news story transcripts of ERR News broadcasts scraped from from
the [ERR Archive](https://arhiiv.err.ee/err-audioarhiiv) News generated by an ASR
pipeline paired with the human written summary from the archive.

The original full dataset consists of 10,420 / 523 / 523 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 2,048 split for training,
validation and testing, respectively. The test split is extended with additional examples
from the train split.

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
