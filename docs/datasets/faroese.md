# 🇫🇴 Faroese

This is an overview of all the datasets used in the Faroese part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### FoSent

This dataset was published in [this paper](https://aclanthology.org/2024.lrec-main.690/)
and is based on 170 news articles from the Faroese news sites
[Portalurin](https://portal.fo) and [Dimmalætting](https://dimma.fo). The sentiment
labels were manually annotated by two native speakers.

The original full dataset consists of 245 samples, which consisted of both a news
article, a chosen sentence from the article, and the sentiment label. We use both the
news article and the chosen sentence as two separate samples, to increase the size of
the dataset (keeping them within the same dataset split). In total, we use a 72 / 40 /
279 split for training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Eg koyri teg, tú koyrir meg Hetta er árstíðin, har vit vanliga fara í jólaborðhald at hugna okkum saman við vinum og starvsfeløgum. Og hóast vit kanska ikki hittast og koma saman á júst sama hátt, sum áðrenn korona rakti samfelagið, so eru óivað nógv sum kortini gleða seg til hesa tíðina við hugna og veitslulag Eins og undanfarin ár, fara Ráðið fyri Ferðslutrygd (í samstarvi við Betri Trygging og Trygd) at fremja átak fyri at steðga rúskoyring. Hetta verður gjørt við filminum  ”Eg koyri teg, tú koyrir meg”, ið er úrslitið av stóru hugskotskappingini hjá Ráðnum fyri Ferðslutrygd síðsta vetur. Filmslýsingin verður í hesum døgum víst í sjónvarpi, biografi og á sosialum miðlum. Brynhild Nolsøe í Lágabø úr Vági vann kappingina, og luttekur saman við vinfólki í lýsingini. Brynhild kennir sjálv til avbjóðingarnar av at vera partur av náttarlívinum í aðrari bygd, enn teirri tú býrt í. Tí bygdi hennara hugskot á egnar royndir. Í vinarbólkinum hjá Brynhild hava tey gjørt eina avtalu, ið byggir á tankan: ”Eg koyri teg, tú koyrir meg.” Hetta merkir, at tey skiftast um at koyra: - Avtalan er tann, at um eitt vinfólk er farið í býin og eg liggi heima, so ringja tey til mín, og eg fari upp at koyra tey. Um eg eri farin í býin og okkurt vinfólk liggur heima, so koma tey eisini upp at koyra meg. Tað er líkamikið um tað er morgun, dagur ella nátt, greiddi Brynhild frá í lýsingarfilminum, ið er komin burtur úr hugskotinum hjá Brynhild. Vit valdu at gera eina hugskotskapping, har ung fólk sluppu at seta dagsskránna, og úrslitið gjørdist hesin filmurin, ið byggir á tey hugskot, ið tey ungu sjálvi høvdu, sigur Lovisa Petersen Glerfoss, stjóri í Ráðnum fyri Ferðslutrygd. Eftir at vinnarin varð funnin, hevur Brynhild arbeitt saman við eini lýsingarstovu við at menna hugskotið til eina lidna lýsing. Í lýsingini síggja vit Brynhild og hennara vinfólk í býnum og á veg til hús. Í samráð við Brynhild er lýsingin blivin jalig og uppbyggjandi, heldur enn fordømandi og neilig. Hugburðurin til rúskoyring er broyttur munandi seinastu nógvu árini, og heili 98% av føroyingum siga at rúskoyring verður ikki góðtikin. Men kortini verða bilførarar javnan tiknir við promillu í blóðinum. Harafturat er rúskoyring orsøk til fjórðu hvørja deyðsvanlukku í ferðsluni, vísa tøl úr norðurlondum. Tí er tað eisini í 2021 týdningarmikið at tosa um at steðga rúskoyring! Átakið heldur fram hetta til nýggjárs og løgreglan ger rúskanningar, meðan átakið er. Eisini fer løgreglan at lata bilførarum, sum hava síni viðurskifti í ordan, snøggar lyklaringar við boðskapinum \"Eg koyri teg, tú koyrir meg\". ",
  "label": "positive"
}
```

```json
{
  "text": "Vestmanna skúli hevur hesar leiðreglur í sambandi við sjúkar næmingar: Tað er ógvuliga umráðandi at næmingar, sum ikki eru koppsettir, og hava verið í samband við fólk, sum eru testað positiv fyri koronu, halda tilmælini. ",
  "label": "neutral"
}
```

```json
{
  "text": "Landsverk arbeiður í løtuni við at fáa trailaran, sum er fult lastaður, upp aftur, og arbeiðið fer væntandi at taka nakrar tímar, tí stórar maskinur skulu til, og tær mugu koyra um Eiðiskarð fyri at koma til hjálpar. ",
  "label": "negative"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Her eru nakrir tekstir flokkaðir eftir lyndi, sum kann vera 'positivt', 'neutralt' ella 'negativt'.
  ```

- Base prompt template:

  ```text
  Text: {text}
  Lyndi: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Tekstur: {text}

  Flokka lyndið í tekstinum. Svara við 'positivt', 'neutralt' ella 'negativt'.
  ```

- Label mapping:
  - `positive` ➡️ `positivt`
  - `neutral` ➡️ `neutralt`
  - `negative` ➡️ `negativt`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset fosent
```

## Named Entity Recognition

### FoNE

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.74/)
and is based on news articles from [Sosialurin](http://www.sosialurin.fo/). The named
entities were automatically tagged, but verified manually. They use a superset of the
CoNNL-2003 dataset, with the following additional entity types: `Date`, `Money`,
`Percent` and `Time`. We remove these additional entity types from our dataset and keep
only the original CoNNL-2003 entity types (`PER`, `ORG`, `LOC`, `MISC`).

The original full dataset consists of 6,286 samples, which we split into 1,024 / 256 /
2,048 samples for training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  'tokens': array(['Millum', 'teirra', 'er', 'Tommy', 'Petersen', ',', 'sum', 'eitt', 'skifti', 'hevði', 'ES', 'sum', 'sítt', 'málsøki', 'í', 'Tinganesi', '.'], dtype=object),
  'labels': array(['O', 'O', 'O', 'B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'O', 'O', 'O', 'O', 'B-LOC', 'O'], dtype=object)
}
```

```json
{
  'tokens': array(['Fleiri', 'læraratímar', 'skulu', 'í', 'ár', 'brúkast', 'á', 'HF', '-', 'skúlanum', 'í', 'Klaksvík', ',', 'men', 'sambært', 'leiðaranum', 'á', 'skúlanum', 'hevur', 'tað', 'bara', 'við', 'sær', ',', 'at', 'lærarar', ',', 'sum', 'eru', 'búsitandi', 'í', 'Klaksvík', ',', 'koma', 'at', 'ferðast', 'minni', 'á', 'Kambsdal', 'og', 'ístaðin', 'brúka', 'meira', 'undirvísingartíð', 'í', 'býnum', '.'], dtype=object),
  'labels': array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
```

```json
{
  'tokens': array(['Soleiðis', ',', 'at', 'Starvsstovan', 'kann', 'fylgja', 'við', ',', 'at', 'tað', 'ikki', 'er', 'nýliga', 'heilivágsviðgjørdur', 'fiskur', ',', 'sum', 'tikin', 'verður', '.'], dtype=object),
  'labels': array(['O', 'O', 'O', 'B-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Her eru nakrir setningar og nakrar JSON orðabøkur við nevndar eindir, sum eru í setningunum.
  ```

- Base prompt template:

  ```text
  Setningur: {text}
  Nevndar eindir: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Setningur: {text}

  Greinið nevndu einingarnar í setningunni. Þú ættir að skila þessu sem JSON orðabók með lyklunum 'persónur', 'staður', 'felagsskapur' og 'ymiskt'. Gildin ættu að vera listi yfir nevndu einingarnar af þeirri gerð, nákvæmlega eins og þær koma fram í setningunni.
  ```

- Label mapping:
  - `B-PER` ➡️ `persónur`
  - `I-PER` ➡️ `persónur`
  - `B-LOC` ➡️ `staður`
  - `I-LOC` ➡️ `staður`
  - `B-ORG` ➡️ `felagsskapur`
  - `I-ORG` ➡️ `felagsskapur`
  - `B-MISC` ➡️ `ymiskt`
  - `I-MISC` ➡️ `ymiskt`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset fone
```

### Unofficial: WikiANN-fo

This dataset was part of the WikiANN dataset (also known as PAN-X), published in [this
paper](https://aclanthology.org/P17-1178/). It is based on Wikipedia articles, and the
labels have been automatically annotated using knowledge base mining. There are no
`MISC` entities in this dataset, so we only keep the `PER`, `LOC` and `ORG` entities.

The original full dataset consists of an unknown amount of samples, which we split into
1,024 / 256 / 2,048 samples for training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  'tokens': array(["'", "''", 'Pólland', "''", "'"], dtype=object),
  'labels': array(['O', 'O', 'B-LOC', 'O', 'O'], dtype=object)
}
```

```json
{
  'tokens': array(['Skulu', 'úrvalssvimjararnir', 'betra', 'úrslit', 'síni', ',', 'so', 'er', 'neyðugt', 'hjá', 'teimum', 'at', 'fara', 'uttanlands', 'at', 'venja', '(', 'Danmark', ',', 'USA', ')', ';', 'hinvegin', 'minkar', 'hetta', 'um', 'kappingina', 'hjá', 'teimum', 'heimligu', 'svimjarunum', '.'], dtype=object),
  'labels': array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
```

```json
{
  'tokens': array(['Norðuramerika', '-', '16', '%'], dtype=object),
  'labels': array(['B-LOC', 'O', 'O', 'O'], dtype=object)
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Her eru nakrir setningar og nakrar JSON orðabøkur við nevndar eindir, sum eru í setningunum.
  ```

- Base prompt template:

  ```text
  Setningur: {text}
  Nevndar eindir: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Setningur: {text}

  Greinið nevndu einingarnar í setningunni. Þú ættir að skila þessu sem JSON orðabók með lyklunum 'persónur', 'staður', 'felagsskapur' og 'ymiskt'. Gildin ættu að vera listi yfir nevndu einingarnar af þeirri gerð, nákvæmlega eins og þær koma fram í setningunni.
  ```

- Label mapping:
  - `B-PER` ➡️ `persónur`
  - `I-PER` ➡️ `persónur`
  - `B-LOC` ➡️ `staður`
  - `I-LOC` ➡️ `staður`
  - `B-ORG` ➡️ `felagsskapur`
  - `I-ORG` ➡️ `felagsskapur`
  - `B-MISC` ➡️ `ymiskt`
  - `I-MISC` ➡️ `ymiskt`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset wikiann-fo
```

## Linguistic Acceptability

### ScaLA-fo

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Faroese Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Faroese-FarPaHC) by assuming that
the documents in the treebank are correct, and corrupting the samples to create
grammatically incorrect samples. The corruptions were done by either removing a word
from a sentence, or by swapping two neighbouring words in a sentence. To ensure that
this does indeed break the grammaticality of the sentence, a set of rules were used on
the part-of-speech tags of the words in the sentence.

The original dataset consists of 1,621 samples, from which we use 1,024 / 256 / 1,024
samples for training, validation and testing, respectively (so 3,328 samples used in
total). These splits are used as-is in the framework.

Here are a few examples from the training split:

```json
{
  "text": "Hann talaði tí í samkomuhúsinum við Jödarnar og við teir, sum óttaðust Guð, og á torginum hvönn dag við teir, sum hann har hitti við.",
  "label": "correct"
}
```

```json
{
  "text": "Hann finnur fyrst bróður sín, Símun, og sigur við hann: \"hava Vit funnið Messias\" sum er tað sama sum Kristus; tað er: salvaður.",
  "label": "incorrect"
}
```

```json
{
  "text": "Hetta hendi tríggjar ferðir, og alt fyri eitt varð luturin tikin upp aftur himmals til.",
  "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Hetta eru nakrir setningar og um teir eru mállæruliga rættir.
  ```

- Base prompt template:

  ```text
  Setningur: {text}
  Mállæruliga rættur: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Setningur: {text}

  Greinið hvort setningurin er mállæruliga rættur ella ikki. Svarið skal vera 'ja' um setningurin er rættur og 'nei' um hann ikki er.
  ```

- Label mapping:
  - `correct` ➡️ `ja`
  - `incorrect` ➡️ `nei`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-fo
```

## Reading Comprehension

### FoQA

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2502.07642)
and is based on the Faroese Wikipedia. The questions and answers were automatically
generated using GPT-4-turbo, which were verified by a native speaker, and some of them
were also corrected by the same native speaker.

The original full dataset consists of 2,000 samples, and we split these into 848 / 128 /
1,024 samples for training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "context": "Felagsskapur ST fyri undirvísing, vísindum og mentan (á enskum: United Nations Educational, Scientific and Cultural Organization, stytt UNESCO) er ein serstovnur undir Sameindu Tjóðum, stovnaður í 1946. Endamálið við felagskapinum er at menna útbúgving, gransking og mentan og at fremja samstarv millum tey 195 limalondini og teir 8 atlimirnar, ið eru Føroyar, Curaçao, Aruba, Jomfrúoyggjar, Caymanoyggjar, Makao, Niðurlendsku Antillurnar og Tokelau. Føroyar fingu atlimaskap í 2009 . Atlimaskapur gevur øll tey somu rættindi sum limaskapur. Limalondini skipa seg við hvør síni UNESCO nevnd. Fyrsta føroyska UNESCO nevndin varð skipað í mai 2012. \n\nUNESCO tekur sær millum annað av at meta um, hvørji pláss í heiminum skulu fáa status sum World Heritage Sites (heimsarvur). Limalond UNESCO samtyktu í 1972 millumtjóðasáttmálan um at verja heimsins mentanar- og náttúruarv. Orsøkin er vandin fyri, at náttúruøki, fornfrøðilig minnismerki og mentanarvirði forfarast orsakað av ferðafólkavinnu, dálking, kríggi ella vanligari órøkt.\n\nHygg eisini at \n\n Millumtjóðasáttmáli UNESCO um vernd av heimsins mentanar- og náttúruarvi.\n\nKeldur\n\nSlóðir úteftir \n\n UNESCO World Heritage Centre\n\nST\nHeimsarvar",
  "question": "Hvat góðkendu UNESCO-limalondini í 1972?",
  "answers": {
    "answer_start": array([806]),
    "text": array(["millumtjóðasáttmálan um at verja heimsins mentanar- og náttúruarv"], dtype=object)
  }
}
```

```json
{
  "context": "Levi Niclasen, sum yrkjari betri kendur sum Óðin Ódn (føddur 1. mai 1943 á Tvøroyri, uppvaksin í Hvalba) er ein føroyskur rithøvundur, tónleikari, lærari og politikari. \n\nAftan á barnaskúlan arbeiddi hann í kolinum í Hvalba. Í 1957 stovnaði hann saman við brøðum sínum ein tónleikabólk, og brátt blivu teir kendir sum Hvalbiarbrøðurnir. Teir góvu út tvær stak plátur í 1962. Hann var í Grønlandi 1960 og 1961 og arbeiddi á landi í Føroyingahavnini fyri Nordafar. \nHann fór síðan á læraraskúla í Havn og tók prógv frá Føroya Læraraskúla í 1967. Var settur sum lærari við Hvalbiar skúla 1. august 1967. Hevur verið skúlaleiðari við Hvalbiar skúla frá 1. august 1979. Hann hevur eisini verið á Fróðskaparsetri Føroya og fullført nám í føroyskum og bókmentum 1969-70. Hann hevur útgivið fleiri yrkingasøvn og eisini eitt stuttsøgusavn og eina bók við bæði yrkingum og stuttsøgum. Hann hevur eisini týtt tvær bøkur til føroyskt.\n\nÚtgávur  \nGivið út á egnum forlagi:\nHvirlur (yrkingasavn) 1970\nEg eri í iva (yrkingasavn) 1970 \nTey í urðini (søgusavn) 1973 \nReyðibarmur (yrkingar og stuttsøgur) 1974\nViðrák og Mótrák (yrkingasavn) 1975\nÓttast ikki (yrkingasavn) 1975\nNívandi niða (yrkingasavn) 1983 \nLovað er lygnin (yrkingasavn) 1983 \nEg eigi eina mynd (yrkingasavn) 1987\n\nTýðingar \nEydnuríki prinsurin (Oscar Wilde) (Føroya Lærarafelag 1977). \nHeilaga landið (Pär Lagerkvist) (felagið Varðin 1986).\n\nFamilja \nForeldur: Thomasia Niclasen, f. Thomasen á Giljanesi í Vágum og Hentzar Niclasen, kongsbóndi á Hamri í Hvalba. Giftist í 1971 við Súsonnu Niclasen, f. Holm. Hon er fødd í Hvalba í 1950. Tey eiga tríggjar synir: Tórarinn, Tóroddur og Njálur.\n\nKeldur \n\nFøroyskir týðarar\nFøroyskir rithøvundar\nFøroyskir yrkjarar\nFøroyskir lærarar\nHvalbingar\nFøðingar í 1943",
  "question": "Hvar var Levi Niclasen settur í starv í Grønlandi í 1961?",
  "answers": {
    "answer_start": array([431]),
    "text": array(["Føroyingahavnini"], dtype=object)
  }
}
```

```json
{
  "context": "Giro d'Italia (á føroyskum Kring Italia) er ein av teimum trimum stóru teinasúkklukappingunum og verður hildin hvørt ár í mai/juni og varir í 3 vikur. Kappingin fer fram í Italia, men partar av kappigini kunnu eisini fara fram í onkrum ørðum landi í Evropa, t.d. byrjaði Giro d'Italia í Niðurlondum í 2016 og í Danmark í 2014.\n\nGiro d'Italia varð fyrstu ferð hildið í 1909, har ið tilsamans 8 teinar á 2448\xa0km vóru súkklaðir. Kappingin er saman við Tour de France og Vuelta a España ein av teimum trimum klassisku teinakappingunum, har Tour de France tó er tann mest týðandi.\n\nHar tann fremsti súkklarin í Tour de France er kendur fyri at súkkla í gulari troyggju, so súkklar fremsti súkklarin í Giro d´Italia í ljósareyðari troyggju, á italskum nevnd Maglia rosa. Tann fremsti fjallasúkklarin súkklar í grønari troyggju (Maglia Verde), meðan súkklarin við flestum stigum koyrir í lilla (Maglia ciclimano). Í 2007 varð tann hvíta ungdómstroyggjan innførd aftur, eftir at hon hevði verið burturi í nøkur ár, hon nevnist Maglia Bianca.\n\nTríggir súkklarar hava vunnið kappingina fimm ferðir: Alfredo Binda, Fausto Coppi og Eddy Merckx. Italiumaðurin Felice Gimondi hevur staðið á sigurspallinum níggju ferðir, har hann tríggjar ferðir hevur vunnið, tvær ferðir á øðrum plássi og fýra ferðir á triðjaplássi.\n\nYvirlit yvir vinnarar\n\nByrjan í øðrum londum\n\nKeldur \n\nGiro d'Italia",
  "question": "Hvør hevur fimm ferðir vunnið Giro d'Italia?",
  "answers": {
    "answer_start": array([1089]),
    "text": array(["Alfredo Binda, Fausto Coppi og Eddy Merckx"], dtype=object)
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Hetta eru tekstir saman við spurningum og svar.
  ```

- Base prompt template:

  ```text
  Tekstur: {text}
  Spurningur: {question}
  Svara við í mesta lagi trimum orðum: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Tekstur: {text}

  Svara hesum spurninginum um tekstin uppiyvir við í mesta lagi trimum orðum.

  Spurningur: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset foqa
```

### Unofficial: MultiWikiQA-fo

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
    "context": 'Ali Babba- og 49 aðrar blaðgreinir er eitt savn við fimmti greinum, ið Høgni Mohr hevur skrivað og latið prentað í Dimmalætting og Vinnuvitan frá desember 2004 til februar 2006.\n\nSøgugongd \nGreinasavnið snýr seg um fólk, sum búgva í Føroyum, og onnur, ið hava tilknýti til hetta landið, men búgva uttanlands. Tekstirnir hava sum innihald trý eyðkend sløg av menniskjum: tey ávísu ókendu, sum standa aftan fyri tey kendu; onnur, ið eru mitt í einum serliga spennandi starvi; og hini, ið virka fremst í vinnulívinum. Savnið er sostatt grundað á tríggjar greinarøðir, ið júst eru greiddar úr hondum eftir hesum trimum leistum.\n\nLes eisini \nMohr, Høgni (2010) Tá deyðin verður avdúkaður. Øgiliga egið forlag. ISBN 9789991880518Styrkin í bókini er tann beinrakna tekstin, tær hugtakandi, men knøppu orðingarnar, miðlingin av sterkum menniskjaligum kenslum, stúran, gleði, ótta og sorg, og so tann einfalda, positiva mennsikjafatanin \xa0- Erhard Jacobsen, ummælari.Mohr, Høgni (2017) Fractura nasi. Øgiliga egið forlag. ISBN 9789991880525. Kirsten Brix týtt til danskt 2019. Danskt heiti Rejse for livet. forlag Amanda Books. Seld til filmframleiðslu í 2018.Hon er í passandi flogferð, skrivingin. Floygd, sum eingin annar tekstur eg nýligani havi lisið. Síðst eg kendi meg so væl í felag við hin skrivandi var, tá eg læs Bommhjarta hjá Jóanesi Nielsen, sum kom í fjør. Ein smittandi respektleys søga, sum hemningsleys gongur sínar egnu leiðir. Men aftanfyri hómast ein leitan eftir egnum upphavi. Hví bleiv eg sum eg bleiv, er skuggaspurningur høvundans \xa0- Birgir Kruse, ummælari.Mohr, Høgni (2018) Slepp tær til heiti fani. Øgiliga egið forlag. ISBN 9789991880532. Tekningar: Astrid Andreasen.Tað smakkar bara so væl at lesa hasi orðini. Ikki tí eg havi nakað ímóti Gerhardi ella Javnaðarflokkinum í Avhaldslosjuni, men bara tí at eg síggi spælandi orðalagið, sum ikki er eitt stívrent kvæðaørindi at fáa bókstavarím til skúlabrúks, men beint fram brúksføroyskt loyst úr lagdi \xa0- Birgir Kruse, ummælari.Mohr, Høgni (2019) mær dámar ikki høgna hoydal. Øgiliga egið forlag. ISBN 9789991880549\n\nTýtt og ritstjórnað \n2006 - Askur og Embla (týtt), Bókadeild Føroya lærarafelags, 204 síður.\n\n2013 - Sannleikin um ástarævintýrið (týtt og ritstjórnað), Øgiliga egið forlag, 35 síður.\n\nKeldur',
    "question": 'Hvør er útgávandi av bókini "Mær dámar ikki Høgna Hoydal?"',
    "answers": {
        "answer_start": array([684]),
        "text": array(['Øgiliga egið forlag'], dtype=object)
    }
}
```

```json
{
    "context": 'Ævintýr eru sum skaldskaparslag munnbornar søgur um vanlig folk í einum yvirnatúrligum heimi. Heitið veður nýtt um fleiri sløg av søgum, ið als ikki øll hava sama yivrnatúrliga innihald. Antti Aarne og Stith Thompson hava gjørt eina skrá yvir heimsins ævintýr. Har eru tey skift sundur í 5 høvuðsbólkar ella týpur. Sum annar munnborin skaldskapur hava ævintýrini ongan kendan høvund ella upprunaligan form. Tey kennast aftur eftir greining av søgugongd og innihaldi, og á tann hátt hava Aarne og Thompson skift tey sundur í týpur hvørja við sínum nummari og stavunum AT frammanfyri. Hesar týpur og høvuðsbólkar eru: I Djóraævintýr (AT 1-299), II Eginlig ævintýr (AT 300-1199), III Skemtiævintýr (AT 1200-1999), IV Formilævintýr (AT 2000-2399) og V Ymisk ævintýr (AT 2400.2499). Hesin seinasti bólkurin umfatar tey ævintýr, ið høvundarnir ikki fingu at hóska til hinar bólkarnar. \n\nÍ øllum vanligum brúki verður oftast hugsað um søgurnar í bólki II, tá talan er um ævintýr. Serstakliga kanska undirbólk A, ið verður kallaður Gandaævintýr (AT 300-749). Í hesum bólki eru m.a. tær væl kendu søgurnar um ein fátækan drong, ið bjargar eini prinsessu, sum trøll við níggju høvdum ella onkur onnur yvirnatúrlig vera hevur tikið; í endanum giftist drongurin við prinsessuni og verður kongur. Ella eina fátæka gentu, ið bjargar einum prinsi, sum ofta er umskaptur til okkurt andskræmiligt, og síðani giftist við honum og gerst drotning. Øll liva síðani lukkuliga. \n\nHóast ævintýr sum skaldskaparslag upprunaliga eru munnbornar søgur, kenna vit tey nú í tíðini best og ivaleyst bert úr ritstjórnaðum, prentaðum útgávum. Charles Perrault (1628-1703) var hin fyrsti at geva út eitt savn við søgum, ið eru ritstjórnað ævintýr. Bókin kom í 1697 og nenvdist Søgur og frásagnir úr farnum tíðum við undirheitinum "Gásamóðir sigur frá" (Les Contes de ma Mère l’Oye). Millum søgurnar í hesum savni eru so víðagitnar søgur sum Reyðhetta, Tornarósa og Øskufía. Perrault óttaðist bókmentaliga og mentanarliga smakkin í tíðini, lagaði søgurnar til, sum honum tókti best og gav tær út í navninum á 10 ára gamla syni sínum. Bókin gjørdist ómetaliga væl umtókt og var sum frá leið týdd til flestøll fjølment evropeisk mál. Seinni fóru fólk aðrastaðni at savna og skriva upp ævintýr, og summpart við beinleiðis fyrimynd í søgunum hjá Perrault komu serliga í 19. øld fleiri kend søvn við ritstjórnaðum ævintýrum. Kendast eru ævintýrini hjá týskarunum Jacob og Wilhelm Grimm. Eisini í Norðurlondum vaks áhugin, og millum kendastu útgávur eru tær hjá Ewald Tang Christensen í Danmark, Asbjørnsen og Moe í Noregi, og Jóni Árnasyni í Íslandi. \n\nÍ Føroyum tók Jakob Jakobsen tráðin upp, og í árunum 1898-1901 gav hann út savn sítt við føroyskum sagnum og ævintýrum. Eisini hann ritstjórnaði søgurnar, sum hann savnaði, so vit kunnu siga, at soleiðis sum vit lesa tær hjá honum, hava tær ikki verið sagdar honum. Hansara ritstjórnan er mest av málsligum slag. Hann flytur munnliga frásøgn í skrift við teimum tillagingum, ið tá eru neyðugar, og hartil reinsar hann frásøgnina fyri útlendskan málburð. Mangt bendir á, at ævintýr valla eru gamal skaldskapur í Føroyum. Tað tykist, sum tey eru komin í munnliga frásøgn í Føroyum eftir fólksligum, einahelst donskum útgávum. Men sum væntandi er í munnligari søgulist, hava fólk lagað tey til so við og við, so tey ofta hava føroyskan dám í mongum lutum. Summi teirra eru tó ivaleyst gomul í Føroyum.\n\nKeldur \n\n Kirsten Brix: "Drongurin, ið burturtikin varð av sjótrøllakonginum", Varðanum bd. 59 1992, s. 188-219. \n Jakob Jakobsen: Færøske Folkesagn og Æventyr 1899-1901.\n\nÆvintýr\nFólkaminni',
    "question": 'Hvat var heitið á bókini eftir Charles Perrault?',
    "answers": {
        "answer_start": array([1743]),
        "text": array(['Søgur og frásagnir úr farnum tíðum við undirheitinum "Gásamóðir sigur frá" (Les Contes de ma Mère l’Oye)'], dtype=object)
    }
}
```

```json
{
    "context": 'Trøllakampar (frøðiheiti Asplenium) hoyra til tann bólkin av plantum, ið verður kallaður blómuleysar plantur. Tað finnast 20.000 sløg av trøllakampum í heiminum, og er hetta slagríkasta fylki, aftaná fylkið við blómuplantum, ið telur 250.000 sløg. Flestu sløgini av trøllakampum finnast í tropunum og trívast best har vátt er. Trøllakampar verða mettir at vera "primitivt" plantuslag, ið er nær í ætt við upprunaplanturnar. Teir hava ikki blómur og seta ikki fræ, men nørast við grókornum, ið hjá summum trøllakampum sita í gróhópum aftanfyri á blaðnum, vardir av einum skjøldri, sum opnar seg, tá grókornini eru búgvin, so at tey kunnu spjaðast. Hjá øðrum sita teir á blaðkantinum, sum er rullaður inneftir, so leingi grókornini ikki eru búgvin. \n\nSummi trøllakampasløg hava tvey sløg av bløðum, eitt slag ið er “sterilt” og eitt sum er “fertilt”. Tað “fertila” blaðið kann hjá summum sløgum vera heilt ymiskt frá tí “sterila”. Trøllakampur kann hava grókorn í milliónatali, men bert fáar nýggjar plantur koma burturúr. Bløðini hava ymiskt skap. Tey kunnu verða innskorin eina, tvær og fleiri ferðir ella als ikki innskorin. Við sínum sermerkta vakstrarlagi líkist trøllakampur, áður enn hann er fullvaksin, einum fiólhøvdi ella tí evsta á fiólini.\n\nÚtbreiðsla\n\nTrøllakampar vóru nógv vanligari í Føroyum, áðrenn fólk settu búgv her. Hetta prógva sákornskanningar. Vøksturin í Føroyum er sum heild ávirkaður av seyðabiti, og hevur hann verið tað, síðan fólk settu búgv her. Seyðurin legðist beinanvegin eftir tí fruktagóða gróðri, sum landið var avvaksið við. Hesin gróðurin hvarv eftir stuttari tíð og broyttist til tættbitna gróðurin, sum vit kenna í dag.  Sáðkornskanningar vísa, at trøllakampar sum heild fóru nógv aftur aftan á landnám. Teir eru av elstu plantusløgum á jørð og vuksu her fyri meira enn 300 mió árum síðan. Í koltíðini vuksu trøllakampur, javni og bjølluvísa sum stórir skógir.\n\nIkki allastaðni er seyður sloppin framat at bíta. Tí sæst enn tann mest upprunaligi gróðurin í gjáum og bakkum, har seyður ikki er sloppin framat. Her er gróðurin stórur og fjølbroyttur, og kanningar bera prógv um, at hann hevur verið støðugur í langa tíð av teirri orsøk, at seyður og fólk ikki sluppu framat. Av teimum trøllakampum, ið eru vanligir í Føroyum, eru fyrst og fremst tann stórvaksni trøllakalskampurin, tann heldur fínari mjúki kvennkampurin og dimmgrøni ekstur blóðkampurin. Hesir trøllkampar eru nógv vanligari í londunum sunnan fyri enn norðan fyri okkum.\n\nFleiri sløg av trøllakampum finnast í brattlendi. Lættast er at fáa eyga á tann stórvaksna trøllakallskampin og tann næstan líka stórvaksna mjúka kvennkampin. Sáðkornskanningar hava víst, at útbreiðslan av trøllakampum minkaði ógvuliga nógv, tá ið fólk settu búgv í Føroyum og høvdu húsdjór síni við sær.\n\nFimtan sløg av trøllakampum finnast í Føroyum. Flestu av teimum dámar best at vaksa í klettarivum, har vátt og skuggi er - men eisini í grýtutum lendi, brattlendi og gjáum. Ein tann mest vanligi trøllakampurin í Føroyum er fínur klettakampur, meðan svartur trøllakampur og strálhærdur trøllakampur eru sera sjáldsamir og bert finnast á einum stað. \n\nÍ 2007 varð nýtt trøllakampaslag funnið í brattlendi í Norðuroyggjum. Hetta er tungutrøllakampur (Asplenium scolopendrium). Hesin trøllakampur er eisini sjáldsamur í hinum Norðurlondunum.\n\nKelda\n Stamps.fo\n\nSí eisini\n Plantulívið í Føroyum\n\nPlantur í Føroyum\nPlantur',
    "question": 'Hvussu mong trøllakamps sløg eru til í Føroyum?',
    "answers": {
        "answer_start": array([2782]),
        "text": array(['Fimtan'], dtype=object)
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Hetta eru tekstir saman við spurningum og svar.
  ```

- Base prompt template:

  ```text
  Tekstur: {text}
  Spurningur: {question}
  Svara við í mesta lagi trimum orðum: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Tekstur: {text}

  Svara hesum spurninginum um tekstin uppiyvir við í mesta lagi trimum orðum.

  Spurningur: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-fo
```

## Grammatical Error Detection

### Unofficial: GerLangMod-fo

This dataset is based on the [GerLangMod](https://github.com/noahmanu/gerlangmod)
collection and derived from the Faroese Universal Dependencies treebank. Assuming UD
annotations are accurate and sentences are well-formed, the dataset contains permuted
versions of these UD sentences where half of the verbs have been misplaced within their
phrase boundaries. Noun-headed groups of tokens are treated as impermeable units so
misplaced verbs cannot split them up, and no verb can be placed in the first position of
the first phrase of each sentence to avoid creating correct polar question syntax.

The original dataset consists of 2,809 samples derived from the
[UD_Faroese-FarPaHC](https://github.com/UniversalDependencies/UD_Faroese-FarPaHC) and
[UD_Faroese-OFT](https://github.com/UniversalDependencies/UD_Faroese-OFT) treebanks.
We use a sample of 1,024 / 256 / 2,048 of these for training, validation and testing,
respectively.

Here are a few examples from the training split:

```json
{
    "tokens": [
        "nei",
        "til",
        "tess",
        "eri",
        "eg",
        "komin",
        "at",
        "hesum",
        "tíma"
    ],
    "labels": [
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O"
    ]
}
```

```json
{
    "tokens": [
        "landið",
        "limur",
        "í",
        "bretska",
        "samveldinum",
        "er"
    ],
    "labels": [
        "O",
        "O",
        "O",
        "O",
        "O",
        "B-ERR"
    ]
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Niðanfyri eru setningar og JSON orðabøkur við málvillum, ið eru í givnu setningunni.
  ```

- Base prompt template:

  ```text
  Setning: {text}
  Málvillur: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Setning: {text}

  Kenn aftur málvillurnar í setningunni. Tú skalt prenta hetta sum ein JSON orðabók við lyklinum 'villa'. Virðið skal vera listi yvir rangt sett orð, beint sum tey síggjast í setningunni.
  ```

- Label mapping:
  - `B-ERR` ➡️ `villa`
  - `I-ERR` ➡️ `villa`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset gerlangmod-fo
```

## Hallucination Detection

### Unofficial: MultiWikiHalluQA-fo

This dataset uses the same data as [Unofficial: MultiWikiQA-fo](#unofficial-multiwikiqa-fo), published in
[this paper](https://doi.org/10.48550/arXiv.2509.04111), containing Wikipedia articles
with LLM-generated questions and answers in 300+ languages. Rather than evaluating the
correctness of the generated answer, this task evaluates the degree to which the model
hallucinates, i.e., generates tokens that are not grounded in the provided context.

The hallucination detection is performed using the
[LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect) library, which uses a
transformer-based classifier to predict hallucination at the token level. The metric
reported is the hallucination rate, computed as the ratio of hallucinated tokens to
total tokens in the generated answers.

Here are a few examples from the training split:

```json
{
    "context": 'Ali Babba- og 49 aðrar blaðgreinir er eitt savn við fimmti greinum, ið Høgni Mohr hevur skrivað og latið prentað í Dimmalætting og Vinnuvitan frá desember 2004 til februar 2006.\n\nSøgugongd \nGreinasavnið snýr seg um fólk, sum búgva í Føroyum, og onnur, ið hava tilknýti til hetta landið, men búgva uttanlands. Tekstirnir hava sum innihald trý eyðkend sløg av menniskjum: tey ávísu ókendu, sum standa aftan fyri tey kendu; onnur, ið eru mitt í einum serliga spennandi starvi; og hini, ið virka fremst í vinnulívinum. Savnið er sostatt grundað á tríggjar greinarøðir, ið júst eru greiddar úr hondum eftir hesum trimum leistum.\n\nLes eisini \nMohr, Høgni (2010) Tá deyðin verður avdúkaður. Øgiliga egið forlag. ISBN 9789991880518Styrkin í bókini er tann beinrakna tekstin, tær hugtakandi, men knøppu orðingarnar, miðlingin av sterkum menniskjaligum kenslum, stúran, gleði, ótta og sorg, og so tann einfalda, positiva mennsikjafatanin \xa0- Erhard Jacobsen, ummælari.Mohr, Høgni (2017) Fractura nasi. Øgiliga egið forlag. ISBN 9789991880525. Kirsten Brix týtt til danskt 2019. Danskt heiti Rejse for livet. forlag Amanda Books. Seld til filmframleiðslu í 2018.Hon er í passandi flogferð, skrivingin. Floygd, sum eingin annar tekstur eg nýligani havi lisið. Síðst eg kendi meg so væl í felag við hin skrivandi var, tá eg læs Bommhjarta hjá Jóanesi Nielsen, sum kom í fjør. Ein smittandi respektleys søga, sum hemningsleys gongur sínar egnu leiðir. Men aftanfyri hómast ein leitan eftir egnum upphavi. Hví bleiv eg sum eg bleiv, er skuggaspurningur høvundans \xa0- Birgir Kruse, ummælari.Mohr, Høgni (2018) Slepp tær til heiti fani. Øgiliga egið forlag. ISBN 9789991880532. Tekningar: Astrid Andreasen.Tað smakkar bara so væl at lesa hasi orðini. Ikki tí eg havi nakað ímóti Gerhardi ella Javnaðarflokkinum í Avhaldslosjuni, men bara tí at eg síggi spælandi orðalagið, sum ikki er eitt stívrent kvæðaørindi at fáa bókstavarím til skúlabrúks, men beint fram brúksføroyskt loyst úr lagdi \xa0- Birgir Kruse, ummælari.Mohr, Høgni (2019) mær dámar ikki høgna hoydal. Øgiliga egið forlag. ISBN 9789991880549\n\nTýtt og ritstjórnað \n2006 - Askur og Embla (týtt), Bókadeild Føroya lærarafelags, 204 síður.\n\n2013 - Sannleikin um ástarævintýrið (týtt og ritstjórnað), Øgiliga egið forlag, 35 síður.\n\nKeldur',
    "question": 'Hvør er útgávandi av bókini "Mær dámar ikki Høgna Hoydal?"',
    "answers": {
        "answer_start": array([684]),
        "text": array(['Øgiliga egið forlag'], dtype=object)
    }
}
```

```json
{
    "context": 'Ævintýr eru sum skaldskaparslag munnbornar søgur um vanlig folk í einum yvirnatúrligum heimi. Heitið veður nýtt um fleiri sløg av søgum, ið als ikki øll hava sama yivrnatúrliga innihald. Antti Aarne og Stith Thompson hava gjørt eina skrá yvir heimsins ævintýr. Har eru tey skift sundur í 5 høvuðsbólkar ella týpur. Sum annar munnborin skaldskapur hava ævintýrini ongan kendan høvund ella upprunaligan form. Tey kennast aftur eftir greining av søgugongd og innihaldi, og á tann hátt hava Aarne og Thompson skift tey sundur í týpur hvørja við sínum nummari og stavunum AT frammanfyri. Hesar týpur og høvuðsbólkar eru: I Djóraævintýr (AT 1-299), II Eginlig ævintýr (AT 300-1199), III Skemtiævintýr (AT 1200-1999), IV Formilævintýr (AT 2000-2399) og V Ymisk ævintýr (AT 2400.2499). Hesin seinasti bólkurin umfatar tey ævintýr, ið høvundarnir ikki fingu at hóska til hinar bólkarnar. \n\nÍ øllum vanligum brúki verður oftast hugsað um søgurnar í bólki II, tá talan er um ævintýr. Serstakliga kanska undirbólk A, ið verður kallaður Gandaævintýr (AT 300-749). Í hesum bólki eru m.a. tær væl kendu søgurnar um ein fátækan drong, ið bjargar eini prinsessu, sum trøll við níggju høvdum ella onkur onnur yvirnatúrlig vera hevur tikið; í endanum giftist drongurin við prinsessuni og verður kongur. Ella eina fátæka gentu, ið bjargar einum prinsi, sum ofta er umskaptur til okkurt andskræmiligt, og síðani giftist við honum og gerst drotning. Øll liva síðani lukkuliga. \n\nHóast ævintýr sum skaldskaparslag upprunaliga eru munnbornar søgur, kenna vit tey nú í tíðini best og ivaleyst bert úr ritstjórnaðum, prentaðum útgávum. Charles Perrault (1628-1703) var hin fyrsti at geva út eitt savn við søgum, ið eru ritstjórnað ævintýr. Bókin kom í 1697 og nenvdist Søgur og frásagnir úr farnum tíðum við undirheitinum "Gásamóðir sigur frá" (Les Contes de ma Mère l’Oye). Millum søgurnar í hesum savni eru so víðagitnar søgur sum Reyðhetta, Tornarósa og Øskufía. Perrault óttaðist bókmentaliga og mentanarliga smakkin í tíðini, lagaði søgurnar til, sum honum tókti best og gav tær út í navninum á 10 ára gamla syni sínum. Bókin gjørdist ómetaliga væl umtókt og var sum frá leið týdd til flestøll fjølment evropeisk mál. Seinni fóru fólk aðrastaðni at savna og skriva upp ævintýr, og summpart við beinleiðis fyrimynd í søgunum hjá Perrault komu serliga í 19. øld fleiri kend søvn við ritstjórnaðum ævintýrum. Kendast eru ævintýrini hjá týskarunum Jacob og Wilhelm Grimm. Eisini í Norðurlondum vaks áhugin, og millum kendastu útgávur eru tær hjá Ewald Tang Christensen í Danmark, Asbjørnsen og Moe í Noregi, og Jóni Árnasyni í Íslandi. \n\nÍ Føroyum tók Jakob Jakobsen tráðin upp, og í árunum 1898-1901 gav hann út savn sítt við føroyskum sagnum og ævintýrum. Eisini hann ritstjórnaði søgurnar, sum hann savnaði, so vit kunnu siga, at soleiðis sum vit lesa tær hjá honum, hava tær ikki verið sagdar honum. Hansara ritstjórnan er mest av málsligum slag. Hann flytur munnliga frásøgn í skrift við teimum tillagingum, ið tá eru neyðugar, og hartil reinsar hann frásøgnina fyri útlendskan málburð. Mangt bendir á, at ævintýr valla eru gamal skaldskapur í Føroyum. Tað tykist, sum tey eru komin í munnliga frásøgn í Føroyum eftir fólksligum, einahelst donskum útgávum. Men sum væntandi er í munnligari søgulist, hava fólk lagað tey til so við og við, so tey ofta hava føroyskan dám í mongum lutum. Summi teirra eru tó ivaleyst gomul í Føroyum.\n\nKeldur \n\n Kirsten Brix: "Drongurin, ið burturtikin varð av sjótrøllakonginum", Varðanum bd. 59 1992, s. 188-219. \n Jakob Jakobsen: Færøske Folkesagn og Æventyr 1899-1901.\n\nÆvintýr\nFólkaminni',
    "question": 'Hvat var heitið á bókini eftir Charles Perrault?',
    "answers": {
        "answer_start": array([1743]),
        "text": array(['Søgur og frásagnir úr farnum tíðum við undirheitinum "Gásamóðir sigur frá" (Les Contes de ma Mère l’Oye)'], dtype=object)
    }
}
```

```json
{
    "context": 'Trøllakampar (frøðiheiti Asplenium) hoyra til tann bólkin av plantum, ið verður kallaður blómuleysar plantur. Tað finnast 20.000 sløg av trøllakampum í heiminum, og er hetta slagríkasta fylki, aftaná fylkið við blómuplantum, ið telur 250.000 sløg. Flestu sløgini av trøllakampum finnast í tropunum og trívast best har vátt er. Trøllakampar verða mettir at vera "primitivt" plantuslag, ið er nær í ætt við upprunaplanturnar. Teir hava ikki blómur og seta ikki fræ, men nørast við grókornum, ið hjá summum trøllakampum sita í gróhópum aftanfyri á blaðnum, vardir av einum skjøldri, sum opnar seg, tá grókornini eru búgvin, so at tey kunnu spjaðast. Hjá øðrum sita teir á blaðkantinum, sum er rullaður inneftir, so leingi grókornini ikki eru búgvin. \n\nSummi trøllakampasløg hava tvey sløg av bløðum, eitt slag ið er “sterilt” og eitt sum er “fertilt”. Tað “fertila” blaðið kann hjá summum sløgum vera heilt ymiskt frá tí “sterila”. Trøllakampur kann hava grókorn í milliónatali, men bert fáar nýggjar plantur koma burturúr. Bløðini hava ymiskt skap. Tey kunnu verða innskorin eina, tvær og fleiri ferðir ella als ikki innskorin. Við sínum sermerkta vakstrarlagi líkist trøllakampur, áður enn hann er fullvaksin, einum fiólhøvdi ella tí evsta á fiólini.\n\nÚtbreiðsla\n\nTrøllakampar vóru nógv vanligari í Føroyum, áðrenn fólk settu búgv her. Hetta prógva sákornskanningar. Vøksturin í Føroyum er sum heild ávirkaður av seyðabiti, og hevur hann verið tað, síðan fólk settu búgv her. Seyðurin legðist beinanvegin eftir tí fruktagóða gróðri, sum landið var avvaksið við. Hesin gróðurin hvarv eftir stuttari tíð og broyttist til tættbitna gróðurin, sum vit kenna í dag.  Sáðkornskanningar vísa, at trøllakampar sum heild fóru nógv aftur aftan á landnám. Teir eru av elstu plantusløgum á jørð og vuksu her fyri meira enn 300 mió árum síðan. Í koltíðini vuksu trøllakampur, javni og bjølluvísa sum stórir skógir.\n\nIkki allastaðni er seyður sloppin framat at bíta. Tí sæst enn tann mest upprunaligi gróðurin í gjáum og bakkum, har seyður ikki er sloppin framat. Her er gróðurin stórur og fjølbroyttur, og kanningar bera prógv um, at hann hevur verið støðugur í langa tíð av teirri orsøk, at seyður og fólk ikki sluppu framat. Av teimum trøllakampum, ið eru vanligir í Føroyum, eru fyrst og fremst tann stórvaksni trøllakalskampurin, tann heldur fínari mjúki kvennkampurin og dimmgrøni ekstur blóðkampurin. Hesir trøllkampar eru nógv vanligari í londunum sunnan fyri enn norðan fyri okkum.\n\nFleiri sløg av trøllakampum finnast í brattlendi. Lættast er at fáa eyga á tann stórvaksna trøllakallskampin og tann næstan líka stórvaksna mjúka kvennkampin. Sáðkornskanningar hava víst, at útbreiðslan av trøllakampum minkaði ógvuliga nógv, tá ið fólk settu búgv í Føroyum og høvdu húsdjór síni við sær.\n\nFimtan sløg av trøllakampum finnast í Føroyum. Flestu av teimum dámar best at vaksa í klettarivum, har vátt og skuggi er - men eisini í grýtutum lendi, brattlendi og gjáum. Ein tann mest vanligi trøllakampurin í Føroyum er fínur klettakampur, meðan svartur trøllakampur og strálhærdur trøllakampur eru sera sjáldsamir og bert finnast á einum stað. \n\nÍ 2007 varð nýtt trøllakampaslag funnið í brattlendi í Norðuroyggjum. Hetta er tungutrøllakampur (Asplenium scolopendrium). Hesin trøllakampur er eisini sjáldsamur í hinum Norðurlondunum.\n\nKelda\n Stamps.fo\n\nSí eisini\n Plantulívið í Føroyum\n\nPlantur í Føroyum\nPlantur',
    "question": 'Hvussu mong trøllakamps sløg eru til í Føroyum?',
    "answers": {
        "answer_start": array([2782]),
        "text": array(['Fimtan'], dtype=object)
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Hetta eru tekstir saman við spurningum og svar.
  ```

- Base prompt template:

  ```text
  Tekstur: {text}
  Spurningur: {question}
  Svara við í mesta lagi trimum orðum: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Tekstur: {text}

  Svara hesum spurninginum um tekstin uppiyvir við í mesta lagi trimum orðum.

  Spurningur: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-hallucination-qa-fo
```
