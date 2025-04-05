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
  ```
  Her eru nakrir tekstir flokkaðir eftir lyndi, sum kann vera 'positivt', 'neutralt' ella 'negativt'.
  ```
- Base prompt template:
  ```
  Text: {text}
  Lyndi: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Tekstur: {text}

  Flokka lyndið í tekstinum. Svara við 'positivt', 'neutralt' ella 'negativt'.
  ```
- Label mapping:
    - `positive` ➡️ `positivt`
    - `neutral` ➡️ `neutralt`
    - `negative` ➡️ `negativt`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset fosent
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
  ```
  Her eru nakrir setningar og nakrar JSON orðabøkur við nevndar eindir, sum eru í setningunum.
  ```
- Base prompt template:
  ```
  Setningur: {text}
  Nevndar eindir: {label}
  ```
- Instruction-tuned prompt template:
  ```
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
$ euroeval --model <model-id> --dataset fone
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
  ```
  Her eru nakrir setningar og nakrar JSON orðabøkur við nevndar eindir, sum eru í setningunum.
  ```
- Base prompt template:
  ```
  Setningur: {text}
  Nevndar eindir: {label}
  ```
- Instruction-tuned prompt template:
  ```
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
$ euroeval --model <model-id> --dataset wikiann-fo
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

The original dataset consists of 1,621 samples, from which we use 1,024 / 256 / 1,024 samples for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
used as-is in the framework.

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
  ```
  Hetta eru nakrir setningar og um teir eru mállæruliga rættir.
  ```
- Base prompt template:
  ```
  Setningur: {text}
  Mállæruliga rættur: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Setningur: {text}

  Greinið hvort setningurin er mállæruliga rættur ella ikki. Svarið skal vera 'ja' um setningurin er rættur og 'nei' um hann ikki er.
  ```
- Label mapping:
    - `correct` ➡️ `ja`
    - `incorrect` ➡️ `nei`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset scala-fo
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
  'context': 'Felagsskapur ST fyri undirvísing, vísindum og mentan (á enskum: United Nations Educational, Scientific and Cultural Organization, stytt UNESCO) er ein serstovnur undir Sameindu Tjóðum, stovnaður í 1946. Endamálið við felagskapinum er at menna útbúgving, gransking og mentan og at fremja samstarv millum tey 195 limalondini og teir 8 atlimirnar, ið eru Føroyar, Curaçao, Aruba, Jomfrúoyggjar, Caymanoyggjar, Makao, Niðurlendsku Antillurnar og Tokelau. Føroyar fingu atlimaskap í 2009 . Atlimaskapur gevur øll tey somu rættindi sum limaskapur. Limalondini skipa seg við hvør síni UNESCO nevnd. Fyrsta føroyska UNESCO nevndin varð skipað í mai 2012. \n\nUNESCO tekur sær millum annað av at meta um, hvørji pláss í heiminum skulu fáa status sum World Heritage Sites (heimsarvur). Limalond UNESCO samtyktu í 1972 millumtjóðasáttmálan um at verja heimsins mentanar- og náttúruarv. Orsøkin er vandin fyri, at náttúruøki, fornfrøðilig minnismerki og mentanarvirði forfarast orsakað av ferðafólkavinnu, dálking, kríggi ella vanligari órøkt.\n\nHygg eisini at \n\n Millumtjóðasáttmáli UNESCO um vernd av heimsins mentanar- og náttúruarvi.\n\nKeldur\n\nSlóðir úteftir \n\n UNESCO World Heritage Centre\n\nST\nHeimsarvar',
  'question': 'Hvat góðkendu UNESCO-limalondini í 1972?',
  'answers': {
    'answer_start': array([806]),
    'text': array(['millumtjóðasáttmálan um at verja heimsins mentanar- og náttúruarv'], dtype=object)
  }
}
```
```json
{
  'context': 'Levi Niclasen, sum yrkjari betri kendur sum Óðin Ódn (føddur 1. mai 1943 á Tvøroyri, uppvaksin í Hvalba) er ein føroyskur rithøvundur, tónleikari, lærari og politikari. \n\nAftan á barnaskúlan arbeiddi hann í kolinum í Hvalba. Í 1957 stovnaði hann saman við brøðum sínum ein tónleikabólk, og brátt blivu teir kendir sum Hvalbiarbrøðurnir. Teir góvu út tvær stak plátur í 1962. Hann var í Grønlandi 1960 og 1961 og arbeiddi á landi í Føroyingahavnini fyri Nordafar. \nHann fór síðan á læraraskúla í Havn og tók prógv frá Føroya Læraraskúla í 1967. Var settur sum lærari við Hvalbiar skúla 1. august 1967. Hevur verið skúlaleiðari við Hvalbiar skúla frá 1. august 1979. Hann hevur eisini verið á Fróðskaparsetri Føroya og fullført nám í føroyskum og bókmentum 1969-70. Hann hevur útgivið fleiri yrkingasøvn og eisini eitt stuttsøgusavn og eina bók við bæði yrkingum og stuttsøgum. Hann hevur eisini týtt tvær bøkur til føroyskt.\n\nÚtgávur  \nGivið út á egnum forlagi:\nHvirlur (yrkingasavn) 1970\nEg eri í iva (yrkingasavn) 1970 \nTey í urðini (søgusavn) 1973 \nReyðibarmur (yrkingar og stuttsøgur) 1974\nViðrák og Mótrák (yrkingasavn) 1975\nÓttast ikki (yrkingasavn) 1975\nNívandi niða (yrkingasavn) 1983 \nLovað er lygnin (yrkingasavn) 1983 \nEg eigi eina mynd (yrkingasavn) 1987\n\nTýðingar \nEydnuríki prinsurin (Oscar Wilde) (Føroya Lærarafelag 1977). \nHeilaga landið (Pär Lagerkvist) (felagið Varðin 1986).\n\nFamilja \nForeldur: Thomasia Niclasen, f. Thomasen á Giljanesi í Vágum og Hentzar Niclasen, kongsbóndi á Hamri í Hvalba. Giftist í 1971 við Súsonnu Niclasen, f. Holm. Hon er fødd í Hvalba í 1950. Tey eiga tríggjar synir: Tórarinn, Tóroddur og Njálur.\n\nKeldur \n\nFøroyskir týðarar\nFøroyskir rithøvundar\nFøroyskir yrkjarar\nFøroyskir lærarar\nHvalbingar\nFøðingar í 1943',
  'question': 'Hvar var Levi Niclasen settur í starv í Grønlandi í 1961?',
  'answers': {
    'answer_start': array([431]),
    'text': array(['Føroyingahavnini'], dtype=object)
  }
}
```
```json
{
  'context': "Giro d'Italia (á føroyskum Kring Italia) er ein av teimum trimum stóru teinasúkklukappingunum og verður hildin hvørt ár í mai/juni og varir í 3 vikur. Kappingin fer fram í Italia, men partar av kappigini kunnu eisini fara fram í onkrum ørðum landi í Evropa, t.d. byrjaði Giro d'Italia í Niðurlondum í 2016 og í Danmark í 2014.\n\nGiro d'Italia varð fyrstu ferð hildið í 1909, har ið tilsamans 8 teinar á 2448\xa0km vóru súkklaðir. Kappingin er saman við Tour de France og Vuelta a España ein av teimum trimum klassisku teinakappingunum, har Tour de France tó er tann mest týðandi.\n\nHar tann fremsti súkklarin í Tour de France er kendur fyri at súkkla í gulari troyggju, so súkklar fremsti súkklarin í Giro d´Italia í ljósareyðari troyggju, á italskum nevnd Maglia rosa. Tann fremsti fjallasúkklarin súkklar í grønari troyggju (Maglia Verde), meðan súkklarin við flestum stigum koyrir í lilla (Maglia ciclimano). Í 2007 varð tann hvíta ungdómstroyggjan innførd aftur, eftir at hon hevði verið burturi í nøkur ár, hon nevnist Maglia Bianca.\n\nTríggir súkklarar hava vunnið kappingina fimm ferðir: Alfredo Binda, Fausto Coppi og Eddy Merckx. Italiumaðurin Felice Gimondi hevur staðið á sigurspallinum níggju ferðir, har hann tríggjar ferðir hevur vunnið, tvær ferðir á øðrum plássi og fýra ferðir á triðjaplássi.\n\nYvirlit yvir vinnarar\n\nByrjan í øðrum londum\n\nKeldur \n\nGiro d'Italia",
  'question': "Hvør hevur fimm ferðir vunnið Giro d'Italia?",
  'answers': {
    'answer_start': array([1089]),
    'text': array(['Alfredo Binda, Fausto Coppi og Eddy Merckx'], dtype=object)
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:
  ```
  Hetta eru tekstir saman við spurningum og svar.
  ```
- Base prompt template:
  ```
  Tekstur: {text}
  Spurningur: {question}
  Svara við í mesta lagi trimum orðum: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Tekstur: {text}

  Svara hesum spurninginum um tekstin uppiyvir við í mesta lagi trimum orðum.

  Spurningur: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset foqa
```
