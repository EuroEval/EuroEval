# 🇱🇹 Lithuanian

This is an overview of all the datasets used in the Lithuanian part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### Lithuanian Emotions

This dataset is a combination of machine translated versions of the [GoEmotions
dataset](https://doi.org/10.48550/arXiv.2005.00547) and the [Kaggle emotions
dataset](https://www.kaggle.com/datasets/nelgiriyewithana/emotions). GoEmotions consists
of English Reddit comments and the Kaggle dataset contains English Twitter messages.
Both datasets have been machine translated to Lithuanian.

The original dataset contains 377k / 47.1k / 5.43k / 41.7k samples for the combined
training, combined validation, Lithuanian GoEmotions test, and Lithuanian Twitter
emotions test splits, respectively. We use 1,024 / 256 / 2,048 samples for our training,
validation and test splits, respectively. Our splits are based on the original splits.

Here are a few examples from the training split:

```json
{
  "text": "Aš jaučiuosi taip nekantrus, kad turiu laukti daugiau nei mėnesį ir tuo pačiu labai stengiuosi nelinkėti to laiko",
  "label": "positive"
}
```

```json
{
  "text": "Jaučiuosi gana bendras šeimininkas Toros",
  "label": "negative"
}
```

```json
{
  "text": "Florida, jis gavo du",
  "label": "neutral"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Toliau pateikti dokumentai ir jų nuotaika,
  kuri gali būti 'teigiamas', 'neutralus' arba 'neigiamas'.
  ```

- Base prompt template:

  ```text
  Dokumentas: {text}
  Nuotaika: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Dokumentas: {text}

  Klasifikuokite nuotaiką dokumente. Atsakykite su 'teigiamas', 'neutralus' arba 'neigiamas', ir nieko kito.
  ```

- Label mapping:
  - `positive` ➡️ `teigiamas`
  - `neutral` ➡️ `neutralus`
  - `negative` ➡️ `neigiamas`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset lithuanian-emotions
```

## Named Entity Recognition

### WikiANN-lt

This dataset was published in [this paper](https://doi.org/10.18653/v1/P17-1178) and is
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
  "tokens": array(["'", "''", 'Michael', 'Schumacher', "''", "'"], dtype=object),
  "labels": ["O", "O", "B-PER", "I-PER", "O", "O"]
}
```

```json
{
  "tokens": array(['Keliu', 'sujungtas', 'su', 'Alta', '.'], dtype=object),
  "labels": ["O", "O", "O", "B-LOC", "O"]
}
```

```json
{
  "tokens": array(['Amazonės', 'lamantinas', '(', "''Trichechus", 'inunguis', "''",
       ')'], dtype=object),
  "labels": ["B-LOC", "I-LOC", "O", "O", "O", "O", "O"]
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Toliau pateikti sakiniai ir JSON žodynai su vardiniais vienetais, kurie pateikiame sakinyje.
  ```

- Base prompt template:

  ```text
  Sakinys: {text}
  Vardiniai vienetai: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Sakinys: {text}

  Identifikuokite vardinius vienetus sakinyje. Turėtumėte pateikti tai kaip JSON žodyną su raktais 'asmuo', 'vieta', 'organizacija' ir 'kita'. Reikšmės turi būti to tipo vardinių vienetų sąrašai, tiksliai taip, kaip jie rodomi sakinyje.
  ```

- Label mapping:
  - `B-PER` ➡️ `asmuo`
  - `I-PER` ➡️ `asmuo`
  - `B-LOC` ➡️ `vieta`
  - `I-LOC` ➡️ `vieta`
  - `B-ORG` ➡️ `organizacija`
  - `I-ORG` ➡️ `organizacija`
  - `B-MISC` ➡️ `kita`
  - `I-MISC` ➡️ `kita`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset wikiann-lt
```

## Linguistic Acceptability

### ScaLA-lt

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Lithuanian Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Lithuanian-ALKSNIS) by assuming
that the documents in the treebank are correct, and corrupting the samples to create
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
  "text": "Be to, tai, kad turi man neįprastų drabužių, primena, jog ir daugiau man nežinomo gyvenimo.",
  "label": "incorrect"
}
```

```json
{
  "text": "Juos sukelia kokia nors konkreti organinė ir šiuo atveju galvos skausmas yra tik tam tikros ligos simptomas.",
  "label": "incorrect"
}
```

```json
{
  "text": "Juos sukelia kokia nors konkreti organinė ir šiuo atveju galvos skausmas yra tik tam tikros ligos simptomas.",
  "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Toliau pateikti sakiniai ir ar jie yra gramatiškai teisingi.
  ```

- Base prompt template:

  ```text
  Sakinys: {text}
  Gramatiškai teisingas: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Sakinys: {text}

  Nustatykite, ar sakinys yra gramatiškai teisingas, ar ne. Atsakykite su 'taip' arba 'ne', ir nieko kito.
  ```

- Label mapping:
  - `correct` ➡️ `taip`
  - `incorrect` ➡️ `ne`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-lt
```

## Reading Comprehension

### MultiWikiQA-lt

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
    "context": "Gadas Elmalė (; ; g. 1971\xa0m. balandžio 19 d.)\xa0– Maroko ir Prancūzijos komikas ir aktorius. Jo naujausias spektaklis vadinasi „Sans tambour“ (pažodž. „be būgno“, kas yra žodžių žaismas tarp prancūzų kalbos žodžių tabou ('tabū') ir tambour ('būgnas')). Jis vaidino daugelyje filmų, įskaitant „Coco“, „Hors de prix“, „La doublure“ ir „Midnight in Paris“.\n\nAnkstyvasis gyvenimas ir šeima \nG. Elmalė gimė Kasablankoje, Maroke. Jis turi sefardų žydų kraujo. Augo mišrių kultūrų apsuptyje, kur buvo kalbama arabiškai, hebrajiškai ir prancūziškai. Tėvo vardas Davidas, motinos\xa0– Reginė (mergautinė pavardė Aymard). Turi brolį Arié, kuris yra aktorius ir dainininkas, ir seserį Judith, kuri yra artistė ir režisierio padėjėja.\n\nKarjera \n\nG. Elmalė lankė Kasablankos licėjų. Vėliau jis ketverius metus studijavo politologiją Monrealyje, o kiek vėliau persikėlė į Paryžių, kur jis studijavo dramą. Pirmą savo vaidmenį teatre G. Elmalė atliko monospektaklyje „Décalages“ 1997\xa0m. Šis spektaklis buvo autobiografinis. Pirmosios G. Elmalė sąsajos su kinu buvo Merzak Allouache filme „Salut cousin“. Vėliau jis vaidino filmuose „L’homme est une femme comme les autres“ ir „Train de vie“. G. Elmalė šlovė augo su antro monospektaklio „La vie normale“ pasisekimu ir su filmu „La vérité si je mens 2“ (2000\xa0m.), kuriame jis atliko vaidmenį Dov.\n\nAsmeninis gyvenimas \nG. Elmalė gyveno su prancūze artiste Anne Brochet nuo 1998\xa0m. iki 2002\xa0m. Jie susilaukė sūnaus Noé. Nuo 2009\xa0m. iki 2010\xa0m. jo antroji pusė buvo prancūzė žurnalistė Marie Drucker. Nuo 2011\xa0m. aktorius gyvena su Charlotte Casiraghi. Judviejų sūnus Raphaël gimė 2013\xa0m. gruodžio 17 d.\n\nIšnašos \n\nPrancūzijos aktoriai\nMaroko asmenybės",
    "question": "Su kuo gyveno G. Elmalė nuo 2011 metų?",
    "answers": {"answer_start": [1559], "text": ["Charlotte Casiraghi"]}
}
```

```json
{
    "context": "Bus kraujo () – 2007 m. JAV epinės dramos filmas, kurio scenarijaus autorius ir režisierius Paul Thomas Anderson. Filmas dalinai remiasi Upton Sinclair romano „Oil!“ motyvais. Pasakojama apie auksakasį, kuris XIX a. pab. – XX a. pr Pietų Kalifornijoje kilusio naftos bumo metu nusprendė užsiimti naftos gavyba ir taip pralobti iš šio verslo.\n\nSiužetas \n\n1902-ieji. Aukso ieškotojas Danielis Pleinvju (akt. Day-Lewis) atranda naftos klodą ir įkuria nedidelę naftos gavybos įmonę. Vieno nelaimingo atsitikimo metu žuvus jo darbininkui, Pleinvju įsivaikina jo sūnų. Berniukas, vardu H.V., tampa jo formaliu verslo „partneriu“.\n\nPo devynerių metu Pleinvju sutinka Polą Sandėjų (akt. Dano), kuris jam prasitaria apie naftos klodą, esantį po jo žeme. Pleinvju mėgina nupirkti sklypą už nusiderėtą kainą, bet Polo brolis dvynys Elis, žinodamas apie jo ketinimus, primygtinai pareikalauja $5 000, už kuriuos būtų pastatyta vietinė bažnyčia, kurios pastoriumi taptų Elis. Tačiau Pleinvju įtikina Elio tėvą sudaryti sandorį už nusiderėtą kainą. Vėliau avarijos metu įvykęs sprogimas pažeidžia H.V. klausą.\n\nVieną dieną, Pleinvju aplanko vyriškis, teigiantis esąs jo pusiau brolis Henris. Pleinvju jį priima, nors jo istorijoje ir randa spragų. Vėliau berniukas pabando nužudyti Henrį padegdamas jo antklodę. Pasipiktinęs sūnaus poelgiu, Pleinvju išsiunčia berniuką į mokyklą San Franciske. Įmonės „Standard Oil“ atstovas pasisiūlo nupirkti Pleinvju žemę, bet Pleinvju sudaro sutartį su „Union Oil“ ir nutiesia vamzdyną į Kalifornijos pakrantę. Pleinvju kyla įtarimas dėl Henrio ir šiam papasakojus tikrąją istoriją, Pleinvju jį nužudo ir užkasa lavoną.\n\n1927-ieji. H.V. jau suaugęs ir vedęs. Jis susitinka su tėvu, kuris ne tik, kad tapo turtingu, bet ir įniko į alkoholį, ir paprašo jo nutraukti judviejų sutartį, kad jis galėtų įsteigti savo verslą. Pleinvju išjuokia jo kurtumą ir papasakoja jam apie jo kilmę, ir H.V. išvyksta.\n\nElis aplanko Pleinvju ir pasiūlo jam dar kartą įsigyti dalį jo žemės, kuri priklausė ponui Bendžiui. Pleinvju atskleidžia, kad jis jau seniausiai išgavo visą naftą iš jo nuosavybės per aplinkinius naftos gręžinius. Elis vis tiek paprašo sumokėti, bet Pleinvju įniršta ir užmuša jį boulingo kėgliu.\n\nApdovanojimai \n Oskarų apdovanojimai: geriausias aktorius (Daniel Day-Lewis), geriausia kinematografija (Robert Elswit)\n BAFTA: geriausias pirmo plano aktorius (Daniel Day-Lewis)\n Auksiniai gaubliai: geriausias draminio filmo aktorius (Daniel Day-Lewis)\n Ekrano aktorių gildijos apdovanojimai: geriausias aktorius (Daniel Day-Lewis)\n\nIšnašos \n\n2007 filmai\nJAV filmai\nDramos\nEpiniai filmai",
    "question": "Kokią kompaniją įsteigė Danielis Pleinvju?",
    "answers": {"answer_start": [448], "text": ["nedidelę naftos gavybos įmonę"]}
}
```

```json
{
    "context": "Ero ežeras (, arabanų k. Kati Thanda)\xa0– ežeras centrinėje Australijoje, Pietų Australijos valstijoje, didžiausias visame žemyne.\n\nEro ežeras yra žemiausiame Australijos taške\xa0– jo dugnas yra 15\xa0m žemiau jūros lygio. Dvi ežero dalys\xa0– šiaurinis Ero ežeras ir pietinis Ero ežeras\xa0– kartu užima apie 9 600\xa0km² plotą. Šiuos ežerus jungia 15\xa0km ilgio Godjero sąsiauris. Paviršiaus altitudė\xa0– 9,5 metro. Ero ežero baseino plotas apie 1,3 mln. km².\n\nPavadinimą ežerui davė Edvardas Eras (Edward Eyre), kuris 1839\xa0m. išvyko iš Adelaidės norėdamas tapti pirmuoju europiečiu, kirtusiu Australiją iš pietų į šiaurę. Įveikęs Flinderso kalnagūbrį jis susidūrė su neįveikiama sūrių ežerų juosta ir buvo priverstas grįžti atgal. Po keleto metų Eras dar kartą išvyko į kelionę ir pasiekė ežerą, kuris buvo pavadintas jo vardu. Aborigenai arabanai ežerą vadino Kati Thanda.\n\nUpėms pripildžius Ero ežerą, jo pakrantėse įsikuria didžiulė pelikanų kolonija ir susuka dešimtis tūkstančių lizdų. Tam, kad čia patektų, šie paukščiai įveikia didžiulius atstumus skrisdami virš kaitrios dykumos. Vandens srautui nutrūkus, didelėje kaitroje ežeras greitai garuoja ir tampa dar sūresnis.\n\nDidžiąją metų dalį Ero ežeras\xa0– uždruskėjusi pelkė, vandens prisipildo tik vasarą. Ilgą laiką tyrinėtojai manė, kad Ero ežeras\xa0– didžiulis gėlo vandens ežeras. Šiandien jau aišku, kad Ero ežeras gali būti didžiulėmis gėlo vandens platybėmis\xa0– tačiau vos kartą per aštuonerius ar dešimt metų. Šis ciklas jau kartojasi apie 20 tūkstančių metų. Smarkūs lietūs dvi vasaras iš eilės\xa0– retas įvykis šiame regione: pirmųjų metų lietus susigeria į žemę, antraisiais metais žemė sugeria mažiau vandens, jis atiteka į Ero ežerą iš kalnų ir jį pripildo.\n\nAplink ežerą įsteigtas Ero ežero nacionalinis parkas.\n\nŠaltiniai\n\nNuorodos \n Ero ežeras: Pelikanų rojus  \n Ero ežeras: Jachtklubas \n \nAustralijos ežerai\nPietų Australija",
    "question": "Koks vandens kelias sujungia šiaurinę ir pietinę Ero ežero dalis?",
    "answers": {"answer_start": [346], "text": ["Godjero sąsiauris"]}
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Toliau pateikti tekstai su atitinkamais klausimais ir atsakymais.
  ```

- Base prompt template:

  ```text
  Tekstas: {text}
  Klausimas: {question}
  Atsakykite ne daugiau kaip 3 žodžiais: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Tekstas: {text}

  Atsakykite į šį klausimą apie aukščiau pateiktą tekstą ne daugiau kaip 3 žodžiais.

  Klausimas: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-lt
```

## Knowledge

### LT-History

This dataset was published in [this paper](https://aclanthology.org/2025.nbreal-1.1/),
and consists of Lithuanian national and general history questions and answers.

The dataset originally contains 593 samples, which are reduced to 559 after filtering.
We use a 64 / 32 / 463 split for training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Abiejų Tautų Respublikos Ketverių metų seimo nutarimu:\nPasirinkimai:\na. valstiečiams suteikta asmens laisvė.\nb. bajorai atleisti nuo valstybinių mokesčių;\nc. miestiečiams leista užimti valstybines tarnybas;\nd. įteisinta absoliuti monarcho valdžia;",
  "label": "c"
}
```

```json
{
  "text": "Kurioje eilutėje visos išvardytos asmenybės gyveno Renesanso epochoje?\nPasirinkimai:\na. Vaskas da Gama, Maksimiljenas Robespjeras, Johanas Gutenbergas.\nb. Nikola Makiavelis, Šarlis Monteskjė, Kristupas Kolumbas.\nc. Mikalojus Kopernikas, Ferdinandas Magelanas, Leonardas da Vinčis.\nd. Johanas Gutenbergas, Žanas Žakas Ruso, Leonardas da Vinčis.",
  "label": "c"
}
```

```json
{
  "text": "Lietuvos teritorija suskirstyta į 10 apskričių: Vilniaus, Kauno, Klaipėdos, Šiaulių, Panevėžio, Alytaus ir...\nPasirinkimai:\na. Tauragės, Utenos, Marijampolės ir Telšių;\nb. Tauragės, Trakų, Kėdainių ir Plungės;\nc. Utenos, Marijampolės, Šalčininkų ir Telšių.\nd. Marijampolės, Telšių, Ukmergės ir Neringos;",
  "label": "a"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Toliau pateikti daugiavariančiai klausimai (su atsakymais).
  ```

- Base prompt template:

  ```text
  Klausimas: {text}
  Pasirinkimai:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Atsakymas: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Klausimas: {text}
  Pasirinkimai:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Atsakykite į aukščiau pateiktą klausimą atsakydami 'a', 'b', 'c' arba 'd', ir nieko daugiau.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset lt-history
```

## Common-sense Reasoning

### Winogrande-lt

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2506.19468)
and is a translated and filtered version of the English [Winogrande
dataset](https://doi.org/10.1145/3474381).

The original full dataset consists of 47 / 1,210 samples for training and testing, and
we use 128 of the test samples for validation, resulting in a 47 / 128 / 1,085 split for
training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Derrick negalėjo susikoncentruoti darbe, skirtingai nei Justin, nes _ turėjo smagų darbą. Ką reiškia tuščia vieta _?\nPasirinkimai:\na. Derrick\nb. Justin",
  "label": "b"
}
```

```json
{
  "text": "Vieną kartą Lenkijoje Dennis mėgavosi kelione labiau nei Jason, nes _ turėjo paviršutinišką lenkų kalbos supratimą. Ką reiškia tuščia vieta _?\nPasirinkimai:\na. Dennis\nb. Jason",
  "label": "b"
}
```

```json
{
  "text": "Natalie mano, kad smaragdai yra gražūs brangakmeniai, bet Betty taip nemano. _ nusipirko vėrinį su dideliu smaragdu. Ką reiškia tuščia vieta _?\nPasirinkimai:\na. Natalie\nb. Betty",
  "label": "a"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Toliau pateikti daugiavariančiai klausimai (su atsakymais).
  ```

- Base prompt template:

  ```text
  Klausimas: {text}
  Pasirinkimai:
  a. {option_a}
  b. {option_b}
  Atsakymas: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Klausimas: {text}
  Pasirinkimai:
  a. {option_a}
  b. {option_b}

  Atsakykite į aukščiau pateiktą klausimą atsakydami 'a' arba 'b', ir nieko daugiau.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset winogrande-lt
```

## Summarisation

### Lrytas

This dataset contains news articles and their corresponding summaries from the Lithuanian
public media news portal [Lrytas.lt](https://www.lrytas.lt/).

Samples were collected using the
[lrytas_scraper](https://github.com/alexandrainst/lrytas). We use 1,024 / 256 / 2,048
samples for training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "2025 m. „Tesla Model Y Performance“ testas: superautomobilis visai šeimai Karščiausia „Tesla“ naujiena – jau Lietuvoje\n\nŠiemet nuo 2022 metų gaminamas „Model Y“ buvo atnaujintas, o šių metų pradžioje aš bandžiau modelį iki atnaujinimo. Testą rasite čia: Vizualiai jis nemažai skiriasi nuo ankstesnės versijos – vientisa šviesų juosta priekyje, kitokie sujungti žibintai gale. Salone priekinės sėdynės dabar ne tik šildomos, bet ir vėdinamos, buvo sumontuotas papildomas ekranas gale sėdintiems keleiviams, pagerinta apdailos medžiagų kokybė, pagerintos važiavimo savybės. Įdomu tai, kad atnaujinto bazinio modelio kaina nepadidėjo ir prasideda nuo 45 970 eurų – tai yra automobilis su mažesnės 60 kWh talpos akumuliatoriumi. O modelis su 82 kWh akumuliatoriumi (įkraunama 79 kWh, gamintojo nurodomas nuvažiuojamas atstumas – 622 kilometrai) ir galiniais varančiaisiais ratais atsieis vis dar įmones tenkinančius 49 970 eurų (taikoma 4 metų garantija, papildomai galima įsigyti pratęstą „Mango Insurance“ garantiją, o pasinaudojus nuoroda po vaizdo apžvalga, bus suteikta papildoma 10 proc. nuolaida). Tačiau čia – „Performance“ modelis, kurio du elektros motorai pasiekia net 461 kilovatą arba 627 arklio galias ir 741 Nm – tai leidžia šiam nemažam ir erdviam (JAV galima ir septynių sėdimų vietų versija) elektromobiliui iki 100 km/h įsibėgėti vos per 3,5 sekundes, maksimalus greitis siekia 250 km/h, o kaina yra 62 970 eurų – už modelį iki atnaujinimo gamintojas prašė 4000 eurų mažiau. Bet ką ten ta kaina, kai iš po kojos veržiasi tokia elektros galia – išbandžiau beveik visus, išskyrus maksimalų greitį, dinaminius parametrus. Ir turiu pasakyti, kad kol kas nematau kitų elektromobilių, kurie savo kainos, praktiškumo ir dinamikos santykiu prilygtų šiam modeliui. Nebent išskyrus vis daugiau mūsų šalyje atsirandančių elektromobilių iš Kinijos – atrodo, kad bent jau kol kas „Tesla“ yra vienintelė atsvara šioms mašinoms iš Rytų. Tad nieko keisto, kad tai – ir populiariausias elektromobilis tiek Lietuvoje, tiek ir visame pasaulyje. O dar labai patiko „Performance“ modelyje komplektuojama nustatomo standumo pakaba – galima važiuoti tiek komfortiškai, oriai linguojant per nelygumus, tiek ir gerokai standžiau brėžiant posūkius, bet be dantis kratančio kietumo – puikiai suderintos savybės tarp komforto ir sportiškumo. Ir šis 4,8 m ilgio elektromobilis su 2,9 m atstumu tarp ašių puikiai tinka azartiškam važiavimui ne tik tiesiai – žemas svorio centras, standus kėbulas ir, svarbiausia, dėl kėbului daug naudoto lengvo aliuminio sąlyginai kaip elektromobiliui nedidelis svoris (tai yra sunkiausia „Model Y“ versija, sverianti 2108 kilogramus), jis pasižymi tiek geru valdymu (žinoma, žemesnis tos pačios platformos „Model 3“ būtų smagesnis), tiek ir jau ne kartą įtikinusiomis mažomis energijos sąnaudomis – net ir po aktyvaus važiavimo, jos buvo tik 18,3 kWh/100 km. Daugiau apie naują „Tesla Model Y Performance“ žiūrėkite vaizdo medžiagoje: automobilio bandymas elektromobilis Pakaba Rodyti daugiau žymių",
  "target_text": "Vos prieš kelias savaites Europoje pristatyta galingiausia 2025 metų „Tesla Model Y“ versija „Performance“ labai greitai pasiekė ir Lietuvą, tad pasinaudojau galimybe vienas pirmųjų sėsti prie jo vairo."
}
```

```json
{
  "text": "Žalgirietis Ą. Tubelis prieš debiutą Eurolygoje: „Neperspaudžiu savęs“\n\n„Tikrai laukiu, tačiau per daug neperspaudžiu savęs ir stengiuosi žiūrėti į tai, kaip į dar vienas rungtynes. Nekeisiu savo rutinos, neplanuoju daryti kažko naujo – tiesiog ruošiamės su komanda, atiduodame visą save, dirbame per treniruotes, tad manau, kad to ir reikia pergalei pasiekti – susikaupimo ir juodo darbo“, – apie asmeninį ir komandos nusiteikimą kalbėjo aukštaūgis. Paklaustas apie padidėjusį fiziškumą, Ą.Tubelis neslėpė, kad kartais atsilaikyti nėra lengva. „Stiprūs kūnai, kartais būna sunku, bet gavau patirties tiek vasarą grumdamasis su Eurolygos ir NBA žaidėjais, tiek ir Turkijoje dar spėjau sužaisti prieš „Anadolu Efes“, tai skirtumas tikrai jaučiasi, kartais sunku pastumti, bet tuomet reikia naudoti savo protą ir bandyti tuos kūnus apeiti“, – perėjimą į aukštesnį lygį įvertino lietuvis. Ą.Tubelis pabrėžė ir LKL susitikimų svarbą prieš Eurolygos sezono startą. „LKL rungtynės tikrai padėjo – ne tik man asmeniškai, bet ir susilipdyti visai komandai. Taip pat naudingos ir treniruotės, kuriose gal labai daug naujų dalykų ir neišmokstame, tačiau pastoviai taisome savo klaidas, žiūrime labai daug video medžiagos. Poros treniruočių dar gal ir reiktų, bet manau, kad viskas labai gerai juda į priekį“, – teigiamai apie pasiruošimą sezonui atsiliepė aukštaūgis. Ą.Tubelis kalbėjo ir apie žaidimo planą prieš „Monaco“ ekipą, tad kviečiame įsijungti pilną video ir išgirsti „Žalgirio“ naujoko mintis apie svarbiausią ateinančio mačo žaidimo aspektą. Po šių rungtynių, jau penktadienį žalgiriečiai žais pirmąsias Eurolygos naujojo sezono namų rungtynes, kuriose sausakimšoje arenoje į kovą stos prieš Šarūno Jasikevičiaus Stambulo „Fenerbahce“. Kitos Eurolygos namų rungtynės laukia spalio 16-ąją prieš Milano „Olimpia“. Ąžuolas Tubelis Kauno Žalgiris Eurolyga",
  "target_text": "Kauno „Žalgirio“ krepšininkai jau trečiadienį išvykoje pradės naująjį Eurolygos sezoną susitikime su praėjusio sezono vicečempionų komanda iš Monako. Po vienos iš paskutinių žalgiriečių treniruočių prieš Vassilio Spanoulio auklėtinių iššūkį, mintimis pasidalijo ir Ąžuolas Tubelis, kurio laukia Eurolygos debiutas."
}
```

```json
{
  "text": "Iš „aušriečių“ – žinia dėl naujo kultūros ministro\n\n„Šiai dienai ji (Kultūros ministerija – ELTA) priklauso „Nemuno aušrai“ ir mes ieškosime kito kandidato, kuris ir mums tiktų, ir visuomenei tiktų, ir Vyriausybei, ir visiems žmonėms“, – žurnalistams Seime sakė „Nemuno aušros“ pirmininko pavaduotojas Robertas Puchovičius. „Prioritetas yra partinis (kandidatas – ELTA), bet tikrai svarstysime visus variantus“, – pridūrė jis. Politiko teigimu, artimiausiu metu šį klausimą svarstys partijos organai. „Bus sušaukta mūsų valdyba, taryba ir ieškosime sprendimo. Šiai dienai Kultūros ministerija priklauso „Nemuno aušrai“ ir niekas nesikeičia“, – sakė jis. Adomavičius galėtų užimti kitas pareigas Kultūros ministerijoje Tuo metu kalbėdamas apie I. Adomavičiaus ateitį, R. Puchovičius patikino, kad jis galėtų užimti viceministro, kanclerio ar ministro patarėjo pareigas Kultūros ministerijoje. „Galės grįžti į savo senas pareigas – būti Raimundo Šukio patarėju, gal jam kitas pareigas pasiūlys, kol kas neskubėkime dalinti pareigų“, – tikino „aušrietis“. „Jis parodė, kad jis tikrai yra geras ministras, darbuotojas. Jam tikrai rūpėjo kultūra, jis tikrai galėtų užimti pareigas ir tęsti darbą. Matyčiau jį kiekvienoje pozicijoje, tačiau svarbiausia, kuris ateis ministras ir jis jau matys“, – akcentavo politikas. Pasak jo, I. Adomavičius nusprendė trauktis ne dėl pastarųjų pasisakymų apie Krymą. „Čia nebuvo nei Krymo klausimas, nei kiti dalykai. Matėme, kad buvo labai didelis spaudimas, labai daug puolama. Turbūt žmogus atsikelia su spaudimu, eina miegoti su spaudimu. Jis yra ministras, bet jis yra ir vyras, ir tėvas, turi šeimą, reikia grįžti namo ir tuos nervus atlaikyti“, – kalbėjo R. Puchovičius. „Ignotas padarė tikrai didelį darbą, matome, kad jis gali toliau tęsti savo karjerą politikoje ir matome, kad jis turi didelį pasitikėjimą visuomenėje“, – tikino jis. Kaip skelbta anksčiau, I. Adomavičius sakė, jog klausimai apie tai, kam priklauso Rusijos aneksuotas Krymas, yra provokuojantys. Interviu naujienų portalui „Lrytas“ politikas tvirtino nenorįs kalbėti šiais klausimais. Tačiau netrukus po interviu ministras savo poziciją patikslino. Tačiau atsakyti, kaip įsivaizduoja ir ką jam reiškia Ukrainos pergalė, politikas vengė atsakyti. Ministerijos atsisakyti nesvarsto Apie galimą ministerijos atidavimą kol kas nesvarstoTiesa, kultūros bendruomenei piktinantis, kad ministeriją nuspręsta patikėti „Nemuno aušrai“, R. Puchovičius sako, kad kol kas „aušriečiai“ nesvarsto apie galima jos perdavimą koalicijos partneriams. „Kalbėti visada galime, derėtis visada galime, tačiau kol kas galime daryti tik sąmokslo teoriją. Šiai dienai ji (ministerija – ELTA) priklauso „Nemuno aušrai“ ir mes ieškosime kito kandidato, kuris ir mums tiktų, ir visuomenei tiktų“, – kalbėjo politikas. „Yra ta visuomenė, kuri nepalaiko. Su ja laukia sunkus darbas, kad pakeistume jų nuomonę. Bet yra ir kita pusė, kuri mus palaiko. Mes negalime nusileisti vienai pusei nepabandžius to klausimo išspręsti ir pavesti kitą pusę, kuri mus palaiko. Ieškosime sprendimo, kad visi būtų patenkinti“, – pabrėžė jis. ELTA primena, kad po skandalą sukėlusio pasisakymo apie Krymą „aušriečių“ deleguotas kultūros ministras I. Adomavičius pranešė, kad traukiasi iš pareigų. Apie tai, kad I. Adomavičius turėtų trauktis, penktadienį užsiminė ir premjerė Inga Ruginienė. Šalies vadovui paskyrus „aušrietį“ Ignotą Adomavičių naujuoju kultūros ministru, kilo pasipiktinimo banga – keliami klausimai dėl politiko kompetentingumo, sugebėjimų, nevienareikšmiškų pasisakymų. Netrukus dėmesio centre atsidūrė politiko pasisakymai apie Krymą, Ukrainą, klausimų kėlė kuriam laikui iš ministerijos patalpų dingusios Ukrainos vėliavos. Savo ruožtu praėjusią savaitę Simono Daukanto aikštėje prie Prezidentūros buvo surengtas protestas – bene tūkstantis susirinkusių kultūros sektoriaus atstovų reikalavo, kad Kultūros ministerija neliktų „Nemuno aušros“ rankose. Kultūrininkai taip pat platina ir peticiją – ją jau pasirašė per 67 tūkst. žmonių. Sekmadienį, spalio 5 d., menininkai rengia įspėjamąjį streiką „Tai gali būti paskutinis kartas“. Robert Puchovič Nemuno aušra Ignotas Adomavičius Rodyti daugiau žymių",
  "target_text": "Ignotui Adomavičiui nusprendus trauktis iš kultūros ministro pareigų, politiką delegavusi „Nemuno aušra“ ieškos naujo kandidato šioms pareigoms."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:

  ```text
  Žemiau pateikiami dokumentai su pridėtomis santraukomis.
  ```

- Base prompt template:

  ```text
  Dokumentas: {text}
  Santrauka: {target_text}
  ```

- Instruction-tuned prompt template:

  ```text
  Dokumentas: {text}

  Parašykite aukščiau pateikto dokumento santrauką.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset lrytas
```
