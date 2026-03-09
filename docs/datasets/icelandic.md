# 🇮🇸 Icelandic

This is an overview of all the datasets used in the Icelandic part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### Hotter and Colder Sentiment

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2502.16987),
and consists of texts from Icelandic blog post, annotated with sentiment labels (and
many others) via a crowdsourcing platform.

The original full dataset consists of 2,901 samples, and we use a 1,021 / 255 / 1,607
split for training, validation and testing, respectively (so all samples are used in
total).

Here are a few examples from the training split:

```json
{
  "text": "Til hamingju með gott framtak. Þetta eru góðir útgangspunktar með stjórnarskrána, þó margt fleira þurfi að laga svo hún þjóni vel  nýju lýðveldi framtíðarinnar.Ég styð heils hugar þetta framtak ykkar.",
  "label": "positive"
}
```

```json
{
  "text": "Jú, jú, auðvita á hann ekki að vera samstarfsmaður eða einu sinni í sama húsi og sérstakir ríkissaksóknarar í þessu máli. Sérstakir ríkissaksóknarar fyrir þetta mál eiga að liggja liggja beint undir ráðuneytinu og vera algerlega sjálfstæðir, \"untouchables\". Ég hef ekki enn séð nein rök fyrir því að Valtýr þurfi að víkja úr sínu starfi ef þessi leið verður valin? Best væri ef sérstakir ríkissaksóknarar í þessu máli væri þrepinu hærri í valdastiganum en Valtýr, ef það er hægt að koma því í gegn með snöggum lagabreytingum? Varla er þetta Stjórnarskrármál?",
  "label": "neutral"
}
```

```json
{
  "text": "Meira að segja hörðustu klappstýrur Þórólfs hljóta að hugsa, þó ekki væri í nema augnablik: Mikið er skrýtið að hann sé ekki með á hreinu af hverju fáir handleggir eru að bjóða sig í þriðju sprautuna!Annars er bara sama handritið að fara spilast aftur: Nú er haustið komið og árstíðarbundnar pestir munu rjúka upp, allar sem ein, og þá verður skellt í lás og talað um að hafa opnað of snemma.",
  "label": "negative"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Hér fyrir neðan eru textabrot ásamt lyndisgildi þeirra sem getur verið 'jákvætt', 'hlutlaust' eða 'neikvætt'.
  ```

- Base prompt template:

  ```text
  Textabrot: {text}
  Lyndi: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Texti: {text}

  Flokkaðu tilfinninguna í textanum. Svaraðu með 'jákvætt', 'hlutlaust' eða 'neikvætt'.
  ```

- Label mapping:
  - `positive` ➡️ `jákvætt`
  - `neutral` ➡️ `hlutlaust`
  - `negative` ➡️ `neikvætt`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset hotter-and-colder-sentiment
```

## Named Entity Recognition

### MIM-GOLD-NER

This dataset was published in [this
paper](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/230) and is
based on the [Tagged Icelandic Corpus (MIM)](https://clarin.is/en/resources/mim/), which
consists of Icelandic books, news articles, periodicals, parliament speeches, legal
texts, adjudications and government websites. It has been annotated with named entities
in a semi-automated fashion, where each labels has been manually verified. The entity
types in the dataset is a superset of the CoNLL-2003 tags, with the following additional
labels: `DATE`, `TIME`, `MONEY`, `PERCENT`. These labels have been removed.

The original full dataset consists of 1,000,000 tokens. We use a 1,024 / 256 / 2,048
split for training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  'tokens': array(['Sjálfsagt', 'er', 'að', 'miða', 'endurgreiðsluna', 'verði', 'núverandi', 'heimild', 'framlengd', 'við', 'EUROIII', 'í', 'stað', 'EUROII', 'eins', 'og', 'nú', 'er', '.'], dtype=object),
  'labels': array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'O', 'O', 'B-MISC', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
```

```json
{
  'tokens': array(['Það', 'var', 'bróðir', 'Sandlers', 'sem', 'hvatti', 'hann', 'til', 'að', 'leggja', 'grínið', 'fyrir', 'sig', 'þegar', 'hann', 'var', '17', 'ára', 'að', 'aldri', '.'], dtype=object),
  'labels': array(['O', 'O', 'O', 'B-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
```

```json
{
  'tokens': array(['2.-', 'Erla', 'Guðný', 'Gylfad.', ',', 'Smyrill', 'frá', 'Stokkhólma', ',', '7,01', '.'], dtype=object),
  'labels': array(['O', 'B-PER', 'I-PER', 'I-PER', 'O', 'B-PER', 'O', 'B-LOC', 'O', 'O', 'O'], dtype=object)
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Eftirfarandi eru setningar ásamt JSON lyklum með nefndum einingum sem koma fyrir í setningunum.
  ```

- Base prompt template:

  ```text
  Setning: {text}
  Nafneiningar: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Setning: {text}

  Greindu nefndu einingarnar í setningunni. Þú ættir að skila þessu sem JSON orðabók með lyklunum 'einstaklingur', 'staðsetning', 'stofnun' og 'ýmislegt'. Gildin ættu að vera listi yfir nefndu einingarnar af þeirri gerð, nákvæmlega eins og þær koma fram í setningunni.
  ```

- Label mapping:
  - `B-PER` ➡️ `einstaklingur`
  - `I-PER` ➡️ `einstaklingur`
  - `B-LOC` ➡️ `staðsetning`
  - `I-LOC` ➡️ `staðsetning`
  - `B-ORG` ➡️ `stofnun`
  - `I-ORG` ➡️ `stofnun`
  - `B-MISC` ➡️ `ýmislegt`
  - `I-MISC` ➡️ `ýmislegt`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset mim-gold-ner
```

## Linguistic Acceptability

### ScaLA-is

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Icelandic Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Icelandic-Modern) by assuming that
the documents in the treebank are correct, and corrupting the samples to create
grammatically incorrect samples. The corruptions were done by either removing a word
from a sentence, or by swapping two neighbouring words in a sentence. To ensure that
this does indeed break the grammaticality of the sentence, a set of rules were used on
the part-of-speech tags of the words in the sentence.

The original dataset consists of 3,535 samples, from which we use 1,024 / 256 / 2,048
samples for training, validation and testing, respectively (so 3,328 samples used in
total). These splits are used as-is in the framework.

Here are a few examples from the training split:

```json
{
  "text": "Utanrrh.: Ég hef Ég hefði óskað þess að hæstv. utanríkisráðherra hefði meiri áhrif á forsætisráðherra en raun ber vitni Gripið fram í. því að hann er sem betur fer ekki að tala niður þá atvinnugrein sem tengist sjávarútveginum eins og hæstv. forsætisráðherra gerir alla jafna.",
  "label": "correct"
}
```

```json
{
  "text": "Það væri mun skárra, það hefði verið hægt að gera það meiri með sátt, en það var einfaldlega ekki gert.",
  "label": "incorrect"
}
```

```json
{
  "text": "Mig líka að koma að, ég gleymdi því áðan og kom því heldur ekki að, komugjöldunum eins og þau heita víst núna, ekki legugjöld lengur.",
  "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Hér fyrir neðan eru setningar ásamt mati á því hvort þær eru málfræðilega réttar.
  ```

- Base prompt template:

  ```text
  Setning: {text}
  Málfræðilega rétt: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Setning: {text}

  Greindu hvort setningin er málfræðilega rétt. Svaraðu með 'já' ef setningin er rétt og 'nei' ef hún er það ekki.
  ```

- Label mapping:
  - `correct` ➡️ `já`
  - `incorrect` ➡️ `nei`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-is
```

### Unofficial: IceEC

This dataset was published [here](https://github.com/antonkarl/iceErrorCorpus) and
consists of texts in modern Icelandic from student essays, online news texts and
Wikipedia articles, annotated for mistakes related to spelling, grammar, and other
issues.

The original full dataset consists of 58,200 / 5,270 samples for training and testing,
respectively. We use a 1,024 / 256 / 2,048 split for training, validation and testing,
where the training and testing splits are subsets of the original training and testing
splits, and the validation split is a disjoint subset of the training split.

Here are a few examples from the training split:

```json
{
  "text": "Kannski erum við með meiri sölu í öðrum skrokkhlutum en síðum t.d., “ segir Steinþór.",
  "label": "correct"
}
```

```json
{
  "text": "Þó svo að hann sé leiðinlegur og ekkert tívolí gaman, þá er miðlar hann þekkingu til okkar og án hans mundi enginn menntun vera.",
  "label": "incorrect"
}
```

```json
{
  "text": "Síminn er hvers manns ábyrgð.",
  "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Hér fyrir neðan eru setningar ásamt mati á því hvort þær eru málfræðilega réttar.
  ```

- Base prompt template:

  ```text
  Setning: {text}
  Málfræðilega rétt: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Setning: {text}

  Greindu hvort setningin er málfræðilega rétt. Svaraðu með 'já' ef setningin er rétt og 'nei' ef hún er það ekki.
  ```

- Label mapping:
  - `correct` ➡️ `já`
  - `incorrect` ➡️ `nei`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ice-ec
```

### Unofficial: IceLinguistic

This dataset was published
[here](https://github.com/stofnun-arna-magnussonar/ice_linguistic_benchmarks), with the
source of the documents unknown. It consists of Icelandic sentences annotated with
whether they are grammatically correct or not (along with other linguistic properties).

The original full dataset consists of 382 samples, and we use a 94 / 32 / 256 split for
training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Ég aflaði upplýsinganna og þú peninganna.",
  "label": "correct"
}
```

```json
{
  "text": "Af hverju fór þú ekki heim?",
  "label": "incorrect"
}
```

```json
{
  "text": "Þú borðaðir kökuna og ég kleinuhringurinn.",
  "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Hér fyrir neðan eru setningar ásamt mati á því hvort þær eru málfræðilega réttar.
  ```

- Base prompt template:

  ```text
  Setning: {text}
  Málfræðilega rétt: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Setning: {text}

  Greindu hvort setningin er málfræðilega rétt. Svaraðu með 'já' ef setningin er rétt og 'nei' ef hún er það ekki.
  ```

- Label mapping:
  - `correct` ➡️ `já`
  - `incorrect` ➡️ `nei`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ice-linguistic
```

## Reading Comprehension

### NQiI

This dataset was published in [this paper](https://aclanthology.org/2022.lrec-1.477/)
and is based on articles from the Icelandic Wikipedia. Annotators were asked to write
both questions (only seeing the beginning of the article) as well as answers as they
appear in the article.

The original full dataset consists of 2,234 / 259 / 244 samples for training, validation
and testing, respectively. We use a 1,024 / 256 / 1,024 split for training, validation
and testing, respectively. Our splits are new, and there can thus be some overlap
between the new test split and the old training and validation splits.

Here are a few examples from the training split:

```json
{
  "context": 'Gróðurhúsalofttegund er lofttegund , í lofthjúpi sem drekkur í sig og gefur frá sér innrauða geislun . Það ferli er aðal ástæða gróðurhúsaáhrifa . Helstu gróðurhúsalofttegundirnar í lofthjúpi jarðar eru vatnsgufa , koldíoxíð , metan , tvíköfnunarefnisoxíð og óson . Án gróðurhúsalofttegunda væri meðalhiti yfirborðs jarðar − 18 ° C , núverandi meðaltals 15 ° C . Í sólkerfinu , eru Venus , Mars og Títan einnig með lofthjúp sem veldur gróðurhúsaáhrifum .',
  "question": 'Hverjar eru gróðurhúsalofttegundirnar ?',
  "answers": {
    "answer_start": array([202], dtype=int32),
    "text": array([' vatnsgufa , koldíoxíð , metan , tvíköfnunarefnisoxíð og óson'], dtype=object)
  }
}
```

```json
{
  "context": 'Hvannadalshnúkur eða Hvannadalshnjúkur er hæsti tindur eldkeilunnar undir Öræfajökli og jafnframt hæsti tindur Íslands . Samkvæmt nýjustu mælingu er hæð hans 2.109,6 metrar yfir sjávarmáli . Tindurinn er staðsettur innan Vatnajökulsþjóðgarðs og er vinsæll hjá fjallgöngufólki , reyndu sem og óreyndu . Tindurinn er ekki flókinn uppgöngu og þarfnast ekki mikillar reynslu eða tækni í fjallgöngum , gangan krefst samt mikils úthalds þar sem oftast er gengið á tindinn og niður aftur á sama deginum . Hækkunin er rúmir 2000 metrar , gangan tekur oftast 12 - 14 klst í heild .',
  "question": 'Hvert er hæsta fjall á Íslandi ?',
  "answers": {
    "answer_start": array([20,  0, 20], dtype=int32),
    "text": array([' Hvannadalshnjúkur', 'Hvannadalshnúkur', ' Hvannadalshnjúkur er hæsti tindur eldkeilunnar undir Öræfajökli og jafnframt hæsti tindur Íslands'], dtype=object)
  }
}
```

```json
{
  "context": 'Falklandseyjar er lítill eyjaklasi út af Suður-Ameríku , um 500 km til suðausturs frá Argentínu . Þær eru undir stjórn Bretlands en Argentína hefur einnig gert tilkall til þeirra og olli það Falklandseyjastríðinu milli þjóðanna 1982 .',
  "question": 'Hvar eru Falklandseyjar ?',
  "answers": {
    "answer_start": array([34, 34], dtype=int32),
    "text": array([' út af Suður-Ameríku', ' út af Suður-Ameríku , um 500 km til suðausturs frá Argentínu'], dtype=object)
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Eftirfarandi eru textar með tilheyrandi spurningum og svörum.
  ```

- Base prompt template:

  ```text
  Texti: {text}
  Spurning: {question}
  Svaraðu með að hámarki 3 orðum: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Texti: {text}

  Svaraðu eftirfarandi spurningu um textann að hámarki í 3 orðum.

  Spurning: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset nqii
```

### Unofficial: IcelandicQA

This dataset was published
[here](https://huggingface.co/datasets/mideind/icelandic_qa_scandeval) and consists of
an automatically created Icelandic question-answering dataset based on the Icelandic
Wikipedia as well as Icelandic news articles from the RÚV corpus.

Both questions and answers were generated automatically, meaning that the answers might
not appear in the context. To remedy this, we used GPT-4o to rephrase the answers to
ensure that they appear in the context.

The original full dataset consists of 2,000 samples, and we use a 531 / 128 / 1,024
split for training, validation and testing, respectively. These are all the samples
where the (rephrased) answer appears in the context.

Here are a few examples from the training split:

```json
{
  "context": 'Ómar Ragnarsson - Syngur fyrir börnin  er 33 snúninga LP hljómplata gefin út af SG - hljómplötum árið 1981. Á henni syngur Ómar Ragnarsson þrettán barnalög. Platan er safnplata af áður útgefnum "hit" lögum af 45 snúninga plötum.\n\nLagalisti \n Ég er að baka - Lag - texti: E. Shuman/B. Bower - Ómar Ragnarsson\n Bróðir minn - Lag - texti: W. Holt -Ómar Ragnarsson\n Eitthvað út í loftið - Lag - texti: P. McCartney - Ómar Ragnarsson \n Lok, lok og læs - Lag - texti: Brezkt þjóðlag - Ómar Ragnarsson\n Aha, sei-sei, já-já - Lag - texti: Ómar Ragnarsson\n Ligga, ligga lá - Lag - texti: Ómar Ragnarsson \n Hláturinn lengir lífið - Lag - texti: Ortega - Ómar Ragnarsson\n Sumar og sól - Lag - texti: Ómar Ragnarsson\n Jói útherji - Lag - texti: Ástralskt þjóðlag - Ómar Ragnarsson\n Óli drjóli - Lag - texti: Ómar Ragnarsson)\n Minkurinn í hænsnakofanum - Lag - texti: Norskt þjóðlag - Ómar Ragnarsson \n Kennið mér krakkar - Lag - texti: A. Johansen - Ómar Ragnarsson\n Hí á þig - Lag - texti: Amerískt þjóðlag - Ómar Ragnarsson\n\nSG-hljómplötur\nHljómplötur gefnar út árið 1981\nÓmar Ragnarsson',
  "question": 'Hvaða ár var LP-hljómplatan „Ómar Ragnarsson - Syngur fyrir börnin“ gefin út?',
  "answers": {
    "answer_start": 102,
    "text": array(['1981'], dtype=object)
  }
}
```

```json
{
  "context": 'Tjörn er kirkjustaður í Dalvíkurbyggð í Svarfaðardal. Bærinn stendur að vestanverðu í dalnum um 5 km innan við Dalvík. Þórarinn Kr. Eldjárn lét reisa núverandi íbúðarhús 1931. Tjarnartjörn er lítið og grunnt stöðuvatn á flatlendinu neðan við bæinn. Tjörnin er innan Friðlands Svarfdæla sem teygir sig allt til strandar. Þar er mikið fuglalíf. Tjörn er með stærri jörðum í Svarfaðardal og að líkindum landnámsjörð þótt bæjarins sé ekki getið í Landnámu. Þar hafa verið stundaðar úrkomumælingar á vegum Veðurstofunnar frá árinu 1970. Í hlíðinni ofan við Tjörn eru volgrur og í framhaldi af þeim er jarðhitinn í Laugahlíð þar sem Sundskáli Svarfdæla fær vatn sitt.\nKristján Eldjárn forseti fæddist á Tjörn 1916 og ólst þar upp.\nSönghópurinn Tjarnarkvartettinn var kenndur við Tjörn í Svarfaðardal.\n\nTjarnarbændur á 20. öld:\n Sr. Kristján Eldjárn Þórarinsson og Petrína Soffía Hjörleifsdóttir\n Þórarinn Kr. Eldjárn og Sigrún Sigurhjartardóttir\n Hjörtur Eldjárn Þórarinsson og Sigríður Hafstað\n Kristján Eldjárn Hjartarson og Kristjana Arngrímsdóttir\n\nTjarnarkirkja \n\nKirkja hefur líklega verið reist á Tjörn fljótlega eftir að kristni var lögleidd í landinu. Hennar er þó ekki getið með beinum hætti í heimildum fyrr en í Auðunarmáldaga frá 1318. Þar segir að kirkjan sé helguð Maríu guðsmóður, Mikjáli erkiengli, Jóhannesi skírara og Andrési postula. Kirkjan átti þá hálft heimalandið, Ingvarastaðaland og hólminn Örgumleiða. Á 16. öld er Tjörn orðin beneficium, þ.e. öll komin í eigu kirkjunnar og þannig hélst þar til sr. Kristján Eldjárn Þórarinsson (1843-1917) keypti jörðina árið 1915. Sr. Kristján var síðasti prestur á Tjörn. Í Svarfaðardal voru lengi fjórar sóknir en þrír prestar því Urðakirkja var annexía frá Tjörn. Upsasókn var síðan lögð undir Tjarnarprest 1859 en 1917 var Tjarnarprestakall með sínum þremur sóknum sameinað Vallaprestakalli. Eftir að prestssetrið var flutt frá Völlum 1969 hefur Tjarnarkirkju verið þjónað af frá Dalvík. Tjarnarsókn nær frá Steindyrum að Ytraholti.\n\nNúverandi kirkja var reist 1892. Hún er úr timbri á hlöðnum grunni og tekur 60-70 manns í sæti. Í henni eru steindir gluggar teiknaðir af Valgerði Hafstað listmálara. Kirkjugarður er umhverfis kirkjuna. Kirkjan skemmdist nokkuð í Kirkjurokinu svokallaða, miklu óveðri sem gekk yfir landið þann 20. september árið 1900. Þá eyðilögðust kirkjurnar á Urðum og Upsum og Vallakirkja varð fyrir skemmdum. Tjarnarkirkja snaraðist á grunni sínum og hallaðist mjög til norðurs en járnkrókar miklir, sem héldu timburverkinu við hlaðinn grunninn, vörnuðu því að verr færi. Nokkru eftir fárviðrið gerði hvassviðri af norðri sem færði hana til á grunninum og rétti hana að mestu við á ný. Mörgum þóttu þetta stórmerki. Gert var við kirkjuna eftir þetta og m.a. voru útbúin á hana járnstög sem lengi settu skemmtilegan svip á bygginguna og minntu á hið mikla fárviðri sem hún hafði staðið af sér. Kirkjan stóð einnig af sér Dalvíkurskjálftann 1934 en þó urðu skemmdir á grunni hennar.\n\nHeimildir \n \n \n Kirkjur Íslands 9. bindi. Tjarnarkirkja bls. 271-307. Reykjavík 2007\n\nTenglar\nTjarnarkirkja á kirkjukort.net \n\nÍslenskir sveitabæir\nKirkjustaðir í Eyjafjarðarsýslu\nKirkjur á Íslandi\nSvarfaðardalur',
  "question": 'Á hvaða bæ í Svarfaðardal hafa verið stundaðar úrkomumælingar á vegum Veðurstofunnar frá árinu 1970?',
  "answers": {
    "answer_start": 0,
    "text": array(['Tjörn'], dtype=object)
  }
}
```

```json
{
  "context": 'Fyrir greinina um þáttinn sem er í gangi í dag, sjá Kastljós (dægurmálaþáttur)\nKastljós var fréttaskýringaþáttur sem var á dagskrá Ríkisútvarpsins frá 1974 til 1998. Hann hóf göngu sína sem fréttaskýringaþáttur um innlendar fréttir árið 1974 og tók þá við af þætti sem nefndist Landshorn. Þátturinn var um fjörutíu mínútna langur, í umsjón fréttastofunnar og sýndur á föstudögum á besta tíma. Umsjónarmenn voru mismunandi fréttamenn í hvert skipti. Annar þáttur á miðvikudögum fjallaði þá um erlendar fréttir. 1980 var þáttunum tveimur slegið saman í eitt Kastljós á föstudögum í umsjón tveggja stjórnenda. 1987 var þættinum aftur breytt í fréttaskýringaþátt um innlend málefni stutt skeið. 1988 hét þátturinn Kastljós á sunnudegi og 1990 Kastljós á þriðjudegi eftir breyttum útsendingartíma en 1992 var þátturinn aftur fluttur á besta tíma á föstudegi. 1993 var Kastljós tekið af dagskrá um skeið þegar dægurmálaþátturinn Dagsljós hóf göngu sína. \n\nÍslenskir sjónvarpsþættir',
  "question": 'Á hvaða árum var fréttaskýringaþátturinn Kastljós upphaflega á dagskrá Ríkisútvarpsins?',
  "answers": {
    "answer_start": 147,
    "text": array(['Frá 1974 til 1998'], dtype=object)
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Eftirfarandi eru textar með tilheyrandi spurningum og svörum.
  ```

- Base prompt template:

  ```text
  Texti: {text}
  Spurning: {question}
  Svaraðu með að hámarki 3 orðum: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Texti: {text}

  Svaraðu eftirfarandi spurningu um textann að hámarki í 3 orðum.

  Spurning: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset icelandic-qa
```

### Unofficial: BeleBele-is

This dataset was published in [this paper](https://aclanthology.org/2024.acl-long.44/)
and features multiple-choice reading comprehension questions across 122 languages.

The original dataset contains 900 unique multiple-choice reading comprehension passages
and questions. From these, we use a 256 / 64 / 580 split for training, validation and
testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Texti: Í Frelsisstríðinu mynduðu ríkin þrettán veikburða ríkisstjórn – með Þjóðþingið sem eina þátt þess – skv. fyrstu stjórnarskránni. Þingið var ekki með nægar valdheimildir til að leggja á skatta, og vegna þess að ekki var neinn alríkisstjóri eða dómsvald til staðar, treysti það á yfirvöld í hverju ríki fyrir sig, sem voru oft og tíðum ósamvinnuþýð, til að framfylgja lögum þess. Það hafði heldur engar valdheimildir til að fella niður skattalög og tolla á milli ríkja. Greinarnar gerðu kröfu um samhljóða samþykki allra ríkjanna áður en hægt var að breyta þeim og ríkin sýndu ríkisvaldinu svo mikla lítilsvirðingu að fulltrúar þeirra voru oft fjarverandi.\nSpurning: Samkvæmt því sem fram kemur í kaflanum, hvaða fullyrðing á nákvæmlega við um ástand ríkisvaldsins í frelsisstríðinu?\nSvarmöguleikar:\na. Skattar voru innheimtir af þinginu og ríkisstofnunum\nb. Breytingar á stjórnarskránni þurftu samþykki þingsins\nc. Fulltrúar ríkjanna voru oft fjarverandi\nd. Hin miðlæga ríkisstjórn var mynduð í kringum tvo meginþætti",
  "label": "c"
}
```

```json
{
  "text": "Texti: İzmir er þriðja stærsta borg Tyrklands með um 3,7 milljónir íbúa, næststærstu höfnina á eftir Istanbúl og er mjög góð samgöngumiðstöð. Hin forna borg Smyrna er núna nútímaleg, þróuð og iðandi viðskiptamiðstöð sem staðsett er við gríðarstóran flóa og umkringd er fjöllum. Hinar breiðu breiðgötur, byggingar með framhliðum úr gleri og nútímalegar verslunarmiðstöðvar með hefðbundnum rauðum þakskífum, 18. aldar markaðurinn og gamlar moskur og kirkjur, þó að andrúmsloft borgarinnar tengist meira Miðjarðarhafssvæði Evrópu en hefðbundnu Tyrklandi.\nSpurning: Hvert eftirfarandi einkennir Izmir er frá fornri tíð?\nSvarmöguleikar:\na. Breiðar breiðgötur\nb. Byggingar með framhliðum úr gleri\nc. Verslanamiðstöðvar\nd. rauðar þakskífur",
  "label": "d"
}
```

```json
{
  "text": "Texti: Dæmigert fyrir það tímabil er Kirby Muxloe Castle sem er frekar víggirt hús en raunverulegur kastali. Stóru gljáðu gluggarnir og þunnu veggirnir hefðu ekki getað staðist stórárás í langan tíma. Árið 1480, þegar Hastings lávarður hóf byggingarframkvæmdirnar, ríkti friður í nánast öllu landinu og aðeins var þörf á varnarmúrum gegn litlum ræningjahópum.\nSpurning: Hvert af eftirtöldu hefði verið talið óvenjulegt við byggingu Kirby Muxloe kastala á þeim tíma sem talað er um í kaflanum?\nSvarmöguleikar:\na. Stórir gluggar\nb. Grunnur sem á að standast árásir\nc. Minna af varnarútbúnaði en í öðrum köstulum\nd. Þunnir veggir",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Eftirfarandi eru fjölvalsspurningar (með svörum).
  ```

- Base prompt template:

  ```text
  Spurningar: {text}
  Svarmöguleikar:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svara: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Spurningar: {text}
  Svarmöguleikar:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Svaraðu eftirfarandi spurningum með 'a', 'b', 'c' eða 'd', og engu öðru.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset belebele-is
```

### Unofficial: MultiWikiQA-is

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
    "context": "Eldfell er rétt rúmlega 200 m hátt eldfjall á Heimaey í Vestmannaeyjaklasanum. Það myndaðist í eldgosi sem hófst 23. janúar 1973 en lauk 3. júlí 1973, þetta eldgos er kallað Heimaeyjargosið.\n\nHeimaeyjargosið \nÍ upphafi gossins opnaðist stór sprunga frá norðri til suðurs á austasta hluta Heimaeyjar, og náði hún að höfninni  í norðri en niður að Skarfatanga í suðri. Fljótlega minnkaði sprungan þó og megineldvarpið varð þar sem nú stendur Eldfell. Gosefnið í upphafi gossins var nánast ísúrt, en þó varð það fljótlega basískt (SiO2 > 52%). Efnainnihald kvikunnar bendir til að kvikuhólf og megineldstöð séu að myndast á þessum slóðum. \n\nStrax og tilkynning barst um að eldgos væri hafið hófst brottflutningur fólks af eynni. Af 5.500 íbúum eyj[48;55;272;1980;3808tarinnar voru um 4.000 fluttir burt um nóttina, mestmegnis með skipum. Á næstu vikum voru búslóðir fólks fluttar burt að mestu, en hús tóku mjög fljótlega að hverfa undir hraun.\n\nEinn maður dó í gosinu og var það af völdum koldíoxíðeitrunar - mikið af lífshættulegum lofttegundum kom upp úr jörðinni með vikrinum og gjóskunni. Mikil mildi þótti að ekki skyldi hafa farið verr, þar sem að sprungan kom upp rétt austan við austasta hús bæjarins (þó munaði ekki nema nokkrum metrum). \n\nUm helmingur húsa bæjarins ýmist lenti undir hrauni eða á annan hátt eyðilagðist í gosinu, en uppbyggingin eftir gosið var mjög snögg.\n\nGosið í Heimaey byrjaði 23. janúar 1973 og lauk 3. júlí sama ár. Þetta er fyrsta gos sem hefst við þéttbýli á Íslandi. Það var loftskeytamaðurinn Hjálmar Guðnason og vinur hans, Ólaf Granz, sem voru í sínum vanalega miðnæturgöngutúr þegar hinn tilkomumikla sýn birtist þeim þegar þeir skoðuðu bæinn frá Helgafellstoppi. Þar sáu þeir jörðina opnast og eldtungurnar stóðu marga metra upp í loftið. Strax var haft samband við lögreglu þar sem tilkynnt var að jarðeldur væri kominn upp austan við Kirkjubæ. Lögreglan tók upplýsingarnar ekki trúanlegar í fyrstu en fór strax að athuga hvað væri í gangi og þegar á staðinn var komið sáu þeir að gos var hafið á 1600 metra langri sprungu og magnaðist hratt á fyrstu mínútunum. Kveikt var á brunalúðrum og á mjög skömmum tíma var allur bærinn vaknaður og fólk streymdi úr húsum sínum og niður á bryggju. Flestir þeir sem upplifðu gosið eru sammála um að klukkuna hafi vantað fimm mínútur í tvö þegar að gosið hófst.\n\nEldfellshraun er um 2,5 ferkílómetrar og stækkaði Heimaey um 20%.\n\nTenglar \n Átta tímar í eyjum; greinar í Morgunblaðinu 1973\n kort af götum sem fóru undir hraun\nVestmannaeyjar\nEldfjöll á Íslandi\nEldgos á Íslandi",
    "question": "Hvað er Eldfell hátt?",
    "answers": {
        "answer_start": array([11]),
        "text": array(["rétt rúmlega 200 m"], dtype=object)
    }
}
```

```json
{
    "context": "Edduverðlaunin 2007 eru afhending Edduverðlauna Íslensku kvikmynda- og sjónvarpsakademíunnar sem fór fram á Hótel Hilton Nordica sunnudaginn 11. nóvember 2007. Aðalkynnar kvöldsins voru Þorsteinn Guðmundsson og Ólafía Hrönn Jónsdóttir.\n\nÞær breytingar urðu á verðlaunaflokkum að flokknum „Leikari/leikkona í aðalhlutverki“ var skipt í tvennt og þrír tilnefndir í hvorum flokknum „leikari í aðalhlutverki“ og „leikkona í aðalhlutverki“. Fyrir sjónvarpsefni var flokknum „sjónvarpsþáttur ársins“ skipt í „frétta- og/eða viðtalsþáttur“ ársins annars vegar og „menningar- og/eða lífstílsþáttur ársins“ sem ásamt flokknum „skemmtiþáttur ársins“ gera þrjá flokka fyrir sjónvarpsþætti í stað tveggja áður. Flokkurinn „myndataka og klipping“ sem hafði verið með árið 2005 var aftur tekinn upp. Alls voru því veitt verðlaun í sextán flokkum, auk heiðursverðlauna ÍKSA. \n\nSigurmynd hátíðarinnar var kvikmyndin Foreldrar eftir Ragnar Bragason með sex verðlaun. Tvær myndir með tilvísun í Breiðavíkurmálið voru tilnefndar þetta árið, heimildarmyndin Syndir feðranna og kvikmynd Guðnýjar Halldórsdóttur, Veðramót. Tveir sjónvarpsþættir fengu verðlaun sem besti frétta-/viðtalsþáttur ársins; Kompás á Stöð 2 og Út og suður á RÚV. Egill Helgason var bæði valinn sjónvarpsmaður ársins og bókmenntaþáttur hans, Kiljan, var valinn menningar-/lífstílsþáttur ársins.\n\nTilnefningar og handhafar Edduverðlauna 2007\nHandhafar Edduverðlaunanna í hverjum flokki eru feitletraðir og gulllitaðir.\n\nKvikmynd ársins\n\nLeikið sjónvarpsefni ársins\n\nStuttmynd ársins\n\nLeikstjóri ársins\n\nHandrit ársins\n\nLeikkona í aðalhlutverki\n\nLeikari í aðalhlutverki\n\nLeikari/leikkona í aukahlutverki\n\nHeimildarmynd ársins\n\nFrétta- og/eða viðtalsþáttur ársins\n\nMenningar- og/eða lífstílsþáttur ársins\n\nSkemmtiþáttur ársins\n\nSjónvarpsmaður ársins\n\nMyndataka og klipping\n\nHljóð og tónlist\n\nÚtlit myndar\n\nHeiðursverðlaun ÍKSA 2007\n\nFramlag Íslands til forvals Óskarsins\n\nEdduverðlaunin",
    "question": "Undir hvaða nafni er bókmenntaþáttur Egils Helgasonar þekktur, sem hlaut viðurkenningu sem menningar- eða lífstílsþáttur ársins?",
    "answers": {
        "answer_start": array([1294]),
        "text": array(["Kiljan"], dtype=object)
    }
}
```

```json
{
    "context": "Edinborgarhúsið er friðað hús og menningarmiðstöð á Ísafirði. Húsið var byggt af Edinborgarversluninni sem var kringum aldamótin 1900 eitt stærsta verslunarfyrirtæki landsins um aldamótin 1900. Edinborgarverslunin var stofnuð í Reykjavík árið 1895 og var í eigu  Ásgeirs Sigurðsson sem ættaður var frá Ísafirði og  skoska verslunarfyrirtækisins Copland and Berrie í Leith. Edinborgarverslunin færði út kvíarnar og opnaði verslunarbúð á Ísafirði árið 1902. Árið 1903 varð Karl Olgeirsson, verslunarstjóri Edinborgarverslunar á Ísafirði og meðeigandi fáum árum síðar. \n\nBygging Edinborgarhússins hófst eftir að fengin var byggingarlóð fyrir húsið árið 1907  við Pollinn. Þar var byggt hús eftir teikningu Rögnvald Ágúst Ólafsson og bryggja og bryggjuhús. Edinborgarhúsið og bryggjan voru lengi ein mesta mannvirki á Ísafirði. Edinborgarverslun hætti starfsemi á Ísafirði árið 1917 og seldi hlut sinn til Karls verslunarstjóra. Árið 1918 varð Jóhann E. Þorsteinsson meðeigandi og var verslunin rekin undir nafninu Karl & Jóhann til  1923 en þá seldi Karl sinn hluta og Sigurjón Þ. Jónsson  kom inn og ráku Sigurjón og Jóhann E. Þorsteinson verslunina til ársins 1926.\n\nTogarafélag Ísfirðinga h.f. sem var stofnað 1925  var til húsa í Edinborgarhúsinu. Félagið keypti og rak togarann Hávarð Ísfirðing frá 1925 til 1939. Á kreppuárunum gekk reksturinn illa og árið 1935 tók Landsbankinn yfir reksturinn, hlutafé var aukið og nafni breytt í h.f. Hávarður. Árið 1938 varð þar félag gjaldþrota og stofnað nýtt hlutafélag með aðkomu Kaupfélags Ísfirðinga. Nýja hlutafélagið var nefnt Valur og var togarinn Hávarður endurskírður og nefndur Skutull.\n\nKaupfélag Ísfirðinga elfdist mjög á millistríðsárunum og keypti upp ýmsar eignir. Árið 1937 eignaðist kaupfélagið eignir sem höfðu tilheyrt Edinborgarversluninni  og þar á meðal Edinborgarhúsið og fiskreiti á lóð hússins. Kaupfélagið átti stóran hlut í útgerðarfélaginu Nirði en það félag gerði út báta sem kallaðir voru Dísirnar. Kaupfélagið verkaði fisk frá Nirði á fiskreitunum  og skömmu eftir árið 1945 var settur upp þurrklefi fyrir fisk í Edinborgarhúsinu. Þessi þurrklefi gerði mögulegt að þurrka fisk innan dyra á veturna. Kaupfélag Ísfirðinga átti Edinborgarhúsið í rúmlega 50 ár eða þangað til SÍS tók yfir eigur þess.\n\nStofnað var  einkahlutafélag um menningarmiðstöð í Edinborgarhúsinu 9. september 1992.\n\nHeimild \n Saga hússins (af vefnum edinborg.is)\n\nTenglar \n Glæsileg menningarmiðstöð í Edinborgarhúsi, Morgunblaðið B, 11. janúar 1998, bls. 6-7\n Stefnt að opnun fjölnotasalar eftit eitt ár, Morgunblaðið, 20. maí 2006, bls. 22\n Formleg opnun Edinborgarhússins, Bæjarins besta, 31. maí 2007, bls. 2\n\nÍsafjörður\nByggingar á Íslandi",
    "question": "Hver gegndi stöðu verslunarstjóra hjá Edinborgarversluninni á Ísafirði árið 1903?",
    "answers": {
        "answer_start": array([471]),
        "text": array(["Karl Olgeirsson"], dtype=object)
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Eftirfarandi eru textar með tilheyrandi spurningum og svörum.
  ```

- Base prompt template:

  ```text
  Texti: {text}
  Spurning: {question}
  Svaraðu með að hámarki 3 orðum: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Texti: {text}

  Svaraðu eftirfarandi spurningu um textann að hámarki í 3 orðum.

  Spurning: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-is
```

## Knowledge

### IcelandicKnowledge

This dataset was published
[here](https://huggingface.co/datasets/mideind/icelandic_qa_scandeval) and consists of
an automatically created Icelandic question-answering dataset based on the Icelandic
Wikipedia as well as Icelandic news articles from the RÚV corpus.

The dataset was converted into a multiple-choice knowledge dataset by removing the
contexts and using GPT-4o to generate 3 plausible wrong answers for each correct answer,
using the following prompt for each `row` in the original dataset:

```python
messages = [
    {
        "role": "user",
        "content": f"For the question: {row.question} where the correct answer is: {row.answer}, please provide 3 plausible alternatives in Icelandic. You should return the alternatives in a JSON dictionary, with keys 'first', 'second', and 'third'. The values should be the alternatives only, without any numbering or formatting. The alternatives should be unique and not contain the correct answer."
    }
]

completion = client.beta.chat.completions.parse(
    model="gpt-4o", messages=messages, response_format=CandidateAnswers
)
```

where `CandidateAnswers` is a Pydantic model that is used to ensure [structured
outputs](https://platform.openai.com/docs/guides/structured-outputs).

The original dataset has 2,000 samples, but only 1,994 unique questions, and the total
length of this dataset is therefore 1,994. The split is given by 842 / 128 / 1024 for
train, val, and test, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Hver var talinn heilagur maður eftir dauða sinn, er tákngervingur alþýðuhreyfingar vestanlands og talinn góður til áheita?\nSvarmöguleikar:\na. Þórður Jónsson helgi\nb. Guðmundur Arason\nc. Snorri Þorgrímsson\nd. Jón Hreggviðsson",
  "label": "a"
}
```

```json
{
  "text": "Í kringum hvaða ár hófst verslun á Arngerðareyri?\nSvarmöguleikar:\na. 1895\nb. 1884\nc. 1870\nd. 1902",
  "label": "b"
}
```

```json
{
  "text": "Hvenær var ákveðið að uppstigningardagur skyldi vera kirkjudagur aldraðra á Íslandi?\nSvarmöguleikar:\na. Árið 1975\nb. Árið 1985\nc. Árið 1982\nd. Árið 1990",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Eftirfarandi eru fjölvalsspurningar (með svörum).
  ```

- Base prompt template:

  ```text
  Spurningar: {text}
  Svarmöguleikar:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svara: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Spurningar: {text}
  Svarmöguleikar:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Svaraðu eftirfarandi spurningum með 'a', 'b', 'c' eða 'd', og engu öðru.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset icelandic-knowledge
```

### Unofficial: ARC-is

This dataset is a machine translated version of the English [ARC
dataset](https://doi.org/10.48550/arXiv.1803.05457) and features US grade-school science
questions. The dataset was translated by Miðeind using the Claude 3.5 Sonnet model.

The original full dataset consists of 1,110 / 297 / 1,170 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 1,024 split for training,
validation and testing, respectively (so 2,304 samples used in total). All new splits
are subsets of the original splits.

Here are a few examples from the training split:

```json
{
  "text": "Líkamar manna hafa flókna uppbyggingu sem styður vöxt og lífslíkur. Hver er grundvallaruppbygging líkamans sem stuðlar að vexti og lífslíkum?\nSvarmöguleikar:\na. fruma\nb. vefur\nc. líffæri\nd. líffærakerfi",
  "label": "a"
}
```

```json
{
  "text": "Veðurfræðingur skráir gögn fyrir borg á ákveðnum degi. Gögnin innihalda hitastig, skýjahulu, vindhraða, loftþrýsting og vindátt. Hvaða aðferð ætti veðurfræðingurinn að nota til að skrá þessi gögn fyrir fljótlega tilvísun?\nSvarmöguleikar:\na. skriflega lýsingu\nb. töflu\nc. stöðvarlíkan\nd. veðurkort",
  "label": "b"
}
```

```json
{
  "text": "Hvaða breytingar urðu þegar reikistjörnurnar hitnnuðu á meðan þær mynduðust?\nSvarmöguleikar:\na. Massi þeirra jókst.\nb. Þær töpuðu meirihluta geislavirkra samsæta sinna.\nc. Uppbygging þeirra aðgreindist í mismunandi lög.\nd. Þær byrjuðu að snúast í kringum sólina.",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Eftirfarandi eru fjölvalsspurningar (með svörum).
  ```

- Base prompt template:

  ```text
  Spurningar: {text}
  Svarmöguleikar:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svara: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Spurningar: {text}
  Svarmöguleikar:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Svaraðu eftirfarandi spurningum með 'a', 'b', 'c' eða 'd', og engu öðru.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset arc-is
```

### Unofficial: MMLU-is

This dataset is a machine translated version of the English [MMLU
dataset](https://openreview.net/forum?id=d7KBjmI3GmQ) and features questions within 57
different topics, such as elementary mathematics, US history and law. The dataset was
translated using [Miðeind](https://mideind.is/english.html)'s Greynir translation model.

The original full dataset consists of 269 / 1,410 / 13,200 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 2,048 split for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
new and there can thus be some overlap between the original validation and test sets and
our validation and test sets.

Here are a few examples from the training split:

```json
{
  "text": "Af hverju er öruggara að horfa á tunglið en að horfa á sólina?\nSvarmöguleikar:\na. Tunglið er minna bjart.\nb. Tunglið er nær jörðinni.\nc. Tunglið skín aðallega á nóttunni.\nd. Tunglið er aðeins fullt einu sinni í mánuði.",
  "label": "a"
}
```

```json
{
  "text": "Hvaða lög jarðar eru aðallega gerð úr föstu efni?\nSvarmöguleikar:\na. innri kjarni og ytri kjarni\nb. skorpu og innri kjarni\nc. skorpu og möttli\nd. möttli og ytri kjarni",
  "label": "b"
}
```

```json
{
  "text": "Bekkur er að rannsaka þéttleika bergsýna. Hvaða vísindalegan búnað þurfa þau til að ákvarða þéttleika bergsýnanna?\nSvarmöguleikar:\na. smásjá og vog\nb. bikar og mæliglös\nc. mæliglös og vog\nd. smásjá og mæliglös",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Eftirfarandi eru fjölvalsspurningar (með svörum).
  ```

- Base prompt template:

  ```text
  Spurningar: {text}
  Svarmöguleikar:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svara: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Spurningar: {text}
  Svarmöguleikar:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Svaraðu eftirfarandi spurningum með 'a', 'b', 'c' eða 'd', og engu öðru.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset mmlu-is
```

### Unofficial: Icelandic Language Tests

This dataset is based on the old Icelandic standardised tests held from 2013 to 2017,
available at [mms.is](https://mms.is/eldri-prof-og-svor). The tests were administered at
the primary school level (10th grade) and cover the Icelandic language. Only the
multiple-choice questions from the tests have been included.

There are 571 samples, and we use the 16 oldest samples for train, and the rest for
test.

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Eftirfarandi eru fjölvalsspurningar (með svörum).
  ```

- Base prompt template:

  ```text
  Spurningar: {text}
  Svarmöguleikar:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svara: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Spurningar: {text}
  Svarmöguleikar:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Svaraðu eftirfarandi spurningum með 'a', 'b', 'c' eða 'd', og engu öðru.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset icelandic-lang-tests
```

### Unofficial: Icelandic Mathematics Tests

This dataset is based on the old Icelandic standardised tests held from 2013 to 2017,
available at [mms.is](https://mms.is/eldri-prof-og-svor). The tests were administered at
the primary school level (10th grade) and cover mathematics. Only the multiple-choice
questions from the tests have been included.

There are 242 samples in total, and we use the oldest 16 samples for train, and the rest
for test.

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Eftirfarandi eru fjölvalsspurningar (með svörum).
  ```

- Base prompt template:

  ```text
  Spurningar: {text}
  Svarmöguleikar:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svara: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Spurningar: {text}
  Svarmöguleikar:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Svaraðu eftirfarandi spurningum með 'a', 'b', 'c' eða 'd', og engu öðru.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset icelandic-math-tests
```

## Common-sense Reasoning

### Winogrande-is

This dataset was published in [this paper](https://aclanthology.org/2022.lrec-1.464/)
and is a manually translated and adapted version of the English [WinoGrande
dataset](https://doi.org/10.1145/3474381). The samples are sentences containing two
nouns and an ambiguous pronoun, and the task is to determine which of the two nouns the
pronoun refers to.

The original full dataset consists of 1,095 samples, and we use a 64 / 128 / 896 split
for training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Eiginmaðurinn hennar Myrru keypti handa henni hálsmen með perlu og hún hélt að það væri ekki ekta. _ var of gyllt.\nSvarmöguleikar:\na. perlan\nb. hálsmenið",
  "label": "a"
}
```

```json
{
  "text": "Bergfinnur lét sem hann heyrði ekki í lekanum í krananum en hann hafði ekkert um að velja þegar hundurinn gelti. _ er háværari.\nSvarmöguleikar:\na. lekinn\nb. hundurinn",
  "label": "b"
}
```

```json
{
  "text": "Danía var spenntari fyrir því að heimsækja ritstjórann en Þorláksína vegna þess að _ fannst nýja bókin geggjuð.\nSvarmöguleikar:\na. Þorláksínu\nb. Daníu",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Eftirfarandi eru fjölvalsspurningar (með svörum).
  ```

- Base prompt template:

  ```text
  Spurningar: {text}
  Svarmöguleikar:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svara: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Spurningar: {text}
  Svarmöguleikar:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Svaraðu eftirfarandi spurningum með 'a', 'b', 'c' eða 'd', og engu öðru.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset winogrande-is
```

### Unofficial: HellaSwag-is

This dataset is a machine translated version of the English [HellaSwag
dataset](https://aclanthology.org/P19-1472/). The original dataset was based on both
video descriptions from ActivityNet as well as how-to articles from WikiHow. The dataset
was translated using [Miðeind](https://mideind.is/english.html)'s Greynir translation
model.

The original full dataset consists of 9,310 samples. We use a 1,024 / 256 / 2,048 split
for training, validation and testing, respectively (so 3,328 samples used in total).

Here are a few examples from the training split:

```json
{
  "text": "[höf.] Hvernig finna má samræmi í lífinu [titill] Skuldbinda þig til breytinga. [skref] Fyrsta skrefið til að ná fram breytingum í lífinu er að skuldbinda sig til breytinga. Með því að gefa meðvitaða, viljasetta yfirlýsingu til sjálfs síns um að þú munir halda þig við efnið og ná settum árangri getur það hjálpað þér að halda þér við efnið og ýtt þér áfram í átt að því markmiði.\nSvarmöguleikar:\na. Þá ættir þú að vera að skuldbinda þig til að lifa stöðugra og samræmdara lífi. [Undirskrefi] Hugsaðu um ástæðurnar fyrir því að þú vilt lifa samræmdara lífi.\nb. [undirefni] Byrjaðu á því að skuldbinda þig til að breyta einhverju sem kemur þér úr jafnvægi. Ef þú gerir það ekki þá siturðu uppi með eitthvað sem loðir við þig heima hjá þér, sem verður ekki auðveldara að koma í staðinn fyrir þá tilfinningu.\nc. [Undirefni] Ekki láta skoðanir þínar eða skoðanir stangast á við sjálfsvirðingu þína. Viðurkenndu að þú sért fullorðinn og því óhræddur við að taka þínar eigin ákvarðanir varðandi það sem þú vilt í lífinu.\nd. [Efnisorð] Þegar einhver annar hvetur þig til að breyta, þá skaltu verðlauna þig fyrir það góða sem þú nærð fram þó að það hafi kannski ekki litið út á einhvern hátt. [Titill] Ekki ætlast til þess að fólk breyti sér af skyldurækni.",
  "label": "a"
}
```

```json
{
  "text": "Maður er að vinna á sporöskjulaga vél. það\nSvarmöguleikar:\na. grípur og stýrir tækinu.\nb. sýnir skjáinn á vélinni.\nc. er sýnd í tveimur hlutum, sem hver um sig er festur af manneskju.\nd. virðist vera vinsæll eftir því sem hann vinnur sig upp.",
  "label": "b"
}
```

```json
{
  "text": "Sleðastúlka á uppblásnum bát heldur á streng framan á mann, allt í einu dettur hún í holu. Fólk ber sleðabáta og sleðastúlkan er á sleðabáti. eftir hóp af fólki\nSvarmöguleikar:\na. sleða saman kanóum, svo sleða aðrir í vatninu.\nb. sleða hliðar vatnsvatn á hestum við hliðina á brú báta.\nc. sleða niður brekkuna þangað til hitta aðra einstaklinga.\nd. Sleðamenn ganga á torgi, á milli annarra og síðan hlaupa allir um.",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Eftirfarandi eru fjölvalsspurningar (með svörum).
  ```

- Base prompt template:

  ```text
  Spurningar: {text}
  Svarmöguleikar:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svara: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Spurningar: {text}
  Svarmöguleikar:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Svaraðu eftirfarandi spurningum með 'a', 'b', 'c' eða 'd', og engu öðru.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset hellaswag-is
```

## Summarisation

### RRN

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.3/)
and consists of news articles and their summaries from RÚV, the Icelandic National
Broadcasting Service, from years 2021 and 2022.

The original full dataset consists of 3,960 samples, and we use a 1,024 / 256 / 1,024
split for training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Við erum að sjá ótta um truflanir á framleiðslukeðjum og efnahagsstarfsemi eitthvað í líkingu við það sem var fyrr á árinu.\nsegir Jón Bjarki Bentsson aðalhagfræðingur Íslandsbanka. Áhrif Delta afbrigðisins sjást víða. Eftirspurn hefur ekki haldist í hendur við væntingar sem meðal annars hefur orsakað mikla verðlækkun á olíu á heimsmarkaði undanfarnar vikur. Hefur verðið á ekki verið lægra í þrjá mánuði.\nBílaframleiðeindur eru einnig í vanda, en þar er vandamálið ekki skortur á eftirspurn heldur skortur á aðföngum, á svokölluðum hálfleiðurum nánar tiltekið. Þeir eru aðallega framleiddir í Asíu og hefur útbreiðsla Delta afbrigðisins raskað framleiðslu og framkallað skort. Margir af stærstu bílaframleiðendum heims hafa tilkynnt um að þeir neyðist til að draga úr framleiðslu og þarf Toyota, stærsti bílaframleiðandi heims, að minnka framleiðslu sína um 40 prósent.\nÁstandið hefur sömuleiðis valdið mikilli styrkingu dollars. Miðgengi seðlabanka Íslands í dag er 128 krónur en var í byrjun sumars 121 króna. Á sama tíma hefur krónan haldist stöðug gagnvart öðrum myntum. Auk útbreiðslu Delta afbrigðisins hafa atburðir liðinna vikna í Afganistan þrýst á styrkingu dollarsins.\nÞetta hefur allt áhrif til þess að hvetja til ótta í öryggi eins og svo er kallað og dollarinn nýtur oft góðs af svoleiðis ótta. Þykir náttúrlega gríðarlega örugg eign að hafa og seljanleiki hans er náttúrlega meiri en nokkurs annars eigna flokks.",
  "target_text": "Útbreiðsla Delta afbrigðis kórónuveirunnar ógnar bata heimshagkerfisins. Olíuverð hefur hríðfallið á undanförnum vikum, bílaframleiðendur fá ekki aðföng og fjárfestar flykkjast í bandaríkjadollar. "
}
```

```json
{
  "text": "Veðurfar hefur verið óvenjulegt á suðvesturhorni landsins. Lítið snjóaði í vetur og síðustu vikur hefur úrkoma verið með allra minnsta móti. Jón Þór Ólason, formaður Stangveiðifélags Reykjavíkur, segir að veiðimenn séu vissulega orðnir langeygir eftir rigningunni, en bætir við að eitt helsta einkenni íslenskra veiðimanna sé óbilandi bjartsýni.\nJón Þór segir að norðan- og austanlands séu horfurnar betri. Þurrkatíðin hefur þó ekki haft áhrif á sölu veiðileyfa. Óvissan um veðurfar fylgi með í kaupunum og nú þegar eru margar af ám félagsins uppseldar. Þá er von á fleiri útlendingum í ár en í fyrra, en kórónuveirufaraldurinn hafði mjög mikil áhrif á sölu veiðileyfa í fyrra.",
  "target_text": "Formaður Stangaveiðifélags Reykjavíkur segir veiðimenn á suðvesturhorni landsins dansa nú regndans í von um að langvarandi þurrkatíð sé senn á enda."
}
```

```json
{
  "text": "Í morgun fjarlægðu bæjarstarfsmenn áberandi kosningaborða framboðsins Vina Kópavogs á horni Digranesvegar og Grænutungu. Jóhann Sigurbjörnsson, sem er í18. sæti á lista Vina Kópavogs, setti borðana upp og er afar ósáttur við þeir hafi verið fjarlægðir. Hann segir að vegið sé að tjáningarfrelsi sínu.\nÉg hengi upp borða vegna þess að ég tel mig vera í fullum rétti til að tjá mig um þær framkvæmdir sem eru í gangi hérna á móti mér. Ég hengi upp þessa borða á grindverkið sem er rétt fyrir innan lóðamörk síðan koma hingað menn í gulum fötum í morgun frá bænum sem fjarlægja borðana.\nBæjarstarfsmenn hafa undanfarið verið í samskiptum við framboðið um að brotið hafi verið gegn lögreglusamþykkt og byggingarreglugerð með því að setja upp auglýsingaborða á lóðamörkum og utan þeirra, og einnig svo stóra auglýsingaborða að sérstakt leyfi þurfi.\nSigríður Björg Tómasdóttir upplýsingafulltrúi Kópavogsbæjar segir í samtali við fréttastofu að skýrar reglur gildi um uppsetningu auglýsingaskilta. Reglur um slíka uppsetningu hafi verið sendar að gefnu tilefni á alla framboðsflokka í Kópavogi fyrir helgi. Þá hafi stórt auglýsingaskilti á vegum Framsóknarflokksins í Skógarlind verið fjarlægt af bæjaryfirvöldum í síðustu viku. Sigríður segir að skiltin verði að vera undir tveimur fermetrum til að mega vera uppi - annars þurfi að sækja um leyfi frá byggingarfulltrúa Kópavogsbæjar. Reglurnar séu skýrar.\nHelga, Oddviti Vina Kópavogsbæjar segist hissa yfir framgangi bæjaryfirvalda, þetta geti ekki staðist skoðun og að framboðið muni leita réttar síns.",
  "target_text": "Auglýsingaskilti og framboðsborðar hafa verið fjarlægð af bæjaryfirvöldum í Kópavogi víðs vegar um bæinn síðustu daga. "
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:

  ```text
  Eftirfarandi eru fréttagreinar með tilheyrandi samantektum.
  ```

- Base prompt template:

  ```text
  Fréttagrein: {text}
  Samantekt: {target_text}
  ```

- Instruction-tuned prompt template:

  ```text
  Fréttagrein: {text}

  Skrifaðu samantekt um ofangreindu grein.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset rrn
```

## European Values

### ValEU-is

This dataset is the official Icelandic version of questions from the [European values
study](https://europeanvaluesstudy.eu/). The dataset contains multiple-choice
questions regarding people's values and beliefs across a variety of topics, such as
politics, religion and society.

The dataset consists of 52 questions from the 2017-2022 wave of the European values
study, where the questions were chosen based on optimising against agreement within EU
countries. We use only zero-shot evaluation on this dataset, and thus require no splits.

Here are a few examples from the training split:

```json
{
  "question_id": "G063",
  "text": "Fólk upplifir sig á misjafnan hátt og hvernig það tengist heiminum í kringum sig. Með því að nota þetta spjald, getur þú sagt mér hversu sterkum böndum þú tengist...?\nHeiminum\nSvarmöguleikar:\na. Mjög sterkum böndum\nb. Sterkum böndum\nc. Veikum böndum\nd. Mjög veikum böndum"
}
```

```json
{
  "question_id": "E265_08",
  "text": "Að þínu mati, hversu oft á eftirfarandi sér stað í íslenskum kosningum?\nKjósendum er hótað ofbeldi á kjörstað\nSvarmöguleikar:\na. Mjög oft\nb. Nokkuð oft\nc. Ekki oft\nd. Alls ekki oft"
}
```

```json
{
  "question_id": "E233",
  "text": "Þrátt fyrir að margir þættir séu æskilegir, eru ekki allir þeirra nauðsynleg einkenni lýðræðisríkja. Vinsamlegast segðu mér hvaða einkenni þér finnst vera nauðsynleg í lýðræðisríkjum. Miðaðu við mælistikuna hér á spjaldinu þar sem 1 merkir „alls ekki nauðsynlegt í lýðræðisríki“ og 10 merkir að það sé tvímælalaust „nauðsynlegt í lýðræðisríki“. Hvaða tala lýsir best þinni skoðun?\nKonur hafa sömu réttindi og karlar\nSvarmöguleikar:\na. Það er andstætt lýðræði (spontant).\nb. Alls ekki nauðsynlegt einkenni í lýðræðisríki\nc. 2\nd. 3\ne. 4\nf. 5\ng. 6\nh. 7\ni. 8\nj. 9\nk. Nauðsynlegt einkenni í lýðræðisríki"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 0
- Prefix prompt:

  ```text
  Eftirfarandi eru fjölvalsspurningar (með svörum).
  ```

- Base prompt template:

  ```text
  Spurningar: {text}
  Svarmöguleikar:
  a. {option_a}
  b. {option_b}
  (...)
  k. {option_k}
  Svara: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Spurningar: {text}
  Svarmöguleikar:
  a. {option_a}
  b. {option_b}
  (...)
  k. {option_k}

  Svaraðu eftirfarandi spurningum með 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
  eða 'k', og engu öðru.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset valeu-is
```
