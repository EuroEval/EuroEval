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

  ```text
  Följande är recensioner och deras sentiment, som kan vara 'positiv', 'neutral' eller 'negativ'.
  ```

- Base prompt template:

  ```text
  Recension: {text}
  Sentiment: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Recension: {text}

  Klassificera sentimentet i recensionen. Svara med 'positiv', 'neutral' eller 'negativ'.
  ```

- Label mapping:
  - `positive` ➡️ `positiv`
  - `neutral` ➡️ `neutral`
  - `negative` ➡️ `negativ`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset swerec
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

  ```text
  Följande är meningar och JSON-ordböcker med de namngivna enheter som förekommer i den givna meningen.
  ```

- Base prompt template:

  ```text
  Mening: {text}
  Namngivna entiteter: {label}
  ```

- Instruction-tuned prompt template:

  ```text
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
euroeval --model <model-id> --dataset suc3
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

The original dataset consists of 6,026 samples, from which we use 1,024 / 256 / 2,048
samples for training, validation and testing, respectively (so 3,328 samples used in
total). These splits are used as-is in the framework.

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

  ```text
  Följande är meningar och huruvida de är grammatiskt korrekta.
  ```

- Base prompt template:

  ```text
  Mening: {text}
  Grammatisk korrekt: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Mening: {text}

  Bestäm om meningen är grammatiskt korrekt eller inte. Svara med 'ja' om meningen är korrekt och 'nej' om den inte är.
  ```

- Label mapping:
  - `correct` ➡️ `ja`
  - `incorrect` ➡️ `nej`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-sv
```

## Reading Comprehension

### MultiWikiQA-sv

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
    "context": "Juan Mayorga, född 6 april 1965 i Madrid, är en spansk dramatiker och manusförfattare.\n\nBiografi\nJuan Mayorga har en examen i matematik och filosofi från Universidad Complutense de Madrid 1988. Därefter arbetade han som forskarassistent i filosofi vid Consejo Superior de Investigaciones Científicas. Fortsatta studier i Münster, Berlin och Paris ledde till en doktorsexamen i filosofi 1997 med en avhandling om Walter Benjamin. Han debuterade som dramatiker 1989 med Siete hombres buenos (Sju goda män). Tillsammans med dramatiker kollegorna José Ramón Fernández, Luis Miguel González Cruz och Raúl Hernández Garrido grundade han 1993 teatergruppen Teatro del Astillero. 2011 grundade han en ny teatergrupp, La Loca de la Casa. Sedan 1998 är Mayorga har han undervisat i filosofi och dramatik vid Escuela Superior de Arte Dramático i Madrid. För närvarande (2017) är han chef för avdelningen för scenkonst vid Universidad Carlos III de Madrid. Han har skrivit ett trettiotal pjäser (2017) och hans dramatik har spelats i 18 länder och översatts till 16 språk. Han eftersträvar en filosofiskt präglad teater som tvingar publiken till ställningstaganden. Till hans förebilder hör Harold Pinter. Bland utmärkelser han tilldelats kan nämnas de spanska priserna Nacional de Teatro 2007 och Nacional de Literatura Dramática 2013 samt Premio Europa New Theatrical Realities 2016.\n\n2007 regisserade Alexander Mørk-Eidem Mayorgas Himmelweg på Nationaltheatret i Oslo. 2014 skulle han även ha regisserat den på Stockholms stadsteater under titeln Himlavägen i Jens Nordenhöks översättning men det ställdes in.\n\nReferenser\n\nKällor\n Pressrelease, Premio Europa 14/3 2016 \n Juan Mayorga, The Playwrights Database (läst 5 april 2017)\n Juan Mayorga, France culture (läst 5 april 2017)\n Juan Mayorga, theatre-contemporain.net (läst 5 april 2017)\n Juan Mayorga, Théâtre de Rond-Point, Paris (läst 5 april 2017)\n Juan Mayorga, madridesteatro.com (läst 5 april 2017)\n Arkiv, Kulturhuset Stadsteatern (läst 5 april 2017)\n Lillian Bikset: Teater, løgn og bedrag, Dagbladet 31/8 2007\n Elisabeth Leinslie: Kjenn din besøkelsestid, Dagsavisen 2/9 2007\n Rocío García: Juan Mayorga: las obsesiones de un matemático y autor de éxito, El País 1/6 2016\n\nNoter\n\nExterna länkar\n Juan Mayorga, Internet Movie Database (IMDb)\n\nSpanska dramatiker\nSpanskspråkiga dramatiker\nSpanska manusförfattare\nSpanska författare under 1900-talet\nSpanska författare under 2000-talet\nDramatiker under 1900-talet\nDramatiker under 2000-talet\nPersoner från Madrid\nFödda 1965\nLevande personer\nMän",
    "question": "Vilka akademiska examina har Juan Mayorga avlagt vid Universidad Complutense de Madrid?",
    "answers": {
        "answer_start": array([126]),
        "text": array(["matematik och filosofi"], dtype=object)
    }
}
```

```json
{
    "context": "Janka Kupala (egentligen Ivan Daminikavitj Lutsevitj) född 7 juli 1882 i Vjazynka utanför Minsk, död 28 juni 1942 i Moskva, var en belarusisk författare. Tillsammans med Jakub Kolas räknas han som en av den moderna belarusiska litteraturens grundare.\n\nKupala var till stor del självlärd som författare. Han blev aktiv i den \"belarusiska pånyttfödelsen\" (1903–1921) och redaktör för den belarusiska tidskriften Nasja niva (1914–1915). 1928 blev han ledamot av den belarusiska och ukrainska vetenskapsakademin.\n\nHans tidiga diktning var patriotisk idealiserade den östslaviska statsbildningen i Polotsk under 900- till 1200-talet som ett slags vision för Belarus. Han var även en hård kritiker av både det tidiga polsk-litauiska och ryska väldet över Belarus, varför många av hans verk förbjöds av sovjetregimen. På grund av den politiska förföljelsen under Stalin försökte han 1930 begå självmord och därefter blev han mindre produktiv som författare. Under de sista årtiondena var hans diktning en lång hyllning till socialismen och sovjetmakten. 1941 fick han ta emot Leninorden för sin diktsamling Ад сэрца (1940).\n\nVid Nazitysklands ockupation av Vitryska SSR 1941 flyttade han till Moskva och senare till Tatarstan. \n \nHans fru grundade ett museum över honom i Minsk där många av hans verk finns samlade. Staden Hrodna namngav ett universitet efter honom Janka Kupala Statsuniversitet 1978.\n\nBibliografi i urval \n Sjalejka 1908 (diktsamling)\n Husljar 1910 (diktsamling)\n Advetsjnaja pesnja 1910 (poem)\n Paulinka 1912 (skådespel)\n Sjljacham sjytstsia 1913 (diktsamling)\n Son na kurgane 1913 (poem)\n Raskidanaje hnjazdo 1913 (skådespel)\n\nKällor\n\nNoter\n\nBelarusiska författare\nBelarusiskspråkiga författare\nSovjetiska författare\nPersoner från Minsk voblast\nMän\nFödda 1882\nAvlidna 1942",
    "question": "Vilket datum är Janka Kupalas födelsedag?",
    "answers": {
        "answer_start": array([59]),
        "text": array(["7 juli 1882"], dtype=object)
    }
}
```

```json
{
    "context": "Storsäl (Erignathus barbatus) är en sälart som lever i Norra ishavet.\n\nUtseende och anatomi \n\nStorsälen har gråbrun päls som är ljusare på buken än på ryggen. På vintern får den ett mycket tjock fettskikt så att huvudet ser ovanligt litet ut. Vikten är på vintern omkring 360\xa0kg (ibland upp till 430\xa0kg) och på sommaren ungefär 230\xa0kg. Djuret är vanligen mellan 230 och 250\xa0cm långt och har ett långt vitt skägg. Bröstfenorna har en kännetecknande fyrkantig form. Mellan mars och augusti byter individerna pälsens hår.\n\nUtbredning \n\nStorsälen förekommer på isflaken i hela Arktis. Många individer lever i Berings hav. Under vandringen händer det ibland att några djur simmar fel så att de kommer till europeiska kustlinjer. En gång har djuret observerats i norra Portugal. Liknande iakttagelser rapporterades i norra Kina och från den japanska ön Hokkaido.\n\nEkologi \n\nStorsälen lever mestadels ensam. De vistas alltid i närheten av vattnet, så de kan flytta sig när en isbjörn närmar sig. Djuret kan dyka till 200\xa0meters djup men föredrar att förbli i närheten av havsytan. Under sommaren när antalet isflak minskar vilar den ibland på land. Med skägget letar den på havsbotten efter räkor, musslor och snäckor. Dessutom ingår fiskar i födan.\n\nHannen skapar under vattnet ett ljud som liknar valarnas sång. Troligtvis är ljudet till för att skydda sälens territorium eller för att imponera på honan.\n\nHonan är dräktig i omkring elva månader och föder i april eller maj ett ungdjur. Under dessa elva månader stannar embryots utveckling av en tid så att ungen inte föds för tidig. Kuten väger vid födelsen omkring 34\xa0kg. Honan ger di till kuten två till tre veckor (18 till 24 dagar) och lämnar den sedan ensam på isen. Oftast kan kuten simma redan vid denna ålder. Unga honor blir efter 3 till 8 år könsmogna och unga hannar efter 6 till 7 år. Vanligen blir storsälar inte äldre än 25 år men enskilda individer med en livslängd på 31 år är dokumenterade.\n\nStorsälar jagas aktiv av isbjörnar och de dödas ibland av späckhuggare. Sällsynt faller ungar offer för valross.\n\nStorsäl och människan \n\nJakt på storsäl för kött och hud har bedrivits länge. Men storsälen lever inte i flockar och är därför inte lika lättjagad som andra sälarter. Huden används till exempel för umiak och skor. Själva pälsen är inte eftertraktad. Under senare 1900-talet och början av 2000-talet uppskattades antalet dödade individer per år till 6\xa0800 i Alaska, 2\xa0400 i Kanada och 500 till 1\xa0000 på Grönland. En mera omfattande jakt skedde efter andra världskriget i de sovjetiska delarna av Arktiska havet, där upp till 13\xa0000 individer dödades per år. När arten blev sällsynt under 1970-talet minskade jakten betydlig. Den sovjetiska/ryska fångsten under 1980-talet uppgick bara till 2\xa0000 individer per år. IUCN listar arten som livskraftig (LC) på grund av det stora utbredningsområdet och eftersom beståndsutvecklingen bedöms som stabil.\n\nNoter \n\nÖronlösa sälar\nDäggdjur i palearktiska regionen\nDäggdjur i nearktiska regionen",
    "question": "Vilken bevarandestatus har storsälen enligt IUCN?",
    "answers": {
        "answer_start": array([2804]),
        "text": array(["livskraftig (LC)"], dtype=object)
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Nedan följer texter med tillhörande frågor och svar.
  ```

- Base prompt template:

  ```text
  Text: {text}
  Fråga: {question}
  Svar på max 3 ord: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Text: {text}

  Besvara följande fråga om texten ovan med högst 3 ord.

  Fråga: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-sv
```

### Unofficial: ScandiQA-sv

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

  ```text
  Nedan följer texter med tillhörande frågor och svar.
  ```

- Base prompt template:

  ```text
  Text: {text}
  Fråga: {question}
  Svar på max 3 ord: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Text: {text}

  Besvara följande fråga om texten ovan med högst 3 ord.

  Fråga: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scandiqa-sv
```

### Unofficial: BeleBele-sv

This dataset was published in [this paper](https://aclanthology.org/2024.acl-long.44/)
and features multiple-choice reading comprehension questions across 122 languages.

The original dataset contains 900 unique multiple-choice reading comprehension passages
and questions. From these, we use a 256 / 64 / 580 split for training, validation and
testing, respectively.

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

  ```text
  Följande är flervalsfrågor (med svar).
  ```

- Base prompt template:

  ```text
  Fråga: {text}
  Svarsalternativ:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svar: {label}
  ```

- Instruction-tuned prompt template:

  ```text
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
euroeval --model <model-id> --dataset belebele-sv
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

  ```text
  Följande är flervalsfrågor (med svar).
  ```

- Base prompt template:

  ```text
  Fråga: {text}
  Svar: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Fråga: {text}

  Besvara följande fråga med 'a', 'b', 'c' eller 'd', och inget annat.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset mmlu-sv
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

  ```text
  Följande är flervalsfrågor (med svar).
  ```

- Base prompt template:

  ```text
  Fråga: {text}
  Svarsalternativ:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svar: {label}
  ```

- Instruction-tuned prompt template:

  ```text
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
euroeval --model <model-id> --dataset arc-sv
```

### Unofficial: Skolprov

This dataset contains data from six Swedish knowledge tests and was published at [this
HuggingFace repository](https://huggingface.co/datasets/Ekgren/swedish_skolprov). The
dataset features multiple-choice questions from official Swedish examinations including
the Swedish Scholastic Aptitude Test (högskoleprovet), medical doctor test (kunskapsprov
läkare), dentist test (kunskapsprov tandläkare), audiologist test (kunskapsprov
audionom), pharmacist test (kunskapsprov apotekare), and mathematics and physics test
(matematik och fysikprovet).

The original dataset consists of 545 samples, which we filter down to 474 samples. We
use a 32 / 32 / 410 split for training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "En man, 63 år, har en skelett-, lymfkörtel-, och levermetastaserad prostatacancer där aktiv onkologisk behandling är avslutad. Han är inskriven i ett specialiserat palliativt hemsjukvårdsteam. Han har en långverkande smärtlindring med T. oxykodon 20 mg morgon och kväll. Nu söker han akut vård på grund av buksmärta och ihållande kräkningar. Vad bör du ordinera mot mannens smärtor?\nSvarsalternativ:\na. Tablett kortverkande oxykodon 5 mg peroralt\nb. Tablett ibuprofen 400 mg peroralt\nc. Tablett kortverkande oxykodon 20 mg peroralt\nd. Inj. 5 mg kortverkande oxykodon subkutant\ne. Paracetamol 1 g intravenöst",
  "label": "d"
}
```

```json
{
  "text": "Vilken typ av hörapparat är att föredra för barn under 3 år med en permenent sensorineural hörselnedsättning?\nSvarsalternativ:\na. RIC (reciever in the canal) med öppen dome\nb. benledd hörapparat\nc. bakom-örat-apparat med individuellt anpassad insats\nd. endast mikrofonsystem för att förbättra signal/brus förhållandet",
  "label": "c"
}
```

```json
{
  "text": "Varför är det viktigt med utredande samtal?\nSvarsalternativ:\na. Det kan etablera ett ömsesidigt förtroende mellan audionom och patient som underlag inför utförandet av en individuell, kvalitativ rehabilitering.\nb. Det ger audionomen möjlighet att observera patientens kommunikation som underlag inför utförandet av en individuell, kvalitativ rehabilitering.\nc. Det kan förbättra taluppfattningen som en del i utförandet av en individuell, kvalitativ rehabilitering.\nd. Det samlar nödvändig information om patienten, som underlag inför utförandet av en individuell, kvalitativ rehabilitering.",
  "label": "d"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Följande är flervalsfrågor (med svar).
  ```

- Base prompt template:

  ```text
  Fråga: {text}
  Svarsalternativ:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svar: {label}
  ```

- Instruction-tuned prompt template:

  ```text
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
euroeval --model <model-id> --dataset skolprov
```

### Unofficial: SwedishFacts

This is a benchmark for factual knowledge about Sweden.
The questions are based on topics related to the hosts of the Swedish radio program
[Sommar i P1](https://www.sverigesradio.se/sommar-i-p1) as well as Swedish sporting
events, such as those featured in [En Svensk Klassiker](https://ensvenskklassiker.se).
In the [dataset card](https://huggingface.co/datasets/liu-nlp/swedish-facts-v1)
it is mentioned that a paper with more information is coming soon.

Since the dataset does not include candidate answers, we generate them using GPT-4o.
The original dataset consists of 1,289 samples. We
use a 128 / 64 / 1,097 split for training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Hur många gånger befodrades Micael Bydén till en högre militär grad under 1990-talet?\nSvarsalternativ:\na. Tre, 3\nb. Fyra\nc. Fem\nd. Två",
  "label": "a"
}
```

```json
{
  "text": "Vad heter skivbolaget Titiyo Jah kontrakt med år 1988?\nSvarsalternativ:\na. Virgin Records\nb. Telegram\nc. Sony Music\nd. Warner Music",
  "label": "b"
}
```

```json
{
  "text": "I vilken ort föddes PM Nilsson?\nSvarsalternativ:\na. Göteborg\nb. Lund\nc. Helsingborg\nd. Malmö",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Följande är flervalsfrågor (med svar).
  ```

- Base prompt template:

  ```text
  Fråga: {text}
  Svarsalternativ:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svar: {label}
  ```

- Instruction-tuned prompt template:

  ```text
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
euroeval --model <model-id> --dataset swedish-facts
```

### Unofficial: MultiLoKo-sv

This dataset was published in [this paper](https://arxiv.org/abs/2504.10356) and is part
of MultiLoKo, a multilingual local knowledge benchmark covering 31 languages. The Swedish
questions are separately sourced and designed to target locally relevant topics for
Swedish-speaking populations.

We use the 'dev' split (250 samples) from this dataset. The dataset contains open-ended
questions with correct answers in the 'targets' column. We use the first target answer as
the correct option and use GPT-4.1 to generate 3 plausible but incorrect alternatives per
question. We create a 16 / 234 split for training and testing, respectively.

Here are a few examples from the training split:

```json
{
    "text": "Vilket språk talades när tv-serien Der Kommissar und das Meer visades i Sverige?\nSvarsalternativ:\na. Svenska\nb. Tyska\nc. Engelska\nd. Franska",
    "label": "a"
}
```

```json
{
    "text": "Felix Sandman och Farah Abadi var programledare för musikhjälpen. Vem var den tredje som deltog?\nSvarsalternativ:\na. Molly Sandén\nb. Zackari\nc. Gina Dirawi\nd. Oscar Zia",
    "label": "b"
}
```

```json
{
    "text": "Vilket var det tredje landet utanför Sverige där Robin Olsen spelade klubblagsfotboll?\nSvarsalternativ:\na. Spanien\nb. Frankrike\nc. Italien\nd. Tyskland",
    "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Följande är flervalsfrågor (med svar).
  ```

- Base prompt template:

  ```text
  Fråga: {text}
  Svar: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Fråga: {text}

  Besvara följande fråga med 'a', 'b', 'c' eller 'd', och inget annat.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multiloko-sv
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

  ```text
  Följande är flervalsfrågor (med svar).
  ```

- Base prompt template:

  ```text
  Fråga: {text}
  Svarsalternativ:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svar: {label}
  ```

- Instruction-tuned prompt template:

  ```text
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
euroeval --model <model-id> --dataset hellaswag-sv
```

### Unofficial: GoldenSwag-sv

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
  "text": "Hur man staplar hö. Placera dina pallar på en tillgänglig plats. Det måste vara lätt att ta sig till staplarna, så välj en plats som du kan nå utan problem. Undvik att stapla balar direkt på marken.\nSvarsalternativ:\na. Håll balarna i detta område inom armlängds avstånd så att du inte skadar dig själv och andra i byggnaden. Tänk på att det bör finnas ca 6 balar så att varje person i rummet ska kunna komma åt en bal.\nb. Undvik att stapla balar på asfalterade ytor. Hitta en ojämn yta att stapla pallar på.\nc. Dina pallar måste vara lådliknande och hålla dina ben ut åt sidorna, samt stödja din fulla vikt. Använd betongblock om du har tillgång till sådana.\nd. Höet suger åt sig fukt och blir mögligt. För att förhindra detta, använd träpallar som grund.",
  "label": "d"
}
```

```json
{
  "text": "Hur du väljer vem som ska följa dig till altaret. Identifiera den viktigaste familjemedlemmen i ditt liv. Det kan vara bra att börja med att fundera på vem som är den viktigaste familjemedlemmen i ditt liv och sedan fundera på att be den personen att följa dig till altaret. Du kanske anser att din bror är den viktigaste personen i ditt liv.\nSvarsalternativ:\na. Eller så tänker du på din mamma, vars liv du älskade mest. Att identifiera din familjemedlem kan hjälpa dig att gå igenom några av de mer upplyftande stunderna, eftersom din familjemedlem sannolikt kan vara din make eller sambo.\nb. Kanske är din bror din pappas bästa vän och din mamma har varit din mammas bästa vän under en mycket lång tid. Att göra en lista över dessa personer kan hjälpa dig att förstå varför din mamma är viktig för dig och vad som motiverar henne att följa dig till altaret.\nc. Eller så kanske den första personen som dyker upp i ditt huvud är din ensamstående mamma som uppfostrade dig på egen hand. Du kan skriva ner några personer som är viktiga för dig i din familj på ett papper och sedan välja en från listan.\nd. Eller så kanske du väljer din mammas pappa som din hedersbrudtärna. En lista kan dyka upp framför dig när du organiserar dina kalendrar och möten, så det är viktigt att hitta några ord som tydligt hjälper dig.",
  "label": "c"
}
```

```json
{
  "text": "Hur man får gratis uppgraderingar på smekmånaden. Registrera dig för medlemskort för frequent flier miles. Om du har ett favoritflygbolag, registrera dig för dess bonusprogram så snart du kan, särskilt om du gör många affärsresor. Frekventa flygmil kan snabbt läggas ihop och leda till gratisbiljetter och uppgraderingar.\nSvarsalternativ:\na. Vissa flygbolag tillåter till och med att dina vänner och familj ger dig sina miles, så uppmuntra dem att också registrera sig.. Fråga ditt kreditkortsföretag om incitament.\nb. Välj en destination som du är villig att spendera pengar på. Det är en bra idé att prova några destinationer som du skulle älska att besöka, inklusive Japan, som öppnar upp ekonomiska möjligheter omedelbart.\nc. Skicka uppgifterna till det flygbolag du föredrar. Om du har för avsikt att använda bonuspoäng för affärsändamål ska du skicka uppgifterna till ditt favoritflygbolag på något av följande sätt.\nd. Besök webbplatsen för det flygbolag som arrangerar flygningar för dig, eller leta online för att hitta en resplan som specificerar miles. Förutom gratis resor kan du också använda back to back-bokningstjänster.",
  "label": "a"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Följande är flervalsfrågor (med svar).
  ```

- Base prompt template:

  ```text
  Fråga: {text}
  Svarsalternativ:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svar: {label}
  ```

- Instruction-tuned prompt template:

  ```text
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
euroeval --model <model-id> --dataset goldenswag-sv
```

### Unofficial: Winogrande-sv

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2506.19468)
and is a translated and filtered version of the English [Winogrande
dataset](https://doi.org/10.1145/3474381).

The original full dataset consists of 47 / 1,210 samples for training and testing, and
we use 128 of the test samples for validation, resulting in a 47 / 128 / 1,085 split for
training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Sushin ruttnade på disken om den inte placerades i kylen, eftersom _ utsatte den för kontaminering. Vad syftar tomrummet _ på?\nSvarsalternativ:\na. Alternativ A: disken\nb. Alternativ B: kylen",
  "label": "a"
}
```

```json
{
  "text": "Elena skulle ta deras lager i butikens baksida för Megan att sälja varje gång eftersom _ var en affärsperson. Vad syftar tomrummet _ på?\nSvarsalternativ:\na. Alternativ A: Elena\nb. Alternativ B: Megan",
  "label": "a"
}
```

```json
{
  "text": "Att hantera nödsituationer var aldrig särskilt svårt för Kevin men det var det för Nelson eftersom _ inte kunde förbli lugn under press. Vad syftar tomrummet _ på?\nSvarsalternativ:\na. Alternativ A: Kevin\nb. Alternativ B: Nelson",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Följande är flervalsfrågor (med svar).
  ```

- Base prompt template:

  ```text
  Fråga: {text}
  Svarsalternativ:
  a. {option_a}
  b. {option_b}
  Svar: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Fråga: {text}
  Svarsalternativ:
  a. {option_a}
  b. {option_b}

  Besvara följande fråga med 'a' eller 'b', och inget annat.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset winogrande-sv
```

## Summarisation

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

  ```text
  Nedan följer artiklar med tillhörande sammanfattningar.
  ```

- Base prompt template:

  ```text
  Artikel: {text}
  Sammanfattning: {target_text}
  ```

- Instruction-tuned prompt template:

  ```text
  Artikel: {text}

  Skriv en sammanfattning av artikeln ovan.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset swedn
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

  ```text
  Nedan följer artiklar med tillhörande sammanfattningar.
  ```

- Base prompt template:

  ```text
  Artikel: {text}
  Sammanfattning: {target_text}
  ```

- Instruction-tuned prompt template:

  ```text
  Artikel: {text}

  Skriv en sammanfattning av artikeln ovan.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset schibsted-sv
```

### Unofficial: SVD SEO Title

This dataset was published
[here](https://huggingface.co/datasets/Schibsted/svd-seo-title) and features SEO titles
of news articles from Schibsted Media's Swedish newsroom, from Svenska Dagbladet (SVD).

Here are a few examples from the training split:

```json
{
    "article_id": "https://www.svd.se/a/9z5KQM/",
    "text": "Med en pistol mot huvudet – om han tvingas välja – så blir det Biden. Sverigedemokraten Mattias Karlsson har fått nog av republikaner, Trump och ”Make America great again”. Mattias Karlsson sitter i fiket. Han är deprimerad. I förrgår kom han hem från Washington, och där är ingenting roligt längre.  – Det är fruktansvärt illa ställt. Han tar en snus. – Ingen vet var det ska ta vägen. Det känns som att vad som helst kan hända med USA och med det republikanska partiet. – Men, säger jag, gillade inte du den amerikanska rörelsen, gillade inte du ”Make America great again”? – Jag vet inte. Hans blick är nere i bordet. – Jag vet inte vad den här rörelsen är längre. Jag tänker på 2016. När Trump valdes till president fanns det ett par sverigedemokrater som öppet sympatiserade med honom. Mikael Jansson, den förre partiledaren, som satt i försvarsutskottet och Gustav Kasselstrand, som nyss varit ordförande i ungdomsförbundet. Jansson försvann snart också från partiet. Många svenskar förfärades sedan över vad Trump gjorde och sa som president. Fast då blev fler och fler sverigedemokrater nyfikna. Han som så länge har kallats för partiets chefsideolog blev rentav imponerad. Jag har aldrig trott på Trump som person, jag tål inte karln. När Trump skulle återväljas, 2020, sa Mattias Karlsson i SVT: – Jag stöttar den republikanska sidan därför att det är två världsbilder som står mot varandra, och det är viktigare än de här två personerna som det pratas så mycket om. Han sa: – Det är en reell konflikt som man kan se över hela västvärlden. Det handlar om nationell suveränitet kontra mer globalism, om patriotism kontra mångkulturalism. Nu byter han snus. Det är en trött onsdagseftermiddag i riksdagsfiket. – Jag har aldrig trott på Trump som person, säger han, jag tål inte karln. Men jag trodde på rörelsen och på positionsförflyttningen. Republikanerna lämnade neoliberala ekonomiska teorier och aggressiv utrikespolitik, partiet fick ett mer kommunitärt drag. Jag gillade det. Att man ville ha in arbetarklassen i ett konservativt projekt, landsbygden, vi var alla ”somewheres” och mot globalism. Det fanns en bra grund för en enad rörelse. – Och nu? – Det blev splittring redan under coronan, det dök upp en massa konspirationsteorier. Och nu är det geopolitiken, och där är det nästan revolutionär kursomläggning. Stora delar av den amerikanska högern har släppt idén om sig själv som demokratins banerförare, den har släppt idén om väst.  – Väst? – Ja? – När började du bry dig om väst? Han skrattar. – Det var inget centralt begrepp i min idévärld när jag började i politiken. Det har väl växt fram. Jag läste Roger Scruton, vi som parti började resa mer. Sverigedemokraterna är ett parti som länge inte hade någon utrikespolitik. Ett parti vars första internationella partner hette Le Pen. Ett parti där människor avskydde USA, eftersom det var ett invandrarland med mångkultur. Sverigedemokraterna har aldrig varit ett transatlantiskt parti. – Vad är väst för dig, Mattias? frågar jag. – Det är en uppsättning filosofiska idéer som går tillbaka till Rom och Grekland. Kristendomsarvet. Klassisk musik, klassisk arkitektur, konst. I viss mån delade högtider och traditioner. – Är det geografiskt? Det finns ingen annan allians som försvarar väst.  – Jag ser det inte så primärt, men om vi som parti från början bara tog in Norden så har vi sedan gått längre. Vi är en del av västerlandet och västerlandet är Europa och USA. – Är Turkiet en del av väst? – Nej, det skulle jag inte säga. – Så varför vill du gå in i en försvarsallians med Turkiet? – Det finns ingen annan allians som försvarar väst.  Det är något konstigt i det här samtalet, tänker jag",
    "keyword": "trump",
    "target_text": "Torbjörn Nilsson: SD:s Mattias Karlsson väljer Biden, inte Trump",
    "newsroom": "SVD"
}
```

```json
{
    "article_id": "https://www.svd.se/a/abgaR5/",
    "text": "Både regionstyret och oppositionen i Stockholm vill snabbutreda det medicinska behovet av intensivvårdsrehabilitering på privata Remeokliniken – vars framtid är hotad. ”Vi vill ha en snabb genomgång av läget just nu”, säger sjukvårdsregionrådet Talla Alkurdi (S). Under tisdagen var den laddade frågan uppe i hälso- och sjukvårdsnämnden i region Stockholm om vad man ska göra angående den uppkomna situationen på Remeokliniken i Stora Sköndal. Beslutet blev att både regionstyret och oppositionens skrivelser gick vidare till  förvaltningen för skyndsam hantering. Vänsterpartiets regionråd Jonas Lindberg, som normalt samarbetar tätt med mittkoalitionen med S, MP och C, har annars stuckit ut hakan rejält just i den här frågan. Han var beredd att tillsammans med bland andra Moderaterna rädda ett privat bolag från konkurs. – Det handlar inte om att rädda bolaget utan om vården för svårt sjuka patienter, säger han. De senaste fem åren har Remeokliniken sammanlagt fått över en halv miljard kronor för den intensivvårdsrehabilitering som man sålt till främst Karolinska i Solna och Huddinge. Men nu har beställningarna mattats av och Remeoklinikens framtid är hotad – vilket lett till politisk strid. V och tre av de borgerliga partierna (M, KD, L) har tidigare tagit initiativ till att utreda klinikens framtid både på kort och lång sikt. Ett av utredningsuppdragen handlar om att regionen skriver ett direkt avtal med Remeo. Styret lägger fram nytt förslag Inför hotet att röstas ned under tisdagens möte valde den styrande mittenkoalitionen att lägga fram ett liknande utredningsförslag. Dock med den skillnaden att förvaltningen inte ska undersöka möjligheterna till ett regionavtal med Remeo. – Det skulle vara en direktupphandling som kan vara olaglig. Vi vill att juristerna klarlägger hur vi kan använda oss av kliniken om vi behöver den av medicinska skäl, säger sjukvårdsregionrådet Talla Alkurdi (S). Hon fortsätter: – Det finns ju även andra uppgifter som säger att regionen och Nya Karolinska delvis klarar behovet av platser utan att anlita Remeo. Charlotte Broberg (M) tycker att S spelar bort korten då de inte ser vikten av den livsnödvändiga vård som Remeo bedriver. Det är bedrövligt att S försöker lura stockholmarna på det här sättet, säger Charlotte Broberg (M) och tillägger: - Vi föreslår en direktupphandling, ett övertagande av kostnadsansvar från akutsjukhusen eller en annan laglig modell för att rädda Remeokliniken. Att S försöker skjutsa dessa patienter under mattan och påstå att den här vården skulle finnas på våra akutsjukhus idag är inget annat än lögn och båg, säger Charlotte Broberg (M). SD valde i sista stund att ansluta sig till förslaget från S, MP och C, därmed fanns det egentligen majoritet för den skrivelsen. Men beslutet blev alltså att skicka båda skrivelserna. – Vi har satt tryck i frågan och har nu en enig politik, genom två initiativ, signalerat tydligt att denna vård måste säkerställas så att regionen inte tappar någon intensivvårdskapacitet, kommenterade Jonas Lindberg (V). Kajsa Giesecke är narkosläkare och tidigare verksamhetschef på privata Remeokliniken i Stora Sköndal, som hon var med och startade. I mer än tio år har kliniken avlastat den underdimensionerade intensivvården i Stockholm, vilket inte minst visade sig under pandemin.  Här finns 14 vårdplatser för långvarigt kritiskt sjuka, svaga patienter som legat länge på någon av sjukhusens intensivvårdskliniker. Det kallas intensivvårdsrehabilitering och kan exempelvis vara patienter som inte klarar att andas själva, som behöver andningsstöd med respirator. Normalt är kliniken fullbelagd men sedan i november finns det flera lediga sängar, i dagsläget fem lediga vårdplatser. Anledningen är att Karolinska i Solna och Huddinge remitterat färre patienter än vanligt till Remeokliniken",
    "keyword": "remeokliniken",
    "target_text": "Nya bud kring Remeokliniken",
    "newsroom": "SVD"
}
```

```json
{
    "article_id": "https://www.svd.se/a/nQz9md/",
    "text": "Skildringen av en nyförlöst mors oro har vissa drag av Netflixdrama, och översättningen haltar något – men hajpade ”Mjölkbaren” är ändå riktigt bra, tycker Emi-Simone Zawall. I novellen ”Den gula tapeten” av Charlotte Perkins Gilman (1892) har en nyförlöst kvinna tillfälligt flyttat in i ett stort hus på landet tillsammans med sin make John. John har bestämt att de ska bo i husets väldiga barnkammare (the nursery), men eftersom han är på sitt arbete om dagarna lämnas hon mest åt sig själv. Barnet sköts av en hushållerska, tanken är att modern ska bota sin underliga nedstämdhet med en fullständig passivitet. De sysslolösa dagarna går och barnkammarens gula tapet börjar allt mer uppta hennes uppmärksamhet. Varför är den så ful, undrar hon, och varför har den så konstiga mönster? Och en dag: varför rör mönstren på sig? En annan dag: varför är en kvinna instängd bakom tapeten? Och senare: varför är det flera kvinnor som kryper omkring där? Hennes kusliga syner och tilltagande galenskap har av eftervärlden tolkats som en kritik av kvinnans plats i det patriarkala systemets fängelse. Men i sin debutroman ”The Nursery”, som nyligen översatts till svenska med titeln ”Mjölkbaren” (Bakhåll), har Szilvia Molnar tagit fasta på att det faktiskt är en förlossningspsykos som skildras. I Molnars moderna parafras bor berättarjaget i en stad någonstans i USA, arbetar som översättare av svensk skönlitteratur och har nyss fött barn, lilla Ärtan. Maken John är snäll och omtänksam men hänvisas snabbt efter förlossningen till sitt arbete. Den första tiden med barnet blir alltså en ensam, snudd på klaustrofobisk upplevelse. Kroppen värker, tvivlen växer, barnet skriker, utmattningen är gränslös – ändå måste barnet ammas, bytas på, vyssjas. Hemmets fyra väggar tränger sig på, minnen från det förflutna gör sig påminda och vem är egentligen den märklige grannen Peter med syrgastanken som insisterar på att komma på besök? Molnar viker aldrig med blicken från den kroppsliga chock och sipprande sörja som en förlossning innebär eller de fullkomligt bisarra tankar en nyförlöst kvinna kan drabbas av. Bildspråket är vackert och drastiskt, en bebis kan ha tulpanens tunga huvud eller liknas vid en glänsande färsk kycklingkropp som snart ska styckas. Man förstår att Molnar förstår att det är litteratur hon skriver, varken loggbok eller dagbok: här får overklighetens lungsjuka män, spindlar, och dragspelsmusik tränga in och göra verkligheten verkligare – det är så uppfriskande att hon tar sig an sitt ämne på det sättet. Det gör i sin tur att man förlåter romanens lätta dragning åt den Netflixartade dramaturgi som dagens samtidslitteratur blivit så impregnerad av – alltså den litterära motsvarigheten till tv-seriernas många sekvenser med ”förklarande återblickar”, ”dröjande närbilder” av ”målande detaljer”; en stil som låter läsaren se framför sig hur romanfigurernas gester skulle utföras av skådespelare i en framtida filmatisering. En liten invändning kan också riktas mot den smått koketta retorik som hos Molnar visserligen är i det mildaste laget men i övrigt verkar inbyggd i erfarenhetsgenren som sådan, och som kan sammanfattas i frågan: varför har ingen annan skrivit om det jag skriver om? Det är kanske sant (som Molnar påstår) att hela världslitteraturen är renons på beskrivningar av blöjbyten eller de förtärande tomma timmar där man bara sitter och håller ett sovande eller skrikande barn i famnen.  Men måste det påstås? Räcker det inte med att bara skapa det som inte finns? Författare som själva framhåller sin egen betydelse är en olat",
    "keyword": "mjölkbaren",
    "target_text": "Recension: Szilvia Molnars Mjölkbaren är riktigt bra",
    "newsroom": "SVD"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:

  ```text
  Nedan följer artiklar med tillhörande SEO-rubriker.
  ```

- Base prompt template:

  ```text
  Artikel: {text}
  SEO-rubrik: {target_text}
  ```

- Instruction-tuned prompt template:

  ```text
  Artikel: {text}

  Skriv en SEO-rubrik för ovanstående artikel.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset svd-seo-title
```

## Instruction-following

### IFEval-sv

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
    "text": "Förklara för mig hur man cyklar som om jag vore ett barn. Inkludera inte heller nyckelorden \"långsam\", \"som\" och \"barn\".",
    "target_text": {
        "instruction_id_list": [
            "keywords:forbidden_words",
            "language:response_language"
        ],
        "kwargs": [
            {
                "forbidden_words": [
                    "långsam",
                    "som",
                    "barn"
                ],
            },
            {
                "language": "sv"
            }
        ]
    }
}
```

```json
{
    "text": "Vilka är fördelarna och nackdelarna med att ha övernaturliga krafter? Gör det kort. Packa in hela utdata i JSON-format. Du kan använda markdown-ticks som ```.",
    "target_text": {
        "instruction_id_list": [
            "detectable_format:json_format",
            "language:response_language"
        ],
        "kwargs": [
            {},
            {
                "language": "sv"
            }
        ]
    }
}
```

```json
{
    "text": "Skriv en gåta om Camilla utan att använda kommatecken.",
    "target_text": {
        "instruction_id_list": [
            "punctuation:no_comma",
            "language:response_language"
        ],
        "kwargs": [
            {},
            {
                "language": "sv"
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
euroeval --model <model-id> --dataset ifeval-sv
```

## European Values

### ValEU-sv

This dataset is the official Swedish version of questions from the [European values
study](https://europeanvaluesstudy.eu/). The dataset contains multiple-choice
questions regarding people's values and beliefs across a variety of topics, such as
politics, religion and society.

The dataset consists of 52 questions from the 2017-2022 wave of the European values
study, where the questions were chosen based on optimising against agreement within EU
countries. We use only zero-shot evaluation on this dataset, and thus require no splits.

Here are a few examples from the training split:

```json
{
  "question_id": "E265_01",
  "text": "Enligt dig hur ofta förekommer följande saker vid val i Sverige?\nAtt röster räknas korrekt\nSvarsalternativ:\na. Väldigt ofta\nb. Ganska ofta\nc. Inte ofta\nd. Inte alls ofta"
}
```

```json
{
  "question_id": "E114",
  "text": "Jag skall nu beskriva några olika politiska system för att styra Sverige. Anser du att de är mycket bra, ganska bra, ganska dåliga eller mycket dåliga?\nAtt ha en stark ledare som inte behöver bekymra sig om riksdag och politiska val?\nSvarsalternativ:\na. Mycket bra\nb. Ganska bra\nc. Ganska dåliga\nd. Mycket dåliga"
}
```

```json
{
  "question_id": "D078",
  "text": "För vart och ett av de påståenden som jag läser upp, kan du vänligen ange om du instämmer eller tar avstånd från. Instämmer du helt och hållet, instämmer du, tar du avstånd från, eller tar du helt och hållet avstånd från?\nI stort sett är män bättre lämpade att vara företagsledare än kvinnor\nSvarsalternativ:\na. Instämmer helt och hållet\nb. Instämmer\nc. Tar avstånd från\nd. Tar helt och hållet avstånd från"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 0
- Prefix prompt:

  ```text
  Följande är flervalsfrågor (med svar).
  ```

- Base prompt template:

  ```text
  Fråga: {text}
  Svarsalternativ:
  a. {option_a}
  b. {option_b}
  (...)
  k. {option_k}
  Svar: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Fråga: {text}
  Svarsalternativ:
  a. {option_a}
  b. {option_b}
  (...)
  k. {option_k}

  Besvara följande fråga med 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j' eller 'k',
  och inget annat.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset valeu-sv
```

## Grammatical Error Detection

### Unofficial: GerLangMod-sv

This dataset is based on the [GerLangMod](https://github.com/noahmanu/gerlangmod)
collection and derived from the Swedish Universal Dependencies treebank. Assuming UD
annotations are accurate and sentences are well-formed, the dataset contains permuted
versions of these UD sentences where half of the verbs have been misplaced within their
phrase boundaries. Noun-headed groups of tokens are treated as impermeable units so
misplaced verbs cannot split them up, and no verb can be placed in the first position of
the first phrase of each sentence to avoid creating correct polar question syntax.

The original dataset consists of 11,485 samples derived from the
[UD_Swedish-Talbanken](https://github.com/UniversalDependencies/UD_Swedish-Talbanken),
[UD_Swedish-LinES](https://github.com/UniversalDependencies/UD_Swedish-LinES) and
[UD_Swedish-PUD](https://github.com/UniversalDependencies/UD_Swedish-PUD) treebanks.
We use a sample of 1,024 / 256 / 2,048 of these for training, validation and testing,
respectively.

Here are a few examples from the training split:

```json
{
    "tokens": [
        "mörkt",
        "regn",
        "på",
        "eftermiddagen",
        "i",
        "london",
        "när",
        "planet",
        "startade",
        "flygplatsen",
        "i",
        "rom",
        "ett",
        "väldigt",
        "surögt",
        "skyltfönster",
        "blanka",
        "flödiga",
        "färger",
        "genom",
        "regnet"
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
        "O",
        "O",
        "O",
        "O",
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
        "de",
        "flesta",
        "bilar",
        "i",
        "dag",
        "redan",
        "är",
        "från",
        "fabriken",
        "utrustade",
        "med",
        "radialdäck"
    ],
    "labels": [
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "B-ERR",
        "O",
        "O",
        "O",
        "O",
        "O"
    ]
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Nedan är meningar och JSON-ordböcker med de grammatiska fel som förekommer i den givna meningen.
  ```

- Base prompt template:

  ```text
  Mening: {text}
  Grammatiska fel: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Mening: {text}

  Identifiera de grammatiska felen i meningen. Du ska skriva ut detta som en JSON-ordbok med nyckeln 'fel'. Värdet ska vara en lista över felplacerade ord, precis som de visas i meningen.
  ```

- Label mapping:
  - `B-ERR` ➡️ `fel`
  - `I-ERR` ➡️ `fel`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset gerlangmod-sv
```

## Hallucination Detection

### MultiWikiHalluQA-sv

This dataset uses the same data as [MultiWikiQA-sv](#multiwikiqa-sv), published in
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
    "context": "Juan Mayorga, född 6 april 1965 i Madrid, är en spansk dramatiker och manusförfattare.\n\nBiografi\nJuan Mayorga har en examen i matematik och filosofi från Universidad Complutense de Madrid 1988. Därefter arbetade han som forskarassistent i filosofi vid Consejo Superior de Investigaciones Científicas. Fortsatta studier i Münster, Berlin och Paris ledde till en doktorsexamen i filosofi 1997 med en avhandling om Walter Benjamin. Han debuterade som dramatiker 1989 med Siete hombres buenos (Sju goda män). Tillsammans med dramatiker kollegorna José Ramón Fernández, Luis Miguel González Cruz och Raúl Hernández Garrido grundade han 1993 teatergruppen Teatro del Astillero. 2011 grundade han en ny teatergrupp, La Loca de la Casa. Sedan 1998 är Mayorga har han undervisat i filosofi och dramatik vid Escuela Superior de Arte Dramático i Madrid. För närvarande (2017) är han chef för avdelningen för scenkonst vid Universidad Carlos III de Madrid. Han har skrivit ett trettiotal pjäser (2017) och hans dramatik har spelats i 18 länder och översatts till 16 språk. Han eftersträvar en filosofiskt präglad teater som tvingar publiken till ställningstaganden. Till hans förebilder hör Harold Pinter. Bland utmärkelser han tilldelats kan nämnas de spanska priserna Nacional de Teatro 2007 och Nacional de Literatura Dramática 2013 samt Premio Europa New Theatrical Realities 2016.\n\n2007 regisserade Alexander Mørk-Eidem Mayorgas Himmelweg på Nationaltheatret i Oslo. 2014 skulle han även ha regisserat den på Stockholms stadsteater under titeln Himlavägen i Jens Nordenhöks översättning men det ställdes in.\n\nReferenser\n\nKällor\n Pressrelease, Premio Europa 14/3 2016 \n Juan Mayorga, The Playwrights Database (läst 5 april 2017)\n Juan Mayorga, France culture (läst 5 april 2017)\n Juan Mayorga, theatre-contemporain.net (läst 5 april 2017)\n Juan Mayorga, Théâtre de Rond-Point, Paris (läst 5 april 2017)\n Juan Mayorga, madridesteatro.com (läst 5 april 2017)\n Arkiv, Kulturhuset Stadsteatern (läst 5 april 2017)\n Lillian Bikset: Teater, løgn og bedrag, Dagbladet 31/8 2007\n Elisabeth Leinslie: Kjenn din besøkelsestid, Dagsavisen 2/9 2007\n Rocío García: Juan Mayorga: las obsesiones de un matemático y autor de éxito, El País 1/6 2016\n\nNoter\n\nExterna länkar\n Juan Mayorga, Internet Movie Database (IMDb)\n\nSpanska dramatiker\nSpanskspråkiga dramatiker\nSpanska manusförfattare\nSpanska författare under 1900-talet\nSpanska författare under 2000-talet\nDramatiker under 1900-talet\nDramatiker under 2000-talet\nPersoner från Madrid\nFödda 1965\nLevande personer\nMän",
    "question": "Vilka akademiska examina har Juan Mayorga avlagt vid Universidad Complutense de Madrid?",
    "answers": {
        "answer_start": array([126]),
        "text": array(["matematik och filosofi"], dtype=object)
    }
}
```

```json
{
    "context": "Janka Kupala (egentligen Ivan Daminikavitj Lutsevitj) född 7 juli 1882 i Vjazynka utanför Minsk, död 28 juni 1942 i Moskva, var en belarusisk författare. Tillsammans med Jakub Kolas räknas han som en av den moderna belarusiska litteraturens grundare.\n\nKupala var till stor del självlärd som författare. Han blev aktiv i den \"belarusiska pånyttfödelsen\" (1903–1921) och redaktör för den belarusiska tidskriften Nasja niva (1914–1915). 1928 blev han ledamot av den belarusiska och ukrainska vetenskapsakademin.\n\nHans tidiga diktning var patriotisk idealiserade den östslaviska statsbildningen i Polotsk under 900- till 1200-talet som ett slags vision för Belarus. Han var även en hård kritiker av både det tidiga polsk-litauiska och ryska väldet över Belarus, varför många av hans verk förbjöds av sovjetregimen. På grund av den politiska förföljelsen under Stalin försökte han 1930 begå självmord och därefter blev han mindre produktiv som författare. Under de sista årtiondena var hans diktning en lång hyllning till socialismen och sovjetmakten. 1941 fick han ta emot Leninorden för sin diktsamling Ад сэрца (1940).\n\nVid Nazitysklands ockupation av Vitryska SSR 1941 flyttade han till Moskva och senare till Tatarstan. \n \nHans fru grundade ett museum över honom i Minsk där många av hans verk finns samlade. Staden Hrodna namngav ett universitet efter honom Janka Kupala Statsuniversitet 1978.\n\nBibliografi i urval \n Sjalejka 1908 (diktsamling)\n Husljar 1910 (diktsamling)\n Advetsjnaja pesnja 1910 (poem)\n Paulinka 1912 (skådespel)\n Sjljacham sjytstsia 1913 (diktsamling)\n Son na kurgane 1913 (poem)\n Raskidanaje hnjazdo 1913 (skådespel)\n\nKällor\n\nNoter\n\nBelarusiska författare\nBelarusiskspråkiga författare\nSovjetiska författare\nPersoner från Minsk voblast\nMän\nFödda 1882\nAvlidna 1942",
    "question": "Vilket datum är Janka Kupalas födelsedag?",
    "answers": {
        "answer_start": array([59]),
        "text": array(["7 juli 1882"], dtype=object)
    }
}
```

```json
{
    "context": "Storsäl (Erignathus barbatus) är en sälart som lever i Norra ishavet.\n\nUtseende och anatomi \n\nStorsälen har gråbrun päls som är ljusare på buken än på ryggen. På vintern får den ett mycket tjock fettskikt så att huvudet ser ovanligt litet ut. Vikten är på vintern omkring 360\xa0kg (ibland upp till 430\xa0kg) och på sommaren ungefär 230\xa0kg. Djuret är vanligen mellan 230 och 250\xa0cm långt och har ett långt vitt skägg. Bröstfenorna har en kännetecknande fyrkantig form. Mellan mars och augusti byter individerna pälsens hår.\n\nUtbredning \n\nStorsälen förekommer på isflaken i hela Arktis. Många individer lever i Berings hav. Under vandringen händer det ibland att några djur simmar fel så att de kommer till europeiska kustlinjer. En gång har djuret observerats i norra Portugal. Liknande iakttagelser rapporterades i norra Kina och från den japanska ön Hokkaido.\n\nEkologi \n\nStorsälen lever mestadels ensam. De vistas alltid i närheten av vattnet, så de kan flytta sig när en isbjörn närmar sig. Djuret kan dyka till 200\xa0meters djup men föredrar att förbli i närheten av havsytan. Under sommaren när antalet isflak minskar vilar den ibland på land. Med skägget letar den på havsbotten efter räkor, musslor och snäckor. Dessutom ingår fiskar i födan.\n\nHannen skapar under vattnet ett ljud som liknar valarnas sång. Troligtvis är ljudet till för att skydda sälens territorium eller för att imponera på honan.\n\nHonan är dräktig i omkring elva månader och föder i april eller maj ett ungdjur. Under dessa elva månader stannar embryots utveckling av en tid så att ungen inte föds för tidig. Kuten väger vid födelsen omkring 34\xa0kg. Honan ger di till kuten två till tre veckor (18 till 24 dagar) och lämnar den sedan ensam på isen. Oftast kan kuten simma redan vid denna ålder. Unga honor blir efter 3 till 8 år könsmogna och unga hannar efter 6 till 7 år. Vanligen blir storsälar inte äldre än 25 år men enskilda individer med en livslängd på 31 år är dokumenterade.\n\nStorsälar jagas aktiv av isbjörnar och de dödas ibland av späckhuggare. Sällsynt faller ungar offer för valross.\n\nStorsäl och människan \n\nJakt på storsäl för kött och hud har bedrivits länge. Men storsälen lever inte i flockar och är därför inte lika lättjagad som andra sälarter. Huden används till exempel för umiak och skor. Själva pälsen är inte eftertraktad. Under senare 1900-talet och början av 2000-talet uppskattades antalet dödade individer per år till 6\xa0800 i Alaska, 2\xa0400 i Kanada och 500 till 1\xa0000 på Grönland. En mera omfattande jakt skedde efter andra världskriget i de sovjetiska delarna av Arktiska havet, där upp till 13\xa0000 individer dödades per år. När arten blev sällsynt under 1970-talet minskade jakten betydlig. Den sovjetiska/ryska fångsten under 1980-talet uppgick bara till 2\xa0000 individer per år. IUCN listar arten som livskraftig (LC) på grund av det stora utbredningsområdet och eftersom beståndsutvecklingen bedöms som stabil.\n\nNoter \n\nÖronlösa sälar\nDäggdjur i palearktiska regionen\nDäggdjur i nearktiska regionen",
    "question": "Vilken bevarandestatus har storsälen enligt IUCN?",
    "answers": {
        "answer_start": array([2804]),
        "text": array(["livskraftig (LC)"], dtype=object)
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Nedan följer texter med tillhörande frågor och svar.
  ```

- Base prompt template:

  ```text
  Text: {text}
  Fråga: {question}
  Svar på max 3 ord: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Text: {text}

  Besvara följande fråga om texten ovan med högst 3 ord.

  Fråga: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-hallucination-qa-sv
```
