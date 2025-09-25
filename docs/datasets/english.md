# 🇬🇧 English

This is an overview of all the datasets used in the English part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### SST-5

This dataset was published in [this paper](https://aclanthology.org/D13-1170/) and is
based on movie reviews from rottentomatoes.com, labelled by crowdsourced workers on
Amazon Mechanical Turk.

The original full dataset consists of 8,540 / 1,100 / 2,210 samples for the training,
validation and test splits, respectively. We use 1,024 / 256 / 2,048 samples for our
training, validation and test splits, respectively. All the new splits are subsets of
the original splits.

The original dataset consists of 5 labels instead of our usual 3, but we map them to
`positive`, `neutral` and `negative` as follows:

- `very negative` ➡️ `negative`
- `negative` ➡️ `negative`
- `neutral` ➡️ `neutral`
- `positive` ➡️ `positive`
- `very positive` ➡️ `positive`

Here are a few examples from the training split:

```json
{
  "text": "the leads are natural and lovely , the pace is serene , the humor wry and sprightly .",
  "label": "positive"
}
```

```json
{
  "text": "labute ca n't avoid a fatal mistake in the modern era : he 's changed the male academic from a lower-class brit to an american , a choice that upsets the novel 's exquisite balance and shreds the fabric of the film .",
  "label": "neutral"
}
```

```json
{
  "text": "no cliche escapes the perfervid treatment of gang warfare called ces wild .",
  "label": "negative"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  The following are texts and their sentiment, which can be 'positive', 'neutral' or 'negative'.
  ```

- Base prompt template:

  ```text
  Text: {text}
  Sentiment: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Text: {text}

  Classify the sentiment in the text. Answer with 'positive', 'neutral' or 'negative'.
  ```

- Label mapping:
  - `positive` ➡️ `positive`
  - `neutral` ➡️ `neutral`
  - `negative` ➡️ `negative`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset sst5
```

## Named Entity Recognition

### CoNLL-en

This dataset was published in [this paper](https://aclanthology.org/W03-0419/) and was
part of the CoNNL-2003 shared task. The data comes from the [Reuters
Corpus](http://www.reuters.com/researchandstandards) and consists of news articles
between August 1996 and August 1997, labelled with named entities.

The original full dataset consists of 14,041 / 3,250 / 3,453 samples for the training,
validation and test splits, respectively. We use 1,024 / 256 / 2,048 samples for our
training, validation and test splits, respectively. All the new splits are subsets of
the original splits.

Here are a few examples from the training split:

```json
{
  'tokens': array(['SK', 'Slavia', 'Praha', '3', '1', '2', '0', '6', '3', '5'], dtype=object),
  'labels': array(['B-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
```

```json
{
  'tokens': array(['Guy', 'Whittingham', 'stole', 'three', 'points', 'for', 'the', 'Yorkshire', 'side', 'with', 'a', 'goal', '10', 'minutes', 'from', 'time', '.'], dtype=object),
  'labels': array(['B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
```

```json
{
  'tokens': array(['Dean', 'Palmer', 'hit', 'his', '30th', 'homer', 'for', 'the', 'Rangers', '.'], dtype=object),
  'labels': array(['B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'O'], dtype=object)
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Below are sentences and JSON dictionaries with the named entities that occur in the given sentence.
  ```

- Base prompt template:

  ```text
  Sentence: {text}
  Named entities: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Sentence: {text}

  Identify the named entities in the sentence. You should output this as a JSON dictionary with the keys being 'person', 'location', 'organization' and 'miscellaneous'. The values should be lists of the named entities of that type, exactly as they appear in the sentence.
  ```

- Label mapping:
  - `B-PER` ➡️ `person`
  - `I-PER` ➡️ `person`
  - `B-LOC` ➡️ `location`
  - `I-LOC` ➡️ `location`
  - `B-ORG` ➡️ `organization`
  - `I-ORG` ➡️ `organization`
  - `B-MISC` ➡️ `miscellaneous`
  - `I-MISC` ➡️ `miscellaneous`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset conll-en
```

## Linguistic Acceptability

### ScaLA-En

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [English Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_English-GUM/) by assuming that the
documents in the treebank are correct, and corrupting the samples to create
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
  "text": "And so we have to labour and to work, and to work hard, to give reality to our dreams.",
  "label": "correct"
}
```

```json
{
  "text": "This couch is also quite big, it fits three people quite comfortably, and if I have or friends staying over, it opens up into a full double bed.",
  "label": "incorrect"
}
```

```json
{
  "text": "While studies the psychology of art have focused on individual works and distinctions between representative / non-representative topics, no work has been completed on the aesthetic appreciation of collections or of devotional themes.",
  "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  The following are sentences and whether they are grammatically correct.
  ```

- Base prompt template:

  ```text
  Sentence: {text}
  Grammatically correct: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Sentence: {text}

  Determine whether the sentence is grammatically correct or not. Reply with 'yes' if the sentence is correct and 'no' if it is not.
  ```

- Label mapping:
  - `correct` ➡️ `yes`
  - `incorrect` ➡️ `no`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-en
```

## Reading Comprehension

### SQuAD

This dataset was published in [this paper](https://aclanthology.org/D16-1264/), which is
based on English Wikipedia articles and the questions and answers are written by
crowdworkers.

The original full dataset consists of 130,000 / 11,900 samples for training and
validation, respectively. We use 1,024 / 256 / 2,048 samples for training, validation
and testing, respectively (so 3,328 samples used in total). The new training split is a
subset of the original training split, and the new validation and test splits are
disjoint subsets of the original validation split.

Here are a few examples from the training split:

```json
{
  "context": 'The Federation of International Gymnastics (FIG) was founded in Liege in 1881. By the end of the nineteenth century, men\'s gymnastics competition was popular enough to be included in the first "modern" Olympic Games in 1896. From then on until the early 1950s, both national and international competitions involved a changing variety of exercises gathered under the rubric, gymnastics, that would seem strange to today\'s audiences and that included for example, synchronized team floor calisthenics, rope climbing, high jumping, running, and horizontal ladder. During the 1920s, women organized and participated in gymnastics events. The first women\'s Olympic competition was primitive, only involving synchronized calisthenics and track and field. These games were held in 1928, in Amsterdam.',
  "question": 'When was gymnastics included in the Olympics?',
  "answers": {
    "answer_start": array([219], dtype=int32),
    "text": array(['1896'], dtype=object)
  }
}
```

```json
{
  "context": "London's buildings are too diverse to be characterised by any particular architectural style, partly because of their varying ages. Many grand houses and public buildings, such as the National Gallery, are constructed from Portland stone. Some areas of the city, particularly those just west of the centre, are characterised by white stucco or whitewashed buildings. Few structures in central London pre-date the Great Fire of 1666, these being a few trace Roman remains, the Tower of London and a few scattered Tudor survivors in the City. Further out is, for example, the Tudor period Hampton Court Palace, England's oldest surviving Tudor palace, built by Cardinal Thomas Wolsey c.1515.",
  "question": "The area west of London's city is characterized by what type of building?",
  "answers": {
    "answer_start": array([328], dtype=int32),
    "text": array(['white stucco or whitewashed'], dtype=object)
  }
}
```

```json
{
  "context": 'Along with the rest of South West England, Plymouth has a temperate oceanic climate (Köppen Cfb) which is generally wetter and milder than the rest of England. This means a wide range of exotic plants can be grown. The annual mean temperature is approximately 11 °C (52 °F). Due to the modifying effect of the sea the seasonal range is less than in most other parts of the UK. As a result of this summer highs are lower than its southerly latitude should warrant, but as a contrast the coldest month of February has mean minimum temperatures as mild as between 3 and 4 °C (37 and 39 °F). Snow is rare, not usually equating to more than a few flakes, but there have been exclusions, namely the European winter storms of 2009-10 which, in early January, covered Plymouth in at least 1 inch (2.5 cm) of snow; more on higher ground. Another period of notable snow occurred from 17–19 December 2010 when up to 8 inches (20 cm) of snow fell through the period – though only 2 inches (5.1 cm) would lie at any one time due to melt. Over the 1961–1990 period, annual snowfall accumulation averaged less than 7 cm (3 in) per year. July and August are the warmest months with mean daily maxima over 19 °C (66 °F).',
  "question": 'What month in Plymouth has the lowest temperatures?',
  "answers": {
    "answer_start": array([503], dtype=int32),
    "text": array(['February'], dtype=object)
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  The following are texts with accompanying questions and answers.
  ```

- Base prompt template:

  ```text
  Text: {text}
  Question: {question}
  Answer in max 3 words:
  ```

- Instruction-tuned prompt template:

  ```text
  Text: {text}

  Answer the following question about the above text in at most 3 words.

  Question: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset squad
```

### Unofficial: XQuAD-en

This dataset was published in [this paper](https://aclanthology.org/2020.acl-main.421/)
and contains 1190 question-answer pairs from [SQuAD
v1.1](https://rajpurkar.github.io/SQuAD-explorer/) translated into ten languages by
professional translators.

The dataset is split intro 550 / 128 / 512 question-answer pairs for training,
validation, and testing, respectively.

Here are a few examples from the training split:

```json
{
    "context": "Newcastle replaced him in January 1756 with Lord Loudoun, with Major General James Abercrombie as his second in command. Neither of these men had as much campaign experience as the trio of officers France sent to North America. French regular army reinforcements arrived in New France in May 1756, led by Major General Louis-Joseph de Montcalm and seconded by the Chevalier de Lévis and Colonel François-Charles de Bourlamaque, all experienced veterans from the War of the Austrian Succession. During that time in Europe, on May 18, 1756, England formally declared war on France, which expanded the war into Europe, which was later to be known as the Seven Years" War.",
    "question": "Who led New France reinforcements in 1756?",
    "answers": {
        "answer_start": array([305], dtype=int32),
        "text": array(["Major General Louis-Joseph de Montcalm"], dtype=object)
    }
}
```

```json
{
    "context": "Jacksonville is in the First Coast region of northeast Florida and is centered on the banks of the St. Johns River, about 25 miles (40 km) south of the Georgia state line and about 340 miles (550 km) north of Miami. The Jacksonville Beaches communities are along the adjacent Atlantic coast. The area was originally inhabited by the Timucua people, and in 1564 was the site of the French colony of Fort Caroline, one of the earliest European settlements in what is now the continental United States. Under British rule, settlement grew at the narrow point in the river where cattle crossed, known as Wacca Pilatka to the Seminole and the Cow Ford to the British. A platted town was established there in 1822, a year after the United States gained Florida from Spain; it was named after Andrew Jackson, the first military governor of the Florida Territory and seventh President of the United States.",
    "question": "Prior to the arrival of the French, the area now known as Jacksonville was previously inhabited by what people?",
    "answers": {
        "answer_start": array([329], dtype=int32),
        "text": array(["the Timucua"], dtype=object)
    }
}
```

```json
{
    "context": "Luther\"s hymns were frequently evoked by particular events in his life and the unfolding Reformation. This behavior started with his learning of the execution of Johann Esch and Heinrich Voes, the first individuals to be martyred by the Roman Catholic Church for Lutheran views, prompting Luther to write the hymn "Ein neues Lied wir heben an" ("A new song we raise"), which is generally known in English by John C. Messenger\"s translation by the title and first line "Flung to the Heedless Winds" and sung to the tune Ibstone composed in 1875 by Maria C. Tiddeman.",
    "question": "What is the hymn known as in English?",
    "answers": {
        "answer_start": array([469], dtype=int32),
        "text": array(["Flung to the Heedless Winds"], dtype=object)
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  The following are texts with accompanying questions and answers.
  ```

- Base prompt template:

  ```text
  Text: {text}
  Question: {question}
  Answer in max 3 words:
  ```

- Instruction-tuned prompt template:

  ```text
  Text: {text}

  Answer the following question about the above text in at most 3 words.

  Question: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset xquad-en
```

### Unofficial: BeleBele-en

This dataset was published in [this paper](https://aclanthology.org/2024.acl-long.44/)
and features reading comprehension questions across 122 languages. The dataset was
created by professional translators who translated 900 multiple-choice questions from
English into other languages, with answers carefully validated by native speakers.

The original dataset consists of 900 samples, and we use 256 / 64 / 580 samples for
training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": 'Text: """We will endeavour to cut carbon dioxide emissions per unit of GDP by a notable margin by 2020 from the 2005 level,"" Hu said. He did not set a figure for the cuts, saying they will be made based on China\'s economic output. Hu encouraged developing countries ""to avoid the old path of polluting first and cleaning up later."" He added that ""they should not, however, be asked to take on obligations that go beyond their development stage, responsibility and capabilities."""\nQuestion: What did Hu suggest that developing countries do?\nChoices:\na. Take on obligations that push their development stage\nb. Focus on economic output\nc. Go beyond their current responsibilities\nd. Avoiding old paths of pollution',
  "label": "d"
}
```

```json
{
  "text": 'Text: "All of the cave entrances, which were named ""The Seven Sisters"", are at least 100 to 250 meters (328 to 820 feet) in diameter. Infrared images show that the temperature variations from night and day show that they are likely caves. ""They are cooler than the surrounding surface in the day and warmer at night. Their thermal behavior is not as steady as large caves on Earth that often maintain a fairly constant temperature, but it is consistent with these being deep holes in the ground,"" said Glen Cushing of the United States Geological Survey (USGS) Astrogeology Team and of Northern Arizona University located in Flagstaff, Arizona."\nQuestion: What information suggests that The Seven Sisters are caves?\nChoices:\na. Temperature variations\nb. The diameter of the cave entrances\nc. Geological surveys\nd. Pictures of caves on Earth',
  "label": "a"
}
```

```json
{
  "text": 'Text: The proposed amendment already passed both houses in 2011. A change was made this legislative session when the second sentence was deleted first by the House of Representatives and then was passed in a similar form by the Senate Monday. The failure of the second sentence, which proposes to ban same-sex civil unions, could possibly open the door for civil unions in the future. Following the process, HJR-3 will be reviewed again by the next elected legislature in either 2015 or 2016 to remain in process.\nQuestion: According to the passage, when was the second sentence deleted?\nChoices:\na. During the legislative session\nb. In 2011\nc. On Monday\nd. In 2015',
  "label": "a"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  The following are texts with accompanying questions and answers.
  ```

- Base prompt template:

  ```text
  Text: {text}
  Question: {question}
  Answer in max 3 words:
  ```

- Instruction-tuned prompt template:

  ```text
  Text: {text}

  Answer the following question about the above text in at most 3 words.

  Question: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset belebele-en
```

### Unofficial: MultiWikiQA-en

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
    "context": "Stagecoach in Norfolk (formerly Norfolk Green) was a bus operator based in King's Lynn in Norfolk, England. It operated public bus services in the counties of Norfolk, Cambridgeshire and Lincolnshire as well as numerous school and college services. It was a subsidiary of Stagecoach.\n\nIn April 2018, Stagecoach ceased operations in Norfolk. Services were taken over by First Norfolk & Suffolk, Lynx, Sanders Coaches, Stagecoach in Peterborough (the Interconnect 505) and West Norfolk Community Transport.\n\nHistory\n\nNorfolk Green was formed in 1996 with a fleet of four buses. In 1999 the Saham Toney depot was sold to Konectbus with four coaches.\n\nIn April 2011, Norfolk Green purchased the King's Lynn based services of First East England.\n\nOn 17 December 2013, Norfolk Green was sold to Stagecoach following the retirement of Ben Colson after ill health. Unusually, Stagecoach did not immediately apply its corporate brand, but retained the Norfolk Green trading name and livery, although the fleet received Stagecoach fleet numbers. All buses were rebranded between 2015 and late 2017.\n\nIn January 2018, Stagecoach announced it was reviewing its operations in Norfolk in response to the challenging economic environment, blaming a combination of rising operating costs and pressure on public sector budgets. The company said it met with trade union representatives to minimise the impact on staff and launched a consultation with employees over the potential closure of its King's Lynn depot. The company hoped to relocate the majority of its staff with other operators or elsewhere within the Stagecoach East area, which includes Bedford, Cambridge, Huntingdon and Peterborough.\n\nRoutes\nRoutes operated by Stagecoach Norfolk included the very popular  Coasthopper services between King's Lynn and Cromer, the Interconnect 505 between King's Lynn and Spalding, a town service network in King's Lynn, a city service in Ely and many rural and interurban bus services across Norfolk, Cambridgeshire and Lincolnshire.\n\nFleet\nAs at July 2013, the fleet consisted of 74 buses. Fleet livery is two tone green. Twelve Optare Solo Slimlines wear a dark blue, yellow and green livery for the Coasthopper group of services. A large proportion of buses are also named after local characters and personalities.\n\nUpon Stagecoach's purchase of Norfolk Green, in the summer of 2016 Stagecoach Norfolk went onto replace the fleet of Coasthopper Optare Solo's with Alexander Dennis Enviro200s. In addition, and later on, they purchased brand new Optare Solos. These new buses feature a new updated Coasthopper 'Flying Kite' livery, free Wi-Fi, USB charging points and leather seating.\n\nReferences\n\nExternal links\n\nCompany website\n\nStagecoach Group bus operators in England\nTransport companies established in 1966\nTransport companies disestablished in 2018\n1996 establishments in England\n2018 disestablishments in England\nBritish companies established in 1996\nBritish companies disestablished in 2018\nFormer bus operators in Norfolk\nFormer bus operators in Cambridgeshire\nFormer bus operators in Lincolnshire",
    "question": "What is the date of formation of Norfolk Green?",
    "answers": {
        "answer_start": array([543]),
        "text": array(["1996"], dtype=object)
    }
}
```

```json
{
    "context": "Lara Stalder (born 15 May 1994) is a Swiss ice hockey forward and member of the Swiss national ice hockey team, currently playing with Brynäs IF Dam of the Swedish Women's Hockey League (SDHL). She played with the Minnesota Duluth Bulldogs women's ice hockey team from 2013 to 2017, and with Linköping HC from 2017 to 2019.\n\nPlaying career \nAcross four seasons with Minnesota-Duluth, Stalder put up 148 points in 134 games, leading the team in points in her final season, as well as being named WCHA Player of the Year and Student-Athlete of the Year, and being a top-three finalist for the Patty Kazmaier Award. In 2016, she was drafted 20th overall by the Boston Pride of the National Women's Hockey League (NWHL).\n\nAfter missing most of the 2018–19 season due to a shoulder injury, Stalder left Linköping to sign with Brynäs. In 2020, she was named SDHL Player of the Year after putting up 71 points in 36 games, being the first woman to win Guldhjälmen. The 42 goals she would score that year is the second highest single-season total in SDHL history, and her 71 points the third highest single-season total in SDHL history.\n\nInternational  \nStalder made her senior national team debut at the 2011 IIHF Women's World Championship. She has represented Switzerland at the Winter Olympics in 2014 and won the bronze medal after defeating Sweden in the bronze medal playoff. She would score 6 points in 6 games at the 2018 Winter Olympics, as Switzerland finished in 5th place.\n\nCareer statistics\n\nAwards and honors\n\nNCAA\nWCHA Offensive Player of the Week (Week of 17 January 2017)\nWCHA Offensive Player of the Week (Week of 24 January 2017)\nWCHA Offensive Player of the Week (Week of 31 January 2017)\nWCHA Offensive Player of the Month, January 2017\nWomen's Hockey Commissioners' Association National Division I Player of the Month, January 2017\nPatty Kazmaier Award Top-3 Finalist, 2016–17 season\n2016-17 AHCA-CCM Women's University Division I First-Team All-American\n\nSDHL \n\n Guldhjälmen (Golden Helmet), MVP of the SDHL as selected by players, 2019–20 season\n SDHL Forward of the Year, 2019–20 season\n\nReferences\n\nExternal links\n\nMinnesota Duluth bio\n\n1994 births\nLiving people\nSportspeople from Lucerne\nSwiss women's ice hockey forwards\nIce hockey players at the 2014 Winter Olympics\nIce hockey players at the 2018 Winter Olympics\nIce hockey players at the 2022 Winter Olympics\nOlympic bronze medalists for Switzerland\nOlympic ice hockey players for Switzerland\nOlympic medalists in ice hockey\nMedalists at the 2014 Winter Olympics\nBrynäs IF (women) players\nLinköping HC (women) players\nMinnesota Duluth Bulldogs women's ice hockey players\nSwiss expatriate ice hockey people\nSwiss expatriate sportspeople in Sweden\nSwiss expatriate sportspeople in the United States",
    "question": "Which SDHL award did Lara Stalder receive during the 2019-2020 season?",
    "answers": {
        "answer_start": array([945]),
        "text": array(["Guldhjälmen"], dtype=object)
    }
}
```

```json
{
    "context": "TCG Barbaros (F 244) is the lead ship of  of the Turkish Navy.\n\nDevelopment and design \n\nBarbaros-class frigates were designed in Germany and are part of the MEKO group of modular warships, in this case the MEKO 200 design. Two ships were built in Germany and two in Turkey with German assistance. They are larger than the previous s and are also faster due to using CODOG machinery rather than pure diesels.\n\nThe first two vessels (F 244 and F 245) are defined as the Barbaros class (MEKO 200 TN Track II-A) while the last two vessels (F 246 and F 247) are defined as the Salih Reis class (MEKO 200 TN Track II-B) by the Turkish Navy.\n\nSalih Reis subclass ships are built with 8-cell Mk. 41 VLS and longer than Barbaros class vessels to accommodate 16-cell Mk. 41 VLS upgrade in the future while Barbaros-class vessels built with  Mk.29 Sea Sparrow launchers that planned to be replaced by 8-cell Mk. 41 VLS.\n\nConstruction and career \nBarbaros was launched on 29 September 1993 by Blohm+Voss in Hamburg and commissioned on 23 May 1997.\n\nOn 9 March 2019, her crew saluted to the tomb of Barbaros Hayreddin while crossing Bosporus.\n\nOn 26 August 2020, TCG Barbaros and  sailed alongside  in Eastern Mediterranean Sea. Later that year on 3 October, she underwent alongside USS Roosevelt.\n\nReferences\n\nExternal links\n\n The First Upgraded MEKO 200 Frigate Of Turkish Navy\n BARBAROS CLASS ( MEKO 200 Track II) (Turkey)\n\n1993 ships\nShips built in Germany\nFrigates of the Turkish Navy\nBarbaros-class frigates of the Turkish Navy",
    "question": "Could you tell me about the MEKO group?",
    "answers": {
        "answer_start": array([172]),
        "text": array(["modular warships"], dtype=object)
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  The following are texts with accompanying questions and answers.
  ```

- Base prompt template:

  ```text
  Text: {text}
  Question: {question}
  Answer in max 3 words:
  ```

- Instruction-tuned prompt template:

  ```text
  Text: {text}

  Answer the following question about the above text in at most 3 words.

  Question: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-en
```

## Knowledge

### Life in the UK

This dataset was published
[here](https://huggingface.co/datasets/oliverkinch/life-in-the-uk-multiple-choice) was
scraped from [lifeintheuktestweb.co.uk](https://lifeintheuktestweb.co.uk/test-1/) and
contains multiple choice questions about UK history, culture, and citizenship
requirements. The website was created to help people pass the Life in the UK Test for UK
citizenship.

The original dataset consists of 1,450 samples. After processing (removing questions
with overly short or long texts, repetitive content, and true/false questions), we have
1,206 samples remaining. From these, we use 438 / 256 / 512 samples for our training,
validation and test splits, respectively.

Here are a few examples from the training split:

```json
{
  "text": "What is the capital of the United Kingdom?\nChoices:\na. London\nb. Manchester\nc. Birmingham\nd. Edinburgh",
  "label": "a"
}
```

```json
{
  "text": "What TWO houses were confronted during the Wars of the Roses?\nChoices:\na. The House of Lancaster\nb. The House of Leicester\nc. The House of Canterbury\nd. The House of York",
  "label": "a"
}
```

```json
{
  "text": "What is the name of the War Memorial located in Whitehall?\nChoices:\na. Dumfries\nb. Cenotaph\nc. Royal Crescent\nd. The White Tower",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  The following are multiple choice questions (with answers).
  ```

- Base prompt template:

  ```text
  Question: {text}
  Options:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Answer: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Question: {text}
  Options:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Answer the above question by replying with 'a', 'b', 'c' or 'd', and nothing else.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset life-in-the-uk
```

### Unofficial: MMLU

This dataset was published [in this paper](https://doi.org/10.48550/arXiv.2009.03300)
and features questions within 57 different topics, such as elementary mathematics, US
history and law.

The original full dataset consists of 269 / 1,410 / 13,200 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 2,048 split for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
new and there can thus be some overlap between the original validation and test sets and
our validation and test sets.

Here are a few examples from the training split:

```json
{
  "text": "Use the following key to translate the given formula of PL to natural, English sentences. A: Marina reads a Percy Jackson book. B: Izzy plays Minecraft. C: Emily stops working. D: Russell makes dinner. E: Ashleigh stops by. ~(A ⊃ B) • (B ⊃ ~E)\nChoices:\na. It's not the case that Marina's reading a Percy Jackson book entails that Izzy plays Minecraft, but Izzy's playing Minecraft does entail that Ashleigh doesn't stop by.\nb. If Marina doesn't read a Percy Jackson book, then Izzy plays Minecraft, which entails that Ashleigh doesn't stop by.\nc. Marina's reading a Percy Jackson book does not entail that Izzy plays Minecraft, but Izzy plays Minecraft provided that Ashleigh doesn't stop by.\nd. It's not true that Marina reads a Percy Jackson book only when Izzy plays Minecraft, but Izzy plays Minecraft only when Ashleigh stops by.",
  "label": "a"
}
```

```json
{
  "text": "As of 2017, the share of GDP spent on the military by the United States is about\nChoices:\na. 1%\nb. 3%\nc. 6%\nd. 10%",
  "label": "b"
}
```

```json
{
  "text": "Question 13. A buyer sent a signed letter to a seller that stated: \"Ship 100 boxes of nails at $3 per box, the price quoted in your circular.\" The seller mailed the buyer a signed form acknowledgment that agreed to the buyer's terms and stated on the reverse side: \"Disputes regarding quality shall be arbitrated.\" The buyer did not reply to the seller's acknowledgment, and the seller shipped the nails. When the buyer received the nails, it found their quality to be unsatisfactory and sued the seller for breach of warranty. The seller has asked an attorney whether the parties' contract requires arbitration of the buyer's claim. What is the best advice the attorney can provide?\nChoices:\na. A contract was formed pursuant to conduct when the buyer received the nails, and a court would exclude the arbitration provision from the contract.\nb. A contract was formed when the seller mailed its acknowledgment, and the arbitration term became part of the contract. arbitration term became part of the contract.\nc. A contract was formed when the seller mailed its acknowledgment, and the court must decide whether the arbitration term should be excluded as a material alteration of the contract.\nd. No contract exists, because the arbitration term in the seller's acknowledgment created a counteroffer that the buyer never accepted.",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  The following are multiple choice questions (with answers).
  ```

- Base prompt template:

  ```text
  Question: {text}
  Options:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Answer: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Question: {text}
  Options:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Answer the above question by replying with 'a', 'b', 'c' or 'd', and nothing else.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset mmlu
```

### Unofficial: ARC

This dataset was published [in this paper](https://doi.org/10.48550/arXiv.1803.05457)
and features US grade-school science questions.

The original full dataset consists of 1,110 / 297 / 1,170 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 1,024 split for training,
validation and testing, respectively (so 2,304 samples used in total). All new splits
are subsets of the original splits.

Here are a few examples from the training split:

```json
{
  "text": "Several horses grazed in a fenced area across from a home. On rainy days, soil would wash down a slope and run toward the home. After the horses were moved a few years later, the soil no longer washed down when it rained. What could account for this change?\nChoices:\na. The grass grew and kept the soil intact.\nb. The fence kept the soil contained.\nc. The soil was completely gone.\nd. The amount of rain decreased.",
  "label": "a"
}
```

```json
{
  "text": "How do moose use a learned behavior to protect themselves?\nChoices:\na. They have hollow hair to keep warm in the winter.\nb. They roll in a pool of muddy water to avoid fly bites.\nc. They have keen hearing to sense danger in the forest.\nd. They use their wide hooves to prevent sinking in deep snow.",
  "label": "b"
}
```

```json
{
  "text": "A plant that grows red flowers was crossed with the same kind of plant that grows white flowers. Their offspring grew pink flowers. Which best explains why the offspring grew pink flowers?\nChoices:\na. The offspring experienced a genetic mutation.\nb. The offspring resulted from asexual reproduction.\nc. The genes for flower color exhibited incomplete dominance.\nd. A gene for pink-colored flowers was recessive in one of the parents.",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  The following are multiple choice questions (with answers).
  ```

- Base prompt template:

  ```text
  Question: {text}
  Options:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Answer: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Question: {text}
  Options:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Answer the above question by replying with 'a', 'b', 'c' or 'd', and nothing else.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset arc
```

## Common-sense Reasoning

### HellaSwag

This dataset was published in [this paper](https://aclanthology.org/P19-1472/) and is
based on both video descriptions from ActivityNet as well as how-to articles from
WikiHow.

The original full dataset consists of 9,310 samples. We use a 1,024 / 256 / 2,048 split
for training, validation and testing, respectively (so 3,328 samples used in total).

Here are a few examples from the training split:

```json
{
  "text": "[header] How to solo travel to chile [title] Decide how you will get to chile. [step] Start by figuring out how you will get to chile. If you live in north america, you may decide to fly into a major city in the country, such as santiago, and then take public transit to get around or fly within chile.\nChoices:\na. If you live in south america, it may be possible to take a bus or a train into chile, depending on your budget and your timeframe for the trip. [substeps] There is no special visa required for you to travel into chile and no fee to cross the border into chile.\nb. If you live in australia, you will need to negotiate a road trip, such as a train or bus, to get around chile. [substeps] Plan out the route in advance of arrival so that you can do the same to chile in the future.\nc. If you live in a rural area or you do not plan to travel for a long time, you may opt to take a bus. Using a bus or subway to get around chile is a good route to travel.\nd. If you live in a smaller area, or if you live near a large tourist attraction, you may decide to fly in the opposite direction. [substeps] Skiing, mountain climbing, and bicycle riding are examples of solo travel.",
  "label": "a"
}
```

```json
{
  "text": "The video begins with a title sequence. a young man\nChoices:\na. prepares to black out.\nb. is shown in a gym performing tricks with a jump rope as music plays in the background.\nc. is seen talking continuously about slamming the mouth of a chimpanzee into the camera.\nd. is standing outside with a basketball in his hand, alternating between shots of dribbling for the ball.",
  "label": "b"
}
```

```json
{
  "text": "A herb garden appears with a woman standing next to it in a large garden next to a wheelbarrow filled with mulch. the woman\nChoices:\na. moves the mulch across the ground in the wheelbarrow, falling backwards on attempts.\nb. takes some of the mulch away and starts bagging it in the wheelbarrow.\nc. begins to talk to the camera while gesturing to the flowerbed and the mulch, before eventually picking up a handful of the mulch.\nd. then begins to mulch close to the wheelbarrow with mulching tool in her hand and while waving her arms in the air.",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  The following are multiple choice questions (with answers).
  ```

- Base prompt template:

  ```text
  Question: {text}
  Options:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Answer: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Question: {text}
  Options:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Answer the above question by replying with 'a', 'b', 'c' or 'd', and nothing else.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset hellaswag
```

### Unofficial: Winogrande

This dataset was published in [this paper](https://doi.org/10.1145/3474381). The
original full dataset consists of 47 / 1,210 samples for training and testing, and we
use 128 of the test samples for validation, resulting in a 47 / 128 / 1,085 split for
training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Elena would grab their inventory in the back of the store for Megan to sell each time because _ was a businessperson. What does the blank _ refer to?\nChoices:\na. Elena\nb. Megan",
  "label": "a"
}
```

```json
{
  "text": "Once in Poland, Dennis enjoyed the trip more than Jason because _ had a deeper understanding of the Polish language. What does the blank _ refer to?\nChoices:\na. Dennis\nb. Jason",
  "label": "a"
}
```

```json
{
  "text": "Handling emergencies was never very difficult for Kevin but it was for Nelson because _ wasn't able to remain calm under pressure. What does the blank _ refer to?\nChoices:\na. Kevin\nb. Nelson",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  The following are multiple choice questions (with answers).
  ```

- Base prompt template:

  ```text
  Question: {text}
  Options:
  a. {option_a}
  b. {option_b}
  Answer: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Question: {text}
  Options:
  a. {option_a}
  b. {option_b}

  Answer the above question by replying with 'a' or 'b', and nothing else.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset winogrande
```

## Summarisation

### CNN/DailyMail

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.1506.03340)
and is based on news articles from CNN and DailyMail, with the summaries derived from
bullet points written by the authors of the articles.

The original full dataset consists of 287,113 / 13,368 / 11,490 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 2,048 split for training,
validation and testing, respectively (so 3,328 samples used in total). All new splits
are subsets of the original splits.

Here are a few examples from the training split:

```json
{
  "text": "Reality TV star and conservative firebrand Sarah Palin said today she's 'interested' in running for president in 2016 but stopped short of saying she'd actually seek higher office. 'Yeah, I mean, of course, when you have a servant’s heart, when you know that there is opportunity to do all you can to put yourself forward in the name of offering service, anybody would be interested,' Palin told ABC News reporter Neal Karlinsky. Later stating, 'America has had enough of seeing that...sign on the Oval Office door saying, \"No Girls Allowed.\" ' 'It doesn't necessarily have to be me though,' she said. Scroll down for video . Conservative firebrand Sarah Palin said today she's 'interested' in running for president in 2016 but stopped short of saying she'd actually seek higher office . GIRL POWER: 'America has had enough of seeing that...sign on the Oval Office door saying, \"No Girls Allowed,\" ' Palin said . NOM NOM NOM: Palin made the comments while serving wild boar chili to the Salvation Army in Las Vegas, Nevada, on Friday. She was hosting an episode of Sportsman Channel program Hunt.Fish.Feed . ABC News caught up with Palin while she was serving wild boar chili to the homeless at a Las Vegas, Nevada, Salvation Army for an episode of Sportsman Channel program Hunt.Fish.Feed. She's also in the midst of promoting her hunting show, Amazing Alaska, about to begin its second season. Palin said the GOP needs to nominate a candidate 'who can take on Hillary' and 'show the nation what it is going to take to get the country back on the right track.' 'Because we can't afford status quo,' the former Alaska governor said in a clip of the interview released by ABC this afternoon. 'Status quo lately has been Latin for, \"We're getting screwed,' and status quo has got to go.' The Republican nominee out to be someone can 'turn things around, someone who will, in some respects, I don’t know, maybe be considered a bit avant garde, to the establishment anyway, because this next person has got to realize this is war, this is war for our hunters’ future,' she said at another point in the interview, according to ABC. Asked about former Florida Gov. Jeb Bush's candidacy and 2012 Republican presidential nominee Mitt Romney, Palin snarked, 'I can’t wait for new energy.' Moments later asserting that the GOP primary 'had better be a competition and not a coronation.' Palin, the 2008 vice presidential nominee, said she doesn't 'have to be' the Republican candidate for president but she's 'happy to drive that competition, because competition will make everyone better and produce more and be more candid regarding their solutions they will offer this country. 'I am very interested in that competitive process and, again, not necessarily me.' Former Alaska Palin is pictured here on Thursday at an event to promote her television show, Amazing America with Sarah Palin, at the Shooting, Hunting and Outdoor Trade Show in Las Vegas . The hard-charging, Tea Party icon appears to have a change of heart in the last week about the Oval Office needing a female touch. 'I don't give a flying flip about what gender the person will be,' Palin told Inside Edition after host Deborah Norville asker her about the importance of electing a female president. 'I want the absolute best because America deserves the best, in terms of leadership, getting this country on the right track,' she continued. She ultimately concluded 'it would be nice' to have a woman president, though, 'and it will be nice to see women jump into the ring.' Voicing her support for female candidates in December, Palin told Extra TV, 'I would love to see a woman on both sides of the aisle shooting for that top spot.'",
  "target_text": "'When you know that there is opportunity to do all you can to put yourself forward in the name of offering service, anybody would be interested'\nPalin added: 'It doesn't necessarily have to be me though'\nThe conservative firebrand appears to have a change of heart about the Oval Office needing a female touch .\nLast week she said: 'I don't give a flying flip about what gender the person will be'"
}
```

```json
{
  "text": "By . Amanda Williams . The dictionary makers have taken to Twitter to find new words for the next edition of the lexicon - asking users to choose which words should make the final edition . The latest edition of the Collins English Dictionary could include Twitter slang words such as 'adorkable' and 'fatberg'. The dictionary makers have taken to Twitter to find new words for the next edition of the lexicon - asking users to choose which words should make the final edition. The list of suggested words includes fracktivist - someone who protests against fracking - and felfie, a term used to describe a farmer who takes a selfie, or photograph of themselves. The 12th edition of the dictionary will be the first to contain a word that has been voted for by Twitter users - who have until midnight on May 28 to vote for the new word. Once selected, it will be included in the next edition of the dictionary, which is released in October. The dictionary publisher says that the rise of social media and the hashtag has seen new words and ideas - that they scout for every year - become mainstream much quicker than in the past. Andrew Freeman, associate publisher at Collins, said: 'Twitter offers us an immediate snapshot of how much a word is used. 'The tried and tested approach to compiling dictionaries has to adapt to embrace the ways in which language is developing through use on social media, and this is a fun way to get Twitter users involved in defining the English language.' Collins has been publishing the dictionary since 1819 and is the largest single volume dictionary in print, with the words it contains sourced from the Collins Corpus, which contains more than 4.5 billion words, as well as the open source site collinsdictionary.com, where users can submit words for consideration. The latest edition of the Collins English Dictionary could include Twitter slang words such as 'adorkable' The word felfie, a term used to describe a farmer who takes a selfie, or photograph of themselves could also be included . Nomakeupselfie - a selfie of a woman without make-up, posted online to raise awareness for a charity - is also in the running to be used in the dictionary . Lucy Mangan, a blogger for collinsdictionary.com and a contributor to the Collins English Dictionary, said: 'Twitter is the perfect place to find out what people are really saying and how they’re saying it. 'It’s a space in which you’re freer than almost anywhere else to combine old words, resurrect others or invent totally new ones whenever the need arises.' According to language experts, the list, which also contains the word adorkable, referring to someone who is dorky in an adorable way, is a sign of the way language is changing in the 21st century. Ian Brookes, lexicographer and consultant editor to the Collins English Dictionary, said: 'Language has always had to develop in response to changes in society and technology. In the 20th century the development of the motor car, air travel, television, and the personal computer changed the things that people did and so brought many new words into the language. 'In the 21st century, the growth of social media has had a comparable effect. Twitter users can vote for their choice by visiting twictionary.collinsdictionary.com . Adorkable - dorky in an adorable way . Fatberg - a large mass of solid waste, grease etc, clogging a sewage system . Felfie - a farmer selfie . Gaybourhood - a gay-friendly neighbourhood, e.g. Castro in San Francisco . Nomakeupselfie - a selfie of a woman without make-up, posted online to raise awareness for a charity . Vaguebooking - posting a deliberately vague status updates on social media to prompt a response . Duckface - the traditional pouting facial expression in selfies . Fracktivist - an activist who protests against fracking . Euromaiden - the original pro-Europe protests in Ukraine, named for Maidan Square in Kiev .",
  "target_text": "Dictionary makers have taken to Twitter to find new words for next edition .\nThe suggested words include fracktivist - an anti-fracking protester .\nFelfie - a term used to describe a farmer who takes a selfie - also included ."
}
```

```json
{
  "text": "There were three of them, one of them probably a child, and at least one met a gruesome end at the hands of a terrifying predator. About 67 million years later, a Wyoming rancher led scientists to their remains. Now experts are digging out one of the most complete skeletons yet of a Triceratops, the three-horned, plant-eating dinosaur that was one of the last of the giant reptiles. \"There's only three other skeletons that will match the completeness of one of the specimens we're excavating right now,\" said paleontologist Peter Larson, president of the Black Hills Institute of Geological Research. Most of the remains found before now have included fewer than half of the prehistoric creatures' bones, Larson said Monday. The most complete to date, now on display at the Houston Museum of Natural Science in Texas, has about 76% of its skeleton. \"The largest, more mature individual appears to be the most complete,\" Larson said. \"One is just a bit smaller, and there's another one that by live weight is probably only half the size.\" Will mammoths be brought back to life? Liquid blood fuels cloning hopes . The dig is going on near Newcastle, Wyoming, more than 200 miles north of Cheyenne. \"The fact that there are three of them together is really cool,\" Larson said. The trio could be male and female and their young, or they could be two females looking after a juvenile dinosaur, he said. And before now, there was no indication that the Triceratops moved in groups. The Black Hills Institute is working with the Naturalis Biodiversity Center, from the Netherlands, on the dig. Larson called the discovery of a young Triceratops a \"very significant\" find as well, since it will give scientists an insight into how the great lizards grew up. Newly discovered dinosaur fossil is a primitive bird . Triceratops lived in the twilight of the Cretaceous Period, about a half a million years before the dinosaurs' extinction. Much of what is now the Great Plains and southern Canada was once part of a vast inland sea, and the region is rich in fossils. \"Like most of the specimens that were found, it was brought to our attention by a rancher,\"  Larson said. The rancher sent photos to the Black Hills Institute, located in neighboring South Dakota, in late 2012. Excavation began in May and is expected to take about a month. So far, the bones that have turned up point to a violent end, probably at the hands of the feared Tyrannosaurus rex. On the largest of the three specimens, at least two of the major limb bones were \"bitten through,\" Larson said. \"If you can imagine, this is a bone that is nearly four feet long,\" he said. But a T.rex \"would kind of chop the carcass up with their giant, shearing jaws,\" ripping through flesh and bone alike. \"I think we also have a feeding site for Tyrannosaurus rex, which is very exciting,\" he said. \"This is potentially a site where we can learn the behavior of two different species.\" More science news on CNN's Light Years blog .",
  "target_text": "A rancher led scientists to the remains of three Triceratops .\nOne of the three may be the most complete skeleton yet found .\nA young dinosaur is among the trio .\nAt least one may have been killed by a Tyrannosaurus rex ."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:

  ```text
  The following are articles with accompanying summaries.
  ```

- Base prompt template:

  ```text
  News article: {text}
  Summary: {target_text}
  ```

- Instruction-tuned prompt template:

  ```text
  News article: {text}

  Write a summary of the above article.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset cnn-dailymail
```
