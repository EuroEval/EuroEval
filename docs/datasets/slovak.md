# 🇸🇰 Slovak

This is an overview of all the datasets used in the Slovak part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### CSFD Sentiment-sk

This dataset was published in [this paper](https://aclanthology.org/R13-1016/) and
consists of reviews from the the Czech/Slovak Movie
Database (CSFD).

The original dataset contains 25,000 / 2,500 / 2,500 samples for the training,
validation, and test splits, respectively. We use 1,024 / 256 / 2,048 samples for
our training, validation and test splits, respectively. All the new splits are
subsets of the original splits.

Here are a few examples from the training split:

```json
{
    "text": "Jó Steve Buacemi...jinak sračka",
    "label": "negative"
}
```

```json
{
    "text": "Letny oddychovy comicsovy blockbuster. Po celkom fresh traileri a hlavne podla momentalnych hodnoteni (89%, 76. najlepsi film!!!) som cakal, ze to bude daka svieza pecka a prijemne prekvapenie. Ale nakoniec je to dost taky priemer. Taka ta klasika, universe s roznymi rasami, (anti)hrdinova, neoriginalna zapletka, kopa akcie.. Co vycnievalo boli zaujimave postavy a hlavne vtipne hlasky. Prave tymto sa mohol film viac odlisit od ostatnych comicsoviek, vtedy by som isiel s hodnotenim vyssie.. Od polky filmu mi bolo jasne, ze to je kvazi ochutnavka na (minimalne) dalsie 2 casti, tak uvidime kam to posunu. [#40/2013]",
    "label": "neutral"
}
```

```json
{
    "text": "Prevapivo príjemné, vtipné, rozprávkové. Konečne fantasy film, ktorý sa sústredí na rozprávanie príbehu a nepotrebuje k tomu zbesilé tempo ani veľkolepé počítačové armády. Vo svojej podstate to nie je až také originálne, ale je tam pár zaujímavých nápadov a ako celok to skvelo funguje - všetko je skrátka na svojom mieste...",
    "label": "positive"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Nižšie sú dokumenty a ich sentiment, ktorý môže byť 'pozitívne', 'neutrálne' alebo 'negatívne'.
  ```

- Base prompt template:

  ```text
  Dokument: {text}
  Sentiment: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Dokument: {text}

  Klasifikujte pocit v dokumente. Odpovedzte so 'pozitívne', 'neutrálne', alebo 'negatívne', a nič iné.
  ```

- Label mapping:
  - `positive` ➡️ `pozitívne`
  - `neutral` ➡️ `neutrálne`
  - `negative` ➡️ `negatívne`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset csfd-sentiment-sk
```

## Named Entity Recognition

### UNER-sk

This dataset was published in
[this paper](https://aclanthology.org/2024.naacl-long.243/).

The original dataset consists of 8,482 / 1,059 / 1,060 samples for the
training, validation, and test splits, respectively. We use 1,024 / 256 / 2,048
samples for our training, validation and test splits, respectively. The train and
validation splits are subsets of the original splits, while the test split is
created using additional samples from the train split.

Here are a few examples from the training split:

```json
{
  "tokens": ["Bude", "mať", "názov", "Shanghai", "Noon", "a", "režisérom", "bude", "debutujúci", "Tom", "Dey", "."],
  "labels": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "B-PER", "I-PER", "O"]
}
```

```json
{
  "tokens": ["Ako", "šesťročného", "(", "o", "rok", "skôr", ",", "než", "bolo", "zvykom", ")", "ho", "na", "základe", "zvláštnej", "výnimky", "prijali", "medzi", "Zvedov", "a", "ako", "deväťročný", "sa", "stal", "vedúcim", "skupiny", "."],
  "labels": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-ORG", "O", "O", "O", "O", "O", "O", "O", "O"]
}
```

```json
{
  "tokens": ["To", "predsa", "stojí", "za", "pokus", "!"],
  "labels": ["O", "O", "O", "O", "O", "O"]
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Nasledujúce sú vety a JSON-objekty s pomenovanými entitami, ktoré sa nachádzajú v danej vete.
  ```

- Base prompt template:

  ```text
  Veta: {text}
  Pomenované entity: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Veta: {text}

  Identifikujte pomenované entity vo vete. Výstup by mal byť vo forme JSON-objektu s kľúčmi 'osoba', 'miesto', 'organizácia' a 'rôzne'. Hodnoty by mali byť zoznamy pomenovaných entít danej kategórie, presne tak, ako sa vyskytujú vo vete.
  ```

- Label mapping:
  - `B-PER` ➡️ `osoba`
  - `I-PER` ➡️ `osoba`
  - `B-LOC` ➡️ `miesto`
  - `I-LOC` ➡️ `miesto`
  - `B-ORG` ➡️ `organizácia`
  - `I-ORG` ➡️ `organizácia`
  - `B-MISC` ➡️ `rôzne`
  - `I-MISC` ➡️ `rôzne`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset uner-sk
```

## Linguistic Acceptability

### CS-GEC

This dataset is extracted by postprocessing data from
[this paper](https://aclanthology.org/D19-5545/). Specifically,
grammatically incorrect sentences and their corresponding corrections
were extracted.

The original full dataset consists of 59,493 training and 4,668 test
samples, respectively. We use a 1,024 / 256 / 2,048 split for training,
validation, and testing, respectively. The train and test splits are
subsets of the original splits, and the validation split is created
using examples from the train split.

Here are a few examples from the training split:

```json
{
  "text": "Musíme ochutnát pivo a knedlíky .",
  "label": "incorrect"
}
```

```json
{
  "text": "V budoucnosti bych chtěla mít velkou rodinu a dům mých snů .",
  "label": "correct"
}
```

```json
{
  "text": "Dědeček i babička po druhé světové válce několik let žili v ČR a pak se zase vratili do Lužice .",
  "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Následující jsou věty a zda jsou gramaticky správné.
  ```

- Base prompt template:

  ```text
  Věta: {text}
  Gramaticky správná: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Věta: {text}

  Určete, zda je věta gramaticky správná nebo ne. Odpovězte 'ano', pokud je věta správná, a 'ne', pokud není. Odpovězte pouze tímto slovem, a ničím jiným.
  ```

- Label mapping:
  - `correct` ➡️ `ano`
  - `incorrect` ➡️ `ne`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset cs-gec
```

### Unofficial: ScaLA-cs

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Czech Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Slovak-SNK) by assuming that the
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
    "text": "Niektorí pozorovatelia považujú ropné záujmy USA za jednu z hlavných motivácií vstupu do vojny v Iraku.",
    "label": "correct"
}
```

```json
{
    "text": "Popáliť sa na jedinom písmene je klasický prípad, ktorý sa môže vyskytnúť v rôznych podobách.",
    "label": "correct"
}
```

```json
{
    "text": "Zo strachu o seba, pre svoju povýšenú zbabelosť zaprel svojho Majstra Pána.",
    "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Nasledujú vety a či sú gramaticky správne.
  ```

- Base prompt template:

  ```text
  Veta: {text}
  Gramaticky správna: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Veta: {text}

  Určite, či je veta gramaticky správna alebo nie. Odpovedzte so 'áno', ak je veta správna, a 'nie', ak nie je. Odpovedzte iba týmto slovom, a nič iné.
  ```

- Label mapping:
  - `correct` ➡️ `áno`
  - `incorrect` ➡️ `nie`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-cs
```

### CS-GEC

This dataset is extracted by postprocessing data from
[this paper](https://doi.org/10.18653/v1/D19-5545). Specifically,
grammatically incorrect sentences and their corresponding corrections
were extracted.

The original full dataset consists of 59,493 training and 4,668 test
samples, respectively. We use a 1,024 / 256 / 2,048 split for training,
validation, and testing, respectively. The train and test splits are
subsets of the original splits, and the validation split is created
using examples from the train split.

Here are a few examples from the training split:

```json
{
  "text": "Musíme ochutnát pivo a knedlíky .",
  "label": "incorrect"
}
```

```json
{
  "text": "V budoucnosti bych chtěla mít velkou rodinu a dům mých snů .",
  "label": "correct"
}
```

```json
{
  "text": "Dědeček i babička po druhé světové válce několik let žili v ČR a pak se zase vratili do Lužice .",
  "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Následující jsou věty a zda jsou gramaticky správné.
  ```

- Base prompt template:

  ```text
  Věta: {text}
  Gramaticky správná: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Věta: {text}

  Určete, zda je věta gramaticky správná nebo ne. Odpovězte 'ano', pokud je věta správná, a 'ne', pokud není. Odpovězte pouze tímto slovem, a ničím jiným.
  ```

- Label mapping:
  - `correct` ➡️ `ano`
  - `incorrect` ➡️ `ne`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset cs-gec
```

## Reading Comprehension

### MultiWikiQA-sk

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
  "context": "Register toxických účinkov chemických látok (anglicky Registry of Toxic Effects of Chemical Substances, RTECS) je databáza toxikologických informácií zostavených z voľne dostupnej vedeckej literatúry bez odkazu na platnosť alebo užitočnosť publikovaných štúdií. Do roku 2001 bola databáza spravovaná americkou organizáciou NIOSH (National Institute for Occupational Safety and Health, slov. Národný ústav pre bezpečnosť a ochranu zdravia pri práci) ako verejne dostupná publikácia. Teraz ju spravuje súkromná spoločnosť Symyx Technologies a je dostupná len za poplatok.\n\nObsah \nDatabáza obsahuje šesť typov toxikologických informácií:\n primárne podráždenie\n mutagénne účinky\n reprodukčné účinky\n karcinogénne účinky\n akútna toxicita\n toxicita viacnásobných dávok\nV databáze sa spomínajú ako špecifické číselné hodnoty, ako napríklad LD50, LC50, TDLo alebo TCLo, tak aj študované organizmy a spôsob podávania látky. Pre všetky dáta sú uvedené bibliografické zdroje. Štúdie pritom nie sú nijako hodnotené.\n\nHistória \nDatabáza RTECS bola aktivitou schválenou americkým Kongresom, zakotvenou v Sekcii 20(a)(6) zákona Occupational Safety and Health Act z roku 1970 (PL 91-596). Pôvodné vydanie, známe ako Zoznam toxických látok (Toxic Substances List), bolo publikované 28. júna 1971 a obsahovalo toxikologické dáta o približne 5 000 chemikáliách. Názov bol neskôr zmenený na dnešný Register toxických účinkov chemických látok (Registry of Toxic Effects of Chemical Substances). V januári 2001 databáza obsahovala 152 970 chemikálií. V decembri 2001 bola správa RTECS prevedená z NIOSH do súkromnej firmy Elsevier MDL. Túto firmu kúpila v roku 2007 spoločnosť Symyx, súčasťou akvizície bola aj databáza RTECS. Tá je teraz dostupná len za poplatok vo forme ročného predplatného.\n\nRTECS je k dispozícii v angličtine, francúzštine a španielčine, a to prostredníctvom Kanadského centra pre bezpečnosť a ochranu zdravia pri práci. Predplatitelia majú prístup cez web, na CD-ROM a vo formáte pre intranet. Databáza je dostupná na webe aj cez NISC (National Information Services Corporation) a ExPub (Expert Publishing, LLC).\n\nExterné odkazy \n\n RTECS overview \n Symyx website \n Expert Publishing, LLC Website\n\nZdroj \n\nChemické názvy a kódy\nToxikológia",
  "question": "Aké sú tri možnosti prístupu k databáze RTECS, ak som predplatiteľ?",
  "answers": {"answer_start": [1949], "text": ["cez web, na CD-ROM a vo formáte pre intranet"]}}
```

```json
{
  "context": "Herta Naglová-Docekalová (* 29. máj 1944, Wels, Rakúsko) je rakúska filozofka a profesorka, členka vedenia Medzinárodnej asociácie filozofiek (IAPf), Österreichische Akademie der Wissenschaften, Institut International de Philosophie (Paríž), viceprezidentka Fédération Internationale des Sociétés de Philosophie (FISP), zakladajúca členka interdisciplinárnych pracovných skupín Frauengeschichte a Philosophische Frauenforschung na Viedenskej univerzite, členka redakčných rád popredných vedeckých časopisov, napr. Philosophin, L´Homme, Deutsche Zeitschrift für Philosophie.\n\nŽivotopis \nVyštudovala históriu, filozofiu a germanistiku na Viedenskej univerzite. V roku 1967 získala na svojej alma mater doktorát z histórie prácou o filozofovi dejín Ernstovi von Lasaulx). V rokoch 1968 - 1985 bola asistentkou na Inštitúte filozofie Viedenskej univerzity. V lete 1980 prednášala na Millersville University of Pennsylvania v USA.\n\nV roku 1981 sa habilitovala z filozofie na Viedenskej univerzite dielom Die Objektivität der Geschichtswissenschaft. V rokoch 1985 až 2009 bola profesorkou Inštitútu filozofie Viedenskej univerzity. Od roku 2009 je univerzitnou profesorkou na dôchodku (Universitätsprofessorin i. R.)\n\nBola hosťujúcou profesorkou v roku 1990 na Universiteit Utrecht v holandskom Utrechte; v Nemecku 1991/1992 na Goethe-Universität Frankfurt vo Frankfurte nad Mohanom; 1993 na Universität Konstanz v Konstanzi; 1994/1995 na Freie Universität Berlin v Berlíne. V rokoch 1995/1996 prednášala na Universität Innsbruck a 2011 na univerzite v Petrohrade v Rusku.\n\nDielo (výber) \n Jenseits der Säkularisierung. Religionsphilosophische Studien. - Berlin 2008 (Hg., gem.m. Friedrich Wolfram).\n Viele Religionen - eine Vernunft? Ein Disput zu Hegel. - Wien/Berlin 2008 (Hg., gem.m. Wolfgang Kaltenbacher und Ludwig Nagl).\n Glauben und Wissen. Ein Symposium mit Jürgen Habermas. - Wien/Berlin 2007 (Hg., gem.m. Rudolf Langthaler).\n Geschichtsphilosophie und Kulturkritik. - Darmstadt 2003 (Hrsg., gem.m. Johannes Rohbeck).\n Feministische Philosophie. Ergebnisse, Probleme, Perspektiven. - Frankfurt a.M. 2000 a 2004 \n Continental Philosophy in Feminist Perspective. - Pennsylviania State University Press 2000 (Hg. gem.m. Cornelia Klingler).\n Der Sinn des Historischen. - Frankfurt a.M. 1996 (Hrsg.).\n Politische Theorie. Differenz und Lebensqualität. - Frankfurt a.M. 1996 (Hrsg. gem.m. Herlinde Pauer-Studer).\n Postkoloniales Philosophieren: Afrika. - Wien/München 1992 (Hrsg., gem.m. Franz Wimmer).\n Tod des Subjekts? - Wien/München 1987 (Hrsg., gem.m. Helmuth Vetter).\n Die Objektivität der Geschichtswissenschaft. Systematische Untersuchungen zum wissenschaftlichen Status der Historie. - Wien/München 1982\n spoluvydavateľka: Wiener Reihe. Themen der Philosophie (od 1986). \n spoluvydavateľka: Deutsche Zeitschrift für Philosophie (1993-2004). \n spoluvydavateľka: L'Homme. Europäische Zeitschrift für feministische Geschichtswissenschaft (1990 - 2003).\n\nOcenenia \n Förderpreis mesta Viedeň, 1983\n Käthe Leichter Preis (rakúska štátna cena), 1997 \n Preis für Geistes- und Sozialwissenschaften der Stadt Wien, 2009\n\nReferencie\n\nExterné odkazy \n Oficiálna stránka, Universität Wien \n Austria Forum, Wissenssammlungen/Biographien: Herta Nagl-Docekal\n\nZdroj \n\nRakúski filozofi",
  "question": "Kedy prišla na svet Herta Naglová-Docekalová?",
  "answers": {"answer_start": [28], "text": ["29. máj 1944"]}}
```

```json
{"context": "Martin Bareš (* 25. november 1968, Brno) je český profesor neurológie, od septembra 2019 rektor Masarykovej univerzity, predtým od februára 2018 do septembra 2019 dekan Lekárskej fakulty Masarykovej univerzity.\n\nRiadiace funkcie \nVo februári 2018 sa stal dekanom Lekárskej fakulty Masarykovej univerzity. Funkciu prevzal po Jiřím Mayerovi, ktorý zastával pozíciu dekana v období 20102018. S nástupom na post dekana ukončil svoje pôsobenie ako prorektor univerzity, ako i zástupca prednostu I. neurologickej kliniky pre vedu a výskum.\n\nDo funkcie rektora univerzity bol zvolený 1. apríla 2019 Akademickým senátom Masarykovej univerzity. V prvom kole tajnej voľby získal Bareš 36 hlasov z 50 prítomných senátorov. Protikandidáta, prodekana Prírodovedeckej fakulty Jaromíra Leichmana, volilo 11 senátorov. 3 odovzdané hlasy boli neplatné.\n\nSkúsenosti s pôsobením vo vedení školy zbieral Bareš v rokoch 20112018, kedy pôsobil najskôr ako jej prorektor pre rozvoj a potom ako prorektor pre akademické záležitosti. Za svoje priority označil Bareš v dobe voľby posilňovanie role univerzity ako piliera slobody v súčasnej spoločnosti a zvýšenie kvality vzdelávania, vedy a výskumu na medzinárodnej úrovni.\n\nDo funkcie rektora ho vymenoval 11. júna 2019 prezident Miloš Zeman s účinnosťou od 1. septembra 2019. Vo funkcii tak nahradil Mikuláša Beka, ktorému sa skončilo druhé volebné obdobie a o zvolenie sa teda už opäť uchádzať nemohol. Bareš k 1. septembru 2019 rezignoval na post dekana Lekárskej fakulty.\n\nVedecká činnosť \nJe prednášajúcim v odboroch všeobecné lekárstvo, zubné lekárstvo, optometria, fyzioterapia, neurofyziológia pre študentov prírodných vied Lekárskej fakulty Masarykovej univerzity a školiteľ doktorandov odborovej rady neurológia a neurovedy.\n\nPôsobí v týchto vedeckých radách: Masarykova univerzita, Lekárska fakulta Masarykovej univerzity a CEITEC MU. Ďalej tiež Univerzita Palackého v Olomouci, Lekárska fakulta UPOL, Fakulta veterinárního lékařství VFU, ďalej je tiež členom Českej lekárskej komory, Českej neurologickej spoločnosti, Českej spoločnosti klinickej neurofyziológie, Českej lekárskej spoločnosti Jana Evangelisty Purkyně, Movement Disorders Society, Society for the Research on the Cerebellum a Society for Neuroscience. Takisto je členom redakčnej rady časopisov Clinical Neurophysiology, Behavioural Neurology, Tremor and Other Hyperkinetic Movements a Biomedical Papers.\n\nOsobný život \nJe ženatý, má dvoch synov a dcéru.\n\nReferencie\n\nExterné odkazy \n Martin Bareš\n\nZdroj \n\nČeskí lekári\nNeurológovia\nRektori Masarykovej univerzity\nČeskí univerzitní profesori\nDekani Lekárskej fakulty Masarykovej univerzity\nAbsolventi Lekárskej fakulty Masarykovej univerzity\nOsobnosti z Brna",
"question": "Akú pozíciu mal Martin Bareš na Masarykovej univerzite počnúc septembrom 2019?",
"answers": {"answer_start": [89], "text": ["rektor"]}}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Nasledujú texty s pridruženými otázkami a odpoveďami.
  ```

- Base prompt template:

  ```text
  Text: {text}
  Otázka: {question}
  Odpoveď na maximálne 3 slová:
  ```

- Instruction-tuned prompt template:

  ```text
  Text: {text}

  Odpovedzte na nasledujúcu otázku týkajúcu sa textu uvedeného vyššie maximálne 3 slovami.

  Otázka: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-sk
```

## Knowledge

### MMLU-sk

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
  "text": "V akých smeroch je prípad pre humanitárnu intervenciu, ako je uvedené v tejto kapitol... mocnými štátmi.\nd. Všetky tieto možnosti.",
  "label": "d",
}
```

```json
{
  "text": "FAKTORIÁLOVÝ ANOVA sa používa v prípade, že štúdia zahŕňa viac ako 1 VI. Aký je INTER...činok VI na rovnakej úrovni ako ostatné VI",
  "label": "a"
}
```

```json
{
  "text": "Pre ktorú z týchto dvoch situácií urobí hlavná postava (ktorá používa ja/mňa/môj) nie...ie zlé\nc. Nie zlé, zlé\nd. Nie zlé, nie zlé",
  "label": "d",
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Nasledujú otázky s viacerými možnosťami (s odpoveďami).
  ```

- Base prompt template:

  ```text
  Otázka: {text}
  Odpoveď: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Otázka: {text}

  Odpovedzte na nasledujúcu otázku použitím 'a', 'b', 'c' alebo 'd', a nič iné.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset mmlu-sk
```

## Common-sense Reasoning

### HellaSwag-cs

This dataset is a machine translated version of the English [HellaSwag
dataset](https://doi.org/10.18653/v1/P19-1472). The dataset was translated using
[LINDAT Translation Service](https://lindat.mff.cuni.cz/services/translation/docs).

The original dataset has 10,000 samples. We use a 1,024 / 256 / 2,048 split for training,
validation and testing, respectively.

Here are a few examples from the training split (which have _not_ been post-edited):

```json
{
  "text": "Rybaření na ledu: Vidíme úvodní titulní obrazovku. Na sněhu a ledové rybě sedí muž a chlapec. My\nVýběr:\na. vidíme města a změny kolem nich.\nb. vidíme dole kreslenou animaci bocku.\nc. pak vidíme sport.\nd. vidíme titulní obrazovku a letadlo letí na obloze a v dálce vidíme lidi na ledu a náklaďák.",
  "label": "d"
}
```

```json
{
  "text": "Běh maratonu: Sportovci dávají rozhovory a někteří předvádějí medaile za účast. Sportovci nastupují do bílých autobusů. Autobusy\nVýběr:\na. se pohybují po silnici.\nb. odstartují z rampy.\nc. se pohybují po dráze a lidé skáčou po rampách.\nd. míjí několik sportovců sedících na zelených baldachýnech.",
  "label": "a"
}
```

```json
{
  "text": "Family Life: Jak uspořádat havajskou svatební hostinu. Vyberte tradiční havajský oděv pro nevěstu a ženicha. Havajská nevěsta tradičně nosí bílé dlouhé splývavé šaty s věncem z haku neboli prstenem z hawajských květin kolem hlavy. Havajský ženich tradičně nosí bílé kalhoty a bílou košili s pestrobarevnou šerpou kolem pasu.\nVýběr:\na. Nošení hawajského věnce při příležitosti vaší recepce může také pomoci cementovat hawajské svatební sliby. Havajské splývavé šaty jsou stále tradiční se svatebním oděvem, navzdory povaze svatby.\nb. Ženich také nosí kolem krku zelenou poštolku lei.. Vyberte hawajský oděv pro svatební hostinu.\nc. Tyto prvky spolu velmi dobře splývají. Fotografie se budou odehrávat ve velkém studiu na letišti v mělké vodě.\nd. Vyberte si neformální oděv na svatbu na pláži. Havajské svatby bývají velmi formální, takže si vyberte havajské svatební šaty s motivem kasina.",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Následující jsou otázky s výběrem z více možností (s odpověďmi).
  ```

- Base prompt template:

  ```text
  Otázka: {text}
  Možnosti:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Odpověď: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Otázka: {text}
  Možnosti:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Odpovězte na výše uvedenou otázku pomocí 'a', 'b', 'c' nebo 'd', a nic jiného.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset hellaswag-cs
```

## Summarisation

### Czech News

This dataset was published in
[this paper](https://doi.org/10.48550/arXiv.2307.10666) and contains news articles
from major online news outlets collected from 2000-2022.

The original dataset consists of 1,641,471 / 144,836 / 144,837 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 2,048 split for training,
validation and testing, respectively, sampled from the original splits.

Here are a few examples from the training split:

```json
{
  "text": "Vymetám zákoutí, oči na šťopkách, kde ještě něco zůstalo, abych to mohl popsat a tím popsáním zaevidovat, zkatalogizovat. Třeba na tom Smíchově, tam je to o život. Ale výsledky jsou. Před hostincem U Smolíků, šikmo naproti Smíchovskému nádraží, bylo na tabuli křídou, kterou vedla pevná ruka, naškrábáno jako součást nabídky: V úterý od 18 na hoře bez koz. Autor je bezesporu velkým básníkem, jeho rozmáchlá gesta nepotřebují dodatečné korekce. Představte si tu nádheru - po šesté večerní nastupuje na plac servírka plochá jak lineál. Chlapům sklapne čelist a o to více vypijí hladinek. Mezitím, než jsem dofabuloval, vy už přecházíte do haly zmíněného nádraží, na jehož pravém konci si četbymilovný odjížděč či přijížděč může zakoupit knihy v antikvariátu. A já zde zase jako vandal vyloupávám zasazené démanty, kterých si nikdo nevšímá. Nad přihrádkou, kde na obálkách knih a časopisů převažuje ta partie ženského těla, o které byla řeč výše, umístil prodejce sémanticky neobyčejně komplikovaný nápis: Erotika není k prohlížení! No není to krása? To, co "dělá" erotiku erotikou, je zde výslovně zapovězeno. Je třeba kupovat, a ne jen listovat a zadarmiko se vzrušovat! A já hned vytahuji zápisníček a v tom dusném nádražním prostředí zachycuji tuto opozdilou slzu ztracenou z grálu. Co s tím má co dělat ta sebelítost? Zatímco si tady hraju na soukromého badatele, který pak plody práce věnuje svému národu, v centru Prahy se dějou zásadní věci, proti kterým je tohle moje motýlkaření pouhým okresním přeborem. A je mi to líto. www.desir.cz 5. října 2004 proběhla v nejpoužívanějším pražském demonstračním prostoru Demonstrace za nic. Demonstranti nesli prázdné transparenty (povolené byly pouze tečky, vykřičníky a otazníky), dokonce i průhledné transparenty (ty byly absolutně transparentní) a rozdávali prázdné letáky. Akce byla řádně nahlášena, proto ji doprovázeli orgáni vpředu a vzadu. Končilo se (150-200 osob) pod ocasem, kde byla držena minuta ticha za nic. Geniální pakárna, švejkárna i kafkárna. Tento zásadní názor občanské angažovanosti proběhl pod taktovkou partičky jménem DĚSÍR (DĚti SÍdlištní Recese), která pořádá recesní a hravé akce v Praze se zaměřením na školáky ze SŠ a VŠ. Ve svém programu mají napsáno: Chceme využít městských prvků ve prospěch blaha hravých jedinců. Na jejich stránkách najdete mj. položky: Fotogalerie, Kronika, Kalendář akcí, Pravidla her. Podle návodu si můžete sami zahrát třeba hry Lapni dav nebo Piškvorky nabíjené tramvají. Domnívám se, už bez lítosti, že DĚSÍR je daleko více literárnější než mnoho praktikujících spisovatelů. Tohle je živá abeceda, tamti kladou už jen mrtvé litery.",
  "target_text": "Už dlouho jsem neprováděl cvičení v sebelítosti. Čas běží tak rychle, že zapomínám věnovat se těmto laciným koníčkům. Tak se v tom zas trošku procvičím.Jiní si užívají života, a já se tady pachtím jako motýlkář za prchavými křídly, za okamžiky, za těmi Hrabalovými perličkami"
}
```

```json
{
  "text": "Dillí - Indický nejvyšší soud zakázal turistiku ve stanovených zónách více než 40 tygřích rezervací pod správou centrální vlády. Šesti státům, které nedodržovaly předchozí směrnice, navíc uložil pokuty. Ve volné přírodě subkontinentu žije podle posledního sčítání z loňského roku kolem 1700 tygrů. Ještě před 100 lety přitom v indické divočině podle BBC žilo na 100 tisíc těchto kočkovitých šelem. Organizace na ochranu přírody verdikt soudu uvítaly. Rozhodnutí vychází vstříc příslušné petici, která žádala vytlačení komerčních turistických aktivit z oblastí nejčastějšího výskytu tygrů v rezervacích. V zónách stanovených soudem žije většina indických tygrů. Tygrům se daří také v pražské zoo: Související Pražská zoo představila tygří mláďata, jsou to samičky 6 fotografií I když je rozhodnutí soudu označováno za významné, není jasné, jaký dopad bude mít na turismus. Ten se soustřeďuje do takzvaných nárazníkových zón, což jsou až deset kilometrů široká pásma kolem vymezených zón. Soudní verdikt je jedním z řady kroků, které indické orgány v poslední době podnikly na ochranu tygrů. V únoru byla ve státě Rádžasthán přestěhována celá vesnice, jež musela zvířatům ustoupit. Opatření zjevně zabírají. Podle úřadů počet tygrů v Indii opět roste. Nadále je ale ohrožují lidé žijící uvnitř nebo na okraji rezervací.",
  "target_text": "Nejvyšší soud zakázal vstup do 40 tygřích rezervací"
}
```

```json
{
  "text": "V Klementinu byly například objeveny tři studny, pozůstatky kamenných domů nebo část trativodu z období 16. až 17. století. V základech barokní stavby byly objeveny části klenebních žeber či ostění oken, které s největší pravděpodobností pocházejí z konstrukcí středověkého kláštera odstraněného při výstavbě Klementina. Novinky o tom informovala Irena Maňáková z Národní knihovny ČR. Archeologové slaví v Národní knihovně mnoho úspěchů. FOTO: Národní knihovna ČR K nejvýznamnějšímu nálezu podle ní došlo při přesunu výzkumu ze západního křídla do traktu mezi Studentským a Révovým nádvořím, kde byly pod barokní podlahou suterénu odkryty zbytky zdiv náležejících k dominikánskému klášteru, který zde stál od 30. let 13. století. Odkryli i množství reliktů „Význam nálezu spočívá především v tom, že se jedná o první hmotný doklad tohoto kláštera, o němž jsme dosud věděli pouze z písemných pramenů,“ vysvětlil vedoucí archeolog Jan Havrda. Nálezy dokládají i výstavnost gotické stavby, jež ve své době představovala jednu z nejvýznamnějších pražských církevních institucí. Při výzkumu byly rovněž odkryty četné relikty středověké a raně novověké zástavby, která byla odstraněna v souvislosti s výstavbou této části barokního areálu v roce 1654. Vysoká památková hodnota Kromě pozůstatků kamenných domů byly odkryty i tři středověké studny, z nichž některé později sloužily jako odpadní jímky. Nejvýznamnějším nálezem v těchto prostorách byla lineární zděná konstrukce vystavěná románskou technikou. Zaznamenaná délka 0,7 metru široké zdi dosahuje 11,7 metru a její koruna se nalézala bezprostředně pod současnou podlahou sklepa. Archeologové učinili přes zimu hned několik zajímavých objevů. Pozůstatky dominikánského kláštera ze 13. století jsou však nejvýznamnější. FOTO: Národní knihovna ČR „Jedná se o unikátní architektonickou památku náležející ke skupině pražských profánních románských staveb, které představují nejstarší horizont kamenné architektury na území Pražské památkové rezervace a jejichž památková hodnota je nesporná. Interpretace tohoto nálezu není jednoznačná, mohlo by se však jednat o severní obvodovou zeď rozlehlejšího románského domu,“ uvedl Havrda.",
  "target_text": "Archeologové pracují přes zimu v Národní knihovně jako o život. V poslední době zkoumali klementinské suterény pod západním křídlem bývalé jezuitské koleje a sklepní trakt mezi Studentským a Révovým nádvořím. Nejvýznamnějším nálezem je objev zbytků zdiv, které poprvé hmotně dokládají existenci zdejšího dominikánského kláštera ze 13. století"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:

  ```text
  Následující jsou dokumenty s přiloženými souhrny.
  ```

- Base prompt template:

  ```text
  Dokument: {text}
  Souhrn: {target_text}
  ```

- Instruction-tuned prompt template:

  ```text
  Dokument: {text}

  Napište souhrn výše uvedeného dokumentu.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset czech-news
```
