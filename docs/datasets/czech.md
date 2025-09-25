# 🇨🇿 Czech

This is an overview of all the datasets used in the Czech part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### CSFD Sentiment

This dataset was published in [this paper](https://aclanthology.org/R13-1016/) and
consists of reviews from the the Czech Movie
Database (CSFD).

The original dataset contains 85,948 / 894 / 1503 samples for the training, validation, and
and test splits, respectively. We use 1,024 / 256 / 2,048 samples for our training,
validation and test splits, respectively. The train and validation splits are subsets
of the original splits. For the test split, we use all available test samples and
supplement with additional samples from the training set to reach 2,048 samples in
total.

Here are a few examples from the training split:

```json
{
    "text": "Stoprocentně nejlepší film všech dob... a tím samozřejmě zůstává a navždy zůstane. Tehdy ještě neznámý James Cameron dokázal obrovskou věc a jedním velkofilmem... ne, jedním zázrakem se dostal mezi špičku nejlepších režisérů filmové historie. A ano, jak už se někdo ptal, jak z toho někdo může být natšený - já z toho nadšený jsem a to se nezmění.",
    "label": "positive"
}
```

```json
{
    "text": "První film Woodyho Allena? Jen tak na půl. Vzhledem k tomu, že vzal již natočený japonský brak, sestříhal ho a předaboval, tak bych mu dal spíše titul režisér anglického znění. Ani to se však příliš nepovedlo – je zde pár pokusů o typický allenovský humor, ovšem je to ještě takové nijaké – jeho pravé komediální období má teprve přijít! Takže doporučuji spíše jen jako zajímavost pro milovníky tvorby Woodyho Allena.",
    "label": "neutral"
}
```

```json
{
    "text": "Tak jako písek v přesípacích hodinách, ubíhají dny našich životů, no já nevím, sledovat tenhle seriál, tak mi život neubíhá, ale pekelně se vleče...Je neuvěřitelné, kolik dokážou natočit dílů nesmyslného seriálu o ničem a je neuvěřitelné, kolik lidí se na to dokáže dívat, díky čemuž vznikají podobné katastrofy dodnes....",
    "label": "negative"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```
  Následují dokumenty a jejich sentiment, který může být 'pozitivní', 'neutrální' nebo 'negativní'.
  ```

- Base prompt template:

  ```
  Dokument: {text}
  Sentiment: {label}
  ```

- Instruction-tuned prompt template:

  ```
  Dokument: {text}

  Klasifikujte sentiment v dokumentu. Odpovězte pouze s 'pozitivní', 'neutrální', nebo 'negativní', a nic jiného.
  ```

- Label mapping:
  - `positive` ➡️ `pozitivní`
  - `neutral` ➡️ `neutrální`
  - `negative` ➡️ `negativní`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset csfd-sentiment-mini
```

## Named Entity Recognition

### PONER

This dataset was created [in this master thesis](https://dspace.vut.cz/items/6092e1b0-3d75-4451-8582-28573ac30404).
The dataset consists of 9,310 Czech sentences with 14,639 named entities.
Source data are Czech historical chronicles mostly from the first half of the 20th century.

The original dataset consists of 4,188 / 465 / 4,655 samples for the training, validation
and test splits, respectively.
We use 1,024 / 256 / 2,048 samples for our training, validation and test splits, respectively.
All the new splits are subsets of the original splits.

Here are a few examples from the training split:

```json
{
  "tokens": ["Předseda", "finanční", "komise", "města", "Julius", "Hegr"],
  "labels": ["O", "O", "O", "O", "B-PER", "I-PER"]
}
```

```json
{
  "tokens": ["Fot", ".", "dok", ".", "SV.", "I", "f.", "č.", "6."],
  "labels": ["O", "O", "O", "O", "O", "O", "O", "O", "O"],
}
```

```json
{
  "tokens": ["Konala", "se", "valná", "hromada", "Čtenářského", "spolku"],
  "labels": ["O", "O", "O", "O", "B-ORG", "I-ORG"],
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```
  Následující jsou věty a JSON slovníky s pojmenovanými entitami, které se v dané větě vyskytují.
  ```

- Base prompt template:

  ```
  Věta: {text}
  Pojmenované entity: {label}
  ```

- Instruction-tuned prompt template:

  ```
  Věta: {text}

  Identifikujte pojmenované entity ve větě. Měli byste to vypsat jako JSON slovník s klíči 'osoba', 'místo', 'organizace' a 'různé'. Hodnoty by měly být seznamy pojmenovaných entit tohoto typu, přesně tak, jak se objevují ve větě.
  ```

- Label mapping:
  - `B-PER` ➡️ `osoba`
  - `I-PER` ➡️ `osoba`
  - `B-LOC` ➡️ `místo`
  - `I-LOC` ➡️ `místo`
  - `B-ORG` ➡️ `organizace`
  - `I-ORG` ➡️ `organizace`
  - `B-MISC` ➡️ `různé`
  - `I-MISC` ➡️ `různé`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset poner-mini
```

### Unofficial: WikiANN-lv

This dataset was published in [this paper](https://aclanthology.org/P17-1178/) and is
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
    "tokens": array(["Iezīmē", "robežu", "starp", "Greiema", "Zemi", "ziemeļos", "un",
       "Pālmera", "Zemi", "Antarktīdas", "pussalas", "dienvidos", ",",
       "kā", "arī", "starp", "Faljēra", "krastu", "ziemeļos", "un",
       "Raimila", "krastu", "dienvidos", "."], dtype=object),
       "labels": ["O", "O", "O", "B-LOC", "I-LOC", "O", "O", "B-LOC", "I-LOC", "B-LOC", "I-LOC", "O", "O", "O", "O", "O", "B-LOC", "I-LOC", "O", "O", "B-LOC", "I-LOC", "O", "O"]
}
```

```json
{
    "tokens": array(["'", "''", "x-", "''", "Detroitas", "``", "Pistons", "''"],
      dtype=object),
      "labels": ["O", "O", "O", "O", "B-ORG", "I-ORG", "I-ORG", "I-ORG"]
}
```

```json
{
    "tokens": array(["Kārlis", "Gustavs", "Jēkabs", "Jakobi"], dtype=object),
    "labels": ["B-PER", "I-PER", "I-PER", "I-PER"]
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```
  Tālāk ir teikumi un JSON vārdnīcas ar nosauktajiem objektiem, kas parādās dotajā teikumā.
  ```

- Base prompt template:

  ```
  Teikums: {text}
  Nosauktie objekti: {label}
  ```

- Instruction-tuned prompt template:

  ```
  Teikums: {text}

  Identificējiet nosauktos objektus teikumā. Jums jāizvada šī informācija kā JSON vārdnīcu ar atslēgām 'persona', 'vieta', 'organizācija' un 'dažādi'. Vērtībām jābūt šī tipa nosaukto objektu sarakstiem, tieši tā, kā tie parādās teikumā.
  ```

- Label mapping:
  - `B-PER` ➡️ `persona`
  - `I-PER` ➡️ `persona`
  - `B-LOC` ➡️ `vieta`
  - `I-LOC` ➡️ `vieta`
  - `B-ORG` ➡️ `organizācija`
  - `I-ORG` ➡️ `organizācija`
  - `B-MISC` ➡️ `dažādi`
  - `I-MISC` ➡️ `dažādi`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset wikiann-lv
```

## Linguistic Acceptability

### ScaLA-cs

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Czech Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Czech-CAC) by assuming that the
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
  "text": "Tato skutečnost zásadně určuje i obsah politické čestnosti.",
  "label": "correct"
}
```

```json
{
  "text": "normálním průběhu sdělení se to, co je v předchozí větě jádrem, stává v další větě základem.",
  "label": "incorrect"
}
```

```json
{
  "text": "Zásady ukládají věnovat maximální pozornost hospodaření vodou a negativnímu ovlivňování životního prostředí, především čistoty vod ovzduší.",
  "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```
  Následující jsou věty a zda jsou gramaticky správné.
  ```

- Base prompt template:

  ```
  Věta: {text}
  Gramaticky správná: {label}
  ```

- Instruction-tuned prompt template:

  ```
  Věta: {text}

  Určete, zda je věta gramaticky správná nebo ne. Odpovězte 'ano', pokud je věta správná, a 'ne', pokud není. Odpovězte pouze tímto slovem, a ničím jiným.
  ```

- Label mapping:
  - `correct` ➡️ `ano`
  - `incorrect` ➡️ `ne`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-cs
```

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

  ```
  Následující jsou věty a zda jsou gramaticky správné.
  ```

- Base prompt template:

  ```
  Věta: {text}
  Gramaticky správná: {label}
  ```

- Instruction-tuned prompt template:

  ```
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

### SQAD

This dataset was published in
[this paper](https://nlp.fi.muni.cz/raslan/2019/paper14-medved.pdf)
and has been harvested from Czech Wikipedia articles by students and
annotated with appropriate question, answer sentence, exact answer,
question type and answer type.

The original full dataset has 11,569 / 2,819 train, test samples,
respectively. We use a 1,024 / 256 / 2,048 split for training,
validation and testing, respectively. The train and test splits are
subsets of the original splits, and the validation split is created
using examples from the train split.

Here are a few examples from the training split:

```json
{
  "context": "Právnická fakulta Masarykovy univerzity (PrF MU) je jedna z devíti fakult Masarykovy univerzity. Založena byla spolu s celou univerzitou v Brně roku 1919. V meziválečném období se proslavila školou normativní teorie práva, v roce 1950 byla zrušena a obnovena roku 1969. Sídlí v klasicizující budově na Veveří, nabízí vysokoškolské právní vzdělání na bakalářské (Bc.), magisterské (Mgr. a JUDr.) i doktorské (Ph.D.) úrovni a ve srovnání všech čtyř českých veřejných právnických fakult je pravidelně hodnocena jako nejlepší z nich. Související informace naleznete také v článku Seznam děkanů Právnické fakulty Masarykovy univerzity. Tradice univerzitní výuky práva na Moravě pochází z konce 17. století, v Brně se ale právo přednášelo jen v krátkém období 1778–1782, kdy sem byla přeložena olomoucká univerzita. Po zrušení její právnické fakulty v roce 1855 vznikla citelná potřeba existence nejen právnických studií, veškeré snahy o zřízení druhé české univerzity, která by byla situována do moravského hlavního města Brna a samozřejmě měla svou právnickou fakultu, v nichž se mj. angažoval tehdejší profesor a pozdější československý prezident T. G. Masaryk, však vyšly naprázdno. Bylo tomu tak zejména kvůli odporu Němců, kteří chtěli zachovat převážně německý charakter města, pouze některé právní obory byly vyučovány na české technice. Až po vzniku československé republiky mohla být tato myšlenka uskutečněna, roku 1919 vznikla Masarykova univerzita se sídlem v Brně a její právnická fakulta spolu s lékařskou zahájily výuku ještě ve školním roce 1919/1920.",
  "question": "Kolik fakult má Masarykova univerzita?",
  "answers": {
    "answer_start": array([60], dtype=int32),
    "text": array(["devíti"], dtype=object)
  }
}
```

```json
{
  "context": "Rovnátka (též označovaný jako ortodontický aparát) jsou druh zdravotnické pomůcky, která slouží k narovnání, napravení, či usměrnění růstu zubů. Mohou se nandávat jak na horní, tak i dolní čelist. Fixní rovnátka jsou v ústní dutině nepřetržitě po celou dobu léčby. Jsou nalepena buď z tvářové či jazykové strany (tzv. lingvální). Častějším typem je aplikace z tvářové strany. Důvody jsou takové, že linguální rovnátka jsou dražší, jejich zavedení je náročnější a klade větší nároky na lékaře i pacienta. Snímací rovnátka se vyznačují tím, že je lze vyjmout z ústní dutiny. Používají se pro méně závažné zubní anomálie a vady. Jsou určena pro dočasný a smíšený chrup. Fóliová rovnátka (tzv. neviditelná rovnátka) jsou měkké plastové fólie vyrobené pacientu na míru podle otisku čelistí. Tyto nosiče se v průběhu léčby obměňují. Jedná se v podstatě o speciální druh snímacích rovnátek, protože je lze z úst kdykoliv vyjmout. Neviditelná rovnátka jsou americkým patentem pod názvem Invisalign. Fixní aparát klade větší požadavky na ústní hygienu pacienta, neboť bylo prokázáno, že tvorba plaku je v průběhu nošení tohoto typu rovnátek vyšší. Pacientům s nedostatečnou ústní hygienou se fixní aparát nedoporučuje či mu přímo není umožněn. Fixním aparátem se dosahuje lepších výsledků než snímacím. Používá se jej spíše u závažnějších zubních anomálií. Snímací aparát je výrazně levnější, méně náročnější na hygienu. Lze jej kdykoliv sejmout, což je jeho nevýhoda - pacient není nucen nosit ho. Vstupní pohovor, vyšetření a jeho zadokumentování. Ortodontista pacienta seznámí o výsledcích vyšetření. Návrh léčebného plánu, schválení pacientem, zubní otisky, rentgenové snímky. Léčba, která se skládá ze dvou částí: Aktivní léčba je samotný proces, který by měl vést k nápravě chrupu a estetiky obličeje. Retenční fáze následuje po aktivní léčbě. Proces má za úkol udržet výsledky ortodontické léčby co nejdéle. Pokud je zanedbána, hrozí částečný či celkový návrat k původnímu stavu chrupu. Nejčastějším typem rovnátek je fixní aparát, a proto právě jeho skladba je zde rozebrána: Ortodontický drát (označovaný též jako oblouk) je speciální typ drátu užívaný v ortodoncii. Slouží k posunování zubu/ů. Drát je fixován do zámečků. V místě požádované změny pozice zubů pak mírně ohnutý. Díky svým vlastnostem (tzv. tvarové paměti) má pak v místě ohybu tendenci se rovnat (vrátit do původní polohy). Tím se vytváří síly, které tlačí na zuby.",
  "question": "Lze snímací rovnátka vyjmout z ústní dutiny?",
  "answers": {
    "answer_start": array([504], dtype=int32),
    "text": array(["Snímací rovnátka se vyznačují tím, že je lze vyjmout z ústní dutiny."], dtype=object)
  }
}
```

```json
{
  "context": "Patří mezi ně například switch, router, síťová karta apod. Pasivní prvky jsou součásti, které se na komunikaci podílejí pouze pasivně (tj. nevyžadují napájení) – propojovací kabel (strukturovaná kabeláž, optické vlákno, koaxiální kabel), konektory, u sítí Token Ring i pasivní hub. Opačným protipólem k sítím LAN jsou sítě WAN, jejichž přenosovou kapacitu si uživatelé pronajímají od specializovaných firem a jejichž přenosová kapacita je v poměru k LAN drahá. Uprostřed mezi sítěmi LAN a WAN najdeme sítě MAN. == Od historie k současnosti == První sítě LAN vznikly na konci 70. let 20. století. Sloužily k vysokorychlostnímu propojení sálových počítačů. Na začátku existovalo mnoho technologií, které navzájem nebyly kompatibilní (ARCNET, DECnet, Token ring a podobně). V současné době jsou nejpopulárnější LAN sítě vystavěné s pomocí technologie Ethernet. U osobních počítačů (PC) došlo k rozmachu budování LAN sítí po roce 1983, kdy firma Novell uvedla svůj produkt NetWare. Firma Novell byla v polovině 90. let odsunuta na okraj trhu nástupem firmy Microsoft s produkty Windows for Workgroups a Windows NT. Na počátku sítě LAN s osobními počítači používaly pro svoji jednoduchost rodinu protokolů IPX/SPX (případně NETBEUI, AppleTalk a další specializované proprietární protokoly), avšak s nástupem WWW byly na konci 90. let minulého století nahrazeny rodinou protokolů TCP/IP. == Moderní prvky LAN == V moderních sítích dnes nalézáme pokročilé technologie, které zvyšují jejich propustnost a variabilitu. Jednoduché propojovací prvky (opakovač, resp. HUB) jsou nahrazovány inteligentními zařízeními (bridge, resp. switch, router), které odstraňují kolize, omezují nežádoucí provoz v síti (broadcasty), umožňují monitorování, zabezpečení a další pokročilé zásahy do provozu sítě (např. detekce DoS, filtrování provozu a podobně).",
  "question": "Jak se jmenuje produkt firmy Novell, který způsobil rozmach LAN sítí?",
  "answers": {
    "answer_start": array([969], dtype=int32),
    "text": array(["NetWare"], dtype=object)
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```
  Následující texty obsahují otázky a odpovědi.
  ```

- Base prompt template:

  ```
  Text: {text}
  Otázka: {question}
  Odpověď maximálně 3 slovy: {label}
  ```

- Instruction-tuned prompt template:

  ```
  Text: {text}

  Odpovězte na následující otázku k výše uvedenému textu maximálně 3 slovy.

  Otázka: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset sqad-mini
```

## Knowledge

### MMLU-lv

This dataset is a machine translated version of the English [MMLU
dataset](https://openreview.net/forum?id=d7KBjmI3GmQ) and features questions within 57
different topics, such as elementary mathematics, US history and law. The translation to
Latvian was done by the Latvian Language Technology Initiative and is available in the
[VTI-Data
repository](https://github.com/LUMII-AILab/VTI-Data/tree/refs/heads/main/mmlu).

The full Latvian MMLU dataset consists of 13,941 samples. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
  "text": "Atrodiet skaitli, kas padara apgalvojumu patiesu: 48 m = lodziņš mm.\nIzvēles:\na. 48,000\nb. 480\nc. 4,800\nd. 5",
  "label": "a",
}
```

```json
{
  "text": "Darbinieks sagaida lielāku konsekvenci sava priekšnieka uzvedībā, nekā tas ir pamatoti. Kurš no šiem faktoriem vislabāk izskaidro darbinieka gaidas?\nIzvēles:\na. pārmērīgas attaisnošanas hipotēze\nb. pašlabuma aizspriedumainība\nc. fundamentālais atribūcijas aizspriedums\nd. pašpārbaudes teorija",
  "label": "c",
}
```

```json
{
  "text": "Kurā no šiem diviem scenārijiem galvenais varonis (kurš lieto es/jē/man) dara kaut ko, kas ir acīmredzami morāli nepareizs, saskaņā ar parastajām morāles normām ASV no 2020. gada? scenārijs | Es atslēdzu automašīnas kruīza kontroli. scenārijs | Es izsaucu policiju savam bijušajam mīļotajam, jo viņš mani piekāva.\nIzvēles:\na. Nepareizi, nepareizi\nb. Nepareizi, nav nepareizi\nc. Nav nepareizi, nepareizi\nd. Nav nepareizi, nav nepareizi",
  "label": "d",
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```
  Tālāk seko jautājumi ar vairākām atbilžu izvēlēm (ar atbildēm).
  ```

- Base prompt template:

  ```
  Jautājums: {text}
  Izvēles:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Atbilde: {label}
  ```

- Instruction-tuned prompt template:

  ```
  Jautājums: {text}
  Izvēles:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Atbildiet uz iepriekšējo jautājumu, atbildot ar 'a', 'b', 'c' vai 'd', un nekas cits.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset mmlu-lv
```

## Common-sense Reasoning

### COPA-lv

This dataset was published in [this
paper](https://aclanthology.org/2025.resourceful-1.22/) and is a translated version of
the English [COPA dataset](https://aclanthology.org/S12-1052/), which was created from
scratch by the authors. The dataset was machine translated using the [Tilde Translation
service](https://tilde.ai/machine-translation/), and the test samples were manually
post-edited.

The original full dataset consists of 214 / 57 / 132 samples, and we keep the splits
as-is.

Here are a few examples from the training split (which have _not_ been post-edited):

```json
{
  "text": "Īrnieki tika izlikti no dzīvokļa.\nIzvēles:\na. Viņi savu īri nemaksāja.\nb. Viņi sapratās ar savu saimnieku.",
  "label": "a"
}
```

```json
{
  "text": "Svešinieks man svešvalodā kliedza.\nIzvēles:\na. ES truli blenzu uz viņu.\nb. ES apstājos, lai papļāpātu ar viņu.",
  "label": "a"
}
```

```json
{
  "text": "Pagriezu gaismas slēdzi uz augšu un uz leju.\nIzvēles:\na. Gaisma izdzisa.\nb. Gaisma mirgoja.",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```
  Tālāk seko jautājumi ar vairākām atbilžu izvēlēm (ar atbildēm).
  ```

- Base prompt template:

  ```
  Jautājums: {text}
  Izvēles:
  a. {option_a}
  b. {option_b}
  Atbilde: {label}
  ```

- Instruction-tuned prompt template:

  ```
  Jautājums: {text}
  Izvēles:
  a. {option_a}
  b. {option_b}

  Atbildiet uz iepriekšējo jautājumu, atbildot ar 'a' vai 'b', un nekas cits.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset copa-lv
```

### Unofficial: Winogrande-lv

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2506.19468)
and is a translated and filtered version of the English [Winogrande
dataset](https://doi.org/10.1145/3474381).

The original full dataset consists of 47 / 1,210 samples for training and testing, and
we use the same splits.

Here are a few examples from the training split:

```json
{
  "text": "Pērkot māju, Patrīcijai nav tik daudz naudas, ko tērēt kā Tanjai, tāpēc _ nopērk vienas guļamistabas māju. Ko norāda tukšums _?\nIzvēles:\na. Opcija A: Patrīcija\nb. Opcija B: Tanja",
  "label": "a"
}
```

```json
{
  "text": "Es nevarēju kontrolēt mitrumu, kā es kontrolēju lietu, jo _ nāca no visām pusēm. Ko norāda tukšums _?\nIzvēles:\na. Opcija A: mitrums\nb. Opcija B: lietus",
  "label": "a"
}
```

```json
{
  "text": "Derriks nespēja koncentrēties darbā, atšķirībā no Džastina, jo _ bija jautrs darbs. Ko norāda tukšums _?\nIzvēles:\na. Opcija A: Derriks\nb. Opcija B: Džastins",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```
  Tālāk seko jautājumi ar vairākām atbilžu izvēlēm (ar atbildēm).
  ```

- Base prompt template:

  ```
  Jautājums: {text}
  Izvēles:
  a. {option_a}
  b. {option_b}
  Atbilde: {label}
  ```

- Instruction-tuned prompt template:

  ```
  Jautājums: {text}
  Izvēles:
  a. {option_a}
  b. {option_b}

  Atbildiet uz iepriekšējo jautājumu, atbildot ar 'a' vai 'b', un nekas cits.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset winogrande-lv
```

## Summarisation

### LSM

This dataset contains news articles and their corresponding summaries from the Latvian
public media news portal [LSM.lv](https://www.lsm.lv/).

Samples were collected using the
[lsm_scraper](https://github.com/alexandrainst/lsm_scraper). We use 1,024 / 256 / 2,048
samples for training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "FOTO: Raimonda Paula un Elīnas Garančas satikšanās koncertā «Ja tevis nebūtu...»\n\nIdeja svinēt apaļo jubileju uz vienas skatuves ar izcilo operdziedātāju Elīnu Garanču Maestro radusies, kopā uzstājoties jau pirms pieciem gadiem. Maestro neslēpj gandarījumu, ka pandēmijas dēļ pārceltais koncerts beidzot notiks. Raimonds Pauls koncertprogrammā “Ja tevis nebūtu...” dziedātājai veltījis divus jaunus dziesmu ciklus ar kopīgi atlasītu Vizmas Belševicas un Ojāra Vācieša dzeju. Savukārt koncerta otrajā daļā iekļautas Paula dziesmas no kinofilmām un teātra izrādēm. Kamerorķestra “Simfonietta Rīga” pavadījumā populāras melodijas atšķirīgās noskaņās izskanēs jaunos aranžējumos, ko veidojuši tādi izcili komponisti kā Lolita Ritmane, Rihards Dubra, Jēkabs Jančevskis un Raimonds Macats. “Man šī otrā daļa ar kino un teātra mūziku ir tāds sapnis, kas ir piepildījies. Jo šis žanrs mani vienmēr ir ļoti interesējis. Varētu teikt, ka es operas žanrā esmu nokļuvusi faktiski nejauši, jo sirds aicinājums no paša sākuma bija tieši teātris,” atklāj Elīna Garanča. Ojārs Rubenis atzīst: “Es varu tikai apbrīnot gan Maestro 85 gados – izturību un to darbu, ko viņš var izdarīt. Un, protams, arī Elīnu Garanču, kura vienkārši ir apbrīnojama savā neambiciozitātē pret visu pārējo un ambiciozitātē pret mākslu. Tas ir tas lielmākslinieku kods!” Maestro un Elīnas Garančas atkalsatikšanās Nacionālajā teātrī būs skatāma piektdien un sestdien, savukārt Latvijas Televīzijā šo koncertu varēs vērot šī gada rudenī.",
  "target_text": "Viņiem bija iecerēts tikties jau šī gada sākumā, bet pandēmijas dēļ Raimonda Paula 85. jubilejai veltītais koncerts ar pasaulslavenās operdziedātājas Elīnas Garančas piedalīšanos tika pārcelts. Šajā nedēļas nogalē Nacionālo teātri beidzot pieskandinās abu izcilo mūzikas personību atkalsatikšanās ar skatītājiem koncertā “Ja tevis nebūtu...”."
}
```

```json

{
"text": "Ukrainā tūkstošiem cilvēku protestē pret korupcijas apkarotāju vājināšanu; Zelenskis sola jaunu likumu\n\nCilvēki pauž neapmierinātību par\xa0korupcijas apkarotāju vājināšanu Trešdienas vakarā Kijivā\xa0bija pilns\xa0Ivana Franka laukums, kas ir tuvākā vieta pie prezidenta Volodimira Zelenska darba vietas, kur var brīvi piekļūt cilvēki. Pārsvarā gados jauni cilvēki bija sanākuši, lai paustu protestu, nožēlu un neapmierinātību ar Augstākās Radas pieņemto likumprojektu, kas paredz atcelt Ukrainas Korupcijas apkarošanas biroja un specializētās pretkorupcijas prokuratūras neatkarību, iestāžu pārraudzību nododot ģenerālprokuroram, kas ir politiski izraudzīts. Cilvēki skandēja visdažādākos saukļus – arī \"Rokas nost no NABU!\", \"Neklusē!\", \"Kauns!\", \"Slava Ukrainai!\", \"Varoņiem slava!\" un daudzus citus. Tā kā pamatā tie bija jaunieši, viņi bija ļoti skaļi un aktīvi. Rokās daudziem bija pašdarināti plakāti. Piemēram, \"Augstākā nodevība\" – spēlējoties ar Augstākās Radas jeb parlamenta nosaukumu. Kāds jaunietis arī bija izveidojis plakātu, kur puse sejas bija no prezidenta Zelenska, otra puse – no bēdīgi slavenā prokrieviskā eksprezidenta Viktora Janukoviča, kurš 2014.\xa0gadā pēc Eiromaidana jeb Pašcieņas revolūcijas asiņainajiem notikumiem aizbēga no Ukrainas un šobrīd slēpjas Krievijā. Aktīvisti Ukrainas protestā pret korupcijas apkarotāju vājināšanu 00:00 / 01:09 Lejuplādēt Indra Sprance Latvijas Radio parunājās ar dažiem no aktīvistiem. Marina: Esmu šeit, jo esmu ļoti sašutusi par pašreizējo situāciju ar likumprojektu. Ir pieņemts likums, kas pilnībā neatbilst Eiropas Savienības un tautas prasībām. Mēs atgriežamies pie tā stāvokļa, kāds bija 2013. gadā, kad mūsu tauta cīnījās par savu ceļu uz Eiropas Savienību. Mans brālis pašlaik karo Pokrovskas tuvumā. Visa šī situācija man šķiet kā spļāviens sejā visiem tiem karavīriem, kas mūs sargā, riskējot ar dzīvībām,\xa0– vara viņiem demonstrē, ka esam tuvāk nevis Eiropas Savienībai un mūsu Rietumu partneriem, bet Krievijai. Ihors: Man gandrīz visi vīriešu kārtas radinieki šobrīd karo, un man nav tiesību šobrīd stāvēt malā. Aleksa: Ukrainā šobrīd notiek ļoti briesmīgas lietas – kamēr daži cilvēki atdod savas dzīvības, lai mēs varētu šeit normāli dzīvot, kāds sagrauj valsti. Un tas nav labi. Mums šeit ir jābūt.\xa0 Tas ir svarīgi. Trešdienas vakarā protesta akcija notika arī Ukrainas otrā lielākajā pilsētā Harkivā, tur pēc \"Radio Brīvība\" aplēsēm bijis līdz pustūkstotim cilvēku. Protesti notikuši arī Černihivā, Zaporižjā, Ļvivā, Dņipro, Krivijrihā, Ivanofrankivskā, Ternopiļā, Odesā un citur. Šī ir jau otrā diena, kad cilvēki iziet ielās. Iepriekš tie bija spontāni protesti, reaģējot uz Augstākās Radas lēmumu, bet trešdien jau daudzviet cilvēkus ielās aicinājušas dažādas sabiedriskās organizācijas. Zelenskis sola jaunu likumu Prezidents Volodimirs Zelenskis trešdien bija noorganizējis tikšanos ar visu Ukrainas tiesību aizsardzības iestāžu vadītājiem, tajā skaitā abu pretkorupcijas iestāžu – NABU un specializētās prokuratūras vadītājiem. Saruna bijusi atklāta un vērtīga. Nākamnedēļ notikšot dziļāka darba tikšanās saistībā ar kopīgajiem darbiem. Pēcāk videouzrunā Zelenskis sacīja, ka ir sadzirdējis cilvēku bažas. Zelenskis piedāvās Augstākajai Radai savu – prezidenta likumprojektu, kas nodrošinās tiesību aizsardzības sistēmas spēku un to, ka nebūs nekāda Krievijas iejaukšanās iestāžu darbā. Jau vēlāk Zelenskis likumprojektu iesniedzis. Vēl gan nav skaidrs, kas tieši šajā likumprojektā ir un kad tieši par to balsos parlaments. Kā likumprojektu komentējis Zelenskis, tas paredz pilnīgas korupcijas apkarošanas iestāžu neatkarības garantijas. Tas arī paredzot reālas iespējas pārliecināties, ka iestāžu darbībā neiejaucas Krievija. Ikvienam, kam ir pieeja valsts noslēpumiem - ne tikai Nacionālajam pretkorupcijas birojam un Specializētajai pretkorupcijas prokuratūrai, bet arī Valsts izmeklēšanas birojam un Valsts policijai - ir jāveic melu detektora pārbaudes un tām jābūt regulārām, likumprojekta saturu komentēja Zelenskis. Likumprojektā ir iekļauti arī noteikumi, kas aizsargā pret dažādiem pārkāpumiem, piebilda prezidents. Pēc jaunā likumprojekta pārskatīšanas Nacionālais pretkorupcijas birojs paziņojumā norādīja, ka ierosinātais likumprojekts patiesi atjaunos visas procesuālās pilnvaras un neatkarības garantijas gan birojā, gan Specializētajā pretkorupcijas prokuratūrā. Arī Ukrainas Korupcijas apkarošanas rīcības centrs, kas ir uzraudzības iestāde, atbalstīja iniciatīvu, sakot, ka tā atjaunos principus, ko iepriekš bija nojaukusi Augstākā Rada. Centrs gan brīdināja, ka pat vienas nedēļas kavēšanās var būt pietiekama, lai iznīcinātu virkni abās pretkorupcijas iestādēs esošās tiesvedības pret augstākajām korumpētajām amatpersonām. KONTEKSTS: Ukrainas parlaments 22. jūlijā apstiprināja likuma grozījumus, kas mazina Ukrainas korupcijas apkarošanas iestāžu neatkarību. Ukrainas Nacionālais pretkorupcijas birojs (NABU) un specializētā prokuratūra turpmāk būs pakļauti Ukrainas ģenerālprokuroram, kas ir Ukrainas prezidenta Volodimira Zelenska izvirzīta amatpersona. Tas izraisījis bažas par korupcijas apkarošanas dienestu pakļaušanu Zelenska komandas interesēm. Ukrainas Drošības dienests iepriekš veicis plaša mēroga kratīšanas pie NABU un specializētās prokuratūras darbiniekiem. Šie soļi izraisījuši protestus Ukrainas iekšienē, kā arī kritiku no Ukrainas partneriem, kas raizējas par demokrātijas standartu vājināšanu un nepietiekamo aktivitāti korupcijas apkarošanā. Tas varētu apgrūtināt Ukrainas izredzes kļūt par Eiropas Savienības dalībvalsti.",
"target_text": "Ukrainā trešdienas vakarā, reaģējot uz šonedēļ lielā steigā pieņemto likumu, kas atceļ pretkorupcijas iestāžu neatkarību, tūkstošiem cilvēku izgāja ielās. Latvijas Radio bija klāt Kijivā, kur pulcējās liels skaits cilvēku."
}
```

```json
{
"text": "Norvēģijas dziesma Eirovīzijā – folkmūzikas, elektronikas un viduslaiku estētikas sintēze\n\nAlessandro ir spāņu izcelsmes, viņš runā četrās valodās, iedvesmojas no dažādu pasaules tautu mūzikas, kā arī ir labs dejotājs. Alessandro dziesma \"Lighter\" ieturēta popmūzikas stilistikā, kurā ievīti daudz dažādi elementi. Te var sadzirdēt gan norvēģu folkmūzikas, gan elektroniskās deju mūzikas notis, gan Balkānu popmūzikai raksturīgos ritmus un pat viduslaiku estētiku. Dziesma aicina noticēt sev un būt pašam savai dzirkstij. Dziesmas \"Lighter \" vārdi Golden girl dressed in ice A heart as dark as night You got me to dim my light No more, (no more) I really think I bought your lies Did anything to keep you mine You kept me hooked on your line No more, (no more) Somewhere along the way I lost my mind I had to walk a hundred thousand miles I’m not afraid to set it all on fire I won’t fall again, I’ll be my own lighter (Eh-Eh-Eh-Eh) Nothing can burn me now (Eh-Eh-Eh-Eh) I’ll be my own lighter I feel a spark inside me I don’t need saving (No way, no way) ‘Cause I’m my own, I’m my own lighter I’m tired of a million tries To fight, the signs And when everybody tried to tell me I should’ve known that it was time to break free Your reigns that kept me at your mercy I’ll burn them to the ground No more, no more Ignite the fire Somewhere along the way I lost my mind I had to walk a hundred thousand miles I’m not afraid to set it all on fire I won’t fall again, I’ll be my own lighter (Eh-Eh-Eh-Eh) Nothing can burn me now (Eh-Eh-Eh-Eh) I’ll be my own lighter I feel a spark inside me I don’t need saving (No way, no way) ‘Cause I’m my own, I’m my own lighter Silence fills the room And I’ve taken off my jewels I wish none of this was true But there’s a fire growing too Yeah! (Eh-Eh-Eh-Eh) Nothing can burn me now (Eh-Eh-Eh-Eh) I’ll be my own lighter I feel a spark inside me I don’t need saving (No way, no way) ‘Cause I’m my own, I’m my own lighter (Eh-Eh-Eh-Eh) Nothing can burn me down (Eh-Eh-Eh-Eh) I’m my own, I’m my own lighter Eirovīzija\xa02025 – dalībnieki Vairāk KONTEKSTS: 2025. gada Eirovīzijas dziesmu konkurss notiks Šveicē, Bāzelē, un savu dalību tajā apstiprinājušas 37 valstis. 31 no visām dalībvalstīm sacentīsies pusfinālos\xa013. maijā un 15. maijā. Desmit\xa0labākie no katra pusfināla kvalificēsies lielajam finālam 17. maijā, pievienojoties pērnā gada uzvarētājai Šveicei un \"lielajam piecniekam\" – Spānijai, Apvienotajai\xa0Karalistei, Vācijai, Itālijai un Francijai. Eirovīzijas konkursa pusfināli un fināli šogad sāksies pulksten 22.00 pēc Latvijas laika. Tiešraides būs skatāmas Latvijas Sabiedriskā medija portālā LSM.lv un satura atskaņotājā REplay.lv, kā arī LTV1. Šī gada Latvijas nacionālajā atlasē \"Supernova\" uzvarēja un uz Eirovīziju dosies grupa \"Tautumeitas\" . \"Tautumeitas\" kāps uz skatuves Eirovīzijas konkursa otrajā pusfinālā. Tajā kopā ar Latviju piedalīsies arī Armēnija, Austrālija, Austrija, Grieķija, Īrija, Lietuva, Melnkalne, Čehija, Dānija, Somija, Gruzija, Izraēla, Luksemburga, Malta un Serbija.",
"target_text": "Norvēģiju Eirovīzijas dziesmu konkursā pārstāv jaunais dziedātājs Kails Alessandro ( Kyle Alessandro ). Plašāka auditorija dziedātāju iepazina jau 10 gadu vecumā, kad viņš veiksmīgi piedalījās\xa0TV šovā \"Norway’s Got Talent\"."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:

  ```
  Tālāk ir dokumenti ar pievienotām kopsavilkumiem.
  ```

- Base prompt template:

  ```
  Dokuments: {text}
  Kopsavilkums: {target_text}
  ```

- Instruction-tuned prompt template:

  ```
  Dokuments: {text}

  Uzrakstiet kopsavilkumu par iepriekš minēto dokumentu.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset lsm
```
