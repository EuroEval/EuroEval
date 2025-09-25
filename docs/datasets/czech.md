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

  ```
  Následující jsou dokumenty s přiloženými souhrny.
  ```

- Base prompt template:

  ```
  Dokument: {text}
  Souhrn: {target_text}
  ```

- Instruction-tuned prompt template:

  ```
  Dokument: {text}

  Napište souhrn výše uvedeného dokumentu.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset czech-news
```
