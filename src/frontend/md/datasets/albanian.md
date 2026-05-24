# 🇦🇱 Albanian

This is an overview of all the datasets used in the Albanian part of EuroEval. The
datasets are grouped by their task – see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### MMS-sq

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2306.07902).
The corpus consists of 79 manually selected datasets from over 350 datasets reported in
the scientific literature based on strict quality criteria.

The original dataset contains a single split with 44,284 Albanian samples.
We use 1,024 / 256 / 2,048 samples for our training, validation, and test splits,
respectively.
We have employed stratified sampling based on the label column from the original
dataset to ensure balanced splits.

Here are a few examples from the training split:

```json
{
    "text": "Cirku politik në Nju Jork nga Frank SHKRELI",
    "label": "positive"
}
```

```json
{
    "text": "Balkanweb - Kulturë | \"Si manipuloheshin tekstet e këngëve polifonike para 1990-ës\" - http://t.co/hk0LNEGYah",
    "label": "negative"
}
```

```json
{
    "text": "RT @fislami3: Mos trokit në derën e dikujt që ia hap gjithkujt !!",
    "label": "negative"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Më poshtë janë dokumentet dhe ndjenjat e tyre, të cilat mund të jenë pozitive, neutrale, ose negative.
  ```

- Base prompt template:

  ```text
  Dokument: {text}
  Ndjenja: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Dokument: {text}

  Klasifikoni ndjenjën në dokument. Përgjigjuni vetëm me pozitive, neutrale, ose negative, dhe asgjë tjetër.
  ```

- Label mapping:
  - `positive` ➡️ `pozitive`
  - `neutral` ➡️ `neutrale`
  - `negative` ➡️ `negative`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset mms-sq
```

## Named Entity Recognition

### WikiANN-sq

This dataset was published in [this paper](https://aclanthology.org/P17-1178/) and is
part of a cross-lingual named entity recognition framework for 282 languages from
Wikipedia. It uses silver-standard annotations transferred from English through
cross-lingual links and performs both name tagging and linking to an english Knowledge
Base.

The original full dataset consists of 5,000 / 1,000 / 1,000 samples for the training,
validation and test splits, respectively. We use 1,024 / 256 / 2,048 samples for our
training, validation and test splits, respectively. All the new splits are subsets of
the original splits.

Here are a few examples from the training split:

```json
{
  "tokens": ["Enver", "Hoxha", ",", "politikan", ",", "ministër", ",", "kryeministër", ",", "burrë", "shteti", ",", "diktator"],
  "labels": ["B-PER", "I-PER", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"],
}
```

```json
{
  "tokens": ["'", "''", "Meksika", "''", "'", "-"],
  "labels": ["O", "O", "B-LOC", "O", "O", "O"],
}
```

```json
{
    "tokens": ["Devil", "May", "Cry", "(", "anime", ")"],
    "labels": ["B-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG"],
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Më poshtë janë fjali dhe fjalorë JSON me entitetet e emërtuara që shfaqen në fjalinë e dhënë.
  ```

- Base prompt template:

  ```text
  Fjali: {text}
  Entitete të emërtuara: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Fjali: {text}

  Identifikoni entitetet e emërtuara në fjali. Duhet t’i jepni ato si një fjalor JSON me çelësat 'person', 'vendndodhje', 'organizatë' dhe 'të ndryshme'. Vlerat duhet të jenë lista të entiteteve të emërtuara të atij lloji, saktësisht ashtu siç shfaqen në fjali.
  ```

- Label mapping:
  - `B-PER` ➡️ `person`
  - `I-PER` ➡️ `person`
  - `B-LOC` ➡️ `vendndodhje`
  - `I-LOC` ➡️ `vendndodhje`
  - `B-ORG` ➡️ `organizatë`
  - `I-ORG` ➡️ `organizatë`
  - `B-MISC` ➡️ `të ndryshme`
  - `I-MISC` ➡️ `të ndryshme`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset wikiann-sq
```

## Linguistic Acceptability

### ScaLA-sq

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Albanian Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Albanian-STAF) and
[Albanian TSA](https://github.com/UniversalDependencies/UD_Albanian-TSA/tree/master)
by assuming that
the documents in the treebank are correct, and corrupting the samples to create
grammatically incorrect samples. The corruptions were done by either removing a word
from a sentence, or by swapping two neighbouring words in a sentence. To ensure that
this does indeed break the grammaticality of the sentence, a set of rules were used on
the part-of-speech tags of the words in the sentence.

The original full dataset consists of 160 / 20 / 80 samples for training,
validation and testing, respectively.
We use a 128 / 64 / 210 split for training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Ishte fillimi i shtatorit '99, vjeshta e fundit e mijëvjeçarit të dytë.",
  "label": "correct"
}
```

```json
{
  "text": "Ja, përsëri rashë brenda me këmbët e mia, siç bie miza në rrjetën e merimangës.",
  "label": "correct"
}
```

```json
{
  "text": "Do të vënë buzën në gaz, do t'u shkojë mendja te nofka ime.",
  "label": "correct"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Më poshtë janë fjali dhe nëse janë gramatikisht të sakta.
  ```

- Base prompt template:

  ```text
  Fjali: {text}
  Gramatikisht e saktë: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Fjali: {text}

  Përcaktoni nëse fjalia është gramatikisht e saktë apo jo. Përgjigjuni me po, ose jo, dhe asgjë tjetër.
  ```

- Label mapping:
  - `correct` ➡️ `po`
  - `incorrect` ➡️ `jo`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-sq
```

## Reading Comprehension

### MultiWikiQA-sq

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

The original full dataset consists of 5,006 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
  "context": "E drejta e lindjes është koncepti i gjërave që i detyrohen një personi me ose nga fakti i lindjes së tij, ose për shkak të rendit të lindjes së tyre. Këto mund të përfshijnë të drejtat e shtetësisë bazuar në vendin ku ka lindur personi ose shtetësinë e prindërve të tyre, dhe të drejtat e trashëgimisë për pronën e prindërve ose të tjerëve.\n\nKoncepti i të drejtës së lindjes është i lashtë dhe shpesh përkufizohet pjesërisht me konceptet e patriarkatit dhe rendit të lindjes. Për shembull, \"[në] gjithë Biblën koncepti i së drejtës së lindjes është absolutisht i ndërthurur me të parëlindurin. Kjo do të thotë, i parëlinduri trashëgon të drejtën e lindjes dhe ka pritshmëri të parësore\",  që historikisht i referohej së drejtës, me ligj ose zakoni, që fëmija i parëlindur legjitim të trashëgojë të gjithë pasurinë ose pasurinë kryesore të prindit në përparësi ndaj trashëgimisë së përbashkët midis të gjithë ose disa fëmijëve, çdo fëmije jashtëmartesor ose ndonjë të afërmi kolateral.  Në shekullin e shtatëmbëdhjetë, aktivisti anglez John Lilburne përdori termin në lidhje me të drejtat e anglezëve \"për të nënkuptuar gjithçka që i takon një qytetari\" të Anglisë, gjë që \"pretendohet nga ligji anglez tek autoritetet më të larta\".  Termi u popullarizua në mënyrë të ngjashme në Indi nga avokati i vetëqeverisjes Bal Gangadhar Tilak në vitet 1890, kur Tilak miratoi sloganin e shpikur nga bashkëpunëtori i tij Kaka Baptista: \"Swaraj (vetëqeverisja) është e drejta ime e lindjes dhe unë do ta kem atë\".  Termi më pas \"arriti statusin e një slogani politik\". \n\nNë kontekstin e të drejtave të qytetarisë, \"[t] termi e drejta e lindjes sinjalizon jo vetëm që anëtarësimi fitohet në lindje ose në bazë të lindjes, por gjithashtu se anëtarësimi është supozuar një status i përjetshëm për individin dhe i vazhdueshëm përgjatë brezave për qytetarin. si kolektiv”.  Shtetësia e të drejtës së lindjes ka qenë prej kohësh një tipar i ligjit të përbashkët anglez .  Rasti i Calvinit, [9] ishte veçanërisht i rëndësishëm pasi vendosi se, sipas ligjit të zakonshëm anglez, \"statusi i një personi ishte dhënë në lindje, dhe bazuar në vendin e lindjes - një person i lindur brenda dominimit të mbretit i detyrohej besnikërisë ndaj sovranit, dhe në nga ana tjetër, kishte të drejtën e mbrojtjes së mbretit.\"  I njëjti parim u pranua nga Shtetet e Bashkuara si \"i lashtë dhe themelor\", d.m.th., e drejta e zakonshme e themeluar mirë, siç thuhet nga Gjykata e Lartë në interpretimin e saj të vitit 1898 të Amendamentit të Katërmbëdhjetë të Kushtetutës së Shteteve të Bashkuara në Shtetet e Bashkuara. v. Wong Kim Ark: \"Amendamenti i Katërmbëdhjetë pohon rregullin e lashtë dhe themelor të shtetësisë me lindje brenda territorit, në besnikëri dhe nën mbrojtjen e vendit, duke përfshirë të gjithë fëmijët e lindur këtu nga të huajt rezidentë, me përjashtime ose kualifikime ( aq i vjetër sa vetë rregulli) të fëmijëve të sovranëve të huaj ose ministrave të tyre, ose të lindur në anije publike të huaja, ose të armiqve brenda dhe gjatë një pushtimi armiqësor të një pjese të territorit tonë, dhe me përjashtimin e vetëm shtesë të fëmijëve të anëtarëve të Fiset indiane për shkak të besnikërisë së drejtpërdrejtë ndaj disa fiseve të tyre\". \n\nKoncepti i së drejtës së lindjes që rrjedh nga pjesëmarrja në një kulturë të caktuar është demonstruar në programin Birthright Israel, i iniciuar në 1994.  Programi ofron udhëtime falas për të vizituar Izraelin për personat që kanë të paktën një prind me prejardhje të njohur hebreje, ose që janë konvertuar në judaizëm nëpërmjet një lëvizjeje të njohur hebraike dhe që nuk praktikojnë në mënyrë aktive një fe tjetër. Ata gjithashtu duhet të jenë nga mosha 18 deri në 32 vjeç, pas shkollës së mesme, as të kenë udhëtuar më parë në Izrael në një udhëtim arsimor ose program studimi për bashkëmoshatarët pas moshës 18 vjeç dhe as të kenë jetuar në Izrael mbi moshën 12 vjeç.\n\nShiko gjithashtu \n\n Shtetësia\n Diskriminim\n Monarki trashëgimore\n Monarkia\n Pabarazia ekonomike\n\nReferencat \n\nTë drejtat e njeriut",
  "question": "Cilat koncepte lidhen me konceptin e së drejtës për t'u lindur?",
  "answers": {
    "answer_start": [440],
    "text": ["patriarkatit dhe rendit të lindjes"]
  }
}
```

```json
{
  "context": "Në fizikë, nxitimi këndor (simboli α, alfa) është shkalla kohore e ndryshimit të shpejtësisë këndore. Pas dy llojeve të shpejtësisë këndore, shpejtësia këndore e rrotullimit dhe shpejtësia këndore orbitale, llojet përkatëse të nxitimit këndor janë: nxitimi këndor rrotullues, që përfshin një trup të ngurtë rreth një boshti rrotullimi që kryqëzon qendrën e trupit; dhe nxitimi këndor orbital, që përfshin një pikë materiale dhe një bosht të jashtëm.\n\nNxitimi këndor ka dimensione fizike të këndit për kohë në katror, të matur në njësi SI të radianeve për sekondë në katror (rad ⋅ s⁻²). Në dy dimensione, nxitimi këndor është një pseudoskalar, shenja e të cilit merret si pozitive nëse shpejtësia këndore rritet në të kundërt ose zvogëlohet në drejtim të akrepave të orës, dhe merret si negative nëse shpejtësia këndore rritet ose zvogëlohet në drejtim të kundërt. Në tre dimensione, nxitimi këndor është një pseudovektor.\n\nPër trupat e ngurtë, nxitimi këndor duhet të shkaktohet nga një çift rrotullues i jashtëm neto. Megjithatë, kjo nuk është kështu për trupat jo të ngurtë: Për shembull, një patinator mund të përshpejtojë rrotullimin e tij (duke marrë kështu një nxitim këndor) thjesht duke kontraktuar krahët dhe këmbët nga brenda, gjë që nuk përfshin asnjë çift rrotullues të jashtëm.\n\nNxitimi këndor orbital i një pike materiale\n\nPika në dy dimensione\nNë dy dimensione, nxitimi këndor orbital është shpejtësia me të cilën ndryshon shpejtësia këndore orbitale dydimensionale e grimcës rreth origjinës. Shpejtësia këndore e çastit në çdo moment të kohës jepet nga ...\n\nPrandaj, nxitimi këndor i çastit α i grimcës jepet nga ...\n\nNë rastin e veçantë kur grimca pëson lëvizje rrethore rreth origjinës, ...\n\nPika materiale në tre dimensione\nNë tre dimensione, nxitimi këndor orbital është shpejtësia në të cilën vektori i shpejtësisë këndore orbitale tredimensionale ndryshon me kalimin e kohës. Vektori i shpejtësisë këndore të çastit në çdo moment në kohë jepet nga ...\n\nPrandaj, nxitimi këndor orbital është vektori i përcaktuar nga ...\n\nNë rastin kur largësia e grimcës nga origjina nuk ndryshon me kalimin e kohës (e cila përfshin lëvizjen rrethore si nënrast), formula e mësipërme thjeshtohet në ...\n\nNga ekuacioni i mësipërm, mund të rikuperohet nxitimi kryq rrezor në këtë rast të veçantë si ...",
  "question": "Cilat janë njësitë e nxitimit këndor?",
  "answers": {
    "answer_start": [489],
    "text": ["të këndit për kohë në katror"]
  }
}
```

```json
{
  "context": "\n\nNgjarje \n 1910 – Theodore Roosevelt mbajti fjalimin e njohur si Njeriu në Arenë.\n 1932 – U dogj De Adriaan Windmill 153 vite i vjetër në Haarlem, Holandë.\n 1985 – Coca-Cola ndryshon formulën dhe paraqet në treg produktin e ri New Coke. Pritja ishte e pa pritur, negative, dhe brenda tre muajve u rikthye formula e vjetër.\n 1993 – Eritreanët votuar për pavarësi nga Etiopia. Referendumi u vrojtua Bashkimi Evropian.\n 1997 – U realizua masakra në Omaria, Algjeri ku mbetën të vrarë 42 fshatarë.\n 2003 – Në Bejxhin u mbyllën shkollat për dy javë për shkak të virusit SARS.\n\nLindje \n 1185 - Afonso II, mbret i Portugalisë (v. 1233) \n 1858 - Max Planck, fizikan gjerman (v. 1947)\n 1564 - William Shakespeare, shkrimtar anglez (v. 1616)\n 2001 Berat Emini futbollist i njohur Shqipetar\n\nVdekje \n 997 - Shën Vojciech Sławnikowic, peshkop i Pragës, mbrojtës i Polonisë\n 1605 - Boris Godunov, perandor i Rusisë (lindi më 1551)\n 1616 - William Shakespeare\n 1998 - Konstantinos Karamanlis, politikan grek (l. 1907)\n 2007 - Boris Yeltsin, politikan rus, presidenti i parë i Rusisë (l. 1931)\n\nFesta dhe përvjetore \n Dita ndërkombetare e librit dhe të drejtave të autorit\n\nPrill",
  "question": "Cili ishte mulli me erë që u shkatërrua nga zjarri në Haarlem të Holandës më 1932?",
  "answers": {
    "answer_start": [98],
    "text": ["De Adriaan Windmill"]
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

```text
Më poshtë janë tekste me pyetje dhe përgjigje.
```

- Base prompt template:

```text
Teksti: {text}
Pyetja: {question}
Përgjigje me jo më shumë se 3 fjalë:
```

- Instruction-tuned prompt template:

```text
Teksti: {text}

Përgjigjuni pyetjes së mëposhtme rreth tekstit të mësipërm me jo më shumë se 3 fjalë.

Pyetja: {question}
```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-sq
```

## Knowledge

### GlobalMMLU-lite-sq

Global-MMLU is a machine translated version of the English [MMLU
dataset](https://openreview.net/forum?id=d7KBjmI3GmQ) and features questions within 57
different topics, such as elementary mathematics, US history and law. The translation to
Ukrainian was done by the [Cohere Labs Community](https://cohere.com/research).

The original full dataset consists of 215 / 400 samples for
validation and testing, respectively. We use a 128 / 64 / 404 split for training,
validation and testing, respectively (so 3,328 samples used in total).
The train set is sample from the validation set. Remaining validation samples
are used in the test set.

Here are a few examples from the training split:

```json
{
  "text": "Projektimi ambiental që sjell bashkë kafshë shtëpiake dhe bimë në shtëpi kujdesi, si dhe të rinj e fëmijë si vizitorë të rregullt, njihet si\nOpsione:\na. Projektimi Instrumental i Jetesës\nb. Zgjedhja humaniste\nc. Harmonia njeri-ambient\nd. Alternativa e Edenit",
  "label": "d",
}
```

```json
{
  "text": "Cilën nga të mëposhtmet duhet të bëjë një përdorues kompjuteri për të parandaluar lodhjen e syve të shkaktuar nga ekrani?\nOpsione:\na. Të përdorë pika sysh rregullisht\nb. Të ndryshojë zgjedhjen e softuereve\nc. Të kufizojë rezolucionin në monitor\nd. Të bëjë pushime të rregullta për të parë dritën natyrore",
  "label": "d",
}
```

```json
{
  "text": "Cila nga alternativat vijuese është më e gjatë?\nOpsione:\na. Jetëgjatësia në lindje (LEAB)\nb. Kohëzgjatja e jetës\nc. Jetëgjatësia deri në një moshë të caktuar (LEASA)\nd. Nuk ka mënyrë për ta ditur",
  "label": "b",
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Më poshtë janë pyetje me zgjedhje të shumëfishtë (me përgjigje).
  ```

- Base prompt template:

  ```text
  Pyetje: {text}
  Përgjigje: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Pyetje: {text}

  Përgjigjuni pyetjes së mësipërme duke u përgjigjur me {labels_str}, dhe asgjë tjetër.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset global-mmlu-lite-sq
```

### Unofficial: INCLUDE-sq

This dataset is part of [INCLUDE](https://doi.org/10.48550/arXiv.2411.19799), a
comprehensive knowledge- and reasoning-centric benchmark that evaluates multilingual
LLMs across 44 languages. It contains 4-option multiple-choice questions extracted from
academic and professional exams, covering 57 topics including regional knowledge.

The original dataset consists of a 'validation' split used as training data and a 'test'
split. We use the 'validation' split as the training split, which has 25 samples. We
sample 64 samples from the 'test' split for the validation split, and use the remaining
512 samples for the test split. The sampling is done stratified by the subject column.

Here are a few examples from the dataset:

```json
{
    "text": "Në cilë teori sociologjike pohohet se stratifikimi është i arsyeshëm dhe i pashmangshëm dhe se kontribuon për barazpeshë në shoqëri?\nOpsione:\na. Teorinë funksionaliste.\nb. Teorinë e konflikteve.\nc. Teorinë Marksiste.\nd. Teorinë e Maks Veberit.",
    "label": "a",
    "subject": "Sociology"
}
```

```json
{
    "text": "Cilët janë stilet kryesore të menaxhmentit?\nOpsione:\na. autoritar, demokratik dhe „qetësi në shtëpi“.\nb. autoritar, demokratik, fleksibil.\nc. Totalitar, demokratik, fleksibil.\nd. autoritar, fleksibil dhe „qetësi në shtëpi“.",
    "label": "b",
    "subject": "Business"
}
```

```json
{
    "text": "Në cilën periudhë mbisundon kuptimi që bukuria ka prejardhje hyjnore?\nOpsione:\na. periudhën antike.\nb. post modernizëm.\nc. mesjetë.\nd. klasicizëm.",
    "label": "c",
    "subject": "Visual Arts"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Më poshtë janë pyetje me zgjedhje të shumëfishtë (me përgjigje).
  ```

- Base prompt template:

  ```text
  Pyetje: {text}
  Përgjigje: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Pyetje: {text}

  Përgjigjuni pyetjes së mësipërme duke u përgjigjur me {labels_str}, dhe asgjë tjetër.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset include-sq
```

## Common-sense Reasoning

### Winogrande-sq

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2506.19468)
and is a translated and filtered version of the English [Winogrande
dataset](https://doi.org/10.1145/3474381).

The original full dataset consists of 47 / 1,210 samples for training and testing, and
we use 128 of the test samples for validation, resulting in a 47 / 128 / 1,085 split for
training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Nuk mund ta kontrolloja lagështinë siç kontrolloja shiun, sepse _ po vinte nga kudo. Çfarë i referohet boshllëku _?\nOpsione:\na. lagështia\nb. shiu",
  "label": "a"
}
```

```json
{
  "text": "Jessica mendoi se \"Sandstorm\" ishte kënga më e mirë e shkruar ndonjëherë, por Patricia e urrente atë. _ bleu një biletë për koncertin e xhazit. Çfarë i referohet boshllëku _?\nOpsione:\na. Jessica\nb. Patricia",
  "label": "b"
}
```

```json
{
  "text": "Termostati tregoi se ishte njëzet gradë më ftohtë poshtë se sa ishte lart, kështu që Byron qëndroi në _ sepse ai ishte ftohtë. Çfarë i referohet boshllëku _?\nOpsione:\na. poshtë\nb. lart",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Më poshtë janë pyetje me zgjedhje të shumëfishtë (me përgjigje).
  ```

- Base prompt template:

  ```text
  Pyetje: {text}
  Opsione:
  a. {option_a}
  b. {option_b}
  Përgjigje: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Pyetje: {text}
  Opsione:
  a. {option_a}
  b. {option_b}

  Përgjigjuni pyetjes së mësipërme duke u përgjigjur me 'a' ose 'b', dhe asgjë tjetër.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset winogrande-sq
```

## Summarisation

### LR-Sum-sq

This dataset was published in [this paper](https://aclanthology.org/2023.findings-acl.427/).
The source data is public domain newswire collected from Voice of America websites,
and the summaries are human-written.

The original dataset contains 18,312 / 2,289 / 2,289 samples for the training, validation,
and test splits, respectively. We use 1,024 / 256 / 2,048 samples for our training,
validation and test splits, respectively. The train and validation splits are subsets
of the original splits. For the test split, we use all available test samples and
supplement with additional samples from the training set to reach 2,048 samples in
total.

Here are a few examples from the training split:

```json
{
  "text": "Mbeten të forta marrëdhëniet ushtarake Izrael - SHBA\n\nNë fillim të korrikut, ushtria izraelite njoftoi se kishte testuar me sukses një sistem për rrëzimin e raketave që i kanosen Izraelit. I quajtur \"Kubeja e Hekurt\", ky sistem ka për qëllim të mbrojë vendbanimet izraelite nga raketat e lëshuara nga Libani dhe Gaza. Administrata amerikane i kërkon Kongresit një fond prej 205 milionë dollarësh për ta mbështetur këtë sistem, si shembull tjetër i bashkëpunimit ushtarak SHBA-Izrael. Ndihmëssekretari amerikan për çështjet politike-ushtarake, Endru Shapiro, deklaron: \"Sistemi Kubeja e Hekurt do të ndihmojë në neutralizimin e kërcënimeve të Hezbollahut dhe Hamasit.\" Shtetet e Bashkuara gjithashtu ofrojnë miliona dollarë për sisteme të tjera të mbrojtjes kundër raketave. Presidenti amerikan Barak Obama vlerëson: \"Vitin e kaluar, mbi 1.000 ushtarë amerikanë morën pjesë në stërvitjen më të madhe të këtij lloji.\" Këtë vit, Kongresi miratoi paketën më të madhe të mbështetjes për Izraelin deri tani, me 2.7 miliardë dollarë, ndërsa për vitin e ardhshëm janë kërkuar 3 miliardë. Administrata shpreson që kjo ndihmë të mbështesë bisedimet e paqes që ndërmjetësohen nga SHBA. Analistë nga Lindja e Mesme mendojnë se ndihma mund të rrisë shanset për paqe, por mbeten dyshime mbi qëndrimin e qeverisë dhe ushtrisë izraelite. Qeveria izraelite e sheh Iranin dhe programin e tij bërthamor si kërcënim kryesor. Së fundmi, Kombet e Bashkuara, SHBA dhe shtete të tjera vendosën sanksione të reja ndaj Iranit, që pretendon se programi i tij bërthamor ka qëllime paqësore. Martin Indik, drejtor në Institutin Brukings, thekson: \"Suksesi i procesit të paqes dobëson përpjekjet e Iranit në rajon.\" Presidenti Obama dhe kryeministri Netanjahu kanë patur mospërputhje, kryesisht sa i përket zgjerimit të vendbanimeve në Jeruzalem. Pavarësisht mosmarrëveshjeve, bisedimet e paqes SHBA-Izrael-Palestinë kanë vijuar. Marrëdhëniet u tendosën kur Izraeli shpalli zgjerimin e vendbanimeve hebraike në Jeruzalemin Lindor gjatë një vizite nga nënpresidenti amerikan Xho Bajden. Po ashtu, një ndërhyrje izraelite ndaj një flotilje ndihmash në Gaza u përfundua tragjikisht për disa aktivistë pro-palestinezë. Megjithatë, këto nuk e kanë ndalur dialogun. I dërguari i SHBA, Xhorxh Miçëll, ka ndërmjetësuar për rifillimin e bisedimeve të drejtpërdrejta. \"Kjo lidhje është forcuar së tepërmi në përpjekjet për siguri dhe bashkëpunim ushtarak,\" thekson Obama. Shapiro shton: \"Administrata mbetet e përkushtuar për financimin vjetor të një pakete 10-vjeçare me vlerë 30 miliardë dollarë për shitje armësh dhe siguri për Izraelin.\"",
  "target_text": "Ndërsa vëmendja është përqendruar në mosmarrëveshjet diplomatike ndërmjet Izraelit dhe SHBA, bashkëpunimi ushtarak mbetet i fortë."
}
```

```json
{
  "text": "Siri - \"Gjenerata e Humbur\" e fëmijëve të luftës\n\nDisa prej fëmijëve në Siri e kanë kaluar të gjithë jetën në kushte lufte. Madje, edhe ata që u larguan nga konflikti, po vuajnë pasojat. Ekspertët thonë se efektet e luftës së gjatë janë shkatërruese për zhvillimin fizik, mendor dhe arsimor për një gjeneratë të tërë sirianësh. Siç na njeh edhe materiali në vazhdim, fati i fëmijëve sirianë u diskutua nga ekspertët në një forum këtu në kryeqytetin amerikan, Uashington. Sipas përllogaritjeve rreth 2 milionë fëmijë sirianë nuk shkojnë në shkollë, dhe afro 1 milion e atyre që janë larguar nuk po marrin arsimin e duhur. Gjysma e fëmijëve që shkojnë në shkollë në Siri thonë se nuk ndihen të sigurtë. Grupet humanitare kanë intervistuar disa fëmijë. Amy Richmond është me organizatën \"Save the Children\". Amy Richmond ka punuar me fëmijët e shpërngulur sirianë në Jordani, Liban dhe Turqi, si dhe të zhvendosurit brenda Sirisë. Ekspertët thonë se stresi dhe trauma pengojnë zhvillimin fizik dhe mendor tek fëmijët. Fëmijët e traumatizuar rëndë nuk mund të mësojnë. Pothuajse çdo fëmijë në Siri është prekur nga lufta në një farë mënyre. Të mbijetuarit ka të ngjarë të kenë të paktën një të afërm të vrarë. Fëmijët në kampet e refugjatëve shpesh u nënshtrohen poshtërimeve dhe abuzimeve. Dr. Mohamed Khaled Hamza është me Shoqatën Mjekësore Siriano-Amerikane. Departamenti amerikan i Shtetit po punon me disa bashkësi lokale në verilindje të Sirisë për të ndihmuar në rehabilitimin e fëmijëve të moshës shkollore. Arti, aktrimi e vallëzimi i ndihmojnë fëmijët që të përballen me dhimbjen. Ata po ashtu duhet të mësojnë se si të bashkëveprojnë para mësimit. Catherine Bou-Maroun është me Departamentin e Shtetit. Ekspertët paralajmërojnë se lufta në Siri po krijon një \"gjeneratë të humbur\" që do të ketë pasoja shkatërrimtare për vendin, i cili do të ketë nevojë për rindërtim kur të përfundojë lufta.",
  "target_text": "Ekspertët thonë se lufta në Siri do të ketë efekte afatgjata fizike, mendore dhe arsimore tek fëmijët"
}
```

```json
{
  "text": "Kalendari Historik, 9 qershor - 2003-06-07\n\nMë 9 qershor të vitit 1877, shkrimtari amerikan Samuel Clemens tregoi sesi e kishte zgjedhur pseudonimin e tij letrar, Mark Twain. Në një letër drejtuar gazetës \"Daily Alta Californian\", autori shkruante: \"Fjalët 'Mark' dhe 'Twain' i referohen thellësisë së ujit, që matej me një sistem të caktuar në lumin Missisipi.\" Ky sistem përdorej nga ata që udhëtonin me varkë në lumin ku kishte turma peshqish dhe cektina të herëpashershme. Pseudonimi Mark Twain i përshtatej mjaft mirë Samuel Clemensit, i cili ka shkruar mjaft për jetën në brigjet e Missisipit. Më 9 qershor 1934, personazhi i kartonave Donald Duck u shfaq për herë të parë në ekran, në një rol të dorës së dytë në filmin vizatimor të Walt Disney-t \"The Wise Little Hen\". Walt Disney krijoi një rosak tepër gjaknxehtë në kontrast me personazhin e urtë dhe të qetë Micky Mouse, që ishte protagonisti kryesor i Walt Disney-t. Donald Duck nuk u bë asnjëherë aq popullor sa Micky Mouse, por ai është personazh në më tepër se 100 filma të shkurtër vizatimorë, që janë ndjekur në të gjithë botën. Krijuesi i këtyre personazheve, Walt Disney, fliste për to sikur ata të ishin miq të tij të ngushtë, apo kushërinj. Në një tjetër rast, zoti Disney i përshkroi marrëdhëniet midis Donaldit dhe anëtarëve të tjerë të familjes duke thënë: \"Donald Duck është e vetmja vezë e keqe e familjes.\" Walt Disney vdiq në vitin 1966 por personazhi i tij, Donald Duck, vazhdon të jetojë.",
  "target_text": "Në vitin 1934, shfaqet për herë të parë në ekran personazhi i filmave me kartona Donald Duck."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:

  ```text
  Më poshtë janë dokumente me përmbledhje të bashkëngjitura.
  ```

- Base prompt template:

  ```text
  Dokument: {text}
  Përmbledhje: {target_text}
  ```

- Instruction-tuned prompt template:

  ```text
  Dokument: {text}

  Shkruani një përmbledhje të dokumentit të mësipërm.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset lr-sum-sq
```
