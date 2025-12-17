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

## Summarisation

### LR-Sum-bs

This dataset was published in [this paper](https://aclanthology.org/2023.findings-acl.427/).
The source data is public domain newswire collected from Voice of America websites,
and the summaries are human-written.

The original dataset contains 5,784 / 722 / 723 samples for the training, validation, and
and test splits, respectively. We use 1,024 / 256 / 2,048 samples for our training,
validation and test splits, respectively. The train and validation splits are subsets
of the original splits. For the test split, we use all available test samples and
supplement with additional samples from the training set to reach 2,048 samples in
total.

Here are a few examples from the training split:

```json
{
    "text": "Komisija 9/11: američki dužnosnici nisu shvaćali razmjere opasnosti od al-Qaide (23/7/04) - 2004-07-23\n\nKomisija koja je istraživala terorističke napade na Sjedinjene Države 2001. godine ocjenjuje da američki dužnosnici nisu shvaćali razmjere opasnosti koju je predstavljala al-Qaidina mreža. Neovisni panel objavio je svoje zaključke na tiskovnoj konferenciji u Washingtonu. Iznoseći osnovne zaključke izvještaja, predsjedatelj komisije Thomas Kean rekao je da američka vlast nije bila dovoljno aktivna u borbi protiv opasnosti koju je predstavljala al-Qaida. Panel je ocijenio da je u svim dijelovima vlasti bilo propusta glede “razumijevanja, određivanja politike, osposobljenosti i rukovođenja”. Vojska je – kako se navodi – ponudila tek ograničene opcije u vezi s napadima na al-Qaidu, a djelovanje obavještajnih službi bilo je otežano krutim budžetom i birokratskim suparništvom. U izvještaju se navodi da nitko ne može znati jesu li postojale neke mjere koje su mogle onemogućiti napade, ali se dodaje da planove al-Qaide nije ni omelo, niti odgodilo ništa što su poduzele vlade predsjednika Clintona i Busha. Komisija je pozvala na formiranje saveznog centra za kontra-terorizam, na čijem bi čelu bio direktor ministarskog ranga s obavezom da nadzire rad svih američkih obavještajnih službi. Predsjedatelj komisije Thomas Kean je ocijenio da su Sjedinjene Države i dalje sučeljene sa – kako je rekao – “jednim od najvećih sigurnosnih izazova u našoj povijesti.” Obrazlažući potrebu stvaranja ministarskog položaja za obavještajni rad, član komisije Lee Hamilton rekao je da su informacije i odgovornost sada razvučeni po brojnim obavještajnim službama. On se također založio za davanje više ovlasti kongresnim tijelima za nadzor obavještajnih službi. Samo nekoliko sati nakon što je komisija objavila svoje nalaze, predsjednik Bush je rekao da je suglasan sa zaključkom da su teroristi 2001. iskoristili duboke institucionalne propuste u obrani zemlje: “Preporuke komisije podudarne su sa strategijom koju moja administracija slijedi u nadilaženju propusta i u borbi do pobjede nad terorizmom.” Predsjednik Bush se još nije službeno obvezao na provedbu bilo koje od komisijinih preporuka, ali je rekao da će one biti pažljivo razmotrene. U izvješću komisije navodi se da nisu pronađeni nikakvi dokazi da je bivši irački predsjednik Saddam Hussein ikada “operativno surađivao” s al-Qaidom. Obavještajni podaci ukazuju na “prijateljske kontakte” Iraka s al-Qaidom prije 11.rujna 2001.godine, ali komisija nije pronašla nikakve dokaze da su Bagdad i al-Qaida surađivali u planiranju i izvršenju napada na Sjedinjene Države. Što se Irana tiče, komisija nije pronašla dokaze da je Teheran bio upoznat s napadima na New York i Washington. No, kako se dodaje, to pitanje treba dalje istraživati. Također se navodi da su iranske vlasti omogućile al-Qaidinim članovima da putuju preko Irana bez da im se u pasoše ubilježi kad su ušli i izašli iz te zemlje.",
    "target_text": "Izvješće komisije navodi se da nisu pronađeni nikakvi dokazi da je Saddam Hussein ikada “operativno surađivao” s al-Qaidom"
}
```

```json
{
    "text": "Vlada prihvaća odluku Žalbenog vijeća Haškog suda u predmetu Bobetko (29/11/02) - 2002-11-29\n\nVijest da je Žalbeno vijeće Haškog suda odbilo oba podneska hrvatske Vlade u vezi s optužniom protiv generala Janka Bobetka u hrvatskoj pravnoj i političkoj javnosti – ako je suditi po prvim reakcijama – nikoga nije posebno iznenadila. Odvjetnik Goran Mikuličić, pravni savjetnik hrvatske Vlade u odnosima s Haškim sudom, ovako je prokomentirao vijest o odbijanju hrvatskih podnesaka u Haagu. “Naša Vlada prihvaća odluku. Vlada ne polemizira s odlukom i ne komentira odluku jer to je odluka nadležnog suda s kojim nema dalje nikakve pravne rasprave”. Mikuličić je objasnio koji su daljnji koraci Vlade nakon ovakvih vijesti iz Haaga. “Daljnji postupak Vlade će biti objaveštavanje tajništva Haškog tribunala o nalazu liječničkih ekperata koje je angažirao Županijski sud u Zagrebu. Oni su utvrdili da general Bobetko nije sposoban aktivno sudjelovati u postupku, zbog svog lošeg zdravstvenog stanja, i Vlada će posegnuti za odredbama pravila 59., i izvjestiti tajništvo o nemogućnosti udovoljenja zahtjevu zbog objektivnih okolnosti. Osim pravne donosimo i političke reakcije na odluku Žalbenog vijeća u slučaju Bobetko. SDP-ovac Mato Arlović, koji je i predsjednik saborskog Odbora za ustav i poslovnik, kaže da je vladajuća koalicija bila spremna i na povoljnu i na nepovoljnu odluku Žalbenog vijeća Haškog suda. “U tom poledu mislim da je najveća vrijetnost da je haški sud, raspravljajući o prigovorima Republike Hrvatske priznao Hrvatskoj da se može koristiti pravom koje ovi dokumenti daju i da raspravljajući o našim navodima i našim argumentima donio odluku. Drugo je pitanje što mi nismo imali dostatne dokaze da svoja stajališta i potvrdimo i da ih Haški sud prihvati.” Iako je Vlada za pravne korake koje je poduzela oko optužnice protiv generala Bobetka imala potporu ne samo stranaka vladajuće koalicije nego i opozicije, oporbene stranke danas izražavaju negodovanje zbog načina na koji je Vlada branila interese haških optuženika, svojih državljana. Predsjednik Hrvatskog bloka, Ivić Pašalić, smatra da je Račanova Vlada od samog početka svog mandata povela pogrešnu politiku prema Haškom sudu. Problem, po njemu, potječe od saborske deklaracije koju je vladajuća koalicija izglasala još u svibnju 2000., a u kojoj je priznala nadležnost haškog suda za akcije “Bljesak” i “Oluja”. “Prema tome riječ je o promašenoj strategiji sadašnje Vlade koja je jednostavno kulminirala dolaskom nekoliko optužnica u kojima se Vlada ponašala različito. U slučaju generala Gotovine nije napravila ništa nego je dala žalbu Carli del Ponte koja ju je ekspresno vratila natrag, a u slučaju optužnice protiv generala Bobetka, pritisnuta reakcijama u parlamentu i javnosti pokušali su nešto napraviti, ali očito pravno i politički loše”. Pašalić, međutim, ne spominje ustavni zakon o suradnji s Haškim sudom koji obvezuje hrvatske vlasti na suradnju sa sudom, a kojeg je 1996. donijela Hrvatska demokratska zajednica, stranka kojoj je i sam tada pripadao.",
    "target_text": "Odbijanje hrvatskih podnesaka nikoga nije posebno iznenadilo u pravnim i političkim krugovima"
}
```

```json
{
    "text": "Lječnici udvostručavaju napore na promoviranju vakcinacije kao najbolje zaštite protiv H1N1\n\nZemlje zapadne hemisfere su odpočeledistribuirati H1N1 vakcine u okviru obimnog programa imunizacije protiv virusnepandemije svinjske gripe. Roditelji i neki profesionalci su zabrinuti okosigirnosti vakcine, dok neki doktori dovode u sumnju sposobnosti bolnica da senose sa težim slučajevima. Veliki broj ljudi u Sjedinjenim Državamadolazi u klinike za vakcinaciju. Michelle Lowrey ima troje djece itrudna je sa četvrtim: \"Ja imam sve razloge da budemovdje.\" Trudne žene su izložene većem rizikukomplikacija ukoliko se zaraze virusom H1N1. I do sada je najmanje 86 američkedjece umrlo od novog virusa. Katherine Blake brine za svog sina: \"On je u visoko rizičnoj grupi.Kao dijete je imao otvorenu operaciju srca, i jako me je strah da se nezarazi.\" Američki centar za kontrolu bolestije izvjestio da se novi virus prehlade raširio kroz veći dio zemlje. I poredtoga, neki Amerikanci kažu da neće primiti vakcinu. Mi živimo u Sacramentu. Ima nekihslučajeva svinjske gripe, ali ne mnogo, tako da nas to, zaista, nije pogodilo,kaže jedan čovjek na ulici Washingtona. Neke brine koliko je vakcinasigurna, jer je tako brzo proizvedena, i zato što sadrži konzervanse za kojeneki roditelji tvrde da mogu uzrokovati autizam. Dr. Anne Schuchat iz AmeričkogCentra za kontrolu bolesti kaže da je vakcina sigurna i može se dobiti i bezkonzervansa: \"Mi nismo zanemarili sigurnostu proizvodnji ovih vakcina, ili testiranju i nadgledanju ovih vakcina. I veomaje važno da se ovaj proces obavi pažljivo i sigurno.\" Zdravstveni zvaničnici i lječniciudvostručavaju napore na promoviranju vakcinacije kao najbolje zaštite protivH1N1 virusa. Dr. Peter Holbrooke iz Medicinskogcentra za zaštitu djece u Washingtonu kaže da ljudi griješe kada misle da jeova groznica slična običnoj prehladi: \"Veoma je važno da se dobro razmislio vakcini i bolesti koju ona spriječava. To nije blaga, nego značajnabolest.\" Dr. Holbrooke kaže da čak i umjerenislučajevi izazivaju ozbiljnu bolest i teži slučajevi mogu ubrzano pogoršatistanje. Doktora Arthura Kellermanna saMedicinskog fakulteta Emory brine gdje smjestiti pacijente koji trebajuintenzivnu njegu: \"Mi trebamo pripremiti našekapacitete za intenzivnu njegu i naš zdravstveni sistem za mogućnost donošenjateških odluka - ko može dobiti intenzivnu njegu, a ko ne može.\" Ukoliko H1N1 se virus nastavirazvijati onim tempom kakvim je krenuo nakon što se pojavio u martu, bolestdostiže vrhunac i počinje da opada za otprilike sedam sedmica. Ako je to tako,moglo bi biti da je ona već na vrhuncu u Sjedinjenim Državama, smatra dr.Holbrooke: \"Ali treba shvatiti da veomalako može usljediti drugi val tokom zime.\" Svi se specijalisti slažu u tome daje izbijanje nove groznice nepredvidljivo. I nema dovoljno vakcine H1N1, čak niu Sjedinjenim Državama. Što se tiče zemalja u razvoju, izSvjetske zdravstvene organizacije kažu da bi za njih medjunarodne donacijevakcine trebale početi stizati za nekoliko sedmica.",
    "target_text": "Zemlje zapadne hemisfere su odpočele distribuirati H1N1 vakcine u okviru obimnog programa imunizacije protiv virusne pandemije svinjske"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:

  ```text
  Slijede dokumenti s priloženim sažecima.
  ```

- Base prompt template:

  ```text
  Dokument: {text}
  Sažetak: {target_text}
  ```

- Instruction-tuned prompt template:

  ```text
  Dokument: {text}

  Napišite sažetak gornjeg dokumenta.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset lr-sum-bs
```
