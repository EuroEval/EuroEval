# 🇧🇦 Bosnian

This is an overview of all the datasets used in the Bosnian part of EuroEval. The
datasets are grouped by their task – see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### MMS-bs

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2306.07902).
The corpus consists of 79 manually selected datasets from over 350 datasets reported in
the scientific literature based on strict quality criteria.

The original dataset contains a single split with 36,183 Bosnian samples. We use 1,024 /
256 / 2,048 samples for our training, validation, and test splits, respectively. We have
employed stratified sampling based on the label column from the original dataset to
ensure balanced splits.

Here are a few examples from the training split:

```json
{
  "text": "Jaoo kako cjadko, izasla si s momkom  ju ar filing loved, o maj gash! Awwww. POV RA CA CU",
  "label": "negative"
}
```

```json
{
  "text": "@aneldzoko sta se to desava u Neumu?",
  "label": "neutral"
}
```

```json
{
  "text": "Zasto se inspirator zove inspirator kad se s njim usisava?",
  "label": "neutral"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Slijede dokumenti i njihova osjetila, koja mogu biti pozitivno, neutralno ili negativno.
  ```

- Base prompt template:

  ```text
  Dokument: {text}
  Osjetilo: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Dokument: {text}

  Klasificirajte osjećaj u dokumentu. Odgovorite samo s pozitivno, neutralno, ili negativno, i ništa drugo.
  ```

- Label mapping:
  - `positive` ➡️ `pozitivno`
  - `neutral` ➡️ `neutralno`
  - `negative` ➡️ `negativno`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset mms-bs
```

## Named Entity Recognition

### WikiANN-bs

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
  "tokens": [
    "Čehoslovačka",
    ",",
    "Francuska",
    ",",
    "Mađarska",
    ",",
    "Meksiko",
    ",",
    "Švicarska",
    ",",
    "Urugvaj"
  ],
  "labels": [
    "B-LOC",
    "O",
    "B-LOC",
    "O",
    "B-LOC",
    "O",
    "B-LOC",
    "O",
    "B-LOC",
    "O",
    "B-LOC"
  ]
}
```

```json
{
  "tokens": [
    "godine",
    ",",
    "naselje",
    "je",
    "ukinuto",
    "i",
    "pripojeno",
    "naselju",
    "Bribir",
    "."
  ],
  "labels": ["O", "O", "O", "O", "O", "O", "O", "O", "B-LOC", "O"]
}
```

```json
{
  "tokens": ["Administrativno", "središte", "oblasti", "je", "Tjumenj", "."],
  "labels": ["O", "O", "O", "O", "B-LOC", "O"]
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Slijede rečenice i JSON riječnici s imenovanim entitetima koji se pojavljuju u rečenicama.
  ```

- Base prompt template:

  ```text
  Rečenica: {text}
  Imenovani entiteti: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Rečenica: {text}

  Identificirajte imenovane entitete u rečenici. Prikažite ih kao JSON riječnik s ključevima 'osoba', 'mjesto', 'organizacija' i 'razno'. Vrijednosti trebaju biti popisi imenovanih entiteta navedenog tipa, točno kako se pojavljuju u rečenici.
  ```

- Label mapping:
  - `B-PER` ➡️ `osoba`
  - `I-PER` ➡️ `osoba`
  - `B-LOC` ➡️ `mjesto`
  - `I-LOC` ➡️ `mjesto`
  - `B-ORG` ➡️ `organizacija`
  - `I-ORG` ➡️ `organizacija`
  - `B-MISC` ➡️ `razno`
  - `I-MISC` ➡️ `razno`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset wikiann-bs
```

## Reading Comprehension

### MultiWikiQA-bs

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
  "context": "NGC 3803 (također poznat kao PGC 36204) je eliptična galaksija koja je udaljena oko 164 miliona sg od Zemlje i nalazi se u sazviježđu Lav. Najveći prečnik je 0,40 (19 hiljada sg) a najmanji 0,4 uglovnih minuta (19 hiljada sg). Prvo otkriće je napravio R. J. Mitchell 27. marta 1856. godine.\n\nNajbliži NGC/IC objekti \nSljedeći spisak sadrži deset najbližih NGC/IC objekata.\n\nTakođer pogledajte \n Novi opći katalog\n Spisak NGC objekata\n Spisak galaksija\n\nBilješke \n  Prividna magnituda od 15,5 – Apsolutna magnituda: M = m - 5 ((log10 DL) - 1), gdje je m=15,5 i DL=50,4 * 106.\n  0,40 uglovnih minuta – S = A * D * 0,000291 * P, gdje je A=0,40, D=50,4 i P = 3,2616.\n  Bazirano na euklidsku udaljenost.\n\nReference\n\nLiteratura\n\nVanjski linkovi\n\nNGC 3803 \n\n  NGC 3803 na Aladin pregledaču\n\nNGC katalog \n  Interaktivni NGC Online Katalog\n  Astronomska baza podataka SIMBAD\n  NGC katalog na Messier45.com \n  NGC/IC projekt\n  NGC2000 na NASA sajtu\n  NGC na The Night Sky Atlas sajtu\n\nEliptične galaksije\nLav (sazviježđe)\nNGC objekti\nPGC objekti",
  "question": "Koliki je najmanji kutni promjer NGC 3803 izražen u kutnim minutama?",
  "answers": {
    "answer_start": [158],
    "text": ["0,4"]
  }
}
```

```json
{
    "context": "Po popisu stanovništva, domaćinstava i stanova 2011. u  Srbiji, koji je proveden od 1. do 15. oktobra 2011, u općini Crna Trava živjelo je ukupno 1663 stanovnika, što predstavlja 0,02% od ukupnog broja stanovnika Srbije, odnosno 0,77% od od ukupnog broja stanovnika Jablaničkog okruga.  Popis stanovništva provoden je na temelju Zakona o popisu stanovništva, domaćinstava i stanova u 2011. Godini ("Službeni glasnik RS", br. 104/09 i 24/11).\n\nRezultati popisa\n\nNacionalna pripadnost\n\nMaternji jezik\n\nVjeroispovijest\n\nStarosna piramida \nOd ukupnog broja stanovnika u općini Crna Trava bilo je 838 (50,39%) muškaraca i 825 (49,61%) žena, što predstavlja omjer muškaraca i žena 1.016:1000. Prosječna starost stanovništva bila je 53,7 godina, muškaraca 51,4 godina, a žena 56,1 godina. Udio osoba starijih od 18 godina je 91,5% (1.521), kod muškaraca 92,0% (771), a kod žena 90,9% (750).\n\nTakođer pogledajte\n\nNapomene\n\nReference\n\nVanjski linkovi \n Republički zavod za statistiku Srbije \n\nCrna Trava\nCrna Trava",
    "question": "Koliko godina u prosjeku imaju stanovnici općine Crna Trava?",
    "answers": {
        "answer_start": [726],
        "text": ["53,7 godina"]
    }
}
```

```json
{
  "context": "IC 910 (također poznat kao IRAS 13387+2331, MCG 4-32-25 i PGC 48424) je spiralna galaksija koja je udaljena oko 374 miliona sg od Zemlje i nalazi se u sazviježđu Volar. Najveći prečnik je 0,50 (54 hiljade sg) a najmanji 0,4 uglovnih minuta (44 hiljade sg). Prvo otkriće je napravio Stephane Javelle 16. juna 1892. godine.\n\nNajbliži NGC/IC objekti \nSljedeći spisak sadrži deset najbližih NGC/IC objekata.\n\nTakođer pogledajte \n Novi opći katalog\n Spisak IC objekata\n Spisak galaksija\n\nBilješke \n  Prividna magnituda od 14,4 – Apsolutna magnituda: M = m - 5 ((log10 DL) - 1), gdje je m=14,4 i DL=114,6 * 106.\n  0,50 uglovnih minuta – S = A * D * 0,000291 * P, gdje je A=0,50, D=114,6 i P = 3,2616.\n  Bazirano na euklidsku udaljenost.\n\nReference\n\nLiteratura\n\nVanjski linkovi\n\nIC 910 \n\n  IC 910 na Aladin pregledaču\n\nIC katalog \n  Interaktivni NGC Online Katalog\n  Astronomska baza podataka SIMBAD\n  IC katalog na Messier45.com \n  NGC/IC projekt\n  NGC2000 na NASA sajtu\n  IC na The Night Sky Atlas sajtu\n\nIC objekti\nIRAS objekti\nMCG objekti\nPGC objekti\nSpiralne galaksije\nVolar (sazviježđe)",
  "question": "Kolika je distanca između Zemlje i galaksije IC 910?",
  "answers": {
    "answer_start": [108],
    "text": ["oko 374 miliona sg"]
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

```text
Slijede tekstovi s pitanjima i odgovorima.
```

- Base prompt template:

```text
Tekst: {text}
Pitanje: {question}
Odgovor s najviše 3 riječi:
```

- Instruction-tuned prompt template:

```text
Tekst: {text}

Odgovorite na sljedeće pitanje o gornjem tekstu s najviše 3 riječi.

Pitanje: {question}
```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-bs
```

## Summarisation

### LR-Sum-bs

This dataset was published in
[this paper](https://aclanthology.org/2023.findings-acl.427/). The source data is public
domain newswire collected from Voice of America websites, and the summaries are
human-written.

The original dataset contains 5,784 / 722 / 723 samples for the training, validation,
and test splits, respectively. We use 1,024 / 256 / 2,048 samples for our training,
validation and test splits, respectively. The train and validation splits are subsets of
the original splits. For the test split, we use all available test samples and
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

## Instruction-following

### MultiIFEval-bs

MultiIFEval-bs is part of the MultiIFEval benchmark spanning 305 languages. It is
generated by translating and localising the English IFEval dataset using a structured
LLM generation pipeline. For each target language, a randomly selected Wikipedia article
in that language provides contextual grounding to reduce hallucination and improve
cultural localisation. The pipeline preserves instruction_id_list values for
traceability to the original English samples, and retains kwargs keys with values
localised where appropriate, enabling programmatic constraint verification. The dataset
was published [here](https://huggingface.co/datasets/EuroEval/multi-ifeval-bs).

This dataset is part of the MultiIFEval benchmark introduced in
[this draft paper](https://raw.githubusercontent.com/alexandrainst/multi_ifeval/refs/heads/feat/add-paper/paper/acl_latex.tex).

We use the dataset as the test split, and do not include other splits, as we only
evaluate models zero-shot and the size is too small to warrant a validation set.

Here are a few examples from the test split:

```json
{
  "text": "Napišite sažetak Wikipedia stranice \"https://bs.wikipedia.org/wiki/Bosanski_jezik\" sa najmanje 200 riječi. Nemojte koristiti nijednu zarez i istaknite najmanje 3 sekcije koje imaju naslove u Markdown formatu, na primjer *istaknuta sekcija Dio 1*, *istaknuta sekcija Dio 2*, *istaknuta sekcija Dio 3*.",
  "target_text": {
    "instruction_id_list": [
      "punctuation:no_comma",
      "detectable_format:number_highlighted_sections",
      "length_constraints:number_words"
    ],
    "kwargs": [
      {},
      { "num_highlights": 3 },
      { "num_words": 200, "relation": "at least" }
    ]
  }
}
```

```json
{
  "text": "Planiram putovanje u Bosnu i Hercegovinu i želim da mi napišeš plan putovanja u Shakespeareovom stilu. Nije ti dozvoljeno koristiti zareze u svom odgovoru.",
  "target_text": {
    "instruction_id_list": ["punctuation:no_comma"],
    "kwargs": [{}]
  }
}
```

```json
{
  "text": "Napravite CV za svježe diplomiranog studenta koji se prijavljuje za svoj prvi posao. Obavezno uključite najmanje 12 mjesta držača u uglatim zagradama, kao što su [Ime] ili [Adresa].",
  "target_text": {
    "instruction_id_list": ["detectable_content:number_placeholders"],
    "kwargs": [{ "num_placeholders": 12 }]
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

You can evaluate a model on this dataset as follows:

```bash
euroeval --model <model-id> --dataset multi-ifeval-bs
```

## Hallucination Detection

### RAGTruth-bs

This dataset is a Bosnian translation of the
[RAGTruth](https://aclanthology.org/2024.acl-long.585/) hallucination benchmark, which
contains retrieval-augmented generation (RAG) prompts together with model-generated
answers annotated for hallucinations. Rather than evaluating the correctness of the
generated answer, this task evaluates the degree to which the model hallucinates, i.e.,
generates tokens that are not grounded in the provided context.

The hallucination detection is performed using the
[LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect) library, which uses a
[transformer-based classifier](https://arxiv.org/abs/2605.02504) to predict
hallucination at the token level. The metric reported is the hallucination rate,
computed as the ratio of hallucinated tokens to total tokens in the generated answers.

Here are a few examples from the test split:

```json
{
  "prompt": "Kratko odgovorite na sljedeće pitanje:\nšta uzrokuje da lubenice eksplodiraju\nImajte na umu da vaš odgovor treba biti strogo zasnovan na sljedeća tri odlomka:\nodlomak 1: Stavite gomilu gumica oko sredine lubenice. Budite oprezni jer će sav taj pritisak sa spolja početi da eksplodira lubenicu. Gumice su odličan primjer potencijalne energije. Potencijalna energija je energija koja je pohranjena, koja se ne koristi, samo čeka da bude oslobođena.\n\nodlomak 2: Loša je ideja pokušati ponovo stvoriti eksplodirajuću lubenicu. Moguće je ponovo stvoriti eksploziju hemijskim postupkom, ali to uključuje nestabilne gasove i tečnosti. Pitajte svog nastavnika hemije ili nauke - možda znaju kako to učiniti i biti voljni da ponovo stvore eksplodirajuću lubenicu za vašu klasu!\n\nodlomak 3: Gas će se truditi da pobegne iz lubenice, ali kako se polako povećava zbog truljenja u lubenici, pritisak će nastaviti da raste. Kada koža lubenice više ne bude dovoljno jaka da zadrži gas unutra, eksplodiraće, često prskajući sve obližnje površine sa trulom lubenicom.\n\nU slučaju da odlomci ne sadrže potrebne informacije za odgovor na pitanje, molimo odgovorite sa: \"Nemoguće odgovoriti na osnovu datih odlomaka.\"\noutput:",
  "answer": "Na osnovu datih odlomaka, glavni uzrok eksplodiranja lubenica je nakupljanje gasa unutar voća zbog truljenja ili fermentacije. Kako se gas akumulira, pritisak unutar lubenice raste, na kraju postajući dovoljno jak da pukne kožu i izazove eksploziju. Ovaj proces je sličan konceptu potencijalne energije, gdje se energija skladišti i čeka da bude oslobođena.",
  "labels": []
}
```

```json
{
  "prompt": "Uputstvo:\nNapišite objektivan pregled o sljedećem lokalnom poslu temeljen isključivo na dostavljenim strukturiranim podacima u JSON formatu. Trebali biste uključiti detalje i obuhvatiti informacije navedene u recenzijama kupaca. Pregled bi trebao biti dug 100 - 200 riječi. Ne izmišljajte informacije. Strukturirani podaci:\n{'ime': 'El Pastorcito', 'adresa': '1915 De La Vina St', 'grad': 'Santa Barbara', 'država': 'CA', 'kategorije': 'Meksička, Restorani', 'radno vrijeme': {'Četvrtak': '17:0-22:0', 'Petak': '17:0-22:0', 'Subota': '17:0-22:0'}, 'atributi': {'PoslovnoParkiranje': None, 'RestoraniRezervacije': None, 'VanjskoSjedenje': None, 'WiFi': None, 'RestoraniPonijeti': None, 'RestoraniDobroZaGrupe': None, 'Muzika': None, 'Ambijent': None}, 'zvijezde_poslovanja': 5.0, 'informacije_o_recenzijama': [{'zvijezde_recenzije': 5.0, 'datum_recenzije': '2021-11-28 04:51:03', 'tekst_recenzije': \"Posjetili smo ovo mjesto 2 noći zaredom i da sam bio u gradu još jednu noć, ponovo bih to učinio. Mulitas de pastor su bile odlične, imali smo i mulitas de pollo, tacos de birria i pastor. Horchata pića. Velika. Vidimo se sljedeći put.\"}, {'zvijezde_recenzije': 5.0, 'datum_recenzije': '2021-06-30 03:39:00', 'tekst_recenzije': \"Da! Tako dobro. Prošli smo pored El Pastorcito Mobile 3 noći zaredom na putu do našeg hotela i svake noći smo vidjeli gomilu ljudi kod ovog taco kamiona. (East Beach na S Milpas St) Imaju postavljene stolove i neka svjetla. To je zabavna scena. Odlučili smo to provjeriti. Tako smo srećni što smo to učinili. Idite po al Pastor tacose i burritose i sve to. \\nImali smo i birria tacose - koji su bili u redu, ali ništa posebno. Ali el pastor... stvarno dobro! Iscijedite malo limete, malo salsa verde. Igra POČINJE! \\nPonovo bih se vratio opet i opet. U stvari, jesmo - ponovo smo stali u red i uzeli još jedan taco i burrito iste noći jer je bilo tako dobro.\\nIDITE!\"}, {'zvijezde_recenzije': 5.0, 'datum_recenzije': '2021-06-18 02:32:15', 'tekst_recenzije': 'Prvi put ih isprobavam. Naručio sam tripa tacose i asada mulitu s horchatom. Sve je bilo ukusno. Definitivno ću se vratiti da probam ostatak meniju'}]}\nPregled:",
  "answer": "Naravno! Evo objektivnog pregleda El Pastorcito na osnovu dostavljenih strukturiranih podataka:\n\nEl Pastorcito je meksički restoran smješten u Santa Barbari, CA, sa prosječnom ocjenom od 5 zvjezdica <HAL>na osnovu tri recenzije</HAL>. Restoran nudi raznovrsna jela kao što su mulitas de pastor, mulitas de pollo, tacos de birria i pastor, kao i horchata napitke. Kupci su pohvalili restoran zbog njegove ukusne hrane, pri čemu je jedan recenzent naveo da su mulitas de pastor bile \"savršene\", a drugi preporučio el pastor tacose i burritos. Restoran ima <HAL>opuštenu</HAL> atmosferu i <HAL>ne nudi rezervacije</HAL> <HAL>ili sjedenje na otvorenom</HAL>, ali <HAL>ima WiFi</HAL> i <HAL>prima narudžbe za ponijeti</HAL>. Prema recenzijama, restoran je popularan među lokalnim stanovništvom i posjetiteljima, te je odlično mjesto za grupe. Sve u svemu, El Pastorcito je visoko ocijenjen meksički restoran u Santa Barbari koji poslužuje autentična i ukusna jela.",
  "labels": [
    {
      "start": 204,
      "end": 227,
      "label": "Evident Baseless Info"
    },
    {
      "start": 559,
      "end": 567,
      "label": "Evident Baseless Info"
    },
    {
      "start": 591,
      "end": 610,
      "label": "Evident Baseless Info"
    },
    {
      "start": 622,
      "end": 647,
      "label": "Evident Baseless Info"
    },
    {
      "start": 664,
      "end": 672,
      "label": "Evident Baseless Info"
    },
    {
      "start": 686,
      "end": 712,
      "label": "Subtle Baseless Info"
    }
  ]
}
```

```json
{
  "prompt": "Sažmite sljedeće vijesti u 90 riječi:\nTeksasni predstavnik vjeruje da bi migranti trebali platiti 2.000 dolara za ulazak u zemlju\nPrvi mandat Teksasnog predstavnika Eddiesa Moralesa iznio je zanimljiv prijedlog za administraciju u vezi s migrantima koji prelaze južnu granicu. Njegova izborna jedinica obuhvata dio El Pasa, Del Rija i Eagle Passa, koji su bili neka od najviših mjesta prelaska za migrante.\nNa Fox 4, predstavnik Morales razgovarao je sa Stevenom Dialom o migrantskoj krizi. Demokratski predstavnik rekao je da predsjednik Joe Biden ne radi dovoljno da podrži države na granici dok migranti nastavljaju prelaziti granicu.\nRanije ove godine, predsjednik Biden je donio nekoliko zakona o imigraciji kako bi bolje upravljao migrantima koji prelaze granicu. No, ove godine, brojevi su gotovo jednaki rekordnom broju prošle godine, prema Carinskoj i graničnoj zaštiti.\nKonačan broj za ovu fiskalnu godinu nije objavljen, ali prema Graničnoj zaštiti, susreti s migrantima bili su oko 100.000 ispod rekordnog broja prošle godine od 2.378.944 migranta.\nU intervjuu, predstavnik Morales došao je na ideju da migranti trebaju platiti SAD-u 2.000 dolara za ulazak u zemlju.\nPrema Yahoo News, predstavnik Morales je rekao:\n\"Možemo imati ove sporazume o radnoj snazi između guvernera u Meksiku i guvernera Teksasa. Trebamo 100 radnika, recimo u Uvaldeu za berbu usjeva, mogu napraviti zajednički sporazum o suradnji gdje ih šalju ovamo na određeno vrijeme i ponovo plaćaju, zapošljeni su. Poslodavci su također zbrinuti kako ne bi došli u bilo kakvo kriminalno ponašanje i zapošljavali ovakve migrante. I to je situacija u kojoj svi pobjeđuju.\"\n\"Na primjer, imali smo 1.600 prelazaka dnevno koji su se događali samo u sektoru Eagle Pass/Del Rio. Ako bismo naplatili 2.000 dolara za tu početnu naknadu za obradu i za njih da dobiju neglasovnu ID karticu s RFID čipom, koja bi im također služila kao kreditna kartica, mogli bismo generirati 3,2 miliona dolara dnevno i više od 1,2 milijarde dolara godišnje samo u sektoru Eagle Pass/Del Rio.\"\nMožete pogledati intervju s predstavnikom Moralesom u nastavku:\nMislite li da je naplaćivanje migrantima 2.000 dolara situacija u kojoj svi pobjeđuju za zemlju?\noutput:",
  "answer": "Teksas predstavnik Eddie Morales predložio je naplaćivanje migrantima 2.000 dolara za ulazak u zemlju, navodeći potencijalni prihod od 1,2 milijarde dolara godišnje. Vjeruje da bi to moglo riješiti problem migranata koji prelaze južnu granicu stvaranjem sporazuma o radnoj snazi između guvernera u Meksiku i Teksasu. Međutim, <HAL>kritičari tvrde da ovaj plan možda neće riješiti osnovne uzroke migracije i može dovesti do eksploatacije radnika migranata.</HAL>",
  "labels": [
    {
      "start": 331,
      "end": 455,
      "label": "Evident Baseless Info"
    }
  ]
}
```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ragtruth-bs
```
