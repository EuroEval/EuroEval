# 🇳🇴 Norwegian

This is an overview of all the datasets used in the Norwegian part of ScandEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.


## Sentiment Classification

### NoReC

This dataset was published in [this paper](https://aclanthology.org/L18-1661/) and is
based on reviews from three different media organisations: Schibsted Media Group, Aller
Media and NRK.

The original full dataset consists of 680,792 / 101,106 / 101,594 samples for training,
validation and test, respectively. We use a split of 1,024 / 256 / 2,048 samples for
training, validation and test, respectively. All the new splits are subsets of the
original splits.

Here are a few examples from the training split:

```json
{
  "text": "Den som ikke blir rystende berørt av « De utvalgte » , må være forherdet til det immune .",
  "label": "positive"
}
```
```json
{
  "text": "Under er noen av funksjonene som er dels unike for LG G3 :",
  "label": "neutral"
}
```
```json
{
  "text": "Tilsvarende får vi også lavere score i 3DMark enn hva tilfellet er for f.eks . Xperia Z2 og Galaxy S5 .",
  "label": "negative"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Følgende er anmeldelser og deres sentiment, som kan være 'positiv', 'nøytral' eller 'negativ'.
  ```
- Base prompt template:
  ```
  Anmeldelse: {text}
  Sentiment: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Anmeldelse: {text}

  Klassifiser sentimentet i anmeldelsen. Svar med 'positiv', 'nøytral' eller 'negativ'.
  ```
- Label mapping:
    - `positive` ➡️ `positiv`
    - `neutral` ➡️ `nøytral`
    - `negative` ➡️ `negativ`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset norec
```


## Named Entity Recognition

### NorNE-nb

This dataset was published in [this paper](https://aclanthology.org/2020.lrec-1.559/)
and is a manually NER annotated version of the [Bokmål Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Norwegian-Bokmaal). The NER labels
almost follow the CoNLL-2003 standard, but with some additional labels.

The original full dataset consists of 15,696 / 2,410 / 1,939 samples for training,
validation and test, respectively. We use a split of 1,024 / 256 / 2,048 samples for
training, validation and test, respectively. The splits we use are new, so there might
be some samples from the training split in the validation or test splits.

We have mapped the labels into the CoNLL-2003 standard as follows:

- `LOC` ➡️ `LOC`
- `PER` ➡️ `PER`
- `ORG` ➡️ `ORG`
- `MISC` ➡️ `MISC`
- `GPE_LOC` ➡️ `LOC`
- `GPE_ORG` ➡️ `ORG`
- `PROD` ➡️ `MISC`
- `DRV` ➡️ `MISC`
- `EVT` ➡️ `MISC`

Here are a few examples from the training split:

```json
{
  "tokens": array(['Det', 'fremkommer', 'av', 'årsmeldingene', 'fra', 'Bergen', 'helseråd', 'i', 'årene', '1952', '-', '66', '.'], dtype=object),
  "labels": array(['O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
```
```json
{
  "tokens": array(['Viktig', 'var', 'det', 'også', 'at', 'Kina', 'allerede', 'var', 'blitt', 'så', 'avhengig', 'av', 'det', 'amerikanske', 'markedet', 'og', 'av', 'dollaren', ',', 'at', 'en', 'nedgang', 'i', 'USA', 'også', 'ville', 'ramme', 'Kina', 'hardt', '.'], dtype=object),
  "labels": array(['O', 'O', 'O', 'O', 'O', 'B-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'O', 'O', 'B-ORG', 'O', 'O'], dtype=object)
}
```
```json
{
  'tokens': array(['Han', 'tok', 'fram', 'pistolen', 'og', 'dro', 'tilbake', 'til', 'Skaregata', '2', '.'], dtype=object),
  'labels': array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'O'], dtype=object)
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:
  ```
  Følgende er fraser og JSON-ordbøker med de navngitte enhetene som forekommer i den gitte frasen.
  ```
- Base prompt template:
  ```
  Frase: {text}
  Navngitte enheter: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Frase: {text}

  Identifiser de navngitte enhetene i frasen. Du bør outputte dette som en JSON-ordbok med nøklene 'person', 'sted', 'organisasjon' og 'diverse'. Verdiene skal være lister over de navngitte enhetene av den typen, akkurat som de vises i frasen.
  ```
- Label mapping:
    - `B-PER` ➡️ `person`
    - `I-PER` ➡️ `person`
    - `B-LOC` ➡️ `sted`
    - `I-LOC` ➡️ `sted`
    - `B-ORG` ➡️ `organisasjon`
    - `I-ORG` ➡️ `organisasjon`
    - `B-MISC` ➡️ `diverse`
    - `I-MISC` ➡️ `diverse`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset norne-nb
```


### NorNE-nn

This dataset was published in [this paper](https://aclanthology.org/2020.lrec-1.559/)
and is a manually NER annotated version of the [Nynorsk Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Norwegian-Nynorsk). The NER labels
almost follow the CoNLL-2003 standard, but with some additional labels.

The original full dataset consists of 14,174 / 1,890 / 1,511 samples for training,
validation and test, respectively. We use a split of 1,024 / 256 / 2,048 samples for
training, validation and test, respectively. The splits we use are new, so there might
be some samples from the training split in the validation or test splits.

We have mapped the labels into the CoNLL-2003 standard as follows:

- `LOC` ➡️ `LOC`
- `PER` ➡️ `PER`
- `ORG` ➡️ `ORG`
- `MISC` ➡️ `MISC`
- `GPE_LOC` ➡️ `LOC`
- `GPE_ORG` ➡️ `ORG`
- `PROD` ➡️ `MISC`
- `DRV` ➡️ `MISC`
- `EVT` ➡️ `MISC`

Here are a few examples from the training split:

```json
{
  "tokens": array(['-', 'Ulfr', 'provoserer', 'kjapt', 'fram', 'eit', 'slagsmål', ',', 'og', 'han', 'drep', 'hovdingen', '.'], dtype=object),
  "labels": array(['O', 'B-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
```
```json
{
  "tokens": array(['I', 'haust', 'blei', 'det', 'avslørt', 'at', 'minst', 'to', 'tolvåringar', 'på', 'mellomtrinnet', 'ved', 'Gimle', 'skule', 'hadde', 'med', 'seg', 'alkohol', 'på', 'ein', 'skuletur', '.'], dtype=object),
  "labels": array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
```
```json
{
  "tokens": array(['Krigen', 'mot', 'Irak', 'skulle', 'aldri', 'ha', 'vore', 'gjennomførd', '.'], dtype=object),
  "labels": array(['O', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:
  ```
  Følgende er fraser og JSON-ordbøker med de navngitte enhetene som forekommer i den gitte frasen.
  ```
- Base prompt template:
  ```
  Frase: {text}
  Navngitte enheter: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Frase: {text}

  Identifiser de navngitte enhetene i frasen. Du bør outputte dette som en JSON-ordbok med nøklene 'person', 'sted', 'organisasjon' og 'diverse'. Verdiene skal være lister over de navngitte enhetene av den typen, akkurat som de vises i frasen.
  ```
- Label mapping:
    - `B-PER` ➡️ `person`
    - `I-PER` ➡️ `person`
    - `B-LOC` ➡️ `sted`
    - `I-LOC` ➡️ `sted`
    - `B-ORG` ➡️ `organisasjon`
    - `I-ORG` ➡️ `organisasjon`
    - `B-MISC` ➡️ `diverse`
    - `I-MISC` ➡️ `diverse`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset norne-nn
```


## Linguistic Acceptability

### ScaLA-nb

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Bokmål Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Norwegian-Bokmaal) by
assuming that the documents in the treebank are correct, and corrupting the samples to
create grammatically incorrect samples. The corruptions were done by either removing a
word from a sentence, or by swapping two neighbouring words in a sentence. To ensure
that this does indeed break the grammaticality of the sentence, a set of rules were used
on the part-of-speech tags of the words in the sentence.

The original full dataset consists of 1,024 / 256 / 2,048 samples for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
used as-is in the framework.

Here are a few examples from the training split:

```json
{
  "text": "En vellykket gjennomføring av denne reformen vil bli en avgjørende prøve på Regjeringens handlekraft.",
  "label": "correct"
}
```
```json
{
  "text": "Lunde var ikke blant, mener Andreassen.",
  "label": "incorrect"
}
```
```json
{
  "text": "72 kjoler går hver med sesong.",
  "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Følgende er setninger og hvorvidt de er grammatisk korrekte.
  ```
- Base prompt template:
  ```
  Setning: {text}
  Grammatisk korrekt: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Setning: {text}

  Bestem om setningen er grammatisk korrekt eller ikke. Svar med 'ja' hvis setningen er korrekt og 'nei' hvis den ikke er.
  ```
- Label mapping:
    - `correct` ➡️ `ja`
    - `incorrect` ➡️ `nei`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset scala-nb
```


### ScaLA-nn

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Nynorsk Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Norwegian-Nynorsk) by
assuming that the documents in the treebank are correct, and corrupting the samples to
create grammatically incorrect samples. The corruptions were done by either removing a
word from a sentence, or by swapping two neighbouring words in a sentence. To ensure
that this does indeed break the grammaticality of the sentence, a set of rules were used
on the part-of-speech tags of the words in the sentence.

The original full dataset consists of 1,024 / 256 / 2,048 samples for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
used as-is in the framework.

Here are a few examples from the training split:

```json
{
  "text": "Dersom Noreg snart går forbi Danmark i folketal, slik framskrivingane tilseier, kan også dette langt på veg forklarast med naturressursar.",
  "label": "correct"
}
```
```json
{
  "text": "Eg kan ikkje sjå at det er grunn til å ha ei slik grense i lova, det kan vurderast i, seier ho.",
  "label": "incorrect"
}
```
```json
{
  "text": "SV har elles levert og i dag framsett ei gode forslag som kan bidra til å gjera noko med straumprisproblematikken og straumforbruket, om viljen vår er der.",
  "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Følgende er setninger og hvorvidt de er grammatisk korrekte.
  ```
- Base prompt template:
  ```
  Setning: {text}
  Grammatisk korrekt: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Setning: {text}

  Bestem om setningen er grammatisk korrekt eller ikke. Svar med 'ja' hvis setningen er korrekt og 'nei' hvis den ikke er.
  ```
- Label mapping:
    - `correct` ➡️ `ja`
    - `incorrect` ➡️ `nei`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset scala-nn
```


## Reading Comprehension

### NorQuAD

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.17/)
and is a manually annotated dataset based on data from the Bokmål Wikipedia.

The original full dataset consists of 3,810 / 472 / 472 samples for training, validation
and test, respectively. We use a split of 1,024 / 256 / 2,048 samples for training,
validation and test, respectively. When creating the splits, we only select samples that
contain an answer in the associated context. The splits we use are new, so there might
be some samples from the training split in the validation or test splits.

Here are a few examples from the training split:

```json
{
  "context": 'Sprekpodden: Denne treningen gjør deg smartere og lykkeligere\nHJERNEFORSKER: – Hjernen er i utgangspunktet programmert for latskap. Derfor må vi i større grad tvinge oss selv til å være mer aktive, sier forsker Ole Petter Hjelle. Foto: Tor Stenersen (arkiv)\nSPREKPODDEN: Denne uken har programleder Daniel Røed-Johansen og Malene Indrebø-Langlo besøk av Ole Petter Hjelle. Foto: Morten Uglum\n– Vi var rett og slett lei av å sitte og fortelle pasientene våre at de måtte være i fysisk aktivitet, uten at noe skjedde.\nFor noen år siden startet hjerneforsker og fastlege Ole Petter Hjelle, og de andre legene på Åsgårdstrand legekontor, en treningsgruppe for pasientene sine. Det ble stor suksess.\n– Folk vet at det er bra å trene for den fysiske helsen, men at fysisk aktivitet også er bra for den mentale helse, er et underkommunisert tema, sier han.\nBedre enn sudoku og kryssord\n– Er fysisk aktivitet bedre hjernetrim enn sudoku og kryssord?\n– Løser du masse kryssord, så blir du veldig til å løse kryssord. Men det har ikke de store ringvirkningene på våre kognitive funksjoner, som det å huske, planlegge og gjennomføre, sier Hjelle.\nHan forklarer at når pulsen vår øker, skilles det ut vekstfaktorer i hjernen som beskytter hjernecellene våre og gjør at cellene kommuniserer bedre.\nForskning viser også at det dannes nye hjerneceller i enkelte deler av hjernen, under aktivitet.\n– Men skal man få denne effekten, må man rett og slett være i aktivitet.\nFå opp pulsen\nForskning viser også at fysisk aktivitet reduserer risiko for depresjon og demens, øker intelligensen, bedrer hukommelsen, gjør deg mer kreativ og gir deg et lengre og bedre liv.\nHjelle forteller at det viktigste for å hente ut disse fordelene er å få opp pulsen.\n– Men dersom du skulle valgt en aktivitet – som i størst mulig grad stimulerte flest mulig hjerneområder – pleier jeg å si ballspill. Da får du opp pulsen, du samarbeider, har taktikk, koordinasjon, balanse og strategi, sier Hjelle.\nHør mer fra «treningslegen» i ukens Sprekpodden her.',
  "question": 'Hva jobber Daniel som?',
  "answers": {
    "answer_start": array([286]),
    "text": array(['programleder'], dtype=object)
  }
}
```
```json
{
  "context": 'Litauiske medier: En utvekslingsavtale skal være på plass for Frode Berg\nFrode Berg ble dømt til 14 års fengsel i Russland. Foto: Tore Meek / NTB scanpix\nRussland og Litauen er enige om å utveksle en spiondømt russer mot to litauere og en nordmann, opplyser kilder i den litauiske sikkerhetstjenesten til den litauiske nyhetstjenesten Baltic News Service (BNS).\n– Utvekslingsavtalen inkluderer også en norsk statsborger som er dømt i Russland, sier en anonym tjenestemann i den litauiske sikkerhetstjenesten.\nAvisen navngir ikke Frode Berg, men Berg er den eneste nordmannen som soner en slik dom i Russland.\nAftenposten og en rekke norske medier omtalte saken onsdag ettermiddag. Flere russiske medier melder også om det samme, alle med BNS som kilde\n– Håper en avtale foreligger\nFrode Bergs norske advokat Brynjulf Risnes kan ikke bekrefte opplysningene.\n– Jeg har ikke informasjon som verken bekrefter eller avkrefter en slik avtale. Vi håper selvsagt at en slik avtale foreligger, sier Risnes til NTB.\nUD vil ikke kommentere saken.\n– Norske myndigheter ønsker å få Frode Berg hjem. Vi håndterer saken på den måten som vi mener er best for å ivareta hans interesser. Utover det kommenterer vi ikke saken, sier underdirektør Ane Haavardsdatter Lunde i Utenriksdepartementet til NTB.\nBergs russiske forsvarer, advokat Ilja Novikov, ikke vil kommentere saken, ifølge NRK.\nStøttegruppen for Frode Berg håper opplysningene stemmer.\n– Dersom det viser seg at dette er riktig, er det en ufattelig god nyhet som vi har ventet på skulle skje, sier støttegruppemedlem Thorbjørn Brox Webber til NTB.\n– En slik avtale må bety at Frode kan komme tilbake til Norge og Kirkenes, legger han til.\nDømt for spionasje\nBerg er dømt til 14 års fengsel for spionasje. Han ble pågrepet i Moskva i desember 2017 og har sittet fengslet siden.\nNRK meldte i august at UD er i forhandlinger med Russland om å få Berg hjem og har informert hans nærmeste familie om dette.\nMuligheten for en utvekslingsavtale har vært antydet, men et problem har vært hvem den i så fall skal omfatte.',
  "question": 'Hvilken norske advokat representerer Frode Berg?',
  "answers": {
    "answer_start": array([808]),
    "text": array(['Brynjulf Risnes'], dtype=object)
  }
}
```
```json
{
  "context": 'Ny nedtur for Ruud\nCasper Ruud røk torsdag ut av challengerturneringen i Koblenz. Bildet er fra en tidligere turnering.\nAv Ole Henrik Tveten\nDet ble en frustrerende kamp mot nederlandske Tallpon Griekspoor torsdag. Casper Ruud vant første sett 6-4, men etter det var det lite som stemte for nordmannen i Tyskland.\nI andre sett ble Ruud utspilt og tapte 1-6, mens feilene fortsatte å florere også i tredje sett og Ruud tapte settet 2-6.\nDen norske 20-åringen gikk rett inn i 2. runde i Koblenz-turneringen etter å ha fått walkover i den første. Der slet han seg til seier mot italienske Raul Brancaccio onsdag. Torsdagens motstander, Tallpon Griekspoor, er nummer 233 på verdensrankingen.\nDet startet bra for Snarøya-gutten da han i første sett brøt nederlenderens serve og tok ledelsen 4-3. Servebruddet ble avgjørende for settet som Ruud vant 6-4, etter blant annet å ha reddet en breakball etter en lengre ballveksling.\nI andre sett begynte problemene for Casper Ruud. Griekspoor brøt Ruuds serve ved første anledning og gikk opp i 2-0-ledelse. Deretter vant han egen serve, brøt Ruuds serve på ny og vant så egen serve. Da ledet plutselig nederlenderen 5-0.\nNordmannen servet inn til 5-1, men det var dessverre ikke starten på noen snuoperasjon. Nederlenderen vant settet 6-1.\nNordmannen hadde ikke ristet av seg problemene i pausen, og ble feid av banen av Griekspoor. Ruud kom under 0-4 i tredje sett før han omsider reduserte til 1-4. Men da var det for sent.\nNederlenderen servet inn 5-1, Ruud reduserte, før Griekspoor servet seieren i land. Dermed tapte Ruud tredje sett 6-2 og røk ut av turneringen.\nÅ ryke ut i Tyskland hjelper ikke nordmannens jakt på rankingpoeng for å komme seg inn i topp 100 i verden. Han risikerer å falle flere plasser ettersom han mister de 70 rankingpoengene han skaffet seg da han tok seg til 2. runde i Australian Open i fjor. Ruud er akkurat nå nummer 112 på verdensrankingen. (NTB)',
  "question": 'Hvordan endte 1. sett mellom Ruud og Griekspoor?',
  "answers": {
    "answer_start": array([244]),
    "text": array(['6-4'], dtype=object)
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 2
- Prefix prompt:
  ```
  Her følger tekster med tilhørende spørsmål og svar.
  ```
- Base prompt template:
  ```
  Tekst: {text}
  Spørsmål: {question}
  Svar på maks 3 ord: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Tekst: {text}

  Besvar følgende spørsmål om teksten ovenfor med maks 3 ord.

  Spørsmål: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset norquad
```


### Belebele

This dataset was published in [this paper](https://aclanthology.org/2024.acl-long.44/) and is a large-scale multilingual reading comprehension dataset covering 122 languages. The questions are generated from Wikipedia articles and are designed to test various aspects of reading comprehension, including factual understanding, inference, and numerical reasoning.

The dataset provides training, validation, and test splits with human-verified question-answer pairs. The questions are generated to be answerable from the given context and cover diverse topics from Wikipedia articles.

When evaluating generative models, we use the following setup (see the [methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:
  ```
  Her følger tekster med tilhørende spørsmål og svar.
  ```
- Base prompt template:
  ```
  Tekst: {text}
  Spørsmål: {question}
  Svar: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Tekst: {text}

  Besvar følgende spørsmål om teksten ovenfor.

  Spørsmål: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset belebele-no
```


### Unofficial: NorGLM Multi QA

This dataset was released in [this paper](https://doi.org/10.48550/arXiv.2312.01314) and
features a manually annotated reading comprehension dataset based on Norwegian news
articles. This dataset is an _abstractive_ question answering dataset, meaning that the
answers do not always feature in the context. To fix this, they were rephrased using
[this
script](https://github.com/ScandEval/ScandEval/blob/main/src/scripts/create_norglm_multiqa.py),
which utilised the `gpt-4o-2024-05-13` model.

The original dataset contains 2,406 samples, which we split into 1,024 / 256 / 1,126
samples for training, validation and test, respectively.

Here are a few examples from the training split:

```json
{
  "context": ' Kommer det melding om at ansatte kjøper aksjer i eget selskap, kan det være gode grunner til at du også bør gjøre det. – Vær på lag med innsiderne, er ekspertens råd.Har du lyst til å prøve deg som aksjeinvestor helt gratis og uten reell risiko? Meld deg på Aksje-NM her!Mange assosierer innsidehandel med kjøp og salg av aksjer basert på tilgang på selskapsnyheter før de blir offentliggjort i markedet. Slik handel kan gi stor økonomisk gevinst, og er ulovlig.Det finnes derimot også en lovlig form for innsidehandel, og denne kan det være lurt å følge med på, skal vi tro forskningssjef Geir Linløkken i Investtech. Aksjeskolen er en del av E24s Aksje-NM. En tidligere versjon av denne artikkelserien ble publisert i 2020.Når man snakker om «innsidehandel» i børssammenheng, siktes det som regel til handler som direktører, styremedlemmer og andre nøkkelmedarbeidere gjør. Disse handlene må rapporteres inn til Oslo Børs, og kjøpet eller salget blir offentlig informasjon. Denne informasjonen kan være gull verdt, skal vi tro forskningen til Investtech.– Nøkkelpersoner som direktører og styremedlemmer sitter på veldig mye kunnskap om bedriften. Når disse enten selger eller kjøper aksjer i eget selskap, kan det ses på som et signal til andre aktører, sier Linløkken. Linløkken har forsket på innsidehandel og tatt utgangspunkt i over 11.000 rapporterte innsidekjøp i norske og svenske selskaper. Han har sett nærmere på hvordan kursen utviklet seg i tiden etter innsidekjøpet. – Vi fant at disse selskapene på årlig basis steg med 7,1 prosentpoeng mer enn andre selskaper. Det kan altså være et godt tips å følge med på innsidekjøp.Dersom det tikker inn meldinger om at innsidere selger aksjene sine, er det også lurt å følge nøye med. Investtech har tatt utgangspunkt i over 6.900 slike tilfeller i Norge og Sverige, og gjorde spennende funn. – I snitt gjorde disse aksjene det 3,0 prosentpoeng svakere enn børsen, sier han. Linløkken forteller at noen av aksjene kan ha falt for eksempel 50 prosent etter innsidesalg, mens det kan ha gått ganske bra i andre selskaper med innsidesalg.– Men i gjennomsnitt har disse aksjene gjort det dårlig, fastslår han.Linløkken sier at Investtech anser innsidehandelanalyse som en forenklet fundamental analyse, altså en analyse av om aksjen er billig eller dyr i forhold til verdiene i selskapet. Har man ikke tid eller kunnskap til å gjøre slik analyse selv, er det et godt alternativ å se til innsiderne. – Historisk og statistisk sett, har det vært riktig å følge innsiderne og være på lag med dem, svarer Linløkken.',
  "question": 'Hva kan man gjøre dersom man ikke har tid eller kunnskap til å gjøre en analyse av aksjene til et selskap?',
  "answers": {
    "answer_start": 2434,
    "text": array(['Se til innsiderne.'], dtype=object)
  }
}
```
```json
{
  "context": ' Alt om pubertet, penis, psyken og livet sjæl. Nok en fullkommen bok fra duoen bak et par av de største boksuksessene de siste årene. «De har gjort det igjen», skrev jeg i VG for ganske nøyaktig to år siden, da jeg satt her og leste og anmeldte «Jenteboka» av legene Nina Brochmann og Ellen Støkken Dahl. Da hadde det gått to år siden de brak-debuterte med «Gleden med skjeden». Jeg gav «Jenteboka» terningkast 6. Vel, vel. Du har kanskje gjettet det nå, men nå har de altså gjort det enda en gang: Laget en knallgod, fullkommen bok vi får håpe mange leser.For jeg tør påstå at guttene trenger sin Guttebok vel så mye som jentene trenger sin. For selv om det er jentene vi har snakket mest om, er det mange unge gutter som sliter. Unge gutter faller oftere ut av skolen, er mer deprimerte og har mindre fremtidsoptimisme enn før. Det finnes dyster statistikk, kort fortalt: De opplever også stress og press og uhelse. Og så er de ikke så flinke til å snakke om det. I «Gutteboka» tar Brochmann og Dahl for seg alt man må vite og forstå når man er på vei inn i eller står midt i puberteten. (Eller senere i livet, for den saks skyld, jeg plukket opp noen gode tips selv, jeg.) De skriver om kroppshår, kviser, stemmeskifte,  legning, penisstørrelse, pung, kjønn, sæd, kåthet, ereksjonsknipe (!) og svettelukt, for å nevne noen av mange høydepunkter.  Legeduoen havnet på denne lista: De ti heteste norske forfatterne i utlandet! Foruten alle de rent kroppslige og fysiske forandringene man kan oppleve på veien fra gutt til mann, inneholder boka gode kapitler om de psykiske aspektene og livet sjæl. Grensesetting, samtykke, nettvett, om å trenge en pornopause, om psykisk uhelse, stress og press. «Alle har det vondt iblant, men ingen har det vondt for alltid. Du kommer til å bli glad igjen!» Det er noe med tonen i boka, som er så fin. Lett, åpen, sympatisk, avvæpnende. Smart, kul og og med faglig tyngde. Men aldri formanende, ingen pekefinger. «Onani er godt og sunt. Onani er ikke bare ufarlig – det er bra for deg.» «Kroppen din er laget for å brukes og nytes.»  «Det er synd at trening ender opp med å handle om bare utseendet. Å trene er nemlig bra for deg. Det er ikke jakten på «drømmekroppen».» Selv de mer alvorlige og kliniske temaene er dessuten en fornøyelse å bla om til, også takket være de fantastiske illustrasjonene til Magnhild Wisnes. De er fargerike og morsomme, og gjør boka komplett. Så mange peniser har jeg ikke sett siden vi fniste og lo av «Penisatlaset» på et nachspiel i studietiden. Så kan man jo stille seg spørsmålet, om denne boka når frem til dem som trenger å lese den. Den burde egentlig vært pensum, tenker jeg, eller i alle fall utgangspunkt for et prosjekt på skolen. Å sette seg ned med en bok, som attpåtil handler om puberteten, står vel ikke høyest på lista over hva tenåringsgutter flest vil bruke fritiden sin på. Prøv likevel.  Jeg vet ikke, kanskje betale gutten noen kroner for å lese den, om det er det som skal til. Jeg føler meg sikker på at det vil være verdt det. For hvis de unge guttene våre leser denne boka, er jeg sikker på at livet blir lettere å leve og verden et morsommere sted. Anmeldt av: Trine Saugestad Hatlen',
  "question": 'Hvem står for illustrasjonene i «Gutteboka»?',
  "answers": {
    "answer_start": 2321,
    "text": array(['illustrasjonene til Magnhild Wisnes'], dtype=object)
  }
}
```
```json
{
  "context": ' Regjeringen lanserer ny handlingsplan for å beskytte den truede villaksen. – Altfor slapt, sier SV-politiker.Regjeringen lanserer nå en handlingsplan for å bevare den truede villaksen.– Villaksen kan nå bli rødlistet i Norge for første gong. Det er helt klart at det trengs konkrete tiltak for å snu denne utviklingen, sier Sveinung Rotevatn i pressemeldingen fra regjeringen.Handlingsplanen inneholder tiltak mot blant annet lakselus, rømt oppdrettsfisk, lakseparasitten Gyro, vannkraftregulering, forsuring, overbeskatning og fremmende fiskearter som pukkellaks.Regjeringen viser til at lakselus utgjør den største risikoen for å gjøre ytterligere skade på vill atlantisk laks, ifølge Vitenskapelig råd for lakseforvaltning.– Lakselus utgjør en stor risiko for villaksen. Regjeringen vil blant annet utrede krav om nullutslipp av lakselus fra oppdrettsanlegg fra og med 2030, sier Rotevatn.Det vil i så fall innebære krav om lukkede anlegg.Lakselus finnes naturlig i alle havområder på den nordlige halvkule, og er den vanligste parasitten på laksefisk.Blir forekomsten av lus høy, kan det være en utfordring både for oppdrettsfisk og vill laksefisk.Havbruk medfører at antall fisk i sjøen øker, og dermed øker også antall verter for lakselus. Nivåene med lakselus i anleggene må derfor holdes lavest mulig, slik at de samlede lusemengdene i sjøen ikke blir for store.Som følge av omfattende resistens hos lusen mot kjemiske behandlingsmidler, har næringen de siste årene vært tvunget til å ta i bruk mekaniske metoder for å fjerne lusen, med negative konsekvenser for fiskens velferd.Kilde: Lusedata, MattilsynetDagens trafikklyssystem som regulerer veksten i næringen i forhold til luseutviklingen, skal også utvikles og forbedres.Planen inneholder også tiltak mot en rekke andre påvirkningsfaktorer. Utfisking av rømt oppdrettslaks skal økes, og det skal vurderes nye metoder for å spore og merke oppdrettslaks og hindre at rømt oppdrettslaks gyter.Hele 80 prosent av villaksbestandene i Norge når for tiden ikke minstemålet for god kvalitet. Rømt oppdrettslaks og lakselus er regnet som de to største truslene, skriver regjeringen.Fremmende fiskearter utgjør også en risiko for både biologisk mangfold, produktiviteten til lokal laksefisk og akvakultur.I år har Norge hatt den største invasjonen av pukkellaks noensinne, og regjeringen vil derfor opprette en nasjonal kompetansegruppe for å koordinere arbeidet med dette.SVs nestleder Torgeir Knag Fylkesnes er ikke fornøyd med tiltakene.– Dette er altfor, altfor slapt. Regjeringen tar ikke tak i elefanten i rommet, nemlig den lite bærekraftige forvaltningen av oppdrettsnæringa. Vi må stille strengere miljøkrav til alle nye oppdrettstillatelser, og fase inn disse kravene hos de med eksisterende tillatelser, skriver han i en kommentar til E24.Han påpeker at det i dag tildeles oppdrettstillatelser til den høystbydende, og ikke til de med den mest miljøvennlige teknologien. – Skal vi redde villaksen og sikre en bærekraftig vekst for oppdrettsnæringen, må vi legge om systemet slik at vi gjennom å gi billigere tillatelser, men med krav om nullutslipp, null rømming og null ressurser på avveie.Fylkesnes understreker videre at teknologien finnes, og at næringen har god råd.– Når man for eksempel ser på Salmars investeringsaktivitet de siste ukene, så ser vi at næringen både kan betale for ny teknologi og skatt på formue og grunnrente.Fylkesnes gikk tidligere denne uken hardt ut mot Salmar-eier Gustav Witzøe, etter at laksemilliardæren uttalte seg kritisk mot økning i formuesskatten tidligere i sommer.',
  "question": 'Hva inneholder regjeringens nye handlingsplan for villaksen?',
  "answers": {
    "answer_start": 377,
    "text": array(['Handlingsplanen inneholder tiltak mot blant annet'], dtype=object)
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 2
- Prefix prompt:
  ```
  Her følger tekster med tilhørende spørsmål og svar.
  ```
- Base prompt template:
  ```
  Tekst: {text}
  Spørsmål: {question}
  Svar på maks 3 ord: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Tekst: {text}

  Besvar følgende spørsmål om teksten ovenfor med maks 3 ord.

  Spørsmål: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset norglm-multi-qa
```


## Knowledge

### MMLU-no

This dataset is a machine translated version of the English [MMLU
dataset](https://openreview.net/forum?id=d7KBjmI3GmQ) and features questions within 57
different topics, such as elementary mathematics, US history and law. The translation to
Norwegian was conducted using the [DeepL translation
API](https://www.deepl.com/en/products/api).

The original full dataset consists of 269 / 1,410 / 13,200 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 2,048 split for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
new and there can thus be some overlap between the original validation and test sets and
our validation and test sets.

Here are a few examples from the training split:

```json
{
  "text": "Hvorfor er Mahavira en viktig person i jainatradisjonene?\nSvaralternativer:\na. Han er den siste av de asketiske profetene.\nb. Han er den første av de asketiske profetene\nc. Han er den mest lærde av de asketiske profetene\nd. Han er den helligste av de asketiske profetene",
  "label": "a"
}
```
```json
{
  "text": "En enfaset fullbroomformer kan drives i lastkommuteringsmodus hvis belastningen består av\nSvaralternativer:\na. RL.\nb. RLC underdempet.\nc. RLC overdempet.\nd. RLC kritisk dempet.",
  "label": "b"
}
```
```json
{
  "text": "En professor, som var eneeier av en boligblokk, skrev et skjøte med følgende ordlyd: \"Jeg overdrar herved min boligblokk til min sønn og datter som leietakere i fellesskap.\" I skjøtet, som var korrekt utferdiget, forbeholdt professoren seg en livsvarig eiendomsrett. Professoren fortalte deretter barna sine om overdragelsen og la den i familiehvelvet i biblioteket for oppbevaring. Deretter giftet sønnen seg med en lege. Professoren, som mislikte legen, utferdiget deretter et nytt skjøte som han kalte \"et korreksjonsskjøte\". I \"korreksjonsskjøtet\" overførte professoren bygården \"til min sønn og datter som sameiere med overlevelsesrett.\" Ifølge det nye skjøtet forbeholdt professoren seg igjen livsvarig eiendomsrett. Begge barna aksepterte overdragelsen av \"korreksjonsskjøtet.\" Et halvt år senere døde sønnen, og etterlot seg legen som eneste arving. Eiendomsretten til boligblokken er i datterens og\nSvaralternativer:\na. datteren og legen som sameiere.\nb. datteren med forbehold om professorens livstidsarv.\nc. datteren og legen som sameiere, med forbehold om professorens livsarvinger.\nd. datteren og legen som sameiere med overlevelsesrett, med forbehold for professorens livsarvinger.",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Følgende er flervalgsspørsmål (med svar).
  ```
- Base prompt template:
  ```
  Spørsmål: {text}
  Svaralternativer:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svar: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Spørsmål: {text}
  Svaralternativer:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Besvar følgende spørsmål med 'a', 'b', 'c' eller 'd', og engu öðru.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset mmlu-no
```


### Unofficial: ARC-no

This dataset is a machine translated version of the English [ARC
dataset](https://doi.org/10.48550/arXiv.1803.05457) and features US grade-school science
questions. The translation to Norwegian was conducted using the [DeepL translation
API](https://www.deepl.com/en/products/api).

The original full dataset consists of 1,110 / 297 / 1,170 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 1,024 split for training,
validation and testing, respectively (so 2,304 samples used in total). All new splits
are subsets of the original splits.

Here are a few examples from the training split:

```json
{
  "text": "Hvorfor er det tryggere å se på månen enn på solen?\nSvaralternativer:\na. Månen er mindre lyssterk.\nb. Månen er nærmere jorden.\nc. Månen skinner mest om natten.\nd. Månen er full bare én gang i måneden.",
  "label": "a"
}
```
```json
{
  "text": "Hvilket av følgende er et biprodukt av celleånding hos dyr?\nSvaralternativer:\na. oksygen\nb. varme\nc. sukker\nd. protein",
  "label": "b"
}
```
```json
{
  "text": "Big Bang-teorien sier at universet\nSvaralternativer:\na. trekker seg sammen.\nb. ikke har noen begynnelse.\nc. startet som én enkelt masse.\nd. hele tiden danner hydrogen.",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Følgende er flervalgsspørsmål (med svar).
  ```
- Base prompt template:
  ```
  Spørsmål: {text}
  Svaralternativer:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svar: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Spørsmål: {text}
  Svaralternativer:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Besvar følgende spørsmål med 'a', 'b', 'c' eller 'd', og engu öðru.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset arc-no
```


## Common-sense Reasoning

### HellaSwag-no

This dataset is a machine translated version of the English [HellaSwag
dataset](https://aclanthology.org/P19-1472/). The original dataset was based on both
video descriptions from ActivityNet as well as how-to articles from WikiHow. The dataset
was translated to Norwegian using the [DeepL translation
API](https://www.deepl.com/en/products/api).

The original full dataset consists of 9,310 samples. We use a 1,024 / 256 / 2,048 split
for training, validation and testing, respectively (so 3,328 samples used in total).

Here are a few examples from the training split:

```json
{
  "text": "[header] Slik holder du deg kjølig og føler deg frisk om sommeren [title] Dusj hver dag. [step] Bruk en eksfolierende dusjsåpe for å fjerne smuss. Sett vannet på varmt i starten av dusjen (fordi det rengjør deg mer effektivt), men mot slutten av dusjen setter du vannet på lunkent eller kjølig.\nSvaralternativer:\na. Dette senker kroppstemperaturen slik at du føler deg kjøligere (og våkner opp om morgenen!). [Smør deg med fuktighetskrem rett etter at du har gått ut av dusjen.\nb. Påfør denne gelen på svetten under armene eller på kroppen. Tenk på det som å spyle den ene armhulen med vann (du kan lage din egen dusjsåpe med armene eller bena, og du kan vaske av deg litt med en gang).\nc. Alternativt kan du åpne døren og la kjølig vann strømme gjennom det åpne vinduet i minst en time. [Bruk en ansiktsmaske mens du dusjer.\nd. Vannet skal være varmt nok til å skylle ut smuss og død hud som henger over ansiktet. Påfør kroppssåpe (eller la den være åpen for lufting) på hudoverflaten i korte riller.",
  "label": "a"
}
```
```json
{
  "text": "En løper løper på en bane foran en folkemengde. en mann\nSvaralternativer:\na. kaster en ball som hunden skal fange.\nb. snakker til kameraet.\nc. løper ikke når han hopper ned i en sandkasse.\nd. gir en kort introduksjon før han fortsetter og konkurrerer mot mannen i svart.",
  "label": "b"
}
```
```json
{
  "text": "[header] Slik vet du om hunden din liker deg best [title] Legg merke til at hunden din følger mye etter deg. [En måte å bevise at en hund liker deg best, er når den er mye sammen med deg. Så hold øye med om hunden din liker å være i nærheten av deg.\nSvaralternativer:\na. [Hold øye med eventuell fysisk atferd. [Et godt eksempel på denne atferden er hvis den presser rumpa opp mot låret ditt og sjekker hva du har på deg.\nb. [Se etter tegn på at hunden din kan være flørtende. [Et godt tegn på at hunden din liker deg er at den klapper deg mye eller stirrer på deg i intime øyeblikk.\nc. [Finn ut om hunden din liker å leke med deg. [Hvis det er en hund som elsker leker, kan du leke med dem, og hvis den er veldig glad i å leke, så liker den at du leker med den.\nd. Legg merke til at hunden din følger deg rundt i huset hver dag når du er ute og går. Selv om du kanskje ikke har lyst til det, kan det å tilbringe mye tid sammen med en hund få den til å føle seg komfortabel med deg.",
  "label": "c"
}
