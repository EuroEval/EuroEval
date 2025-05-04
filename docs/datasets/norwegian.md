# 🇳🇴 Norwegian

This is an overview of all the datasets used in the Norwegian part of EuroEval. The
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
$ euroeval --model <model-id> --dataset norec
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
$ euroeval --model <model-id> --dataset norne-nb
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
$ euroeval --model <model-id> --dataset norne-nn
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

The original dataset consists of 20,044 samples, from which we use 1,024 / 256 / 2,048 samples for training,
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
$ euroeval --model <model-id> --dataset scala-nb
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

The original dataset consists of 17,575 samples, from which we use 1,024 / 256 / 2,048 samples for training,
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
$ euroeval --model <model-id> --dataset scala-nn
```


### Unofficial: NoCoLA

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.60/)
and is based on the annotated language learner corpus
[ASK](https://aclanthology.org/L06-1345/). Notably, the individual types of errors are
also annotated in this dataset. We use the error types to ensure that there is an equal
representation of each error type, but then collapse the error types into `correct` and
`incorrect`.

The original dataset consists of 116,199 / 14,293 / 14,387 samples for training,
validation and test, respectively. We use 1,024 / 256 / 2,048 samples for training,
validation and test, respectively, where we sample each error type equally. All splits
are subsets of the original splits.

Here are a few examples from the training split:

```json
{
  "text": "Vi har hatt krig i nesten ti år. Jeg føler meg noen ganger trist fordi jeg har mistet flere venner og min far på grunn av krigen.",
  "label": "correct"
}
```
```json
{
  "text": "Hvis jeg ikke sier in n genting, kan han spille hele dagen.",
  "label": "incorrect"
}
```
```json
{
  "text": "De føler at samfunnet trenger ikke dem.",
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
$ euroeval --model <model-id> --dataset no-cola-binary
```


### Unofficial: Jentoft

This dataset was published in [this Master's thesis](https://www.duo.uio.no/handle/10852/103885) by Matias Jentoft.

The original dataset consists of 85,771 / 10,827 / 10487 samples for training,
validation and test, respectively. We use a split of 1,024 / 256 / 2,048 samples for
training, validation and test, respectively. In each split, the distribution of
`correct` and `incorrect` is 50/50.

Here are a few examples from the training split:

```json
{
  "text": "For to uker siden var jeg på en fotoutstilling om Erytrea.",
  "label": "incorrect"
}
```
```json
{
  "text": "Det viser seg at folk ikke kan leve uten mobiltelefonen.",
  "label": "correct"
}
```
```json
{
  "text": "Mobiltelefoner dominerer mange av oss, og vi bruker dem over alt, på gatene 'hvert hjørne', i gatene, holdeplasser, kaffeteriaene og i parken, der folk burde tilbringe koselig tid sammen i naturen.",
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
$ euroeval --model <model-id> --dataset jentoft
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
$ euroeval --model <model-id> --dataset norquad
```


### Unofficial: NorGLM Multi QA

This dataset was released in [this paper](https://doi.org/10.48550/arXiv.2312.01314) and
features a manually annotated reading comprehension dataset based on Norwegian news
articles. This dataset is an _abstractive_ question answering dataset, meaning that the
answers do not always feature in the context. To fix this, they were rephrased using
[this
script](https://github.com/EuroEval/EuroEval/blob/main/src/scripts/create_norglm_multiqa.py),
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
$ euroeval --model <model-id> --dataset norglm-multi-qa
```


### Unofficial: BeleBele-no

This dataset was published in [this paper](https://aclanthology.org/2024.acl-long.44/) and features multiple-choice reading comprehension questions across 122 languages.

The original dataset contains 900 unique multiple-choice reading comprehension passages and questions. From these, we use a 256 / 64 / 580 split for training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Tekst: Det kinesiske nyhetsbyrået Xinhua meldte tidligere fra om at et fly var kapret. Det ble senere rapportert at flyet fikk en bombetrussel og ble veiledet mot retur til Afghanistan med landing i Kandahar. Ifølge de første rapportene ble flyet dirigert tilbake til Afghanistan etter å ha blitt nektet nødlanding i Ürümqi.\nSpørsmål: Hva ble ikke sagt i den nyeste rapporten fra nyhetsbyrået Xinhua?\nSvaralternativer:\na. Flyet fikk en bombetrussel\nb. Flyet landet i Ürümqi\nc. Flyet ble dirigert til Afghanistan\nd. Flyet landet i Kandahar",
  "label": "b"
}
```
```json
{
  "text": "Tekst: Tyskland begynte å gjøre seg klare til å invadere Storbritannia da kampen om Frankrike var over. Tyskland gav angrepet kodenavnet «operasjon sjøløve». Mesteparten av den britiske hærens tunge våpen og forsyninger hadde gått tapt da den flyktet fra Dunkirk, så de var svært sårbar. Den britiske marinen var imidlertid fremdeles mye kraftigere enn den tyske («Kriegsmarine») og kunne ha ødelagt en eventuell invasjonsflåte sendt over den engelske kanal. Det var likevel svært få skip fra Royal Navy som ble stasjonert nær de sannsynlige invasjonsrutene siden admiralene var engstelige for at de kom til å bli senket av tyske luftangrep.\nSpørsmål: Hva kalte Tyskland angrepet på Storbritannia?\nSvaralternativer:\na. Dunkirk\nb. Operasjon sjøløve\nc. Kriegsmarine\nd. Royal Navy",
  "label": "b"
}
```
```json
{
  "text": "Tekst: Det italienske og det tyske landslaget er de nest beste lagene i verden og var FIFA World Cup-mestere i 2006. Fotball, basketball, volleyball, vannpolo, fekting, rugby, sykling, ishockey, rullehockey og Formel-1 bilsport er godt likte sportsgrener. Vintersport er mest populært i nordlige områder, der italienere deltar i internasjonale konkurranser og OL-arrangementer.\nSpørsmål: Hvilke av følgende sporter vant et verdensmesterskap for Italia, basert på informasjonen i avsnittet?\nSvaralternativer:\na. Fotball\nb. Vannpolo\nc. Basketball\nd. Sykling",
  "label": "a"
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

  Besvar følgende spørsmål med 'a', 'b', 'c' eller 'd', og ikke noe annet.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset belebele-no
```


## Knowledge

### NRK Quiz QA

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2501.11128)
and is a multiple-choice question answering (QA) dataset designed for evaluation of the
Norwegian language and culture, including both Bokmål and Nynorsk. The dataset consists
of quizzes from NRK, the national public broadcaster in Norway.

The original dataset contains 4,930 samples, spread across 549 quizzes. We keep the
top-256 quizzes, allowing us to create splits stratified across all the remaining
quizzes. We 635 / 256 / 2048 samples for training, validation and test, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Gunnar har hatt plutselige og sterke smerteanfall siden han var liten gutt. Det var vondt å tisse og det gjorde vondt i ryggen og magen. Det hjalp litt å drikke vann. Reseptbelagte medisiner kan være nødvendig under anfall.\nSvaralternativer:\na. Nyrestein, kronisk\nb. Irritabel tarmsyndrom\nc. Angst\nd. Urinveisinfeksjon",
  "label": "a"
}
```
```json
{
  "text": "80 år gamle Harrison Ford er nok ein gong aktuell i rolla som Indiana Jones. Kva heiter filmen?\nSvaralternativer:\na. Indiana Jones and the Nasty Nazis\nb. Indiana Jones and the Dial of Destiny\nc. Indiana Jones and the Hunt for Power\nd. Indiana Jones Forever",
  "label": "b"
}
```
```json
{
  "text": "I 1980 måtte denne bassisten overnatte ni netter i fengsel i Japan fordi han prøvde å få med seg ca. 200 gram marihuana inn i landet. Hvem var det?\nSvaralternativer:\na. Sting\nb. Lemmy Kilmister\nc. Paul McCartney\nd. Bootsy Collins",
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

  Besvar følgende spørsmål med 'a', 'b', 'c', eller 'd', og ikke noe annet.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset nrk-quiz-qa
```


### Unofficial: MMLU-no

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

  Besvar følgende spørsmål med 'a', 'b', 'c' eller 'd', og ikke noe annet.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset mmlu-no
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

  Besvar følgende spørsmål med 'a', 'b', 'c' eller 'd', og ikke noe annet.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset arc-no
```


## Common-sense Reasoning

### NorCommonSenseQA

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2501.11128)
and is a manually translated and localised version of the English CommonSenseQA dataset.
There are samples in both Bokmål and Nynorsk, but with the vast majority being Bokmål.

The original dataset contains 1,093 samples. We use a 128 / 128 / 787 split for
training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Hvor er det sannsynlig at en fugl lager hjemmet sitt?\nSvaralternativer:\na. I skogen\nb. I et rede\nc. På taket\nd. På blader\ne. I himmelen",
  "label": "a"
}
```
```json
{
  "text": "Hvis et hjem har et abonnoment, hva får de sannsyneligvis hver dag i posten?\nSvaralternativer:\na. Delestykker\nb. En avis\nc. En gate\nd. En vaskemaskin\ne. Jordas overflate",
  "label": "b"
}
```
```json
{
  "text": "Når du ikke klarer å gjøre noe ferdig, hva feilet du i da?\nSvaralternativer:\na. Å vinne\nb. Å bestå\nc. Å fullfør\nd. Å gjøre det bra\ne. Å lykkes",
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
  e. {option_e}
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
  e. {option_e}

  Besvar følgende spørsmål med 'a', 'b', 'c', 'd' eller 'e', og ikke noe annet.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset nor-common-sense-qa
```


### Unofficial: HellaSwag-no

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

  Besvar følgende spørsmål med 'a', 'b', 'c' eller 'd', og ikke noe annet.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset hellaswag-no
```


## Summarization

### NoSammendrag

This dataset is a combination of [the SNL and VG summarisation
datasets](https://nmbu.brage.unit.no/nmbu-xmlui/handle/11250/3079868) as well as a
translated version of the English [XSum dataset](https://aclanthology.org/D18-1206/),
based on British BBC news articles. The SNL dataset is based on the Norwegian
encyclopedia Store Norske Leksikon, while the VG dataset is based on the Norwegian
articles from the newspaper VG. The translation of the XSum dataset was done using
the [NLLB model](https://doi.org/10.48550/arXiv.2207.04672).

The original full dataset consists of 472,000 samples, and we use a 1,024 / 256 / 2,048
split for training, validation and testing, respectively (so 3,328 samples used in
total).

Here are a few examples from the training split:

```json
{
  "text": "På Akvariet i Bergen har pingvinene fått et ekstra fristende sommertilbud denne uken. – Vi fikk en litt artig idé, og bestemte oss for å gi pingvinene våre en slags «slush-is» i går. Det ble til en morsom aktivisering for pingvinene, og det falt virkelig i god smak hos dem, sier dyrepasser Jannicke Johannessen. Hun forteller at de eldre pingvinene først var litt skeptiske, og at det var de yngste som ledet an i isleken. – Ett- og toåringene var veldig interesserte da vi kom ut med isen, og hoppet opp på den og storkoste seg. En av pingvinene ble faktisk liggende oppå isen helt til den smeltet, ler hun. Hun forteller at isen falt i så god smak, at de skal gjenta suksessen lørdag, slik at flere gjester i parken også kan få med seg aktiviteten.Selv om sommeren har satt flere varmerekorder i hele landet, forteller Johannessen at dyrene i Akvariet slettes ikke har lidd noen nød. – Vi har California-sjøløver, som overhodet ikke har hatt noen problemer med varmen. Tvert imot, de elsker å ligge å sole seg. Vi har også europeiske otere, som takler klimaet godt, da det er dyr man finner naturlig i sørlige deler av Europa. Dessuten er vi ekstremt heldige her på Akvariet, og pumper opp nytt saltvann hele tiden, og dyrene har mange muligheter til å kjøle seg ned på. Hun gir imidlertid et viktig råd til dyreeiere som vil kjøle ned dyrene sine: – Jeg har fått med meg at folk gir is som hundene kan spise for eksempel, og det er ikke akkurat et sjakktrekk. Når man kjøler ned dyrene fra innsiden samtidig som det er veldig varmt ute, tuller det med kroppstemperaturen. Kroppen jobber for å varme opp innsiden samtidig som de får varme utenfra. Du gir dem egentlig et heteslag, sier hun. – Det beste er å kjøle dem ned på utsiden. Dusj dem under «armhulene», eller generelt der de har tynn hud.Også i Tyskland har det vært høye temperaturer i sommer, og dyrepassere har måttet ta grep for å avkjøle dyrene i varmen. I Osnabrück, nord i landet, ble det registrert rundt 35 varmegrader onsdag. For tapirene i dyrehagen ble maten strategisk servert i skyggen, slik at dyrene ikke blir solbrent. Dyrepasser Daniel Chirico bestemte seg dessuten for å spyle tapirene med en hageslange, for å kjøle dem ned ytterligere. – Spesielt de nordiske artene i dyreparken har merket hetebølgen, og tilbringer mesteparten av dagen i skyggen, sier Tobias Klumpe, biolog i Osnabrück Zoo til den tyske avisen Osnabrücker Zeitung . Svartbjørnene tar mer enn gjerne en kald dukkert i sola, samtidig som de nyter kalde forfriskninger med frukt og bær.I Finland har også sommervarmen slått inn for fullt. I Korkeasaari Zoo i Helsinki ble det torsdag registrert 30 varmegrader. Løsningen har blant annet vært å installere en «regnskog» for kenguruene, mens papegøyene har fått egne dusjer de kan bruke. Bjørnene har fått iskald vannmelon, som de nyter i det kalde vannet, og tigerne får frosne kaniner – såfremt de faktisk ønsker å spise. – Appetitten deres blir mindre i varmen. For eksempel spiser hunnene i snitt bare annenhver dag, sier dyrepasser Jonne Stenroth til den finske avisen MTV . Ellers tilbringer tigrene mesteparten av dagen i skyggen mens de slapper av i bassenget, skriver avisen.",
  "target_text": "Mens solen skinner og temperaturene er som høyest, tar dyreparker rundt om i Europa i bruk kreative løsninger for å holde dyrene avkjølte."
}
```
```json
{
  "text": "Nick Corsellis, advokat for Carl Wood, sa at en \"innendørs mann\" må ha vært involvert i razzia, men hans klient manglet ekspertise til å være den personen. Mr Wood og tre andre menn nekter å ha deltatt i £ 14m røveriet. Fire andre har allerede erklært seg skyldig for deres roller i røveriet. \"Og dette er en av grunnene til at Mr. Wood ikke er skyldig. Hva tok han med seg til bordet?\" sa han. Mr. Corsellis sa at det ikke fulgte at hans klient var mannen som ble identifisert av anklagemyndigheten som \"Man F\" i CCTV-opptak av razzia. \"Male F var faktisk en spiller. En innsider, eller knyttet til innsiden, som var fullt kjent med det indre arbeidet i Hatton Garden Safe Deposit\". Mr. Wood manglet slik kunnskap og ville bare ha vært i stand til å fungere som en \"generell hundekrop\", sa advokaten. Corsellis spurte juryen om profesjonelle kriminelle ville vært forberedt på å gi opp en del av sine millioner til en person som bare ville ha vært et \"ekstrapar hender (EPH)\". Han kalte det \"ilogisk\" og \"utrolig\" at en slik person var involvert da \"kriminelle ikke er veldedig folk\". \"Men hvem ville spille Carl Wood - EPH? Tror du at Mr. Tom Hardy eller Mr. Vinnie Jones vil haste å ta rollen som... EPH?\" spurte han.",
  "target_text": "En av mennene som er anklaget for å være en del av Hatton Garden-raiden, kunne ikke ha vært involvert fordi han manglet noen ferdigheter å tilby gjengen, har en domstol hørt."
}
```
```json
{
  "text": "Verdenshjelpen forlot klubben i fjor på grunn av arbeids- og studietilbud, pluss behovet for å komme seg fra en ryggskade. Manager Jamie Sherwood sa til klubbens nettside: \"Jeg er virkelig glad for å ha brakt Natalie tilbake til klubben. \"Hennes erfaring, lederskap og åpenbare evne blir et utmerket tillegg til vår tropp for 2017\". Haigh la til: \"Etter skaden jeg fikk på ryggen for nesten 15 måneder siden, trodde jeg aldri at jeg ville spille igjen, enn si på dette nivået. \"Det er flott å være tilbake i og rundt klubben - det er en ekte buzz etter den suksessen de oppnådde i fjor\".",
  "target_text": "Yeovil Town Ladies har gjenforenet tidligere kaptein Natalie Haigh før damer Super League One klubbens første sesong i toppklassen."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:
  ```
  Her følger nyhetsartikler med tilhørende sammendrag.
  ```
- Base prompt template:
  ```
  Nyhetsartikkel: {text}
  Sammendrag: {target_text}
  ```
- Instruction-tuned prompt template:
  ```
  Nyhetsartikkel: {text}

  Skriv et sammendrag av den ovennevnte artikkelen.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset no-sammendrag
```


### Unofficial: NorGLM Multi Sum

This dataset was released in [this paper](https://doi.org/10.48550/arXiv.2312.01314) and
features a manually annotated summarisation dataset based on Norwegian news articles.

The original dataset contains 467 samples, which we split into 147 / 64 / 256 samples
for training, validation and test, respectively.

Here are a few examples from the training split:

```json
{
  "text": " En sel i England ble fanget i plast. Det kunne gått galt. Hver dag blir også dyr i Norge fanget i plast. Et vondt syn møtte nylig dyrevernere på en strand i England. Der lå en sel som hadde tuklet seg inn i plast. Det kunne gått veldig galt.– Det var tydelig at selen hadde det vondt, forteller en kvinne som så selen på stranden, til kanalen BBC.Men dyrlegene fra den britiske dyrevernsorganisasjonen BDMLR kom heldigvis i tide. De klarte å fri selen fra plasten. Selen ble sluppet tilbake i sjøen.Heldigvis ble ikke selen skadet denne gangen, forklarte dyrevernsorganisasjonen til BBC.Men mange dyr er ikke så heldige når de blir fanget i plast. Dyr setter seg fast i plast over hele verden. Norske sjødyr setter seg fast i plast hver eneste dag, forteller Per-Erik Schulze. Han jobber i Naturvernforbundet og er ekspert på plast og forurensing i havet. – Mange av dyrene står fast i mange dager eller måneder uten å slippe løs. Det er helt grusomt, sier Schulze.Han forteller at disse dyrene ofte setter seg fast i plast: SjøfuglerFiskSelerSmåhvalerHummerSkilpadderDet er også dyr på land som setter seg fast i plast, for eksempel sauer og reinsdyr. Hvert år havner over åtte millioner tonn plast i havet, ifølge Verdens naturfond (WWF). Det meste synker til havbunnen, resten skyller inn på strender eller flyter på havoverflaten.Det er farlig for dyr som lever i og rundt havet, fordi de kan sette seg fast i plasten eller få den i magen.Hva skjer med dyrene som setter seg fast i plast?– Det er det største dyreplageriet i verden. Det er veldig vondt å hekte seg fast. Mange dør kanskje ikke av plasten, men av sult, fordi de ikke kommer seg løs så de kan dra og spise, sier han.Derfor er det viktig ikke å kaste plast som forsøpler naturen, mener Schulze.– En fin tanke er at hver plastbit vi rydder opp, kanskje kan redde et dyr. For det finnes også en god nyhet: De siste årene har mange ryddet søppel i naturen og langs kysten i Norge. Har det hjulpet? – Ja, det har vært en kjempe-ryddedugnad i Norge de siste fem årene. Noen steder er det så rent nå at det er vanskelig å finne noe plast. Det er et godt tegn, sier Schulze.",
  "target_text": " En sel i England som var fanget i plast ble reddet av dyrevernere. Dette er en vanlig situasjon, både i Norge og andre steder i verden, da mange dyr setter seg fast og lider lenge fordi de ikke kan komme seg løs. Per-Erik Schulze, en ekspert fra Naturvernforbundet, oppfordrer folk til å fortsette ryddearbeidet for å minimere risikoen for dyr å komme til skade assosiert med plastforsøpling. Han bekrefter at ryddedugnadene i Norge har vært en suksess."
}
```
```json
{
  "text": " Det drar seg til mot sommer, ferietid, og ikke minst helg. Usikker på hva du skal vie den til? Her har du et lite knippe velmente tips.Denne guiden gjelder fra fredag 10. juni til søndag 12. juni.Fredag og lørdag er det duket for folkefest og musikkbonanza på Viking stadion i Jåttåvågen.Anledningen er to konserter fra det folkekjære Stavangerbandet Mods, som er tilbake igjen på arenaen hvor de i 2012 og i 2017 spilte foran flere titalls tusen elleville fans. Også Kvelertak er med på å innramme en meget sterk musikkhelg i regionen. På fredag går de nemlig opp på scenen på Folken i Stavanger, og skal by på de herligste toner med både hardrock og metall. Også i utelivets verden skjer det ting i helgen. Fredag kveld gjør et nytt nattklubb- og cocktailbar-konsept sitt inntog i Stavanger når LouLou åpner dørene i de gamle Hot-lokalene i Skagen. – Vi har sett at Stavanger manglet en annen og kanskje litt mer eksklusiv plass, hvor man kan feire bursdager og andre store begivenheter, sa daglig leder i Rekom, Frederik Mygind til Byas i forrige uke.Også på Show Bar, nysatsingen til duoen Dennis Poppe og Øyvind Sørensen, blir det åpning til helgen. «Ein liden (ein) pre-opening i morgen (lørdag) og søndag på Show Bar! Sees kl. 20:00», skriver Poppe på sin Instagram-konto. Etter seieren borte mot Sverige sist søndag, er det en revansjelysten «söta bror» som gjester Ullevaal kommende søndag. Flere rogalendinger figurerer i viktige roller på landslaget, med Erling Braut Haaland, Veton Berisha, Kristian Thorstvedt og Birger Meling som navnene. Kampen kan sees på flere utesteder i Stavanger, men kan også nytes fra sofaen fra klokken 20:45. I det Aftenbladet omtaler som «superdagene», med en hel rekke arrangementer den kommende uken, finner flere av de sted denne helgen. Det 91 kilometer lange sykkelløpet, Nordsjørittet, fra Egersund til Sandnes går av stabelen lørdag, og kan la svettekjertlene få fri utfoldelse. Rittet så dagens lys tilbake i 1998 og er et samarbeid mellom flere lokale sykkelklubber. Og på Sola blir det moro for både store og små når Sola Airshow 2022, flystevnet som har vist fram gamle og nye luftmaskiner i en årrekke, holdes på lørdagen og søndagen. Er du derimot mer opptatt av folkelivet, så kan enten Tanangerdagene, eller Solafestivalen være for deg. I Sola kulturhus er det på fredag og lørdag duket for ungdomsfestival.Arrangementet er gratis, for de mellom 13 og 20 år, og byr blant annet på musikk fra den norske rapperen Hkeem, samt Stavanger-bandet Kriminell Kunst. Og et lite stykke unna, fra onsdag denne uken og fram til og med søndag, blir det folkeliv i Tananger, når Tanagerdagene går av stabelen. Arrangementet holdes i regi av Lions Club Tananger, og lover fem dager fulle av aktiviteter for familier, barn, ungdom og voksne. – Her er noe for alle og mye for mange. Hjertelig velkommen, skriver arrangøren på Facebook-arrangementet sitt. Fra 10. til 12. juni holder fem kunstnere pop up-utstilling i Pedersgata.Kunstnerne det er snakk om er ragnhild.kristine, pryl.art, hwks.art, corneliussen.art og Rosa Ottestad.Det hele finner sted i Pedersgata 43, og det er ventet flere besøkende til arrangementet. Utstillingen åpner kl. 18 på fredag, og holder åpent gjennom helga. Vet du bedre enn oss hva skjer neste helg? Send en e-post til helga@byas.no!",
  "target_text": " Artikkelen handler om hvilke arrangementer som skal holdes i perioden fra 10. juni til 12. juni. Blant arrangementene er konserter med bandene Mods og Kvelertak, landskamp i fotball på Ullevaal, og flystevnet Sola Airshow 2022 på Sola der det skal vises fram gamle og nye luftmaskiner. I tillegg arrangeres Tanangerdagene og Solafestivalen."
}
```
```json
{
  "text": " Regjeringen foreslår å åpne nye områder for oppdrettsnæringen, men med strenge miljøkrav. – Gir betydelige muligheter for å øke produksjonen, sier fiskeriministeren.Nærings- og fiskeridepartementet foreslår nå en ny tillatelsesordning for oppdrett med miljøkrav.Det første året kan det tildeles tillatelser på maksimalt 15.000 tonn biomasse (fisk). Hver enkelt søker kan maksimalt få tildelt ti tillatelser, og det vil stilles strenge miljøkrav til søkerne, heter det i meldingen fra departementet.– Dagens produksjon i åpne merder vil fortsatt være grunnstammen i norsk oppdrett. I tillegg har vi lagt til rette for landbasert oppdrett og havbruk til havs. Med denne ordningen peker vi ut en ny retning som gir oppdrettsnæringen mulighet til å ta i bruk nye arealer langs kysten, sier fiskeri- og sjømatminister Odd Emil Ingebrigtsen (H).Til sammenligning ble det produsert rundt 1,4 millioner tonn laks i Norge i 2019, ifølge SSB.Tillatelsene i den nye miljøteknologiordningen kommer i tillegg til veksten som blir tilbudt på ordinær måte gjennom trafikklyssystemet.– Samlet sett gir dette norsk havbruksnæring betydelige muligheter for å øke produksjonen fremover, sier ministeren.Forslaget innebærer følgende miljøkrav: Null utslipp av egg og frittsvømmende stadier av lakselus, minimum 60 prosent oppsamling av slam, samt krav til rømningssikkerhet.Prisen for tillatelsene vil bli satt med utgangspunkt i auksjonsprisene som er oppnådd i forbindelse med ordinære kapasitetsjusteringer, men med et rimelig fradrag.– Havbruksnæringen skaper store verdier for Norge. Men videre vekst må skje innenfor bærekraftige rammer. Hensynet til natur generelt, og villaksen spesielt, er av avgjørende betydning, sier klima- og miljøminister Sveinung Rotevatn (V).Til tross for bedring på viktige områder, er antallet norsk laks i havet mer enn halvert siden 1980-tallet, ifølge Vitenskapelig råd for lakseforvaltning.Det er flere grunner til det, også overfiske, men rådet slår fast at rømt oppdrettslaks og lakselus nå er de største truslene mot villaks.Forslaget skal på kort tid ut på høring.E24 skrev tidligere at siste sitat i saken var fra Ingebrigtsen, mens det egentlig var fra Rotevatn. E24 beklager og har nå rettet feilen.",
  "target_text": " Regjeringen foreslår en ny tillatelsesordning for oppdrett med strenge miljøkrav for å muliggjøre bærekraftig vekst i havbruksnæringen. Denne ordningen vil åpne nye områder for oppdrett, tillate hver søker å få maksimalt ti tillatelser, og krever null utslipp av egg og frittsvømmende stadier av lakselus, minimum 60 prosent oppsamling av slam, samt krav til rømningssikkerhet. Dette skal gi næringen mulighet til å øke produksjonen på bærekraftig måte."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:
  ```
  Her følger nyhetsartikler med tilhørende sammendrag.
  ```
- Base prompt template:
  ```
  Nyhetsartikkel: {text}
  Sammendrag: {target_text}
  ```
- Instruction-tuned prompt template:
  ```
  Nyhetsartikkel: {text}

  Skriv et sammendrag av den ovennevnte artikkelen.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset norglm-multi-sum
```


### Unofficial: Schibsted-no

This dataset was released
[here](https://huggingface.co/datasets/Schibsted/schibsted-article-summaries) and
features summaries of news articles from Schibsted Medias Norwegian newsrooms.

The original dataset contains 1,240 / 347 / 374 samples for training, validation and
testing, respectively. We use these splits as-is.

Here are a few examples from the training split:

```json
{
  "text": "Klubblegenden med innrømmelse under VAR-debatten: – Vanskelig å stå her : VAR-opprøret tok en knusende seier i Trondheim. Til og med styremedlem Ola By Rise måtte innrømme at det var mange gode argumenter imot videodømmingen.  Den gamle keeperhelten talte RBK-styrets sak for VAR sammen med medstyremedlem Tore Reginiussen:  – Det er en veldig vanskelig sak. Det er ikke to VAR-tilhengere som står her, sa en engasjert By Rise fra talerstolen.  VAR-debatten hadde kommet til Rosenborgs medlemmer torsdag, som skulle stemme for at Rosenborg aktivt skulle arbeide for å fjerne VAR eller ikke.  489 stemte for å avvikle VAR. 157 stemte for å beholde VAR. Stemmene ble lest opp til enorm applaus fra salen.  Forslaget om at RBK-styret skulle få «utrede ulike modeller for å få kapital inn i klubben» ble også stemt ned med god margin. – Medlemmene har definitivt makta i Rosenborg og de bruker den. Dette er et gedigent nederlag for det sittende styret og leder Cecilie Gotaas Johnsen, sier Adresseavisens kommentator Birger Løfaldli til VG.  – Særlig investorsaken tror jeg er tung å svelge, der det foreløpig kun var snakk om en utredning. Jeg er spent på hvordan Gotaas Johnsen vil reagere på dette og hvordan hun vurderer arbeidsbetingelsene det kommende året, sier Løfaldli.  VAR-debatten var den som tok lengst tid:  – Jeg har forståelse for klubbens posisjon og forstår at måten oppleves som uvanlig detaljstyrende. Men for mange er dette en ekstraordinær sak. Det er viktig at styret forstår: VAR må ikke forbedres, VAR må fjernes! sa forslagsstiller Ole Christian Gullvåg.  – Talelista begynner å bli lang, var meldingen fra ordstyrer etter at et par stykker hadde snakket sin side i VAR-saken.  Styremedlem By Rise argumenterte med at det ville bli vanskelig å «sette tannkremen tilbake på tuben». Forslagsstiller Gullvåg svarte:  – For oss oppleves det som at noen har sprøytet tannkrem på stua midt under fredagstacoen. Vi har ikke bedt om det, vil ikke ha det.  Ola By Rise har tidligere vært ute på Twitter og vært kritisk til VAR. Han innrømmet også sin tvil rundt temaet.  – Det er vanskelig å stå her. Man må ikke stå hver kamp på Øvre Øst for å reagere på hvordan VAR praktiseres i dag. Så er det ikke sikkert den blir god nok. Involveringen av supporterne burde definitivt blitt bedre. Men det er ikke sikkert det er verktøyet som er problemet, men gjennomføringen, sa By Rise.  Han og Reginiussen listet opp både negative og positive sider ved VAR, og pekte som flere andre klubber på det potensielle økonomiske tapet ved å fjerne VAR.  Styret argumenterte for at Rosenborg skulle være en kritisk meningsbærer rundt videodømming. Et titalls medlemmer tok ordet og sa seg svært uenige, og til slutt var det forslaget fra medlemmene som vant frem.  RBK-medlem Emil Almås var forslagsstiller sammen med Gullvårg. Han sier følgende til VG: – Det vi har fått til i norsk toppfotball de siste dagene er en seier for fotballen og en seier for medlemsdemokratiet. Ved å takke nei til VAR, har norske supportere startet et jordskred, som kommer til å rase gjennom fotballeuropa i årene som kommer! Den dagen VAR er historie, skal jeg med stolthet si at jeg, og mange andre norske fotballsupportere var med på å trille de første steinene nedover dalsiden, sier Almås.  PS. En rørt Rune Bratseth mottok tittelen som æresmedlem i Rosenborg, etter en lang karriere som spiller, sportssjef og styremedlem. - Det er veldig spesielt for meg, sa Bratseth. ",
  "target_text": "489 RBK-medlemmer stemte for å avvikle VAR ved et møte torsdag, med 157 mot Styremedlem Ola By Rise innrømmet gode argumenter mot videodømming, men argumenterte for at Rosenborg skulle være en kritisk stemme imot. RBK-medlem Emil Almås hevder \"norske supportere starter et jordskred\" mot VAR i Europa Medlemmene ga også sitt nei til at RBK-styret skulle få «utrede ulike modeller for å få kapital inn i klubben».  – Et gedigent nederlag for det sittende styret, mener Adresseavisens kommentator Birger Løfaldli "
}
```
```json
{
  "text": "Gazas befolkning sultes med vilje, sier FN-ekspert: Krigen har ødelagt matproduksjonen. Samtidig slippes det ikke inn nok nødhjelp. Israel driver en aktiv politikk for å sulte ut Gazas befolkning, mener FNs spesialrapportør. Israel har som mål å begrense Gazas sivilbefolkning tilgang til mat. Det hevder FNs spesialrapportør for retten til mat, Michael Fakhri, til The Guardian. – Det finnes ingen grunn til å med vilje stoppe leveringen av humanitær hjelp eller ødelegger små fiskebåter, drivhus og fruktåkere, bortsett fra å nekte folk tilgang til mat, sier Fakhri til den britiske avisen. Han mener at Israel med dette gjør seg skyldig i både krigsforbrytelser og folkemord. Jan Egeland: – Fullstendig galskap Sentrale israelske politikere er flere ganger blitt anklaget for å ha brukt retorikk som oppfordrer til folkemord. Dette ble blant annet lagt til grunn da Sør-Afrika klaget Israel inn til ICJ. – Som en menneskerettighetsekspert ved FN mener jeg at dette nå er en folkemord-situasjon, understreker Fakhri. Fakhri er ikke den eneste som har advart om konsekvensene av hungersnøden i Gaza. En FN-rapport konkluderte nylig: Flyktninghjelpens generalsekretær, Jan Egeland, reiste tirsdag inn i Gaza. Han beskriver rystende scener med desperate mennesker som gjør alt i sin makt for å kare til seg mat. – Jeg er fullstendig sjokkert over forholdene her. Folk slåss som ville og gale over madrasser og sekker med mat, sier Egeland til VG. – Det er fullstendig galskap at verden har latt en befolkning bestående av stort sett helt uskyldige kvinner og barn bli utsatt for bombardement og utsulting siden midten av oktober. Hevder Israel trosser FN-domstol Situasjonen er ikke blitt bedre de siste ukene. Det sier bistandsorganisasjoner. Det til tross for at Den internasjonale domstolen (ICJ), FNs viktigste domstol, for én måned siden bestemte at Israel må gjøre alt i sin makt for å sørge for å stoppe et folkemord og sørge for at palestinere har tilgang til bistand. Human Rights Watch (HRW) og Amnesty International påpeker at det slippes inn 30 prosent færre lastebiler med nødhjelp hver dag nå sammenlignet med før ICJs pålegg 26. januar. I februar slapp det inn halvparten så mye nødhjelp i Gaza som måneden før, ifølge FNs organisasjon for palestinske flyktninger (Unrwa). – Den israelske regjeringen sulter 2,4 millioner palestinere i Gaza.  Det sier Omar Shakir, som er lederen for HRWs virksomhet i Israel og Palestina. – Den israelske regjeringen har ganske enkelt oversett domstolens pålegg, føyer han til. Tirsdag redegjorde Ramesh Rajasingham ved FNs kontor for koordinering av humanitær innsats (UNOCHA) om situasjonen for FNs sikkerhetsråd. Han advarte om at jordbruket i Gaza vil kollapse innen mai hvis situasjonen ikke blir bedre, og hvis det ikke blir pause i krigshandlingene. – Vi understreker derfor nok en gang vårt krav om en våpenhvile, sa han. USA blokkerte i februar enda en gang en resolusjon i Sikkerhetsrådet om våpenhvile. Begrunnelsen var at resolusjonen kunne ødelegge forhandlinger om våpenhvile og fangeutveksling som pågår mellom Egypt, Israel og Qatar. – Hvis ingenting skjer, frykter vi at storskala sult i Gaza nesten er uunngåelig, og det vil føre til mange flere ofre, sa Rajasingham til Sikkerhetsrådet.",
  "target_text": "FN mener Israel prøver å sulte ut befolkningen på Gazastripen. Målrettede angrep hindrer matproduksjon og levering av nødhjelp.  Akutt underernæring truer hele befolkningen. Barn og kvinner i Nord-Gaza og Rafah er mest utsatt.  Israel overser FN-domstolens pålegg om å gi palestinere tilgang til bistand. Hjelpeorganisasjoner ser mindre nødhjelp komme inn."
}
```
```json
{
  "text": "Marokkanske og albanske mafianettverk dominerer. Svenskene blir en stadig større trussel.: Flere er bygd på lojalitet til familie og klan, ifølge ny rapport fra Kripos. Om kort tid legger politiet frem sin trusselvurdering. Der vil Politi-Norge peke på de største truslene mot det norske samfunnet. En av truslene som vil bli viet mye plass, er organiserte kriminelle nettverk. I Norge er det rundt hundre slike nettverk. Kripos mener politiet har kapasitet til å følge med på 40 av dem. Nettverkene smugler og selger enorme mengder narkotika. De står bak skyteepisoder, eksplosjoner, menneskesmugling og bedragerier. Målet er profitt. Midlene er vold og hard indre justis. Noen av de mektigste nettverkene er bygd på lojalitet til familie og klan. Nå letter Kripos på sløret. For første gang går politiet ut med en egen rapport om nettverkene som dominerer i den kriminelle underverdenen: I rapporten trekker Kripos frem fem store trusler: 1. Marokkanske narkonettverk En av de aller største truslene er marokkanske narkonettverk. – De er utrolig sentrale, ikke bare i Norge og Norden, sier Eivind Borge fra Kripos. Norskmarokkanere dukker også opp i etterforskninger i andre europeiske land. Aftenposten har tidligere omtalt Zakariya Rahali, som har vært på rømmen siden 2017. Rahali er pekt ut som lederen av Norges største narkonettverk. 2. Albanske narkonettverk Etter marokkanerne, er det albanske nettverk som utgjør den største trusselen. Disse regnes for å være blant de største nettverkene som driver med kokain i hele Europa.  3. Svenske narkonettverk Borges skrekkscenario er at Norge kommer dit Sverige er i dag. Der har gjengkrigen herjet og deler av samfunnet er i ferd med å bli infiltrert av kriminelle. I Norge har samtlige politidistrikt støtt på svenske kriminelle nettverk. Og trusselen er økende, vurderer Kripos. 4. Litauiske kriminelle nettverk For å frakte narkotika, trengs det logistikk. For å gjøre dette, tar mange kriminelle i bruk litauiske nettverk.  5. Norge som transittland I fjor opplevde Europa en «kokaintsunami». Enorme mengder kokain ble tatt av politi og tollere, også i Norge. Men prisene gikk ikke opp. Et tegn på at store mengder kokain er i omløp.  I flere år har havnene i Rotterdam og Antwerpen vært stedet hvor kokain er blitt smuglet inn til Europa. Men der har myndighetene kastet seg rundt. Dermed må de kriminelle se seg om etter nye havner for å få det hvite pulveret til kundene. De store beslagene i fjor, kan peke mot at Norge i større grad er i ferd med å bli et av disse stedene. Enn så lenge er det for tidlig å konkludere om Norge er blitt en del av kokainruten til Europa, mener Borge og Ole Jørgen Arvesen, avdelingsleder med ansvar for etterretning i Kripos. Går sammen med kartellene Hvordan kan Kripos være så sikre i sin sak? Mye kommer fra pågående etterforskninger, men de siste årene har de også fått et unikt innblikk i hvordan de kriminelle jobber og samarbeider. De har fått meldinger og bilder fra Encrochat, Sky ECC og Anom. Det har ledet til flere store saker, men likevel er trusselen fra de kriminelle nettverkene blitt større. – Den er betydelig og økende for hele Europa, også Norge, sier Arvesen. Nettverkene er blitt mer profesjonelle og samarbeider mer med kriminelle i andre land.  – Vi ser tydelig at norske nettverk har direkte kontakt med karteller i Sør-Amerika, sier Eivind Borge fra Kripos. Han sier bakmennene de jobber for å ta, ikke lar seg stoppe med forebygging. Det krever mye etterforskning og samarbeid med politi i andre land.",
  "target_text": "For første gang går politiet ut med en egen rapport om kriminelle nettverk. Rapporten peker på fem store trusler: marokkanske og albanske narkonettverk, svenske narkonettverk, litauiske kriminelle nettverk og at Norge blir et transittland for kokain. Nettverkene i Norge er blitt mer profesjonelle, har direkte kontakt med karteller i Sør-Amerika. Dette krever mer etterforskning og internasjonalt samarbeid."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:
  ```
  Her følger nyhetsartikler med tilhørende sammendrag.
  ```
- Base prompt template:
  ```
  Nyhetsartikkel: {text}
  Sammendrag: {target_text}
  ```
- Instruction-tuned prompt template:
  ```
  Nyhetsartikkel: {text}

  Skriv et sammendrag av den ovennevnte artikkelen.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset schibsted-no
```


### Unofficial: Personal Sum

This dataset was released [here](https://github.com/SmartmediaAI/PersonalSum) and contains human annotated summaries that reflect individual user preferences.

The original dataset contains 1,099 summaries based on 441 unique articles. The dataset has been restructured into 441 samples, where each sample represents a unique article paired with all of its corresponding summaries (1 or more). The dataset has been split such that we have 121 / 64 / 256 samples for training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
    "text": "I en ny bok forteller Abid Rajas søster Abida Raja (49) at hun over lengre tid levde i et voldelig forhold. I en pressemelding avviser eksmannen anklagene. – Min klient ønsker å påpeke at han nekter straffeskyld for partnervold og\nvoldtektsanklager. Han vedkjenner at ekteskapet har hatt sine utfordringer, og at de derfor skilte seg i 2015, skriver eksmannens advokat Javeed H. Shah i en pressemelding. I boken «Frihetens Øyeblikk», beskriver Raja at eksmannen hennes var voldelig, og at hun flere ganger forsøkte å unnslippe mannen. I boken skriver forfatter Håkon F. Høydal:«De siste tjue årene hadde vært en kamp mot seg selv: Hun ønsket å gå fra mannen. Men hun måtte bli. På grunn av barna, og på grunn av familien, på grunn av frykten for fattigdom og skam. Nå hadde hun verken barna, penger eller hus.»VG har tidligere vært i kontakt med Abida Rajas eksmann i forbindelse med bokutgivelsen, som tirsdag ikke hadde lest boken.– Jeg er i utlandet og har ikke lest boken, så kan ikke kommentere uten å lese det, skriver han i en SMS til VG.I boka skriver forfatteren at Abida etter stort press fra familien, skal ha møtt én av ektemannkandidatene, en 23 år gammel inngiftet onkel i Pakistan. Hun var 18 år og skulle gått i andre klasse på videregående hjemme i Norge.«Abida husker ikke om hun sa ja. Men hun sa heller ikke nei. Hun ville bare bort», heter det i boken.Onsdag svarer eksmannen via sin advokat, at han har levd i god tro om at Abida giftet seg av fri vilje slik hun selv uttrykte ovenfor han. – Derfor er opplysningene om tvangsekteskap noe han ble kjent med først i 2020. Boken kommer ett år etter at venstrepolitiker og tidligere statsråd Abid Raja kom med sin bok\xa0«Min skyld». Boken er skrevet av VG-journalist Håkon F. Høydal og ble lansert tirsdag morgen\xa0etter mye hemmelighold. VG har ikke hatt noe med utgivelsen å gjøre.",
    "target_text": ["I en ny bok forteller Abid Rajas søster Abida Raja om hennes erfaringer med et voldelig ekteskap, hvor hun beskriver flere forsøk på å unnslippe. Eksmannen avviser anklagene og hevder at han levde i god tro om at ekteskapet var av fri vilje, noe han først ble klar over i 2020.",
    "Abida Raja beskriver i en ny bok et voldelig forhold med sin eksmann, som avviser anklagene om partnervold og voldtektsanklager. Boken avslører også at Abida ble presset til å møte en ektemannkandidat i en tvangssituasjon, noe eksmannen hevder han ikke var klar over før i 2020.",
    "I boken «Frihetens øyeblikk» forteller forfatteren Håkon F. Høydal at Rajas eksmann var voldelig og hun ønsket å forlate ham. Hun ble værende fordi hun var redd for barnas lidelser, redd for fattigdom og hun skammet seg."]
}
```
```json
{
    "text": "Flere lakseaksjer falt igjen tungt, dagen etter at skatteforslag ga børsras for sjømatselskaper. Samtidig steg Norwegian etter anbefaling fra storbank.Det Ble en noe vinglete dag på Oslo Børs torsdag.Etter en positiv start vendte Børsen snuten nedover i tidlig handel, før den hentet seg inn igjen til forsiktig oppgang omtrent halvveis ut i handelsdagen. Utover ettermiddagen snudde Børsen så nedover igjen.Hovedindeksen endte til slutt dagen ned 1,58 prosent.Nedgangen tiltok den siste timen med handel, samtidig som Wall Street falt kraftig.Oljeprisen steg solid gjennom gårsdagen, og handles rundt én dollar høyere enn da Børsen stengte onsdag. Et fat Nordsjøolje (brent spot) koster ved stengetid torsdag 88,4 dollar, ned rundt 0,9 prosentsiden midnatt.Oljeselskapene Equinor og Aker BP falt i overkant av én prosent, mens Vår Energi endte ned 3,82 prosent.Onsdag falt Hovedindeksen 2,76 prosent etter at lakseselskapene fikk gjennomgå etter regjeringens foreslåtte grunnrenteskatt på havbruk. Verst gikk det for Salmar som stupte 30 prosent, samtidig som Lerøy Seafood falt 27,5 prosent. Torsdag fortsetter nedgangen for lakseaksjene. Sjømatindeksen endte ned 5,05 prosent.Slik så det ut for lakseaksjene ved stengetid (utvikling onsdag i parentes): Salmar falt 1,05 prosent (stupte 30,3 prosent)Grieg Seafood falt 2,75 prosent (falt 26,6 prosent)Mowi falt 3,15 prosent (falt 18,9 prosent) Lerøy Seafood falt 8,10 prosent (raste 27,5 prosent)Austevoll Seafood falt 6,28 prosent (falt 21,7 prosentNorway Royal Salmon falt 8,94 prosent (endte ned 22,9 prosent)Bakkafrost-aksjen falt samtidig 12,83 prosent.Selskapet har virksomhet på Færøyene og understreket onsdag at de ikke påvirkes av det nye norske skatteforslaget. Samtidig understreket de at det arbeides med et forslag om justeringer av skattesatsen på Færøyene.I USA peker pilene solid nedover på børsene torsdag ettermiddag.Det er kraftig nedgang på Wall Street, der den brede S&P 500-indeksen faller godt over to prosent. Teknologiindeksen Nasdaq faller samtidig mer enn tre prosent.I Europa er det også bred, kraftig nedgang på de viktigste børsene. London-børsen, Frankfurt-børsen og Paris-børsen er alle ned i overkant av to prosent rundt stengetid i Oslo.Storbanken HSBC har gjenopptatt dekning på flyselskapet Norwegian, ifølge Bloomberg. Banken anbefaler kjøp og har satt et kursmål på 14,50 kroner. Dermed ser banken for seg en oppside på hele 119 prosent i aksjen, skriver nyhetsbyrået. Norwegian-aksjen steg 6,81 prosent.– Nye Norwegian er en annen forretning enn den før pandemien, som har omstrukturert operasjonelt og økonomisk, skriver HSBC i analysen.– Den nye ledelsen har en solid strategi, en enkel og kostnadseffektiv\nforretningsmodell med en enkelt type fly, et sterkt fokus på sine nøkkelmarkeder i Norden og en solid balanse og likviditet, alt innenfor et gunstig konkurranselandskap som bør tillate ny NAS å ta markedsandeler fra sine konkurrenter, heter det videre i analysen.Storbanken begrunner også sin nye dekning på flyselskapet ved at dets konkurrenter venter mye motvind og ny etterspørsel for Norwegian kan komme ut av det. I tillegg nevnes Norges sikkerhetsnett rundt høye energi- og strømpriser.- Mens Europa står overfor høy inflasjon og lav forbrukertillit, har Norge betydelig lysere utsikter med sine omfattende energiressurser, statlig finansiering og høy inntekt per innbygger.HSBC viser også til høy reiseetterspørsel blant nordmenn.Fornybarselskapet Scatec er i fokus i forbindelse med at selskapet har kommet med nye målsetninger. Selskapet vil investere 10 milliarder kroner av egenkapitalen i nye kraftverk frem mot 2027. Investeringene har som mål å utvide kapasiteten med 1,5 gigawatt hvert år i perioden. Scatec-aksjen endte dagen ned 2,93 prosentXXL er samtidig blant børstaperne torsdag. Aksjen til sportsbutikk-kjeden falt 11,66 prosent.",
    "target_text": ["Lakseaksjer opplever fortsatt betydelig nedgang på Oslo Børs etter regjeringens foreslåtte grunnrenteskatt på havbruk. Hovedindeksen endte ned 1,58 prosent, og sjømatindeksen falt ytterligere 5,05 prosent. Samtidig steg Norwegian-aksjen etter anbefaling fra HSBC, som gjenopptok dekning på selskapet og anbefalte kjøp med et kursmål på 14,50 kroner, med en forventet oppside på 119 prosent."]
}
```
```json
{
    "text": "(Minnesota Wild – St. Louis Blues 4–6) Mats Zuccarello (34) var svært kritisk til seg selv og lagkameratene i Minnesota Wild etter nattens tap mot St. Louis Blues i 23 minusgrader foran 38.000 tilskuere.– Jeg har egentlig ikke ord. Det er pinlig når du har 40.000 mennesker som kommer og fryser ræva av seg, og så spiller vi sånn, sa Zuccarello på pressekonferansen etter «Winter Classic»-oppgjøret på Target Field – et baseballstadion i Minneapolis. Før siste periode ledet Blues 6–2, og Zuccarello beskriver de to første periodene som at de ble «lett utspilt» av Blues. Zuccarello hadde én assist – da Ryan Hartman scoret lagets tredje mål . Wild reduserte to ganger i siste periode og fastsatte sluttresultatet til 4–6. 34-åringen mener det ikke nytter å forklare tapet med kulden, vanskelige forhold og det faktum at de ikke har spilt kamp siden 20. desember: – Det er ingen unnskyldninger ... Det er kaldt for begge lag, isen er humpete for begge lag. Vi spilte ikke smart hockey som vi har gjort i store deler av sesongen. Det var Wilds femte strake tap i en sesong der Zuccarello og laget jevnt over har levert meget bra. – Dessverre skjedde det på en stor kveld som dette. Folk forlater hjemmene sine i kulden for å støtte oss, og så serverer vi dem dette. Vi har skuffet oss selv og alle andre. Det var på forhånd varslet sprengkulde, og målingene viste 23 minusgrader. Zuccarello beskriver opplevelsen slik:– Jeg var skikkelig kald under oppvarmingen, men når kampen starter slår adrenalinet inn. Men jeg tror aldri jeg har vært så kald i hele mitt liv før sisteperioden da vi lå under 6–2, eller hva det var. Det var ingen god følelse. – Det store bildet nå er at vi har fem strake tap, og vi må finne tilbake til måten å vinne på og hvordan vi skal spille som et lag, sier Zuccarello. Zuccarello har scoret åtte mål og lagt 17 målgivende pasninger i løpet av 25 kamper denne sesongen. Det vil si ett målpoeng per kamp i snitt. I sine beste målpoengsesonger for New York Rangers – 2013/14, 2015/16 og 2016/17 – oppnådde han henholdsvis 59 målpoeng på 77 kamper, 61 målpoeng på 81 kamper og 59 på 80 kamper.PS! Natt til fredag spiller Minnesota Wild borte mot Boston Bruins. To dager senere er det hjemmekamp mot Washington Capitals.",
    "target_text": ["Minnesota Wild led et nederlag mot St. Louis Blues under ekstreme værforhold på Target Field. Mats Zuccarello uttrykte sin skuffelse over lagets ytelse foran 38 000 tilskuere, og tilskrev tapet til dårlig spill heller enn kulden. Til tross for Zuccarellos bidrag med en assist, endte Wild med sitt femte strake tap, noe som førte til et press for å finne tilbake til seiersformen før kommende kamper mot Boston Bruins og Washington Capitals.",
    "Det er ingen unnskyldninger for Wilds femte strake tap, til tross for at både Zuccarello og resten av laget generelt har spilt bra denne sesongen. Forholdene var like for begge lag, men laget spilte ikke smart hockey slik de har gjort tidligere i sesongen."]
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:
  ```
  Her følger nyhetsartikler med tilhørende sammendrag.
  ```
- Base prompt template:
  ```
  Nyhetsartikkel: {text}
  Sammendrag: {target_text}
  ```
- Instruction-tuned prompt template:
  ```
  Nyhetsartikkel: {text}

  Skriv et sammendrag av den ovennevnte artikkelen.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset personal-sum
```
