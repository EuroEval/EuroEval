# 🇭🇷 Croatian

This is an overview of all the datasets used in the Croatian part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### MMS-hr

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2306.07902).
The corpus consists of 79 manually selected datasets from over 350 datasets reported in
the scientific literature based on strict quality criteria.

The original dataset contains a single split with 77,594 Croatian samples. We use 1,024
/ 256 / 2,048 samples for our training, validation, and test splits, respectively. We
have employed stratified sampling based on the label column from the original dataset to
ensure balanced splits.

Here are a few examples from the training split:

```json
{
  "text": "ali kako mozete biti ovako trijezni u ovo doba ajde molim vas",
  "label": "negative"
}
```

```json
{
  "text": "RT @bsimun: Thompson okupio 100 000 ljudi u Čavoglavama. Sad će valjda platiti porez. #domoljublje #DanPobjede",
  "label": "neutral"
}
```

```json
{
  "text": "\n Šesti \"El Clásico\" za\n \n  Luku Modrića\n \n bio je i najdraži. Real je dobio Barçu 3-1, a hrvatski veznjak bio je jedan od najboljih igrača \"kraljeva\".\n\n\n\n - Otkako sam u Madridu, meni je to djelovalo kao\n \n  najuvjerljivija demonstracija moći\n \n . Barça je izgledala manje moćno jer je Real odigrao impresivno. Meni ta pobjeda više govori o snazi naše momčadi, o potvrdi kako forma koju iskazujemo već osam-devet utakmica nije slučajna - rekao je Luka za\n \n  SN\n \n .\n\n\n - Imali su psihološku prednost zbog stanja na ljestvici i manjeg imperativa. Zato je\n \n  Realov uspjeh impresivan\n \n , tim prije što smo gubili 0-1 - dodao je.\n\n\n\n  Izvorni članak pročitajte u\n  \n   Sportskim novostima\n  \n  .\n \n\n\n Pohvalio suigrače\n  \n\n\n -\n \n  Čudesna utakmica\n \n cijele momčadi i pobjeda protiv Barcelone. Ajmo, halá Madrid! - napisao je Modrić na društvenim mrežama.\n  \n",
  "label": "positive"
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
euroeval --model <model-id> --dataset mms-hr
```

## Named Entity Recognition

### WikiANN-hr

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
    "tokens": array(["Ubrzo", "su", "uslijedile", "narudžbe", "iz", "cijele", "Britanske", "zajednice", "naroda", "."], dtype=object),
    "labels": ["O", "O", "O", "O", "O", "O", "B-ORG", "I-ORG", "I-ORG", "O"]
}
```

```json
{
    "tokens": array(["``", "(", "Cole", "Porter", ")"], dtype=object),
    "labels": ["O", "O", "B-PER", "I-PER", "O"]
}
```

```json
{
    "tokens": array(["'", "''", "La", "Liga", "2009.", "/", "10", "."], dtype=object),
    "labels": ["O", "O", "B-ORG", "I-ORG", "O", "O", "O", "O"]
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Sljedeće su rečenice i JSON rječnici s imenicama koje se pojavljuju u rečenicama.
  ```

- Base prompt template:

  ```text
  Rečenica: {text}
  Imenovane entiteti: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Rečenica: {text}

  Identificirajte imenovane entitete u rečenici. Prikažite ih kao JSON rječnik s ključevima 'osoba', 'mjesto', 'organizacija' i 'razno'. Vrijednosti trebaju biti popisi imenovanih entiteta navedenog tipa, točno kako se pojavljuju u rečenici.
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
euroeval --model <model-id> --dataset wikiann-hr
```

## Linguistic Acceptability

### ScaLA-hr

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the
[Croatian Universal Dependencies treebank](https://github.com/UniversalDependencies/UD_Croatian-SET)
by assuming that the documents in the treebank are correct, and corrupting the samples
to create grammatically incorrect samples. The corruptions were done by either removing
a word from a sentence, or by swapping two neighbouring words in a sentence. To ensure
that this does indeed break the grammaticality of the sentence, a set of rules were used
on the part-of-speech tags of the words in the sentence.

The original full dataset consists of 1,024 / 256 / 2,048 samples for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
used as-is in the framework.

Here are a few examples from the training split:

```json
{
  "text": "Nakon kratke intervencije, tijekom koje sam saznala kada se taj osjećaj prvog puta pojavio i zbog čega, sve je nestalo i već mjesecima živim bez opterećenja koji me pratilo cijelog života.",
  "label": "correct"
}
```

```json
{
  "text": "Svaki od tih sklopova, i dijelova mora biti homologiran i sukladan s ostalima.",
  "label": "incorrect"
}
```

```json
{
  "text": "Prvi među njima je Laurent Blanc, koji drži Romu na čekanju, a s Parkom prinčeva povezivan je i Fabio Capello.",
  "label": "correct"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Sljedeće su rečenice i jesu li gramatički ispravne.
  ```

- Base prompt template:

  ```text
  Rečenica: {text}
  Gramatički ispravna: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Rečenica: {text}

  Odredite je li rečenica gramatički ispravna ili ne. Odgovorite s 'da' ako je ispravna, i s 'ne' ako nije. Odgovorite samo tom riječju i ničim drugim.
  ```

- Label mapping:
  - `correct` ➡️ `da`
  - `incorrect` ➡️ `ne`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-hr
```

## Reading Comprehension

### MultiWikiQA-hr

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
    "context": "Arkadija je pokrajina u središnjem dijelu Peloponeza, Grčka.\n\nOsnovni podaci\nGlavni grad Arkadije je Tripoli; populacija pokrajine je 100 611 (podatci iz 2005.), na 38. mjestu u Grčkoj; Površina joj je 4419 km² što je čini 5. po veličini; Gustoća naseljenosti je 22,8/km²; sastoji se od 4 provincije, 22 općine i 1 županije (okruga); poštanski broj je 22, registracijske pločice s oznakom TP; službena web stranica je www.arcadia.gr.\n\nOpćine\n\nPovijest\n\nGradska naselja u Arkadiji su se razvila razmjerno kasno (Mantineja, Orhomen, Tegeja). Bili su saveznici Sparte do sloma njezine hegemonije (371. pr. Kr.), otada tvore samostalan savez pod vodstvom novoosnovanog polisa Megalopola. Samostalnost saveza dokrajčili su Makedonci. U 3. st. pr. Kr. dio gradova u Arkadiji pristupa Ahajskom, a dio Etolskom savezu. Pod rimskom vlašću od 168. pr. Kr.\n\nSimbolika Arkadije\n\nPrema grčkoj tradiciji Arkadija je postojbina Pana, domovina jednostavnih, priprostih i poštenih ljudi (pastira). Kao simbol nepokvarena i idilična života javlja se tzv. bukolska (pastirska) poezija. Obnovljena u doba renesanse pod utjecajem idiličnog romana "Arkadija" talijanskog pisca J. Sannazzara. \n\nPo Arkadiji je ime dobila i čuvena knjižnica Akademija (Accademia degli Arcadi), osnovana 1690. g. u Rimu, a pod njenim utjecajem osnovana su i mnoga slična društva diljem Italije i hrvatske obale (Zadar, Split, Dubrovnik).\n\nVanjske poveznice\n\nPan-Arkadski Kongres.\nhttp://www.arcadians.gr\nSveučilište u Patrasu, Arkadia-Project.\nArkadija, Grčka.\nNepoznata Arkadija.\nhttp://flyingbrick.freeyellow.com/arcadia.htm \nhttp://www.arcadianet.gr/en/\nhttp://www.tripolis.gr\n\nZemljopis Grčke",
    "question": "Koji je naziv za pjesništvo pastira koje simbolizira neiskvareni i idiličan život?",
    "answers": {
        "answer_start": [1037],
        "text": ["bukolska"]
    }
}
```

```json
{
  "context": "Hans Emil Alexander Gaede (Kolberg, 19. veljače 1852. -  Freiburg im Breisgau, 16. rujna 1916.) je bio njemački general i vojni zapovjednik. Tijekom Prvog svjetskog rata zapovijedao je Armijskim odjelom B na Zapadnom bojištu.\n\nVojna karijera\nHans Gaede rođen je 19. veljače 1852. u Kolbergu (danas Kolobrzeg u Poljskoj). Sin je Alexandera Gaede i Emilie Franke. Gaede je u prusku vojsku stupio 1870. godine, te je sudjelovao u Prusko-francuskom ratu u kojem je i ranjen. Nakon rata pohađa Prusku vojnu akademiju, te nakon završetka iste služi u raznim vojnim jedinicama kao u i pruskom ministarstvu rata. Čin pukovnika dostigao 1897. godine kada postaje zapovjednikom i tvrđave Thorn. General bojnikom je postao 1900. godine, dok je 1904. godine promaknut u čin general poručnika kada dobiva zapovjedništvo nad 33. pješačkom divizijom smještenom u Metzu koji se tada nalazio u okviru Njemačkog Carstva. Godine 1907. Gaede je stavljen na raspolaganje.\n\nPrvi svjetski rat\nNa početku Prvog svjetskog rata Gaede je reaktiviran, te postaje zamjenikom zapovjednika XIV. korpusa koji je bio u sastavu 7. armije koja se nalazila pod zapovjedništvom Josiasa von Heeringena. U rujnu 1914. postaje zapovjednikom Armijskog odjela Gaede koji je kasnije preimenovan u Armijski odjel B koji je držao front u Gornjem Alzasu. Za zapovijedanje u borbama u Alzasu Gaede je 25. rujna 1915. godine odlikovan ordenom Pour le Mérite. U prosincu 1915. Gaedeu je na Sveučilištu u Freiburgu dodijeljen počasni doktorat.\n\nSmrt\nU rujnu 1916. godine Gaede se teško razbolio zbog čega je 3. rujna 1916. morao napustiti zapovjedništvo armijskog odjela. Umro je 16. rujna 1916. godine u 64. godini života u bolnici Freiburgu im Breisgau od posljedica operacije.\n\nVanjske poveznice\n     Hans Gaede na stranici Prussianmachine.com\n     Hans Gaede na stranici Deutschland14-18.de\n\nNjemački vojni zapovjednici u Prvom svjetskom ratu",
  "question": "Koju nagradu je Gaede primio 25. rujna 1915.?",
  "answers": {
    "answer_start": [1395],
    "text": ["Pour le Mérite"]
  }
}
```

```json
{
    "context": "Žiroglavci (Enteropneusta) su u klasičnoj sistematici životinjski razred s manje od 100 poznatih vrsta. Ubraja ih se u kojeno polusvitkovce (Hemichordata) i preko njih u drugousti (Deuterostomia), jer im se tijekom embrionalnog razvoja usta razvijaju a ne proizlaze iz "prausta", prvog otvora ranog embrionalnog životnog stadija, gastrule. Njihovo znanstveno ime znači, što izražava i tradicionalno mišljenje da su oni praoblik svitkovaca, u koje spadaju i kralježnjaci.\n\nNo, mjesto žiroglavaca u sistematici je danas sporno. Tako se razmatra moguća srodnost žiroglavaca ne samo sa svitkovcima, nego i s bodljikašima (Echinodermata) u koje spadaju na primjer zvjezdače (Asteroidea) i ježinci (Echinoidea). Čak se sve više smatra vjerojatnijim da žiroglavci ne čine monofiletsku skupinu, što znači da oni nisu svi potomci istih zajedničkih predaka.\n\nGrađa i izgled\nTijelo žiroglavaca je meko, crvoliko, i osim grube podjele na tri dijela, nesegmentirano. Veličinom su vrlo različiti, neke vrste su duge samo nekoliko milimetara, dok druge mogu biti duge i 2,5 metra. Boja im je različita, od bijele do tamno ljubičaste.\n  \nMeđu beskralješnjacima, žiroglavci su neobični jer imaju neke osobine koje su tipične za kralježnjake: \n Njihov živčani sustav satoji se od živčanih vrpci koje se protežu leđnom i trbušnom stranom životinje. U predjelu "glave" i oko crijeva ove dvije živčane vrpce kružno su međusobno povezane a od njih se odvajaju živčane niti koje završavaju u vanjskoj koži. Leđna živčana vrpca smještena je u posebnom naboru. Zbog njegovog nastanka u embrionalnom razvoju ponekad ga se smatra homolognim leđnoj moždini svitkovaca.\n Žiroglavci imaju i do 100 ždrijelnih pukotina koje imaju isto anatomsko porijeklo kao i škrge kod riba. Voda koja im uđe na usni otvor nakon zadržavanja djelića hrane, izlazi iz tijela kroz te pukotine.\n\nHrana, životni prostor i rasprostranjenost\nŽiroglavci se hrane na dva različita načina: ili kopaju kroz sediment morskog dna, što znači da uzimaju mulj dna i probavljaju u njemu sadržan organski sadržaj (kao kišne gliste), ili filtriraju iz vode sadržane djeliće organske materijekao na primjer alge. Zbog toga žive uglavnom u ili neposredno ispod dijela izloženog plimi i oseci, na ili u morskom dnu (bentos) dijelom i do dubine od 5.000 metara, i tamo često žive u kanalićima u obliku slova U. Samo rijetke vrste žive u otvorenom moru (pelagijal). Žiroglavci žive u svim morskim područjima, od tropa pa sve do u polarna područja.\n\nRazmnožavanje\nŽiroglavci su odvojenih spolova, no izgledom se gotovo ne razlikuju. Iz oplođenog jajašca najčešće se prvo razvijaju trepetljive larve vrlo slične larvama bodljikaša. Dio životnog ciklusa prije metamorfoze provodi kao plankton hraneći se djelićima hrane koji se zadrže na trepetljikama larve i od tamo se prenose do ustiju. Kod nekih vrsta razvoj se odvija direktno, bez larvenog stadija.\n\nDrugi projekti i vanjske poveznice\nTaksonomija žiroglavaca  (engleski)\nFilogeneza žiroglavaca (engleski)\n\nPolusvitkovci",
    "question": "Koliki je broj poznatih vrsta žiroglavaca?",
    "answers": {
        "answer_start": [75],
        "text": ["manje od 100"]
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

```text
Sljedeći tekstovi sadrže pitanja i odgovore.
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
euroeval --model <model-id> --dataset multi-wiki-qa-hr
```

## Knowledge

### MMLU-hr

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2410.08928)
and is a machine translated version of the English
[MMLU dataset](https://openreview.net/forum?id=d7KBjmI3GmQ). It features questions
within 57 different topics, such as elementary mathematics, US history, and law. DeepL
was used to translate the dataset to Croatian.

The original full dataset consists of 254 / 12,338 samples for validation and testing.
These splits were merged, duplicates removed, and new splits were created with 1,024 /
256 / 2048 samples for training, validation, and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Kako se odvija lateralna komunikacija u organizaciji?\nIzbori:\na. Informacije se prenose prema gore.\nb. Informacije se prenose prema dolje.\nc. Informacije su dvosmjerni proces.\nd. Informacije se prenose između različitih odjela i funkcija.",
  "label": "d"
}
```

```json
{
  "text": "Kako astronomi misle da Jupiter generira svoju unutarnju toplinu?\nIzbori:\na. kroz egzotermne kemijske reakcije koje pretvaraju kemijsku potencijalnu energiju u toplinsku energiju\nb. nuklearna fuzija\nc. kontrakcijom koja mijenja gravitacijsku potencijalnu energiju u toplinsku energiju\nd. unutarnje trenje zbog njegove brze rotacije i diferencijalne rotacije",
  "label": "c"
}
```

```json
{
  "text": "Ako se parabola $y_1 = x^2 + 2x + 7$ i pravac $y_2 = 6x + b$ sijeku u samo jednoj točki, koja je vrijednost $b$?\nIzbori:\na. 7\nb. 3\nc. 12\nd. 4",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Sljedeća su pitanja s višestrukim izborom (s odgovorima).
  ```

- Base prompt template:

  ```text
  Pitanje: {text}
  Izbori:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Odgovor: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Pitanje: {text}

  Odgovorite na gornje pitanje koristeći 'a', 'b', 'c' ili 'd', i ništa drugo.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset mmlu-hr
```

### Unofficial: INCLUDE-hr

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
  "text": "U čemu se prema Ludwigu Wittgensteinu otkriva značenje riječi?\nIzbori:\na. u kritici metafizike\nb. u idealnome jeziku\nc. u upotrebi riječi\nd. u prinudnoj šutnji",
  "label": "a",
  "subject": "Philosophy"
}
```

```json
{
  "text": "Na koji od navedenih uzročnika bolesti antibiotici neće djelovati?\nIzbori:\na. na Salmonella typhi\nb. na Herpes simplex\nc. na Streptococcus mutans\nd. na Escherichia coli",
  "label": "b",
  "subject": "Biology"
}
```

```json
{
  "text": "Koje vrste veza prevladavaju između molekula metana?\nIzbori:\na. kovalentne\nb. vodikove\nc. van der Waalsove\nd. peptidne",
  "label": "c",
  "subject": "Chemistry"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Sljedeća su pitanja s višestrukim izborom (s odgovorima).
  ```

- Base prompt template:

  ```text
  Pitanje: {text}
  Odgovor: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Pitanje: {text}

  Odgovorite na gornje pitanje koristeći {labels_str}, i ništa drugo.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset include-hr
```

## Common-sense Reasoning

### Winogrande-hr

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2506.19468)
and is a translated and filtered version of the English
[Winogrande dataset](https://doi.org/10.1145/3474381). DeepL was used to translate the
dataset to Croatian.

The original full dataset consists of 47 / 1,210 samples for training and testing, and
we use 128 of the test samples for validation, resulting in a 47 / 128 / 1,085 split for
training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Nisam mogao kontrolirati vlagu kao što sam kontrolirao kišu, jer je _ dolazila odasvud. Na što se odnosi praznina _?\nIzbori:\na. vlaga\nb. kiša",
  "label": "a"
}
```

```json
{
  "text": "Jessica je mislila da je Sandstorm najbolja pjesma ikad napisana, ali Patricia ju je mrzila. _ je kupila kartu za jazz koncert. Na što se odnosi praznina _?\nIzbori:\na. Jessica\nb. Patricia",
  "label": "b"
}
```

```json
{
  "text": "Termostat je pokazivao da je dolje dvadeset stupnjeva hladnije nego gore, pa je Byron ostao u _ jer mu je bilo hladno. Na što se odnosi praznina _?\nIzbori:\na. dolje\nb. gore",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Sljedeća su pitanja s višestrukim izborom (s odgovorima).
  ```

- Base prompt template:

  ```text
  Pitanje: {text}
  Mogućnosti:
  a. {option_a}
  b. {option_b}
  Odgovor: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Pitanje: {text}
  Mogućnosti:
  a. {option_a}
  b. {option_b}

  Odgovorite na gornje pitanje koristeći 'a' ili 'b', i ništa drugo.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset winogrande-hr
```

## Instruction-following

### MultiIFEval-hr

MultiIFEval-hr is part of the MultiIFEval benchmark spanning 305 languages. It is
generated by translating and localising the English IFEval dataset using a structured
LLM generation pipeline. For each target language, a randomly selected Wikipedia article
in that language provides contextual grounding to reduce hallucination and improve
cultural localisation. The pipeline preserves instruction_id_list values for
traceability to the original English samples, and retains kwargs keys with values
localised where appropriate, enabling programmatic constraint verification. The dataset
was published [here](https://huggingface.co/datasets/EuroEval/multi-ifeval-hr).

This dataset is part of the MultiIFEval benchmark introduced in
[this draft paper](https://raw.githubusercontent.com/alexandrainst/multi_ifeval/refs/heads/feat/add-paper/paper/acl_latex.tex).

We use the dataset as the test split, and do not include other splits, as we only
evaluate models zero-shot and the size is too small to warrant a validation set.

Here are a few examples from the test split:

```json
{
  "text": "Napišite sažetak Wikipedijske stranice \"https://hr.wikipedia.org/wiki/Hrvatski_jezik\" s najmanje 200 riječi. Nemojte koristiti nijednu zarez i istaknite najmanje 3 odjeljka koji imaju naslove u Markdown formatu, na primjer *istaknuti odjeljak Dio 1*, *istaknuti odjeljak Dio 2*, *istaknuti odjeljak Dio 3*.",
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
  "text": "Planiram putovanje u Hrvatsku i želim da mi napišeš plan putovanja u Shakespeareovom stilu. Nije ti dopušteno koristiti zareze u svom odgovoru.",
  "target_text": {
    "instruction_id_list": ["punctuation:no_comma"],
    "kwargs": [{}]
  }
}
```

```json
{
  "text": "Napravite životopis za svježe diplomiranog studenta koji se prijavljuje za svoj prvi posao. Obavezno uključite najmanje 12 mjesta za popunjavanje u uglatim zagradama, kao što su [Ime] ili [Adresa].",
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
euroeval --model <model-id> --dataset multi-ifeval-hr
```

## Hallucination Detection

### RAGTruth-hr

This dataset is a Croatian translation of the
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
  "prompt": "Upute:\nNapišite objektivan pregled o sljedećem lokalnom poslovanju temeljen isključivo na dostavljenim strukturiranim podacima u JSON formatu. Trebali biste uključiti detalje i obraditi informacije navedene u recenzijama kupaca. Pregled bi trebao biti dug 100 - 200 riječi. Ne izmišljajte informacije. Strukturirani podaci:\n{'ime': 'Crushcakes & Co', 'adresa': '1150 Coast Village Rd', 'grad': 'Montecito', 'država': 'CA', 'kategorije': 'Restorani, Kafići, Hrana, Pekarnice, Cupcakes, Kava i čaj, Deserti', 'radno vrijeme': {'Ponedjeljak': '7:30-17:0', 'Utorak': '7:30-17:0', 'Srijeda': '7:30-17:0', 'Četvrtak': '7:30-17:0', 'Petak': '7:30-17:0', 'Subota': '7:30-17:0', 'Nedjelja': '7:30-17:0'}, 'atributi': {'PoslovniParking': None, 'RestoraniRezervacije': False, 'VanjskoSjedenje': True, 'WiFi': 'besplatno', 'RestoraniZaPonijeti': True, 'RestoraniDobroZaGrupe': False, 'Glazba': None, 'Ambijent': None}, 'zvijezde_poslovanja': 4.5, 'informacije_o_recenzijama': [{'zvijezde_recenzije': 5.0, 'datum_recenzije': '2015-09-14 17:52:35', 'tekst_recenzije': 'Želite li vidjeti kako se moje unutarnje dijete ponaša kao budala pred svima, oči velike kao Marge? Pustite me u ovo mjesto! Ukusne slatke poslastice koje natjeraju moje malene okusne pupoljke da vrište! Hansel i Gretel bi riskirali svoje male živote za ove poslastice!!!!'}, {'zvijezde_recenzije': 4.0, 'datum_recenzije': '2015-09-08 02:17:26', 'tekst_recenzije': 'Latte su bili ukusni s zdravim mliječnim alternativama. šarmantno, ali malo mjesto za jelo i slatki cupcakes također'}, {'zvijezde_recenzije': 5.0, 'datum_recenzije': '2015-07-23 18:02:48', 'tekst_recenzije': 'Najbolji hamburger u gradu. Pretpostavljajući da ih volite onako kako ja: čokoladna torta, prekrivena kremom, bez mesnih peciva i povrća.'}]\nPregled:",
  "answer": "Crushcakes & Co je lokalni posao smješten na 1150 Coast Village Rd u Montecitu, CA. To je restoran i kafić koji se specijalizirao za posluživanje ukusne hrane, pekarskih proizvoda, kolačića, kave, čaja i deserta. Poslovanje je otvoreno od 7:30 do 17:00 svakog dana u tjednu.\n\nKupci su pohvalili Crushcakes & Co zbog njegovih ukusnih poslastica i aromatičnih ponuda. Jedan recenzent izrazio je svoje uzbuđenje zbog sočnih deserata koji čine da im okusni pupoljci vrište od radosti. Uporedili su iskustvo s njihovim unutarnjim djetetom koje se ponaša glupo u javnosti. Drugi kupac pohvalio je ukusne latte, spominjući dostupnost zdravih mliječnih alternativa. Također su spomenuli šarmantno, ali malo područje za jelo i slatke kolačiće.\n\nCrushcakes & Co je dobio ukupnu ocjenu od 4.5 zvjezdica, što ukazuje na visoku kvalitetu njihovih proizvoda i usluga. Poslovanje nudi besplatan WiFi i vanjsko sjedenje za kupce. Međutim, rezervacije nisu dostupne, a poslovanje možda nije prikladno za velike grupe.\n\nU sažetku, Crushcakes & Co je visoko cijenjena lokalna ustanova poznata po svojim ukusnim poslasticama i raznolikosti hrane i pića. Njegova prikladna lokacija i ugodne ponude čine ga popularnim izborom za lokalne stanovnike i posjetitelje.",
  "labels": []
}
```

```json
{
  "prompt": "Kratko odgovorite na sljedeće pitanje:\nšto je tranzistor i kako radi\nImajte na umu da vaš odgovor treba biti strogo temeljen na sljedeće tri odlomka:\nodlomak 1: Dizajn tranzistora omogućuje mu da funkcionira kao pojačalo ili prekidač. To se postiže korištenjem male količine električne energije za kontrolu vrata na mnogo većem izvoru električne energije, slično kao što se okrene ventil za kontrolu opskrbe vodom. Osnova privlači te elektrone iz emitera jer ima pozitivniji napon od emitera. Ovaj pokret elektrona stvara protok električne energije kroz tranzistor. Struja prolazi od emitera do kolektora kroz osnovu.\n\nodlomak 2: Tranzistor je minijaturna elektronička komponenta koja može raditi dva različita posla. Može raditi ili kao pojačalo ili kao prekidač: 1 Kada radi kao pojačalo, prima malu električnu struju na jednom kraju (ulazna struja) i proizvodi mnogo veću električnu struju (izlazna struja) na drugom. 2 Tranzistori također mogu raditi kao prekidači. Obično, rupe u bazi djeluju kao prepreka, sprječavajući bilo kakav značajan protok struje od emitera do kolektora dok je tranzistor u svom isključenom stanju. Tranzistor radi kada elektroni i rupe počnu kretati preko dva spoja između n-tipa i p-tipa silicija.\n\nodlomak 3: Tranzistor je poluvodički uređaj koji se koristi za pojačavanje i prebacivanje elektroničkih signala i električne energije. Sastoji se od poluvodičkog materijala s najmanje tri priključka za povezivanje s vanjskim krugom. Kolektorska struja je otprilike β (dobitak struje zajedničkog emitera) puta struja baze. Obično je veća od 100 za tranzistore malih signala, ali može biti manja kod tranzistora dizajniranih za visokonaponske primjene. Za razliku od tranzistora s poljima (vidi dolje), BJT je uređaj s niskim ulaznim otporom.\n\nU slučaju da odlomci ne sadrže potrebne informacije za odgovor na pitanje, molimo odgovorite s: \"Nemoguće je odgovoriti na temelju danih odlomaka.\"\noutput:",
  "answer": "Na temelju danih odlomaka, tranzistor je elektronička komponenta napravljena od poluvodičkog materijala s najmanje tri terminala za povezivanje s vanjskim krugom. Funkcionira kao pojačalo ili prekidač ovisno o protoku elektrona kroz svoju bazu. Kada radi kao pojačalo, prima mali električni trenutak na jednom kraju i proizvodi veći električni trenutak na drugom. Protoku elektrona se upravlja korištenjem male količine električne energije za okretanje strukture nalik ventilu u tranzistoru. U svom isključenom stanju, praznine u bazi djeluju kao barijera, sprječavajući bilo kakav značajan protok trenutka od emitera do kolektora. Tranzistori mogu imati konfiguraciju zajedničkog emitera gdje je kolektorski trenutak otprilike β puta bazni trenutak.",
  "labels": []
}
```

```json
{
  "prompt": "Kratko odgovorite na sljedeće pitanje:\nprednosti msm praha\nImajte na umu da vaš odgovor treba biti strogo temeljen na sljedeće tri odlomka:\nodlomak 1: MSM koristi tijelu pomažući da se kisik učinkovitije unese u krv uz istu količinu rada. 3. MSM prah pomaže povećati energiju. Još jedna prednost MSM-a je da razine energije rastu jer MSM pomaže da stanične membrane postanu propusnije.\n\nodlomak 2: 1. MSM prah koristi našim tijelima pomažući apsorbirati više hranjivih tvari (vitamina i minerala). Koenzim Q10 se spaja s MSM-om, što znači da je za tijelo potrebno imati MSM (sumpor) kako bi u potpunosti iskoristilo ovu hranjivu tvar.\n\nodlomak 3: MSM prah pomaže ublažiti mišićne bolove i nelagodu. 7. MSM prah koristi tijelu smanjujući upalu zbog ozljede ili upalnih bolesti kao što je artritis. Kada je tlak vode unutar stanice veći od tlaka vode izvan stanice, dolazi do upale i oticanja.\n\nU slučaju da odlomci ne sadrže potrebne informacije za odgovor na pitanje, molimo odgovorite s: \"Nemoguće je odgovoriti na temelju danih odlomaka.\"\noutput:",
  "answer": "Na temelju danih odlomaka, prednosti MSM praha uključuju poboljšanje oksigenacije krvi, povećanje razine energije, poboljšanje apsorpcije hranjivih tvari, ublažavanje mišićnih bolova i smanjenje upale.",
  "labels": []
}
```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ragtruth-hr
```
