# Catalan

This is an overview of all the datasets used in the Croatian part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### GuiaCat

This dataset was published in [here](https://huggingface.co/datasets/projecte-aina/GuiaCat).
The dataset consists of 5,750 restaurant reviews in Catalan, with five associated scores
and a sentiment label. The data was provided by [GuiaCat](https://guiacat.cat/).

The original full dataset consists of 4,750 / 500 / 500 samples for the training,
validation and test splits, respectively. We use 1,024 / 256 / 2,048 samples for our
training, validation and test splits, respectively. The train and validation splits
are subsets of the original dataset, while the test split has additional samples from
the train split to reach the desired size.

Here are a few examples from the training split:

```json
{
    "text": "Ens han servit un menú d'allò més bo, amb la característica de la cuina catalana, canelons, i conill, el preu correcte, hem sortit satisfets.",
    "label": "positive"
}
```

```json
{
    "text": "Cuina catalana típica de la regió amb un menú molt variat. El menú és econòmic i de qualitat",
    "label": "positive"
}
```

```json
{
    "text": "Lloc petit,molt tranquil i acollidor. Molt bons els embotits i les carns, que les agafen de la  propia xarcuteria que tenen abaix. Sempre que hi he anat sempre acabo comprant alguna que altre cosa de lo que m'ha agradat. Gran lloc per anar acompanyat. Molt recomanable als amants dels embotits que disfrutaran com nens petits . ",
    "label": "positive"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Els documents següents i el seu sentiment, que pot ser positiu, neutral o negatiu.
  ```

- Base prompt template:

  ```text
  Document: {text}
  Sentiment: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Document: {text}

  Classifiqueu el sentiment en el document. Contesteu només amb positiu, neutral, o negatiu, i res més.
  ```

- Label mapping:
  - `positive` ➡️ `positiu`
  - `neutral` ➡️ `neutral`
  - `negative` ➡️ `negatiu`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset guia-cat
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
and was automatically created from the [Croatian Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Croatian-SET) by assuming that the
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
    "text": "El portaveu del govern català ha demanat a Maragall que \"ens deixi fer, perquè al final les coses les hem de fer nosaltres dins de la nostra coalició i que cadascú s' ocupi de casa seva\".",
    "label": "correct"
}
```

```json
{
    "text": "La petrolera francesa Total llançarà una oferta de compra (OPA) sobre el 41 % de la Petrofina belga per 1,8 bilions de pessetes.",
    "label": "incorrect"
}
```

```json
{
    "text": "Els fluids de l'organisme poden trencar les boles, i això provoca la intoxicació mortal.",
    "label": "correct"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Les següents oracions indiquen si són gramaticalment correctes.
  ```

- Base prompt template:

  ```text
  Oració: {text}
  Gramaticalment correcta: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Oració: {text}

  Determina si l'oració és gramaticalment correcta o no. Respon amb 'sí' si és correcta, i amb 'no' si no ho és. Respon només amb aquesta paraula i res més.
  ```

- Label mapping:
  - `correct` ➡️ `sí`
  - `incorrect` ➡️ `no`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-ca
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

This dataset was published in
[this paper](https://doi.org/10.48550/arXiv.2410.08928) and is a machine
translated version of the English [MMLU dataset](https://openreview.net/forum?id=d7KBjmI3GmQ).
It features questions within 57 different topics, such as elementary mathematics, US
history, and law. DeepL was used to translate the dataset to Croatian.

The original full dataset consists of 254 / 12,338 samples for
validation and testing. These splits were merged, duplicates removed, and
new splits were created with 1,024 / 256 / 2048 samples for training, validation, and
testing, respectively.

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

## Common-sense Reasoning

### Winogrande-hr

This dataset was published in
[this paper](https://doi.org/10.48550/arXiv.2506.19468) and is a translated
and filtered version of the English
[Winogrande dataset](https://doi.org/10.1145/3474381). DeepL was used to
translate the dataset to Croatian.

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
