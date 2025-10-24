# 🇷🇸 Serbian

This is an overview of all the datasets used in the Serbian part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### MMS-sr

Information about the dataset can be found [here](https://brand24-ai.github.io/mms_benchmark/).
The corpus consists of 79 manually selected datasets from over 350 datasets reported in the
scientific literature based on strict quality criteria.

The original dataset contains a single split with 6,165,262 samples. We use
1,024 / 256 / 2,048 samples for our training, validation and test splits, respectively.

Here are a few examples from the training split:

```json
{
    "text": "Primiti manje od 10 trojki je uspeh za Radonjica.",
    "label": "neutral"
}
```

```json
{
    "text": "RT @Susanna_SQ: Osecati se dobro u sopstvenoj kozi, mozda je jedna od najvecih umetnosti zivljenja.",
    "label": "positive"
}
```

```json
{
    "text": "RT @aleksitimija_: ljubav je prolazna. prijatelji su prolazni. strast je prolazna. sve je prolazno. jedino sto mi ostaje jeste puno misli i...",
    "label": "negative"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  U nastavku su dokumenti i njihov sentiment, koji može biti 'pozitivan', 'neutralan' ili 'negativan'.
  ```

- Base prompt template:

  ```text
  Dokument: {text}
  Sentiment: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Dokument: {text}

  Klasifikujte sentiment u dokumentu. Odgovorite sa 'pozitivan', 'neutralan', ili 'negativan', i ništa drugo.
  ```

- Label mapping:
  - `positive` ➡️ `pozitivan`
  - `neutral` ➡️ `neutralan`
  - `negative` ➡️ `negativan`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset mms-sr
```

## Named Entity Recognition

### UNER-sr

This dataset was published in
[this paper](https://aclanthology.org/2024.naacl-long.243/).

The original dataset consists of 3,328 / 536 / 520 samples for the
training, validation, and test splits, respectively. We use 1,024 / 256 / 2,048
samples for our training, validation and test splits, respectively. The train and
validation splits are subsets of the original splits, while the test split is
created using additional samples from the train and validation splits.

Here are a few examples from the training split:

```json
{
    "tokens": ["Pre", "samo", "dve", "decenije", "Hrvatska", "je", "proglasila", "nezavisnost", "od", "bivše", "Jugoslavije", "."],
    "labels": ["O", "O", "O", "O", "B-LOC", "O", "O", "O", "O", "O", "B-LOC", "O"]
}
```

```json
{
    "tokens": ["Vratio", "se", "makartizam", ",", "samo", "su", "progonitelji", "sada", "iz", "liberalne", "elite", "i", "oni", "kontrolišu", "frakciju", "u", "državi", "koja", "se", "otela", "od", "države", "i", "bori", "se", "protiv", "izabrane", "vlasti", "."],
    "labels": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]
}
```

```json
{
    "tokens": ["Ne", "smatra", "da", "su", "pregovori", "sa", "Srbijom", "prvi", "prioritet", "."],
    "labels": ["O", "O", "O", "O", "O", "O", "B-LOC", "O", "O", "O"]
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Sledeće su rečenice i JSON rečnici sa imenovanim entitetima koji se pojavljuju u datoj rečenici.
  ```

- Base prompt template:

  ```text
  Rečenica: {text}
  Imenovani entiteti: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Rečenica: {text}

  Identifikujte imenovane entitete u rečenici. Trebalo bi da ovo ispišete kao JSON rečnik sa ključevima 'osoba', 'mesto', 'organizacija' i 'razno'. Vrednosti treba da budu liste imenovanih entiteta te kategorije, tačno onako kako se pojavljuju u rečenici.
  ```

- Label mapping:
  - `B-PER` ➡️ `osoba`
  - `I-PER` ➡️ `osoba`
  - `B-LOC` ➡️ `mesto`
  - `I-LOC` ➡️ `mesto`
  - `B-ORG` ➡️ `organizacija`
  - `I-ORG` ➡️ `organizacija`
  - `B-MISC` ➡️ `razno`
  - `I-MISC` ➡️ `razno`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset uner-sr
```

## Linguistic Acceptability

### ScaLA-sr

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Serbian Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Serbian-SET) by assuming that the
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
    "text": "Hrvatski ministar odbrane Branko Vukelić i njegov srpski kolega Dragan Šutanovac potpisaće u utorak (8. juna) u Zagrebu bilateralni sporazum o saradnji na polju odbrane.",
    "label": "correct"
}
```

```json
{
    "text": "Žene vlasnici i rukovodioci pokazale su veliku upornost u očuvanju svojih, posebno tokom ekonomske krize.",
    "label": "incorrect"
}
```

```json
{
    "text": "Očekuje se da snimanje bude završeno do kraja leta, a montaža bi trebalo da bude gotova do aprila sledeće godine.",
    "label": "correct"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  U nastavku su rečenice i da li su gramatički ispravne.
  ```

- Base prompt template:

  ```text
  Rečenica: {text}
  Gramatički ispravna: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Rečenica: {text}

  Odredite da li je rečenica gramatički ispravna ili ne. Odgovorite sa {labels_str}, i ništa drugo.
  ```

- Label mapping:
  - `correct` ➡️ `da`
  - `incorrect` ➡️ `ne`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-sr
```

## Reading Comprehension

### MultiWikiQA-sr

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
    "context": "Клеопатра Карађорђевић (Крајова, 14/26. новембар 1835 — Глајхенберг, 1/13. јул 1855) је била ћерка кнеза Александра Карађорђевића и кнегиње Персиде.\n\nБиографија \nРођена је у Влашкој од оца Александра Карађорђевића (1806—1885) и мајке Персиде, рођене Ненадовић. Породица Карађорђевић је од 1817. до 1831. живела у Хотину, а затим у Влашкој до 1839. У Србију су дошли октобра 1839. и Александар је априла 1840. ступио у војну службу као ађутант кнеза Михаила Обреновића.\n\nАлександар је изабран за кнеза Србије 1842. године, а после две године је прешао у двор, кућу купљену од Стојана Симића. Клеопатра је одрастала са две године старијом сестром Полексијом (1833—1914), и када су напуниле 10 и 12 година поставило се питање њиховог образовања. На препоруку Илије Гарашанина и Јована Хаџића за приватног учитеља је изабран Матија Бан, Дубровчанин који је из Цариграда дошао у Србију 1844. године. На дужност приватног учитеља кнежевих ћерки Полексије и Клеопатре ступио је 13. јула 1845.\n\nЧешки композитор и пијаниста Алојз Калауз који је у Србију дошао 1843. године и у Београду давао приватне часове клавира, компоновао је песму „Што се боре мисли моје“ за Клеопатрин 15. рођендан. Средином педесетих година 19. века поново је компоновао Корнелије Станковић и та песма је за време друге владе кнеза Михаила редовно певана на баловима у Београду.\n\nСестра Полексија се удала 1849. за Константина Николајевића. Клеопатра је са њима 1852. путовала у Цариград, у пасошу је именована као „принцеза србска“. До удаје је живела у двору. Удата је 9. фебруара 1855. за Милана Петронијевића, сина Аврама Петронијевића који је био председник Владе 1844—1852. Венчање је било у Саборној цркви, кум је био Стефан Стефановић Тенка, стари сват аустријски конзул Теодор Радосављевић, а венчао их је митрополит београдски Петар.\n\nУмрла је 1/13. јула 1855. године у бањи Глајхенберг у Штајерској и сахрањена у породичној гробници у Тополи, касније у цркви Светог Ђорђа на Опленцу.\n\nУ Неменикућама постоји Клеопатрина чесма.\n\nПородично стабло\n\nПородица\n\nСупружник\n\nВиди још \n Карађорђевићи\n Петронијевићи\n\nРеференце\n\nЛитература \n Радомир Ј. Поповић: Принцеза Клеопатра Карађорђевић-Петронијевић, Даница за 2012. годину, Вукова задужбина, Београд (2011). стр. 352-363.\n\nСпољашње везе \n Музичка честитка за Клеопатру Карађорђевић („Политика”, 5. август 2017)\n\nРођени 1835.\nУмрли 1855.\nКлеопатра', 'question': 'Који је датум рођења Клеопатре Карађорђевић?",
    "answers": {
        "answer_start": [33],
        "text": ["14/26. новембар 1835"]
    }
}
```

```json
{
    "context": "Доња Гуштерица је насеље у општини Липљан на Косову и Метохији. По законима самопроглашене Републике Косово насеље се налази у саставу општине Грачаница. Атар насеља се налази на територији катастарске општине Доња Гуштерица површине 1133 -{ha}-.\n\nИсторија \nДоња Гуштерица је почетком 20. века сматрана за највеће село на Косову Пољу. Ту је 1904. године завршена градња српског православног храма. Градњу су помогли ктитори и побожни народ из места.\n\nПорекло становништва по родовима \nСрпски родови подаци из 1932. године)\n\n Доганџићи (32 k., Св. Јован). Имали су две славе, јер су, поред старе славе Св. Јована, завели доцније и славу Св. Николе. Стари су досељеници и оснивачи села. Доселили се од Тетова да избегну освету, „јер су поубијали арамије у својој кући“. Досељење им је старије од оних помена соколарства у овом селу средином XVIII века.\n\n Шкуртови (3 k., Св. Никола) и Сталићи (1 k., Ђурђиц), досељеници непознатог порекла.\n\n Аладанци (5 k., Св. Никола). Досељени крајем XVIII века из Гњиланске Мораве.\n\n Терзићи (6 k., Св. Никола). Досељени крајем XVIII века из околине Гњилана из села Понеша.\n\n Живанчићи (7 k., Св. Никола). Доселили се из Ибарског Колашина почетком XIX века.\n\n Бакшићани (6 k., Св. Јанићије Девички). Пресељени из Бакшије почетком XIX века.\n\n Сојевићи (12 k., Ђурђиц). Досељени око 1820. године из Сојева. Исти су род са Сојевићима у Топличану.\n\n Шубарићи (10 k., Митровдан). Пресељени из Плешине после Сојевића.\n\n Подримци (4 k., Св. Никола). Избегли око 1830. године из Мовљана у Метохији да избегну крвну освету, јер су убили неког Арбанаса што је хтео да им отме Волове.\n\n Грбићовци (6 k., Св. Никола). Пресељени из Гребна око 1830. године.\n\n Кукурегџићи (5 k., Св. Никола). Пресељени из Гувног Села око 1830. године.\n\n Јерци или Јерцићи (1 k., Св. Арханђео). Пресељени средином XIX века из истоименог рода у Горњој Гуштерици, старином из Ибарског Колашина.\n\n Декићи (2 k., Св. Арханђео). Пресељени из Горње Брњице око 1870. године.\n\n Сиринићани (1 k., Ваведење). Досељени 1916. године из Сушића у Сиринићкој Жупи.\n\nДемографија \n\nНасеље има српску етничку већину.\nБрој становника на пописима:\n\n попис становништва 1948. године: 974\n попис становништва 1953. године: 1097\n попис становништва 1961. године: 1187\n попис становништва 1971. године: 1158\n попис становништва 1981. године: 1210\n попис становништва 1991. године: 1269\n\nРеференце \n\nВикипројект географија/Насеља у Србији\n\nНасељена места у Липљану\nНасељена места на Косову и Метохији",
    "question": "Одакле су Подримци побегли отприлике 1830. године?",
    "answers": {
        "answer_start": [1506],
        "text": ["Мовљана у Метохији"]
    }
}
```

```json
{
    "context": "Тржић Примишљански је насељено мјесто града Слуња, на Кордуну, Карловачка жупанија, Република Хрватска.\n\nГеографија \nТржић Примишљански се налази око 18 км сјеверозападно од Слуња.\n\nИсторија \nПоп Никола Гаћеша је ту у свом родном месту (рођ. 1785) хтео да преведе православне парохијане у унију. Али када је примио унију, убио га је 18. јуна 1820. године у његовој кући хајдук из Збега, Благоје Бараћ. Тако је спречена унија у Тржићу код Примишља.\n\nТо село је током ратова са Турцима у 16. и 17. веку било скоро потпуно опустошено. Остала је само католичка црква Св. Миховила и неколико околних кућа. Граничарски пуковник Оршић је 1686. године ту населио православне Србе из Цазина. На два километра од католичког храма подигли су православци себи богомољу посвећену Св. апостолу Петру.\n\nТржић Примишљански се од распада Југославије до августа 1995. године налазио у Републици Српској Крајини.\n\nСтановништво \nПрема попису становништва из 2011. године, насеље Тржић Примишљански је имало 20 становника.\n\nИзвори\n\nСпољашње везе \n\nСлуњ\nКордун\nНасељена места у Хрватској\nНасељена места у Карловачкој жупанији\nВикипројект географија/Насеља у Хрватској",
    "question": "Ко је одговоран за смрт попа Николе Гаћеше?",
    "answers": {
        "answer_start": [370],
        "text": ["хајдук из Збега, Благоје Бараћ"]
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Следе текстови са одговарајућим питањима и одговорима.
  ```

- Base prompt template:

  ```text
  Текст: {text}
  Питање: {question}
  Одговор у максимум 3 речи:
  ```

- Instruction-tuned prompt template:

  ```text
  Текст: {text}

  Одговорите на следеће питање о горњем тексту у максимум 3 речи.

  Питање: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-sr
```

## Knowledge

### MMLU-sk

This dataset is a machine translated version of the English [MMLU
dataset](https://openreview.net/forum?id=d7KBjmI3GmQ) and features questions within 57
different topics, such as elementary mathematics, US history and law. The translation to
Slovak was done by the University of Oregon as part of [this
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
  Možnosti:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
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

### Winogrande-sk

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2506.19468)
and is a translated and filtered version of the English [Winogrande
dataset](https://doi.org/10.1145/3474381).

The original full dataset consists of 47 / 1,210 samples for training and testing, and
we use 128 of the test samples for validation, resulting in a 47 / 128 / 1,085 split for
training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Nedokázal som ovládať vlhkosť ako som ovládal dážď, pretože _ prichádzalo odvšadiaľ. Na koho sa vzťahuje prázdne miesto _?\nMožnosti:\na. vlhkosť\nb. dážď",
  "label": "a"
}
```

```json
{
  "text": "Jessica si myslela, že Sandstorm je najlepšia pieseň, aká bola kedy napísaná, ale Patricia ju nenávidela. _ si kúpila lístok na jazzový koncert. Na koho sa vzťahuje prázdne miesto _?\nMožnosti:\na. Jessica\nb. Patricia",
  "label": "b"
}
```

```json
{
  "text": "Termostat ukazoval, že dole bolo o dvadsať stupňov chladnejšie ako hore, takže Byron zostal v _ pretože mu bola zima. Na koho sa vzťahuje prázdne miesto _?\nMožnosti:\na. dole\nb. hore",
  "label": "b"
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
  Možnosti:
  a. {option_a}
  b. {option_b}
  Odpoveď: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Otázka: {text}
  Možnosti:
  a. {option_a}
  b. {option_b}

  Odpovedzte na nasledujúcu otázku použitím 'a' alebo 'b', a nič iné.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset winogrande-sk
```
