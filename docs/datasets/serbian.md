# 🇷🇸 Serbian

This is an overview of all the datasets used in the Serbian part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### MMS-sr

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2306.07902).
The corpus consists of 79 manually selected datasets from over 350 datasets reported in
the scientific literature based on strict quality criteria.

The original dataset contains a single split with 76,368 Serbian samples. We use
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

### MMLU-sr

This dataset was published [in this paper](https://doi.org/10.48550/arXiv.2009.03300)
and features questions within 57 different topics, such as elementary mathematics, US
history and law.

The original full dataset consists of 276 / 1,439 / 13,173 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 2,048 split for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
new and there can thus be some overlap between the original validation and test sets and
our validation and test sets.

Here are a few examples from the training split:

```json
{
    "text": "Kruti, čvrsti kontejner konstantne zapremine sadrži idealni gas zapremine v1, pritiska P1 i temperature T1. Temperatura se povećava u izohornom procesu. Koja od sledećih tvrdnji NIJE tačna?\nOpcije:\na. Prosečna brzina molekula se povećava.\nb. Pritisak se povećava.\nc. Kinetička energija sistema se povećava.\nd. Zapremina se povećava.",
    "label": "d"
}
```

```json
{
    "text": "Kakav tip kovalentnih veza veže amino kiseline u proteinu?\nOpcije:\na. Peptidne veze\nb. Vodonične veze\nc. Jonske veze\nd. Glikozidne veze",
    "label": "a",
}
```

```json
{
    "text": "Teorija __________ predviđa postojanje tri nivoa moralnog rasuđivanja u kojima pojedinac može razmatrati etičke probleme, zavisno od svog kognitivnog kapaciteta.\nOpcije:\na. Egoizam\nb. Kognitivni moralni razvoj\nc. Razlika u moći\nd. Izbjegavanje neizvjesnosti",
    "label": "b",
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Slede pitanja višestrukog izbora (sa odgovorima).
  ```

- Base prompt template:

  ```text
  Pitanje: {text}
  Opcije:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Odgovor: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Pitanje: {text}

  Odgovorite na navedeno pitanje koristeći 'a', 'b', 'c' ili 'd', i ništa drugo.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset mmlu-sr
```

### Unofficial: INCLUDE-sr

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
  "text": "Koji je glavni grad Srbije?\nOpcije:\na. Novi Sad\nb. Niš\nc. Kragujevac\nd. Beograd",
  "label": "d"
}
```

```json
{
  "text": "Ko je napisao roman 'Na Drini ćuprija'?\nOpcije:\na. Branko Ćopić\nb. Mesa Selimović\nc. Ivo Andrić\nd. Dobrica Ćosić",
  "label": "c"
}
```

```json
{
  "text": "Koji ćelijski organoid je odgovoran za proizvodnju energije?\nOpcije:\na. Ribosom\nb. Hloroplast\nc. Golgijev aparat\nd. Mitohondrija",
  "label": "d"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Slede pitanja višestrukog izbora (sa odgovorima).
  ```

- Base prompt template:

  ```text
  Pitanje: {text}
  Odgovor: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Pitanje: {text}

  Odgovorite na navedeno pitanje koristeći {labels_str}, i ništa drugo.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset include-sr
```

## Common-sense Reasoning

### Winogrande-sr

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2506.19468)
and is a translated and filtered version of the English [Winogrande
dataset](https://doi.org/10.1145/3474381).

The original full dataset consists of 47 / 1,210 samples for training and testing, and
we use 128 of the test samples for validation, resulting in a 47 / 128 / 1,085 split for
training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
    "text": "Nisam mogao kontrolisati vlagu kao što sam kontrolisao kišu, jer je _ dolazilo sa svih strana. Na šta se odnosi prazno _?\nOpcije:\na. vlaga\nb. kiša",
    "label": "a"
}
```

```json
{
    "text": "Jessica je mislila da je Sandstorm najbolja pesma ikada napisana, ali Patricia ju je mrzela. _ je kupila kartu za džez koncert. Na šta se odnosi prazno _?\nOpcije:\na. Jessica\nb. Patricia",
    "label": "b"
}
```

```json
{
    "text": "Rukovanje hitnim slučajevima nikada nije bilo veoma teško za Kevina, ali jeste za Nelsona jer _ nije mogao da ostane smiren pod pritiskom. Na šta se odnosi prazno _?\nOpcije:\na. Kevin\nb. Nelson",
    "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Slede pitanja višestrukog izbora (sa odgovorima).
  ```

- Base prompt template:

  ```text
  Pitanje: {text}
  Opcije:
  a. {option_a}
  b. {option_b}
  Odgovor: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Pitanje: {text}
  Opcije:
  a. {option_a}
  b. {option_b}

  Odgovorite na navedeno pitanje koristeći 'a' ili 'b', i ništa drugo.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset winogrande-sr
```

## Summarisation

### LR-Sum-sr

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
    "text": "Desnica pobedila na izborima za EP\n\nPrema raspoloživim, još uvijek neslužbenim podacima, ključna većina zastupnika novog saziva Evropskog parlamenta ostaje iz partija desnog centra. Grupa evropskih narodnjačkih partija će od 735 mjesta držati 267, gubeći dvadesetak mjesta, dok su najveći gubitnici izbora, socijalisti, ostali bez čak 58 mjesta u Parlamentu i sada će ga popunjavati sa 159 zastupnika. Izgubili su i liberali 19 mjesta od sadašnjih stotinu, dok su evropski zeleni blago porasli sa sadašnjih 43 na 51 mjesto zastupnika u EP. "", kazao je predsjednik Evropskog parlamenta, . A ultra desne, anti-imigrantske, anti-islamske i anti-evropske partije su zabilježile porast glasača u Holandiji, Grčkoj, Finskoj, Italiji, Mađarskoj, Rumuniji i Velikoj Britaniji, dok im je popularnost opala u Belgiji, Francuskoj i Poljskoj. Neki od analitičara tvrde kako je krajnja desnica iskoristila činjenicu da su glasači u rekordno malom broju izašli na glasanje. Od direktnih izbora za EP 1979. nije zabilježen tako slab odziv glasača. Razlog za to se traži i u činjenici da je predizborna kampanja u brojnim evropskim državama u prvom planu imala nacionalnu, prije evropske agende. "", nagasila je potpredsjednica Evropske komisije, . Evropski parlament i ostatak evropskih institucija će se od danas morati pomiriti i sa činjenicom da ultra desni, anti-evropski, anti-imigrantski raspoloženi novi stanari parlamenta već pakuju kofere za Brisel gdje će \"kvariti posao\" pro-evropskim snagama. "", kazao je potpredsjednik Evropske komisije,. Već su, inače, počele najave prvih poteza pobjedničkih političkih grupacija. Tako su evropski narodnjaci najavili kako će od lidera Unije, što se krajem naredne sedmice okupljaju u Briselu, tražiti da dozvole njihovom kandidatu, sadašnjem predsjedniku Evropske komisije, , da ostane na toj funkciji i naredni mandat. Socijalisti i još više zeleni su oštro protiv, kritikujući Baroza da je premalo uradio na pitanju socijalne zaštite Evropljana od efekata aktuelne krize. Liberalna grupa u parlamentu ne bi imala ništa protiv da podrži Baroza, ukoliko bi ušla u koaliciju sa narodnjacima i ako bi tada njihovi koalicioni partneri odustali od postavljanja svoga čovjeka i na mjesto predsjednika parlamenta.",
    "target_text": "Rezultati tek održanih evropskih izbora su pokazali da su Evropljani ozbiljno zabrinuti za život, posao, ličnu egzistenciju. Povjerili "
}
```

```json
{
    "text": "SAD razvijaju \"značajan režim sankcija\" za Iran\n\nIranski funkcioneri saopštili su da je obogaćivanje uranijuma nastavljeno u utorak, u postrojenju Natanz, u prisustvu inspektora međunarodne agencije za atomsku energiju. Iran insistira da je svrha toga miroljubiva, ali zapadne zemlje podozrevaju da se program koristi za izgradnju nuklearnog oružja. Juče, u Beloj kući, predsednik Obama je rekao da su vrata još otvorena za Teheran da ponovo stekne poverenje međunarodne zajednice, ili da se suoči sa novim sankcijama. \"Tokom narednih nekoliko nedelja razvićemo značajan režim sankcija koji će im ukazati na to koliko su izolovani od međunarodne zajednice u celini,\" rekao je Obama. Savet bezbednosti Ujedinjenih nacija zaveo je već tri kruga sankcija prema Iranu zbog toga što nije obustavio obogaćivanje uranijuma. Francuska podržava američku inicijativu za uvođenje novih sankcija. \"Sada smo uvereni da je međunarodna zajednica ujedinjena u vezi sa ovakvim ponašanjem Irana,\" rekao je američki predsednik. Grupa svetskih sila i Ujedinjene nacije predložili su Teheranu da pošalje uranijum na obogaćivanje u inostranstvo, kako bi ga dobio natrag u obliku goriva za reaktor. U odgovor na to Iran je poslao mešane signale, što je Obama kritikovao. \"To nam pokazuje da, uprkos tvrdnjama da je svrha njihovog nuklearnog programa miroljubiva, oni nastavljaju kursem koji vodi ka izgradnji oružja, što je neprihvatljivo,\" rekao je Barak Obama. Sekretar ruskog Saveta bezbednosti, , rekao je da odluka Irana o daljem obogaćivanju uranijuma izaziva sumnju u miroljubivost tog programa. \"Iran insistira da ne želi nuklearno oružje već da razvija mirnodopski nuklearni program. Ali akcije koje preduzima, kao što je obogaćivanje uranijuma do 20 odsto, sa razlogom pokreću sumnje u drugim zemljama,\" rekao je Patrušev. Kina, koja se protivi dodatnim sankcijama Iranu, juče se založila za rešavanje tog spora putem dijaloga.",
    "target_text": "Predsednik Barak Obama kaže da Sjedinjene Države razvijaju, \"značajan režim mogućih sankcija,\" kao odgovor na iranski nuklearni program"
}
```

```json
{
    "text": "Rusija o situaciji vezanoj za Irak - 2002-09-13\n\nNa kraju trodnevnih razgovora Džon Bolton je rekao novinarima da æe Vašington u roku od nekoliko nedelja uputiti izaslanika u Moskvu radi razgovora o situaciji u Iraku. On je naglasio da æe Vašington -- pre nego što preduzme bilo kakvu akciju -- saslušati sve što Moskva ima da kaže na tu temu. ”Kao što je to bio sluèaj i sa zalivskom krizom pre 12 godina, oèevidno je da na ruskoj strani postoji zabrinutost koju æe SAD obiljno uzeti u obzir.“ Džon Bolton je dodao da su o tom pitanju ruski i amerièki predsednik veæ vodili intenzivne ragovore i da oni nisu završeni. Rusija se protivi unilateralnoj akciji protiv Iraka i zalaže se za povratak inspektora U-N-a u Irak. U juèerašnjem govoru u svetskoj organizaicji amerièki predsednik Buš je pozvao U-N da prozovu Irak zbog oružja masovnog uništenja. Amerièki državni podsekretar Bolton izjavio je da Vašinton neæe praviti sa Moskvom nikakve nagodbe u vezi sa Irakom. Prema njegovim reèima, Amerika ne planira da ignoriše moguæe ruske napade na èeèenska uporišta u Gruziji ukoliko Rusija za uzvrat odobri evenutalni amerièki napad na Irak: ”Mislim da je ignorisanje autoritera Saveta bezbednosti preozbiljno da bi se oko tog pitanja pravile nagodbe. Mislim da to ruska strana i ne oèekuje od amerièke.“ Džon Bolton je dodao da su amerièki argumenti protiv Iraka veæ sami po sebi dovoljno èvrsti.", "target_text": "Amerièki državi podsekretar Džon Bolton izrazio je uverenje da æe Rusija i SAD biti u stanju da postignu saglasnost u vezi sa Irakom. "
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:

  ```text
  Slede dokumenti sa odgovarajućim sažecima.
  ```

- Base prompt template:

  ```text
  Dokument: {text}
  Sažetak: {target_text}
  ```

- Instruction-tuned prompt template:

  ```text
  Dokument: {text}

  Napišite sažetak gorenavedenog dokumenta.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset lr-sum-sr
```
