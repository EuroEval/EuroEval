# 🇧🇬 Bulgarian

This is an overview of all the datasets used in the Bulgarian part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### Cinexio

This data was published in [this paper](https://aclanthology.org/2023.acl-long.487/)
and contains movie reviews in Bulgarian from the Cinexio ticket-booking website
(which is not available anymore).

The original dataset contains 7,118 / 807 / 856 samples for the training, validation,
and test splits, respectively. We use 1,024 / 256 / 2,048 samples for our training,
validation and test splits, respectively. The train and validation splits are subsets of
the original splits, while the test split is created using additional samples from the
train split.

The original dataset uses a 0-5 rating scale (with 0.5 increments)
which is converted to three sentiment categories:

- [0, 1.5] ➡️ `negative`
- [2.0, 3.5] ➡️ `neutral`
- [4.0, 5.0] ➡️ `positive`

Here are a few examples from the training split:

```json
{
    "text": "Страхотен филм,също като първа част! Става и за малки и за големи.",
    "label": "positive"
}
```

```json
{
    "text": "Рядко тъп филм. Филмът на актьора-тийн-секс-идол Ники Илиев е с крайно нелеп сюжет, а актьорска игра е неадекватна.",
    "label": "negative"
}
```

```json
{
    "text": "Поредена бозарика. Празно и несилно.",
    "label": "neutral"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Следват документи и техният сентимент, който може да бъде позитивен, неутрален или негативен.
  ```

- Base prompt template:

  ```text
  Документ: {text}
  Сентимент: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Документ: {text}

  Класифицирайте сентимента в документа. Отговорете с позитивен, неутрален, или негативен, и нищо друго.
  ```

- Label mapping:
  - `positive` ➡️ `позитивен`
  - `neutral` ➡️ `неутрален`
  - `negative` ➡️ `негативен`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset cinexio
```

## Named Entity Recognition

### BG-NER-BSNLP

This dataset was published in [this paper](https://aclanthology.org/W19-3709/)
and consists of Web documents.

The original dataset consists of 6,701 / 2,179 samples for the
training and test splits, respectively. We use 1,024 / 256 / 2,048
samples for our training, validation and test splits, respectively. The train and
tests splits are subsets of the original splits, and the validation split is
created from the training split.

Here are a few examples from the training split:

```json
{
    "tokens": ["За", "всички", "нас", "е", "по-добре", "да", "постигнем", "споразумение", "възможно", "най-скоро", "\"", ",", "каза", "Захариева", "."],
    "labels": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-PER", "O"]
}
```

```json
{
    "tokens": ["Но", "съм", "уверена", ",", "че", "можем", "да", "сключим", "споразумение", "\"", "."],
    "labels": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]
}
```

```json
{
    "tokens": ["Според", "тях", "130_000", "граждани", "на", "ЕС", "са", "напуснали", "страната", "до", "септември", "миналата", "година", ",", "което", "е", "най-големият", "брой", "от", "2008", "г.", "досега", "."],
    "labels": ["O", "O", "O", "O", "O", "B-ORG", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  По-долу са изречения и JSON речници с именуваните обекти, които се срещат в дадените изречения.
  ```

- Base prompt template:

  ```text
  Изречение: {text}
  Именувани обекти: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Изречение: {text}

  Идентифицирайте именуваните обекти в изречението. Трябва да изведете това като JSON речник с ключовете 'лице', 'място', 'организация' и 'разни'. Стойностите трябва да бъдат списъци на именуваните обекти от този тип, точно както се появяват в изречението.
  ```

- Label mapping:
  - `B-PER` ➡️ `лице`
  - `I-PER` ➡️ `лице`
  - `B-LOC` ➡️ `място`
  - `I-LOC` ➡️ `място`
  - `B-ORG` ➡️ `организация`
  - `I-ORG` ➡️ `организация`
  - `B-MISC` ➡️ `разни`
  - `I-MISC` ➡️ `разни`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset bg-ner-bsnlp
```

## Linguistic Acceptability

### ScaLA-bg

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Bulgarian Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Bulgarian-BTB) by assuming that
the documents in the treebank are correct, and corrupting the samples to create
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
    "text": "Затова съвсем неслучайно в интервю по БиТиВи Първанов не забрави да се похвали, че столетницата има 240 хиляди души членска маса.",
    "label": "correct"
}
```

```json
{
    "text": "Формулата на нашето време рязко и ясно се очертава с лаконичното противоречие: прогрес технологията, регрес в морала.",
    "label": "incorrect"
}
```

```json
{
    "text": "Част от фойерверките са купени от сергиите, където часове преди полунощ (по време на официалната забрана) те се продаваха свободно.",
    "label": "correct"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Следват изречения и дали са граматически правилни.
  ```

- Base prompt template:

  ```text
  Изречение: {text}
  Граматически правилно: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Изречение: {text}

  Определете дали изречението е граматически правилно или не. Отговорете с 'да', ако е правилно, и 'не', ако не е. Отговорете само с тази дума, и нищо друго.
  ```

- Label mapping:
  - `correct` ➡️ `да`
  - `incorrect` ➡️ `не`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-bg
```

## Reading Comprehension

### MultiWikiQA-bg

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
    "context": "„Нихоншоки“ (; по английската Система на Хепбърн Nihonshoki) или „Нихонги“ (, Nihongi), „Японска хроника“ е една от трите книги смятани за свещени за шинтоизма, заедно с „Коджики“ (, Kojiki) и „Куджики“ (, Kujiki). Считана е за първата същинска японска историография и проследява историята на Япония от сътворението на архипелага до управлението на императрица Джито (, Jitō-tennō) (686\xa0– 697).\n\nИстория \nПаметникът е завършен през 720 г. Съставен е по нареждане и под ръководството на принц Тонери (, Toneri-shinō) заедно с другия главен съставител Фудживара-но Фухито (, Fujiwara-no Fuhito). За образец на съставителите служат древните китайски исторически хроники. Интересен е фактът, че „Нихоншоки“ е написан едва 8 години след съставянето на „Коджики“\xa0– документ с твърде сходно съдържание на неговото. Една от вероятните причини е, че „Нихоншоки“ е написан изцяло на китайски език, което по това време устройвало повече целите на Японския императорски двор.\n\nСъдържание \n„Нихоншоки“ се състои от тридесет свитъка (, maki), разделени на три части. Документът основно се концентрира върху историчски събития и факти, както и върху генеалогията на императорите. Първата част започва с два свитъка, които се занимават с японските митове за сътворението и последвалите го събития. А следващите свитъци дават началото на проследяването на историята на Япония по хронологичен ред на управление на императори, като започват с легендарния първи император Джимму (, Jimmu-tennō). Втората част от свитъци е за управлението на император Темму (, Temmu-tennō) (673\xa0– 686). Третата и последна част обхваща управлението на императрица Джито.\n\nВ документа присъстват и 129 песни/поеми „ута“ (, uta). 50 от тях съвпадат с песните от „Коджики“, а останалите 79 се срещат за първи път. За записването им е използвана същата писмена система, както и в „Коджики“ – Маньогана (,\u3000Man'yōgana) т.е. ползване на китайски писмени знаци само с фонетичната им стойност, не като идеограми.\n\nИзточници \n Калве, Робер „Японците: историята на един народ“. Рива (2005)\n Цигова, Бойка „Пътят на словото в Япония“. Университетско издателство „Св. Климент Охридски“ (2006)\n\nВижте също \n Коджики\n\nВъншни препратки \n  The Internet Sacred Text Archive – Преводи на японски шинтоистки текстове, сред които и на „Нихоншоки“, на английски.\n  和漢籍の書棚  – Текстът на „Нихоншоки“ на японски.\n\nЯпонска литература\nШинтоизъм",
    "question": "Колко на брой свитъка има в „Нихоншоки“?",
    "answers": {
        'answer_start': [1002],
        "text": ["тридесет"]
    }
}
```

```json
{
    "context": "5 старс или срещано на латиница като 5 stars е музикално шоу излъчвано по bTV. Водещи са Светозар Христов, Лора Владова, Владимир Димов, Надя Казакова и Любена Нинова, които са открития от Шоуто на Слави от музикалните проекти „Аз пея в Ку-ку бенд“ през 2004 г. и „Музикална академия Ку-ку бенд“ през 2005 г. В първия сезон на предаването водещ е и Росен Петров.\n\nСезони\n\nПърви сезон \nПървото предаване е излъчено на 11 април 2005 и са излъчени 12 епизода. Първия сезон на шоуто се снима в зала 2 на НДК. В шоуто гостуват известни личности, а петимата водещи им дават кратко интервю, след което изпълняват 5-те му любими песни.\n\nВтори сезон \nВтория сезон на шоуто се излъчва през есента на същата година под името „5 stars: Продължението“. Той е заснет във виртуално студио с 3D ефекти, като сюжета на шоуто е сменен. В него водещите стават репортери и интервюират хора от различни социални групи, съсловия и гилдии за изпълнителя. Предпочитаните песни са определени чрез проучване от специално наета социологическа агенция. В предаването гостува виден представител на анализираната група, а за негово удоволствие пее някой от фаворитите на гилдията.\n\nТрети сезон \nТретия и последен сезон на шоуто е през пролетта на 2006 г., отново пред публика. С началото на третия сезон предаването променя интрото и мелодията.\n\nВъншни препратки \n\nБългарски реалити предавания\nПредавания на БТВ\nОсновани в България през 2005 година",
    "question": "Как се избират любимите песни от втори сезон?",
    "answers": {
        "answer_start": [967],
        "text": ["чрез проучване от специално наета социологическа агенция"]
        }
}
```

```json
{
    "context": "Селскостопанската академия (съкратено ССА) е научна организация със седалище в София, България.\n\nПризвана е да извършва научни изследвания и приложна дейност в областта на земеделието, животновъдството и хранителната промишленост. Има право да подготвя докторанти по съответните научни дисциплини.\n\nИстория \nПо примера на Всесъюзната академия на селскостопанските науки „Ленин“ в Москва като български аналог е създадена отрасловата научна академия под името Академия на селскостопанските науки (АСН) със седалище в София през 1961 г. Тя има физически лица за членове, наричани с научното звание „академик“ (действителен член). Закрита е през 1971 г., като нейните членове и учени преминават към общонаучната БАН.\n\nСамо след година (1972) е създадена научно-образователната Селскостопанска академия „Георги Димитров“. Тя обединява научни институти и всички висши училища в страната, подготвящи специалисти в посочените най-горе научни направления. Закрита е след само 3 години (1975). Подобна научно-образователна функция и структура има и повече просъществувала обединена Медицинска академия (1972 – 1990).\n\nОтново е учредена Селскостопанска академия с Указ 1008 (ДВ, бр. 34) на Държавния съвет през 1982 г. ССА обаче – за разлика от предишната ССА „Г. Димитров“, е само научна организация и не се занимава с обучение на студенти. В нея няма членство на учени, каквото има в първоначалната АСН.\n\nАкадемията е преобразувана (ДВ, бр. 113/1999) в Национален център за аграрни науки (НЦАН) в края на 1999 г.\n\nНЦАН е преименуван (ДВ, бр. 43/2008) с предишното название Селскостопанска академия през 2008 г. През лятото на 2018 г. е приет новият Устройствен правилник на ССА, с който се внасят важни изменения в нейната структура. Въведени са научните звания академик и член-кореспондент от 1 август 2018 г. Академичният състав включва: академици, член-кореспонденти, чуждестранни членове и почетни членове на Академията, хабилитирани и нехабилитирани учени, доктори на науките и доктори от системата на Академията, докторанти\n\nСтруктура \n Централно управление\nСелскостопанската академия се ръководи от управителен съвет с мандат от 4 години начело с председател, подпомаган от заместник-председател и главен научен секретар.\n\nПредседателят на Управителния съвет е също председател на ССА и председател на Изпълнителното бюро, което включва още заместник-председателя и главния научен секретар.\n\nКоординати: адрес: 1373, гр. София, ул. „Суходолска“ 30; тел.: 02/9299481; Факс: 02/9202067; e-mail: aa@acad.bg\n\n Научни институти\n\n Други звена\n „Система за агропазарна информация“ (САПИ), София (държавно предприятие към ССА)\n 19 опитни станции (самостоятелни държавни предприятия) – във Варна, Видин, Враца, Кърджали, Лозница, Лом, Павликени, Пазарджик, Поморие, Самоков, Септември, Силистра, Сливен, Смолян, Средец, Търговище, с. Хан Крум (Област Шумен), Хасково, Ямбол.\n 2 експериментални бази (държавни предприятия към институти)\n Национален земеделски музей, София\n\nИзточници\n\nВъншни препратки \n Официален сайт\n Закон за Селскостопанската академия (заглавие изм., ДВ, бр. 43/2008) – обн., ДВ, бр. 113/1999; изм., ДВ, бр. 15/2003; изм., ДВ, бр. 43/2008; изм., ДВ, бр. 54/2008; изм., ДВ, бр. 10/2009; изм., ДВ, бр. 74/2009; изм., ДВ, бр. 99/2009; изм., ДВ, бр. 78/2010\n Устройствен правилник на Селскостопанската академия – приет с ПМС 226 от 15 септември 2008; обн., ДВ, бр. 83/2008; изм., ДВ, бр. 42/2009; изм., ДВ, бр. 79/2009; изм., ДВ, бр. 84/2010; изм., ДВ, бр. 101/2010; изм. и доп., ДВ, бр. 14/2012",
    "question": "Къде е главната квартира на Академията за селско стопанство?",
    "answers": {
        "answer_start": [79],
        "text": ["София, България"]
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Следват текстове със съответни въпроси и отговори.
  ```

- Base prompt template:

  ```text
  Текст: {text}
  Въпрос: {question}
  Отговор с максимум 3 думи:
  ```

- Instruction-tuned prompt template:

  ```text
  Текст: {text}

  Отговорете на следния въпрос относно текста по-горе с максимум 3 думи.

  Въпрос: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-bg
```

## Knowledge

### Exams-bg

This dataset was published in [this paper](https://aclanthology.org/2023.acl-long.487/)
and contains questions collected from high school (HS) examinations in Bulgaria.

The original full dataset consists of 1,329 / 365 / 1,472 samples for
training, validation and testing, respectively. We only keep samples that have 4 choices,
and we thus use a 1,024 / 94 / 2,048 split for training, validation and testing,
respectively. The train and validation set are sampled from the original splits, but
the test set has additional samples from both the original train and validation sets.

Here are a few examples from the training split:

```json
{
    "text": "При свързването на три аминокиселини се образува:\nВъзможности:\na. тризахарид\nb. трипептид\nc. тринуклеотид\nd. триглицерид",
    "label": "b"
}
```

```json
{
    "text": "През 1911 г. Българското книжовно дружество се преименува на:\nВъзможности:\na. Народна библиотека „Кирил и Методий”\nb. Софийски държавен университет\nc. Българска академия на науките\nd. Висше педагогическо училище",
    "label": "c"
}
```

```json
{
    "text": "Коя земеделска култура се отглежда само в Южна България?\nВъзможности:\na. тютюн\nb. слънчоглед\nc. ориз\nd. царевица",
    "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Следват въпроси с множествен избор (с отговори).
  ```

- Base prompt template:

  ```text
  Въпрос: {text}
  Възможности:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Отговор: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Въпрос: {text}

  Отговорете на горния въпрос като отговорите с 'a', 'b', 'c' или 'd', и нищо друго.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset exams-bg
```

### Unofficial: INCLUDE-bg

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
    "text": "Първият от два монохроматични източника на светлина излъчва сноп с дължина на вълната nm 630λ1 = , а вторият – с nm 450λ2 = . По тези данни можем да твърдим, че двата снопа са:\nВъзможности:\na. с различен цвят\nb. кохерентни\nc. с еднакъв цвят\nd. успоредни",
    "label": "a",
    "subject": "Physics"
}
```

```json
{
    "text": "При кой тип стопанство решенията за производство се взимат в зависимост от  регулиращата роля на пазара, в условията на конкуренция и свободна лична   инициатива?\nВъзможности:\na. централизирано\nb. пазарно\nc. аграрно\nd. натурално",
    "label": "b",
    "subject": "Geography"
}
```

```json
{
    "text": "Теорията, която приема генетичния фактор за основен при формирането на личността, наричаме:\nВъзможности:\na. социогенетична\nb. когнитивна\nc. биогенетична\nd. психогенегична",
    "label": "c",
    "subject": "Philosophy"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Следват въпроси с множествен избор (с отговори).
  ```

- Base prompt template:

  ```text
  Въпрос: {text}
  Отговор: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Въпрос: {text}

  Отговорете на горния въпрос като отговорите с {labels_str}, и нищо друго.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset include-bg
```

## Common-sense Reasoning

### Winogrande-bg

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2506.19468)
and is a translated and filtered version of the English [Winogrande
dataset](https://doi.org/10.1145/3474381).

The original full dataset consists of 47 / 1,210 samples for training and testing, and
we use 128 of the test samples for validation, resulting in a 47 / 128 / 1,085 split for
training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
    "text": "Не можех да контролирам влагата както контролирах дъжда, защото _ идваше отвсякъде. На какво се отнася празното място _?\nВъзможности:\na. влага\nb. дъжд",
    "label": "a"
}
```

```json
{
    "text": "Джесика смяташе, че "Sandstorm" е най-великата песен, писана някога, но Патриция я мразеше. _ купи билет за джаз концерта. На какво се отнася празното място _?\nВъзможности:\na. Джесика\nb. Патриция",
    "label": "b"
}
```

```json
{
    "text": "Термостатът показа, че долу е двадесет градуса по-хладно, отколкото горе, затова Байрон остана в _ защото му беше студено. На какво се отнася празното място _?\nВъзможности:\na. долу\nb. горе",
    "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Следват въпроси с множествен избор (с отговори).
  ```

- Base prompt template:

  ```text
  Въпрос: {text}
  Възможности:
  a. {option_a}
  b. {option_b}
  Отговор: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Въпрос: {text}
  Възможности:
  a. {option_a}
  b. {option_b}

  Отговорете на горния въпрос като отговорите с 'a' или 'b', и нищо друго.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset winogrande-bg
```

## Hallucination Detection

### MultiWikiHalluQA-bg

This dataset uses the same data as [MultiWikiQA-bg](#multiwikiqa-bg), published in
[this paper](https://doi.org/10.48550/arXiv.2509.04111), containing Wikipedia articles
with LLM-generated questions and answers in 300+ languages. Rather than evaluating the
correctness of the generated answer, this task evaluates the degree to which the model
hallucinates, i.e., generates tokens that are not grounded in the provided context.

The hallucination detection is performed using the
[LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect) library, which uses a
transformer-based classifier to predict hallucination at the token level. The metric
reported is the hallucination rate, computed as the ratio of hallucinated tokens to
total tokens in the generated answers.

Here are a few examples from the training split:

```json
{
    "context": "„Нихоншоки“ (; по английската Система на Хепбърн Nihonshoki) или „Нихонги“ (, Nihongi), „Японска хроника“ е една от трите книги смятани за свещени за шинтоизма, заедно с „Коджики“ (, Kojiki) и „Куджики“ (, Kujiki). Считана е за първата същинска японска историография и проследява историята на Япония от сътворението на архипелага до управлението на императрица Джито (, Jitō-tennō) (686\xa0– 697).\n\nИстория \nПаметникът е завършен през 720 г. Съставен е по нареждане и под ръководството на принц Тонери (, Toneri-shinō) заедно с другия главен съставител Фудживара-но Фухито (, Fujiwara-no Fuhito). За образец на съставителите служат древните китайски исторически хроники. Интересен е фактът, че „Нихоншоки“ е написан едва 8 години след съставянето на „Коджики“\xa0– документ с твърде сходно съдържание на неговото. Една от вероятните причини е, че „Нихоншоки“ е написан изцяло на китайски език, което по това време устройвало повече целите на Японския императорски двор.\n\nСъдържание \n„Нихоншоки“ се състои от тридесет свитъка (, maki), разделени на три части. Документът основно се концентрира върху историчски събития и факти, както и върху генеалогията на императорите. Първата част започва с два свитъка, които се занимават с японските митове за сътворението и последвалите го събития. А следващите свитъци дават началото на проследяването на историята на Япония по хронологичен ред на управление на императори, като започват с легендарния първи император Джимму (, Jimmu-tennō). Втората част от свитъци е за управлението на император Темму (, Temmu-tennō) (673\xa0– 686). Третата и последна част обхваща управлението на императрица Джито.\n\nВ документа присъстват и 129 песни/поеми „ута“ (, uta). 50 от тях съвпадат с песните от „Коджики“, а останалите 79 се срещат за първи път. За записването им е използвана същата писмена система, както и в „Коджики“ – Маньогана (,\u3000Man'yōgana) т.е. ползване на китайски писмени знаци само с фонетичната им стойност, не като идеограми.\n\nИзточници \n Калве, Робер „Японците: историята на един народ“. Рива (2005)\n Цигова, Бойка „Пътят на словото в Япония“. Университетско издателство „Св. Климент Охридски“ (2006)\n\nВижте също \n Коджики\n\nВъншни препратки \n  The Internet Sacred Text Archive – Преводи на японски шинтоистки текстове, сред които и на „Нихоншоки“, на английски.\n  和漢籍の書棚  – Текстът на „Нихоншоки“ на японски.\n\nЯпонска литература\nШинтоизъм",
    "question": "Колко на брой свитъка има в „Нихоншоки“?",
    "answers": {
        'answer_start': [1002],
        "text": ["тридесет"]
    }
}
```

```json
{
    "context": "5 старс или срещано на латиница като 5 stars е музикално шоу излъчвано по bTV. Водещи са Светозар Христов, Лора Владова, Владимир Димов, Надя Казакова и Любена Нинова, които са открития от Шоуто на Слави от музикалните проекти „Аз пея в Ку-ку бенд“ през 2004 г. и „Музикална академия Ку-ку бенд“ през 2005 г. В първия сезон на предаването водещ е и Росен Петров.\n\nСезони\n\nПърви сезон \nПървото предаване е излъчено на 11 април 2005 и са излъчени 12 епизода. Първия сезон на шоуто се снима в зала 2 на НДК. В шоуто гостуват известни личности, а петимата водещи им дават кратко интервю, след което изпълняват 5-те му любими песни.\n\nВтори сезон \nВтория сезон на шоуто се излъчва през есента на същата година под името „5 stars: Продължението“. Той е заснет във виртуално студио с 3D ефекти, като сюжета на шоуто е сменен. В него водещите стават репортери и интервюират хора от различни социални групи, съсловия и гилдии за изпълнителя. Предпочитаните песни са определени чрез проучване от специално наета социологическа агенция. В предаването гостува виден представител на анализираната група, а за негово удоволствие пее някой от фаворитите на гилдията.\n\nТрети сезон \nТретия и последен сезон на шоуто е през пролетта на 2006 г., отново пред публика. С началото на третия сезон предаването променя интрото и мелодията.\n\nВъншни препратки \n\nБългарски реалити предавания\nПредавания на БТВ\nОсновани в България през 2005 година",
    "question": "Как се избират любимите песни от втори сезон?",
    "answers": {
        "answer_start": [967],
        "text": ["чрез проучване от специално наета социологическа агенция"]
        }
}
```

```json
{
    "context": "Селскостопанската академия (съкратено ССА) е научна организация със седалище в София, България.\n\nПризвана е да извършва научни изследвания и приложна дейност в областта на земеделието, животновъдството и хранителната промишленост. Има право да подготвя докторанти по съответните научни дисциплини.\n\nИстория \nПо примера на Всесъюзната академия на селскостопанските науки „Ленин“ в Москва като български аналог е създадена отрасловата научна академия под името Академия на селскостопанските науки (АСН) със седалище в София през 1961 г. Тя има физически лица за членове, наричани с научното звание „академик“ (действителен член). Закрита е през 1971 г., като нейните членове и учени преминават към общонаучната БАН.\n\nСамо след година (1972) е създадена научно-образователната Селскостопанска академия „Георги Димитров“. Тя обединява научни институти и всички висши училища в страната, подготвящи специалисти в посочените най-горе научни направления. Закрита е след само 3 години (1975). Подобна научно-образователна функция и структура има и повече просъществувала обединена Медицинска академия (1972 – 1990).\n\nОтново е учредена Селскостопанска академия с Указ 1008 (ДВ, бр. 34) на Държавния съвет през 1982 г. ССА обаче – за разлика от предишната ССА „Г. Димитров“, е само научна организация и не се занимава с обучение на студенти. В нея няма членство на учени, каквото има в първоначалната АСН.\n\nАкадемията е преобразувана (ДВ, бр. 113/1999) в Национален център за аграрни науки (НЦАН) в края на 1999 г.\n\nНЦАН е преименуван (ДВ, бр. 43/2008) с предишното название Селскостопанска академия през 2008 г. През лятото на 2018 г. е приет новият Устройствен правилник на ССА, с който се внасят важни изменения в нейната структура. Въведени са научните звания академик и член-кореспондент от 1 август 2018 г. Академичният състав включва: академици, член-кореспонденти, чуждестранни членове и почетни членове на Академията, хабилитирани и нехабилитирани учени, доктори на науките и доктори от системата на Академията, докторанти\n\nСтруктура \n Централно управление\nСелскостопанската академия се ръководи от управителен съвет с мандат от 4 години начело с председател, подпомаган от заместник-председател и главен научен секретар.\n\nПредседателят на Управителния съвет е също председател на ССА и председател на Изпълнителното бюро, което включва още заместник-председателя и главния научен секретар.\n\nКоординати: адрес: 1373, гр. София, ул. „Суходолска“ 30; тел.: 02/9299481; Факс: 02/9202067; e-mail: aa@acad.bg\n\n Научни институти\n\n Други звена\n „Система за агропазарна информация“ (САПИ), София (държавно предприятие към ССА)\n 19 опитни станции (самостоятелни държавни предприятия) – във Варна, Видин, Враца, Кърджали, Лозница, Лом, Павликени, Пазарджик, Поморие, Самоков, Септември, Силистра, Сливен, Смолян, Средец, Търговище, с. Хан Крум (Област Шумен), Хасково, Ямбол.\n 2 експериментални бази (държавни предприятия към институти)\n Национален земеделски музей, София\n\nИзточници\n\nВъншни препратки \n Официален сайт\n Закон за Селскостопанската академия (заглавие изм., ДВ, бр. 43/2008) – обн., ДВ, бр. 113/1999; изм., ДВ, бр. 15/2003; изм., ДВ, бр. 43/2008; изм., ДВ, бр. 54/2008; изм., ДВ, бр. 10/2009; изм., ДВ, бр. 74/2009; изм., ДВ, бр. 99/2009; изм., ДВ, бр. 78/2010\n Устройствен правилник на Селскостопанската академия – приет с ПМС 226 от 15 септември 2008; обн., ДВ, бр. 83/2008; изм., ДВ, бр. 42/2009; изм., ДВ, бр. 79/2009; изм., ДВ, бр. 84/2010; изм., ДВ, бр. 101/2010; изм. и доп., ДВ, бр. 14/2012",
    "question": "Къде е главната квартира на Академията за селско стопанство?",
    "answers": {
        "answer_start": [79],
        "text": ["София, България"]
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Следват текстове със съответни въпроси и отговори.
  ```

- Base prompt template:

  ```text
  Текст: {text}
  Въпрос: {question}
  Отговор с максимум 3 думи:
  ```

- Instruction-tuned prompt template:

  ```text
  Текст: {text}

  Отговорете на следния въпрос относно текста по-горе с максимум 3 думи.

  Въпрос: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-hallucination-qa-bg
```
