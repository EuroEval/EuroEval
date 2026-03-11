# 🇧🇾 Belarusian

This is an overview of all the datasets used in the Belarusian part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### BeSLS

This dataset was introduced in [this paper](https://aclanthology.org/2025.acl-long.25/).
It comprises 2,000 sentences that have been manually annotated for sentiment polarity:
positive (1) or negative (0).

The original split of the dataset consists of 1,500 samples for training, 250 for
validation, and 250 for testing. In EuroEval, we use 256 samples for training, 128 for
validation, and 1,616 for testing. The train and validation splits are subsets of the
original train/validation splits, while the test split includes the remaining samples
from the original training and validation sets.

Here are a few examples from the training split:

```json
{
  "text": "Пры вельмі сціплым бюджэце ў 20 млн даляраў Стахельскі зняў эталонны экшэн.",
  "label": "positive",
}
```

```json
{
    "text": "Гэта лічба толькі пацвярджае, што фестываль з кожным годам набірае моцы, пашыраючы сваю геаграфію.",
    "label": "positive",
}
```

```json
{
    "text": "Яна цудоўна абудзіла апетыт, апетыт да падрабязнасцяў, да разгадвання, да спазнання.",
    "label": "positive",
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Ніжэй прыведзены дакументы і іх сентымент, які можа быць 'станоўчы', 'нейтральны' або 'адмоўны'.
  ```

- Base prompt template:

  ```text
  Дакумент: {text}
  Сентымент: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Дакумент: {text}

  Класіфікуйце сентымент у дакуменце. Адкажыце толькі 'станоўчы', 'нейтральны' або 'адмоўны', і нічога іншага.
  ```

- Label mapping:
  - `positive` ➡️ `станоўчы`
  - `neutral` ➡️ `нейтральны`
  - `negative` ➡️ `адмоўны`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset besls
```

## Named Entity Recognition

### WikiANN-be

This dataset was published in [this paper](https://aclanthology.org/P17-1178/) and is
part of a cross-lingual named entity recognition framework for 282 languages from
Wikipedia. It uses silver-standard annotations transferred from English through
cross-lingual links and performs both name tagging and linking to an english Knowledge
Base.

The original full dataset consists of 15,000 / 1,000 / 1,000 samples for the training,
validation and test splits, respectively. We use 1,024 / 256 / 1,000 samples for our
training, validation and test splits, respectively. All the new splits are subsets of
the original splits.

Here are a few examples from the training split:

```json
{
  "tokens": ["Сцюарт", "Бінэм", "(", "4", ")"],
  "labels": ["B-PER", "I-PER", "O", "O", "O"],
}
```

```json
{
  "tokens": ["Пасля", "гуляў", "таксама", "за", "моладзевую", "зборную", "Беларусі", "."],
  "labels": ["O", "O", "O", "O", "B-ORG", "I-ORG", "I-ORG", "O"],
}
```

```json
{
  "tokens": ["Горад", "Кампен", ",", "Нідэрланды"],
  "labels": ["B-LOC", "I-LOC", "I-LOC", "I-LOC"],
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Ніжэй прыведзены сказы і JSON-слоўнікі з іменаванымі сутнасцямі, якія прысутнічаюць у дадзеным сказе.
  ```

- Base prompt template:

  ```text
  Сказ: {text}
  Іменаваныя сутнасці: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Сказ: {text}

  Ідэнтыфікуйце іменаваныя сутнасці ў сказе. Вы павінны вывесці гэта як JSON-слоўнік з ключамі 'асоба', 'месца', 'арганізацыя' і 'рознае'. Значэнні павінны быць спісамі іменаваных сутнасцей гэтага тыпу, дакладна такімі, як яны з'яўляюцца ў сказе.
  ```

- Label mapping:
  - `B-PER` ➡️ `асоба`
  - `I-PER` ➡️ `асоба`
  - `B-LOC` ➡️ `месца`
  - `I-LOC` ➡️ `месца`
  - `B-ORG` ➡️ `арганізацыя`
  - `I-ORG` ➡️ `арганізацыя`
  - `B-MISC` ➡️ `рознае`
  - `I-MISC` ➡️ `рознае`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset wikiann-be
```

## Linguistic Acceptability

### ScaLA-be

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Belarusian Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Belarusian-HSE) by assuming that
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
    "text": "Скончыла Беларускую акадэмію мастацтваў (курс Міхаіла Жданоўскага) і курс дакументальнага кіно Doc Pro у Школе Вайды (Варшава).",
    "label": "correct"
}
```

```json
{
    "text": "Дзяржаўныя СМІ не расказалі пра тыя рэкамэндацыі WHO, якіх Беларусь не выконвае",
    "label": "correct"
}
```

```json
{
    "text": "Але праз 19 гадоў Статут новы ВКЛ скасаваў большасьць палажэньняў Люблінскай уніі.",
    "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Ніжэй прыведзены сказы і ці з'яўляюцца яны граматычна правільнымі.
  ```

- Base prompt template:

  ```text
  Сказ: {text}
  Граматычна правільны: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Сказ: {text}

  Вызначце, ці сказ граматычна правільны ці не. Адкажыце толькі {labels_str}, і нічога іншага.
  ```

- Label mapping:
  - `correct` ➡️ `так`
  - `incorrect` ➡️ `не`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-be
```

## Reading Comprehension

### MultiWikiQA-be

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
  "context": "Кавальская Слабада () — станцыя першай чаргі Зеленалужскай лініі Мінскага метрапалітэна, паміж станцыямі «Вакзальная» і «Аэрадромная».\n\nНазва станцыі звязаная з існаваннем у гэтым раёне рамесніцкага прадмесця, дзе жылі кавалі. Рамесніцтва стала асноўным матывам афармлення станцыі, створанага скульптарамі П. Вайніцкім, М. Тарлецкай і мастачкай М. Жвірбля.\n\nСтанцыя размешчана на скрыжаванні вуліц Жукоўскага і Варанянскага і ўведзена ў эксплуатацыю 6 лістапада 2020 года.\n\nКанструкцыя станцыі – скляпеністага тыпу. Пасажырская платформа знаходзіцца пасярэдзіне, з абодвух бакоў размяшчаюцца калеі. Архітэктары праекту – У. Целяпнёў, В. Целяпнёва, А. Кузьміч, Г. Васільеў.\n\nДля ўзмацнення бяспекі пасажыраў абапал станцыйнай платформы ўсталяваны празрыстыя брамкі абсталяваныя аўтаматычным механізмам адкрыцця-закрыцця, якія адкрываюцца толькі для высадкі і пасадкі пасажыраў. Станцыя абсталявана таксама ліфтам і прыстасавана да патрэб асоб з абмежаванымі фізічнымі магчымасцямі.\n\nСтанцыі Мінскага метрапалітэна\nЗеленалужская лінія метро\n2020 год у Мінску\nЗ’явіліся ў 2020 годзе",
  "question": "Хто распрацаваў праект станцыі?",
  "answers": {
    "answer_start": [622],
    "text": ["У. Целяпнёў, В. Целяпнёва, А. Кузьміч, Г. Васільеў"]
  }
}
```

```json
{
  "context": "Каралеўства Ісландыя (, ) — гістарычная дзяржава, існаваўшая ў 1918 – 1944 у асабістай уніі з Даніяй.\n\nЗ 1380 Ісландыя практычна ўвесь час знаходзілася пад кантролем дацкага манарха (да 1814 фармальна належала Нарвегіі). У 1874 ёй ўпершыню была нададзена ўласная Канстытуцыя і самакіраванне, якое было пашыранае ў 1904. 1 снежня 1918 быў падпісаны Акт аб уніі, паводле якога Ісландыя прызнавалася як цалкам незалежная дзяржава на чале з дацкім каралём Крысціянам X.\n\nПасля акупацыі Даніі Нацысцкай Германіяй 9 красавіка 1940 сувязь з дацкім урадам была парушаная. 10 красавіка ўлада над Ісландыяй перайшла да мясцовага парламента – Альтынга, які выбраў часовым кіраўніком краіны Свейна Б'ёрнсана.\n\nПасля разрыву з Даніяй Ісландыя заняла пазіцыю нейтралітэту, не падтрымаўшы ні Вось, ні Саюзнікаў. Але 10 мая 1940 Вялікабрытанія пачала ўварванне ў Ісландыю, высадзіўшы 800 вайскоўцаў у Рэйк’явіку. Ісландскі ўрад выразіў пратэст супраць «гвалтоўнага парушэння» нейтралітэту, але ўжо 17 мая востраў быў цалкам акупаваны брытанскімі войскамі, а на тэрыторыю краіны дэсантавалася яшчэ 4.000, і прэм'ер-міністр Ісландыі Герман Ёўнасан звярнуўся па радыё да грамадзян краіны з просьбай ветліва аднесціся да брытанскіх салдат. У ліпні 1941 кантроль над востравам перайшоў да ЗША, бо Брытаніі патрабавалася армія ў іншых месцах. Да канца Другой сусветнай вайны Ісландыя заставалася пад акупацыяй Саюзнікаў.\n\n24 мая 1944 адбыўся рэферэндум, на якім большая частка прагаласавала за незалежнасць ад Даніі і рэспубліканскую форму кіравання. Першым прэзідэнтам Ісландыі стаў Свейн Б’ёрнсан. Тым часам Данія заставалася пад нацысцкай акупацыяй, і многія датчане былі абражаныя тым, што гэтае рашэнне было прынятае ў такі час. Тым не менш, дацкі кароль Крысціян X накіраваў віншавальную тэлеграму маладой рэспубліцы.\n\nГістарычныя дзяржавы Еўропы\nГісторыя Ісландыі",
  "question": "Калі ЗША ўзялі пад свой кантроль Ісландыю?",
  "answers": {
    "answer_start": [1228],
    "text": ["1941"]
  }
}
```

```json
{
  "context": "Ка́ртлі () — адна з асноўных гісторыка-геаграфічных абласцей Грузіі, калыска яе дзяржаўнасці. З адміністрацыйнага пункта гледжання, Картлі складаецца з Квема-Картлі і Шыда-Картлі. Частка тэрыторыі адносіцца да Мцхета-Мтыянецкага і Самцхэ-Джавахецкага краёў. Горад Тбілісі таксама знаходзіцца на тэрыторыі гістарычнай вобласці Картлі. Тэрыторыя Шыда-Картлі ўключае тэрыторыю былой Паўднёва-Асяцінскай аўтаномнай вобласці, якая цяпер непадкантрольная ўладам Грузіі.\n\nКартлійская гаворка ляжыць у аснове грузінскай мовы. У сучаснай грузінскай мове лінгвісты вылучаюць картлійскі дыялект.\n\nНа тэрыторыі Картлі размешчаны гарады: Тбілісі, Руставі, Горы, Хашуры, Карэлі, Каспі, Цхінвалі, Ахалгоры, Балнісі, Марнеулі, Мцхета, Душэты, Баржомі.\n\nУ вобласці шмат гістарычных помнікаў найстаражытнай і найноўшай гісторыі Грузіі.\n\nДля Картлі характэрны неаднастайны этнічны склад насельніцтва. Пераважае грузінскае насельніцтва, таксама маюцца кампактныя арэалы рассялення армян, азербайджанцаў, асецін і грэкаў.\n\nСпасылкі \n\n Этнакаўказ. Этнакарта Картлі 1959 г.\n\nГістарычныя вобласці Грузіі",
  "question": "З якіх гісторыка-геаграфічных рэгіёнаў складаецца Картлі ў адміністрацыйным плане?",
  "answers": {
    "answer_start": [152],
    "text": ["Квема-Картлі і Шыда-Картлі"]
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

```text
Ніжэй прыведзены тэксты з адпаведнымі пытаннямі і адказамі.
```

- Base prompt template:

```text
Тэкст: {text}
Пытанне: {question}
Адказ максімум 3 словамі: {label}
```

- Instruction-tuned prompt template:

```text
Тэкст: {text}

Адкажыце на наступнае пытанне пра тэкст вышэй максімум 3 словамі.

Пытанне: {question}
```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-be
```

## Common-sense Reasoning

### BE-WSC

This dataset was published in
[this paper](https://aclanthology.org/2025.acl-long.25/) and is a Belarusian
version of the Winograd schema challenge (WSC).

The original full dataset consists of 570 / 200 / 200 samples for training,
validation, and testing.
We use 128 of the test samples for validation, resulting in a 128 / 64 / 720
split for training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Хаця абедзве яны беглі прыблізна аднолькава хутка, Зоя апярэдзіла Свету, бо _ дрэнна стартавала. Да каго або чаго адносіцца пропуск _?\nВарыянты:\na. Зоя\nb. Свету",
  "label": "b"
}
```

```json
{
  "text": "Зміцер папрасіў Рамана дапамагчы, але _ атрымаў адмову. Да каго або чаго адносіцца пропуск _?\nВарыянты:\na. Рамана\nb. Зміцер",
  "label": "b"
}
```

```json
{
  "text": "Томсан наведаў магілу Купера ў 1765 годзе. На той момант _ ужо пяць гадоў як памёр. Да каго або чаго адносіцца пропуск _?\nВарыянты:\na. Купера\nb. Томсан",
  "label": "a"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Ніжэй прыведзены пытанні з некалькімі варыянтамі адказу (з адказамі).
  ```

- Base prompt template:

  ```text
  Пытанне: {text}
  Варыянты:
  a. {option_a}
  b. {option_b}
  Адказ: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Пытанне: {text}
  Варыянты:
  a. {option_a}
  b. {option_b}

  Адкажыце на пытанне вышэй, адказаўшы 'a' або 'b', і нічога іншага.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset be-wsc
```

## Hallucination Detection

### MultiWikiHalluQA-be

This dataset uses the same data as [MultiWikiQA-be](#multiwikiqa-be), published in
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
  "context": "Кавальская Слабада () — станцыя першай чаргі Зеленалужскай лініі Мінскага метрапалітэна, паміж станцыямі «Вакзальная» і «Аэрадромная».\n\nНазва станцыі звязаная з існаваннем у гэтым раёне рамесніцкага прадмесця, дзе жылі кавалі. Рамесніцтва стала асноўным матывам афармлення станцыі, створанага скульптарамі П. Вайніцкім, М. Тарлецкай і мастачкай М. Жвірбля.\n\nСтанцыя размешчана на скрыжаванні вуліц Жукоўскага і Варанянскага і ўведзена ў эксплуатацыю 6 лістапада 2020 года.\n\nКанструкцыя станцыі – скляпеністага тыпу. Пасажырская платформа знаходзіцца пасярэдзіне, з абодвух бакоў размяшчаюцца калеі. Архітэктары праекту – У. Целяпнёў, В. Целяпнёва, А. Кузьміч, Г. Васільеў.\n\nДля ўзмацнення бяспекі пасажыраў абапал станцыйнай платформы ўсталяваны празрыстыя брамкі абсталяваныя аўтаматычным механізмам адкрыцця-закрыцця, якія адкрываюцца толькі для высадкі і пасадкі пасажыраў. Станцыя абсталявана таксама ліфтам і прыстасавана да патрэб асоб з абмежаванымі фізічнымі магчымасцямі.\n\nСтанцыі Мінскага метрапалітэна\nЗеленалужская лінія метро\n2020 год у Мінску\nЗ’явіліся ў 2020 годзе",
  "question": "Хто распрацаваў праект станцыі?",
  "answers": {
    "answer_start": [622],
    "text": ["У. Целяпнёў, В. Целяпнёва, А. Кузьміч, Г. Васільеў"]
  }
}
```

```json
{
  "context": "Каралеўства Ісландыя (, ) — гістарычная дзяржава, існаваўшая ў 1918 – 1944 у асабістай уніі з Даніяй.\n\nЗ 1380 Ісландыя практычна ўвесь час знаходзілася пад кантролем дацкага манарха (да 1814 фармальна належала Нарвегіі). У 1874 ёй ўпершыню была нададзена ўласная Канстытуцыя і самакіраванне, якое было пашыранае ў 1904. 1 снежня 1918 быў падпісаны Акт аб уніі, паводле якога Ісландыя прызнавалася як цалкам незалежная дзяржава на чале з дацкім каралём Крысціянам X.\n\nПасля акупацыі Даніі Нацысцкай Германіяй 9 красавіка 1940 сувязь з дацкім урадам была парушаная. 10 красавіка ўлада над Ісландыяй перайшла да мясцовага парламента – Альтынга, які выбраў часовым кіраўніком краіны Свейна Б'ёрнсана.\n\nПасля разрыву з Даніяй Ісландыя заняла пазіцыю нейтралітэту, не падтрымаўшы ні Вось, ні Саюзнікаў. Але 10 мая 1940 Вялікабрытанія пачала ўварванне ў Ісландыю, высадзіўшы 800 вайскоўцаў у Рэйк’явіку. Ісландскі ўрад выразіў пратэст супраць «гвалтоўнага парушэння» нейтралітэту, але ўжо 17 мая востраў быў цалкам акупаваны брытанскімі войскамі, а на тэрыторыю краіны дэсантавалася яшчэ 4.000, і прэм'ер-міністр Ісландыі Герман Ёўнасан звярнуўся па радыё да грамадзян краіны з просьбай ветліва аднесціся да брытанскіх салдат. У ліпні 1941 кантроль над востравам перайшоў да ЗША, бо Брытаніі патрабавалася армія ў іншых месцах. Да канца Другой сусветнай вайны Ісландыя заставалася пад акупацыяй Саюзнікаў.\n\n24 мая 1944 адбыўся рэферэндум, на якім большая частка прагаласавала за незалежнасць ад Даніі і рэспубліканскую форму кіравання. Першым прэзідэнтам Ісландыі стаў Свейн Б’ёрнсан. Тым часам Данія заставалася пад нацысцкай акупацыяй, і многія датчане былі абражаныя тым, што гэтае рашэнне было прынятае ў такі час. Тым не менш, дацкі кароль Крысціян X накіраваў віншавальную тэлеграму маладой рэспубліцы.\n\nГістарычныя дзяржавы Еўропы\nГісторыя Ісландыі",
  "question": "Калі ЗША ўзялі пад свой кантроль Ісландыю?",
  "answers": {
    "answer_start": [1228],
    "text": ["1941"]
  }
}
```

```json
{
  "context": "Ка́ртлі () — адна з асноўных гісторыка-геаграфічных абласцей Грузіі, калыска яе дзяржаўнасці. З адміністрацыйнага пункта гледжання, Картлі складаецца з Квема-Картлі і Шыда-Картлі. Частка тэрыторыі адносіцца да Мцхета-Мтыянецкага і Самцхэ-Джавахецкага краёў. Горад Тбілісі таксама знаходзіцца на тэрыторыі гістарычнай вобласці Картлі. Тэрыторыя Шыда-Картлі ўключае тэрыторыю былой Паўднёва-Асяцінскай аўтаномнай вобласці, якая цяпер непадкантрольная ўладам Грузіі.\n\nКартлійская гаворка ляжыць у аснове грузінскай мовы. У сучаснай грузінскай мове лінгвісты вылучаюць картлійскі дыялект.\n\nНа тэрыторыі Картлі размешчаны гарады: Тбілісі, Руставі, Горы, Хашуры, Карэлі, Каспі, Цхінвалі, Ахалгоры, Балнісі, Марнеулі, Мцхета, Душэты, Баржомі.\n\nУ вобласці шмат гістарычных помнікаў найстаражытнай і найноўшай гісторыі Грузіі.\n\nДля Картлі характэрны неаднастайны этнічны склад насельніцтва. Пераважае грузінскае насельніцтва, таксама маюцца кампактныя арэалы рассялення армян, азербайджанцаў, асецін і грэкаў.\n\nСпасылкі \n\n Этнакаўказ. Этнакарта Картлі 1959 г.\n\nГістарычныя вобласці Грузіі",
  "question": "З якіх гісторыка-геаграфічных рэгіёнаў складаецца Картлі ў адміністрацыйным плане?",
  "answers": {
    "answer_start": [152],
    "text": ["Квема-Картлі і Шыда-Картлі"]
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

```text
Ніжэй прыведзены тэксты з адпаведнымі пытаннямі і адказамі.
```

- Base prompt template:

```text
Тэкст: {text}
Пытанне: {question}
Адказ максімум 3 словамі: {label}
```

- Instruction-tuned prompt template:

```text
Тэкст: {text}

Адкажыце на наступнае пытанне пра тэкст вышэй максімум 3 словамі.

Пытанне: {question}
```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-hallucination-qa-be
```
