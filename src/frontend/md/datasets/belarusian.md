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
  "label": "positive"
}
```

```json
{
  "text": "Гэта лічба толькі пацвярджае, што фестываль з кожным годам набірае моцы, пашыраючы сваю геаграфію.",
  "label": "positive"
}
```

```json
{
  "text": "Яна цудоўна абудзіла апетыт, апетыт да падрабязнасцяў, да разгадвання, да спазнання.",
  "label": "positive"
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
  "labels": ["B-PER", "I-PER", "O", "O", "O"]
}
```

```json
{
  "tokens": [
    "Пасля",
    "гуляў",
    "таксама",
    "за",
    "моладзевую",
    "зборную",
    "Беларусі",
    "."
  ],
  "labels": ["O", "O", "O", "O", "B-ORG", "I-ORG", "I-ORG", "O"]
}
```

```json
{
  "tokens": ["Горад", "Кампен", ",", "Нідэрланды"],
  "labels": ["B-LOC", "I-LOC", "I-LOC", "I-LOC"]
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
and was automatically created from the
[Belarusian Universal Dependencies treebank](https://github.com/UniversalDependencies/UD_Belarusian-HSE)
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

This dataset was published in [this paper](https://aclanthology.org/2025.acl-long.25/)
and is a Belarusian version of the Winograd schema challenge (WSC).

The original full dataset consists of 570 / 200 / 200 samples for training, validation,
and testing. We use 128 of the test samples for validation, resulting in a 128 / 64 /
720 split for training, validation and testing, respectively.

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

## Instruction-following

### MultiIFEval-be

MultiIFEval-be is part of the MultiIFEval benchmark spanning 305 languages. It is
generated by translating and localising the English IFEval dataset using a structured
LLM generation pipeline. For each target language, a randomly selected Wikipedia article
in that language provides contextual grounding to reduce hallucination and improve
cultural localisation. The pipeline preserves instruction_id_list values for
traceability to the original English samples, and retains kwargs keys with values
localised where appropriate, enabling programmatic constraint verification. The dataset
was published [here](https://huggingface.co/datasets/EuroEval/multi-ifeval-be).

This dataset is part of the MultiIFEval benchmark introduced in
[this draft paper](https://raw.githubusercontent.com/alexandrainst/multi_ifeval/refs/heads/feat/add-paper/paper/acl_latex.tex).

We use the dataset as the test split, and do not include other splits, as we only
evaluate models zero-shot and the size is too small to warrant a validation set.

Here are a few examples from the test split:

```json
{
  "text": "Напішыце зводку старонкі Wikipedia \"https://be.wikipedia.org/wiki/Беларуская_мова\" не менш за 250 слоў. Не выкарыстоўвайце ніякіх косак і вылучыце не менш за 3 раздзелы, якія маюць загалоўкі, у фарматаце Markdown, напрыклад *выдзелены раздзел Частка 1*, *выдзелены раздзел Частка 2*, *выдзелены раздзел Частка 3*.",
  "target_text": {
    "instruction_id_list": [
      "punctuation:no_comma",
      "detectable_format:number_highlighted_sections",
      "length_constraints:number_words"
    ],
    "kwargs": [
      {},
      { "num_highlights": 3 },
      { "num_words": 250, "relation": "at least" }
    ]
  }
}
```

```json
{
  "text": "Я планую паездку ў Беларусь і хачу, каб ты напісаў мне план падарожжа ў стылі Шэкспіра. Табе не дазваляецца выкарыстоўваць коскі ў адказе.",
  "target_text": {
    "instruction_id_list": ["punctuation:no_comma"],
    "kwargs": [{}]
  }
}
```

```json
{
  "text": "Стварыце рэзюмэ для нядаўняга выпускніка школы, які падае заяўку на сваю першую працу. Пераканайцеся, што вы ўключылі не менш за 12 месцаў у квадратных дужках, такіх як [Імя] або [Адрас].",
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
euroeval --model <model-id> --dataset multi-ifeval-be
```

## Hallucination Detection

### RAGTruth-be

This dataset is a Belarusian translation of the
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
  "prompt": "Падсумуйце наступныя навіны ў межах 87 слоў:\n'Маладыя і бяссмертныя' Спойлеры: Нік прымушае Салі выбіраць -- яго або Адама?\nСпойлеры 'Маладыя і бяссмертныя' (Y&R) на тыдзень з 23 кастрычніка паведамляюць, што Нік Ньюман (Джошуа Мораў) прымушае Салі Спектру (Кортні Хоуп) выбіраць паміж ім і Адамам Ньюманам (Марк Гросман). Штотыднёвы прамо CBS паказвае Ніка ў бойцы з Адамам, патрабуючы ад Салі выбару паміж імі.\nЖурнал Soap Opera Digest паведамляе, што Нік адмаўляецца дазволіць Адаму разрушаць яго адносіны з Салі. Даведаўшыся, што ў Салі ёсць неразрешаныя пачуцці да Адама, ён разрывае адносіны з рудой. Ён настойваў, каб яна разабралася ў сваіх пачуццях да яго брата, каб яны маглі рухацца наперад.\nСалі адчувае канфлікт паміж братамі. Яна глыбока клапоціцца пра Ніка і пачала ў яго закахвацца. Аднак у яе моцная сувязь з Адамам, і яна не можа адмовіцца ад сваіх пачуццяў да яго. Яна не можа выбраць паміж імі. Яна адмаўляецца гэта зрабіць і разглядае магчымасць пакінуць абодвух, каб пазбегнуць прыняцця рашэння.\nСпойлеры 'Маладыя і бяссмертныя' кажуць, што Салі і Адам пераходзяць да інтымных адносін на тыдні з 23 кастрычніка. Гэта не змяняе яе пачуццяў да Ніка, але ўмацоўвае яе сувязь з Адамам.\nНік і Адам хочуць, каб Салі прыняла рашэнне паміж імі. Яны абодва абяцаюць ёй бяспеку і нязменную падтрымку. Яна не можа ўявіць сваё жыццё без іх у сваім жыцці. Нік адмаўляецца сядзець у баку, пакуль Адам спрабуе заваяваць Салі. Ён нападае на Адама, спрабуючы прымусіць яго пакінуць Салі ў спакоі.\nБраты ўступаюць у фізічную сутычку. Салі разрывае іх, загадаўшы спыніць бойку. Нік звяртаецца да Салі і патрабуе, каб яна вызначылася, з кім хоча быць.\nСпойлеры 'Маладыя і бяссмертныя' кажуць, што рашэнне Салі здзівіць гледачоў. Яна павінна прыняць рашэнне, якое будзе лепшым для яе будучыні.\nФанаты Y&R, якога брата Ньюмана, на вашу думку, выбярэ Салі? Працягвайце глядзець 'Маладыя і неспакойныя', якія выходзяць па буднях на CBS і транслююцца на Paramount Plus. Калі ласка, падзяліце сваімі думкамі ў каментарах ніжэй. Не забудзьцеся падпісвацца, каб чытаць больш майго кантэнту.\noutput:"
}
```

```json
{
  "prompt": "Каротка адказвайце на наступнае пытанне:\nяк пасадзіць бульбу, якая прарасла\nУлічвайце, што ваш адказ павінен быць строга заснаваны на наступных трох фрагментах:\nфрагмент 1: 1. Нарэжце вялікія, прараслыя бульбы на кавалкі з двума або трыма вачыма кожны, выкарыстоўваючы востры нож, асцярожна, каб не зламаць доўгія парасткі. Пакладзіце кавалкі бульбы на паднос або бляху для печы і пакіньце іх у прахалодным, цёмным месцы на ноч, каб зрэзаныя канцы маглі высахнуць. Маленькія бульбы можна садзіць цэлымі. Добра палівайце бульбу. Працягвайце рэгулярна паліваць, калі зямля сухая на дотык. Дадайце 4-дюймавы пласт мульчы вакол раслін, адцягваючы яго не менш чым на 4 дюймы ад прараслых бульбяных вянкоў. Мульча захоўвае зямлю халоднай, падтрымліваючы пастаянны ўзровень вільготнасці.\n\nфрагмент 2: Агульны агляд. Часам, нават з найлепшымі намерамі, насенне бульбы пачынае прарастаць, бо вы не змаглі ўкласці іх у зямлю. Гэта можа быць звязана з надвор'ем або проста таму, што вы не падрыхтавалі садовую прастору своечасова. Насенне бульбы праходзіць тры стадыі: маладыя насенне, сярэдняга ўзросту і старыя. Агульны агляд. Часам, нават з найлепшымі намерамі, насенне бульбы пачынае прарастаць, бо вы не змаглі ўкласці іх у зямлю.\n\nфрагмент 3: Інструкцыі. Перакульціце зямлю ў вашай запланаванай зоне пасадкі з дапамогай сапкі або культыватара на глыбіню 12 дюймаў і дробна разаб'іце зямлю. Выдаліце любыя вялікія камяні і іншыя адыходы падчас сапкі. Выкапайце траншэю каля 4 дюймаў шырынёй і 6-8 дюймаў глыбінёй. Калі вы плануеце пасадзіць больш за адзін рад, аддзяліце іх прыкладна на 24-36 дюймаў. Нарэжце бульбу на кубікі памерам каля 1-2 дюймаў з кожнага боку, пакідаючы парастак на кожным кубіку. Непрараслыя часткі павінны мець два або тры вока на кожным. Пакладзіце кубікі на дно траншэі, аддзяляючы кожны кубік ад іншага прыкладна на 6-8 дюймаў. Пакладзіце бок з парасткамі або вачамі кубікаў уверх. Накрыйце бульбу пластом кампосту, а затым замясціце зямлю на вяршыні пасадкі. Інструкцыі. Перакульціце зямлю ў вашай запланаванай зоне пасадкі з дапамогай сапкі або культыватара на глыбіню 12 дюймаў і дробна разаб'іце зямлю. Выдаліце любыя вялікія камяні і іншыя адыходы падчас сапкі. Выкапайце траншэю каля 4 дюймаў шырынёй і 6-8 дюймаў глыбінёй.\n\nКалі фрагменты не ўтрымліваюць неабходнай інфармацыі для адказу на пытанне, калі ласка, адкажыце: \"Немагчыма адказаць на аснове дадзеных фрагментаў.\"\noutput:"
}
```

```json
{
  "prompt": "Сумуйце наступныя навіны ў 76 словах:\nАпрацоўка працоўных дазволаў для мігрантаў будзе працягвацца нават у тым выпадку, калі федэральны ўрад зачыніцца\nНават калі федэральны ўрад зачыніцца на гэтых выходных, федэральнае агенцтва, адказнае за апрацоўку працоўных дазволаў і іншых заяў ад шукачоў прытулку, працягне сваю працу.\nУрад Байдена аб'явіў на мінулым тыдні, што венесуэльскія грамадзяне, якія ўвайшлі ў ЗША да 31 ліпеня, цяпер будуць уключаны ў групу мігрантаў, якія маюць права на часовы абаронены статус. Для тысяч шукачоў прытулку ў Нью-Ёрку гэта можа паскорыць працэс атрымання працоўных дазволаў.\nДжадду наведала горад у пятніцу, каб назіраць за тым, як мясцовыя, дзяржаўныя і федэральныя арганізацыі справляюцца з наплывам мігрантаў там.\nЯна наведала цэнтр прыёму мігрантаў, дзе федэральныя супрацоўнікі аказваюць дапамогу ў працы з мігрантамі, раздаючы ўлёткі і адказваючы на запыты адносна заяў на працоўны дазвол. Таксама былі бачныя цэнтр падтрымкі працоўнай аўтарызацыі і аддзяленне Амерыканскага Чырвонага Крыжа.\nКрокі для паскарэння працоўных дазволаў\nЗгодна з прадстаўніком USCIS, арганізацыя прыняла меры для паскарэння працэсу падачы заяў на працоўныя дазволы, праводзіць адукацыйныя сесіі для імігрантаў у бібліятэках і выпусціла электронныя і SMS-апавяшчэнні для мігрантаў аб іх праве на працоўны дазвол. Акрамя таго, арганізацыя прапануе мабільныя біяметрычныя паслугі, уключаючы адбіткі пальцаў.\nДжадду заяўляе, што USCIS \"прызначана\" завяршыць заяўкі на працоўныя дазволы ў сярэднім за 30 дзён.\nПатрабуюцца дадатковыя сродкі\nХоць USCIS зможа функцыянаваць падчас зачынення дзякуючы зборам за падачу розных іміграцыйных формаў, Джадду сцвярджае, што Кангрэсу варта даць арганізацыі больш фінансавання, каб яна магла апрацоўваць больш працоўных дазволаў і заяў на TPS.\nУ нас ёсць толькі пэўная колькасць грошай з-за грошай, якія людзі даюць нам. Гэта часам можа быць абмежавальным, заявіла Джадду. Чэсна кажучы, мы маглі б дасягнуць значна больш, калі б Кангрэс падтрымліваў нас.\noutput:"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information):

- Number of few-shot examples: 0 (zero-shot only)
- Instruction prompt:

  ```text
  {prompt}
  ```

  I.e., we just use the instruction directly as the prompt.

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ragtruth-be
```

## Logical Reasoning

### ZebraPuzzleEasy-be

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2511.03553)
and consists of logic grid puzzles (also known as Einstein's riddles or Zebra puzzles),
where the task is to determine which attributes belong to which house based on a set of
clues. This is the easy variant with 2 houses and 3 attribute categories.

The original full dataset consists of 128 / 128 / 1,024 samples for training, validation
and testing, respectively (so 1,280 samples used in total). We use the same splits.

Here are a few examples from the training split:

```json
{
  "text": "Шэраг дамоў пранумараваны ад 1 да 2 злева направа.\n\nУ кожным доме жыве чалавек з унікальнай характарыстыкай у кожнай з наступных катэгорый:\n\nНацыянальнасці: Іспанія і Данія.\nХатнія жывёлы: зебра і кот.\nНапоі: какава і малако.\n\nМы таксама ведаем наступнае:\n\n1. Чалавек, які глядзіць скачкі з трамплінам, лічыць другім лепшым фруктам манга.\n2. Чалавек з сястрой мае татуіроўку.\n3. Данец не п'е какавы.\n4. Чалавек, які гуляе ў відэагульні, носіць акуляры.\n5. Данец жыве ў доме нумар 1.\n6. Чалавек, які п'е какаву, ведае, што сонечная сістэма рухаецца са хуткасцю каля 200 км/с вакол цэнтра галактыкі.\n7. Чалавек з катом не жыве ў доме нумар 2.\n8. Чалавек, які наведаў Канаду, не мае кактуса.",
  "target_text": {
    "object_1": [
      "Данія",
      "кот",
      "малако"
    ],
    "object_2": [
      "Іспанія",
      "зебра",
      "какава"
    ]
  }
}
```

```json
{
  "text": "Шэраг дамоў пранумараваны ад 1 да 2 злева направа.\n\nУ кожным доме жыве чалавек з унікальнай характарыстыкай у кожнай з наступных катэгорый:\n\nНацыянальнасці: Ісландыя і Фарэрскія астравы.\nЛюбімыя садавіна і ягады: банан і чорная парэчка.\nХатнія жывёлы: зебра і пёс.\n\nМы таксама ведаем наступнае:\n\n1. Ісландзец жыве побач з чалавекам з рыжымі валасамі.\n2. Чалавек з псом жыве злева ад чалавека з зeбрай.\n3. Чалавек, які гуляе на гітары, мае ступень магістра па матэматыцы.\n4. Ісландзец любіць бананы.\n5. Чалавек у акулярах жыве ў доме нумар 2.\n6. Прыхільнік бананаў мае марскую свінку.\n7. Чалавек з роварам жыве ў доме нумар 1.\n8. Прыхільнік чорнай парэчкі не жыве ў доме нумар 1.",
  "target_text": {
    "object_1": [
      "Ісландыя",
      "банан",
      "пёс"
    ],
    "object_2": [
      "Фарэрскія астравы",
      "чорная парэчка",
      "зебра"
    ]
  }
}
```

```json
{
  "text": "Шэраг дамоў пранумараваны ад 1 да 2 злева направа.\n\nУ кожным доме жыве чалавек з унікальнай характарыстыкай у кожнай з наступных катэгорый:\n\nНацыянальнасці: Нідэрланды і Францыя.\nПрафесіі: медсястра і пекар.\nНапоі: какава і малако.\n\nМы таксама ведаем наступнае:\n\n1. Чалавек у акулярах не жыве ў доме нумар 2.\n2. Медсястра не жыве ў доме нумар 1.\n3. Медсястра глядзіць скачкі з трамплінам.\n4. Чалавек, які часта займаецца парусным спортам, не жыве ў доме нумар 2.\n5. Француз п'е какаву.\n6. Француз ведае, што на вуліцы шмат аўтамабіляў.\n7. Чалавек, які п'е какаву, і чалавек без кактуса — добрыя сябры.\n8. Француз пекар.",
  "target_text": {
    "object_1": [
      "Францыя",
      "пекар",
      "какава"
    ],
    "object_2": [
      "Нідэрланды",
      "медсястра",
      "малако"
    ]
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt: (empty)
- Instruction prompt:

  ```text
  Вось загадка:
  <riddle>
  {text}
  </riddle>

  Хто мае якія ўласцівасці і жыве ў якім доме?

  Калі ласка, дайце свой адказ у выглядзе JSON dictionary. Кожны key павінен быць object_X, дзе X — нумар дома. Кожнае value павінна быць спісам уласцівасцей з вышэйпералічаных катэгорый, якія належаць чалавеку ў доме нумар X.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset zebra-puzzles-easy-be
```

### Unofficial: ZebraPuzzleHard-be

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2511.03553)
and consists of logic grid puzzles (also known as Einstein's riddles or Zebra puzzles),
where the task is to determine which attributes belong to which house based on a set of
clues. This is the hard variant with 4 houses and 5 attribute categories.

The original full dataset consists of 128 / 128 / 1,024 samples for training, validation
and testing, respectively (so 1,280 samples used in total). We use the same splits.

Here are a few examples from the training split:

```json
{
  "text": "Шэраг дамоў пранумараваны ад 1 да 4 злева направа.\n\nУ кожным доме жыве чалавек з унікальнай характарыстыкай у кожнай з наступных катэгорый:\n\nНацыянальнасці: Вялікабрытанія, Нарвегія, Фарэрскія астравы і Францыя.\nПрафесіі: настаўнік, паліцэйскі, пекар і праграміст.\nЛюбімыя садавіна і ягады: апельсін, банан, чорная парэчка і яблык.\nЛюбімыя жанры кніг: дэтэктыў, жахі, навукова-папулярная і паэзія.\nХатнія жывёлы: кот, кролік, пёс і хвалісты папугай.\n\nМы таксама ведаем наступнае:\n\n1. Чалавек з татуіроўкай не жыве ў доме нумар 2.\n2. Праграміст жыве паміж настаўнікам і чытачом жахаў.\n3. Паміж французам і чалавекам з кролікам знаходзіцца адзін дом.\n4. Прыхільнік чорнай парэчкі не мае пса.\n5. Паміж прыхільнікам бананаў і чытачом паэзіі знаходзіцца 2 дамы.\n6. Прыхільнік чорнай парэчкі жыве ў доме нумар 2.\n7. Чалавек з кролікам мае рыжыя валасы.\n8. Брытанец жыве побач з чалавекам з марской свінкай.\n9. Нарвежац жыве справа ад фарэрца.\n10. Паміж французам і настаўнікам знаходзіцца 2 дамы.\n11. Брытанец любіць яблыкі.\n12. Пекар жыве адразу справа ад чытача паэзіі.\n13. Чалавек, які наведаў Канаду, не жыве ў доме нумар 1.\n14. Прыхільнік чорнай парэчкі жыве побач з чалавекам з хвалістым папугаем.\n15. Фарэрац жыве адразу злева ад чытача навукова-папулярнай літаратуры.\n16. Чытач жахаў жыве побач з чалавекам са ступенню магістра па матэматыцы.\n17. Нарвежац настаўнік.",
  "target_text": {
    "object_1": [
      "Францыя",
      "паліцэйскі",
      "апельсін",
      "паэзія",
      "хвалісты папугай"
    ],
    "object_2": [
      "Фарэрскія астравы",
      "пекар",
      "чорная парэчка",
      "жахі",
      "кот"
    ],
    "object_3": [
      "Вялікабрытанія",
      "праграміст",
      "яблык",
      "навукова-папулярная",
      "кролік"
    ],
    "object_4": [
      "Нарвегія",
      "настаўнік",
      "банан",
      "дэтэктыў",
      "пёс"
    ]
  }
}
```

```json
{
  "text": "Шэраг дамоў пранумараваны ад 1 да 4 злева направа.\n\nУ кожным доме жыве чалавек з унікальнай характарыстыкай у кожнай з наступных катэгорый:\n\nНацыянальнасці: Іспанія, Нарвегія, Фарэрскія астравы і Швецыя.\nПрафесіі: медсястра, настаўнік, пекар і праграміст.\nЛюбімыя садавіна і ягады: апельсін, банан, груша і чорная парэчка.\nЛюбімыя жанры кніг: жахі, любоўны раман, навуковая фантастыка і паэзія.\nХобі: гандбол, маляванне, настольныя гульні і тэніс.\n\nМы таксама ведаем наступнае:\n\n1. Чытач любоўных раманаў жыве паміж настаўнікам і чалавекам, які гуляе ў тэніс.\n2. Чалавек з марской свінкай не жыве ў доме нумар 2.\n3. Паміж фарэрцам і прыхільнікам бананаў знаходзіцца адзін дом.\n4. Фарэрац ведае, што сонечная сістэма рухаецца са хуткасцю каля 200 км/с вакол цэнтра галактыкі.\n5. Паміж пекарам і чытачом любоўных раманаў знаходзіцца адзін дом.\n6. Паміж чалавекам, які гуляе ў тэніс і чалавекам, які малюе знаходзіцца адзін дом.\n7. Прыхільнік груш мае ровар.\n8. Медсястра чытае жахі.\n9. Чытач навуковай фантастыкі і чалавек, які глядзіць скачкі з трамплінам, — добрыя сябры.\n10. Нарвежац жыве злева ад фарэрца.\n11. Прыхільнік груш жыве паміж прыхільнікам апельсінаў і чалавекам, які малюе.\n12. Прыхільнік груш жыве паміж прыхільнікам апельсінаў і чытачом навуковай фантастыкі.\n13. Ва ўсіх дамах ёсць вялікія вокны.\n14. Швед жыве ў доме нумар 3.\n15. Чалавек, які гуляе ў тэніс, жыве паміж іспанцам і чалавекам, які гуляе ў настольныя гульні.\n16. Іспанец жыве адразу справа ад настаўніка.",
  "target_text": {
    "object_1": [
      "Нарвегія",
      "настаўнік",
      "чорная парэчка",
      "навуковая фантастыка",
      "маляванне"
    ],
    "object_2": [
      "Іспанія",
      "праграміст",
      "банан",
      "любоўны раман",
      "гандбол"
    ],
    "object_3": [
      "Швецыя",
      "медсястра",
      "груша",
      "жахі",
      "тэніс"
    ],
    "object_4": [
      "Фарэрскія астравы",
      "пекар",
      "апельсін",
      "паэзія",
      "настольныя гульні"
    ]
  }
}
```

```json
{
  "text": "Шэраг дамоў пранумараваны ад 1 да 4 злева направа.\n\nУ кожным доме жыве чалавек з унікальнай характарыстыкай у кожнай з наступных катэгорый:\n\nЛюбімыя садавіна і ягады: клубніца, суніца, чорная парэчка і яблык.\nЛюбімыя жанры кніг: дэтэктыў, жахі, навукова-папулярная і паэзія.\nХатнія жывёлы: зебра, кролік, слімак і хвалісты папугай.\nНапоі: кава, какава, сок і чай.\nХобі: вязанне кручком, настольныя гульні, тэніс і футбол.\n\nМы таксама ведаем наступнае:\n\n1. Прыхільнік клубніц і чалавек, чый хатні жывёл стары для свайго віду, — добрыя сябры.\n2. Чалавек, які гуляе ў футбол, і чалавек з роварам — добрыя сябры.\n3. Паміж чытачом навукова-папулярнай літаратуры і чалавекам з хвалістым папугаем знаходзіцца 2 дамы.\n4. Паміж чалавекам са слімакам і чалавекам, які гуляе ў футбол знаходзіцца 2 дамы.\n5. Чытач навукова-папулярнай літаратуры жыве побач з чалавекам, які п'е каву.\n6. Чытач паэзіі жыве паміж прыхільнікам суніц і чалавекам, які п'е сок.\n7. Прыхільнік суніц чытае жахі.\n8. Паміж прыхільнікам яблыкаў і прыхільнікам клубніц знаходзіцца адзін дом.\n9. Чытач навукова-папулярнай літаратуры займаецца вязаннем кручком.\n10. Чалавек, які гуляе на гітары, жыве ў доме нумар 1.\n11. Чалавек без кактуса не жыве ў доме нумар 2.\n12. Чалавек, які п'е сок, жыве адразу злева ад чалавека, які гуляе ў тэніс.\n13. Прыхільнік яблыкаў жыве злева ад чалавека, які займаецца вязаннем кручком.\n14. Прыхільнік клубніц не жыве паміж чалавекам з зeбрай і чалавекам, які п'е какаву, і гэта тры розных людзі.\n15. Чалавек, які гуляе ў відэагульні, мае ступень магістра па матэматыцы.",
  "target_text": {
    "object_1": [
      "чорная парэчка",
      "дэтэктыў",
      "хвалісты папугай",
      "сок",
      "футбол"
    ],
    "object_2": [
      "яблык",
      "паэзія",
      "кролік",
      "какава",
      "тэніс"
    ],
    "object_3": [
      "суніца",
      "жахі",
      "зебра",
      "кава",
      "настольныя гульні"
    ],
    "object_4": [
      "клубніца",
      "навукова-папулярная",
      "слімак",
      "чай",
      "вязанне кручком"
    ]
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt: (empty)
- Instruction prompt:

  ```text
  Вось загадка:
  <riddle>
  {text}
  </riddle>

  Хто мае якія ўласцівасці і жыве ў якім доме?

  Калі ласка, дайце свой адказ у выглядзе JSON dictionary. Кожны key павінен быць object_X, дзе X — нумар дома. Кожнае value павінна быць спісам уласцівасцей з вышэйпералічаных катэгорый, якія належаць чалавеку ў доме нумар X.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset zebra-puzzles-hard-be
```
