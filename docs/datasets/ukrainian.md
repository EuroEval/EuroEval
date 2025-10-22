# 🇺🇦 Ukrainian

This is an overview of all the datasets used in the Ukrainian part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### Cross-Domain UK Reviews

The dataset can be found [here](https://huggingface.co/datasets/vkovenko/cross_domain_uk_reviews).
The data is scrapped from [Tripadvisor](https://www.tripadvisor.com/) and [Rozetka](https://rozetka.com.ua/).

The [original dataset](https://huggingface.co/datasets/vkovenko/cross_domain_uk_reviews/blob/main/processed_data.csv)
contains 611,863 samples. We use 1,024 / 256 / 2,048 samples for our training,
validation and test splits, respectively.

Here are a few examples from the training split:

```json
{
    "text": "як і всі Mc Donalds, якість дуже низька, але рахунок високий за те, що ви їсте. . шкода, але не доходить до достатності",
    "label": "negative"
}
```

```json
{
    "text": "Посудомийною машиною користуюсь давно, роботою цілком заоволена. Працює дуже тихо і прекрасно справляється з забрудненим посудом. Вміщає в себе 12 комплектів посуду.",
    "label": "positive"
}
```

```json
{
    "text": "Зупинилися в готелі в липні 2021 року з сім'єю ( 4 людини ) , номер був обраний за категоріями люкс . У номері просторо і чисто . При бронюванні вони попросили викласти диван , що і було зроблено . У ванній кімнаті були всі витратні матеріали та рушники , в достатній кількості . У номері є невеликий холодильник , сейф . Але розчарований СНІДАНОК . Оголошений сніданок шведський стіл був , але це було повне розчарування . Вибачте , але можна зробити більш різноманітним і корисним , без майонезу на овочах ? ? . мухи літали і було неприємно перебувати в приміщенні .",
    "label": "neutral"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Нижче наведені документи і їх настрій, який може бути 'позитивний', 'нейтральний' або 'негативний'.
  ```

- Base prompt template:

  ```text
  Документ: {text}
  Настрій: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Документ: {text}

  Класифікуйте настрій у документі. Відповідайте 'позитивний', 'нейтральний', або 'негативний', і нічого більше.
  ```

- Label mapping:
  - `positive` ➡️ `позитивний`
  - `neutral` ➡️ `нейтральний`
  - `negative` ➡️ `негативний`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset cross-domain-uk-reviews
```

## Named Entity Recognition

### NER-uk

The dataset can be found [here](https://github.com/lang-uk/ner-uk).
The dataset primarily consists of text from the
[Open Corpus of Ukrainian Texts](https://github.com/brown-uk/corpus).

The original dataset consists of 10,833 / 668 / 1,307 samples for the
training, validation, and test splits, respectively. We use 1,024 / 256 / 2,048
samples for our training, validation and test splits, respectively. The train and
validation splits are subsets of the original splits, while the test split is
created using additional samples from the train split.

Here are a few examples from the training split:

```json
{
  "tokens": ["Хоча", "непросто", "про", "неї", "розповісти", "»", ".", "Ведмідь", "замовк", ",", "подивився", "на", "друзів", ",", "які", "уважно", "його", "слухали", ",", "і", "запитав", ":"],
  "labels": ["O", "O", "O", "O", "O", "O", "O", "B-PER", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]
}
```

```json
{
  "tokens": ["Експериментальний", "матеріал", "було", "оброблено", "статистично", ".", "Метою", "запропонованої", "статті", "є", "аналіз", "структурно-змістових", "особливостей", "перетворень", "у", "районній", "пресі", "Тернопільщини", "означеного", "періоду", "."],
  "labels": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-LOC", "O", "O", "O"]
}
```

```json
{
  "tokens": ["Як", "відомо", ",", "рішення", "«", "Про", "вихід", "зі", "складу", "засновників", "редакції", "газети", "«", "Житомирщина", "»", "з", "ініціативи", "голови", "обласної", "ради", "було", "прийнято", "на", "другій", "сесії", "обласної", "ради", "24", "грудня", "минулого", "року", "—", "саме", "того", "дня", ",", "коли", "Верховна", "Рада", "ухвалила", "в", "остаточній", "редакції", "Закон", "про", "реформування", "преси", "."],
  "labels": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-ORG", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-ORG", "I-ORG", "O", "O", "O", "O", "O", "O", "O", "O", "O"]
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Нижче наведені речення та JSON-словники з іменованими сутностями, які присутні у даному реченні.
  ```

- Base prompt template:

  ```text
  Речення: {text}
  Іменовані сутності: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Речення: {text}

  Ідентифікуйте іменовані сутності у реченні. Ви повинні вивести це як JSON-словник з ключами 'особа', 'місце', 'організація' та 'різне'. Значення мають бути списками іменованих сутностей цього типу, точно такими, як вони з'являються у реченні.
  ```

- Label mapping:
  - `B-PER` ➡️ `особа`
  - `I-PER` ➡️ `особа`
  - `B-LOC` ➡️ `місце`
  - `I-LOC` ➡️ `місце`
  - `B-ORG` ➡️ `організація`
  - `I-ORG` ➡️ `організація`
  - `B-MISC` ➡️ `різне`
  - `I-MISC` ➡️ `різне`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ner-uk-mini
```

## Linguistic Acceptability

### ScaLA-uk

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Ukrainian Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Ukrainian-ParlaMint) by assuming that
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
  "text": "Під патронатом Президента України в цьому році проведено ІІ Всеукраїнські літні спортивні ігри, які стали важливим етапом у підготовці до кваліфікаційних змагань по відбору до Літньої олімпіади в Афінах, сприяли зміцненню фізкультурно-спортивного руху, охопивши всі верстви населення.",
  "label": "correct"
}
```

```json
{
  "text": "І прошу, давайте подякуємо за допомогу нашим білоруським сусідам.",
  "label": "correct"
}
```

```json
{
  "text": "Шановні колеги, тепер переходимо до наступного.",
  "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Нижче наведені речення і їхня граматична правильність.
  ```

- Base prompt template:

  ```text
  Речення: {text}
  Граматично правильно: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Речення: {text}

  Визначте, чи речення граматично правильне чи ні. Відповідайте 'так', якщо речення правильне, і 'ні', якщо ні. Відповідайте лише цим словом, і нічим більше.
  ```

- Label mapping:
  - `correct` ➡️ `так`
  - `incorrect` ➡️ `ні`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-uk
```

## Reading Comprehension

### MultiWikiQA-uk

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
  "context": "Thalassema thalassema\xa0— вид ехіур родини Thalassematidae.\n\nПоширення \nВид поширений у припливній зоні вздовж європейського узбережжя Атлантичного океану та Середземного моря.\n\nОпис \nЧерв'як завдовжки 2-7\xa0см. На передньому кінці тіла розташований червоний м'язистий хоботок, який може розтягуватись на 10-20\xa0см у довжину. Рот знаходиться біля основи хоботка. Забарвлення тіла може бути різноманітним\xa0— синє, сіре, жовте, помаранчеве, рожеве. На передньому кінці тіла дві черевці щетинки, на задньому кінці вони відсутні.\n\nСпосіб життя \nМешкає у піску та мулі припливної зони. Живиться детритом та мікроорганізмами. Активний вночі.\n\nРозмноження \nСтатевий диморфізм відсутній. Запліднення зовнішнє. Плаваюча личинка трохофора живе деякий час як зоопланктон, потім осідає на дно і перетворюється на черв'яка.\n\nПосилання \n Lexikon der Biologie: Thalassema\n Saskiya Richards: A spoon worm (Thalassema thalassema)  MarLIN, The Marine Life Information Network, 2009.\n\nЕхіури\nКільчасті черви Атлантичного океану\nФауна Середземного моря\nТварини, описані 1774",
  "question": "Яка кількість черевних щетинок розташована на передній частині тіла Thalassema thalassema?",
  "answers": {
    "answer_start": [466],
    "text": ["дві"]
    }
}
```

```json
{
  "context": "Сезон 2007–08 в Серії A\xa0— футбольне змагання у найвищому дивізіоні чемпіонату Італії, що проходило між 26 серпня 2007 та 18 травня 2008 року. Став 76-м турніром з моменту заснування Серії A. Участь у змаганні брали 20 команд, у тому числі 3 команди, які попереднього сезону підвищилися у класі з Серії B. За результатами сезону 17 команд продовжили виступи в елітному дивізіоні, а три найгірших клуби вибули до Серії B.\n\nПереможцем турніру став міланський «Інтернаціонале», який здобув свій третій поспіль та 16-й в історії чемпіонський титул. Майбутні чемпіони захопили одноосібне лідерство у 6 турі турніру, після чого вже не залишали першого рядка турнірної таблиці. Хоча посеред змагання відрив основного переслідувача, «Роми», від лідера сягав 11 очок, перед останнім туром команди розділяв лише один заліковий пункт. «Інтер» забезпечив перемогу в сезоні, здолавши в цьому останньому турі одного з аутсайдерів сезону, «Парму», з рахунком 2:0.\n\nКоманди \n\nУчасть у турнірі Серії A сезону 2007–08 брали 20 команд:\n\nТурнірна таблиця\n\nРезультати\n\nБомбардири \nЗа результатами сезону таблицю найкращих бомбардирів Серії А очолила пара нападників туринського «Ювентуса»\xa0— Алессандро Дель П'єро та Давід Трезеге, які забили відповідно 21 та 20 голів в матчах турніру.\n\nПовний перелік гравців, що забили принаймні 10 голів в рамках Серії A сезону 2007—08:\n\n 21 гол\n  Алессандро Дель П'єро («Ювентус»)\n 20 голів\n  Давід Трезеге («Ювентус»)\n 19 голів\n  Марко Боррієлло («Дженоа»)\n 17 голів\n  Антоніо Ді Натале («Удінезе»)\n  Златан Ібрагімович («Інтернаціонале»)\n  Адріан Муту («Фіорентина»)\n 15 голів\n  Амаурі («Палермо»)\n  Кака («Мілан»)\n 14 голів\n  Горан Пандев («Лаціо»)\n  Томмазо Роккі («Лаціо»)\n  Франческо Тотті («Рома»)\n 13 голів\n  Хуліо Рікардо Крус («Інтернаціонале»)\n  Массімо Маккароне («Сієна»)\n 12 голів\n  Нікола Аморузо («Реджина»)\n  Клаудіо Белуччі («Сампдорія»)\n  Крістіано Доні («Аталанта»)\n  Фабіо Квальярелла («Удінезе»)\n 11 голів\n  Філіппо Індзагі («Мілан»)\n 10 голів\n  Роберт Аквафреска («Кальярі»)\n  Антоніо Кассано («Сампдорія»)\n  Франческо Тавано («Ліворно»)\n\nАльберто Джилардіно, Давід Трезеге і Нікола Аморузо забили по сто м'ячів у матчах Серії «А». По завершенні сезону, до десятки найвлучніших голеадорів ліги входять: Сільвіо Піола (275), Гуннар Нордаль (225), Джузеппе Меацца (216), Жозе Алтафіні (216), Роберто Баджо (205), Курт Хамрін (190), Джузеппе Сіньйорі (188), Габрієль Батістута (184), Джамп'єро Боніперті (178), Амедео Амадеї (174).\n\nПосилання \n Серія A 2007–08 на RSSSF  \n\n2007-2008\n2007 у футболі\n2008 у футболі\n2007 в італійському спорті\n2008 в італійському спорті", "question": "Яка кількість голів була забита Алессандро Дель П'єро протягом сезону Серії А 2007–2008 років?",
  "answers": {
    "answer_start": [1353],
    "text": ['21 гол']
    }
}
```

```json
{
  "context": "Тім Смолдерс (,  26 серпня 1980, Гел)\xa0— бельгійський футболіст, що грав на позиції півзахисника. По завершенні ігрової кар'єри\xa0— тренер.\n\nІгрова кар'єра \nУ дорослому футболі дебютував 1998 року виступами за команду «Брюгге», в якій провів шість сезонів, взявши участь у 63 матчах чемпіонату. За цей час виборов титул чемпіона Бельгії.\n\nЗгодом з 2004 по 2015 рік грав у складі нідерландського «Росендала», а також на батьківщині за «Шарлеруа», «Гент» та «Серкль».\n\nПротягом 2015—2018 років грав за нижчоліговий «Звевезеле».\n\nКар'єра тренера\nПерший досвід тренерської роботи отримав ще граючи на полі як помічник головного тренера «Серкля» у 2014—2015 роках.\n\nЗгодом входив до тренерських штабів юнацької збірної Бельгії (U-19) та молодіжної команди «Брюгге».\n\n2021 року очолив футбольну академію «Брюгге».\n\nТитули і досягнення\n Чемпіон Бельгії (1):\n«Брюгге»: 2002-2003\n Володар Кубка Бельгії (2):\n«Брюгге»: 2002, 2004\n Володар Суперкубка Бельгії (3):\n«Брюгге»: 1998, 2002, 2003\n\nПосилання \n\nбельгійські футболісти\nбельгійські футбольні тренери\nФутболісти «Брюгге»\nФутболісти «Росендала»\nФутболісти «Шарлеруа»\nФутболісти «Гента»\nФутболісти «Серкля»\nТренери ФК «Серкль»\nТренери юнацької збірної Бельгії з футболу\nТренери ФК «Брюгге»\nбельгійські футбольні легіонери\nФутбольні легіонери в Нідерландах",
  "question": "Де Смолдерс вперше спробував себе в ролі тренера?",
  "answers": {
    "answer_start": [629],
    "text": ["«Серкля»"]
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Нижче наведені тексти з відповідними питаннями та відповідями.
  ```

- Base prompt template:

  ```text
  Текст: {text}
  Питання: {question}
  Відповідь максимум 3 словами:
  ```

- Instruction-tuned prompt template:

  ```text
  Текст: {text}

  Відповідь на наступне питання про вищезазначений текст максимум 3 словами.

  Питання: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-uk
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
  "text": "Nedokázal som ovládať vlhkosť ako som ovládal dážď, pretože _ prichádzalo odvšadiaľ. Na koho sa vzťahuje prázdne miesto _?\nMožnosti:\na. Možnosť A: vlhkosť\nb. Možnosť B: dážď",
  "label": "a"
}
```

```json
{
  "text": "Jessica si myslela, že Sandstorm je najlepšia pieseň, aká bola kedy napísaná, ale Patricia ju nenávidela. _ si kúpila lístok na jazzový koncert. Na koho sa vzťahuje prázdne miesto _?\nMožnosti:\na. Možnosť A: Jessica\nb. Možnosť B: Patricia",
  "label": "b"
}
```

```json
{
  "text": "Termostat ukazoval, že dole bolo o dvadsať stupňov chladnejšie ako hore, takže Byron zostal v _ pretože mu bola zima. Na koho sa vzťahuje prázdne miesto _?\nMožnosti:\na. Možnosť A: dole\nb. Možnosť B: hore",
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
