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

### MultiWikiQA-sk

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
  "context": "Register toxických účinkov chemických látok (anglicky Registry of Toxic Effects of Chemical Substances, RTECS) je databáza toxikologických informácií zostavených z voľne dostupnej vedeckej literatúry bez odkazu na platnosť alebo užitočnosť publikovaných štúdií. Do roku 2001 bola databáza spravovaná americkou organizáciou NIOSH (National Institute for Occupational Safety and Health, slov. Národný ústav pre bezpečnosť a ochranu zdravia pri práci) ako verejne dostupná publikácia. Teraz ju spravuje súkromná spoločnosť Symyx Technologies a je dostupná len za poplatok.\n\nObsah \nDatabáza obsahuje šesť typov toxikologických informácií:\n primárne podráždenie\n mutagénne účinky\n reprodukčné účinky\n karcinogénne účinky\n akútna toxicita\n toxicita viacnásobných dávok\nV databáze sa spomínajú ako špecifické číselné hodnoty, ako napríklad LD50, LC50, TDLo alebo TCLo, tak aj študované organizmy a spôsob podávania látky. Pre všetky dáta sú uvedené bibliografické zdroje. Štúdie pritom nie sú nijako hodnotené.\n\nHistória \nDatabáza RTECS bola aktivitou schválenou americkým Kongresom, zakotvenou v Sekcii 20(a)(6) zákona Occupational Safety and Health Act z roku 1970 (PL 91-596). Pôvodné vydanie, známe ako Zoznam toxických látok (Toxic Substances List), bolo publikované 28. júna 1971 a obsahovalo toxikologické dáta o približne 5 000 chemikáliách. Názov bol neskôr zmenený na dnešný Register toxických účinkov chemických látok (Registry of Toxic Effects of Chemical Substances). V januári 2001 databáza obsahovala 152 970 chemikálií. V decembri 2001 bola správa RTECS prevedená z NIOSH do súkromnej firmy Elsevier MDL. Túto firmu kúpila v roku 2007 spoločnosť Symyx, súčasťou akvizície bola aj databáza RTECS. Tá je teraz dostupná len za poplatok vo forme ročného predplatného.\n\nRTECS je k dispozícii v angličtine, francúzštine a španielčine, a to prostredníctvom Kanadského centra pre bezpečnosť a ochranu zdravia pri práci. Predplatitelia majú prístup cez web, na CD-ROM a vo formáte pre intranet. Databáza je dostupná na webe aj cez NISC (National Information Services Corporation) a ExPub (Expert Publishing, LLC).\n\nExterné odkazy \n\n RTECS overview \n Symyx website \n Expert Publishing, LLC Website\n\nZdroj \n\nChemické názvy a kódy\nToxikológia",
  "question": "Aké sú tri možnosti prístupu k databáze RTECS, ak som predplatiteľ?",
  "answers": {"answer_start": [1949], "text": ["cez web, na CD-ROM a vo formáte pre intranet"]}}
```

```json
{
  "context": "Herta Naglová-Docekalová (* 29. máj 1944, Wels, Rakúsko) je rakúska filozofka a profesorka, členka vedenia Medzinárodnej asociácie filozofiek (IAPf), Österreichische Akademie der Wissenschaften, Institut International de Philosophie (Paríž), viceprezidentka Fédération Internationale des Sociétés de Philosophie (FISP), zakladajúca členka interdisciplinárnych pracovných skupín Frauengeschichte a Philosophische Frauenforschung na Viedenskej univerzite, členka redakčných rád popredných vedeckých časopisov, napr. Philosophin, L´Homme, Deutsche Zeitschrift für Philosophie.\n\nŽivotopis \nVyštudovala históriu, filozofiu a germanistiku na Viedenskej univerzite. V roku 1967 získala na svojej alma mater doktorát z histórie prácou o filozofovi dejín Ernstovi von Lasaulx). V rokoch 1968 - 1985 bola asistentkou na Inštitúte filozofie Viedenskej univerzity. V lete 1980 prednášala na Millersville University of Pennsylvania v USA.\n\nV roku 1981 sa habilitovala z filozofie na Viedenskej univerzite dielom Die Objektivität der Geschichtswissenschaft. V rokoch 1985 až 2009 bola profesorkou Inštitútu filozofie Viedenskej univerzity. Od roku 2009 je univerzitnou profesorkou na dôchodku (Universitätsprofessorin i. R.)\n\nBola hosťujúcou profesorkou v roku 1990 na Universiteit Utrecht v holandskom Utrechte; v Nemecku 1991/1992 na Goethe-Universität Frankfurt vo Frankfurte nad Mohanom; 1993 na Universität Konstanz v Konstanzi; 1994/1995 na Freie Universität Berlin v Berlíne. V rokoch 1995/1996 prednášala na Universität Innsbruck a 2011 na univerzite v Petrohrade v Rusku.\n\nDielo (výber) \n Jenseits der Säkularisierung. Religionsphilosophische Studien. - Berlin 2008 (Hg., gem.m. Friedrich Wolfram).\n Viele Religionen - eine Vernunft? Ein Disput zu Hegel. - Wien/Berlin 2008 (Hg., gem.m. Wolfgang Kaltenbacher und Ludwig Nagl).\n Glauben und Wissen. Ein Symposium mit Jürgen Habermas. - Wien/Berlin 2007 (Hg., gem.m. Rudolf Langthaler).\n Geschichtsphilosophie und Kulturkritik. - Darmstadt 2003 (Hrsg., gem.m. Johannes Rohbeck).\n Feministische Philosophie. Ergebnisse, Probleme, Perspektiven. - Frankfurt a.M. 2000 a 2004 \n Continental Philosophy in Feminist Perspective. - Pennsylviania State University Press 2000 (Hg. gem.m. Cornelia Klingler).\n Der Sinn des Historischen. - Frankfurt a.M. 1996 (Hrsg.).\n Politische Theorie. Differenz und Lebensqualität. - Frankfurt a.M. 1996 (Hrsg. gem.m. Herlinde Pauer-Studer).\n Postkoloniales Philosophieren: Afrika. - Wien/München 1992 (Hrsg., gem.m. Franz Wimmer).\n Tod des Subjekts? - Wien/München 1987 (Hrsg., gem.m. Helmuth Vetter).\n Die Objektivität der Geschichtswissenschaft. Systematische Untersuchungen zum wissenschaftlichen Status der Historie. - Wien/München 1982\n spoluvydavateľka: Wiener Reihe. Themen der Philosophie (od 1986). \n spoluvydavateľka: Deutsche Zeitschrift für Philosophie (1993-2004). \n spoluvydavateľka: L'Homme. Europäische Zeitschrift für feministische Geschichtswissenschaft (1990 - 2003).\n\nOcenenia \n Förderpreis mesta Viedeň, 1983\n Käthe Leichter Preis (rakúska štátna cena), 1997 \n Preis für Geistes- und Sozialwissenschaften der Stadt Wien, 2009\n\nReferencie\n\nExterné odkazy \n Oficiálna stránka, Universität Wien \n Austria Forum, Wissenssammlungen/Biographien: Herta Nagl-Docekal\n\nZdroj \n\nRakúski filozofi",
  "question": "Kedy prišla na svet Herta Naglová-Docekalová?",
  "answers": {"answer_start": [28], "text": ["29. máj 1944"]}}
```

```json
{
  "context": "Martin Bareš (* 25. november 1968, Brno) je český profesor neurológie, od septembra 2019 rektor Masarykovej univerzity, predtým od februára 2018 do septembra 2019 dekan Lekárskej fakulty Masarykovej univerzity.\n\nRiadiace funkcie \nVo februári 2018 sa stal dekanom Lekárskej fakulty Masarykovej univerzity. Funkciu prevzal po Jiřím Mayerovi, ktorý zastával pozíciu dekana v období 20102018. S nástupom na post dekana ukončil svoje pôsobenie ako prorektor univerzity, ako i zástupca prednostu I. neurologickej kliniky pre vedu a výskum.\n\nDo funkcie rektora univerzity bol zvolený 1. apríla 2019 Akademickým senátom Masarykovej univerzity. V prvom kole tajnej voľby získal Bareš 36 hlasov z 50 prítomných senátorov. Protikandidáta, prodekana Prírodovedeckej fakulty Jaromíra Leichmana, volilo 11 senátorov. 3 odovzdané hlasy boli neplatné.\n\nSkúsenosti s pôsobením vo vedení školy zbieral Bareš v rokoch 20112018, kedy pôsobil najskôr ako jej prorektor pre rozvoj a potom ako prorektor pre akademické záležitosti. Za svoje priority označil Bareš v dobe voľby posilňovanie role univerzity ako piliera slobody v súčasnej spoločnosti a zvýšenie kvality vzdelávania, vedy a výskumu na medzinárodnej úrovni.\n\nDo funkcie rektora ho vymenoval 11. júna 2019 prezident Miloš Zeman s účinnosťou od 1. septembra 2019. Vo funkcii tak nahradil Mikuláša Beka, ktorému sa skončilo druhé volebné obdobie a o zvolenie sa teda už opäť uchádzať nemohol. Bareš k 1. septembru 2019 rezignoval na post dekana Lekárskej fakulty.\n\nVedecká činnosť \nJe prednášajúcim v odboroch všeobecné lekárstvo, zubné lekárstvo, optometria, fyzioterapia, neurofyziológia pre študentov prírodných vied Lekárskej fakulty Masarykovej univerzity a školiteľ doktorandov odborovej rady neurológia a neurovedy.\n\nPôsobí v týchto vedeckých radách: Masarykova univerzita, Lekárska fakulta Masarykovej univerzity a CEITEC MU. Ďalej tiež Univerzita Palackého v Olomouci, Lekárska fakulta UPOL, Fakulta veterinárního lékařství VFU, ďalej je tiež členom Českej lekárskej komory, Českej neurologickej spoločnosti, Českej spoločnosti klinickej neurofyziológie, Českej lekárskej spoločnosti Jana Evangelisty Purkyně, Movement Disorders Society, Society for the Research on the Cerebellum a Society for Neuroscience. Takisto je členom redakčnej rady časopisov Clinical Neurophysiology, Behavioural Neurology, Tremor and Other Hyperkinetic Movements a Biomedical Papers.\n\nOsobný život \nJe ženatý, má dvoch synov a dcéru.\n\nReferencie\n\nExterné odkazy \n Martin Bareš\n\nZdroj \n\nČeskí lekári\nNeurológovia\nRektori Masarykovej univerzity\nČeskí univerzitní profesori\nDekani Lekárskej fakulty Masarykovej univerzity\nAbsolventi Lekárskej fakulty Masarykovej univerzity\nOsobnosti z Brna",
  "question": "Akú pozíciu mal Martin Bareš na Masarykovej univerzite počnúc septembrom 2019?",
  "answers": {
    "answer_start": [89],
    "text": ["rektor"]
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Nasledujú texty s pridruženými otázkami a odpoveďami.
  ```

- Base prompt template:

  ```text
  Text: {text}
  Otázka: {question}
  Odpoveď na maximálne 3 slová:
  ```

- Instruction-tuned prompt template:

  ```text
  Text: {text}

  Odpovedzte na nasledujúcu otázku týkajúcu sa textu uvedeného vyššie maximálne 3 slovami.

  Otázka: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-sk
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
