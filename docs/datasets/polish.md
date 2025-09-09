# 🇵🇱 Polish

This is an overview of all the datasets used in the Polish part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### PolEmo2
This dataset was published in [this paper](https://aclanthology.org/K19-1092/) and consists of Polish online reviews from the medicine and hotels domains, annotated for sentiment. Each review is labelled as positive, negative, neutral, or ambiguous. We have filtered out the ambiguous samples.

The original full dataset consists of 6,573 / 823 / 820 samples for the training, validation and test splits, respectively. We use 1,024 / 256 / 2,048 samples for our training, validation and test splits, respectively. The train and validation splits are subsets of the original splits. For the test split, we use all available test samples and supplement with additional samples from the training set to reach 2,048 samples in total.

The distribution of sentiment labels across the combined splits is as follows:
- **Negative**: 1,592 samples
- **Positive**: 1,119 samples
- **Neutral**: 617 samples

Here are a few examples from the training split:

```json
{
    "text": "Stary , bardzo zaniedbany hotel , obsluga czesto nie w humorze nie wykluczajac wlasciciela hotelu . Sniadania malo urozmaicone , powtarzajace sie przez caly tydzien dwa rodzaje byle jakiej wedliny , jednego rodzaju zoltego sera i jajecznicy ze sproszkowanych jajek . Obiadokolacja bardzo pozno 19 . 30 . Dla malych dzieci i zmeczonych narciarzy stanowczo za pozno . Napewno odwiedze Livignio , ale nigdy wiecej hotel Europa .",
    "label": "negative"
}
```
```json
{
    "text": "Arkadiusz Miszuk został powołany na stanowisko prezesa , zaś Dariusz Rutowicz na stanowisko wiceprezesa , giełdowej spółki hotelowej Interferie SA , poinformowała spółka w komunikacie z 16 marca : „ Zarząd spółki Interferie INTERFERIE S . A . w Lubinie , informuje iż Rada Nadzorcza Spółki na posiedzeniu w dniu 16 . 03 . 2012 roku odwołała ze składu Zarządu : 1 ) Pana Adama Milanowskiego , 2 ) Pana Radosława Besztygę . Jednocześnie Zarząd INTERFERIE S . A . w Lubinie , informuje iż w dniu 16 . 03 . 2012 roku Rada Nadzorcza Spółki powołała w skład Zarządu : 1 ) Pana Arkadiusza Miszuka - na stanowisko Prezesa Zarządu , 2 ) Pana Dariusza Rutowicza - na stanowisko Wiceprezesa Zarządu .",
    "label": "neutral"
}
```
```json
{
    "text": "Hotel znajduje się w idealnym miejscu dla fanów pieszych wycieczek . Z dala od zgiełku Krupówek - blisko szlaków wychodzących w góry . Pokoje przestronne i czyste . Obsługa bardzo miła . Basen jest aczkolwiek swoim urokiem nie zachwyca . Bardzo bogate i smaczne śniadania . Również jedzenie w restauracji jest naprawdę godne polecenia . Byli śmy gośćmi hotelu już dwa razy za równo jako para jaki i rodzina z dziećmi i za każdym razem byli śmy zadowoleni .",
    "label": "positive"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Liczba przykładów few-shot: 12
- Prefiks promptu:
  ```
  Poniżej znajdują się dokumenty i ich sentyment, który może być 'pozytywny', 'neutralny' lub 'negatywny'.
  ```
- Szablon podstawowy promptu:
  ```
  Dokument: {text}
  Sentyment: {label}
  ```
- Szablon promptu instrukcyjnego:
  ```
  Dokument: {text}

  Klasyfikuj sentyment w dokumencie. Odpowiedz z 'pozytywny', 'neutralny' lub 'negatywny', i nic więcej.
  ```
- Label mapping:
    - `positive` ➡️ `positive`
    - `neutral` ➡️ `neutral`
    - `negative` ➡️ `negative`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset polemo2
```


## Named Entity Recognition

### KPWr-NER

This dataset was published in [this paper](https://aclanthology.org/L12-1574/) and is part of the KPWr (KrakówPoland Wrocław) corpus - a free Polish corpus annotated with various types of linguistic entities including named entities. The corpus was created to serve as training and testing material for Machine Learning algorithms and is released under a Creative Commons licence. The named entity annotations include persons, locations, organizations, and miscellaneous entities, which are mapped to standard BIO format labels.

The original dataset uses the train and test splits from the source corpus. The original data train split has 13,959 samples and test split has 4,323 samples. The validation split is created from the original training split. We use 1,024 / 256 / 2,048 samples for our training, validation and test splits, respectively. The train and validation splits are subsets of the original training split, while the test split is a subset of the original test split.

Here are a few examples from the training split:

```json
{
  "tokens": array(['Rublowka', '(', 'ros', '.', 'Рублёвка', ')', '–', 'potoczna',
       'nazwa', 'zachodniego', 'przedmieścia', 'Moskwy', '.'], dtype=object),
  "labels": array(['B-LOC', 'O', 'O', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O'], dtype=object)
}
```
```json
{
  "tokens": array(['Wiele', 'z', 'nich', 'zebrał', 'w', 'tomie', 'Cymelium', '(',
       '1978', ')', '.'], dtype=object),
  "labels": array(['O', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'O', 'O', 'O', 'O'], dtype=object)
}
```
```json
{
  "tokens": array(['Raul', 'Lozano', ':', 'Żeby', 'nie', 'było', ',', 'że',
       'faworyzuje', 'mistrza', 'Polski', 'w', 'siatkówce', ',', 'nie',
       'przyjechał', 'na', 'mecze', 'rozgrywane', 'w', 'Bełchatowie', '.'],
      dtype=object),
  "labels": array(['B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O'], dtype=object)
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:
  ```
  Poniżej znajdują się zdania i słowniki JSON z nazwanymi jednostkami występującymi w danym zdaniu.
  ```
- Base prompt template:
  ```
  Zdanie: {text}
  Nazwane jednostki: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Zdanie: {text}

  Zidentyfikuj nazwane jednostki w zdaniu. Powinieneś wypisać to jako słownik JSON z kluczami 'osoba', 'lokalizacja', 'organizacja' i 'różne'. Wartości powinny być listami nazwanych jednostek tego typu, dokładnie tak jak pojawiają się w zdaniu.
  ```
- Label mapping:
    - `B-PER` ➡️ `osoba`
    - `I-PER` ➡️ `osoba`
    - `B-LOC` ➡️ `lokalizacja`
    - `I-LOC` ➡️ `lokalizacja`
    - `B-ORG` ➡️ `organizacja`
    - `I-ORG` ➡️ `organizacja`
    - `B-MISC` ➡️ `różne`
    - `I-MISC` ➡️ `różne`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset kpwr-ner
```


## Linguistic Acceptability

### ScaLA-pl

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Polish Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Polish-PDB) by assuming that the
documents in the treebank are correct, and corrupting the samples to create
grammatically incorrect samples. The corruptions were done by either removing a word
from a sentence, or by swapping two neighbouring words in a sentence. To ensure that
this does indeed break the grammaticality of the sentence, a set of rules were used on
the part-of-speech tags of the words in the sentence.

The original full dataset consists of 22,152 samples, from which we use 1,024 / 256 / 2,048 samples for training,
validation and testing, respectively.

Here are a few examples from the training split:

```json
{
    "text": "Papierową śmierć zafundowaliśmy zafundowali śmy już kilku osobom.",
    "label": "correct"
}
```
```json
{
    "text": "To tylko mały krok; znam doskonale jego rozmiar; jestem świadomy, że polityka nieustanny wysiłek, a kiedy jedno zadanie się kończy, zaraz znajdzie się następne.",
    "label": "incorrect"
}
```
```json
{
    "text": "Tutaj interesuje mnie etyczny kontekst transferu naukowej wiedzy psychologicznej z laboratorium badacza do sali wykładowej i laboratorium studenckiego - czynniki ułatwiające i utrudniające, ale lokowane na stosunkowo wysokim poziomie ogólności.",
    "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Poniżej znajdują się teksty i czy są gramatycznie poprawne.
  ```
- Base prompt template:
  ```
  Tekst: {text}
  Gramatycznie poprawny: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Tekst: {text}

  Określ czy tekst jest gramatycznie poprawny czy nie. Odpowiedz {labels_str}, i nic więcej.
  ```
- Label mapping:
    - `correct` ➡️ `tak`
    - `incorrect` ➡️ `nie`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset scala-pl
```


## Reading Comprehension

### PoQuAD

PoQuAD is a Polish Question Answering dataset with contexts from Polish Wikipedia. It follows the SQuAD format with innovations including lower annotation density, abstractive answers, polar questions, and impossible questions.

This dataset was published in [this paper](https://dl.acm.org/doi/10.1145/3587259.3627548).

The original dataset consists of 51,951 samples. We use 1,024 / 256 / 2,048 samples for training, validation and testing, respectively.
We do not use the impossible questions in this version of the dataset.

Here are a few examples from the training split:

```json
{
  "context": "Luna (Karol Sevilla) jest nastolatką z Meksyku, która szczęśliwie jedzie przez życie na wrotkach. Jak każda dziewczyna w jej wieku, mieszka wraz ze swoją rodziną, chodzi do szkoły i ma swoją grupę znajomych. Ma również pracę jako dostawca w restauracji typu fast food. Luna spędza większość swojego czasu na wrotkach na nabrzeżu swego ukochanego miasta, słuchając piosenek skomponowanych przez jej najlepszego przyjaciela, Simóna (Michael Ronda). Ale jej życie przybiera jednak niespodziewany obrót, gdy jej rodzice otrzymują propozycję niemożliwą do odrzucenia..., jutro rodzina Valente musi opuścić swój ukochany dom i przenieść się do innego kraju, do Argentyny. Luna musi przystosować się do nowego życia, nowych przyjaciół i nowej szkoły, gdzie spotyka się świat luksusu i elit, który niewiele ma z nią wspólnego. Luna szuka schronienia w swojej jeździe na wrotkach, a przez nie odkrywa tor wrotkarski, Jam & Roller, który oferuje jej nowy wszechświat na kołach. Podczas tego nowego etapu w swoim życiu Luna rozwija swoją pasję do jazdy i tańca na wrotkach oraz odkrywa drogę do nowych przyjaciół i pierwszej miłości, którą znajduje w osobie zupełnie innej od niej samej, Matteo (Ruggero Pasquarelli). Na przeszkodzie stoi jednak najpopularniejsza dziewczyna w szkole i dziewczyna Matteo, Ámbar (Valentina Zenere), która za wszelką cenę chce uczynić życie Luny niemożliwym. Również podczas rozwijania swych pasji, Luna może być o krok od odkrycia swojej prawdziwej tożsamości.",
  "question": "Gdzie przeprowadza się Luna?",
  "answers": {'text': array(['do Argentyny'], dtype=object), 'answer_start': array([652], dtype=int32), 'generative_answer': array(['do Argentyny'], dtype=object)}}
```
```json
{
  "context": "W sezonie 1933 Ruch zdobył mistrzostwo Polski. Katzy zagrał w dziewiętnastu kolejkach ligowych. Jedynym meczem, w którym nie wystąpił, było spotkanie inauguracyjne sezon przeciwko Garbarni Kraków (6:0, 2 kwietnia 1933 roku). Podczas wyjazdowego meczu towarzyskiego z Polonią Karwina (4:1, 14 maja 1933 roku) został usunięty z boiska za krytykowanie decyzji sędziego. W październiku zagrał w przegranym sparingu reprezentacji Śląska, której przeciwnikiem była reprezentacja Polski (1:2, 4 października 1933 roku).",
  "question": "W ilu rundach spotkań wziął udział Stefan Katzy?",
  "answers": {'text': array(['w dziewiętnastu'], dtype=object), 'answer_start': array([60], dtype=int32), 'generative_answer': array(['W dziewiętnastu'], dtype=object)}}
```
```json
{
  "context": "Następnego dnia Amerykanie wysłali nad stację kolejową w Ploeszti 136 B-24 i 94 B-17 w asyście 132 P-38 i 48 P-47. 1 Grupa wysłała na przechwycenie 23 myśliwce IAR, ale tylko część z nich odnalazła bombowce meldując o zestrzeleniu trzech B-24. Sierż. Raghiga Dumitrescu stoczył walkę z czterema P-38, uszkadzając jeden z nich, jednak później sam został zestrzelony. Dwa inne samoloty lądowały na brzuchach. 5 Grupa poderwała 8 IAR-80 i 4 Bf 109E z 51 eskadry oraz 7 Bf 109E z 52 eskadry. Ich piloci odnotowali pięć zestrzeleń pewnych i jedno prawdopodobne. Kpt. Iliescu lądował awaryjnie uszkodzonym samolotem. 6 Grupa wykonała 49 lotów na IAR odnotowując pięć zwycięstw, w tym trzy potwierdzone, bez strat własnych. 7 Grupa wysłała 15 IAR-81C i 13 Bf 109G, meldując o trzech zwycięstwach przy stracie jednego samolotu. Piloci niemieckiego III/JG 77 meldowali o 16 zestrzelonych B-24 ze stratą 7 Bf 109G. O strąceniu 4 B-24 i 1 B-17 meldowali piloci z 10./JG 301. Sześć kolejnych Liberatorów mieli zestrzelić piloci II/JG 51, jednego B-17 lotnicy 12./NJG 6, a jednego P-51 pilot 1./JG 302. Prawdziwe straty Amerykanów wyniosły 10 B-24 (po pięć z 450. i 451. BG), trzy B-17 oraz jeden P-38 z 14. FG. Myśliwce eskorty nie odnotowały ani jednego zestrzelenia.", "question": "Czy sierżantowi Raghiga Dumitrescu udało się doprowadzić do awarii którego z samolotów P-38?",
  "answers": {'text': array(['Sierż. Raghiga Dumitrescu stoczył walkę z czterema P-38, uszkadzając jeden z nich'],
      dtype=object), 'answer_start': array([244], dtype=int32), 'generative_answer': array(['tak'], dtype=object)}}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:
  ```
  Poniżej znajdują się teksty z towarzyszącymi pytaniami i
  odpowiedziami.
  ```
- Base prompt template:
  ```
  Tekst: {text}
  Pytanie: {question}
  Odpowiedź w maksymalnie 3 słowach: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Tekst: {text}

  Odpowiedz na następujące pytanie dotyczące powyższego tekstu w maksymalnie 3 słowach.

  Pytanie: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset poquad
```


## Knowledge

### LLMzSzŁ

This dataset was created based on Polish national exams extracted from the archives of the Polish Central Examination Board. LLMzSzŁ (LLMs Behind the School Desk) represents the first comprehensive benchmark for the Polish language at this scale. The dataset features both academic and professional tests covering 4 types of exams from 154 different domains. The dataset was created to evaluate the ability of language models to transfer knowledge between languages and to assess their performance on Polish educational content.

The original dataset consisted of almost 19,000 closed-ended questions in a single test split. We use a 1,024 / 256 / 2,048 split for training, validation and testing, respectively (so 3,328 samples used in total).

Here are a few examples from the training split:

```json
{
  "text": "Czujnik do pomiaru poziomu obciążenia, stosowany w wozach paszowych jako element systemu zdalnego ważenia masy mieszanki, jest czujnikiem\nChoices:\na. tensometrycznym.\nb. podczerwieni.\nc. indukcyjnym.\nd. optycznym.",
  "label": "a"
}
```
```json
{
  "text": "Wybierz prawidłową kolejność wykonania operacji remontowych maszyny.\nChoices:\na. Weryfikacja, regeneracja, oczyszczenie, demontaż, badanie i odbiór maszyny po remoncie.\nb. Demontaż, weryfikacja, oczyszczenie, regeneracja, badanie i odbiór maszyny po remoncie.\nc. Oczyszczenie, demontaż, weryfikacja, regeneracja, naprawa zespołów, montaż, badanie i odbiór maszyny po remoncie.\nd. Regeneracja, demontaż, weryfikacja, oczyszczenie, naprawa zespołów, regeneracja, badanie i odbiór maszyny po remoncie.",
  "label": "c"
}
```
```json
{
  "text": "Cieczą ciężką jednorodną nazywamy substancję ciekłą, której gęstość jest\nChoices:\na. równa gęstości wody.\nb. większa od gęstości wody.\nc. mniejsza od gęstości wody.\nd. wypadkową gęstości cieczy ciężkiej i wody.",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Poniżej znajdują się pytania wielokrotnego wyboru (z odpowiedziami).
  ```
- Base prompt template:
  ```
  Pytanie: {text}
  Odpowiedź: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Pytanie: {text}

  Odpowiedz na powyższe pytanie, odpowiadając {labels_str}, i nic więcej.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset llmzszl
```


## Common-sense Reasoning

## Summarization

### PSC

The Polish Summaries Corpus (PSC) was published in [this paper](https://aclanthology.org/L14-1145/) and is a resource created for automated single-document summarization of Polish. The corpus contains manual summaries of news articles, with multiple independently created summaries for single texts to overcome annotator bias. It includes both abstract free-word summaries and extraction-based summaries created by selecting text spans from the original documents.

The original dataset consists only of a training split. We use 1,024 / 256 / 2,048 samples for our training, validation and test splits, respectively. All splits are subsets of the original training data, with the validation and test splits sampled from the original training set.

Here are a few examples from the training split:

```json
{
  "text": "Rozpoczynający się 31 grudnia 2000 roku The Race ma stać się pokazem możliwości technicznych współczesnego jachtingu, rozwoju technologii telekomunikacyjnych, ma dowieść siły marketingowej wielkich wydarzeń sportowych, a także potęgi finansowej sponsorów tego przedsięwzięcia. Około dziesięciu superjachtów wystartuje 31 grudnia 2000 roku o północy z Barcelony. Najlepszy po około dwóch miesiącach powinien wpłynąć do Starego Portu w Marsylii.",
  "target_text": "31 grudnia 2000 roku rozpoczynają się regaty The Race, będące rozwinięciem regat dookoła świata - Jules Verne Trophy. Jachty wystartują z Barcelony i przepłyną bez pomocy  i zawijania do portów trzy oceany.  Organizatorzy regat chcą dotrzeć do miliardów odbiorców.  By pobić rekordy oglądalności i zaprezentować sponsorów wykorzystana zostanie najnowsza technika m.in kamery na jachtach."
}
```
```json
{
  "text": "jeśli w polskich przedsiębiorstwach nie zostanie przeprowadzona restrukturyzacja, z ograniczeniem zatrudnienia i wzrostem wydajności, nie ma co marzyć, aby stały się one konkurencyjne w momencie wejścia Polski do Unii Europejskiej. wejście zagranicznego inwestora często oznacza zmniejszenie zatrudnienia. Do zmniejszania liczby pracowników prowadzą fuzje przedsiębiorstw. Na ochronny parasol pakietów socjalnych i odprawy dla zwalnianych mogą liczyć zatrudnieni górnictwie i hutnictwie. Na osłonę nie mogą liczyć pracownicy przemysłu lekkiego.",
  "target_text": "W firmach konieczne są zwolnienia restrukturyzacyjne i wzrost wydajności pracy. Jeśli porównamy polskie przedsiębiorstwa z ich zachodnimi odpowiednikami, okazuje się, że w stosunku do wielkości produkcji zatrudnienie u nas jest drastycznie większe. Głęboka restrukturyzacja jest konieczna, jeśli polscy producenci chcą być konkurencyjni po wstąpieniu Polski do Unii Europejskiej. Wymusza ją też kryzys na Wschodzie. Często są one również wynikami wejścia zagranicznego inwestora lub fuzji. Oprócz zwolnień potrzebne są inwestycje."
}
```
```json
{
  "text": "Podczas II Kongresu Filmu Polskiego ogromne poruszenie środowiska filmowego wywołał list ministra Andrzeja Zakrzewskiego. Minister Zakrzewski zaatakował środowisko filmowe za to, że dotąd nie ma nowego prawa filmowego.  Filmowcy Poczuli się skrzywdzeni ocenami, bo straty były przy zmianie ustrojowej i likwidacji państwowego mecenatu nieuniknione. A Polska najlepiej chyba ze wszystkich krajów postkomunistycznych przeprowadziła swoją kinematografię przez ten trudny okres.",
  "target_text": "Środowisko filmowe jest poruszone listem ministra kultury, który krytykuje polskie kino i atakuje filmowców m.in. za niewypracowanie nowego prawa filmowego. Twórcy czują się skrzywdzeni bezpodstawnymi zarzutami. Zaznaczają, że to ministerstwo odpowiada za zatrzymanie prac nad ustawą o kinematografii. Publiczna krytyka i niedbałość o interesy środowiska twórczego są oburzające. Minister potwierdza, że jest autorem listu, i nie akceptuje obecnej formuły Komitetu Kinematografii."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:
  ```
  Poniżej znajdują się artykuły z towarzyszącymi streszczeniami.
  ```
- Base prompt template:
  ```
  Artykuł: {text}
  Streszczenie: {target_text}
  ```
- Instruction-tuned prompt template:
  ```
  Artykuł: {text}

  Napisz streszczenie powyższego artykułu.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset psc
```
