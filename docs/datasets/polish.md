# 🇵🇱 Polish

This is an overview of all the datasets used in the Polish part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### PolEmo2

This dataset was published in [this paper](https://doi.org/10.18653/v1/K19-1092) and
consists of Polish online reviews from the medicine and hotels domains, annotated for
sentiment. Each review is labelled as positive, negative, neutral, or ambiguous. We have
filtered out the ambiguous samples.

The original full dataset consists of 6,573 / 823 / 820 samples for the training,
validation and test splits, respectively. We use 1,024 / 256 / 2,048 samples for our
training, validation and test splits, respectively. The train and validation splits are
subsets of the original splits. For the test split, we use all available test samples
and supplement with additional samples from the training set to reach 2,048 samples in
total.

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

- Number of few-shot examples: 12
- Prompt prefix:

  ```text
  Poniżej znajdują się dokumenty i ich sentyment, który może być 'pozytywny', 'neutralny' lub 'negatywny'.
  ```

- Base prompt template:

  ```text
  Dokument: {text}
  Sentyment: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Dokument: {text}

  Klasyfikuj sentyment w dokumencie. Odpowiedz jednym słowem: 'pozytywny', 'neutralny' lub 'negatywny'.
  ```

- Label mapping:
  - `positive` ➡️ `pozytywny`
  - `neutral` ➡️ `neutralny`
  - `negative` ➡️ `negatywny`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset polemo2
```

## Named Entity Recognition

### KPWr-NER

This dataset was published in [this paper](https://aclanthology.org/L12-1574/) and is
part of the KPWr ("Korpus Języka Polskiego Politechniki Wrocławskiej" - "Polish Corpus
of Wrocław University of Technology") corpus - a free Polish corpus annotated with
various types of linguistic entities including named entities. The corpus was created to
serve as training and testing material for Machine Learning algorithms and is released
under a Creative Commons licence. The named entity annotations include persons,
locations, organizations, and miscellaneous entities, which are mapped to standard BIO
format labels.

The original dataset uses the train and test splits from the source corpus. The original
data train split has 13,959 samples and test split has 4,323 samples. The validation
split is created from the original training split. We use 1,024 / 256 / 2,048 samples
for our training, validation and test splits, respectively. The train and validation
    splits are subsets of the original training split, while the test split is a subset
    of the original test split.

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

  ```text
  Poniżej znajdują się zdania i słowniki JSON z jednostkami nazewniczymi, które występują w danym zdaniu.
  ```

- Base prompt template:

  ```text
  Zdanie: {text}
  Jednostki nazewnicze: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Zdanie: {text}

  Zidentyfikuj jednostki nazewnicze w zdaniu. Wypisz je jako słownik JSON z kluczami 'osoba', 'miejsce', 'organizacja' i 'inne'. Wartości odpowiadające kluczom powinny być listami jednostek nazewniczych danego typu, dokładnie tak, jak pojawiają się w zdaniu.
  ```

- Label mapping:
  - `B-PER` ➡️ `osoba`
  - `I-PER` ➡️ `osoba`
  - `B-LOC` ➡️ `miejsce`
  - `I-LOC` ➡️ `miejsce`
  - `B-ORG` ➡️ `organizacja`
  - `I-ORG` ➡️ `organizacja`
  - `B-MISC` ➡️ `inne`
  - `I-MISC` ➡️ `inne`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset kpwr-ner
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

The original full dataset consists of 22,152 samples, from which we use 1,024 / 256 /
2,048 samples for training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
    "text": "Papierową śmierć zafundowaliśmy już kilku osobom.",
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

  ```text
  Poniżej znajdują się teksty i informacja, czy są gramatycznie poprawne.
  ```

- Base prompt template:

  ```text
  Tekst: {text}
  Gramatycznie poprawny: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Tekst: {text}

  Określ, czy tekst jest gramatycznie poprawny. Odpowiedz używając wyłącznie {labels_str}.
  ```

- Label mapping:
  - `correct` ➡️ `tak`
  - `incorrect` ➡️ `nie`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-pl
```

## Reading Comprehension

### PoQuAD

PoQuAD was published in [this paper](https://doi.org/10.1145/3587259.3627548) and is a
Polish Question Answering dataset with contexts from Polish Wikipedia. It follows the
SQuAD format with innovations including lower annotation density, abstractive answers,
polar questions, and impossible questions.

The original dataset consists of 51,951 samples. We use 1,024 / 256 / 2,048 samples for
training, validation and testing, respectively. We do not use the impossible questions
in this version of the dataset.

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

  ```text
  Poniżej znajdują się teksty z towarzyszącymi pytaniami i
  odpowiedziami.
  ```

- Base prompt template:

  ```text
  Tekst: {text}
  Pytanie: {question}
  Odpowiedź z użyciem maksymalnie 3 słów: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Tekst: {text}

  Odpowiedz na następujące pytanie dotyczące powyższego tekstu, używając maksymalnie 3 słów.

  Pytanie: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset poquad
```

### Unofficial: MultiWikiQA-pl

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
    'context': 'Marcus Terrell Thornton (ur. 5 czerwca 1987 w Baton Rouge) – amerykański koszykarz, występujący na pozycji rzucającego obrońcy, wybrany do drugiego składu najlepszych debiutantów NBA.\n\n25 lipca 2015 roku podpisał umowę z Houston Rockets.\n\n18 lutego 2016 w ramach wymiany między trzema klubami miał trafić do Detroit Pistons. Jednak cztery dni później umowa została anulowana, ponieważ inny gracz biorący udział w wymianie, litewski skrzydłowy Donatas Motiejūnas, nie przeszedł testów medycznych i tym samym Thornton pozostał w drużynie Houston Rockets. 26 lutego 2016 roku został zwolniony przez klub Rockets. 9 marca 2016 roku podpisał umowę do końca sezonu z klubem Washington Wizards.\n\n22 lutego 2017 został wytransferowany wraz z Andrew Nicholsonem oraz przyszłym wyborem I rundy draftu 2017 do Brooklyn Nets w zamian za Bojana Bogdanovicia i Chrisa McCullougha. Kolejnego dnia został zwolniony przez Nets.\n\nOsiągnięcia \nStan na 29 grudnia 2020, na podstawie, o ile nie zaznaczono inaczej.\n College\n Uczestnik turnieju NCAA (2009)\n Mistrz sezonu regularnego konferecji Southeastern NCAA (SEC – 2009)\n Zawodnik roku konferencji Southeastern (2009)\n MVP turnieju NJCAA Basketball Coaches Association Classic\n Najlepszy nowo przybyły zawodnik konferencji SEC (2008)\n Zaliczony do:\n I składu: \n SEC (2008, 2009)\n All-Louisiana (2008)\n NJCAA All-American (2007)\n\n NBA\n Wybrany do II składu debiutantów NBA (2010)\n\n Inne\n Uczestnik meczu gwiazd G-League (2018)\n\nPrzypisy\n\nLinki zewnętrzne \n Profil na NBA.com \n Statystyki na basketball-reference.com \n Profil na landofbasketball.com \n\nAmerykańscy koszykarze\nKoszykarze Boston Celtics\nKoszykarze New Orleans Hornets\nKoszykarze Sacramento Kings\nKoszykarze Phoenix Suns\nKoszykarze Houston Rockets\nKoszykarze LSU Tigers\nKoszykarze Grand Rapids Drive\nKoszykarze Washington Wizards\nKoszykarze Brooklyn Nets\nKoszykarze Beijing Ducks\nUrodzeni w 1987\nLudzie urodzeni w Baton Rouge',
    'question': 'Gdzie Thornton przyszedł na świat?',
    'answers': {
        'answer_start': array([46]),
        'text': array(['Baton Rouge'], dtype=object)
    }
}
```

```json
{
    "context": "Leonowo – dawny folwark. Tereny, na których był położony leżą obecnie na Białorusi, w obwodzie mińskim, w rejonie miadzielskim, w sielsowiecie Krzywicze.\n\nHistoria \nW czasach zaborów folwark prywatny w powiecie wilejskim, w guberni wileńskiej Imperium Rosyjskiego. W 1866 roku liczył 18 mieszkańców w 1 domu.\n\nW latach 1921–1945 folwark leżał w Polsce, w województwie wileńskim, w powiecie wilejskim, w gminie Krzywicze.\n\nWedług Powszechnego Spisu Ludności z 1921 roku zamieszkiwały tu 24 osoby, 17 było wyznania rzymskokatolickiego a 7 mahometańskiego. Jednocześnie 17 mieszkańców zadeklarowało polską a 7 białoruską przynależność narodową. Były tu 3 budynki mieszkalne. W 1931 w 2 domach zamieszkiwało 17 osób.\n\nWierni należeli do parafii rzymskokatolickiej i prawosławnej w Krzywiczach. Miejscowość podlegała pod Sąd Grodzki w Krzywicze i Okręgowy w Wilnie; właściwy urząd pocztowy mieścił się w Krzywiczach.\n\nW wyniku napaści ZSRR na Polskę we wrześniu 1939 miejscowość znalazła się pod okupacją sowiecką. 2 listopada została włączona do Białoruskiej SRR. Od czerwca 1941 roku pod okupacją niemiecką. W 1944 miejscowość została ponownie zajęta przez wojska sowieckie i włączona do Białoruskiej SRR.\n\nUwagi\n\nPrzypisy\n\nLinki zewnętrzne \n\n \n\nRejon miadzielski\nOpuszczone miejscowości na Białorusi\nMiejscowości województwa wileńskiego (II Rzeczpospolita)",
    "question": "Jaka była liczba ludności Leonowa w 1921 roku?",
    "answers": {
        "answer_start": array([486]),
        "text": array(["24"], dtype=object)
    }
}
```

```json
{
    "context": "Carlos Manuel Brito Leal de Queiroz (wym. ; ur. 1 marca 1953 w Nampuli w Mozambiku) – portugalski trener piłkarski i piłkarz.\n\nKariera szkoleniowa \nBył bramkarzem miejscowego klubu Nampuli. W 1976 z powodu kontuzji musiał zakończyć piłkarską karierę. Pracę szkoleniową rozpoczął w Portugalii, z reprezentacją młodzieżową, z którą dwukrotnie – w 1989 i 1991 – zdobył tytuł mistrza świata. Jest twórcą największych sukcesów w historii młodzieżowej piłki portugalskiej i wychowawcą „Złotego pokolenia” portugalskich piłkarzy, którego najwybitniejsi przedstawiciele – Luís Figo, Rui Costa, Jorge Costa i Fernando Couto – stanowili później o sile dorosłej kadry.\n\nW 1990 został selekcjonerem reprezentacji A, ale nie udało mu się z nią awansować do Mundialu 1994. Do światowego czempionatu wprowadził za to Republikę Południowej Afryki, lecz został zwolniony na kilka miesięcy przed turniejem. Ponadto szkolił zespoły w Stanach Zjednoczonych, Japonii i Zjednoczonych Emiratach Arabskich; jest jednym z nielicznych trenerów, którzy pracowali na czterech różnych kontynentach.\n\nW 2003 dostał szansę od Realu Madryt, ale sezon spędzony w stolicy Hiszpanii – IV miejsce w Primera División i szybkie odpadnięcie z Ligi Mistrzów – był jednym z gorszych w całej historii klubu. W 2004 ponownie (wcześniej w latach 2002–2003) został asystentem Aleksa Fergusona w Manchesterze United. W tym czasie klub zdobył m.in. Puchar Mistrzów i dwa tytuły mistrza Anglii. Zdaniem wielu obserwatorów był szykowany na następcę Fergusona, jednak w lipcu 2008 zdecydował się przyjąć propozycję szefów Portugalskiego Związku Piłki Nożnej i po raz drugi w karierze poprowadził reprezentację Portugalii.\n\nBrał z nią udział w kwalifikacjach do Mundialu 2010. W grupie eliminacyjnej Portugalia zajęła drugie miejsce, za Danią. Do mistrzostw awansowała dzięki wygranej w barażach z Bośnią i Hercegowiną. Na samym turnieju jego podopieczni, wśród których znajdowali się m.in. Cristiano Ronaldo, Deco, Paulo Ferreira i Ricardo Carvalho, doszli do drugiej rundy, gdzie przegrali 0:1 z przyszłymi mistrzami świata Hiszpanami. W rozgrywkach grupowych wygrali z Koreą Północną i Wybrzeżem Kości Słoniowej oraz zremisowali z Brazylią.\n\nPo mistrzostwach Queiroz został zawieszony na pół roku za obrażenie kontrolerów antydopingowych. W tym czasie Portugalczycy (prowadzeni na boisku przez Agostinho Oliveirę) rozpoczęli eliminacje do Euro 2012; po dwu pierwszych meczach mieli na koncie tylko jeden punkt, po remisie z Cyprem (4:4) i porażce z Norwegią (0:1). 9 września, dwa dni po tym ostatnim spotkaniu, portugalska federacja postanowiła rozwiązać kontrakt z trenerem.\n\n4 kwietnia 2011 został selekcjonerem reprezentacji Iranu. Dwa lata później świętował z nią awans do Mundialu 2014.\n\n8 września 2021 roku został trenerem reprezentacji Egiptu.\n\nSukcesy szkoleniowe \n mistrzostwo świata U-20 1989 i 1991 z młodzieżową reprezentacją Portugalii\n wicemistrzostwo Portugalii 1996 ze Sportingiem\n awans do Mundialu 2002 z reprezentacją RPA\n awans do Mundialu 2010 i start w tym turnieju (1/8 finału) z reprezentacją Portugalii\n awans do Mundialu 2014 z reprezentacją Iranu\n\nOdznaczenia \n  Komandor Orderu Infanta Henryka (1989, Portugalia)\n\nZobacz też \n Złote pokolenie piłkarzy portugalskich\n\nPortugalscy trenerzy piłkarscy\nSelekcjonerzy reprezentacji Portugalii w piłce nożnej mężczyzn\nSelekcjonerzy reprezentacji Zjednoczonych Emiratów Arabskich w piłce nożnej mężczyzn\nSelekcjonerzy reprezentacji Południowej Afryki w piłce nożnej mężczyzn\nSelekcjonerzy reprezentacji Iranu w piłce nożnej mężczyzn\nSelekcjonerzy reprezentacji Kolumbii w piłce nożnej mężczyzn\nSelekcjonerzy reprezentacji Egiptu w piłce nożnej mężczyzn\nSelekcjonerzy reprezentacji Kataru w piłce nożnej mężczyzn\nTrenerzy piłkarzy Realu Madryt\nTrenerzy piłkarzy Sportingu CP\nTrenerzy piłkarzy Nagoya Grampus\nTrenerzy piłkarzy New York Red Bulls\nOdznaczeni Orderem Infanta Henryka\nLudzie urodzeni w Nampuli\nUrodzeni w 1953",
    "question": "Kiedy Carlos Queiroz był selekcjonerem reprezentacji Portugalii na Mistrzostwach Świata?",
    "answers": {
        "answer_start": array([1720]),
        "text": array(["2010"], dtype=object)
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Poniżej znajdują się teksty z towarzyszącymi pytaniami i
  odpowiedziami.
  ```

- Base prompt template:

  ```text
  Tekst: {text}
  Pytanie: {question}
  Odpowiedź z użyciem maksymalnie 3 słów: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Tekst: {text}

  Odpowiedz na następujące pytanie dotyczące powyższego tekstu, używając maksymalnie 3 słów.

  Pytanie: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-pl
```

## Knowledge

### LLMzSzŁ

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2501.02266)
and is based on Polish national exams extracted from the archives of the Polish Central
Examination Board. LLMzSzŁ (LLMs Behind the School Desk) represents the first
comprehensive benchmark for the Polish language at this scale. The dataset features both
academic and professional tests covering 4 types of exams from 154 different domains.
The dataset was created to evaluate the ability of language models to transfer knowledge
between languages and to assess their performance on Polish educational content.

The original dataset consisted of almost 19,000 closed-ended questions in a single test
split. We use a 1,024 / 256 / 2,048 split for training, validation and testing,
respectively (so 3,328 samples used in total).

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

  ```text
  Poniżej znajdują się pytania wielokrotnego wyboru (z odpowiedziami).
  ```

- Base prompt template:

  ```text
  Pytanie: {text}
  Opcje:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Odpowiedź: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Pytanie: {text}
  Opcje:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Odpowiedz na powyższe pytanie, używając 'a', 'b', 'c' lub 'd' i niczego więcej.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset llmzszl
```

### Unofficial: INCLUDE-pl

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
    "text": "Proces tłumaczenia kodu źródłowego pisanego przez programistę na zrozumiały dla komputera kod maszynowy to\nOpcje:\na. kompilowanie.\nb. debugowanie.\nc. uruchamianie.\nd. implementowanie.",
    "label": "a",
    "subject": "Professional certification"
}
```

```json
{
    "text": "Dane są punkty 𝐾 = (−3, −7) oraz 𝑆 = (5, 3). Punkt 𝑆 jest środkiem odcinka 𝐾𝐿. Wtedy punkt 𝐿 ma współrzędne\nOpcje:\na. (13, 10)\nb. (13, 13)\nc. (1, −2)\nd. (7, −1)",
    "label": "b",
    "subject": "Math"
}
```

```json
{
    "text": "Przepisy konstytucji jednego z państw Art. 9. Prezydent […] przewodniczy Radzie Ministrów. Art. 21. Premier kieruje działalnością Rządu. […] Art. 24.1. Parlament uchwala ustawy. Kontroluje działalność Rządu.  Ocenia jego politykę. 2. Parlament składa się ze Zgromadzenia Narodowego i Senatu. 3. Deputowani do Zgromadzenia Narodowego […] są wybierani w wyborach bezpośrednich. 4. Senat […] wybierany jest w wyborach pośrednich. […] Art. 39.1. Inicjatywa ustawodawcza przysługuje zarówno Premierowi, jak i członkom Parlamentu. 2. Rządowe projekty ustaw są rozpatrywane przez Radę Ministrów […] i wnoszone do prezydium jednej z izb. […] Art. 45.1. Każdy rządowy lub parlamentarny projekt ustawy jest rozpatrywany kolejno przez obie izby Parlamentu w celu przyjęcia tekstu w tym samym brzmieniu. […] biblioteka.sejm.gov.pl (tekst uwzględniający zmiany z 23 lipca 2008 r.). Zaznacz nazwę państwa, z którego ustawy zasadniczej pochodzą przytoczone przepisy prawne.\nOpcje:\na. Republika Włoska\nb. Federacja Rosyjska\nc. Republika Francuska\nd. Stany Zjednoczone Ameryki",
    "label": "c",
    "subject": "Sociology"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Poniżej znajdują się pytania wielokrotnego wyboru (z odpowiedziami).
  ```

- Base prompt template:

  ```text
  Pytanie: {text}
  Odpowiedź: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Pytanie: {text}

  Odpowiedz na powyższe pytanie, używając {labels_str} i niczego więcej.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset include-pl
```

## Common-sense Reasoning

### Winogrande-pl

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2506.19468)
and is a translated and filtered version of the English [Winogrande
dataset](https://doi.org/10.1145/3474381).

The original full dataset consists of 47 / 1,210 samples for training and testing, and
we use 128 of the test samples for validation, resulting in a 47 / 128 / 1,085 split for
training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Gęsi wolą gniazdować na polach niż w lasach, ponieważ na _ drapieżniki są bardzo widoczne. Do kogo odnosi się puste miejsce _?\nOpcje:\na. pola\nb. lasy",
  "label": "a"
}
```

```json
{
  "text": "Kyle czuł się bardziej komfortowo, mówiąc przed dużymi grupami niż Nick, ponieważ _ brał udział w kursach przemówień publicznych na studiach. Do kogo odnosi się puste miejsce _?\nOpcje:\na. Kyle\nb. Nick",
  "label": "a"
}
```

```json
{
  "text": "Nie mogłem kontrolować wilgoci tak jak kontrolowałem deszcz, ponieważ _ wchodziła w jednym miejscu. Do kogo odnosi się puste miejsce _?\nOpcje:\na. wilgoci\nb. deszcz",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Poniżej znajdują się pytania wielokrotnego wyboru (z odpowiedziami).
  ```

- Base prompt template:

  ```text
  Pytanie: {text}
  Opcje:
  a. {option_a}
  b. {option_b}
  Odpowiedź: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Pytanie: {text}
  Opcje:
  a. {option_a}
  b. {option_b}

  Odpowiedz na powyższe pytanie, używając 'a' lub 'b' i niczego więcej.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset winogrande-pl
```

### Unofficial: GoldenSwag-pl

This dataset is a filtered and machine translated version of the English [HellaSwag
dataset](https://aclanthology.org/P19-1472/), featuring both video descriptions from
ActivityNet as well as how-to articles from WikiHow. The machine translated version was
published in [this paper](https://doi.org/10.48550/arXiv.2410.08928) and was done using
DeepL, and the filtering was published in [this
paper](https://doi.org/10.48550/arXiv.2504.07825), which resulted in higher quality
samples.

The original full dataset consists of 1530 / 1530 samples for training and validation,
respectively. However, they are exactly equal. We use a split of 660 / 256 / 2,048
samples for training, validation, and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Jak usunąć samoopalacz ze skóry? Nałóż oliwkę dla dzieci. W większości przypadków wilgoć pomaga zachować kolor opalenizny. Jednak oliwka dla dzieci ma odwrotne działanie i rozluźnia komórki skóry zabarwione samoopalaczem.\nOpcje:\na. Stosowanie oliwki dla dzieci może złagodzić lub całkowicie usunąć samoopalacz bez szkody dla skóry. Namocz skórę w oliwce dla niemowląt i pozostaw na dziesięć minut.\nb. Nakładaj oliwkę dla niemowląt po trochu i ugniataj ją, aby rozluźnić suche plamy. Możesz użyć niewielkiej ilości oliwki dla niemowląt na raz i użyć tylko odrobiny olejku na skórze bez słońca.\nc. Użyj miękkiej bawełnianej ściereczki lub gąbki, aby delikatnie wchłonąć trochę płynu. Następnie możesz nałożyć miękki bawełniany ręcznik na głowę i przykryć twarz.\nd. Spróbuj użyć niewielkiej ilości oliwki dla niemowląt 3 razy w tygodniu i delikatnie zetrzyj martwe komórki naskórka opuszkami palców. Jak wspomniano wcześniej, nie należy myć, nawilżać ani usuwać kremu do opalania bez słońca, ponieważ może to uniemożliwić przyleganie samoopalacza do skóry.",
  "label": "a"
}
```

```json
{
  "text": "Jak śledzić swoją dietę na fitbit. Załóż konto fitbit, jeśli jeszcze go nie masz. Kliknij łącze \"zaloguj się\" w prawym górnym rogu strony i wprowadź dane logowania do konta, aby się zalogować. Zaloguj się do swojego konta fitbit na stronie www.fitbit.com lub w aplikacji.\nOpcje:\na. Zostaniesz poproszony o podanie swoich danych osobowych (imię i nazwisko, numer telefonu, adres e-mail itp. ) oraz hasła, a następnie możesz ustawić hasło.\nb. To tutaj będziesz śledzić swoje dzienne spożycie kalorii. Pod pulpitem nawigacyjnym konta kliknij \" dziennik.\nc. Możesz pominąć ten krok, jeśli chcesz zalogować się z aplikacji apple.com myfitbit na swoim iPhonie lub iPadzie. Jeśli masz już konto na tym samym iPhonie lub iPadzie, nie musisz się logować.\nd. Wybierz, określ i wprowadź swoje dane żywieniowe. W prawym górnym rogu ekranu pojawi się lista wszystkich produktów o równej wadze.",
  "label": "b"
}
```

```json
{
  "text": "Jak mrozić warzywa ogrodowe. Wybieraj młode, świeżo zebrane warzywa. Warzywa ogrodowe, które leżały przez kilka dni, stracą część swojej świeżości, a zamrażanie może spowodować utratę ich dodatkowego smaku. Aby zapewnić, że warzywa pozostaną świeże tak długo, jak to możliwe, wybieraj warzywa, które są ledwo dojrzałe lub lekko niedojrzałe.\nOpcje:\na. Sprawdź, czy warzywa są w sezonie. Owoce, które są już dojrzałe lub dojrzałe, będą lepiej przechowywać się w zamrażarce, a produkty, które jeszcze nie wyschły, będą trwać dłużej, jeśli zamrażasz produkty bez ich odcedzania.\nb. Większe warzywa mogą dodać dodatkowego smaku i objętości poprzez szybsze wysychanie, ale nadal będą trwać dłużej i będą mniej przetworzone. Staraj się wybierać warzywa o długości około 10 cm (4 cale).\nc. Unikaj przejrzałych warzyw, które mogą się zepsuć nawet po zamrożeniu. Zbieraj warzywa wcześnie rano i natychmiast rozpocznij proces zamrażania, aby uzyskać najlepsze rezultaty.\nd. Jeśli nie jesteś fanem brokułów i ziemniaków, wybierz warzywa, które można zjeść bez zmiany koloru na pomarańczowy. Zdecyduj, jak mokre lub suche mają być warzywa.",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Poniżej znajdują się pytania wielokrotnego wyboru (z odpowiedziami).
  ```

- Base prompt template:

  ```text
  Pytanie: {text}
  Opcje:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Odpowiedź: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Pytanie: {text}
  Opcje:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Odpowiedz na powyższe pytanie, używając 'a', 'b', 'c' lub 'd' i niczego więcej.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset goldenswag-pl
```

## Summarisation

### PSC

The Polish Summaries Corpus (PSC) was published in [this
paper](https://aclanthology.org/L14-1145/) and is a resource created for automated
single-document summarization of Polish. The corpus contains manual summaries of news
articles, with multiple independently created summaries for single texts to overcome
annotator bias. It includes both abstract free-word summaries and extraction-based
summaries created by selecting text spans from the original documents.

The original dataset consists only of a training split. We use 1,024 / 256 / 2,048
samples for our training, validation and test splits, respectively. All splits are
subsets of the original training data, with the validation and test splits sampled from
the original training set.

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

  ```text
  Poniżej znajdują się artykuły z towarzyszącymi im streszczeniami.
  ```

- Base prompt template:

  ```text
  Artykuł: {text}
  Streszczenie: {target_text}
  ```

- Instruction-tuned prompt template:

  ```text
  Artykuł: {text}

  Napisz streszczenie powyższego artykułu.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset psc
```

## European Values

### ValEU-pl

This dataset is the official Polish version of questions from the [European values
study](https://europeanvaluesstudy.eu/). The dataset contains multiple-choice
questions regarding people's values and beliefs across a variety of topics, such as
politics, religion and society.

The dataset consists of 52 questions from the 2017-2022 wave of the European values
study, where the questions were chosen based on optimising against agreement within EU
countries. We use only zero-shot evaluation on this dataset, and thus require no splits.

Here are a few examples from the training split:

```json
{
  "question_id": "E069_02",
  "text": "Teraz wymienię różne instytucje i organizacje. Proszę powiedzieć, w jakim stopniu darzy je Pan/i/ zaufaniem: w bardzo dużym stopniu, raczej w dużym stopniu, w niezbyt dużym stopniu, czy w ogóle nie darzy zaufaniem?\nWojsko\nOpcje:\na. W bardzo dużym stopniu\nb. Raczej w dużym stopniu\nc. W niezbyt dużym stopniu\nd. W ogóle nie darzę zaufaniem"
}
```

```json
{
  "question_id": "E265_07",
  "text": "Jak często, Pana/i/ zdaniem, w czasie wyborów w Polsce mają miejsce następujące sytuacje?\nBogaci ludzie kupują wybory\nOpcje:\na. Bardzo często\nb. Raczej często\nc. Raczej rzadko\nd. W ogóle nie, bardzo rzadko"
}
```

```json
{
  "question_id": "E265_02",
  "text": "Jak często, Pana/i/ zdaniem, w czasie wyborów w Polsce mają miejsce następujące sytuacje?\nKandydaci opozycji nie mogą startować w wyborach\nOpcje:\na. Bardzo często\nb. Raczej często\nc. Raczej rzadko\nd. W ogóle nie, bardzo rzadko"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 0
- Prefix prompt:

  ```text
  Poniżej znajdują się pytania wielokrotnego wyboru (z odpowiedziami).
  ```

- Base prompt template:

  ```text
  Pytanie: {text}
  Opcje:
  a. {option_a}
  b. {option_b}
  (...)
  k. {option_k}
  Odpowiedź: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Pytanie: {text}
  Opcje:
  a. {option_a}
  b. {option_b}
  (...)
  k. {option_k}

  Odpowiedz na powyższe pytanie, używając 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
  'j' lub 'k' i niczego więcej.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset valeu-pl
```

## Hallucination Detection

### Unofficial: MultiWikiHalluQA-pl

This dataset uses the same data as [Unofficial: MultiWikiQA-pl](#unofficial-multiwikiqa-pl), published in
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
    'context': 'Marcus Terrell Thornton (ur. 5 czerwca 1987 w Baton Rouge) – amerykański koszykarz, występujący na pozycji rzucającego obrońcy, wybrany do drugiego składu najlepszych debiutantów NBA.\n\n25 lipca 2015 roku podpisał umowę z Houston Rockets.\n\n18 lutego 2016 w ramach wymiany między trzema klubami miał trafić do Detroit Pistons. Jednak cztery dni później umowa została anulowana, ponieważ inny gracz biorący udział w wymianie, litewski skrzydłowy Donatas Motiejūnas, nie przeszedł testów medycznych i tym samym Thornton pozostał w drużynie Houston Rockets. 26 lutego 2016 roku został zwolniony przez klub Rockets. 9 marca 2016 roku podpisał umowę do końca sezonu z klubem Washington Wizards.\n\n22 lutego 2017 został wytransferowany wraz z Andrew Nicholsonem oraz przyszłym wyborem I rundy draftu 2017 do Brooklyn Nets w zamian za Bojana Bogdanovicia i Chrisa McCullougha. Kolejnego dnia został zwolniony przez Nets.\n\nOsiągnięcia \nStan na 29 grudnia 2020, na podstawie, o ile nie zaznaczono inaczej.\n College\n Uczestnik turnieju NCAA (2009)\n Mistrz sezonu regularnego konferecji Southeastern NCAA (SEC – 2009)\n Zawodnik roku konferencji Southeastern (2009)\n MVP turnieju NJCAA Basketball Coaches Association Classic\n Najlepszy nowo przybyły zawodnik konferencji SEC (2008)\n Zaliczony do:\n I składu: \n SEC (2008, 2009)\n All-Louisiana (2008)\n NJCAA All-American (2007)\n\n NBA\n Wybrany do II składu debiutantów NBA (2010)\n\n Inne\n Uczestnik meczu gwiazd G-League (2018)\n\nPrzypisy\n\nLinki zewnętrzne \n Profil na NBA.com \n Statystyki na basketball-reference.com \n Profil na landofbasketball.com \n\nAmerykańscy koszykarze\nKoszykarze Boston Celtics\nKoszykarze New Orleans Hornets\nKoszykarze Sacramento Kings\nKoszykarze Phoenix Suns\nKoszykarze Houston Rockets\nKoszykarze LSU Tigers\nKoszykarze Grand Rapids Drive\nKoszykarze Washington Wizards\nKoszykarze Brooklyn Nets\nKoszykarze Beijing Ducks\nUrodzeni w 1987\nLudzie urodzeni w Baton Rouge',
    'question': 'Gdzie Thornton przyszedł na świat?',
    'answers': {
        'answer_start': array([46]),
        'text': array(['Baton Rouge'], dtype=object)
    }
}
```

```json
{
    "context": "Leonowo – dawny folwark. Tereny, na których był położony leżą obecnie na Białorusi, w obwodzie mińskim, w rejonie miadzielskim, w sielsowiecie Krzywicze.\n\nHistoria \nW czasach zaborów folwark prywatny w powiecie wilejskim, w guberni wileńskiej Imperium Rosyjskiego. W 1866 roku liczył 18 mieszkańców w 1 domu.\n\nW latach 1921–1945 folwark leżał w Polsce, w województwie wileńskim, w powiecie wilejskim, w gminie Krzywicze.\n\nWedług Powszechnego Spisu Ludności z 1921 roku zamieszkiwały tu 24 osoby, 17 było wyznania rzymskokatolickiego a 7 mahometańskiego. Jednocześnie 17 mieszkańców zadeklarowało polską a 7 białoruską przynależność narodową. Były tu 3 budynki mieszkalne. W 1931 w 2 domach zamieszkiwało 17 osób.\n\nWierni należeli do parafii rzymskokatolickiej i prawosławnej w Krzywiczach. Miejscowość podlegała pod Sąd Grodzki w Krzywicze i Okręgowy w Wilnie; właściwy urząd pocztowy mieścił się w Krzywiczach.\n\nW wyniku napaści ZSRR na Polskę we wrześniu 1939 miejscowość znalazła się pod okupacją sowiecką. 2 listopada została włączona do Białoruskiej SRR. Od czerwca 1941 roku pod okupacją niemiecką. W 1944 miejscowość została ponownie zajęta przez wojska sowieckie i włączona do Białoruskiej SRR.\n\nUwagi\n\nPrzypisy\n\nLinki zewnętrzne \n\n \n\nRejon miadzielski\nOpuszczone miejscowości na Białorusi\nMiejscowości województwa wileńskiego (II Rzeczpospolita)",
    "question": "Jaka była liczba ludności Leonowa w 1921 roku?",
    "answers": {
        "answer_start": array([486]),
        "text": array(["24"], dtype=object)
    }
}
```

```json
{
    "context": "Carlos Manuel Brito Leal de Queiroz (wym. ; ur. 1 marca 1953 w Nampuli w Mozambiku) – portugalski trener piłkarski i piłkarz.\n\nKariera szkoleniowa \nBył bramkarzem miejscowego klubu Nampuli. W 1976 z powodu kontuzji musiał zakończyć piłkarską karierę. Pracę szkoleniową rozpoczął w Portugalii, z reprezentacją młodzieżową, z którą dwukrotnie – w 1989 i 1991 – zdobył tytuł mistrza świata. Jest twórcą największych sukcesów w historii młodzieżowej piłki portugalskiej i wychowawcą „Złotego pokolenia” portugalskich piłkarzy, którego najwybitniejsi przedstawiciele – Luís Figo, Rui Costa, Jorge Costa i Fernando Couto – stanowili później o sile dorosłej kadry.\n\nW 1990 został selekcjonerem reprezentacji A, ale nie udało mu się z nią awansować do Mundialu 1994. Do światowego czempionatu wprowadził za to Republikę Południowej Afryki, lecz został zwolniony na kilka miesięcy przed turniejem. Ponadto szkolił zespoły w Stanach Zjednoczonych, Japonii i Zjednoczonych Emiratach Arabskich; jest jednym z nielicznych trenerów, którzy pracowali na czterech różnych kontynentach.\n\nW 2003 dostał szansę od Realu Madryt, ale sezon spędzony w stolicy Hiszpanii – IV miejsce w Primera División i szybkie odpadnięcie z Ligi Mistrzów – był jednym z gorszych w całej historii klubu. W 2004 ponownie (wcześniej w latach 2002–2003) został asystentem Aleksa Fergusona w Manchesterze United. W tym czasie klub zdobył m.in. Puchar Mistrzów i dwa tytuły mistrza Anglii. Zdaniem wielu obserwatorów był szykowany na następcę Fergusona, jednak w lipcu 2008 zdecydował się przyjąć propozycję szefów Portugalskiego Związku Piłki Nożnej i po raz drugi w karierze poprowadził reprezentację Portugalii.\n\nBrał z nią udział w kwalifikacjach do Mundialu 2010. W grupie eliminacyjnej Portugalia zajęła drugie miejsce, za Danią. Do mistrzostw awansowała dzięki wygranej w barażach z Bośnią i Hercegowiną. Na samym turnieju jego podopieczni, wśród których znajdowali się m.in. Cristiano Ronaldo, Deco, Paulo Ferreira i Ricardo Carvalho, doszli do drugiej rundy, gdzie przegrali 0:1 z przyszłymi mistrzami świata Hiszpanami. W rozgrywkach grupowych wygrali z Koreą Północną i Wybrzeżem Kości Słoniowej oraz zremisowali z Brazylią.\n\nPo mistrzostwach Queiroz został zawieszony na pół roku za obrażenie kontrolerów antydopingowych. W tym czasie Portugalczycy (prowadzeni na boisku przez Agostinho Oliveirę) rozpoczęli eliminacje do Euro 2012; po dwu pierwszych meczach mieli na koncie tylko jeden punkt, po remisie z Cyprem (4:4) i porażce z Norwegią (0:1). 9 września, dwa dni po tym ostatnim spotkaniu, portugalska federacja postanowiła rozwiązać kontrakt z trenerem.\n\n4 kwietnia 2011 został selekcjonerem reprezentacji Iranu. Dwa lata później świętował z nią awans do Mundialu 2014.\n\n8 września 2021 roku został trenerem reprezentacji Egiptu.\n\nSukcesy szkoleniowe \n mistrzostwo świata U-20 1989 i 1991 z młodzieżową reprezentacją Portugalii\n wicemistrzostwo Portugalii 1996 ze Sportingiem\n awans do Mundialu 2002 z reprezentacją RPA\n awans do Mundialu 2010 i start w tym turnieju (1/8 finału) z reprezentacją Portugalii\n awans do Mundialu 2014 z reprezentacją Iranu\n\nOdznaczenia \n  Komandor Orderu Infanta Henryka (1989, Portugalia)\n\nZobacz też \n Złote pokolenie piłkarzy portugalskich\n\nPortugalscy trenerzy piłkarscy\nSelekcjonerzy reprezentacji Portugalii w piłce nożnej mężczyzn\nSelekcjonerzy reprezentacji Zjednoczonych Emiratów Arabskich w piłce nożnej mężczyzn\nSelekcjonerzy reprezentacji Południowej Afryki w piłce nożnej mężczyzn\nSelekcjonerzy reprezentacji Iranu w piłce nożnej mężczyzn\nSelekcjonerzy reprezentacji Kolumbii w piłce nożnej mężczyzn\nSelekcjonerzy reprezentacji Egiptu w piłce nożnej mężczyzn\nSelekcjonerzy reprezentacji Kataru w piłce nożnej mężczyzn\nTrenerzy piłkarzy Realu Madryt\nTrenerzy piłkarzy Sportingu CP\nTrenerzy piłkarzy Nagoya Grampus\nTrenerzy piłkarzy New York Red Bulls\nOdznaczeni Orderem Infanta Henryka\nLudzie urodzeni w Nampuli\nUrodzeni w 1953",
    "question": "Kiedy Carlos Queiroz był selekcjonerem reprezentacji Portugalii na Mistrzostwach Świata?",
    "answers": {
        "answer_start": array([1720]),
        "text": array(["2010"], dtype=object)
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Poniżej znajdują się teksty z towarzyszącymi pytaniami i
  odpowiedziami.
  ```

- Base prompt template:

  ```text
  Tekst: {text}
  Pytanie: {question}
  Odpowiedź z użyciem maksymalnie 3 słów: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Tekst: {text}

  Odpowiedz na następujące pytanie dotyczące powyższego tekstu, używając maksymalnie 3 słów.

  Pytanie: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-hallucination-qa-pl
```
