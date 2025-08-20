# 🇱🇻 Latvian

This is an overview of all the datasets used in the Latvian part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.



## Reading Comprehension


### Unofficial: MultiWikiQA-lv

This dataset will be published in an upcoming paper, and contains Latvian Wikipedia
articles with generated questions and answers, using the LLM Gemini-1.5-pro.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
    "context": "Zvjaheļa (, līdz 2022. gadam — Novohrada-Volinska) ir pilsēta Ukrainas ziemeļrietumos, Žitomiras apgabala rietumos, Slučas upes krastā. Tā ir Zvjaheļas rajona administratīvais centrs. Attālums līdz apgabala centram Žitomirai ir .\n\nZvjaheļa ir ukraiņu tautas dzejnieces Lesjas Ukrainkas dzimtā pilsēta.\nŠeit ir dzimis Ukrainas armijas virspavēlnieks ģenerālis Valerijs Zalužnijs.\n\nVēsture \nVēstures avotos apdzīvotā vieta pirmoreiz minēta 1256. gadā Slučas labajā krastā kā Vozvjaheļa (Возвягель) Galīcijas-Volīnijas hronikā. Gadu vēlāk to par nepaklausību nodedzināja Galīcijas karalis Danila. Nākamo reizi apdzīvotā vieta minēta 1432. gadā jau Slučas kreisajā krastā kā Vzvjahoļas (Взвяголь) miests, bet 1499. gadā\xa0— Zvjahoļa (Звяголь). 1507. gadā miests ieguva tiesības būvēt pili un veidot pilsētu. Pēc Ļubļinas ūnijas 1569. gadā miests saukts par Zvjaheļu (Звягель, ).\n\n1793. gadā Zvjaheļa nonāca Krievijas Impērijas sastāvā. 1795. gadā miests ieguva Novohradas-Volinskas nosaukumu un pilsētas tiesības, un kļuva par jaunizveidotās Volīnijas guberņas centru (līdz 1804. gadam).\n\n2022. gada 16. jūnijā Novohradas-Volinskas domes deputāti nobalsoja par pilsētas pārdēvēšanu tās vēsturiskajā nosaukumā — Zvjaheļa. Vēlāk šo lēmumu apstiprināja Žitomiras apgabala dome. Ar Ukrainas Augstākās Radas dekrētu 2022. gada 16. novembrī pilsēta tika pārdēvēta par Zvjaheļu.\n\nAtsauces\n\nĀrējās saites",
    "question": "Kāds Ukrainas bruņoto spēku komandieris nāk no Zvjaheļas?",
    "answers": {
        "answer_start": array([349]),
        "text": array(["ģenerālis Valerijs Zalužnijs"], dtype=object)
    }
}
```
```json
{
    "context": "Bogota (), saukta arī Santafe de Bogota (Santa Fe de Bogotá), ir pilsēta Kolumbijas centrālajā daļā, 2640 metri virs jūras līmeņa. Kolumbijas galvaspilsēta, galvenais valsts politiskais, ekonomiskais un kultūras centrs. Kaut arī pilsēta atrodas tropiskajā joslā, augstkalnu apstākļu dēļ pilsētā nav karsts (vidējā gaisa temperatūra visu gadu - apmēram +15 grādi).\n\nVēsture \n\nPirms konkistadoru ierašanās Bogotas vietā bija čibču indiāņu galvenais centrs, kuru sauca par Bakatu (Bacatá).\n\nMūsdienu pilsētu nodibināja konkistadors Gonsalo Himeness de Kvesada (Gonzalo Jiménez de Quesada) 1538. gadā.\n\n1718. gadā Bogota kļuva par spāņu Jaunās Granādas vicekaralistes (Virreinato de Nueva Granada) centru.\n\n1810. gadā iedzīvotāji sacēlās pret spāņu varu, tomēr sacelšanās tika apspiesta. 1819. gadā Bogotu ieņēma Simona Bolivāra karaspēks.\n\n1819. gadā vicekaraliste ieguva neatkarību no Spānijas un Bogota kļuva par Lielkolumbijas (Gran Colombia) galvaspilsētu. Tomēr 1830. gadā Lielkolumbija sabruka un izveidojās Jaunā Granāda (mūsdienu Kolumbija), Ekvadora un Venecuēla. 1903. gadā ar ASV atbalstu pret solījumiem atļaut būvēt Panamas kanālu, neatkarību no Kolumbijas ieguva Panama.\n\n1948. gadā Bogotā tika nogalināts populārais kolumbiešu poltiķis Horhe Gaitans. Pilsētā izcēlās plaši nemieri un ielu kaujas. Sākās politiskās nestabilitātes periods (La Violencia), kurš turpinājās 10 gadus, gāja bojā no 180 000 līdz 300 000 kolumbiešu.\n\nCilvēki \n\nBogotā dzimuši:\n\n Egans Bernals (Egan Bernal, 1997) — riteņbraucējs;\n Ingrīda Betankūra (Íngrid Betancourt, 1961) — politiķe;\n Huans Pablo Montoija (Juan Pablo Montoya, 1975) — Formula 1 pilots;\n Katalina Sandino Moreno (Catalina Sandino Moreno, 1981) — aktrise;\n Kamilo Toress Restrepo (Camilo Torres Restrepo, 1929-1966) — revolucionārs.\n\nĀrējās saites \n\nDienvidamerikas galvaspilsētas\nKolumbijas pilsētas",
    "question": "Kad Bogata tika iecelta par Jaunās Granādas vicekaralistes centru Spānijas pakļautībā?",
    "answers": {
        "answer_start": array([599]),
        "text": array(["1718. gadā"], dtype=object)
    }
}
```
```json
{
    "context": "Džastins Šulcs (; dzimis ) ir kanādiešu hokejists, aizsargs. Pašlaik (2020) Šulcs spēlē Nacionālās hokeja līgas kluba Vašingtonas "Capitals" sastāvā.\n\nSpēlētāja karjera \nPēc vairākām NCAA čempionātā aizvadītām sezonām, profesionāļa karjeru Šulcs sāka 2012.—13. gada sezonā, tajā spēles laiku dalot starp NHL klubu Edmontonas "Oilers" un AHL vienību Oklahomsitijas "Barons". "Oilers" Šulcs aizvadīja 48 spēles, savukārt AHL kļuva par līgas rezultatīvāko aizsargu, tiekot atzīts arī par līgas labāko aizsargu. 2013.—14. gada sezonu Šulcs jau pilnībā aizvadīja "Oilers" sastāvā.\n\nPēc neveiksmīga 2015.—16. gada sezonas ievada Šulcs tika aizmainīts uz Pitsburgas "Penguins". Tās sastāvā 2016. un 2017. gadā viņš izcīnīja Stenlija kausu. "Penguins" sastāvā spēlēja līdz 2020. gadam, kad pievienojās Vašingtonas "Capitals".\n\nĀrējās saites \n\n1990. gadā dzimušie\nKanādas hokejisti\nEdmontonas "Oilers" spēlētāji\nPitsburgas "Penguins" spēlētāji\nVašingtonas "Capitals" spēlētāji\nStenlija kausa ieguvēji\nBritu Kolumbijā dzimušie",
    "question": "Kad Džastins Šulcs uzsāka savu profesionālo karjeru?",
    "answers": {
        "answer_start": array([251]),
        "text": array(["2012.—13. gada sezonā"], dtype=object)
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:
  ```
  Turpmāk seko teksti ar atbilstošiem jautājumiem un atbildēm.
  ```
- Base prompt template:
  ```
  Teksts: {text}
  Jautājums: {question}
  Atbildēt ar maksimāli 3 vārdiem:
  ```
- Instruction-tuned prompt template:
  ```
  Teksts: {text}

  Atbildiet uz šo jautājumu par iepriekš minēto tekstu ar maksimāli 3 vārdiem.

  Jautājums: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset multi-wiki-qa-lv
```
