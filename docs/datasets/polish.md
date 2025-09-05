# 🇵🇱 Polish

This is an overview of all the datasets used in the Polish part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

## Named Entity Recognition

## Linguistic Acceptability

## Reading Comprehension

### PoQuAD

PoQuAD is a Polish Question Answering dataset with contexts from Polish Wikipedia. It follows the SQuAD format with innovations including lower annotation density, abstractive answers, polar questions, and impossible questions.

This dataset was published in [this paper](https://dl.acm.org/doi/10.1145/3587259.3627548).

The original dataset consists of 51,951 samples. We use 1,024 / 256 / 2,048 samples for training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
    "context": "Gdański MKS reprezentuje 388 zakładów. Lech Wałęsa zwraca się do SB, by zaprzestały szykan wobec strajkujących i opozycyjnych działaczy. O godzinie 14 do stoczni przybywa wojewoda gdański w celu ustalenia szczegółów rozmów z delegacją rządową. 6 godzin później przybywa ona z wicepremierem Jagielskim na czele do Gdańska i rozpoczyna rozmowy z MKS-em. Transmituje się je na cały zakład. Nieco później do rozmów dołącza delegacja szczecińskiego MKS-u (skupiał on wtedy 134 zakłady). Ukazuje się pierwszy numer Strajkowego Biuletynu Informacyjnego „Solidarność”.\nW Gdańskiej Stoczni Remontowej na placu przed budynkiem Dyrekcji GSR odbywa się Msza Święta, którą koncelebruje ks. Henryk Jankowski.",
    "question": "Kto odprawił mszę w stoczni?",
    "answers": {
        "text": array(["ks. Henryk Jankowski"], dtype=object),
        "answer_start": array([673], dtype=int32)
    }
}
```
```json
{
    "context": "W swojej kampanii reklamowej, uruchomionej z okazji otwarcia roller coastera, park błędnie określił Zadrę jako kolejkę górską drewnianą (ang. wooden coaster), podczas gdy według standardów branży model I-Box stanowi kolejkę górską stalową (stalowy tor) o hybrydowej konstrukcji podpór (stalowo-drewnianej). Niejasna była również sytuacja dotycząca ostatecznej wysokości konstrukcji, którą park określał w swoich materiałach jako 61, 63, a nawet 63,8 m. Podana wysokość 62,8 m pochodzi z wywiadu udzielonego w 2018 roku magazynowi branżowemu First Drop przez projektanta kolejki, Alana Schilke. W 2021 roku producent atrakcji potwierdził wymienione w wywiadzie parametry techniczne.", "question": "Z jakich materiałów zostały wykonane filary hybrydowej kolejki?",
    "answers": {
        "text": array(["stalowo-drewnianej"], dtype=object),
        "answer_start": array([286], dtype=int32)
        }
}
```
```json
{
    "context": "Król spędzał swój czas w Aubonne, małej szwajcarskiej wiosce nad jeziorem Leman, ale od kilku lat mieszkał też w swojej rezydencji w zachodniej Rumunii (w Săvârşin, zamku kupionym przez rodzinę królewską w 1943 roku). Rodzina królewska korzysta także z prywatności zapewnionej przez Pałac Elżbiety, posiadłości w zielonym obszarze na północ od Bukaresztu, udostępnionym rodzinie królewskiej przez władze Rumunii. Tutaj właśnie, na pierwszym piętrze pałacu, król Michał zmuszony został do abdykacji 30 grudnia 1947 roku.",
    "question": "Jaki organ pozwolił rodzinie królewskiej dysponować Pałacem Elżbiety?",
    "answers": {
        "text": array(["władze Rumunii"], dtype=object),
        "answer_start": array([397], dtype=int32)
    }
}
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

## Common-sense Reasoning

## Summarization
