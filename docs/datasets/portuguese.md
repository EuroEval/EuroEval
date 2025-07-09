# 🇵🇹 Portuguese

This is an overview of all the datasets used in the European Portuguese part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### Unofficial: SST-2 PT

This dataset was published in [this paper](https://arxiv.org/abs/2404.05333) and is part of the extraglue dataset. It is created by taking the original SST-2 dataset and using machine translation (DeepL) to translate it.

The original dataset contains 67,300 training, 872 validation, and 1,820 test samples. We use 1,024 / 256 / 2,048 samples for train / val / test respectively. Given that the original validation dataset only has 1,820 sample for testing, we derive that split from the training split, while ensuring no overlaps occur. This dataset only includes positive and negative labels, no neutrals.

Here are a few examples from the training split:

```json
{
  "text": "um drama psicológico absorvente e inquietante .",
  "label": "positive"
}
```

```json
{
  "text": "tudo o que não se pode suportar",
  "label": "negative"
}
```

```json
{
  "text": "má escrita",
  "label": "negative"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Abaixo encontras documentos e os seus sentimentos correspondentes, que podem ser 'positivo' ou 'negativo'.
  ```
- Base prompt template:
  ```
  Documento: {text}
  Sentimento: {label}
  ```
- Instruction-tuned prompt template:

  ```
  Texto: {text}

  Clasifica o sentimento do documento. Responde apenas com 'positivo' ou 'negativo'.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset sst2-pt
```

## Reading Comprehension

### Unofficial: BoolQ-PT

This dataset was published in [this paper](https://arxiv.org/abs/2404.05333) and is part of  the extraglue dataset. It is created by taking the original BoolQ dataset and using machine translation (DeepL) to translate it.

The original dataset has a passage, question, and yes/no label. We adapt this dataset by taking the original passage, question, and yes/no options, and turning it into a Q/A style question where the model can answer yes or no.

The original dataset contains 9,430 training, 3,270 validation, and 3,250 test samples. We use 1,024 / 256 / 2,048 samples for train / val / test respectively. We've observed some overlap in the splits, so decided to concatenate all splits into a single dataset, shuffling it, and extract splits.

Here are a few examples from the training split:

```json
{
  "text": "Texto: Animais Fantásticos e Onde Encontrá-los -- Fantastic Beasts and Where to Find Them é um livro de 2001 escrito pela autora britânica J.K. Rowling (sob o pseudónimo do autor fictício Newt Scamander) sobre as criaturas mágicas do universo Harry Potter. A versão original, ilustrada pela própria autora, pretende ser a cópia de Harry Potter do livro didático com o mesmo nome mencionado em Harry Potter e a Pedra Filosofal (ou Harry Potter and the Sorcerer's Stone nos EUA), o primeiro romance da série Harry Potter. Inclui várias notas no seu interior, supostamente escritas à mão por Harry, Ron Weasley e Hermione Granger, detalhando as suas próprias experiências com algumas das bestas descritas e incluindo piadas relacionadas com a série original.\nPergunta: Animais fantásticos e onde encontrá-los está relacionado com Harry Potter?\nOpções:\na. sim\nb. não",
  "label": "a"
}
```

```json
{
  "text": "Texto: Leslie Aun, vocero de la Fundación Komen, informó que rige una nueva normativa en la organización conforme la cual no procederá el otorgamiento de subvenciones o fondos en favor de entidades que sean objeto de investigación oficial. La política de Komen desacreditó a Planned Parenthood a raíz de una investigación en curso que dirige el representante Cliff Stearns sobre la forma en la que esta organización informa y utiliza sus fondos. En su rol de director del Subcomité de Supervisión e Investigación, que se encuentra bajo el paraguas del Comité de Energía y Comercio, Stearns conduce una investigación para determinar si los impuestos se usan para financiar interrupciones de embarazos a través de Paternidad Planificada.\nPregunta: ¿Qué comité preside Cliff Stearns?\nOpciones:\na. Comité de Energía y Comercio de la Cámara de Representantes\nb. La Fundación Komen\nc. Planned Parenthood\nd. El Subcomité de Supervisión e Investigación",
  "label": "d"
}
```

```json
{
  "text": "Texto: Payola -- Payola, na indústria musical, é a prática ilegal de pagamento ou outro incentivo por parte das empresas discográficas para a difusão de gravações na rádio comercial, em que a canção é apresentada como fazendo parte da emissão normal do dia, sem o anunciar antes da emissão. Ao abrigo da legislação dos EUA, uma estação de rádio pode tocar uma canção específica em troca de dinheiro, mas tal deve ser divulgado no ar como sendo um tempo de antena patrocinado, e essa reprodução da canção não deve ser contada como uma "emissão regular".\nPergunta: A payola é legal no Canadá e nos Estados Unidos?\nOpções:\na. sim\nb. não",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  As seguintes são perguntas de escolha múltipla (com respostas).
  ```
- Base prompt template:
  ```
  Pergunta: {text}
  Opções:
  a. {option_a}
  b. {option_b}
  Resposta: {label}
  ```
- Instruction-tuned prompt template:

  ```
  Pergunta: {text}
  Opções:
  a. {option_a}
  b. {option_b}

  Responde à pergunta acima usando só 'a' ou 'b', e nada mais.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset boolq-pt
```
