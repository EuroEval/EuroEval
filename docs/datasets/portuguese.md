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

This dataset was published in [this paper](https://arxiv.org/abs/2404.05333) and is part of the extraglue dataset. It is created by taking the original BoolQ dataset and using machine translation (DeepL) to translate it.

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
  "text": "Texto: Oceano Antártico -- O Oceano Antártico, também conhecido como Oceano Antártico ou Oceano Austral, compreende as águas mais a sul do Oceano Mundial, geralmente consideradas a sul de 60° de latitude sul e circundando a Antárctida. Como tal, é considerado como a quarta maior das cinco principais divisões oceânicas: mais pequeno do que os oceanos Pacífico, Atlântico e Índico, mas maior do que o oceano Ártico. Esta zona oceânica é o local onde as águas frias da Antárctida, que fluem para norte, se misturam com as águas subantárcticas, mais quentes.\nPergunta: Existe um oceano chamado oceano Austral?\nOpções:\na. sim\nb. não",
  "label": "a"
}
```

```json
{
  "text": "Texto: Lista dos votos de desempate dos vice-presidentes dos Estados Unidos -- O vice-presidente dos Estados Unidos é o presidente ex officio do Senado, como previsto no artigo I, secção 3, cláusula 4, da Constituição dos Estados Unidos, mas só pode votar para desempatar. De acordo com o Senado dos Estados Unidos, até 28 de fevereiro de 2018, o voto de desempate foi dado 264 vezes por 36 vice-presidentes.\nPergunta: O vice-presidente já desempatou alguma vez no Senado?\nOpções:\na. sim\nb. não"
  "label": "a"
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

## Knowledge

### MMLU-pt

This dataset is a subset of [MMLUx](https://huggingface.co/datasets/LumiOpen/opengpt-x_mmlux). Originally from openGPT-X/mmlux.

The original full dataset consists of 270 / 1,439 / 14,774 samples for training, validation, and testing, respectively. As this is not expected from EuroEval, we merged them, removed any duplicates, and then created new splits with 256 / 2048 / 1024 samples for validation, test, and training, respectively.

Here are a few examples from the training split:

```json
{
  "text": "De que tipo de direitos gozam os Estados costeiros sobre a sua plataforma continental?\nOpções:\na. O Estado costeiro goza ipso facto e ab initio de direitos soberanos sobre a sua plataforma continental para efeitos de exploração e aproveitamento dos seus recursos naturais\nb. O Estado costeiro só pode exercer direitos soberanos sobre a sua plataforma continental mediante declaração\nc. O Estado costeiro exerce direitos soberanos sobre a sua plataforma continental para efeitos de exploração dos seus recursos haliêuticos\nd. O Estado costeiro só pode exercer direitos limitados sobre a sua plataforma continental e apenas com o consentimento dos Estados vizinhos",
  "label": "a"
}
```

```json
{
  "text": "Qual delas não é uma competência-chave reconhecida da gestão?\nOpções:\na. Competências conceptuais\nb. Competências humanas\nc. Competências técnicas\nd. Competências de redação",
  "label": "d"
}
```

```json
{
    "text": "O presidente executa um "veto de bolso" fazendo qual das seguintes opções?\nOpções:\na. Manifestando publicamente a rejeição de um projeto de lei\nb. Emitindo uma ordem executiva que invalida um projeto de lei recentemente aprovado\nc. Não assinando um projeto de lei após o encerramento do Congresso\nd. Retirando embaixadores de uma negociação de paz",
    "label": "c",
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Las siguientes son preguntas de opción múltiple (con respuestas).
  ```
- Base prompt template:
  ```
  Pergunta: {text}
  Opções:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Resposta: {label}
  ```
- Instruction-tuned prompt template:

  ```
  Pergunta: {text}
  Opções:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Responde à pergunta acima usando só 'a' ou 'b', 'c' ou 'd', e nada mais.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset mmlu-pt
```

## Common-sense Reasoning

### HellaSwag-pt

This dataset is a subset of [HellaswagX](https://huggingface.co/datasets/LumiOpen/opengpt-x_goldenswagx). Originally from openGPT-X/hellaswagx.

The original full dataset consists of 1530 / 1530 samples for training and validation, respectively. However, they are exactly equal. For EuroEval, we use a split of 660 / 256 / 2.05k samples for training, validation, and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Como fazer com que o seu namorado à distância se sinta especial. Escreva uma carta de amor à moda antiga para enviar por correio normal. Embora seja possível enviar um e-mail instantaneamente, receber um pacote ou uma carta pelo correio é um esforço muito mais íntimo e sincero. As cartas também criam uma recordação que não pode ser feita por correio eletrónico.\nOpções:\na. Não se preocupe em escrever o poema perfeito ou algo profundo, o facto de se ter esforçado por escrever é suficiente. Pode fazer um desenho, encontrar um cartão pré-fabricado ou até enviar um postal de um local especial.\nb. Considere a possibilidade de criar um álbum de recortes com as notas do seu casamento como forma de surpreender o seu namorado com flores, um colar sentido ou até uma caixa com os brinquedos favoritos dele. A carta irá acompanhar a maioria dos filmes favoritos dele, dos quais você e o seu homem gostam de falar.\nc. Numa carta, escrevem-se palavras que vão até ao coração da pessoa. Se quiser enganar alguém para que lhe conte um pequeno segredo que lhe contou, tem de ter cuidado.\nd. Escreva-o em silêncio, não em voz alta e clara, e peça ao destinatário que o leia duas vezes. Utilize a linha de assunto para explicar a razão pela qual está a escrever ao seu namorado.",
  "label": "a"
}
```

```json
{
  "text": "Como cultivar inhame. Comece a cultivar os rebentos. Os inhames não são cultivados a partir de sementes como a maioria dos outros vegetais - eles crescem a partir de estacas, que são derivadas dos rebentos de inhames adultos. Para fazer crescer os rebentos, corte um inhame ao meio e mergulhe uma das partes num copo de água fria.\nOpções:\na. Mesmo antes de as plantas começarem a brotar, escave um pedaço do caule e coloque-o debaixo da água para que fique nivelado com o fundo do copo. Repita este processo até ter cerca de 5 cm de caule.\nb. A meio do processo de imersão, feche a outra metade num balde de água comercial. Pense em usar latas, baldes tupperware e outros recipientes que sejam grandes o suficiente para conter vários inhames de uma vez.\nc. Você deve ver as sementes brotarem. Se não conseguir, corte pequenas secções e mantenha os rebentos no copo de água fria.\nd. Insira palitos de dentes em três pontos à volta do meio do inhame e suspenda-o sobre o recipiente, meio submerso na água. Certifique-se de que o inhame escolhido tem um aspeto saudável.",
  "label": "d"
}
```

```json
{
"text": "Como detetar o plágio. Utilize aplicações online gratuitas que não requerem subscrições ou inscrições para verificar documentos electrónicos. Pesquise no Google "verificador de plágio" para encontrar uma série de aplicações Web gratuitas que contêm caixas onde pode colar o texto suspeito. Carregue no botão verificar e deixe que a aplicação analise a Internet em busca de instâncias de texto duplicado.\nOpções:\na. Qualquer coisa que apareça indica que está a utilizar uma destas aplicações gratuitas. Normalmente, é necessário iniciar sessão no início da aplicação.\nb. Cuidado! Utilizar os motores de busca para descobrir alguns sites oficiais de educação e classificá-los como "falsos". Exemplo: ' math problem manuscript for mr.\nc. Se quiser converter pdfs em texto, pode fazê-lo. Alguém que entregue um documento pdf, embora não seja inerentemente suspeito, pode ser um sinal de que está a tentar evitar ser apanhado.\nd. Aparecerá uma janela de teste a perguntar se precisa de uma aplicação de pesquisa. Se não precisar, escolha google ' anti-pasteurização.",
"label": "c",
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Las siguientes son preguntas de opción múltiple (con respuestas).
  ```
- Base prompt template:
  ```
  ergunta: {text}
  Opções:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Resposta: {label}
  ```
- Instruction-tuned prompt template:

  ```
  ergunta: {text}
  Opções:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Responde à pergunta acima usando só 'a' ou 'b', 'c' ou 'd', e nada mais.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset hellaswag-pt
```
