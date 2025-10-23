# 🇵🇹 Portuguese

This is an overview of all the datasets used in the European Portuguese part of
EuroEval. The datasets are grouped by their task - see the [task overview](/tasks) for
more information about what these constitute.

## Sentiment Classification

### SST2-PT

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2404.05333)
and is part of the ExtraGLUE dataset. It is created by taking the original SST-2 dataset
and using machine translation (DeepL) to translate it.

The original dataset contains 67,300 training, 872 validation, and 1,820 test samples.
We use 1,024 / 256 / 2,048 samples for train / val / test respectively. Given that the
original validation dataset only has 1,820 sample for testing, we derive that split from
the training split, while ensuring no overlaps occur. This dataset only includes
positive and negative labels, no neutrals.

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

  ```text
  Abaixo encontras documentos e os seus sentimentos correspondentes, que podem ser 'positivo' ou 'negativo'.
  ```

- Base prompt template:

  ```text
  Documento: {text}
  Sentimento: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Texto: {text}

  Clasifica o sentimento do documento. Responde apenas com 'positivo' ou 'negativo'.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset sst2-pt
```

## Named Entity Recognition

### HAREM

This dataset was published in [this paper](https://aclanthology.org/L06-1027/) and is
based on the [Primeiro HAREM](https://www.linguateca.pt/harem/) evaluation campaign for
**Portuguese from Portugal**, using the manually annotated **Colecção Dourada**. The
text sources come from varied sources: web, news, fiction books, politics, email,
speeches, technical, expository.

We extract only documents where `<ORIGEM>` is `PT`, i.e., of **Portuguese origin**. The
raw XML annotations are parsed and converted to token-level BIO labels. Tags are mapped
to standard CoNLL categories:

- `PER` (pessoa)
- `LOC` (local)
- `ORG` (organização)
- `MISC` (diverso)

Labels follow the standard CoNLL BIO scheme with numeric encoding:

```python
{
  "O": 0,
  "B-PER": 1,
  "I-PER": 2,
  "B-ORG": 3,
  "I-ORG": 4,
  "B-LOC": 5,
  "I-LOC": 6,
  "B-MISC": 7,
  "I-MISC": 8
}
```

In addition to tokenization and label alignment, each document is split into individual
sentences, using punctuation-based heuristics. This makes the dataset better suited for
sentence-level inference and generation.

Due to the limited number of PT-origin documents (1,965 examples total), we couldn’t
reach the target of 2,304 (1,024 + 256 + 1,024). The final split is:

- Train: 873 examples
- Validation: 218 examples
- Test: 874 examples

```json
{
  "tokens": array(["Na", "Covilhã", "ainda", "não", "havia", "liceu", "nessa", "altura", "."], dtype=object),
  "labels": array([0, 5, 0, 0, 0, 0, 0, 0, 0], dtype=object)
}
```

```json
{
 "tokens": array(["Por", "exemplo", ",", "em", "Filosofia", "está", "muito", "boa", "."], dtype=object),
  "labels": array([0, 0, 0, 0, 7, 0, 0, 0, 0], dtype=object)
}
```

```json
{
  "tokens": array(["Sabe", "qual", "a", "origem", "da", "sua", "família", "?"], dtype=object),
  "labels": array([0, 0, 0, 0, 0, 0, 0, 0], dtype=object)
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Seguem-se frases e dicionários JSON com as entidades mencionadas presentes na frase indicada.
  ```

- Base prompt template:

  ```text
  Frase: {text}
  Entidades mencionadas: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Frase: {text}

  Identifica as entidades mencionadas na frase. Deves devolver um dicionário JSON com as chaves 'pessoa', 'organização', 'local' e 'diverso' . Os valores devem ser listas contendo as entidades mencionadas desse tipo, tal como ocorrem na frase.
  ```

- Label mapping:
  - `B-PER` ➡️ `pessoa`
  - `I-PER` ➡️ `pessoa`
  - `B-LOC` ➡️ `local`
  - `I-LOC` ➡️ `local`
  - `B-ORG` ➡️ `organização`
  - `I-ORG` ➡️ `organização`
  - `B-MISC` ➡️ `diverso`
  - `I-MISC` ➡️ `diverso`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset harem
```

## Linguistic Acceptability

### ScaLA-pt

This dataset is a Portuguese version of ScaLA, which was originally published in [this
paper](https://aclanthology.org/2023.nodalida-1.20/), created by corrupting
grammatically correct sentences from the [Universal Dependencies Portuguese-Bosque
treebank](https://github.com/UniversalDependencies/UD_Portuguese-Bosque), filtered to
only include samples from the European Portuguese source *CETEMPúblico*. The treebank is
based on the Constraint Grammar conversion of the Bosque corpus, part of the Floresta
Sintá(c)tica treebank.

Corruptions were applied by either **removing a word** from the sentence or **swapping
two neighbouring words**. Rules based on part-of-speech tags were used to ensure that
these corruptions lead to grammatical errors.

The final dataset contains:

- **Training set**: 1,024 examples
- **Validation set**: 256 examples
- **Test set**: 2,048 examples

These splits are used as-is in the framework.

Here are a few examples from the training split:

```json
{
    "text": "Nos Em os mercados orientais, Tóquio foi a excepção e, ao o meio da de a manhã, a bolsa tendia para uma alta marginal, com o índice Nikkei a marcar 12,07 pontos no em o fim da de a sessão da de a manhã.",
    "label": "incorrect"
}
```

```json
{
    "text": "A equipa está a mostrar progressos, mas ainda há muito para fazer.",
    "label": "correct"
}
```

```json
{
    "text": "Vários estudos têm mostrado que estes linfomas regridem depois de tratamentos dirigidos à a HP a, o que sugere uma relação entre os dois.",
    "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Seguem-se abaixo textos e se são gramaticalmente corretos.
  ```

- Base prompt template:

  ```text
    Texto: {text}
    Gramaticalmente correcto: {label}
  ```

- Instruction-tuned prompt template:

  ```text
    Texto: {text}

    Determina se o texto é gramaticalmente correcto ou não. Responde com 'sim' ou 'não', e nada mais.
  ```

- Label mapping:
  - `correct` ➡️ `sim`
  - `incorrect` ➡️ `não`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-pt
```

## Reading Comprehension

### MultiWikiQA-pt

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

As Portuguese Wikipedia is a mixture of both European Portuguese and Brazilian
Portuguese, we filtered the Wikipedia articles with [this
classifier](https://hf.co/liaad/PtVId), published in [this
paper](https://doi.org/10.1609/aaai.v39i24.34705), keeping only the articles in European
Portuguese.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
    "context": "Manuel Frederico Tojal de Valsassina Heitor (Lisboa, 21 de setembro de 1958) é um professor universitário e político português, que foi ministro da Ciência, Tecnologia e Ensino Superior do XXI e XXII Governos Constitucionais.\n\nBiografia \nFilho de Frederico Lúcio de Valsassina Heitor (17 de Julho de 1930 - 2010), Comendador da Ordem da Instrução Pública a 9 de Junho de 1995, trineto por via feminina dum Barão de Valsassina na Áustria, Diretor do Colégio Valsassina, Membro-Honorário da Ordem da Instrução Pública a 9 de Novembro de 1985, e Comendador da Ordem da Instrução Pública, e de sua mulher Maria Manuela de Oliveira Tojal (1933 - Lisboa, 25 de Março de 2017), irmão do arquiteto Frederico Valsassina e neto materno do também arquitecto Raul Tojal, Manuel Heitor frequentou o Colégio Valsassina.\n\nFormação académica\nManuel Heitor licenciou-se em Engenharia Mecânica pelo Instituto Superior Técnico da Universidade Técnica de Lisboa em 1981, doutorou-se, em 1985, na mesma área, no domínio da Combustão Experimental, pelo Imperial College de Londres e obteve o título de agregado pela Universidade Técnica de Lisboa em 1992.\nRealizou um pós-doutoramento na Universidade da Califórnia, em San Diego.\n\nEnsino e investigação\nÉ professor catedrático do Instituto Superior Técnico, instituição onde tem desenvolvido a sua carreira académica, inicialmente na área de Mecânica de Fluidos e Combustão Experimental, e, mais recentemente, coordenando os programas de doutoramento daquele Instituto nas áreas da «Engenharia e Políticas Públicas» e da «Engenharia de Concepção e Sistemas Avançados de Manufactura».\n\nDesempenhou as funções de Presidente Adjunto do Instituto Superior Técnico entre 1993 e 1998.\n\nDesde o início dos anos 90 do século XX tem-se dedicado ao estudo de políticas de ciência, tecnologia e inovação, incluindo as políticas e gestão do ensino superior.\n\nDirige o «Centro de Estudos em Inovação, Tecnologia e Politicas de Desenvolvimento, IN+», do Instituto Superior Técnico, cuja fundação promoveu em 1998.\nEm 2005, este Centro foi nomeado como um dos Top 50 global centres of research on Management of Technology, pela International Association for the Management of Technology.\n\nFoi Professor Visitante na Universidade Harvard no ano letivo de 2011-2012.\n\nÉ Research Fellow da Universidade do Texas em Austin, no Instituto IC2, Innovation, Creativity and Capital.\n\nEm julho 2015 promoveu em Portugal o Manifesto «O Conhecimento como Futuro» e, mais recentemente, a declaração internacional «Knowledge as Our Common Future».\n\nAtividade política\n\nFoi Secretário de Estado da Ciência, Tecnologia e Ensino Superior, dos XVII e XVIII Governos, entre março de 2005 e junho de 2011.\n\nNestas funções participou ativamente na modernização do sistema de ensino português e no aumento do financiamento público e privado para atividades de ciência e tecnologia.\n\nNesta funções desenvolveu igualmente a conceção e concretização de consórcios internacionais em investigação e formação avançada entre universidades portuguesas e norte americanas, envolvendo redes temáticas de ciência e tecnologia.\n\nÉ ministro da Ciência, Tecnologia e Ensino Superior desde 2015.\n\nEm 2021 Manuel Heitor anunciou a criação de mais três escolas de Medicina em Évora, Aveiro e Vila Real.\n\nPortugueses de ascendência italiana\nPortugueses de ascendência austríaca\nNaturais de Lisboa\nAlunos do Instituto Superior Técnico\nAlunos da Universidade da Califórnia\nEngenheiros mecânicos de Portugal\nProfessores universitários de Portugal\nPolíticos do Partido Socialista (Portugal)\nSecretários de Estado de Portugal\nMinistros da Ciência de Portugal\nMinistros de Portugal\nPolíticos de Portugal\nGoverno de Portugal",
    "question": "Quando Manuel Heitor divulgou os planos para estabelecer três novas faculdades de medicina em Portugal?",
    "answers": {
        "answer_start": array([3176]),
        "text": array(["2021"], dtype=object)
    }
}
```

```json
{
    "context": "Multibanco é uma rede portuguesa de caixas automáticos (ATM) e de terminais de pagamento automático (POS) pertencente à SIBS, que tem como acionistas praticamente a totalidade das instituições bancárias portuguesas. Apesar do nome multibanco ser uma marca registada, propriedade da empresa SIBS, o termo é frequentemente empregue para designar de forma genérica um sistema interbancário que disponibilize serviços como o levantamento de dinheiro num dispositivo automático ou o pagamento de compras em lojas físicas.\n\nAtualmente, a utilização da rede Multibanco não se encontra limitada à utilização de um cartão bancário sendo possível usufruir de alguns dos serviços Multibanco através da aplicação MB Way, ao possibilitar o levantamento de numerário em qualquer caixa automático Multibanco ou pagamentos de compras nos terminais de pagamento automático da rede Multibanco através da leitura de um código QR, por aproximação do telemóvel ou usando o número de telemóvel.\n\nHistória \n\nO funcionamento do Multibanco teve início em setembro de 1985, com a instalação de 12 caixas automáticos (ATM) nas duas principais cidades do país (Lisboa e Porto). Enquanto Portugal foi um dos últimos países da Europa ocidental a instalá-las, o equipamento usado representou o que havia de mais avançado, baseado nas experiências de outros países, muitos dos quais gastam agora imenso dinheiro para substituir e atualizar máquinas obsoletas. Segundo um estudo britânico, o Multibanco seria o mais funcional de toda a Europa (com 60 funcionalidades), permitindo fazer operações que outros sistemas europeus não conseguem (por exemplo, o da Noruega não permite mais do que levantar dinheiro, saber os saldos e carregar o telemóvel). Em Portugal, os multibancos têm tido muito sucesso, o que levou ao aparecimento de novos serviços não bancários, como a venda de bilhetes ou o pagamento de serviços (água, eletricidade, gás, telefone, Internet, carregamento de telemóvel, Via Verde, etc.)\n\nEm 1987, foram introduzidos os terminais de pagamento automático (POS) Multibanco que permitiam pagar em lojas físicas com a utilização de cartões bancários, mesmo com cartões não exclusivos da rede Multibanco. Em 2008, estes sistemas passaram a permitir pagar faturas, carregar o telemóvel, consultar o saldo e movimentar contas, sendo neste caso, ao contrário do que acontece com os caixas automáticos Multibanco, as operações feitas pelos comerciantes.\n\nUtilização \n\nEm 2014, haviam cerca de 270 mil terminais de pagamento automático Multibanco. Em 2018, existiam cerca de 12 mil caixas multibanco de norte a sul do país, incluindo as regiões autónomas dos Açores e da Madeira. Diariamente, são levantados das máquinas de Multibanco cerca de 71 milhões de euros. A SIBS gere cerca de três mil milhões de operações financeiras por ano com um valor superior a 4,5 mil milhões de euros e conta com mais de 300 milhões de utilizadores, nacionais e estrangeiros.\n\nCom a exceção de 2019, o número de terminais no país tem vindo a diminuir ano após ano. Esta redução surge em paralelo com a redução acelerada da utilização dos terminais em favor do uso de aplicações móveis e web-sites.\n\nVer também\n Rede interbancária\n Caixa automático\n Plus\n Cirrus\n\nLigações externas \n\n SIBS - instituição de pagamento gestora dos sistemas Multibanco em Portugal\n\nRedes interbancárias\nCaixas eletrônicos\nSistema bancário\nInvenções e descobertas portuguesas",
    "question": "Quando é que os terminais de pagamento automático Multibanco começaram a ser usados?",
    "answers": {
        "answer_start": array([1976]),
        "text": array(["1987"], dtype=object)
    }
}
```

```json
{
    "context": "O furacão do Dia do Trabalho de 1935 foi o ciclone tropical mais forte da temporada de furacões no oceano Atlântico de 1935. Tem sido um dos mais intensos dos que têm tocado terra nos Estados Unidos e o primeiro dos três furacões de categoria 5 que têm açoitado este país durante o século XX, sendo os outros o Furacão Camille em 1969 e o Furacão Andrew em 1992. Depois de ter-se gerado como uma débil tempestade tropical ao leste das Bahamas a 29 de agosto de 1935, avançou lentamente para o oeste, se convertendo em furacão a 1 de setembro, intensificando rapidamente a sua potência antes de golpear a parte norte das Florida Keys a 2 de setembro. Após tocar terra em seu pico de intensidade, seguiu ao noroeste ao longo da costa oeste da Flórida, e debilitado anteriormente a terra para perto de Cedar Keys a 4 de setembro.\n\nO furacão causou graves danos na zona norte das Florida Keys, vendo-se toda a região afectada por uma forte marejada, com ondas dentre 4 e 9 metros aproximadamente. Por causa dos fortes ventos a maioria dos edifícios na zona de Islamorada ficaram destruídos. As linhas ferroviárias da Key West Flórida viram-se gravemente danificadas ou destruídas. O furacão também causou danos a seu passo pelo noroeste da Flórida, Geórgia e as Carolinas. Calcula-se que ao todo morreram mais de 400 pessoas. Este furacão iguala o recorde com o Furacão Dorian por ter sido o furacão mais potente que tenha golpeado os Estados Unidos quanto a pressão barométrica.\n\n1935 nos Estados Unidos",
    "question": "Qual o furacão mais intenso que ocorreu na época dos furacões no Atlântico em 1935?",
    "answers": {
        "answer_start": array([0]),
        "text': array(["O furacão do Dia do Trabalho de 1935"], dtype=object)
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Os textos que se seguem são acompanhados de perguntas e respostas.
  ```

- Base prompt template:

  ```text
  Texto: {text}
  Pergunta: {question}
  Resposta com um máximo de 3 palavras: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Texto: {text}

  Responde à seguinte pergunta sobre o texto acima num máximo de 3 palavras.

  Pergunta: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-pt
```

### Unofficial: BoolQ-PT

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2404.05333)
and is part of the ExtraGLUE dataset. It is created by taking the original BoolQ dataset
and using machine translation (DeepL) to translate it.

The original dataset has a passage, question, and yes/no label. We adapt this dataset by
taking the original passage, question, and yes/no options, and turning it into a Q/A
style question where the model can answer yes or no.

The original dataset contains 9,430 training, 3,270 validation, and 3,250 test samples.
We use 1,024 / 256 / 2,048 samples for train / val / test respectively. We've observed
some overlap in the splits, so decided to concatenate all splits into a single dataset,
shuffling it, and extract splits.

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

  ```text
  As seguintes são perguntas de escolha múltipla (com respostas).
  ```

- Base prompt template:

  ```text
  Pergunta: {text}
  Opções:
  a. {option_a}
  b. {option_b}
  Resposta: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Pergunta: {text}
  Opções:
  a. {option_a}
  b. {option_b}

  Responde à pergunta acima usando só 'a' ou 'b', e nada mais.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset boolq-pt
```

## Knowledge

### MMLU-pt

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2410.08928)
and is a machine translated version of the English [MMLU
dataset](https://openreview.net/forum?id=d7KBjmI3GmQ) and features questions within 57
different topics, such as elementary mathematics, US history and law. The translation to
Portuguese was done using DeepL.

The original full dataset consists of 270 / 1,439 / 14,774 samples for training,
validation, and testing, respectively. These splits were merged, duplicates removed, and
new splits were created with 1,024 / 256 / 2048 samples for training, validation, and
testing, respectively.

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

  ```text
  As seguintes são perguntas de escolha múltipla (com respostas).
  ```

- Base prompt template:

  ```text
  Pergunta: {text}
  Opções:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Resposta: {label}
  ```

- Instruction-tuned prompt template:

  ```text
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
euroeval --model <model-id> --dataset mmlu-pt
```

## Common-sense Reasoning

### GoldenSwag-pt

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

  ```text
  As seguintes são perguntas de escolha múltipla (com respostas).
  ```

- Base prompt template:

  ```text
  Pergunta: {text}
  Opções:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Resposta: {label}
  ```

- Instruction-tuned prompt template:

  ```text
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
euroeval --model <model-id> --dataset goldenswag-pt
```

### Unofficial: Winogrande-pt

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2506.19468)
and is a translated and filtered version of the English [Winogrande
dataset](https://doi.org/10.1145/3474381).

The original full dataset consists of 47 / 1,210 samples for training and testing, and
we use 128 of the test samples for validation, resulting in a 47 / 128 / 1,085 split for
training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Megan tem muito mais dinheiro do que Jessica porque _ acabou de comprar o bilhete de loteria vencedor. A que se refere o espaço em branco _?\nOpções:\na. Megan\nb. Jessica",
  "label": "a"
}
```

```json
{
  "text": "Elena pegaria o inventário na parte de trás da loja para Megan vender cada vez porque _ era uma empresária. A que se refere o espaço em branco _?\nOpções:\na. Elena\nb. Megan",
  "label": "a"
}
```

```json
{
  "text": "Joseph tinha que ter unhas bem cuidadas para o trabalho, mas não Kevin, porque _ trabalhava em uma fazenda. A que se refere o espaço em branco _?\nOpções:\na. Joseph\nb. Kevin",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Følgende er multiple choice spørgsmål (med svar).
  ```

- Base prompt template:

  ```text
  Spørgsmål: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}
  Svar: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Spørgsmål: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}

  Besvar ovenstående spørgsmål ved at svare med 'a' eller 'b', og intet andet.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset winogrande-pt
```

## Summarisation

### Publico

This dataset contains 3,304 news articles from the Portuguese newspaper *Público* paired
with extractive-style summaries. The samples all come from the [CCNews
corpus](https://commoncrawl.org/blog/news-dataset-available).

To create summary–document pairs, we extract the **first two sentences** of each article
as the `target_text` (summary), and concatenate the **title and the remainder** of the
article into `text`. This heuristic is grounded in the journalistic convention of
placing concise leads at the beginning of articles.

We provide 3 splits:

- Train: 1,024 examples
- Validation: 256 examples
- Test: 2,024 examples

Here are a few examples from the training split:

```json
{
    "text": "As grandes transições, o risco de disrupção\nPor que razão se acumulam tantos riscos elevados e com tal perigosidade? A razão principal, quero crer, reside no afloramento dos impactos das grandes transições - climática, ecológica, energética, demográfica, digital, migratória, laboral, sociocultural e sociopolítica - e numa inusitada convergência de todos os seus efeitos, internos e externos, nas décadas mais próximas e, bem assim, na impotência da política, tal como a conhecemos, para lidar com tantos eventos de tal amplitude. Senão vejamos. A vertigem digital e as suas inúmeras provações Eis o vórtice em que estamos metidos: chips e sensores, drones e câmaras de vigilância, interfaces cérebro-computacionais e nano-implantes, máquinas inteligentes e mestres algoritmos, robots e veículos autónomos, torres e antenas. Neste ambiente congestionado e num campo eletromagnético 4G+5G cada vez mais preenchido, seria impossível não acontecerem interações fortuitas, incidentes imprevistos, impactos inusitados, descobertas acidentais. Estamos, assim, obrigados a multiplicar os ângulos de observação e as perspetivas de olhar para os problemas. A surpresa pode ser, deveras, surpreendente. Basta, apenas, que aconteçam alguns acidentes graves cuja responsabilidade seja atribuída, “afinal”, à utilização abusiva de sistemas de inteligência automáticos e veículos autónomos. Estou convencido de que neste novo ambiente de virtualidade real a descontextualização que a inteligência artificial e automática carrega consigo nos fará passar inúmeras provações. Velocidade e colisão Com a chegada das redes 4G e 5G chegam as tecnologias mais disruptivas, mas chega, também, o risco de mais imersão, invasão e intrusão, ou seja, o risco iminente de uma grande colisão. Dito de outro modo, com a chegada das redes distribuídas as tecnologias imersivas, intrusivas e invasivas irão colidir, tarde ou cedo, com os seus destinatários potenciais. O que importa sublinhar nesta altura, no preciso momento em que a alta velocidade da rede 5G está para chegar, é o risco muito elevado de uma “grande colisão por excesso de velocidade”. De facto, a pandemia da covid-19 mostra-nos que está iminente uma grande colisão entre o infinitamente grande dos macro-organismos, os seres humanos que nós somos, e o infinitamente pequeno dos microorganismos, como é o caso da covid-19. Efeitos assimétricos Os efeitos assimétricos destas grandes transições vão deixar muitos territórios para trás. Cada transição tem o seu ciclo de vida específico, com uma duração variável, e é completamente impossível abordar todas as suas consequências no âmbito limitado de uma escala de tempo ou geografia em concreto. Ou seja, cada território nacional ou regional acabará por sofrer, tarde ou cedo, os danos colaterais de medidas erradas tomadas pelos territórios seus vizinhos. É nesta altura, justamente, que organizações supranacionais como a União Europeia ou subnacionais como as comunidades intermunicipais poderão e deverão mostrar toda a sua relevância geoeconómica e geopolítica. As interações fortuitas e os imponderáveis do acaso As características principais da rede 5G são a hipervelocidade, baixa latência, alta conectividade, elevada densidade e intensidade, curto alcance. Se pensarmos, agora, no polígono digital que esta rede nos oferece – Big data e computação na nuvem, (BDCC), Internet dos objetos (IOT), Inteligência artificial (IA), Realidade aumentada e virtual (RAV), Computação periférica (EC) – e na interação intensa entre estes e outros dispositivos tecnológicos e digitais, estamos cada vez mais próximos das chamadas “propriedades emergentes” do “serendipismo” (do inglês serenpidity), a saber, interações fortuitas, incidentes imprevistos, impactos inusitados, descobertas acidentais. Ou seja, perante a interdependência máxima crescem extraordinariamente os imponderáveis do acaso. Dispositivos tecnológicos e assistentes inteligentes Na sociedade da informação e da comunicação a inteligência deixou de estar contida nos limites humanos originais. Com efeito, nos dias que correm, a inteligência está dispersa e difusa, manifesta-se sob múltiplas formas e interage com praticamente tudo o que nos envolve. Deste ponto de vista, a realidade não para de aumentar todos os dias à medida que a inteligência se transfere para ambientes inteligentes que são extensões da nossa própria inteligência. Hoje tudo é smart, desde a realidade virtual e aumentada aos interfaces cérebro-computacionais, desde a inteligência dos objetos até à inteligência das máquinas. De facto, a nossa inteligência e as faculdades humanas estão a transitar para fora do seu habitat biológico e o corpo humano instala-se em dispositivos tecnológicos transumanos e pós-humanos cuja configuração futura nem sequer imaginamos. Entre a distração e a alucinação Somos screeners muitas horas por dia, é impossível manter a atenção num ambiente completamente saturado de notificações e avisos. A multiplicação dos dispositivos tecnológicos e digitais – uma espécie de sexto continente - exige de nós uma atualização constante. Todos os dias mergulhamos num imenso oceano de informação, experimentamos uma vertigem permanente para separar o essencial do acessório e lutamos com imensas dificuldades para administrar a nossa economia da atenção. No final do dia estamos exaustos e no dia seguinte, ainda debilitados, tudo recomeça. Na vertigem o foco da atenção converte-se num turbilhão, talvez, mesmo, em delírio e alucinação. As mudanças paradigmáticas Entre tantas transições previsíveis e excecionais haverá, também, mudanças paradigmáticas, cujos sinais de longo alcance só alguns vislumbrarão. O drama das mudanças paradigmáticas é que elas não se compadecem com a duração dos ciclos políticos curtos e muito menos com programas de governo reativos. A redução dos passivos climáticos, tais como o sequestro e armazenamento de carbono, ou a mudança de alguns aspetos nucleares do modelo de desenvolvimento dominante, por exemplo, a revisão de algumas cadeias de valor no sentido da sua reterritorialização, ou, ainda, a mudança de aspetos fundamentais do nosso comportamento quotidiano, por exemplo, no que diz respeito ao cumprimento de regras base de economia circular. Quer dizer, temos de estar avisados, não podemos permitir que os efeitos contraproducentes ou paradoxais das várias transições acabem por absorver os pequenos/grandes sinais das mudanças paradigmáticas. Notas Finais Como se observa, o risco de disrupção está sempre presente, seja o carácter invasivo e intrusivo das tecnologias 5G, a histeria coletiva de informação e comunicação num ambiente totalmente saturado, a crença nos mestres-algoritmos e na metalinguagem normalizadora das plataformas digitais. Digamos que, doravante, crescerá bastante o risco sistémico da economia digital 4G e 5G e, nesse sentido, estamos obrigados a desenvolver treino específico e capacidades especiais para entender e antecipar como se forjam e desenvolvem as interações fortuitas, os incidentes imprevistos e, por via deles, também, as descobertas acidentais. Este é o grande paradoxo do nosso tempo. Mais liberdade, mais incerteza, mais episódios acidentais. Por outro lado, os sinais dessas interações acidentais podem ser de tal modo fortuitos e furtivos que dificilmente caberão no interior das nossas métricas conceptuais e instrumentais habituais. O nosso arsenal teórico e, muito em especial, o campo das ciências sociais e humanas, com origem no iluminismo moderno e na cultura analógica, estão definitivamente postos em causa e a academia deve preparar-se para rever o seu estatuto científico eminente se não quiser ser um ator secundário que corre pelo lado de fora da realidade da cultura tecnológica e digital.",
    "target_text": "O nosso tempo não corre de feição. Desastres ambientais motivados por alterações climáticas, campos de refugiados em número crescente, pandemia da covid-19, elevado número de abalos de terra e erupções vulcânicas, adição digital e ódio nas redes sociais, polarização social e radicalização política, crise da transição energética, precariedade nos mercados de trabalho e baixos salários, dívidas públicas acumuladas gigantescas, crescente tensão geopolítica entre grandes potências."
}
```

```json
{
    "text": "Sloane Stephens bateu todas as probabilidades\nFiel às indicações do treinador – “Respira, bate na bola, mexe os pés” –, Stephens soube controlar melhor as emoções, embora, na véspera não soubesse o que fazer para lidar com o nervosismo. “Estive a ler revistas de carros, críticas sobre a segurança… é um pouco estranho mas foi o que fiz. Estava muito nervosa, mas sabia que ela, provavelmente, sentia o mesmo”, contou a norte-americana de 24 anos. Não foi preciso muito tempo para se saber quem estava mais à vontade no Arthur Ashe Stadium: três erros directos de Keys conduziram ao primeiro break e deram uma vantagem de 3-2 à compatriota. A ansiedade da jogadora de 22 anos foi aumentando, o que não ajudou a que reencontrasse o seu ténis poderoso. E Keys terminou com somente metade dos pontos disputados com o seu primeiro serviço e sem concretizar nenhum dos três break-points – todos no segundo jogo do segundo set. Mais conservadora no seu estilo de jogo, Stephens não precisou muito mais do que manter a bola em campo para manter o ascendente no encontro. Mas também serviu bem, contra-atacou e defendeu-se muito bem nas esporádicas tentativas de reacção de Keys, e fechou o encontro ao fim dos 61 minutos. “Fiz seis erros em todo o encontro? Inacreditável! Acho que isso nunca me aconteceu antes”, confessou Stephens, já na conferência de imprensa, após a vitória por 6-3, 6-0. Antes, também tinha ficado boquiaberta quando Keys cometeu o último erro, no terceiro match-point. “’Ganhei mesmo o Open dos EUA’. Fiquei assim um bocadinho… Uau!”, admitiu. E foi de estupefacção o seu ar quando recebeu o cheque de três milhões de euros. Pelo meio, abraçou longamente na rede a amiga Maddy, que não conseguiu conter as lágrimas. E depois de subir às bancadas para abraçar treinador, família e o namorado (o futebolista Jozy Altidore), foi sentar-se ao lado dela, fazendo-a sorrir. “Sendo a amiga que é, Sloane apoiou-me muito”, contou Keys, que também reconheceu os nervos. “Estive nervosa toda a manhã, obviamente. Sloane é uma adversária difícil de defrontar, especialmente quando não metemos muitas bolas e ela também não falha. Não sabia o que fazer quando estava no court, o que intensificou ainda mais o nervosismo”, admitiu Keys. Por causa das paragens forçadas, nenhuma delas vai surgir no "top-10" do ranking desta segunda-feira. Stephens, que regressou à competição em Julho, após uma paragem de 11 meses e uma operação ao pé direito, vai surgir no 17.º lugar. Já Keys, operada por duas vezes ao pulso esquerdo, a segunda em Junho, vai subir ao 12.º lugar. Mas com o regresso em pleno das veteranas Serena Williams, Victoria Azarenka e Maria Sharapova, o confronto com a nova geração, em que se incluem as duas norte-americanas, mas também as campeãs Jelena Ostapenko (Roland Garros) e Garbiñe Muguruza (Wimbledon), vai elevar o interesse sobre o circuito feminino em 2018. Quatro anos e meio depois de derrotar Serena Williams, ser apontada como sua sucessora e chegar às meias-finais do Open da Austrália, Stephens está orgulhosa por ter confirmado as expectativas. “Um dia, vou poder mostrar aos meus filhos que venci o Open dos EUA. Quantas pessoas podem dizer isto? Até já gravaram o meu nome no vestuário. Isto é espantoso”, disse Stephens, ainda incrédula.",
    "target_text": "Se já era altamente improvável que duas jogadoras vindas de recentes intervenções cirúrgicas pudessem, poucos meses depois, estar numa final de um torneio do Grand Slam, as hipóteses de Sloane Stephens vencer o Open dos EUA eram mais reduzidas depois da sua amiga Madison Keys ter realizado uma exibição de sonho nas meias-finais. Mas, no derradeiro encontro entre duas estreantes em finais de majors, o maior nervosismo de Keys impediu-a de produzir o ténis que a levou a eliminar Venus Williams e, com 30 erros não forçados, contribuiu com metade dos pontos ganhos por Stephens e suficientes para erguer o seu primeiro troféu do Grand Slam."
}
```

```json
{
    "text": "Praia algarvia entre as seis melhores do mundo, destaca TripAdvisor\nDesta vez, deixa a segunda metade da tabela para firmar-se entre os dez melhores areais do planeta, subindo seis lugares em relação a 2021. “É uma praia deslumbrante... o sol bate nos diferentes tons de areia laranja e amarela das falésias altas que reflectem uma cor quente”, lê-se no comentário de um utilizador, destacado pela TripAdvisor em comunicado. “A própria areia da praia é um amarelo dourado de grão fino. As ondas quebram na praia com uma ferocidade gentil que cria um surf branco para nadadores e surfistas.” Para criar a lista, renovada anualmente, a TripAdvisor revê “dezenas de milhões de avaliações enviadas por milhões de viajantes globais nos últimos 12 meses”, analisando “a qualidade e a quantidade das avaliações” para “determinar as praias favoritas dos viajantes” no ano anterior, antecipando tendências para os próximos meses. Este ano, as escapadelas para praias insulares surgem particularmente “populares”, “com quase três quartos das dez melhores do mundo a situarem-se em locais remotos”, destaca a empresa em comunicado. É o caso da praia vencedora de 2023, a brasileira Baía do Sancho, localizada na ilha Fernando de Noronha. É um regresso ao topo da lista, subindo seis posições relativamente ao ano passado. Outro destaque é uma “nova e empolgante entrada”: a “dramática” praia de Reynisfjara, em Vik, na Islândia. “É uma praia como nenhuma outra”, assegura a nota de imprensa. “Com as suas mundialmente famosas areias negras e imponentes formações rochosas que se elevam sobre a costa, alguns podem reconhecer o impressionante cenário de A Guerra dos Tronos.” Apesar de “popular entre os observadores de pássaros devido aos vários tipos de aves marinhas avistadas nas proximidades, principalmente os papagaios-do-mar”, as “águas geladas” e as ondas, que podem atingir os 40 metros de altitude, não convidam a banhos. “É uma praia mais bem admirada da segurança do litoral.” “Além das adoradas praias do Havai, das Caraíbas e da Europa continental, a nossa comunidade está mesmo à procura de melhorar as suas experiências ao abraçar as falésias de Cannon Beach, na costa de Oregon, no Oeste dos Estados Unidos, e destinos mais frios, como a praia de Reynisfjara, na Islândia”, nota Sarah Firshein, chefe editorial da TripAdvisor, em comunicado.",
    "target_text": "As “exuberantes falésias de areia vermelha” que emolduram “uma praia de areia branca que parece estender-se infinitamente”, terminando num “oceano azul-esverdeado”, valeram à praia da Falésia, situada em Olhos de Água, no concelho de Albufeira, o sexto lugar do ranking das melhores praias do mundo, eleito anualmente pelos Traveler's Choice Awards, da TripAdvisor. Há anos que o areal algarvio surge entre as preferências dos utilizadores da plataforma internacional, mantendo-se a única praia portuguesa no top mundial.",
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:

  ```text
  Abaixo encontras documentos com resumos associados.
  ```

- Base prompt template:

  ```text
  Documento: {text}
  Resumo: {target_text}
  ```

- Instruction-tuned prompt template:

  ```text
  Documento: {text}

  Escreve um resumo do documento anterior.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset publico
```
