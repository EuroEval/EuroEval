# 🇫🇷 French

This is an overview of all the datasets used in the French part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.


## Sentiment Classification

### AlloCiné

This dataset was published in [this Github
repository](https://github.com/TheophileBlard/french-sentiment-analysis-with-bert) and
features reviews from the French movie review website [AlloCiné](https://www.allocine.fr/). The reviews range from
0.5 to 5 (inclusive), with steps of 0.5. The negative samples are reviews with a rating
of at most 2, and the positive ones are reviews with a rating of at least 4. The reviews
in between were discarded.

The original full dataset consists of 160,000 / 20,000 / 20,000 samples for training,
validation, and testing, respectively. We use 1,024 / 256 / 2,048 samples for training,
validation, and testing, respectively. All our splits are subsets of the original ones.

Here are a few examples from the training split:

```json
{
  "text": "Ce 7ème volet ne mérite pas de notre part une grande attention, au vu du précédent New Police Story. À la limite du huis clos, Jackie évolue dans une boîte de nuit, sorte de piège du méchant cherchant à se venger, ou du moins à découvrir la vérité sur la mort de sa sœur. Notre cascadeur acteur ne bénéficie pas d'un décors à la hauteur de son potentiel acrobatique et le film d'un scénario à la hauteur d'une production, et cette production d'une large distribution, ce qui explique son arrivée direct tout étagère.",
  "label": "negative"
}
```
```json
{
  "text": "Meme pour ceux qui n'aime pas les Chevaliers du Fiel allez voir. 1 il est meilleur que le 1 et cela est rare de voir une suite qui est meilleur que le 1. Des scènes qui peuvent faire rire les petit et les grands. On ne s'ennuie pas. Super film allez le voir. L'interpretation des acteurs sont super. Bonne journée",
  "label": "positive"
}
```
```json
{
  "text": "Une ambiance envoûtante, un récit où se mélangent sorcellerie, croyances indiennes, enquête policière sur fond de trafic de drogue, tout est conforme au livre de Tony Hillerman, même si ce dernier a \"renié\" le film. Personnellement j'adore. Hélas introuvable en France et diffusé seulement sur canal , il y a ..... un certain temps.",
  "label": "positive"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:
  ```
  Voici des textes et leur sentiment, qui peut être 'positif' ou 'négatif'.
  ```
- Base prompt template:
  ```
  Texte: {text}
  Sentiment: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Texte: {text}

  Classez le sentiment dans le texte. Répondez par ‘positif' ou ‘négatif'.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset allocine
```


## Named Entity Recognition

### ELTeC

This dataset was published in [this paper](https://doi.org/10.3828/mlo.v0i0.364) and
consists of sentences from 100 novels in French during the period 1840-1920, all of
which are in the public domain. These novels were automatically labelled with named
entities using Stanza-NER, and then manually corrected.

The original dataset consists of 100 samples, one for each novel. We split the novels
into sentences using the French NLTK sentence splitter, resulting in 4,815 samples. We
use 1,024 / 256 / 2,048 samples for training, validation, and testing, respectively.

We have furthermore converted the OntoNotes 5.0 labelling scheme to the CoNLL-2003
labelling scheme, which is more common in the NER literature. The mapping is as follows:

- `PERS` ➡️ `PER`
- `LOC` ➡️ `LOC`
- `ORG` ➡️ `ORG`
- `OTHER` ➡️ `MISC`
- `DEMO` ➡️ `O`
- `ROLE` ➡️ `O`
- `EVENT` ➡️ `O`

Here are a few examples from the training split:

```json
{
  'tokens': array(['Jamais', 'ils', 'ne', 'firent', 'de', 'provisions', ',', 'excepté', 'quelques', 'bottes', "d'ail", 'ou', "d'oignons", 'qui', 'ne', 'craignaient', 'rien', 'et', 'ne', 'coûtaient', 'pas', "grand'chose", ';', 'le', 'peu', 'de', 'bois', "qu'ils", 'consommaient', 'en', 'hiver', ',', 'la', 'Sauviat', "l'achetait", 'aux', 'fagotteurs', 'qui', 'passaient', ',', 'et', 'au', 'jour', 'le', 'jour', '.'], dtype=object),
  'labels': array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
```
```json
{
  'tokens': array(['I', 'Il', 'y', 'avait', 'plus', 'de', 'soixante', 'ans', 'que', "l'empereur", 'Napoléon', ',', 'pressé', "d'argent", ',', 'avait', 'vendu', 'les', 'provinces', 'de', 'la', 'Louisiane', 'à', 'la', 'République', 'des', 'États-Unis', ';', 'mais', ',', 'en', 'dépit', 'de', "l'infiltration", 'yankee', ',', 'les', 'traditions', 'des', 'créoles', 'français', 'se', 'perpétuaient', '.'], dtype=object),
  'labels': array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
```
```json
{
  'tokens': array(['Les', 'fenêtres', 'de', 'la', 'vieille', 'demeure', 'royale', ',', 'ordinairement', 'si', 'sombres', ',', 'étaient', 'ardemment', 'éclairées', ';', 'les', 'places', 'et', 'les', 'rues', 'attenantes', ',', 'habituellement', 'si', 'solitaires', ',', 'dès', 'que', 'neuf', 'heures', 'sonnaient', 'à', "Saint-Germain-l'Auxerrois", ',', 'étaient', ',', "quoiqu'il", 'fût', 'minuit', ',', 'encombrées', 'de', 'populaire', '.'], dtype=object),
  'labels': array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:
  ```
  Vous trouverez ci-dessous des phrases et des dictionnaires JSON avec les entités nommées qui apparaissent dans la phrase donnée.
  ```
- Base prompt template:
  ```
  Sentence: {text}
  Entités nommées: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Sentence: {text}

  Identifiez les entités nommées dans la phrase. Vous devez produire ceci sous forme de dictionnaire JSON avec les clés 'personne', 'lieu', 'organisation' et 'divers'. Les valeurs doivent être des listes des entités nommées de ce type, exactement comme elles apparaissent dans la phrase.
  ```

- Label mapping:
    - `B-PER` ➡️ `personne`
    - `I-PER` ➡️ `personne`
    - `B-LOC` ➡️ `lieu`
    - `I-LOC` ➡️ `lieu`
    - `B-ORG` ➡️ `organisation`
    - `I-ORG` ➡️ `organisation`
    - `B-MISC` ➡️ `divers`
    - `I-MISC` ➡️ `divers`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset eltec
```


## Linguistic Acceptability

### ScaLA-fr

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [French Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_French-GSD/tree/master) by
assuming that the documents in the treebank are correct, and corrupting the samples to
create grammatically incorrect samples. The corruptions were done by either removing a
word from a sentence, or by swapping two neighbouring words in a sentence. To ensure
that this does indeed break the grammaticality of the sentence, a set of rules were used
on the part-of-speech tags of the words in the sentence.

The original dataset consists of 16,342 samples, from which we use 1,024 / 256 / 2,048 samples for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
used as-is in the framework.

Here are a few examples from the training split:

```json
{
  "text": "Le dessert est une part minuscule de gâteau.",
  "label": "correct"
}
```
```json
{
  "text": "Le trafic international sera normal vendredi sur Eurostar, Thalys, et sur les trains à grande vitesse à destination de l', a indiqué la SNCF dans un communiqué.",
  "label": "incorrect"
}
```
```json
{
  "text": "Certains craignent qu' un avantage compétitif trop net et trop durable favorise les positions dominantes, monopoles et oligopoles, qui limitent la et concurrence finissent par peser sur le consommateur.",
  "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Les phrases suivantes indiquent si elles sont grammaticalement correctes.
  ```
- Base prompt template:
  ```
  Phrase: {text}
  Correct du point de vue grammatical: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Phrase: {text}

  Déterminez si la phrase est grammaticalement correcte ou non. Répondez par 'oui' si la phrase est correcte et par 'non' si elle ne l'est pas, et rien d'autre.
  ```
- Label mapping:
    - `correct` ➡️ `oui`
    - `incorrect` ➡️ `non`

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset scala-fr
```


## Reading Comprehension

### FQuAD

This dataset was published in [this
paper](https://aclanthology.org/2020.findings-emnlp.107/), and is a manually annotated
dataset of questions and answers from the French Wikipedia.

The original full dataset consists of 20,731 / 3,188 / 2,189 samples for training,
validation and testing, respectively. Note that the testing split is not publicly
accessible, however, so we only use the training and validation split. We use 1,024 /
256 / 2,048 samples for training, validation, and testing, respectively. Our training
split is a subset of the original training split, and our validation and testing splits
are subsets of the original validation split.

Here are a few examples from the training split:

```json
{
  'context': "Parmi leurs thèmes récurrents, on en trouve qui sont communs à beaucoup d'autres groupes contemporains ou plus anciens : les Stranglers ont décrit, à plusieurs reprises, la vie d'un groupe de rock dans toutes ses dimensions (fans, autres groupes, vie en tournée). Le thème rebattu - chez les groupes des années 1960-1970 - de la drogue, est abordée sur une demi-douzaine de chansons (Don't Bring Harry), tandis que la vision angoissée du futur, dans le contexte de la guerre froide ou en lien avec les avancées de la science, a donné lieu à plusieurs titres (Curfew). On retrouve également chez eux des préoccupations écologiques (Dreamtime) ou sociales. La guerre, notamment les deux guerres mondiales (Northwinds), mais aussi les guerres contemporaines (I Don't Agree), sont à l'origine de divers textes. Mais le thème qui les a le plus inspirés, c'est de loin les femmes (The Man They Love to Hate).",
  'question': 'Sur combien de chanson le thème de la drogue est il abordé ?',
  'answers': {
    'answer_start': array([353]),
    'text': array(['une demi-douzaine'], dtype=object)
  }
}
```
```json
{
  'context': "Au cours de cette période, Cavour se distingue par son talent de financier. Il contribue de manière prépondérante à la fusion de la Banque de Gênes et de la nouvelle Banque de Turin au sein de la Banque Nationale des États sardes (Banca Nazionale degli Stati Sardi). Après le succès électoral de décembre 1849, Cavour devient également une des figures dominantes de la politique piémontaise et il prend la fonction de porte-parole de la majorité modérée qui vient de se créer. Fort de cette position, il fait valoir que le moment des réformes est arrivé, favorisé par le Statut albertin qui a créé de réelles perspectives de progrès. Le Piémont peut ainsi s'éloigner du front catholique et réactionnaire, qui triomphe dans le reste de l'Italie. ",
  'question': "En quel année sort-il vainqueur d'une élection ?",
  'answers': {
    'answer_start': array([305]),
    'text': array(['1849'], dtype=object)
  }
}
```
```json
{
  'context': "Pour autant, le phénomène météorologique se décline sous d'autres variantes : ocelles du paon, évoquant les cent yeux d'Argus, fleurs champêtres et ornant les jardins où s'établit l'osmose entre couleurs complémentaires. La poésie tient en main la palette du peintre,, celle de Claude Gellée ou de Poussin. Pour autant, il ne s'agit pas là d'une posture habituelle chez lui, qui privilégie les paysages quasi-monochromes.",
  'question': "Qu'est ce que l'auteur préfère décrire ?",
  'answers': {
    'answer_start': array([394]),
    'text': array(['paysages'], dtype=object)
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:
  ```
  Les textes suivants sont accompagnés de questions et de réponses.
  ```
- Base prompt template:
  ```
  Texte: {text}
  Question: {question}
  Réponse en 3 mots maximum: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Texte: {text}

  Répondez à la question suivante sur le texte ci-dessus en 3 mots maximum.

  Question: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset fquad
```


### Unofficial: BeleBele-fr

This dataset was published in [this paper](https://aclanthology.org/2024.acl-long.44/) and features multiple-choice reading comprehension questions across 122 languages.

The original dataset contains 900 unique multiple-choice reading comprehension passages and questions. From these, we use a 256 / 64 / 580 split for training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Texte: Lorsqu’un petit groupe d’êtres vivants (une petite population) est séparé de la population principale dont il est issu (par exemple, s’il se déplace au-dessus d’une chaîne de montagnes ou d’une rivière, ou s’il se déplace vers une nouvelle île de sorte qu’il ne peut pas facilement revenir en arrière), il se retrouve souvent dans un environnement différent de celui dans lequel il était auparavant. Ce nouvel environnement a des ressources et des concurrents différents, de sorte que la nouvelle population aura besoin de caractéristiques ou d'adaptations nouvelles pour être un concurrent puissant par rapport à ce dont elle avait besoin auparavant. La population d'origine n'a pas changé du tout,\xa0elle a toujours besoin des mêmes adaptations. Au fil du temps, à mesure que la nouvelle population s'adapte à son nouvel environnement, elle commence à ressembler de moins en moins à l'autre population. Enfin, après des milliers ou même des millions d'années, les deux populations paraîtront tellement différentes qu'elles ne pourront plus être considérées comme appartenant à la même espèce. Nous appelons ce processus «\u2009spéciation\u2009», ce qui signifie simplement la formation de nouvelles espèces. La spéciation est une conséquence inévitable et une partie très importante de l’évolution.\nQuestion: D’après l’extrait et parmi les exemples ci-dessous, qu’est-ce qui gênerait le processus d’évolution\xa0?\nChoix:\na. La difficulté pour un petit groupe à s’épanouir dans un nouvel endroit\nb. La migration d’une portion d’une population vers un nouvel environnement\nc. L’ajustement par une population de son adaptation à un nouvel environnement\nd. Le fait qu’une population finisse par devenir deux populations distinctes",
  "label": "a"
}
```
```json
{
  "text": "Texte: Le pillage généralisé se serait poursuivi pendant la nuit, les forces de l'ordre n'étant pas présentes dans les rues de Bichkek. Un observateur a décrit Bichkek comme étant en train de sombrer dans un état d’« anarchie », tandis que la population se déplaçait en bandes dans les rues et pillait les magasins de biens de consommation. Plusieurs habitants de Bichkek ont reproché les manifestants du sud d'être responsables de l'anarchie.\nQuestion: Qui a accusé les manifestants du sud de pillage\xa0?\nChoix:\na. Des habitants de Bichkek\nb. Les forces de l’ordre\nc. Les anarchistes\nd. Des bandes de personnes",
  "label": "a"
}
```
```json
{
  "text": "Texte: Dans de nombreuses régions du monde, faire un signe de la main est un geste amical signifiant «\u2009bonjour\u2009». En revanche, en Malaisie, du moins chez les Malais des zones rurales, cela signifie « viens par ici », comme le fait de plier l'index vers soi, geste utilisé dans certains pays occidentaux, et il ne devrait être utilisé qu'en ce sens. De même, un voyageur britannique en Espagne pourrait confondre un signe d'adieu fait par une personne qui tourne la paume de sa main vers elle-même (plutôt que vers la personne à qui elle adresse le signe) avec une invitation à revenir.\nQuestion: Dans les zones rurales de la Malaisie, quel geste signifie « viens par ici » ?\nChoix:\na. Plier l’index\nb. Faire un signe de la main\nc. Faire un « high five »\nd. Lever le pouce",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Les questions suivantes sont des questions à choix multiples (avec réponses).
  ```
- Base prompt template:
  ```
  Question: {text}
  Choix:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Réponse: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Question: {text}
  Choix:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Répondez à la question ci-dessus par 'a', 'b', 'c' ou 'd', et rien d'autre.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset belebele-fr
```


## Knowledge

### MMLU-fr

This dataset is a machine translated version of the English [MMLU
dataset](https://openreview.net/forum?id=d7KBjmI3GmQ) and features questions within 57
different topics, such as elementary mathematics, US history and law. The translation to
French was done by the University of Oregon as part of [this
paper](https://aclanthology.org/2023.emnlp-demo.28/), using GPT-3.5-turbo.

The original full dataset consists of 269 / 1,410 / 13,200 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 2,048 split for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
new and there can thus be some overlap between the original validation and test sets and
our validation and test sets.

Here are a few examples from the training split:

```json
{
  "text": "En 2013, la part des personnes en Ethiopie qui pensent que les partis politiques sont corrompus est\nChoix:\na. 24%\nb. 44%\nc. 64%\nd. 84%",
  "label": "a"
}
```
```json
{
  "text": "Combien de nombres entiers positifs et négatifs $12$ est-il un multiple?\nChoix:\na. 3\nb. 12\nc. 4\nd. 6",
  "label": "b"
}
```
```json
{
  "text": "Quelle affirmation suivante concernant les réactions dépendantes de la lumière de la photosynthèse est correcte?\nChoix:\na. Ils fournissent le carbone qui est incorporé dans le sucre.\nb. Ils produisent du PGA, qui est converti en glucose par la fixation du carbone dans les réactions indépendantes de la lumière.\nc. L'eau est séparée en fournissant des ions hydrogène et des électrons à la NADP pour un stockage temporaire.\nd. Ils se produisent dans le stroma des chloroplastes.",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Les questions suivantes sont des questions à choix multiples (avec réponses).
  ```
- Base prompt template:
  ```
  Question: {text}
  Choix:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Réponse: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Question: {text}
  Choix:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Répondez à la question ci-dessus par 'a', 'b', 'c' ou 'd', et rien d'autre.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset mmlu-fr
```


## Common-sense Reasoning

### HellaSwag-fr

This dataset is a machine translated version of the English [HellaSwag
dataset](https://aclanthology.org/P19-1472/). The original dataset was based on both
video descriptions from ActivityNet as well as how-to articles from WikiHow. The dataset
was translated by the University of Oregon as part of [this
paper](https://aclanthology.org/2023.emnlp-demo.28/), using GPT-3.5-turbo.

The original full dataset consists of 9,310 samples. We use a 1,024 / 256 / 2,048 split
for training, validation and testing, respectively (so 3,328 samples used in total).

Here are a few examples from the training split:

```json
{
  "text": "[header] Comment dire à vos enfants que vous allez divorcer [title] Contrôlez vos émotions. [step] Vos enfants seront probablement en colère et bouleversés lorsque vous leur annoncerez le divorce, essayez donc de ne pas réagir de la même manière. Attendez de rompre la nouvelle lorsque vous pourrez discuter du sujet de manière efficace et rester maître de vos émotions.\nChoix:\na. Rappelez-vous, le but de la discussion est d'être là pour les enfants - ils ne devraient pas avoir à vous réconforter. [title] Essayez de le faire ensemble, si possible.\nb. [substeps] Trouvez un moyen d'éviter que vos enfants ne vous agressent verbalement. Assurez-vous d'être calme et posé et ne donnez pas l'impression que la nouvelle du divorce est quelque chose qui vous dérange.\nc. [substeps] Si vos enfants ont du mal à comprendre la nouvelle à distance, posez-leur des questions lors d'une conversation intime et privée. Laissez-les utiliser les questions pour traiter et comprendre ce qu'ils ressentent à propos de l'annonce.\nd. [substeps] Si vous ne voulez pas qu'ils le sachent immédiatement, partez en silence et réfléchissez un peu plus longtemps avant de leur dire. Cherchez un endroit confortable pour vous deux pour parler en privé, afin que vous puissiez tous deux prendre du temps pour traiter vos sentiments et accepter la situation.",
  "label": "a"
}
```
```json
{
  "text": "Certains stands servent des hot-dogs aux gens alors qu'ils pêchent sur la glace. Un petit garçon et une petite fille tentent d'attraper un poisson. ils\nChoix:\na. attrapent un poisson et continuent de nager.\nb. sont interviewés pendant qu'ils pêchent.\nc. essaient à plusieurs reprises, errant tout près de leur poisson.\nd. sont rapidement emportés par le courant alors qu'ils luttent pour s'éloigner du banc de la rivière et pagayent pour échapper à de légères infestations de poissons dans l'eau",
  "label": "b"
}
```
```json
{
  "text": "[header] Comment se calmer [title] Respirer. [step] Respirer. Lentement.\nChoix:\na. Concentrez-vous sur votre respiration et détendez votre corps. Continuez à inspirer et expirer lentement par le nez, en mettant une pression sur votre diaphragme et vos muscles fessiers (vos poumons).\nb. Si votre cœur bat vite ou fort, vous pourriez être en danger de tachycardie, d'AVC ou de toute autre crise cardiaque. [title] Allongez-vous sur le dos et inspirez et expirez profondément.\nc. Inspirez pendant 5 secondes; retenez votre souffle pendant 5 secondes, puis expirez pendant 5 secondes. Cela fonctionne parce que vous faites l'opposé de ce qu'une personne excitée ferait.\nd. Inspirez pendant un compte de cinq et abaissez-vous. Expirez, expirez quatre fois de plus, aussi profondément que vous pouvez sentir, et répétez pour un total de dix.",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Les questions suivantes sont des questions à choix multiples (avec réponses).
  ```
- Base prompt template:
  ```
  Question: {text}
  Choix:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Réponse: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Question: {text}
  Choix:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Répondez à la question ci-dessus par 'a', 'b', 'c' ou 'd', et rien d'autre.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset hellaswag-fr
```


## Summarization

### Orange Sum

This dataset was published in [this
paper](https://aclanthology.org/2021.emnlp-main.740/) and consists of news articles from
[Orange Actu](https://actu.orange.fr/). The summaries were written by the journalists
themselves (the "abstract" field in the original dataset).

The original full dataset consists of 21,401 / 1,500 / 1,500 samples for training,
validation and testing, respectively. We use 1,024 / 256 / 1,024 samples for training,
validation, and testing, respectively. All our splits are subsets of the original ones.

Here are a few examples from the training split:

```json
{
  "text": "Réclamé puis annoncé par Emmanuel Macron, le débat parlementaire sur l'immigration s'est ouvert ce lundi 7 octobre avec une allocution d'Edouard Philippe devant les députés. Le Premier ministre a commencé son discours en empruntant les mots d'un de ses prédécesseurs, Michel Rocard. Il a ensuite fait état d'un système français d'asile \"saturé\". \"En 2018, la France a enregistré le record de 123.000 demandes d'asile\", a t-il rappelé, estimant que la France \"n'a pas atteint tous\" ses objectifs en matière de politique migratoire et de lutte contre l'immigration irrégulière. \"La question d'un pilotage par objectifs de l'admission au séjour n'est pas tabou. Je n'ai pas peur de réfléchir à l'idée de quotas. Il nous faut donc regarder sujet après sujet. On sait depuis longtemps que les quotas ne s'appliquent ni à l'asile ni à l'immigration familiale. Pour autant, celle-ci ne pourrait échapper à toute maîtrise. Il faut lutter contre les abus et les fraudes, et resserrer les critères là où cela s'impose\" a t-il poursuivi.Le Premier ministre a en revanche balayé l'idée de la fin du droit du sol, réclamée par des élus de droite. \"Je ne vois pas bien en quoi à l'échelle du pays, la fin du droit du sol serait une réponse\". Il a également adressé une critique virulente à l'égard de la théorie de \"l'immigration de remplacement\", un \"vocable d'une laideur certaine qui fait appel aux ressorts les plus détestables du complotisme.Ces théories \"inspiraient encore récemment des discours dont j'ai eu l'occasion de dire qu'ils étaient profondément contraires à l'idée dont nous nous faisons de la France et de la République\" a t-il encore asséné, en référence à la récente \"Convention de la droite\" organisée le 28 septembre dernier autour de Marion Maréchal et Eric Zemmour.",
  "target_text": "Le Premier ministre a ouvert ce lundi 7 octobre le débat sur l'immigration à l'Assemblée nationale, déclarant que le système français d'asile est aujourd'hui \"saturé\". Il a au passage pourfendu la théorie de \"l'immigration de remplacement\", qui fait selon lui appel \"aux ressorts les plus détestables du complotisme\"."
}
```
```json
{
  "text": "Un supermarché a été détruit par une explosion, samedi 2 janvier, à Grasse, dans les Alpes-Maritimes, a rapporté France 3. Aucun blessé n'est à déplorer.L'explosion s'est produite vers 6h du matin dans ce supermarché Aldi de Grasse. Elle a été suivie par un violent incendie. Le bâtiment a été \"totalement détruit\", selon le maire de la ville, qui a évoqué une cause \"accidentelle\" sur sa page Facebook. Une centaine de pompiers, ainsi que des policiers ont été mobilisés pour lutter contre le sinistre et sécuriser le périmètre.Selon Nice-Matin, deux employées du supermarché ont été soufflées par l'explosion en allumant la lumière au moment d'arriver sur leur lieu de travail. Aucune des deux n'a été blessée physiquement, mais elles sont très choquées.Vers 9h, le feu était maîtrisé, a indiqué à France 3 un porte-parole du Service d'incendie et de secours des Alpes-Maritimes. Soixante pompiers et 40 engins de secours étaient toujours mobilisés sur place.",
  "target_text": "Une centaine de pompiers ont été mobilisés pour lutter contre l'incendie."
}
```
```json
{
  "text": "Trois ans et demi après la décision des Britanniques de quitter l'Union européenne, le Brexit est finalement intervenu vendredi 31 janvier. Une mesure qui va sérieusement changer la donne pour les Britanniques qui siègent aujourd'hui dans les conseils municipaux en France. Comme tous les citoyens européens, les Britanniques avaient jusqu'à présent le droit de vote et d'éligibilité aux élections municipales françaises. Actuellement sur 2.493 conseillers étrangers, 757 viennent du Royaume-Uni, soit environ 30%, selon le Répertoire national des élus. Ils sont nettement plus nombreux que les Belges (544 élus) et les Portugais (357). Ils résident pour la plupart dans un grand quart Sud-Ouest de la France : Charente (70 élus), Dordogne (59), Aude (52), Haute-Vienne (40), Lot-et-Garonne (31), Hérault (30), Deux Sèvres (28), Gers (26), Lot (23)...Or, avec le Brexit, ils ne pourront pas briguer de nouveau mandat, à moins d'avoir acquis une autre nationalité européenne depuis les dernières élections. C'est notamment le cas à Poupas, village de 85 habitants dans le Tarn-et-Garonne, où deux des trois conseillers municipaux britanniques, sur les 11 au total que compte la commune, ont obtenu la nationalité française. Le droit \"de payer et de se taire\"Pour certaines petites communes, où il est souvent difficile de trouver des candidats, c'est un vrai casse-tête. À Perriers-en-Beauficel, dans la Manche, Patrick Head , originaire du Wiltshire (sud de l'Angleterre), va ainsi terminer son mandat. Le sexagénaire avait raflé pas moins de 89,74% des suffrages dans ce petit village normand, où il a élu domicile en 2004. Soit le meilleur score de cette commune de 216 habitants, où les électeurs peuvent rayer ou ajouter un nom. \"Ça va nous manquer car Patrick nous aidait beaucoup\", regrette la maire Lydie Brionne, qui explique que son colistier faisait \"le lien\" avec la cinquantaine de Britanniques installés dans ce coin de campagne normande. À Perriers-en-Beauficel, sur les onze élus de 2014, deux sont Britanniques. \"Il va falloir trouver deux nouveaux candidats. C'est difficile de trouver des gens motivés dans une petite commune\", souligne la maire, par ailleurs éleveuse de vaches laitières. \"Depuis 20 ans, beaucoup de Britanniques se sont installés, ils ont repeuplé la commune, ça a donné du dynamisme\", raconte l'élue. Avec le Brexit, \"j'ai peur qu'ils soient obligés de repartir.\"Loin d'être isolé, le cas de ce village normand se retrouve partout où les Britanniques sont fortement implantés. À Bellegarde-du-Razès, commune de 240 habitants dans l'Aude, les deux élus d'Outre-Manche \"apportent une valeur ajoutée\" au village, avec \"leur importante implication dans le milieu associatif\", estime le maire Gilbert De Paoli. L'Écossaise Alisson Mackie, 63 ans, installée depuis 2011, est dépitée de ne plus pouvoir se représenter en mars. \"On a construit notre maison ici, on paye des impôts ici, on consomme ici mais on a été rayés des listes électorales\", déplore-t-elle.À Jouac, village de 180 habitants en Haute-Vienne, la maire Virginie Windridge, 39 ans, elle-même mariée à un Britannique, trouve aussi \"très injuste que des gens qui sont là depuis des années, payent des impôts et contribuent à la vie de la commune, aient du jour au lendemain le droit 'de payer et de se taire'\". \"C'est dur à avaler\", dit-elle.Les deux élus britanniques actuels ont \"un apport important\", souligne la maire. \"Déjà ils sont un relais avec la communauté britannique de la commune. Et puis ils apportent des idées différentes, une autre façon de fonctionner, de voir les choses\", décrit Mme Windridge. \"Ils amènent parfois un regard sur ce qui existe ou se fait ailleurs, une autre perspective\". \"Et, il faut bien le dire, culturellement, quelquefois, les Britanniques sont plus ouverts aux changements que nous, ont un peu moins peur de l'inconnu\", ajoute-t-elle en donnant en exemple la décision d'éteindre l'éclairage public nocturne. \"Les élus britanniques étaient naturellement les plus ouverts sur cette idée-là, ils voyaient de suite le gagnant-gagnant, pour l'environnement et le budget de la commune\", estime-t-elle.",
  "target_text": "À l'heure actuelle, plus de 750 Britanniques siègent dans les conseils municipaux en France. Or, avec la sortie du Royaume-Uni de l'Union européenne, ils ne pourront pas se représenter en mars prochain."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:
  ```
  Les articles suivants sont accompagnés d'un résumé.
  ```
- Base prompt template:
  ```
  Article de presse: {text}
  Résumé: {target_text}
  ```
- Instruction-tuned prompt template:
  ```
  Article de presse: {text}

  Rédigez un résumé de l'article ci-dessus.
  ```

You can evaluate this dataset directly as follows:

```bash
$ euroeval --model <model-id> --dataset orange-sum
```
