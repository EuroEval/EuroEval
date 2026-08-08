# 🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scots

This is an overview of all the datasets used in the Scots part of EuroEval. The
datasets are grouped by their task – see the [task overview](/tasks) for more
information about what these constitute.

## Logical Reasoning

### ZebraPuzzleEasy-sco

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2511.03553)
and consists of logic grid puzzles (also known as Einstein's riddles or Zebra puzzles),
where the task is to determine which attributes belong to which house based on a set of
clues. This is the easy variant with 2 houses and 3 attribute categories.

The original full dataset consists of 128 / 128 / 1,024 samples for training, validation
and testing, respectively (so 1,280 samples used in total). We use the same splits.

Here are a few examples from the training split:

```json
{
  "text": "A row o hooses is nummered 1 tae 2 fae left tae richt.\n\nIn ilka hoose bides a body wi a unique attribute in ilka ane o the follaein categories:\n\nNationalities: Iceland an Spain.\nFavourite book genres: poetry an romance.\nHobbies: bouldering an tennis.\n\nWe ken the follaein forby:\n\n1. The poetry reader bides tae the left o the romance reader.\n2. The Icelander disnae bide in hoose nummer 1.\n3. The body that boulders bides niest tae the body that watches ski jumping.\n4. The body that boulders kens that there ar a lot o cars on the street.\n5. The romance reader enjoys solvin puzzles.\n6. The body that plays tennis kens that aw the hooses hae muckle windaes.\n7. The body that boulders bides in hoose nummer 2.\n8. The body that haes been tae Canada bides in hoose nummer 1.",
  "target_text": {
    "object_1": [
      "Spain",
      "poetry",
      "tennis"
    ],
    "object_2": [
      "Iceland",
      "romance",
      "bouldering"
    ]
  }
}
```

```json
{
  "text": "A row o hooses is nummered 1 tae 2 fae left tae richt.\n\nIn ilka hoose bides a body wi a unique attribute in ilka ane o the follaein categories:\n\nPets: dog an rabbit.\nFavourite book genres: non-fiction an poetry.\nFavourite fruits: banana an blackcurrant.\n\nWe ken the follaein forby:\n\n1. Coffee haes caffeine.\n2. The dug owner bides tae the richt o the body that loes blackcurrants.\n3. The body that aften sails disnae bide in hoose nummer 2.\n4. The body wi a guinea pig bides in hoose nummer 1.\n5. The body wi glesses loes physics.\n6. The body that loes blackcurrants is guid pals wi the body wi a tattoo.\n7. The poetry reader disnae like bananas.",
  "target_text": {
    "object_1": [
      "rabbit",
      "poetry",
      "blackcurrant"
    ],
    "object_2": [
      "dog",
      "non-fiction",
      "banana"
    ]
  }
}
```

```json
{
  "text": "A row o hooses is nummered 1 tae 2 fae left tae richt.\n\nIn ilka hoose bides a body wi a unique attribute in ilka ane o the follaein categories:\n\nJobs: nurse an police officer.\nPets: budgerigar an stick insect.\nFavourite book genres: poetry an romance.\n\nWe ken the follaein forby:\n\n1. The body wi a guinea pig bides in hoose nummer 1.\n2. The nurse disnae read romance novels.\n3. The body that plays the guitar bides in hoose nummer 1.\n4. The body that watches ski jumping bides in hoose nummer 2.\n5. The stick insect owner kens that aw the hooses hae muckle windaes.\n6. There ar a lot o cars on the street.\n7. The stick insect owner bides in hoose nummer 1.\n8. The nurse disnae bide in hoose nummer 1.",
  "target_text": {
    "object_1": [
      "police officer",
      "stick insect",
      "romance"
    ],
    "object_2": [
      "nurse",
      "budgerigar",
      "poetry"
    ]
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt: (empty)
- Instruction prompt:

  ```text
  Here's a raivel:
  <riddle>
  {text}
  </riddle>

  Whae has whit attributes and bides in whit hoose?

  Please gie yer answer as a JSON dictionary. Ilka key maun be object_X whaur X is the hoose nummer. Ilka value maun be a leet o the attributes frae the forsaid categories that belang tae the body in hoose nummer X.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset zebra-puzzles-easy-sco
```

### Unofficial: ZebraPuzzleHard-sco

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2511.03553)
and consists of logic grid puzzles (also known as Einstein's riddles or Zebra puzzles),
where the task is to determine which attributes belong to which house based on a set of
clues. This is the hard variant with 4 houses and 5 attribute categories.

The original full dataset consists of 128 / 128 / 1,024 samples for training, validation
and testing, respectively (so 1,280 samples used in total). We use the same splits.

Here are a few examples from the training split:

```json
{
  "text": "A row o hooses is nummered 1 tae 4 fae left tae richt.\n\nIn ilka hoose bides a body wi a unique attribute in ilka ane o the follaein categories:\n\nPets: cat, dog, stick insect an zebra.\nDrinks: juice, smoothie, soda an tea.\nFavourite book genres: crime, fantasy, poetry an romance.\nHobbies: board games, bouldering, football an tennis.\nFavourite fruits: banana, blackcurrant, orange an strawberry.\n\nWe ken the follaein forby:\n\n1. The dug owner bides jist tae the left o the body that loes blackcurrants.\n2. The juice drinker bides niest tae the body that watches ski jumping.\n3. There ar 2 hooses atween the smoothie drinker an the body that plays tennis.\n4. The ginger drinker bides tae the richt o the romance reader.\n5. There is ane hoose atween the stick insect owner an the smoothie drinker.\n6. The fantasy reader haes a guinea pig.\n7. The body wi a pet that is auld for its kind bides in hoose nummer 3.\n8. The zebra owner bides tae the left o the juice drinker.\n9. There is ane hoose atween the smoothie drinker an the romance reader.\n10. The ginger drinker bides atween the cat owner an the poetry reader.\n11. The body wi reid hair disnae bide in hoose nummer 2.\n12. The body that boulders loes oranges.\n13. The cat owner bides niest tae the body that plays board games.\n14. The body that loes bananas aften sails.\n15. There ar 2 hooses atween the fantasy reader an the body that loes bananas.",
  "target_text": {
    "object_1": [
      "zebra",
      "tea",
      "poetry",
      "tennis",
      "banana"
    ],
    "object_2": [
      "stick insect",
      "juice",
      "romance",
      "bouldering",
      "orange"
    ],
    "object_3": [
      "dog",
      "soda",
      "crime",
      "board games",
      "strawberry"
    ],
    "object_4": [
      "cat",
      "smoothie",
      "fantasy",
      "football",
      "blackcurrant"
    ]
  }
}
```

```json
{
  "text": "A row o hooses is nummered 1 tae 4 fae left tae richt.\n\nIn ilka hoose bides a body wi a unique attribute in ilka ane o the follaein categories:\n\nJobs: police officer, shop assistant, software developer an teacher.\nDrinks: cocoa, milk, soda an tea.\nFavourite book genres: crime, horror, romance an science fiction.\nHobbies: board games, crocheting, painting an tennis.\nFavourite fruits: apple, blackcurrant, pear an wild strawberry.\n\nWe ken the follaein forby:\n\n1. The tea drinker loes blackcurrants.\n2. The dominie bides jist tae the richt o the body that loes aiples.\n3. The body that plays the guitar bides in hoose nummer 1.\n4. The body that haes a cactus disnae bide in hoose nummer 3.\n5. The body wi a pet that is auld for its kind disnae bide in hoose nummer 4.\n6. There is ane hoose atween the ginger drinker an the body that loes pears.\n7. The dominie crochets.\n8. There ar 2 hooses atween the shop assistant an the romance reader.\n9. Snails ar molluscs.\n10. There is ane hoose atween the horror reader an the body that loes aiples.\n11. The cocoa drinker bides atween the polisman an the body that loes blackcurrants.\n12. The body wi a bike bides in hoose nummer 4.\n13. There is ane hoose atween the science fiction reader an the body that paints.\n14. The cocoa drinker loes wild strawberries.\n15. The ginger drinker plays board games.",
  "target_text": {
    "object_1": [
      "police officer",
      "soda",
      "romance",
      "board games",
      "apple"
    ],
    "object_2": [
      "teacher",
      "cocoa",
      "science fiction",
      "crocheting",
      "wild strawberry"
    ],
    "object_3": [
      "software developer",
      "milk",
      "horror",
      "tennis",
      "pear"
    ],
    "object_4": [
      "shop assistant",
      "tea",
      "crime",
      "painting",
      "blackcurrant"
    ]
  }
}
```

```json
{
  "text": "A row o hooses is nummered 1 tae 4 fae left tae richt.\n\nIn ilka hoose bides a body wi a unique attribute in ilka ane o the follaein categories:\n\nNationalities: Netherlands, Norway, Spain an Sweden.\nJobs: baker, nurse, police officer an teacher.\nPets: budgerigar, cat, dog an stick insect.\nDrinks: coffee, juice, smoothie an soda.\nHobbies: board games, bouldering, crocheting an painting.\n\nWe ken the follaein forby:\n\n1. The body wi reid hair disnae bide in hoose nummer 4.\n2. There is ane hoose atween the juice drinker an the body that paints.\n3. The polisman bides tae the richt o the cat owner.\n4. The coffee drinker bides atween the dominie an the body that plays board games.\n5. The Norwegian bides jist tae the left o the budgie owner.\n6. The body wi a guinea pig disnae bide in hoose nummer 2.\n7. The stick insect owner bides niest tae the body wi a bike.\n8. The nurse disnae hae a cat.\n9. The body wi glesses disnae bide in hoose nummer 1.\n10. The body that loes physics disnae bide in hoose nummer 1.\n11. The Dutchman disnae bide atween the Spaniard an the baxter, an they ur three different fowk.\n12. The coffee drinker paints.\n13. There ar 2 hooses atween the budgie owner an the dug owner.\n14. The body that boulders bides tae the left o the body that paints.\n15. The polisman bides in hoose nummer 3.\n16. The smoothie drinker bides in hoose nummer 1.\n17. There ar 2 hooses atween the Dutchman an the body that crochets.",
  "target_text": {
    "object_1": [
      "Netherlands",
      "teacher",
      "dog",
      "smoothie",
      "bouldering"
    ],
    "object_2": [
      "Sweden",
      "baker",
      "cat",
      "coffee",
      "painting"
    ],
    "object_3": [
      "Norway",
      "police officer",
      "stick insect",
      "soda",
      "board games"
    ],
    "object_4": [
      "Spain",
      "nurse",
      "budgerigar",
      "juice",
      "crocheting"
    ]
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt: (empty)
- Instruction prompt:

  ```text
  Here's a raivel:
  <riddle>
  {text}
  </riddle>

  Whae has whit attributes and bides in whit hoose?

  Please gie yer answer as a JSON dictionary. Ilka key maun be object_X whaur X is the hoose nummer. Ilka value maun be a leet o the attributes frae the forsaid categories that belang tae the body in hoose nummer X.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset zebra-puzzles-hard-sco
```
