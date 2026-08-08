# 🇮🇪 Irish

This is an overview of all the datasets used in the Irish part of EuroEval. The
datasets are grouped by their task – see the [task overview](/tasks) for more
information about what these constitute.

## Logical Reasoning

### ZebraPuzzleEasy-ga

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2511.03553)
and consists of logic grid puzzles (also known as Einstein's riddles or Zebra puzzles),
where the task is to determine which attributes belong to which house based on a set of
clues. This is the easy variant with 2 houses and 3 attribute categories.

The original full dataset consists of 128 / 128 / 1,024 samples for training, validation
and testing, respectively (so 1,280 samples used in total). We use the same splits.

Here are a few examples from the training split:

```json
{
  "text": "Tá sraith tithe uimhrithe 1 go 2 ó chlé go deas.\n\nTá duine ina chónaí i ngach teach, agus tá saintréith ar leith aige i ngach ceann de na catagóirí seo a leanas:\n\nNáisiúntachtaí: Iodáil agus Ísiltír.\nPoist: forbróir bogearraí agus garda.\nTorthaí is fearr: cuirín dubh agus úll.\n\nTá an méid seo a leanas ar eolas againn freisin:\n\n1. Tá cónaí ar an nÍsiltíreach in aice leis an nduine a sheolann go minic.\n2. Tá a fhios ag an bhforbróir bogearraí go bhfuil fuinneoga móra ag gach teach.\n3. Níl cónaí ar an nÍsiltíreach i dteach uimhir 1.\n4. Tá an duine a bhfuil dúil aige i gcuiríní dubha ina dhuine a chaitheann spéaclaí.\n5. Tá an garda ina dhuine a bhfuil peata aige atá sean dá speiceas.\n6. Tá dúil ag an bhforbróir bogearraí i gcuiríní dubha.\n7. Tá cónaí ar an nduine a bhfuil dúil aige san fhisic i dteach uimhir 1.\n8. Tá dúil ag an nIodálach in úlla.",
  "target_text": {
    "object_1": [
      "Iodáil",
      "garda",
      "úll"
    ],
    "object_2": [
      "Ísiltír",
      "forbróir bogearraí",
      "cuirín dubh"
    ]
  }
}
```

```json
{
  "text": "Tá sraith tithe uimhrithe 1 go 2 ó chlé go deas.\n\nTá duine ina chónaí i ngach teach, agus tá saintréith ar leith aige i ngach ceann de na catagóirí seo a leanas:\n\nPoist: altra agus garda.\nDeochanna: caife agus smoothie.\nSeánraí leabhar: bleachtaireacht agus ficsean eolaíochta.\n\nTá an méid seo a leanas ar eolas againn freisin:\n\n1. Tá caidreamh maith idir an duine a ólann caife agus an duine a bhí i gCeanada.\n2. Tá an duine a cheapann gurb é mango an dara toradh is fearr ina dhuine a bhfuil deirfiúr aige.\n3. Tá fuinneoga móra ag gach teach.\n4. Tá an duine a bhfuil tatú air ina dhuine a fhéachann ar léimneach sciála.\n5. Níl cónaí ar an nduine a léann bleachtaireacht i dteach uimhir 2.\n6. Níl cónaí ar an nduine a bhfuil rothar aige i dteach uimhir 1.\n7. Tá cónaí ar an n-altra ar chlé den dhuine a ólann smoothie.",
  "target_text": {
    "object_1": [
      "altra",
      "caife",
      "bleachtaireacht"
    ],
    "object_2": [
      "garda",
      "smoothie",
      "ficsean eolaíochta"
    ]
  }
}
```

```json
{
  "text": "Tá sraith tithe uimhrithe 1 go 2 ó chlé go deas.\n\nTá duine ina chónaí i ngach teach, agus tá saintréith ar leith aige i ngach ceann de na catagóirí seo a leanas:\n\nNáisiúntachtaí: Oileáin Fharó agus Ríocht Aontaithe.\nPoist: altra agus múinteoir.\nPeataí: budragár agus madra.\n\nTá an méid seo a leanas ar eolas againn freisin:\n\n1. Tá cónaí ar an mBriotanach in aice leis an nduine a bhfuil tatú air.\n2. Tá an duine a bhfuil máistreacht sa mhatamaitic aige ina dhuine a sheinneann an ghiotár.\n3. Tá cónaí ar an bhFaróch ar chlé den dhuine a bhfuil budragár aige.\n4. Níl cónaí ar an nduine rua i dteach uimhir 1.\n5. Is moilisc iad seilidí.\n6. Tá cónaí ar an múinteoir ar dheis den dhuine a bhfuil madra aige.\n7. Tá an duine a bhfuil deirfiúr aige ina dhuine nach bhfuil cachtas aige.",
  "target_text": {
    "object_1": [
      "Oileáin Fharó",
      "altra",
      "madra"
    ],
    "object_2": [
      "Ríocht Aontaithe",
      "múinteoir",
      "budragár"
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
  Seo tomhas duit:
  <riddle>
  {text}
  </riddle>

  Cé a bhfuil na tréithe seo aige agus a chónaíonn i cén teach?

  Cuir do fhreagra isteach le do thoil mar fhoclóir JSON. Caithfidh gach key a bheith object_X áit a bhfuil X mar uimhir an tí. Caithfidh gach value a bheith ina liosta de na tréithe ó na catagóirí thuasluaite a bhaineann leis an duine i dteach uimhir X.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset zebra-puzzles-easy-ga
```

### Unofficial: ZebraPuzzleHard-ga

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2511.03553)
and consists of logic grid puzzles (also known as Einstein's riddles or Zebra puzzles),
where the task is to determine which attributes belong to which house based on a set of
clues. This is the hard variant with 4 houses and 5 attribute categories.

The original full dataset consists of 128 / 128 / 1,024 samples for training, validation
and testing, respectively (so 1,280 samples used in total). We use the same splits.

Here are a few examples from the training split:

```json
{
  "text": "Tá sraith tithe uimhrithe 1 go 4 ó chlé go deas.\n\nTá duine ina chónaí i ngach teach, agus tá saintréith ar leith aige i ngach ceann de na catagóirí seo a leanas:\n\nNáisiúntachtaí: Iodáil, Oileáin Fharó, Sualainn agus Íoslainn.\nPoist: cúntóir siopa, forbróir bogearraí, garda agus múinteoir.\nDeochanna: bainne, cócó, smoothie agus sú.\nSeánraí leabhar: ficsean eolaíochta, filíocht, neamhfhicsean agus scéalta uafáis.\nCaitheamh aimsire: cluichí boird, cróiseáil, leadóg agus péintéireacht.\n\nTá an méid seo a leanas ar eolas againn freisin:\n\n1. Tá an duine a bhfuil dúil aige san fhisic ina dhuine nach bhfuil cachtas aige.\n2. Tá cónaí ar an nÍoslannach díreach ar chlé den Iodálach.\n3. Tá cónaí ar an nduine a ólann cócó díreach ar chlé den dhuine a phéinteálann.\n4. Tá 2 teach idir an tÍoslannach agus an duine a léann scéalta uafáis.\n5. Tá cónaí ar an gcúntóir siopa ar chlé den dhuine a léann neamhfhicsean.\n6. Tá cónaí ar an bhforbróir bogearraí i dteach uimhir 4.\n7. Tá an duine a fhéachann ar léimneach sciála ina dhuine a bhfuil muc ghuine aige.\n8. Níl cónaí ar an nduine a chaitheann spéaclaí i dteach uimhir 3.\n9. Tá cónaí ar an nduine a imríonn cluichí boird i dteach uimhir 1.\n10. Tá cónaí ar an nduine a ólann smoothie i dteach uimhir 3.\n11. Tá an Faróch ina gharda.\n12. Tá cónaí ar an bhFaróch díreach ar dheis den dhuine a léann ficsean eolaíochta.\n13. Níl cónaí ar an nduine a ólann sú in aice leis an nduine a phéinteálann, agus daoine difriúla iad.\n14. Tá cónaí ar an nduine a bhfuil peata aige atá sean dá speiceas i dteach uimhir 3.\n15. Tá an duine a léann filíocht ina dhuine a bhfuil máistreacht sa mhatamaitic aige.\n16. Tá cónaí ar an nduine a ólann cócó idir an cúntóir siopa agus an duine a chróiseálann.",
  "target_text": {
    "object_1": [
      "Íoslainn",
      "cúntóir siopa",
      "sú",
      "filíocht",
      "cluichí boird"
    ],
    "object_2": [
      "Iodáil",
      "múinteoir",
      "cócó",
      "ficsean eolaíochta",
      "leadóg"
    ],
    "object_3": [
      "Oileáin Fharó",
      "garda",
      "smoothie",
      "neamhfhicsean",
      "péintéireacht"
    ],
    "object_4": [
      "Sualainn",
      "forbróir bogearraí",
      "bainne",
      "scéalta uafáis",
      "cróiseáil"
    ]
  }
}
```

```json
{
  "text": "Tá sraith tithe uimhrithe 1 go 4 ó chlé go deas.\n\nTá duine ina chónaí i ngach teach, agus tá saintréith ar leith aige i ngach ceann de na catagóirí seo a leanas:\n\nPoist: aire, altra, cúntóir siopa agus forbróir bogearraí.\nDeochanna: bainne, caife, cócó agus tae.\nSeánraí leabhar: fantaisíocht, ficsean eolaíochta, filíocht agus rómánsaíocht.\nCaitheamh aimsire: bollánóireacht, leadóg, liathróid láimhe agus péintéireacht.\nTorthaí is fearr: banana, cuirín dubh, oráiste agus piorra.\n\nTá an méid seo a leanas ar eolas againn freisin:\n\n1. Tá cónaí ar an nduine a léann ficsean eolaíochta in aice leis an nduine a imríonn cluichí físeáin.\n2. Tá 2 teach idir an t-altra agus an duine a imríonn leadóg.\n3. Tá cónaí ar an nduine a bhfuil dúil aige i bpiorraí in aice leis an nduine a bhfuil dúil aige in oráistí.\n4. Tá a fhios ag an nduine a imríonn leadóg gurb éisc iad na scadáin.\n5. Níl cónaí ar an nduine a léann fantaisíocht in aice leis an nduine a léann ficsean eolaíochta, agus daoine difriúla iad.\n6. Níl cónaí ar an nduine a ólann tae in aice leis an nduine a ólann cócó, agus daoine difriúla iad.\n7. Tá cónaí ar an nduine a imríonn liathróid láimhe i dteach uimhir 4.\n8. Níl cónaí ar an nAire idir an cúntóir siopa agus an duine a bhfuil dúil aige i mbananaí, agus triúr daoine difriúla iad.\n9. Tá 2 teach idir an duine a ólann cócó agus an duine a ólann caife.\n10. Níl cónaí ar an n-altra in aice leis an nduine a dhéanann bollánóireacht, agus daoine difriúla iad.\n11. Níl cónaí ar an nduine a léann ficsean eolaíochta i dteach uimhir 4.\n12. Níl cónaí ar an nduine a léann ficsean eolaíochta i dteach uimhir 1.\n13. Tá cónaí ar an nduine a léann fantaisíocht in aice leis an nduine a bhfuil dúil aige i bpiorraí.\n14. Tá cónaí ar an bhforbróir bogearraí díreach ar dheis den dhuine a ólann tae.\n15. Tá caidreamh maith idir an duine a léann ficsean eolaíochta agus an duine a bhí i gCeanada.\n16. Tá cónaí ar an gcúntóir siopa díreach ar chlé den dhuine a léann úrscéalta rómánsúla.\n17. Tá an duine a léann úrscéalta rómánsúla ina dhuine nach bhfuil cachtas aige.\n18. Tá an duine a bhfuil dúil aige san fhisic ina dhuine a bhfuil peata aige atá sean dá speiceas.",
  "target_text": {
    "object_1": [
      "aire",
      "caife",
      "filíocht",
      "leadóg",
      "cuirín dubh"
    ],
    "object_2": [
      "cúntóir siopa",
      "tae",
      "ficsean eolaíochta",
      "bollánóireacht",
      "oráiste"
    ],
    "object_3": [
      "forbróir bogearraí",
      "bainne",
      "rómánsaíocht",
      "péintéireacht",
      "piorra"
    ],
    "object_4": [
      "altra",
      "cócó",
      "fantaisíocht",
      "liathróid láimhe",
      "banana"
    ]
  }
}
```

```json
{
  "text": "Tá sraith tithe uimhrithe 1 go 4 ó chlé go deas.\n\nTá duine ina chónaí i ngach teach, agus tá saintréith ar leith aige i ngach ceann de na catagóirí seo a leanas:\n\nPoist: báicéir, cúntóir siopa, garda agus múinteoir.\nPeataí: budragár, cat, coinín agus seilide.\nDeochanna: cócó, cóla, sú agus tae.\nSeánraí leabhar: bleachtaireacht, filíocht, neamhfhicsean agus scéalta uafáis.\nTorthaí is fearr: banana, oráiste, piorra agus sú talún fiáin.\n\nTá an méid seo a leanas ar eolas againn freisin:\n\n1. Tá teach amháin idir an duine a léann filíocht agus an duine a léann scéalta uafáis.\n2. Níl cónaí ar an nduine a bhfuil dúil aige san fhisic i dteach uimhir 1.\n3. Tá a fhios ag an nduine a ólann cóla gurb éisc iad na scadáin.\n4. Tá cónaí ar an ngarda díreach ar dheis den dhuine a ólann tae.\n5. Tá a fhios ag an nduine a léann filíocht go bhfuil doras glas ag roinnt de na tithe.\n6. Tá cónaí ar an nduine a ólann cóla díreach ar chlé den dhuine a léann filíocht.\n7. Tá cónaí ar an mbáicéir ar dheis den dhuine a bhfuil dúil aige i bpiorraí.\n8. Tá cónaí ar an nduine a ólann tae díreach ar dheis den dhuine a ólann cóla.\n9. Níl cónaí ar an múinteoir idir an duine a ólann sú agus an duine a léann bleachtaireacht, agus triúr daoine difriúla iad.\n10. Níl dúil ag an gcúntóir siopa in oráistí.\n11. Níl cónaí ar an nduine a bhfuil seilide aige in aice leis an nduine a bhfuil dúil aige i sú talún fiáin, agus daoine difriúla iad.\n12. Níl cónaí ar an nduine a ólann cócó i dteach uimhir 1.\n13. Tá an grianchóras ag gluaiseacht ar luas thart ar 200 km/s timpeall lár na réaltra.\n14. Níl cónaí ar an nduine a ólann cócó in aice leis an nduine a bhfuil dúil aige i mbananaí, agus daoine difriúla iad.\n15. Tá cónaí ar an nduine a bhfuil coinín aige i dteach uimhir 1.\n16. Tá 2 teach idir an múinteoir agus an duine a bhfuil cat aige.\n17. Is moilisc iad seilidí.",
  "target_text": {
    "object_1": [
      "múinteoir",
      "coinín",
      "cóla",
      "neamhfhicsean",
      "banana"
    ],
    "object_2": [
      "cúntóir siopa",
      "seilide",
      "tae",
      "filíocht",
      "piorra"
    ],
    "object_3": [
      "garda",
      "budragár",
      "cócó",
      "bleachtaireacht",
      "oráiste"
    ],
    "object_4": [
      "báicéir",
      "cat",
      "sú",
      "scéalta uafáis",
      "sú talún fiáin"
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
  Seo tomhas duit:
  <riddle>
  {text}
  </riddle>

  Cé a bhfuil na tréithe seo aige agus a chónaíonn i cén teach?

  Cuir do fhreagra isteach le do thoil mar fhoclóir JSON. Caithfidh gach key a bheith object_X áit a bhfuil X mar uimhir an tí. Caithfidh gach value a bheith ina liosta de na tréithe ó na catagóirí thuasluaite a bhaineann leis an duine i dteach uimhir X.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset zebra-puzzles-hard-ga
```
