# 🇩🇪 German

This is an overview of all the datasets used in the German part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### SB10k

This dataset was published in [this paper](https://aclanthology.org/W17-1106/) and is
based on German tweets, which were manually annotated by three annotators.

The original full dataset consists of 1,840 / 324 / 870 samples, and we use a 1,024 /
256 / 1,024 split for training, validation and testing, respectively. The splits are new
and there can thus be some overlap between the original validation and test sets and our
validation and test sets.

Here are a few examples from the training split:

```json
{
  "text": "ALEMANHA (4-5-1): Neuer; Schmelzer, Hummels, Mertesacker, Lahm; Gündogan, Khedira, Özil, Müller, Reus; Klose",
  "label": "positive"
}
```

```json
{
  "text": "@user ok. Bin jetzt dann hernach gleich nochmal weg, aber schreib ruhig.",
  "label": "neutral"
}
```

```json
{
  "text": "@user Schwüle 34°, Tendenz steigend. #schrecklich",
  "label": "negative"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Im Folgenden sind Tweets und ihre Stimmung aufgeführt, die 'positiv', 'neutral' oder 'negativ' sein kann.
  ```

- Base prompt template:

  ```text
  Tweet: {text}
  Stimmungslage: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Tweet: {text}

  Klassifizieren Sie die Stimmung im Tweet. Antworten Sie mit 'positiv', 'neutral' oder 'negativ'.
  ```

- Label mapping:
  - `positive` ➡️ `positiv`
  - `neutral` ➡️ `neutral`
  - `negative` ➡️ `negativ`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset sb10k
```

## Named Entity Recognition

### GermEval

This dataset was published in [this paper](https://aclanthology.org/L14-1251/) and is
based on German Wikipedia as well as news articles, and was manually annotated. It
roughly follows the CoNLL-2003 format, but also allows overlapping entities and derived
entities (such as "English" for "England"). We remove the derived entities and convert
the partially overlapping entities to non-overlapping entities (e.g., `B-ORGpart` to
`B-ORG`).

The original full dataset consists of 24,000 / 2,200 / 5,100 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 1,024 split for training,

Here are a few examples from the training split:

```json
{
  'tokens': array(['Am', 'Ende', 'der', 'Saison', '2006/07', 'soll', 'es', 'für', 'die', 'Löwen', 'wieder', 'zu', 'einem', 'Europapokal-Platz', 'reichen', '.'], dtype=object),
  'labels': array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'O', 'O', 'O', 'B-LOC', 'O', 'O'], dtype=object)
}
```

```json
{
  'tokens': array(['In', 'einer', 'Stichwahl', 'gegen', 'seinen', 'Vorgänger', 'Georg', 'Kronawitter', 'wurde', 'Erich', 'Kiesl', 'am', '1.', 'April', '1984', 'abgewählt', '.'], dtype=object),
  'labels': array(['O', 'O', 'O', 'O', 'O', 'O', 'B-PER', 'I-PER', 'O', 'B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
```

```json
{
  'tokens': array(['Noch', 'im', '13.', 'Jahrhundert', 'wurde', 'sie', 'in', 'manchen', 'Handschriften', 'mit', 'der', 'Christherre-Chronik', 'verschmolzen', '.'], dtype=object),
  'labels': array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'O', 'O'], dtype=object)
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Es folgen Sätze und JSON-Wörterbücher mit den benannten Entitäten, die in der angegebenen Phrase vorkommen.
  ```

- Base prompt template:

  ```text
  Satz: {text}
  Benannte Entitäten: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Satz: {text}

  Identifizieren Sie die benannten Entitäten im Satz. Sie sollten dies als JSON-Wörterbuch mit den Schlüsseln 'person', 'ort', 'organisation' und 'verschiedenes' ausgeben. Die Werte sollten Listen der benannten Entitäten dieses Typs sein, genau wie sie im Satz erscheinen.
  ```

- Label mapping:
  - `B-PER` ➡️ `person`
  - `I-PER` ➡️ `person`
  - `B-LOC` ➡️ `ort`
  - `I-LOC` ➡️ `ort`
  - `B-ORG` ➡️ `organisation`
  - `I-ORG` ➡️ `organisation`
  - `B-MISC` ➡️ `verschiedenes`
  - `I-MISC` ➡️ `verschiedenes`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset germeval
```

## Linguistic Acceptability

### ScaLA-de

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the
[German Universal Dependencies treebank](https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD)
by assuming that the documents in the treebank are correct, and corrupting the samples
to create grammatically incorrect samples. The corruptions were done by either removing
a word from a sentence, or by swapping two neighbouring words in a sentence. To ensure
that this does indeed break the grammaticality of the sentence, a set of rules were used
on the part-of-speech tags of the words in the sentence.

The original dataset consists of 15,590 samples, from which we use 1,024 / 256 / 2,048
samples for training, validation and testing, respectively (so 3,328 samples used in
total). These splits are used as-is in the framework.

Here are a few examples from the training split:

```json
{
  "text": "Im In dem Sommer draußen zu sitzen ist immer wieder eine \"Wonne\", so man noch einen Platz bekommt",
  "label": "correct"
}
```

```json
{
  "text": "Eine 65 m lange Betonmauer trägt nachts einen Leucht - Schriftzug \"HOSTAL HOSTILE HOTEL HOSTAGE GOSTIN OSTILE HOSTEL HOSTIL HOST\", was in seinem etymologischen Wortspiel so viel bedeutet, dass aus einem feindlichen ein gastfreundlicher Ort geworden ist, in Anspielung auf das auf dem Gelände des ehemaligen Frauenlagers genau gegenüber liegende Novotel Goldene Bremm (heute Mercure Saarbrücken - Süd), das konzeptionell insoweit in die Idee einbezogen ist.",
  "label": "incorrect"
}
```

```json
{
  "text": "Allerdings wurde nachgewiesen, dass sich der ebenfalls in Extremlebensräumen vorkommende Nematode Halicephalobus mephisto im in dem Labor bevorzugt Desulforudis audaxviator ernährt, wenn er eine Wahl hat (Alternative: E. coli).",
  "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Die folgenden Sätze und ob sie grammatikalisch korrekt sind.
  ```

- Base prompt template:

  ```text
  Satz: {text}
  Grammatikalisch richtig: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Satz: {text}

  Bestimmen Sie, ob der Satz grammatikalisch korrekt ist oder nicht. Antworten Sie mit 'ja', wenn der Satz korrekt ist und 'nein', wenn er es nicht ist.
  ```

- Label mapping:
  - `correct` ➡️ `ja`
  - `incorrect` ➡️ `nein`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-de
```

## Reading Comprehension

### GermanQuAD

This dataset was published in [this paper](https://aclanthology.org/2021.mrqa-1.4/) and
is based on German Wikipedia articles, and was manually annotated.

The original full dataset consists of 11,518 / 2,204 samples for training and testing,
respectively. We use a 1,024 / 256 / 2,048 split for training, validation and testing,
respectively (so 3,328 samples used in total). These splits are new and there can thus
be some overlap between the original validation and test sets and our validation and
test sets.

Here are a few examples from the training split:

```json
{
  "context": "Mali\n\n=== Verwaltungsgliederung ===\nDer Staat gliedert sich in zehn Regionen und den Hauptstadtdistrikt. Diese teilen sich in 49 Kreise ''(cercles)'' und 703 Gemeinden ''(communes)''. Die Regionen sind nach ihren Hauptstädten benannt. Zwei dieser zehn Regionen, Ménaka und Taoudénit, wurden 2012 per Gesetzesbeschluss gebildet. Die Einrichtung ist seit 2016 im Gange.\nDie Angaben der Regionen Gao und Timbuktu, aus denen die Regionen Ménaka und Taoudénit ausgegliedert wurden, spiegeln noch den Stand vor der Aufspaltung wider.\nUm auch Flüchtlinge und vor allem Nomaden in das Verwaltungssystem eingliedern zu können, entstanden sogenannte ''Fractions'' (''Fractions Nomades'', ein Begriff, den schon die Kolonialregierung nutzte), die es dementsprechend vor allem im Norden in der Nähe von Dörfern gibt. Seit den großen Trockenphasen entstanden durch Wanderungsbewegungen solche Verwaltungseinheiten allerdings auch verstärkt im Süden.",
  "question": 'Wie viele verschiedene Regionen hat Mali? ',
  "answers": {
    "answer_start": array([63], dtype=int32),
    "text": array(['zehn Regionen und den Hauptstadtdistrikt'], dtype=object)
  }
}
```

```json
{
  "context": 'Iran\n\n=== Automobilindustrie ===\nIn der Automobilindustrie waren 2010 rund 500.000 Menschen beschäftigt, damit ist die Branche der zweitgrößte Arbeitgeber nach der Ölindustrie und der Iran der größte Automobilproduzent im Mittleren Osten. 2012 ist die Automobilproduktion des Iran jedoch scharf eingebrochen; es wurden nur noch 989.110 Fahrzeuge produziert – 40 Prozent weniger als 2011. Darunter fallen 848.000 PKW und 141.110 Nutzfahrzeuge.\nDie beiden größten Automobilhersteller sind die staatliche SAIPA – derzeit im Privatisierungsprozess – und Iran Khodro (IKCO). Die IKCO produziert neben einheimischen Modellen wie Dena und Runna in Lizenz Modelle u.\xa0a. von Peugeot. SAIPA hat die IKCO im Jahr 2010 das erste Mal in der Rangfolge überholt. Nach Ansicht des Business Monitor International’s Iran Autos Report wird sich die Belastbarkeit der iranischen Automobilindustrie erst in den nächsten Jahren zeigen, wenn der einheimische Markt gesättigt ist und der Iran zunehmend auf dem internationalen Markt agiert, denn bisher ist der Produktionsanstieg noch überwiegend auf die Unterstützung der Regierung zurückzuführen. 12,64 % der zugelassenen Kraftfahrzeuge werden mit Gas betrieben. Der Iran liegt damit weltweit an fünfter Stelle der Nutzung von gasbetriebenen Kraftfahrzeugen.\nDer schwedische LKW-Produzent Scania eröffnete 2011 eine neue Produktionslinie in Qazvin und löst damit Daimler-Chrysler ab, das seine Geschäftskontakte mit dem Iran abgebrochen hat.',
  "question": 'Wie heißen die Automodelle von Iran Khodro?',
  "answers": {
    "answer_start": array([622], dtype=int32),
    "text": array([' Dena und Runna'], dtype=object)
  }
}
```

```json
{
  "context": 'Griechenland\n\n=== Klima ===\nGriechenland hat überwiegend ein mediterranes Klima mit feucht-milden Wintern und trocken-heißen Sommern. An der Küste ist es im Winter sehr mild und es regnet häufig; Schnee fällt nur selten. Die Sommer sind relativ heiß und es gibt nur gelegentlich Sommergewitter. Mit 48° wurde 1977 in Griechenland der kontinentaleuropäische Hitzerekord gemessen.\nIm Landesinneren ist es vor allem im Winter deutlich kühler und es gibt häufig Nachtfrost, manchmal auch starke Schneefälle. Der Frühling ist kurz, verwöhnt aber „mit einem Feuerwerk aus Lavendel und Anemonen, Klatschmohn und Kamille“. Im Sommer ist es ähnlich wie an der Küste heiß und trocken. Die jährlichen Niederschläge schwanken zwischen 400 und 1000\xa0mm. Da Griechenland sehr gebirgig ist, ist Wintersport durchaus möglich, es existieren 19 Wintersportgebiete unterschiedlicher Größe. Ein kleiner Teil im Nordwesten des Festlandes liegt in der gemäßigten Klimazone.',
  "question": 'Wie oft schneit es in Griechenland?',
  "answers": {
    "answer_start": array([209], dtype=int32),
    "text": array(['nur selten'], dtype=object)
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Im Folgenden finden Sie Texte mit den dazugehörigen Fragen und Antworten.
  ```

- Base prompt template:

  ```text
  Text: {text}
  Fragen: {question}
  Fragen Antwort in maximal 3 Wörtern: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Text: {text}

  Beantworten Sie die folgende Frage zum obigen Text in höchstens 3 Wörtern.

  Frage: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset germanquad
```

### Unofficial: XQuAD-de

This dataset was published in [this paper](https://aclanthology.org/2020.acl-main.421/)
and contains 1190 question-answer pairs from
[SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) translated into ten languages
by professional translators.

The dataset is split into 550 / 128 / 512 question-answer pairs for training,
validation, and testing, respectively.

Here are a few examples from the training split:

```json
{
    "context": "Irgendwann vor etwa einer Milliarde Jahren drang ein freilebendes Cyanobakterium in eine frühe eukaryotische Zelle ein, entweder als Nahrung oder als interner Parasit, schaffte es aber, aus der phagozytischen Vakuole zu entkommen, in der es enthalten war. Die innersten Lipiddoppelschicht-Membranen, die alle Chloroplasten umgeben, entsprechen der äußeren und inneren Membran der gramnegativen Zellwand des anzestralen Cyanobakteriums und nicht der phagosomalen Membran des Wirts, die wahrscheinlich verloren ging. Der neue Zellenbewohner wurde schnell zu einem Vorteil, indem er dem eukaryotischen Wirt, welcher ihm erlaubte, in ihm zu leben, Nahrung zur Verfügung stellte. Mit der Zeit wurde das Cyanobakterium assimiliert und viele seiner Gene gingen verloren oder in den Nukleus des Wirts übertragen. Einige seiner Proteine wurden dann im Zellplasma der Wirtzelle synthetisiert und zurück in den Chloroplasten (das ehemalige Cyanobakterium) importiert.",
    "question": "In welche Art von Zell drangen vor langer Zeit Cynaobakterien ein?",
    "answers": {
        "answer_start": array([95], dtype=int32),
        "text": array(["eukaryotische"], dtype=object)
    }
}
```

```json
{
    "context": "Im tibetischen Buddhismus werden die Dharma-Lehrer/innen gewöhnlich als Lama bezeichnet. Ein Lama, der sich durch Phowa und Siddhi bewusst zur Wiedergeburt, häufig mehrere Male, entschlossen hat, um sein Bodhisattva-Gelübte fortsetzen zu können, wird Tulku genannt.",
    "question": "Zu wie vielen Wiedergeburten hat sich ein Lama bereiterklärt?",
    "answers": {
        "answer_start": array([164], dtype=int32),
        "text": array(["mehrere Male"], dtype=object)
    }
}
```

```json
{
    "context": "Trotz ihrer weichen, gallertartigen Körper wurden Fossilien in Lagerstätten gefunden, die bis ins frühe Kambrium vor etwa 515 Millionen Jahren zurückreichen und von denen man annimmt, dass sie Rippenquallen darstellen. Die Fossilien verfügen nicht über Tentakel, haben aber viel mehr Kammreihen als moderne Formen. Die Position der Rippenquallen im evolutionären Stammbaum der Tiere ist seit langem umstritten. Die heutigen Mehrheitsmeinung, die auf der molekularen Phylogenese basiert, geht davon aus, dass Nesseltiere und Bilateria enger miteinander verwandt sind als die Rippenquallen selbst. Eine kürzlich durchgeführte Analyse der molekularen Phylogenese kam zu dem Schluss, dass der gemeinsame Vorfahre aller modernen Rippenquallen cydippida-ähnlich war und alle modernen Gruppen erst relativ spät auftauchten, wahrscheinlich nach der Kreide-Paläogen-Grenze vor 66 Millionen Jahren. Die seit den 1980er Jahren ansammelnden Beweise deuten darauf hin, dass „Cydippida“ nicht monophyletisch sind. Mit anderen Worten, sie beinhalten nicht alle Nachkommen, sondern nur die Nachkommen eines einzigen gemeinsamen Vorfahren. Dies wird angenommen, da alle anderen traditionellen Gruppen von Rippenquallen Nachkommen verschiedener Cydippida sind.",
    "question": "Was fehlte den Rippenquallen-Fossilien, über das heutige Rippenquallen verfügen?",
    "answers": {
        "answer_start": array([253], dtype=int32),
        "text": array(["Tentakel"], dtype=object)
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Im Folgenden finden Sie Texte mit den dazugehörigen Fragen und Antworten.
  ```

- Base prompt template:

  ```text
  Text: {text}
  Fragen: {question}
  Fragen Antwort in maximal 3 Wörtern: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Text: {text}

  Beantworten Sie die folgende Frage zum obigen Text in höchstens 3 Wörtern.

  Frage: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset xquad-de
```

### Unofficial: BeleBele-de

This dataset was published in [this paper](https://aclanthology.org/2024.acl-long.44/)
and features multiple-choice reading comprehension questions across 122 languages. The
dataset was created by professional translators who translated 900 multiple-choice
questions from English into other languages, with answers carefully validated by native
speakers.

The original dataset contains 900 unique multiple-choice reading comprehension passages
and questions. From these, we use a 256 / 64 / 580 split for training, validation and
testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Text: Es gibt viele Dinge, die Sie vor und während einer Reise berücksichtigen müssen. Erwarten Sie nicht, dass die Dinge beim Reisen genau so sind wie „zuhause“. Umgangsformen, Gesetze, Essen, Verkehr, Unterkünfte, Standards, Spache und so weiter werden zu einem gewissen Grad anders sein als dort, wo Sie leben. Dies ist etwas, was man immer im Hinterkopf behalten sollte, um Enttäuschung oder gar Abneigung über lokale Vorgehensweisen zu vermeiden.\nFragen: Was kann Reisenden dem Abschnitt nach helfen, Enttäuschung beim Besuch neuer Orte zu vermeiden?\nAntwortmöglichkeiten:\na. Ähnliche Standards wie zuhause erwarten\nb. Essen probieren, das ungewohnt ist\nc. Die gleichen Gesetze wie zuhause einhalten\nd. Nicht vorher nach Unterkünften recherchieren",
  "label": "b"
}
```

```json
{
  "text": "Text: Genehmigungen müssen im Voraus bestellt werden. Sie benötigen eine Genehmigung, um in La Sirena zu übernachten. Sirena ist die einzige Rangerstation, die neben Zelten auch Übernachtung im Schlafsaal und warme Mahlzeiten anbietet. La Leona, San Pedrillo und Los Patos bieten nur Camping ohne Verpflegung an. Es ist möglich, eine Parklizenz direkt bei der Rangerstation in Puerto Jiménez zu bekommen, aber sie akzeptieren keine Kreditkarten Die Parkverwaltung (MINAE) stellt Genehmigungen  für den Park nicht früher als einen Monat vor der geplanten Ankunft aus. CafeNet El Sol bietet einen Reservierungsservice gegen eine Gebühr von 30 US-Dollar bzw. 10 US-Dollar für Tageskarten an. Einzelheiten dazu findet man auf deren Corcovado-Seite.\nFragen: Welche der folgenden Rangerstationen bietet zwei Übernachtungsmöglichkeiten an?\nAntwortmöglichkeiten:\na. Sirena\nb. Los Patos\nc. La Leona\nd. San Pedrillo",
  "label": "a"
}
```

```json
{
  "text": "Text: Naturnaher Tourismus zieht Leute an, die daran interessiert sind, Naturgebiete zu besuchen, um die Landschaft zu genießen, einschließlich der wilden Pflanzen und Tiere. Beispiele für Aktivitäten vor Ort sind Jagen, Angeln, Fotografie, Vogelbeobachtung, der Besuch von Parks und das Lernen von Informationen über das Ökosystem. Ein Beispiel dafür ist der Besuch, das Fotografieren und das Studieren von Orangutangs in Borneo.\nFragen: Welche der folgenden Aktivitäten ist kein Beispiel für naturnahen Tourismus?\nAntwortmöglichkeiten:\na. Wandern zu einem Wasserfall\nb. Fotografieren von Wildblumen\nc. Besuch eines Wissenschaftsmuseum\nd. Fliegenfischen",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Die folgenden Fragen sind Multiple-Choice-Fragen (mit Antworten).
  ```

- Base prompt template:

  ```text
  Frage: {text}
  Antwort: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Frage: {text}
  Antwortmöglichkeiten:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Beantworten Sie die obige Frage mit 'a', 'b', 'c' oder 'd', und nichts anderes.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset belebele-de
```

### Unofficial: MultiWikiQA-de

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
    "context": "Claire Patricia Grogan (* 17. März 1962 in Glasgow, Schottland) ist eine britische Schauspielerin, Popsängerin sowie Kinder- und Jugendbuchautorin.\n\nTrotz abweichender Schreibweise ist sie seit Beginn ihrer Karriere unter dem Namen Clare Grogan bekannt. Im Fernsehen trat sie später als C.P. Grogan auf, da es in der britischen Künstlergewerkschaft Equity eine andere Person gleichen Namens gab.\n\nLeben \nClare Grogan wurde vom schottischen Filmregisseur Bill Forsyth in Glasgow entdeckt, wo sie in einem Restaurant als Kellnerin arbeitete. Im Alter von 19 Jahren spielte sie die Rolle der Susan im Spielfilm Gregory’s Girl. Zu diesem Zeitpunkt feierte sie bereits als Sängerin der New-Wave-Band Altered Images erste Erfolge. Mit Titeln wie Happy Birthday und I Could Be Happy wurde die Band Anfang der 1980er-Jahre auch außerhalb Großbritanniens bekannt. Sie löste sich 1984 nach der Produktion des dritten Albums aufgrund nachlassenden Publikumszuspruchs auf.\n\n1987 startete Grogan den Versuch einer Solokarriere, hatte mit ihrer Single Love Bomb jedoch keinen Erfolg. Auch ihr Album Trash Mad wurde nie veröffentlicht. Musikalisch trat sie danach nur noch selten in Erscheinung. 1993 war sie an der Produktion des Musikvideos Young at Heart der Gruppe Bluebells beteiligt. Der Titel stand vier Wochen lang auf dem ersten Platz der britischen Singlecharts. 2000 steuerte sie den Gesang im Song Night Falls Like A Grand Piano aus dem Album Hyacinths and Thistles der Band The 6ths bei. Im zwei Jahre später veröffentlichten The Ultimate Celtic Album ist sie mit dem Stück Her Hooped Dream vertreten. Für das 2003 erschienene Album A Tribute to Frankie Miller nahm sie eine neue Version von Angels With Dirty Faces auf. Nach einer 18-jährigen Pause trat sie in den 2000er-Jahren mit wechselnden Musikern mehrmals bei der Here and Now Tour, beim Rewind Festival sowie ähnlichen Revival-Veranstaltungen in Großbritannien und Irland unter dem Namen Altered Images auf.\n\nIm Jahr 1985 setzte sie ihre zweite Karriere als Schauspielerin mit einer kleinen Rolle als Empfangsdame in der sechsteiligen BBC-Produktion Blott on the Landscape fort. In der Science-Fiction-Fernsehserie Red Dwarf spielte sie die Kristine Kochanski, wurde später aber durch die Schauspielerin Chloë Annett ersetzt. Weitere Auftritte in den Serien Father Ted und EastEnders sowie in den britischen Spielfilmen Bury It und The Penalty King folgten. Grogan war auch Moderatorin im Musiksender VH1 und Gastgeberin einer Talkshow. Zuweilen half sie als Sprecherin beim Radiosender BBC Radio 6 Music aus.\n\nAls Autorin debütierte Grogan im Oktober 2008 mit dem Kinderbuch Tallulah and the Teenstars. Es erzählt die Geschichte einer Schülerin, die eine Popband gründet und den aufkommenden Erfolg bewältigen muss. Ende 2011 erschien eine Fortsetzung mit dem Titel Tallulah on Tour.\n\n1994 heiratete Grogan den Produzenten Steve Lironi, früher selbst Gitarrist und Schlagzeuger der Altered Images. Das Paar adoptierte 2005 ein Mädchen und lebt im Londoner Stadtbezirk London Borough of Haringey.\n\nWerke\n\nKinofilme und Fernsehproduktionen \n 1980: Gregory’s Girl\n 1984: Comfort and Joy\n 1985: Blott on the Landscape (britische Fernsehserie)\n 1988: Red Dwarf (britische Fernsehserie), Episoden The End, Balance of Power und Stasis Leak\n 1993: Red Dwarf, Episode Psirens\n 1996: Father Ted (britische Fernsehserie), Episode Rock-a-Hula Ted\n 1997: Jilting Joe\n 1997: EastEnders (britische Fernsehserie), zwei Episoden\n 2002: Bury It\n 2006: The Penalty King\n 2007: Legit (britische Fernsehserie), Episoden Birthday, Manitoba und Night of the Lobster\n 2011: Skins – Hautnah, Episode Mini\n 2012: Waterloo Road (britische Fernsehserie), Episode Future Proof\n\nBücher \n 2008: Tallulah and the Teenstars\n 2011: Tallulah on Tour\n\nWeblinks\n\nEinzelnachweise \n\nFilmschauspieler\nAutor\nPopsänger\nLiteratur (21. Jahrhundert)\nLiteratur (Englisch)\nKinder- und Jugendliteratur\nMusiker (Vereinigtes Königreich)\nPerson (Glasgow)\nSchotte\nBrite\nGeboren 1962\nFrau",
    "question": "Was war Clare Grogans Tätigkeit, bevor sie von Bill Forsyth entdeckt wurde?",
    "answers": {
        "answer_start": array([519]),
        "text": array(["Kellnerin"], dtype=object)
    }
}
```

```json
{
    "context": "Claris International Inc. (bis August 2019 FileMaker, Inc.) ist eine hundertprozentige US-amerikanische Tochtergesellschaft des kalifornischen Computerherstellers Apple, die die Datenbanksoftware FileMaker entwickelt. Die Firma FileMaker entstand 1998 als Nachfolgerin von Claris, die ihrerseits 1987 als Ableger von Apple gegründet worden war.\n\nGeschichte \nClaris wurde Anfang 1998 aufgelöst. Das Programm FileMaker Pro wurde Grundlage des neu gegründeten Unternehmens FileMaker, Inc.\n\nProdukte von Claris waren:\n ClarisCAD, ein CAD-Programm\n Claris MacDraw, ein Zeichenprogramm\n Claris Em@iler, ein E-Mail-Programm\n FileMaker, später FileMaker Pro, das dominierende Datenbankprogramm auf der Macintosh-Plattform\n Claris Home Page, ein HTML-Editor\n Claris Impact, ein Präsentationsprogramm\n Claris MacWrite Pro, eine Textverarbeitung\n Claris Organizer, ein Personal Information Manager\n Claris Resolve, eine Tabellenkalkulation\n ClarisWorks, ein Büropaket, das später von Apple als AppleWorks weitergeführt wurde\n Claris MacPaint, ein Bildbearbeitungsprogramm\n\nVon 2008 bis 2013 wurde die persönliche Datenbankanwendung Bento verkauft.\n\nIm August 2019 gab das Unternehmen bekannt, zum alten Unternehmensnamen Claris zurückzukehren.\n\nEinzelnachweise \n\nApple\nSoftwarehersteller (Vereinigte Staaten)\nUnternehmen (Santa Clara, Kalifornien)\nGegründet 1998",
    "question": "Unter welchem Namen war FileMaker, Inc. früher bekannt, bevor es in Claris International Inc. umbenannt wurde?",
    "answers": {
        "answer_start": array([31]),
        "text": array(["August 2019"], dtype=object)
    }
}
```

```json
{
    "context": "Augusta Marie Gertrude von Hanau (* 21. September 1829 in Niederdorfelden; † 18. September 1887 in Halle) war die unehelich geborene älteste Tochter des Kurfürsten Friedrich Wilhelm I. von Hessen-Kassel (1802–1875) und seiner erst späteren Ehefrau Gertrude, spätere Fürstin von Hanau und zu Hořowitz (1803–1882).\n\nKurprinz Friedrich Wilhelm lernte seine Frau kennen, als diese noch mit dem Leutnant Karl Michael Lehmann (1787–1882) verheiratet war, beging mit ihr Ehebruch, erreichte schließlich die Scheidung und heiratete sie 1831. Augusta Marie Gertrude wurde so zu einer Zeit geboren, als ihre Mutter noch eine verheiratete Lehmann war. Sie wurde deshalb zunächst vom damaligen Mann ihrer Mutter als ehelich anerkannt. Erst nach der Scheidung und der Heirat von Gertrude Lehmann mit dem Kurprinzen verzichtete Karl Michael Lehmann auf die Vaterschaftsrechte. Augusta Marie Gertrude Lehmann wurde nun von ihrem leiblichen Vater zur Gräfin Schaumburg und später zur Prinzessin von Hanau erhoben.\n\nAm 17. Juli 1849 heiratete sie den Grafen Ferdinand Maximilian zu Ysenburg-Büdingen (* 24. Oktober 1823; † 5. Mai 1903). Dieser war mental wohl etwas gestört. Nachdem eine Kasseler Zeitung 1853 seine Frau „Erlaucht“ statt „Durchlaucht“ betitelt hatte, griff er den Ersten Minister seines Schwiegervaters, Ludwig Hassenpflug, tätlich an und verletzte ihn mit Stockschlägen. Er kam darauf vorübergehend in eine Klinik. 1865 wurde er durch den Kurfürsten in den Fürstenstand erhoben und nannte sich nun Ferdinand-Maximillian I.\n\nFürstin Augusta Marie Gertrude hatte ein sehr enges Verhältnis zu ihrem Vater. Als er 1866 nach dem gegen Preußen verlorenen Krieg in Stettin als Kriegsgefangener einsaß, besuchte sie ihn.\n\nSie starb in Halle, wohin sie ihren Mann begleitet hatte, der sich dort einer Operation unterziehen musste.\n\nLiteratur \n Rüdiger Ham: Ludwig Hassenpflug: Staatsmann und Jurist zwischen Revolution und Reaktion. Eine politische Biographie = Studien zur Geschichtsforschung der Neuzeit 50. Hamburg 2007. ISBN 978-3-8300-2764-5\nMichel Huberty: L' Allemagne dynastique: Les 15 familles qui ont fait l'empire. Bd. 1: Hesse - Reuss - Saxe. Le Perreux-sur-Marne 1976. ISBN 2-901138-01-2\n Philipp Losch: Die Fürstin von Hanau und ihre Kinder. In: Hanauer Geschichtsblätter 13 (1939), S. 33.\n\nWeblinks\n\nEinzelnachweise \n\nFriedrich Wilhelm I. (Hessen-Kassel)\nTitularfürst (Isenburg)\nFamilienmitglied des Hauses Hanau-Hořovice\n⚭Augusta Marie Gertrude #Hanau\nDeutscher\nGeboren 1829\nGestorben 1887\nFrau",
    "question": "Wann wurde Ferdinand Maximilian von Ysenburg-Büdingen Fürst?",
    "answers": {
        "answer_start": array([1416]),
        "text": array(["1865"], dtype=object)
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Im Folgenden finden Sie Texte mit den dazugehörigen Fragen und Antworten.
  ```

- Base prompt template:

  ```text
  Text: {text}
  Fragen: {question}
  Fragen Antwort in maximal 3 Wörtern: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Text: {text}

  Beantworten Sie die folgende Frage zum obigen Text in höchstens 3 Wörtern.

  Frage: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-de
```

## Knowledge

### INCLUDE-de

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
  "text": "Wann dürfen Sie in einem Tunnel Ihr Fahrzeug wenden?\nAntwortmöglichkeiten:\na. Wenn Einsatzkräfte das Wenden ausdrücklich anordnen\nb. Wenn ich aus einer Gefahrensituation flüchten möchte\nc. Wenn ich unter Zeitdruck bin und sich vor mir ein Stau gebildet hat\nd. Nur wenn ich mit meinem Fahrzeug in einem Zug umkehren kann",
  "label": "a",
  "subject": "Driving License"
}
```

```json
{
  "text": "Das Industrieland hat in einer Wirtschaftstätigkeit einen komparativen Vorteil, wenn\nAntwortmöglichkeiten:\na. in einer anderen Tätigkeit sein absoluter Vorteil größer ist.\nb. in dieser Tätigkeit sein absoluter Vorteil am größten ist.\nc. es keinen absoluten Vorteil hat.\nd. in dieser Tätigkeit sein absoluter Nachteil am geringsten ist.",
  "label": "b",
  "subject": "Economics"
}
```

```json
{
  "text": "Ein Schiff fährt mit einer geradlinigen, gleichförmigen Bewegung auf dem offenen Meer. Zu gleicher Zeit fliegt auch ein Albatros mit einer in Bezug auf das Meer geradlinigen, gleichförmigen Bewegung in der Luft. Wie bewegt sich der Albatros in Bezug auf das Schiff?\nAntwortmöglichkeiten:\na. Die Bahn des Vogels ist geradlinig, aber seine Geschwindigkeit in Bezug auf das Schiff ist nicht konstant.\nb. Abhängig vom Winkel der zwei Geschwindigkeitsvektoren kann die Bahn des Vogels sowohl krummlinig, als auch geradlinig sein und auch seine Geschwindigkeit in Bezug auf das Schiff kann veränderlich sein.\nc. Der Vogel führt in Bezug auf das Schiff eine gleichförmige, geradlinige Bewegung aus.\nd. In bestimmten Fällen kann die Bahn des Vogels in Bezug auf das Schiff auch krummlinig sein, aber seine Geschwindigkeit hat einen konstanten Betrag.",
  "label": "c",
  "subject": "Physics"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Die folgenden Fragen sind Multiple-Choice-Fragen (mit Antworten).
  ```

- Base prompt template:

  ```text
  Frage: {text}
  Antwort: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Frage: {text}

  Beantworten Sie die obige Frage mit {labels_str}, und nichts anderes.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset include-de
```

### MultiLoKo-de

This dataset was published in [this paper](https://arxiv.org/abs/2504.10356) and is part
of MultiLoKo, a multilingual local knowledge benchmark covering 31 languages. The German
questions are separately sourced and designed to target locally relevant topics for
German-speaking populations.

We use the 'dev' split (250 samples) from this dataset. The dataset contains open-ended
questions with correct answers in the 'targets' column. We use the first target answer
as the correct option and use GPT-4.1 to generate 3 plausible but incorrect alternatives
per question. We create a 16 / 234 split for training and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "In welchem Bereich sammelte Reiner Calmund erste berufliche und sportliche Erfahrungen, die später für seine Tätigkeit als Manager bei Bayer 04 Leverkusen relevant wurden?\nAntwortmöglichkeiten:\na. Trainer\nb. Spielerberater\nc. Physiotherapeut\nd. Sportjournalist",
  "label": "a"
}
```

```json
{
  "text": "Wie lange war Malu Dreyer Mitglied der SPD, als sie zum ersten Mal Landespräsidentin für den Wahlkreis Trier wurde?\nAntwortmöglichkeiten:\na. vierzehn Jahre\nb. elf Jahre\nc. sieben Jahre\nd. zwanzig Jahre",
  "label": "b"
}
```

```json
{
  "text": "Wie lautet der Name des Jungen, der in der vierten Staffel der Fernsehserie „Bettys Diagnose“ zu sehen ist und der der Hauptfigur in der Schule das Leben schwer macht?\nAntwortmöglichkeiten:\na. Tim Weigel\nb. Lukas Kramer\nc. Frank Stern\nd. Jonas Becker",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Die folgenden Fragen sind Multiple-Choice-Fragen (mit Antworten).
  ```

- Base prompt template:

  ```text
  Frage: {text}
  Antwort: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Frage: {text}

  Beantworten Sie die obige Frage mit 'a', 'b', 'c' oder 'd', und nichts anderes.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multiloko-de
```

### Unofficial: MMLU-de

This dataset is a machine translated version of the English
[MMLU dataset](https://openreview.net/forum?id=d7KBjmI3GmQ) and features questions
within 57 different topics, such as elementary mathematics, US history and law. The
translation to German was done by the University of Oregon as part of
[this paper](https://aclanthology.org/2023.emnlp-demo.28/), using GPT-3.5-turbo.

The original full dataset consists of 269 / 1,410 / 13,200 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 2,048 split for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
new and there can thus be some overlap between the original validation and test sets and
our validation and test sets.

Here are a few examples from the training split:

```json
{
  "text": "Teotihuacán wurde im Becken von Mexiko bekannt, nachdem sein Rivale Cuicuilco,\nAntwortmöglichkeiten:\na. von einem Vulkanausbruch gelähmt wurde.\nb. einem Bürgerkrieg unter seinen herrschenden Familien erlag.\nc. unter einer Ernteplage litt.\nd. von einem Hurrikan an der Golfküste überschwemmt wurde.",
  "label": "a"
}
```

```json
{
  "text": "Wer von den folgenden ist der industrielle Philanthrop?\nAntwortmöglichkeiten:\na. Frederick Taylor\nb. Seebohm Rowntree\nc. Henry Ford\nd. Max Weber",
  "label": "b"
}
```

```json
{
  "text": "Verglichen mit der Varianz der Maximum-Likelihood-Schätzung (MLE) ist die Varianz der Maximum-A-Posteriori (MAP)-Schätzung ________\nAntwortmöglichkeiten:\na. höher\nb. gleich\nc. niedriger\nd. es kann jede der obigen Optionen sein",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Die folgenden Fragen sind Multiple-Choice-Fragen (mit Antworten).
  ```

- Base prompt template:

  ```text
  Frage: {text}
  Antwortmöglichkeiten:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Antwort: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Frage: {text}
  Antwortmöglichkeiten:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Beantworten Sie die obige Frage mit 'a', 'b', 'c' oder 'd', und nichts anderes.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset mmlu-de
```

### Unofficial: ARC-de

This dataset is a machine translated version of the English
[ARC dataset](https://doi.org/10.48550/arXiv.1803.05457) and features US grade-school
science questions. The translation to German was done by the University of Oregon as
part of [this paper](https://aclanthology.org/2023.emnlp-demo.28/), using GPT-3.5-turbo.

The original full dataset consists of 1,110 / 297 / 1,170 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 1,024 split for training,
validation and testing, respectively (so 2,304 samples used in total). All new splits
are subsets of the original splits.

Here are a few examples from the training split:

```json
{
  "text": "Callahan zitiert die Ergebnisse des Oregon Death with Dignity Legal Defense and Education Center, wonach es \"nach vier vollen Jahren keine Missteps, Missbräuche oder Zwangstendenzen\" bezüglich der Gesetze zur Euthanasie gab. Er argumentiert dagegen, dass\nAntwortmöglichkeiten:\na. sie dies ohne eine anonyme Umfrage nicht sicher wissen können.\nb. andere Studien haben widersprüchliche Ergebnisse gefunden.\nc. selbst wenn das Ergebnis wahr ist, ist es irrelevant für den moralischen Status der Euthanasie.\nd. die Ergebnisse sind verdächtig, weil die Studie von Befürwortern der Euthanasie durchgeführt wurde.",
  "label": "a"
}
```

```json
{
  "text": "Eine Frau besaß ein Land im absoluten Besitz. Die Frau übertrug das Land an einen Freund “auf Lebenszeit” und als der Freund starb, sollte das Land an den Nachbarn der Frau \"und ihre Erben\" weitergegeben werden. Der Nachbar starb und in ihrem ordnungsgemäß beglaubigten Testament vermachte sie ihre gesamte Hinterlassenschaft an eine örtliche Wohltätigkeitsorganisation. Wenn sie intestat gestorben wäre, wäre ihre Tochter ihre einzige Erbin gewesen. Ein Jahr nach dem Tod des Nachbarn übertrug dessen Tochter durch eine Quitclaims-Erklärung jegliches Interesse, das sie an dem Land haben könnte, an die Freundin der Frau. Das Common Law Rule Against Perpetuities ist in der Gerichtsbarkeit unverändert. Es gibt keine weiteren anwendbaren Gesetze. Wer hat welches Interesse an dem Land?\nAntwortmöglichkeiten:\na. Der Freund hat einen absoluten Besitz, weil sein Lebensbesitz mit dem verbleibenden Teil verschmolzen ist, den ihm die Tochter übertragen hat.\nb. Der Freund hat einen Lebensbesitz und die Wohltätigkeitsorganisation hat einen unverfallbaren Rest, weil das Interesse des Nachbarn übertragbar war.\nc. Der Freund hat einen Lebensbesitz und die Tochter hat einen unverfallbaren Rest, weil die Erklärung der Frau ein Interesse an den Erben des Nachbarn schuf.\nd. Der Freund hat einen Lebensbesitz und die Frau hat einen Rückfall, weil das Restinteresse des Nachbarn gemäß dem Rule Against Perpetuities ungültig war.",
  "label": "b"
}
```

```json
{
  "text": "Für welches dieser beiden Szenarien tut der Hauptcharakter (der ich/mich/mein verwendet) etwas, das nach den gewöhnlichen moralischen Standards in den USA im Jahr 2020 eindeutig moralisch falsch ist? Szenario 1 | Ich gehe gerne in die Kirche, weil es mich in höchstem Maße Gott nahe fühlen lässt. Szenario 2 | Ich habe den Test meines Nachbarn angesehen, um eine gute Note zu bekommen.\nAntwortmöglichkeiten:\na. Falsch, Falsch\nb. Falsch, Nicht falsch\nc. Nicht falsch, Falsch\nd. Nicht falsch, Nicht falsch",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Die folgenden Fragen sind Multiple-Choice-Fragen (mit Antworten).
  ```

- Base prompt template:

  ```text
  Frage: {text}
  Antwortmöglichkeiten:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Antwort: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Frage: {text}
  Antwortmöglichkeiten:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Beantworten Sie die obige Frage mit 'a', 'b', 'c' oder 'd', und nichts anderes.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset arc-de
```

## Common-sense Reasoning

### HellaSwag-de

This dataset is a machine translated version of the English
[HellaSwag dataset](https://aclanthology.org/P19-1472/). The original dataset was based
on both video descriptions from ActivityNet as well as how-to articles from WikiHow. The
dataset was translated by the University of Oregon as part of
[this paper](https://aclanthology.org/2023.emnlp-demo.28/), using GPT-3.5-turbo.

The original full dataset consists of 9,310 samples. We use a 1,024 / 256 / 2,048 split
for training, validation and testing, respectively (so 3,328 samples used in total).

Here are a few examples from the training split:

```json
{
  "text": "[header] Wie man sich trennt, wenn Kinder involviert sind [title] Erstellen Sie einen Trennungsplan mit Ihrem Partner. [step] Sie sollten sich auch auf das Gespräch mit Ihren Kindern vorbereiten, indem Sie vorher mit Ihrem Partner einen Plan für die Zukunft erstellen. Sie sollten gemeinsam besprechen, wer wo leben wird, wer für bestimmte tägliche Bedürfnisse und Aktivitäten der Kinder verantwortlich sein wird und wann der offizielle Scheidungsprozess beginnen wird.\nAntwortmöglichkeiten:\na. Indem Sie hierüber klare Vorstellungen haben, können Sie Ihre Kinder besser beruhigen und einheitlich auftreten. [substeps] Zum Beispiel, könnten Sie vereinbaren, dass Ihr Partner auszieht und in einer nahegelegenen Wohnung oder einem anderen Haus lebt.\nb. Sie beide sollten Ihre Aktionen in den Monaten bis zur Eheschließung sowie darüber, wie Sie alles tun werden, planen, sobald das Kind wieder mit seinem Vater vereint ist. [title] Entscheiden Sie, was Sie mit dem Kind machen werden.\nc. Stellen Sie sicher, dass Ihr Partner einverstanden ist und zustimmt, immer Pausen zu machen. [substeps] Sie sollten sich nun auf die Urlaubsdaten und Reisepläne einigen, zu denen Ihre Kinder gehen werden.\nd. Der erste Schritt zu diesem Plan ist, ein Telefongespräch zu vereinbaren, damit Sie mit Ihrem Partner persönlich sprechen können. Sprechen Sie ruhig und deutlich, um den Ton für dieses Gespräch zu setzen.",
  "label": "a"
}
```

```json
{
  "text": "[header] Wie man Festival-Make-up macht [title] Bereiten Sie Ihr Gesicht vor. [step] Bevor Sie Ihr Augen-Make-up auftragen, müssen Sie eine Basis schaffen. Dies hilft sicherzustellen, dass Ihr Augen-Make-up den ganzen Tag hält.\nAntwortmöglichkeiten:\na. [substeps] Zeichnen Sie eine runde, quadratische oder diagonale Linie um Ihr Auge. Verfolgen Sie den Kreis um Ihr Auge und ziehen Sie dann einen rechteckigen Streifen in der Mitte.\nb. [substeps] Beginnen Sie mit einem sauberen, mit Feuchtigkeit versorgten Gesicht. Reinigen Sie Ihr Gesicht zunächst mit einem sanften Reinigungsmittel und tragen Sie dann einen leichten Feuchtigkeitsspender auf Ihr Gesicht und Ihren Hals auf, um das Erscheinungsbild feiner Linien zu reduzieren.\nc. Bevor Sie Lidschatten auftragen, wählen Sie einen einzelnen Lidschatten aus und messen Sie ihn so aus, dass er etwas größer ist als das Auge, das Sie verblenden möchten. Tragen Sie den Lidschatten auf die Spitze jedes Auges auf und streichen Sie mit einem Verblendpinsel darüber.\nd. Make-up am frühen Morgen zu tragen ist nicht immer eine Option, aber Sie können es am Abend tun. [substeps] Duschen Sie, um Ihre Haut sauber und mit Feuchtigkeit versorgt zu halten.",
  "label": "b"
}
```

```json
{
  "text": "Wir sehen einen Mann in einem Orchester Grimassen schneiden. Der Mann steht dann auf und spielt die Violine. Wir sehen Menschen an Spinden. wir\nAntwortmöglichkeiten:\na. sehen Menschen in einem Bus.\nb. sehen Menschen beim Üben von Kampfsport und Musik spielen.\nc. kehren zum Mann zurück, der die Violine spielt.\nd. sehen den Mann am Keyboard wieder.",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Die folgenden Fragen sind Multiple-Choice-Fragen (mit Antworten).
  ```

- Base prompt template:

  ```text
  Frage: {text}
  Antwortmöglichkeiten:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Antwort: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Frage: {text}
  Antwortmöglichkeiten:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Beantworten Sie die obige Frage mit 'a', 'b', 'c' oder 'd', und nichts anderes.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset hellaswag-de
```

### Unofficial: GoldenSwag-de

This dataset is a filtered and machine translated version of the English
[HellaSwag dataset](https://aclanthology.org/P19-1472/), featuring both video
descriptions from ActivityNet as well as how-to articles from WikiHow. The machine
translated version was published in
[this paper](https://doi.org/10.48550/arXiv.2410.08928) and was done using DeepL, and
the filtering was published in [this paper](https://doi.org/10.48550/arXiv.2504.07825),
which resulted in higher quality samples.

The original full dataset consists of 1530 / 1530 samples for training and validation,
respectively. However, they are exactly equal. We use a split of 660 / 256 / 2,048
samples for training, validation, and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Wie man Rouge aufträgt. Verwenden Sie die richtige Art von Pinsel. Die Art des Pinsels hängt davon ab, wo Sie das Rouge auftragen wollen. Da Sie das Rouge nicht nur auf die Wangenäpfel auftragen werden, sollten Sie für kleinere Bereiche einen kleineren Pinsel verwenden.\nAntwortmöglichkeiten:\na. Für die Wangen können Sie einen normalen Rougepinsel verwenden. Manche empfehlen, für die kleineren Gesichtspartien einen Abdeckpinsel zu verwenden.\nb. Je kleiner der Pinsel ist, desto mehr Rouge müssen Sie auf Ihre Wangen auftragen. Überprüfen Sie auf der Verpackung die richtige Pinselgröße für diesen Bereich.\nc. Wählen Sie den Pinsel, der am besten zu Ihrem Haartyp passt. Bei lockigem Haar verwenden Sie einen größeren Pinsel für dünneres Haar und einen kleineren Pinsel für dünnes Haar.\nd. Für größere Flächen können Sie einen Tubenpinsel oder einen Pinsel in einer anderen Farbe verwenden, um ein Zusammenfallen zu vermeiden. Verwenden Sie einen Pinsel mit Borsten in der Farbe Ihrer Grundierung, damit die abgerundeten Borsten weniger auffallen.",
  "label": "a"
}
```

```json
{
  "text": "Wie Sie einen Redakteur auf sich aufmerksam machen können. Lesen und befolgen Sie die Einreichungsrichtlinien der Publikation. Publikationen erstellen Einreichungsrichtlinien, um es sowohl den Autoren als auch den Redakteuren leichter zu machen. Wenn Sie die Richtlinien lesen und befolgen, erstellen Sie einen Beitrag, der den Anforderungen der Publikation entspricht, was es für Sie als Autor einfacher macht, und zwar in einem Format, das die Redakteure leichter auf Eignung und Qualität prüfen können.\nAntwortmöglichkeiten:\na. Vermeiden Sie es, den Namen und die Veröffentlichungsseite der Publikation vollständig zu blockieren. Wenn die Publikation nicht sehr künstlerisch ist, wird sie vielleicht gar nicht veröffentlicht.\nb. Vergewissern Sie sich, dass Ihr Artikel diesen Richtlinien entspricht, wenn Sie sich um eine Stelle als Redakteur bewerben. Bei einigen Stellen müssen Sie eine bestimmte Menge an Arbeit leisten, um eine Redakteursstelle zu erhalten, während bei anderen ein Minimum von 30 Arbeitsstunden erforderlich ist.\nc. Die meisten Publikationen mit Internetpräsenz bieten ihre Richtlinien für die Einreichung von Beiträgen auf ihren Websites an. Wenn dies nicht der Fall ist, können Sie die Richtlinien erhalten, indem Sie an die angegebene Adresse der Publikation schreiben.\nd. Bitten Sie die Autoren am Ende der Veröffentlichung, Ihre Arbeit regelmäßig zu veröffentlichen. Heben Sie in Ihrem Beitrag wichtige Aspekte hervor, damit Sie nicht von der Publikation ausgeschlossen werden.",
  "label": "c"
}
```

```json
{
  "text": "Wie Sie Hundegeruch aus Ihrem Auto entfernen. Waschen Sie alle abnehmbaren Teile Ihres Autos. Alle Teile Ihres Autos, die Sie abnehmen können, sollten Sie in der Waschmaschine waschen. Dadurch wird der Hundegeruch entfernt und Ihr Auto riecht wieder frischer.\nAntwortmöglichkeiten:\na. Wenn Sie feststellen, dass Ihr Auto nach Ihnen riecht, wenn Sie es ausstecken, sollten Sie die Teile 5 Minuten in warmem Wasser und 20 Minuten in kaltem Wasser einweichen. Wenn Sie ein Straßenfest veranstalten, verwenden Sie einen Trichter, um Plastikteile in die Waschmaschine zu befördern, während die äußeren Teile weggeworfen werden.\nb. Gummimatten, Autositzbezüge und alle Decken, die Sie für Ihren Hund aufbewahren, können entfernt und gewaschen werden. Waschen Sie die Teile Ihres Autos sicherheitshalber bei einer kühlen Temperatur.\nc. Eine Schicht Antitranspirant hingegen entfernt nur das Produkt, und Ihr Auto riecht wahrscheinlich schon nach Urin. Wenn Ihr Auto mit Ledersitzen ausgestattet ist, wischen Sie das Produkt, das sich dort angesammelt hat, ab.\nd. Am sichersten ist es, alle abnehmbaren Teile Ihres Autos zu entfernen, einschließlich der "Fifflers". Diese Teile können bei heißem Wetter leicht stinken, aber sie können auch schwitzen und den Eigengeruch Ihres Hundes produzieren.",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Die folgenden Fragen sind Multiple-Choice-Fragen (mit Antworten).
  ```

- Base prompt template:

  ```text
  Frage: {text}
  Antwortmöglichkeiten:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Antwort: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Frage: {text}
  Antwortmöglichkeiten:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Beantworten Sie die obige Frage mit 'a', 'b', 'c' oder 'd', und nichts anderes.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset goldenswag-de
```

### Unofficial: Winogrande-de

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2506.19468)
and is a translated and filtered version of the English
[Winogrande dataset](https://doi.org/10.1145/3474381).

The original full dataset consists of 47 / 1,210 samples for training and testing, and
we use 128 of the test samples for validation, resulting in a 47 / 128 / 1,085 split for
training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Einmal in Polen genoss Dennis die Reise mehr als Jason, weil _ ein tieferes Verständnis der polnischen Sprache hatte. Worauf bezieht sich der leere _?\nAntwortmöglichkeiten:\na. Dennis\nb. Jason",
  "label": "a"
}
```

```json
{
  "text": "Johns Zertifizierung war weniger wichtig als Jims Abschluss, weil die _ von einer unbedeutenden Universität war. Worauf bezieht sich der leere _?\nAntwortmöglichkeiten:\na. Zertifizierung\nb. Abschluss",
  "label": "a"
}
```

```json
{
  "text": "Um Verhaltensverzerrungen zu überwinden, müssen wir uns mehr darauf konzentrieren, die bewussten Handlungen zu ändern, anstatt die unbewussten Handlungen, weil die _ Handlungen freiwillig sind. Worauf bezieht sich der leere _?\nAntwortmöglichkeiten:\na. unbewusst\nb. bewusst",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Die folgenden Fragen sind Multiple-Choice-Fragen (mit Antworten).
  ```

- Base prompt template:

  ```text
  Frage: {text}
  Antwortmöglichkeiten:
  a. {option_a}
  b. {option_b}
  Antwort: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Frage: {text}
  Antwortmöglichkeiten:
  a. {option_a}
  b. {option_b}

  Beantworten Sie die obige Frage mit 'a' oder 'b', und nichts anderes.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset winogrande-de
```

## Summarisation

### MLSum-de

This dataset was published in
[this paper](https://aclanthology.org/2020.emnlp-main.647/) and features news articles
and their summaries in five languages, including German. The German part of the dataset
is based on news articles from Süddeutsche Zeitung, with human-written summaries.

The original full dataset consists of 221,000 / 11,400 / 10,700 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 2,048 split for training,
validation and testing, respectively (so 3,328 samples used in total). All new splits
are subsets of the original splits.

Here are a few examples from the training split:

```json
{
  "text": "Jede neue Schlagzeile ein Stich ins Herz: Führende Muslime beklagen in einem offenen Brief die wachsende \"Feindseligkeit\" gegen Migranten in Deutschland. Sie fordern Bundespräsident Wulff auf, Stellung zu beziehen. In einem offenen Brief haben 15 namhafte deutsche Muslime Bundespräsident Christian Wulff aufgefordert, in der schwelenden Debatte um Integrationsprobleme Stellung zu beziehen. Auslöser der Kontroverse war das Buch Deutschland schafft sich ab des SPD-Politikers und scheidenden Bundesbankvorstandes Thilo Sarrazin. Detailansicht öffnen In der von SPD-Politiker und Noch-Bundesbanker Thilo Sarrazin ausgelösten Integrationsdebatte fordern namhafte deutsche Muslime nun von Bundespräsident Christian Wulff, Stellung zu beziehen. (Foto: dpa) Intellektuelle wie der Regisseur Fatih Akin und der Schriftsteller Feridun Zaimoglu beklagten in dem in der taz veröffentlichten Brief wachsende \"Feindseligkeit\" gegen Muslime in Deutschland. Wörtlich heißt es: \"Für Musliminnen und Muslime ist derzeit nicht einmal der Gang zum Zeitungshändler leicht, weil sie nie wissen, welche Schlagzeile, welches stereotype Bild sie dort erwartet.\" Die Unterzeichner erinnerten Wulff an seine Antrittsrede, in der er die Chancen der Integration betont hatte. \"Wir bitten Sie, gerade in der derzeitigen angespannten Stimmung für die Leitsätze einer offenen, von gegenseitigem Respekt geprägten demokratischen Kultur einzustehen und öffentlich für sie zu werben\", heißt es in dem Appell an Wulff. Auslöser für den offenen Brief sei der Aufruf der Bild-Zeitung gewesen, an Präsident Wulff zu schreiben, sagte Shermin Langhoff, Intendantin des Berliner Theaters Ballhaus Naunynstraße. \"Wir dachten uns, das können wir nicht so stehen lassen\", sagte die Mitunterzeichnerin zur SZ. Sie sprach von \"biologistischen Wahnthesen\" Sarrazins und hofft auf ein \"Wort der Vernunft\" aus Bellevue. Auch andere Unterzeichnerinnen setzen darauf, dass sich das Staatsoberhaupt in die Debatte einschaltet. Aylin Selcuk, Initiatorin des Vereins Deukische Generation, wünscht sich ein starkes Zeichen Wulffs. Der Präsident möge zeigen, dass die Muslime in Deutschland dazugehören. \"Wir bitten Sie: Bekennen Sie sich zu uns.\" Lamya Kaddor vom Liberal-Islamischen Bund sprach von einem \"öffentlichen Bekenntnis\" des Präsidenten. In der laufenden Debatte gehe es nicht nur um Muslime, sondern um den \"Zusammenhalt in der Gesellschaft\", warnte Selcuk. Die Studentin hatte Sarrazin nach seinen Äußerungen zur vererbten Intelligenz wegen Volksverhetzung angezeigt. Seitdem erreichten sie unzählige E-Mails, in denen sie geschmäht und bedroht werde, sagte Selcuk. Nun hofft sie auf Wulff. \"Wir werden dieses Land nicht aufgeben\", heißt es in dem Brief an Christian Wulff. \"Dieses Land ist unsere Heimat und Sie sind unser Präsident.\"",
  "target_text": "Jede neue Schlagzeile ein Stich ins Herz: Führende Muslime beklagen in einem offenen Brief die wachsende \"Feindseligkeit\" gegen Migranten in Deutschland. Sie fordern Bundespräsident Wulff auf, Stellung zu beziehen."
}
```

```json
{
  "text": "Hoch flog der erste Schläger in die Luft, und viele andere Gegenstände folgten ihm. Überall auf dem Eis lag die Ausrüstung der deutschen Mannschaft zerstreut, Handschuhe, Helme, Schläger, weg damit, wer braucht so etwas schon, wenn er hemmungslos jubeln kann? In einer Ecke des Eises versammelten sich die Spieler der deutschen Eishockey-Mannschaft. Sie hüpften und tanzten und schrien, und wenn es nicht zu den Gepflogenheiten des Sports zählen würde, irgendwann zum Händeschütteln mit dem Gegner in der Mitte des Feldes zu erscheinen, dann hätten sie wahrscheinlich noch eine ganze Weile so weitergemacht. Es war nun wirklich ein sporthistorischer Moment, den das Team des Deutschen Eishockey-Bundes (DEB) dort zelebrierte. Mit 4:3 (1:0, 3:1, 0:2) hatte es in einem phänomenalen Spiel den Rekord-Olympiasieger Kanada bezwungen und sich damit für das Finale des Turniers gegen die Olympischen Athleten aus Russland (5.10 Uhr MEZ) qualifiziert. Zum ersten Mal überhaupt kann eine deutsche Mannschaft Olympiasieger werden, es ist der größte Erfolg in der Geschichte des deutschen Eishockeys. \"Verrückt, ne, verrückt, verrückte Welt\", sagte Bundestrainer Marco Sturm: \"Das ist einmalig.\" Ein ohnehin schon irres Turnier kulminiert in diesem 4:3 im Halbfinale Ja, einmalig war es in der Tat, was seine Mannschaft da geleistete hatte. Und es war interessant mitzuerleben, wie nach dem Spiel ein Akteur nach dem anderen in die Kabine trottete und sich unterwegs kurz den Journalisten stellte. Da war etwa der Torwart Danny aus den Birken, der völlig ausgelaugt war. Oder Defensivspieler Moritz Müller, der seine Tränen kaum halten konnte. Oder die NHL-gestählten Routiniers Christian Erhoff und Marcel Goc, die schon so viel erlebt haben, aber so etwas wie an diesem Abend dann doch noch nicht. Keiner hatte schon so recht begriffen, was da geschehen war, und keiner wollte zu großen sportfachlichen Analysen ansetzen, als es um die Gründe für den Erfolg ging. Ein jeder sagte nur: Team. Mannschaft. Teamgeist. Mannschaftsgeist. Diese Wörter fallen oft im Sport, aber soweit sich das von außen beurteilen lässt, trifft das bei den Eishockey-Spielern tatsächlich zu. Sturm hat in den drei Jahren eine bemerkenswerte Mannschaft geformt, die ohnehin ein irres Turnier spielt. Das knappe 0:1 gegen Schweden in der Vorrunde, der Penalty-Sieg über Norwegen, der Erfolg nach Verlängerung gegen die Schweiz, das denkwürdige 4:3 gegen Schweden im Viertelfinale. Aber all das kulminierte jetzt in diesem 4:3 gegen Kanada im Halbfinale. In einem \"Jahrhundertspiel\", wie Alfons Hörmann, Präsident des Deutschen Olympischen Sportbundes, nicht ganz zu Unrecht schwärmte.",
  "target_text": "Nach dem sensationellen 4:3-Sieg gegen Kanada kann das deutsche Eishockey-Team erstmals Olympiasieger werden. Im Finale ist der Gegner der Favorit - doch die Mannschaft von Marco Sturm glaubt an sich."
}
```

```json
{
  "text": "Monatelang haben Sicherheitsbehörden nach Salah Abdeslam gefahndet. Jetzt ist der 26-jährige Terrorverdächtige festgenommen worden. Er soll an den Anschlägen von Paris beteiligt gewesen sein, bei denen am 13. November drei Killerkommandos 130 Menschen getötet hatten. Was man bisher über den Mann weiß Salah Abdeslam ist in Brüssel geboren, aber französischer Staatsbürger. Er ist der Bruder des Selbstmordattentäters Brahim, der ebenfalls bei den Anschlägen dabei war. Die verstümmelte Leiche des 31-jährigen Brahim Abdeslam hatte die Polizei am Tag des Anschlags am Boulevard Voltaire in der Nähe des Konzertsaals Bataclan gefunden, wo er sich in die Luft gesprengt hatte. Salah wohnte im Brüsseler Vorort Molenbeek, der als eine Hochburg von gewaltbereiten Islamisten in Belgien gilt. Abdeslam soll in Deutschland gewesen sein Laut Recherchen des SWR soll sich Abdeslam Anfang Oktober 2015 kurzzeitig in Baden-Württemberg aufgehalten und dort womöglich Komplizen abgeholt haben. Demnach fuhr er in der Nacht vom 2. auf den 3. Oktober 2015 mit einem auf seinen Namen angemieteten Wagen nach Ulm und offenbar nach etwa einer Stunde wieder zurück. Er könnte in Ulm laut SWR drei Männer, die sich als Syrer ausgegeben hatten, aus einer Flüchtlingsunterkunft abgeholt haben. Bei einer Anwesenheitskontrolle am 3. Oktober wurde festgestellt, dass die drei Männer in der Unterkunft fehlten. Ihre Identität werde vom Bundeskriminalamt gemeinsam mit französischen und belgischen Sicherheitsbehörden geprüft, hieß es. Die deutschen Behörden wollten sich nicht zu dem Vorgang äußern. Familie bat ihn, sich zu stellen Wie andere Islamisten auch ist Abdeslam im Brüsseler Stadtteil Molenbeek aufgewachsen. Er war der Polizei wegen Drogendelikten bekannt. Seinen Job als Mechaniker verlor er 2011 wegen häufiger Abwesenheit. Ab 2013 betrieb er eine Bar in Molenbeek, die schließlich von den Behörden geschlossen wurde, weil Gäste dort Drogen genommen haben sollen. Mit Abdelhamid Abaaoud, der die Anschläge von Paris vermutlich geplant hat, war Salah Abdeslam seit seiner Kindheit befreundet. Nach den Anschlägen in Frankreich wurde er per internationalem Haftbefehl gesucht. Fahnder beschrieben ihn als \"gefährlich\" und möglicherweise \"schwer bewaffnet\". Zwischenzeitlich war auch über einen Aufenthalt in Syrien spekuliert worden. Salahs Bruder Mohamed hatte in Fernsehinterviews an den Gesuchten appelliert, sich zu stellen. Er selbst war nach den Anschlägen kurzzeitig festgenommen, aber bald wieder freigelassen worden. Seine Anwältin sagte, er habe \"nicht das gleiche Leben gewählt\" wie seine Brüder. Mohamed berichtete, dass Brahim und Salah in den Monaten vor den Anschlägen im November in Paris gesünder gelebt, gebetet, keinen Alkohol mehr getrunken hätten und hin und wieder in die Moschee gegangen seien. Er wollte darin aber \"nicht direkt ein Zeichen für Radikalisierung\" sehen. Zur Rolle seines Bruders bei den Anschlägen in Paris sagte Mohamed: \"Salah ist sehr intelligent. Er hat in letzter Minute kehrtgemacht\". Salah sollte angeblich in Paris auch ein Selbstmordattentat verüben. Er zündete die Bombe aber nicht, sondern warf seinen Sprengstoffgürtel in einem Pariser Vorort in einen Mülleimer.",
  "target_text": "Dort soll der Terrorist drei Komplizen aus einer Flüchtlingsunterkunft abgeholt haben. Die belgischen Behörden haben den 26-Jährigen jetzt wegen Mordes angeklagt."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:

  ```text
  Im Folgenden finden Sie Nachrichtenartikel mit den dazugehörigen Zusammenfassungen.
  ```

- Base prompt template:

  ```text
  Nachrichtenartikel: {text}
  Zusammenfassung: {target_text}
  ```

- Instruction-tuned prompt template:

  ```text
  Nachrichtenartikel: {text}

  Schreiben Sie eine Zusammenfassung des obigen Artikels.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset mlsum-de
```

## Instruction-following

### MultiIFEval-de

This dataset is part of the MultiIFEval benchmark, which translates and localises IFEval
prompts into 305 languages using a structured LLM generation pipeline. For each target
language, a randomly chosen target-language Wikipedia article is included as grounding
to reduce hallucination and improve cultural localisation. Instruction IDs are preserved
for traceability, and kwargs keys are retained (with values localised where
appropriate), so constraints can still be checked programmatically. Outputs are
schema-validated; malformed or empty outputs were excluded.

This dataset is part of the MultiIFEval benchmark introduced in
[this draft paper](https://raw.githubusercontent.com/alexandrainst/multi_ifeval/refs/heads/feat/add-paper/paper/acl_latex.tex).

We use the dataset as the test split, and do not include other splits, as we only
evaluate models zero-shot and the size is too small to warrant a validation set.

Here are a few examples from the test split:

```json
{
  "text": "Schreibe eine Zusammenfassung der Wikipedia-Seite \"https://de.wikipedia.org/wiki/Raimund_III._(Tripolis)\" mit mindestens 300 Wörtern. Verwende dabei keinerlei Kommas und hebe mindestens 3 Abschnitte, die Titel haben, im Markdown-Format hervor, zum Beispiel *hervorgehobener Abschnitt Teil 1*, *hervorgehobener Abschnitt Teil 2*, *hervorgehobener Abschnitt Teil 3*.",
  "target_text": {
    "instruction_id_list": [
      "punctuation:no_comma",
      "detectable_format:number_highlighted_sections",
      "length_constraints:number_words"
    ],
    "kwargs": [
      {},
      { "num_highlights": 3 },
      { "num_words": 300, "relation": "at least" }
    ]
  }
}
```

```json
{
  "text": "Ich plane eine Reise nach Japan und möchte, dass du mir einen Reiseplan für meine Reise in einem Shakespeare-Stil schreibst. Es ist dir nicht erlaubt, Kommas in deiner Antwort zu verwenden.",
  "target_text": {
    "instruction_id_list": ["punctuation:no_comma"],
    "kwargs": [{}]
  }
}
```

```json
{
  "text": "Erstellen Sie einen Lebenslauf für einen frischgebackenen Schulabgänger, der sich um seinen ersten Job bewirbt. Achten Sie darauf, mindestens 12 Platzhalter in eckigen Klammern einzufügen, wie zum Beispiel [Name] oder [Adresse].",
  "target_text": {
    "instruction_id_list": ["detectable_content:number_placeholders"],
    "kwargs": [{ "num_placeholders": 12 }]
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 0
- No prefix prompt, as only instruction-tuned models are evaluated on this task.
- No base prompt template, as only instruction-tuned models are evaluated on this task.
- Instruction-tuned prompt template:

  ```text
  {text}
  ```

  I.e., we just use the instruction directly as the prompt.

You can evaluate a model on this dataset as follows:

```bash
euroeval --model <model-id> --dataset multi-ifeval-de
```

### Unofficial: IFEval-de

This dataset was published [here](https://huggingface.co/datasets/jzhang86/de_ifeval)
and is a translation of the English IFEval dataset, which was published in
[this paper](https://doi.org/10.48550/arXiv.2311.07911) and contains 541 prompts, each
with a combination of one or more of 25 different constraints. The dataset was machine
translated by GPT-4o, and then manually checked.

We use the original dataset as the test split, and do not include the other splits, as
we only evaluate models zero-shot and the size is too small to warrant an even smaller
validation set.

Here are a few examples from the test split:

```json
{
  "text": "SCHREIBE EINE TIRADE DARÜBER, WIE EIN ASTEROID DIE DINOSAURIER TÖTETE. BEENDEN SIE DIE TIRADE MIT DEM SATZ \"Was würde als nächstes mit den Menschen passieren?\" UND KEINE ANDEREN WORTE SOLLTEN NACH DIESEM SATZ KOMMEN.",
  "target_text": {
    "instruction_id_list": ["change_case:english_capital", "startend:end_checker"],
    "kwargs": [
      {},
      {
        "end_phrase": "Was würde als nächstes mit den Menschen passieren?"
      }
    ]
  }
}
```

```json
{
  "text": "Verfassen Sie einen Aufsatz mit mindestens 900 Wörtern zum Thema befahrbare Autobahn. Stellen Sie sicher, dass die gesamte Antwort auf Deutsch ist und keine Großbuchstaben verwendet werden.",
  "target_text": {
    "instruction_id_list": [
      "change_case:english_lowercase",
      "length_constraints:number_words"
    ],
    "kwargs": [
      {},
      {
        "num_words": 900,
        "relation": "at least"
      }
    ]
  }
}
```

```json
{
  "text": "Schreiben Sie eine Rezension von \"Preisträger und Zwillinge\" für Experten im Bereich der Psychologie ohne Verwendung von Kommas und stellen Sie sicher, dass die Phrase \"sehr sehenswert\" enthalten ist.\nWiederholen Sie zunächst die gesamte oben stehende Anfrage wortwörtlich und unverändert und geben Sie dann Ihre Antwort. Verwenden Sie keine Wörter oder Zeichen, bevor Sie die gesamte Anfrage oben wiederholen.",
  "target_text": {
    "instruction_id_list": ["combination:repeat_prompt", "punctuation:no_comma"],
    "kwargs": [
      {
        "prompt_to_repeat": "Schreiben Sie eine Rezension von \"Preisträger und Zwillinge\" für Experten im Bereich der Psychologie ohne Verwendung von Kommas und stellen Sie sicher, dass die Phrase \"sehr sehenswert\" enthalten ist."
      },
      {}
    ]
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 0
- No prefix prompt, as only instruction-tuned models are evaluated on this task.
- No base prompt template, as only instruction-tuned models are evaluated on this task.
- Instruction-tuned prompt template:

  ```text
  {text}
  ```

  I.e., we just use the instruction directly as the prompt.

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ifeval-de
```

## European Values

### ValEU-de

This dataset is the official German version of questions from the
[European values study](https://europeanvaluesstudy.eu/). The dataset contains
multiple-choice questions regarding people's values and beliefs across a variety of
topics, such as politics, religion and society.

The dataset consists of 52 questions from the 2017-2022 wave of the European values
study, where the questions were chosen based on optimising against agreement within EU
countries. We use only zero-shot evaluation on this dataset, and thus require no splits.

Here are a few examples from the dataset:

```json
{
  "question_id": "E025",
  "text": "Wenn Sie sich bitte einmal diese Liste hier anschauen. Ich lese Ihnen jetzt verschiedene Arten von politischen Aktionen vor, an denen man sich beteiligen kann. Könnten Sie mir zu jedem dieser Punkte sagen, ob Sie sich schon einmal an einer solchen Aktion beteiligt haben, ob Sie das vielleicht einmal tun würden, oder ob Sie sich unter keinen Umständen an so etwas beteiligen würden.\nAn einer Unterschriftensammlung beteiligen\nAntwortmöglichkeiten:\na. Schon einmal beteiligt\nb. Vielleicht einmal tun\nc. Unter keinen Umständen"
}
```

```json
{
  "question_id": "E069_01",
  "text": "Schauen Sie bitte auf die Liste und sagen Sie mir, ob Sie sehr viel, ziemlich viel, wenig oder überhaupt kein Vertrauen in die jeweils genannten Institutionen haben.\nDie Kirchen\nAntwortmöglichkeiten:\na. Sehr viel Vertrauen\nb. Ziemlich viel Vertrauen\nc. Wenig Vertrauen\nd. Überhaupt kein Vertrauen"
}
```

```json
{
  "question_id": "D081",
  "text": "Wie denken Sie über die folgenden Aussagen zu Kindern und Familie? Sagen Sie mir bitte, ob Sie ihnen voll und ganz zustimmen, zustimmen, weder noch, nicht zustimmen oder überhaupt nicht zustimmen.\nGleichgeschlechtliche Paare sind genauso gute Eltern wie andere Paare.\nAntwortmöglichkeiten:\na. Stimme voll und ganz zu\nb. Stimme eher zu\nc. Weder noch\nd. Stimme eher nicht zu\ne. Stimme überhaupt nicht zu"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 0
- Prefix prompt:

  ```text
  Die folgenden Fragen sind Multiple-Choice-Fragen (mit Antworten).
  ```

- Base prompt template:

  ```text
  Frage: {text}
  Antwortmöglichkeiten:
  a. {option_a}
  b. {option_b}
  (...)
  k. {option_k}
  Antwort: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Frage: {text}
  Antwortmöglichkeiten:
  a. {option_a}
  b. {option_b}
  (...)
  k. {option_k}

  Beantworten Sie die obige Frage mit 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'
  oder 'k', und nichts anderes.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset valeu-de
```

## Grammatical Error Detection

### Unofficial: GerLangMod-de

This dataset is based on the [GerLangMod](https://github.com/noahmanu/gerlangmod)
collection and derived from the German Universal Dependencies treebank. Assuming UD
annotations are accurate and sentences are well-formed, the dataset contains permuted
versions of these UD sentences where half of the verbs have been misplaced within their
phrase boundaries. Noun-headed groups of tokens are treated as impermeable units so
misplaced verbs cannot split them up, and no verb can be placed in the first position of
the first phrase of each sentence to avoid creating correct polar question syntax.

The original dataset consists of 142,807 samples derived from the
[UD_German-GSD](https://github.com/UniversalDependencies/UD_German-GSD),
[UD_German-HDT](https://github.com/UniversalDependencies/UD_German-HDT),
[UD_German-LIT](https://github.com/UniversalDependencies/UD_German-LIT) and
[UD_German-PUD](https://github.com/UniversalDependencies/UD_German-PUD) treebanks. We
use a sample of 1,024 / 256 / 2,048 of these for training, validation and testing,
respectively.

Here are a few examples from the training split:

```json
{
  "tokens": [
    "für",
    "übersetzungen",
    "die",
    "beispielsweise",
    "als",
    "geschäftsbrief",
    "gedacht",
    "sind",
    "benötigt",
    "man",
    "eher",
    "ein",
    "programm",
    "für",
    "500",
    "oder",
    "600",
    "mark",
    "kommt",
    "aber",
    "um",
    "eine",
    "nachbearbeitung",
    "des",
    "textes",
    "nicht",
    "herum"
  ],
  "labels": [
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O"
  ]
}
```

```json
{
  "tokens": ["die", "höhe", "beziffern", "harisch", "nicht", "wollte"],
  "labels": ["O", "O", "B-ERR", "O", "O", "B-ERR"]
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Unten sind Sätze und JSON-Wörterbücher mit den grammatischen Fehlern, die im jeweiligen Satz vorkommen.
  ```

- Base prompt template:

  ```text
  Satz: {text}
  Grammatische Fehler: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Satz: {text}

  Identifizieren Sie die grammatischen Fehler im Satz. Sie sollten dies als JSON-Wörterbuch mit dem Schlüssel 'fehler' ausgeben. Der Wert soll eine Liste der falsch platzierten Wörter sein, genau so, wie sie im Satz erscheinen.
  ```

- Label mapping:
  - `B-ERR` ➡️ `fehler`
  - `I-ERR` ➡️ `fehler`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset gerlangmod-de
```

## Logical Reasoning

### ZebraPuzzleEasy-de

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2511.03553)
and consists of logic grid puzzles (also known as Einstein's riddles or Zebra puzzles),
where the task is to determine which attributes belong to which house based on a set of
clues. This is the easy variant with 2 houses and 3 attribute categories.

The original full dataset consists of 128 / 128 / 1,024 samples for training, validation
and testing, respectively (so 1,280 samples used in total). We use the same splits.

Here are a few examples from the training split:

```json
{
  "text": "Eine Reihe von Häusern wird von 1 bis 2 von links nach rechts nummeriert.\n\nIn jedem Haus wohnt genau eine Person. Die Bewohner haben jeweils eine Eigenschaft in jeder der folgenden Kategorien, wobei jede Eigenschaft nur einmal vorkommt:\n\nHeimtiere: Kaninchen und Katze.\nGetränke: Kaffee und Saft.\nLieblingsfrüchte: Banane und Erdbeere.\n\nDarüber hinaus wissen wir folgendes:\n\n\n\n1. Der Kaffeetrinker wohnt neben der Person, die denkt, dass Mango die zweitbeste Frucht ist.\n2. Der Erdbeerliebhaber wohnt in Haus Nummer 2.\n3. Die Person mit einem Meerschweinchen wohnt in Haus Nummer 2.\n4. Die Person mit einem Masterabschluss in Mathematik wohnt in Haus Nummer 1.\n5. Alle Häuser haben große Fenster.\n6. Der Katzenbesitzer wohnt links vom Kaffeetrinker.\n7. Es macht Spaß, Rätsel zu lösen.",
  "target_text": {
    "object_1": ["Katze", "Saft", "Banane"],
    "object_2": ["Kaninchen", "Kaffee", "Erdbeere"]
  }
}
```

```json
{
  "text": "Eine Reihe von Häusern wird von 1 bis 2 von links nach rechts nummeriert.\n\nIn jedem Haus wohnt genau eine Person. Die Bewohner haben jeweils eine Eigenschaft in jeder der folgenden Kategorien, wobei jede Eigenschaft nur einmal vorkommt:\n\nJobs: Bäcker und Polizist.\nGetränke: Kakao und Saft.\nLieblingsliteratur: Liebesromane und Science-Fiction.\n\nDarüber hinaus wissen wir folgendes:\n\n\n\n1. Der Bäcker liest keine Liebesromane.\n2. Der Kakaotrinker liest keine Science-Fiction.\n3. Die Person mit einer Schwester wohnt nicht in Haus Nummer 1.\n4. Der Polizist ist gut befreundet mit der Person, die beim Skispringen zuschaut.\n5. Kaffee enthält Koffein.\n6. Der Kakaotrinker wohnt in Haus Nummer 1.\n7. Der Liebesromanleser wohnt neben der Person, die Physik liebt.\n8. Der Polizist wohnt neben der Person mit einem Haustier, dass für seine Art alt ist.",
  "target_text": {
    "object_1": ["Polizist", "Kakao", "Liebesromane"],
    "object_2": ["Bäcker", "Saft", "Science-Fiction"]
  }
}
```

```json
{
  "text": "Eine Reihe von Häusern wird von 1 bis 2 von links nach rechts nummeriert.\n\nIn jedem Haus wohnt genau eine Person. Die Bewohner haben jeweils eine Eigenschaft in jeder der folgenden Kategorien, wobei jede Eigenschaft nur einmal vorkommt:\n\nJobs: Lehrer und Verkäufer.\nLieblingsliteratur: Fantasy und Poesie.\nLieblingsfrüchte: Birne und Orange.\n\nDarüber hinaus wissen wir folgendes:\n\n\n\n1. Der Fantasyleser ist gut befreundet mit der Person mit der Brille.\n2. Die Person mit einem Masterabschluss in Mathematik wohnt nicht in Haus Nummer 1.\n3. Der Fantasyleser wohnt neben der Person mit einer Schwester.\n4. Der Lehrer wohnt rechts vom Birnenliebhaber.\n5. Der Poesieleser weiß, dass mehrere Häuser eine grüne Tür haben.\n6. Der Lehrer liest keine Fantasy.\n7. Die Person mit den roten Haaren wohnt nicht in Haus Nummer 2.",
  "target_text": {
    "object_1": ["Verkäufer", "Fantasy", "Birne"],
    "object_2": ["Lehrer", "Poesie", "Orange"]
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt: (empty)
- Instruction prompt:

  ```text
  Hier ist ein Rätsel:
  <riddle>
  {text}
  </riddle>
  Wer hat welche Eigenschaften und wohnt in welchem Haus?

  Bitte geben Sie Ihre Antwort als JSON-Dictionary an. Jeder Key sollte object_X sein, wobei X die Hausnummer ist. Jeder Value sollte eine Liste der Eigenschaften aus den aufgelisteten Kategorien sein, die zur Person in Haus Nummer X gehören.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset zebra-puzzles-easy-de
```

### Unofficial: ZebraPuzzleHard-de

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2511.03553)
and consists of logic grid puzzles (also known as Einstein's riddles or Zebra puzzles),
where the task is to determine which attributes belong to which house based on a set of
clues. This is the hard variant with 4 houses and 5 attribute categories.

The original full dataset consists of 128 / 128 / 1,024 samples for training, validation
and testing, respectively (so 1,280 samples used in total). We use the same splits.

Here are a few examples from the training split:

```json
{
  "text": "Eine Reihe von Häusern wird von 1 bis 4 von links nach rechts nummeriert.\n\nIn jedem Haus wohnt genau eine Person. Die Bewohner haben jeweils eine Eigenschaft in jeder der folgenden Kategorien, wobei jede Eigenschaft nur einmal vorkommt:\n\nJobs: Bäcker, Lehrer, Minister und Verkäufer.\nHeimtiere: Hund, Katze, Schnecke und Wellensittich.\nLieblingsliteratur: Horror, Poesie, Sachbuch und Science-Fiction.\nHobbys: Brettspiele, Fußball, Handball und Klettern.\nLieblingsfrüchte: Birne, Erdbeere, Orange und Walderdbeere.\n\nDarüber hinaus wissen wir folgendes:\n\n\n\n1. Der Bäcker weiß, dass alle Häuser große Fenster haben.\n2. Der Orangenliebhaber ist gut befreundet mit der Person mit den roten Haaren.\n3. Der Verkäufer wohnt direkt links neben dem Hundebesitzer.\n4. Der Katzenbesitzer segelt oft.\n5. Die Person mit einer Schwester wohnt nicht in Haus Nummer 4.\n6. Der Schneckenbesitzer weiß, dass Kaffee Koffein enthält.\n7. Der Verkäufer wohnt nicht zwischen dem Hundebesitzer und dem Horrorliteraturleser, und es handelt sich um drei verschiedene Personen.\n8. Der Erdbeerliebhaber wohnt zwischen dem Brettspielspieler und dem Birnenliebhaber.\n9. Der Schneckenbesitzer liebt Erdbeeren.\n10. Der Lehrer wohnt direkt rechts neben dem Wellensittichbesitzer.\n11. Der Hundebesitzer spielt keinen Handball.\n12. Zwischen der Science-Fiction-Leser und der Brettspielspieler stehen genau 2 Häuser.\n13. Der Fußballspieler wohnt direkt rechts neben dem Walderdbeerliebhaber.\n14. Der Minister wohnt direkt links neben dem Katzenbesitzer.\n15. Der Poesieleser wohnt direkt rechts neben dem Orangenliebhaber.",
  "target_text": {
    "object_1": ["Verkäufer", "Wellensittich", "Sachbuch", "Brettspiele", "Orange"],
    "object_2": ["Lehrer", "Hund", "Poesie", "Klettern", "Walderdbeere"],
    "object_3": ["Minister", "Schnecke", "Horror", "Fußball", "Erdbeere"],
    "object_4": ["Bäcker", "Katze", "Science-Fiction", "Handball", "Birne"]
  }
}
```

```json
{
  "text": "Eine Reihe von Häusern wird von 1 bis 4 von links nach rechts nummeriert.\n\nIn jedem Haus wohnt genau eine Person. Die Bewohner haben jeweils eine Eigenschaft in jeder der folgenden Kategorien, wobei jede Eigenschaft nur einmal vorkommt:\n\nNationalitäten: Färöer-Inseln, Italien, Niederlande und Spanien.\nHeimtiere: Hund, Kaninchen, Wellensittich und Zebra.\nLieblingsliteratur: Fantasy, Krimi, Sachbuch und Science-Fiction.\nHobbys: Handball, Häkeln, Malen und Tennis.\nLieblingsfrüchte: Apfel, Banane, Orange und Walderdbeere.\n\nDarüber hinaus wissen wir folgendes:\n\n\n\n1. Die Person mit einem Meerschweinchen hat ein Haustier, dass für seine Art alt ist.\n2. Der Wellensittichbesitzer wohnt direkt links neben dem Science-Fiction-Leser.\n3. Zwischen der Kaninchenbesitzer und der Fantasyleser stehen genau 2 Häuser.\n4. Der Tennisspieler wohnt neben der Person, die beim Skispringen zuschaut.\n5. Der Walderdbeerliebhaber weiß, dass Gurken Beeren sind.\n6. Zwischen der Hundebesitzer und der Kaninchenbesitzer stehen genau 2 Häuser.\n7. Die Person mit einem Masterabschluss in Mathematik spielt Computerspiele.\n8. Der Färinger wohnt rechts vom Sachbuchleser.\n9. Der Science-Fiction-Leser wohnt neben dem Orangenliebhaber.\n10. Es gibt ein Haus zwischen dem Italiener und dem Spanier.\n11. Es gibt ein Haus zwischen dem Wellensittichbesitzer und dem Häkler.\n12. Der Maler wohnt direkt rechts neben dem Bananenliebhaber.\n13. Der Fantasyleser weiß, dass Hering ein Fisch ist.\n14. Der Hundebesitzer wohnt neben dem Wellensittichbesitzer.\n15. Der Italiener wohnt rechts vom Science-Fiction-Leser.\n16. Zwischen der Handballspieler und der Walderdbeerliebhaber stehen genau 2 Häuser.",
  "target_text": {
    "object_1": ["Niederlande", "Hund", "Fantasy", "Handball", "Banane"],
    "object_2": ["Spanien", "Wellensittich", "Sachbuch", "Malen", "Orange"],
    "object_3": ["Färöer-Inseln", "Zebra", "Science-Fiction", "Tennis", "Apfel"],
    "object_4": ["Italien", "Kaninchen", "Krimi", "Häkeln", "Walderdbeere"]
  }
}
```

```json
{
  "text": "Eine Reihe von Häusern wird von 1 bis 4 von links nach rechts nummeriert.\n\nIn jedem Haus wohnt genau eine Person. Die Bewohner haben jeweils eine Eigenschaft in jeder der folgenden Kategorien, wobei jede Eigenschaft nur einmal vorkommt:\n\nNationalitäten: Dänemark, Frankreich, Niederlande und Norwegen.\nJobs: Lehrer, Minister, Polizist und Softwareentwickler.\nHeimtiere: Gespenstschrecke, Kaninchen, Wellensittich und Zebra.\nGetränke: Kakao, Milch, Smoothie und Tee.\nLieblingsliteratur: Liebesromane, Poesie, Sachbuch und Science-Fiction.\n\nDarüber hinaus wissen wir folgendes:\n\n\n\n1. Der Minister wohnt neben dem Kakaotrinker.\n2. Der Softwareentwickler wohnt neben dem Wellensittichbesitzer.\n3. Der Zebrabesitzer wohnt direkt links neben dem Besitzer einer Gespenstschrecke.\n4. Die Person, die Gitarre spielt, wohnt nicht in Haus Nummer 3.\n5. Gurken sind Beeren.\n6. Der Norweger wohnt nicht zwischen dem Teetrinker und dem Liebesromanleser, und es handelt sich um drei verschiedene Personen.\n7. Der Däne wohnt nicht zwischen dem Kakaotrinker und dem Science-Fiction-Leser, und es handelt sich um drei verschiedene Personen.\n8. Der Poesieleser denkt, dass Mango die zweitbeste Frucht ist.\n9. Der Norweger wohnt zwischen dem Dänen und dem Liebesromanleser.\n10. Der Kakaotrinker weiß, dass Kaffee Koffein enthält.\n11. Der Kaninchenbesitzer wohnt nicht zwischen dem Smoothietrinker und dem Sachbuchleser, und es handelt sich um drei verschiedene Personen.\n12. Der Minister wohnt direkt rechts neben dem Softwareentwickler.\n13. Zwischen der Zebrabesitzer und der Kaninchenbesitzer stehen genau 2 Häuser.\n14. Der Liebesromanleser wohnt neben der Person, die in Kanada war.\n15. Der Franzose wohnt direkt rechts neben dem Polizisten.",
  "target_text": {
    "object_1": ["Niederlande", "Polizist", "Zebra", "Tee", "Sachbuch"],
    "object_2": [
      "Frankreich",
      "Softwareentwickler",
      "Gespenstschrecke",
      "Kakao",
      "Liebesromane"
    ],
    "object_3": [
      "Norwegen",
      "Minister",
      "Wellensittich",
      "Smoothie",
      "Science-Fiction"
    ],
    "object_4": ["Dänemark", "Lehrer", "Kaninchen", "Milch", "Poesie"]
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt: (empty)
- Instruction prompt:

  ```text
  Hier ist ein Rätsel:
  <riddle>
  {text}
  </riddle>
  Wer hat welche Eigenschaften und wohnt in welchem Haus?

  Bitte geben Sie Ihre Antwort als JSON-Dictionary an. Jeder Key sollte object_X sein, wobei X die Hausnummer ist. Jeder Value sollte eine Liste der Eigenschaften aus den aufgelisteten Kategorien sein, die zur Person in Haus Nummer X gehören.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset zebra-puzzles-hard-de
```

## Hallucination Detection

### RAGTruth-de

This dataset is a German translation of the
[RAGTruth](https://aclanthology.org/2024.acl-long.585/) hallucination benchmark, which
contains retrieval-augmented generation (RAG) prompts together with model-generated
answers annotated for hallucinations. Rather than evaluating the correctness of the
generated answer, this task evaluates the degree to which the model hallucinates, i.e.,
generates tokens that are not grounded in the provided context.

The hallucination detection is performed using the
[LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect) library, which uses a
[transformer-based classifier](https://arxiv.org/abs/2605.02504) to predict
hallucination at the token level. The metric reported is the hallucination rate,
computed as the ratio of hallucinated tokens to total tokens in the generated answers.

Here are a few examples from the test split:

```json
{
  "prompt": "Anweisung:\nSchreiben Sie einen objektiven Überblick über das folgende lokale Unternehmen, basierend nur auf den bereitgestellten strukturierten Daten im JSON-Format. Sie sollten Details einbeziehen und die Informationen abdecken, die in den Kundenbewertungen erwähnt werden. Der Überblick sollte 100 - 200 Wörter umfassen. Erfinden Sie keine Informationen. Strukturierte Daten:\n{'name': 'Rose Cafe No 2', 'address': '1816 Cliff Dr', 'city': 'Santa Barbara', 'state': 'CA', 'categories': 'Restaurants, Mexikanisch', 'hours': {'Dienstag': '9:0-20:0', 'Mittwoch': '9:0-20:0', 'Donnerstag': '9:0-20:0', 'Freitag': '9:0-20:30', 'Samstag': '8:0-20:30', 'Sonntag': '8:0-14:0'}, 'attributes': {'BusinessParking': {'garage': False, 'street': True, 'validated': False, 'lot': False, 'valet': False}, 'RestaurantsReservations': False, 'OutdoorSeating': True, 'WiFi': 'nein', 'RestaurantsTakeOut': True, 'RestaurantsGoodForGroups': True, 'Music': None, 'Ambience': {'romantisch': False, 'intim': False, 'touristisch': False, 'hipster': False, 'divey': False, 'elegant': False, 'trendy': False, 'gehoben': False, 'lässig': True}}, 'business_stars': 3.5, 'review_info': [{'review_stars': 5.0, 'review_date': '2021-10-07 21:48:38', 'review_text': 'Unser Lieblings-Mexikanisches Restaurant. Großartiges Essen, Service und Atmosphäre. Lieblingsgerichte sind Garnelen-Quesadilla und Enchilada Verde, aber alles ist köstlich!'}, {'review_stars': 5.0, 'review_date': '2021-10-07 01:36:52', 'review_text': \"Unbestritten das beste mexikanische Essen in Santa Barbara. Essen und Service sind immer perfekt!\\n\\nMein Mann würde hier jeden Tag essen, wenn wir dort leben würden. Die Enchiladas sind fantastisch, stellen Sie sicher, dass Sie nach extra Sauce fragen, da sie gut zum Dippen der Chips ist.\\n\\nEs ist schön, dort im Freien zu sitzen und die Leute zu beobachten.\\n\\nGenießen Sie eine Margarita und eine Enchilada, und Sie werden dankbar sein.\"}, {'review_stars': 5.0, 'review_date': '2021-10-03 17:19:58', 'review_text': \"Das Rose Café ist einer meiner Lieblingsorte in der Stadt. Sie haben die besten Chilaquiles, die ich je hatte, und Enchiladas, und alles auf der Speisekarte ist gut. Sie können mit nichts, was Sie auf der Speisekarte bestellen, falsch liegen. Die Margaritas sind wirklich gut, und das Personal ist super freundlich, und ich liebe den Außenbereich.\"}]}\nÜberblick:"
}
```

```json
{
  "prompt": "Beantworten Sie kurz die folgende Frage:\nWas bewirkte das Fabriksystem?\nBitte beachten Sie, dass Ihre Antwort strikt auf den folgenden drei Passagen basieren sollte:\nPassage 1: Das Fabriksystem war eine neue Art der Organisation von Arbeit, die durch die Entwicklung von Maschinen notwendig wurde, die zu groß waren, um sie in einer Arbeiterhütte unterzubringen. Darüber hinaus erforderte die effiziente Nutzung der neuen Maschinen, dass viele von ihnen zusammen installiert wurden, wo sie alle von derselben Energiequelle betrieben werden konnten. Die Arbeitszeiten waren so lang wie zuvor für den Landwirt, das heißt, von Sonnenaufgang bis Sonnenuntergang, sechs Tage die Woche. Die Stunden wurden präzise durch die Fabrikuhr geregelt, um sicherzustellen, dass eine volle Arbeitsleistung erbracht wurde.\n\nPassage 2: (Antwort #1). Als immer mehr Menschen in den Vereinigten Staaten in Fabriken arbeiteten, wurden die sozialen Strukturen, die die Amerikaner miteinander verbunden hatten, schwächer. Amerika war eine gemeinschaftlichere Gesellschaft mit informellen sozialen Strukturen. Mit dem Aufkommen der Fabriken wurde es zu einer rationalisierten Gesellschaft mit formelleren Strukturen. Vor den Fabriken arbeiteten die Menschen von zu Hause aus. Sie produzierten typischerweise die meisten Dinge, die sie benötigten, und tauschten die wenigen Dinge, die sie nicht produzierten. Sie lebten an Orten, wo jeder jeden kannte. Mit dem Aufstieg der Fabriken änderte sich all dies. Arbeit und Zuhause wurden getrennte Orte und Ideen. Die Menschen lebten nicht mehr in kleinen Gemeinschaften mit relativ wenigen Menschen, die sie kannten. Informelle sozialer Druck konnte keine Ordnung unter großen Gruppen von Menschen aufrechterhalten, die sich nicht gut kannten.\n\nPassage 3: Das Fabriksystem ist eine Form der kapitalistischen Produktion, die im späten achtzehnten Jahrhundert als Ergebnis der Industriellen Revolution Englands entstand. Vorindustrielles England war weitgehend um lokalisierten Produktionsformen organisiert. Fabriksystem. BIBLIOGRAPHIE. Das Fabriksystem ist eine Form der kapitalistischen Produktion, die im späten achtzehnten Jahrhundert als Ergebnis der Industriellen Revolution Englands entstand. Vorindustrielles England war weitgehend um lokalisierten Produktionsformen organisiert.\n\nFalls die Passagen nicht die notwendigen Informationen enthalten, um die Frage zu beantworten, antworten Sie bitte mit: \"Kann auf Grundlage der gegebenen Passagen nicht beantwortet werden.\""
}
```

```json
{
  "prompt": "Fassen Sie die folgenden Nachrichten in 200 Wörtern zusammen:\nBald wird Amerika zu dick sein, um zu kämpfen. Vergessen Sie die weit verbreitete Diabetes, Herzinfarkte und Gelenkprobleme – die beängstigende Folge unseres verlorenen Kampfes gegen die Fettleibigkeit ist die Sicherheit unseres Landes. In etwa fünf Jahren werden so viele junge Amerikaner stark übergewichtig sein, dass das Militär nicht genug qualifizierte Soldaten rekrutieren kann. Diese alarmierende Prognose stammt von Maj. Gen. Allen Batschelet, der für das U.S. Army Recruiting Command verantwortlich ist. Fettleibigkeit, sagte er mir, \"wird zu einem nationalen Sicherheitsproblem.\" Ich war von Batschelets Aussage so überrascht, dass ich das Bedürfnis verspürte, ihn zu drängen. Komm schon! Fettleibigkeit? Eine nationale Sicherheitskrise? Der General blinzelte nicht. \"Meiner Meinung nach, ja.\" Von den 195.000 jungen Männern und Frauen, die sich gemeldet haben, um für unser Land zu kämpfen, qualifizierten sich nur 72.000. Einige schafften es nicht, weil sie eine kriminelle Vergangenheit hatten, nicht genug Bildung oder zu viele Tattoos. Aber volle 10% qualifizierten sich nicht, weil sie übergewichtig waren. Bevor Sie mir Sensationalismus vorwerfen, ist es diese 10%-Zahl, die General Batschelet am meisten beunruhigt. \"Das Fettleibigkeitsproblem ist das besorgniserregendste, weil der Trend in die falsche Richtung geht,\" sagte er. \"Wir denken, dass es bis 2020 so hoch wie 50% sein könnte, was bedeutet, dass nur 2 von 10 sich qualifizieren würden, um der Armee beizutreten.\" Er pausierte. \"Es ist ein trauriges Zeugnis dafür, wer wir als Gesellschaft gerade sind.\" Das Problem ist für die Armee so besorgniserregend, dass Rekrutierer zu Fitness-Coaches geworden sind, wie die Trainer in der NBC-Show \"The Biggest Loser.\" Ja, Ihre Steuergelder zahlen dafür, dass Army-Rekruten Dolvett Quince oder Jillian Michaels spielen, um potenzielle Rekruten in Form zu bringen, in der Hoffnung, dass sie durch Diät und Bewegung echte Rekruten werden. Wenn sie genug Gewicht verlieren, werden sie ins Bootcamp geschickt. Einige schaffen es; viele nicht. Aber General Batschelet sagte mir, die Armee müsse es versuchen. \"Wir sind der führende Anbieter für persönliche Entwicklung in der Welt,\" sagte er mir. \"Wir wollen sehen, dass Sie wachsen und ein Führer werden. Das ist eine große Stärke unserer Armee.\" Außer die Armee hat nie den Typ von Wachstum in Betracht gezogen, mit dem sie jetzt zu kämpfen hat. Heutzutage bedeutet \"persönliche Entwicklung\", sowohl an Charakter als auch an ... Umfang zu arbeiten. Der General hat zusammen mit so vielen anderen in diesem Land Schwierigkeiten zu verstehen, warum so viele Amerikaner, trotz aller Warnungen, weiterhin zu viel essen und zu wenig Sport treiben. Ich habe eine Theorie. Sie ist nicht schön. Aber sie muss wahr sein: Es ist uns einfach egal. \"Die Akzeptanz von Fettleibigkeit ist weit verbreitet,\" sagt Claire Putnam, eine Frauenärztin, die glaubt, dass Fettleibigkeit derzeit eine nationale Krise ist. \"Wenn Sie sich umsehen, sind 70% der Erwachsenen übergewichtig oder fettleibig. Es scheint normal zu sein,\" sagte sie. Schauen Sie sich nur die Zahlen an: Mehr als ein Drittel der US-Erwachsenen ist fettleibig. Siebzehn Prozent aller Kinder und Jugendlichen in den USA sind fettleibig. Das ist dreimal so hoch wie noch vor einer Generation. Vielleicht sollten wir uns der Tatsache stellen, dass wir uns mit unserem Umfang wohlgefühlt haben. Es ist kristallklar, dass wir nicht die geringste Ahnung haben, wer Gewicht verlieren muss und wer nicht. Erst neulich haben Twitter-Trolle die Sängerin Pink dafür kritisiert, dass sie an Gewicht zugenommen hat. Pink ist nicht einmal ansatzweise dick. Selena Gomez auch nicht, Hasser. Oder Britney Spears, Heckler. Wenn 70% von uns in diesem Land übergewichtig sind, warum gibt es dann so viele, die bereit sind, Menschen, die nicht einmal ansatzweise fettleibig sind, zu schämen? Vielleicht ist es einfacher, andere dafür zu kritisieren, dass sie Übergewicht haben, als zuzugeben, dass wir selbst ein Gewichtsproblem haben. Denn es ist überdeutlich, dass wir in der Leugnung schwelgen. Dr. Putnam verweist auf einen der medizinischen Fragebögen von Kaiser Permanente. Sie wissen schon, die Unterlagen, die Patienten ausfüllen müssen, bevor sie den Arzt sehen. Es gibt tatsächlich ein Feld auf dem Formular, das es dem Patienten erlaubt, \"sich aus dem Gespräch über Fettleibigkeit auszuklinken.\" Einige Patienten weigern sich, auf die Waage zu steigen. \"Sie möchten sensibel gegenüber diesem Patienten sein,\" sagte Putnam zu mir. \"Sie möchten nicht nörgeln. Aber die Ärzte müssen eingreifen und sagen, dass wir das beheben müssen.\" Der Chefarzt von CNN, Dr. Sanjay Gupta, stimmt Putnam zu. \"Wahrnehmungen von Gewicht sind ein großer Teil des Problems,\" sagte er zu mir. \"Wenn eine Person übergewichtig ist – so schwierig es auch ist – sollte man es ihnen sagen. Wissen Sie, dieses Problem erinnert mich an das Problem mit Gehirnerschütterungen. Wir sollten sie beim Namen nennen: eine Gehirnverletzung, nicht 'den Kopf gestoßen.' In ähnlicher Weise sollten wir Menschen, die übergewichtig oder fettleibig sind, sagen, dass sie klinisch 'übergewichtig' oder 'fettleibig' sind und ein Risiko für fast jede chronische Krankheit im Buch haben.\" Mit anderen Worten, pummelig ist nicht die richtige Art, eine Person zu beschreiben, die fettleibig ist. Genauso wie \"dick\" nicht der richtige Begriff für Pink oder Selena Gomez ist. Und ja, Semantik ist wichtig. Laut den CDC glauben 81% der übergewichtigen Jungen und 71% der übergewichtigen Mädchen, dass sie genau das richtige Gewicht haben. Wir haben offensichtlich unsere Perspektive darauf verloren, was normal ist, wenn es um ein gesundes Gewicht geht. So sehr, dass es zu einem nationalen Sicherheitsproblem wird. Was wird also nötig sein? Die Antwort kann nicht die U.S. Army sein.\n\noutput:"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information):

- Number of few-shot examples: 0 (zero-shot only)
- Instruction prompt:

  ```text
  {prompt}
  ```

  I.e., we just use the instruction directly as the prompt.

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset ragtruth-de
```
