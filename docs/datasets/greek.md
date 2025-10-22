# 🇬🇷 Greek

This is an overview of all the datasets used in the Greek part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### Greek-SA

This dataset was published in [this paper](https://doi.org/10.1145/2801948.2802010) and
consists of data from Twitter/X.

The original dataset contains 5,936 / 383 / 767 samples for the training,
validation, and test splits, respectively. We use 1,024 / 256 / 2,048 samples for
our training, validation and test splits, respectively. The test split is created using
additional samples from the training set.

Here are a few examples from the training split:

```json
{
    "text": "Εκτός του ότι ήταν προβληματικό το τηλέφωνό από την πρώτη μέρα όταν μπόρεσα να μιλήσω με το service γιατί το τηλέφωνό δεν το σηκώνουν μου είπαν πως δεν γίνεται αντικατάσταση αλλά service ....σε τηλέφωνό του κουτιού....και όταν το έστειλα να το δουν....αυτα!!!!! Τοχούν 15 μέρες ! Με έχουν αφήσει δίχως τηλέφωνο 15 μέρες και ούτε ένα τηλέφωνο. Ούτε να μου πούνε κάτι. Τσαμπίκα κρατάνε Όμηρο και μένα και τα χρήματα μου. Απαράδεκτοι. Λυπάμαι γιατί cubot τηλέφωνα αγοράζαμε πάντα. Μην πέσεις στην ανάγκη τους. Μακριά!",
    "label": "negative"
}
```

```json
{
    "text": "Το τηλεφωνακι ειναι φανταστικο ειδικα για τα  λεφτα αυτα! Εξαιρετικη εμπειρια χειρισμου κ πλοηγησης, καποιες απειροελαχιστες καθυστερησεις ΔΕΝ χαλουν την τελικη εμπειρια. Πρωινες φωτογραφιες πολυ καλες, βραδυνες πολυ μετριες αλλα οχι εντελως χαλια. Το ηχειο ακουγεται παντα, δεν καλυπτεται κι ειναι αξιοπρεπες για κινητο. Οθονη, το δυνατο χαρτι, εξαιρετικη, μπαταρια, με βγαζει 48 συνεχομενες ωρες με λιγο gaming πολυ social texting κ γενικως ψαχουλεμα αρκετο. Καποιο αρνητικο, θα μπορουσα να ελεγα η ταχυτητα του επεξεργαστη , μετα τις βραδυνες φωτο, αλλα με 200+ euros ηξερα τι επαιρνα κ ειμαι πολυ ευχαριστημενος.",
    "label": "positive"
}
```

```json
{
    "text": "Τόση αυτολύπηση στο ΠΑΣΟΚ, είναι πια να τους λυπάσαι. #dendiavasetomnimonio",
    "label": "negative"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Τα ακόλουθα είναι έγγραφα και το συναίσθημά τους, το οποίο μπορεί να είναι 'θετικό', 'ουδέτερο' ή 'αρνητικό'.
  ```

- Base prompt template:

  ```text
  Έγγραφο: {text}
  Συναίσθημα: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Έγγραφο: {text}

  Ταξινομήστε το συναίσθημα στο έγγραφο. Απαντήστε με 'θετικό', 'ουδέτερο', ή 'αρνητικό', και τίποτα άλλο.
  ```

- Label mapping:
  - `positive` ➡️ `θετικό`
  - `negative` ➡️ `αρνητικό`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset greek-sa
```

## Named Entity Recognition

### elNER

This dataset was published in
[this paper](https://doi.org/10.1145/3411408.3411437).

The original dataset consists of 16,798 / 1,901 / 2,103 samples for the
training, validation, and test splits, respectively. We use 1,024 / 256 / 2,048
samples for our training, validation and test splits, respectively. The new splits
are subsets of the original splits.

Here are a few examples from the training split:

```json
{
    "tokens": ["Επιστήμονες", "στη", "Βρετανία", "μετέτρεψαν", "αρσενικά", "ποντίκια", "σε", "θηλυκά", "κάνοντας", "παρεμβάσεις", "στο", "γονιδίωμά", "τους", "και", "αφαιρώντας", "ορισμένα", "τμήματα", "του", "DNA", "τους", "."],
    "labels": ["O", "O", "B-LOC", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]
}
```

```json
{
    "tokens": ["Στο", "μεταξύ", ",", "μία", "άλλη", "δέσμη", "κατευθυντήριων", "γραμμών", "της", "ΕΚΤ", ",", "που", "αφορά", "τα", "δάνεια", "που", "θα", "γίνουν", "«", "κόκκινα", "»", "στο", "μέλλον", ",", "συνάντησε", "αντιδράσεις", "από", "αρκετά", "μέλη", "της", "Ευρωβουλής", ",", "ιδιαίτερα", "από", "την", "Ιταλία", ",", "καθώς", "και", "από", "λομπίστες", "."],
    "labels": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "B-ORG", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-ORG", "O", "O", "O", "O", "B-LOC", "O", "O", "O", "O", "O", "O"]
}
```

```json
{
    "tokens": ["Είναι", "σημαντικό", "η", "Ελλάδα", "να", "συνεχίσει", "σε", "αυτή", "την", "πορεία", "για", "να", "ανακτήσει", "σταθερή", "πρόσβαση", "στις", "αγορές", "...", "Παρά", "την", "ενισχυμένη", "εμπιστοσύνη", "των", "αγορών", ",", "η", "ελληνική", "οικονομία", "αντιμετωπίζει", "ένα", "δύσκολο", "οικονομικό", "και", "χρηματοοικονομικό", "περιβάλλον", "."],
    "labels": ["O", "O", "O", "B-LOC", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-MISC", "O", "O", "O", "O", "O", "O"]
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Ακολουθούν προτάσεις και λεξικά JSON με τις ονομαστικές οντότητες που εμφανίζονται στην δεδομένη πρόταση.
  ```

- Base prompt template:

  ```text
  Πρόταση: {text}
  Ονομαστικές οντότητες: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Πρόταση: {text}

  Αναγνωρίστε τις ονομαστικές οντότητες στην πρόταση. Θα πρέπει να παράγετε αυτό ως λεξικό JSON με κλειδιά 'πρόσωπο', 'τοποθεσία', 'οργανισμός' και 'διάφορα'. Οι τιμές πρέπει να είναι λίστες των ονομαστικών οντοτήτων αυτού του τύπου, ακριβώς όπως εμφανίζονται στην πρόταση.
  ```

- Label mapping:
  - `B-PER` ➡️ `πρόσωπο`
  - `I-PER` ➡️ `πρόσωπο`
  - `B-LOC` ➡️ `τοποθεσία`
  - `I-LOC` ➡️ `τοποθεσία`
  - `B-ORG` ➡️ `οργανισμός`
  - `I-ORG` ➡️ `οργανισμός`
  - `B-MISC` ➡️ `διάφορα`
  - `I-MISC` ➡️ `διάφορα`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset elner
```

## Linguistic Acceptability

### ScaLA-el

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Greek Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Greek-GUD) by assuming that the
documents in the treebank are correct, and corrupting the samples to create
grammatically incorrect samples. The corruptions were done by either removing a word
from a sentence, or by swapping two neighbouring words in a sentence. To ensure that
this does indeed break the grammaticality of the sentence, a set of rules were used on
the part-of-speech tags of the words in the sentence.

The original full dataset consists of 1,024 / 256 / 2,048 samples for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
used as-is in the framework.

Here are a few examples from the training split:

```json
{
    "text": "Πίσω σ το γραφείο, προσπαθώ να βάλω σε τάξη τις σκέψεις μου.",
    "label": "correct"
}
```

```json
{
    "text": "Πρώτα έκανε την κουτσουκέλα της και μετά άρχιζε τις διαχύσεις.",
    "label": "correct"
}
```

```json
{
    "text": "Αν αποφασίσει ότι πρέπει να συνεχίσουμε την έρευνα, θα ζητήσουμε γραπτή εντολή.",
    "label": "correct"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Οι ακόλουθες είναι προτάσεις και εάν είναι γραμματικά σωστές.
  ```

- Base prompt template:

  ```text
  Πρόταση: {text}
  Γραμματικά σωστή: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Πρόταση: {text}

  Προσδιορίστε εάν η πρόταση είναι γραμματικά σωστή ή όχι. Απαντήστε με 'ναι' αν είναι σωστή, ή 'όχι' αν δεν είναι. Απαντήστε μόνο με αυτή τη λέξη, και τίποτα άλλο.
  ```

- Label mapping:
  - `correct` ➡️ `ναι`
  - `incorrect` ➡️ `όχι`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-el
```

## Reading Comprehension

### MultiWikiQA-el

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
    "context": "Για συνώνυμους οικισμούς στην Ελλάδα δείτε το λήμμα: Σαρακήνα (αποσαφήνιση)\n\nΗ Σαρακήνα είναι ορεινό χωριό της περιφερειακής ενότητας Γρεβενών σε υψόμετρο 750 μέτρα.\n\nΓεωγραφία - Ιστορία \nΗ Σαρακήνα βρίσκεται προς στα ανατολικά όρια με την περιφερειακή ενότητα Κοζάνης και στην περιοχή των Βεντζίων. Απέχει 29 χλμ. Α.-ΝΑ. από τα Γρεβενά και 28,5 χλμ. ΒΔ. από τη Δεσκάτη. Νότια του χωριού περνάει ο χείμαρρος Γκιώνης που καταλήγει νοτιότερα στον ποταμό Αλιάκμονα. Σύμφωνα με την τοπική παράδοση το χωριό βρισκόταν πάντα στην ίδια τοποθεσία και είχε το ίδιο όνομα. Την περίοδο της τουρκοκρατίας ήταν κεφαλοχώρι, είχε στενή σχέση με την Ιερά Μονή Ζαβόρδας, από την οποία απέχει 25 χλμ. βορειοδυτικά και σε αυτό έμεινε για μεγάλο διάστημα ο Όσιος Νικάνορας. Πολλοί κάτοικοί του δούλεψαν για λογαριασμό της Μονής Ζάβορδας, όπου και προστατεύονταν από τους Τούρκους. Στον Κώδικα της Ζάβορδας αναφέρονται πολλοί κάτοικοί της ως αφιερωτές  της Ιεράς Μονής Αγίου Νικάνορος ή Ζάβορδας τόσο την περίοδο 1534 έως 1692 (Α΄ Γραφή) όσο και 1692 και μετά (Β΄Γραφή).\n\nΗ Σαρακήνα είναι ο τόπος καταγωγής του δάσκαλου - λαογράφου Κώστα Καραπατάκη (1906 - 2000).\n\nΑξιοθέατα\nΗ τρίκλιτη θολωτή βασιλική εκκλησία του Αγίου Νικολάου έχει χαρακτηριστεί από το 1995 ως \"μνημείο χρήζον ειδικής κρατικής προστασίας. Πρόκειται για ενοριακό ναό μεγάλων διαστάσεων, στον τύπο της θολωτής βασιλικής, που κτίστηκε το 1858. Eσωτερικά το μνημείο είναι κατάγραφο με τοιχογραφίες που χρονολογούνται στο δεύτερο μισό του 19ου αι.\".\n\nΔιοικητικά \nΤο χωριό αναφέρεται επίσημα 1918 στο ΦΕΚ 260Α - 31/12/1918 με το όνομα Σαρακίνα να ορίζεται έδρα τη ομώνυμης κοινότητας που ανήκε στο νομό Κοζάνης. Το όνομα διορθώνεται σε Σαρακήνα το 1940 και το 1964 με το ΦΕΚ 185Α - 30/10/1964 αποσπάται στο νομό Γρεβενών. Το χωριό σύμφωνα με το σχέδιο Καλλικράτης, μαζί με το Νεοχώρι και το Δίπορο αποτελούν την τοπική κοινότητα Σαρακήνας που ανήκει στη δημοτική ενότητα Βεντζίου του Δήμου Γρεβενών και σύμφωνα με την απογραφή 2011ως κοινότητα έχει πληθυσμό 378 κατοίκους, ενώ ως οικισμός 177.\n\nΕξωτερικοί σύνδεσμοι\nΝαός Αγίου Νικολάου Σαρακήνας  από τον ιστότοπο Οδυσσεύς του Υπουργείου Πολιτισμού\n\nΠαραπομπές\n\nΔήμος Γρεβενών\nΣαρακήνα', 'question': 'Που βρίσκεται η Σαρακήνα;",
    "answers": {
        "answer_start": [134],
        "text": ["Γρεβενών"]
    }
}
```

```json
{
    "context": "Η Ρεάλ Κλουμπ Θέλτα ντε Βίγο, Σ.Α.Ν. «Β» (Real Club Celta de Vigo, S.A.D. \"B\") είναι ισπανικός ποδοσφαιρικός σύλλογος με έδρα το Βίγο της Ποντεβέρδα, στην αυτόνομη κοινότητα της Γαλικίας. Ιδρύθηκε το 1927, είναι η δεύτερη ομάδα της Θέλτα Βίγο και συμμετέχει σήμερα στη Σεγούντα Ντιβισιόν Β. Παίζει τους εντός έδρας αγώνες της στο Μουνιθιπάλ ντε Μπαρέιρο, χωρητικότητας 4.500 θέσεων.\n\nΙστορία \nΤο 1927, ιδρύθηκε η Σπορτ Κλουμπ Τουρίστα (Sport Club Turista), που μετονομάστηκε σε Κλουμπ Τουρίστα (Club Turista) εννέα χρόνια αργότερα. Το 1988, προβλεπόταν μία συγχώνευση με τη Γκραν Πένια ΦΚ, αλλά τελικά η Τουρίστα απορροφήθηκε από τη Θέλτα Βίγο και μετονομάστηκε Θέλτα Τουρίστα (Celta Turista). Η Γκραν Πένια συνέχισε να αγωνίζεται τις επόμενες δεκαετίες ως ανεξάρτητος σύλλογος.\n\nΣτην πρώτη σεζόν επαγγελματικού ποδοσφαίρου, η Θέλτα Τουρίστα αγωνίστηκε στην Περιφερειακή Κατηγορία, τερματίζοντας στην πρώτη θέση με 57 πόντους. Αγωνίστηκε για πρώτη φορά στη Σεγούντα Ντιβισιόν Β τη σεζόν 1992-93, όπου υποβιβάστηκε την επόμενη σεζόν. Το 1996, για να συμμορφωθεί με τους νέους κανονισμούς της Βασιλικής Ισπανικής Ομοσπονδίας Ποδοσφαίρου, ο σύλλογος άλλαξε την ονομασία του σε Θέλτα Βίγο Β (Celta de Vigo Β).\n\nΙστορικό ονομασιών \n\n Σπορτ Κλουμπ Τουρίστα (1927-1936) \n Κλουμπ Τουρίστα (1936-1988) \n Θέλτα Τουρίστα (1988-1996) \n Θέλτα Βίγο Β (1996-)\n\nΔιακρίσεις \n\n Τερθέρα Ντιβισιόν: 1957-58, 1999-2000, 2000-01 \n Κύπελλο Βασιλικής Ισπανικής Ομοσπονδίας Ποδοσφαίρου: 2001-02\n\nΕξωτερικοί σύνδεσμοι \n\n Επίσημος ιστότοπος Θέλτα Βίγο \n Προφίλ ομάδας στο Futbolme  \n\nΙδρύσεις ποδοσφαιρικών ομάδων το 1927\nΠοδοσφαιρικές ομάδες Ισπανίας\nΘέλτα Βίγο\nΓαλικία",
    "question": "Πότε άλλαξε όνομα η Θέλτα Τουρίστα σε Θέλτα Βίγο Β;",
    "answers": {
        "answer_start": [1036],
        "text": ["1996"]
    }
}
```

```json
{
    "context": "Ο Ντανίλο Α΄ Τσέπτσεβιτς Πέτροβιτς-Νιέγκος (σερβικά: Данило I Шћепчевић Петровић Његош, 1670 – 11 Ιανουαρίου 1735), πιο γνωστός ως Βλάντικα (επίσκοπος) Ντανίλο, από τον Οίκο των Πέτροβιτς-Νιέγκος ήταν Σέρβος Ορθόδοξος Μητροπολίτης του Τσετίνγιε την περίοδο 1697-1735 και Πρίγκιπας-Επίσκοπος του Μαυροβουνίου. Αυτοαποκαλείτο ως "vojevodič srpskoj zemlji" (Δούκας των σερβικών εδαφών).\n\nΒιογραφία\n\nΟ Ντανίλο Τσέπτσεβιτς γεννήθηκε περίπου το 1670, στο Νιεγκούσι. Θεωρείται ως ο ιδρυτής της δυναστείας των Πέτροβιτς-Νιέγκος στο Μαυροβούνιο το 1696. Αφού πρώτα διηύθυνε αμυντικές επιχειρήσεις και οχυρωματικές εργασίες, ενώ, έστω σε περιορισμένο βαθμό, κατάφερε να θέσει ένα τέρμα στις αντιπαλότητες μεταξύ των διαφορετικών μεγάλων οικογενειών του Μαυροβουνίου, ο Ντανίλο ξεκίνησε πολεμικές επιχειρήσεις κατά των Οθωμανών το 1711. Στη διάρκεια της βασιλείας του, έκαναν, για πρώτη φορά, την εμφάνισή τους διπλωματικές σχέσεις και πολιτικοί δεσμοί μεταξύ της Ρωσίας και του Μαυροβουνίου.\n\nΈνας Ρώσος ιστορικός, ο Πάβελ Ροβίνσκι, έγραψε με εντυπωσιακό βαθμό αντικειμενικότητας αναφορικά με τις σχέσεις μεταξύ Μαυροβουνίου και Ρωσίας, χωρίς να τον απασχολούν τα συμφέροντα της κυβέρνησης της χώρας του. Ο Ροβίνσκι έβγαλε ως συμπέρασμα ότι ήταν η απειλή της Αυστρίας και της Τουρκίας (και κατά καιρούς κι αυτή της Δημοκρατίας της Βενετίας) που έστρεψαν το Μαυροβούνιο προς τη Ρωσία. Χωρίς να έχουν πού να αποταθούν στον σκληρό αγώνα που έδιναν για την επιβίωσή τους, οι ηγέτες της σερβικής περιοχής του Μαυροβουνίου στράφηκαν προς το παρελθόν τους, στις μυθικές τους καταβολές και την πηγή όλων των σλαβικών φύλων, όχι μόνο επειδή αποτελούσε μια Μεγάλη Δύναμη, αλλά και για το γεγονός ότι αποτελούσε ισχυρό αντίβαρο απέναντι στους Τούρκους και τους Αυστριακούς.\n\nΤο 1715, ο Ντανίλο επισκέφτηκε τον Τσάρο Πέτρο Α΄ στην Αγία Πετρούπολη και εξασφάλισε τη συμμαχία του ενάντια στους Οθωμανούς—ένα ταξίδι το οποίο έγινε παράδοση για τους διαδόχους του στην ηγεσία του Μαυροβουνίου και σε όλες τις υπόλοιπες σερβικές περιοχές των Βαλκανίων. Στη συνέχεια, επανέκτησε τον έλεγχο της Ζέτα, η οποία βρισκόταν υπό οθωμανική κατοχή, άνοιξε, εκ νέου, το μοναστήρι στο Τσετίνγιε, και ανήγειρε οχυρώσεις γύρω από το Μοναστήρι του Πόντοστρογκ-Πόντμανιε στην Μπούντβα το οποίο ανακατασκευάστηκε το 1630 και χρησίμευε ως θερινή κατοικία για την ηγέτιδα οικογένεια του Μαυροβουνίου. Σε κείμενο που σώζεται σε χειρόγραφο, δώρο του προς το Σερβικό Πατριαρχείο του Πετς, το 1732, ο Ντανίλο συστήνεται με υπερηφάνεια ως \"Ντανίλ Νιέγκος, Επίσκοπος του Τσετίνγιε, ηγεμόνας της σερβικής γης.\"\n\nΤον Ντανίλο διαδέχτηκαν δύο μέλη της οικογένειάς του και στενοί συγγενείς, αρχικά ο ξάδερφός του Σάββας Β΄ Πέτροβιτς-Νιέγκος και στη συνέχεια ο ανιψιός του Βασίλειος Πέτροβιτς-Νιέγκος, ο οποίος για περισσότερες από δύο δεκαετίες κατάφερε να εκτοπίσει από την εξουσία τον αντιδημοφιλή Σάββα και να γίνει η σημαντικότερη ηγετική αρχή του Μαυροβουνίου και εκπρόσωπός του εκτός συνόρων. Η επιλογή του Σάββα Β΄ Πέτροβιτς-Νιέγκος από τον Ντανίλο είχε κυρίως να κάνει με τους οικογενειακούς δεσμούς και το γεγονός ότι συγκαταλεγόταν μεταξύ των υποστηρικτών του τελευταίου, καθώς η οικογένεια του Σάββα προερχόταν από τη γενέτειρα των Πέτροβιτς, Νιεγκούσι. Όπως κι ο Ντανίλο, ο Σάββας έγινε μοναχός, υπηρετώντας στο Μοναστήρι του Μάνιε στις ακτές του Μαυροβουνίου, όπου χρίστηκε αρχιερέας το 1719 από τον Σέρβο Πατριάρχη του Πετς, Μωυσή (1712–1726). Από τη στιγμή που χρίστηκε κι έπειτα, ο Ντανίλο άρχισε σταδιακά να μυεί τον νεαρό Σάββα στην πολιτική ζωή, δίνοντας του ρόλο συγκυβερνήτη ως προετοιμασία ενόψει του μελλοντικού του ρόλου. Ωστόσο, λίγα πράγματα από τη μετέπειτα πολιτική δραστηριότητα του Σάββα δείχνουν ότι είχε οποιοδήποτε κέρδος από τα "πολιτικά γυμνάσια" στα οποία τον υπέβαλλε ο Ντανίλο, με εξαίρεση το γεγονός ότι εξακολούθησε να διατηρεί μια στάση status quo, ενώ, ταυτόχρονα, άφηνε τους τοπικούς φυλάρχους ελεύθερους να αυτοδιαχειρίζονται τις περιοχές τους.\n\nΕξωτερικοί σύνδεσμοι\n Ο Βλάντικα Ντανίλο στο montenet.org - στα αγγλικά\n Διατηρημένες Επιστολές του Βλάντικα Ντανίλο - στα σερβικά\n Επίσημη ιστοσελίδα του Βασιλικού Οίκου του Μαυροβουνίου.\n The Njegoskij Fund Public Project : Private family archives-based digital documentary fund focused on history and culture of Royal Montenegro.\n Άρθρο στη μαυροβουνίτικη γλώσσα σχετικά με τον Μητροπολίτη Ντανίλο Πέτροβιτς-Νιέγκος\n\nΠαραπομπές\n\nΒλαντικα Ντανιλο\nΟίκος των Πέτροβιτς-Νιέγκος', 'question': 'Σε ποια τοποθεσία έχτισε οχυρά ο Ντανίλο;",
    "answers": {
        "answer_start": [2180],
        "text": ["γύρω από το Μοναστήρι του Πόντοστρογκ-Πόντμανιε στην Μπούντβα"]
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  Ακολουθούν κείμενα με σχετικές ερωτήσεις και απαντήσεις.
  ```

- Base prompt template:

  ```text
  Κείμενο: {text}
  Ερώτηση: {question}
  Απάντηση (έως 3 λέξεις):
  ```

- Instruction-tuned prompt template:

  ```text
  Κείμενο: {text}

  Απαντήστε στην παρακάτω ερώτηση που σχετίζεται με το κείμενο παραπάνω χρησιμοποιώντας το πολύ 3 λέξεις.

  Ερώτηση: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-el
```

## Knowledge

### MMLU-sk

This dataset is a machine translated version of the English [MMLU
dataset](https://openreview.net/forum?id=d7KBjmI3GmQ) and features questions within 57
different topics, such as elementary mathematics, US history and law. The translation to
Slovak was done by the University of Oregon as part of [this
paper](https://aclanthology.org/2023.emnlp-demo.28/), using GPT-3.5-turbo.

The original full dataset consists of 269 / 1,410 / 13,200 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 2,048 split for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
new and there can thus be some overlap between the original validation and test sets and
our validation and test sets.

Here are a few examples from the training split:

```json
{
  "text": "V akých smeroch je prípad pre humanitárnu intervenciu, ako je uvedené v tejto kapitol... mocnými štátmi.\nd. Všetky tieto možnosti.",
  "label": "d",
}
```

```json
{
  "text": "FAKTORIÁLOVÝ ANOVA sa používa v prípade, že štúdia zahŕňa viac ako 1 VI. Aký je INTER...činok VI na rovnakej úrovni ako ostatné VI",
  "label": "a"
}
```

```json
{
  "text": "Pre ktorú z týchto dvoch situácií urobí hlavná postava (ktorá používa ja/mňa/môj) nie...ie zlé\nc. Nie zlé, zlé\nd. Nie zlé, nie zlé",
  "label": "d",
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Nasledujú otázky s viacerými možnosťami (s odpoveďami).
  ```

- Base prompt template:

  ```text
  Otázka: {text}
  Možnosti:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Odpoveď: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Otázka: {text}

  Odpovedzte na nasledujúcu otázku použitím 'a', 'b', 'c' alebo 'd', a nič iné.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset mmlu-sk
```

### Winogrande-sk

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2506.19468)
and is a translated and filtered version of the English [Winogrande
dataset](https://doi.org/10.1145/3474381).

The original full dataset consists of 47 / 1,210 samples for training and testing, and
we use 128 of the test samples for validation, resulting in a 47 / 128 / 1,085 split for
training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Nedokázal som ovládať vlhkosť ako som ovládal dážď, pretože _ prichádzalo odvšadiaľ. Na koho sa vzťahuje prázdne miesto _?\nMožnosti:\na. Možnosť A: vlhkosť\nb. Možnosť B: dážď",
  "label": "a"
}
```

```json
{
  "text": "Jessica si myslela, že Sandstorm je najlepšia pieseň, aká bola kedy napísaná, ale Patricia ju nenávidela. _ si kúpila lístok na jazzový koncert. Na koho sa vzťahuje prázdne miesto _?\nMožnosti:\na. Možnosť A: Jessica\nb. Možnosť B: Patricia",
  "label": "b"
}
```

```json
{
  "text": "Termostat ukazoval, že dole bolo o dvadsať stupňov chladnejšie ako hore, takže Byron zostal v _ pretože mu bola zima. Na koho sa vzťahuje prázdne miesto _?\nMožnosti:\na. Možnosť A: dole\nb. Možnosť B: hore",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Nasledujú otázky s viacerými možnosťami (s odpoveďami).
  ```

- Base prompt template:

  ```text
  Otázka: {text}
  Možnosti:
  a. {option_a}
  b. {option_b}
  Odpoveď: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Otázka: {text}
  Možnosti:
  a. {option_a}
  b. {option_b}

  Odpovedzte na nasledujúcu otázku použitím 'a' alebo 'b', a nič iné.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset winogrande-sk
```
