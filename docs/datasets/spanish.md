# 🇪🇸 Spanish

This is an overview of all the datasets used in the Spanish part of EuroEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.

## Sentiment Classification

### SentimentHeadlines-es

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2208.13947)
and features political news headlines.

The original full dataset consists of 1,371 /  609 / 459 samples for training,
validation, and testing, respectively. We use 861 /  256 / 1,024 samples for training,
validation, and testing, respectively. All our splits are subsets of the original ones.
The label distribution for the splits are as follows:

| Split | positive | negative | neutral | Total |
|-------|----------|----------|---------|-------|
| Train | 368      | 248      | 245     | 861   |
| Val   | 88       | 90       | 78      | 256   |
| Test  | 417      | 293      | 314     | 1,024 |
| Total | 873      | 631      | 637     | 2,141 |

Here are a few examples from the training split:

```json
{
    "text": "Mauricio Macri, en el cierre de campaña: “Esta marcha no termina hoy acá, sino en noviembre”",
    "label": "neutral"
}
```

```json
{
    "text": "Lavagna reforzó su discurso económico y pidió más consumo",
    "label": "positive"
}
```

```json
{
    "text": "Sin la aprobación del Fondo, Macri quema reservas para la fuga",
    "label": "negative"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Lo siguiente son reseñas y su sentimiento, que puede ser 'positivo', 'neutral' o 'negativo'.
  ```

- Base prompt template:

  ```text
  Texto: {text}
  Sentimiento: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Texto: {text}

  Clasifica el sentimiento de la reseña. Responde con 'positivo', 'neutral' o 'negativo', y nada más.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset sentiment-headlines-es
```

## Named Entity Recognition

### CoNLL-es

This dataset was published in [this paper](https://aclanthology.org/W02-2024/) and
contains 8,324 / 1,916 / 1,518 samples for training, validation, and testing,
respectively. We use 1,024 / 256 / 1,024 samples for training, validation, and testing,
respectively. All the new splits are subsets of the original splits.

Here are a few examples from the training split:

```json
{
    "tokens": array(["Todo", "estará", "integrado", ",", "la", "relación", "entre", "los", "espacios", "y", "entre", "los", "músicos", "y", "la", "audiencia", "."], dtype=object),
    "labels": array(["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"], dtype=object),
}
```

```json
{
  "tokens": array(["(", "NA2428-NH4437", ")", "PSOE", "PIDE", "QUE", "COMISION", "CONTROL", "DE", "RTVE", "CONOZCA", "PRESUPUESTO", "ENTE", "Madrid", "(", "EFE", ")", "."], dtype=object),
  "labels": array(["O", "O", "O", "B-ORG", "O", "O", "B-MISC", "I-MISC", "O", "B-ORG", "O", "O", "O", "B-LOC", "O", "B-ORG", "O", "O"], dtype=object),
}
```

```json
{
  "tokens": array(["(", "NA2428-NH4437", ")", "PSOE", "PIDE", "QUE", "COMISION", "CONTROL", "DE", "RTVE", "CONOZCA", "PRESUPUESTO", "ENTE", "Madrid", "(", "EFE", ")", "."], dtype=object),
  "labels": array(["O", "O", "O", "B-ORG", "O", "O", "B-MISC", "I-MISC", "O", "B-ORG", "O", "O", "O", "B-LOC", "O", "B-ORG", "O", "O"], dtype=object),
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:

  ```text
  Lo siguiente son oraciones y diccionarios JSON con las entidades nombradas que aparecen en la oración dada.
  ```

- Base prompt template:

  ```text
  Oración: {text}
  Entidades nombradas: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Oración: {text}

  Identifica las entidades nombradas en la oración. Debes producir esto como un diccionario JSON con las claves 'persona', 'lugar', 'organización' y 'misceláneo'. Los valores deben ser listas de las entidades nombradas de ese tipo, exactamente como aparecen en la oración.
  ```

- Label mapping:
  - `B-PER` ➡️ `persona`
  - `I-PER` ➡️ `persona`
  - `B-LOC` ➡️ `lugar`
  - `I-LOC` ➡️ `lugar`
  - `B-ORG` ➡️ `organización`
  - `I-ORG` ➡️ `organización`
  - `B-MISC` ➡️ `misceláneo`
  - `I-MISC` ➡️ `misceláneo`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset conll-es
```

## Linguistic Acceptability

### ScaLA-es

This dataset was published in [this paper](https://aclanthology.org/L08-1222/) and was
automatically created from the [Spanish Universal
Dependencies](https://github.com/UniversalDependencies/UD_Spanish-AnCora) by assuming
that the documents in the treebank are correct, and corrupting the samples to create
grammatically incorrect samples. The corruptions were done by either removing a word
from a sentence, or by swapping two neighbouring words in a sentence. To ensure that
this does indeed break the grammaticality of the sentence, a set of rules were used on
the part-of-speech tags of the words in the sentence.

The original dataset consists of 17,662 samples, from which we use 1,024 / 256 / 2,048
samples for training, validation and testing, respectively (so 3,328 samples used in
total). These splits are used as-is in the framework.

Here are a few examples from the training split:

```json
{
    "text": "El fuego obligó al a el desalojo preventivo de algunas casas y del de el observatorio del de el Roque de los Muchachos, del de el Instituto de Astrofísica de Canarias.",
    "label": "correct"
}
```

```json
{
    "text": "El libro que leemos intenta explicarlo explicar, pero sin exagerar las posturas de tirios y troyanos.",
    "label": "incorrect"
}
```

```json
{
    "text": "Por su parte, el Consejo de Ministros dio ayer otra vuelta de tuerca al a el control urbanístico de las ciudades autónomas de Ceuta y de Melilla para evitar la urbanística por parte del de el Grupo Independiente Liberal (GIL), que gobierna en Ceuta.",
    "label": "incorrect"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:

  ```text
  Lo siguiente son textos y si son gramaticalmente correctos.
  ```

- Base prompt template:

  ```text
    Texto: {text}
    Gramaticalmente correcto: {label}
  ```

- Instruction-tuned prompt template:

  ```text
    Texto: {text}

    Determina si el texto es gramaticalmente correcto o no. Responde con 'sí' si el texto es correcto, y 'no' si no lo es.
  ```

- Label mapping:
  - `correct` ➡️ `sí`
  - `incorrect` ➡️ `no`

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset scala-es
```

## Reading Comprehension

### MLQA-es

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.1910.07475)
and contains 0 / 500 / 5,253 samples for training, validation, and testing,
respectively. We have made a 1,024 / 256 / 2,048 split, where we use the 500 validation
samples + 524 test samples for training. Then we split the remaining test set into
validation (256 samples) and test (2048 samples).

Here are a few examples from the training split:

```json
{
    "context": "En 1978, el Banco Estatal de Vietnam introdujo los primeros billetes de 5 hao, 1, 5, 10, 20 y 50 đồng fechados en 1976. En 1980 se añadieron los billetes de 2 y 10 đồng, seguidos de los de 30 y 100 đồng en 1981.",
    "question": "¿Cuándo añadió el Banco Estatal de Vietnam los billetes de 2 y 10 đồng?",
    "answers": {
      "answer_start": [120],
      "text": ["En 1980"]
    }
}
```

```json
{
    "context": "Como otros terópodos de la familia Dromaeosauridae, Saurornitholestes era un dinosaurio carnívoro bípedo, equipado con una garra retráctil con forma de oz en el segundo dedo de cada pie. Saurornitholestes era más ligero y tenía las patas más largas que otros dromaeosáuridos como Velociraptor o Dromaeosaurus. Se asemeja a Velociraptor en tener dientes grandes, parecidos a colmillos, en la parte frontal de las mandíbulas.",
    "question": "¿Dónde se encuentra la garra de Saurornitholestes?",
    "answers": {
        "answer_start": [161],
        "text": ["segundo dedo de cada pie"]
    }
}
```

```json
{
    "context": "En cinco ediciones (en las tres primeras, 1896, 1900 y 1904, así como en las de 1988 y 1992) fueron entregadas por prueba dos medallas de bronce (una a cada uno de los perdedores de las semifinales); en el resto de ediciones se ha disputado adicionalmente un partido por el tercer lugar para definir al ganador de la medalla de bronce.",
    "question": "¿De qué material fueron las medallas entregadas a los semifinalistas en 1896?",
    "answers": {
        "answer_start": [138], "text": ["bronce"]
        }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  A continuación se presentan textos con sus preguntas y respuestas correspondientes.
  ```

- Base prompt template:

  ```text
  Texto: {text}
  Pregunta: {question}
  Respuesta en máximo 3 palabras: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Texto: {text}

  Responda la siguiente pregunta sobre el texto anterior en máximo 3 palabras.

  Pregunta: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset xquad-es
```

### Unofficial: XQuAD-es

This dataset was published in [this paper](https://aclanthology.org/2020.acl-main.421/)
and contains 1190 question-answer pairs from [SQuAD
v1.1](https://rajpurkar.github.io/SQuAD-explorer/) translated into ten languages by
professional translators.

The dataset is split intro 550 / 128 / 512 question-answer pairs for training,
validation, and testing, respectively.

Here are a few examples from the training split:

```json
{
    "context": "El Mercado del Grainger reemplazó a un mercado anterior construido originalmente en 1808 llamado el Mercado del Carnicero. El Mercado del Grainger en sí mismo, se abrió en 1835 y fue el primer mercado interior de Newcastle. En el momento de su apertura en 1835 se dijo que era uno de los mercados más grandes y hermosos de Europa. La inauguración se celebró con una gran cena a la que asistieron 2000 invitados, y la Galería de Arte Laing tiene un cuadro de este evento. Con la excepción del techo de madera, que fue destruido por un incendio en 1901 y sustituido por arcos de celosía de acero, el mercado se encuentra en su mayor parte en su estado original. La arquitectura del Mercado del Grainger, como la mayoría de las de Grainger Town, que están clasificadas en el grado I o II, fue clasificada en el grado I en 1954 por Patrimonio Inglés.",
    "question": "¿Cuántos invitados asistieron a la cena de inauguración del Mercado del Grainger?",
    "answer": {
      "answer_start": [396],
      "text": ["2000"]
    }
}
```

```json
{
    "context": "Los avances realizados en Oriente Medio en botánica y química llevaron a la medicina en el Islam medieval a desarrollar sustancialmente la farmacología. Muhammad ibn Zakarīya Rāzi (Rhazes) (865-915), por ejemplo, actuó para promover los usos médicos de los compuestos químicos. Abu al-Qasim al-Zahrawi (Abulcasis) (936-1013) fue pionero en la preparación de medicamentos por sublimación y destilación. Su Liber servitoris es de particular interés, ya que proporciona al lector recetas y explica cómo preparar los 'simples' a partir de los cuales se componían los complejos medicamentos que se utilizaban entonces de forma generalizada. Sabur Ibn Sahl (d 869), fue, sin embargo, el primer médico en iniciar la farmacopedia, describiendo una gran variedad de medicamentos y remedios para las dolencias. Al-Biruni (973-1050) escribió una de las obras islámicas más valiosas sobre farmacología, titulada Kitab al-Saydalah (El libro de los medicamentos), en la que detallaba las propiedades de los medicamentos y esbozaba el papel de la farmacia, así como las atribuciones y los deberes de los farmacéuticos. Avicena también describió nada menos que 700 preparados, sus propiedades, modos de acción y sus indicaciones. De hecho, dedicó todo un volumen a los medicamentos simples en El canon de la medicina. De gran impacto fueron también las obras de al-Maridini de Bagdad y El Cairo, y de Ibn al-Wafid (1008-1074), ambas impresas en latín más de cincuenta veces, apareciendo como De Medicinis universalibus et particularibus de 'Mesue' el más joven, y el Medicamentis simplicibus de 'Abenguefit'. Pedro de Abano (1250-1316) tradujo y añadió un suplemento a la obra de al-Maridini bajo el título De Veneris. Las contribuciones de Al-Muwaffaq en este campo también son pioneras. En su vida en el siglo X, escribió Los fundamentos de las verdaderas propiedades de los remedios, describiendo, entre otras cosas, el óxido arsenioso, y conociendo el ácido silícico. Hizo una clara distinción entre carbonato de sodio y carbonato de potasio y llamó la atención sobre la naturaleza venenosa de los compuestos de cobre, especialmente el vitriolo de cobre, y también los compuestos de plomo. También describe la destilación de agua de mar para beber [se requiere verificación].",
    "question": "¿Cuáles fueron los desarrollos en los que los científicos influyeron en la creación de la farmacología en el Islam medieval?",
    "answer": {
      "answer_start": [43],
      "text": ["botánica y química"]
    }
}
```

```json
{
    "id": "5725c91e38643c19005acced",
    "context": "A pesar de sus cuerpos blandos y gelatinosos, los fósiles que se cree que representan a los ctenóforos, aparentemente sin tentáculos pero con muchas más filas de púas que las formas modernas, han sido encontrados en lagerstätten en los primeros tiempos de la época de la era Cámbrica, hace alrededor de 515 millones de años. La posición de los ctenóforos en el árbol genealógico evolutivo de los animales se ha discutido durante mucho tiempo, y la opinión mayoritaria en la actualidad, basada en la filogenética molecular, es que los cnidarios y los bilaterianos están más estrechamente relacionados entre sí que cualquiera de ellos con los ctenóforos. Un análisis reciente de filogenética molecular concluyó que el antepasado común de todos los ctenóforos modernos era similar a los cidípidos, y que todos los grupos modernos aparecieron relativamente recientemente, probablemente después del evento de extinción del Cretácico-Paleógeno hace 66 millones de años. Las pruebas acumuladas desde la década de 1980 indican que los "cidípidos" no son monofiléticos, es decir, no incluyen a todos y solo a los descendientes de un único antepasado común, ya que todos los demás grupos tradicionales de ctenóforos son descendientes de varios cidípidos.",
    "question": "¿Qué edad tienen los fósiles encontrados que representan los ctenóforos?",
    "answer": {
      "answer_start": [303],
      "text": ["515 millones de años"]
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  A continuación se presentan textos con sus preguntas y respuestas correspondientes.
  ```

- Base prompt template:

  ```text
  Texto: {text}
  Pregunta: {question}
  Respuesta en máximo 3 palabras: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Texto: {text}

  Responda la siguiente pregunta sobre el texto anterior en máximo 3 palabras.

  Pregunta: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset xquad-es
```

### Unofficial: BeleBele-es

This dataset was published in [this paper](https://aclanthology.org/2024.acl-long.44/)
and features multiple-choice reading comprehension questions across 122 languages.

The original dataset contains 900 unique multiple-choice reading comprehension passages
and questions. From these, we use a 256 / 64 / 580 split for training, validation and
testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Texto: Beba alcohol con moderación. Este afecta a cada persona de manera diferente y conocer sus propios límites es sumamente importante. La ingesta excesiva de alcohol puede causar problemas de salud crónicos como daño hepático e incluso ceguera y muerte. El peligro potencial se incrementa con el consumo de alcohol elaborado de forma ilegal. En las bebidas alcohólicas ilegales puede haber varias impurezas amenazantes, como el metanol, capaz de provocar ceguera o incluso la muerte, aun cuando se ingiera poca cantidad.\nPregunta: Según el fragmento, ¿cuál de los siguientes sentidos puede verse afectado por el consumo excesivo de alcohol?\nOpciones:\na. Audición\nb. Vista\nc. Gusto\nd. Olfato",
  "label": "b"
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
  "text": "Texto: El elemento del determinismo cultural se encontraba muy presente en el romanticismo, según estudiosos como Goether, Fichte y Schlegel. En el contexto del Romanticismo, la geografía moldeó a las personas y, con el transcurso del tiempo, se desarrollaron costumbres y culturas relacionadas con esa geografía que, al estar en armonía con la localización de esa sociedad, eran preferibles a leyes que se impusieran de forma arbitraria.\nPregunta: De acuerdo con el texto, ¿qué moldeó a las personas durante el período del Romanticismo?\nOpciones:\na. Leyes\nb. Geografía\nc. Costumbres\nd. Cultura",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Las siguientes son preguntas de opción múltiple (con respuestas).
  ```

- Base prompt template:

  ```text
  Pregunta: {text}
  Opciones:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Respuesta: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Pregunta: {text}
  Opciones:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Responda la pregunta anterior usando solo 'a', 'b', 'c' o 'd', y nada más.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset belebele-es
```

### Unofficial: MultiWikiQA-es

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2509.04111)
and contains Wikipedia articles with LLM-generated questions and answers in 300+
languages.

The original full dataset consists of 5,000 samples in a single split. We use a 1,024 /
256 / 2,048 split for training, validation and testing, respectively, sampled randomly.

Here are a few examples from the training split:

```json
{
    "context": "El moretum es una especie de queso untable tradicional que se servía como acompañamiento de algunos de los platos de la Antigua Roma. Se trata de un tipo de aderezo o salsa cuyos ingredientes se machacan en un mortero, del cual toma el nombre.\n\nCitas \n\nEn el Appendix Vergiliana, obra que, como indica su nombre, se atribuye a Virgilio, se dice que los ingredientes del moretum son hierbas aromáticas, ajo, queso, vinagre, aceite de oliva y sal, y se describe su preparación como desayuno por un campesino. Con respecto a la combinación de los ingredientes, en la línea 103 se lee \n\nEl moretum también es nombrado por Columela en el libro XII de su obra De re rustica.\n\nVéase también \n\n Pesto\n Almodrote\n\nBibliografía \n\n Rodríguez-Pantoja Márquez, Miguel: El \"Moretum\", estudio lingüístico y literario. , n.º 8, 1977, pp. 117 - 148. Departamento de Prehistoria y Arqueología. Universidad de Sevilla. ISSN 0210-7694\n Texto en PDF.\n\nNotas y referencias\n\nEnlaces externos \n\nGastronomía de la Antigua Roma\nDesayunos\nAlimentos untables\nPlatos de queso\nSalsas de Italia",
    "question": "¿Quién es considerado el autor del poema Moretum, que forma parte del Appendix Vergiliana?",
    "answers": {
        "answer_start": array([327]),
        "text": array(["Virgilio"], dtype=object)
    }
}
```

```json
{
    "context": "La culebra-viborera mexicana (Clelia scytalina) también conocida como zopilota de altura, es una especie de serpiente que pertenece a la familia Colubridae. Es nativo del sur de México, América Central y Colombia.\nComo las demás especies de musurana, se alimenta principalmente de otras serpientes, especialmente de serpientes venenosas del género Bothrops.\n\nDescripción \nLos adultos poseen una coloración negra grisácea iridiscente o negra azulada en el dorso. Los juveniles tienen un dorso rojo, cabeza negra y un collar nucal amarillo opaco que está rodeado de pigmento negro; el vientre es color crema inmaculado; escamas dorsales en 17 hileras.\n\nDistribución \nClelia scytalina se distribuye a bajas y moderadas elevaciones (hasta 1,200 ) de Veracruz en la vertiente del Atlántico y desde Jalisco en la vertiente del Pacífico hacia el sur a través de América Central hasta Colombia. Está especie es generalmente rara en México excepto en la Sierra de los Tuxtlas en el sur de Veracruz donde es considerada como relativamente común. Es conocida de localidades dispersas en la vertiente del Atlántico en el centro de Veracruz, Oaxaca, Chiapas, Tabasco, suroeste de Campeche y sur de Quintana roo, y en el vertiente del Pacífico\xa0 en Jalisco, Colima, Guerrero y Chiapas.\n\nHábitat \nEsta serpiente grande y activa habita el bosque caducifolio tropical, el bosque estacional perennifolio y el bosque lluvioso. Es principalmente terrestre y nocturna, pero también se puede encontrar activa durante el día. Por lo general, forrajea por la noche en bosques primarios o secundarios, a menudo a lo largo de arroyos. Se alimenta principalmente de serpientes, incluidas las nauyacas (Bothrops asper) y otras serpientes venenosas, que a veces pueden ser tan largas como ella. Ocasionalmente comen ranas, lagartijas y mamíferos. Clelia scytalina es ovípara.\n\nEstado de conservación \nSe encuentra catalogada dentro de la lista roja de la IUCN como una especie con preocupación menor (LC).\n\nReferencias\n\nEnlaces externos \n . Encyclopedia of Life.\n\nscytalina\nReptiles de América Central\nAnimales descritos en 1867\nTaxones descritos por Edward Drinker Cope",
    "question": "¿Cuál es el límite altitudinal de la distribución de Clelia scytalina?",
    "answers": {
        "answer_start": array([735]),
        "text": array(["1,200"], dtype=object)
    }
}
```

```json
{
    "context": "Coslada Central es una estación de la línea 7 del Metro de Madrid, situada entre la calle de Pablo Neruda y el paseo de Francisco Javier Sauquillo, en el municipio madrileño de Coslada.\n\nOfrece una conexión con la estación de Coslada de las líneas C-2, C-7 y C-8 de Cercanías Madrid, formando ambas un intercambiador de transporte.\n\nHistoria y características \nLa estación fue inaugurada el 5 de mayo de 2007 y está decorada con grandes murales por los andenes y el vestíbulo, realizados por Raúl Díaz Reyes, los cuales, bajo el título de \"De Madrid al cielo\", reflejan diferentes imágenes de cielos de Madrid.\n\nLa estación ha sufrido varias obras de rehabilitación desde su inauguración para garantizar la seguridad y aliviar las grietas que se han formado encima de los túneles por los que discurre el tramo MetroEste. Véase Obras de rehabilitación en Línea 7 para más detalles.\n\nAccesos \nVestíbulo Coslada Central (Metro de Madrid)\n Doctor Fleming C/ Doctor Fleming, s/n (en el parque Doctor Fleming)\n  Ascensor C/ Doctor Fleming, s/n (en el parque Doctor Fleming)\n Renfe Abierto de 6:00 a 0:30 Correspondencia con Cercanías Renfe\nVestíbulo Renfe\n  Luis Braille C/ Luis Braille, s/n (Correspondencia con Cercanías Renfe)\n\nLíneas y conexiones\n\nMetro\n\nCercanías\n\nAutobuses\n\nReferencias\n\nVéase también \n Línea 7 (Metro de Madrid)\n MetroEste\n Estaciones del Metro de Madrid\n Coslada, ,\n\nEnlaces externos y referencias \n\n Ficha de la estación en metromadrid.es\n Página oficial del Metro de Madrid\n\nCoslada\nCoslada\nEstaciones de metro de España inauguradas en 2007",
    "question": "¿Cómo se llaman las obras de arte que adornan esta estación de metro?",
    "answers": {
        "answer_start": array([539]),
        "text": array(["\"De Madrid al cielo\""], dtype=object)
    }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:

  ```text
  A continuación se presentan textos con sus preguntas y respuestas correspondientes.
  ```

- Base prompt template:

  ```text
  Texto: {text}
  Pregunta: {question}
  Respuesta en máximo 3 palabras: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Texto: {text}

  Responda la siguiente pregunta sobre el texto anterior en máximo 3 palabras.

  Pregunta: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset multi-wiki-qa-es
```text

## Knowledge

### MMLU-es

This dataset is a machine translated version of the English [MMLU
dataset](https://openreview.net/forum?id=d7KBjmI3GmQ) and features questions within 57
different topics, such as elementary mathematics, US history and law. The translation to
French was done by the University of Oregon as part of [this
paper](https://aclanthology.org/2023.emnlp-demo.28/), using GPT-3.5-turbo.

The original full dataset consists of 272 / 1,465 / 13,334 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 2,048 split for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
new and there can thus be some overlap between the original validation and test sets and
our validation and test sets.

Here are a few examples from the training split:

```json
{
    "text": "¿Qué método de los siguientes utiliza el método de loci como ayuda para la memoria?\nOpciones:\na. Codificación semántica\nb. Imaginería visual\nc. Señales auditivas\nd. Memoria ecoica",
    "label": "b",
}
```

```json
{
    "text": "Cuando una medida realmente cuantifica lo que afirma medir, decimos que tiene buena\nOpciones:\na. precisión\nb. validez\nc. confiabilidad\nd. valor asociativo",
    "label": "b",
}
```

```json
{
    "text": "Un ranchero, siendo el propietario en un simple título, transfirió la propiedad mediante una escritura de garantía a una mujer. La mujer opignoró la finca a favor de su sobrina para asegurar un préstamo de la sobrina a la mujer por la cantidad de $500,000. La hipoteca fue inmediatamente registrada. Dos años después, la mujer transfirió la finca a un granjero mediante una escritura de renuncia. La mujer, entonces, incumplió con la hipoteca, y la sobrina entabló una acción in personam contra el granjero para recuperar la cantidad adeudada por la hipoteca. Se presume que la escritura de renuncia de la mujer al granjero no hacía referencia a la hipoteca. Es probable que el acreedor hipotecario\nOpciones:\na. tenga éxito, porque la transferencia de la propiedad de la mujer al granjero resultó en una delegación implícita de responsabilidades.\nb. tenga éxito, porque la sobrina era una beneficiaria de tercera parte en la transferencia entre la mujer y el granjero.\nc. no tenga éxito, porque el granjero no prometió pagar la deuda hipotecaria.\nd. no tenga éxito, a menos que el granjero tuviera conocimiento constructivo de la existencia de la hipoteca.",
    "label": "c",
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Las siguientes son preguntas de opción múltiple (con respuestas).
  ```

- Base prompt template:

  ```text
  Pregunta: {text}
  Opciones:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Respuesta: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Pregunta: {text}
  Opciones:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Responda la pregunta anterior usando solo 'a', 'b', 'c' o 'd', y nada más.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset mmlu-es
```

### Unofficial: INCLUDE-es

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
    "text": "Hormona que actúa sobre el metabolismo de agua, sodio, potasio y cloruro de sodio:\nOpciones:\na. Aldosterona\nb. Cortisol\nc. Corticosterona\nd. Cortisona",
    "label": "a",
    "subject": "Medicine"
}
```

```json
{
    "text": "Nervio que inerva a los músculos esternocleidomastoideo y trapecio:\nOpciones:\na. Hipogloso\nb. Espinal\nc. Vago\nd. Acústico",
    "label": "b",
    "subject": "Medicine"
}
```

```json
{
    "text": "Si el precio del bien sustituto disminuye, la curva de la demanda se\nOpciones:\na. expandirá\nb. incrementará\nc. desplazará hacia la izquierda\nd. mantendrá constante",
    "label": "c",
    "subject": "Economics"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Las siguientes son preguntas de opción múltiple (con respuestas).
  ```

- Base prompt template:

  ```text
  Pregunta: {text}
  Respuesta: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Pregunta: {text}

  Responda la pregunta anterior usando solo {labels_str}, y nada más.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset include-es
```

## Common-sense Reasoning

### HellaSwag-es

This dataset is a machine translated version of the English [HellaSwag
dataset](https://aclanthology.org/P19-1472/). The original dataset was based on both
video descriptions from ActivityNet as well as how-to articles from WikiHow. The dataset
was translated by the University of Oregon as part of [this
paper](https://aclanthology.org/2023.emnlp-demo.28/), using GPT-3.5-turbo.

The original full dataset consists of 9,374 samples. We use a 1,024 / 256 / 2,048 split
for training, validation and testing, respectively (so 3,328 samples used in total).

Here are a few examples from the training split:

```json
{
    "text": "[header] Cómo crear tinta de tatuaje de prisión [title] Encuentra una lata o un contenedor de metal. [step] Debe poder contener de 4 a 6 onzas de aceite para bebés, junto con un poco de algodón apretado. Prueba usando una lata de pulimento para botas vacía y limpia.\nOpciones:\na. [title] Usa alcohol isopropílico como lubricante. [step] Mientras que algunos pulidores de escritorio tradicionalmente utilizados para tatuajes se manchan o amenazan, mezcla ¼ taza de agua tibia y ¼ taza de detergente regular para platos.\nb. Si no tienes acceso a un contenedor pre-hecho: usa una herramienta afilada para cortar una lata de aluminio de 12 onzas por la mitad, y usa la mitad inferior como tu contenedor. [substeps] Puede que puedas comprar una lata de pulimento para botas en la tienda de la cárcel.\nc. No se recomienda el vidrio ya que se piensa que es muy frágil y es probable que reaccione mal al metal. [title] Quita cualquier objeto extranjero o bordado del contenedor.\nd. [title] Vierte el pulimento blanco en un tubo de plástico como fluido sellante. [step] Un tubo ligero y bastante delgado funciona mejor como reservorio.",
    "label": "b",
}
```

```json
{
  "text": "Entonces, la niña baja firmemente sus manos hacia su costado, junta sus pies y hace una reverencia, continuando con una rutina de varios movimientos de karate. la niña\nOpciones:\na. luego da una triunfante ola mientras levanta una mano derecha en el aire y continúa su rutina.\nb. cae en un tatami alto en el aire y un hombre se acerca y le ayuda mientras desmonta.\nc. finalmente desmonta y coloca su instrumento en su soporte, sin hacer una reverencia, su postura seria cambia a una de plena concentración mientras levanta sus manos en el aire y eleva sus brazos.\nd. termina su rutina un poco más lejos del punto donde comenzó, baja firmemente sus manos hacia su costado y hace una pequeña reverencia, luego abre sus piernas a la altura de los hombros y vuelve a la misma posición en la que estaba cuando empezó.",
  "label": "d",
}
```

```json
{
"text": "[header] Cómo llevar tu peinado del día a la noche [title] Humedece tu cabello. [step] Crear ondas a partir de un moño es una gran opción para cabello largo. Cuando quieras usar un moño para crear ondas en tu cabello, lo mejor es comenzar con el cabello al menos parcialmente húmedo.\nOpciones:\na. Así que antes de comenzar, usa una toalla para secar en el lugar donde quieres poner el cabello. [substeps] Una buena regla es secar el cabello con una toalla antes de ponerlo en un moño.\nb. Si te lavas el cabello por la mañana, sécalo con secadora o al aire hasta la mitad antes de hacer el moño. Si no planeas lavar tu cabello, rocíalo ligeramente con una botella rociadora llena de agua.\nc. [substeps] El cabello rizado se verá sin esfuerzo y más esponjado con la cabeza húmeda porque es suave y brillante. Si tu cabello no está tan seco como quieres, no te vuelvas loca.\nd. Si quieres dejarlo suelto durante la noche, usa una secadora. [substeps] Una secadora de cabello normalmente funciona mejor.",
"label": "b",
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Las siguientes son preguntas de opción múltiple (con respuestas).
  ```

- Base prompt template:

  ```text
  Pregunta: {text}
  Opciones:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Respuesta: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Pregunta: {text}
  Opciones:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Responda la pregunta anterior usando solo 'a', 'b', 'c' o 'd', y nada más.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset hellaswag-es
```

### Unofficial: GoldenSwag-es

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
  "text": "Cómo desmaquillarse. Empapa un disco de algodón con desmaquillante de ojos. Un desmaquillante de ojos bifásico sirve para la mayoría del maquillaje de ojos. Combina el poder disolvente de un desmaquillante a base de aceite con las cualidades suaves y calmantes del agua limpiadora.\nOpciones:\na. Es una buena opción para el maquillaje de ojos intenso; sólo asegúrate de agitar bien el envase antes de usarlo, ya que la fórmula tiende a separarse. Si utilizas máscara y delineador de ojos resistentes al agua o un maquillaje muy resistente, utiliza un limpiador a base de aceite.\nb. Normalmente, el objetivo es hacer que el proceso de limpieza específico sea una experiencia más agradable, en lugar de que sea completamente autocalmante. Como alternativa, puedes simplemente humedecer tu disco de algodón con una solución de agua fría y aplicarla desde el rabillo del ojo hacia las esquinas interiores.\nc. Compra un bote de este desmaquillante en una farmacia o por Internet. Cuanto más tiempo lo apliques, más oscura será la capa externa de maquillaje de los ojos.\nd. Aunque estos productos son menos caros que los limpiadores faciales normales, no siempre son infalibles. Utilizarlos en exceso afectará a la eficacia de tu maquillaje.",
  "label": "a"
}
```

```json
{
  "text": "Cómo hacer turrón. Forre el molde. Forre el fondo y los lados de un molde de 20 cm por 20 cm con papel pergamino. Resérvalo para utilizarlo más tarde.\nOpciones:\na. Si quieres enfriar el pan sin papel pergamino, vierte 2 cucharadas (45 ml) de azúcar en un bol y mételo en el congelador para que se enfríe. También puede refrigerar la mezcla durante al menos un día.\nb. Lleve el agua a ebullición. En el fuego, ponga el fuego a tope para que el agua hierva.\nc. Verterá el sirope de arce en el cazo después de cocer la miel y el azúcar.. Calienta el agua en el cazo.\nd. Como alternativa, puedes engrasar el fondo y los lados del molde con mantequilla, manteca o spray antiadherente para cocinar. El papel de pergamino facilitará la limpieza del molde.",
  "label": "d"
}
```

```json
{
  "text": "Cómo seguir amamantando después de volver al trabajo. Prepárate. Antes de volver al trabajo, tienes que planificarte y prepararte con tiempo. Esto significa hacer acopio de leche materna extraída (ebm) y establecer la infraestructura necesaria para extraer leche materna con éxito en el trabajo.\nOpciones:\na. Hacer acopio parece consolidar la síntesis eficaz de un ciclo de lactancia sano y constante. Sin embargo, tener un suministro fresco de ebm puede dificultar el mantenimiento de una relación sana y productiva con tu bebé.\nb. Determina la edad ideal de tu hija. Si tu hija sólo tiene seis meses, reserva un poco de tiempo para empezar a amamantarla.\nc. Si inviertes tiempo y esfuerzo en conseguirlo antes, será menos complicado una vez que vuelvas al trabajo. Haz acopio de ebm mientras estés de baja por maternidad.\nd. Escribe todo lo que quieras saber en una hoja aparte y guárdalo para más tarde o sustitúyelo por una nota escrita a mano. Si es posible, planifica tomarte un día libre en el trabajo.",
  "label": "c"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Las siguientes son preguntas de opción múltiple (con respuestas).
  ```

- Base prompt template:

  ```text
  Pregunta: {text}
  Opciones:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Respuesta: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Pregunta: {text}
  Opciones:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Responda la pregunta anterior usando solo 'a', 'b', 'c' o 'd', y nada más.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset goldenswag-es
```

### Unofficial: Winogrande-es

This dataset was published in [this paper](https://doi.org/10.48550/arXiv.2506.19468)
and is a translated and filtered version of the English [Winogrande
dataset](https://doi.org/10.1145/3474381).

The original full dataset consists of 47 / 1,210 samples for training and testing, and
we use 128 of the test samples for validation, resulting in a 47 / 128 / 1,085 split for
training, validation and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Joseph tenía que tener las uñas bien cuidadas para el trabajo, pero no Kevin, porque _ trabajaba en un banco. ¿A qué se refiere el espacio en blanco _?\nOpciones:\na. Joseph\nb. Kevin",
  "label": "a"
}
```

```json
{
  "text": "Craig realmente ama limpiar todo el tiempo pero Derrick no porque _ es muy ordenado. ¿A qué se refiere el espacio en blanco _?\nOpciones:\na. Craig\nb. Derrick",
  "label": "a"
}
```

```json
{
  "text": "Una vez en Polonia, Dennis disfrutó del viaje más que Jason porque _ tenía un conocimiento superficial del idioma polaco. ¿A qué se refiere el espacio en blanco _?\nOpciones:\na. Dennis\nb. Jason",
  "label": "b"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:

  ```text
  Las siguientes son preguntas de opción múltiple (con respuestas).
  ```

- Base prompt template:

  ```text
  Pregunta: {text}
  Opciones:
  a. {option_a}
  b. {option_b}
  Respuesta: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Pregunta: {text}
  Opciones:
  a. {option_a}
  b. {option_b}

  Responda la pregunta anterior usando solo 'a' o 'b', y nada más.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset winogrande-es
```

## Summarisation

### MLSum-es

The dataset was published in [this paper](https://aclanthology.org/2020.emnlp-main.647/)
and is obtained from online newspapers.

The original full dataset consists of 266,367 / 10,358 / 13,920 samples for training,
validation, and testing, respectively. We use 1,024 / 256 / 2,024 samples for training,
validation, and testing, respectively. All our splits are subsets of the original ones.

Here are a few examples from the training split:

```json
{
    "text": "El todopoderoso secretario general de los populares bajo la presidencia de José María Aznar, Francisco Álvarez-Cascos, ha desencadenado en su partido una tormenta en un vaso de agua. Amparándose en una retórica de servicio a Asturias que apenas alcanza a disimular la frustración de sus ambiciones personales, Álvarez-Cascos ha anunciado su baja en el partido de Mariano Rajoy y ha insinuado la creación de una nueva fuerza política para concurrir como candidato a la presidencia de Asturias en las elecciones autonómicas de mayo. Nada tiene de extraño que quien fuera uno de los máximos adalides del 'todo vale' desde la oposición y también desde el Gobierno, aplique ahora esta máxima a su propio partido. Álvarez-Cascos, durante sus años de protagonismo, tensó la vida política española hasta bordear los límites de la estabilidad institucional, arremetiendo contra sus adversarios con instrumentos que despreciaban normas elementales del juego democrático. Su intento de regresar a la política activa, rechazado por la dirección nacional de su partido, no responde al deseo de ofrecer un programa diferente a los asturianos, sino al de saciar su sed de poder tras años de obligada abstinencia. En la comparecencia para explicar las razones de su marcha dejó entrever ajustes de cuentas y venganzas, pero ni una sola idea sobre la que articular el proyecto político que defiende. Es cierto que la democracia interna que Álvarez-Cascos reclama ahora en el PP fue abolida mientras fue él quien tuvo las riendas. Pero no porque sea Álvarez-Cascos su repentino y paradójico abanderado deja de ser una reclamación justa: el PP ha recurrido a la cooptación para decidir la candidatura a la presidencia de Asturias, reafirmándose en un método que aplica a todos los niveles, tanto municipal como autonómico. E, incluso, nacional, como lo atestigua la presidencia de Mariano Rajoy por una decisión personal de su antecesor en el cargo. La aventura de Álvarez-Cascos no solo tendrá dificultades para prosperar por las mezquinas razones que la impulsan, sino por el momento elegido para emprenderla. Un partido que se ve en la antesala del poder cierra filas con su dirección y no destruye sus expectativas desangrándose en luchas internas. Si el PP se encuentra en esta tesitura es por la forma de entender la política de Álvarez-Cascos, pero también por la fragilidad del liderazgo de Rajoy. Dirigentes regionales como la presidenta de la Comunidad de Madrid no dudan en aprovechar cualquier circunstancia para desafiarlo. Álvarez-Cascos ha conseguido mostrar con un único movimiento cuál es la realidad interna de un partido que se considera en vísperas de alcanzar el Gobierno. El vaso de agua donde se desarrolla la ruidosa tormenta que ha desencadenado tiene el valor de un síntoma. Estas son las fuerzas que conviven en el PP y estas son las formas con las que los populares dirimen sus diferencias. * Este artículo apareció en la edición impresa del Martes, 4 de enero de 2011",
    "target_text": "El histórico dirigente del PP se revuelve contra Rajoy al ver frustrada su ambición en Asturias"
}
```

```json
{
    "text": "Eladio Loizaga tiene un bigote fino y un hablar pausado. El Ministro de Relaciones Exteriores de Paraguay, de 66 años, ha estado en Madrid para preparar la visita del presidente de su país, Horacio Cartes, el próximo junio. Después de una charla en Casa de América, Loizaga reflexiona sobre las relaciones diplomáticas en América Latina, la actualidad en Venezuela y Cuba, y los lazos de la región con Estados Unidos, Europa y China. Pregunta. ¿Qué tipo de relación hay entre los países de América Latina? Respuesta. Las relaciones diplomáticas, comerciales y políticas son óptimas. Se basan en respetar el principio de pluralidad y no injerencia en los asuntos internos de cada Estado, a menos que sea una decisión tan grosera que choque con los principios democráticos y las normas constitucionales. En América Latina hemos aprendido a convivir dentro de esa pluralidad, sin que esa pluralidad se uniforme. Cada uno tiene su filosofía y eso tiene que ser respetado. No hay conflictos que pongan en peligro las relaciones entre nosotros. Hemos entendido que podemos convivir con esas diferencias ideológicas. La no inferencia es una piedra angular. P. ¿Incluso en Venezuela con la situación de los presos políticos? R. Paraguay tiene una consolidación democrática plena. En nuestro país ya no hay presos por expresar una idea política distinta a la del Gobierno. Somos miembro del Consejo de Derechos Humanos de Naciones Unidas. En ese sentido, pensamos que acallar voces no contribuye a la libertad de la nación. P. ¿Condena pues las decisiones de Nicolás Maduro? R. Tenemos una posición expresada a través de Unasur. Constituyó una decisión de tres cancilleres, Colombia, Brasil y Ecuador, para cooperar en el diálogo con todos los sectores políticos democráticos de Venezuela. Queremos que Venezuela encuentre una salida conforme a sus propias reglas constitucionales. Hay una línea muy fina en lo que es una injerencia interna, y nosotros somos muy celosos porque la hemos sufrido. Estados Unidos tuvo por mucho tiempo, no un abandono, sino una negligencia benigna hacia América Latina. Como Europa. P. ¿Por qué la mayoría de gobiernos latinoamericanos guardaron silencio? R. Varios gobiernos han mostrado su preocupación y ratificado su posición de que las partes dialoguen, que el Gobierno y la oposición se sienten para encontrar una salida democrática. Tenemos que evitar una salida traumática. Queremos apoyar al pueblo venezolano, porque sabemos las necesidades que están pasando. Estamos en contacto con el Gobierno para ayudar y proveer alimentos y otros productos que se necesitan. P. ¿Apoya la labor que pretende hacer Felipe González? R. No me puedo referir a eso. Hay situaciones en las que, sin desconocer los derechos fundamentales de la persona, hay que tener cierto respeto por el marco interno de cada país. P. ¿Cuál es la salud de los derechos humanos en América Latina? R. Los derechos humanos no se definen hoy solo como derechos políticos. América Latina estaba gobernada por dictaduras, por posiciones extremas, de izquierda y de derecha. Hoy tenemos un adelanto político en toda la región y también la necesidad de ir dando respuesta a los derechos humanos de cuarta generación, la vivienda, la salud, el agua potable... Avanzamos en la lucha contra la pobreza. Y en que los chicos vayan a la escuela. Sin educación no vamos a desarrollarnos. P. ¿Puede América Latina tener una voz única en cuanto a política exterior? R. Hoy no va a ser posible. Sabemos muy bien las posiciones ideológicas de cada uno. En lo posible tratamos de consensuar en la educación, el desarrollo social, pero tener una sola voz política es difícil. Tenemos visiones distintas de cómo vemos el mundo y las relaciones con otros Estados. P. Colombia está en un proceso de paz. ¿Qué es más importante, justicia o paz? R. No es fácil. Hay muchas aristas que deben tenerse en cuenta en el campo penal. El Gobierno busca las medidas jurídicas que den garantía al proceso. P. En otra mesa se sientan Cuba y EE UU. ¿Normalizarán plenamente sus relaciones? R. Era la última rémora de la guerra fría. Obama ha tomado una decisión de mucho coraje, en un momento político interno difícil, y con un sentido pragmático. Señaló que las conductas hacia Cuba no daban resultado y que había que buscar otro camino. La Cumbre de las Américas en Panamá fue histórica. El presidente Castro se expresó con mucha honestidad. Y Obama reconoció que no son perfectos, que tienen problemas. Ojalá se restablezcan las embajadas y el pueblo cubano camine por la senda de la democracia. P. ¿Cuál es el papel del papa Francisco en la política exterior en Latinoamérica? R. El Papa ha tenido un rol muy activo en asuntos de interés general en el mundo, como los problemas de la mujer, el cambio climático, Cuba y Estados Unidos... su presencia en el mundo social es importante. Nos recuerda que existe gente, gente marginada, necesitada. Los países más ricos tienen que contribuir a que tengamos un mundo más equilibrado. P. ¿Qué tipo de relación hay entre EE UU y América Latina? R. Estados Unidos tuvo por mucho tiempo, no un abandono, sino una negligencia benigna hacia América Latina. Como Europa. ¿Quién ocupó ese espacio? China. Con Europa tenemos valores compartidos, y la independencia paraguaya está inspirada en la revolución francesa. De España, como puente, necesitábamos más acompañamiento. China ocupó ese espacio. A Estados Unidos se le mira con diversos cristales. Para Paraguay es un país amigo. P. ¿La relación con Argentina? R. Es un socio comercial importante. Pero hay cuestiones del día a día que pueden enturbiar nuestras relaciones. Queremos hacer un Mercosur más abierto, sin trabas.",
    "target_text": "El ministro paraguayo reflexiona sobre las relaciones diplomáticas en América Latina y la actualidad en Venezuela y Cuba"
}
```

```json
{
    "text": "La Audiencia Nacional ha aprobado extraditar al empresario egipcio Husein Salem a Egipto, donde está siendo juzgado por su supuesta implicación en el caso de corrupción que se sigue contra el expresidente Hosni Mubarak, según informó el Ministerio de Exteriores egipcio. El tribunal también aprobó la entrega de Jaled, hijo de Salem, mientras se estudia si su hija Magda será extraditada. La fiscalía acusa a Salem de haber obtenido favores políticos a cambio de la donación a la familia Mubarak de cinco mansiones, camuflada como una venta ficticia. Esos favores se tradujeron en la asignación de terrenos a su favor y la adquisición fraudulenta de contratos públicos de venta y exportación de gas a Israel, en la localidad de Sharm El Sheik. Esta venta hizo perder al Estado egipcio 536 millones. El empresario, detenido en España el 16 junio de 2011, fue condenado el jueves a 15 años de cárcel por otro caso de corrupción. Y en octubre ya fue sentenciado a siete años, al igual que sus hijos Jaled y Magda, por blanquear 1,7 millones.",
    "target_text": "La fiscalía acusa a Salem de haber obtenido favores políticos a cambio de la donación al exdictador de cinco mansiones, como una venta ficticia",
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:

  ```text
  Los siguientes son artículos de noticias con sus resúmenes.
  ```

- Base prompt template:

  ```text
  Artículo: {text}
  Resumen: {target_text}
  ```

- Instruction-tuned prompt template:

  ```text
  Artículo: {text}

  Escribe un resumen del artículo anterior.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset mlsum-es
```

### Unofficial: DACSA-es

This dataset was published in [this
paper](https://aclanthology.org/2022.naacl-main.434/). The original DACSA dataset
consists of Spanish and Catalan news articles, but this configuration (DACSA-es)
contains only Spanish articles.

The original full dataset consists of 636,596 / 35,376 samples for the training and
validation splits, respectively. The dataset has two test splits with 35,376 and
17,836 samples. The first test split contains samples present in the train and
validation splits, while the second split only has samples that are not present in
the training and validation splits.

We use a 1,024 / 256 / 2,048 split for training, validation, and testing, respectively
(totaling 3,328 samples). All new splits are subsets of the original splits.
Importantly, our new test split is exclusively derived from the original test split
that does not overlap with the training and validation splits.

Here are a few examples from the training split:

```json
{
  "text": "El Popocatépetl está más activo que nunca. La noche del domingo, el volcán mexicano registró una explosión de material incandescente y una nube de vapor, agua y ceniza de hasta 2.000 metros; según Webcams de México, que captó el momento, se trata de la mayor exhalación en tres años.. Las autoridades de Protección Civil y el Centro Nacional de Prevención de Desastres han pedido a la población que no se acerquen al cráter pues se prevé que continúe la actividad volcánica; pidieron también que se cubran alcantarillas y depósitos de agua para evitar la contaminación con ceniza. El volcán ha empezado a registrar cada vez más actividad en los últimos meses. La última gran erupción del Popocatépetl fue durante el mes de diciembre del año 2000 cuando más de 40.000 personas fueron evacuadas.",
  "target_text": "Las cámaras de monitoreo registraron que el volcán mexicano lanzó material incandescente y una emisión continua de vapor de agua."
}
```

```json
{
  "text": "El Power Electronics Valencia derrotó con solvencia al Baloncesto Fuenlabrada (86-60) en la décima jornada de la Liga ACB y mantiene vivo el sueño de clasificarse para la próxima edición de la Copa del Rey, que hace pocas semanas era sólo una utopía. El equipo valenciano controló en todo momento el choque liderado por un brillante Víctor Claver. Asestó un primer golpe al equipo madrileño en los primeros minutos, mantuvo a raya los intentos de remontada de su voluntarioso rival en el segundo tiempo y le desbordó al aprovechar el intenso desgaste al que le había sometido durante treinta minutos. Fiel a la personalidad que se está labrando desde la llegada al banquillo de Svetislav Pesic, el conjunto valenciano saltó a la cancha con una enérgica defensa que ahogó por completo el juego de su rival y que, combinada con diez puntos casi consecutivos de Víctor Claver, le permitió romper el choque en pocos minutos (27-11 m.7).La salida a la pista de los veteranos Ferran Laviña y Salva Guardia le dio al Fuenlabrada cierta tranquilidad. El conjunto madrileño dejó de tratar de igualar el ritmo valenciano , bajó la velocidad de su juego y, por momentos, pareció meterse en el encuentro (29-23, m.12).Pero el que no cambió su apuesta fue el Power Electronics Valencia y con un inmenso despliegue físico de Florent Pietrus volvió a abrir brecha en el marcador, impidiendo que le hicieran daño bajo los aros tanto Ayón como Batista, las dos referencias interiores del Fuenlabrada (50-33, m.20).El paso por el vestuario redujo el acierto ofensivo de los locales y el equipo de Salva Maldonado lo aprovechó. Más concentrado en defensa y bien dirigido por Colom, pudo sacar partido a los puntos de Rabaseda y Kus pero no consiguió que los locales perdieran los nervios. Cuando más atascado estaba el conjunto valenciano dos triples casi consecutivos de Claver permitieron a los locales mantener la calma y encarar el último cuarto con un tranquilizador colchón y, de nuevo, con una agresiva defensa (61-49, m.30).En el último cuarto, el equipo de Pesic recuperó sus mejores armas. Adelantó su línea de presión defensiva, ahogó la salida de balón de los visitantes, les cerró todos los caminos al aro y blindó la zona para no conceder rebotes ofensivos, un recital que en ataque completó con nueve puntos seguidos de Savanovic. Así remató al Fuenlabrada, lo que permite a los locales sumar un nuevo triunfo que mantiene vivo su sueño copero.",
  "target_text": "En un último cuarto espléndido el Power ha terminado arrollando al Fuenlabrada por 86-60 gracias a un excepcional trabajo de equipo en el que ha destacado Víctor Claver con sus 18."
}
```

```json
{
  "text": "El 'tanking', la estrategia de acumular derrotas adrede para elegir antes en el 'draft', pone en jaque a la Liga estadounidens. La frustración de Marc Gaso. Las grandes competiciones deportivas son un reducto comunista de Estados Unidos. Los ingresos se reparten entre todos, hay un techo de gasto (y se penaliza a quien lo supera, aunque pueda permitírselo), los salarios están limitados por convenio y se concede a los peores equipos la mejor oportunidad de conseguir el talento joven que entra en la liga. El objetivo es sacar a flote a los de abajo para favorecer la igualdad pero, hecho el sistema, hecha la trampa: con los playoffs a la vuelta de la esquina, casi un tercio de la liga intenta perder tantos partidos como sea posible. Y sin disimular demasiado. El fenómeno se llama tanking, aunque la palabra es tabú en la NBA. A Mark Cuban, el dueño de los Dallas Mavericks, se le ocurrió decir en alto que lo mejor para su equipo era perder partidos y la liga le puso una multa con 600.000 dólares, una de las más altas de la historia. Además, la NBA se preocupó de que se filtrara a la prensa un mensaje interno en el que advertía a las 30 franquicias de lo negativo que eso era para la imagen de la liga. Hay que guardar las formas, aunque en la clasificación haya un elefante: para nueve equipos ganar es un disgusto. Son franquicias que pierden deliberadamente para tener más opciones de conseguir una buena elección de draft. El tanking puede hacerse en pretemporada, construyendo un equipo flojo, o a mitad de curso si algo se tuerce, sentando a los mejores jugadores, simulando lesiones o dejando marchar a otros. Eso es lo que se está viendo ahora. Los Memphis Grizzlies sentaron durante una semana sin motivo a su mejor anotador, Tyreke Evans, y de aquí a final de temporada Marc Gasol descansará cuando tengan dos partidos seguidos. Los Atlanta Hawks dieron la carta de libertad a Ilyasova y Belinelli, un titular y su mejor jugador de banquillo. Los Bulls anunciaron que en los últimos dos meses sentarían a dos titulares para dar sus minutos a los jóvenes y se han llevado un aviso de la liga. Tan serio, que los ambos han vuelto a jugar... 10 minutos. El orden del draft se decide en una lotería donde se sortean las tres primeras plazas: a más derrotas, más papeletas para ganarla. El resto se establece por orden inverso a la clasificación, así que, con un poco de ojo, todas esas derrotas pueden convertirse en la estrella del futuro. La estrategia más agresiva para perder fue la de los Philadelphia 76ers entre 2013 y 2016. Fue tan descarada que tuvo nombre, The Process (El proceso), y su ideólogo, Sam Hinkie, acabó despedido con un pequeño empujón de la NBA. Pasaron cuatro años en los bajos fondos, firmando algunas de las peores temporadas de la historia, pero sacaron a Joel Embiid y Ben Simmons, dos estrellas para una década. Su futuro es deslumbrante y muchos quieren copiarles, pese a que por cada éxito hay muchos fracasos. Equipos como Sacramento o Phoenix llevan años perdiendo y su futuro sigue negro. Ahora la NBA busca soluciones para un problema que daña seriamente su imagen. En verano aprobó una reforma del draft que intenta combatir esta plaga. Hasta ahora, el peor equipo tenía un 25% de posibilidades de conseguir el número uno y el porcentaje descendía gradualmente. Desde 2019, los tres peores tendrán un 14% y subirán las opciones del resto, para no premiar tanto la derrota. Pero lo cierto es que la reforma es una versión aguada de las propuestas anteriores, mera cosmética que no impedirá que en años como éste, tan cargado de grandes talentos, los equipos sigan dejándose caer. Porque al final toda derrota es asumible si permite obtener a DeAndre Ayton o Luka Doncic, jóvenes que pueden cambiar de inmediato el curso de una franquicia. La NBA es un correcalles donde no les da tiempo casi a bajarse del avión. y aquí en Europa vamos por el mismo camino con dos ligas paralelas.",
  "target_text": "El 'tanking', la estrategia de acumular derrotas adrede para elegir antes en el 'draft', pone en jaque a la Liga estadounidenseLa frustración de Marc Gasol."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:

  ```text
  A continuación se presentan documentos con resúmenes adjuntos.
  ```

- Base prompt template:

  ```text
  Documento: {text}
  Resumen: {target_text}
  ```

- Instruction-tuned prompt template:

  ```text
  Documento: {text}

  Escribe un resumen del documento anterior.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset dacsa-es
```

## Instruction-following

### IFEval-es

This dataset was published [here](https://huggingface.co/datasets/BSC-LT/IFEval_es)
and is a translation of the English IFEval dataset, which was published in [this
paper](https://doi.org/10.48550/arXiv.2311.07911) and contains 541 prompts, each with a
combination of one or more of 25 different constraints. The dataset was manually
translated by a professional translator.

We use the original dataset as the test split, and do not include the other splits, as
we only evaluate models zero-shot and the size is too small to warrant an even smaller
validation set.

Here are a few examples from the test split:

```json
{
    "text": "He intentado que me devuelvan el dinero de un producto que compré por Internet, pero la empresa se niega a reembolsármelo. ¿Puedes ayudarme a escribirles una carta? Quiero que la carta incluya las palabras confianza, marca, cliente, ley, política e inutilizable.",
    "target_text": {
        "instruction_id_list": [
            "es:keywords:existence"
        ],
        "kwargs": [
            {
                "keywords": [
                    "confianza",
                    "marca",
                    "cliente",
                    "ley",
                    "política",
                    "inutilizable"
                ]
            }
        ]
    }
}
```

```json
{
    "text": "Escribe una historia sobre un hombre que está enamorado de una mujer que tiene Tourette. La historia debe tener al menos 4 secciones y cada sección debe empezar con Sección X (donde X es 1, 2, 3, 4) y toda la respuesta debe tener al menos 100 frases.",
    "target_text": {
        "instruction_id_list": [
            "es:detectable_format:multiple_sections",
            "es:length_constraints:number_sentences"
        ],
        "kwargs": [
            {
                "num_sections": 4,
                "section_spliter": "Sección"
            },
            {
                "num_sentences": 100,
                "relation": "at least"
            }
        ]
    }
}
```

```json
{
    "text": "Escribe una entrada de blog sobre las últimas noticias de España, con un título entre paréntesis angulares dobles, es decir, <<título>>, y que tenga menos de 5 frases (excluyendo 5). Las frases deben ser largas para que el número total de palabras de tu respuesta sea de 250 o más.",
    "target_text": {
        "instruction_id_list": [
            "es:detectable_format:title",
            "es:length_constraints:number_sentences",
            "es:length_constraints:number_words"
        ],
        "kwargs": [
            {
            },
            {
                "num_sentences": 5,
                "relation": "less than"
            },
            {
                "num_words": 250,
                "relation": "at least"
            }
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
euroeval --model <model-id> --dataset ifeval-es
```

## European Values

### ValEU-es

This dataset is the official Spanish version of questions from the [European values
study](https://europeanvaluesstudy.eu/). The dataset contains multiple-choice
questions regarding people's values and beliefs across a variety of topics, such as
politics, religion and society.

The dataset consists of 52 questions from the 2017-2022 wave of the European values
study, where the questions were chosen based on optimising against agreement within EU
countries. We use only zero-shot evaluation on this dataset, and thus require no splits.

Here are a few examples from the training split:

```json
{
  "question_id": "A072",
  "text": "Por favor lea atentamente la siguiente lista de organizaciones de voluntariado y diga a cuál de ellas pertenece, en caso de que pertenezca a alguna\nAsociaciones profesionales\nOpciones:\na. No\nb. Sí"
}
```

```json
{
  "question_id": "A079",
  "text": "Por favor lea atentamente la siguiente lista de organizaciones de voluntariado y diga a cuál de ellas pertenece, en caso de que pertenezca a alguna\nOtros grupos/Otras organizaciones\nOpciones:\na. No\nb. Sí"
}
```

```json
{
  "question_id": "D026_05",
  "text": "¿Qué es lo que piensa sobre las afirmaciones siguientes? ¿Está Vd. de acuerdo o en desacuerdo con ellas?\nLos hijos adultos tienen el deber de proporcionar cuidados de larga duración a sus padres.\nOpciones:\na. Muy de acuerdo\nb. De acuerdo\nc. Ni de acuerdo ni en desacuerdo\nd. Desacuerdo\ne. Muy en desacuerdo"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 0
- Prefix prompt:

  ```text
  Las siguientes son preguntas de opción múltiple (con respuestas).
  ```

- Base prompt template:

  ```text
  Pregunta: {text}
  Opciones:
  a. {option_a}
  b. {option_b}
  (...)
  k. {option_k}
  Respuesta: {label}
  ```

- Instruction-tuned prompt template:

  ```text
  Pregunta: {text}
  Opciones:
  a. {option_a}
  b. {option_b}
  (...)
  k. {option_k}

  Responda la pregunta anterior usando solo 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
  'j', o 'k', y nada más.
  ```

You can evaluate this dataset directly as follows:

```bash
euroeval --model <model-id> --dataset valeu-es
```
