---
hide:
    - toc
---
# Tasks

<span class="viewport-desktop">Choose a task from the menu on the left to see detailed
information about that task.</span><span class="viewport-mobile">👆 Choose a task
from the top left menu to see detailed information about that task.</span>

## 📚 Overview

This page covers all the evaluation tasks used in EuroEval. These tasks fall under two
categories, corresponding to whether the models should merely _understand_ the input
documents (NLU), or rather they are also required to _generate_ new text (NLG).

### NLU Tasks

NLU tasks are tasks where the model is required to understand the natural language input
and provide an output based on this understanding. The outputs are typically very short,
often just a single label or a couple of words. The performance on these tasks is thus
relevant to you if you primarily aim to use the language models for processing documents
rather than generating entirely new documents. Both encoder and decoder models can be
evaluated on these tasks, enabling you to compare the performance across all language
models out there. The tasks in this category are:

1. [Classification](/tasks/classification)
2. [Sentiment Classification](/tasks/sentiment-classification)
3. [Named Entity Recognition](/tasks/named-entity-recognition)
4. [Linguistic Acceptability](/tasks/linguistic-acceptability)
5. [Reading Comprehension](/tasks/reading-comprehension)
6. [Natural Language Inference](/tasks/natural-language-inference)
7. [Grammatical Error Detection](/tasks/grammatical-error-detection)

### NLG Tasks

NLG tasks are tasks where the model is required to generate natural language output
based on some input. The outputs are typically longer than in NLU tasks, often multiple
paragraphs. The performance on these tasks is thus relevant to you if you aim to use the
language models for generating new documents. Only decoder models can be evaluated on
these tasks, as encoder models do not have the capability to generate text. The tasks in
this category are:

1. [Summarization](/tasks/summarization)
2. [Knowledge](/tasks/knowledge)
3. [Common-sense Reasoning](/tasks/common-sense-reasoning)
4. [Simplification](/tasks/simplification)
5. [European Values](/tasks/european-values)
6. [Instruction-following](/tasks/instruction-following)
7. [Bias Detection](/tasks/bias-detection)
8. [Hallucination Detection](/tasks/hallucination-detection)
9. [Logical Reasoning](/tasks/logical-reasoning)
10. [Grammatical Error Correction](/tasks/grammatical-error-correction)
11. [Tool Calling](/tasks/tool-calling)
12. [Translation](/tasks/translation)

### Other

- [Speed](/tasks/speed): a utility benchmark measuring how quickly a model processes
  input, rather than an NLU or NLG evaluation.
