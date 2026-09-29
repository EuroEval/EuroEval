---
hide:
    - toc
---
# Leaderboards

<span class="viewport-desktop">Choose a leaderboard from the menu on the left to see the
results.</span><span class="viewport-mobile">👆 Choose a leaderboard from the top
left menu to see the results.</span>

## 🏷️ Types of Leaderboards

Each language has four leaderboards:

- **Chat Leaderboard**: This leaderboard shows instruction-tuned and reasoning models
  on _all_ [tasks](/tasks): standard NLU and NLG tasks plus tasks specific to these
  models (e.g. instruction following, tool use, bias evaluation). Evaluations here are
  **zero-shot**.
- **Generative Leaderboard**: This leaderboard covers the standard NLU and NLG
  [tasks](/tasks) for any text-generating model, whether base, instruction-tuned, or
  reasoning. Evaluations are **few-shot** by default, unless specified otherwise; the
  model name notes exceptions (e.g. `model-name (zero-shot)`).
- **Understanding Leaderboard**: This leaderboard compares encoder and generative
  models on sequence/token classification, extractive question answering, and the
  multiple-choice tasks eligible for this comparison. Encoder models are evaluated by
  finetuning; generative models use their recorded evaluation mode, including labelled
  zero-shot results where available. It focuses on language understanding rather than
  text generation.
- **All Models Leaderboard**: This leaderboard compares all eligible model families,
  including encoders, generative models, and zero-shot classifiers such as Laya, on
  sequence classification and eligible multiple-choice tasks. Evaluation modes
  differ by model: encoders may be finetuned, generative models may be few-shot or
  labelled zero-shot, and zero-shot classifiers use their classifier inference mode.

Each leaderboard computes its ranking over its own task set. Rank scores and positions
should therefore be compared within a category, not across categories.

## 📊 How to Read the Leaderboards

The main score column is the `Rank score`, showing the
[mean rank score](/methodology) of the model across all the tasks in the leaderboard.
The lower the score, the better the model. The `Rank` column to the left is a dense
ordinal ranking derived from the rank score (see the methodology page for how ties
are decided).

The columns that follow the rank columns are metadata about the model:

- `Type`: The type of model:
  - 🔍 indicates that it is an encoder model (e.g., BERT)
  - 🧠 indicates that it is a base generative model (e.g., GPT-2)
  - 📝 indicates that it is an instruction-tuned model (e.g., ChatGPT)
  - 🤔 indicates that it is a reasoning model (e.g., o1)
- `Parameters`: The total number of parameters in the model, in millions.
- `Vocabulary`: The size of the model's vocabulary, in thousands.
- `Context`: The maximum number of tokens that the model can process at a time.
- `Commercial`: Whether the model can be used for commercial purposes. See [here](/faq)
  for more information.
- `Merge`: Whether the model is a merge of other models.

After these metadata columns, the individual scores for each dataset is shown. Each
dataset has a primary and secondary score - see what these are on the [task
page](/tasks). Lastly, the final columns show the EuroEval version used to benchmark
the given model on each of the datasets.

To read more about the individual datasets, see the [datasets](/datasets) page. If
you're interested in the methodology behind the benchmark, see the
[methodology](/methodology) page.
