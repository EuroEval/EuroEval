# Contamination Detection

## 📚 Overview

Contamination detection checks whether a generative model may have been exposed to a
private, prospective canary corpus. It complements EuroEval's public benchmarks, whose
test data may appear in model training corpora, but it cannot determine whether a model
has seen every public benchmark example.

The canary consists of 256 rows arranged into 32 independently keyed groups. Evaluators
receive the corpus text, but not the secret associations, matched controls, or scoring
key. For each row, EuroEval collects a short greedy completion and stores limited
auxiliary evidence with the ordinary evaluation results. Full prompts, private targets,
controls, and keys are not included in the evidence.

Encoder models do not generate canary completions and are marked as not applicable.

## 📊 Interpretation

The canary does not produce a public metric or leaderboard score. During private
leaderboard processing, EuroEval compares exact completions of hidden exposed
associations with matched controls across the independent groups. A positive detection
can cause the canonical model, including its datasets, revisions, and variants, to be
excluded from generated leaderboards. The evidence and decision remain private.

A negative result only means that this detector found no evidence of exposure to the
private canary. It does not prove that the model's training data contains no EuroEval
benchmark data.

## 🛠️ How to run

Contamination detection is an ordinary task. It is automatically included in suite,
ordinary task, and language evaluations whenever no dataset is selected. Targeted
`--dataset` evaluations omit the canary by design. To select it explicitly (including
alongside no other task), run:

```bash
euroeval --model <model-id> --task contamination-detection
```

There is no bespoke canary flag, and no separate Hugging Face login is required.
