# Logical Reasoning

## 📚 Overview

Logical reasoning evaluates a model's ability to solve constraint satisfaction problems,
specifically Zebra puzzles (also known as Einstein puzzles). The model is given a set of
clues describing relationships between objects across multiple categories, and it must
deduce the correct configuration.

For example, a puzzle might describe 4 houses, each with a person having attributes like
nationality, pet, drink, and colour. Given clues like "The Norwegian lives in the first
house" and "The person who drinks coffee lives in the green house", the model must
determine the complete assignment of attributes to houses.

The puzzles come in two difficulty levels:

- **Easy**: 2 objects (houses) with 3 attributes each
- **Hard**: 4 objects (houses) with 5 attributes each

When evaluating generative models, we allow the model to generate 256 tokens on this
task. Models must produce structured JSON output with keys `object_1`, `object_2`, etc.,
where each value is a list of attributes belonging to that object.

## 📊 Metrics

We report two accuracy metrics for the logical reasoning task:

- **Cell-wise accuracy**: The micro-average accuracy across all individual
  attribute assignments. A prediction is correct for each cell (object × attribute) if
  the predicted attribute matches the gold label. This gives credit for partially
  correct solutions.
  
- **Puzzle-level accuracy**: The strict accuracy where a puzzle is only counted as
  correct if all attribute assignments are correct. This measures the model's ability to
  solve the complete puzzle.

Both metrics are computed after parsing the model's JSON output. Models that fail to
produce valid JSON receive zero scores.

## 🛠️ How to run

In the command line interface of the [EuroEval Python package](/python-package), you
can benchmark your favorite model on the logical reasoning task like so:

```bash
euroeval --model <model-id> --task logical-reasoning
```

To run on a specific difficulty level and language:

```bash
euroeval --model <model-id> --dataset zebra-puzzles-easy-da
euroeval --model <model-id> --dataset zebra-puzzles-hard-da
```
