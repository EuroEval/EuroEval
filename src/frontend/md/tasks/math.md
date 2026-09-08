# Math

## 📚 Overview

The math task evaluates a model's ability to solve mathematical problems and return the
correct answer. When an Inspect AI `eval.yaml` uses the `math` scorer, EuroEval loads it
as this task.

A math dataset should include a `prompt_template` solver asking the model to box its
answer. EuroEval warns when that instruction is missing because answer extraction
prefers the last `\boxed{...}` or `\fbox{...}` in the completion. It then falls back to
answer markers, delimited mathematics, short whole-text answers, the last non-empty line
and finally the last number in the text.

## 📊 Metrics

Plain numbers are compared exactly, with a percentage interpreted as its value divided
by 100. Expressions are compared as values: EuroEval rewrites the LaTeX used in
benchmark answers — including fractions, roots, powers, products and `\pi` — into
arithmetic and evaluates it with SymPy. For example, `\frac{1}{2}` matches `0.5`, `2\pi`
matches `6.283185307179586`, and `x + y` matches `y + x`.

Values that SymPy compares exactly, such as rationals and `\sqrt{2}`, are never compared
approximately. A one-word answer is read as a name rather than text, making it
case-sensitive, as in Inspect AI; an answer of two or more words is compared as
case-insensitive text.

EuroEval uses a restricted rewrite rather than Inspect AI's full LaTeX grammar.
Unsupported structures, such as matrices, integrals and piecewise braces, are compared
as normalised text.

## 🛠️ How to run

In the command line interface of the [EuroEval Python package](/python-package), run all
math datasets with:

```bash
euroeval --model <model-id> --task math
```
