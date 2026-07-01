# Token / cost optimisation plan — `create_ragtruth.py`

## Problem

`src/scripts/dataset_creation/create_ragtruth.py` should cost ~$18–20 per
language on `gpt-4o-mini` (~16M input tokens × $0.15/1M + ~26M output tokens ×
$0.60/1M ≈ $18). Observed bills exceed $1000 per language — a ~50–60× blowup.

The estimate is **output-token-dominated** (~$15.60 of the ~$18 is output), so
the blowup is almost certainly on the output side. The changes below are ranked
by expected impact.

---

## 1. Add a `max_tokens` cap — highest priority

**Why:** the request payload sets no output limit:

```python
payload = {
    "model": model,
    "messages": [{"role": "user", "content": translation_prompt}],
    "temperature": 0.4,
}
```

Translation prompts here wrap the text in `======START======` /
`======END======` delimiters over long, messy RAG contexts. Models frequently
**degenerate into repetition loops** and emit output up to the model ceiling
(16,384 tokens for `gpt-4o-mini`). Expected average output is only ~700
tokens/call; a runaway call bills ~20× that. If even a third of calls loop out,
that alone takes $18 → $1000+.

**Change:** in `translate_text` (`create_ragtruth.py`, payload at ~L210), add a
`max_tokens` sized to the input, since a faithful translation is roughly the
length of the source:

```python
# translation output should be ~the length of the input (plus a margin for
# EN->DA/DE expansion); cap generously to truncate repetition-loop runaways
# instead of billing to the model ceiling.
# len(text)/3 overestimates token count (~3 chars/token) so the cap is safe.
max_tokens = min(4096, int(len(text) / 3 * 1.5) + 256)
payload = {
    "model": model,
    "messages": [{"role": "user", "content": translation_prompt}],
    "temperature": 0.4,
    "max_tokens": max_tokens,
}
```

**Base the cap on the length of the text being translated, NOT on any
length limit stated inside the prompt.** ~925/1000 RAGTruth prompts mention a
target length (e.g. `inden for 200 ord`, `100-200 ord lang`, `Besvar ...
kort`), but those describe the length of the *answer the original model was
asked to generate* — this script only *translates* already-existing text, so:

- the real `len(text)` is a stricter, directly-measurable signal than a parsed
  limit;
- the limit only bounds the answer, not the (expensive) prompt-context call;
- the mentions are inconsistent (number / range / vague `kort` / mixed
  translated phrasing), so a regex over them mis-fires.

**Add detection/logging:** when a response comes back with
`finish_reason == "length"`, log a warning with the sample so truncated/looping
translations are visible rather than silently expensive.

---

## 2. Confirm the model actually used

**Why:** `--model` is free-form (default `gpt-4o-mini`). If runs used `gpt-4o`
that is ~16× the price on its own ($18 → ~$290); an o-series/reasoning model
also bills hidden reasoning tokens and rejects `temperature != 1`.

**Change:** no code change strictly required, but:

- Log the resolved model name prominently at start (already logged at L956 —
  verify it in the logs of the expensive runs).
- Consider validating the model against an allowlist, or at least warn loudly if
  a non-`mini` model is selected.

---

## 3. Add jitter to retries; reduce re-billing

**Why:** `translate_text` retries up to 5× on 429/5xx
(`create_ragtruth.py`, `@retry` at ~L153), re-sending the full prompt and being
billed for every attempt (including any partial output). `wait_exponential` has
no jitter, so all `max_workers=30` requests back off in lockstep and re-hammer
the API, worsening the rate-limiting that triggers the retries.

**Change:**

- Switch to `wait_random_exponential(multiplier=1, max=60)` (jittered) from
  `tenacity`.
- Consider lowering the default `--max-workers` (currently 30) if 429s are
  common, and honour `retry-after` (already partially done at L238–244).

---

## 4. Fix resume accounting (drift → re-translation)

**Why:** `num_processed = len(translated_data.samples)` counts only *successful*
samples, but failed/empty samples (`translate_sample` returns `None` at
L442–446 and L503–507) are never stored. On resume,
`remaining_samples = data.samples[num_processed:]` re-includes an
already-processed window equal to the cumulative failure count. Over a long run
restarted many times (crashes / rate limits), this redundant work compounds.

**Change:** track the number of *source* samples consumed, not the number of
successful outputs. Options:

- Persist a `last_processed_index` (source index) alongside the output, and
  resume from that, or
- Store failed/skipped samples as explicit `None`/placeholder markers so
  `len(translated_data.samples)` matches the consumed source count.

---

## 5. (Unconfirmed) Redundant re-translation of shared prompts

**Status: NOT VERIFIED — verify before implementing.**

RAGTruth was originally built by collecting responses from 6 LLMs per source
instance, which *would* mean ~6 samples share the same (long) `prompt` context,
translated independently once per sample — an input-side multiplier.

**However**, this was not confirmed for the `ragtruth_data.json` this script
actually consumes, and the one available empirical check argues *against* it:
the cached translated dataset
(`EuroEval/ragtruth-translated-hallucinations-da-mini`) has 100% unique prompts
(1000/1000 test, 256/256 val). Those are random subsets so the check is not
conclusive, but zero collisions in 1000 draws leans away from heavy duplication.

**Action before any code change:** measure the real ratio on the source file:

```python
import json
from collections import Counter
data = json.load(open("ragtruth_data.json"))
prompts = [s["prompt"] for s in data["samples"]]
print(len(prompts), "rows;", len(set(prompts)), "unique;",
      f"ratio={len(prompts) / len(set(prompts)):.2f}")
```

- If ratio ≈ 1.0 → no input-side redundancy; **drop this item**.
- If ratio > 1 → dedup: translate each unique prompt once, cache by prompt text,
  and reuse the translation across all samples sharing it.

---

## Suggested implementation order

1. **#1 `max_tokens` + `finish_reason` logging** — do first; largest lever and
   low risk.
2. **#2 confirm model** — a logs check, may explain a large constant factor.
3. **#3 retry jitter** — small, reduces re-billing under load.
4. **#4 resume accounting** — correctness fix, matters for long/restarted runs.
5. **#5** — only if the ratio measurement (above) shows duplication.

## Verification

- Add a `--test` run (already supported, L641–642) and log total input/output
  tokens from the `usage` field of each response to measure real per-sample
  cost before a full run.
- After #1, confirm no responses return `finish_reason == "length"` on a sample
  batch.

## Scope note

This is a `src/scripts/` change only — no `CHANGELOG.md` entry needed (changelog
covers `src/euroeval` only).
