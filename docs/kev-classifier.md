# Kev classifier backend

EuroEval can evaluate checkpoints from the upstream
[Kev repository](https://github.com/jaredpalmer/kev) directly in-process. It uses
`kev.checkpoint.Checkpoint` and `LoadOptions` to load Kev's trained pointer head and
either the adapter or full backbone; it does not use a serving API. Complete checkpoints
are recognized by `head.pt` plus either `adapter_config.json` /
`adapter_model.safetensors`, or `config.json` / `model*.safetensors`.

Kev is not the unrelated `kev` package on PyPI. Install upstream from its Git repository
(not `pip install kev`) in the EuroEval environment:

```bash
uv pip install --no-deps 'git+https://github.com/jaredpalmer/kev.git'
```

`--no-deps` avoids replacing EuroEval's pinned `scikit-learn==1.6.1`. Kev currently
requires `scikit-learn>=1.9.1`, `torch>=2.6,<2.9`, and `transformers>=5.17,<6`,
which do not all match EuroEval's environment. The loader only uses Kev's inference
code, but this dependency override is **not validated** against real weights. Check
upstream compatibility before relying on benchmark results. A checkpoint is detected
from its files even if Kev is not installed; loading then reports how to install it.

The adapter supports sequence classification and multiple-choice classification in
zero-shot mode only. It returns a probability for each choice, not generated-token
probabilities; few-shot demonstrations are not supported. Model detection reads small
Hub file metadata and does not fetch model weights; weight files are fetched only when
the model is loaded.
