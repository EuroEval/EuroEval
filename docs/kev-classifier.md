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

`--no-deps` avoids replacing EuroEval's pinned `scikit-learn==1.6.1`; upstream metadata
currently requests scikit-learn >=1.9.1 and may impose Python bounds that conflict with
EuroEval. This command installs only Kev itself, so install/validate the runtime
requirements listed by the upstream project manually in an environment compatible with
both projects. EuroEval checks that `kev.checkpoint` imports before recognizing a
checkpoint; it does not claim that the upstream dependency set is fully compatible.

The adapter supports sequence classification and multiple-choice classification in
zero-shot mode only. Like other zero-shot classifier backends, it does not support
few-shot demonstrations or token-level log probabilities. Model detection reads small
Hub file metadata and does not fetch model weights; weight files are fetched only when
the model is loaded.
