# Kev classifier backend

EuroEval can evaluate checkpoints from the upstream
[Kev repository](https://github.com/jaredpalmer/kev) directly in-process. It uses
`kev.checkpoint.Checkpoint` and `LoadOptions` to load Kev's trained pointer head and
either the adapter or full backbone; it does not use a serving API. Complete checkpoints
are recognized by `head.pt` plus either `adapter_config.json` /
`adapter_model.safetensors`, or `config.json` / `model*.safetensors`.

Kev is not the unrelated `kev` package on PyPI. Install the upstream project from its
repository into the EuroEval environment before selecting a checkpoint. EuroEval
intentionally does not add it as a mandatory dependency or install it from PyPI: the
current upstream dependency metadata requires scikit-learn >=1.9.1, conflicting with
EuroEval's pinned 1.6.1, and upstream Python bounds may also differ. Do not resolve this
by upgrading EuroEval's scikit-learn pin; use an environment/dependency revision
compatible with both projects.

The adapter supports sequence classification and multiple-choice classification in
zero-shot mode only. Like other zero-shot classifier backends, it does not support
few-shot demonstrations or token-level log probabilities. Model detection reads small
Hub file metadata and does not fetch model weights; weight files are fetched only when
the model is loaded.
