<div align='center'>
<img src="https://raw.githubusercontent.com/EuroEval/EuroEval/main/gfx/euroeval.png" height="500" width="372">
</div>

### The robust European language model benchmark.

_(formerly known as ScandEval)_

______________________________________________________________________
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://euroeval.com)
[![PyPI Status](https://badge.fury.io/py/euroeval.svg)](https://pypi.org/project/euroeval/)
[![First paper](https://img.shields.io/badge/arXiv-2304.00906-b31b1b.svg)](https://arxiv.org/abs/2304.00906)
[![Second paper](https://img.shields.io/badge/arXiv-2406.13469-b31b1b.svg)](https://arxiv.org/abs/2406.13469)
[![License](https://img.shields.io/github/license/EuroEval/EuroEval)](https://github.com/EuroEval/EuroEval/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/EuroEval/EuroEval)](https://github.com/EuroEval/EuroEval/commits/main)
[![Code Coverage](https://img.shields.io/badge/Coverage-67%25-yellow.svg)](https://github.com/EuroEval/EuroEval/tree/main/tests)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](https://github.com/EuroEval/EuroEval/blob/main/CODE_OF_CONDUCT.md)


## Maintainers

- Dan Saattrup Nielsen ([@saattrupdan](https://github.com/saattrupdan),
  dan.nielsen@alexandra.dk)
- Kenneth Enevoldsen ([@KennethEnevoldsen](https://github.com/KennethEnevoldsen),
  kenneth.enevoldsen@cas.au.dk)


## Installation
To install the package simply write the following command in your favorite terminal:
```
$ pip install euroeval[all]
```

This will install the EuroEval package with all extras. You can also install the
minimal version by leaving out the `[all]`, in which case the package will let you know
when an evaluation requires a certain extra dependency, and how you install it.

## Quickstart
### Benchmarking from the Command Line
The easiest way to benchmark pretrained models is via the command line interface. After
having installed the package, you can benchmark your favorite model like so:
```
$ euroeval --model <model-id>
```

Here `model` is the HuggingFace model ID, which can be found on the [HuggingFace
Hub](https://huggingface.co/models). By default this will benchmark the model on all
the tasks available. If you want to benchmark on a particular task, then use the
`--task` argument:
```
$ euroeval --model <model-id> --task sentiment-classification
```

We can also narrow down which languages we would like to benchmark on. This can be done
by setting the `--language` argument. Here we thus benchmark the model on the Danish
sentiment classification task:
```
$ euroeval --model <model-id> --task sentiment-classification --language da
```

Multiple models, datasets and/or languages can be specified by just attaching multiple
arguments. Here is an example with two models:
```
$ euroeval --model <model-id1> --model <model-id2>
```

The specific model version/revision to use can also be added after the suffix '@':
```
$ euroeval --model <model-id>@<commit>
```

This can be a branch name, a tag name, or a commit id. It defaults to 'main' for latest.

See all the arguments and options available for the `euroeval` command by typing
```
$ euroeval --help
```

### Benchmarking from a Script
In a script, the syntax is similar to the command line interface. You simply initialise
an object of the `Benchmarker` class, and call this benchmark object with your favorite
model:
```
>>> from euroeval import Benchmarker
>>> benchmark = Benchmarker()
>>> benchmark(model="<model>")
```

To benchmark on a specific task and/or language, you simply specify the `task` or
`language` arguments, shown here with same example as above:
```
>>> benchmark(model="<model>", task="sentiment-classification", language="da")
```

If you want to benchmark a subset of all the models on the Hugging Face Hub, you can
simply leave out the `model` argument. In this example, we're benchmarking all Danish
models on the Danish sentiment classification task:
```
>>> benchmark(task="sentiment-classification", language="da")
```

### Benchmarking from Docker
A Dockerfile is provided in the repo, which can be downloaded and run, without needing
to clone the repo and installing from source. This can be fetched programmatically by
running the following:
```
$ wget https://raw.githubusercontent.com/EuroEval/EuroEval/main/Dockerfile.cuda
```

Next, to be able to build the Docker image, first ensure that the NVIDIA Container
Toolkit is
[installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation)
and
[configured](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker).
Ensure that the the CUDA version stated at the top of the Dockerfile matches the CUDA
version installed (which you can check using `nvidia-smi`). After that, we build the
image as follows:
```
$ docker build --pull -t euroeval -f Dockerfile.cuda .
```

With the Docker image built, we can now evaluate any model as follows:
```
$ docker run -e args="<euroeval-arguments>" --gpus 1 --name euroeval --rm euroeval
```

Here `<euroeval-arguments>` consists of the arguments added to the `euroeval` CLI
argument. This could for instance be `--model <model-id> --task
sentiment-classification`.


### Reproducing the datasets
All datasets used in this project are generated using the scripts located in the
[src/scripts](src/scripts) folder. To reproduce a dataset, run the corresponding script
with the following command

```shell
$ uv run src/scripts/<name-of-script>.py
```

Replace <name-of-script> with the specific script you wish to execute, e.g.,

```shell
$ uv run src/scripts/create_allocine.py
```

## Contributors :pray:

A huge thank you to all the contributors who have helped make this project a success!

<a href="https://github.com/peter-sk"><img src="https://avatars.githubusercontent.com/u/6168908" width=50 alt="Contributor avatar for peter-sk"/></a>
<a href="https://github.com/AJDERS"><img src="https://avatars.githubusercontent.com/u/38854604" width=50 alt="Contributor avatar for AJDERS"/></a>
<a href="https://github.com/oliverkinch"><img src="https://avatars.githubusercontent.com/u/71556498" width=50 alt="Contributor avatar for oliverkinch"/></a>
<a href="https://github.com/versae"><img src="https://avatars.githubusercontent.com/u/173537" width=50 alt="Contributor avatar for versae"/></a>
<a href="https://github.com/viggo-gascou"><img src="https://avatars.githubusercontent.com/u/94069687" width=50 alt="Contributor avatar for viggo-gascou"/></a>
<a href="https://github.com/mathiasesn"><img src="https://avatars.githubusercontent.com/u/27091759" width=50 alt="Contributor avatar for mathiasesn"/></a>
<a href="https://github.com/Alkarex"><img src="https://avatars.githubusercontent.com/u/1008324" width=50 alt="Contributor avatar for Alkarex"/></a>
<a href="https://github.com/marksverdhei"><img src="https://avatars.githubusercontent.com/u/46672778" width=50 alt="Contributor avatar for marksverdhei"/></a>
<a href="https://github.com/Mikeriess"><img src="https://avatars.githubusercontent.com/u/19728563" width=50 alt="Contributor avatar for Mikeriess"/></a>
<a href="https://github.com/pakagronglb"><img src="https://avatars.githubusercontent.com/u/178713124" width=50 alt="Contributor avatar for pakagronglb"/></a>
<a href="https://github.com/ThomasKluiters"><img src="https://avatars.githubusercontent.com/u/8137941" width=50 alt="Contributor avatar for ThomasKluiters"/></a>
<a href="https://github.com/BramVanroy"><img src="https://avatars.githubusercontent.com/u/2779410" width=50 alt="Contributor avatar for BramVanroy"/></a>
<a href="https://github.com/peregilk"><img src="https://avatars.githubusercontent.com/u/9079808" width=50 alt="Contributor avatar for peregilk"/></a>

### Special Thanks
- Thanks to [Google](https://google.com/) for sponsoring Gemini credits as part of their
  [Google Cloud for Researchers Program](https://cloud.google.com/edu/researchers).
- Thanks [@Mikeriess](https://github.com/Mikeriess) for evaluating many of the larger
  models on the leaderboards.
- Thanks to [OpenAI](https://openai.com/) for sponsoring OpenAI credits as part of their
  [Researcher Access Program](https://openai.com/form/researcher-access-program/).
- Thanks to [UWV](https://www.uwv.nl/) and [KU
  Leuven](https://www.arts.kuleuven.be/ling/ccl) for sponsoring the Azure OpenAI
  credits used to evaluate GPT-4-turbo in Dutch.
- Thanks to [Miðeind](https://mideind.is/english.html) for sponsoring the OpenAI
  credits used to evaluate GPT-4-turbo in Icelandic and Faroese.
- Thanks to [CHC](https://chc.au.dk/) for sponsoring the OpenAI credits used to
  evaluate GPT-4-turbo in German.


## Citing EuroEval
If you want to cite the framework then feel free to use this:

```
@article{nielsen2024encoder,
  title={Encoder vs Decoder: Comparative Analysis of Encoder and Decoder Language Models on Multilingual NLU Tasks},
  author={Nielsen, Dan Saattrup and Enevoldsen, Kenneth and Schneider-Kamp, Peter},
  journal={arXiv preprint arXiv:2406.13469},
  year={2024}
}
@inproceedings{nielsen2023scandeval,
  author = {Nielsen, Dan Saattrup},
  booktitle = {Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)},
  month = may,
  pages = {185--201},
  title = {{ScandEval: A Benchmark for Scandinavian Natural Language Processing}},
  year = {2023}
}
```
