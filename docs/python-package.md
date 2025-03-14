---
hide:
    - navigation
---
# The `euroeval` Python Package

The `euroeval` Python package is the Python package used to evaluate language models in
EuroEval. This page will give you a brief overview of the package and how to use it.
You can also check out the [full API reference](/api/euroeval/) for more details.


## Installation

To install the package simply write the following command in your favorite terminal:

```bash
$ pip install euroeval[all]
```

This will install the EuroEval package with all extras. You can also install the
minimal version by leaving out the `[all]`, in which case the package will let you know
when an evaluation requires a certain extra dependency, and how you install it.


## Quickstart

### Benchmarking from the Command Line

The easiest way to benchmark pretrained models is via the command line interface. After
having installed the package, you can benchmark your favorite model like so:

```bash
$ euroeval --model <model-id>
```

Here `model` is the HuggingFace model ID, which can be found on the [HuggingFace
Hub](https://huggingface.co/models). By default this will benchmark the model on all
the tasks available. If you want to benchmark on a particular task, then use the
`--task` argument:

```bash
$ euroeval --model <model-id> --task sentiment-classification
```

We can also narrow down which languages we would like to benchmark on. This can be done
by setting the `--language` argument. Here we thus benchmark the model on the Danish
sentiment classification task:

```bash
$ euroeval --model <model-id> --task sentiment-classification --language da
```

Multiple models, datasets and/or languages can be specified by just attaching multiple
arguments. Here is an example with two models:

```bash
$ euroeval --model <model-id1> --model <model-id2>
```

The specific model version/revision to use can also be added after the suffix '@':

```bash
$ euroeval --model <model-id>@<commit>
```

This can be a branch name, a tag name, or a commit id. It defaults to 'main' for latest.

See all the arguments and options available for the `euroeval` command by typing

```bash
$ euroeval --help
```

### Benchmarking from a Script

In a script, the syntax is similar to the command line interface. You simply initialise
an object of the `Benchmarker` class, and call this benchmark object with your favorite
model:

```python
>>> from euroeval import Benchmarker
>>> benchmark = Benchmarker()
>>> benchmark(model="<model>")
```

To benchmark on a specific task and/or language, you simply specify the `task` or
`language` arguments, shown here with same example as above:

```python
>>> benchmark(model="<model>", task="sentiment-classification", language="da")
```

If you want to benchmark a subset of all the models on the Hugging Face Hub, you can
simply leave out the `model` argument. In this example, we're benchmarking all Danish
models on the Danish sentiment classification task:

```python
>>> benchmark(task="sentiment-classification", language="da")
```

### Benchmarking from Docker

A Dockerfile is provided in the repo, which can be downloaded and run, without needing
to clone the repo and installing from source. This can be fetched programmatically by
running the following:

```bash
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

```bash
$ docker build --pull -t euroeval -f Dockerfile.cuda .
```

With the Docker image built, we can now evaluate any model as follows:

```bash
$ docker run -e args="<euroeval-arguments>" --gpus 1 --name euroeval --rm euroeval
```

Here `<euroeval-arguments>` consists of the arguments added to the `euroeval` CLI
argument. This could for instance be `--model <model-id> --task
sentiment-classification`.
