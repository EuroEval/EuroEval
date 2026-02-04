"""Minimal COMET regression metric implementation (in-repo)."""

from __future__ import annotations

import collections.abc as c
import logging
import typing as t
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from transformers import (
    BertConfig,
    BertModel,
    BertTokenizerFast,
    XLMRobertaConfig,
    XLMRobertaModel,
    XLMRobertaTokenizerFast,
)

from ..exceptions import InvalidBenchmark
from ..logging_utils import log, log_once
from .base import Metric

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import BenchmarkConfig, DatasetConfig


class Encoder(torch.nn.Module):
    """Minimal encoder interface for COMET."""

    tokenizer: t.Any
    model: t.Any

    @property
    def output_units(self) -> int:
        raise NotImplementedError

    @property
    def max_positions(self) -> int:
        raise NotImplementedError

    @property
    def num_layers(self) -> int:
        raise NotImplementedError

    @property
    def size_separator(self) -> int:
        raise NotImplementedError

    @property
    def uses_token_type_ids(self) -> bool:
        raise NotImplementedError

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model: str,
        load_pretrained_weights: bool = True,
        local_files_only: bool = False,
    ) -> "Encoder":
        raise NotImplementedError

    def freeze_embeddings(self) -> None:
        raise NotImplementedError

    def prepare_sample(self, sample: list[str]) -> dict[str, torch.Tensor]:
        tokenizer_output = self.tokenizer(
            sample,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_positions - 2,
        )
        return tokenizer_output


class BERTEncoder(Encoder):
    """BERT encoder."""

    def __init__(
        self,
        pretrained_model: str,
        load_pretrained_weights: bool = True,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained(
            pretrained_model, use_fast=True, local_files_only=local_files_only
        )
        if load_pretrained_weights:
            self.model = BertModel.from_pretrained(
                pretrained_model, add_pooling_layer=False
            )
        else:
            self.model = BertModel(
                BertConfig.from_pretrained(
                    pretrained_model, local_files_only=local_files_only
                ),
                add_pooling_layer=False,
            )
        self.model.encoder.output_hidden_states = True

    @property
    def output_units(self) -> int:
        return self.model.config.hidden_size

    @property
    def max_positions(self) -> int:
        return self.model.config.max_position_embeddings - 2

    @property
    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers + 1

    @property
    def size_separator(self) -> int:
        return 1

    @property
    def uses_token_type_ids(self) -> bool:
        return True

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model: str,
        load_pretrained_weights: bool = True,
        local_files_only: bool = False,
    ) -> "Encoder":
        return BERTEncoder(pretrained_model, load_pretrained_weights, local_files_only)

    def freeze_embeddings(self) -> None:
        for param in self.model.embeddings.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        **kwargs: t.Any,
    ) -> dict[str, torch.Tensor]:
        last_hidden_states, pooler_output, all_layers = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=False,
        )
        return {
            "sentemb": pooler_output,
            "wordemb": last_hidden_states,
            "all_layers": all_layers,
            "attention_mask": attention_mask,
        }


class XLMREncoder(BERTEncoder):
    """XLM-RoBERTa encoder."""

    def __init__(
        self,
        pretrained_model: str,
        load_pretrained_weights: bool = True,
        local_files_only: bool = False,
    ) -> None:
        super(Encoder, self).__init__()
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(
            pretrained_model, local_files_only=local_files_only
        )
        if load_pretrained_weights:
            self.model = XLMRobertaModel.from_pretrained(
                pretrained_model, add_pooling_layer=False
            )
        else:
            self.model = XLMRobertaModel(
                XLMRobertaConfig.from_pretrained(
                    pretrained_model, local_files_only=local_files_only
                ),
                add_pooling_layer=False,
            )
        self.model.encoder.output_hidden_states = True

    @property
    def output_units(self) -> int:
        return self.model.config.hidden_size

    @property
    def max_positions(self) -> int:
        return self.model.config.max_position_embeddings - 2

    @property
    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers + 1

    @property
    def size_separator(self) -> int:
        return 2

    @property
    def uses_token_type_ids(self) -> bool:
        return False

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model: str,
        load_pretrained_weights: bool = True,
        local_files_only: bool = False,
    ) -> "Encoder":
        return XLMREncoder(pretrained_model, load_pretrained_weights, local_files_only)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: t.Any
    ) -> dict[str, torch.Tensor]:
        last_hidden_states, _, all_layers = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=False,
        )
        return {
            "sentemb": last_hidden_states[:, 0, :],
            "wordemb": last_hidden_states,
            "all_layers": all_layers,
            "attention_mask": attention_mask,
        }


ENCODER_MAP: dict[str, t.Type[Encoder]] = {
    "BERT": BERTEncoder,
    "XLM-RoBERTa": XLMREncoder,
}


def average_pooling(
    tokens: torch.Tensor,
    embeddings: torch.Tensor,
    mask: torch.Tensor,
    padding_index: int,
) -> torch.Tensor:
    padding_mask = tokens.eq(padding_index).unsqueeze(-1)
    wordemb = embeddings.float().masked_fill_(padding_mask, 0.0).type_as(embeddings)
    sentemb = torch.sum(wordemb, 1)
    sum_mask = mask.unsqueeze(-1).expand(embeddings.size()).float().sum(1)
    return sentemb / sum_mask


def max_pooling(
    tokens: torch.Tensor, embeddings: torch.Tensor, padding_index: int
) -> torch.Tensor:
    padding_mask = tokens.eq(padding_index).unsqueeze(-1)
    wordemb = embeddings.float().masked_fill_(padding_mask, float("-inf")).type_as(
        embeddings
    )
    return wordemb.max(dim=1)[0]


class LayerwiseAttention(torch.nn.Module):
    """Layer-wise attention with optional sparsemax."""

    def __init__(
        self,
        num_layers: int,
        layer_norm: bool = False,
        dropout: float | None = None,
        layer_transformation: str = "softmax",
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layer_norm = layer_norm
        self.dropout = dropout

        if layer_transformation == "sparsemax":
            try:
                from entmax import sparsemax
            except Exception:
                log_once(
                    "entmax not available; falling back to softmax for COMET.",
                    level=logging.WARNING,
                )
                self.transform_fn = torch.softmax
            else:
                self.transform_fn = sparsemax
        else:
            self.transform_fn = torch.softmax

        self.scalar_parameters = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.FloatTensor([0.0]), requires_grad=True)
                for _ in range(num_layers)
            ]
        )
        self.gamma = torch.nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

        if self.dropout:
            dropout_mask = torch.zeros(len(self.scalar_parameters))
            dropout_fill = torch.empty(len(self.scalar_parameters)).fill_(-1e20)
            self.register_buffer("dropout_mask", dropout_mask)
            self.register_buffer("dropout_fill", dropout_fill)

    def forward(
        self, tensors: list[torch.Tensor], mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if len(tensors) != self.num_layers:
            raise InvalidBenchmark(
                "LayerwiseAttention received the wrong number of layers."
            )

        weights = torch.cat([p for p in self.scalar_parameters])
        if self.training and self.dropout:
            weights = torch.where(
                self.dropout_mask.uniform_() > self.dropout,
                weights,
                self.dropout_fill,
            )
        normed_weights = self.transform_fn(weights, dim=0)
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)
        pieces = [w * t for w, t in zip(normed_weights, tensors)]
        return self.gamma * sum(pieces)


class FeedForward(torch.nn.Module):
    """Feed-forward head used by COMET."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden_sizes: list[int] | None = None,
        activations: str = "Tanh",
        final_activation: str | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_sizes = hidden_sizes or [3072, 1024]
        modules: list[torch.nn.Module] = [
            torch.nn.Linear(in_dim, hidden_sizes[0]),
            getattr(torch.nn, activations.title())(),
            torch.nn.Dropout(dropout),
        ]
        for i in range(1, len(hidden_sizes)):
            modules.append(torch.nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            modules.append(getattr(torch.nn, activations.title())())
            modules.append(torch.nn.Dropout(dropout))
        modules.append(torch.nn.Linear(hidden_sizes[-1], int(out_dim)))
        if final_activation is not None:
            modules.append(getattr(torch.nn, final_activation.title())())
        self.ff = torch.nn.Sequential(*modules)

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        ff_dtypes = {param.dtype for param in self.ff.parameters()}
        if ff_dtypes == {torch.float16} and in_features.dtype != torch.float16:
            in_features = in_features.to(torch.float16)
        return self.ff(in_features)


@dataclass
class RegressionPrediction:
    score: torch.Tensor


class CometRegressionModel(torch.nn.Module):
    """Minimal COMET regression model for inference."""

    def __init__(self, hparams: dict[str, t.Any]) -> None:
        super().__init__()
        encoder_model = hparams.get("encoder_model", "XLM-RoBERTa")
        encoder_cls = ENCODER_MAP.get(encoder_model)
        if encoder_cls is None:
            raise InvalidBenchmark(
                f"Unsupported COMET encoder model: {encoder_model!r}."
            )

        self.encoder = encoder_cls.from_pretrained(
            hparams["pretrained_model"],
            load_pretrained_weights=False,
            local_files_only=False,
        )

        self.pool = hparams.get("pool", "avg")
        self.layer = hparams.get("layer", "mix")
        self.use_context = False

        self.layerwise_attention: LayerwiseAttention | None = None
        if self.layer == "mix":
            self.layerwise_attention = LayerwiseAttention(
                num_layers=self.encoder.num_layers,
                layer_norm=hparams.get("layer_norm", True),
                dropout=hparams.get("dropout", 0.1),
                layer_transformation=hparams.get("layer_transformation", "softmax"),
            )

        hidden_sizes = hparams.get("hidden_sizes", [3072, 1024])
        activations = hparams.get("activations", "Tanh")
        final_activation = hparams.get("final_activation", None)
        dropout = hparams.get("dropout", 0.1)

        self.estimator = FeedForward(
            in_dim=self.encoder.output_units * 6,
            hidden_sizes=hidden_sizes,
            activations=activations,
            final_activation=final_activation,
            dropout=dropout,
        )

    def get_sentence_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoder_out = self.encoder(
            input_ids, attention_mask, token_type_ids=token_type_ids
        )
        if self.layerwise_attention is not None:
            embeddings = self.layerwise_attention(
                encoder_out["all_layers"], attention_mask
            )
        elif isinstance(self.layer, int) and 0 <= self.layer < self.encoder.num_layers:
            embeddings = encoder_out["all_layers"][self.layer]
        else:
            raise InvalidBenchmark(f"Invalid COMET layer setting: {self.layer!r}.")

        if self.pool == "max":
            sentemb = max_pooling(
                input_ids, embeddings, self.encoder.tokenizer.pad_token_id
            )
        elif self.pool == "avg":
            sentemb = average_pooling(
                input_ids,
                embeddings,
                attention_mask,
                self.encoder.tokenizer.pad_token_id,
            )
        elif self.pool == "cls":
            sentemb = embeddings[:, 0, :]
        else:
            raise InvalidBenchmark(f"Invalid COMET pooling: {self.pool!r}.")

        return sentemb

    def forward(
        self,
        src_input_ids: torch.Tensor,
        src_attention_mask: torch.Tensor,
        mt_input_ids: torch.Tensor,
        mt_attention_mask: torch.Tensor,
        ref_input_ids: torch.Tensor,
        ref_attention_mask: torch.Tensor,
        src_token_type_ids: torch.Tensor | None = None,
        mt_token_type_ids: torch.Tensor | None = None,
        ref_token_type_ids: torch.Tensor | None = None,
    ) -> RegressionPrediction:
        src_sentemb = self.get_sentence_embedding(
            src_input_ids, src_attention_mask, token_type_ids=src_token_type_ids
        )
        mt_sentemb = self.get_sentence_embedding(
            mt_input_ids, mt_attention_mask, token_type_ids=mt_token_type_ids
        )
        ref_sentemb = self.get_sentence_embedding(
            ref_input_ids, ref_attention_mask, token_type_ids=ref_token_type_ids
        )

        diff_ref = torch.abs(mt_sentemb - ref_sentemb)
        diff_src = torch.abs(mt_sentemb - src_sentemb)
        prod_ref = mt_sentemb * ref_sentemb
        prod_src = mt_sentemb * src_sentemb

        embedded_sequences = torch.cat(
            (mt_sentemb, ref_sentemb, prod_ref, diff_ref, prod_src, diff_src),
            dim=1,
        )
        return RegressionPrediction(score=self.estimator(embedded_sequences).view(-1))

    def prepare_sample(self, sample: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        inputs = {k: [dic[k] for dic in sample] for k in sample[0]}
        src_inputs = self.encoder.prepare_sample(inputs["src"])
        mt_inputs = self.encoder.prepare_sample(inputs["mt"])
        ref_inputs = self.encoder.prepare_sample(inputs["ref"])

        src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
        mt_inputs = {"mt_" + k: v for k, v in mt_inputs.items()}
        ref_inputs = {"ref_" + k: v for k, v in ref_inputs.items()}
        return {**src_inputs, **mt_inputs, **ref_inputs}


def _load_comet_checkpoint(
    model_id: str, cache_dir: str
) -> tuple[Path, dict[str, t.Any]]:
    snapshot_path = snapshot_download(repo_id=model_id, cache_dir=cache_dir)
    checkpoint_path = Path(snapshot_path) / "checkpoints" / "model.ckpt"
    hparams_path = Path(snapshot_path) / "hparams.yaml"
    if not checkpoint_path.exists() or not hparams_path.exists():
        raise InvalidBenchmark(
            f"COMET files missing for {model_id!r} in {snapshot_path}."
        )
    with hparams_path.open() as f:
        hparams = yaml.safe_load(f)
    return checkpoint_path, hparams


class CometMetric(Metric):
    """Reference-based COMET regression metric."""

    def __init__(
        self,
        name: str,
        pretty_name: str,
        model_id: str,
        batch_size: int = 8,
        postprocessing_fn: t.Callable[[float], tuple[float, str]] | None = None,
    ) -> None:
        super().__init__(
            name=name, pretty_name=pretty_name, postprocessing_fn=postprocessing_fn
        )
        self.model_id = model_id
        self.batch_size = batch_size
        self.model: CometRegressionModel | None = None
        self.device: torch.device | None = None

    def download(self, cache_dir: str) -> "CometMetric":
        checkpoint_path, hparams = _load_comet_checkpoint(
            model_id=self.model_id, cache_dir=cache_dir
        )
        model = CometRegressionModel(hparams=hparams)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            log(
                f"Unexpected COMET keys: {unexpected[:5]}",
                level=logging.DEBUG,
            )
        if missing:
            log(
                f"Missing COMET keys: {missing[:5]}",
                level=logging.DEBUG,
            )
        self.model = model
        self.model.eval()
        return self

    def __call__(
        self,
        predictions: c.Sequence,
        references: c.Sequence,
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
    ) -> float | None:
        if dataset is None:
            raise InvalidBenchmark("COMET requires the dataset to obtain sources.")

        if self.model is None:
            self.download(cache_dir=benchmark_config.cache_dir)
        assert self.model is not None

        if self.device != benchmark_config.device:
            self.device = benchmark_config.device
            self.model.to(self.device)
            self.model.eval()

        sources = dataset["text"]
        if not len(sources) == len(predictions) == len(references):
            raise InvalidBenchmark(
                "COMET expects sources, predictions, and references to be the same "
                f"length, got {len(sources)}, {len(predictions)}, and {len(references)}."
            )

        samples = [
            {"src": src, "mt": pred, "ref": ref}
            for src, pred, ref in zip(sources, predictions, references)
        ]

        scores: list[torch.Tensor] = []
        for i in range(0, len(samples), self.batch_size):
            batch_samples = samples[i : i + self.batch_size]
            model_inputs = self.model.prepare_sample(batch_samples)
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
            with torch.inference_mode():
                preds = self.model(**model_inputs).score
            scores.append(preds.detach().cpu())

        mean_score = torch.cat(scores).mean().item()
        return float(mean_score)


comet_metric = CometMetric(
    name="comet",
    pretty_name="COMET",
    model_id="Unbabel/wmt22-comet-da",
    postprocessing_fn=lambda x: (x, f"{x:.4f}"),
)
