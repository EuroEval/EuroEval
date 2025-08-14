"""Callbacks for use while training a model."""

from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.trainer_callback import TrainerCallback
from transformers.training_args import TrainingArguments

from .benchmark_modules.custom import CustomGenerativeModel
from .benchmarker import Benchmarker
from .dataset_configs import get_all_dataset_configs


class EuroEvalCallback(TrainerCallback):
    """A callback that evaluates a model on the EuroEval benchmark during evaluation."""

    def __init__(self, datasets: list[str]) -> None:
        """Initialise the EuroEvalCallback.

        Args:
            datasets:
                A list of EuroEval dataset names to evaluate the model on during
                evaluation.

        Raises:
            ValueError:
                If any of the datasets do not exist in EuroEval.
        """
        # Check that the datasets exist in EuroEval
        all_datasets = get_all_dataset_configs()
        non_existing_datasets = [
            dataset for dataset in datasets if dataset not in all_datasets
        ]
        if non_existing_datasets:
            raise ValueError(
                "The following datasets do not exist: "
                f"{', '.join(non_existing_datasets)}"
            )

        self.benchmarker = Benchmarker(
            dataset=datasets, num_iterations=2, save_results=False
        )

    def on_evaluate(
        self,
        args: TrainingArguments,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        **kwargs,
    ) -> None:
        """Evaluate the model on the EuroEval benchmark during evaluation.

        Args:
            args:
                The training arguments.
            model:
                The model being trained.
            tokenizer:
                The tokenizer used by the model.
            **kwargs:
                Additional keyword arguments.

        Raises:
            ValueError:
                If the output directory is not specified in the training arguments.
        """
        if args.output_dir is None:
            raise ValueError(
                "Output directory must be specified in TrainingArguments for the "
                "EuroEvalCallback to work."
            )
        benchmark_module = CustomGenerativeModel(model=model, tokenizer=tokenizer)
        self.benchmarker.benchmark(model=benchmark_module)
