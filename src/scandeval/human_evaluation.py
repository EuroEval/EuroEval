"""Gradio app for conducting human evaluation of the tasks."""
# mypy: disable-error-code="name-defined"

import importlib.util
import json
import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import click
from datasets import Dataset
from transformers import AutoConfig, AutoTokenizer
from transformers.utils import ModelOutput

from scandeval.benchmark_dataset import BenchmarkDataset

from .benchmark_config_factory import build_benchmark_config
from .config import ModelConfig
from .dataset_configs import SPEED_CONFIG, get_all_dataset_configs
from .dataset_factory import DatasetFactory
from .enums import Framework, ModelType
from .exceptions import NeedsExtraInstalled
from .tasks import NER
from .utils import create_model_cache_dir, enforce_reproducibility

if importlib.util.find_spec("gradio") is not None:
    import gradio as gr

if TYPE_CHECKING:
    pass


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HumanEvaluator:
    """An app for evaluating human performance on the ScandEval benchmark."""

    def __init__(
        self,
        annotator_id: int,
        title: str,
        description: str,
        dummy_model_id: str = "mistralai/Mistral-7B-v0.1",
    ) -> None:
        """Initialize the HumanEvaluator.

        Args:
            annotator_id:
                The annotator ID for the evaluation.
            title:
                The title of the app.
            description:
                The description of the app.
            dummy_model_id:
                The model ID to use for generating prompts.
        """
        self.annotator_id = annotator_id
        self.title = title
        self.description = description
        self.dummy_model_id = dummy_model_id

        self.sample_idx: int
        self.benchmark_dataset: BenchmarkDataset
        self.active_dataset: Dataset

        self.dataset_configs = {
            name: cfg
            for name, cfg in get_all_dataset_configs().items()
            if not cfg.unofficial
        }
        self.tasks = sorted(
            {
                cfg.task.name.replace("-", " ").title()
                for cfg in self.dataset_configs.values()
                if cfg != SPEED_CONFIG
            }
        )
        self.languages = sorted(
            {
                language.name
                for cfg in self.dataset_configs.values()
                if cfg != SPEED_CONFIG
                for language in cfg.languages
                if language.name not in {"Norwegian Bokmål", "Norwegian Nynorsk"}
            }
        )

    def create_app(self) -> "gr.Blocks":
        """Create the Gradio app for human evaluation.

        Returns:
            The Gradio app for human evaluation.
        """
        with gr.Blocks(title=self.title, theme=gr.themes.Monochrome()) as app:
            gr.components.HTML(f"<center><h1>{self.title}</h1></center>")
            gr.components.Markdown(self.description)
            with gr.Row(variant="panel"):
                language_dropdown = gr.Dropdown(
                    label="Language", choices=self.languages
                )
                task_dropdown = gr.Dropdown(label="Task", choices=self.tasks)
                dataset_dropdown = gr.Dropdown(label="Dataset", choices=[""])
            with gr.Row(variant="panel"):
                with gr.Column():
                    task_examples = gr.Markdown("Task Examples")
                with gr.Column():
                    question = gr.Markdown(label="Question")
                    answer = gr.Textbox(label="Answer", interactive=True)
                    submit_button = gr.Button("Submit")

            language_dropdown.change(
                fn=self.update_dataset_choices,
                inputs=[language_dropdown, task_dropdown],
                outputs=dataset_dropdown,
            )
            task_dropdown.change(
                fn=self.update_dataset_choices,
                inputs=[language_dropdown, task_dropdown],
                outputs=dataset_dropdown,
            )
            dataset_dropdown.change(
                fn=partial(self.update_dataset, iteration=self.annotator_id),
                inputs=dataset_dropdown,
                outputs=[task_examples, question, answer],
            )
            submit_button.click(
                fn=partial(self.submit_answer, annotator_id=self.annotator_id),
                inputs=[dataset_dropdown, question, answer],
                outputs=[question, answer],
            )
            answer.submit(
                fn=partial(self.submit_answer, annotator_id=self.annotator_id),
                inputs=[dataset_dropdown, question, answer],
                outputs=[question, answer],
            )
        return app

    def update_dataset_choices(self, language: str, task: str) -> "gr.Dropdown":
        """Update the dataset choices based on the selected language and task.

        Args:
            language:
                The language selected by the user.
            task:
                The task selected by the user.

        Returns:
            A list of dataset names that match the selected language and task.
        """
        if language is None or task is None:
            return gr.Dropdown(choices=[])

        dataset_configs = [
            cfg
            for cfg in get_all_dataset_configs().values()
            if language in {language.name for language in cfg.languages}
            and task.lower().replace(" ", "-") == cfg.task.name
            and not cfg.unofficial
        ]
        assert len(dataset_configs) > 0

        choices = sorted([cfg.name for cfg in dataset_configs])

        logger.info(
            f"User selected {language} and {task}, which resulted in the datasets "
            f"{choices}, with {choices[0]!r} being chosen by default."
        )

        return gr.Dropdown(choices=choices, value=choices[0])

    def update_dataset(self, dataset_name: str, iteration: int) -> tuple[str, str, str]:
        """Update the dataset based on a selected dataset name.

        Args:
            dataset_name:
                The dataset name selected by the user.
            iteration:
                The iteration index of the datasets to evaluate.

        Returns:
            A tuple (task_examples, question, answer) for the selected
            dataset.
        """
        if not dataset_name:
            return "", "", ""

        logger.info(f"User selected dataset {dataset_name} - loading dataset...")

        benchmark_config = build_benchmark_config(
            progress_bar=False,
            save_results=True,
            task=None,
            dataset=None,
            language=[
                language.code
                for cfg in get_all_dataset_configs().values()
                for language in cfg.languages
                if not cfg.unofficial
            ],
            model_language=None,
            dataset_language=None,
            framework=None,
            device=None,
            batch_size=1,
            evaluate_train=False,
            raise_errors=False,
            cache_dir=".scandeval_cache",
            token=None,
            openai_api_key=None,
            prefer_azure=False,
            azure_openai_api_key=None,
            azure_openai_endpoint=None,
            azure_openai_api_version=None,
            force=False,
            verbose=False,
            trust_remote_code=False,
            load_in_4bit=None,
            use_flash_attention=None,
            clear_model_cache=False,
            only_validation_split=True,
            few_shot=True,
            num_iterations=iteration + 1,
            run_with_cli=True,
        )
        dataset_factory = DatasetFactory(benchmark_config=benchmark_config)
        dataset_config = get_all_dataset_configs()[dataset_name]

        model_id = f"human-{iteration}"
        model_config = ModelConfig(
            model_id=model_id,
            revision="main",
            framework=Framework.HUMAN,
            task="text-generation",
            languages=dataset_config.languages,
            model_type=ModelType.HUMAN,
            model_cache_dir=create_model_cache_dir(
                cache_dir=benchmark_config.cache_dir, model_id=model_id
            ),
        )

        self.benchmark_dataset = dataset_factory.build_dataset(dataset=dataset_config)
        self.sample_idx = 0

        dataset_path = (
            Path(".scandeval_cache")
            / "human-evaluation"
            / dataset_name
            / f"human-{iteration}.csv"
        )
        if dataset_path.exists():
            self.active_dataset = Dataset.from_csv(str(dataset_path))
            try:
                while self.active_dataset["answer"][self.sample_idx] is not None:
                    self.sample_idx += 1
            except IndexError:
                scores = self.compute_scores()
                gr.Info(
                    "If you want to evaluate another dataset then please select a new "
                    "one from the menus."
                )
                for metric_name, score in scores.items():
                    gr.Info(f"\n\n{metric_name}: {score:.2%}")
                gr.Info(
                    "You have already completed this dataset! Here are your scores:"
                )
                return "", "", ""
        else:
            rng = enforce_reproducibility(framework=Framework.PYTORCH)
            train, val, tests = self.benchmark_dataset._load_data(rng=rng)
            _, _, tests = self.benchmark_dataset._load_prepared_data(
                train=train,
                val=val,
                tests=tests,
                model_config=model_config,
                hf_model_config=AutoConfig.from_pretrained(self.dummy_model_id),
                tokenizer=AutoTokenizer.from_pretrained(self.dummy_model_id),
                benchmarking_generative_model=True,
            )
            self.active_dataset = (
                tests[iteration]
                .remove_columns(column_names=["input_ids", "attention_mask"])
                .add_column(name="answer", column=[None] * len(tests[iteration]))
            )

        task_examples, question = self.example_to_markdown(
            example=self.active_dataset[self.sample_idx]
        )

        if self.benchmark_dataset.dataset_config.task == NER:
            ner_tags = list()
            for ner_tag in dataset_config.prompt_label_mapping.values():
                if ner_tag not in ner_tags:
                    ner_tags.append(ner_tag)
            answer = json.dumps({ner_tag: [] for ner_tag in ner_tags})
        else:
            answer = ""

        logger.info(
            f"Loaded dataset {dataset_name}, with the following task examples:\n\n"
            f"{task_examples}"
        )

        return task_examples, question, answer

    def submit_answer(
        self, dataset_name: str, question: str, answer: str, annotator_id: int
    ) -> tuple[str, str]:
        """Submit an answer to the dataset.

        Args:
            dataset_name:
                The name of the dataset.
            question:
                The question for the dataset.
            answer:
                The answer to the question.
            annotator_id:
                The annotator ID for the evaluation.

        Returns:
            A tuple (question, answer), with `question` being the next question, and
            `answer` being an empty string.
        """
        if not answer:
            gr.Warning("Please provide an answer before submitting.")
            logger.info("User tried to submit without providing an answer.")
            return question, answer

        # Custom NER validation
        if self.benchmark_dataset.dataset_config.task == NER:
            try:
                json.loads(answer)
            except json.JSONDecodeError:
                gr.Warning("Please provide a valid JSON object as an answer.")
                logger.info("User tried to submit an invalid JSON object as an answer.")
                return question, answer

            if not isinstance(json.loads(answer), dict):
                gr.Warning(
                    "Please provide a JSON object with a dictionary as an answer."
                )
                logger.info(
                    "User tried to submit a JSON object without a dictionary as an answer."
                )
                return question, answer

            ner_tags = list(
                self.benchmark_dataset.dataset_config.prompt_label_mapping.values()
            )
            for ner_tag in ner_tags:
                if ner_tag not in json.loads(answer).keys():
                    gr.Warning(
                        f"Please provide a JSON object with the key {ner_tag!r}."
                    )
                    logger.info(
                        f"User tried to submit a JSON object without the key {ner_tag!r}."
                    )
                    return question, answer

        samples_left = len(self.active_dataset) - self.sample_idx - 1
        if samples_left:
            gr.Info(f"Submitted - {samples_left} to go!")

        # Store the user's answer
        answers = self.active_dataset["answer"]
        answers[self.sample_idx] = answer
        self.active_dataset = self.active_dataset.remove_columns("answer").add_column(
            name="answer", column=answers
        )
        logger.info(
            f"User submitted the answer {answer!r} to the question {question!r}, with "
            f"sample index {self.sample_idx}."
        )

        dataset_path = (
            Path(".scandeval_cache")
            / "human-evaluation"
            / dataset_name
            / f"human-{annotator_id}.csv"
        )
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.active_dataset.to_csv(dataset_path)

        try:
            # Attempt to get the next question
            self.sample_idx += 1
            _, question = self.example_to_markdown(
                example=self.active_dataset[self.sample_idx]
            )

            if self.benchmark_dataset.dataset_config.task == NER:
                ner_tags = list()
                for ner_tag in (
                    self.benchmark_dataset.dataset_config.prompt_label_mapping.values()
                ):
                    if ner_tag not in ner_tags:
                        ner_tags.append(ner_tag)
                answer = json.dumps({ner_tag: [] for ner_tag in ner_tags})
            else:
                answer = ""

        except IndexError:
            scores = self.compute_scores()
            gr.Info(
                "If you want to evaluate another dataset then please select a new "
                "one from the menus."
            )
            for metric_name, score in scores.items():
                gr.Info(f"\n\n{metric_name}: {score:.2%}")
            gr.Info("You have completed the dataset! Here are your scores:")
            question = ""
            answer = ""

        return question, answer

    def example_to_markdown(self, example: dict) -> tuple[str, str]:
        """Convert an example to a Markdown string.

        Args:
            example:
                The example to convert.

        Returns:
            A tuple (task_examples, question) for the example.
        """
        task_examples: str | list[str] = [
            sample.replace("\n", "\n\n")
            for sample in example["text"].split("\n\n")[:-1]
        ]
        task_examples = "\n\n**Example**\n\n".join(task_examples)

        question = "**Question**\n\n"
        question += "\n\n".join(example["text"].split("\n\n")[-1].split("\n")[:-1])
        question += "\n\n" + example["text"].split("\n\n")[-1].split("\n")[-1]

        return task_examples, question

    def compute_scores(self) -> dict[str, float]:
        """Compute the scores for the dataset.

        Returns:
            The scores for the dataset.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.dummy_model_id)
        tokenizer.pad_token = tokenizer.eos_token
        sequences = tokenizer(
            self.active_dataset["answer"],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        ).input_ids
        model_output = ModelOutput(sequences=sequences)
        all_preds = self.benchmark_dataset._extract_labels_from_generation(
            input_batch=self.active_dataset.to_dict(),
            model_output=model_output,
            tokenizer=tokenizer,
        )
        ground_truth = self.active_dataset["label"]
        itr_scores: dict[str, float] = self.benchmark_dataset._compute_metrics(
            model_outputs_and_labels=(all_preds, ground_truth),
            id2label=self.benchmark_dataset.dataset_config.id2label,
        )
        return itr_scores


@click.command()
@click.option(
    "--annotator-id",
    "-id",
    type=int,
    required=True,
    help="""The annotator ID to use for the evaluation. Needs to be between 0 and 10,
    inclusive.""",
)
def main(annotator_id: int) -> None:
    """Start the Gradio app for human evaluation."""
    if importlib.util.find_spec("gradio") is None:
        raise NeedsExtraInstalled(extra="human_evaluation")

    evaluator = HumanEvaluator(
        annotator_id=annotator_id,
        title="ScandEval Human Evaluation",
        description="""
        In this app we will evaluate your performance on a variety of tasks, with the
        goal of comparing human performance to language model performance.

        When you select a language and a task then you will be given a brief
        description of the task, as well as examples of how to solve it. Please read
        through these examples before proceeding with the task.

        Please do not use any additional aids (such as search engines) when completing
        these tasks.

        Note that the Enter key will also submit your answer!
        """,
    )
    evaluator.create_app().launch()


if __name__ == "__main__":
    main()
