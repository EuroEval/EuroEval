"""Gradio app for conducting human evaluation of the tasks."""

import importlib.util
import json
import logging
import typing as t
from collections import defaultdict
from functools import partial
from pathlib import Path

import click
from datasets import Dataset

from .benchmark_config_factory import build_benchmark_config
from .data_loading import load_data
from .data_models import BenchmarkResult, GenerativeModelOutput
from .dataset_configs import SPEED_CONFIG, get_all_dataset_configs
from .enums import GenerativeType, TaskGroup
from .exceptions import NeedsExtraInstalled
from .scores import aggregate_scores
from .task_group_utils import (
    question_answering,
    sequence_classification,
    text_to_text,
    token_classification,
)
from .tasks import NER
from .utils import enforce_reproducibility

if importlib.util.find_spec("gradio") is not None:
    import gradio as gr
    from gradio.components import HTML, Button, Dropdown, Markdown, Textbox

if t.TYPE_CHECKING:
    from .types import ComputeMetricsFunction, ExtractLabelsFunction, ScoreDict

logger = logging.getLogger("euroeval")


class HumanEvaluator:
    """An app for evaluating human performance on the EuroEval benchmark."""

    def __init__(
        self,
        annotator_id: int,
        title: str,
        description: str,
        dummy_model_id: str = "mistralai/Mistral-7B-v0.1",
    ) -> None:
        """Initialise the HumanEvaluator.

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

        self.extract_labels_from_generation: "ExtractLabelsFunction"
        self.compute_metrics: "ComputeMetricsFunction"

    def create_app(self) -> "gr.Blocks":
        """Create the Gradio app for human evaluation.

        Returns:
            The Gradio app for human evaluation.
        """
        with gr.Blocks(title=self.title, theme=gr.themes.Monochrome()) as app:
            HTML(f"<center><h1>{self.title}</h1></center>")
            Markdown(self.description)
            with gr.Row(variant="panel"):
                language_dropdown = Dropdown(label="Language", choices=self.languages)
                task_dropdown = Dropdown(label="Task", choices=self.tasks)
                dataset_dropdown = Dropdown(label="Dataset", choices=[""])
            with gr.Row(variant="panel"):
                with gr.Column():
                    task_examples = Markdown("Task Examples", visible=False)
                with gr.Column():
                    question = Markdown(label="Question", visible=False)
                    with gr.Row():
                        ner_tag_dropdown = Dropdown(
                            label="Entity type",
                            choices=[""],
                            interactive=True,
                            visible=False,
                            scale=0.5,  # type: ignore[arg-type]
                        )
                        ner_tag_answer = Textbox(
                            label="Entity", interactive=True, visible=False, scale=1
                        )
                        with gr.Column(scale=0.2):  # type: ignore[arg-type]
                            ner_tag_add_button = Button("Add entity", visible=False)
                            ner_tag_reset_button = Button(
                                "Reset entities", visible=False
                            )
                    answer = Textbox(label="Answer", visible=False)
                    submit_button = Button("Submit", visible=False)

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
                outputs=[
                    task_examples,
                    question,
                    ner_tag_dropdown,
                    ner_tag_answer,
                    ner_tag_add_button,
                    ner_tag_reset_button,
                    answer,
                    submit_button,
                ],
            )
            ner_tag_add_button.click(
                fn=self.add_entity_to_answer,
                inputs=[question, ner_tag_dropdown, ner_tag_answer, answer],
                outputs=[ner_tag_answer, answer],
            )
            ner_tag_answer.submit(
                fn=self.add_entity_to_answer,
                inputs=[question, ner_tag_dropdown, ner_tag_answer, answer],
                outputs=[ner_tag_answer, answer],
            )
            ner_tag_reset_button.click(fn=self.reset_entities, outputs=answer)
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

    def update_dataset_choices(
        self, language: str | None, task: str | None
    ) -> "Dropdown":
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
            return Dropdown(choices=[])

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

        return Dropdown(choices=choices, value=choices[0])

    def update_dataset(
        self, dataset_name: str, iteration: int
    ) -> (
        "tuple[Markdown, Markdown, Dropdown, Textbox, Button, Button, Textbox, Button]"
    ):
        """Update the dataset based on a selected dataset name.

        Args:
            dataset_name:
                The dataset name selected by the user.
            iteration:
                The iteration index of the datasets to evaluate.

        Returns:
            A tuple (task_examples, question, entity_type, entity, entity_add_button,
            entity_reset_button, answer, submit_button) for the selected dataset.
        """
        blank_answer = (
            Markdown("", visible=False),
            Markdown("", visible=False),
            Dropdown(visible=False),
            Textbox(visible=False),
            Button(visible=False),
            Button(visible=False),
            Textbox("", visible=False),
            Button(visible=False),
        )

        if not dataset_name:
            return blank_answer

        logger.info(f"User selected dataset {dataset_name} - loading dataset...")
        gr.Info(f"Loading dataset {dataset_name}...")

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
            device=None,
            batch_size=1,
            raise_errors=False,
            cache_dir=".euroeval_cache",
            api_key=None,
            force=False,
            verbose=False,
            trust_remote_code=False,
            clear_model_cache=False,
            evaluate_test_split=False,
            few_shot=True,
            num_iterations=iteration + 1,
            api_base=None,
            api_version=None,
            gpu_memory_utilization=0.9,
            debug=False,
            run_with_cli=True,
            only_allow_safetensors=False,
        )
        self.dataset_config = get_all_dataset_configs()[dataset_name]

        # TODO: Is this needed?
        # model_id = f"human-{iteration}"
        # model_config = ModelConfig(
        #     model_id=model_id,
        #     revision="main",
        #     task="text-generation",
        #     languages=dataset_config.languages,
        #     model_type=ModelType.HUMAN,
        #     model_cache_dir=create_model_cache_dir(
        #         cache_dir=benchmark_config.cache_dir, model_id=model_id
        #     ),
        #     adapter_base_model_id=None,
        # )

        self.sample_idx = 0

        dataset_path = (
            Path(".euroeval_cache")
            / "human-evaluation"
            / dataset_name
            / f"human-{iteration}.csv"
        )
        if dataset_path.exists():
            active_dataset = Dataset.from_csv(str(dataset_path))
            assert isinstance(active_dataset, Dataset)
            self.active_dataset = active_dataset
            try:
                while self.active_dataset["answer"][self.sample_idx] is not None:
                    self.sample_idx += 1
            except IndexError:
                self.compute_and_log_scores()
                return blank_answer
        else:
            rng = enforce_reproducibility()
            datasets = load_data(
                rng=rng,
                dataset_config=self.dataset_config,
                benchmark_config=benchmark_config,
            )
            # TODO: Prepare data?
            self.active_dataset = (
                datasets[iteration]["test"]
                .remove_columns(
                    column_names=["input_ids", "attention_mask"],
                    new_fingerprint=datasets[iteration]["test"]._fingerprint,
                )
                .add_column(
                    name="answer",
                    column=[None] * len(datasets[iteration]["test"]),
                    new_fingerprint=datasets[iteration]["test"]._fingerprint,
                )
            )
            if self.dataset_config.task == NER:
                labels_in_train: set[str] = {
                    tag
                    for tag_list in self.active_dataset["labels"]
                    for tag in tag_list
                }
                self.has_misc_tags = (
                    "B-MISC" in labels_in_train or "I-MISC" in labels_in_train
                )

        match self.dataset_config.task.task_group:
            case TaskGroup.SEQUENCE_CLASSIFICATION:
                self.compute_metrics = partial(
                    sequence_classification.compute_metrics,
                    dataset_config=self.dataset_config,
                )
                self.extract_labels_from_generation = partial(
                    sequence_classification.extract_labels_from_generation,
                    dataset_config=self.dataset_config,
                )
            case TaskGroup.TEXT_TO_TEXT:
                self.compute_metrics = partial(
                    text_to_text.compute_metrics,
                    dataset_config=self.dataset_config,
                    benchmark_config=benchmark_config,
                )
                self.extract_labels_from_generation = (
                    text_to_text.extract_labels_from_generation
                )
            case TaskGroup.TOKEN_CLASSIFICATION:
                self.compute_metrics = partial(
                    token_classification.compute_metrics,
                    has_misc_tags=self.has_misc_tags,
                    dataset_config=self.dataset_config,
                )
                self.extract_labels_from_generation = partial(
                    token_classification.extract_labels_from_generation,
                    dataset_config=self.dataset_config,
                )
            case TaskGroup.QUESTION_ANSWERING:
                self.compute_metrics = partial(
                    question_answering.compute_metrics,
                    dataset_config=self.dataset_config,
                )
                self.extract_labels_from_generation = (
                    question_answering.extract_labels_from_generation
                )
            case TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION:
                raise NotImplementedError
            case _:
                raise NotImplementedError(
                    f"Task group {self.dataset_config.task.task_group} is not "
                    "supported."
                )

        task_examples, question = self.example_to_markdown(
            example=self.active_dataset[self.sample_idx]
        )

        logger.info(
            f"Loaded dataset {dataset_name}, with the following task examples:\n\n"
            f"{task_examples}"
        )

        if self.dataset_config.task == NER:
            ner_tags = list()
            for ner_tag in self.dataset_config.prompt_label_mapping.values():
                if ner_tag not in ner_tags:
                    ner_tags.append(ner_tag)
            return (
                Markdown(task_examples, visible=True),
                Markdown(question, visible=True),
                Dropdown(
                    label="Entity type",
                    choices=ner_tags,
                    value=ner_tags[0],
                    visible=True,
                ),
                Textbox(label="Entity", interactive=True, visible=True),
                Button("Add entity", visible=True),
                Button("Reset entities", visible=True),
                Textbox(
                    json.dumps({ner_tag: [] for ner_tag in ner_tags}),
                    interactive=False,
                    visible=True,
                ),
                Button("Submit", visible=True),
            )
        else:
            return (
                Markdown(task_examples, visible=True),
                Markdown(question, visible=True),
                Dropdown(label="Entity type", choices=[], visible=False),
                Textbox(label="Entity", interactive=True, visible=False),
                Button("Add entity", visible=False),
                Button("Reset entities", visible=False),
                Textbox("", interactive=True, visible=True),
                Button("Submit", visible=True),
            )

    def add_entity_to_answer(
        self, question: str, entity_type: str, entity: str, answer: str
    ) -> "tuple[Textbox, Textbox]":
        """Add an entity to the answer.

        Args:
            question:
                The current question.
            entity_type:
                The entity type selected by the user.
            entity:
                The entity provided by the user.
            answer:
                The current answer.

        Returns:
            A tuple (entity, answer) with a (blank) entity and answer.
        """
        if not entity_type or not entity:
            return Textbox(""), Textbox("")

        if entity not in question:
            gr.Warning(
                f"The entity {entity!r} is not present in the question. Please "
                "write it *exactly* as it appears in the question."
            )
            return Textbox(entity), Textbox(answer)

        current_answer_obj = json.loads(answer)
        if entity not in current_answer_obj[entity_type]:
            current_answer_obj[entity_type].append(entity)

        answer = json.dumps(current_answer_obj)
        return Textbox(""), Textbox(answer)

    def reset_entities(self) -> "Textbox":
        """Reset the entities in the answer.

        Returns:
            A blank answer.
        """
        ner_tags = list()
        for ner_tag in self.dataset_config.prompt_label_mapping.values():
            if ner_tag not in ner_tags:
                ner_tags.append(ner_tag)
        return Textbox(json.dumps({ner_tag: [] for ner_tag in ner_tags}))

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
        if self.dataset_config.task == NER:
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
                    "User tried to submit a JSON object without a dictionary as an "
                    "answer."
                )
                return question, answer

            ner_tags = list(self.dataset_config.prompt_label_mapping.values())
            for ner_tag in ner_tags:
                if ner_tag not in json.loads(answer).keys():
                    gr.Warning(
                        f"Please provide a JSON object with the key {ner_tag!r}."
                    )
                    logger.info(
                        "User tried to submit a JSON object without the key "
                        f"{ner_tag!r}."
                    )
                    return question, answer

        samples_left = len(self.active_dataset) - self.sample_idx - 1
        if samples_left:
            gr.Info(f"Submitted - {samples_left} to go!")

        # Store the user's answer
        answers = self.active_dataset["answer"]
        answers[self.sample_idx] = answer
        self.active_dataset = self.active_dataset.remove_columns(
            column_names=["answer"], new_fingerprint=self.active_dataset._fingerprint
        ).add_column(
            name="answer",
            column=answers,
            new_fingerprint=self.active_dataset._fingerprint,
        )
        logger.info(
            f"User submitted the answer {answer!r} to the question {question!r}, with "
            f"sample index {self.sample_idx}."
        )

        dataset_path = (
            Path(".euroeval_cache")
            / "human-evaluation"
            / dataset_name
            / f"human-{annotator_id}.csv"
        )
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.active_dataset.to_csv(dataset_path)

        # Attempt to get the next question
        try:
            self.sample_idx += 1
            _, question = self.example_to_markdown(
                example=self.active_dataset[self.sample_idx]
            )

            if self.dataset_config.task == NER:
                ner_tags = list()
                for ner_tag in self.dataset_config.prompt_label_mapping.values():
                    if ner_tag not in ner_tags:
                        ner_tags.append(ner_tag)
                answer = json.dumps({ner_tag: [] for ner_tag in ner_tags})
            else:
                answer = ""

        # If we fail to get the next question it means that the user has finished
        # annotating the dataset, so we compute and log the scores
        except IndexError:
            self.compute_and_log_scores()
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

    def compute_and_log_scores(self) -> None:
        """Computes and logs the scores for the dataset."""
        model_output = GenerativeModelOutput(sequences=self.active_dataset["answer"])

        active_dataset_dict = self.active_dataset.to_dict()
        assert isinstance(active_dataset_dict, dict)

        all_preds = self.extract_labels_from_generation(
            input_batch=active_dataset_dict, model_output=model_output
        )
        ground_truth = self.active_dataset["label"]
        itr_scores: dict[str, float] = self.compute_metrics(
            model_outputs_and_labels=(all_preds, ground_truth),
            dataset=self.active_dataset,
        )

        # We reverse the order, as the Info messages are printed in reverse order
        scores = list(itr_scores.items())
        scores.reverse()
        gr.Info(
            "If you want to evaluate another dataset then please select a new "
            "one from the menus."
        )
        for metric_name, score in scores:
            gr.Info(f"\n\n{metric_name}: {score:.2%}")
        gr.Info("You have completed this dataset! Here are your scores:")
        logger.info(
            f"User completed the dataset {self.dataset_config.name!r}"
            f", with the following scores: {itr_scores}"
        )

        # Load previous human results, if any. We do this since the human evaluation is
        # only a single iteration, so the results from the current annotation should be
        # added to the previous results.
        results_path = Path.cwd() / "euroeval_benchmark_results.jsonl"
        results: "ScoreDict" = defaultdict(list)
        if results_path.exists():
            all_results = [
                json.loads(line.strip())
                for line in results_path.read_text().strip().split("\n")
                if line.strip()
            ]
            human_result_candidates = [
                result
                for result in all_results
                if result["model"] == "human"
                and result["dataset"] == self.dataset_config.name
            ]
            if human_result_candidates:
                results = human_result_candidates[0]["results"]

        # Append to results
        results["raw"].append(  # type: ignore[union-attr]
            {f"test_{metric_name}": score for metric_name, score in itr_scores.items()}
        )

        # Aggregate scores
        total_dict: dict[str, float] = dict()
        for metric in self.dataset_config.task.metrics:
            test_score, test_se = aggregate_scores(
                scores=results["raw"],  # type: ignore[arg-type]
                metric=metric,
            )
            test_score, _ = metric.postprocessing_fn(test_score)
            test_se, _ = metric.postprocessing_fn(test_se)
            total_dict[f"test_{metric.name}"] = test_score
            total_dict[f"test_{metric.name}_se"] = test_se
        results["total"] = total_dict

        benchmark_result = BenchmarkResult(
            dataset=self.dataset_config.name,
            task=self.dataset_config.task.name,
            dataset_languages=[
                language.code for language in self.dataset_config.languages
            ],
            model="human",
            results=results,
            num_model_parameters=-1,
            max_sequence_length=-1,
            vocabulary_size=-1,
            merge=False,
            generative=True,
            generative_type=GenerativeType.INSTRUCTION_TUNED,
            few_shot=True,
            validation_split=True,
        )
        benchmark_result.append_to_results(results_path=results_path)


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
        title="EuroEval Human Evaluation",
        description="""
        In this app we will evaluate your performance on a variety of tasks, with the
        goal of comparing human performance to language model performance.

        When you select a language and a task then you will be given a brief
        description of the task, as well as examples of how to solve it. Please read
        through these examples before proceeding with the task.

        Please do not use any additional aids (such as search engines) when completing
        these tasks.

        Note that several examples appear more than once - this is intentional, as it
        allows us to compare your performance across multiple examples.

        Note that the Enter key will also submit your answer!
        """,
    )
    evaluator.create_app().queue().launch()


if __name__ == "__main__":
    main()
