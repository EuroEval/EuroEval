"""Exceptions to used by other functions."""


class InvalidBenchmark(Exception):
    """The (model, dataset) combination cannot be benchmarked."""

    def __init__(
        self, message: str = "This model cannot be benchmarked on the given dataset."
    ) -> None:
        """Initialise the exception.

        Args:
            message:
                The message to display.
        """
        self.message = message
        super().__init__(self.message)


class InvalidModel(Exception):
    """The model cannot be benchmarked on any datasets."""

    def __init__(
        self, message: str = "The model cannot be benchmarked on any datasets."
    ) -> None:
        """Initialise the exception.

        Args:
            message:
                The message to display.
        """
        self.message = message
        super().__init__(self.message)


class HuggingFaceHubDown(Exception):
    """The Hugging Face Hub seems to be down."""

    def __init__(
        self, message: str = "The Hugging Face Hub is currently down."
    ) -> None:
        """Initialise the exception.

        Args:
            message:
                The message to display.
        """
        self.message = message
        super().__init__(self.message)


class NoInternetConnection(Exception):
    """There seems to be no internet connection."""

    def __init__(
        self, message: str = "There is currently no internet connection."
    ) -> None:
        """Initialise the exception.

        Args:
            message:
                The message to display.
        """
        self.message = message
        super().__init__(self.message)


class NaNValueInModelOutput(Exception):
    """There is a NaN value in the model output."""

    def __init__(
        self, message: str = "There is a NaN value in the model output."
    ) -> None:
        """Initialise the exception.

        Args:
            message:
                The message to display.
        """
        self.message = message
        super().__init__(self.message)


class FlashAttentionNotInstalled(Exception):
    """The `flash-attn` package has not been installed."""

    def __init__(
        self,
        message: str = (
            "The model you are trying to load requires Flash Attention. To use Flash "
            "Attention, please install the `flash-attn` package, which can be done by "
            "running `pip install -U wheel && FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE "
            "pip install flash-attn --no-build-isolation`."
        ),
    ) -> None:
        """Initialise the exception.

        Args:
            message:
                The message to display.
        """
        self.message = message
        super().__init__(self.message)


class NeedsExtraInstalled(InvalidModel):
    """The evaluation requires extra to be installed."""

    def __init__(self, extra: str) -> None:
        """Initialise the exception.

        Args:
            extra:
                The extra that needs to be installed.
        """
        self.extra = extra
        self.message = (
            f"The model you are trying to load requires the `{extra}` extra to be "
            f"installed. To install the `{extra}` extra, please run `pip install "
            f"euroeval[{extra}]` or `pip install euroeval[all]`."
        )
        super().__init__(self.message)


class NeedsManualDependency(InvalidModel):
    """The evaluation requires a dependency to be manually installed."""

    def __init__(self, package: str) -> None:
        """Initialise the exception.

        Args:
            package:
                The package that needs to be manually installed.
        """
        self.package = package
        self.message = (
            f"The model you are trying to load requires the `{package}` package to be "
            f"installed - please run `pip install {package}` and try again."
        )
        super().__init__(self.message)


class NeedsAdditionalArgument(InvalidModel):
    """The evaluation requires additional arguments to the `euroeval` command."""

    def __init__(
        self, cli_argument: str, script_argument: str, run_with_cli: bool
    ) -> None:
        """Initialise the exception.

        Args:
            cli_argument:
                The argument that needs to be passed to the `euroeval` command.
            script_argument:
                The argument that needs to be passed to the `Benchmarker` class.
            run_with_cli:
                Whether the benchmark is being run with the CLI.
        """
        self.cli_argument = cli_argument
        self.script_argument = script_argument
        if run_with_cli:
            self.message = (
                f"The model you are trying to load requires the `{cli_argument}` "
                "argument to be passed to the `euroeval` command. Please pass the "
                "argument and try again."
            )
        else:
            self.message = (
                f"The model you are trying to load requires the `{script_argument}` "
                "argument  to be passed to the `Benchmarker` class. Please pass the "
                "argument and try again."
            )
        super().__init__(self.message)


class NeedsEnvironmentVariable(InvalidModel):
    """The evaluation requires an environment variable to be set."""

    def __init__(self, env_var: str) -> None:
        """Initialise the exception.

        Args:
            env_var:
                The environment variable that needs to be set.
        """
        self.env_var = env_var
        self.message = (
            f"The model you are trying to load requires the `{env_var}` environment "
            "variable to be set. Please set the environment variable and try again."
        )
        super().__init__(self.message)
