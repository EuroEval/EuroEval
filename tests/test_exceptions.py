"""Tests for the `exceptions` module."""

import inspect

from euroeval import exceptions


def test_all_classes_are_exceptions() -> None:
    """Test that all classes in `exceptions` are exceptions."""
    all_classes = [
        getattr(exceptions, obj_name)
        for obj_name in dir(exceptions)
        if not obj_name.startswith("_")
        and inspect.isclass(object=getattr(exceptions, obj_name))
    ]
    for obj in all_classes:
        assert issubclass(obj, Exception), f"Class {obj.__name__} is not an exception."


class TestNeedsExtraInstalled:
    """Tests for the NeedsExtraInstalled exception."""

    def test_needs_extra_installed_message_format(self) -> None:
        """Test that message contains extra name."""
        extra_name = "test-extra"
        exc = exceptions.NeedsExtraInstalled(extra=extra_name)

        assert extra_name in exc.message
        assert "euroeval" in exc.message
        assert extra_name in exc.message
        assert exc.extra == extra_name


class TestNeedsManualDependency:
    """Tests for the NeedsManualDependency exception."""

    def test_needs_manual_dependency_message_format(self) -> None:
        """Test that message contains package name."""
        package_name = "test-package"
        exc = exceptions.NeedsManualDependency(package=package_name)

        assert package_name in exc.message
        assert "pip install" in exc.message
        assert exc.package == package_name


class TestNeedsSystemDependency:
    """Tests for the NeedsSystemDependency exception."""

    def test_needs_system_dependency_message_format(self) -> None:
        """Test that message contains dependency and instructions."""
        dependency = "test-dependency"
        instructions = "Please install it using your package manager."
        exc = exceptions.NeedsSystemDependency(
            dependency=dependency, instructions=instructions
        )

        assert dependency in exc.message
        assert instructions in exc.message
        assert exc.dependency == dependency
        assert instructions in exc.message


class TestNeedsAdditionalArgument:
    """Tests for the NeedsAdditionalArgument exception."""

    def test_needs_additional_argument_cli_mode_message(self) -> None:
        """Test message format for CLI mode."""
        cli_argument = "--test-arg"
        script_argument = "test_arg"
        exc = exceptions.NeedsAdditionalArgument(
            cli_argument=cli_argument,
            script_argument=script_argument,
            run_with_cli=True,
        )

        assert cli_argument in exc.message
        assert exc.cli_argument == cli_argument
        assert exc.script_argument == script_argument

    def test_needs_additional_argument_script_mode_message(self) -> None:
        """Test message format for script mode."""
        cli_argument = "--test-arg"
        script_argument = "test_arg"
        exc = exceptions.NeedsAdditionalArgument(
            cli_argument=cli_argument,
            script_argument=script_argument,
            run_with_cli=False,
        )

        assert script_argument in exc.message
        assert exc.cli_argument == cli_argument
        assert exc.script_argument == script_argument


class TestNeedsEnvironmentVariable:
    """Tests for the NeedsEnvironmentVariable exception."""

    def test_needs_environment_variable_message_format(self) -> None:
        """Test that message contains environment variable name."""
        env_var = "TEST_API_KEY"
        exc = exceptions.NeedsEnvironmentVariable(env_var=env_var)

        assert env_var in exc.message
        assert exc.env_var == env_var


class TestBaseExceptions:
    """Tests for base exception classes."""

    def test_invalid_benchmark_default_message(self) -> None:
        """Test InvalidBenchmark with default message."""
        exc = exceptions.InvalidBenchmark()
        assert exc.message == "This model cannot be benchmarked on the given dataset."

    def test_invalid_benchmark_custom_message(self) -> None:
        """Test InvalidBenchmark with custom message."""
        custom_msg = "Custom error message"
        exc = exceptions.InvalidBenchmark(message=custom_msg)
        assert exc.message == custom_msg

    def test_invalid_model_default_message(self) -> None:
        """Test InvalidModel with default message."""
        exc = exceptions.InvalidModel()
        assert exc.message == "The model cannot be benchmarked on any datasets."

    def test_invalid_task_default_message(self) -> None:
        """Test InvalidTask with default message."""
        exc = exceptions.InvalidTask()
        assert exc.message == "The task is invalid."

    def test_hugging_face_hub_down_default_message(self) -> None:
        """Test HuggingFaceHubDown with default message."""
        exc = exceptions.HuggingFaceHubDown()
        assert exc.message == "The Hugging Face Hub is currently down."

    def test_no_internet_connection_default_message(self) -> None:
        """Test NoInternetConnection with default message."""
        exc = exceptions.NoInternetConnection()
        assert exc.message == "There is currently no internet connection."

    def test_nan_value_in_model_output_default_message(self) -> None:
        """Test NaNValueInModelOutput with default message."""
        exc = exceptions.NaNValueInModelOutput()
        assert exc.message == "There is a NaN value in the model output."
