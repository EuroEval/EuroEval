# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = []
# ///

"""Checks related to the .env file in the repository."""

from pathlib import Path

# List of all the environment variables that are desired
DESIRED_ENVIRONMENT_VARIABLES = dict(
    GIT_NAME="Enter your full name, to be shown in Git commits:\n> ",
    GIT_EMAIL="Enter your email, as registered on your Github account:\n> ",
    EUROEVAL_PYPI_API_TOKEN="Enter your EuroEval PyPI API token, or leave empty "
    "if you do not want to use it:\n> ",
    SCANDEVAL_PYPI_API_TOKEN="Enter your ScandEval PyPI API token, or leave empty "
    "if you do not want to use it:\n> ",
)


def fix_dot_env_file() -> None:
    """Ensures that the .env file exists and contains all desired variables."""
    # Create path to the .env file
    env_file_path = Path(".env")

    # Ensure that the .env file exists
    env_file_path.touch(exist_ok=True)

    # Otherwise, extract all the lines in the .env file
    env_file_lines = env_file_path.read_text().splitlines(keepends=False)

    # Extract all the environment variables in the .env file
    env_vars = [line.split("=")[0] for line in env_file_lines]

    # For each of the desired environment variables, check if it exists in the .env
    # file
    env_vars_missing = [
        env_var
        for env_var in DESIRED_ENVIRONMENT_VARIABLES.keys()
        if env_var not in env_vars
    ]

    # Create all the missing environment variables
    with env_file_path.open("a") as f:
        for env_var in env_vars_missing:
            value = ""
            if value == "":
                value = input(DESIRED_ENVIRONMENT_VARIABLES[env_var])
            f.write(f'{env_var}="{value}"\n')


if __name__ == "__main__":
    fix_dot_env_file()
