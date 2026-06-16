# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = []
# ///

"""Checks related to the .env file in the repository."""

from pathlib import Path

# The prompt shown for each desired environment variable when it is missing.
DESIRED_ENVIRONMENT_VARIABLES = {
    "GIT_NAME": "Enter your full name, to be shown in Git commits:\n> ",
    "GIT_EMAIL": "Enter your email, as registered on your Github account:\n> ",
    "EUROEVAL_PYPI_API_TOKEN": "Enter your EuroEval PyPI API token, or leave empty "
    "if you do not want to use it:\n> ",
    "SCANDEVAL_PYPI_API_TOKEN": "Enter your ScandEval PyPI API token, or leave empty "
    "if you do not want to use it:\n> ",
}


def main() -> None:
    """Ensure that the .env file exists and contains all desired variables."""
    env_file_path = Path(".env")
    env_file_path.touch(exist_ok=True)

    env_file_lines = env_file_path.read_text().splitlines(keepends=False)
    env_vars = [line.split("=")[0] for line in env_file_lines]
    env_vars_missing = [
        env_var for env_var in DESIRED_ENVIRONMENT_VARIABLES if env_var not in env_vars
    ]

    with env_file_path.open("a") as f:
        for env_var in env_vars_missing:
            value = input(DESIRED_ENVIRONMENT_VARIABLES[env_var])
            f.write(f'{env_var}="{value}"\n')


if __name__ == "__main__":
    main()
