# Welcome to the EuroEval Contributing guide

Thank you for investing your time in contributing to our project! :sparkles:.

Read our [Code of Conduct](./CODE_OF_CONDUCT.md) to keep our community approachable and
respectable.

In this guide you will get an overview of the contribution workflow from opening an
issue, creating a PR, reviewing, and merging the PR.

## New contributor guide

To get an overview of the project, read the [README](README.md). Here are some
resources to help you get started with open source contributions:

- [Finding ways to contribute to open source on GitHub]
- [Set up Git]
- [GitHub flow]
- [Collaborating with pull requests]

## Getting started

### Adding datasets

EuroEval welcomes contributions of new datasets that help evaluate language models
across European languages. A guide for adding datasets to EuroEval can be found
[here](NEW_DATASET_GUIDE.md).

### Issues

#### Create a new issue

If you spot a problem with the package, [search if an issue already exists]. If a
related issue doesn't exist, you can open a new issue using a relevant [issue form].

#### Solve an issue

Scan through our [existing issues] to find one that interests you. You can narrow down
the search using `labels` as filters. If you find an issue to work on, you are welcome
to open a PR with a fix.

### Make Changes

1. Fork the repository.

- Using GitHub Desktop:
  - [Getting started with GitHub Desktop] will guide you through setting up Desktop.
  - Once Desktop is set up, you can use it to [fork the repo in Github Desktop]!

- Using the command line:
  - [Fork the repo] so that you can make your changes without affecting the original
    project until you're ready to merge them.

1. Run `make install` from within the repo to get set up.

2. Create a working branch and start with your changes!

### Commit your update

Commit the changes once you are happy with them. Make sure your code follows the
[Python Conventions](#python-conventions) below.

Once your changes are ready, don't forget to self-review to speed up the review
process:zap:.

### Pull Request

When you're finished with the changes, create a pull request, also known as a PR.

- Fill the "Ready for review" template so that we can review your PR. This template
  helps reviewers understand your changes as well as the purpose of your pull request.
- Don't forget to [link PR to issue] if you are solving one.
- Enable the checkbox to [allow maintainer edits] so the branch can be updated for a
  merge.

Once you submit your PR, a team member will review your proposal. We may ask questions
or request for additional information.

- We may ask for changes to be made before a PR can be merged, either using [suggested
  changes] or pull request comments. You can apply suggested changes directly through
  the UI. You can make any other changes in your fork, then commit them to your branch.
- As you update your PR and apply changes, mark each conversation as [resolved].
- If you run into any merge issues, checkout this [git tutorial] to help you resolve
  merge conflicts and other issues.

### Your PR is merged

Congratulations :tada::tada: The EuroEval team thanks you :sparkles:.

## Python Conventions

### Code Organisation

- Code modules go in `src/euroeval/` (imported, not executed)
- Scripts go in `src/scripts/` (executed with `uv run`)
- Tests go in `tests/`

### Code Formatting, Linting, Type Checking and Testing

- Run `make check` to run formatters, linters, and type checkers
- Run `make test` to run the tests locally
- Use relative imports in modules:

  ```python title="src/mypackage/module.py"
  from .another_module import some_function
  ```

- Use absolute imports in scripts:

  ```python title="src/scripts/script.py"
  from mypackage.module import some_function
  from another_script import some_other_function
  ```

### Docstrings

Use [Google-style docstrings] for all public functions, classes, and modules. Here is an
example:

```python
def process_items(items: list[Item]) -> list[Result]:
    """Process items and return results.

    Args:
        items:
          List of items to process.

    Returns:
        List of processed results.

    Raises:
        ValueError:
          If items list is empty.
    """
    return batch_process(items=items)
```

### Type Annotations

- Fully type-annotate all functions, methods, and variables
- Target Python 3.12+ syntax:
  - Use `list[T]`, `dict[K, V]`, `set[T]` (not `List`, `Dict`, `Set` from typing)
  - Use `X | Y` for unions (not `Union[X, Y]`)
  - Use `X | None` for optional types (not `Optional[X]`)
- Always use `import typing as t` and the `t.` prefix for typing module types like
  `t.Literal`, `t.TypeAlias`, or `t.TYPE_CHECKING`
- For `Iterable`, `Generator`, and `Callable`, use `collections.abc` instead of `typing`.
  Import as `import collections.abc as c` and refer to types as `c.Iterable`,
  `c.Generator`, `c.Callable`, etc.
- Avoid `Any` when possible. You can often use `t.TypeVar` with meaningful names instead
  of single letters like `T`. The main acceptable use of `Any` is for dictionaries with
  mixed outputs, e.g., `dict[str, t.Any]`

### Functions

- Use a single leading underscore (`_`) for protected functions (not importable from
  outside the module) or protected methods (not usable outside their class)
- Always use keyword arguments when calling functions — never positional arguments

Example:

```python
def process_items(items: list[Item]) -> list[Result]:
    ...

process_items(items=items)
```

## References

[Finding ways to contribute to open source on GitHub]:
  https://docs.github.com/en/get-started/exploring-projects-on-github/finding-ways-to-contribute-to-open-source-on-github
[Set up Git]:
  https://docs.github.com/en/get-started/quickstart/set-up-git
[GitHub flow]:
  https://docs.github.com/en/get-started/quickstart/github-flow
[Collaborating with pull requests]:
  https://docs.github.com/en/github/collaborating-with-pull-requests
[search if an issue already exists]:
  https://docs.github.com/en/github/searching-for-information-on-github/searching-on-github/searching-issues-and-pull-requests#search-by-the-title-body-or-comments
[issue form]:
  https://github.com/EuroEval/EuroEval/issues
[existing issues]:
  https://github.com/EuroEval/EuroEval/issues
[Getting started with GitHub Desktop]:
  https://docs.github.com/en/desktop/installing-and-configuring-github-desktop/getting-started-with-github-desktop
[fork the repo in Github Desktop]:
  https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/cloning-and-forking-repositories-from-github-desktop
[Fork the repo]:
  https://docs.github.com/en/github/getting-started-with-github/fork-a-repo#fork-an-example-repository
[link PR to issue]:
  https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue
[allow maintainer edits]:
  https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/allowing-changes-to-a-pull-request-branch-created-from-a-fork
[suggested changes]:
  https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/incorporating-feedback-in-your-pull-request
[resolved]:
  https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/commenting-on-a-pull-request#resolving-conversations
[git tutorial]:
  https://github.com/skills/resolve-merge-conflicts
[Google-style docstrings]:
  https://google.github.io/styleguide/pyguide.html#383-functions-and-methods
