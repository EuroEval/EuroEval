# Publishing mutates shared package metadata; serialise every make invocation so
# package builds, release creation, and version changes cannot overlap.
.NOTPARALLEL:

# This ensures that we can call `make <target>` even if `<target>` exists as a file or
# directory.
.PHONY: help docs install install-frontend install-vercel leaderboards force-leaderboards \
	volunteer-worker-plan volunteer-worker-check publish publish-euroeval publish-scandeval \
	publish-major publish-minor publish-patch add-dev-version

# Exports all variables defined in the makefile available to scripts
.EXPORT_ALL_VARIABLES:

# Read-only targets deliberately never import or export the local dotenv file.
# Set gRPC environment variables, which prevents some errors with the `grpcio` package
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1

# Set the PATH env var used by cargo and uv
export PATH := ${HOME}/.local/bin:${HOME}/.cargo/bin:$(PATH)

# Set the shell to bash, enabling the use of `source` statements
SHELL := /bin/bash

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	@echo "Installing the 'EuroEval' project..."
	@$(MAKE) --quiet install-rust
	@$(MAKE) --quiet install-uv
	@$(MAKE) --quiet install-dependencies
	@$(MAKE) --quiet setup-environment-variables
	@$(MAKE) --quiet install-frontend
	@$(MAKE) --quiet install-vercel
	@echo "Installed the 'EuroEval' project."

install-rust:
	@if [ "$(shell which rustup)" = "" ]; then \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
		echo "Installed Rust."; \
	fi

install-uv:
	@if [ "$(shell which uv)" = "" ]; then \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
			echo "Installed uv."; \
		else \
			echo "Updating uv..."; \
			uv self update || true; \
	fi

install-dependencies:
	@uv python install 3.12
	@uv sync --all-extras --all-groups --python 3.12

setup-environment-variables:
	@uv run python src/scripts/fix_dot_env_file.py

setup-environment-variables-non-interactive:
	@uv run python src/scripts/fix_dot_env_file.py --non-interactive

install-frontend: ## Install frontend dependencies
	@echo "Installing frontend dependencies..."
	@npm install
	@echo "Installed frontend dependencies."

install-vercel:
	@if [ "$(shell which vercel)" = "" ]; then \
		npm install -g vercel; \
		echo "Installed Vercel CLI."; \
	fi

install-pre-commit:
	@uv run pre-commit install
	@uv run pre-commit autoupdate

test:  ## Run tests
	@uv run pytest && uv run readme-cov && rm .coverage*

frontend:  ## Build and deploy the frontend
	@uv run python src/scripts/prepare_vercel_worktree.py && vercel build --prod --yes && uv run python src/scripts/verify_vercel_functions.py && vercel deploy --prebuilt --prod

leaderboards:  ## Collect finished evaluation results and regenerate leaderboards
	@uv run python src/scripts/collect_evaluation_results.py

force-leaderboards:
	@uv run python src/scripts/collect_evaluation_results.py --force

volunteer-worker-plan:
	@uv run python src/scripts/volunteer_worker_operations.py plan

volunteer-worker-check:
	@uv run python src/scripts/volunteer_worker_operations.py check

tree:  ## Print directory tree
	@tree -a --gitignore -I .git .

check:  ## Lint, format, and type-check the code
	@npm run check:vercel-source
	@git add . && uv run pre-commit run --all-files
	@if command -v llama-server >/dev/null 2>&1 && pgrep -x llama-server >/dev/null; then \
		echo "Running Slopo code duplication detection..."; \
		uv run python src/scripts/run_slopo_check.py; \
	else \
		echo "Slopo skipped (llama.cpp server not running)"; \
	fi

refresh-dataset-revisions:
	@uv run python src/scripts/refresh_dataset_revisions.py
	@uv run ruff format src/euroeval/dataset_configs/
	@uv run pytest -q --no-cov tests/test_dataset_configs.py -k test_builtin_hub_datasets_have_pinned_revisions

bump-major: refresh-dataset-revisions
	@uv run python -m src.scripts.versioning --major
	@echo "Bumped major version!"

bump-minor: refresh-dataset-revisions
	@uv run python -m src.scripts.versioning --minor
	@echo "Bumped minor version!"

bump-patch: refresh-dataset-revisions
	@uv run python -m src.scripts.versioning --patch
	@echo "Bumped patch version!"

add-dev-version:
	@uv run src/scripts/versioning.py --add-dev

publish:
	@set -e; set -a; [ ! -f .env ] || source .env; set +a; \
	if [ -z "$${EUROEVAL_PYPI_API_TOKEN:-}" ] || [ -z "$${SCANDEVAL_PYPI_API_TOKEN:-}" ]; then \
		echo "Both EUROEVAL_PYPI_API_TOKEN and SCANDEVAL_PYPI_API_TOKEN must be specified to publish." >&2; exit 1; \
	fi; \
	uv run src/scripts/versioning.py --validate-publish; \
	echo "Publishing to PyPI..."; \
	$(MAKE) --quiet publish-euroeval; \
	$(MAKE) --quiet publish-scandeval; \
	uv run src/scripts/versioning.py --release-only "$$(uv run python -c 'from scripts.versioning import get_project_version; print(get_project_version())')"; \
	$(MAKE) --quiet add-dev-version; \
	$(MAKE) --quiet frontend; \
	echo "Published!"

publish-euroeval:
	@test -n "${EUROEVAL_PYPI_API_TOKEN}" || { echo "EUROEVAL_PYPI_API_TOKEN must be specified to publish." >&2; exit 1; }
	@uv run src/scripts/versioning.py --validate-stable
	@rm -rf build/ dist/
	@uv build
	@uv publish --username "__token__" --password ${EUROEVAL_PYPI_API_TOKEN}

publish-scandeval:
	@test -n "${SCANDEVAL_PYPI_API_TOKEN}" || { echo "SCANDEVAL_PYPI_API_TOKEN must be specified to publish." >&2; exit 1; }
	@uv run src/scripts/versioning.py --validate-stable
	@set -e; backup=$$(mktemp); cp -p pyproject.toml "$$backup"; \
	cleanup() { cp "$$backup" pyproject.toml; rm -f "$$backup"; \
		if [ -d src/scandeval ] && [ ! -e src/euroeval ]; then mv src/scandeval src/euroeval; fi; }; \
	trap cleanup EXIT; \
	if [ "$$(uname)" = "Darwin" ]; then \
		sed -i '' 's/^name = "EuroEval"/name = "ScandEval"/' pyproject.toml; \
	else \
		sed -i 's/^name = "EuroEval"/name = "ScandEval"/' pyproject.toml; \
	fi; \
	mv src/euroeval src/scandeval; \
	rm -rf build/ dist/; \
	uv build; \
	uv publish --username "__token__" --password ${SCANDEVAL_PYPI_API_TOKEN}

publish-major:  ## Publish a major version
	@$(MAKE) install && $(MAKE) check && $(MAKE) bump-major && $(MAKE) publish

publish-minor:  ## Publish a minor version
	@$(MAKE) install && $(MAKE) check && $(MAKE) bump-minor && $(MAKE) publish

publish-patch:  ## Publish a patch version
	@$(MAKE) install && $(MAKE) check && $(MAKE) bump-patch && $(MAKE) publish

loc: ## Count the number of lines of code in the project
	@git ls-files | grep '\.py' | xargs wc -l | tail -n 1
