.PHONY: setup setup-dev lint test update-deps clean

# Default Python interpreter
PYTHON ?= python3
UV ?= uv

setup:
	$(UV) venv
	$(UV) pip install -e .

setup-dev:
	$(UV) venv
	$(UV) pip install -e ".[dev]"

lint:
	$(UV) run ruff check .
	$(UV) run black --check .

format:
	$(UV) run ruff check --fix .
	$(UV) run black .
	$(UV) run isort .

test:
	$(UV) run pytest tests/

update-deps:
	$(UV) pip compile pyproject.toml --output-file requirements.lock

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type f -name "coverage.xml" -delete
	find . -type f -name "*.log" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf .venv/
