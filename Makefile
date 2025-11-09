.PHONY: install lint format test clean

install:
poetry install --with dev

lint:
poetry run ruff check src

format:
poetry run black src

test:
poetry run pytest --cov=valentina --cov-report=term-missing

clean:
rm -rf .pytest_cache .ruff_cache
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
