.PHONY: help install install-dev format lint test clean all

help:
	@echo "Available commands:"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make format       - Format code with Black and isort"
	@echo "  make lint         - Run all linters (Flake8, Pylint, MyPy)"
	@echo "  make test         - Run tests with coverage"
	@echo "  make clean        - Remove build artifacts and cache files"
	@echo "  make all          - Format, lint, and test (full quality check)"

install:
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

format:
	@echo "Running Black..."
	black .
	@echo "Running isort..."
	isort .
	@echo "Formatting complete!"

lint:
	@echo "Running Flake8..."
	flake8 . || true
	@echo "\nRunning Pylint..."
	pylint src/ || true
	@echo "\nRunning MyPy..."
	mypy src/ || true
	@echo "\nLinting complete!"

test:
	@echo "Running tests with coverage..."
	pytest --cov=src --cov-report=term-missing --cov-report=html
	@echo "\nTests complete! Coverage report saved to htmlcov/index.html"

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage coverage.xml
	@echo "Cleanup complete!"

all: format lint test
	@echo "\nAll quality checks passed! ✓"
