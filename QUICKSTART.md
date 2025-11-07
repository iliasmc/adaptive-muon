# Quick Reference Guide

## Setup Commands

### Using Conda (Recommended)
```bash
# GPU version (CUDA 11.8)
conda env create -f environment.yml
conda activate adaptive-muon

# CPU-only version
conda env create -f environment-cpu.yml
conda activate adaptive-muon-cpu

# Install pre-commit hooks
pre-commit install
```

### Using pip
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
pre-commit install
```

## Daily Development Workflow

### 1. Format Code
```bash
black .
isort .
```

Or use make:
```bash
make format
```

### 2. Run Linters
```bash
flake8 .
pylint src/
mypy src/
```

Or use make:
```bash
make lint
```

### 3. Run Tests
```bash
pytest --cov=src
```

Or use make:
```bash
make test
```

### 4. Run Everything
```bash
make all
```

## Pre-commit Hooks

Hooks run automatically before each commit. To run manually:
```bash
pre-commit run --all-files
```

## Tool Configuration

| Tool | Config File | Purpose |
|------|-------------|---------|
| Black | pyproject.toml | Code formatting |
| isort | pyproject.toml | Import sorting |
| Flake8 | .flake8 | Linting |
| Pylint | pyproject.toml | Advanced linting |
| MyPy | pyproject.toml | Type checking |
| pytest | pyproject.toml | Testing |
| Pre-commit | .pre-commit-config.yaml | Git hooks |

## Code Style Rules

- **Line length**: 100 characters
- **Python versions**: 3.8+
- **Import order**: stdlib, third-party, local (managed by isort)
- **Type hints**: Encouraged but not required
- **Docstrings**: Required for public functions and classes

## Common Issues

### "Command not found: black/isort/flake8"
```bash
# Conda users
conda activate adaptive-muon

# pip users
source venv/bin/activate
pip install -r requirements-dev.txt
```

### Pre-commit hook fails
```bash
# Run formatting
black .
isort .

# Try commit again
git commit
```

### Tests fail
```bash
# Install test dependencies
pip install pytest pytest-cov torch

# Run tests
pytest -v
```

## CI/CD

GitHub Actions runs automatically on:
- Push to main/develop
- Pull requests to main/develop

Checks performed:
- ✅ Black formatting
- ✅ isort import sorting
- ✅ Flake8 linting
- ✅ Pylint code analysis
- ✅ MyPy type checking
- ✅ pytest test suite

## Resources

- **Full documentation**: See CONTRIBUTING.md
- **Project setup**: See README.md
- **PyTorch docs**: https://pytorch.org/docs/
- **Code style guide**: https://pep8.org/
