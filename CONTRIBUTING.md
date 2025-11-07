# Contributing to adaptive-muon

Thank you for your interest in contributing to this PyTorch project! This document provides guidelines and best practices for maintaining high code quality.

## Development Setup

We support two methods for setting up the development environment: Conda (recommended) and pip with venv.

### Option 1: Using Conda (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/iliasmc/adaptive-muon.git
   cd adaptive-muon
   ```

2. **Create conda environment**
   
   For GPU support (CUDA 11.8):
   ```bash
   conda env create -f environment.yml
   conda activate adaptive-muon
   ```
   
   For CPU-only:
   ```bash
   conda env create -f environment-cpu.yml
   conda activate adaptive-muon-cpu
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

### Option 2: Using pip and venv

1. **Clone the repository**
   ```bash
   git clone https://github.com/iliasmc/adaptive-muon.git
   cd adaptive-muon
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

### Managing Dependencies

#### With Conda

To update the conda environment:
```bash
# Update all packages
conda update --all

# Add a new package
conda install package-name

# Export environment
conda env export > environment.yml
```

To update environment file from your current environment:
```bash
conda env export --no-builds > environment.yml
```

#### With pip

To update pip dependencies:
```bash
pip install --upgrade -r requirements-dev.txt
```

## Code Quality Tools

This project uses several tools to enforce code quality:

### 1. Black (Code Formatter)
Black is an opinionated code formatter that ensures consistent code style.

```bash
# Format all Python files
black .

# Check formatting without making changes
black --check .
```

**Configuration**: See `[tool.black]` in `pyproject.toml`
- Line length: 100 characters
- Target Python versions: 3.8+

### 2. isort (Import Sorter)
isort automatically sorts and organizes imports.

```bash
# Sort imports
isort .

# Check import sorting
isort --check-only .
```

**Configuration**: See `[tool.isort]` in `pyproject.toml`
- Profile: black (compatible with Black)
- Line length: 100 characters

### 3. Flake8 (Linter)
Flake8 checks for code style and potential errors.

```bash
# Run flake8
flake8 .

# Run with specific file
flake8 src/my_module.py
```

**Configuration**: See `.flake8`
- Max line length: 100
- Ignores conflicts with Black (E203, W503, E501)

### 4. Pylint (Advanced Linter)
Pylint provides more comprehensive linting and code analysis.

```bash
# Run pylint on source directory
pylint src/

# Run with specific file
pylint src/my_module.py
```

**Configuration**: See `[tool.pylint]` in `pyproject.toml`
- Max line length: 100
- Some rules disabled for PyTorch projects

### 5. MyPy (Type Checker)
MyPy performs static type checking.

```bash
# Run mypy
mypy src/

# Run on specific file
mypy src/my_module.py
```

**Configuration**: See `[tool.mypy]` in `pyproject.toml`
- Checks untyped definitions
- Ignores missing imports (useful for PyTorch)

### 6. Pre-commit Hooks
Pre-commit hooks automatically run checks before each commit.

```bash
# Install hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Update hooks to latest versions
pre-commit autoupdate
```

**Configuration**: See `.pre-commit-config.yaml`

## Running All Checks

To run all code quality checks at once:

```bash
# Format code
black .
isort .

# Run linters
flake8 .
pylint src/
mypy src/

# Run tests
pytest tests/ --cov=src
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_my_module.py

# Run in parallel
pytest -n auto
```

**Configuration**: See `[tool.pytest.ini_options]` in `pyproject.toml`

## Best Practices

### Code Style
1. Follow PEP 8 guidelines
2. Use type hints where possible
3. Keep line length to 100 characters
4. Write descriptive variable and function names
5. Add docstrings to classes and functions

### PyTorch-Specific Guidelines
1. Use `torch.nn.Module` for neural network components
2. Prefer `torch.nn.functional` for stateless operations
3. Use `torch.utils.data.Dataset` and `DataLoader` for data handling
4. Set random seeds for reproducibility
5. Use device-agnostic code (`to(device)`)

### Git Workflow
1. Create a feature branch for each change
2. Write clear commit messages
3. Ensure all tests pass before committing
4. Pre-commit hooks will run automatically
5. Request code review for pull requests

### Documentation
1. Add docstrings to all public functions and classes
2. Update README.md for significant changes
3. Include usage examples in docstrings
4. Document PyTorch tensor shapes in comments

## Continuous Integration

This project uses GitHub Actions for CI/CD:
- Runs on Python 3.8, 3.9, 3.10, 3.11
- Checks code formatting (Black, isort)
- Runs linters (Flake8, Pylint)
- Performs type checking (MyPy)
- Runs test suite with coverage

See `.github/workflows/code-quality.yml` for details.

## Questions?

If you have any questions about contributing or code quality tools, please open an issue on GitHub.
