# adaptive-muon

A PyTorch project template with comprehensive code quality enforcement tools.

[![Code Quality](https://github.com/iliasmc/adaptive-muon/actions/workflows/code-quality.yml/badge.svg)](https://github.com/iliasmc/adaptive-muon/actions/workflows/code-quality.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

This repository demonstrates best practices for maintaining high code quality in PyTorch projects. It includes:

- **Code Formatting**: Black and isort for consistent code style
- **Linting**: Flake8 and Pylint for catching code issues
- **Type Checking**: MyPy for static type analysis
- **Pre-commit Hooks**: Automated checks before each commit
- **CI/CD**: GitHub Actions workflow for continuous integration
- **Testing**: pytest with coverage reporting

## Quick Start

### Installation

#### Option 1: Using Conda (Recommended)

Conda provides better dependency management and includes optimized PyTorch builds.

1. Clone the repository:
```bash
git clone https://github.com/iliasmc/adaptive-muon.git
cd adaptive-muon
```

2. Create and activate the conda environment:

**For GPU (CUDA 11.8):**
```bash
conda env create -f environment.yml
conda activate adaptive-muon
```

**For CPU-only:**
```bash
conda env create -f environment-cpu.yml
conda activate adaptive-muon-cpu
```

**Note:** Edit `environment.yml` to change CUDA version:
- For CUDA 12.1: `pytorch-cuda=12.1`
- For CPU-only: `cpuonly`

3. Install pre-commit hooks:
```bash
pre-commit install
```

#### Option 2: Using pip and venv

1. Clone the repository:
```bash
git clone https://github.com/iliasmc/adaptive-muon.git
cd adaptive-muon
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

## Code Quality Tools

### 1. Black (Code Formatter)
Black automatically formats Python code to a consistent style.

```bash
# Format all files
black .

# Check without modifying
black --check .
```

**Configuration**: `pyproject.toml` - Line length: 100 characters

### 2. isort (Import Sorter)
Automatically sorts and organizes imports.

```bash
# Sort imports
isort .

# Check only
isort --check-only .
```

**Configuration**: `pyproject.toml` - Compatible with Black

### 3. Flake8 (Linter)
Checks code for style violations and potential errors.

```bash
flake8 .
```

**Configuration**: `.flake8` - Max line length: 100, ignores Black conflicts

### 4. Pylint (Advanced Linter)
Provides comprehensive code analysis.

```bash
pylint src/
```

**Configuration**: `pyproject.toml` - Customized for PyTorch projects

### 5. MyPy (Type Checker)
Performs static type checking.

```bash
mypy src/
```

**Configuration**: `pyproject.toml` - Configured for PyTorch compatibility

### 6. Pre-commit Hooks
Automatically runs checks before each commit.

```bash
# Run on all files manually
pre-commit run --all-files

# Update hook versions
pre-commit autoupdate
```

**Configuration**: `.pre-commit-config.yaml`

## Running All Checks

```bash
# Format code
black .
isort .

# Run all linters
flake8 .
pylint src/
mypy src/

# Run tests with coverage
pytest --cov=src --cov-report=html
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test
pytest tests/test_model.py
```

## Project Structure

```
adaptive-muon/
├── src/                      # Source code
│   ├── __init__.py
│   ├── model.py             # PyTorch models
│   └── utils.py             # Utility functions
├── tests/                   # Test files
│   ├── __init__.py
│   ├── test_model.py
│   └── test_utils.py
├── .github/
│   └── workflows/
│       └── code-quality.yml # CI/CD workflow
├── .flake8                  # Flake8 configuration
├── .pre-commit-config.yaml  # Pre-commit hooks
├── environment.yml          # Conda environment (GPU)
├── environment-cpu.yml      # Conda environment (CPU-only)
├── pyproject.toml           # Project and tool configuration
├── requirements-dev.txt     # Development dependencies (pip)
├── setup.py                 # Setup script
├── Makefile                 # Convenient commands
├── CONTRIBUTING.md          # Contribution guidelines
└── README.md               # This file
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:
- Development setup
- Code quality standards
- Testing requirements
- Git workflow
- PyTorch-specific best practices

## CI/CD

The project uses GitHub Actions for continuous integration:
- ✅ Runs on Python 3.8, 3.9, 3.10, 3.11
- ✅ Code formatting checks (Black, isort)
- ✅ Linting (Flake8, Pylint)
- ✅ Type checking (MyPy)
- ✅ Test suite with coverage

See `.github/workflows/code-quality.yml` for configuration.

## Best Practices Summary

1. **Use Conda for dependency management** - Better PyTorch integration and environment isolation
2. **Always format code** with Black and isort before committing
3. **Run linters** to catch potential issues early
4. **Add type hints** to improve code clarity and catch errors
5. **Write tests** for new functionality
6. **Use pre-commit hooks** to automate quality checks
7. **Follow PyTorch conventions** for model and data handling
8. **Document code** with clear docstrings and comments

## Dependency Management

### Why Conda?

Conda is recommended for PyTorch projects because:
- **Optimized PyTorch builds**: Conda packages include optimized BLAS/LAPACK libraries
- **CUDA management**: Automatic CUDA toolkit installation and version management
- **Cross-platform**: Consistent environments across Linux, macOS, and Windows
- **Non-Python dependencies**: Can manage C/C++ libraries PyTorch depends on

### Conda vs pip

| Feature | Conda | pip |
|---------|-------|-----|
| PyTorch optimization | ✅ Optimized builds | ⚠️ Generic builds |
| CUDA handling | ✅ Automatic | ❌ Manual |
| Dependency conflicts | ✅ Better resolution | ⚠️ Can have conflicts |
| Environment isolation | ✅ Complete | ⚠️ Python-only |
| Speed | ⚠️ Slower solver | ✅ Faster |

### Common Commands

**Conda:**
```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate adaptive-muon

# Update environment
conda env update -f environment.yml

# Remove environment
conda env remove -n adaptive-muon

# Export environment
conda env export > environment.yml
```

**pip:**
```bash
# Create virtual environment
python -m venv venv

# Activate
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements-dev.txt
```

## License

This project is provided as a template for PyTorch projects with code quality enforcement.

## Resources

- [Conda Documentation](https://docs.conda.io/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [Black Documentation](https://black.readthedocs.io/)
- [isort Documentation](https://pycqa.github.io/isort/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [Pylint Documentation](https://pylint.pycqa.org/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [Pre-commit Documentation](https://pre-commit.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [pytest Documentation](https://docs.pytest.org/)