# adaptive-muon

This repo contains the code for the project in the Deep Learning project at ETH Zurich. Our project proposal is [here](docs/our_project_proposal.pdf).

[![Code Quality](https://github.com/iliasmc/adaptive-muon/actions/workflows/code-quality.yml/badge.svg)](https://github.com/iliasmc/adaptive-muon/actions/workflows/code-quality.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

We want to investigate using an adaptive version of Muon, to preserve its benefits in performance and accuracy while minimizing the compuational cost. We also investigate the effect of Muon on the Hessian of the loss landscape.

## Quick Start

### Training model
`python -m src.train_model`

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
