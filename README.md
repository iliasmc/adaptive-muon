# adaptive-muon

This repo contains the code for the project in the Deep Learning project at ETH Zurich. Our project proposal is [here](docs/our_project_proposal.pdf).


## Overview

We want to investigate using an adaptive version of Muon, to preserve its benefits in performance and accuracy while minimizing the compuational cost. We also investigate the effect of Muon on the Hessian of the loss landscape.

## Installation

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
pip install -r requirements.txt
```

## Running experiments
To run a series of experiments, set the experimental configs in `run_config.json` as seen below. In this case, it takes the cartesian product / all possible combinations of the specified params and runs and experiment for each combination. For instance the setup below will run 4 experiments.
```python
{
    "block1": 24,
    "block2": 48,
    "block3": 48,
    "batch_size": 512,
    "val_split": 0.1,
    "whitening": true,
    "num_epochs": 100,
    "bias_lr": 0.01,
    "head_lr": 0.6,
    "conv_optimizer": ["muon", "sgd"],
    "conv_lr": [0.01, 0.03],
    "conv_momentum": 0.85,
    "scheduler": "cosine",
    "sgd_momentum": 0.85,
    "results_root": "."
}
```

Then, run `python -m src.launcher` to run the expriments provided in `run_config.json`. After running, you can see the results on https://wandb.ai/adaptive-muon, under `cnn-training` for logs related to training and `hessian-metrics` for logs related to the hessian statistics. While all needed results should be hosted on WANDB now, you will also get additional results in the `results_root` directory that you specified in `run_config.json` (by default it will be in the directory of adaptive-muon). In the local results, the model_weights are stored as well as a plot for the accuracy and a plot for the hessian metric results.

Make sure to cleanup the results under `results_root` from time to time to avoid cluttering and going over the 20gb space limit.

To run experiments as a slurm job, you can call the following, adjusting the GPU time (max time is 24hr I think). One full experiment run takes about 10min for training + 40min for hessian stats = 50mins, so keep that in mind. Doing hyperaparamter optimization will be easy and fast, but studing hessian metrics will be a bottleneck.

```batch
sbatch --time=04:00:00 --mem-per-cpu=10000 -A deep_learning --wrap="python -m src.launcher"
```

## Hyperparamter tuning
We used sweeps from Weights & Biases to tune our hyperparamters. You can run it with:

```bash
wandb sweep sweep_muon.yaml --project muon_sweep --entity adaptive-muon
```

This will return a `sweep_id` which you can then use to kick off a sweep with:

```bash
wandb agent adaptive-muon/muon_sweep/{sweep_id}
```

If you want to do this on the cluster, you should create a new sweep locally (or on the login node) and then run this:

```bash
sbatch --time=08:00:00 --mem-per-cpu=10000 -A deep_learning --wrap="wandb agent adaptive-muon/muon_sweep/{sweep_id} --count 1"
```


## Code Quality Tools

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
