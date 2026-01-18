import itertools
import json
import os
import subprocess
import sys
from datetime import datetime


def run_experiments(config_path):
    # 1. Load the configuration
    with open(config_path, 'r') as f:
        config_space = json.load(f)

    # 2. Extract keys and values
    keys = list(config_space.keys())
    values = []
    
    # Handle both scalars and lists in config
    for key in keys:
        val = config_space[key]
        if isinstance(val, list):
            values.append(val)
        else:
            values.append([val])

    # 3. Generate all combinations (Cartesian Product)
    combinations = list(itertools.product(*values))
    
    print(f"Total experiments to run: {len(combinations)}")

    # 4. Loop through each combination and run the script
    for i, combo in enumerate(combinations):
        # Create a dictionary for this specific run
        current_params = dict(zip(keys, combo))
        
        # Generate a timestamp for this run (to ensure we know the output path)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Build the command line arguments for training
        # Use sys.executable and -m to run as module
        cmd = [sys.executable, "-m", "src.train_model"]
        
        # Pass the timestamp explicitly
        cmd.append("--timestamp")
        cmd.append(ts)

        for key, value in current_params.items():
            cmd.append(f"--{key}")
            # All arguments in train_model.py now expect a value (e.g. type=str2bool),
            # so we pass the value explicitly instead of using --flag/--no-flag.
            if isinstance(value, bool):
                cmd.append(str(value).lower())
            else:
                cmd.append(str(value))
        
        print(f"\n>>> Running Experiment {i+1}/{len(combinations)}")
        print(f"Command: {' '.join(cmd)}")
        
        # Determine the expected output path
        results_root = current_params.get("results_root", ".")
        run_path = os.path.join(results_root, ts)
        print(f"Expected Output Path: {run_path}")
        
        # Execute the process and wait for it to finish
        try:
            # Run directly, letting stdout flow to terminal naturally
            subprocess.run(cmd, check=True)

            # After training, run hessian metrics
            if os.path.exists(run_path):
                print(f"Dataset detected at: {run_path}")
                hessian_cmd = [
                    sys.executable, 
                    "-m", 
                    "src.hessian_metrics", 
                    "--run_path", 
                    run_path
                ]
                print(f"Running Hessian Metrics: {' '.join(hessian_cmd)}")
                subprocess.run(hessian_cmd, check=True)
            else:
                print(f"Warning: Expected output directory {run_path} does not exist. Skipping Hessian metrics.")

        except subprocess.CalledProcessError as e:
            print(f"Experiment {i+1} failed with error: {e}")

if __name__ == "__main__":
    # Ensure these filenames match your actual files
    run_experiments("run_configs/muon_optimal_config.yaml")
    run_experiments("run_configs/sgd_optimal_config.yaml")
    # run_experiments("run_configs/ablation_study_config.yaml")
