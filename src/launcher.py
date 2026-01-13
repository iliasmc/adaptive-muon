import json
import subprocess
import itertools
from pathlib import Path

def run_experiments(config_path, script_path):
    # 1. Load the configuration
    with open(config_path, 'r') as f:
        config_space = json.load(f)

    # 2. Extract keys and values
    keys = list(config_space.keys())
    values = list(config_space.values())

    # 3. Generate all combinations (Cartesian Product)
    combinations = list(itertools.product(*values))
    
    print(f"Total experiments to run: {len(combinations)}")

    # 4. Loop through each combination and run the script
    for i, combo in enumerate(combinations):
        # Create a dictionary for this specific run
        current_params = dict(zip(keys, combo))
        
        # Build the command line arguments
        cmd = ["python", script_path]
        for key, value in current_params.items():
            cmd.append(f"--{key}")
            # Handle boolean flags or strings/numbers
            if isinstance(value, bool):
                cmd.append(str(value).lower())
            else:
                cmd.append(str(value))
        
        print(f"\n>>> Running Experiment {i+1}/{len(combinations)}")
        print(f"Command: {' '.join(cmd)}")
        
        # Execute the process and wait for it to finish
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Experiment {i+1} failed with error: {e}")

if __name__ == "__main__":
    # Ensure these filenames match your actual files
    run_experiments("run_config.json", "train_model.py")