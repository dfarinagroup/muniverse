import subprocess
import os
import yaml
from easydict import EasyDict as edict

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return edict(config)

def generate_dataset(config_path):
    """
    Generate a dataset using the specified configuration file.

    Args:
        config_path (str): Path to the YAML configuration file containing:
            - engine: "docker" or "singularity"
            - sim_script: Path to the data generation script
            - output_dir: Output directory for the generated data
            - input_config: Path to the input configuration for run_neuromotion.py

    Returns:
        str: Absolute path to the output directory containing the generated dataset.
    """
    # Load configuration
    config = load_config(config_path)
    
    # Get required parameters with defaults
    engine = config.get("engine", "docker")
    sim_script = os.path.abspath(config["sim_script"])
    output_dir = os.path.abspath(config["output_dir"])
    input_config = os.path.abspath(config["input_config"])

    # Execute the shell script using subprocess
    try:
        subprocess.run(
            ["./run.sh", engine, sim_script, input_config, output_dir],
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        print(f"[INFO] Data generated successfully at {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Data generation failed: {e}")
        raise

    return output_dir
