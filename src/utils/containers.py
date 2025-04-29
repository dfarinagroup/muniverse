import os
import subprocess
import platform
from pathlib import Path

def check_docker_image_exists(image_name):
    """Check if a Docker image exists locally."""
    try:
        result = subprocess.run(
            ['docker', 'image', 'inspect', image_name],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False

def check_singularity_image_exists(image_name):
    """
    Check if a Singularity image exists in the src/environment directory.
    
    Args:
        image_name (str): The Docker image name to check for
        
    Returns:
        bool: True if the Singularity image exists in src/environment, False otherwise
    """
    # Convert Docker image name to Singularity image name
    sif_name = f"{image_name.split('/')[-1].replace(':', '_')}.sif"
    
    # Get the path to environment directory
    current_dir = Path(__file__).parent.parent.parent
    environment_dir = current_dir / "environment"

    print(sif_name, environment_dir)
    # Check if the image exists in the environment directory
    image_path = environment_dir / sif_name
    return image_path.exists()

def pull_docker_image(image_name):
    """Pull a Docker image."""
    print(f"[INFO] Pulling Docker image '{image_name}'...")
    try:
        subprocess.run(['docker', 'pull', '--platform', 'linux/amd64', image_name], check=True)
        print(f"[INFO] Successfully pulled Docker image '{image_name}'")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to pull Docker image '{image_name}': {e}")
        raise

def pull_singularity_image(image_name):
    """
    Pull a Singularity image and save it to environment directory.
    
    Args:
        image_name (str): The Docker image name to pull
    """
    # Convert Docker image name to Singularity image name
    sif_name = f"{image_name.split('/')[-1].replace(':', '_')}.sif"
    
    # Get the path to environment directory
    current_dir = Path(__file__).parent.parent.parent
    environment_dir = current_dir / "environment"
    
    # Create environment directory if it doesn't exist
    environment_dir.mkdir(exist_ok=True)
    
    # Full path for the Singularity image
    image_path = environment_dir / sif_name
    
    print(f"[INFO] Pulling Singularity image '{sif_name}' to {environment_dir}...")
    try:
        # Change to environment directory before pulling
        original_dir = os.getcwd()
        os.chdir(environment_dir)
        
        subprocess.run(['singularity', 'pull', sif_name, f"docker://{image_name}"], check=True)
        print(f"[INFO] Successfully pulled Singularity image '{sif_name}'")
        
        # Change back to original directory
        os.chdir(original_dir)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to pull Singularity image '{sif_name}': {e}")
        raise
    finally:
        # Ensure we change back to original directory even if there's an error
        os.chdir(original_dir)

def pull_container(name, engine="docker"):
    """
    Pull or verify a container image.

    Args:
        name (str): The identifier for the container image (e.g., "pranavm19/muniverse-test:neuromotion")
        engine (str): The container engine to use ("docker" or "singularity")

    Raises:
        ValueError: If the engine is not supported
        subprocess.CalledProcessError: If pulling the container fails
    """
    if engine not in ["docker", "singularity"]:
        raise ValueError(f"Unsupported container engine: {engine}")

    if engine == "docker":
        if not check_docker_image_exists(name):
            pull_docker_image(name)
        else:
            print(f"[INFO] Docker image '{name}' already exists locally")
    else:  # singularity
        if not check_singularity_image_exists(name):
            pull_singularity_image(name)
        else:
            print(f"[INFO] Singularity image for '{name}' already exists in environment")

def verify_container_engine(engine):
    """
    Verify that the specified container engine is installed and available.

    Args:
        engine (str): The container engine to verify ("docker" or "singularity")

    Returns:
        bool: True if the engine is available, False otherwise
    """
    try:
        if engine == "docker":
            subprocess.run(['docker', '--version'], capture_output=True, check=True)
        elif engine == "singularity":
            subprocess.run(['singularity', '--version'], capture_output=True, check=True)
        else:
            print(f"[WARNING] Unsupported container engine: {engine}")
            return False
        return True
    except FileNotFoundError:
        print(f"[WARNING] Container engine '{engine}' is not installed")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Container engine '{engine}' is installed but failed to run: {e}")
        return False
