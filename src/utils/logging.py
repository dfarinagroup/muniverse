"""
Logging utilities for tracking runs and their metadata.
"""

import os
import json
import time
import platform
import subprocess
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

class RunLogger:
    """Class for logging run metadata and execution details."""
    
    def __init__(self, run_id: Optional[str] = None):
        """
        Initialize the run logger.
        
        Args:
            run_id: Optional run ID. If not provided, one will be generated.
        """
        self.run_id = run_id or self._generate_run_id()
        self.start_time = datetime.now()
        self.log_data = {
            "Run_id": self.run_id,
            "Simulation_config": None,
            "Runtime_environment": {
                "Host": {
                    "Hostname": None,
                    "OS": None,
                    "Kernel": None,
                    "CPU": None,
                    "GPU": [],
                    "RAM_GB": None
                },
                "Container": {
                    "Engine": None,
                    "Engine_version": None,
                    "Image": None,
                    "Image_id": None
                }
            },
            "Code": { 
                "Repository": None,
                "URL": None,
                "Branch": None,
                "Commit": None,
            },
            "Execution": {
                "Timing": {
                    "Start": self.start_time.isoformat(),
                    "End": None,
                },
                "Return_codes": {}
            },
            "Outputs": {
                "Files": [],
                "Metadata": {}
            }
        }

        # Add code information
        git_info = self._get_git_info()
        self.log_data["Code"] = git_info
        
        # Add host information
        self.log_data["Runtime_environment"]["Host"] = self._get_host_info()
    
    def _generate_run_id(self) -> str:
        """Generate a unique run ID based on timestamp and random string."""
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        random_str = os.urandom(4).hex()
        return f"{timestamp}_{random_str}"
    
    def _get_git_info(self) -> Dict[str, Any]:
        """Get git repository information."""
        try:
            # Get repository URL
            repo_url = subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            # Extract repository name from URL
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            
            # Get current branch
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            # Get commit hash
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            return {
                "Name": repo_name,
                "URL": repo_url,
                "Branch": branch,
                "Commit": commit,
            }
        except subprocess.CalledProcessError:
            return {
                "Name": "unknown",
                "URL": "unknown",
                "Branch": "unknown",
                "Commit": "unknown",
            }
    
    def _get_host_info(self) -> Dict[str, Any]:
        """Get host system information."""
        return {
            "OS": platform.system() + " " + platform.release(),
            "Kernel": platform.version(),
            "CPU": platform.processor(),
            "GPU": self._get_gpu_info(),
            "RAM_GB": self._get_ram_info()
        }
    
    def _get_gpu_info(self) -> List[str]:
        """Get GPU information if available."""
        try:
            nvidia_smi = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            return [line for line in nvidia_smi.split("\n") if line]
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []
    
    def _get_ram_info(self) -> float:
        """Get total RAM in GB."""
        try:
            if platform.system() == "Linux":
                with open("/proc/meminfo") as f:
                    for line in f:
                        if "MemTotal" in line:
                            return float(line.split()[1]) / (1024 * 1024)  # Convert to GB
            elif platform.system() == "Darwin":
                total = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode()
                return float(total) / (1024 * 1024 * 1024)  # Convert to GB
            else:
                return 0.0
        except:
            return 0.0
    
    def set_config(self, config_content: Dict[str, Any]):
        """
        Set configuration information.
        
        Args:
            config_content: Complete configuration content
        """
        self.log_data["Simulation_config"] = config_content
    
    def set_container_info(self, engine: str, engine_version: str, image: str, image_id: str):
        """
        Set container information.
        
        Args:
            engine: Container engine (e.g., "docker", "singularity")
            engine_version: Version of the container engine
            image: Container image name
            image_id: Container image ID
        """
        self.log_data["Runtime_environment"]["Container"] = {
            "Engine": engine,
            "Engine_version": engine_version,
            "Image": image,
            "Image_id": image_id
        }
    
    def add_output(self, path: str, size_bytes: int, checksum: Optional[str] = None):
        """
        Add an output file to the log.
        
        Args:
            path: Path to the output file
            size_bytes: Size of the file in bytes
            checksum: Optional checksum of the file
        """
        if checksum is None:
            checksum = self._calculate_file_checksum(path)
            
        self.log_data["Outputs"]["Files"].append({
            "Path": path,
            "Size_bytes": size_bytes,
            "Checksum": checksum
        })
    
    def set_return_code(self, script_name: str, code: int):
        """
        Set the return code for a script.
        
        Args:
            script_name: Name of the script
            code: Return code
        """
        self.log_data["Execution"]["Return_codes"][script_name] = code
    
    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of a file."""
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except:
            return "unknown"
    
    def finalize(self, output_dir: str):
        """
        Finalize the log and save it to a file.
        
        Args:
            output_dir: Directory to save the log file
        """
        # Update timing information
        end_time = datetime.now()
        self.log_data["Execution"]["Timing"].update({
            "End": end_time.isoformat(),
        })
        
        # Save log file
        log_path = os.path.join(output_dir, f"run_log_{self.run_id}.json")
        with open(log_path, "w") as f:
            json.dump(self.log_data, f, indent=2)
        
        return log_path 