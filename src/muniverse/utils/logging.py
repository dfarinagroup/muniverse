"""
Logging utilities for tracking runs and their metadata.
"""

import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class BaseMetadataLogger:
    """Base class for logging metadata with BIDS-compatible structure."""

    def __init__(self):
        """Initialize the base metadata logger."""
        self.start_time = datetime.now()
        self.log_data = {
            "BIDSVersion": "1.8.0",
            "DatasetType": "derivative",
            "PipelineDescription": {
                "Name": "MUniverse",
                "Version": "0.0.1",
                "License": "MIT",
            },
            "GeneratedBy": [],
            "InputData": {},
            "OutputData": {},
            "RuntimeEnvironment": {
                "Host": {
                    "Hostname": None,
                    "OS": None,
                    "Kernel": None,
                    "CPU": None,
                    "GPU": [],
                    "RAM_GB": None,
                },
                "Container": {
                    "Engine": None,
                    "Engine_version": None,
                    "Image": None,
                    "Image_id": None,
                },
            },
            "Execution": {
                "Timing": {
                    "Start": self.start_time.isoformat(),
                    "End": None,
                },
                "ReturnCodes": {},
            },
        }

        # Add host information
        self.log_data["RuntimeEnvironment"]["Host"] = self._get_host_info()

    def _get_host_info(self) -> Dict[str, Any]:
        """Get host system information."""
        return {
            "Hostname": platform.node(),
            "OS": platform.system() + " " + platform.release(),
            "Kernel": platform.version(),
            "CPU": platform.processor(),
            "GPU": self._get_gpu_info(),
            "RAM_GB": self._get_ram_info(),
        }

    def _get_gpu_info(self) -> List[str]:
        """Get GPU information if available."""
        try:
            nvidia_smi = (
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,driver_version",
                        "--format=csv,noheader",
                    ],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
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
                            return float(line.split()[1]) / (
                                1024 * 1024
                            )  # Convert to GB
            elif platform.system() == "Darwin":
                total = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode()
                return float(total) / (1024 * 1024 * 1024)  # Convert to GB
            else:
                return 0.0
        except:
            return 0.0

    def _get_container_info(self, engine: str, container: str) -> Dict[str, str]:
        """Get container information based on the engine type.

        Args:
            engine: Container engine ('docker' or 'singularity')
            container: Container name (for Docker) or full path (for Singularity)

        Returns:
            Dict containing container information (name, id, created)
        """
        try:
            # Get container info using inspect
            inspect_cmd = [engine, "inspect"]
            if engine == "singularity":
                inspect_cmd.append("--json")
            inspect_cmd.append(container)

            inspect_output = subprocess.check_output(inspect_cmd).decode()
            inspect_data = json.loads(inspect_output)

            # Extract info based on engine
            if engine == "docker":
                inspect_data = inspect_data[0]  # Docker returns a list
                return {
                    "name": (
                        inspect_data["RepoTags"][0]
                        if inspect_data["RepoTags"]
                        else container
                    ),
                    "id": inspect_data["Id"],
                }
            else:  # singularity
                return {
                    "name": os.path.basename(container),
                    "id": inspect_data.get("data", {})
                    .get("attributes", {})
                    .get("id", "unknown"),
                }

        except Exception as e:
            print(f"Warning: Could not get {engine} container info: {e}")
            # Return appropriate name based on engine
            return {
                "name": (
                    os.path.basename(container)
                    if engine == "singularity"
                    else container
                ),
                "id": "unknown",
            }

    def set_return_code(self, script_name: str, code: int):
        """Set the return code for a script."""
        self.log_data["Execution"]["ReturnCodes"][script_name] = code

    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of a file."""
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except:
            return "unknown"

    def finalize(
        self,
        engine: Optional[str] = None,
        container: Optional[str] = None,
    ) -> None:
        """Finalize the log by adding the end time and container information.

        Args:
            engine: Optional container engine name
            container: Optional container name/path
        """
        end_time = datetime.now()
        self.log_data["Execution"]["Timing"]["End"] = end_time.isoformat()

        # Update container info if engine and container are provided
        if engine and container:
            image_info = self._get_container_info(engine, container)
            self.log_data["RuntimeEnvironment"]["Container"] = {
                "Engine": engine,
                "Engine_version": subprocess.check_output([engine, "--version"])
                .decode()
                .strip(),
                "Image": image_info["name"],
                "Image_id": image_info["id"],
            }

    def _get_package_root(self) -> Path:
        """Get the root directory of the package, handling both local and pip installations."""
        # Try to get the package root from the current file's location
        current_path = Path(__file__).parent.parent.parent

        # Check if we're in a pip installation (site-packages)
        if "site-packages" in str(current_path):
            # In pip installation, we need to get the actual package root
            try:
                import muniverse

                return Path(muniverse.__file__).parent
            except ImportError:
                # If we can't import the package, fall back to current path
                return current_path
        else:
            # We're in local development
            return current_path

    def _get_git_info(self, repo_path: str) -> Dict[str, Any]:
        """Get git repository information for a specific path."""
        try:
            # Get repository URL
            repo_url = (
                subprocess.check_output(
                    ["git", "-C", repo_path, "config", "--get", "remote.origin.url"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )

            # Get current branch
            branch = (
                subprocess.check_output(
                    ["git", "-C", repo_path, "rev-parse", "--abbrev-ref", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )

            # Get commit hash
            commit = (
                subprocess.check_output(
                    ["git", "-C", repo_path, "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )

            return {
                "Name": os.path.basename(repo_path),
                "URL": repo_url,
                "Branch": branch,
                "Commit": commit,
            }
        except subprocess.CalledProcessError:
            return {
                "Name": "Muniverse",
                "URL": "https://github.com/pranavm19/muniverse.git",  # TODO: Replace with final URL
                "Branch": "main",
                "Commit": "unknown",
            }

    def add_generated_by(
        self,
        name: str,
        url: str,
        commit: str,
        branch: Optional[str] = None,
        file: Optional[str] = None,
        license: Optional[str] = None,
    ):
        """Add a generator to the GeneratedBy list.

        Args:
            name: Name of the generator
            url: Repository URL
            commit: Git commit hash
            branch: Optional git branch
            file: Optional specific file responsible for generation
            license: Optional license
        """
        generator = {"Name": name, "URL": url, "Commit": commit}
        if branch:
            generator["Branch"] = branch
        if file:
            generator["File"] = file
        if license:
            generator["License"] = license
        self.log_data["GeneratedBy"].append(generator)


class SimulationLogger(BaseMetadataLogger):
    """Logger for simulation runs."""

    def __init__(self, run_id: Optional[str] = None):
        super().__init__(run_id)
        # Update fields for simulation runs
        self.log_data["DatasetType"] = "raw"
        self.log_data["PipelineDescription"]["Name"] += " Data Generation"

        # Add simulation-specific fields
        self.log_data["InputData"] = {
            "Description": "Simulation configuration for synthetic EMG generation",
            "Configuration": {},
        }
        self.log_data["OutputData"] = {
            "Description": "Generated synthetic EMG data and metadata",
            "Files": [],
            "Metadata": {},
        }

        # Add MUniverse generator info
        package_root = self._get_package_root()
        muniverse_info = self._get_git_info(str(package_root))
        self.add_generated_by(
            name="MUniverse Data Generation",
            url=muniverse_info["URL"],
            commit=muniverse_info["Commit"],
            branch=muniverse_info["Branch"],
            file="muniverse/data_generation/generate_data.py",
            license="MIT",  # TODO: Add license
        )

        # Add NeuroMotion generator info
        self.add_generated_by(
            name="NeuroMotion",
            url="https://github.com/shihan-ma/NeuroMotion.git",
            commit="590369ec2f395e6e228aa9dc58bf0fb87a2c0329",
            license="GPL-3.0 license",
        )

    def set_config(self, config_content: Dict[str, Any]):
        """Set simulation configuration."""
        self.log_data["InputData"]["Configuration"] = config_content

    def add_output(self, path: str, size_bytes: int, checksum: Optional[str] = None):
        """Add an output file to the log."""
        if checksum is None:
            checksum = self._calculate_file_checksum(path)

        self.log_data["OutputData"]["Files"].append(
            {
                "FileName": os.path.basename(path),
                "SizeBytes": size_bytes,
                "Checksum": checksum,
            }
        )


class AlgorithmLogger(BaseMetadataLogger):
    """Logger for algorithm runs."""

    def __init__(self):
        super().__init__()

        # Update fields for algorithm runs
        self.log_data["PipelineDescription"]["Name"] += " Decomposition"

        # Add algorithm-specific fields
        self.log_data["SourceDatasets"] = (
            []
        )  # TODO: Specifies where the source dataset is stored; InputData specified the file

        self.log_data["InputData"] = {
            "FileName": None,
            "FileFormat": None,
            "Description": "Input EMG data for decomposition",
        }
        self.log_data["OutputData"] = {
            "Files": [],
            "Description": "Decomposition results and metadata",
        }
        self.log_data["AlgorithmConfiguration"] = {}
        self.log_data["ProcessingSteps"] = []

        # Add MUniverse generator info
        package_root = self._get_package_root()
        muniverse_info = self._get_git_info(str(package_root))
        self.add_generated_by(
            name="MUniverse Decomposition",
            url=muniverse_info["URL"],
            commit=muniverse_info["Commit"],
            branch=muniverse_info["Branch"],
            file="muniverse/algorithms/decomposition.py",
        )

    def set_input_data(self, file_name: str, file_format: str):
        """Set input data information."""
        self.log_data["InputData"].update(
            {"FileName": file_name, "FileFormat": file_format}
        )

    def set_algorithm_config(self, config_content: Dict[str, Any]):
        """Set algorithm configuration."""
        self.log_data["AlgorithmConfiguration"] = config_content

    def add_processing_step(self, step_name: str, details: Dict[str, Any]):
        """Add a processing step with its details."""
        self.log_data["ProcessingSteps"].append({"Step": step_name, "Details": details})

    def add_output(self, path: str, size_bytes: int, checksum: Optional[str] = None):
        """Add an output file to the log.

        Args:
            path: Path to the output file
            size_bytes: Size of the file in bytes
            checksum: Optional checksum of the file
        """
        if checksum is None:
            checksum = self._calculate_file_checksum(path)

        file_path = Path(path)
        file_info = {
            "FileName": os.path.basename(path),
            "SizeBytes": size_bytes,
            "Checksum": checksum,
        }
        
        self.log_data["OutputData"]["Files"].append(file_info)
