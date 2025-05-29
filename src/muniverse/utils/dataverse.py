import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import time
import pandas as pd
import hashlib
from pyDataverse.api import NativeApi
from pyDataverse.models import Dataverse, Dataset, Datafile

class DataverseUploader:
    def __init__(
        self,
        api_token: str,
        dataverse_url: str,
        dataverse_alias: str,
        config_path: Optional[str] = None,
        dataset_pid: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        checksum_type: str = 'md5'
    ):
        """Initialize the Dataverse uploader.
        
        Args:
            api_token: Dataverse API token
            dataverse_url: URL of the Dataverse instance
            dataverse_alias: Alias of the target Dataverse
            config_path: Optional path to a JSON config file containing the above parameters
            dataset_pid: Optional persistent ID of an existing dataset to work with
            max_retries: Maximum number of retry attempts for failed uploads
            retry_delay: Delay in seconds between retry attempts
            checksum_type: Type of checksum to use ('md5' or 'sha256')
        """
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                api_token = config.get('api_token', api_token)
                dataverse_url = config.get('dataverse_url', dataverse_url)
                dataverse_alias = config.get('dataverse_alias', dataverse_alias)
                dataset_pid = config.get('dataset_pid', dataset_pid)
                max_retries = config.get('max_retries', max_retries)
                retry_delay = config.get('retry_delay', retry_delay)
                checksum_type = config.get('checksum_type', checksum_type)
        
        self.api = NativeApi(dataverse_url, api_token)
        self.dataverse_alias = dataverse_alias
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.checksum_type = checksum_type
        self._files_df = pd.DataFrame(columns=['label', 'directoryLabel', 'checksum'])
        self._verify_connection()
        self._print_dataverse_info()
        
        if dataset_pid:
            self.set_dataset(dataset_pid)
    
    def _verify_connection(self) -> None:
        """Verify the connection to Dataverse."""
        response = self.api.get_info_version()
        if not response.json()['status'] == 'OK':
            raise ConnectionError(f"Failed to connect to Dataverse: {response.text}")
        
    def _print_dataverse_info(self) -> None:
        """Pretty print the Dataverse info."""
        datasets = self.list_datasets()
        n_datasets = len(datasets)

        print(f"Connected to Dataverse: {self.dataverse_alias}, with {n_datasets} datasets.")
        if n_datasets > 0:
            print("To proceed, select an existing dataset by calling set_dataset(dataset_pid) with the dataset's persistent ID.")
        else:
            print("No datasets found. Create a new dataset directly in the Dataverse web interface. Bye!")
        
    def get_dataverse_info(self) -> Dict:
        """Get information about the Dataverse instance."""
        response = self.api.get_dataverse(self.dataverse_alias) 
        if not response.json()['status'] == 'OK':
            raise RuntimeError(f"Failed to get dataverse info: {response.text}")
        return response.json()['data']
    
    def list_datasets(self, limit: int = 10, offset: int = 0) -> List[Dict]:
        """List datasets in the Dataverse.
        Returns:
            List[Dict]: List of dataset information dictionaries
        """
        response = self.api.get_dataverse_contents(self.dataverse_alias)
        if not response.json()['status'] == 'OK':
            raise RuntimeError(f"Failed to list datasets: {response.text}")
        
        # Filter for datasets
        datasets = [
            item for item in response.json()['data']
            if item.get('type') == 'dataset'
        ]
        
        return datasets
    
    def get_dataset_info(self, dataset_pid: Optional[str] = None) -> Dict:
        """Get detailed information about a specific dataset.
        
        Args:
            dataset_pid: Optional dataset persistent ID (overrides current dataset)
            
        Returns:
            Dict: Dataset information
        """
        if dataset_pid is None:
            if self.dataset_pid is None:
                raise ValueError("No dataset specified. Either set dataset_pid in constructor, use set_dataset(), or provide dataset_pid parameter.")
            dataset_pid = self.dataset_pid
            
        response = self.api.get_dataset(dataset_pid)
        if not response.json()['status'] == 'OK':
            raise RuntimeError(f"Failed to get dataset info: {response.text}")
        
        return response.json()['data']
    
    def set_dataset(self, dataset_pid: str) -> None:
        """Set the current dataset to work with and update files metadata.
        
        Args:
            dataset_pid: Dataset persistent ID (DOI)
        """
        self.dataset_pid = dataset_pid
        self._update_files_metadata()
    
    def _update_files_metadata(self) -> None:
        """Update the files metadata DataFrame from the dataset."""
        if not self.dataset_pid:
            return
            
        response = self.api.get_datafiles_metadata(self.dataset_pid)
        if not response.json()['status'] == 'OK':
            raise RuntimeError(f"Failed to get files metadata: {response.text}")
            
        files_data = response.json()['data']
        files_list = []
        
        for file in files_data:
            files_list.append({
                'label': file.get('label', ''),
                'directoryLabel': file.get('directoryLabel', ''),
                'checksum': file.get('dataFile', {}).get('md5', '')
            })
            
        self._files_df = pd.DataFrame(files_list)

    def _compute_checksum(self, file_path: Union[str, Path]) -> str:
        """Compute checksum of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: Checksum of the file
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        hash_func = hashlib.md5 if self.checksum_type == 'md5' else hashlib.sha256
        
        with open(file_path, 'rb') as f:
            file_hash = hash_func()
            # Read the file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b''):
                file_hash.update(chunk)
                
        return file_hash.hexdigest()

    def find_duplicate_by_checksum(self, file_path: Union[str, Path]) -> Optional[Dict]:
        """Find a duplicate file in the dataset by comparing checksums.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            Optional[Dict]: Dictionary with duplicate file info if found, None otherwise
        """
        print(f"Finding duplicate by checksum for {file_path}")
        if self._files_df.empty:
            return None
            
        file_checksum = self._compute_checksum(file_path)
        print(f"File checksum: {file_checksum}")
        duplicates = self._files_df[self._files_df['checksum'] == file_checksum]
        
        if not duplicates.empty:
            return duplicates.iloc[0].to_dict()
        return None

    def file_exists(self, local_filename: str, remote_directory: Optional[str] = None, local_path: Optional[Union[str, Path]] = None) -> bool:
        """Check if a local file already exists in the remote dataset.
        
        Args:
            local_filename: Name of the file in the local directory (and remote dataset)
            remote_directory: Optional directory path within the dataset
            local_path: Optional full path to the local file for checksum comparison
            
        Returns:
            bool: True if the file exists, False otherwise
        """
        if self._files_df.empty:
            return False
            
        # Check if the file is a .TSV and if so, check for a .TAB equivalent
        if local_filename.endswith('.tsv'):
            local_filename = local_filename[:-4] + '.tab'
        
        # First check by name and path
        mask = self._files_df['label'] == local_filename
        if remote_directory:
            mask &= self._files_df['directoryLabel'] == remote_directory
            
        if mask.any():
            return True
            
        # If we have a local file path, check by checksum
        if local_path is not None:
            try:
                duplicate = self.find_duplicate_by_checksum(local_path)
                if duplicate is not None:
                    print(f"Found duplicate by checksum: {duplicate['label']}")
                    return True
            except Exception as e:
                print(f"Warning: Could not compute checksum: {str(e)}")
                
        return False

    def upload_file(
        self,
        local_path: Union[str, Path],
        metadata: Optional[Dict] = None,
        dataset_pid: Optional[str] = None,
        remote_directory: Optional[str] = None,
        skip_existing: bool = True,
        dry_run: bool = True
    ) -> Dict:
        """Upload a file to a dataset.
        
        Args:
            local_path: Path to the local file to upload
            metadata: Additional metadata for the file
            dataset_pid: Optional dataset persistent ID (overrides current dataset)
            remote_directory: Optional path within the dataset where the file should be stored
                           (e.g., "sub-01/ses-01/emg" for BIDS structure)
            skip_existing: Whether to skip uploading if the file already exists
            dry_run: If True, only check if file would be uploaded without actually uploading
            
        Returns:
            Dict: Response from the upload operation or dry-run status
        """
        if dataset_pid is None:
            if self.dataset_pid is None:
                raise ValueError("No dataset specified. Either set dataset_pid in constructor, use set_dataset(), or provide dataset_pid parameter.")
            dataset_pid = self.dataset_pid
            
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")
        
        # Check if file already exists
        print(f"Checking if {local_path.name} exists in the remote dataset")
        if skip_existing and self.file_exists(local_path.name, remote_directory, local_path):
            print(f"Skipping {local_path.name} as it already exists in the remote dataset")
            return {"status": "skipped", "message": "File already exists in remote dataset"}
        
        # Prepare the datafile metadata
        df_metadata = {
            'pid': dataset_pid,
            'filename': local_path.name,
            'metadata': metadata or {}
        }
        
        # Add directory path if specified
        if remote_directory:
            df_metadata['directoryLabel'] = remote_directory
        
        df = Datafile()
        df.set(df_metadata)
        
        if dry_run:
            print(f"[DRY RUN] Would upload {local_path.name} to {remote_directory or 'root'}")
            if metadata:
                print(f"[DRY RUN] With metadata: {metadata}")
            return {
                "status": "dry_run",
                "message": "File would be uploaded",
                "metadata": df_metadata
            }
        
        # Implement retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.api.upload_datafile(dataset_pid, str(local_path), df.json())
                if not response.json()['status'] == 'OK':
                    raise RuntimeError(f"Failed to upload file: {response.text}")
                return response.json()
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    print(f"Upload attempt {attempt + 1} failed, retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    raise RuntimeError(f"Failed to upload file after {self.max_retries} attempts: {str(last_error)}")
    
    def upload_directory(
        self,
        local_directory: Union[str, Path],
        metadata: Optional[Dict] = None,
        dataset_pid: Optional[str] = None,
        remote_base_directory: Optional[str] = None,
        skip_existing: bool = True,
        dry_run: bool = True
    ) -> List[Dict]:
        """Upload all files in a directory to a dataset.
        
        Args:
            local_directory: Path to the local directory containing files to upload
            metadata: Additional metadata for the files
            dataset_pid: Optional dataset persistent ID (overrides current dataset)
            remote_base_directory: Optional base path within the dataset where files should be stored
                                (e.g., "sub-01" for BIDS structure)
            skip_existing: Whether to skip uploading if files already exist
            dry_run: If True, only check if files would be uploaded without actually uploading
            
        Returns:
            List[Dict]: List of responses from upload operations or dry-run status
        """
        if dataset_pid is None:
            if self.dataset_pid is None:
                raise ValueError("No dataset specified. Either set dataset_pid in constructor, use set_dataset(), or provide dataset_pid parameter.")
            dataset_pid = self.dataset_pid
            
        local_directory = Path(local_directory)
        if not local_directory.exists():
            raise FileNotFoundError(f"Directory not found: {local_directory}")
        
        responses = []
        for file_path in local_directory.rglob("*"):
            if file_path.is_file():
                try:
                    # Calculate the relative path within the dataset
                    rel_path = file_path.relative_to(local_directory)
                    remote_directory = str(rel_path.parent)
                    
                    # Combine with base directory path if provided
                    if remote_base_directory:
                        remote_directory = f"{remote_base_directory}/{remote_directory}" if remote_directory != "." else remote_base_directory
                    
                    response = self.upload_file(
                        file_path,
                        metadata,
                        dataset_pid,
                        remote_directory if remote_directory != "." else None,
                        skip_existing,
                        dry_run
                    )
                    responses.append(response)
                except Exception as e:
                    print(f"Failed to upload {file_path}: {str(e)}")
        
        self._update_files_metadata()

        return responses
