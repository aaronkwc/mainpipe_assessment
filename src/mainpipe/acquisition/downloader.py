"""
Data acquisition module for fetching and loading data.
"""
import os
import requests
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class DataAcquisition:
    """Handles data acquisition from various sources."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize data acquisition.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_url(self, url: str, filename: Optional[str] = None) -> Path:
        """
        Download data from a URL.
        
        Args:
            url: URL to download from
            filename: Optional filename to save as
            
        Returns:
            Path to downloaded file
        """
        if filename is None:
            filename = url.split("/")[-1]
        
        filepath = self.data_dir / filename
        
        if filepath.exists():
            logger.info(f"File {filepath} already exists, skipping download")
            return filepath
        
        logger.info(f"Downloading {url} to {filepath}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=filename
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Successfully downloaded to {filepath}")
            return filepath
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {url}: {e}")
            raise
    
    def download_urls(self, urls: List[str]) -> List[Path]:
        """
        Download multiple URLs.
        
        Args:
            urls: List of URLs to download
            
        Returns:
            List of paths to downloaded files
        """
        paths = []
        for url in urls:
            try:
                path = self.download_url(url)
                paths.append(path)
            except Exception as e:
                logger.error(f"Error downloading {url}: {e}")
        
        return paths
    
    def load_local_file(self, filepath: str) -> Path:
        """
        Load a local file.
        
        Args:
            filepath: Path to local file
            
        Returns:
            Path object
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File {filepath} not found")
        
        return path
    
    def list_files(self, pattern: str = "*") -> List[Path]:
        """
        List all files in data directory matching pattern.
        
        Args:
            pattern: Glob pattern to match files
            
        Returns:
            List of matching file paths
        """
        return list(self.data_dir.glob(pattern))
