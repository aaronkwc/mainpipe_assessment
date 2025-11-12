"""
Export module for creating training-ready data shards and mixtures.
"""
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


class DataExporter:
    """Handles exporting processed data in training-ready formats."""
    
    def __init__(self, output_dir: str = "data/processed"):
        """
        Initialize data exporter.
        
        Args:
            output_dir: Directory to save exported data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_shards(
        self,
        data: List[Dict],
        shard_size: int = 10000,
        prefix: str = "shard"
    ) -> List[Path]:
        """
        Create data shards for distributed training.
        
        Args:
            data: List of data samples
            shard_size: Number of samples per shard
            prefix: Prefix for shard filenames
            
        Returns:
            List of paths to created shards
        """
        logger.info(f"Creating shards with size {shard_size}")
        
        num_shards = (len(data) + shard_size - 1) // shard_size
        shard_paths = []
        
        for i in range(num_shards):
            start_idx = i * shard_size
            end_idx = min((i + 1) * shard_size, len(data))
            shard_data = data[start_idx:end_idx]
            
            shard_path = self.output_dir / f"{prefix}_{i:05d}.jsonl"
            with open(shard_path, 'w') as f:
                for item in shard_data:
                    f.write(json.dumps(item) + '\n')
            
            shard_paths.append(shard_path)
            logger.info(f"Created shard {i+1}/{num_shards}: {shard_path} ({len(shard_data)} samples)")
        
        return shard_paths
    
    def create_mixture(
        self,
        datasets: Dict[str, List[Dict]],
        mixture_weights: Optional[Dict[str, float]] = None,
        output_name: str = "mixture"
    ) -> Path:
        """
        Create a mixture of multiple datasets.
        
        Args:
            datasets: Dictionary mapping dataset names to data lists
            mixture_weights: Optional weights for mixing datasets
            output_name: Name for output mixture file
            
        Returns:
            Path to created mixture file
        """
        logger.info(f"Creating mixture from {len(datasets)} datasets")
        
        # Default to equal weights
        if mixture_weights is None:
            mixture_weights = {name: 1.0 for name in datasets.keys()}
        
        # Normalize weights
        total_weight = sum(mixture_weights.values())
        mixture_weights = {k: v/total_weight for k, v in mixture_weights.items()}
        
        # Calculate samples per dataset
        total_samples = sum(len(data) for data in datasets.values())
        samples_per_dataset = {
            name: int(len(data) * mixture_weights.get(name, 0))
            for name, data in datasets.items()
        }
        
        # Mix the datasets
        mixed_data = []
        for name, data in datasets.items():
            n_samples = samples_per_dataset[name]
            if n_samples > len(data):
                n_samples = len(data)
            
            # Sample from this dataset
            if n_samples < len(data):
                indices = np.random.choice(len(data), n_samples, replace=False)
                sampled_data = [data[i] for i in indices]
            else:
                sampled_data = data
            
            # Add dataset label
            for item in sampled_data:
                item['dataset'] = name
            
            mixed_data.extend(sampled_data)
        
        # Shuffle the mixture
        np.random.shuffle(mixed_data)
        
        # Save mixture
        mixture_path = self.output_dir / f"{output_name}.jsonl"
        with open(mixture_path, 'w') as f:
            for item in mixed_data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Created mixture: {mixture_path} ({len(mixed_data)} samples)")
        
        # Save mixture metadata
        metadata = {
            'datasets': list(datasets.keys()),
            'mixture_weights': mixture_weights,
            'samples_per_dataset': samples_per_dataset,
            'total_samples': len(mixed_data)
        }
        metadata_path = self.output_dir / f"{output_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return mixture_path
    
    def export_jsonl(self, data: List[Dict], filename: str) -> Path:
        """
        Export data as JSONL format.
        
        Args:
            data: List of data samples
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Exported {len(data)} samples to {filepath}")
        return filepath
    
    def export_pickle(self, data: Any, filename: str) -> Path:
        """
        Export data as pickle format.
        
        Args:
            data: Data to export
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        filepath = self.output_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Exported data to {filepath}")
        return filepath
    
    def export_numpy(self, data: np.ndarray, filename: str) -> Path:
        """
        Export data as numpy array.
        
        Args:
            data: Numpy array to export
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        filepath = self.output_dir / filename
        np.save(filepath, data)
        
        logger.info(f"Exported numpy array to {filepath}")
        return filepath
