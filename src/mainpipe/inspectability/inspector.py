"""
Inspectability module for analyzing and visualizing data statistics.
"""
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)


class DataInspector:
    """Handles data inspection, statistics, and visualization."""
    
    def __init__(self, output_dir: str = "data/reports"):
        """
        Initialize data inspector.
        
        Args:
            output_dir: Directory to save reports and visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {}
    
    def analyze_lengths(self, lengths: List[int], name: str = "text") -> Dict:
        """
        Analyze length distribution.
        
        Args:
            lengths: List of lengths
            name: Name of the measurement
            
        Returns:
            Dictionary of length statistics
        """
        if not lengths:
            return {}
        
        lengths_array = np.array(lengths)
        stats = {
            f'{name}_count': len(lengths),
            f'{name}_min': int(np.min(lengths_array)),
            f'{name}_max': int(np.max(lengths_array)),
            f'{name}_mean': float(np.mean(lengths_array)),
            f'{name}_median': float(np.median(lengths_array)),
            f'{name}_std': float(np.std(lengths_array)),
            f'{name}_p25': float(np.percentile(lengths_array, 25)),
            f'{name}_p75': float(np.percentile(lengths_array, 75)),
            f'{name}_p90': float(np.percentile(lengths_array, 90)),
            f'{name}_p95': float(np.percentile(lengths_array, 95)),
            f'{name}_p99': float(np.percentile(lengths_array, 99)),
        }
        
        self.stats.update(stats)
        return stats
    
    def plot_length_histogram(
        self,
        lengths: List[int],
        title: str = "Length Distribution",
        filename: str = "length_histogram.png",
        bins: int = 50
    ) -> Path:
        """
        Create histogram of length distribution.
        
        Args:
            lengths: List of lengths
            title: Plot title
            filename: Output filename
            bins: Number of histogram bins
            
        Returns:
            Path to saved plot
        """
        plt.figure(figsize=(12, 6))
        plt.hist(lengths, bins=bins, edgecolor='black', alpha=0.7)
        plt.xlabel('Length')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_val = np.mean(lengths)
        median_val = np.median(lengths)
        plt.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.1f}')
        plt.axvline(median_val, color='g', linestyle='--', label=f'Median: {median_val:.1f}')
        plt.legend()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved histogram to {filepath}")
        return filepath
    
    def plot_drop_reasons(
        self,
        drop_reasons: Dict[str, int],
        filename: str = "drop_reasons.png"
    ) -> Path:
        """
        Create bar chart of drop reasons.
        
        Args:
            drop_reasons: Dictionary mapping reasons to counts
            filename: Output filename
            
        Returns:
            Path to saved plot
        """
        if not drop_reasons:
            logger.warning("No drop reasons to plot")
            return None
        
        plt.figure(figsize=(12, 6))
        reasons = list(drop_reasons.keys())
        counts = list(drop_reasons.values())
        
        plt.barh(reasons, counts)
        plt.xlabel('Count')
        plt.ylabel('Drop Reason')
        plt.title('Data Drop Reasons')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add count labels
        for i, count in enumerate(counts):
            plt.text(count, i, f' {count}', va='center')
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved drop reasons chart to {filepath}")
        return filepath
    
    def analyze_language_scores(
        self,
        language_scores: Dict[str, List[float]],
        filename: str = "language_scores.png"
    ) -> Dict:
        """
        Analyze and visualize language detection scores.
        
        Args:
            language_scores: Dictionary mapping languages to score lists
            filename: Output filename for plot
            
        Returns:
            Dictionary of language statistics
        """
        stats = {}
        
        if not language_scores:
            return stats
        
        # Calculate statistics per language
        for lang, scores in language_scores.items():
            if scores:
                stats[f'{lang}_count'] = len(scores)
                stats[f'{lang}_mean_score'] = float(np.mean(scores))
                stats[f'{lang}_median_score'] = float(np.median(scores))
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Box plot of scores by language
        languages = list(language_scores.keys())
        scores_list = [language_scores[lang] for lang in languages]
        
        plt.boxplot(scores_list, labels=languages)
        plt.xlabel('Language')
        plt.ylabel('Confidence Score')
        plt.title('Language Detection Scores')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved language scores plot to {filepath}")
        self.stats.update(stats)
        return stats
    
    def analyze_duplicates(
        self,
        duplicate_markers: List[bool],
        filename: str = "duplicates.png"
    ) -> Dict:
        """
        Analyze duplicate detection results.
        
        Args:
            duplicate_markers: List of boolean duplicate markers
            filename: Output filename for plot
            
        Returns:
            Dictionary of duplicate statistics
        """
        total = len(duplicate_markers)
        duplicates = sum(duplicate_markers)
        unique = total - duplicates
        
        stats = {
            'total_samples': total,
            'duplicate_count': duplicates,
            'unique_count': unique,
            'duplicate_rate': duplicates / total if total > 0 else 0
        }
        
        # Create pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(
            [unique, duplicates],
            labels=['Unique', 'Duplicates'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['#2ecc71', '#e74c3c']
        )
        plt.title(f'Duplicate Detection Results\n(Total: {total})')
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved duplicate analysis to {filepath}")
        self.stats.update(stats)
        return stats
    
    def analyze_pii_hits(
        self,
        pii_hits: Dict[str, int],
        filename: str = "pii_hits.png"
    ) -> Dict:
        """
        Analyze PII detection hit rates.
        
        Args:
            pii_hits: Dictionary mapping PII types to hit counts
            filename: Output filename for plot
            
        Returns:
            Dictionary of PII statistics
        """
        if not pii_hits:
            return {}
        
        total_hits = sum(pii_hits.values())
        stats = {
            'total_pii_hits': total_hits,
            'pii_types': list(pii_hits.keys()),
            'pii_type_counts': pii_hits
        }
        
        # Calculate hit rates
        for pii_type, count in pii_hits.items():
            stats[f'{pii_type}_hit_rate'] = count / total_hits if total_hits > 0 else 0
        
        # Create bar chart
        plt.figure(figsize=(12, 6))
        pii_types = list(pii_hits.keys())
        counts = list(pii_hits.values())
        
        plt.bar(pii_types, counts, color='#e74c3c', alpha=0.7)
        plt.xlabel('PII Type')
        plt.ylabel('Hit Count')
        plt.title(f'PII Detection Results (Total Hits: {total_hits})')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add count labels
        for i, count in enumerate(counts):
            plt.text(i, count, f'{count}', ha='center', va='bottom')
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved PII analysis to {filepath}")
        self.stats.update(stats)
        return stats
    
    def generate_report(self, filename: str = "pipeline_report.json") -> Path:
        """
        Generate comprehensive pipeline report.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved report
        """
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Saved pipeline report to {filepath}")
        return filepath
    
    def get_stats(self) -> Dict:
        """
        Get all collected statistics.
        
        Returns:
            Dictionary of statistics
        """
        return self.stats.copy()
