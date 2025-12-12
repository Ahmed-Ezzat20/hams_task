"""
CSV Writer Module

Writes EOU samples directly to CSV format.
"""

import csv
import logging
from pathlib import Path
from typing import List, Dict, Any


class CSVWriter:
    """Writes EOU samples to CSV files."""
    
    @staticmethod
    def write_samples(
        samples: List[Dict[str, Any]],
        output_file: str,
        mode: str = 'w'
    ) -> None:
        """
        Write samples to CSV file.
        
        Args:
            samples: List of sample dictionaries with keys: utterance, style, label
            output_file: Path to output CSV file
            mode: File mode ('w' for write, 'a' for append)
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file exists and is empty
        file_exists = output_path.exists() and output_path.stat().st_size > 0
        
        with open(output_path, mode, encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            
            # Write header only if file is new or empty
            if mode == 'w' or not file_exists:
                writer.writerow(['utterance', 'style', 'label'])
            
            # Write samples
            for sample in samples:
                writer.writerow([
                    sample['utterance'],
                    sample['style'],
                    sample['label']
                ])
        
        logging.info(f"Wrote {len(samples)} samples to {output_file}")
    
    @staticmethod
    def get_statistics(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the samples.
        
        Args:
            samples: List of sample dictionaries
        
        Returns:
            Dictionary with statistics
        """
        total = len(samples)
        
        if total == 0:
            return {
                "total": 0,
                "eou_count": 0,
                "non_eou_count": 0,
                "eou_percentage": 0.0,
                "non_eou_percentage": 0.0,
                "style_distribution": {}
            }
        
        eou_count = sum(1 for s in samples if s['label'] == 1)
        non_eou_count = total - eou_count
        
        style_counts = {}
        for sample in samples:
            style = sample['style']
            style_counts[style] = style_counts.get(style, 0) + 1
        
        return {
            "total": total,
            "eou_count": eou_count,
            "non_eou_count": non_eou_count,
            "eou_percentage": (eou_count / total) * 100,
            "non_eou_percentage": (non_eou_count / total) * 100,
            "style_distribution": style_counts
        }
    
    @staticmethod
    def print_statistics(samples: List[Dict[str, Any]]) -> None:
        """
        Print statistics about the samples.
        
        Args:
            samples: List of sample dictionaries
        """
        stats = CSVWriter.get_statistics(samples)
        
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        print(f"Total samples: {stats['total']}")
        print("\nEOU Distribution:")
        print(f"  EOU (label=1):     {stats['eou_count']:4d} ({stats['eou_percentage']:.1f}%)")
        print(f"  Non-EOU (label=0): {stats['non_eou_count']:4d} ({stats['non_eou_percentage']:.1f}%)")
        print("\nStyle Distribution:")
        for style, count in stats['style_distribution'].items():
            percentage = (count / stats['total']) * 100
            print(f"  {style:12s}: {count:4d} ({percentage:.1f}%)")
        print("="*50 + "\n")
