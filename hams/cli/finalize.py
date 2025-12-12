"""
CLI command for finalizing datasets to CSV format for Hugging Face.
"""

import argparse
import logging
from pathlib import Path

from ..core.csv_exporter import CSVExporter


logger = logging.getLogger(__name__)


def main():
    """Main entry point for hams-finalize command."""
    parser = argparse.ArgumentParser(
        description='Finalize dataset to CSV format for Hugging Face'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help='Input JSONL file with conversations'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for CSV files'
    )
    parser.add_argument(
        '--create-splits',
        action='store_true',
        help='Create train/val/test splits (default: False)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training data ratio (default: 0.7)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation data ratio (default: 0.15)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Test data ratio (default: 0.15)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['huggingface', 'turn_level', 'conversation_level'],
        default='huggingface',
        help='Export format (default: huggingface)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Validate ratios
        if args.create_splits:
            total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
            if abs(total_ratio - 1.0) > 0.01:
                logger.error(f"Split ratios must sum to 1.0 (got {total_ratio})")
                return
        
        # Create splits or single export
        if args.create_splits:
            logger.info("Creating train/val/test splits...")
            splits = CSVExporter.create_splits(
                input_file=args.input_file,
                output_dir=args.output_dir,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                export_format=args.format
            )
            
            logger.info("\n=== Splits Created ===")
            for split_name, split_file in splits.items():
                # Count lines
                with open(split_file, 'r') as f:
                    num_lines = sum(1 for _ in f) - 1  # Exclude header
                logger.info(f"{split_name}: {split_file} ({num_lines} examples)")
        
        else:
            # Single file export
            logger.info(f"Exporting to {args.format} format...")
            
            from ..core.writer import DatasetWriter
            reader = DatasetWriter(args.input_file)
            conversations = reader.read_conversations()
            
            if not conversations:
                logger.error(f"No conversations found in {args.input_file}")
                return
            
            output_file = Path(args.output_dir) / "dataset.csv"
            exporter = CSVExporter(str(output_file))
            
            if args.format == 'huggingface':
                num_rows = exporter.export_huggingface_format(conversations)
            elif args.format == 'turn_level':
                num_rows = exporter.export_turn_level(conversations)
            elif args.format == 'conversation_level':
                num_rows = exporter.export_conversation_level(conversations)
            
            logger.info("\n=== Export Complete ===")
            logger.info(f"Output file: {output_file}")
            logger.info(f"Rows written: {num_rows}")
        
        logger.info("\nDataset is ready for Hugging Face upload!")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)


if __name__ == '__main__':
    main()
