"""
Generate CLI Command

Generates EOU samples in CSV format.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from ..core.prompt_builder import EOUAwarePromptBuilder
from ..core.generator import ConversationGenerator, GenerationError
from ..core.csv_writer import CSVWriter


def setup_logging(log_file: str = 'generation.log'):
    """Setup logging configuration."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def main():
    """Main entry point for generate command."""
    parser = argparse.ArgumentParser(
        description='Generate Arabic EOU samples in CSV format'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of samples to generate (default: 100)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='data/eou_samples.csv',
        help='Output CSV file path (default: data/eou_samples.csv)'
    )
    parser.add_argument(
        '--samples-per-call',
        type=int,
        default=50,
        help='Number of samples to request per API call (default: 50)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default='logs/generation.log',
        help='Log file path (default: logs/generation.log)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen3-235B-A22B-Instruct-2507',
        help='Model name to use (default: Qwen/Qwen3-235B-A22B-Instruct-2507)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature (default: 0.7)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file)
    
    # Get API key
    api_key = os.getenv('NEBIUS_API_KEY')
    if not api_key:
        logging.error("NEBIUS_API_KEY environment variable not set")
        sys.exit(1)
    
    try:
        # Initialize generator
        generator = ConversationGenerator(
            api_key=api_key,
            model_name=args.model,
            temperature=args.temperature
        )
        
        # Generate samples
        logging.info(f"Generating {args.num_samples} samples...")
        samples = generator.generate_batch(
            num_samples=args.num_samples,
            samples_per_call=args.samples_per_call
        )
        
        # Write to CSV
        CSVWriter.write_samples(samples, args.output_file, mode='w')
        
        # Print statistics
        CSVWriter.print_statistics(samples)
        
        logging.info(f"Successfully generated {len(samples)} samples")
        logging.info(f"Output saved to: {args.output_file}")
        
    except GenerationError as e:
        logging.error(f"Generation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
