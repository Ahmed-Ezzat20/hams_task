"""
CLI command for building complete datasets (wrapper for generate + augment).
"""

import os
import argparse
import logging
import subprocess
import sys
from pathlib import Path


logger = logging.getLogger(__name__)


def main():
    """Main entry point for hams-build command."""
    parser = argparse.ArgumentParser(
        description='Build complete Arabic EOU dataset (generate + augment)'
    )
    parser.add_argument(
        '--num-conversations',
        type=int,
        required=True,
        help='Number of conversations to generate'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory (default: data)'
    )
    parser.add_argument(
        '--style',
        type=str,
        choices=['clean', 'asr_like', 'both'],
        default='both',
        help='Output style: clean, asr_like, or both (default: both)'
    )
    parser.add_argument(
        '--domains',
        type=str,
        nargs='+',
        help='Specific domains to generate (default: all)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature (default: 0.7)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='Nebius API key (default: NEBIUS_API_KEY env var)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen3-235B-A22B-Instruct-2507',
        help='Model name'
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
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get API key
    api_key = args.api_key or os.getenv('NEBIUS_API_KEY')
    if not api_key:
        logger.error("API key not provided. Set NEBIUS_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    try:
        # Step 1: Generate clean conversations
        if args.style in ['clean', 'both']:
            logger.info("=== Step 1: Generating clean conversations ===")
            clean_file = output_dir / 'conversations_clean.jsonl'
            
            cmd = [
                sys.executable, '-m', 'hams.cli.generate',
                '--num-conversations', str(args.num_conversations),
                '--output-file', str(clean_file),
                '--style', 'clean',
                '--temperature', str(args.temperature),
                '--api-key', api_key,
                '--model', args.model,
                '--log-level', args.log_level
            ]
            
            if args.domains:
                cmd.extend(['--domains'] + args.domains)
            
            result = subprocess.run(cmd)
            if result.returncode != 0:
                logger.error("Failed to generate clean conversations")
                sys.exit(1)
        
        # Step 2: Generate ASR-like conversations OR augment clean ones
        if args.style in ['asr_like', 'both']:
            logger.info("=== Step 2: Creating ASR-like conversations ===")
            asr_file = output_dir / 'conversations_asr.jsonl'
            
            if args.style == 'asr_like':
                # Generate directly with ASR style
                cmd = [
                    sys.executable, '-m', 'hams.cli.generate',
                    '--num-conversations', str(args.num_conversations),
                    '--output-file', str(asr_file),
                    '--style', 'asr_like',
                    '--temperature', str(args.temperature),
                    '--api-key', api_key,
                    '--model', args.model,
                    '--log-level', args.log_level
                ]
                
                if args.domains:
                    cmd.extend(['--domains'] + args.domains)
                
                result = subprocess.run(cmd)
            else:
                # Augment clean conversations
                cmd = [
                    sys.executable, '-m', 'hams.cli.asr_augment',
                    '--input-file', str(clean_file),
                    '--output-file', str(asr_file),
                    '--log-level', args.log_level
                ]
                
                result = subprocess.run(cmd)
            
            if result.returncode != 0:
                logger.error("Failed to create ASR-like conversations")
                sys.exit(1)
        
        # Summary
        logger.info("\n=== Dataset Build Complete ===")
        logger.info(f"Output directory: {output_dir}")
        
        if args.style in ['clean', 'both']:
            logger.info(f"Clean conversations: {output_dir / 'conversations_clean.jsonl'}")
        if args.style in ['asr_like', 'both']:
            logger.info(f"ASR-like conversations: {output_dir / 'conversations_asr.jsonl'}")
        
        logger.info("\nNext steps:")
        logger.info("1. Review the generated data")
        logger.info("2. Create train/val/test splits")
        logger.info("3. Upload to Hugging Face")
        logger.info("4. Proceed to model fine-tuning")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
