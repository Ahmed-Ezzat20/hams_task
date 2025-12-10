"""
CLI command for applying ASR-style augmentation to existing datasets.
"""

import argparse
import logging

from ..core.postprocessor import PostProcessor
from ..core.writer import DatasetWriter


logger = logging.getLogger(__name__)


def main():
    """Main entry point for hams-asr-augment command."""
    parser = argparse.ArgumentParser(
        description='Apply ASR-style noise to existing conversations'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help='Input JSONL file with clean conversations'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='Output JSONL file for augmented conversations'
    )
    parser.add_argument(
        '--drop-punct',
        action='store_true',
        default=True,
        help='Remove punctuation (default: True)'
    )
    parser.add_argument(
        '--prob-swap',
        type=float,
        default=0.1,
        help='Probability of word swapping (default: 0.1)'
    )
    parser.add_argument(
        '--prob-filler',
        type=float,
        default=0.2,
        help='Probability of adding filler words (default: 0.2)'
    )
    parser.add_argument(
        '--prob-repeat',
        type=float,
        default=0.1,
        help='Probability of word repetition (default: 0.1)'
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
        # Initialize components
        logger.info("Reading conversations...")
        reader = DatasetWriter(args.input_file)
        conversations = reader.read_conversations()
        
        if not conversations:
            logger.error(f"No conversations found in {args.input_file}")
            return
        
        logger.info(f"Found {len(conversations)} conversations")
        
        # Augment conversations
        logger.info("Applying ASR-style augmentation...")
        postprocessor = PostProcessor()
        augmented = []
        
        noise_params = {
            'drop_punct': args.drop_punct,
            'prob_swap': args.prob_swap,
            'prob_filler': args.prob_filler,
            'prob_repeat': args.prob_repeat
        }
        
        for conv in conversations:
            aug_conv = postprocessor.augment_conversation(
                conv,
                add_noise=True,
                **noise_params
            )
            augmented.append(aug_conv)
        
        # Write augmented conversations
        logger.info(f"Writing augmented conversations to {args.output_file}...")
        writer = DatasetWriter(args.output_file)
        writer.write_conversations(augmented)
        
        # Statistics
        stats = writer.get_statistics()
        logger.info("\n=== Augmentation Complete ===")
        logger.info(f"Total conversations: {stats['total_conversations']}")
        logger.info(f"Total turns: {stats['total_turns']}")
        logger.info(f"Output file: {args.output_file}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)


if __name__ == '__main__':
    main()
