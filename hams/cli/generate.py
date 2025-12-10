"""
CLI command for generating conversations.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List

from ..core.prompt_builder import PromptBuilder
from ..core.generator import ConversationGenerator, GenerationError
from ..core.postprocessor import PostProcessor
from ..core.writer import DatasetWriter


logger = logging.getLogger(__name__)


def main():
    """Main entry point for hams-generate command."""
    parser = argparse.ArgumentParser(
        description='Generate Arabic EOU conversations'
    )
    parser.add_argument(
        '--num-conversations',
        type=int,
        required=True,
        help='Number of conversations to generate'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--style',
        type=str,
        choices=['clean', 'asr_like'],
        default='clean',
        help='Output style: clean or asr_like (default: clean)'
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
        '--delay',
        type=float,
        default=0.5,
        help='Delay between API requests in seconds (default: 0.5)'
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
        help='Model name (default: Qwen/Qwen3-235B-A22B-Instruct-2507)'
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
    
    # Get API key
    api_key = args.api_key or os.getenv('NEBIUS_API_KEY')
    if not api_key:
        logger.error("API key not provided. Set NEBIUS_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    try:
        # Initialize components
        logger.info("Initializing components...")
        prompt_builder = PromptBuilder()
        generator = ConversationGenerator(api_key=api_key, model_name=args.model)
        postprocessor = PostProcessor()
        writer = DatasetWriter(args.output_file)
        
        # Build prompts
        logger.info(f"Building prompts (style={args.style}, domains={args.domains})...")
        prompts = prompt_builder.get_all_prompts(
            style=args.style,
            domains=args.domains
        )
        
        if not prompts:
            logger.error("No prompts available")
            sys.exit(1)
        
        # Cycle through prompts to reach target number
        num_cycles = (args.num_conversations + len(prompts) - 1) // len(prompts)
        all_prompts = (prompts * num_cycles)[:args.num_conversations]
        
        logger.info(f"Generating {args.num_conversations} conversations...")
        logger.info(f"Using {len(prompts)} unique prompts, {num_cycles} cycle(s)")
        
        # Generate conversations
        conversations = []
        failed = 0
        
        for idx, prompt in enumerate(all_prompts, 1):
            logger.info(f"Generating conversation {idx}/{len(all_prompts)}...")
            
            try:
                # Generate
                raw_json = generator.generate(
                    prompt=prompt,
                    temperature=args.temperature
                )
                
                # Normalize
                conversation = postprocessor.normalize(raw_json)
                conversations.append(conversation)
                
                logger.info(f"Success: {conversation.conversation_id} ({len(conversation.turns)} turns)")
                
                # Write incrementally
                if idx % 10 == 0:
                    writer.write_conversations(conversations, append=(idx > 10))
                    conversations = []
                
            except (GenerationError, ValueError) as e:
                logger.warning(f"Failed: {str(e)}")
                failed += 1
                continue
            
            # Delay between requests
            if idx < len(all_prompts):
                import time
                time.sleep(args.delay)
        
        # Write remaining conversations
        if conversations:
            writer.write_conversations(conversations, append=True)
        
        # Statistics
        stats = writer.get_statistics()
        logger.info("\n=== Generation Complete ===")
        logger.info(f"Total conversations: {stats['total_conversations']}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total turns: {stats['total_turns']}")
        logger.info(f"Avg turns/conversation: {stats['avg_turns_per_conversation']:.1f}")
        logger.info(f"Output file: {args.output_file}")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
