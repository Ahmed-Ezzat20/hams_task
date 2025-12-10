#!/usr/bin/env python3
"""
Combine multiple batch files of synthetic conversations into a single dataset.

This script merges multiple JSON files containing synthetic conversations
into a single combined dataset, useful for managing large-scale data generation.

Usage:
    python combine_batches.py --input-files batch_1.json batch_2.json batch_3.json --output combined.json
    python combine_batches.py  # Uses default batch_*.json pattern
"""

import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import glob


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def combine_batches(
    input_files: List[str], output_file: str, remove_duplicates: bool = True
) -> Dict[str, Any]:
    """
    Combine multiple batch files into a single dataset.

    Args:
        input_files: List of input JSON files to combine.
        output_file: Path to output combined file.
        remove_duplicates: Whether to remove duplicate conversations.

    Returns:
        Statistics dictionary with combination results.
    """
    all_conversations = []
    stats = {
        "input_files": len(input_files),
        "total_conversations_before": 0,
        "total_conversations_after": 0,
        "duplicates_removed": 0,
        "total_turns": 0,
        "errors": [],
    }

    logger.info(f"Starting combination of {len(input_files)} files...")

    # Load all conversations
    for input_file in input_files:
        try:
            logger.info(f"Loading {input_file}...")

            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle both direct conversation lists and wrapped data
            if "conversations" in data:
                conversations = data["conversations"]
            else:
                conversations = data if isinstance(data, list) else [data]

            logger.info(
                f"  Loaded {len(conversations)} conversations from {input_file}"
            )
            all_conversations.extend(conversations)
            stats["total_conversations_before"] += len(conversations)

        except FileNotFoundError:
            error_msg = f"File not found: {input_file}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in {input_file}: {str(e)}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
        except Exception as e:
            error_msg = f"Error loading {input_file}: {str(e)}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)

    logger.info(f"Total conversations loaded: {len(all_conversations)}")

    # Remove duplicates if requested
    if remove_duplicates:
        logger.info("Removing duplicate conversations...")
        unique_conversations = []
        seen_ids = set()

        for conv in all_conversations:
            conv_id = conv.get("conversation_id")
            if conv_id and conv_id not in seen_ids:
                unique_conversations.append(conv)
                seen_ids.add(conv_id)
            elif not conv_id:
                # If no ID, add it anyway
                unique_conversations.append(conv)

        stats["duplicates_removed"] = len(all_conversations) - len(unique_conversations)
        all_conversations = unique_conversations
        logger.info(f"Removed {stats['duplicates_removed']} duplicates")

    stats["total_conversations_after"] = len(all_conversations)

    # Count total turns
    for conv in all_conversations:
        if "turns" in conv:
            stats["total_turns"] += len(conv["turns"])

    # Create combined dataset
    combined_data = {
        "metadata": {
            "combination_date": datetime.now().isoformat(),
            "source_files": input_files,
            "total_conversations": len(all_conversations),
            "total_turns": stats["total_turns"],
            "combination_stats": {
                "input_files": stats["input_files"],
                "conversations_before_dedup": stats["total_conversations_before"],
                "conversations_after_dedup": stats["total_conversations_after"],
                "duplicates_removed": stats["duplicates_removed"],
            },
        },
        "conversations": all_conversations,
    }

    # Save combined file
    try:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)

        logger.info(f"✓ Successfully saved combined dataset to {output_file}")
        logger.info(f"  Total conversations: {len(all_conversations)}")
        logger.info(f"  Total turns: {stats['total_turns']}")

    except Exception as e:
        error_msg = f"Error saving combined file: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)

    return stats


def get_batch_files(pattern: str = "batch_*.json") -> List[str]:
    """
    Find batch files matching a pattern.

    Args:
        pattern: Glob pattern to match files.

    Returns:
        List of matching file paths.
    """
    files = sorted(glob.glob(pattern))
    logger.info(f"Found {len(files)} batch files matching pattern '{pattern}'")
    return files


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Combine multiple batch files of synthetic conversations"
    )
    parser.add_argument("--input-files", nargs="+", help="Input JSON files to combine")
    parser.add_argument(
        "--pattern",
        type=str,
        default="batch_*.json",
        help="Glob pattern to find batch files (default: batch_*.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="combined_synthetic_data.json",
        help="Output file path (default: combined_synthetic_data.json)",
    )
    parser.add_argument(
        "--no-dedup", action="store_true", help="Do not remove duplicate conversations"
    )

    args = parser.parse_args()

    # Determine input files
    if args.input_files:
        input_files = args.input_files
    else:
        input_files = get_batch_files(args.pattern)
        if not input_files:
            logger.error(f"No files found matching pattern: {args.pattern}")
            logger.info("Specify files explicitly with --input-files")
            return 1

    if not input_files:
        logger.error("No input files specified")
        return 1

    logger.info(f"Combining {len(input_files)} files...")
    logger.info(f"Output file: {args.output}")

    # Combine batches
    stats = combine_batches(
        input_files, args.output, remove_duplicates=not args.no_dedup
    )

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("COMBINATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Input files: {stats['input_files']}")
    logger.info(f"Conversations (before dedup): {stats['total_conversations_before']}")
    logger.info(f"Conversations (after dedup): {stats['total_conversations_after']}")
    logger.info(f"Duplicates removed: {stats['duplicates_removed']}")
    logger.info(f"Total turns: {stats['total_turns']}")

    if stats["errors"]:
        logger.warning(f"Errors encountered: {len(stats['errors'])}")
        for error in stats["errors"]:
            logger.warning(f"  - {error}")

    logger.info("=" * 50)

    return 0 if not stats["errors"] else 1


if __name__ == "__main__":
    exit(main())
