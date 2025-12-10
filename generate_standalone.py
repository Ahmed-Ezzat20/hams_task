#!/usr/bin/env python3
"""
Standalone Production Dataset Generation Script
Generates Arabic EOU samples in batches WITHOUT using subprocesses.
This avoids Python environment issues on Windows.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

# Import directly from hams package
from hams.core.generator import ConversationGenerator, GenerationError
from hams.core.csv_writer import CSVWriter


def log(message):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def setup_logging(log_file: str):
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


def generate_batch(batch_num, samples_per_batch, output_file, generator):
    """Generate a single batch of samples using direct function calls."""
    log(f"Starting batch {batch_num}...")
    
    try:
        # Generate samples directly
        samples = generator.generate_batch(
            num_samples=samples_per_batch,
            samples_per_call=50
        )
        
        # Write to CSV
        CSVWriter.write_samples(samples, output_file, mode='w')
        
        # Print statistics
        log(f"✅ Batch {batch_num} completed successfully")
        log(f"\nBatch {batch_num} Statistics:")
        CSVWriter.print_statistics(samples)
        
        return True, len(samples)
        
    except GenerationError as e:
        log(f"❌ Batch {batch_num} failed: {e}")
        return False, 0
    except Exception as e:
        log(f"❌ Batch {batch_num} error: {e}")
        return False, 0


def merge_batches(batch_files, output_file):
    """Merge all batch CSV files into a single file."""
    log(f"Merging {len(batch_files)} batches into {output_file}...")
    
    # Read all batch files
    dfs = []
    for batch_file in batch_files:
        if not batch_file.exists():
            log(f"⚠️  Batch file not found: {batch_file}")
            continue
        df = pd.read_csv(batch_file)
        dfs.append(df)
        log(f"  Loaded {len(df)} samples from {batch_file.name}")
    
    if not dfs:
        log("❌ No batch files found to merge!")
        return 0
    
    # Concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Save merged file
    merged_df.to_csv(output_file, index=False)
    
    log(f"✅ Merged {len(merged_df)} total samples")
    log(f"✅ Saved to: {output_file}")
    
    return len(merged_df)


def print_final_statistics(output_file):
    """Print final dataset statistics."""
    df = pd.read_csv(output_file)
    
    log("\n" + "="*60)
    log("FINAL DATASET STATISTICS")
    log("="*60)
    log(f"Total samples: {len(df)}")
    
    # Label distribution
    label_counts = df['label'].value_counts()
    eou_count = label_counts.get(1, 0)
    non_eou_count = label_counts.get(0, 0)
    log(f"\nLabel Distribution:")
    log(f"  EOU (label=1): {eou_count} ({eou_count/len(df)*100:.1f}%)")
    log(f"  Non-EOU (label=0): {non_eou_count} ({non_eou_count/len(df)*100:.1f}%)")
    
    # Style distribution
    style_counts = df['style'].value_counts()
    log(f"\nStyle Distribution:")
    for style, count in style_counts.items():
        log(f"  {style}: {count} ({count/len(df)*100:.1f}%)")
    
    # Duplicates
    duplicates = df.duplicated(subset=['utterance']).sum()
    log(f"\nDuplicates: {duplicates} ({duplicates/len(df)*100:.1f}%)")
    
    # Unique last words
    last_words = df['utterance'].str.split().str[-1].nunique()
    log(f"Unique last words: {last_words}")
    
    log("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate production-scale Arabic EOU dataset (standalone version)"
    )
    parser.add_argument("--total-samples", type=int, default=5000, help="Total number of samples to generate")
    parser.add_argument("--samples-per-batch", type=int, default=500, help="Number of samples per batch")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for batches")
    parser.add_argument("--output-file", type=str, default=None, help="Final merged output file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-235B-A22B-Instruct-2507", help="Model name")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    args = parser.parse_args()
    
    # Calculate number of batches
    num_batches = (args.total_samples + args.samples_per_batch - 1) // args.samples_per_batch
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set output file
    if args.output_file is None:
        args.output_file = output_dir / f"arabic_eou_dataset_{args.total_samples}.csv"
    else:
        args.output_file = Path(args.output_file)
    
    # Setup logging
    log_file = output_dir / "generation.log"
    setup_logging(log_file)
    
    # Get API key
    api_key = os.getenv('NEBIUS_API_KEY')
    if not api_key:
        log("❌ NEBIUS_API_KEY environment variable not set")
        sys.exit(1)
    
    # Print configuration
    log("="*60)
    log("PRODUCTION DATASET GENERATION (STANDALONE)")
    log("="*60)
    log(f"Total samples: {args.total_samples}")
    log(f"Samples per batch: {args.samples_per_batch}")
    log(f"Number of batches: {num_batches}")
    log(f"Output directory: {output_dir}")
    log(f"Final output file: {args.output_file}")
    log(f"Model: {args.model}")
    log(f"Temperature: {args.temperature}")
    log("="*60)
    
    # Initialize generator once (reuse for all batches)
    try:
        generator = ConversationGenerator(
            api_key=api_key,
            model_name=args.model,
            temperature=args.temperature
        )
        log("✅ Generator initialized successfully")
    except Exception as e:
        log(f"❌ Failed to initialize generator: {e}")
        sys.exit(1)
    
    # Generate batches
    batch_files = []
    successful_batches = 0
    total_samples_generated = 0
    
    for i in range(1, num_batches + 1):
        batch_file = output_dir / f"batch_{i:02d}.csv"
        batch_files.append(batch_file)
        
        success, samples_count = generate_batch(i, args.samples_per_batch, batch_file, generator)
        
        if success:
            successful_batches += 1
            total_samples_generated += samples_count
        
        log(f"\nProgress: {i}/{num_batches} batches ({i/num_batches*100:.1f}%)")
        log(f"Successful batches: {successful_batches}/{i}")
        log(f"Total samples generated: {total_samples_generated}")
        log("")
    
    # Merge all batches
    log("\n" + "="*60)
    total_samples = merge_batches(batch_files, args.output_file)
    
    if total_samples > 0:
        # Print final statistics
        print_final_statistics(args.output_file)
        
        log("\n✅ GENERATION COMPLETE!")
        log(f"✅ Final dataset: {args.output_file}")
        log(f"✅ Total samples: {total_samples}")
    else:
        log("\n❌ GENERATION FAILED - No samples generated")
        sys.exit(1)


if __name__ == "__main__":
    main()
