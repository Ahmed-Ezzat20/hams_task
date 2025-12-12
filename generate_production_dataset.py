#!/usr/bin/env python3
"""
Automated Production Dataset Generation Script
Generates 5,000 Arabic EOU samples in batches with progress tracking.
"""

import os
import sys
import time
import csv
import argparse
from pathlib import Path
from datetime import datetime
import subprocess

def log(message):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def generate_batch(batch_num, samples_per_batch, output_file, log_file):
    """Generate a single batch of samples."""
    log(f"Starting batch {batch_num}...")
    
    cmd = [
        "python", "-m", "hams.cli.generate",
        "--num-samples", str(samples_per_batch),
        "--output-file", output_file,
        "--log-file", log_file
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
        if result.returncode == 0:
            log(f"✅ Batch {batch_num} completed successfully")
            return True
        else:
            log(f"❌ Batch {batch_num} failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        log(f"❌ Batch {batch_num} timed out after 2 hours")
        return False
    except Exception as e:
        log(f"❌ Batch {batch_num} error: {e}")
        return False

def merge_batches(batch_files, output_file):
    """Merge all batch CSV files into a single file."""
    log(f"Merging {len(batch_files)} batches into {output_file}...")
    
    all_rows = []
    header = None
    
    for batch_file in batch_files:
        if not os.path.exists(batch_file):
            log(f"⚠️  Batch file not found: {batch_file}")
            continue
            
        with open(batch_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if header is None:
                header = reader.fieldnames
            for row in reader:
                all_rows.append(row)
    
    # Write merged file
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_rows)
    
    log(f"✅ Merged {len(all_rows)} samples into {output_file}")
    return len(all_rows)

def print_statistics(csv_file):
    """Print dataset statistics."""
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    total = len(data)
    eou_count = sum(1 for row in data if row['label'] == '1')
    non_eou_count = total - eou_count
    
    styles = {}
    for row in data:
        style = row['style']
        styles[style] = styles.get(style, 0) + 1
    
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total samples: {total}")
    print("\nLabel Distribution:")
    print(f"  EOU (label=1):     {eou_count:5d} ({eou_count/total*100:.1f}%)")
    print(f"  Non-EOU (label=0): {non_eou_count:5d} ({non_eou_count/total*100:.1f}%)")
    print("\nStyle Distribution:")
    for style, count in sorted(styles.items()):
        print(f"  {style:12s}: {count:5d} ({count/total*100:.1f}%)")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Generate production-scale Arabic EOU dataset")
    parser.add_argument("--total-samples", type=int, default=5000, help="Total number of samples to generate")
    parser.add_argument("--samples-per-batch", type=int, default=500, help="Number of samples per batch")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for batches")
    parser.add_argument("--output-file", type=str, default=None, help="Final merged output file")
    args = parser.parse_args()
    
    # Calculate number of batches
    num_batches = (args.total_samples + args.samples_per_batch - 1) // args.samples_per_batch
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Set output file
    if args.output_file is None:
        args.output_file = str(output_dir / f"arabic_eou_dataset_{args.total_samples}.csv")
    
    log("="*60)
    log("PRODUCTION DATASET GENERATION")
    log("="*60)
    log(f"Total samples: {args.total_samples}")
    log(f"Samples per batch: {args.samples_per_batch}")
    log(f"Number of batches: {num_batches}")
    log(f"Output directory: {output_dir}")
    log(f"Final output file: {args.output_file}")
    log("="*60)
    
    start_time = time.time()
    batch_files = []
    successful_batches = 0
    
    # Generate batches
    for i in range(1, num_batches + 1):
        batch_file = output_dir / f"batch_{i}.csv"
        log_file = logs_dir / f"batch_{i}.log"
        
        batch_files.append(str(batch_file))
        
        if generate_batch(i, args.samples_per_batch, str(batch_file), str(log_file)):
            successful_batches += 1
        else:
            log(f"⚠️  Batch {i} failed, continuing with next batch...")
        
        # Progress update
        log(f"Progress: {i}/{num_batches} batches ({i/num_batches*100:.1f}%)")
        log(f"Successful batches: {successful_batches}/{i}")
    
    # Merge all batches
    total_samples = merge_batches(batch_files, args.output_file)
    
    # Print statistics
    print_statistics(args.output_file)
    
    # Final summary
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    
    log("="*60)
    log("GENERATION COMPLETE")
    log("="*60)
    log(f"Total samples generated: {total_samples}")
    log(f"Successful batches: {successful_batches}/{num_batches}")
    log(f"Time elapsed: {hours}h {minutes}m")
    log(f"Final dataset: {args.output_file}")
    log("="*60)
    
    if successful_batches < num_batches:
        log(f"⚠️  Warning: {num_batches - successful_batches} batches failed")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
