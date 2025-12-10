#!/usr/bin/env python3
"""
Dataset Splitting and Validation CLI
Creates train/validation/test splits with quality validation.
"""

import argparse
import csv
import random
from pathlib import Path
from collections import Counter

def split_dataset(input_file, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split dataset into train/val/test with stratification by label."""
    
    # Read data
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    # Separate by label for stratification
    eou_samples = [row for row in data if row['label'] == '1']
    non_eou_samples = [row for row in data if row['label'] == '0']
    
    # Shuffle
    random.seed(seed)
    random.shuffle(eou_samples)
    random.shuffle(non_eou_samples)
    
    # Calculate split sizes
    def calculate_splits(samples, train_r, val_r, test_r):
        total = len(samples)
        train_size = int(total * train_r)
        val_size = int(total * val_r)
        test_size = total - train_size - val_size
        
        train = samples[:train_size]
        val = samples[train_size:train_size + val_size]
        test = samples[train_size + val_size:]
        
        return train, val, test
    
    eou_train, eou_val, eou_test = calculate_splits(eou_samples, train_ratio, val_ratio, test_ratio)
    non_eou_train, non_eou_val, non_eou_test = calculate_splits(non_eou_samples, train_ratio, val_ratio, test_ratio)
    
    # Combine and shuffle
    train_data = eou_train + non_eou_train
    val_data = eou_val + non_eou_val
    test_data = eou_test + non_eou_test
    
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write splits
    header = ['utterance', 'style', 'label']
    
    splits = {
        'train': train_data,
        'validation': val_data,
        'test': test_data
    }
    
    for split_name, split_data in splits.items():
        output_file = output_path / f"{split_name}.csv"
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(split_data)
        print(f"✅ Created {split_name}.csv ({len(split_data)} samples)")
    
    return train_data, val_data, test_data

def validate_split(split_name, data):
    """Validate a data split and return statistics."""
    total = len(data)
    eou_count = sum(1 for row in data if row['label'] == '1')
    non_eou_count = total - eou_count
    
    styles = Counter(row['style'] for row in data)
    
    print(f"\n{'='*60}")
    print(f"{split_name.upper()} SPLIT STATISTICS")
    print("="*60)
    print(f"Total samples: {total}")
    print(f"\nLabel Distribution:")
    print(f"  EOU (label=1):     {eou_count:5d} ({eou_count/total*100:.1f}%)")
    print(f"  Non-EOU (label=0): {non_eou_count:5d} ({non_eou_count/total*100:.1f}%)")
    print(f"\nStyle Distribution:")
    for style, count in sorted(styles.items()):
        print(f"  {style:12s}: {count:5d} ({count/total*100:.1f}%)")
    print("="*60)
    
    return {
        'total': total,
        'eou': eou_count,
        'non_eou': non_eou_count,
        'eou_ratio': eou_count / total,
        'styles': dict(styles)
    }

def write_report(output_dir, train_stats, val_stats, test_stats):
    """Write validation report to file."""
    report_file = Path(output_dir) / "report.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("DATASET SPLIT VALIDATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Train Split: {train_stats['total']} samples\n")
        f.write(f"  EOU: {train_stats['eou']} ({train_stats['eou_ratio']*100:.1f}%)\n")
        f.write(f"  Non-EOU: {train_stats['non_eou']} ({(1-train_stats['eou_ratio'])*100:.1f}%)\n\n")
        
        f.write(f"Validation Split: {val_stats['total']} samples\n")
        f.write(f"  EOU: {val_stats['eou']} ({val_stats['eou_ratio']*100:.1f}%)\n")
        f.write(f"  Non-EOU: {val_stats['non_eou']} ({(1-val_stats['eou_ratio'])*100:.1f}%)\n\n")
        
        f.write(f"Test Split: {test_stats['total']} samples\n")
        f.write(f"  EOU: {test_stats['eou']} ({test_stats['eou_ratio']*100:.1f}%)\n")
        f.write(f"  Non-EOU: {test_stats['non_eou']} ({(1-test_stats['eou_ratio'])*100:.1f}%)\n\n")
        
        f.write("="*60 + "\n")
        f.write("VALIDATION CHECKS\n")
        f.write("="*60 + "\n\n")
        
        # Check balance
        train_eou_ratio = train_stats['eou_ratio']
        val_eou_ratio = val_stats['eou_ratio']
        test_eou_ratio = test_stats['eou_ratio']
        
        if abs(train_eou_ratio - val_eou_ratio) < 0.05 and abs(train_eou_ratio - test_eou_ratio) < 0.05:
            f.write("✅ Label distribution is balanced across splits\n")
        else:
            f.write("⚠️  Label distribution varies across splits\n")
        
        if train_stats['total'] > val_stats['total'] and train_stats['total'] > test_stats['total']:
            f.write("✅ Train split is largest\n")
        else:
            f.write("⚠️  Train split should be largest\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"\n✅ Validation report saved to {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--input-file", required=True, help="Input CSV file")
    parser.add_argument("--output-dir", required=True, help="Output directory for splits")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio (default: 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.01:
        print("❌ Error: Ratios must sum to 1.0")
        return 1
    
    print("="*60)
    print("DATASET SPLITTING")
    print("="*60)
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Validation ratio: {args.val_ratio}")
    print(f"Test ratio: {args.test_ratio}")
    print(f"Random seed: {args.seed}")
    print("="*60)
    
    # Split dataset
    train_data, val_data, test_data = split_dataset(
        args.input_file,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
    
    # Validate splits
    train_stats = validate_split("train", train_data)
    val_stats = validate_split("validation", val_data)
    test_stats = validate_split("test", test_data)
    
    # Write report
    write_report(args.output_dir, train_stats, val_stats, test_stats)
    
    print("\n✅ Dataset splitting complete!")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
