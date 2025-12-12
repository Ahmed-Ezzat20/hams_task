#!/usr/bin/env python3
"""
Comprehensive Dataset Analysis Script
Analyzes the generated Arabic EOU dataset for quality metrics.
"""

import pandas as pd

def analyze_dataset(csv_file):
    """Perform comprehensive analysis of the dataset."""
    
    print("="*80)
    print("ARABIC EOU DATASET - COMPREHENSIVE ANALYSIS")
    print("="*80)
    print()
    
    # Load dataset
    df = pd.read_csv(csv_file)
    
    # Basic Statistics
    print("📊 BASIC STATISTICS")
    print("-" * 80)
    print(f"Total samples: {len(df):,}")
    print(f"File size: {csv_file}")
    print()
    
    # Label Distribution
    print("🏷️  LABEL DISTRIBUTION")
    print("-" * 80)
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        percentage = count / len(df) * 100
        label_name = "EOU (Complete)" if label == 1 else "Non-EOU (Incomplete)"
        print(f"  Label {label} ({label_name}): {count:,} samples ({percentage:.2f}%)")
    print()
    
    # Style Distribution
    print("🎨 STYLE DISTRIBUTION")
    print("-" * 80)
    style_counts = df['style'].value_counts()
    for style, count in style_counts.items():
        percentage = count / len(df) * 100
        print(f"  {style:12s}: {count:,} samples ({percentage:.2f}%)")
    print()
    
    # Cross-tabulation: Style × Label
    print("📊 STYLE × LABEL DISTRIBUTION")
    print("-" * 80)
    crosstab = pd.crosstab(df['style'], df['label'], margins=True)
    print(crosstab)
    print()
    
    # Duplicates Analysis
    print("🔍 DUPLICATES ANALYSIS")
    print("-" * 80)
    total_duplicates = df.duplicated(subset=['utterance']).sum()
    duplicate_percentage = total_duplicates / len(df) * 100
    print(f"  Total duplicates: {total_duplicates:,} ({duplicate_percentage:.2f}%)")
    print(f"  Unique utterances: {df['utterance'].nunique():,}")
    print()
    
    # Length Statistics
    print("📏 LENGTH STATISTICS")
    print("-" * 80)
    df['word_count'] = df['utterance'].str.split().str.len()
    df['char_count'] = df['utterance'].str.len()
    
    print(f"  Word count - Mean: {df['word_count'].mean():.2f}, Median: {df['word_count'].median():.0f}, "
          f"Min: {df['word_count'].min()}, Max: {df['word_count'].max()}")
    print(f"  Char count - Mean: {df['char_count'].mean():.2f}, Median: {df['char_count'].median():.0f}, "
          f"Min: {df['char_count'].min()}, Max: {df['char_count'].max()}")
    print()
    
    # Length by Label
    print("📏 LENGTH BY LABEL")
    print("-" * 80)
    for label in sorted(df['label'].unique()):
        label_name = "EOU" if label == 1 else "Non-EOU"
        subset = df[df['label'] == label]
        print(f"  {label_name:8s} - Words: {subset['word_count'].mean():.2f} avg, "
              f"Chars: {subset['char_count'].mean():.2f} avg")
    print()
    
    # Last Word Analysis
    print("🔤 LAST WORD ANALYSIS")
    print("-" * 80)
    df['last_word'] = df['utterance'].str.split().str[-1]
    unique_last_words = df['last_word'].nunique()
    print(f"  Unique last words: {unique_last_words:,}")
    
    # Top 20 most common last words
    top_last_words = df['last_word'].value_counts().head(20)
    print("  Top 20 most common last words:")
    for word, count in top_last_words.items():
        percentage = count / len(df) * 100
        print(f"    {word:20s}: {count:4d} ({percentage:.2f}%)")
    print()
    
    # Punctuation Analysis
    print("📝 PUNCTUATION ANALYSIS")
    print("-" * 80)
    punctuation_patterns = {
        'Question marks (؟)': r'؟',
        'Periods (.)': r'\.',
        'Commas (،)': r'،',
        'Exclamation (!)': r'!',
        'Ellipsis (...)': r'\.\.\.|…',
        'Dashes (-)': r'-',
        'Em-dashes (—)': r'—',
    }
    
    for name, pattern in punctuation_patterns.items():
        count = df['utterance'].str.contains(pattern, regex=True).sum()
        percentage = count / len(df) * 100
        print(f"  {name:25s}: {count:5d} ({percentage:.2f}%)")
    print()
    
    # Ellipsis bias check (should be 0%)
    ellipsis_count = df['utterance'].str.contains(r'\.\.\.|…', regex=True).sum()
    if ellipsis_count > 0:
        print(f"  ⚠️  WARNING: Found {ellipsis_count} utterances with ellipsis (should be 0)")
    else:
        print("  ✅ No ellipsis bias detected")
    print()
    
    # Sample Distribution by Label
    print("📋 SAMPLE EXAMPLES")
    print("-" * 80)
    print("  EOU (label=1) examples:")
    for i, row in df[df['label'] == 1].head(5).iterrows():
        print(f"    [{row['style']:10s}] {row['utterance']}")
    print()
    print("  Non-EOU (label=0) examples:")
    for i, row in df[df['label'] == 0].head(5).iterrows():
        print(f"    [{row['style']:10s}] {row['utterance']}")
    print()
    
    # Quality Score
    print("⭐ QUALITY SCORE")
    print("-" * 80)
    
    # Calculate quality metrics
    label_balance_score = 100 - abs(60 - (label_counts.get(1, 0) / len(df) * 100)) * 2
    duplicate_score = 100 - (duplicate_percentage * 10)
    diversity_score = min(100, (unique_last_words / len(df) * 100) * 3)
    ellipsis_score = 100 if ellipsis_count == 0 else max(0, 100 - (ellipsis_count / len(df) * 100) * 10)
    
    overall_score = (label_balance_score + duplicate_score + diversity_score + ellipsis_score) / 4
    
    print(f"  Label balance score:  {label_balance_score:.1f}/100 (target: 60/40 EOU/non-EOU)")
    print(f"  Duplicate score:      {duplicate_score:.1f}/100 (lower is better)")
    print(f"  Diversity score:      {diversity_score:.1f}/100 (unique last words)")
    print(f"  Ellipsis bias score:  {ellipsis_score:.1f}/100 (should be 100)")
    print("  " + "="*60)
    print(f"  OVERALL QUALITY:      {overall_score:.1f}/100")
    print()
    
    if overall_score >= 90:
        print("  ✅ EXCELLENT - Production ready!")
    elif overall_score >= 80:
        print("  ✅ GOOD - Ready for training")
    elif overall_score >= 70:
        print("  ⚠️  ACCEPTABLE - Consider improvements")
    else:
        print("  ❌ NEEDS IMPROVEMENT - Review data quality")
    
    print()
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return df

if __name__ == "__main__":
    import sys
    
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "data/arabic_eou_dataset_10000.csv"
    analyze_dataset(csv_file)
