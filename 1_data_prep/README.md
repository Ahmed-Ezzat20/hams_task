# Data Preparation

This directory contains scripts for generating Arabic conversational datasets with End-of-Utterance (EOU) labels.

## Files

- **generate_dataset.py** - Main script for dataset generation
- **prompts.yaml** - Prompt templates for different conversation domains
- **utils.py** - Helper functions (if needed)

## Quick Start

### Generate Dataset

```bash
# Generate 10,000 samples
python generate_dataset.py --num-samples 10000 --output dataset.csv

# Generate and split into train/val/test
python generate_dataset.py --num-samples 10000 --split --output-dir ./data
```

### Requirements

```bash
pip install openai pyyaml
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Basic Generation

```bash
python generate_dataset.py \
    --num-samples 10000 \
    --output dataset.csv \
    --model gpt-4.1-mini
```

### Advanced Options

```bash
python generate_dataset.py \
    --num-samples 10000 \
    --split \
    --output-dir ./data \
    --model gpt-4.1-mini \
    --temperature 0.7 \
    --samples-per-call 50
```

### Arguments

- `--num-samples`: Number of samples to generate (default: 10000)
- `--output`: Output CSV file (default: dataset.csv)
- `--model`: LLM model to use (default: gpt-4.1-mini)
- `--temperature`: Sampling temperature (default: 0.7)
- `--samples-per-call`: Samples per API call (default: 50)
- `--split`: Split dataset into train/val/test
- `--output-dir`: Output directory for splits (default: ./data)
- `--api-key`: OpenAI API key (or set OPENAI_API_KEY env var)

## Dataset Format

The generated CSV has three columns:

| Column | Description | Values |
|--------|-------------|--------|
| utterance | Arabic text | Any Arabic text |
| style | Conversation style | informal, formal, asr_like |
| label | EOU label | 0 (incomplete), 1 (complete) |

### Example

```csv
utterance,style,label
مرحباً، كيف حالك؟,informal,1
أنا أريد أن,informal,0
الحمد لله بخير,informal,1
هل يمكنك أن,formal,0
```

## Customization

### Modify Prompts

Edit `prompts.yaml` to customize:
- Conversation domains
- Example utterances
- Generation instructions
- Saudi dialect emphasis

### Adjust Generation

Modify `generate_dataset.py` to:
- Change LLM provider
- Add custom validation
- Implement custom post-processing
- Add more features

## Tips

1. **Start small** - Test with 100 samples first
2. **Monitor costs** - Each 10K samples costs ~$1-5 depending on model
3. **Check quality** - Manually review samples before training
4. **Balance labels** - Aim for 40-60% complete utterances
5. **Saudi dialect** - Prompts emphasize Saudi dialect but include MSA

## Troubleshooting

### API Key Error
```
ValueError: API key required
```
**Solution:** Set `OPENAI_API_KEY` environment variable

### No Samples Generated
```
WARNING - No samples generated
```
**Solution:** Check API key, model name, and internet connection

### Invalid CSV Format
```
WARNING - Skipping invalid row
```
**Solution:** Normal - LLM sometimes generates malformed rows, they're skipped automatically

## Next Steps

After generating the dataset:
1. Review samples for quality
2. Check label distribution (should be ~50/50)
3. Proceed to `2_eou_model/` for training
