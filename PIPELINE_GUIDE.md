# Complete Pipeline Guide: Arabic EOU Dataset Generation

This guide provides step-by-step instructions for generating, processing, and finalizing an Arabic EOU detection dataset ready for Hugging Face upload.

---

## 📋 Prerequisites

### 1. **Clone the Repository**
```bash
git clone https://github.com/Ahmed-Ezzat20/hams_task.git
cd hams_task
```

### 2. **Install Dependencies**
```bash
# Using pip
pip install -e .

# Or using uv
uv sync
```

### 3. **Set API Key**
```bash
export NEBIUS_API_KEY="your-nebius-api-key-here"
```

---

## 🚀 Pipeline Steps

### **Step 1: Generate Conversations**

Generate synthetic Arabic conversations using the LLM.

#### Option A: Quick Build (Recommended)
```bash
hams-build \
    --num-conversations 1000 \
    --output-dir data \
    --style both
```

**Output:**
- `data/conversations_clean.jsonl` (1000 clean conversations)
- `data/conversations_asr.jsonl` (1000 ASR-like conversations)

**Time:** ~2-3 hours  
**Cost:** ~$7-10

---

#### Option B: Manual Generation

**Generate clean conversations:**
```bash
hams-generate \
    --num-conversations 1000 \
    --output-file data/conversations_clean.jsonl \
    --style clean \
    --temperature 0.7
```

**Generate ASR-like conversations:**
```bash
hams-generate \
    --num-conversations 1000 \
    --output-file data/conversations_asr.jsonl \
    --style asr_like \
    --temperature 0.7
```

---

### **Step 2: Verify Generated Data**

Check the generated conversations:

```bash
# Count conversations
wc -l data/conversations_clean.jsonl
wc -l data/conversations_asr.jsonl

# View first conversation
head -1 data/conversations_clean.jsonl | jq .

# Check statistics
python -c "
from hams.core.writer import DatasetWriter
writer = DatasetWriter('data/conversations_clean.jsonl')
stats = writer.get_statistics()
print(f'Total conversations: {stats[\"total_conversations\"]}')
print(f'Total turns: {stats[\"total_turns\"]}')
print(f'Avg turns/conv: {stats[\"avg_turns_per_conversation\"]:.1f}')
print(f'Domains: {stats[\"domains\"]}')
"
```

**Expected Output:**
```
Total conversations: 1000
Total turns: 9000-11000
Avg turns/conv: 9-11
Domains: {'restaurant': 200, 'banking': 150, ...}
```

---

### **Step 3: Finalize Dataset for Hugging Face**

Convert JSONL to CSV format with train/val/test splits.

#### Create Splits (Recommended)
```bash
hams-finalize \
    --input-file data/conversations_clean.jsonl \
    --output-dir datasets/arabic_eou_clean \
    --create-splits \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --format huggingface
```

**Output:**
- `datasets/arabic_eou_clean/train.csv` (700 conversations, ~6,300-7,700 turns)
- `datasets/arabic_eou_clean/validation.csv` (150 conversations, ~1,350-1,650 turns)
- `datasets/arabic_eou_clean/test.csv` (150 conversations, ~1,350-1,650 turns)

---

#### Also Finalize ASR Dataset
```bash
hams-finalize \
    --input-file data/conversations_asr.jsonl \
    --output-dir datasets/arabic_eou_asr \
    --create-splits \
    --format huggingface
```

---

### **Step 4: Inspect CSV Files**

Check the CSV format:

```bash
# View header
head -1 datasets/arabic_eou_clean/train.csv

# View first few rows
head -5 datasets/arabic_eou_clean/train.csv | column -t -s,

# Count rows
wc -l datasets/arabic_eou_clean/*.csv
```

**Expected CSV Format:**
```csv
text,label,context,domain,conversation_id,split
السلام عليكم,1,,restaurant,conv_restaurant_001,train
وعليكم السلام,1,السلام عليكم,restaurant,conv_restaurant_001,train
...
```

---

### **Step 5: Upload to Hugging Face**

#### Manual Upload via Web Interface

1. Go to [huggingface.co/new-dataset](https://huggingface.co/new-dataset)
2. Create a new dataset (e.g., `arabic-eou-detection`)
3. Upload the CSV files:
   - `train.csv`
   - `validation.csv`
   - `test.csv`
4. Add dataset card with description
5. Make it public

---

#### Programmatic Upload (Alternative)

```bash
# Install huggingface_hub
pip install huggingface-hub

# Login
huggingface-cli login

# Upload dataset
python -c "
from huggingface_hub import HfApi
api = HfApi()

# Create dataset repository
api.create_repo(
    repo_id='your-username/arabic-eou-detection',
    repo_type='dataset',
    private=False
)

# Upload files
api.upload_folder(
    folder_path='datasets/arabic_eou_clean',
    repo_id='your-username/arabic-eou-detection',
    repo_type='dataset'
)
"
```

---

## 📊 Complete Pipeline Example

Here's a complete end-to-end example:

```bash
#!/bin/bash

# Set API key
export NEBIUS_API_KEY="your-api-key"

# Step 1: Generate 1000 conversations (both clean and ASR)
echo "Step 1: Generating conversations..."
hams-build \
    --num-conversations 1000 \
    --output-dir data \
    --style both

# Step 2: Verify data
echo "Step 2: Verifying data..."
wc -l data/*.jsonl

# Step 3: Finalize clean dataset
echo "Step 3: Finalizing clean dataset..."
hams-finalize \
    --input-file data/conversations_clean.jsonl \
    --output-dir datasets/arabic_eou_clean \
    --create-splits \
    --format huggingface

# Step 4: Finalize ASR dataset
echo "Step 4: Finalizing ASR dataset..."
hams-finalize \
    --input-file data/conversations_asr.jsonl \
    --output-dir datasets/arabic_eou_asr \
    --create-splits \
    --format huggingface

# Step 5: Show statistics
echo "Step 5: Dataset statistics..."
wc -l datasets/arabic_eou_clean/*.csv
wc -l datasets/arabic_eou_asr/*.csv

echo "Done! Ready to upload to Hugging Face."
```

---

## 🎯 Quick Reference

### Generate Data
```bash
hams-build --num-conversations 1000 --output-dir data --style both
```

### Finalize for HuggingFace
```bash
hams-finalize \
    --input-file data/conversations_clean.jsonl \
    --output-dir datasets/arabic_eou \
    --create-splits
```

### Upload to HuggingFace
```bash
huggingface-cli login
huggingface-cli upload your-username/arabic-eou-detection datasets/arabic_eou
```

---

## 📁 Expected Directory Structure

After running the complete pipeline:

```
hams_task/
├── data/
│   ├── conversations_clean.jsonl    # 1000 clean conversations
│   └── conversations_asr.jsonl      # 1000 ASR-like conversations
├── datasets/
│   ├── arabic_eou_clean/
│   │   ├── train.csv                # 70% of data
│   │   ├── validation.csv           # 15% of data
│   │   └── test.csv                 # 15% of data
│   └── arabic_eou_asr/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
└── hams/                            # Package code
```

---

## ⚙️ Advanced Options

### Custom Split Ratios
```bash
hams-finalize \
    --input-file data/conversations.jsonl \
    --output-dir datasets/custom \
    --create-splits \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```

### Different Export Formats

**Turn-level format** (one row per turn):
```bash
hams-finalize \
    --input-file data/conversations.jsonl \
    --output-dir datasets/turn_level \
    --format turn_level
```

**Conversation-level format** (one row per conversation):
```bash
hams-finalize \
    --input-file data/conversations.jsonl \
    --output-dir datasets/conv_level \
    --format conversation_level
```

---

## 🔍 Troubleshooting

### Issue: API quota exceeded
**Solution:** Wait for quota reset or upgrade to paid tier

### Issue: Out of memory during generation
**Solution:** Generate in smaller batches (e.g., 100-200 at a time)

### Issue: CSV encoding errors
**Solution:** Files are UTF-8 encoded by default, ensure your viewer supports UTF-8

### Issue: Missing conversations in splits
**Solution:** Check input file has enough conversations for the split ratios

---

## 📈 Expected Results

### Dataset Size
- **1000 conversations** → ~9,000-11,000 turns
- **Train split (70%)** → ~6,300-7,700 turns
- **Val split (15%)** → ~1,350-1,650 turns
- **Test split (15%)** → ~1,350-1,650 turns

### File Sizes
- **JSONL files:** ~50-60 MB per 1000 conversations
- **CSV files:** ~30-40 MB per 1000 conversations

### Generation Time
- **1000 conversations:** ~2-3 hours
- **100 conversations:** ~15-20 minutes

### Cost (Nebius Token Factory)
- **1000 conversations:** ~$7-10
- **100 conversations:** ~$0.70-1.00

---

## ✅ Validation Checklist

Before uploading to Hugging Face:

- [ ] Generated at least 1000 conversations
- [ ] Verified JSONL files are valid
- [ ] Created train/val/test splits
- [ ] Inspected CSV files for correct format
- [ ] Checked CSV encoding (UTF-8)
- [ ] Verified label distribution (is_eou)
- [ ] Reviewed sample conversations for quality
- [ ] Prepared dataset card/README

---

## 🎓 Next Steps

After uploading to Hugging Face:

1. **Load dataset in Python:**
```python
from datasets import load_dataset
dataset = load_dataset("your-username/arabic-eou-detection")
```

2. **Proceed to model fine-tuning** (Day 2-3)
3. **Evaluate model performance** (Day 3)
4. **Integrate with LiveKit** (Day 3-4)
5. **Create demo video** (Day 4)

---

## 📞 Support

If you encounter issues:
1. Check the logs in the console output
2. Verify API key is set correctly
3. Ensure sufficient disk space
4. Review the troubleshooting section above

---

**Happy dataset building! 🚀**
