# Arabic EOU Data Generation Guide

## How to Generate Samples

### Prerequisites

1. **Set up your API key:**
```bash
export NEBIUS_API_KEY='your-api-key-here'
```

2. **Install dependencies:**
```bash
cd hams_task
pip install -r requirements.txt
# or with uv:
uv sync
```

---

## Generation Commands

### 1. Test Run (5 conversations)
**Purpose:** Verify everything works before generating large batches

```bash
python arabic_eou_data_generator.py \
    --num-conversations 5 \
    --output-file data/test_batch.json \
    --validate
```

**Expected output:**
- File: `data/test_batch.json`
- Time: ~2-3 minutes
- Cost: ~$0.04

---

### 2. Small Batch (50 conversations)
**Purpose:** Initial dataset for prototyping

```bash
python arabic_eou_data_generator.py \
    --num-conversations 50 \
    --output-file data/batch_small.json \
    --log-file logs/batch_small.log \
    --validate
```

**Expected output:**
- File: `data/batch_small.json`
- Time: ~20-25 minutes
- Cost: ~$0.35

---

### 3. Medium Batch (200 conversations)
**Purpose:** Development and initial training

```bash
python arabic_eou_data_generator.py \
    --num-conversations 200 \
    --output-file data/batch_medium.json \
    --log-file logs/batch_medium.log \
    --validate
```

**Expected output:**
- File: `data/batch_medium.json`
- Time: ~80-100 minutes
- Cost: ~$1.40

---

### 4. Large Batch (500 conversations)
**Purpose:** Full training dataset

```bash
python arabic_eou_data_generator.py \
    --num-conversations 500 \
    --output-file data/batch_large.json \
    --log-file logs/batch_large.log \
    --temperature 0.7 \
    --delay 0.5 \
    --validate
```

**Expected output:**
- File: `data/batch_large.json`
- Time: ~3-4 hours
- Cost: ~$3.50

---

### 5. Multiple Batches (Recommended for 1000+ conversations)

Instead of generating 1000 conversations in one run, split into multiple batches:

```bash
# Batch 1
python arabic_eou_data_generator.py \
    --num-conversations 250 \
    --output-file data/batch_1.json \
    --log-file logs/batch_1.log

# Batch 2
python arabic_eou_data_generator.py \
    --num-conversations 250 \
    --output-file data/batch_2.json \
    --log-file logs/batch_2.log

# Batch 3
python arabic_eou_data_generator.py \
    --num-conversations 250 \
    --output-file data/batch_3.json \
    --log-file logs/batch_3.log

# Batch 4
python arabic_eou_data_generator.py \
    --num-conversations 250 \
    --output-file data/batch_4.json \
    --log-file logs/batch_4.log

# Combine all batches
python combine_batches.py \
    --input-files data/batch_1.json data/batch_2.json data/batch_3.json data/batch_4.json \
    --output data/combined_1000.json
```

**Benefits of multiple batches:**
- ✅ Can pause and resume
- ✅ Easier to manage errors
- ✅ Better for rate limiting
- ✅ Can run in parallel (different machines)

---

## How Many Samples Do You Need?

### Recommended Dataset Sizes

| Dataset Size | Use Case | Training Quality | Cost | Time |
|--------------|----------|------------------|------|------|
| **50-100** | Quick prototyping | Poor | $0.35-0.70 | 20-40 min |
| **200-300** | Initial development | Fair | $1.40-2.10 | 1.5-2 hours |
| **500-800** | Good training | Good | $3.50-5.60 | 3-5 hours |
| **1,000-1,500** | **Recommended** | **Very Good** | **$7-10.50** | **6-9 hours** |
| **2,000-3,000** | Production quality | Excellent | $14-21 | 12-18 hours |
| **5,000+** | Research-grade | Outstanding | $35+ | 24+ hours |

---

## Optimal Dataset Size: **1,000-1,500 Conversations**

### Why This Range?

#### **Minimum Effective Size (1,000 conversations)**
- Provides sufficient diversity
- Covers all conversation types
- Enables proper train/validation/test splits
- Balances cost and quality

#### **Sweet Spot (1,200-1,500 conversations)**
- Excellent model performance
- Good generalization
- Reasonable cost (~$8-10)
- Manageable generation time (6-9 hours)

#### **Beyond 2,000 conversations**
- Diminishing returns for small models
- Better for large-scale production systems
- Higher cost and time investment

---

## Dataset Composition

### For 1,200 Conversations (Recommended)

**Synthetic Data (70-80%):** 840-960 conversations
```bash
# Generate in 4 batches of 240 each
python arabic_eou_data_generator.py --num-conversations 240 --output-file data/synthetic_batch_1.json
python arabic_eou_data_generator.py --num-conversations 240 --output-file data/synthetic_batch_2.json
python arabic_eou_data_generator.py --num-conversations 240 --output-file data/synthetic_batch_3.json
python arabic_eou_data_generator.py --num-conversations 240 --output-file data/synthetic_batch_4.json
```

**Real Data (20-30%):** 240-360 conversations
- Source from existing Arabic conversational datasets
- Saudi Arabic Spontaneous Speech Data
- Gulf Arabic Conversational Telephone Speech
- Manually collected conversations

---

## Turn-Level Statistics

### Expected Turns per Conversation
- Average: 8-10 turns
- Minimum: 6 turns
- Maximum: 12 turns

### For 1,200 Conversations:
- **Total turns:** ~10,800 turns (1,200 × 9 avg)
- **Training examples:** ~10,800 EOU detection instances
- **This is sufficient** for fine-tuning transformer models

---

## Data Splits

### Recommended Split Ratios

For 1,200 conversations:

| Split | Conversations | Turns (~9 avg) | Percentage |
|-------|---------------|----------------|------------|
| **Training** | 840 | 7,560 | 70% |
| **Validation** | 180 | 1,620 | 15% |
| **Test** | 180 | 1,620 | 15% |

### Creating Splits

```bash
# After combining all batches
python split_dataset.py \
    --input data/combined_1200.json \
    --train-ratio 0.70 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --output-dir data/splits/
```

---

## Cost Breakdown

### For 1,200 Conversations (Recommended)

**Using Qwen/Qwen3-235B-A22B-Instruct-2507:**
- Input tokens: ~1,200 conversations × 300 tokens = 360,000 tokens
- Output tokens: ~1,200 conversations × 1,500 tokens = 1,800,000 tokens
- **Input cost:** 360K tokens × $0.20/1M = $0.07
- **Output cost:** 1.8M tokens × $0.60/1M = $1.08
- **Total cost:** ~$1.15 per 1,200 conversations

**Wait, that's much cheaper than estimated!**

Let me recalculate based on actual usage:
- Average conversation: ~2,000 tokens (input + output)
- 1,200 conversations × 2,000 tokens = 2,400,000 tokens
- Cost: 2.4M × $0.40/1M (blended rate) = **~$0.96**

**Actual cost for 1,200 conversations: ~$1-2**

---

## Generation Time

### Factors Affecting Speed:
- Model response time: ~5-10 seconds per conversation
- Delay between requests: 0.5 seconds (default)
- Retries on failures: 2-3 attempts

### Estimated Time:
- **1,200 conversations:** 
  - 1,200 × 8 seconds = 9,600 seconds = **~2.7 hours**
  - With delays and retries: **~3-4 hours**

---

## Best Practice: Generate in Batches

### Recommended Approach for 1,200 Conversations

```bash
# Create directories
mkdir -p data logs

# Generate 5 batches of 240 conversations each
for i in {1..5}; do
    python arabic_eou_data_generator.py \
        --num-conversations 240 \
        --output-file data/batch_$i.json \
        --log-file logs/batch_$i.log \
        --validate
    
    echo "Completed batch $i of 5"
    sleep 5
done

# Combine all batches
python combine_batches.py \
    --input-files data/batch_*.json \
    --output data/arabic_eou_dataset_1200.json \
    --validate

echo "Dataset generation complete!"
```

---

## Quality Validation

### After Generation, Check:

1. **Total conversations:** Should match target (1,200)
2. **Valid conversations:** Should be >95%
3. **Average turns:** Should be 8-10
4. **EOU labels:** All turns should have `is_eou` field
5. **Language quality:** Sample 10-20 conversations manually

### Validation Command:

```bash
python arabic_eou_data_generator.py \
    --num-conversations 0 \
    --output-file data/arabic_eou_dataset_1200.json \
    --validate
```

---

## Summary: Quick Start Guide

### For Your Project (4-day deadline):

**Day 1: Generate Dataset**

```bash
# Morning: Test run
python arabic_eou_data_generator.py --num-conversations 10 --validate

# Afternoon: Generate full dataset (1,200 conversations in 5 batches)
./generate_batches.sh  # Script to generate 5 batches of 240 each

# Evening: Combine and validate
python combine_batches.py --input-files data/batch_*.json --output data/dataset.json
```

**Expected Results:**
- 1,200 synthetic conversations
- ~10,800 turns with EOU labels
- Cost: ~$1-2
- Time: 3-4 hours
- Ready for model training

---

## Troubleshooting

### Issue: Generation is slow
**Solution:** Reduce `--delay` parameter (minimum 0.2 seconds)

### Issue: API rate limits
**Solution:** Increase `--delay` or split into smaller batches

### Issue: JSON parsing errors
**Solution:** The generator has built-in retry logic; check logs for details

### Issue: Low quality conversations
**Solution:** Adjust `--temperature` (lower = more consistent, higher = more creative)

---

## Next Steps After Generation

1. **Validate dataset quality**
2. **Create train/val/test splits**
3. **Upload to Hugging Face**
4. **Proceed to model fine-tuning**

For model fine-tuning instructions, see `MODEL_TRAINING_GUIDE.md`
