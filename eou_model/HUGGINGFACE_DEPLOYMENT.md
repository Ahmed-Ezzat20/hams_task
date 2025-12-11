# HuggingFace Hub Deployment Guide

**Complete guide for deploying your Arabic EOU model to HuggingFace Hub for easy sharing and inference.**

---

## Table of Contents

- [Why HuggingFace Hub?](#why-huggingface-hub)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Steps](#detailed-steps)
- [Model Formats](#model-formats)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Why HuggingFace Hub?

### Benefits

✅ **Easy Sharing**: Share your model with a simple URL  
✅ **Version Control**: Track model versions and updates  
✅ **Automatic Inference API**: Test your model in the browser  
✅ **Easy Integration**: Load with one line of code  
✅ **Model Card**: Automatic documentation and metrics display  
✅ **Community**: Reach thousands of ML practitioners  

### What Gets Uploaded

1. **PyTorch Model** - For training and fine-tuning
2. **ONNX Model** - For optimized inference (optional)
3. **Quantized ONNX** - For production deployment (optional)
4. **Tokenizer** - AraBERT tokenizer files
5. **Model Card** - Comprehensive documentation
6. **Config Files** - Model configuration and metadata

---

## Prerequisites

### 1. HuggingFace Account

Create a free account at [huggingface.co](https://huggingface.co/join)

### 2. Access Token

Get your token from [Settings → Access Tokens](https://huggingface.co/settings/tokens)

**Permissions needed:**
- ✅ Write access

### 3. Install Dependencies

```bash
pip install huggingface_hub
```

### 4. Login to HuggingFace

```bash
# Option 1: Login via CLI
huggingface-cli login

# Option 2: Set environment variable
export HF_TOKEN="your_token_here"
```

---

## Quick Start

### Upload Complete Model (Recommended)

```bash
cd eou_model

# Upload PyTorch + ONNX + Quantized models
python scripts/upload_to_huggingface.py \
    --model_path ./models/eou_model \
    --repo_name "your-username/arabic-eou-detector" \
    --onnx_model_path ./models/eou_model.onnx \
    --quantized_model_path ./models/eou_model_quantized.onnx
```

**That's it!** Your model is now available at:
`https://huggingface.co/your-username/arabic-eou-detector`

---

## Detailed Steps

### Step 1: Prepare Your Model

Ensure you have all model files:

```bash
ls -la models/eou_model/
# Should show:
# - config.json
# - pytorch_model.bin (or model.safetensors)
# - tokenizer.json
# - tokenizer_config.json
# - special_tokens_map.json
# - vocab.txt

ls -la models/
# Should show:
# - eou_model.onnx (optional)
# - eou_model_quantized.onnx (optional)
```

### Step 2: Choose Repository Name

Format: `username/model-name`

**Examples:**
- `ahmed-ezzat/arabic-eou-detector`
- `your-org/arabic-turn-detection`
- `your-name/arabert-eou-v1`

**Best practices:**
- Use lowercase
- Use hyphens (not underscores)
- Be descriptive
- Include language if multilingual

### Step 3: Run Upload Script

#### Basic Upload (PyTorch only)

```bash
python scripts/upload_to_huggingface.py \
    --model_path ./models/eou_model \
    --repo_name "your-username/arabic-eou-detector"
```

#### Full Upload (PyTorch + ONNX + Quantized)

```bash
python scripts/upload_to_huggingface.py \
    --model_path ./models/eou_model \
    --repo_name "your-username/arabic-eou-detector" \
    --onnx_model_path ./models/eou_model.onnx \
    --quantized_model_path ./models/eou_model_quantized.onnx
```

#### Private Repository

```bash
python scripts/upload_to_huggingface.py \
    --model_path ./models/eou_model \
    --repo_name "your-username/arabic-eou-detector" \
    --private
```

#### With Custom Token

```bash
python scripts/upload_to_huggingface.py \
    --model_path ./models/eou_model \
    --repo_name "your-username/arabic-eou-detector" \
    --token "hf_your_token_here"
```

### Step 4: Verify Upload

Visit your model page:
```
https://huggingface.co/your-username/arabic-eou-detector
```

You should see:
- ✅ Model card with metrics
- ✅ Files tab with all uploaded files
- ✅ Inference API widget (try it!)

---

## Model Formats

### Which Format to Upload?

| Format | Size | Speed | Use Case | Upload? |
|--------|------|-------|----------|---------|
| **PyTorch** | 516 MB | Baseline | Training, fine-tuning | ✅ Required |
| **ONNX** | 516 MB | 2-3x faster | Production inference | ✅ Recommended |
| **Quantized ONNX** | 130 MB | 3-4x faster | Production (best) | ✅ Highly Recommended |

### Recommended: Upload All Three

**Why?**
- **PyTorch**: For users who want to fine-tune
- **ONNX**: For users who want fast inference
- **Quantized**: For users who want production deployment

**How?**
```bash
python scripts/upload_to_huggingface.py \
    --model_path ./models/eou_model \
    --repo_name "your-username/arabic-eou-detector" \
    --onnx_model_path ./models/eou_model.onnx \
    --quantized_model_path ./models/eou_model_quantized.onnx
```

---

## Usage Examples

### After Upload: How Users Load Your Model

#### PyTorch (Transformers)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "your-username/arabic-eou-detector"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Inference
def predict_eou(text: str):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(-1).item()
    return prediction == 1  # True if EOU

# Test
is_eou = predict_eou("مرحبا كيف حالك")
print(f"Is EOU: {is_eou}")
```

#### ONNX (Production)

```python
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np

model_name = "your-username/arabic-eou-detector"

# Download ONNX model
onnx_path = hf_hub_download(
    repo_id=model_name,
    filename="model_quantized.onnx"  # or "model.onnx"
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
session = ort.InferenceSession(onnx_path)

# Inference
def predict_eou(text: str):
    inputs = tokenizer(text, padding="max_length", max_length=512, return_tensors="np")
    outputs = session.run(None, {
        'input_ids': inputs['input_ids'].astype(np.int64),
        'attention_mask': inputs['attention_mask'].astype(np.int64)
    })
    prediction = np.argmax(outputs[0], axis=-1)[0]
    return prediction == 1

# Test
is_eou = predict_eou("مرحبا كيف حالك")
print(f"Is EOU: {is_eou}")
```

#### LiveKit Integration

```python
from huggingface_hub import hf_hub_download
from livekit.plugins.arabic_turn_detector import ArabicTurnDetector

# Download quantized model
model_path = hf_hub_download(
    repo_id="your-username/arabic-eou-detector",
    filename="model_quantized.onnx"
)

# Create turn detector
turn_detector = ArabicTurnDetector(
    model_path=model_path,
    unlikely_threshold=0.7
)

# Use in agent
session = AgentSession(
    turn_detector=turn_detector,
    # ... other config
)
```

---

## Best Practices

### 1. Model Card Quality

The upload script automatically creates a comprehensive model card with:
- ✅ Model description
- ✅ Performance metrics
- ✅ Usage examples
- ✅ Training details
- ✅ Limitations
- ✅ Citation

**Customize it** by editing the model card template in `upload_to_huggingface.py`

### 2. Versioning

Use git tags to version your model:

```bash
# Tag a version
git tag -a v1.0.0 -m "Initial release"
git push origin v1.0.0

# Upload with version tag
python scripts/upload_to_huggingface.py \
    --model_path ./models/eou_model \
    --repo_name "your-username/arabic-eou-detector" \
    --revision "v1.0.0"
```

### 3. Model Naming

**Good names:**
- `arabic-eou-detector`
- `arabert-turn-detection`
- `arabic-utterance-classifier`

**Avoid:**
- `model` (too generic)
- `eou_model_final_v2` (unclear)
- `test123` (not descriptive)

### 4. Documentation

Include in your model card:
- ✅ What the model does
- ✅ How to use it
- ✅ Performance metrics
- ✅ Limitations
- ✅ Training data
- ✅ License

### 5. License

Choose an appropriate license:
- **Apache 2.0**: Permissive, allows commercial use
- **MIT**: Very permissive
- **CC BY-NC**: Non-commercial only

Set in model card metadata:
```yaml
license: apache-2.0
```

---

## Troubleshooting

### Issue: Authentication Failed

**Error:**
```
HTTPError: 401 Client Error: Unauthorized
```

**Solution:**
```bash
# Re-login
huggingface-cli login

# Or set token
export HF_TOKEN="your_token_here"
```

### Issue: Repository Already Exists

**Error:**
```
Repository already exists
```

**Solution:**
The script uses `exist_ok=True`, so this shouldn't happen. If it does:

```bash
# Delete existing repo (if you own it)
huggingface-cli delete-repo your-username/arabic-eou-detector

# Or use a different name
python scripts/upload_to_huggingface.py \
    --repo_name "your-username/arabic-eou-detector-v2"
```

### Issue: File Too Large

**Error:**
```
File too large to upload
```

**Solution:**
HuggingFace supports files up to 50GB. If your model is larger:

1. Use Git LFS (automatically enabled)
2. Upload in chunks
3. Use quantized model instead

### Issue: Model Card Not Showing

**Problem:** Model card (README.md) not displaying

**Solution:**
1. Check file was uploaded: Files tab → README.md
2. Ensure YAML frontmatter is valid
3. Re-upload with correct formatting

### Issue: Inference API Not Working

**Problem:** Inference widget shows error

**Solution:**
1. Ensure `config.json` has correct `task` field
2. Check model outputs are compatible
3. May take a few minutes to initialize

---

## Advanced Usage

### Update Existing Model

```bash
# Upload new version
python scripts/upload_to_huggingface.py \
    --model_path ./models/eou_model_v2 \
    --repo_name "your-username/arabic-eou-detector"

# The script will update existing files
```

### Upload Only ONNX

```bash
from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="./models/eou_model_quantized.onnx",
    path_in_repo="model_quantized.onnx",
    repo_id="your-username/arabic-eou-detector",
    token="your_token"
)
```

### Download Model Programmatically

```python
from huggingface_hub import snapshot_download

# Download entire repository
model_dir = snapshot_download(
    repo_id="your-username/arabic-eou-detector",
    cache_dir="./models"
)

# Download specific file
from huggingface_hub import hf_hub_download

onnx_path = hf_hub_download(
    repo_id="your-username/arabic-eou-detector",
    filename="model_quantized.onnx"
)
```

---

## Example: Complete Workflow

```bash
# 1. Train model
python scripts/train.py --output_dir ./models/eou_model

# 2. Convert to ONNX
python scripts/convert_to_onnx.py \
    --model_path ./models/eou_model \
    --output_path ./models/eou_model.onnx

# 3. Quantize
python scripts/quantize_model.py \
    --model_path ./models/eou_model.onnx \
    --output_path ./models/eou_model_quantized.onnx

# 4. Upload to HuggingFace
python scripts/upload_to_huggingface.py \
    --model_path ./models/eou_model \
    --repo_name "your-username/arabic-eou-detector" \
    --onnx_model_path ./models/eou_model.onnx \
    --quantized_model_path ./models/eou_model_quantized.onnx

# 5. Share!
# https://huggingface.co/your-username/arabic-eou-detector
```

---

## Resources

- **HuggingFace Hub Documentation**: https://huggingface.co/docs/hub
- **Model Upload Guide**: https://huggingface.co/docs/hub/models-uploading
- **Model Cards**: https://huggingface.co/docs/hub/model-cards
- **ONNX on HuggingFace**: https://huggingface.co/docs/optimum

---

## Summary

### ✅ Benefits of HuggingFace Hub

1. **Easy Sharing**: One URL, anyone can use
2. **Version Control**: Track changes
3. **Automatic API**: Test in browser
4. **Community**: Reach thousands of users
5. **Integration**: Works with Transformers, ONNX Runtime, etc.

### 🚀 Quick Upload

```bash
python scripts/upload_to_huggingface.py \
    --model_path ./models/eou_model \
    --repo_name "your-username/arabic-eou-detector" \
    --onnx_model_path ./models/eou_model.onnx \
    --quantized_model_path ./models/eou_model_quantized.onnx
```

### 📦 What Users Get

```python
# One line to load your model!
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("your-username/arabic-eou-detector")
model = AutoModelForSequenceClassification.from_pretrained("your-username/arabic-eou-detector")
```

---

**Ready to share your model with the world? Let's go!** 🚀
