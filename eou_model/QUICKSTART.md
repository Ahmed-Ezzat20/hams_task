# EOU Model - Quick Start Guide

Get started with the EOU model in 5 minutes!

## Prerequisites

- Python 3.8+
- pip
- 8GB RAM minimum
- GPU recommended (but not required)

## Step 1: Installation (2 minutes)

```bash
# Navigate to eou_model directory
cd eou_model

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Train Model (30-60 minutes)

```bash
# Train with default settings
python scripts/train.py \
    --model_name "aubmindlab/bert-base-arabertv2" \
    --dataset_name "arabic-eou-detection-10k" \
    --output_dir "./models/eou_model" \
    --num_epochs 10
```

**Expected output:**
```
Training complete!
✓ Accuracy: 90%
✓ F1-Score: 0.92
✓ Model saved to: ./models/eou_model
```

## Step 3: Convert to ONNX (2 minutes)

```bash
# Convert trained model to ONNX
python scripts/convert_to_onnx.py \
    --model_path "./models/eou_model" \
    --output_path "./models/eou_model.onnx"
```

**Expected output:**
```
✓ Conversion Complete!
ONNX model saved to: ./models/eou_model.onnx
File size: ~400 MB
```

## Step 4: Quantize Model (1 minute)

```bash
# Quantize for production
python scripts/quantize_model.py \
    --model_path "./models/eou_model.onnx" \
    --output_path "./models/eou_model_quantized.onnx" \
    --tokenizer_path "./models/eou_model"
```

**Expected output:**
```
✓ Quantization Complete!
Size reduction: 75%
Original: 400 MB → Quantized: 100 MB
```

## Step 5: Test Inference (1 minute)

Create `test_inference.py`:

```python
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# Load model and tokenizer
session = ort.InferenceSession(
    "./models/eou_model_quantized.onnx",
    providers=['CPUExecutionProvider']
)
tokenizer = AutoTokenizer.from_pretrained("./models/eou_model")

# Test function
def predict_eou(text: str):
    inputs = tokenizer(
        text,
        padding="max_length",
        max_length=512,
        return_tensors="np"
    )
    
    outputs = session.run(
        None,
        {
            'input_ids': inputs['input_ids'].astype(np.int64),
            'attention_mask': inputs['attention_mask'].astype(np.int64)
        }
    )
    
    logits = outputs[0]
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    is_eou = np.argmax(probs) == 1
    confidence = float(probs[0, 1])
    
    return is_eou, confidence

# Test examples
test_texts = [
    "مرحبا كيف حالك",
    "أنا بخير شكرا",
    "ما هو اسمك",
]

print("Testing EOU Detection:")
print("=" * 60)
for text in test_texts:
    is_eou, conf = predict_eou(text)
    status = "✓ EOU" if is_eou else "✗ Not EOU"
    print(f"{status} | Confidence: {conf:.4f} | Text: {text}")
```

Run the test:

```bash
python test_inference.py
```

**Expected output:**
```
Testing EOU Detection:
============================================================
✓ EOU | Confidence: 0.8523 | Text: مرحبا كيف حالك
✓ EOU | Confidence: 0.9012 | Text: أنا بخير شكرا
✗ Not EOU | Confidence: 0.3421 | Text: ما هو اسمك
```

## Next Steps

### Integrate with LiveKit

See the [README.md](README.md#integration-with-livekit) for LiveKit integration examples.

### Optimize Performance

1. **Use GPU for inference:**
   ```python
   session = ort.InferenceSession(
       model_path,
       providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
   )
   ```

2. **Adjust confidence threshold:**
   ```python
   threshold = 0.7  # Lower = more sensitive, Higher = more conservative
   is_eou = confidence > threshold
   ```

3. **Batch inference:**
   ```python
   # Process multiple texts at once
   texts = ["text1", "text2", "text3"]
   inputs = tokenizer(texts, padding=True, return_tensors="np")
   outputs = session.run(None, onnx_inputs)
   ```

### Fine-tune on Your Data

1. Prepare your dataset in the format:
   ```json
   {"utterance": "text", "label": 0}  # 0 = No EOU, 1 = EOU
   ```

2. Train with your dataset:
   ```bash
   python scripts/train.py \
       --dataset_name "path/to/your/dataset" \
       --output_dir "./models/custom_eou_model"
   ```

## Troubleshooting

### Issue: Import errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: Out of memory

```bash
# Use smaller batch size
python scripts/train.py --batch_size 16
```

### Issue: Slow inference

```bash
# Use quantized model
# Already done in Step 4!
```

## Resources

- **Full Documentation**: [README.md](README.md)
- **Training Analysis**: See the training notebook analysis document
- **LiveKit Integration**: [LiveKit Agents Documentation](https://docs.livekit.io/agents/)

## Support

Need help? Check:
1. [README.md](README.md) - Full documentation
2. [Troubleshooting section](README.md#troubleshooting)
3. GitHub Issues

---

**Congratulations!** 🎉 You now have a production-ready EOU detection model!
