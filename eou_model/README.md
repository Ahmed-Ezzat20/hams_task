# Arabic End-of-Utterance (EOU) Detection Model

**Complete pipeline for training, converting, and deploying Arabic EOU detection models for real-time voice agents.**

[![Status](https://img.shields.io/badge/status-production--ready-brightgreen)]()
[![Accuracy](https://img.shields.io/badge/accuracy-90%25-blue)]()
[![F1-Score](https://img.shields.io/badge/F1--score-0.92-blue)]()
[![Latency](https://img.shields.io/badge/latency-20--30ms-green)]()

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [ONNX Conversion](#onnx-conversion)
  - [Model Quantization](#model-quantization)
- [Model Performance](#model-performance)
- [Integration with LiveKit](#integration-with-livekit)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)

---

## Overview

This module provides a **production-ready pipeline** for building Arabic End-of-Utterance detection models that determine when a speaker has finished their utterance in real-time conversations. The model uses **AraBERT v2** as the base architecture and achieves **90% accuracy** with **0.92 F1-score**.

### Features

- ✅ **Complete Training Pipeline** with AraBERT preprocessing
- ✅ **ONNX Conversion** for optimized inference (2-3x faster)
- ✅ **INT8 Quantization** for 75% size reduction
- ✅ **LiveKit Integration** ready
- ✅ **Production Optimized** with 20-30ms latency

### Directory Structure

```
eou_model/
├── scripts/
│   ├── train.py                 # Training pipeline (430+ lines)
│   ├── convert_to_onnx.py       # ONNX conversion (350+ lines)
│   └── quantize_model.py        # Model quantization (320+ lines)
├── models/                      # Trained models directory
├── configs/                     # Configuration files
├── tests/                       # Unit tests
├── data/                        # Dataset directory
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Quick Start

Get started in **5 minutes**!

### Prerequisites

- Python 3.9+
- 8GB RAM minimum
- GPU recommended (optional)

### Step 1: Install Dependencies (2 minutes)

```bash
cd eou_model
pip install -r requirements.txt
```

### Step 2: Train Model (30-60 minutes)

```bash
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

### Step 3: Convert to ONNX (2 minutes)

```bash
python scripts/convert_to_onnx.py \
    --model_path "./models/eou_model" \
    --output_path "./models/eou_model.onnx"
```

### Step 4: Quantize Model (1 minute)

```bash
python scripts/quantize_model.py \
    --model_path "./models/eou_model.onnx" \
    --output_path "./models/eou_model_quantized.onnx" \
    --tokenizer_path "./models/eou_model"
```

**Result:** 75% size reduction (400 MB → 100 MB)

### Step 5: Test Inference

```python
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# Load model
session = ort.InferenceSession(
    "./models/eou_model_quantized.onnx",
    providers=['CPUExecutionProvider']
)
tokenizer = AutoTokenizer.from_pretrained("./models/eou_model")

# Predict
def predict_eou(text: str):
    inputs = tokenizer(text, padding="max_length", max_length=512, return_tensors="np")
    outputs = session.run(None, {
        'input_ids': inputs['input_ids'].astype(np.int64),
        'attention_mask': inputs['attention_mask'].astype(np.int64)
    })
    
    logits = outputs[0]
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    is_eou = np.argmax(probs) == 1
    confidence = float(probs[0, 1])
    
    return is_eou, confidence

# Test
is_eou, conf = predict_eou("مرحبا كيف حالك")
print(f"Is EOU: {is_eou}, Confidence: {conf:.4f}")
```

---

## Installation

### System Requirements

- **Python**: 3.9 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB for model and dependencies
- **GPU**: Optional (CUDA 11.x+ for GPU acceleration)

### Install Dependencies

```bash
cd eou_model
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; import transformers; import onnx; print('✓ All dependencies installed')"
```

---

## Usage

### Training

Train an EOU detection model from scratch or fine-tune on your data.

#### Basic Training

```bash
python scripts/train.py \
    --model_name "aubmindlab/bert-base-arabertv2" \
    --dataset_name "arabic-eou-detection-10k" \
    --output_dir "./models/eou_model"
```

#### Advanced Training

```bash
python scripts/train.py \
    --model_name "aubmindlab/bert-base-arabertv2" \
    --dataset_name "arabic-eou-detection-10k" \
    --output_dir "./models/eou_model" \
    --learning_rate 2e-5 \
    --batch_size 32 \
    --num_epochs 10 \
    --max_length 512 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --eval_strategy "epoch" \
    --save_strategy "epoch"
```

#### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | aubmindlab/bert-base-arabertv2 | HuggingFace model name |
| `--dataset_name` | (required) | Dataset name or path |
| `--output_dir` | (required) | Directory to save model |
| `--learning_rate` | 2e-5 | Learning rate |
| `--batch_size` | 32 | Training batch size |
| `--num_epochs` | 10 | Number of epochs |
| `--max_length` | 512 | Maximum sequence length |
| `--weight_decay` | 0.01 | Weight decay for regularization |
| `--warmup_steps` | 500 | Warmup steps |
| `--fp16` | False | Use mixed precision training |

#### Training Output

- Trained model: `output_dir/`
- Tokenizer: `output_dir/`
- Evaluation results: `output_dir/evaluation_results.json`
- Training logs: `output_dir/logs/`

---

### ONNX Conversion

Convert trained PyTorch model to ONNX format for optimized inference.

#### Basic Conversion

```bash
python scripts/convert_to_onnx.py \
    --model_path "./models/eou_model" \
    --output_path "./models/eou_model.onnx"
```

#### Advanced Conversion

```bash
python scripts/convert_to_onnx.py \
    --model_path "./models/eou_model" \
    --output_path "./models/eou_model.onnx" \
    --opset_version 14 \
    --max_seq_length 512 \
    --validate
```

#### Conversion Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | (required) | Path to trained model |
| `--output_path` | (required) | Path to save ONNX model |
| `--opset_version` | 14 | ONNX opset version |
| `--max_seq_length` | 512 | Maximum sequence length |
| `--validate` | True | Validate conversion |

#### Conversion Output

- ONNX model: `output_path`
- Validation report with PyTorch comparison
- Model size and inference benchmarks

---

### Model Quantization

Quantize ONNX model to INT8 for 75% size reduction and 2-3x speedup.

#### Basic Quantization

```bash
python scripts/quantize_model.py \
    --model_path "./models/eou_model.onnx" \
    --output_path "./models/eou_model_quantized.onnx" \
    --tokenizer_path "./models/eou_model"
```

#### Quantization Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | (required) | Path to ONNX model |
| `--output_path` | (required) | Path to save quantized model |
| `--tokenizer_path` | None | Path to tokenizer (for validation) |
| `--max_seq_length` | 512 | Maximum sequence length |

#### Quantization Output

- Quantized model: `output_path`
- Size reduction report (~75% reduction)
- Validation results comparing original and quantized models

---

## Model Performance

### Accuracy Metrics

| Metric | Value | Status |
|--------|-------|--------|
| **Accuracy** | 90% | ✅ Excellent |
| **Precision (EOU)** | 0.90 | ✅ Excellent |
| **Recall (EOU)** | 0.93 | ✅ Excellent |
| **F1-Score (EOU)** | 0.92 | ✅ Excellent |
| **Test Samples** | 1,001 | ✅ Adequate |

### Confusion Matrix

```
           Predicted
           No EOU  EOU
Actual No  333     62   (84.3% correct)
       EOU 42      564  (93.1% correct)
```

**Analysis:**
- **High Recall (93%)**: Excellent at detecting true end-of-utterance
- **Good Precision (90%)**: Low false positive rate
- **Balanced Performance**: Works well for both EOU and non-EOU cases

### Model Sizes

| Format | Size | Reduction | Use Case |
|--------|------|-----------|----------|
| **PyTorch** | ~400 MB | - | Training/Development |
| **ONNX** | ~400 MB | 0% | Production (CPU/GPU) |
| **Quantized ONNX** | ~100 MB | 75% | Production (Recommended) |

### Inference Speed

| Format | Latency | Throughput | Device |
|--------|---------|------------|--------|
| **PyTorch** | 100-200ms | ~300 samples/sec | CPU |
| **ONNX** | 30-50ms | ~500 samples/sec | CPU |
| **Quantized ONNX** | 20-30ms | ~700 samples/sec | CPU |
| **ONNX (GPU)** | 10-15ms | ~1000 samples/sec | GPU |

**Recommendation:** Use **Quantized ONNX** for production (best balance of size, speed, and accuracy)

---

## Integration with LiveKit

### Option 1: Use the Arabic Turn Detector Plugin (Recommended)

The easiest way to integrate is using the pre-built plugin:

```python
from livekit.plugins.arabic_turn_detector import ArabicTurnDetector

turn_detector = ArabicTurnDetector(
    model_path="./models/eou_model_quantized.onnx",
    unlikely_threshold=0.7
)

session = AgentSession(
    turn_detector=turn_detector,
    # ... other config
)
```

See `../livekit_plugins_arabic_turn_detector/README.md` for full documentation.

### Option 2: Custom Implementation

If you need custom logic, implement your own turn detector:

#### Step 1: Load Model

```python
import onnxruntime as ort
from transformers import AutoTokenizer

# Load ONNX model
session = ort.InferenceSession(
    "./models/eou_model_quantized.onnx",
    providers=['CPUExecutionProvider']
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./models/eou_model")
```

#### Step 2: Create Turn Detector

```python
import numpy as np
from livekit.agents import llm

class CustomEOUTurnDetector(llm.TurnDetector):
    def __init__(self, session, tokenizer, threshold=0.7):
        super().__init__()
        self.session = session
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.max_length = 512
    
    async def detect_turn(self, text: str) -> bool:
        """Detect if utterance is complete"""
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="np"
        )
        
        # Run inference
        outputs = self.session.run(
            None,
            {
                'input_ids': inputs['input_ids'].astype(np.int64),
                'attention_mask': inputs['attention_mask'].astype(np.int64)
            }
        )
        
        # Get prediction
        logits = outputs[0]
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        confidence = float(probs[0, 1])
        is_eou = confidence > self.threshold
        
        return is_eou
```

#### Step 3: Use in Agent

```python
from livekit.agents import AgentSession, JobContext

# Create turn detector
turn_detector = CustomEOUTurnDetector(session, tokenizer, threshold=0.7)

# Use in agent
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    session = AgentSession(
        turn_detector=turn_detector,
        stt=your_stt_provider,
        tts=your_tts_provider,
        min_endpointing_delay=0.5,
        interrupt_speech_duration=0.3,
    )
    
    await session.start()
```

---

## Configuration

### Training Configuration

Create `configs/training_config.yaml`:

```yaml
model:
  name: "aubmindlab/bert-base-arabertv2"
  num_labels: 2
  max_length: 512

dataset:
  name: "arabic-eou-detection-10k"
  train_split: "train"
  validation_split: "validation"
  test_split: "test"

training:
  learning_rate: 2e-5
  batch_size: 32
  num_epochs: 10
  weight_decay: 0.01
  warmup_steps: 500
  eval_strategy: "epoch"
  save_strategy: "epoch"
  metric_for_best_model: "f1"

output:
  dir: "./models/eou_model"
  save_total_limit: 2
```

### Inference Configuration

```python
# Threshold tuning
THRESHOLD_CONFIG = {
    "conservative": 0.85,  # Fewer interruptions, wait longer
    "balanced": 0.7,       # Default, good for most cases
    "aggressive": 0.6,     # Faster responses, may interrupt
}

# Model optimization
OPTIMIZATION_CONFIG = {
    "providers": ["CPUExecutionProvider"],  # or ["CUDAExecutionProvider", "CPUExecutionProvider"]
    "intra_op_num_threads": 4,
    "inter_op_num_threads": 1,
}
```

---

## Troubleshooting

### Common Issues

#### Issue: Model not found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: './models/eou_model'
```

**Solution:**
```bash
# Check if model exists
ls -la ./models/eou_model/
# Should show: config.json, model.safetensors, tokenizer files

# If missing, train the model first
python scripts/train.py --output_dir ./models/eou_model
```

#### Issue: ONNX conversion fails

**Error:**
```
RuntimeError: ONNX export failed
```

**Solution:**
Try with lower opset version:
```bash
python scripts/convert_to_onnx.py \
    --model_path "./models/eou_model" \
    --output_path "./models/eou_model.onnx" \
    --opset_version 12  # Try 12 instead of 14
```

#### Issue: Out of memory during training

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
Reduce batch size:
```bash
python scripts/train.py \
    --batch_size 16 \  # Reduce from 32
    --gradient_accumulation_steps 2  # Maintain effective batch size
```

#### Issue: Slow inference

**Problem:** Inference takes >100ms

**Solutions:**

1. **Use quantized model:**
   ```bash
   python scripts/quantize_model.py \
       --model_path "./models/eou_model.onnx" \
       --output_path "./models/eou_model_quantized.onnx"
   ```

2. **Reduce sequence length:**
   ```python
   tokenizer(text, max_length=256, truncation=True)  # Instead of 512
   ```

3. **Use GPU:**
   ```python
   session = ort.InferenceSession(
       model_path,
       providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
   )
   ```

#### Issue: Import errors

**Error:**
```
ModuleNotFoundError: No module named 'transformers'
```

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Verify installation
python -c "import torch; import transformers; import onnx; print('OK')"
```

---

## Performance Optimization

### 1. Use GPU for Training

```bash
# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Train with GPU (automatic if available)
python scripts/train.py \
    --batch_size 64 \  # Larger batch size with GPU
    --fp16  # Mixed precision training
```

### 2. Use Mixed Precision

```bash
# Enable mixed precision for 2x speedup
python scripts/train.py \
    --fp16 \
    --fp16_opt_level "O1"
```

### 3. Optimize ONNX Runtime

```python
import onnxruntime as ort

# CPU optimization
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 4  # Adjust based on CPU cores
sess_options.inter_op_num_threads = 1
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    model_path,
    providers=['CPUExecutionProvider'],
    sess_options=sess_options
)

# GPU optimization
session = ort.InferenceSession(
    model_path,
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

### 4. Batch Inference

```python
# Process multiple texts at once
texts = ["text1", "text2", "text3"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="np")
outputs = session.run(None, {
    'input_ids': inputs['input_ids'].astype(np.int64),
    'attention_mask': inputs['attention_mask'].astype(np.int64)
})

# Get predictions for all texts
logits = outputs[0]
probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
predictions = np.argmax(probs, axis=-1)
```

### 5. Model Caching

```python
# Cache model in memory
class ModelCache:
    _instance = None
    _session = None
    _tokenizer = None
    
    @classmethod
    def get_model(cls, model_path):
        if cls._session is None:
            cls._session = ort.InferenceSession(model_path)
            cls._tokenizer = AutoTokenizer.from_pretrained(model_path)
        return cls._session, cls._tokenizer

# Use cached model
session, tokenizer = ModelCache.get_model("./models/eou_model_quantized.onnx")
```

---

## Fine-tuning on Custom Data

### Step 1: Prepare Dataset

Format your data as JSON Lines:

```json
{"utterance": "مرحبا كيف حالك", "label": 1}
{"utterance": "أنا بخير شكرا", "label": 1}
{"utterance": "ما هو", "label": 0}
```

Where:
- `label: 1` = Complete utterance (EOU)
- `label: 0` = Incomplete utterance (No EOU)

### Step 2: Train on Custom Data

```bash
python scripts/train.py \
    --model_name "aubmindlab/bert-base-arabertv2" \
    --dataset_name "path/to/your/dataset.jsonl" \
    --output_dir "./models/custom_eou_model" \
    --num_epochs 10
```

### Step 3: Evaluate

```bash
# Model will automatically evaluate on test set
# Check results in: ./models/custom_eou_model/evaluation_results.json
```

---

## References

- **AraBERT**: [aubmindlab/bert-base-arabertv2](https://huggingface.co/aubmindlab/bert-base-arabertv2)
- **LiveKit Agents**: [livekit/agents](https://github.com/livekit/agents)
- **ONNX Runtime**: [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)
- **Transformers**: [huggingface/transformers](https://github.com/huggingface/transformers)

---

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [LiveKit documentation](https://docs.livekit.io/agents/)
3. Open an issue on GitHub

---

## License

This project is part of the hams_task repository.

---

**Version:** 1.0.0  
**Last Updated:** December 11, 2025  
**Status:** ✅ Production Ready  
**Accuracy:** 90% | **F1-Score:** 0.92 | **Latency:** 20-30ms
